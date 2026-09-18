import gc
import json
import os

import fsspec
import hydra
import lightning as L
from lightning.fabric import Fabric
import numpy as np
import omegaconf
import rich.syntax
import rich.tree
import torch
from tqdm import trange

import algo
import checkpoint_ema
import dataloader
from evaluated_checkpoint import EvaluatedCheckpoint
import model_assets
import nirvana_snapshot
import sandbox_gsm8k
import utils
from ipdb import set_trace as debug

torch.autograd.set_detect_anomaly(True)

utils.register_omegaconf_resolvers()


class PeriodicStepCheckpoint(L.Callback):
  """Save deterministic checkpoints every N completed train batches."""

  def __init__(self, dirpath, every_n_train_steps, snapshot_dir=None,
               snapshot=False):
    self.dirpath = os.fspath(dirpath)
    self.every_n_train_steps = int(every_n_train_steps)
    self._completed_train_batches = 0
    self._last_saved_step = 0
    self.snapshot_dir = os.fspath(snapshot_dir or dirpath)
    self.snapshot = bool(snapshot)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self.every_n_train_steps <= 0:
      return

    self._completed_train_batches += 1
    save_step = (
      self._completed_train_batches // self.every_n_train_steps
    ) * self.every_n_train_steps
    if save_step <= 0 or save_step <= self._last_saved_step:
      return

    os.makedirs(self.dirpath, exist_ok=True)
    ckpt_path = os.path.join(self.dirpath, f'step={save_step:08d}.ckpt')
    # All DDP ranks must enter save_checkpoint because Lightning synchronizes
    # inside the strategy; only rank zero writes the file.
    trainer.save_checkpoint(ckpt_path)
    self._last_saved_step = save_step
    if trainer.is_global_zero:
      print(
        f'Saved periodic checkpoint at train_batch={save_step}: {ckpt_path}',
        flush=True)
      nirvana_snapshot.save(self.snapshot_dir, enabled=self.snapshot,
                            reason=f'periodic train_batch={save_step}')


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    model = diffusion_model(config, tokenizer=tokenizer).to('cuda')
    return model
  
  return diffusion_model.load_from_checkpoint(
    model_assets.resolve_checkpoint(
      config.eval.checkpoint_path, config.get('model_weights_dir')),
    tokenizer=tokenizer,
    config=config,
    weights_only=False)


def _configure_training_mode(config):
  """Validate initialization and configure teacher continuation.

  `training.finetune_path` has always been a weights-only warm start. The new
  flag deliberately keeps that behavior, but forces the ordinary model loss
  even when a shared command line already set `is_distill=True`.
  """
  checkpoint_ema.validate_finetune_weights(config)
  if (config.mode != 'train'
      or not bool(config.training.get('continue_teacher', False))):
    return
  if not config.training.finetune_path:
    raise ValueError(
      'training.continue_teacher=True requires training.finetune_path; '
      'there is no teacher checkpoint to continue.')
  config.adversarial_distill.is_distill = False
  # Distillation performs its own accumulation and its launch commands
  # therefore set Lightning's accumulation to one. Once the ordinary automatic
  # optimization path is selected, restore the accumulation implied by the
  # requested global batch size instead of silently training at a smaller one.
  config.trainer.accumulate_grad_batches = int(
    config.adversarial_distill.accum_batches)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      model.metrics.record_entropy(samples)
      text_samples = model.tokenizer.batch_decode(samples)
      model.metrics.record_generative_perplexity(
        text_samples, config.model.length, model.device)
      all_samples.extend(list(text_samples))
  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'generated_seqs': all_samples}, f, indent=4)
  print('Samples saved at:', samples_path)

def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  if config.training.get('continue_teacher', False):
    logger.info(
      'Continuing the teacher with the standard model objective (no frozen '
      'teacher, fake model, DMD, or MMD reward); Lightning gradient '
      f'accumulation={config.trainer.accumulate_grad_batches}.')
  snapshot = bool(config.checkpointing.get('snapshot', True))
  snapshot_dir = os.fspath(config.checkpointing.save_dir)
  # wandb_logger = None
  # if config.get('wandb', None) is not None:
  #   wandb_logger = L.pytorch.loggers.WandbLogger(
  #     config=omegaconf.OmegaConf.to_object(config),
  #     **config.wandb)
  tensorboard_logger = hydra.utils.instantiate(config.logger)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
    if config.checkpointing.get('use_periodic_train_batch_checkpoint', False):
      callbacks.append(PeriodicStepCheckpoint(
        os.path.join(os.fspath(config.checkpointing.save_dir), 'checkpoints'),
        config.callbacks.checkpoint_every_n_steps.every_n_train_steps,
        snapshot_dir=snapshot_dir, snapshot=snapshot))

  if config.checkpointing.get('save_evaluated_checkpoints', False):
    callbacks.append(EvaluatedCheckpoint(
      dirpath=os.path.join(
        os.fspath(config.checkpointing.save_dir), 'checkpoints'),
      start_step=config.checkpointing.get('save_evaluated_from_step', 0),
      snapshot_dir=snapshot_dir, snapshot=snapshot))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    finetune_path = model_assets.resolve_checkpoint(
      config.training.finetune_path, config.get('model_weights_dir'))
    assert utils.fsspec_exists(finetune_path)
    model = diffusion_model.load_from_checkpoint(
      finetune_path,
      tokenizer=tokenizer,
      config=config,
      weights_only=False)
    if checkpoint_ema.apply_finetune_weights(
        model, config, resuming=ckpt_path is not None):
      logger.info(
        'Initialized student, frozen teacher and fake model from checkpoint '
        'EMA weights; preserved EMA decay and update counter.')
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=tensorboard_logger,
    enable_progress_bar=False if config.debug else True)
  if config.eval.get('validate_before_train', False):
    # Baseline at step 0. For a distillation run the student *is* the teacher
    # here, so this is the teacher's own score under the exact metrics the run
    # is judged by -- the zero point the curve is measured against.
    # Lightning finalizes the logger when `validate` returns, which for a
    # tracker closes the run and makes every later write fail; defer that
    # across the call, `fit` finalizes for real.
    logger.info('Baseline validation before training (step 0).')
    defers = hasattr(tensorboard_logger, 'defer_finalize')
    if defers:
      tensorboard_logger.defer_finalize = True
    try:
      trainer.validate(model, valid_ds)
    finally:
      if defers:
        tensorboard_logger.defer_finalize = False
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)



def _load_generated_gsm8k(config, logger):
  logger.info(f'Loading GSM8K predictions from {config.gsm8k.input_file}')
  with open(config.gsm8k.input_file) as f:
    loaded = json.load(f)
  # Accept either a bare list of records or a previously-saved results.json
  # (which wraps the list under 'records').
  if isinstance(loaded, dict) and 'records' in loaded:
    return loaded['records']
  return loaded


def _pad_prefix_batch(id_lists, device):
  """Pad a list of 1D tensors into a batch.

  Returns:
    padded: (B, max_len) int64 tensor, zero-padded.
    lengths: (B,) int64 tensor of actual lengths.
  """
  lengths = torch.tensor(
    [len(ids) for ids in id_lists], device=device)
  max_len = int(lengths.max())
  padded = torch.zeros(
    len(id_lists), max_len, dtype=torch.long, device=device)
  for i, ids in enumerate(id_lists):
    padded[i, :len(ids)] = ids.to(device)
  return padded, lengths


def _sample_gsm8k(diffusion_model, config, logger, tokenizer, fabric):
  model = _load_from_checkpoint(diffusion_model, config, tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  # Load dataset on rank 0 first, then broadcast to others
  with fabric.rank_zero_first():
    dataset = dataloader.get_gsm8k_test_dataset(config, tokenizer)

  # Interleaved split: rank r gets indices r, r+W, r+2W, ...
  rank = fabric.global_rank
  world_size = fabric.world_size
  local_indices = list(range(rank, len(dataset), world_size))
  batch_size = config.loader.eval_batch_size

  os.makedirs(config.gsm8k.output_dir, exist_ok=True)
  rank_json_path = os.path.join(config.gsm8k.output_dir,
                                f'rank_{rank:04d}.json')
  records = []
  total_nfe = 0
  num_batches = 0
  for batch_start in trange(
      0, len(local_indices), batch_size,
      desc=f'Rank {rank} sampling',
      disable=rank != 0):
    batch_indices = local_indices[
      batch_start:batch_start + batch_size]
    B = len(batch_indices)

    all_ids = [torch.tensor(dataset[idx]['input_ids'])
               for idx in batch_indices]
    padded, lengths = _pad_prefix_batch(
      all_ids, fabric.device)

    samples, metadata = model.generate_samples(
      num_samples=B,
      num_steps=config.sampling.steps,
      prefix_tokens=padded,
      prefix_lengths=lengths,
      return_metadata=True)
    total_nfe += metadata['nfe']
    num_batches += 1

    for i, idx in enumerate(batch_indices):
      prompt_len = int(lengths[i])
      generated_ids = samples[i, prompt_len:].cpu().tolist()

      response = tokenizer.decode(
        generated_ids, skip_special_tokens=True)
      records.append({
        'prompt': dataset[idx]['prompt'],
        'response_ground_truth':
          dataset[idx]['response_ground_truth'],
        'response': response,
        # Forwards this record actually cost. Carried per record rather than
        # per rank so it survives the merge below, and because an adaptive
        # remasking strategy spends a different amount on every batch.
        'nfe': int(metadata['nfe']),
      })
    # Cleanup memory for the next sampling batch
    del samples, metadata
    gc.collect()

  avg_nfe = total_nfe / max(num_batches, 1)
  logger.info(f'Rank {rank}: average NFE per batch: {avg_nfe:.1f}')

  with open(rank_json_path, 'w') as f:
    json.dump(records, f, indent=2)
  logger.info(f'Rank {rank}: saved {len(records)} records to {rank_json_path}')

  fabric.barrier()

  if fabric.global_rank != 0:
    return None

  all_records = []
  for r in range(world_size):
    path = os.path.join(config.gsm8k.output_dir, f'rank_{r:04d}.json')
    with open(path) as f:
      all_records.extend(json.load(f))
  return all_records


@torch.no_grad()
def _gsm8k_eval(diffusion_model, config, logger, tokenizer):
  logger.info('Starting GSM8K eval.')
  fabric = Fabric(
    accelerator=config.trainer.accelerator,
    devices=config.trainer.devices,
    num_nodes=config.trainer.num_nodes)
  fabric.launch()

  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)

  if config.gsm8k.input_file is not None:
    if fabric.global_rank == 0:
      samples = _load_generated_gsm8k(config, logger)
    else:
      samples = None
  else:
    samples = _sample_gsm8k(
      diffusion_model, config, logger, tokenizer, fabric)
  if fabric.global_rank == 0:
    logger.info(f'Total records evaluated: {len(samples)}')
    per_sample_correct = np.array([
      int(sandbox_gsm8k.evaluate_samples(
        rec['response'],
        rec['response_ground_truth'],
        timeout_s=config.gsm8k.timeout))
      for rec in samples], dtype=np.int32)
    correct = int(per_sample_correct.sum())
    # Optionally: bootstrapped 95% CI (test-set bootstrap, percentile method).
    # When bootstrap_size > 1 the reported `accuracy` is the bootstrap mean
    # (per Raschka's recommendation) with 95% CI from the 2.5/97.5 percentiles
    # of the bootstrap distribution. `accuracy_se` is the std of the bootstrap
    # means (= bootstrap-estimated standard error of the accuracy).
    if config.gsm8k.bootstrap_size > 1:
      rng = np.random.default_rng(config.seed)
      idx = rng.integers(0, len(samples),
                         size=(config.gsm8k.bootstrap_size, len(samples)))
      boot_acc = per_sample_correct[idx].mean(axis=1)
      acc = float(boot_acc.mean())
      acc_se = float(boot_acc.std(ddof=1))
      acc_ci_lo = float(np.percentile(boot_acc, 2.5))
      acc_ci_hi = float(np.percentile(boot_acc, 97.5))
      logger.info(
        f'GSM8K accuracy: {acc * 100:.2f}% '
        f'[{acc_ci_lo * 100:.2f}%, {acc_ci_hi * 100:.2f}%] (95% CI) '
        f'({correct}/{len(samples)}, bootstrap N={config.gsm8k.bootstrap_size})')
    else:
      acc = correct / len(samples)
      acc_se = 0.0
      acc_ci_lo = acc
      acc_ci_hi = acc
      logger.info(f'GSM8K accuracy: {acc * 100:.2f}% ({correct}/{len(samples)})')

    os.makedirs(config.gsm8k.output_dir, exist_ok=True)
    merged_path = os.path.join(config.gsm8k.output_dir, 'results.json')
    # `sampling_steps` is only the budget for the fixed-schedule strategies; under
    # an adaptive one it is ignored entirely. Record the realized forward count
    # beside it -- that is the number two runs can be compared on.
    nfes = [rec['nfe'] for rec in samples if 'nfe' in rec]
    strategy = config.sampling.get('remasking_strategy', 'random')
    results = {'accuracy': acc,
               'accuracy_se': acc_se,
               'accuracy_ci_lo': acc_ci_lo,
               'accuracy_ci_hi': acc_ci_hi,
               'num_correct': correct,
               'num_total': len(samples),
               'sampling_steps': config.sampling.steps,
               'remasking_strategy': strategy,
               # Only the adaptive strategy is budgeted by it; recording it
               # elsewhere would put a knob in the row that did nothing.
               'dynamic_threshold': config.sampling.get(
                 'dynamic_threshold', None) if strategy ==
               'low_confidence_dynamic' else None,
               'avg_nfe': sum(nfes) / len(nfes) if nfes else None,
               'records': samples}
    if results['avg_nfe'] is not None:
      logger.info(
        f"GSM8K decode: {results['remasking_strategy']} at "
        f"{results['avg_nfe']:.1f} NFE per sample")
    if config.gsm8k.input_file is not None:
      results['input_file'] = config.gsm8k.input_file
    with open(merged_path, 'w') as f:
      json.dump(results, f, indent=2)
    logger.info(f'Results saved to {merged_path}')

  fabric.barrier()

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  model_assets.configure_model_weights(
    config, base_dir=hydra.utils.get_original_cwd())
  _configure_training_mode(config)
  if config.mode not in ('sample_eval', 'ppl_eval', 'gsm8k_eval'):
    # Restore before writing the current config, initializing the tracker,
    # or checking the resume checkpoint in _train.
    nirvana_snapshot.restore(
      os.fspath(config.checkpointing.save_dir),
      enabled=bool(config.checkpointing.get('snapshot', True)))
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'd3pm':
    diffusion_model = algo.D3PMAbsorb
  elif config.algo.name == 'sedd':
    diffusion_model = algo.SEDDAbsorb
  elif config.algo.name == 'duo':
    diffusion_model = algo.DUO
  elif config.algo.name == 'distillation':
    diffusion_model = algo.Distillation
  elif config.algo.name == 'ot-finetune':
    diffusion_model = algo.OptimalTransportFinetune
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'gsm8k_eval':
    _gsm8k_eval(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
