import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils
import underdeep_logger  # noqa: F401  (registers `underdeep_logger.*` Hydra _target_)
from ipdb import set_trace as debug

torch.autograd.set_detect_anomaly(True)

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


class PeriodicStepCheckpoint(L.Callback):
  """Save deterministic checkpoints every N completed train batches."""

  def __init__(self, dirpath, every_n_train_steps):
    self.dirpath = os.fspath(dirpath)
    self.every_n_train_steps = int(every_n_train_steps)
    self._completed_train_batches = 0
    self._last_saved_step = 0

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


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    model = diffusion_model(config, tokenizer=tokenizer).to('cuda')
    return model
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    weights_only=False)


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


def _prepare_gm_data(model, config, logger, tokenizer):
  """Validation batches, decoded to text, to pair with the generated ones for
  the GPT-2 Gradient Moment -- or `[]` when GM is off or cannot be computed.

  GM (D-MMD arXiv:2603.20155 Eq. 13/14, `metrics.GradientMoment`) is a distance
  *between two distributions*: the reference model's loss-gradient is averaged
  over our samples (`g`) and over real data (`q`), and the metric is the squared
  norm of the difference. Generative perplexity needs no data, so
  `mode=sample_eval` never loaded any; this pulls one validation batch per
  generated batch from the same shuffled loader `mode=ppl_eval` uses
  (`valid_seed=config.seed`), which also makes both sides of a pair
  `loader.eval_batch_size` rows wide. Decoded to text because the reference
  model re-tokenizes with its own tokenizer, exactly as gen-ppl does.

  The number of pairs is `min(sampling.num_sample_batches, eval.gm_num_batches)`
  -- the same `M` the validation path uses -- and is logged and written to the
  report, since it is what the estimator's variance rides on.
  """
  if not config.eval.get('compute_gm', False):
    return []
  if model.metrics.gm is None:
    logger.warning(
      'eval.compute_gm is set but this model was built without a '
      'GradientMoment; skipping GM.')
    return []
  if config.sampling.semi_ar:
    # Same reason generative perplexity is skipped below: semi-AR samples are
    # full of <|endoftext|> and need post-processing before a reference model
    # can score them.
    logger.warning('GM is not computed for semi-AR sampling.')
    return []
  num_pairs = min(config.sampling.num_sample_batches,
                  config.eval.get('gm_num_batches', 4))
  if num_pairs < 2:
    # `GradientMoment.compute` needs two independent draws to form a cross-pair
    # (a single `d_m . d_m` would estimate the norm plus its own variance).
    logger.warning(
      'GM needs at least 2 paired (generated, data) batches but only %d are '
      'configured (min of sampling.num_sample_batches=%d and '
      'eval.gm_num_batches=%d); skipping GM.',
      num_pairs, config.sampling.num_sample_batches,
      config.eval.get('gm_num_batches', 4))
    return []
  model.metrics.gm.reset()
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  texts = []
  for batch in valid_ds:
    texts.append(valid_ds.tokenizer.batch_decode(batch['input_ids']))
    if len(texts) == num_pairs:
      break
  if len(texts) < 2:
    logger.warning(
      'The validation set yielded only %d batch(es); skipping GM.',
      len(texts))
    return []
  logger.info(
    'GM: %d paired (generated, data) batches of %d sequences '
    '(reference model: %s).',
    len(texts), config.loader.eval_batch_size,
    config.eval.get('gm_eval_model_name_or_path', 'gpt2'))
  return texts


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  model.metrics.pred_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  gm_data_texts = _prepare_gm_data(model, config, logger, tokenizer)
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  for _sb in range(config.sampling.num_sample_batches):
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
      # `device=` by keyword: the third positional parameter is `retokenize`, so
      # passing the device there left `device` at its 'cuda' default -- harmless
      # while the model is on a GPU (a torch.device is truthy, so retokenization
      # still happened) but a crash for a CPU eval.
      model.metrics.record_generative_perplexity(
        text_samples, config.model.length, device=model.device)
      all_samples.extend(list(text_samples))
      # GPT-2 GM: one (generated, data) pair per batch. Each pair contributes a
      # gradient difference `d_m`; the estimator in `GradientMoment.compute`
      # averages the `M (M - 1)` cross terms `d_i . d_j`, so this only needs the
      # pairing to be independent across batches, not within one.
      if _sb < len(gm_data_texts):
        model.metrics.gm.accumulate(
          text_samples, gm_data_texts[_sb], config.model.length,
          model.device)
  generative_ppl = 0.
  entropy = 0.
  pred_entropy = 0.
  gpt2_gm = None
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
    if float(model.metrics.pred_entropy.weight) > 0:
      pred_entropy = model.metrics.pred_entropy.compute().item()
      print('Model predictive entropy:', pred_entropy)
    if gm_data_texts:
      # Lower is better; 0 means the reference model cannot tell the samples
      # from the data. Unlike generative perplexity it is not improved by mode
      # collapse, which is the point of reporting it beside gen-ppl.
      gpt2_gm = model.metrics.gm.compute()
      print('GPT-2 gradient moment:', gpt2_gm)
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'pred_entropy': pred_entropy,
               'gpt2_gm': gpt2_gm,
               'gm_num_pairs': len(gm_data_texts),
               'generated_seqs': all_samples}, f, indent=4)
  print('Samples saved at:', samples_path)


def _noise_diversity_eval(diffusion_model, config, logger, tokenizer):
  """Sample the same batch under several input-noise draws and report how much
  the noise -- rather than the sampler -- is driving sample diversity."""
  logger.info('Starting Noise Diversity Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  if not model.has_noise_conditioning:
    logger.warning(
      'This model has no input-noise conditioning '
      '(adversarial_distill.noise_dim = 0), so the noise arm is a replay of '
      'one deterministic function: expect agree_noise = 1.0.')

  model._eval_mode()
  diversity = model.noise_diversity_report(
    num_steps=config.sampling.steps, return_variants=True)
  model._train_mode()

  print('Noise diversity:')
  for name, value in diversity.metrics.items():
    print(f'  {name}: {value:.6f}')
  # The same row under each noise draw, decoding randomness held fixed: what
  # `agree_noise` is measuring, in readable form.
  print('Row 0 under each noise draw:')
  for k, replicate in enumerate(diversity.variants):
    print(f'  [noise {k}] {replicate[0][:200]!r}')
  report_path = config.eval.noise_diversity_path
  with fsspec.open(report_path, 'w') as f:
    json.dump({'metrics': diversity.metrics,
               'noise_variants': diversity.variants}, f, indent=4)
  print('Noise diversity report saved at:', report_path)


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
  # wandb_logger = None
  # if config.get('wandb', None) is not None:
  #   wandb_logger = L.pytorch.loggers.WandbLogger(
  #     config=omegaconf.OmegaConf.to_object(config),
  #     **config.wandb)
  train_logger = hydra.utils.instantiate(config.logger)

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
        config.callbacks.checkpoint_every_n_steps.every_n_train_steps))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config,
      weights_only=False)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=train_logger,
    enable_progress_bar=False if config.debug else True)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM_with_AR
  elif config.algo.name == 'mdlm_dmmd':
    diffusion_model = algo.MDLM_with_DMMD
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
  elif config.mode == 'noise_diversity_eval':
    _noise_diversity_eval(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
