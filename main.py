import json
import os

import fsspec
import hydra
import lightning as L
from lightning.fabric import Fabric
import omegaconf
import rich.syntax
import rich.tree
import torch
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm, trange

import algo
import dataloader
import nirvana_utils
import underdeep_logger  # noqa: F401  (registers `underdeep_logger.*` as a Hydra `_target_`)
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


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
def _print_batch(config, train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    if config.data.modality == 'text':
      first = batch['input_ids'][0, :k]
      last = batch['input_ids'][0, -k:]
      print(f'First {k} tokens:', tokenizer.decode(first))
      print('ids:', first)
      print(f'Last {k} tokens:', tokenizer.decode(last))
      print('ids:', last)


def _build_loggers(config):
  """Loggers the `Trainer` streams metrics to, or None for no logger.

  Two independent sources: a `wandb` block (duo's original tracker) and the
  `/logger` config group (underdeep by default, `logger=tensorboard` for
  TensorBoard, `~logger` for neither). Both may be active at once; wandb is kept
  first because `trainer.logger` -- which `trainer_base` reaches for its
  wandb-only `log_table` calls -- is the first entry of the list.
  """
  loggers = []
  if config.get('wandb', None) is not None:
    loggers.append(L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb))
  if config.get('logger', None) is not None:
    loggers.append(hydra.utils.instantiate(config.logger))
  return loggers or None


@L.pytorch.utilities.rank_zero_only
def _restore_nirvana_snapshot(config):
  """Pull a preempted attempt's run directory back before anything reads it.

  Must run before the `resume_ckpt_path` check below, otherwise the restored
  `last.ckpt` (and the underdeep run id next to it) arrive too late to be used.
  Off Nirvana this is a no-op. Note that it only helps when the run directory is
  stable across attempts, i.e. when `hydra.run.dir` / `checkpointing.save_dir`
  is set explicitly rather than left at the default timestamped path.
  """
  snapshot_dir = config.checkpointing.get('nirvana_snapshot_dir', None)
  if snapshot_dir is None:
    return
  nirvana_utils.copy_snapshot_to_out(os.fspath(snapshot_dir))


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
  for _ in trange(config.sampling.num_sample_batches):
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
    logger.info(f'Generative perplexity: {generative_ppl}')
    logger.info(f'Sample entropy: {entropy}')
  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'generative_ppl': generative_ppl,
               'entropy': entropy,
               'generated_seqs': all_samples}, f, indent=4)
  logger.info(f'Samples saved at: {samples_path}',)

def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=_build_loggers(config))
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  _restore_nirvana_snapshot(config)

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

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(config, train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=_build_loggers(config))
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _eval_fid(diffusion_model, config, logger, tokenizer):
  logger.info('Preparing data and model for FID eval.')
  fabric = Fabric(accelerator=config.trainer.accelerator,
                  devices=config.trainer.devices,
                  num_nodes=config.trainer.num_nodes)
  
  fabric.launch()
  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)
  print(f'(Rank {fabric.global_rank}): seed: {seed}')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  assert config.data.train == 'cifar10', \
                    'FID eval only implemented for CIFAR-10'

  # Like in flow matching papers: FID against train -- un-flipped. The reference
  # is the data distribution itself, and `DiscreteCIFAR10(train=True)` otherwise
  # hands back randomly mirrored images, since `RandomHorizontalFlip` is a
  # *training* augmentation. Going through `get_dataloaders` would apply it.
  dataset = dataloader.DiscreteCIFAR10(
    config.data.cache_dir, train=True, augment=False)

  # How many images to *generate*. `null` means one per training image, i.e. the
  # full 50k that published CIFAR-10 FIDs are quoted at; a smaller number is a
  # cheap preview of a checkpoint, at the cost of an upward-biased score (see
  # the note printed at the end). The reference side stays at the full split
  # either way -- real images cost only an Inception forward, and holding the
  # reference fixed is what makes two runs at different sample counts
  # comparable to each other. `metrics.FID` does the same in-loop.
  num_samples = config.eval.get('fid_eval_num_samples', None)
  num_samples = len(dataset) if num_samples is None else int(num_samples)
  assert 0 < num_samples <= len(dataset), (
    f'eval.fid_eval_num_samples={num_samples} must lie in '
    f'(0, {len(dataset)}]')

  # Every rank must generate the same number of images, so that they reach the
  # metric sync below having done the same amount of work.
  batch_size = config.loader.eval_batch_size
  per_step = batch_size * fabric.world_size
  assert num_samples % per_step == 0, (
    f'{num_samples} generated samples do not divide evenly over '
    f'loader.eval_batch_size={batch_size} x world_size='
    f'{fabric.world_size} = {per_step} images per step. Pick an '
    f'eval.fid_eval_num_samples that is a multiple of {per_step}, or change '
    f'loader.eval_batch_size.')

  # A deterministic permutation, so that the subset is reproducible across runs
  # and identical on every rank -- `L.seed_everything` has already offset the
  # global RNG by the rank, hence the explicit generator on `config.seed`.
  # Shuffling (rather than taking the first N) keeps the class distribution of
  # the subset representative, which matters for class-conditional models:
  # `generate_samples` is handed the labels of the real batch.
  order = torch.randperm(
    len(dataset),
    generator=torch.Generator().manual_seed(config.seed))
  gen_dataset = torch.utils.data.Subset(
    dataset, order[:num_samples].tolist())

  sampler = DistributedSampler(
    gen_dataset,
    num_replicas=fabric.world_size,
    rank=fabric.global_rank,
    shuffle=False)

  loader = torch.utils.data.DataLoader(
    gen_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory)

  # The reference pass covers the whole split. Striding the indices by rank
  # partitions it exactly, where a `DistributedSampler` would pad the tail into
  # duplicates whenever the world size does not divide 50k.
  ref_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(
      dataset,
      range(fabric.global_rank, len(dataset), fabric.world_size)),
    batch_size=batch_size,
    shuffle=False,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory)

  fid_calculator = FrechetInceptionDistance(
    normalize=False).to(fabric.device)
  is_calculator = InceptionScore(
    normalize=False).to(fabric.device)

  desc = f'(Rank {fabric.global_rank}) Reference...'
  for batch in tqdm(ref_loader, desc=desc):
    # Reshape 1D seq -> 2D image
    real_samples = model.tokenizer.batch_decode(
      batch['input_ids']).to(fabric.device)
    fid_calculator.update(real_samples, real=True)

  desc = f'(Rank {fabric.global_rank}) Sampling...'
  for batch in tqdm(loader, desc=desc):
    # Generate images with labels matching the true data
    labels = batch['labels']

    gen_samples = model.generate_samples(
      num_samples=batch['input_ids'].shape[0],
      num_steps=config.sampling.steps,
      labels=labels)
    gen_samples = model.tokenizer.batch_decode(gen_samples)

    fid_calculator.update(gen_samples, real=False)
    is_calculator.update(gen_samples)

  fabric.barrier()
  logger.info('Done sampling. Computing FID & IS...')
  fid = fid_calculator.compute()
  incep_score = is_calculator.compute()

  if fabric.global_rank == 0:
    logger.info(f'FID: {fid}  ({num_samples} generated samples vs '
                f'{len(dataset)} reference images)')
    logger.info(f'IS: {incep_score}')
    if num_samples < len(dataset):
      logger.info(
        f'NOTE: FID is biased upward at {num_samples} samples and is not '
        f'comparable to published CIFAR-10 numbers, which are quoted at '
        f'{len(dataset)}. Leave eval.fid_eval_num_samples unset for the '
        f'comparable figure.')
  fabric.barrier()


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
  elif config.mode == 'fid_eval':
    _eval_fid(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()