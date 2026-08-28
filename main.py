import collections
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
import nirvana_snapshot
import owt_pareto_plot
import pareto_temperature
import utils
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


def _warn_if_resume_unreachable(config, logger):
  """Say so when resume is configured but nothing will write the file it wants.

  `resume_ckpt_path` defaults to `checkpoints/last.ckpt`, which only the
  `checkpoint_every_n_steps` callback writes, and only when `save_last` is on.
  Turn `save_last` off -- the shipped TinyGSM scripts do, and it is an easy
  thing to copy -- and resume degrades in the worst possible way: the job
  restores its snapshot, finds no checkpoint, starts from step 0, and reports
  nothing wrong. Numbered `step=*.ckpt` files may be sitting right beside the
  path it looked for.

  A warning rather than an error: wanting checkpoints without wanting resume is
  a legitimate configuration, and this runs on every start including the ones
  that never get preempted.
  """
  if not config.checkpointing.get('resume_from_ckpt', False):
    return
  wanted = config.checkpointing.get('resume_ckpt_path', None)
  if not wanted or not str(wanted).endswith('last.ckpt'):
    return
  callbacks = config.get('callbacks', None) or {}
  every_n = callbacks.get('checkpoint_every_n_steps', None) or {}
  monitor = callbacks.get('checkpoint_monitor', None) or {}
  if every_n.get('save_last', False) or monitor.get('save_last', False):
    return
  logger.warning(
    f'resume_from_ckpt is on and resume_ckpt_path is {wanted}, but no '
    'checkpoint callback has save_last=True, so that file will never be '
    'written and every restart will silently begin from step 0. Set '
    'callbacks.checkpoint_every_n_steps.save_last=True, or point '
    'checkpointing.resume_ckpt_path at a file something does write.')


class SnapshotSync(L.Callback):
  """Push the run directory to the snapshot store after validation.

  `PeriodicStepCheckpoint` pushes its own writes, but Lightning's
  `ModelCheckpoint` callbacks save on validation and know nothing about
  snapshots. `on_validation_end` runs after they have written, which is the
  first hook where the new file is actually on disk -- `on_save_checkpoint`
  fires *before* the write and would push the previous state.
  """

  def __init__(self, snapshot_dir, enabled=True):
    self.snapshot_dir = os.fspath(snapshot_dir)
    self.enabled = bool(enabled)

  def on_validation_end(self, trainer, pl_module):
    if trainer.sanity_checking or not trainer.is_global_zero:
      return
    nirvana_snapshot.save(self.snapshot_dir, enabled=self.enabled,
                          reason=f'validation step={trainer.global_step}')


class PeriodicStepCheckpoint(L.Callback):
  """Save deterministic checkpoints every N completed train batches."""

  def __init__(self, dirpath, every_n_train_steps, snapshot_dir=None,
               snapshot=False):
    self.dirpath = os.fspath(dirpath)
    self.every_n_train_steps = int(every_n_train_steps)
    self._completed_train_batches = 0
    self._last_saved_step = 0
    # The snapshot covers the whole run directory, not just `dirpath`, so that
    # a restored job also gets its logs and tracker metadata back.
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
                            reason=f'periodic step={save_step}')


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


def _load_student_weights(model, config, logger, path=None):
  """Put a trained student's weights into a model built from the HF teacher.

  `algo.backbone=hf_dit` builds the architecture from `eval.checkpoint_path` --
  an HF hub id -- so that path cannot also carry the trained student. This
  loads the Lightning `.ckpt` written by training on top of it.

  Which weights: the EMA ones by default, because those are what every number
  the run reported was measured on. A checkpoint stores BOTH -- `state_dict`
  holds the raw live parameters, `ema` holds the shadow copy -- and reading
  `state_dict` alone silently evaluates a different model from the one the
  training curves describe. `eval.disable_ema=True` asks for the raw weights.

  Returns a short label naming what was loaded, for the figure and the JSON.

  `path` overrides `eval.student_ckpt_path`, which is how a multi-checkpoint
  sweep walks a list: the architecture is built once and only the weights are
  swapped, since every checkpoint in a sweep shares the `hf_dit` backbone that
  `eval.checkpoint_path` names.
  """
  if path is None:
    path = config.eval.get('student_ckpt_path', '') or ''
  if not path:
    logger.info('No eval.student_ckpt_path: evaluating '
                f'{config.eval.checkpoint_path} as built.')
    return f'{config.eval.checkpoint_path} (no student ckpt)'
  assert utils.fsspec_exists(path), f'no such checkpoint: {path}'
  # Training wrote these on a newer `tokenizers` than this env pins, and every
  # checkpoint embeds a pickled tokenizer under `hyper_parameters`.
  with utils.tolerant_checkpoint_load():
    checkpoint = torch.load(path, map_location='cpu')
  state_dict = collections.OrderedDict(
    (k, v) for k, v in checkpoint['state_dict'].items()
    # A distillation checkpoint carries the frozen teacher and the fake critic
    # alongside the student; only the student is being evaluated.
    if not k.startswith(('teacher.', 'fake_model.')))
  use_ema = not config.eval.disable_ema and 'ema' in checkpoint
  if use_ema:
    model._apply_checkpoint_ema_to_backbone_state(state_dict, checkpoint)
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if missing:
    logger.warning(f'{len(missing)} missing keys, first: {missing[:3]}')
  if unexpected:
    logger.warning(f'{len(unexpected)} unexpected keys, '
                   f'first: {unexpected[:3]}')
  # `_eval_mode` would otherwise copy this freshly-built model's own shadow
  # params over the weights just loaded. The EMA choice is already baked in
  # above, so take that path out of the picture entirely.
  model.ema = None
  step = checkpoint.get('global_step', '?')
  which = 'ema' if use_ema else 'raw'
  logger.info(f'Loaded {which} weights from {path} (global_step={step}).')
  return f'{os.path.basename(path)} @ step {step} ({which})'


def _init_distributed():
  """Join the torchrun process group, if this process was launched by one.

  `mode=pareto_eval` runs no Lightning trainer -- the sweep is a plain loop --
  so nothing else would set this up. All distribution has to buy here is that
  the ranks split the samples and the metrics all-reduce; there is no gradient
  and so no need for DDP to wrap the model.

  Returns `(rank, world_size, local_rank)`, and `(0, 1, 0)` for an ordinary
  single-process run.
  """
  world_size = int(os.environ.get('WORLD_SIZE', 1))
  if world_size <= 1:
    return 0, 1, 0
  local_rank = int(os.environ.get('LOCAL_RANK', 0))
  torch.cuda.set_device(local_rank)
  if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='nccl')
  return torch.distributed.get_rank(), world_size, local_rank


def _pareto_checkpoints(config):
  """The checkpoints this sweep covers, in the order they will be drawn.

  `eval.student_ckpt_paths` wins whenever it is set; `eval.student_ckpt_path`
  is the single-checkpoint spelling and stays the fallback, so every command
  written before the list existed still means what it meant. An empty string
  is a legal ENTRY, not a terminator: it names the model as
  `eval.checkpoint_path` builds it, i.e. the teacher, which is the baseline the
  students in the same list are read against.
  """
  paths = list(config.eval.get('student_ckpt_paths', None) or [])
  if paths:
    return [str(p) for p in paths]
  return [str(config.eval.get('student_ckpt_path', '') or '')]


def _short_labels(paths):
  """Compact, unique legend names for a list of checkpoint paths.

  The basename's stem is what a reader recognises -- `step=00014000` -- and for
  the usual case, one run's checkpoint directory, that is the whole answer. Two
  runs collide on it, so the name widens leftwards a path component at a time
  until the names separate, skipping any component every path shares:
  `checkpoints` is in all of them and discriminates nothing, so the pair reads
  `run_a/step=00014000` rather than `run_a/checkpoints/step=00014000`. An index
  is the last resort. Uniqueness matters beyond the legend, because these also
  name the tracker series.
  """
  parts = [[c for c in os.path.abspath(path).split(os.sep) if c] if path
           else ['teacher'] for path in paths]
  for piece in parts:
    # Only the file loses its extension; the directories keep their names.
    piece[-1] = os.path.splitext(piece[-1])[0]

  def render(depths):
    return ['/'.join(piece[-d] for d in sorted(depths, reverse=True)
                     if len(piece) >= d) for piece in parts]

  deepest = max(len(piece) for piece in parts)
  depths, depth = [1], 1
  labels = render(depths)
  while len(set(labels)) < len(labels) and depth < deepest:
    depth += 1
    # Only over the paths that reach this far: a short entry (`teacher`) would
    # otherwise make a shared component look like a distinguishing one.
    column = {piece[-depth] for piece in parts if len(piece) >= depth}
    if len(column) < 2:
      continue
    depths.append(depth)
    labels = render(depths)

  seen = collections.Counter()
  for i, label in enumerate(labels):
    seen[label] += 1
    if seen[label] > 1:
      labels[i] = f'{label}#{seen[label]}'
  return labels


def _pareto_eval(diffusion_model, config, logger, tokenizer):
  """Sweep checkpoints over sampling temperatures and step budgets."""
  rank, world_size, local_rank = _init_distributed()
  temperatures = config.eval.get('pareto_temperatures', None)
  budgets = config.eval.get('owt_pareto_steps', None)
  assert temperatures, 'set eval.pareto_temperatures, e.g. [0.7,1.0,2.0]'
  assert budgets, 'set eval.owt_pareto_steps, e.g. [8,16,32]'
  paths = _pareto_checkpoints(config)
  shorts = _short_labels(paths)
  if rank == 0:
    logger.info(f'Starting temperature Pareto eval over {len(paths)} '
                'checkpoint(s).'
                + (f' Sharded over {world_size} ranks.'
                   if world_size > 1 else ''))

  # Built once for the whole list: every checkpoint in a sweep is the same
  # architecture -- `eval.checkpoint_path` names it -- so only the weights need
  # swapping, and rebuilding per checkpoint would re-download the hub config
  # and re-allocate the model for nothing.
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model, config=config, tokenizer=tokenizer)
  model = model.to(f'cuda:{local_rank}')
  model.eval()
  # An empty entry means "the model as built". That is only true before a
  # student has been loaded over it, so when the baseline is not first in the
  # list, keep a pristine CPU copy to restore from. Built only when the list
  # actually asks for it, since it is a second copy of the weights.
  pristine = None
  if len(paths) > 1 and any(not p for p in paths):
    pristine = {k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}

  # One tracker run, not one per rank: `logger.force_new_run=true` would
  # otherwise open `world_size` runs and scatter the sweep across all of them.
  tracker = None
  if rank == 0:
    tracker = hydra.utils.instantiate(config.logger)
    try:
      tracker.log_hyperparams({
        'eval/temperatures': str(list(temperatures)),
        'eval/steps': str(list(budgets)),
        'eval/checkpoints': ', '.join(shorts),
        'eval/world_size': world_size})
    except Exception as e:  # pragma: no cover - tracker best-effort
      print(f'[pareto-temp] hyperparam log failed: {e}', flush=True)

  sweeps = []
  for index, (path, short) in enumerate(zip(paths, shorts)):
    if not path and pristine is not None:
      model.load_state_dict(pristine, strict=True)
    label = _load_student_weights(model, config, logger, path=path)
    if rank == 0 and len(paths) > 1:
      print(f'[pareto-temp] === checkpoint {index + 1}/{len(paths)}: '
            f'{short} -- {label} ===', flush=True)
    curves, samples = pareto_temperature.sweep(
      model,
      temperatures=temperatures,
      budgets=budgets,
      seeds=config.eval.get('owt_pareto_seeds', 1),
      num_batches=(config.eval.get('owt_pareto_num_batches', 0)
                   or config.sampling.num_sample_batches),
      batch_size=config.loader.eval_batch_size,
      max_length=config.model.length)
    sweeps.append({'label': label, 'short': short, 'path': path,
                   'curves': curves, 'samples': samples})

  if world_size > 1:
    # Hold the other ranks until rank 0 is done: tearing the group down while a
    # collective is outstanding is what turns a finished sweep into a hang.
    torch.distributed.barrier()
  if rank == 0:
    pareto_temperature.publish_many(
      sweeps, tracker,
      save_dir=os.fspath(config.checkpointing.save_dir),
      reference=owt_pareto_plot.load_reference(
        config.eval.get('owt_pareto_reference', None)))
    tracker.finalize('success')
  if world_size > 1:
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


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

  # Before the existence check below, not after: on a restarted preemptible
  # job the local disk is empty and this is what puts `last.ckpt` back on it.
  # Check first and the run starts from scratch with a full snapshot sitting
  # unused, silently losing every step of prior training.
  snapshot = bool(config.checkpointing.get('snapshot', True))
  snapshot_dir = os.fspath(config.checkpointing.save_dir)
  nirvana_snapshot.restore(snapshot_dir, enabled=snapshot)
  _warn_if_resume_unreachable(config, logger)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming from {ckpt_path}')
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
    if snapshot:
      callbacks.append(SnapshotSync(snapshot_dir, enabled=True))

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
  # `save_cfg` writes a file into the run dir; under torchrun every rank shares
  # that dir, so let one of them write it rather than eight racing on it.
  if int(os.environ.get('RANK', 0)) == 0:
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
  elif config.mode == 'pareto_eval':
    _pareto_eval(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
