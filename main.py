import functools
import json
import os
import time

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import metrics
import nirvana_utils
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


class NirvanaSnapshot(L.Callback):
  """Publish the run directory to the Nirvana snapshot path every N steps.

  This is the write half of Nirvana restartability; the read half is the
  `nirvana_utils.copy_snapshot_to_out` call in `_train`, which has to happen
  before the resume path is resolved. Mirrors `yrDiT/train.py`, which pushes on
  the same schedule it writes its archival checkpoints on.

  `every_n_train_steps` should be a multiple of the `ModelCheckpoint` interval so
  that a push has a fresh `last.ckpt` to carry. Cost is one copy of the run
  directory plus a tar of it, so the interval also sets the I/O the job spends on
  being restartable -- and it bounds the work a preemption can throw away.

  The push is deliberately deferred by one batch. Lightning reorders callbacks to
  put `ModelCheckpoint` last, so at the end of step N this callback runs *before*
  the checkpoint for step N is written; pushing there would ship the previous
  interval's checkpoint and silently cost a whole interval of progress on every
  restart. Waiting for the next batch to start means the file on disk is the one
  the step just wrote.

  Rank zero only, and never fatal: a failed transfer must not take training
  down, it only means the next attempt restarts from an older state.
  """

  def __init__(self, out_dir, every_n_train_steps):
    self.out_dir = os.fspath(out_dir)
    self.every_n_train_steps = int(every_n_train_steps)
    self._last_pushed_step = -1
    self._push_pending = False

  def _push(self, trainer, reason):
    if not trainer.is_global_zero:
      return
    step = int(trainer.global_step)
    if step == self._last_pushed_step:
      return
    self._last_pushed_step = step
    print(f'[nirvana] snapshot at step {step} ({reason})', flush=True)
    try:
      nirvana_utils.copy_out_to_snapshot(self.out_dir)
    except Exception as e:  # pragma: no cover - transfer is best-effort
      print(f'[nirvana] snapshot failed: {e}', flush=True)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self.every_n_train_steps <= 0:
      return
    step = int(trainer.global_step)
    if step > 0 and step % self.every_n_train_steps == 0:
      # `ModelCheckpoint` has not written this step's file yet; push next batch.
      self._push_pending = True

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    if self._push_pending:
      self._push_pending = False
      self._push(trainer, 'periodic')

  def on_train_end(self, trainer, pl_module):
    # Write the final state ourselves rather than pushing whatever the last
    # interval happened to leave behind: a run that reaches `max_steps` should
    # leave its *final* weights in the snapshot. Collective, so every rank calls
    # it; only rank zero then transfers.
    try:
      trainer.save_checkpoint(
        os.path.join(self.out_dir, 'checkpoints', 'last.ckpt'))
    except Exception as e:  # pragma: no cover - best-effort
      print(f'[nirvana] final checkpoint failed: {e}', flush=True)
    self._push(trainer, 'train end')

  def on_exception(self, trainer, pl_module, exception):
    # A crash is exactly when the state is most worth keeping. Preemption by
    # SIGKILL cannot reach this hook -- that case is covered by the periodic
    # push, which is what bounds the loss.
    self._push(trainer, f'exception: {type(exception).__name__}')


class EmaExport(L.Callback):
  """Write an EMA-only `.safetensors` of the backbone every N steps.

  The full Lightning `last.ckpt` is ~4x larger than the weights (optimizer
  moments, EMA shadow, dataloader state) and is overwritten in place, so it
  answers "resume this run" but not "give me the teacher as it was at 500k
  steps". This is the second half of `yrDiT`'s scheme -- `latest.pt` for resume,
  `ema_<step>.pt` for archival -- with the archival file written in the format
  `algo._load_pretrained_backbone` reads: raw backbone keys, EMA values, the
  same layout as the released `jdeschena/duo2-cifar10` teachers. Such a file can
  be handed straight to `adversarial_distill.teacher_ckpt_path`.

  Rank zero only, and never fatal. The live parameters are never touched: the
  shadow params are zipped against the parameter order the EMA was built with
  (`backbone` then `noise`, cf. `TrainerBase._get_parameters`) rather than
  copied over the model, so an export cannot perturb training.
  """

  def __init__(self, dirpath, every_n_train_steps):
    self.dirpath = os.fspath(dirpath)
    self.every_n_train_steps = int(every_n_train_steps)
    self._last_saved_step = -1

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if self.every_n_train_steps <= 0 or not trainer.is_global_zero:
      return
    step = int(trainer.global_step)
    if step <= 0 or step % self.every_n_train_steps != 0:
      return
    if step == self._last_saved_step:
      return
    self._last_saved_step = step
    try:
      self._export(pl_module, step)
    except Exception as e:  # pragma: no cover - archival is best-effort
      print(f'[ema-export] failed at step {step}: {e}', flush=True)

  def _export(self, pl_module, step):
    import safetensors.torch
    if pl_module.ema is None:
      raise RuntimeError('training.ema is off, so there is no EMA to export')
    # Same order `ExponentialMovingAverage` was constructed with, so the
    # positional shadow params can be given names.
    names = ([('backbone', n) for n, _ in pl_module.backbone.named_parameters()]
             + [('noise', n) for n, _ in pl_module.noise.named_parameters()])
    shadow = list(pl_module.ema.shadow_params)
    if len(names) != len(shadow):
      raise RuntimeError(
        f'EMA shadow params ({len(shadow)}) do not line up with the '
        f'parameters they were built from ({len(names)})')
    ema_params = {name: tensor
                  for (module, name), tensor in zip(names, shadow)
                  if module == 'backbone'}
    # Start from the live state dict so buffers come along, then overwrite the
    # parameters with their EMA values. `clone()` because safetensors rejects
    # tensors that share storage.
    state_dict = {k: v.detach().cpu().contiguous().clone()
                  for k, v in pl_module.backbone.state_dict().items()}
    missing = [k for k in ema_params if k not in state_dict]
    if missing:
      raise RuntimeError(f'EMA names absent from the backbone: {missing[:4]}')
    state_dict.update({k: v.detach().cpu().contiguous().clone()
                       for k, v in ema_params.items()})
    os.makedirs(self.dirpath, exist_ok=True)
    path = os.path.join(self.dirpath, f'ema_{step:08d}.safetensors')
    safetensors.torch.save_file(
      state_dict, path,
      metadata={'source': 'ema.shadow_params', 'step': str(step)})
    print(f'[ema-export] wrote {len(state_dict)} tensors to {path}',
          flush=True)


class ThroughputMonitor(L.Callback):
  """Log optimizer steps/sec and the projected hours left to `max_steps`.

  `trainer/loss` alone cannot distinguish "training normally" from "training at
  a third of the speed because the machine is contended", which on a multi-day
  preemptible run is the difference between finishing and not. Timed over
  `trainer.log_every_n_steps` windows, rank-local (no collective), and logged
  through the module so it reaches whichever tracker is configured.
  """

  def __init__(self):
    self._window_start = None
    self._window_steps = 0

  def on_train_start(self, trainer, pl_module):
    self._window_start = time.monotonic()
    self._window_steps = 0

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    window = max(1, int(trainer.log_every_n_steps))
    self._window_steps += 1
    if self._window_steps < window or self._window_start is None:
      return
    elapsed = time.monotonic() - self._window_start
    self._window_start = time.monotonic()
    self._window_steps = 0
    if elapsed <= 0:
      return
    steps_per_sec = window / elapsed
    metrics_to_log = {'trainer/steps_per_sec': steps_per_sec}
    max_steps = int(getattr(trainer, 'max_steps', -1) or -1)
    if max_steps > 0:
      remaining = max(0, max_steps - int(trainer.global_step))
      metrics_to_log['trainer/eta_hours'] = remaining / steps_per_sec / 3600.0
    pl_module.log_dict(metrics_to_log, on_step=True, on_epoch=False,
                       prog_bar=False, rank_zero_only=True, sync_dist=False)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    model = diffusion_model(config, tokenizer=tokenizer).to('cuda')
    return model

  if not config.eval.checkpoint_path:
    # No Lightning checkpoint: the weights must come from somewhere else, and
    # `adversarial_distill.teacher_ckpt_path` is that somewhere -- this is how
    # the *teacher* baseline is evaluated (`scripts/fid_cifar10_teacher.sh`),
    # since a downloaded `.safetensors` denoiser is not a Lightning checkpoint.
    assert config.adversarial_distill.get('teacher_ckpt_path', ''), (
      'eval.checkpoint_path is empty and no '
      'adversarial_distill.teacher_ckpt_path was given, so there are no '
      'weights to evaluate.')
    return diffusion_model(config, tokenizer=tokenizer).to('cuda')

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
def _print_batch(train_ds, valid_ds, tokenizer, k=64, modality='text'):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    if modality == 'image':
      # `decode` returns an image tensor, not a string: print the numbers and
      # the range instead, which is what one actually checks here (pixels must
      # be int64 in [0, 255], with no stray mask/special ids).
      ids = batch['input_ids']
      print(f'First {k} pixel ids:', first.tolist())
      print(f'Last {k} pixel ids:', last.tolist())
      print(f'dtype={ids.dtype} min={int(ids.min())} max={int(ids.max())}')
      continue
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


def _save_image_grid(images, path, nrow=None):
  """Write uint8 `(B, 3, H, W)` images to `path` as a single PNG grid."""
  import torchvision
  n = images.shape[0]
  nrow = int(max(1, n ** 0.5)) if nrow is None else nrow
  grid = torchvision.utils.make_grid(images.cpu(), nrow=nrow, padding=2)
  torchvision.io.write_png(grid, os.fspath(path))
  return path


def _generate_image_samples(diffusion_model, config, logger, tokenizer):
  """`mode=sample_eval` for `data.modality=image`.

  Generative perplexity and GM both score samples with a GPT-2-family language
  model, so neither exists here. What replaces them is a PNG grid (the thing
  one actually looks at) plus, optionally, FID against the validation split --
  the cheap version. For the paper's number use `mode=fid_eval`, which follows
  the 50 000-sample protocol against the train split.
  """
  logger.info('Starting Sample Eval (image).')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.sample_entropy.reset()
  model.metrics.pred_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  fid = None
  if config.eval.get('compute_fid', False):
    fid = metrics.build_fid(device=model.device)
    if fid is None:
      logger.warning('eval.compute_fid is set but FID is unavailable.')
    else:
      _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True)
      seen, budget = 0, config.eval.get('fid_num_samples_val', 10000)
      for batch in valid_ds:
        real = tokenizer.batch_decode(batch['input_ids'][:budget - seen])
        fid.update(real.to(model.device), real=True)
        seen += real.shape[0]
        if seen >= budget:
          break
      logger.info('FID reference: %d real images.', seen)

  model._eval_mode()
  last = None
  num_generated = 0
  for _ in range(config.sampling.num_sample_batches):
    samples = model.generate_samples(
      num_samples=config.loader.eval_batch_size,
      num_steps=config.sampling.steps)
    model.metrics.record_entropy(samples)
    last = tokenizer.batch_decode(samples)
    num_generated += last.shape[0]
    if fid is not None:
      fid.update(last.to(model.device), real=False)
  model._train_mode()

  entropy = model.metrics.sample_entropy.compute().item()
  pred_entropy = 0.
  print('Sample entropy:', entropy)
  if float(model.metrics.pred_entropy.weight) > 0:
    pred_entropy = model.metrics.pred_entropy.compute().item()
    print('Model predictive entropy:', pred_entropy)
  fid_value = None
  if fid is not None:
    fid_value = float(fid.compute())
    print('FID (vs. validation split):', fid_value)

  grid_path = config.eval.get('image_grid_path', None)
  if last is not None and grid_path:
    _save_image_grid(last[:config.sampling.num_sample_log], grid_path)
    print('Sample grid saved at:', grid_path)

  samples_path = config.eval.generated_samples_path
  with fsspec.open(samples_path, 'w') as f:
    json.dump({'entropy': entropy,
               'pred_entropy': pred_entropy,
               'fid': fid_value,
               'num_generated': num_generated,
               'sampling_steps': config.sampling.steps,
               'image_grid_path': str(grid_path) if grid_path else None}, f,
              indent=4)
  print('Report saved at:', samples_path)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  if config.data.get('modality', 'text') == 'image':
    return _generate_image_samples(
      diffusion_model, config, logger, tokenizer)
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


def _eval_fid(diffusion_model, config, logger, tokenizer):
  """`mode=fid_eval` -- the paper's CIFAR-10 protocol (D-MMD Table 1).

  Ported from DUO's `_eval_fid` (`~/duo/main.py`) with one change: it is
  **unconditional**. DUO generates with `labels` matching each real batch;
  here the backbone falls back to the checkpoint's classifier-free-guidance
  null class (see `models.unet.UNet.forward`), which is what D-MMD Sec. 6.1
  reports, so no label plumbing is needed through the sampler.

  `eval.fid_num_samples` (default 50 000, the standard protocol) generated
  images are compared against the same number of **train** images -- FID
  against train is the convention these papers follow. The work is split over
  `trainer.devices` ranks via Fabric; torchmetrics reduces the Inception
  statistics across ranks inside `compute()`, so every rank must reach it.

  The body runs inside `fabric.launch(fn)` rather than after a bare
  `fabric.launch()`: with more than one device Lightning's launcher requires a
  function to run in each process, and errors out otherwise.
  """
  import datetime

  from lightning.fabric import Fabric
  from lightning.fabric.strategies import DDPStrategy

  # A long process-group timeout, not NCCL's 30-minute default: the ranks only
  # meet at the barrier after the whole sampling loop, and at high
  # `sampling.steps` with 50 000 samples that loop runs for hours. Any drift
  # between ranks is then charged against the timeout, and a rank that merely
  # finished early would otherwise tear the job down.
  fabric = Fabric(accelerator=config.trainer.accelerator,
                  devices=config.trainer.devices,
                  num_nodes=config.trainer.num_nodes,
                  strategy=DDPStrategy(
                    timeout=datetime.timedelta(hours=12)))
  fabric.launch(functools.partial(
    _eval_fid_worker, diffusion_model=diffusion_model, config=config,
    logger=logger, tokenizer=tokenizer))


def _eval_fid_worker(fabric, diffusion_model, config, logger, tokenizer):
  """One rank's share of `mode=fid_eval`. See `_eval_fid`."""
  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)
  print(f'(Rank {fabric.global_rank}): seed: {seed}', flush=True)

  assert config.data.get('modality', 'text') == 'image', (
    'mode=fid_eval requires data.modality=image (e.g. data=cifar10).')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  fid = metrics.build_fid(device=fabric.device)
  assert fid is not None, (
    'mode=fid_eval needs torchmetrics image metrics; install torch-fidelity.')

  num_samples = int(config.eval.get('fid_num_samples', 50000))
  batch_size = config.loader.eval_batch_size
  world = fabric.world_size
  # Round the per-rank share up to whole batches so every rank runs the same
  # number of iterations -- an uneven count deadlocks the collective in
  # `compute()`.
  per_rank_batches = -(-(-(-num_samples // world) // batch_size))
  per_rank = per_rank_batches * batch_size
  if fabric.global_rank == 0:
    logger.info(
      'FID: %d images/rank x %d rank(s) = %d generated (requested %d), '
      'at sampling.steps=%s.',
      per_rank, world, per_rank * world, num_samples, config.sampling.steps)
    if per_rank * world < 10000:
      logger.warning(
        'Fewer than 10k samples: FID is strongly biased upward at this size '
        'and is only meaningful as a smoke test.')

  # --- Real images: the train split, un-augmented unless asked otherwise. ---
  real_set = dataloader.get_dataset(
    config.data.train, tokenizer, wrap=config.data.wrap, mode='train',
    cache_dir=config.data.cache_dir,
    insert_eos=config.data.insert_train_eos,
    block_size=config.model.length, streaming=config.data.streaming,
    num_proc=config.loader.num_workers, config=config,
    augment=config.eval.get('fid_augment_real', False))
  # Deliberately NOT `config.loader.num_workers`: that defaults to the machine's
  # full CPU count (255 here), and every worker is a fork holding its own copy of
  # the in-memory CIFAR-10 arrays. Multiplied by the number of ranks that is
  # >1000 processes and enough resident memory to get the rank OOM-killed --
  # which then surfaces as an NCCL watchdog timeout on the *other* ranks. This
  # pass reads at most `per_rank` images once, sequentially, so a couple of
  # workers already hide the decode latency behind the sampler.
  real_workers = min(4, config.loader.num_workers)
  real_loader = torch.utils.data.DataLoader(
    real_set, batch_size=batch_size, num_workers=real_workers,
    pin_memory=config.loader.pin_memory,
    sampler=torch.utils.data.distributed.DistributedSampler(
      real_set, num_replicas=world, rank=fabric.global_rank, shuffle=False))
  real_seen = 0
  for batch in real_loader:
    if real_seen >= per_rank:
      break
    images = tokenizer.batch_decode(
      batch['input_ids'][:per_rank - real_seen]).to(fabric.device)
    fid.update(images, real=True)
    real_seen += images.shape[0]

  # --- Generated images. ---
  for i in range(per_rank_batches):
    samples = model.generate_samples(
      num_samples=batch_size, num_steps=config.sampling.steps)
    fid.update(
      tokenizer.batch_decode(samples).to(fabric.device), real=False)
    if fabric.global_rank == 0 and (i + 1) % 10 == 0:
      logger.info('  generated %d / %d per-rank images.',
                  (i + 1) * batch_size, per_rank)
    if i == 0 and fabric.global_rank == 0:
      grid_path = config.eval.get('image_grid_path', None)
      if grid_path:
        _save_image_grid(
          tokenizer.batch_decode(samples)[:config.sampling.num_sample_log],
          grid_path)

  fabric.barrier()
  logger.info('Done sampling. Computing FID & IS...')
  # Collectives: every rank must call these.
  fid_value = float(fid.compute())
  is_mean, is_std = fid.compute_inception_score()

  if fabric.global_rank == 0:
    print(f'FID: {fid_value}')
    print(f'IS: {float(is_mean)} +/- {float(is_std)}')
    report_path = config.eval.get('fid_report_path', None)
    if report_path:
      with fsspec.open(report_path, 'w') as f:
        json.dump({'fid': fid_value,
                   'inception_score_mean': float(is_mean),
                   'inception_score_std': float(is_std),
                   'num_generated': per_rank * world,
                   'num_real': real_seen * world,
                   'sampling_steps': config.sampling.steps,
                   'sampling_predictor': config.sampling.predictor,
                   'checkpoint_path': config.eval.checkpoint_path,
                   'teacher_ckpt_path': config.adversarial_distill.get(
                     'teacher_ckpt_path', '')}, f, indent=4)
      print('FID report saved at:', report_path)
  fabric.barrier()


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

  # Nirvana restart: pull the preempted attempt's state back into `save_dir`
  # *before* the resume path below is resolved, and before the underdeep logger
  # looks for `underdeep_metadata.pt` there, so a restart both resumes the
  # optimizer state and keeps writing to the same tracker run. Under Lightning's
  # subprocess launcher this runs in the parent, which spawns the ranks only
  # inside `trainer.fit()`, i.e. after the restore has finished; off Nirvana it
  # is a no-op.
  snapshot_every = int(config.checkpointing.get(
    'nirvana_snapshot_every_n_steps', 0) or 0)
  if snapshot_every > 0 and os.environ.get('LOCAL_RANK', '0') == '0':
    nirvana_utils.copy_snapshot_to_out(
      os.fspath(config.checkpointing.save_dir))

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming from checkpoint: {ckpt_path}')
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
  callbacks.append(ThroughputMonitor())
  ema_export_every = int(config.checkpointing.get(
    'ema_export_every_n_steps', 0) or 0)
  if ema_export_every > 0:
    callbacks.append(EmaExport(
      os.path.join(os.fspath(config.checkpointing.save_dir), 'checkpoints'),
      ema_export_every))
  # Ordering relative to `ModelCheckpoint` is not up to us -- Lightning moves
  # checkpoint callbacks to the end -- which is why the push is deferred by a
  # batch inside the callback rather than arranged here.
  if snapshot_every > 0:
    callbacks.append(NirvanaSnapshot(
      os.fspath(config.checkpointing.save_dir), snapshot_every))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer,
               modality=config.data.get('modality', 'text'))

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
  elif config.algo.name == 'mdlm_rl':
    diffusion_model = algo.MDLM_RL
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
  elif config.mode == 'fid_eval':
    _eval_fid(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
