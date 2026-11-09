import collections
import itertools
import json
import os
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

import dataloader
import metrics
import models
from model_assets import pretrained_kwargs
import utils
from ipdb import set_trace as debug


# `pareto/owt_step_*` plots perplexity against entropy, but the tracker indexes
# a series by an integer step. Entropy is carried at this fixed scale, so an x
# of 5062 reads as entropy 5.062.
_ENTROPY_X_SCALE = 1000


def _mean(values):
  return float(sum(values) / len(values)) if values else 0.0


def _std(values):
  """Sample standard deviation; 0 for fewer than two values."""
  if len(values) < 2:
    return 0.0
  mean = _mean(values)
  return float((sum((v - mean) ** 2 for v in values)
                / (len(values) - 1)) ** 0.5)


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  prior_loss: torch.FloatTensor
  num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - t 
    dalpha_t = - (1 - self.eps)
    return dalpha_t, alpha_t


def sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer,
    vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    mask_schedule_cfg = config.adversarial_distill.get('mask_schedule', {}) or {}
    if (hasattr(mask_schedule_cfg, 'get')
        and mask_schedule_cfg.get('mode', 'original') != 'original'
        and config.algo.name != 'mdlm'):
      raise ValueError('adversarial_distill.mask_schedule requires algo=mdlm')
    if hasattr(self.config.algo, 'ignore_bos'):
      self.ignore_bos = config.algo.ignore_bos
    else:
      self.ignore_bos = False
    if hasattr(self.config.algo, 'loss_type'):
      self.loss_type = config.algo.loss_type
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = self.config.sampling.predictor
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        **pretrained_kwargs(
          config.eval.checkpoint_path,
          getattr(config, 'model_weights_dir', None)),
        trust_remote_code=True)

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    self.sampling_temperature = float(
      self.config.sampling.get('temperature', 1.0) or 1.0)
    # Noise Schedule
    self.noise = LogLinear()

    self.metrics = metrics.Metrics(
      gen_ppl_eval_model_name_or_path=\
        self.config.eval.gen_ppl_eval_model_name_or_path,
      eval_ppl_batch_size=\
        self.config.eval.perplexity_batch_size,
      model_weights_dir=getattr(config, 'model_weights_dir', None))

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _validate_configuration(self):
    assert self.config.algo.backbone in {'dit', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def q_xt(self, x, alpha_t):
    raise NotImplementedError

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(),
                           self.noise.parameters())

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def _apply_checkpoint_ema_to_backbone_state(self, state_dict, checkpoint):
    """Overwrite the raw backbone weights in `state_dict` with the EMA ones.

    A checkpoint carries both: `state_dict` holds the live parameters and
    `ema.shadow_params` a parallel list in `backbone.named_parameters()` order.
    Every number a run reports is measured on the EMA copy (`_eval_mode`
    swaps it in), so an eval that reads `state_dict` alone quietly scores a
    different model from the one the curves describe.
    """
    ema_state = checkpoint.get('ema', None)
    if not (ema_state and 'shadow_params' in ema_state):
      return
    for (name, _), shadow in zip(
        self.backbone.named_parameters(), ema_state['shadow_params']):
      key = f'backbone.{name}'
      if key in state_dict:
        state_dict[key] = shadow.detach().clone().to(state_dict[key].dtype)

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed']
    # is 1 iteration behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps,
    # not the number of local steps, so we don't multiply with
    # self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

    if (self.config.data.train == 'tiny_gsm'
        and self.config.eval.get('gsm8k_at_validation', False)
        and self.config.eval.get('validate_before_train', False)
        and self.global_step == 0):
      from tinygsm_validation import log_validation_gsm8k
      self._eval_mode()
      try:
        log_validation_gsm8k(self, initial=True)
      finally:
        self._train_mode()

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _temper(self, log_p_x0):
    """Sharpen or flatten the denoiser's x0 prediction.

    Takes and returns LOG-probabilities, renormalised, so callers can keep
    treating the result as `p(x0 | x_t)`. At temperature 1 the tensor is handed
    straight back: the published sampler must stay bit-identical when the knob
    is untouched.

    Safe on the absorbing parameterisation's hard rows. `MDLM
    ._process_model_output` pins already-unmasked positions to `0` for the
    observed token and `neg_infinity` elsewhere; dividing that by any
    temperature in a sane range leaves it one-hot after renormalisation, so a
    committed token cannot be un-committed by tempering. The same holds for the
    mask token's suppressed column.
    """
    if self.sampling_temperature == 1.0:
      return log_p_x0
    return (log_p_x0 / self.sampling_temperature).log_softmax(dim=-1)

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    raise NotImplementedError

  def forward(self, xt, sigma):
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output = self.backbone(xt, sigma)
    return self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)

  def on_train_epoch_start(self):
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    current_accumulation_step = (
      batch_idx % self.trainer.accumulate_grad_batches)
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'],
                        current_accumulation_step,
                        train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_train_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)

  def on_validation_epoch_start(self):
    self.metrics.reset()
    # Gradient Moment compares the reference LM's gradient on our samples
    # against its gradient on REAL text, so a fresh batch of held-out rows has
    # to be collected alongside the usual validation.
    self._gm_ref_texts = []
    self._eval_mode()
    # Decided once per validation rather than per call: `global_step` does not
    # move during the loop, and re-deciding would print the skip line once per
    # validation batch.
    self._gen_eval_active = self._generative_eval_active()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    del batch_idx
    self._collect_gm_reference(batch)
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    return losses.loss

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    # The held-out NLL above is cheap and it is the loss curve, so it runs at
    # the trainer's cadence throughout. Everything below is the expensive half
    # -- sampling, gen_ppl, Gradient Moment, both Pareto sweeps -- and that is
    # what `eval.eval_step_range` / `eval.eval_at_steps` gate.
    if not getattr(self, '_gen_eval_active', True):
      self._train_mode()
      return
    if (self.config.data.train == 'tiny_gsm'
        and self.config.eval.get('gsm8k_at_validation', False)):
      try:
        from tinygsm_validation import log_validation_gsm8k
        log_validation_gsm8k(self)
      finally:
        self._train_mode()
      return
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      samples, text_samples = None, None
      # Gradient Moment scores a POOL of samples against a pool of real rows,
      # so unlike the per-batch metrics it needs them kept rather than
      # overwritten each iteration.
      gm_samples = []
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self.generate_samples(
          num_samples=self.config.loader.eval_batch_size)
        
        self.metrics.record_entropy(samples)
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if (metrics.GRADIENT_MOMENT_ENABLED
            and self.config.eval.get('owt_gm', False)):
          gm_samples.extend(text_samples)
        if self.config.eval.compute_generative_perplexity:
          self.metrics.record_generative_perplexity(
            text_samples, self.num_tokens, self.device)
      if text_samples is not None:
        if self.trainer.global_rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Log the last generated samples
          text_samples = text_samples[
            : self.config.sampling.num_sample_log]
          # Stable key + explicit `step`: a tracker turns repeated calls
          # under one title into a step-indexed series, so baking the step
          # into the title instead lands every validation as its own
          # unrelated table.
          self.trainer.logger.log_table(
            key='val/samples',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples],
            step=self.global_step)
        if self.config.eval.compute_generative_perplexity:
          self.log('val/gen_ppl',
                   self.metrics.gen_ppl.compute(),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          self.log('val/sample_entropy',
                   self.metrics.sample_entropy.compute(),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
      # Before the Pareto sweep, which resets `self.metrics` per budget.
      self._log_gradient_moment(gm_samples)
    # Deliberately last: the sweep reuses `self.metrics` and resets it per
    # budget, so it must not run before `val/gen_ppl` and `val/sample_entropy`
    # above have been read off. `on_validation_epoch_start` resets again, so
    # leaving it dirty here is harmless.
    try:
      self._log_owt_pareto()
    except Exception as e:
      # A diagnostic must never take the run down with it. The sweep samples at
      # budgets the main loop never uses (2 steps, say), so it can fail where
      # ordinary validation does not -- and it sits inside the validation of
      # runs measured in days.
      print(f'[owt-pareto] sweep failed, continuing: {e}', flush=True)
    try:
      self._log_owt_pareto_temperature()
    except Exception as e:
      print(f'[owt-pareto-temp] sweep failed, continuing: {e}', flush=True)
    self._train_mode()

  def _generative_eval_active(self):
    """Whether this validation should pay for the generative evaluation.

    Sampling, generative perplexity, Gradient Moment and both Pareto sweeps
    cost minutes per validation; the held-out NLL costs seconds. So the
    trainer's `val_check_interval` is really two cadences fused into one, and a
    run that wants the expensive half often has to buy it everywhere.

    These two knobs separate them. When the interesting part of a run is
    already known -- which checkpoints matter, or which stretch of training --
    restricting the expensive half to it is what makes a dense
    `val_check_interval` affordable: an evaluation every 50 steps inside the
    window, and none outside it.

      * `eval.eval_step_range`: `[start, end]`, inclusive, in training steps.
        Either bound may be null for an open end.
      * `eval.eval_at_steps`: the exact steps to evaluate at. A step only fires
        if a validation actually lands on it, i.e. entries have to be multiples
        of `val_check_interval / accumulate_grad_batches`; ones that are not
        are reported once rather than silently never firing.

    Both read `self.global_step`, which counts OPTIMIZER steps -- one per
    global batch, whatever `accumulate_grad_batches` and the device count make
    that -- so they are in the same units as `trainer.max_steps`, as
    `ModelCheckpoint.every_n_train_steps`, and as the x-axis of every metric
    the run logs. They are NOT in micro-batches, which is what
    `val_check_interval` counts.

    Both are null by default and both being null is the old behaviour exactly.
    Set together they must BOTH admit the step. Never applied outside `fit`:
    `mode=ppl_eval` and friends run one validation at step 0, and a window
    written for a training run would silently turn them into no-ops.
    """
    config = self.config.eval
    window = config.get('eval_step_range', None)
    at_steps = config.get('eval_at_steps', None)
    if window is None and at_steps is None:
      return True
    state_fn = getattr(getattr(self.trainer, 'state', None), 'fn', None)
    if str(getattr(state_fn, 'value', state_fn)) != 'fit':
      return True

    step = int(self.global_step)
    reason = None
    if window is not None:
      window = list(window)
      assert len(window) == 2, (
        f'eval.eval_step_range must be [start, end], got {window!r}')
      start, end = window
      if start is not None and step < int(start):
        reason = f'step {step} is before eval.eval_step_range start {start}'
      elif end is not None and step > int(end):
        reason = f'step {step} is past eval.eval_step_range end {end}'
    if reason is None and at_steps is not None:
      wanted = sorted({int(s) for s in at_steps})
      self._warn_unreachable_eval_steps(wanted)
      if step not in wanted:
        reason = f'step {step} is not in eval.eval_at_steps'
    if reason is None:
      return True
    if self.trainer.global_rank == 0:
      print(f'[eval-window] skipping generative eval: {reason}', flush=True)
    return False

  def _warn_unreachable_eval_steps(self, wanted):
    """Say so, once, when an `eval.eval_at_steps` entry can never fire.

    Validation only lands on some steps, so a step that is not one of them is
    never a step this method is called at -- and the failure mode is silence,
    which reads exactly like "the sweep is broken".

    Which steps those are needs one conversion. `eval_at_steps` is in
    `global_step`, i.e. OPTIMIZER steps -- one per global batch, the same axis
    as `trainer.max_steps` and as every logged metric. `val_check_interval` is
    not: Lightning compares it against `batch_idx`, which counts MICRO-batches
    (`_should_check_val_fx`). The two coincide only at
    `accumulate_grad_batches=1`; at 2, `val_check_interval=250` is a validation
    every 125 optimizer steps.
    """
    if getattr(self, '_eval_at_steps_checked', False):
      return
    self._eval_at_steps_checked = True
    interval = getattr(self.trainer, 'val_check_interval', None)
    # A float is a fraction of an epoch, which says nothing about which steps
    # validation lands on; only the integer form is checkable.
    if not isinstance(interval, int) or interval <= 0:
      return
    accum = int(getattr(self.trainer, 'accumulate_grad_batches', 1) or 1)
    if accum > 1 and interval % accum:
      # Validation then lands on optimizer steps that are not evenly spaced,
      # and guessing which would be worse than saying nothing.
      return
    cadence = max(1, interval // max(1, accum))
    stranded = [s for s in wanted if s % cadence]
    if stranded and self.trainer.global_rank == 0:
      print(f'[eval-window] eval.eval_at_steps {stranded} are not multiples '
            f'of the validation cadence ({cadence} optimizer steps = '
            f'val_check_interval {interval} / accumulate_grad_batches {accum})'
            ' and will never fire', flush=True)

  def _collect_gm_reference(self, batch):
    """Keep a capped sample of real validation text for Gradient Moment.

    Decoding is not free, so this stops as soon as `gm_eval_samples` rows are
    held and does nothing at all when the metric is off.
    """
    if (not metrics.GRADIENT_MOMENT_ENABLED
        or not self.config.eval.get('owt_gm', False)):
      return
    # The rows only feed Gradient Moment, which the window may have switched
    # off for this validation; decoding them anyway would be pure waste.
    if not getattr(self, '_gen_eval_active', True):
      return
    wanted = int(self.config.eval.get('gm_eval_samples', 64))
    held = getattr(self, '_gm_ref_texts', None)
    if held is None:
      self._gm_ref_texts = held = []
    if len(held) >= wanted:
      return
    ids = batch['input_ids'][: wanted - len(held)]
    held.extend(self.tokenizer.batch_decode(ids))

  def _log_gradient_moment(self, sample_texts):
    """Score the samples against real data by Gradient Moment, and log it.

    Kept separate from generative perplexity because it answers the question
    perplexity cannot: gen_ppl rewards text the reference LM finds predictable,
    which a collapsing model produces *more* of, so it can improve while the
    model degrades. GM compares gradients against real data, so collapse moves
    it the right way.
    """
    if (not metrics.GRADIENT_MOMENT_ENABLED
        or not self.config.eval.get('owt_gm', False)):
      return
    refs = getattr(self, '_gm_ref_texts', None) or []
    if not sample_texts or not refs:
      print('[gm] no samples or no reference rows, skipping', flush=True)
      return
    try:
      results = self.metrics.record_gradient_moment(
        sample_texts, refs, max_length=self.num_tokens,
        batch_size=int(self.config.eval.get('gm_eval_batch_size', 4)),
        device=self.device)
    except Exception as e:
      # A diagnostic must not take down a run measured in days.
      print(f'[gm] failed, continuing: {e}', flush=True)
      return
    if results is None:
      return
    print(f"[gm] step {self.global_step}: centered {results['centered_gm']:+.4e} "
          f"(n_pairs={results['n_pairs']})", flush=True)
    for key, name in (('centered_gm', 'val/gm'),
                      ('uncentered_gm_samples', 'val/gm_uncentered_samples'),
                      ('uncentered_gm_data', 'val/gm_uncentered_data')):
      self.log(name, torch.tensor(float(results[key]), device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)

  def _log_owt_pareto(self):
    """Sample at several step budgets and record the fluency/diversity curve.

    One denoising-step count gives one (entropy, perplexity) point; the set of
    them is the frontier this checkpoint sits on. Opt-in via
    `eval.owt_pareto_steps`, because the sweep costs the sum of its budgets in
    extra sampling -- [2,4,8,16,32,64] is 126 forward passes per batch on top of
    the validation that already ran.
    """
    config = self.config.eval
    budgets = config.get('owt_pareto_steps', None)
    if not budgets or self.trainer.sanity_checking:
      return
    every = int(config.get('owt_pareto_every_n_validations', 1) or 1)
    self._owt_pareto_count = getattr(self, '_owt_pareto_count', 0) + 1
    if (self._owt_pareto_count - 1) % every:
      return
    if not self.config.eval.compute_generative_perplexity:
      print('[owt-pareto] skipped: needs eval.compute_generative_perplexity',
            flush=True)
      return

    budgets = sorted({int(b) for b in budgets})
    num_batches = int(config.get('owt_pareto_num_batches', 0) or
                      self.config.sampling.num_sample_batches)
    seeds = max(1, int(config.get('owt_pareto_seeds', 1) or 1))

    # Repeats are *seeded*, not merely repeated. Averaging more samples would
    # shrink the error bars either way, but fixing the seeds also makes every
    # checkpoint draw the SAME noise, which turns the frontier's movement
    # between checkpoints into a paired comparison: a shift then reflects the
    # model changing rather than one validation getting luckier draws. That
    # matters more than the raw variance here, since the question the plot
    # answers is which direction the frontier is moving.
    #
    # The seed carries the rank, so data-parallel ranks still draw different
    # samples -- seeding them identically would make the extra ranks redundant
    # rather than additional evidence.
    base_seed = int(getattr(self.config, 'seed', 0) or 0)
    rank = self.trainer.global_rank

    # Sampling here must not perturb the training RNG stream, or enabling the
    # sweep would silently change the trajectory of the run it is measuring.
    cpu_rng = torch.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)
    try:
      # Gradient Moment per budget, when it is on and there is real text to
      # compare against. Worth the cost precisely here: gen_ppl is the frontier
      # axis most vulnerable to collapse -- a degenerate sampler scores *better*
      # on it -- and GM is the one number in this sweep that moves the right way
      # when that happens. Cheap budgets are where collapse shows up first.
      gm_on = (metrics.GRADIENT_MOMENT_ENABLED
               and bool(self.config.eval.get('owt_gm', False)))
      gm_refs = getattr(self, '_gm_ref_texts', None) or []
      if gm_on and not gm_refs:
        print('[owt-pareto] GM requested but no reference rows; '
              'skipping GM per budget', flush=True)
      curve = []
      for budget in budgets:
        entropies, perplexities, moments = [], [], []
        for repeat in range(seeds):
          if seeds > 1:
            # 9973 is prime, so budgets and repeats cannot collide onto a
            # shared seed the way `base + repeat + budget` would.
            torch.manual_seed(base_seed + 9973 * (repeat + 1) + rank)
          # Fresh accumulators per repeat: torchmetrics aggregate until reset,
          # so without this each point would carry the previous budgets' and
          # repeats' samples too and the curve would flatten into a running
          # average.
          self.metrics.reset()
          texts = []
          for _ in range(num_batches):
            samples = self.generate_samples(
              num_samples=self.config.loader.eval_batch_size,
              num_steps=budget)
            self.metrics.record_entropy(samples)
            decoded = self.tokenizer.batch_decode(samples)
            if gm_on and gm_refs:
              texts.extend(decoded)
            self.metrics.record_generative_perplexity(
              decoded, self.num_tokens, self.device)
          entropies.append(float(self.metrics.sample_entropy.compute()))
          perplexities.append(float(self.metrics.gen_ppl.compute()))
          if gm_on and gm_refs:
            # Scored per seed rather than on the pooled samples, so GM is
            # averaged the same way entropy and perplexity are and its spread
            # means the same thing as theirs.
            try:
              gm = self.metrics.record_gradient_moment(
                texts, gm_refs, max_length=self.num_tokens,
                batch_size=int(self.config.eval.get('gm_eval_batch_size', 4)),
                device=self.device)
            except Exception as e:
              print(f'[owt-pareto] GM failed at budget {budget}: {e}',
                    flush=True)
              gm = None
            if gm is not None:
              moments.append(float(gm['centered_gm']))
        point = {'steps': budget,
                 'entropy': _mean(entropies),
                 'gen_ppl': _mean(perplexities),
                 'entropy_std': _std(entropies),
                 'gen_ppl_std': _std(perplexities),
                 'seeds': seeds}
        if moments:
          point['gm'] = _mean(moments)
          point['gm_std'] = _std(moments)
        curve.append(point)
        spread = ('' if seeds < 2 else
                  f" +/- {point['gen_ppl_std']:.2f} / "
                  f"{point['entropy_std']:.4f} over {seeds} seeds")
        gm_txt = '' if 'gm' not in point else f" gm {point['gm']:+.4e}"
        print(f"[owt-pareto] step {self.global_step} budget {budget}: "
              f"gen_ppl {point['gen_ppl']:.2f} entropy {point['entropy']:.4f}"
              f"{gm_txt}{spread}", flush=True)
    finally:
      torch.set_rng_state(cpu_rng)
      if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    self._publish_owt_pareto(curve)

  def _publish_owt_pareto(self, curve):
    """Log one checkpoint's frontier: scalar series, trends, and the figure."""
    def _scalar(name, value):
      self.log(name, torch.tensor(float(value), device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)

    for point in curve:
      # Per-budget trends against the *training* step, so each budget also reads
      # as an ordinary curve over the run.
      tag = f"{point['steps']:g}"
      _scalar(f'val/gen_ppl_s{tag}', point['gen_ppl'])
      _scalar(f'val/entropy_s{tag}', point['entropy'])
      # The spread across seeds, as its own series. The figure already draws it
      # as error bars, but those cannot be read off a tracker's scalar view --
      # and the spread is what decides whether a move in the mean is real. It
      # is also worth watching in its own right: a spread that grows over
      # training says the sampler is becoming erratic, which the mean hides.
      #
      # Logged only when several seeds actually ran. At one seed `_std` is
      # identically 0, and a flat zero line invites being read as "no
      # variance measured" rather than "variance not measured". Since
      # `owt_pareto_seeds` is fixed for a run, the set of series stays stable
      # within it rather than appearing and vanishing between validations.
      if int(point.get('seeds', 1)) > 1:
        _scalar(f'val/gen_ppl_s{tag}_std', point.get('gen_ppl_std', 0.0))
        _scalar(f'val/entropy_s{tag}_std', point.get('entropy_std', 0.0))
      if 'gm' in point:
        _scalar(f'val/gm_s{tag}', point['gm'])
        if int(point.get('seeds', 1)) > 1:
          _scalar(f'val/gm_s{tag}_std', point.get('gm_std', 0.0))

    if self.trainer.global_rank != 0:
      return
    logger = self.trainer.logger
    if logger is None:
      return
    # The frontier itself, as a series whose x-axis is sample entropy rather
    # than the training step -- the same pair of axes the figure draws, so the
    # native scalar graph and `pareto/frontier` tell the same story. `self.log`
    # cannot express that (it always stamps the trainer's step), so go through
    # the logger's own `log_metrics`, which passes `step` straight through. One
    # series per checkpoint, since a single graph cannot hold two curves.
    #
    # The tracker indexes points by an integer step, and entropy is a float
    # around 5, so it is carried at fixed scale: an x of 5062 means entropy
    # 5.062. Three decimals separates budgets that sit ~0.05 apart with room to
    # spare, and keeps the axis monotone in entropy.
    series = f'pareto/owt_step_{self.global_step}'
    for point in curve:
      try:
        logger.log_metrics({series: point['gen_ppl']},
                           step=int(round(point['entropy']
                                          * _ENTROPY_X_SCALE)))
      except Exception as e:  # pragma: no cover - tracker best-effort
        print(f'[owt-pareto] scalar log failed: {e}', flush=True)
        break

    history_path = os.path.join(self.config.checkpointing.save_dir,
                                'owt_pareto_history.json')
    run_uid = getattr(logger, 'version', '') or ''
    if getattr(self, '_owt_pareto_history', None) is None:
      # A *resumed* run resumes the same tracker run, so the figure has to pick
      # up the curves the earlier process drew. A fresh run reusing the same
      # save_dir must not -- it would graft the previous run's frontiers onto
      # this one's plot. The run uid tells the two cases apart.
      self._owt_pareto_history = {}
      if os.path.exists(history_path):
        try:
          with open(history_path) as f:
            stored = json.load(f)
          if stored.get('run_uid') == run_uid:
            self._owt_pareto_history = {int(k): v
                                        for k, v in stored['curves'].items()}
          else:
            print('[owt-pareto] ignoring history from a different run '
                  f"({stored.get('run_uid')!r} != {run_uid!r})", flush=True)
        except Exception as e:  # pragma: no cover
          print(f'[owt-pareto] history read failed: {e}', flush=True)
    self._owt_pareto_history[int(self.global_step)] = curve
    try:
      os.makedirs(self.config.checkpointing.save_dir, exist_ok=True)
      with open(history_path, 'w') as f:
        json.dump({'run_uid': run_uid, 'curves': self._owt_pareto_history},
                  f, indent=2)
    except Exception as e:  # pragma: no cover
      print(f'[owt-pareto] history write failed: {e}', flush=True)

    if not hasattr(logger, 'log_image'):
      return
    try:
      import owt_pareto_plot
      image = owt_pareto_plot.render(
        self._owt_pareto_history,
        reference=owt_pareto_plot.load_reference(
          self.config.eval.get('owt_pareto_reference', None)))
      logger.log_image('pareto/frontier', image, step=self.global_step)
    except Exception as e:  # pragma: no cover - plotting is best-effort
      print(f'[owt-pareto] figure failed: {e}', flush=True)

  def _log_owt_pareto_temperature(self):
    """Sweep the sampler's temperature at this checkpoint.

    `_log_owt_pareto` varies the CHECKPOINT at a fixed sampler and answers "is
    the frontier moving the right way". This varies the SAMPLER at a fixed
    checkpoint, and having both is what lets a move be attributed: a frontier
    that drops may only mean the model got sharper, and the tell is whether the
    new point sits below the checkpoint's own temperature curve or merely
    somewhere along it. The budget sweep cannot answer that at any number of
    checkpoints, because it never draws the curve the point would sit on.

    Off unless `eval.owt_pareto_temp_every_n_validations` is set, and it should
    stay off for most runs: the cost is `len(eval.pareto_temperatures)` times
    the budget sweep this runs on top of -- 5 x 126 forward passes per batch for
    the documented budget list, which is not a diagnostic to run every
    validation. The temperatures themselves come from `eval.pareto_temperatures`
    rather than a knob of their own, so a sweep is configured the same way here
    and in `mode=pareto_eval`.
    """
    config = self.config.eval
    every = int(config.get('owt_pareto_temp_every_n_validations', 0) or 0)
    temperatures = config.get('pareto_temperatures', None)
    budgets = config.get('owt_pareto_steps', None)
    if not every or not temperatures or not budgets:
      return
    if self.trainer.sanity_checking:
      return
    # Its own counter, not the budget sweep's: the two cadences are set
    # independently and sharing a counter would couple them.
    self._owt_pareto_temp_count = getattr(
      self, '_owt_pareto_temp_count', 0) + 1
    if (self._owt_pareto_temp_count - 1) % every:
      return
    if not config.compute_generative_perplexity:
      print('[owt-pareto-temp] skipped: needs '
            'eval.compute_generative_perplexity', flush=True)
      return

    # `sweep` SPLITS its sample count across ranks; `_log_owt_pareto` has every
    # rank draw `num_batches` and pools them. Scaling by the world size is what
    # makes the two protocols agree, so the T=1 point measured here is the same
    # measurement as the budget sweep's point at that budget rather than a
    # smaller one that merely sits near it.
    world_size = max(1, int(getattr(self.trainer, 'world_size', 1) or 1))
    num_batches = int(config.get('owt_pareto_num_batches', 0) or
                      self.config.sampling.num_sample_batches) * world_size
    seeds = max(1, int(config.get('owt_pareto_seeds', 1) or 1))

    # Sampling here must not perturb the training RNG stream, or enabling the
    # sweep would silently change the trajectory of the run it is measuring.
    # `sweep` seeds per (repeat, budget, rank) and never per checkpoint, so
    # every validation draws the SAME noise -- which is what makes the movement
    # across checkpoints a paired comparison rather than a lucky draw.
    cpu_rng = torch.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)
    try:
      import pareto_temperature
      curves, _ = pareto_temperature.sweep(
        self,
        temperatures=list(temperatures),
        budgets=sorted({int(b) for b in budgets}),
        seeds=seeds,
        num_batches=num_batches,
        batch_size=self.config.loader.eval_batch_size,
        max_length=self.num_tokens,
        # No text kept: the training tracker already carries samples from
        # ordinary validation, and a table per validation would bury them.
        keep_samples=0)
    finally:
      torch.set_rng_state(cpu_rng)
      if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    self._publish_owt_pareto_temperature(curves)

  def _publish_owt_pareto_temperature(self, curves):
    """Log this checkpoint's temperature fan, and its history over the run.

    Two figures, because the sweep has three dimensions and no single axes holds
    them: `pareto/temperature_frontier` is this checkpoint alone, one curve per
    budget, and `pareto/temperature_by_checkpoint` is one panel per budget with
    one line per checkpoint, which is where the run's progress actually reads.
    """
    def _scalar(name, value):
      self.log(name, torch.tensor(float(value), device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)

    # One series per (budget, temperature) against the TRAINING step. The
    # eval-mode sweep indexes the same numbers by temperature instead, which
    # cannot work here: every validation would write the same handful of x
    # values, and the graph would be a pile of overwrites rather than a trend.
    for temperature, curve in curves.items():
      for point in curve:
        tag = f"s{point['steps']:g}_t{float(temperature):g}"
        _scalar(f'val/gen_ppl_{tag}', point['gen_ppl'])
        _scalar(f'val/entropy_{tag}', point['entropy'])

    if self.trainer.global_rank != 0:
      return
    logger = self.trainer.logger
    if logger is None:
      return

    history_path = os.path.join(self.config.checkpointing.save_dir,
                                'owt_pareto_temp_history.json')
    run_uid = getattr(logger, 'version', '') or ''
    if getattr(self, '_owt_pareto_temp_history', None) is None:
      # Same rule as the budget sweep's history: a RESUMED run continues the
      # same tracker run and must pick up the curves the earlier process drew,
      # while a fresh run reusing the save_dir must not graft them onto its own
      # figure. The run uid is what tells the two apart.
      self._owt_pareto_temp_history = {}
      if os.path.exists(history_path):
        try:
          with open(history_path) as f:
            stored = json.load(f)
          if stored.get('run_uid') == run_uid:
            self._owt_pareto_temp_history = {
              int(k): v for k, v in stored['curves'].items()}
          else:
            print('[owt-pareto-temp] ignoring history from a different run '
                  f"({stored.get('run_uid')!r} != {run_uid!r})", flush=True)
        except Exception as e:  # pragma: no cover
          print(f'[owt-pareto-temp] history read failed: {e}', flush=True)
    # Keyed by `%g`, because JSON keys are strings and `str(0.7000000001)` is
    # not a temperature anyone set. `_by_budget` parses the key back either way.
    self._owt_pareto_temp_history[int(self.global_step)] = {
      f'{float(t):g}': curve for t, curve in curves.items()}
    try:
      os.makedirs(self.config.checkpointing.save_dir, exist_ok=True)
      with open(history_path, 'w') as f:
        json.dump({'run_uid': run_uid,
                   'curves': self._owt_pareto_temp_history}, f, indent=2)
    except Exception as e:  # pragma: no cover
      print(f'[owt-pareto-temp] history write failed: {e}', flush=True)

    # Everything below publishes the sweep as *numbers* as well as pictures.
    # `pareto/temperature_frontier` is an image, and an image cannot answer
    # "what was gen_ppl at T=0.9, 16 steps, at training step 3000" -- it has to
    # be read off a pixel. The per-(budget, temperature) scalars above answer
    # that one series at a time; these two answer it in a form that parses.
    by_budget = collections.defaultdict(list)
    for temperature, curve in curves.items():
      for point in curve:
        by_budget[int(point['steps'])].append(
          dict(point, temperature=float(temperature)))

    # The frontier itself, as scalars, exactly as `_publish_owt_pareto` logs
    # the budget frontier: x is entropy at fixed scale (5062 means 5.062), y is
    # gen_ppl, one series per budget per checkpoint. `self.log` cannot express
    # it -- it always stamps the trainer's step -- so this goes through the
    # logger's own `log_metrics`, which passes `step` straight through. Named
    # per checkpoint because a single series cannot hold two curves: reusing
    # one would have every validation overwrite the same handful of x values.
    for budget, points in sorted(by_budget.items()):
      series = f'pareto/temp_step_{self.global_step}_s{budget}'
      for point in sorted(points, key=lambda p: p['entropy']):
        try:
          logger.log_metrics({series: point['gen_ppl']},
                             step=int(round(point['entropy']
                                            * _ENTROPY_X_SCALE)))
        except Exception as e:  # pragma: no cover - tracker best-effort
          print(f'[owt-pareto-temp] frontier log failed: {e}', flush=True)
          break

    # And the whole sweep as one table per validation: the exact values at
    # full precision, including the spreads and the seed count that the scalar
    # series drop, in a single place that a script can pull down and parse.
    # Stable key + explicit `step`, so the tracker keeps them as one
    # step-indexed series rather than a pile of unrelated tables.
    if hasattr(logger, 'log_table'):
      rows = []
      for temperature, curve in sorted(curves.items(),
                                       key=lambda kv: float(kv[0])):
        for point in sorted(curve, key=lambda p: int(p['steps'])):
          rows.append([int(self.global_step), float(temperature),
                       int(point['steps']), float(point['entropy']),
                       float(point['gen_ppl']),
                       float(point.get('entropy_std', 0.0)),
                       float(point.get('gen_ppl_std', 0.0)),
                       int(point.get('seeds', 1))])
      try:
        logger.log_table(
          key='pareto/temperature_values',
          columns=['train_step', 'temperature', 'steps', 'entropy',
                   'gen_ppl', 'entropy_std', 'gen_ppl_std', 'seeds'],
          data=rows, step=self.global_step)
      except Exception as e:  # pragma: no cover - tracker best-effort
        print(f'[owt-pareto-temp] value table failed: {e}', flush=True)

    if not hasattr(logger, 'log_image'):
      return
    try:
      import owt_pareto_plot
      reference = owt_pareto_plot.load_reference(
        self.config.eval.get('owt_pareto_reference', None))
      logger.log_image(
        'pareto/temperature_frontier',
        owt_pareto_plot.render_temperature(
          curves, reference=reference,
          subtitle=f'training step {self.global_step}'),
        step=self.global_step)
      history = sorted(self._owt_pareto_temp_history.items())
      # One checkpoint is not a progression, and the grid would say nothing the
      # fan above has not already said.
      if len(history) > 1:
        logger.log_image(
          'pareto/temperature_by_checkpoint',
          owt_pareto_plot.render_checkpoints(
            [{'short': f'{step / 1000:g}k',
              'label': f'training step {step}',
              'curves': stored} for step, stored in history],
            reference=reference,
            title='Fluency vs diversity: temperature at each checkpoint'),
          step=self.global_step)
    except Exception as e:  # pragma: no cover - plotting is best-effort
      print(f'[owt-pareto-temp] figure failed: {e}', flush=True)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def generate_samples(self, num_samples, num_steps, eps):
    raise NotImplementedError

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    self._eval_mode()
    samples = self.generate_samples(
      num_samples=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self._train_mode()
    return samples

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    raise NotImplementedError

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    (input_tokens, output_tokens,
     valid_tokens) = self._process_model_input(
       x0, valid_tokens)
    loss = self.nll(input_tokens, output_tokens,
                    current_accumulation_step, train_mode,
                    valid_tokens=valid_tokens)
    assert loss.ndim == 2
    if self.ignore_bos:
      loss[:, 1:] = loss[:, 1:]
      valid_tokens[:, 1:] = valid_tokens[:, 1:]

    nlls = (loss * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    token_nll = nlls / num_tokens

    return Loss(loss=token_nll,
                nlls=nlls,
                prior_loss=0.0,
                num_tokens=num_tokens)


class Diffusion(TrainerBase):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.noise_removal in {
      'none', 'ancestral', 'greedy'}
    assert self.loss_type in {'elbo', 'low_var'}
    if self.config.sampling.noise_removal == 'greedy':
      assert self.sampler != 'analytic'
      assert self.parameterization in {'mean', 'subs'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      if self.config.adversarial_distill.is_distill:
        t = t.chunk(self.accum_batches)[accum_step]
      else:
        t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype,
                     device=self.device)
    sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
    model_output_t0 = self.forward(x0, sigma_t0)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)


  def nll_per_token(self, model_output, xt, x0, alpha_t,
                    dalpha_t, low_var):
    raise NotImplementedError

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)
    log_x_theta = self.forward(xt, sigma=sigma)
    utils.print_nans(log_x_theta, 'model_output')
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

  def _get_score(self, **kwargs):
    del kwargs
    raise NotImplementedError

  def _denoiser_update(self, x, t):
    raise NotImplementedError

  def _analytic_update(self, x, t, dt):
    raise NotImplementedError

  def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
    raise NotImplementedError

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ancestral':
        _, x = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=None)
      elif self.sampler == 'ancestral_cache':
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x=x,t=t, dt=dt)

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                    device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
      if self.sampler == 'analytic':
        x = self._denoiser_update(x=x, t=t0)
      else:
        _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                 p_x0=p_x0_cache,
                                 noise_removal_step=True)
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
    return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, stride_length, num_strides, dt=0.001):
    # TODO(subham): Test this method after refactoring.
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self.prior_sample(n_samples, self.num_tokens)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    # TODO(subham): Test this method after refactoring.
    self._eval_mode()
    (sampling_steps, samples,
     sequence_lengths) = self._semi_ar_sampler(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    self._train_mode()
    return sampling_steps, samples, sequence_lengths


class AbsorbingState(Diffusion):
  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do 
    # vocab_size = len(tokenizer), so that we account
    # for the special tokens added in dataloader.py.
    # But we use tokenizer.vocab_size so as to to be
    # consistent with the prior checkpoints.
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    self.subs_masking = config.algo.subs_masking
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    assert not (self.parameterization == 'mean'
                and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'mean'

  def q_xt(self, x, alpha_t, valid_tokens=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      alpha_t: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < 1 - alpha_t
    if valid_tokens is not None:
      move_indices = move_indices & valid_tokens.bool()
    xt = torch.where(move_indices, self.mask_index, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      p_x0 = self._temper(self.forward(
        x, self._sigma_from_alphat(alpha_t))).exp()
    
    q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    _x = sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_alphat(self.noise(t)[1])
    sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self._get_score(x, sigma_t)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_alphat(self.noise(t)[1])
    score = self._get_score(x, sigma)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge


class UniformState(Diffusion):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.time_conditioning
    assert self.parameterization == 'mean'
    if self.config.algo.name != 'distillation':
      assert self.T == 0

  def q_xt(self, x, alpha_t, valid_tokens=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      move_chance: float torch.Tensor with shape
        (batch_size, 1).
    """
    move_indices = torch.rand(
      *x.shape, device=x.device) < 1 - alpha_t
    if valid_tokens is not None:
      move_indices = move_indices & valid_tokens.bool()
    uniform_tensor = torch.randint(
      0, self.vocab_size, x.shape, device=x.device)
    xt = torch.where(move_indices, uniform_tensor, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return torch.randint(
      0, self.vocab_size, batch_dims, dtype=torch.int64,
      device=self.device)
