import contextlib
import itertools
import typing
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
import pareto_curve
import utils
from ipdb import set_trace as debug


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


class Cosine(torch.nn.Module):
  """`alpha_t = 1 - cos(pi/2 * (1 - t))`, as in the upstream duo repo.

  Selected with `noise=cosine`; it is the schedule the published CIFAR-10
  checkpoints (`jdeschena/duo2-cifar10`) were trained under, so distilling from
  them requires it -- `alpha_t` is what `q_xt` masks with and what the teacher's
  time conditioning is derived from, and a mismatched schedule silently scores
  the teacher at noise levels it never saw.

  `dalpha_t` is returned with a trailing singleton axis (its callers pair it
  with an already-unsqueezed `alpha_t` against a `(batch, length)` loss, which a
  bare `(batch,)` would not broadcast against; `LogLinear` sidesteps this by
  returning a scalar).
  """

  def __init__(self):
    super().__init__()
    self.eps = 1e-3
    self.half_pi = torch.pi / 2

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - torch.cos(self.half_pi * (1 - t))
    dalpha_t = - torch.sin(self.half_pi * (1 - t)) * self.half_pi
    if dalpha_t.ndim == 1:
      dalpha_t = dalpha_t.unsqueeze(-1)
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


@dataclass
class NoiseDiversity:
  """Result of `TrainerBase.noise_diversity_report`."""
  metrics: typing.Dict[str, float]
  # Decoded samples, `variants[k][b]` = row `b` under noise draw `k`, or None
  # if they were not requested.
  variants: typing.Optional[typing.List[typing.List[str]]]


def _matched_agreement(stack):
  """Mean positional token agreement between *matched rows* of different
  replicates: mean over `k < j`, row `b` and position `l` of
  `stack[k, b, l] == stack[j, b, l]`. 1.0 => the replicates are identical."""
  num_replicates = stack.shape[0]
  total, num_pairs = 0.0, 0
  for i in range(num_replicates):
    for j in range(i + 1, num_replicates):
      total += (stack[i] == stack[j]).float().mean().item()
      num_pairs += 1
  return total / max(num_pairs, 1)


def _independent_agreement(stack):
  """Mean positional agreement between two *different rows* of the same
  replicate, i.e. between fully independent samples. This is the chance-level
  floor the matched-row agreements above should be read against: two unrelated
  unconditional samples still agree wherever both emit a common token."""
  batch_size = stack.shape[1]
  if batch_size < 2:
    return None
  equal = (stack[:, :, None, :] == stack[:, None, :, :]).float().mean(-1)
  rows, cols = torch.triu_indices(
    batch_size, batch_size, offset=1, device=stack.device)
  return equal[:, rows, cols].mean().item()


def _distinct_per_position(stack):
  """Mean number of distinct tokens a single position takes across the
  replicates. 1.0 => every replicate wrote the same token everywhere;
  `num_replicates` => a different token every time."""
  ordered, _ = torch.sort(stack, dim=0)
  return (1 + (ordered[1:] != ordered[:-1]).sum(0)).float().mean().item()


def _unique_sequence_frac(stack):
  """Fraction of the replicates of a row that are distinct sequences, averaged
  over rows. 1/K => all replicates collapsed to one sequence; 1.0 => every
  replicate produced a sequence not seen under any other noise draw."""
  num_replicates = stack.shape[0]
  equal = (stack[:, None] == stack[None, :]).all(-1)  # (K, K, B)
  is_new = torch.ones(equal.shape[0], equal.shape[-1],
                      dtype=torch.bool, device=stack.device)
  for k in range(1, num_replicates):
    is_new[k] = ~ equal[:k, k].any(0)
  return (is_new.float().sum(0) / num_replicates).mean().item()


class TrainerBase(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer,
    vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
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
        config.eval.checkpoint_path, trust_remote_code=True)
    elif self.config.algo.backbone == 'unet':
      # Raw-pixel (CIFAR-10) denoiser. As with `hf_dit`, `eval.checkpoint_path`
      # is where the pretrained weights come from -- for distillation those are
      # the teacher's, and `algo.MDLM_with_AR` copies this backbone into the
      # teacher / fake critics. Accepts a local Lightning `.ckpt` or a
      # `<hf_repo>:<path>` hub spec (see `utils.resolve_checkpoint_path`).
      self.backbone = models.unet.UNet(
        self.config, vocab_size=self.vocab_size)
      if config.eval.checkpoint_path:
        utils.load_backbone_from_lightning_checkpoint(
          self.backbone, config.eval.checkpoint_path,
          use_ema=not config.eval.disable_ema)
        # `algo.time_conditioning=False` makes `_process_sigma` feed the
        # denoiser a constant zero time, which a `*_time_cond` checkpoint was
        # never trained for -- it still runs, and silently scores every noise
        # level as if it were t=0. The algo configs default it to False, so this
        # is easy to leave off by accident.
        if ('time_uncond' not in str(config.eval.checkpoint_path)
            and 'time_cond' in str(config.eval.checkpoint_path)
            and not self.config.algo.time_conditioning):
          print('[checkpoint] WARNING: this looks like a time-conditioned '
                'checkpoint but algo.time_conditioning=False, so the denoiser '
                'will be given t=0 at every noise level. Pass '
                'algo.time_conditioning=True.', flush=True)

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    # Noise Schedule
    if str(self.config.noise.get('type', 'log-linear')) == 'cosine':
      self.noise = Cosine()
    else:
      self.noise = LogLinear()

    self.is_vision = utils.is_vision(self.config)
    self.metrics = metrics.Metrics(
      gen_ppl_eval_model_name_or_path=\
        self.config.eval.gen_ppl_eval_model_name_or_path,
      eval_ppl_batch_size=\
        self.config.eval.perplexity_batch_size,
      compute_gm=(self.config.eval.get('compute_gm', False)
                  and not self.is_vision),
      gm_eval_model_name_or_path=self.config.eval.get(
        'gm_eval_model_name_or_path', 'gpt2'),
      # Images: skip the text reference models/tokenizer entirely and build the
      # image metric instead (see `metrics.Metrics`).
      is_vision=self.is_vision,
      compute_fid=(self.config.eval.get('compute_fid', True)
                   and self.is_vision),
      fid_feature=self.config.eval.get('fid_feature', 2048))

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
    assert self.config.algo.backbone in {'dit', 'hf_dit', 'unet'}
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

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    raise NotImplementedError

  def _pred_entropy_mask(self, xt):
    """Positions of `xt` whose predictive distribution is a genuine model
    prediction (see `metrics.Metrics.record_pred_entropy`). None => all."""
    del xt
    return None

  def _record_pred_entropy(self, dist, xt, log_input=False):
    """Fold one denoiser call's predictive distribution into
    `val/pred_entropy`. Called only from the samplers, so it costs a
    reduction over an already-materialized tensor and no extra forward."""
    if dist is None or not self.config.eval.get(
        'compute_pred_entropy', True):
      return
    self.metrics.record_pred_entropy(
      dist, mask=self._pred_entropy_mask(xt), log_input=log_input)

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
    self._eval_mode()
    # Buffer for GM data batches (decoded to text; see on_validation_epoch_end).
    self._gm_data_texts = []
    # How many validation batches have gone into FID's *real* side this epoch.
    self._fid_real_batches = 0
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    del batch_idx
    # GPT-2 Gradient Moment needs *data* samples to pair with generated ones.
    # Buffer the first `gm_num_batches` val batches as text (not tensors, which
    # would be inference tensors and unusable in the autograd pass GM needs).
    if self.metrics.gm is not None and len(
        self._gm_data_texts) < self.config.eval.get('gm_num_batches', 4):
      self._gm_data_texts.append(
        self.tokenizer.batch_decode(batch['input_ids']))
    # FID's real side: the first `fid_num_batches` validation images, folded in
    # as they stream past rather than buffered (the Inception features are all
    # the estimator keeps).
    if (self.metrics.fid is not None
        and self._fid_real_batches
        < self.config.eval.get('fid_num_batches', 4)):
      self.metrics.record_fid(
        self.tokenizer.batch_decode(batch['input_ids']), real=True)
      self._fid_real_batches += 1
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    return losses.loss

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      samples, text_samples, images = None, None, None
      for _sb in range(
        self.config.sampling.num_sample_batches):
        samples = self.generate_samples(
          num_samples=self.config.loader.eval_batch_size)

        self.metrics.record_entropy(samples)
        # Decode the samples to be re-tokenized by eval model
        decoded = self.tokenizer.batch_decode(samples)
        if self.is_vision:
          # Raw pixels: `batch_decode` returns uint8 (B, 3, H, W) images, and
          # every metric below is text-only. FID takes their place.
          images = decoded
          self.metrics.record_fid(images, real=False)
          continue
        text_samples = decoded
        if self.config.eval.compute_generative_perplexity:
          self.metrics.record_generative_perplexity(
            text_samples, self.num_tokens, self.device)
        # GPT-2 Gradient Moment: pair this generated batch with a buffered data
        # batch (needs trainer.inference_mode=false so autograd is available).
        if (self.metrics.gm is not None
            and _sb < len(self._gm_data_texts)):
          self.metrics.gm.accumulate(
            text_samples, self._gm_data_texts[_sb],
            self.num_tokens, self.device)
      if self.is_vision:
        self._log_image_metrics(images)
      elif text_samples is not None:
        if self.trainer.global_rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Log the last generated samples
          text_samples = text_samples[
            : self.config.sampling.num_sample_log]
          self.trainer.logger.log_table(
            key=f'samples@global_step{self.global_step}',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples])
        if self.config.eval.compute_generative_perplexity:
          # Held in locals so the Pareto curve below reuses these values
          # instead of calling `compute()` again -- under DDP that is a
          # collective, and it has to run on every rank or not at all.
          gen_ppl = self.metrics.gen_ppl.compute()
          sample_entropy = self.metrics.sample_entropy.compute()
          self.log('val/gen_ppl',
                   gen_ppl,
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          self.log('val/sample_entropy',
                   sample_entropy,
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          self._log_pareto_curve(sample_entropy, gen_ppl)
        # Mean entropy of the denoiser's own conditionals over the sampling
        # trajectory (nats; see metrics.record_pred_entropy). Accumulated by
        # the samplers above, so it is only defined for samplers that expose
        # `p_x0` -- `analytic` and DUO's ancestral update do not.
        if float(self.metrics.pred_entropy.weight) > 0:
          self.log('val/pred_entropy',
                   self.metrics.pred_entropy.compute(),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
        if self.metrics.gm is not None:
          self.log('val/gm',
                   torch.tensor(self.metrics.gm.compute(),
                                device=self.device),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
    # How much of that diversity was the injected input noise? Independent of
    # the block above (it generates its own, seed-controlled batches).
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
        and self.config.eval.get('noise_diversity', False)):
      diversity = self.noise_diversity_report(return_variants=True)
      for name, value in diversity.metrics.items():
        self.log(f'val/{name}',
                 torch.tensor(value, device=self.device),
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
      if (diversity.variants is not None
          and self.trainer.global_rank == 0
          and hasattr(self.trainer.logger, 'log_table')):
        # One row per noise draw, same decoding seed: the qualitative version
        # of `noise_div/agree_noise`.
        self.trainer.logger.log_table(
          key=f'noise_variants@global_step{self.global_step}',
          columns=['noise draw', 'sample'],
          data=[[k, replicate[0]]
                for k, replicate in enumerate(diversity.variants)])
    self._train_mode()

  def _log_image_metrics(self, images):
    """Validation metrics for a raw-pixel run (`data.modality=image`).

    The image-domain replacement for the text block above: everything there
    (generative perplexity, the GPT-2 gradient moment, the decoded sample table)
    needs a language model or a tokenizer and says nothing about pixels. What is
    logged instead:

    * `val/fid` -- Frechet Inception Distance of the generated images against
      validation images (`metrics.Metrics.compute_fid`). Over a few batches only,
      so it is a *relative* signal across steps, biased far above a 50k-sample
      FID.
    * `val/sample_entropy` -- unigram entropy of the emitted byte values, i.e.
      how much of the palette a sample uses. Domain-agnostic (it is the same
      statistic the text runs log) and the first thing to collapse if the
      generator mode-collapses to flat images.
    * `val/pred_entropy` -- entropy of the denoiser's own per-pixel
      conditionals over the sampling trajectory, bounded by log(257) ~= 5.55.
    * a grid of the last generated batch, for loggers with `log_image`.

    `val/nll`, `val/bpd` and `val/ppl` are logged by the caller and need no
    special casing -- for images `val/bpd` is bits per *pixel-channel*.
    """
    fid = self.metrics.compute_fid()
    if fid is not None:
      self.log('val/fid',
               torch.as_tensor(float(fid), device=self.device),
               on_epoch=True, on_step=False, sync_dist=True)
    self.log('val/sample_entropy',
             self.metrics.sample_entropy.compute(),
             on_epoch=True, on_step=False, sync_dist=True)
    if float(self.metrics.pred_entropy.weight) > 0:
      self.log('val/pred_entropy',
               self.metrics.pred_entropy.compute(),
               on_epoch=True, on_step=False, sync_dist=True)
    self._log_sample_images(images)

  def _log_sample_images(self, images):
    """Log a grid of generated images. Rank zero, best-effort, never fatal."""
    if (images is None
        or not self.config.eval.get('log_sample_images', True)
        or self.trainer.global_rank != 0
        or not hasattr(self.trainer.logger, 'log_image')):
      return
    try:
      import torchvision
      num = min(images.shape[0],
                int(self.config.eval.get('sample_image_num', 16)))
      grid = torchvision.utils.make_grid(
        images[:num].detach().cpu(),
        nrow=int(self.config.eval.get('sample_image_nrow', 8)),
        padding=2)
      self.trainer.logger.log_image(
        key=self.config.eval.get('sample_image_key', 'val/samples'),
        image=grid.permute(1, 2, 0).numpy(),
        step=self.global_step)
    except Exception as e:  # pragma: no cover - diagnostics are best-effort
      print('[samples] image log failed: {}'.format(e), flush=True)

  def _log_pareto_curve(self, sample_entropy, gen_ppl):
    """Extend the run's quality/diversity Pareto curve by one point.

    Called from `on_validation_epoch_end` right after `val/gen_ppl` and
    `val/sample_entropy` go out, with the very same two numbers, so the curve
    is refreshed on exactly the schedule those scalars are -- once per
    validation, at the same global step. The point is appended to
    `eval.pareto_path` (a JSONL under `checkpointing.save_dir`, so it survives
    resuming), the whole history is re-read and re-rendered, and the image is
    logged under `eval.pareto_image_key`, which makes it a step-indexed series
    in the tracker: scrubbing it replays the run walking its trade-off.

    Free in GPU terms -- both numbers are already computed. Rank zero only, and
    never fatal: a tracker or plotting failure must not take a training run
    down with it, so everything here is caught and reported.
    """
    if not self.config.eval.get('pareto_curve', False):
      return
    if self.trainer.global_rank != 0 or self.trainer.sanity_checking:
      return
    try:
      path = self.config.eval.get('pareto_path', None)
      pareto_curve.append(
        path,
        step=self.global_step,
        sample_entropy=float(sample_entropy),
        gen_ppl=float(gen_ppl))
      if not hasattr(self.trainer.logger, 'log_image'):
        return
      image = pareto_curve.render_history(
        path, label=getattr(self.trainer.logger, 'name', None) or 'this run')
      self.trainer.logger.log_image(
        key=self.config.eval.get('pareto_image_key', 'val/pareto'),
        image=image,
        step=self.global_step)
    except Exception as e:  # pragma: no cover - diagnostics are best-effort
      print('[pareto] curve update failed: {}'.format(e), flush=True)

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

  # --- Input-noise diversity probe -------------------------------------

  @property
  def has_noise_conditioning(self):
    """Whether the backbone injects input noise (`noise_dim > 0`)."""
    return hasattr(self.backbone, 'noise_stream')

  @contextlib.contextmanager
  def _noise_stream(self, seed):
    """Pin the backbone's *input-noise* draws to a private RNG seeded with
    `seed`. A no-op for backbones without noise conditioning, so the probe
    below still runs there -- and correctly reports that varying the noise
    changes nothing."""
    stream = getattr(self.backbone, 'noise_stream', None)
    if stream is None:
      yield
    else:
      with stream(seed):
        yield

  @torch.no_grad()
  def _sample_with_seeds(self, num_samples, decode_seed, noise_seed,
                         num_steps=None):
    """One batch of samples with *both* randomness sources pinned: the global
    RNG (the sampler's categorical / Gumbel draws) to `decode_seed`, and the
    injected input noise to `noise_seed`."""
    torch.manual_seed(decode_seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(decode_seed)
    with self._noise_stream(noise_seed):
      return self.generate_samples(num_samples=num_samples,
                                   num_steps=num_steps)

  @torch.no_grad()
  def noise_diversity_report(self, num_samples=None, num_noises=None,
                             num_steps=None, seed=None,
                             decode_control=None, return_variants=False):
    """How much of the model's sample diversity comes from the *input noise*?

    A masked-diffusion generator has two sources of randomness: the sampler's
    own categorical draws, and -- when `adversarial_distill.noise_dim > 0` --
    the Gaussian noise injected into the embedding residual (D-MMD Sec. 6.5),
    which exists precisely so the generator can stay diverse under a
    moment-matching objective instead of collapsing. This probe separates the
    two by generating the same batch several times with one source replayed
    and the other varied:

    * **noise arm** -- `num_noises` batches sharing one decoding seed, each
      with a different noise seed. Matched rows differ *only* because of the
      noise, so `agree_noise` (their mean positional token agreement) is 1.0
      exactly when the noise channel is unused and falls as it starts to
      matter.
    * **decode arm** (control, `decode_control`) -- the mirror image:
      `num_noises` batches sharing one noise seed with different decoding
      seeds. `agree_decode` says how much diversity the sampler contributes on
      its own. With `sampling.temperature=0` and the confidence predictor
      decoding is deterministic, so this is 1.0 and *all* diversity is noise.
    * **floor** -- different rows of one batch are fully independent samples,
      and `agree_indep` is what their agreement happens to be (nonzero: two
      unrelated texts share ' the', ',', ...). Both arms are reported against
      it as `sensitivity = (1 - agree) / (1 - agree_indep)`: 0 => that source
      contributes nothing, 1 => varying it alone decorrelates samples as much
      as drawing an unrelated sample does.
    * **replay check** -- one extra batch with *both* seeds equal to the noise
      arm's first replicate. `agree_replay` must be ~1.0; anything lower means
      generation is not bit-reproducible on this device and the other numbers
      are correspondingly blurred, not evidence about the noise.

    Also reported: `distinct_per_pos`, the mean number of distinct tokens a
    position takes across the noise arm (1.0 .. `num_noises`), and
    `uniq_seq_frac`, the fraction of a row's noise replicates that are
    distinct sequences (`1/num_noises` .. 1.0) -- the coarse, readable version
    of the same question.

    Replaying a stream is exact for the ancestral predictors: they draw once
    per step from the global RNG no matter what the noise did, so the noise
    arm's decoding draws line up step for step. Two predictors can let a
    replayed stream drift, which is worth knowing before reading a borderline
    number: the confidence predictor takes a data-dependent number of steps,
    so at `sampling.temperature > 0` the noise arm's Gumbel draws can
    desynchronise, and `ancestral_cache` skips the denoiser call on a step
    that reveals nothing, so the decode arm's *noise* draws can. Drift blurs
    an arm towards the floor -- it inflates the measured contribution rather
    than hiding one.

    Costs `2 * num_noises + 1` sampling batches (`num_noises + 2` without the
    control). The global RNG state is saved and restored, so calling this
    during validation does not perturb the training stream.
    """
    eval_config = self.config.eval
    if num_samples is None:
      num_samples = (eval_config.get('noise_diversity_num_samples', None)
                     or self.config.loader.eval_batch_size)
    if num_noises is None:
      num_noises = int(eval_config.get('noise_diversity_num_noises', 4))
    if decode_control is None:
      decode_control = bool(
        eval_config.get('noise_diversity_decode_control', True))
    if seed is None:
      seed = int(eval_config.get('noise_diversity_seed', 0))
    assert num_noises >= 2, (
      f'eval.noise_diversity_num_noises must be >= 2 to compare noise '
      f'draws against each other, got {num_noises}.')
    # Each rank probes a different set of seeds, so the DDP-averaged numbers
    # cover more draws instead of recomputing the same ones.
    seed = int(seed) + 100003 * self.global_rank

    # Validation runs inside inference mode by default, where a captured RNG
    # state would itself be an inference tensor; take/restore it outside.
    with torch.inference_mode(False):
      cpu_rng_state = torch.random.get_rng_state().clone()
      cuda_rng_states = (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None)
    try:
      # Same decoding seed everywhere, one noise seed per replicate.
      noise_arm = torch.stack([
        self._sample_with_seeds(
          num_samples, decode_seed=seed, noise_seed=seed + 1 + k,
          num_steps=num_steps)
        for k in range(num_noises)])
      # Both seeds repeated: must reproduce `noise_arm[0]` token for token.
      replay = self._sample_with_seeds(
        num_samples, decode_seed=seed, noise_seed=seed + 1,
        num_steps=num_steps)
      decode_arm = None
      if decode_control:
        # Mirror image: one noise seed, a different decoding seed each time.
        decode_arm = torch.stack([
          self._sample_with_seeds(
            num_samples, decode_seed=seed + 1 + k, noise_seed=seed + 1,
            num_steps=num_steps)
          for k in range(num_noises)])
    finally:
      with torch.inference_mode(False):
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
          torch.cuda.set_rng_state_all(cuda_rng_states)

    agree_noise = _matched_agreement(noise_arm)
    agree_indep = _independent_agreement(noise_arm)
    report = {
      'noise_div/agree_noise': agree_noise,
      'noise_div/agree_replay': (replay == noise_arm[0]).float().mean().item(),
      'noise_div/distinct_per_pos': _distinct_per_position(noise_arm),
      'noise_div/uniq_seq_frac': _unique_sequence_frac(noise_arm),
      'noise_div/num_noises': float(num_noises)}
    # A single-row batch has no independent pair to normalise against; the raw
    # agreements are still meaningful, the ratios are simply unavailable.
    span = None if agree_indep is None else max(1.0 - agree_indep, 1e-8)
    if span is not None:
      report['noise_div/agree_indep'] = agree_indep
      report['noise_div/sensitivity'] = (1.0 - agree_noise) / span
    if decode_arm is not None:
      agree_decode = _matched_agreement(decode_arm)
      report['noise_div/agree_decode'] = agree_decode
      if span is not None:
        report['noise_div/decode_sensitivity'] = (1.0 - agree_decode) / span

    variants = None
    if return_variants and not self.is_vision:
      # Skipped for images: `batch_decode` gives tensors, and every consumer of
      # `variants` prints or tabulates them as text. The numbers above are
      # token-level and domain-agnostic, so the probe itself still applies.
      variants = [self.tokenizer.batch_decode(replicate)
                  for replicate in noise_arm]
    return NoiseDiversity(metrics=report, variants=variants)

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

  def _remask_update(self, x, t, dt, noise_removal_step=False):
    raise NotImplementedError(
      'Remasking (consistency-style) sampling is only defined for '
      'absorbing-state (masked) diffusion; set sampling.predictor to '
      'a sampler supported by this algo.')

  def _confidence_sampler(self, num_samples, num_steps, eps):
    raise NotImplementedError(
      'Confidence sampling is only defined for absorbing-state '
      '(masked) diffusion; set sampling.predictor to a sampler '
      'supported by this algo.')

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    if self.sampler == 'confidence':
      return self._confidence_sampler(
        num_samples, num_steps=num_steps, eps=eps)
    x = self.prior_sample(num_samples, self.num_tokens)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ancestral':
        p_x0, x_next = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=None)
        self._record_pred_entropy(p_x0, x)
        x = x_next
      elif self.sampler == 'ancestral_cache':
        # Only a cache miss actually runs the denoiser; on a hit `p_x0` is
        # last step's distribution and folding it in again would just
        # reweight the average by how long each call happened to be reused.
        is_fresh = p_x0_cache is None
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=p_x0_cache)
        if is_fresh:
          self._record_pred_entropy(p_x0_cache, x)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      elif self.sampler == 'remask':
        # No caching here: this update rewrites *every* position, so `x`
        # changes on essentially every step and a cached `p_x0` would be
        # conditioned on a latent that no longer exists.
        p_x0, x_next = self._remask_update(x=x, t=t, dt=dt)
        self._record_pred_entropy(p_x0, x)
        x = x_next
      else:
        x = self._analytic_update(x=x,t=t, dt=dt)

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                    device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
      if self.sampler == 'analytic':
        x = self._denoiser_update(x=x, t=t0)
      else:
        # Shared with `remask`: at `alpha_s = 1` its remask rate is pinned to
        # 0 by the admissibility bound, so its update degenerates to exactly
        # this one -- fill every remaining mask, keep what is already there.
        # (`p_x0_cache` is None for that predictor, so the call is fresh.)
        is_fresh = p_x0_cache is None
        p_x0, x_next = self._ancestral_update(x=x, t=t0, dt=None,
                                 p_x0=p_x0_cache,
                                 noise_removal_step=True)
        if is_fresh:
          self._record_pred_entropy(p_x0, x)
        x = x_next
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      log_p_x0 = self.forward(xt=x, sigma=sigma)
      self._record_pred_entropy(log_p_x0, x, log_input=True)
      x = log_p_x0.argmax(dim=-1)
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
    self._validate_remask_configuration()

  def _validate_remask_configuration(self):
    """Checks for `sampling.predictor='remask'`. Split out so subclasses
    that do not chain to `_validate_configuration` (e.g. `MDLM_with_AR`)
    can still run them -- otherwise a typo in `remask_strategy` would only
    surface at the first validation sampling, thousands of steps in."""
    if self.sampler != 'remask':
      return
    strategy = self.config.sampling.get('remask_strategy', 'consistency')
    assert strategy in {'consistency', 'scaled', 'cap'}, (
      f'Unknown sampling.remask_strategy={strategy!r}; expected one of '
      "'consistency', 'scaled', 'cap'.")
    if strategy in {'scaled', 'cap'}:
      eta = float(self.config.sampling.get('remask_eta', 1.0))
      assert 0.0 <= eta <= 1.0, (
        f'sampling.remask_eta must be in [0, 1], got {eta}.')
    assert self.config.sampling.noise_removal != 'none', (
      "sampling.predictor='remask' remasks tokens right up to the last "
      'step, so it needs a final full decode; set '
      "sampling.noise_removal to 'ancestral' or 'greedy'.")

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

  def _pred_entropy_mask(self, xt):
    # Revealed positions are pinned to a one-hot by `_process_model_output`,
    # so their entropy is 0 by construction; only the masked positions carry
    # an actual prediction.
    return xt == self.mask_index

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      p_x0 = self.forward(
        x, self._sigma_from_alphat(alpha_t)).exp()
    
    q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    _x = sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _remask_rate(self, alpha_t, alpha_s):
    """`sigma_t`: the probability that `_remask_update` sends an already
    revealed token back to `[MASK]` on this step. Shape follows `alpha_t`,
    i.e. `(batch, 1)`.

    Clipped to the largest admissible value `(1 - alpha_s) / alpha_t`, above
    which the masked branch's stay-probability would go negative -- there is
    simply not enough mask mass at `s` left to absorb the remasked tokens.
    """
    strategy = self.config.sampling.get('remask_strategy', 'consistency')
    eta = float(self.config.sampling.get('remask_eta', 1.0))
    sigma_max = ((1 - alpha_s)
                 / alpha_t.clamp_min(1e-9)).clamp(0.0, 1.0)
    if strategy == 'consistency':
      # sigma_t = 1 - alpha_s makes both branches of `_remask_update` equal,
      # i.e. the step forgets `x_t` entirely. Always <= sigma_max.
      sigma = 1 - alpha_s
    elif strategy == 'scaled':
      # Straight interpolation between the ancestral sampler (eta=0) and
      # 'consistency' (eta=1).
      sigma = eta * (1 - alpha_s)
    elif strategy == 'cap':
      # ReMDM's max-cap schedule: a constant rate, clipped where the bound
      # bites (which it does at both ends of the trajectory).
      sigma = torch.full_like(alpha_t, eta)
    else:
      raise ValueError(
        f'Unknown sampling.remask_strategy={strategy!r}; expected one of '
        "'consistency', 'scaled', 'cap'.")
    return torch.minimum(sigma.clamp_min(0.0), sigma_max)

  def _remask_update(self, x, t, dt, noise_removal_step=False):
    """One step of remasking ("consistency-style") reverse sampling.

    `_ancestral_update` only ever *adds* tokens: `copy_flag` freezes a
    position the moment it is revealed, so a token committed at step 3 on
    the strength of an almost fully masked context can never be revisited,
    and every such mistake is carried into the final sample. That is the
    dominant error at low step counts, where each step commits many
    positions at once from a weak context.

    This update instead decodes *every* position to a proposal `x0_hat` and
    then renoises, so a revealed token can go back to `[MASK]` and be
    re-decoded later under a much richer context. Per position, going from
    `t` to `s < t`:

      * revealed in `x_t`  -> `[MASK]` with probability `sigma_t`,
                              otherwise kept (`x0_hat` reproduces it, since
                              `_process_model_output` pins revealed
                              positions to a one-hot);
      * masked in `x_t`    -> `[MASK]` with probability
                              `beta = (1 - alpha_s - alpha_t sigma_t)
                                      / (1 - alpha_t)`,
                              otherwise a categorical draw from `p_x0`.

    `beta` is what keeps the expected masked fraction at `1 - alpha_s`
    exactly as the schedule prescribes (`alpha_t sigma_t
    + (1 - alpha_t) beta = 1 - alpha_s`), so `sampling.steps` means the
    same thing here as for the ancestral predictors and the two are
    directly comparable at equal step count. Two special values are worth
    naming (`sampling.remask_strategy`, see `_remask_rate`):

      * `sigma_t = 0` gives back `_ancestral_update` term for term;
      * `sigma_t = 1 - alpha_s` makes both branches identical, i.e. the step
        stops depending on `x_t` at all: decode the whole sequence, then
        renoise it to level `s`. This is the masked-diffusion analogue of
        multistep sampling in a continuous consistency model
        (`x0 = f(x_t, t)`, then `x_s = x0 + sigma_s eps`), and is the
        default.

    The catch, and the reason the rate is a knob rather than a constant:
    remasking is only worth it if the re-decode is better than what it
    threw away. It costs a full denoiser call per step with no `p_x0`
    caching, and under an aggressive rate a short trajectory can spend all
    its steps churning instead of converging. It also breaks the ELBO's
    derivation -- likelihood bounds computed with `_loss` do not describe
    this sampler.
    """
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    assert alpha_t.ndim == 2

    p_x0 = self.forward(
      x, self._sigma_from_alphat(alpha_t)).exp()
    if self.p_nucleus < 1:
      # Truncate only the draw; the returned `p_x0` (and so `val/pred_
      # entropy`) keeps reading the model's own distribution.
      sample_probs = utils.top_k_top_p_filtering(
        p_x0.log(), top_p=self.p_nucleus).softmax(-1)
    else:
      sample_probs = p_x0
    x0_hat = sample_categorical(sample_probs)

    sigma_t = self._remask_rate(alpha_t, alpha_s)
    beta = ((1 - alpha_s - alpha_t * sigma_t)
            / (1 - alpha_t).clamp_min(1e-9)).clamp(0.0, 1.0)
    # (batch, 1) rates broadcast against the (batch, length) predicate.
    mask_prob = torch.where(x == self.mask_index, beta, sigma_t)
    remask = torch.rand(
      x.shape, device=x.device) < mask_prob
    x_next = torch.where(remask, self.mask_index, x0_hat)
    if self.ignore_bos:
      x_next[:, 0] = x[:, 0]
    return p_x0, x_next

  def _denoise_x0(self, x):
    """One denoiser call on the current partially-masked `x`.

    Returns `(x0, prob)`: the proposed clean token at every position and
    the probability the denoiser assigns to it. The time fed to the
    network is read off `x` itself -- the fraction of positions still
    masked is exactly `1 - alpha_t` under the absorbing forward process
    -- rather than taken from a fixed grid, because confidence decoding
    reveals a data-dependent number of tokens per step (and, with
    blocks, leaves later blocks fully masked). For a model with
    `algo.time_conditioning=False` (e.g. MDLM) `_process_sigma` zeroes
    this out anyway.

    The token is greedy by default (`sampling.temperature = 0`), as in
    the LLaDA / dInfer threshold decoders: the confidence the threshold
    tests is then `max_v p(v)`, a clean statement that this position's
    conditional is already nearly deterministic, which is exactly what
    licenses committing several positions in the same step. Sampling the
    token instead can propose an off-mode candidate whose probability
    then fails the threshold, so the position is deferred and re-drawn
    -- temperature and the threshold work against each other.

    A positive `sampling.temperature` switches to Gumbel-max, which is
    the usual softmax temperature (`argmax(log p + T g)` has the same
    argmax as `argmax(log p / T + g)`, i.e. a draw from
    `softmax(log p_x0 / T)`); 1.0 is an exact categorical draw from
    `p_x0`, the draw `sample_categorical` makes in the ancestral
    sampler. That is worth paying for in *unconditional* generation:
    with T=0 there is no prompt, `prior_sample` gives every row the same
    all-mask input and the rest of this sampler is deterministic, so a
    batch collapses to one identical sequence -- unless the student
    injects input noise (`adversarial_distill.noise_dim > 0`), which
    supplies the diversity itself.

    `sampling.p_nucleus` additionally restricts the draw to the nucleus.
    Both knobs affect only *which* token is proposed: the returned
    probability is always read off the unmodified `p_x0`, so the
    threshold in `_confidence_sampler` sees the model's real probability
    and raising the temperature does not loosen it.
    """
    t = (x == self.mask_index).to(torch.float32).mean(
      -1, keepdim=True).clamp(min=self.sampling_eps, max=1.0)
    _, alpha_t = self.noise(t)
    log_p_x0 = self.forward(x, self._sigma_from_alphat(alpha_t))
    self._record_pred_entropy(log_p_x0, x, log_input=True)
    logits = log_p_x0
    if self.p_nucleus < 1:
      # `top_k_top_p_filtering` writes in place; keep `log_p_x0` intact.
      logits = utils.top_k_top_p_filtering(
        logits.clone(), top_p=self.p_nucleus)
    temperature = float(
      self.config.sampling.get('temperature', 0.0))
    if temperature > 0:
      noise = torch.rand_like(logits)
      gumbel = - torch.log(- torch.log(noise + 1e-10) + 1e-10)
      x0 = (logits + temperature * gumbel).argmax(-1)
    else:
      x0 = logits.argmax(-1)
    prob = log_p_x0.gather(-1, x0[..., None]).squeeze(-1).exp()
    return x0, prob

  @torch.no_grad()
  def _confidence_sampler(self, num_samples, num_steps=None,
                          eps=1e-5):
    """Confidence-threshold decoding, as used by block-diffusion models.

    The ancestral sampler (`_ancestral_update`) unmasks a *random*
    subset of positions per step -- each masked position flips with
    probability `(alpha_s - alpha_t) / (1 - alpha_t)` regardless of what
    the denoiser thinks -- and it is the sampler the ELBO is derived
    for. This one instead commits positions by the denoiser's own
    probability, the MaskGIT / LLaDA / Fast-dLLM rule. It is not the
    reverse of the forward masking process, so likelihood bounds no
    longer apply, but it removes most of the factorised-independence
    error at low step counts: the positions committed together are the
    ones whose conditionals are already nearly deterministic, which is
    exactly when treating them as independent is harmless.

    Reveal rule, per row and per step: unmask **every** still-masked
    position whose probability `p_x0[i, x0_i]` is `>=
    sampling.confidence_threshold`, but **at least
    `sampling.min_reveal` (k)** of them -- the top-k by probability --
    so a step where nothing clears the bar still makes progress and the
    loop cannot stall. The number of denoiser calls is therefore
    data-dependent, not fixed: an easy sequence finishes in a handful
    while a hard one falls back to ~`L / k` steps. A threshold above 1
    gives pure top-k-per-step decoding; `min_reveal` = the block length
    gives one-shot decoding.

    Two knobs shape the schedule:

    * `sampling.block_size` (0 => the whole sequence, i.e. no blocking)
      decodes left-to-right one block at a time, finishing a block
      before touching the next. That is the semi-autoregressive schedule
      of block diffusion; with `block_size=1` it degenerates to pure
      left-to-right AR decoding, and with the default it is LLaDA-style
      whole-sequence confidence decoding.
    * `num_steps` (`sampling.steps`) is only a **budget cap** here,
      spread evenly over the blocks. Threshold decoding sets its own
      pace, so the cap normally does not bind; if a block does hit it,
      its remaining masks are revealed in one shot rather than leaving
      mask tokens in the output.

    Already-revealed tokens are never revisited, exactly as in the
    ancestral sampler.
    """
    del eps
    if num_steps is None:
      num_steps = self.config.sampling.steps
    block_size = int(
      self.config.sampling.get('block_size', 0) or self.num_tokens)
    assert self.num_tokens % block_size == 0, (
      f'sampling.block_size={block_size} must divide the sequence '
      f'length {self.num_tokens}.')
    num_blocks = self.num_tokens // block_size
    threshold = float(
      self.config.sampling.get('confidence_threshold', 0.9))
    min_reveal = int(self.config.sampling.get('min_reveal', 1))
    assert min_reveal >= 1, (
      'sampling.min_reveal must be >= 1, otherwise a step that clears '
      'no position would leave the sampler stuck.')
    # Budget cap per block; >= 1 so every block gets at least one call.
    max_steps = max(1, num_steps // num_blocks)

    x = self.prior_sample(num_samples, self.num_tokens)
    for block in range(num_blocks):
      lo, hi = block * block_size, (block + 1) * block_size
      for step in range(max_steps):
        block_masked = (x[:, lo:hi] == self.mask_index)
        num_masked = block_masked.sum(-1)
        if not bool(num_masked.any()):
          break

        x0, prob = self._denoise_x0(x)
        # Only still-masked positions inside the active block compete.
        eligible = torch.zeros_like(x, dtype=torch.bool)
        eligible[:, lo:hi] = block_masked
        prob = prob.masked_fill(~ eligible, - float('inf'))

        # Everything over the bar, but never fewer than k and never
        # more than what is actually left masked in this block.
        num_reveal = torch.clamp(
          (prob >= threshold).sum(-1),
          min=min_reveal).minimum(num_masked)
        if step == max_steps - 1:
          # Budget exhausted: finish the block in one shot.
          num_reveal = num_masked
        # Per-row top-`num_reveal` via ranks (the count varies by row).
        # Thresholding keeps exactly the highest-probability positions,
        # so one ranked selection covers both branches of the rule.
        ranks = prob.argsort(dim=-1, descending=True).argsort(dim=-1)
        reveal = (ranks < num_reveal[:, None]) & eligible
        x = torch.where(reveal, x0, x)
    return x

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
