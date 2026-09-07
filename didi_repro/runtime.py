"""Explicit runtime/checkpoint repairs around pristine upstream source.

No objective, training hyperparameter, noise schedule, or sampling equation is
replaced here. See README.md for the complete deviation ledger. Source edits
are applied in memory with exact-match checks, so the vendored files remain
byte-identical and each repaired method can be compared against its original.
"""

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

from . import UPSTREAM_REVISION

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / 'vendor' / 'didi_instruct'


def verify_upstream():
  manifest = json.loads((Path(__file__).with_name('UPSTREAM.json')).read_text())
  if manifest['commit'] != UPSTREAM_REVISION:
    raise RuntimeError('DiDi upstream revision/manifest mismatch')
  for name, expected in manifest['files'].items():
    path = UPSTREAM_ROOT / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
      raise RuntimeError(f'Vendored DiDi source changed: {path}')
  return manifest


def repaired_function(function, replacements):
  """Compile an upstream function with only the listed, exact source edits."""
  source = textwrap.dedent(inspect.getsource(function))
  for before, after in replacements:
    if source.count(before) != 1:
      raise RuntimeError(f'Upstream patch no longer matches {function.__name__}: {before!r}')
    source = source.replace(before, after, 1)
  namespace = dict(function.__globals__)
  exec(compile(source, f'<didi-repair:{function.__name__}>', 'exec'), namespace)
  result = namespace[function.__name__]
  result.__didi_repaired_source__ = source
  return result


TRAINING_REPAIRS = (
  ('tau_gen, _, sigma_gen, pi_tau = self._get_tau(batch_size)',
   'tau_gen, alpha_gen, sigma_gen, pi_tau = self._get_tau(batch_size)'),
  ('self._get_corrupted_inputs(batch_size, tau_gen, None, sigma_gen)',
   'self._get_corrupted_inputs(batch_size, tau_gen, alpha_gen, sigma_gen)'),
  # Diagnostic typo only: match the scalar actually differentiated above.
  ('total_student_loss = loss_pg + loss_kl - self.entropy_beta * entropy',
   'total_student_loss = loss_pg + self.kl_beta * loss_kl - self.entropy_beta * entropy'),
)


def install_runtime(upstream_main):
  """Install repairs and return the repaired class; original modules stay intact."""
  import torch
  original = upstream_main.algo.DiDiInstruct
  if getattr(original, '_didi_runtime_installed', False):
    return original
  training_step = repaired_function(original.training_step, TRAINING_REPAIRS)
  # Same input gradient; unlike backward(), autograd.grad does not leave
  # discriminator parameter gradients behind during guided evaluation.
  guided_step = repaired_function(original._get_gradient_tilting_step, (
    ('reward.backward()\n      grad = log_p_x0_for_grad.grad',
     'grad = torch.autograd.grad(reward, log_p_x0_for_grad)[0]'),))

  class ReproductionDiDi(original):
    _didi_runtime_installed = True

    def __init__(self, config, tokenizer):
      if config.algo.backbone != 'dit':
        raise ValueError('Strict DiDi reproduction uses upstream algo.backbone=dit '
                         'and a native teacher .ckpt, not the IDLM HF wrapper.')
      if int(config.trainer.accumulate_grad_batches) != 1:
        raise ValueError('Upstream DiDi has no gradient accumulation; keep '
                         'trainer.accumulate_grad_batches=1 for reproduction.')
      if int(config.algo.num_samples_per_prompt) < 2:
        raise ValueError('Upstream sample-standard-deviation reward normalization needs G >= 2.')
      super().__init__(config, tokenizer)
      # Lightning otherwise rejects two optimizers before the first batch.
      self.automatic_optimization = False
      # Missing in the published DiDi YAML; the upstream MDLM value.
      self.loss_type = getattr(config.algo, 'loss_type', 'elbo')
      self._didi_in_training_step = False
      self._didi_teacher_metadata = None
      self._didi_student_metadata = None
      self._didi_loading_checkpoint = False
      effective_names = (
        'discriminator_lr', 'kl_beta', 'entropy_beta', 'sampling_eps',
        'label_smoothing', 'discriminator_warmup_steps', 'student_update_every',
        'remask_prob', 'reward_clip_val', 'log_odds_eps', 'num_samples_per_prompt',
        'advantage_clip_val', 'grad_clip_value', 'omega_min', 'omega_max',
        'tau_mode', 'tau_min_eps', 't0_eps', 'logprob_mode', 'kl_hi_coef',
        'kl_lo_coef', 'importance_weight', 'num_candidates',
        'guidance_scale_start', 'guidance_scale_end', 'rerank_steps_ratio',
        'ema_beta', 'loss_type')
      self._didi_effective_hyperparameters = {
        name: getattr(self, name) for name in effective_names}
      self._didi_effective_hyperparameters.update({
        'scheduler_implementation': 'linear warmup then linear decay, floor=1e-8',
        'reward_std_correction': 1,
        'evaluation_weights': 'backbone',
        'noise_eps': self.noise.eps,
      })

    def initialize_teacher_checkpoint(self, checkpoint, source_path=None):
      """Load native raw MDLM weights into every copy created by __init__."""
      state = checkpoint.get('state_dict', {})
      backbone = {k[len('backbone.'):]: v for k, v in state.items()
                  if k.startswith('backbone.')}
      if not backbone:
        raise ValueError('Teacher checkpoint must contain native MDLM backbone.* '
                         'weights. An HF model ID or DiDi student_ema-only export '
                         'is not the upstream teacher training checkpoint.')
      # Fail on missing/unexpected tensors rather than silently training random
      # layers. The spectral-normalized discriminator head is newly initialized
      # upstream and must NOT be replaced with the teacher vocabulary head.
      self.backbone.load_state_dict(backbone, strict=True)
      self.teacher.load_state_dict(backbone, strict=True)
      self.student_ema.load_state_dict(backbone, strict=True)
      disc_state = {k: v for k, v in backbone.items()
                    if not k.startswith('output_layer.')}
      result = self.discriminator.load_state_dict(disc_state, strict=False)
      if result.unexpected_keys or any(
          not key.startswith('output_layer.') for key in result.missing_keys):
        raise ValueError(f'Teacher/discriminator architecture mismatch: {result}')
      if self.ema is not None:
        # This legacy EMA is saved upstream, but its eval override does not use
        # it. Keep it initialized consistently without changing EMA behavior.
        self.ema.shadow_params = [p.detach().clone() for p in self._get_parameters()
                                  if p.requires_grad]
      metadata = {'path': str(source_path), 'weights': 'raw backbone.*'}
      if source_path is not None:
        digest = hashlib.sha256()
        with open(source_path, 'rb') as stream:
          for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        metadata['sha256'] = digest.hexdigest()
      self._didi_teacher_metadata = metadata
      self._didi_student_metadata = None

    def training_step(self, batch, batch_idx):
      if self._didi_teacher_metadata is None:
        raise RuntimeError('DiDi teacher weights were not initialized from a checkpoint.')
      self._didi_in_training_step = True
      try:
        result = training_step(self, batch, batch_idx)
        self.log('didi/iteration', float(self.global_steps), on_step=True,
                 on_epoch=False, sync_dist=True)
        return result
      finally:
        self._didi_in_training_step = False

    def _get_rewards(self, *args, **kwargs):
      result = super()._get_rewards(*args, **kwargs)
      final, mean, std, clipped, advantage = result
      self.log_dict({
        'reward/mean': mean,
        'reward/std': std,
        'reward/clipped_fraction': clipped,
        'reward/advantage_abs_mean': advantage.abs().mean(),
        'reward/advantage_std': advantage.std(),
        'reward/normalized_abs_mean': final.abs().mean(),
      }, on_step=True, on_epoch=False, sync_dist=True)
      return result

    _get_gradient_tilting_step = guided_step

    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
      steps = self.config.sampling.steps if num_steps is None else num_steps
      if self.sampler == 'guided' and int(steps) < 2:
        raise ValueError('Upstream guided schedule divides by steps-1; use at least 2 steps.')
      # Lightning validation defaults to inference_mode; the original guided
      # sampler needs input gradients. Its own @no_grad still controls ordinary
      # generation and torch.enable_grad still controls only the tilting step.
      with torch.inference_mode(False):
        return super().generate_samples(num_samples, num_steps, eps)

    def on_save_checkpoint(self, checkpoint):
      # Upstream removes the teacher and rewrites manual progress from the
      # automatic optimizer loop. Preserve model/optimizer/schedule progress;
      # this does not promise bitwise data/RNG replay on a mid-epoch resume.
      # Retain Lightning's actual progress and ALL model state instead.
      if self.ema is not None:
        checkpoint['ema'] = self.ema.state_dict()
      checkpoint['didi_repro'] = {
        'upstream_revision': UPSTREAM_REVISION,
        'next_iteration': self.global_steps + int(self._didi_in_training_step),
        'teacher': self._didi_teacher_metadata,
        'student_initialization': self._didi_student_metadata,
        'evaluation_weights': 'backbone',
      }

    def on_load_checkpoint(self, checkpoint):
      state = checkpoint.get('state_dict', {})
      metadata = checkpoint.get('didi_repro')
      self._didi_student_metadata = (metadata.get('student_initialization')
                                     if metadata is not None else None)
      if self.config.mode == 'train':
        if metadata is None or not any(k.startswith('teacher.') for k in state):
          raise ValueError('Training resume needs a didi_repro checkpoint '
                           'with frozen teacher and iteration state. For a fresh '
                           'run, supply the teacher via training.finetune_path.')
        if metadata['upstream_revision'] != UPSTREAM_REVISION:
          raise ValueError('Resume checkpoint uses a different upstream revision.')
        self.global_steps = int(metadata['next_iteration'])
        self._didi_teacher_metadata = metadata['teacher']
      else:
        # Official HF converter emits ONLY student_ema.* for the generator.
        # Upstream sampling reads backbone.*, so populate that otherwise-random
        # copy only when there are genuinely no raw generator weights.
        if not any(k.startswith('backbone.') for k in state):
          ema_weights = {k[len('student_ema.'):]: v for k, v in state.items()
                         if k.startswith('student_ema.')}
          if not ema_weights:
            raise ValueError('Evaluation checkpoint has no generator weights.')
          state.update({f'backbone.{k}': v for k, v in ema_weights.items()})
        if self.sampler == 'guided' and not any(k.startswith('discriminator.') for k in state):
          raise ValueError('Guided evaluation needs trained discriminator weights.')
      if self.ema is not None and 'ema' in checkpoint:
        self.ema.load_state_dict(checkpoint['ema'])
      self._didi_loading_checkpoint = True

    def load_state_dict(self, state_dict, strict=True, **kwargs):
      # Unlike upstream strict=False, reject missing learned weights. For an
      # evaluation-only export the unused teacher/EMA copies may be absent.
      result = super().load_state_dict(state_dict, strict=False, **kwargs)
      if self._didi_loading_checkpoint:
        required = ('backbone.', 'discriminator.') if self.sampler == 'guided' else ('backbone.',)
        if self.config.mode == 'train':
          required = ('backbone.', 'teacher.', 'discriminator.', 'student_ema.')
        missing = [k for k in result.missing_keys if k.startswith(required)]
        if missing or result.unexpected_keys:
          raise ValueError(f'Incomplete DiDi checkpoint: missing={missing}, '
                           f'unexpected={result.unexpected_keys}')
        self._didi_loading_checkpoint = False
      elif strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(str(result))
      return result

  # A top-level, stable class name permits checkpoint hyperparameter pickling.
  ReproductionDiDi.__name__ = 'ReproductionDiDi'
  ReproductionDiDi.__qualname__ = 'ReproductionDiDi'
  globals()['ReproductionDiDi'] = ReproductionDiDi
  upstream_main.algo.DiDiInstruct = ReproductionDiDi
  return ReproductionDiDi


def load_upstream():
  """Load original flat imports in a dedicated process, never mix IDLM modules."""
  verify_upstream()
  for name in ('main', 'algo', 'models', 'trainer_base', 'dataloader', 'metrics', 'utils'):
    module = sys.modules.get(name)
    if module is not None:
      path = getattr(module, '__file__', '') or ''
      if not Path(path).resolve().is_relative_to(UPSTREAM_ROOT):
        raise RuntimeError(f'Conflicting {name} module; launch python -m didi_repro '
                           'in its own process, not inside IDLM main.py.')
  sys.path.insert(0, str(UPSTREAM_ROOT))
  upstream = importlib.import_module('main')
  install_runtime(upstream)
  return upstream
