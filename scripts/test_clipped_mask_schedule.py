"""CPU regressions for TinyGSM's fixed clipped generator mask schedule.

Run: python scripts/test_clipped_mask_schedule.py
Uses actual training methods with tiny denoisers; no checkpoints are needed.
"""
import ast
import copy
import os
import sys
import warnings
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mask_schedule import ClippedMaskSchedule

SAMPLING_EPS = 0.001
NOISE_EPS = 0.001


class Config(SimpleNamespace):
  def get(self, name, default=None):
    return getattr(self, name, default)


def config(mode='clipped', minimum=0.45, maximum=0.95, **overrides):
  values = dict(is_distill=True, high_mask_fraction=0.0,
                high_mask_min=0.7, high_mask_max=1.0,
                dmd_weight=1.0, mmd_reward_weight=0.0,
                accum_batches=2, fake_frequency=1)
  if mode is not None:
    values['mask_schedule'] = {'mode': mode, 'min': minimum, 'max': maximum}
  values.update(overrides)
  return Config(**values)


def schedule(cfg=None, sampling_eps=SAMPLING_EPS, noise_eps=NOISE_EPS):
  return ClippedMaskSchedule.from_config(
    config() if cfg is None else cfg, sampling_eps, noise_eps)


def _owt_reference(base, minimum, maximum, sampling_eps, noise_eps):
  """Independent OWT fixed-mode formula, without a sibling-repo dependency."""
  lower = max(minimum, sampling_eps * (1 - noise_eps))
  upper = min(maximum, 1 - noise_eps)
  quantiles = ((base - sampling_eps) / (1 - sampling_eps)).clamp(0, 1)
  return ((lower + (upper - lower) * quantiles) / (1 - noise_eps)).clamp(
    sampling_eps, 1.0)


def _tree(filename):
  with open(os.path.join(ROOT, filename), encoding='utf-8') as source:
    return ast.parse(source.read(), filename=filename)


def _method(tree, class_name, method_name):
  parent = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name)
  return copy.deepcopy(next(node for node in parent.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == method_name))


def _production_training_parts():
  # Avoid importing the full Lightning/model/tokenizer environment. These are
  # the real sampler, corruption and MDLM loss methods, not rewritten stubs.
  tree = _tree('trainer_base.py')
  nodes = [copy.deepcopy(node) for node in tree.body
           if isinstance(node, (ast.ClassDef, ast.FunctionDef))
           and node.name in ('LogLinear', 'sample_categorical')]
  for parent, name in (('Diffusion', '_sample_t'),
                       ('Diffusion', '_alpha_to_t'),
                       ('Diffusion', '_sigma_from_alphat'),
                       ('Diffusion', '_process_sigma'),
                       ('AbsorbingState', 'q_xt')):
    nodes.append(_method(tree, parent, name))
  namespace = {'torch': torch}
  exec(compile(ast.Module(body=nodes, type_ignores=[]), 'trainer_base.py', 'exec'),
       namespace)
  return namespace


PARTS = _production_training_parts()


class RecordingNoise(PARTS['LogLinear']):
  def __init__(self):
    super().__init__()
    self.times = []

  def forward(self, t):
    self.times.append(t.detach().clone())
    return super().forward(t)


class TinyDenoiser(torch.nn.Module):
  def __init__(self):
    super().__init__()
    generator = torch.Generator().manual_seed(61)
    self.table = torch.nn.Parameter(torch.randn(8, 8, generator=generator))
    self.calls = []

  def forward(self, xt, sigma):
    self.calls.append((xt.detach().clone(), sigma.detach().clone(),
                       torch.is_grad_enabled()))
    return self.table[xt] + sigma.reshape(-1, 1, 1) * 0.03


class RuntimeBase(torch.nn.Module):
  def __init__(self, cfg, tokenizer):
    super().__init__()
    del tokenizer
    self.config = cfg
    self.sampling_eps = SAMPLING_EPS
    self.noise = RecordingNoise()
    self.backbone = TinyDenoiser()
    self.ema = None
    self.sampler = 'ancestral_cache'
    self.mask_index = 7
    self.ignore_bos = True
    self.neg_infinity = -1e6
    self.time_conditioning = True
    self.antithetic_sampling = True
    self.trainer = SimpleNamespace(num_nodes=1, node_rank=0,
                                   num_devices=1, local_rank=0)
    self.base_times = []
    self.corruptions = []

  @property
  def device(self):
    return self.backbone.table.device

  def _sample_t(self, n, accumulation_step):
    t = PARTS['_sample_t'](self, n, accumulation_step)
    self.base_times.append(t.detach().clone())
    return t

  _alpha_to_t = PARTS['_alpha_to_t']
  _sigma_from_alphat = PARTS['_sigma_from_alphat']
  _process_sigma = PARTS['_process_sigma']

  def q_xt(self, x0, alpha_t, valid_tokens=None):
    xt = PARTS['q_xt'](self, x0, alpha_t, valid_tokens)
    self.corruptions.append(xt.detach().clone())
    return xt

  def forward(self, xt, sigma):
    sigma = self._process_sigma(sigma)
    return self._process_model_output(self.backbone(xt, sigma), xt, sigma)


def _set_trainable(model, enabled):
  model.requires_grad_(enabled)


def _runtime_class():
  tree = _tree('algo.py')
  names = ('__init__', '_validate_configuration', '_needs_fake_phase',
           '_process_model_output', '_preserve_prompt_probs',
           '_preserve_prompt_tokens', '_sample_distill_generator_t',
           'multistep_generation', 'q_xt_with_probs',
           'nll_per_token_with_probs', 'generator_loss', 'fake_loss')
  parent = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'MDLM')
  parent = copy.deepcopy(parent)
  parent.bases = [ast.Name(id='RuntimeBase', ctx=ast.Load())]
  parent.body = [_method(tree, 'MDLM', name) for name in names]
  namespace = {'torch': torch, 'F': F, 'copy': copy, 'RuntimeBase': RuntimeBase,
               'ClippedMaskSchedule': ClippedMaskSchedule,
               'utils': SimpleNamespace(
                 freeze_model=lambda model: _set_trainable(model, False),
                 activate_model=lambda model: _set_trainable(model, True)),
               'trainer_base': SimpleNamespace(
                 sample_categorical=PARTS['sample_categorical'])}
  module = ast.fix_missing_locations(ast.Module(body=[parent], type_ignores=[]))
  exec(compile(module, 'algo.py', 'exec'), namespace)
  return namespace['MDLM'], namespace['trainer_base']


MDLM, TRAINER_STUB = _runtime_class()


def runtime(mode='clipped', **overrides):
  cfg = Config(adversarial_distill=config(mode=mode, **overrides),
               training=Config(continue_teacher=False),
               loader=Config(global_batch_size=8))
  return MDLM(cfg, tokenizer=None)


def test_default_original_and_clipped_modes_consume_no_additional_rng():
  for mode in (None, 'original', 'clipped'):
    model = runtime(mode)
    assert model._mask_schedule.mode == ('original' if mode is None else mode)
    torch.manual_seed(72)
    base = model._sample_t(4, 0)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(72)
    actual = model._sample_distill_generator_t(4, 0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    expected = base if mode != 'clipped' else _owt_reference(
      base, 0.45, 0.95, SAMPLING_EPS, NOISE_EPS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  base = torch.linspace(SAMPLING_EPS, 1, 32)
  assert schedule(config(mode=None)).sample(base) is base


def test_original_mode_preserves_legacy_high_mask_values_and_rng():
  for fraction in (0.0, 0.5, 1.0):
    model = runtime('original', high_mask_fraction=fraction,
                    high_mask_min=0.7, high_mask_max=0.9)
    torch.manual_seed(19)
    base = model._sample_t(4, 0)
    if fraction:
      probabilities = 0.7 + 0.2 * torch.rand_like(base)
      high = ((1 - (1 - probabilities)) / (1 - NOISE_EPS)).clamp(1e-5, 1)
      expected = torch.where(torch.rand_like(base) < fraction, high, base)
    else:
      expected = base
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    actual = model._sample_distill_generator_t(4, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    # Old lightweight callers and checkpoints need not attach schedule state.
    del model._mask_schedule
    torch.manual_seed(19)
    torch.testing.assert_close(model._sample_distill_generator_t(4, 0), expected,
                               rtol=0, atol=0)


def test_clipped_matches_owt_formula_for_supported_intervals_and_dtypes():
  for dtype in (torch.float32, torch.float64):
    base = torch.linspace(SAMPLING_EPS, 1.0, 1001, dtype=dtype)
    for lower, upper in ((0.45, 0.95), (0.1, 0.6), (0.0, 1.0), (0.5, 1.0)):
      sched = schedule(config(minimum=lower, maximum=upper))
      before = torch.get_rng_state()
      actual = sched.sample(base)
      expected = _owt_reference(base, lower, upper, SAMPLING_EPS, NOISE_EPS)
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)
      assert actual.dtype == dtype and actual.device == base.device
      assert torch.equal(torch.get_rng_state(), before)
      probability = actual * (1 - NOISE_EPS)
      torch.testing.assert_close(probability[[0, -1]],
                                 torch.tensor(sched.bounds, dtype=dtype))


def test_interval_intersection_does_not_create_endpoint_probability_mass():
  sched = schedule(config(minimum=0, maximum=1), sampling_eps=0.1, noise_eps=0.2)
  quantiles = (torch.arange(10000, dtype=torch.float64) + 0.5) / 10000
  base = 0.1 + 0.9 * quantiles
  probability = sched.sample(base) * 0.8
  assert probability.min() > 0.08 and probability.max() < 0.8
  assert torch.all(probability.diff() > 0)
  torch.testing.assert_close(probability.mean(), torch.tensor(0.44).double(),
                             rtol=0, atol=1e-8)


def test_rank_node_and_accumulation_strata_are_preserved():
  model = runtime()
  model.config.loader.global_batch_size = 48
  model.accum_batches = 3
  model.trainer.num_nodes = 2
  model.trainer.num_devices = 2
  torch.manual_seed(36)
  global_base = PARTS['_sample_t'](model, 48, None)
  expected = schedule().sample(global_base)
  results = []
  for node in range(2):
    model.trainer.node_rank = node
    for rank in range(2):
      model.trainer.local_rank = rank
      for accumulation in range(3):
        torch.manual_seed(36)
        results.append(model._sample_distill_generator_t(4, accumulation))
  actual = torch.cat(results)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  normalized = (actual * (1 - NOISE_EPS) - 0.45) / 0.5
  assert torch.equal((normalized * 48).floor().long(), torch.arange(48))


def test_invalid_configs_fail_during_schedule_initialization():
  cases = [config(mode='adaptive'), config(mode='typo'),
           config(minimum=-0.1), config(maximum=1.1),
           config(minimum=0.95, maximum=0.45),
           config(minimum=0.45, maximum=0.45),
           config(minimum=float('nan')), config(maximum=float('inf')),
           config(minimum=0.9995, maximum=1.0),
           config(high_mask_fraction=0.2), config(generator_force_full_mask=True),
           config(is_distill=False), config(mask_schedule='clipped')]
  for cfg in cases:
    try:
      schedule(cfg)
    except ValueError:
      pass
    else:
      raise AssertionError(f'invalid configuration accepted: {cfg}')
  for sampling_eps, noise_eps in ((0, 0.001), (1, 0.001),
                                  (0.001, -0.1), (0.001, 1)):
    try:
      schedule(sampling_eps=sampling_eps, noise_eps=noise_eps)
    except ValueError:
      pass
    else:
      raise AssertionError('invalid noise support accepted')
  try:
    runtime(high_mask_fraction=0.2)
  except ValueError as error:
    assert 'high_mask_fraction' in str(error)
  else:
    raise AssertionError('MDLM constructor accepted conflicting schedules')


def test_dmd_student_and_fake_phases_preserve_prompts_and_original_critic_noise():
  x0 = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 2, 1, 5, 3, 4],
                     [0, 3, 2, 1, 5, 4], [0, 4, 3, 2, 1, 5]])
  valid = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0]])
  protected = ~valid.bool()
  clean_probs = F.one_hot(x0, 8).float()
  for phase in ('generator_loss', 'fake_loss'):
    model = runtime()
    with torch.no_grad():
      model.fake_model.table.add_(torch.linspace(-0.4, 0.4, 8))
    model.backbone.requires_grad_(phase == 'generator_loss')
    model.fake_model.requires_grad_(phase == 'fake_loss')
    sampled_probs = []

    def record_sample(probs):
      sampled_probs.append(probs.detach().clone())
      return PARTS['sample_categorical'](probs)

    original_x0, original_valid = x0.clone(), valid.clone()
    torch.manual_seed(49)
    with mock.patch.object(TRAINER_STUB, 'sample_categorical', side_effect=record_sample):
      loss = getattr(model, phase)(x0, valid, current_accumulation_step=0)
    assert loss.shape == (4,) and torch.isfinite(loss).all()
    loss.mean().backward()
    trained = model.backbone if phase == 'generator_loss' else model.fake_model
    frozen = model.fake_model if phase == 'generator_loss' else model.backbone
    assert trained.table.grad is not None and trained.table.grad.abs().sum() > 0
    assert frozen.table.grad is None and model.teacher.table.grad is None
    assert len(model.base_times) == len(model.noise.times) == 2
    torch.testing.assert_close(model.noise.times[0],
                               model._mask_schedule.sample(model.base_times[0]))
    torch.testing.assert_close(model.noise.times[1], model.base_times[1], rtol=0, atol=0)
    assert (model.noise.times[1] * (1 - NOISE_EPS) < 0.45).any()
    assert model.backbone.calls[0][2] == (phase == 'generator_loss')
    assert len(sampled_probs) == 1
    torch.testing.assert_close(sampled_probs[0][protected], clean_probs[protected],
                               rtol=0, atol=0)
    for denoiser in (model.backbone, model.teacher, model.fake_model):
      for tokens, _, _ in denoiser.calls:
        assert torch.equal(tokens[protected], x0[protected])
    assert torch.equal(x0, original_x0) and torch.equal(valid, original_valid)
    assert not any('mask_schedule' in key for key in model.state_dict())


def test_generator_return_xt_preserves_prompt_probabilities_and_ignored_bos():
  model = runtime()
  x0 = torch.tensor([[0, 1, 2, 3, 4, 5]]).expand(4, -1).clone()
  valid = torch.tensor([[1, 0, 1, 1, 0, 1]]).expand(4, -1).clone()
  with mock.patch.object(model, 'forward', return_value=torch.full((4, 6, 8), -2.1)):
    # A deliberately imperfect forward makes the explicit prompt restoration
    # observable independently of MDLM's observed-token parameterization.
    probs, xt = model.multistep_generation(x0, valid, 0, return_xt=True)
  protected = ~valid.bool()
  torch.testing.assert_close(probs[protected], F.one_hot(x0, 8).float()[protected],
                             rtol=0, atol=0)
  assert torch.equal(xt[protected], x0[protected])
  assert torch.equal(xt[:, 0], x0[:, 0])


def main():
  torch.set_num_threads(1)
  # Production DMD requests CUDA FP32 autocast; CUDA is deliberately absent.
  warnings.filterwarnings('ignore', message='User provided device_type of.*cuda.*')
  for name, function in sorted(globals().items()):
    if name.startswith('test_'):
      function()
      print(f'PASS {name}')
  print('\nall TinyGSM clipped mask schedule tests passed')


if __name__ == '__main__':
  main()
