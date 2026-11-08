"""Offline CPU checks for the opt-in BD3-LM-style masking schedule.

Run with ``python scripts/test_clipped_mask_schedule.py``.  These checks use
only PyTorch and the schedule module: no model download, GPU, or pytest.
Add ``--distributed`` to also run the bounded, two-process CPU/Gloo check.
"""
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_schedule import (ClippedMaskSchedule,  # noqa: E402
                           DistillMaskScheduleMixin)


SAMPLING_EPS = 1e-3
NOISE_EPS = 1e-3


def config(mode='clipped', minimum=0.45, maximum=0.95, **overrides):
  result = {
    'is_distill': True,
    'high_mask_fraction': 0.0,
    'generator_force_full_mask': False,
    'mask_schedule': {
      'mode': mode,
      'min': minimum,
      'max': maximum,
      'search_batches': 2,
      'search_intervals': [[0.45, 0.95], [0.1, 0.6], [0.25, 0.75]],
    },
  }
  result['mask_schedule'].update(overrides)
  return result


def schedule(cfg=None, **kwargs):
  return ClippedMaskSchedule.from_config(
    config() if cfg is None else cfg,
    sampling_eps=kwargs.get('sampling_eps', SAMPLING_EPS),
    noise_eps=kwargs.get('noise_eps', NOISE_EPS))


def assert_close(actual, expected, **kwargs):
  torch.testing.assert_close(
    torch.as_tensor(actual, dtype=torch.float64),
    torch.as_tensor(expected, dtype=torch.float64),
    **kwargs)


def expect_invalid(cfg, **kwargs):
  try:
    schedule(cfg, **kwargs)
  except (TypeError, ValueError):
    return
  raise AssertionError(f'invalid schedule was accepted: {cfg!r}, {kwargs!r}')


def test_original_is_exact_and_consumes_no_extra_rng():
  # Omitted configuration must preserve existing checkpoints/config files.
  for cfg in ({'is_distill': True}, config(mode='original')):
    sched = schedule(cfg)
    base = torch.linspace(SAMPLING_EPS, 1.0, 101)
    torch.manual_seed(401)
    before = torch.random.get_rng_state()
    actual = sched.sample(base)
    assert torch.equal(actual, base)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_fixed_sampling_has_requested_distribution_and_no_extra_rng():
  sched = schedule()
  torch.manual_seed(317)
  base = SAMPLING_EPS + (1.0 - SAMPLING_EPS) * torch.rand(100_000)
  before = torch.random.get_rng_state()
  actual = sched.sample(base)
  assert torch.equal(torch.random.get_rng_state(), before)
  probability = actual * (1.0 - NOISE_EPS)
  assert probability.min() >= 0.45 - 1e-7
  assert probability.max() <= 0.95 + 1e-7
  assert abs(probability.mean().item() - 0.7) < 0.002
  assert abs((probability > 0.5).float().mean().item() - 0.9) < 0.004
  assert actual.dtype == base.dtype
  assert actual.device == base.device


def test_time_conversion_maps_endpoints_and_preserves_antithetic_order():
  sched = schedule()
  base = torch.linspace(SAMPLING_EPS, 1.0, 1001, dtype=torch.float64)
  probability = sched.sample(base) * (1.0 - NOISE_EPS)
  assert_close(probability[[0, -1]], [0.45, 0.95], atol=1e-12, rtol=0)
  assert_close(probability.diff(), torch.full((1000,), 0.0005),
               atol=1e-10, rtol=0)
  # Candidate probes use the same base draw with an explicit interval.
  alternative = sched.sample(base, bounds=(0.1, 0.6)) * (1.0 - NOISE_EPS)
  assert_close(alternative[[0, -1]], [0.1, 0.6], atol=1e-12, rtol=0)
  assert sched.bounds == (0.45, 0.95)


def test_supported_endpoints_do_not_create_clamping_atoms():
  sched = schedule(config(minimum=0.0, maximum=1.0))
  low = SAMPLING_EPS * (1.0 - NOISE_EPS)
  high = 1.0 - NOISE_EPS
  assert_close(sched.bounds, [low, high], atol=1e-12, rtol=0)
  # Interior quantiles distinguish interval intersection from draw-then-clamp:
  # clipping draws at either endpoint would create repeated extreme values.
  unit = (torch.arange(20_000, dtype=torch.float64) + 0.5) / 20_000
  base = SAMPLING_EPS + (1.0 - SAMPLING_EPS) * unit
  probability = sched.sample(base) * (1.0 - NOISE_EPS)
  assert probability.min() > low
  assert probability.max() < high
  assert torch.all(probability.diff() > 0)
  assert_close(probability.mean(), (low + high) / 2, atol=1e-12, rtol=0)


def test_invalid_configuration_fails_before_training():
  cases = [
    config(mode='typo'),
    config(minimum=-0.1),
    config(maximum=1.1),
    config(minimum=0.95, maximum=0.45),
    config(minimum=0.45, maximum=0.45),
    config(minimum=float('nan')),
    config(maximum=float('inf')),
    config(mode='adaptive', search_batches=0),
    config(mode='adaptive', search_batches=-1),
    config(mode='adaptive', search_intervals=[[0.8, 0.2]]),
    config(mode='adaptive', search_intervals=[[0.1, 0.2, 0.3]]),
    # A nonzero mathematical interval can still be outside noise support.
    config(minimum=0.9995, maximum=1.0),
  ]
  for cfg in cases:
    expect_invalid(cfg)
  for name, value in (('high_mask_fraction', 0.2),
                      ('generator_force_full_mask', True),
                      ('is_distill', False)):
    cfg = config()
    cfg[name] = value
    expect_invalid(cfg)


def test_adaptive_candidates_include_initial_and_use_effective_support():
  sched = schedule(config(
    mode='adaptive', minimum=0.3, maximum=0.8,
    search_intervals=[[0.0, 1.0], [0.3, 0.8], [0.3, 0.8]]))
  assert sched.bounds in sched.candidates
  assert len(sched.candidates) == len(set(sched.candidates))
  assert (SAMPLING_EPS * (1.0 - NOISE_EPS),
          1.0 - NOISE_EPS) in sched.candidates


def test_adaptive_selects_variance_instead_of_mean():
  sched = schedule(config(mode='adaptive'))
  moments = sched.new_statistics(torch.device('cpu'))
  assert moments.shape == (len(sched.candidates), 4)
  assert moments.dtype == torch.float64
  for index in range(len(sched.candidates)):
    # The selected candidate has much higher mean but zero variance.
    losses = [100.0, 100.0, 100.0] if index == 1 else [0.0, 1.0, 2.0]
    sched.observe(moments, index, torch.tensor(losses, requires_grad=True))
  assert not moments.requires_grad
  assert sched.select(moments) == 1
  assert sched.bounds == sched.candidates[1]


def test_adaptive_pooled_moments_match_concatenated_examples():
  sched = schedule(config(mode='adaptive'))
  shards = [sched.new_statistics('cpu') for _ in range(2)]
  combined = sched.new_statistics('cpu')
  for index in range(len(sched.candidates)):
    batches = ([0.0, 0.0], [10.0, 10.0]) if index == 0 else (
      [3.0, 4.0], [5.0, 6.0])
    for shard, values in zip(shards, batches):
      sched.observe(shard, index, torch.tensor(values, dtype=torch.float64))
    sched.observe(combined, index,
                  torch.tensor(batches[0] + batches[1], dtype=torch.float64))
  pooled = shards[0] + shards[1]  # Same reduction as SUM over DDP ranks.
  assert_close(pooled, combined, atol=0, rtol=0)
  # The first interval has zero variance within each shard, but pooled
  # variance 25. Averaging minibatch variances would select it incorrectly.
  assert sched.select(pooled) == 1
  counts, sums, squares = pooled[:, 0], pooled[:, 1], pooled[:, 2]
  assert_close(squares / counts - (sums / counts).square(),
               [25.0, 1.25, 1.25], atol=1e-12, rtol=0)


def test_adaptive_rejects_nonfinite_and_insufficient_candidates():
  sched = schedule(config(mode='adaptive'))
  moments = sched.new_statistics('cpu')
  before = sched.bounds
  assert sched.select(moments) is None
  assert sched.bounds == before
  sched.observe(moments, 0, torch.tensor([0.0, float('nan'), 0.0]))
  sched.observe(moments, 1, torch.tensor([0.0]))
  sched.observe(moments, 2, torch.tensor([2.0, 3.0]))
  assert sched.select(moments) == 2
  assert sched.bounds == sched.candidates[2]
  for bad in (float('inf'), -float('inf')):
    moments = sched.new_statistics('cpu')
    sched.observe(moments, 0, torch.tensor([bad, 0.0, 0.0]))
    assert sched.select(moments) is None


def test_adaptive_ties_keep_current_interval():
  sched = schedule(config(mode='adaptive'))
  moments = sched.new_statistics('cpu')
  sched.observe(moments, 2, torch.tensor([1.0, 1.0]))
  assert sched.select(moments) == 2
  tied = sched.new_statistics('cpu')
  for index in range(len(sched.candidates)):
    sched.observe(tied, index, torch.tensor([1.0, 2.0, 3.0]))
  assert sched.select(tied) == 2
  assert sched.bounds == sched.candidates[2]


def test_checkpoint_restores_only_compatible_adaptive_state():
  cfg = config(mode='adaptive')
  trained = schedule(cfg)
  moments = trained.new_statistics('cpu')
  trained.observe(moments, 1, torch.tensor([2.0, 2.0]))
  assert trained.select(moments) == 1
  state = copy.deepcopy(trained.state_dict())
  restored = schedule(cfg)
  restored.load_state_dict(state)
  assert restored.bounds == trained.bounds
  assert restored.candidates == trained.candidates
  for incompatible in (
      config(mode='original'),
      config(mode='clipped'),
      config(mode='adaptive', search_intervals=[[0.2, 0.7]]),
      config(mode='adaptive', minimum=0.4, maximum=0.9)):
    target = schedule(incompatible)
    initial = target.bounds
    target.load_state_dict(state)
    assert target.bounds == initial


class TinyLogLinear:
  eps = NOISE_EPS

  def __call__(self, t):
    return -(1.0 - self.eps), 1.0 - (1.0 - self.eps) * t


class TinyValidationBase(torch.nn.Module):
  """Tiny trainable denoiser with the real MDLM validation/probe interface."""

  def __init__(self, cfg, tokenizer):
    super().__init__()
    del tokenizer
    self.config = cfg
    self.sampling_eps = SAMPLING_EPS
    self.noise = TinyLogLinear()
    self.mask_index = 7
    self.ignore_bos = False
    self.global_step = 10
    self.trainer = SimpleNamespace(sanity_checking=False)
    self.embedding = torch.nn.Embedding(8, 5)
    self.output = torch.nn.Linear(5, 8)
    self.forward_inputs = []
    self.grad_modes = []
    self.validation_times = []
    self.logged = []
    self.parent_events = []

  @property
  def device(self):
    return self.embedding.weight.device

  def _sample_t(self, n, accum_step):
    assert accum_step is None
    return self.sampling_eps + (1.0 - self.sampling_eps) * torch.rand(n)

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _sigma_from_alphat(self, alpha):
    return -alpha.log()

  def q_xt(self, x0, alpha_t, valid_tokens=None):
    masked = torch.rand_like(x0, dtype=torch.float32) < (1.0 - alpha_t)
    if valid_tokens is not None:
      masked &= valid_tokens.bool()
    return x0.masked_fill(masked, self.mask_index)

  def forward(self, xt, sigma):
    assert torch.isfinite(sigma).all()
    self.forward_inputs.append(xt.clone())
    self.grad_modes.append(torch.is_grad_enabled())
    return self.output(self.embedding(xt)).log_softmax(-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False):
    assert low_var is False
    log_prob = log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    return log_prob * dalpha_t / (1.0 - alpha_t) * (xt == self.mask_index)

  def on_validation_epoch_start(self):
    self.parent_events.append('start')
    self.eval()

  @torch.no_grad()
  def validation_step(self, batch, batch_idx):
    self.parent_events.append(('step', batch_idx))
    x0, valid = batch['input_ids'], batch['attention_mask']
    t = self._sample_t(x0.shape[0], None)
    self.validation_times.append(t.clone())
    dalpha, alpha = self.noise(t)
    alpha = alpha[:, None]
    xt = self.q_xt(x0, alpha, valid)
    log_probs = self.forward(xt, self._sigma_from_alphat(alpha))
    loss = self.nll_per_token(log_probs, xt, x0, alpha, dalpha)
    return (loss * valid).sum() / valid.sum().clamp_min(1)

  def on_validation_epoch_end(self):
    self.parent_events.append('end')
    self.train()

  def log_dict(self, values, **kwargs):
    assert kwargs['sync_dist'] is False
    self.logged.append(values)

  def on_save_checkpoint(self, checkpoint):
    self.parent_events.append('save')
    checkpoint['base_hook_was_called'] = True

  def on_load_checkpoint(self, checkpoint):
    self.parent_events.append('load')


class TinyScheduledModel(DistillMaskScheduleMixin, TinyValidationBase):
  pass


def tiny_model(schedule_mode='adaptive', **kwargs):
  cfg = SimpleNamespace(
    mode=kwargs.pop('mode', 'train'),
    adversarial_distill=config(mode=schedule_mode))
  model = TinyScheduledModel(cfg, tokenizer=None)
  for name, value in kwargs.items():
    setattr(model, name, value)
  return model


def response_batch():
  return {
    'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6],
                               [2, 3, 4, 5, 6, 1],
                               [3, 4, 5, 6, 1, 2]]),
    # Middle row has no response; it must not become a zero-loss observation.
    'attention_mask': torch.tensor([[0, 0, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 0]]),
  }


def test_mixin_preserves_prompts_excludes_empty_rows_and_disables_gradients():
  model = tiny_model()
  batch = response_batch()
  model.on_validation_epoch_start()
  stats = model._mask_schedule_statistics
  original_tokens = batch['input_ids'].clone()
  original_mask = batch['attention_mask'].clone()
  before_rng = torch.random.get_rng_state()
  # Exercise the mixin outside Lightning's no_grad validation context too.
  with torch.enable_grad():
    model._probe_mask_schedule(batch, stats)
  assert torch.equal(torch.random.get_rng_state(), before_rng)
  assert torch.equal(stats[:, 0], torch.full_like(stats[:, 0], 2.0))
  assert len(model.forward_inputs) == len(model._mask_schedule.candidates)
  assert all(enabled is False for enabled in model.grad_modes)
  assert all(parameter.grad is None for parameter in model.parameters())
  assert not stats.requires_grad
  protected = ~batch['attention_mask'].bool()
  for xt in model.forward_inputs:
    assert torch.equal(xt[protected], original_tokens[protected])
  assert torch.equal(batch['input_ids'], original_tokens)
  assert torch.equal(batch['attention_mask'], original_mask)


def test_mixin_preserves_ordinary_validation_and_limits_probe_batches():
  original = tiny_model(schedule_mode='original')
  adaptive = tiny_model()
  adaptive.load_state_dict(original.state_dict())
  batch = response_batch()
  state = torch.random.get_rng_state()
  original.on_validation_epoch_start()
  expected = [original.validation_step(batch, index) for index in range(3)]
  original.on_validation_epoch_end()
  expected_rng = torch.random.get_rng_state()
  torch.random.set_rng_state(state)
  adaptive.on_validation_epoch_start()
  actual = [adaptive.validation_step(batch, index) for index in range(3)]
  adaptive.on_validation_epoch_end()
  assert torch.equal(torch.random.get_rng_state(), expected_rng)
  assert all(torch.equal(left, right) for left, right in zip(actual, expected))
  assert all(torch.equal(left, right) for left, right in zip(
    original.validation_times, adaptive.validation_times))
  assert len(adaptive.forward_inputs) == (
    3 + adaptive._mask_schedule.search_batches * len(adaptive._mask_schedule.candidates))
  assert adaptive.parent_events == original.parent_events
  assert adaptive.training  # Parent end hook still restores training mode.
  assert adaptive._mask_schedule_statistics is None
  assert adaptive.logged[-1]['mask_schedule/probe_rows'] == 4
  assert adaptive.logged[-1]['mask_schedule/updated'] == 1.0


def test_mixin_ignores_sanity_evaluation_startup_and_nonadaptive_modes():
  models = [tiny_model(schedule_mode='original'),
            tiny_model(schedule_mode='clipped'),
            tiny_model(mode='eval'),
            tiny_model(global_step=0),
            tiny_model(trainer=SimpleNamespace(sanity_checking=True))]
  for model in models:
    initial = model._mask_schedule.bounds
    model.on_validation_epoch_start()
    assert model._mask_schedule_statistics is None
    model.validation_step(response_batch(), 0)
    model.on_validation_epoch_end()
    assert len(model.forward_inputs) == 1
    assert not model.logged
    assert model._mask_schedule.bounds == initial


def test_mixin_skips_batches_without_scored_tokens_including_ignored_bos():
  model = tiny_model(ignore_bos=True)
  batch = response_batch()
  batch['attention_mask'].zero_()
  batch['attention_mask'][:, 0] = 1
  saved_mask = batch['attention_mask'].clone()
  model.on_validation_epoch_start()
  initial = model._mask_schedule.bounds
  before_rng = torch.random.get_rng_state()
  model._probe_mask_schedule(batch, model._mask_schedule_statistics)
  assert torch.equal(torch.random.get_rng_state(), before_rng)
  assert torch.equal(batch['attention_mask'], saved_mask)
  assert not model.forward_inputs
  assert not model._mask_schedule_statistics.any()
  model.on_validation_epoch_end()
  assert model._mask_schedule.bounds == initial
  assert model.logged[-1]['mask_schedule/updated'] == 0.0


def test_mixin_checkpoint_hooks_preserve_parent_behavior_and_restore_bounds():
  model = tiny_model()
  stats = model._mask_schedule.new_statistics('cpu')
  model._mask_schedule.observe(stats, 1, torch.tensor([3.0, 3.0]))
  assert model._mask_schedule.select(stats) == 1
  checkpoint = {}
  model.on_save_checkpoint(checkpoint)
  assert checkpoint['base_hook_was_called'] is True
  assert 'distill_mask_schedule' in checkpoint
  restored = tiny_model()
  restored.on_load_checkpoint(checkpoint)
  assert restored._mask_schedule.bounds == model._mask_schedule.bounds
  assert restored.parent_events[-1] == 'load'
  # Old checkpoints keep their original strict parameter loading contract.
  assert not any('mask_schedule' in key for key in restored.state_dict())
  restored.load_state_dict(model.state_dict(), strict=True)
  legacy = tiny_model()
  legacy.on_load_checkpoint({})
  assert legacy._mask_schedule.bounds == (0.45, 0.95)


def _distributed_worker(rank, rendezvous):
  import torch.distributed as dist

  torch.set_num_threads(1)
  dist.init_process_group(
    'gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2,
    timeout=datetime.timedelta(seconds=20))
  try:
    model = tiny_model()
    model.on_validation_epoch_start()
    moments = model._mask_schedule_statistics
    # Different counts and means across ranks are essential: both ranks
    # locally prefer candidate 0, but its pooled variance is much worse.
    local_losses = ([[0.0, 0.0], [2.0, 4.0], [0.0, 4.0]] if rank == 0 else
                    [[10.0, 10.0, 10.0], [3.0, 5.0, 6.0], [1.0, 6.0, 9.0]])
    for index, values in enumerate(local_losses):
      model._mask_schedule.observe(moments, index, torch.tensor(values))
    assert int(model._mask_schedule.variances(moments).argmin()) == 0
    assert_close(moments[:, 0], [2.0 + rank] * 3, atol=0, rtol=0)
    # Invoke the real hook, including its real all_reduce SUM operation.
    model.on_validation_epoch_end()
    variances = model._mask_schedule.variances(moments)
    assert_close(moments[:, 0], [5.0, 5.0, 5.0], atol=0, rtol=0)
    assert_close(variances, [24.0, 2.0, 10.8], atol=1e-12, rtol=0)
    assert model._mask_schedule.bounds == model._mask_schedule.candidates[1]
    assert model.logged[-1]['mask_schedule/denoising_variance'] == 2.0
    assert model.logged[-1]['mask_schedule/probe_rows'] == 5.0
    assert model._mask_schedule_statistics is None
    assert model.parent_events[-1] == 'end'
    print('RANK_RESULT ' + json.dumps({
      'rank': rank, 'bounds': model._mask_schedule.bounds,
      'variance': float(model.logged[-1]['mask_schedule/denoising_variance']),
      'count': float(model.logged[-1]['mask_schedule/probe_rows']),
    }), flush=True)
  finally:
    dist.destroy_process_group()


def run_distributed_test():
  """Real independent processes, with collective and parent wall-clock bounds."""
  processes = []
  with tempfile.TemporaryDirectory(prefix='mask-schedule-gloo-') as directory:
    rendezvous = os.path.join(directory, 'rendezvous')
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = '1'
    environment['MKL_NUM_THREADS'] = '1'
    try:
      for rank in range(2):
        processes.append(subprocess.Popen(
          [sys.executable, os.path.abspath(__file__), '--distributed-worker',
           str(rank), '--rendezvous', rendezvous], env=environment,
          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
      deadline = time.monotonic() + 45.0
      reports = []
      for rank, process in enumerate(processes):
        output, _ = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
        assert process.returncode == 0, f'Gloo rank {rank} failed:\n{output}'
        records = [line[len('RANK_RESULT '):] for line in output.splitlines()
                   if line.startswith('RANK_RESULT ')]
        assert len(records) == 1, f'Gloo rank {rank} returned no result:\n{output}'
        reports.append(json.loads(records[0]))
      assert [report['rank'] for report in reports] == [0, 1]
      for report in reports:
        assert report['bounds'] == [0.1, 0.6]
        assert report['variance'] == 2.0
        assert report['count'] == 5.0
      print('PASS distributed: both ranks selected [0.1, 0.6], '
            'pooled variance=2, rows=5')
    except subprocess.TimeoutExpired as error:
      raise AssertionError('Two-process Gloo check exceeded its 45-second deadline') from error
    finally:
      # No child is allowed to outlive the test, including a broken rendezvous.
      for process in processes:
        if process.poll() is None:
          process.kill()
      for process in processes:
        process.wait(timeout=3)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--distributed', action='store_true',
                      help='also exercise the real two-process Gloo reduction')
  parser.add_argument('--distributed-worker', type=int, help=argparse.SUPPRESS)
  parser.add_argument('--rendezvous', help=argparse.SUPPRESS)
  arguments = parser.parse_args()
  if arguments.distributed_worker is not None:
    _distributed_worker(arguments.distributed_worker, arguments.rendezvous)
    sys.exit(0)
  tests = [value for name, value in list(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
    print(f'PASS {test.__name__}')
  if arguments.distributed:
    run_distributed_test()
  print(f'{len(tests)}/{len(tests)} passed')
