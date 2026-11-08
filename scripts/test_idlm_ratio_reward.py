"""CPU regressions for bounded-memory IDLM log-density ratio scoring.

Run from the repository root: python scripts/test_idlm_ratio_reward.py
No checkpoints, network access, or accelerator are needed.
"""
import ast
import math
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mmd_reward


def _load_ratio_class():
  # Exercise the actual implementation without importing Lightning, model
  # architectures, tokenizers, or the training environment.
  path = os.path.join(ROOT, 'algo.py')
  with open(path, encoding='utf-8') as source:
    tree = ast.parse(source.read(), filename=path)
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef)
             and node.name == 'IdlmRatioReward']
  assert len(classes) == 1
  namespace = {'torch': torch, 'F': F}
  exec(compile(ast.Module(body=classes, type_ignores=[]), path, 'exec'), namespace)
  return namespace['IdlmRatioReward']


IdlmRatioReward = _load_ratio_class()
_MISSING = object()


class _Model(torch.nn.Module):
  """Deterministic contextual logits and a record of each critic forward."""

  def __init__(self, vocab, seed, dtype):
    super().__init__()
    generator = torch.Generator().manual_seed(seed)
    self.table = torch.nn.Parameter(torch.randn(vocab, vocab, generator=generator))
    self.context = torch.nn.Parameter(torch.randn(vocab, generator=generator))
    self.output_dtype = dtype
    self.calls = []

  def forward(self, xt, sigma):
    self.calls.append((xt.detach().clone(),
                       None if sigma is None else sigma.detach().clone(),
                       torch.is_grad_enabled()))
    conditioning = xt.float().mean(-1) * 0.07
    if sigma is not None:
      conditioning = conditioning + sigma * 0.3
    logits = self.table[xt] + conditioning[:, None, None] * self.context
    return logits.to(self.output_dtype)


class _Algo:
  def __init__(self, chunk=5, *, dalpha='vector', ignore_bos=True,
               dtype=torch.float32, sigma_none=False):
    settings = {} if chunk is _MISSING else {'idlm_reward_batch_size': chunk}
    self.config = SimpleNamespace(adversarial_distill=SimpleNamespace(**settings))
    self.vocab_size = 11
    self.mask_index = self.vocab_size - 1
    self.ignore_bos = ignore_bos
    self.dalpha_kind = dalpha
    self.sigma_none = sigma_none
    self.teacher = _Model(self.vocab_size, 29, dtype)
    self.fake_model = _Model(self.vocab_size, 71, dtype)
    self.sample_calls = []

  def _sample_t(self, batch, accumulation_step):
    t = 0.1 + 0.8 * torch.rand(batch)
    self.sample_calls.append((batch, accumulation_step, t.clone()))
    return t

  def noise(self, t):
    alpha = 0.2 + 0.6 * t
    if self.dalpha_kind == 'python_scalar':
      dalpha = -0.37
    elif self.dalpha_kind == 'tensor_scalar':
      dalpha = torch.tensor(-0.37)
    elif self.dalpha_kind == 'vector':
      dalpha = -(0.2 + t)
    elif self.dalpha_kind == 'column':
      dalpha = -(0.2 + t[:, None])
    else:
      raise AssertionError(self.dalpha_kind)
    return dalpha, alpha

  def _sigma_from_alphat(self, alpha):
    return -alpha.log()

  def _process_sigma(self, sigma):
    return None if self.sigma_none else sigma.squeeze(-1)

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    # MDLM's hard-token parameterization, including exact observations. Finite
    # negative logits also preserve the historical dense target multiplication.
    model_output[:, :, self.mask_index] += -1e6
    log_probs = model_output - torch.logsumexp(model_output, -1, keepdim=True)
    observed = xt != self.mask_index
    log_probs[observed] = -1e6
    log_probs[observed, xt[observed]] = 0
    return log_probs


def _inputs(group=3):
  generator = torch.Generator().manual_seed(19)
  clean = torch.randint(0, 10, (4, 7), generator=generator)
  generated = torch.randint(0, 10, (group, 4, 7), generator=generator)
  clean[:, 0] = 0
  generated[:, :, 0] = 0
  selected = torch.tensor([[True, True, False, True, False, True, True],
                           [False, False, False, False, False, False, False],
                           [True, False, True, True, True, False, False],
                           [True, True, True, True, True, True, False]])
  return clean, generated, selected


@torch.no_grad()
def _dense_old_reference(algo, clean, generated, selected, eos_id):
  """Independent pre-chunking implementation, including dense FP32 targets."""
  del clean, eos_id
  group, batch, length = generated.shape
  t = algo._sample_t(batch, 0)
  dalpha, alpha = algo.noise(t)
  alpha = alpha[:, None]
  moved = (torch.rand(batch, length) < 1 - alpha) & selected.bool()
  xt = torch.where(moved[None], algo.mask_index, generated)
  if algo.ignore_bos:
    xt[:, :, 0] = generated[:, :, 0]
  xt = xt.reshape(group * batch, length)
  targets = F.one_hot(generated.reshape(group * batch, length),
                      num_classes=algo.vocab_size).float()
  valid = selected[None].expand(group, -1, -1).reshape(group * batch, length)
  alpha = alpha.repeat(group, 1)
  if torch.is_tensor(dalpha) and dalpha.ndim:
    dalpha = dalpha.reshape(batch, -1).repeat(group, 1)
  sigma = algo._process_sigma(algo._sigma_from_alphat(alpha))
  scores = []
  for model in (algo.teacher, algo.fake_model):
    logits = algo._process_model_output(model(xt, sigma), xt, sigma)
    token_loss = (targets * logits).sum(-1) * dalpha / (1 - alpha)
    scores.append((token_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1))
  reward = (scores[1] - scores[0]).float().reshape(group, batch).T
  active = selected.any(-1)
  return torch.where(active[:, None], reward, torch.zeros_like(reward)), active


class _DenseCritic:
  resolved_layers = []

  def __init__(self, algo):
    self.algo = algo

  def grouped_reward(self, clean, generated, selected, eos_id):
    return _dense_old_reference(self.algo, clean, generated, selected, eos_id)


def _assert_forward_contract(actual, reference, group, batch, chunk):
  expected_batches = [min(chunk, group * batch - start)
                      for start in range(0, group * batch, chunk)]
  assert len(actual.sample_calls) == 1
  assert actual.sample_calls[0][:2] == (batch, 0)
  torch.testing.assert_close(actual.sample_calls[0][2], reference.sample_calls[0][2])
  for actual_model, reference_model in ((actual.teacher, reference.teacher),
                                        (actual.fake_model, reference.fake_model)):
    assert [call[0].shape[0] for call in actual_model.calls] == expected_batches
    assert len(actual_model.calls) == math.ceil(group * batch / chunk)
    assert not any(call[2] for call in actual_model.calls)
    torch.testing.assert_close(torch.cat([call[0] for call in actual_model.calls]),
                               reference_model.calls[0][0], rtol=0, atol=0)
    if actual.sigma_none:
      assert all(call[1] is None for call in actual_model.calls)
    else:
      torch.testing.assert_close(torch.cat([call[1] for call in actual_model.calls]),
                                 reference_model.calls[0][1], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in actual_model.parameters())


def _compare_to_reference(chunk, **settings):
  group = settings.pop('group', 3)
  clean, generated, selected = _inputs(group)
  reference = _Algo(**settings)
  actual = _Algo(chunk, **settings)
  torch.manual_seed(821)
  expected_reward, expected_active = _dense_old_reference(
    reference, clean, generated, selected, 0)
  expected_rng = torch.get_rng_state().clone()
  torch.manual_seed(821)
  with mock.patch.object(F, 'one_hot', side_effect=AssertionError(
      'production ratio scoring allocated dense one-hot targets')):
    reward, active = IdlmRatioReward(actual).grouped_reward(
      clean, generated, selected, 0)
  torch.testing.assert_close(reward, expected_reward, rtol=2e-6, atol=2e-6)
  torch.testing.assert_close(active, expected_active, rtol=0, atol=0)
  torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
  assert reward.shape == (4, group) and reward.dtype == torch.float32
  assert reward.abs().sum() > 0
  assert not reward.requires_grad and reward.grad_fn is None
  assert torch.count_nonzero(reward[1]) == 0
  _assert_forward_contract(actual, reference, group, 4,
                           8 if chunk is _MISSING else chunk)
  return actual, generated, selected


def test_ragged_chunks_preserve_dense_reward_and_rng_for_all_dalpha_shapes():
  for dalpha in ('python_scalar', 'tensor_scalar', 'vector', 'column'):
    for chunk in (1, 5, 8, 12, 99):
      _compare_to_reference(chunk, dalpha=dalpha)


def test_bf16_model_outputs_keep_historical_fp32_nll_weighting():
  # HF DiT may return BF16 even inside an outer disabled-autocast context.
  # Its selected log probabilities must be promoted BEFORE scalar weighting,
  # exactly as multiplying the old dense FP32 targets used to promote them.
  for dalpha in ('python_scalar', 'tensor_scalar', 'vector', 'column'):
    for chunk in (1, 5, 99):
      _compare_to_reference(chunk, dalpha=dalpha, dtype=torch.bfloat16)


def test_shared_times_masks_and_bos_are_unchanged_by_chunk_boundaries():
  for ignore_bos in (False, True):
    actual, generated, selected = _compare_to_reference(5, ignore_bos=ignore_bos)
    inputs = torch.cat([call[0] for call in actual.teacher.calls]).reshape_as(generated)
    corruption = inputs == actual.mask_index
    torch.testing.assert_close(corruption, corruption[:1].expand_as(corruption))
    assert corruption.any()
    assert not corruption[:, ~selected].any()
    torch.testing.assert_close(inputs[~corruption], generated[~corruption])
    if ignore_bos:
      torch.testing.assert_close(inputs[:, :, 0], generated[:, :, 0])
    else:
      assert corruption[:, :, 0].any(), 'fixture must exercise corruptible BOS'
    sigma = torch.cat([call[1] for call in actual.teacher.calls]).reshape(3, 4)
    torch.testing.assert_close(sigma, sigma[:1].expand_as(sigma), rtol=0, atol=0)
    fake_inputs = torch.cat([call[0] for call in actual.fake_model.calls])
    torch.testing.assert_close(fake_inputs, inputs.reshape(12, 7))


def test_missing_setting_defaults_to_eight_and_none_sigma_is_supported():
  _compare_to_reference(_MISSING, sigma_none=True)


def test_group_one_and_paired_return_contract():
  for per_layer in (False, True):
    clean, generated, selected = _inputs(group=1)
    reference = _Algo(dalpha='tensor_scalar')
    torch.manual_seed(13)
    expected, expected_active = _dense_old_reference(
      reference, clean, generated, selected, 0)
    torch.manual_seed(13)
    actual = _Algo(3, dalpha='tensor_scalar')
    output = IdlmRatioReward(actual).paired_reward(
      clean, generated[0], selected, 0, return_per_layer=per_layer)
    assert len(output) == (3 if per_layer else 2)
    torch.testing.assert_close(output[0], expected[:, 0], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(output[1], expected_active)
    if per_layer:
      assert output[2] == {}
    _assert_forward_contract(actual, reference, 1, 4, 3)


def test_all_inactive_rows_return_finite_zero_rewards():
  clean, generated, selected = _inputs()
  selected.zero_()
  algo = _Algo(5, dtype=torch.bfloat16)
  reward, active = IdlmRatioReward(algo).grouped_reward(clean, generated, selected, 0)
  assert not active.any() and torch.isfinite(reward).all()
  assert torch.count_nonzero(reward) == 0


def test_invalid_batch_sizes_and_shapes_fail_before_model_forward():
  for invalid in (0, -1, True, False, 1.5, '8', None):
    algo = _Algo(invalid)
    try:
      IdlmRatioReward(algo)
    except ValueError as error:
      assert 'positive integer' in str(error)
    else:
      raise AssertionError(f'invalid batch size accepted: {invalid!r}')
    assert not algo.teacher.calls and not algo.fake_model.calls
  clean, generated, selected = _inputs()
  for candidates, mask in ((generated[0], selected), (generated, selected[:, :-1])):
    algo = _Algo()
    try:
      IdlmRatioReward(algo).grouped_reward(clean, candidates, mask, 0)
    except ValueError:
      pass
    else:
      raise AssertionError('invalid grouped input shape was accepted')
    assert not algo.sample_calls and not algo.teacher.calls and not algo.fake_model.calls


def test_reinforce_loss_statistics_and_student_gradients_match_dense_reference():
  clean, _, written = _inputs()
  generator = torch.Generator().manual_seed(55)
  initial_logits = torch.randn(4, 7, 11, generator=generator)
  initial_logits[:, :, -1] = -100  # Real student cannot emit the mask token.
  for group, baseline in ((1, 'none'), (3, 'group'), (3, 'batch')):
    for wide_scoring in (False, True):
      scored = written.clone()
      if wide_scoring:
        scored[written.any(-1)] = True
      results = []
      for dense in (True, False):
        logits = initial_logits.clone().requires_grad_()
        algo = _Algo(5, dalpha='python_scalar', dtype=torch.bfloat16)
        critic = _DenseCritic(algo) if dense else IdlmRatioReward(algo)
        torch.manual_seed(178)
        loss, stats = mmd_reward.mmd_reinforce_loss(
          logits.softmax(-1), clean, written, critic, group_size=group,
          eos_id=0, reward_weight=0.7, baseline=baseline, scored=scored)
        loss.backward()
        assert logits.grad.abs().sum() > 0
        assert torch.count_nonzero(logits.grad[~written]) == 0
        assert all(parameter.grad is None for model in (algo.teacher, algo.fake_model)
                   for parameter in model.parameters())
        results.append((loss.detach(), stats, logits.grad, torch.get_rng_state()))
      expected, actual = results
      torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)
      assert actual[1].keys() == expected[1].keys()
      for key in actual[1]:
        torch.testing.assert_close(actual[1][key], expected[1][key],
                                   rtol=2e-6, atol=2e-6)
      torch.testing.assert_close(actual[2], expected[2], rtol=2e-6, atol=2e-6)
      torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def main():
  torch.set_num_threads(1)
  for name, function in sorted(globals().items()):
    if name.startswith('test_'):
      function()
      print(f'PASS  {name}')
  print('\nall IDLM ratio reward tests passed')


if __name__ == '__main__':
  main()
