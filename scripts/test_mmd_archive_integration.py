"""Synthetic CPU integration, real DiT, no network, datasets or tracker."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
  import ipdb
except ImportError:
  import pdb
  sys.modules['ipdb'] = pdb
import algo
import utils
from evaluated_checkpoint import EvaluatedCheckpoint
from mmd.objective import FeatureReward, reinforce_loss
from mmd.objective_fast import FastFeatureReward
from mmd_archive import ArchiveObjective, validate_config


def build(implementation='fast', head=True, draws=2, group=4, dropout=0):
  utils.register_omegaconf_resolvers()
  with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
    cfg = compose(config_name='config', overrides=[
      'data=tiny-gsm', 'algo=mdlm', 'model=small', 'model.hidden_size=32',
      'model.cond_dim=16', 'model.n_blocks=3', 'model.n_heads=4', 'model.length=16',
      f'model.dropout={dropout}', 'trainer.devices=1', 'trainer.accumulate_grad_batches=1',
      'loader.batch_size=4', 'loader.global_batch_size=8', 'loader.num_workers=1',
      'adversarial_distill.is_distill=True', 'adversarial_distill.dmd_weight=0',
      'adversarial_distill.mmd_reward_weight=1', 'adversarial_distill.mmd_kernel=rbf',
      'adversarial_distill.mmd_feature_layer=1', 'adversarial_distill.mmd_rbf_alpha=0.2',
      'adversarial_distill.mmd_noise_level=0', 'adversarial_distill.mmd_kernel_source=draws',
      'adversarial_distill.mmd_rbf_exclude_same_sentence=True',
      'adversarial_distill.mmd_rbf_exclude_same_position=True',
      f'adversarial_distill.mmd_implementation={implementation}',
      f'adversarial_distill.mmd_fast_scored_head={head}',
      f'adversarial_distill.mmd_kernel_batch_size={draws}',
      f'adversarial_distill.mmd_group_size={group}',
      'adversarial_distill.mmd_forward_batch_size=2', 'sampling.predictor=ancestral_cache'])
  tokenizer = SimpleNamespace(vocab_size=13, mask_token='[MASK]', mask_token_id=12,
                              eos_token_id=1, pad_token_id=0)
  torch.manual_seed(45)
  with mock.patch('metrics.Metrics', return_value=SimpleNamespace()):
    model = algo.MDLM(cfg, tokenizer)
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=.1)
  torch.nn.init.normal_(model.backbone.output_layer.linear.bias, std=.1)
  utils.freeze_model(model.fake_model)
  model._trainer = SimpleNamespace(num_nodes=1, node_rank=0, num_devices=1, local_rank=0)
  model.logged = {}
  model.log_dict = lambda stats, **kw: model.logged.update(stats)
  return model


def inputs():
  torch.manual_seed(46)
  clean = torch.randint(2, 12, (4, 16))
  valid = torch.ones_like(clean, dtype=torch.bool)
  valid[:, :3] = False
  clean[:, -2:] = 0
  clean[1, 10] = 1
  valid[0] = False
  selected = valid & clean.ne(0)
  selected[1, 11:] = False
  selected[2, 5:] = False
  return clean, valid, selected


def fixed_draws(probs, num_samples, **kwargs):
  return probs.argsort(-1, descending=True)[:, torch.arange(num_samples) % probs.shape[-1]]


class LossWrapper(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, clean, valid):
    return self.model.generator_loss(clean, valid, 0)


def ddp_worker(rank, init_file):
  torch.set_num_threads(1)
  dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
  try:
    model = build()
    model.config.loader.global_batch_size = 16
    model.accum_batches = 2
    model._trainer.num_devices = 2
    model._trainer.local_rank = rank
    wrapped = torch.nn.parallel.DistributedDataParallel(LossWrapper(model), find_unused_parameters=True)
    clean, valid, _ = inputs()
    if rank == 0:
      valid.zero_()
    optimizer = torch.optim.SGD(model.backbone.parameters(), lr=.1)
    for _ in range(2):
      optimizer.zero_grad()
      wrapped(clean, valid).backward()
      gradient = torch.cat([p.grad.flatten() for p in model.backbone.parameters() if p.grad is not None])
      other = gradient.clone()
      dist.broadcast(other, src=0)
      torch.testing.assert_close(gradient, other, rtol=0, atol=0)
      assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
      optimizer.step()
    assert model._archive_mmd_objective.head.positions is None
  finally:
    dist.destroy_process_group()


class IntegrationTests(unittest.TestCase):
  def test_source_is_verbatim(self):
    manifest = json.loads((ROOT / 'mmd/archive_source.json').read_text())
    for path, expected in manifest['files'].items():
      self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

  def test_loss_gradients_and_stats_match_archive_on_actual_dit(self):
    for draws, group in [(2, 4), (3, 2), (2, 1)]:
      for implementation, head in [('reference', False), ('fast', False), ('fast', True)]:
        model = build(implementation, head, draws, group)
        reference = build('reference', False, draws, group)
        clean, _, selected = inputs()
        xt = torch.where(selected, 12, clean)
        sigma = torch.zeros(len(xt), 1)
        reward = FeatureReward(reference.teacher, layer=1, alpha=.2,
                               feature_batch_size=2, exclude_same_position=True)
        with mock.patch.object(torch, 'multinomial', side_effect=fixed_draws):
          expected, expected_stats = reinforce_loss(reference(xt, sigma).exp(), clean, selected, reward,
              group_size=group, draws_per_candidate=draws, average_candidates=False, log_paper=True)
          actual, actual_stats = ArchiveObjective(model).loss(model, xt, sigma, clean, selected)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for p, q in zip(model.backbone.parameters(), reference.backbone.parameters()):
          torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6)
        for key in expected_stats:
          torch.testing.assert_close(actual_stats[key].float(), expected_stats[key].float(), rtol=2e-5, atol=2e-6)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertEqual(model(xt, sigma).shape, (4, 16, 13))

  def test_fixed_extractor_disables_outer_autocast(self):
    for implementation in ('fast', 'reference'):
      model = build(implementation)
      objective = ArchiveObjective(model)
      observed = []
      handle = model.teacher.register_forward_pre_hook(
        lambda *args: observed.append(torch.is_autocast_cpu_enabled()))
      clean, _, selected = inputs()
      with torch.autocast('cpu', dtype=torch.bfloat16):
        if implementation == 'fast':
          features = objective.reward.features_at(clean, torch.arange(16).expand(4, -1))
        else:
          features = objective.reward.features(clean)
      handle.remove()
      self.assertTrue(observed)
      self.assertFalse(any(observed))
      self.assertEqual(model.teacher.rotary_emb.cos_cached.dtype, torch.float32)
      self.assertEqual(features.dtype, torch.float32)
      self.assertTrue(torch.isfinite(features).all())

  def test_launcher_gpu_scaling_and_overrides(self):
    utils.register_omegaconf_resolvers()
    with tempfile.TemporaryDirectory() as directory:
      shim = Path(directory) / 'python'
      shim.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
      shim.chmod(0o700)
      for devices in (2, 4, 8):
        env = dict(os.environ, DEVICES=str(devices), PATH=directory + os.pathsep + os.environ['PATH'])
        command = ['bash', str(ROOT / 'scripts/train_tinygsm_mmd_fast.sh'),
                   'adversarial_distill.mmd_kernel_batch_size=3', 'adversarial_distill.mmd_group_size=8',
                   'adversarial_distill.mmd_feature_layer=9', 'adversarial_distill.mmd_rbf_alpha=6.17283950617284e-5',
                   'eval.start_step=5500']
        args = json.loads(subprocess.check_output(command, env=env, text=True))
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
          cfg = compose(config_name='config', overrides=args[3:])
        validate_config(cfg)
        self.assertEqual(cfg.trainer.val_check_interval, 500 * cfg.adversarial_distill.accum_batches)
        self.assertEqual(32 * devices * cfg.adversarial_distill.accum_batches, 512)
        self.assertEqual(cfg.adversarial_distill.mmd_kernel_batch_size, 3)
        self.assertEqual(cfg.adversarial_distill.mmd_group_size, 8)
        self.assertTrue(cfg.eval.gsm8k_pareto_fp32)
        self.assertEqual(cfg.eval.start_step, 5500)
        self.assertTrue(cfg.checkpointing.save_evaluated_checkpoints)
        self.assertEqual(cfg.data.separator, r'\n')

  @unittest.skipUnless(torch.cuda.is_available(), 'CUDA optional; run explicitly on an available GPU')
  def test_cuda_bf16_feature_fix_and_finite_backward(self):
    for implementation in ('fast', 'reference'):
      model = build(implementation)
      model.metrics = SimpleNamespace(to=lambda *a, **k: None)
      model = model.to('cuda')
      clean, _, selected = [t.to('cuda') for t in inputs()]
      xt = torch.where(selected, 12, clean)
      sigma = torch.zeros(len(xt), 1, device='cuda')
      objective = ArchiveObjective(model)
      seen = []
      handle = model.teacher.register_forward_pre_hook(lambda *a: seen.append(torch.is_autocast_enabled()))
      with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, stats = objective.loss(model, xt, sigma, clean, selected)
      handle.remove()
      loss.backward()
      self.assertTrue(seen)
      self.assertFalse(any(seen))
      self.assertEqual(model.teacher.rotary_emb.cos_cached.dtype, torch.float32)
      gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
      self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
      self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
      self.assertTrue(torch.isfinite(loss))

  def test_rejects_incompatible_estimator(self):
    cfg = build().config
    for key, value in [('mmd_kernel_batch_size', 1), ('mmd_kernel_source', 'rows'),
                       ('mmd_feature_layer', 3), ('mmd_rbf_alpha', None)]:
      previous = cfg.adversarial_distill[key]
      cfg.adversarial_distill[key] = value
      with self.assertRaises(ValueError):
        validate_config(cfg)
      cfg.adversarial_distill[key] = previous
    cfg.loader.global_batch_size = 7
    with self.assertRaises(ValueError):
      validate_config(cfg)

  def test_lightning_accumulation_ema_and_evaluated_checkpoint(self):
    import lightning as L
    model = build(dropout=.1)
    model.metrics = SimpleNamespace(to=lambda *a, **k: None)
    model.on_train_epoch_start = lambda: None
    model.on_train_epoch_end = lambda: None
    model.on_validation_epoch_start = model._eval_mode
    model.on_validation_epoch_end = model._train_mode
    model.validation_step = lambda batch, batch_idx: None
    clean, valid, _ = inputs()
    dataset = [{'input_ids': clean[i], 'attention_mask': valid[i]} for _ in range(4) for i in range(4)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=1)
    initial = model.backbone.output_layer.linear.weight.detach().clone()
    with tempfile.TemporaryDirectory() as directory:
      callback = EvaluatedCheckpoint(directory, start_step=1)
      trainer = L.Trainer(accelerator='cpu', devices=1, max_steps=2,
          precision='32-true', logger=False, enable_checkpointing=False,
          enable_progress_bar=False, enable_model_summary=False,
          num_sanity_val_steps=0, limit_val_batches=1, val_check_interval=2,
          callbacks=[callback], default_root_dir=directory)
      trainer.validate(model, loader, verbose=False)
      trainer.fit(model, loader, loader)
      self.assertEqual(trainer.global_step, 2)
      self.assertFalse(torch.equal(initial, model.backbone.output_layer.linear.weight))
      self.assertEqual(len(list(Path(directory).glob('eval-step=*.ckpt'))), 2)
      checkpoint_path = Path(directory) / 'eval-step=00000002.ckpt'
      payload = torch.load(checkpoint_path, weights_only=False)
      self.assertEqual(payload['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'], 4)
      restored = build(dropout=.1)
      restored.on_load_checkpoint(payload)
      restored.load_state_dict(payload['state_dict'], strict=True)
      for p, q in zip(model.backbone.parameters(), restored.backbone.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
      for p, q in zip(model.ema.shadow_params, restored.ema.shadow_params):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
      self.assertEqual(len(payload['optimizer_states']), 2)
      restored.metrics = SimpleNamespace(to=lambda *a, **k: None)
      restored.on_train_epoch_start = lambda: None
      restored.on_train_epoch_end = lambda: None
      restored.on_validation_epoch_start = restored._eval_mode
      restored.on_validation_epoch_end = restored._train_mode
      restored.validation_step = lambda batch, batch_idx: None
      updates = restored.ema.num_updates
      resume = L.Trainer(accelerator='cpu', devices=1, max_steps=3,
          precision='32-true', logger=False, enable_checkpointing=False,
          enable_progress_bar=False, enable_model_summary=False,
          num_sanity_val_steps=0, limit_val_batches=1, val_check_interval=2,
          callbacks=[EvaluatedCheckpoint(directory, start_step=1)], default_root_dir=directory)
      resume.fit(restored, loader, loader, ckpt_path=str(checkpoint_path))
      self.assertEqual(resume.global_step, 3)
      self.assertEqual(restored.ema.num_updates, updates + 1)

  def test_two_rank_ddp_with_empty_rank(self):
    with tempfile.TemporaryDirectory() as directory:
      mp.spawn(ddp_worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
  torch.set_num_threads(2)
  warnings.filterwarnings('ignore', message='User provided device_type of')
  unittest.main(verbosity=2)
