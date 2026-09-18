"""Offline tests for the isolated, minimally repaired DiDi upstream runtime.

Run in its own process: conda run -n mdlm python scripts/test_didi_runtime.py
The upstream objective is exercised with a tiny CPU network; no model, dataset,
Underdeep connection, or GPU is required.
"""

import ast
import copy
import inspect
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
from omegaconf import OmegaConf, open_dict
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from didi_repro import UPSTREAM_REVISION
from didi_repro.runtime import (
    TRAINING_REPAIRS, UPSTREAM_ROOT, install_runtime, load_upstream,
    repaired_function, verify_upstream,
)


class ToyTokenizer:
    vocab_size = 8
    mask_token = None
    mask_token_id = None
    pad_token_id = 0

    def __len__(self):
        return self.vocab_size

    def batch_decode(self, rows):
        return [' '.join(map(str, row.tolist())) for row in rows]


class ToyEmbedding(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocab, width) * .1)

    def forward(self, tokens):
        if tokens.ndim == 2:
            return self.embedding[tokens]
        return tokens.softmax(-1) @ self.embedding


class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return x + torch.tanh(self.linear(x + x.mean(1, keepdim=True)))


class ToyOutput(nn.Module):
    def __init__(self, width, vocab):
        super().__init__()
        self.linear = nn.Linear(width, vocab)

    def forward(self, x, c=None):
        return self.linear(x)


class ToyBackbone(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        width = config.model.hidden_size
        self.vocab_embed = ToyEmbedding(vocab_size, width)
        self.blocks = nn.ModuleList([ToyBlock(width) for _ in range(6)])
        self.output_layer = ToyOutput(width, vocab_size)

    def forward(self, tokens, sigma):
        x = self.vocab_embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x, c=sigma)


class EmptyMetrics(dict):
    @property
    def nll(self):
        return SimpleNamespace(mean_value=0, weight=0)


class ToyMetrics:
    def __init__(self, **kwargs):
        self.train_nlls = EmptyMetrics()
        self.valid_nlls = EmptyMetrics()

    def reset(self):
        pass

    def to(self, *args, **kwargs):
        return self


class ToyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        ids = (torch.arange(12) + index) % 8
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class DiDiRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.upstream = load_upstream()
        cls.model_class = cls.upstream.algo.DiDiInstruct
        cls.original_class = cls.model_class.__bases__[0]

    def config(self, tiny=False):
        with hydra.initialize_config_dir(
                version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
            cfg = hydra.compose(config_name='config')
        if tiny:
            cfg.model.hidden_size = 8
            cfg.model.length = 12
            cfg.loader.num_workers = 0
            cfg.trainer.devices = 1
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.max_steps = 4
            cfg.optim.lr = .01
            cfg.algo.discriminator_lr = .01
            cfg.algo.lr_scheduler.warmup_steps = 1
            cfg.algo.save_after_n_steps = 1000000
            cfg.algo.print_every = 0
            cfg.eval.generate_samples = False
        return cfg

    def model(self, cfg=None):
        cfg = self.config(tiny=True) if cfg is None else cfg
        with mock.patch.object(self.upstream.algo.trainer_base.models.dit, 'DIT', ToyBackbone), \
                mock.patch.object(self.upstream.algo.trainer_base.metrics, 'Metrics', ToyMetrics):
            return self.model_class(cfg, ToyTokenizer())

    def initialize(self, model):
        teacher = {f'backbone.{key}': value.detach().clone() + .03
                   for key, value in model.backbone.state_dict().items()}
        model.initialize_teacher_checkpoint({'state_dict': teacher})
        return teacher

    def corrected_config(self, iterations=3):
        cfg = self.config(tiny=True)
        with open_dict(cfg):
            cfg.didi_training = {'iterations': iterations}
            cfg.didi_eval = {'weights': 'student_ema'}
        cfg.algo.lr_scheduler.warmup_steps = 0
        cfg.algo.ema_beta = .9999
        cfg.trainer.max_steps = iterations * 2
        return cfg

    def assert_same(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def trainer(self, directory, max_steps):
        return L.Trainer(
            accelerator='cpu', devices=1, precision='32-true', max_steps=max_steps,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0,
            default_root_dir=directory, gradient_clip_val=0.0,
        )

    def test_manifest_and_runtime_install_idempotence(self):
        manifest = verify_upstream()
        self.assertEqual(manifest['commit'], UPSTREAM_REVISION)
        self.assertGreaterEqual(len(manifest['files']), 40)
        self.assertIn('algo.py', manifest['files'])
        self.assertIn('LICENSE.md', manifest['files'])
        self.assertIs(install_runtime(self.upstream), self.model_class)
        with mock.patch('didi_repro.runtime.UPSTREAM_REVISION', 'not-the-original'):
            with self.assertRaisesRegex(RuntimeError, 'revision/manifest'):
                verify_upstream()

    def test_training_source_changes_are_only_explicit_repairs(self):
        original = textwrap.dedent(inspect.getsource(self.original_class.training_step))
        repaired = repaired_function(self.original_class.training_step, TRAINING_REPAIRS)
        restored = repaired.__didi_repaired_source__
        self.assertEqual(len(TRAINING_REPAIRS), 4)
        for before, after in reversed(TRAINING_REPAIRS):
            # The corrected logging expression also already occurs in the
            # differentiated objective; restore only its LAST occurrence.
            restored = before.join(restored.rsplit(after, 1))
        self.assertEqual(ast.dump(ast.parse(restored)), ast.dump(ast.parse(original)))
        with self.assertRaisesRegex(RuntimeError, 'no longer matches'):
            repaired_function(self.original_class.training_step, (('unknown source', ''),))

    def test_upstream_default_hyperparameters_are_not_rewritten(self):
        cfg = self.config()
        algo_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
        on_disk = OmegaConf.to_container(OmegaConf.load(
            UPSTREAM_ROOT / 'configs/algo/didi_instruct.yaml'), resolve=True)
        self.assertEqual(algo_cfg, on_disk)
        self.assertEqual(cfg.optim.lr, 1e-6)
        self.assertEqual(cfg.algo.discriminator_lr, 1e-6)
        self.assertEqual(cfg.algo.num_samples_per_prompt, 8)
        self.assertEqual(cfg.algo.tau_mode, 'beta11')
        self.assertEqual(cfg.algo.kl_beta, .05)
        self.assertEqual(cfg.algo.entropy_beta, .0005)
        self.assertEqual(cfg.trainer.max_steps, 10000)
        self.assertEqual(cfg.trainer.accumulate_grad_batches, 1)
        self.assertEqual(cfg.sampling.predictor, 'ancestral_cache')
        self.assertEqual(cfg.sampling.steps, 1000)
        self.assertEqual(cfg.training.ema, .9999)
        self.assertEqual(cfg.algo.ema_beta, .999)

    def test_teacher_initialization_all_copies_and_original_head(self):
        model = self.model()
        head = copy.deepcopy(model.discriminator.output_layer.state_dict())
        original_cfg = OmegaConf.to_container(model.config, resolve=True)
        source = self.initialize(model)
        expected = {key[len('backbone.'):]: value for key, value in source.items()}
        for network in (model.backbone, model.teacher, model.student_ema):
            self.assert_same(network.state_dict(), expected)
        for key, value in model.discriminator.state_dict().items():
            if not key.startswith('output_layer.'):
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assert_same(model.discriminator.output_layer.state_dict(), head)
        self.assertEqual(original_cfg, OmegaConf.to_container(model.config, resolve=True))
        self.assertFalse(model.automatic_optimization)
        self.assertEqual(model.loss_type, 'elbo')
        self.assertFalse(any(p.requires_grad for p in model.teacher.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.student_ema.parameters()))
        for index, block in enumerate(model.discriminator.blocks):
            self.assertEqual(all(p.requires_grad for p in block.parameters()), index >= 2)

    def test_optimizer_parameter_ownership_and_reject_incompatible_inputs(self):
        model = self.model()
        optim_student, optim_disc = model.configure_optimizers()
        student_ids = {id(p) for group in optim_student.param_groups for p in group['params']}
        disc_ids = {id(p) for group in optim_disc.param_groups for p in group['params']}
        self.assertEqual(student_ids, {id(p) for p in model.backbone.parameters()})
        self.assertEqual(disc_ids, {id(p) for p in model.discriminator.parameters() if p.requires_grad})
        self.assertFalse(student_ids & disc_ids)
        with self.assertRaisesRegex(ValueError, 'native MDLM'):
            model.initialize_teacher_checkpoint({'state_dict': {'student_ema.fake': torch.ones(1)}})
        for key, value, message in (
                ('algo.backbone', 'hf_dit', 'native teacher'),
                ('trainer.accumulate_grad_batches', 2, 'accumulation'),
                ('algo.num_samples_per_prompt', 1, 'G >= 2')):
            cfg = self.config(tiny=True)
            OmegaConf.update(cfg, key, value)
            with self.assertRaisesRegex(ValueError, message):
                self.model(cfg)

    def test_cpu_manual_training_and_checkpoint_state_resume(self):
        torch.manual_seed(19)
        model = self.model()
        self.initialize(model)
        before_student = copy.deepcopy(model.backbone.state_dict())
        before_disc = copy.deepcopy(model.discriminator.state_dict())
        before_teacher = copy.deepcopy(model.teacher.state_dict())
        with tempfile.TemporaryDirectory(prefix='didi-runtime-test-') as directory:
            trainer = self.trainer(directory, max_steps=4)
            differentiated = []
            original_backward = model.manual_backward

            def capture_backward(loss, *args, **kwargs):
                differentiated.append(loss.detach().clone())
                return original_backward(loss, *args, **kwargs)

            with mock.patch.object(model, 'manual_backward', side_effect=capture_backward):
                trainer.fit(model, train_dataloaders=DataLoader(ToyDataset(), batch_size=1))
            # Lightning counts each manual optimizer.step(), upstream counts batches.
            self.assertEqual(trainer.global_step, 4)
            self.assertEqual(model.global_steps, 2)
            self.assertTrue(any(not torch.equal(v, before_student[k])
                                for k, v in model.backbone.state_dict().items()))
            self.assertTrue(any(not torch.equal(v, before_disc[k])
                                for k, v in model.discriminator.state_dict().items()))
            self.assert_same(model.teacher.state_dict(), before_teacher)
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in model.student_ema.parameters()))
            for block in model.discriminator.blocks[:2]:
                self.assertTrue(all(p.grad is None for p in block.parameters()))
            for value in trainer.logged_metrics.values():
                self.assertTrue(torch.isfinite(torch.as_tensor(value)).all())
            required = {'trainer/loss_discriminator', 'trainer/loss_student_pg',
                        'trainer/loss_student_total', 'reward/mean', 'reward/std',
                        'reward/advantage_abs_mean', 'didi/iteration'}
            self.assertTrue(required.issubset(trainer.logged_metrics))
            self.assertEqual(len(differentiated), 4)
            torch.testing.assert_close(trainer.logged_metrics['trainer/loss_student_total'],
                                       differentiated[-1])
            first = torch.load(str(Path(directory) / 'student_iters_0.ckpt'),
                               map_location='cpu', weights_only=False)
            self.assertEqual(first['didi_repro']['next_iteration'], 1)
            self.assertEqual(first['global_step'], 2)
            path = str(Path(directory) / 'resume.ckpt')
            trainer.save_checkpoint(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['didi_repro']['next_iteration'], 2)
            self.assertTrue(any(k.startswith('teacher.') for k in checkpoint['state_dict']))
            clone = self.model()
            clone.on_load_checkpoint(checkpoint)
            clone.load_state_dict(checkpoint['state_dict'])
            self.assert_same(model.state_dict(), clone.state_dict())
            self.assertEqual(clone.global_steps, 2)
            continuation = self.trainer(directory, max_steps=6)
            continuation.fit(clone, train_dataloaders=DataLoader(ToyDataset(), batch_size=1),
                             ckpt_path=path)
            self.assertEqual(continuation.global_step, 6)
            self.assertEqual(clone.global_steps, 3)
            self.assert_same(clone.teacher.state_dict(), before_teacher)

    def test_checkpoint_validation_and_evaluation_only_ema_export(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'Training resume needs a didi_repro checkpoint'):
            model.on_load_checkpoint({'state_dict': model.state_dict()})
        cfg = self.config(tiny=True)
        cfg.mode = 'sample_eval'
        cfg.sampling.predictor = 'guided'
        evaluated = self.model(cfg)
        ema = {f'student_ema.{key}': value.detach().clone() + .1
               for key, value in evaluated.student_ema.state_dict().items()}
        state = {**ema, **{f'discriminator.{key}': value.detach().clone()
                          for key, value in evaluated.discriminator.state_dict().items()}}
        checkpoint = {'state_dict': state}
        evaluated.on_load_checkpoint(checkpoint)
        evaluated.load_state_dict(checkpoint['state_dict'])
        self.assert_same(evaluated.backbone.state_dict(),
                         {k[len('student_ema.'):]: v for k, v in ema.items()})
        missing_disc = self.model(cfg)
        with self.assertRaisesRegex(ValueError, 'trained discriminator'):
            missing_disc.on_load_checkpoint({'state_dict': dict(ema)})
        with self.assertRaisesRegex(ValueError, 'at least 2 steps'):
            evaluated.generate_samples(1, num_steps=1)

    def test_guided_generation_leaves_parameter_gradients_unchanged(self):
        torch.manual_seed(31)
        cfg = self.config(tiny=True)
        cfg.sampling.predictor = 'guided'
        model = self.model(cfg)
        self.initialize(model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, .25)
        before = {name: None if p.grad is None else p.grad.clone()
                  for name, p in model.named_parameters()}
        with torch.inference_mode():
            samples = model.generate_samples(2, num_steps=4)
        self.assertEqual(tuple(samples.shape), (2, 12))
        self.assertFalse((samples == model.mask_index).any())
        for name, parameter in model.named_parameters():
            if before[name] is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, before[name], rtol=0, atol=0)

    def test_guided_autograd_repair_preserves_original_samples(self):
        torch.manual_seed(47)
        model = self.model()
        self.initialize(model)
        model.eval()
        x = model.prior_sample(2, model.num_tokens)
        t = torch.full((2, 1), .7)
        random_state = torch.get_rng_state()
        with torch.no_grad():
            expected = self.original_class._get_gradient_tilting_step(
                model, x, t, .1, .4, model.discriminator)
        self.assertTrue(any(p.grad is not None for p in model.discriminator.parameters()))
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(random_state)
        with torch.no_grad():
            actual = model._get_gradient_tilting_step(x, t, .1, .4, model.discriminator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_iteration_budget_schedule_and_resume_use_same_units(self):
        cfg = self.corrected_config()
        model = self.model(cfg)
        self.initialize(model)
        initial_ema = [p.detach().clone() for p in model.student_ema.parameters()]

        class StopAtTwo(L.Callback):
            def on_train_batch_end(self, trainer, pl_module, *args):
                if pl_module.global_steps == 2:
                    trainer.should_stop = True

        with tempfile.TemporaryDirectory(prefix='didi-iterations-') as directory:
            trainer = self.trainer(directory, max_steps=6)
            trainer.callbacks.append(StopAtTwo())
            rates = []
            ema_updates = []
            original_lr = model._get_lr
            original_update = model.update_ema

            def capture_lr():
                optimizers = original_lr()
                rates.append([o.param_groups[0]['lr'] for o in optimizers])
                return optimizers

            def capture_ema():
                before = [p.detach().clone() for p in model.student_ema.parameters()]
                original_update()
                for prev, raw, averaged in zip(before, model.backbone.parameters(),
                                              model.student_ema.parameters()):
                    torch.testing.assert_close(averaged, prev.lerp(raw.detach(), .0001))
                ema_updates.append(1)

            with mock.patch.object(model, '_get_lr', side_effect=capture_lr), \
                    mock.patch.object(model, 'update_ema', side_effect=capture_ema):
                trainer.fit(model, train_dataloaders=DataLoader(ToyDataset(), batch_size=1))
            self.assertEqual(model.global_steps, 2)
            self.assertEqual(trainer.global_step, 4)
            self.assertEqual(len(ema_updates), 2)
            self.assertTrue(any(not torch.equal(before, after)
                                for before, after in zip(initial_ema, model.student_ema.parameters())))
            torch.testing.assert_close(torch.tensor(rates),
                                       torch.tensor([[.01, .01], [.01 * 2 / 3] * 2]))
            path = str(Path(directory) / 'resume.ckpt')
            trainer.save_checkpoint(path)
            clone = self.model(self.corrected_config())
            continuation = self.trainer(directory, max_steps=6)
            continuation.fit(clone, train_dataloaders=DataLoader(ToyDataset(), batch_size=1),
                             ckpt_path=path)
            self.assertEqual(clone.global_steps, 3)
            self.assertEqual(continuation.global_step, 6)
            for optimizer in continuation.optimizers:
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .01 / 3)
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                clone.training_step(next(iter(DataLoader(ToyDataset(), batch_size=1))), 0)

    def test_no_warmup_10000_iteration_schedule_reaches_released_floor(self):
        cfg = self.corrected_config(10000)
        cfg.optim.lr = cfg.algo.discriminator_lr = 1e-6
        model = self.model(cfg)
        optimizers = model.configure_optimizers()
        with mock.patch.object(model, 'optimizers', return_value=optimizers):
            for iteration, expected in ((0, 1e-6), (5000, 5e-7), (9999, 1e-8)):
                model.global_steps = iteration
                for optimizer in model._get_lr():
                    self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected, places=14)

    def test_iteration_mode_rejects_wrong_trainer_budget_and_changed_resume(self):
        model = self.model(self.corrected_config())
        self.initialize(model)
        checkpoint = {'state_dict': copy.deepcopy(model.state_dict())}
        model.on_save_checkpoint(checkpoint)
        for changed in ('iterations', 'warmup', 'ema'):
            cfg = self.corrected_config(4 if changed == 'iterations' else 3)
            if changed == 'warmup':
                cfg.algo.lr_scheduler.warmup_steps = 1
            if changed == 'ema':
                cfg.algo.ema_beta = .999
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'protocol differs'):
                self.model(cfg).on_load_checkpoint(copy.deepcopy(checkpoint))
        legacy = copy.deepcopy(checkpoint)
        del legacy['didi_repro']['training_protocol']
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            self.model(self.corrected_config()).on_load_checkpoint(legacy)
        with tempfile.TemporaryDirectory(prefix='didi-wrong-budget-') as directory:
            trainer = self.trainer(directory, max_steps=3)
            with self.assertRaisesRegex(ValueError, 'requires trainer.max_steps=6'):
                trainer.fit(model, train_dataloaders=DataLoader(ToyDataset(), batch_size=1))

    def test_ema_used_for_validation_and_guided_sampling_without_weight_swaps(self):
        cfg = self.corrected_config()
        cfg.sampling.predictor = 'guided'
        model = self.model(cfg)
        self.initialize(model)
        with torch.no_grad():
            for parameter in model.student_ema.parameters():
                parameter.add_(.2)
        model.train()
        model.teacher.eval()
        model.student_ema.eval()
        for p in model.backbone.parameters():
            p.grad = torch.ones_like(p)
        before = copy.deepcopy(model.state_dict())
        identities = {name: id(p) for name, p in model.named_parameters()}
        modes = [module.training for module in model.modules()]
        grads = {name: None if p.grad is None else p.grad.clone()
                 for name, p in model.named_parameters()}
        raw_calls, ema_calls = [], []
        raw_hook = model.backbone.register_forward_pre_hook(lambda *args: raw_calls.append(1))
        ema_hook = model.student_ema.register_forward_pre_hook(lambda *args: ema_calls.append(1))
        try:
            with torch.inference_mode():
                sample = model.generate_samples(2, num_steps=4)
            self.assertEqual(sample.shape, (2, 12))
            self.assertEqual(raw_calls, [])
            self.assertGreater(len(ema_calls), 0)
            ema_calls.clear()

            def validation_forward(instance, batch, batch_idx):
                return instance.forward(batch, torch.ones(len(batch), 1))

            tokens = model.prior_sample(2, 12)
            with mock.patch.object(self.original_class, 'validation_step', validation_forward):
                model.validation_step(tokens, 0)
            self.assertEqual(raw_calls, [])
            self.assertEqual(len(ema_calls), 1)
            model.forward(tokens, torch.ones(2, 1))
            self.assertEqual(len(raw_calls), 1)  # Training forward still raw.
        finally:
            raw_hook.remove()
            ema_hook.remove()
        self.assert_same(before, model.state_dict())
        self.assertEqual(identities, {name: id(p) for name, p in model.named_parameters()})
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(model._didi_evaluation_depth, 0)
        for name, p in model.named_parameters():
            if grads[name] is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)

    def test_ema_context_restores_modes_after_failure_and_rejects_missing_weights(self):
        cfg = self.corrected_config()
        model = self.model(cfg)
        modes = [module.training for module in model.modules()]
        with mock.patch.object(self.original_class, 'generate_samples', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                model.generate_samples(1, num_steps=4)
        self.assertEqual(model._didi_evaluation_depth, 0)
        self.assertEqual(modes, [module.training for module in model.modules()])
        cfg.mode = 'sample_eval'
        evaluated = self.model(cfg)
        state = {f'backbone.{k}': v for k, v in model.backbone.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'requires student_ema'):
            evaluated.on_load_checkpoint({'state_dict': state})
        with open_dict(cfg):
            cfg.didi_eval.weights = 'invalid'
        with self.assertRaisesRegex(ValueError, 'must be backbone or student_ema'):
            self.model(cfg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
