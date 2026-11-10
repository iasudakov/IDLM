"""CPU MMD lifecycle: CDLM warm start, independent frozen features and resume."""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import lightning as L
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from mmd_finetune import MMDFinetune, answer_content_mask
from mmd_objective import reinforce_loss_from_samples
from test_cdlm_training import (Original, Output, Tokenizer, config as cdlm_config)


class Denoiser(torch.nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.output_layer = Output(vocab_size)
        self.dropout = torch.nn.Dropout(.15)

    def forward(self, tokens, sigma):
        hidden = self.embedding(tokens)
        hidden = self.dropout(hidden + hidden.mean(1, keepdim=True))
        for block in self.blocks:
            hidden = block(hidden).tanh()
        return self.output_layer(hidden)


def config(directory):
    cfg = cdlm_config(directory)
    cfg.algo.name = 'mmd_finetune'
    cfg.eval.generate_samples = False
    cfg.checkpointing.snapshot = False
    cfg.lr_scheduler.num_warmup_steps = 0
    OmegaConf.update(cfg, 'mmd', {'feature_layer': 0, 'kernel': 'rbf',
        'rbf_alpha': .3, 'kernel_block_size': 0, 'feature_batch_size': 2,
        'group_size': 8, 'draws_per_candidate': 1, 'average_candidates': False,
        'student_forward_batch_size': 1, 'student_activation_checkpointing': True}, force_add=True)
    return cfg


def trainer(steps, **kwargs):
    return L.Trainer(accelerator='cpu', devices=1, max_steps=steps, max_epochs=-1,
                     accumulate_grad_batches=2, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False,
                     num_sanity_val_steps=0, **kwargs)


class WarmStartTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(741)
        self.enterContext(patch('models.dit.DIT', Denoiser))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = config(str(self.root))
        self.model = MMDFinetune(self.cfg, Tokenizer())
        self.student_file, self.feature_file = self.root / 'cdlm.ckpt', self.root / 'mdlm.ckpt'
        self.student = {k: v.detach().clone() for k, v in self.model.backbone.state_dict().items()}
        self.feature = {k: v + .4 for k, v in self.student.items()}
        self.ema = {'shadow_params': [value + .1 for _, value in self.model.backbone.named_parameters()],
                    'decay': .97, 'num_updates': 1248}
        self.write_student()
        torch.save({'state_dict': {'backbone.' + k: v for k, v in self.feature.items()}}, self.feature_file)

    def write_student(self, *, state=None, ema=True, ema_only=False):
        state = self.student if state is None else state
        payload = {'state_dict': {
            **{'backbone._orig_mod.' + name: value for name, value in state.items()},
            'teacher_head.weight': torch.ones(17, 5),
            'teacher_head.bias': torch.ones(17),
            'teacher.unused': torch.ones(2), 'fake_model.unused': torch.ones(3)},
            'optimizer_states': [{'unrelated_cdlm_optimizer': True}],
            'global_step': 1248, 'epoch': 16,
            'loops': {'old_cdlm_training_state': True}}
        if ema:
            payload['ema'] = self.ema
        if ema_only:
            payload['checkpoint_weights'] = 'ema'
        torch.save(payload, self.student_file)

    def initialize(self, **kwargs):
        self.model.initialize_from_checkpoint(str(self.student_file), str(self.feature_file), **kwargs)

    def test_raw_student_retains_ema_but_does_not_import_teacher_head_or_loops(self):
        self.initialize()
        for key, actual in self.model.backbone.state_dict().items():
            self.assertTrue(torch.equal(actual, self.student[key]))
        for key, actual in self.model.teacher.state_dict().items():
            self.assertTrue(torch.equal(actual, self.feature[key]))
        self.assertEqual(self.model.ema.num_updates, 1248)
        self.assertEqual(self.model.ema.decay, .97)
        for actual, expected in zip(self.model.ema.shadow_params, self.ema['shadow_params']):
            self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(self.model.fast_forward_epochs)
        self.assertIsNone(self.model.fast_forward_batches)
        self.assertFalse(hasattr(self.model, 'teacher_head'))
        self.assertEqual(self.model.global_step, 0)
        optimizers, _ = self.model.configure_optimizers()
        self.assertFalse(optimizers[0].state)
        self.assertTrue(all(not p.requires_grad for p in self.model.teacher.parameters()))
        self.model.train()
        self.assertFalse(self.model.teacher.training)

    def test_raw_reset_policy_and_selected_ema_reset_policy(self):
        for use_ema in (False, True):
            with self.subTest(use_ema=use_ema):
                self.initialize(student_use_ema=use_ema, retain_ema=False)
                expected = (self.ema['shadow_params'] if use_ema else
                            [self.student[k] for k, _ in self.model.backbone.named_parameters()])
                for actual, shadow, target in zip(self.model.backbone.parameters(),
                                                 self.model.ema.shadow_params, expected):
                    self.assertTrue(torch.equal(actual, target))
                    self.assertTrue(torch.equal(shadow, target))
                self.assertEqual(self.model.ema.num_updates, 0)
                self.assertEqual(self.model.ema.decay, self.cfg.training.ema)

    def test_ema_selection_requires_explicit_history_reset(self):
        with self.assertRaisesRegex(ValueError, 'retain_ema=False'):
            self.initialize(student_use_ema=True)

    def test_ema_only_student_requires_selection_and_reset(self):
        self.write_student(ema=False, ema_only=True)
        with self.assertRaisesRegex(ValueError, 'only EMA'):
            self.initialize(retain_ema=False)
        self.initialize(student_use_ema=True, retain_ema=False)
        self.assertEqual(self.model.ema.num_updates, 0)
        for key, actual in self.model.backbone.state_dict().items():
            self.assertTrue(torch.equal(actual, self.student[key]))

    def test_retaining_missing_incompatible_or_invalid_ema_is_rejected(self):
        self.write_student(ema=False)
        with self.assertRaisesRegex(ValueError, 'Retaining EMA'):
            self.initialize()
        original = self.ema
        for change in ({'shadow_params': [torch.zeros(1)]},
                       {'num_updates': -1}, {'decay': float('nan')}):
            with self.subTest(change=change):
                self.ema = {**original, **change}
                self.write_student()
                with self.assertRaises(ValueError):
                    self.initialize()

    def test_strict_student_and_feature_shapes(self):
        wrong = {**self.student, 'output_layer.linear.weight': torch.zeros(2, 3)}
        self.write_student(state=wrong)
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            self.initialize()
        self.write_student()
        torch.save({'state_dict': {'backbone.' + k: v for k, v in wrong.items()}}, self.feature_file)
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            self.initialize()

    def test_optional_feature_ema_affects_only_features(self):
        torch.save({'state_dict': {'backbone.' + k: v for k, v in self.feature.items()},
                    'ema': self.ema}, self.feature_file)
        self.initialize(feature_use_ema=True)
        for actual, expected in zip(self.model.teacher.parameters(), self.ema['shadow_params']):
            self.assertTrue(torch.equal(actual, expected))
        for key, actual in self.model.backbone.state_dict().items():
            self.assertTrue(torch.equal(actual, self.student[key]))

    def test_feature_checkpoint_missing_does_not_silently_copy_student(self):
        self.feature_file.unlink()
        with self.assertRaises(FileNotFoundError):
            self.initialize()

    def test_cdlm_checkpoint_cannot_be_used_as_a_full_mmd_resume(self):
        with self.assertRaisesRegex(ValueError, 'full MMD-finetune checkpoint'):
            self.model.on_load_checkpoint(torch.load(self.student_file, weights_only=False))

    def test_checkpointed_physical_chunks_preserve_gradients_and_full_batch_objective(self):
        self.initialize()
        self.cfg.mmd.student_forward_batch_size = 2
        original = torch.tensor([[0, 4, 5, 6, 1, 2]] * 3)
        selected = torch.tensor([[0, 0, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 0],
                                 [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
        corrupted = torch.where(selected, self.model.mask_index, original)
        sigma = torch.ones(3, 1)
        results = []
        for checkpointed in (False, True):
            self.cfg.mmd.student_activation_checkpointing = checkpointed
            torch.manual_seed(511)
            sizes = []
            handle = self.model.backbone.register_forward_pre_hook(
                lambda module, inputs: sizes.append(inputs[0].shape[0]))
            seen_rewards = []

            def reward_fn(clean, draws, written):
                seen_rewards.append(tuple(clean.shape))
                weights = torch.arange(1, clean.shape[1] + 1, dtype=torch.float32)
                return (draws[0].float() * weights * written).sum(-1), written.any(-1)

            try:
                logs, samples = self.model._sample_student_outputs(corrupted, sigma, original, selected)
                self.assertEqual(tuple(logs.shape), (8, 3, 6))
                self.assertEqual(tuple(samples.shape), (8, 3, 6))
                self.assertEqual(sizes, [2, 1])
                self.assertTrue(torch.equal(samples[:, ~selected], original[~selected].expand(8, -1)))
                loss, _ = reinforce_loss_from_samples(logs, original, selected, samples, reward_fn,
                                                      group_size=8, average_candidates=False)
                # The reward observes all three examples, including the final
                # short physical chunk, exactly once per group member.
                self.assertEqual(seen_rewards, [(3, 6)] * 8)
                with torch.no_grad():
                    reward = torch.stack([reward_fn(original, sample[None], selected)[0]
                                          for sample in samples], dim=-1)
                    advantage = reward - (reward.sum(-1, keepdim=True) - reward) / 7
                expected = -(advantage.T * (logs * selected[None]).sum(-1)).sum() / selected.sum()
                self.assertTrue(torch.allclose(loss, expected, rtol=1e-6, atol=1e-6))
                gradients = torch.autograd.grad(loss, tuple(self.model.backbone.parameters()))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
                self.assertEqual(len(sizes), 4 if checkpointed else 2)
                results.append((samples.detach(), logs.detach(), loss.detach(), gradients))
            finally:
                handle.remove()
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        for ordinary, checkpointed in zip(results[0][3], results[1][3]):
            self.assertTrue(torch.equal(ordinary, checkpointed))

    def test_forward_chunk_configuration_rejects_invalid_values(self):
        for chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=chunk):
                self.cfg.mmd.student_forward_batch_size = chunk
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    MMDFinetune(self.cfg, Tokenizer())

    def test_full_training_resume_and_student_only_ema_inference(self):
        manifest = self.root / 'initialization.json'
        provenance = {'source_namespace': 'cdlm_tinygsm', 'global_step': 1248,
                      'source_sha256': 'abc123'}
        manifest.write_text(json.dumps(provenance))
        OmegaConf.update(self.cfg, 'mmd.initialization_manifest', str(manifest), force_add=True)
        self.initialize()
        # Denoising masks are stochastic; these seed/data ensure real MMD updates.
        torch.manual_seed(8)
        loader = DataLoader([Original()[0] for _ in range(16)], batch_size=2)
        fit = trainer(2, limit_val_batches=0)
        fit.fit(self.model, loader)
        self.assertEqual(fit.global_step, 2)
        self.assertEqual(self.model.ema.num_updates, 1250)
        self.assertTrue(any(not torch.equal(value, self.student[key])
                            for key, value in self.model.backbone.state_dict().items()))
        for key, actual in self.model.teacher.state_dict().items():
            self.assertTrue(torch.equal(actual, self.feature[key]))
        self.assertTrue(all(p.grad is None for p in self.model.teacher.parameters()))
        checkpoint = self.root / 'checkpoints/last.ckpt'
        saved = torch.load(checkpoint, weights_only=False)
        self.assertEqual(saved['global_step'], 2)
        self.assertTrue(saved['optimizer_states'])
        self.assertEqual(saved['mmd_finetune_schema'], 1)
        self.assertEqual(saved['mmd_initialization']['manifest'], provenance)
        self.assertTrue(any(key.startswith('teacher.') for key in saved['state_dict']))
        self.assertFalse(any(key.startswith('teacher_head.') for key in saved['state_dict']))
        self.student_file.unlink()
        self.feature_file.unlink()
        manifest.unlink()
        restored = MMDFinetune(self.cfg, Tokenizer())
        resumed = trainer(3, limit_val_batches=0)
        with patch.object(restored, 'initialize_from_checkpoint', side_effect=AssertionError('not a warm start')):
            resumed.fit(restored, loader, ckpt_path=str(checkpoint))
        self.assertEqual(resumed.global_step, 3)
        self.assertEqual(restored.ema.num_updates, 1251)
        self.assertEqual(restored._mmd_initialization['manifest'], provenance)
        for key, actual in restored.teacher.state_dict().items():
            self.assertTrue(torch.equal(actual, self.feature[key]))
        inference = MMDFinetune(self.cfg, Tokenizer(), inference_only=True)
        inference.load_evaluation_weights(str(checkpoint))
        self.assertFalse(hasattr(inference, 'teacher'))
        self.assertIsNone(inference.ema)
        self.assertEqual(inference._evaluation_weights, 'ema')
        final = torch.load(checkpoint, weights_only=False)
        for actual, expected in zip(inference.backbone.parameters(), final['ema']['shadow_params']):
            self.assertTrue(torch.equal(actual, expected))

    def test_initial_periodic_and_final_evaluation_hooks(self):
        self.initialize()
        self.cfg.eval.validate_before_train = True
        self.cfg.eval.gsm8k_at_validation = True
        loader = DataLoader([Original()[0] for _ in range(16)], batch_size=2)
        seen = []

        def evaluate(module, **kwargs):
            self.assertFalse(module.backbone.training)
            for actual, shadow in zip(module._get_parameters(), module.ema.shadow_params):
                self.assertTrue(torch.equal(actual, shadow))
            seen.append((module.global_step, kwargs))

        with patch('sdtt_eval.log_validation_gsm8k', side_effect=evaluate):
            trainer(3, limit_val_batches=1, val_check_interval=4).fit(self.model, loader, loader)
        self.assertIn((0, {'initial': True}), seen)
        self.assertIn((2, {}), seen)
        self.assertIn((3, {'final': True}), seen)


class ProtocolTest(unittest.TestCase):
    def test_answer_mask_preserves_first_answer_eos_excludes_prompt_eos_and_padding(self):
        tokens = torch.tensor([[0, 1, 3, 4, 1, 5, 2]])
        valid = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
        actual = answer_content_mask(tokens, valid, Tokenizer())
        self.assertTrue(torch.equal(actual, torch.tensor([[False, False, False, True, True, False, False]])))

    def test_final_checkpoint_precedes_evaluation_and_snapshot_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []

            def save(path):
                Path(path).write_bytes(b'checkpoint')
                events.append('checkpoint')

            def evaluate(*args, **kwargs):
                events.append('eval')
                raise RuntimeError('failed evaluation')

            model = SimpleNamespace(
                config=OmegaConf.create({'checkpointing': {'save_dir': directory,
                    'snapshot': True, 'snapshot_namespace': 'mmd_from_cdlm'}}),
                trainer=SimpleNamespace(save_checkpoint=save), global_step=7000,
                _eval_mode=Mock(side_effect=lambda: events.append('ema')),
                _train_mode=Mock(side_effect=lambda: events.append('raw')))
            with patch('sdtt_eval.log_validation_gsm8k', side_effect=evaluate), \
                 patch('nirvana_snapshot.save', side_effect=lambda *a, **k: events.append('snapshot')):
                with self.assertRaisesRegex(RuntimeError, 'failed evaluation'):
                    MMDFinetune.on_train_end(model)
            self.assertEqual(events, ['checkpoint', 'ema', 'eval', 'raw', 'snapshot'])


if __name__ == '__main__':
    unittest.main()
