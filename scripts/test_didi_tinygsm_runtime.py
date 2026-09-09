"""CPU-only conditional DiDi tests; no data/model downloads or GPU are used."""

from pathlib import Path
import copy
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import open_dict
import torch

import test_didi_runtime as runtime_fixtures
from didi_repro.runtime import load_upstream, verify_upstream
from didi_repro.tinygsm_runtime import install_tinygsm_runtime


class TinyGSMRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.upstream = load_upstream()
        cls.owt_class = cls.upstream.algo.DiDiInstruct
        cls.model_class = install_tinygsm_runtime(cls.upstream)
        class TinyGSMTeacherFixture(runtime_fixtures.DiDiRuntimeTests):
            def initialize(self, model):
                teacher = {f'backbone.{key}': value.detach().clone() + .03
                           for key, value in model.backbone.state_dict().items()}
                # Native TinyGSM checkpoints carry separately trained EMA
                # shadows. They must not silently be replaced by raw weights.
                shadows = [teacher[f'backbone.{name}'].clone() + .007
                           for name, _ in model.backbone.named_parameters()]
                model.initialize_teacher_checkpoint({
                    'state_dict': teacher,
                    'ema': {'decay': .9999, 'num_updates': 12000,
                            'shadow_params': shadows},
                })
                return teacher

        cls.fixture = TinyGSMTeacherFixture()
        cls.fixture.upstream = cls.upstream
        cls.fixture.model_class = cls.model_class

    @classmethod
    def tearDownClass(cls):
        cls.upstream.algo.DiDiInstruct = cls.owt_class

    def config(self, source='main', sampler='ancestral_cache'):
        cfg = self.fixture.config(tiny=True)
        with open_dict(cfg):
            cfg.data.train = 'tiny_gsm'
            cfg.data.wrap = False
            cfg.data.train_on_prompt = False
            cfg.data.train_on_pad = True
            cfg.data.separator = '\n'
            cfg.didi_eval = {'weights': 'student_ema', 'sampler_source': source}
        cfg.sampling.predictor = sampler
        cfg.sampling.steps = 4
        cfg.algo.num_samples_per_prompt = 2
        cfg.algo.num_candidates = 2
        cfg.algo.guidance_scale_start = .1
        cfg.algo.guidance_scale_end = .2
        return cfg

    def model(self, **kwargs):
        model = self.fixture.model(self.config(**kwargs))
        self.fixture.initialize(model)
        return model

    def batch(self, train_on_pad=True):
        tokens = (torch.arange(24).reshape(2, 12) % 7).long()
        target = torch.ones_like(tokens)
        target[0, :3] = 0
        target[1, :5] = 0
        if not train_on_pad:
            target[0, 8:] = 0
            target[1, 10:] = 0
            tokens[target == 0] = 0
        return {'input_ids': tokens, 'attention_mask': target,
                'prompt_len': torch.tensor([3, 5])}

    def test_install_is_idempotent_and_owt_class_is_unchanged(self):
        self.assertIs(install_tinygsm_runtime(self.upstream), self.model_class)
        self.assertNotIn('_didi_tinygsm_installed', self.owt_class.__dict__)
        self.assertIs(self.model_class.__bases__[0], self.owt_class)
        fixture = runtime_fixtures.DiDiRuntimeTests()
        fixture.upstream = self.upstream
        fixture.model_class = self.owt_class
        model = fixture.model()
        self.assertTrue(bool((model._get_masked_input(2) == model.mask_index).all()))
        verify_upstream()

    def test_rejects_unconditional_or_wrapped_dataset_configuration(self):
        cfg = self.config()
        cfg.data.train = 'openwebtext-train'
        with self.assertRaisesRegex(ValueError, 'data=tiny-gsm'):
            self.fixture.model(cfg)
        cfg = self.config()
        cfg.data.wrap = True
        with self.assertRaisesRegex(ValueError, 'wrap=false'):
            self.fixture.model(cfg)
        cfg = self.config()
        cfg.data.train_on_prompt = True
        with self.assertRaisesRegex(ValueError, 'train_on_prompt=false'):
            self.fixture.model(cfg)

    def test_real_and_fake_corruption_keep_identical_fixed_context(self):
        model = self.model()
        for train_on_pad in (False, True):
            batch = self.batch(train_on_pad)
            tokens, target = batch['input_ids'], batch['attention_mask'].bool()
            fake = torch.full_like(tokens, 6)
            with model._target_context(tokens, target):
                initial = model._get_masked_input(2)
                self.assertTrue(torch.equal(initial[~target], tokens[~target]))
                self.assertTrue(bool((initial[target] == model.mask_index).all()))
                for clean in (tokens, fake):
                    corrupted, mask = model._corrupt_with_mask_ratio(clean, torch.zeros(2))
                    self.assertTrue(torch.equal(corrupted[~target], tokens[~target]))
                    self.assertTrue(torch.equal(mask, target))
                    self.assertTrue(bool((corrupted[target] == model.mask_index).all()))
            self.assertIsNone(model._didi_target_state)

    def test_both_trajectory_masks_and_logprob_gradients_exclude_prompt(self):
        model = self.model()
        batch = self.batch(False)
        tokens, target = batch['input_ids'], batch['attention_mask'].bool()
        with model._target_context(tokens, target):
            x_t = model._get_masked_input(2)
            x_mid, logp_hi, logits_hi, mask_hi = model._get_backward(x_t, 1., torch.tensor([.5, .5]))
            x_end, logp_lo, logits_lo, mask_lo = model._get_backward(x_mid, torch.tensor([.5, .5]), .001)
            self.assertTrue(torch.equal(mask_hi, target))
            self.assertFalse(bool(mask_lo[~target].any()))
            for output in (x_mid, x_end):
                self.assertTrue(torch.equal(output[~target], tokens[~target]))
            self.assertTrue(torch.isfinite(logp_hi).all())
            self.assertTrue(torch.isfinite(logp_lo).all())
            for logp, logits in ((logp_hi, logits_hi), (logp_lo, logits_lo)):
                gradient, = torch.autograd.grad(logp.sum(), logits, retain_graph=True)
                self.assertTrue(bool((gradient[~target] == 0).all()))

    def test_regularization_has_no_direct_prompt_or_fixed_pad_gradient(self):
        model = self.model()
        batch = self.batch(False)
        tokens, target = batch['input_ids'], batch['attention_mask'].bool()
        logits_hi = torch.randn(2, 12, model.vocab_size, requires_grad=True)
        logits_lo = torch.randn(2, 12, model.vocab_size, requires_grad=True)
        loss_kl, entropy = model._get_regularization(
            2, tokens, tokens, torch.ones(2), logits_hi, target, logits_lo, target)
        (loss_kl - .1 * entropy).backward()
        for value in (logits_hi.grad, logits_lo.grad):
            self.assertTrue(bool((value[~target] == 0).all()))
            self.assertGreater(float(value[target].abs().sum()), 0)

    def test_validation_nll_corrupts_answer_but_never_prompt(self):
        model = self.model()
        batch = self.batch(False)
        tokens, target = batch['input_ids'], batch['attention_mask'].bool()
        seen = []
        def forward_hook(module, inputs, output):
            del module, output
            seen.append(inputs[0].clone())
        hook = model.student_ema.register_forward_hook(forward_hook)
        # _loss is also valid outside the runtime evaluation context and then
        # uses the raw student. Hook both copies to cover either routing.
        other = model.backbone.register_forward_hook(forward_hook)
        try:
            with mock.patch.object(model, '_sample_t', return_value=torch.ones(2)):
                losses = model._loss(tokens, batch['attention_mask'])
        finally:
            hook.remove()
            other.remove()
        self.assertTrue(torch.isfinite(losses.loss))
        self.assertEqual(int(losses.num_tokens), int(target.sum()))
        self.assertTrue(seen)
        for value in seen:
            self.assertTrue(torch.equal(value[~target], tokens[~target]))
        self.assertIsNone(model._didi_target_state)

    def test_generation_masks_whole_suffix_and_ignores_supplied_gold_tail(self):
        model = self.model()
        batch = self.batch(False)
        tokens = batch['input_ids']
        lengths = batch['prompt_len']
        seen = []
        def fake_generation(instance, num_samples, num_steps, eps):
            del num_steps, eps
            value = instance.prior_sample(num_samples, instance.num_tokens)
            seen.append(value.clone())
            return torch.full_like(value, 6)  # final clamp must also work
        with mock.patch.object(self.owt_class, 'generate_samples', fake_generation):
            result = model.generate_samples(2, prefix_tokens=tokens, prefix_lengths=lengths)
        for row, length in enumerate(lengths):
            self.assertTrue(torch.equal(seen[0][row, :length], tokens[row, :length]))
            self.assertTrue(bool((seen[0][row, length:] == model.mask_index).all()))
            self.assertTrue(torch.equal(result[row, :length], tokens[row, :length]))
            self.assertTrue(bool((result[row, length:] == 6).all()))
        self.assertIsNone(model._didi_target_state)
        self.assertEqual(len(model.discriminator._forward_hooks), 0)

    def test_actual_ancestral_and_both_guided_samplers_preserve_uneven_prefixes(self):
        batch = self.batch()
        tokens, lengths = batch['input_ids'], batch['prompt_len']
        for source, sampler in (('main', 'ancestral_cache'), ('main', 'guided'),
                                ('inference', 'guided')):
            with self.subTest(source=source, sampler=sampler):
                model = self.model(source=source, sampler=sampler)
                with torch.inference_mode():
                    output = model.generate_samples(
                        2, num_steps=4, prefix_tokens=tokens[:, :5], prefix_lengths=lengths)
                self.assertEqual(output.shape, (2, 12))
                for row, length in enumerate(lengths):
                    self.assertTrue(torch.equal(output[row, :length], tokens[row, :length]))
                self.assertIsNone(model._didi_target_state)
                self.assertEqual(len(model.discriminator._forward_hooks), 0)
                self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()))

    def test_reranking_scores_only_suffix_for_every_candidate(self):
        model = self.model()
        batch = self.batch()
        target = batch['attention_mask'].bool()
        logits = torch.ones(6, 12, 1, requires_grad=True)
        with model._target_context(batch['input_ids'], target):
            scores = model._conditional_discriminator_output(None, None, logits)
        expected = target.repeat_interleave(3, dim=0)
        self.assertTrue(torch.equal(scores.squeeze(-1), expected.float()))
        scores.sum().backward()
        self.assertTrue(torch.equal(logits.grad.squeeze(-1), expected.float()))

    def test_invalid_prefixes_and_unconditional_sampling_fail_clearly(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'requires prefix_tokens'):
            model.generate_samples(2)
        tokens = torch.zeros(2, 5, dtype=torch.long)
        for lengths in (torch.tensor([0, 5]), torch.tensor([3, 6]),
                        torch.tensor([3., 5.]), torch.tensor([3])):
            with self.assertRaises(ValueError):
                model.generate_samples(2, prefix_tokens=tokens, prefix_lengths=lengths)

    def test_context_and_hooks_restored_after_sampler_failure(self):
        model = self.model()
        batch = self.batch()
        with mock.patch.object(self.owt_class, 'generate_samples', side_effect=RuntimeError('test')):
            with self.assertRaisesRegex(RuntimeError, 'test'):
                model.generate_samples(2, prefix_tokens=batch['input_ids'],
                                       prefix_lengths=batch['prompt_len'])
        self.assertIsNone(model._didi_target_state)
        self.assertEqual(len(model.discriminator._forward_hooks), 0)

    def test_empty_corruption_mask_keeps_reward_diagnostics_finite(self):
        model = self.model()
        model._trainer = types.SimpleNamespace(world_size=1)
        batch = self.batch()
        with mock.patch.object(model, 'log_dict'):
            result = model._get_rewards(batch['input_ids'], torch.ones(2),
                                        torch.zeros(2, 12, dtype=torch.bool), 1, 2)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in result))
        self.assertTrue(bool((result[0] == 0).all()))

    def test_two_optimizer_cpu_training_keeps_teacher_and_fixed_tokens(self):
        model = self.model()
        batch = self.batch(False)
        teacher_before = {key: value.detach().clone() for key, value in model.teacher.state_dict().items()}
        fixed = batch['input_ids'].repeat_interleave(2, dim=0)
        target = batch['attention_mask'].bool().repeat_interleave(2, dim=0)
        original = model._get_backward
        count = []
        discriminator_outputs = []
        def check_discriminator(module, inputs, output):
            del module
            self.assertTrue(torch.equal(inputs[0][~target], fixed[~target]))
            if output.requires_grad:
                output.retain_grad()
                discriminator_outputs.append(output)
        def checked_backward(instance, x_prev, t_hi, t_lo):
            del instance
            result = original(x_prev, t_hi, t_lo)
            count.append(1)
            self.assertTrue(torch.equal(x_prev[~target], fixed[~target]))
            self.assertTrue(torch.equal(result[0][~target], fixed[~target]))
            self.assertFalse(bool(result[3][~target].any()))
            return result
        model._get_backward = types.MethodType(checked_backward, model)
        hook = model.discriminator.register_forward_hook(check_discriminator)
        try:
            with tempfile.TemporaryDirectory() as directory:
                trainer = self.fixture.trainer(directory, max_steps=2)
                trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader([batch], batch_size=None))
        finally:
            hook.remove()
        self.assertEqual(model.global_steps, 1)
        self.assertEqual(len(count), 2)
        self.assertEqual(len(discriminator_outputs), 2)
        for output in discriminator_outputs:
            self.assertTrue(bool((output.grad.squeeze(-1)[~target] == 0).all()))
        self.assertIsNone(model._didi_target_state)
        for key, value in model.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)

    def checkpoint(self, model):
        checkpoint = {'state_dict': {key: value.detach().clone()
                                     for key, value in model.state_dict().items()}}
        model.on_save_checkpoint(checkpoint)
        return checkpoint

    def test_conditional_checkpoint_roundtrip_restores_weights_and_iteration(self):
        model = self.model()
        model.global_steps = 17
        checkpoint = self.checkpoint(model)
        metadata = checkpoint['didi_tinygsm']
        self.assertEqual(metadata['schema'], 'didi-tinygsm-conditional')
        self.assertEqual(metadata['version'], 1)
        self.assertEqual(metadata['data.train'], 'tiny_gsm')
        self.assertEqual(metadata['model.length'], 12)
        self.assertEqual(metadata['data.separator'], '\n')
        self.assertTrue(metadata['data.train_on_pad'])
        restored = self.model()
        restored.on_load_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint['state_dict'])
        self.assertEqual(restored.global_steps, 17)
        self.assertEqual(restored._didi_teacher_metadata, model._didi_teacher_metadata)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_conditional_checkpoint_rejects_owt_and_missing_task_metadata(self):
        model = self.model()
        checkpoint = self.checkpoint(model)
        del checkpoint['didi_tinygsm']
        with self.assertRaisesRegex(ValueError, 'missing conditional-task metadata'):
            model.on_load_checkpoint(checkpoint)
        # A native MDLM teacher is initialization, not a DiDi resume. It still
        # needs no DiDi-specific metadata and populates every teacher copy.
        self.fixture.initialize(model)
        self.assertIsNotNone(model._didi_teacher_metadata)

    def test_conditional_checkpoint_rejects_changed_tokenization_and_mask_semantics(self):
        model = self.model()
        checkpoint = self.checkpoint(model)
        replacements = {
            'schema': 'other-schema',
            'version': 2,
            'data.train': 'openwebtext-train',
            'data.tokenizer_name_or_path': 'different/tokenizer',
            'data.separator': '\n### Answer:\n',
            'data.train_on_prompt': True,
            'data.train_on_pad': False,
            'data.wrap': True,
            'model.length': 1024,
            'target_mask': 'all-tokens',
            'generation_target': 'gold-answer-span',
            'special_token_ids': {'bos': 1, 'eos': 2, 'pad': 3, 'mask': 4},
        }
        for key, value in replacements.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(checkpoint)
                changed['didi_tinygsm'][key] = value
                with self.assertRaisesRegex(ValueError, 'conditional-task mismatch'):
                    model.on_load_checkpoint(changed)
        # Also compare against a genuinely different current configuration,
        # rather than only mutating saved dictionaries.
        cfg = self.config()
        cfg.data.separator = '\nAnswer: '
        different = self.fixture.model(cfg)
        with self.assertRaisesRegex(ValueError, 'data.separator'):
            different.on_load_checkpoint(checkpoint)

    def test_checkpoint_guard_also_protects_evaluation_loading(self):
        cfg = self.config()
        cfg.mode = 'sample_eval'
        model = self.fixture.model(cfg)
        with self.assertRaisesRegex(ValueError, 'conditional-task metadata'):
            model.on_load_checkpoint({'state_dict': {}})

    def test_lightning_checkpoint_file_can_resume_conditional_training(self):
        model = self.model()
        batch = self.batch()
        loader = torch.utils.data.DataLoader([batch], batch_size=None)
        with tempfile.TemporaryDirectory() as directory:
            first = self.fixture.trainer(directory, max_steps=2)
            first.fit(model, train_dataloaders=loader)
            checkpoint_path = str(Path(directory) / 'conditional-resume.ckpt')
            first.save_checkpoint(checkpoint_path)
            on_disk = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.assertEqual(on_disk['didi_tinygsm']['schema'], 'didi-tinygsm-conditional')
            restored = self.model()
            second = self.fixture.trainer(directory, max_steps=4)
            second.fit(restored, train_dataloaders=loader, ckpt_path=checkpoint_path)
            self.assertEqual(restored.global_steps, 2)
            self.assertEqual(second.global_step, 4)
            self.assertEqual(restored._didi_teacher_metadata, model._didi_teacher_metadata)


if __name__ == '__main__':
    # Do not rediscover the imported fixture's OWT suite after installing a
    # different class into the flat upstream module in this dedicated process.
    unittest.main(defaultTest='TinyGSMRuntimeTests')
