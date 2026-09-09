"""Native TinyGSM EMA initialization regression tests; CPU-only and offline."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import open_dict
import torch

import test_didi_tinygsm_runtime as runtime_fixtures


class NativeEMAInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime_fixtures.TinyGSMRuntimeTests.setUpClass()
        cls.helpers = runtime_fixtures.TinyGSMRuntimeTests()

    @classmethod
    def tearDownClass(cls):
        runtime_fixtures.TinyGSMRuntimeTests.tearDownClass()

    def uninitialized(self, source=None):
        config = self.helpers.config()
        if source is not None:
            with open_dict(config):
                config.didi_init = {'tinygsm_student_ema': source}
        return self.helpers.fixture.model(config)

    def native_checkpoint(self, model):
        raw = {f'backbone.{key}': value.detach().clone() + .031
               for key, value in model.backbone.state_dict().items()}
        shadows = []
        for index, (name, _) in enumerate(model.backbone.named_parameters()):
            tensor = raw[f'backbone.{name}']
            # Distinct values for each slot AND within each tensor make wrong
            # positional mapping visible even among same-shaped parameters.
            pattern = (torch.arange(tensor.numel(), dtype=tensor.dtype)
                       .reshape(tensor.shape).remainder(7) * .003)
            shadows.append(tensor.clone() + .017 * (index + 1) + pattern)
        return {'state_dict': raw,
                'ema': {'decay': .9999, 'num_updates': 12000,
                        'shadow_params': shadows}}

    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, value in actual.items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def assert_rejected_without_mutation(self, model, checkpoint):
        before = copy.deepcopy(model.state_dict())
        old_teacher = copy.deepcopy(model._didi_teacher_metadata)
        old_student = copy.deepcopy(model._didi_student_metadata)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            model.initialize_teacher_checkpoint(checkpoint)
        self.assert_state_equal(model.state_dict(), before)
        self.assertEqual(model._didi_teacher_metadata, old_teacher)
        self.assertEqual(model._didi_student_metadata, old_student)

    def test_default_only_student_ema_uses_native_checkpoint_shadows_in_order(self):
        model = self.uninitialized()
        checkpoint = self.native_checkpoint(model)
        discriminator_head = copy.deepcopy(model.discriminator.output_layer.state_dict())
        model.initialize_teacher_checkpoint(checkpoint)
        raw = {key.removeprefix('backbone.'): value for key, value in checkpoint['state_dict'].items()}
        self.assert_state_equal(model.backbone.state_dict(), raw)
        self.assert_state_equal(model.teacher.state_dict(), raw)
        self.assert_state_equal(model.discriminator.output_layer.state_dict(), discriminator_head)
        for name, parameter in model.discriminator.named_parameters():
            if not name.startswith('output_layer.'):
                torch.testing.assert_close(parameter, raw[name], rtol=0, atol=0)
        for (name, parameter), shadow in zip(model.student_ema.named_parameters(), checkpoint['ema']['shadow_params']):
            torch.testing.assert_close(parameter, shadow, rtol=0, atol=0)
            self.assertFalse(torch.equal(parameter, raw[name]), name)
        self.assertEqual(model._didi_student_metadata['kind'], 'native_tinygsm_initialization')
        metadata = model._didi_student_metadata['student_ema']
        self.assertEqual(metadata['source'], 'checkpoint_ema')
        self.assertEqual(metadata['checkpoint_num_updates'], 12000)
        self.assertEqual(metadata['checkpoint_decay'], .9999)
        self.assertEqual(metadata['parameter_count'], len(list(model.backbone.parameters())))
        self.assertEqual(metadata['raw_different_parameter_count'], metadata['parameter_count'])
        self.assertEqual(model.ema_beta, .999)  # Distillation EMA decay is NOT checkpoint decay.

    def test_shadow_tensor_storage_is_independent_of_input_and_other_model_copies(self):
        model = self.uninitialized()
        checkpoint = self.native_checkpoint(model)
        expected = checkpoint['ema']['shadow_params'][0].clone()
        model.initialize_teacher_checkpoint(checkpoint)
        ema_parameter = next(model.student_ema.parameters())
        raw_parameter = next(model.backbone.parameters())
        teacher_parameter = next(model.teacher.parameters())
        self.assertNotEqual(ema_parameter.data_ptr(), checkpoint['ema']['shadow_params'][0].data_ptr())
        self.assertNotEqual(ema_parameter.data_ptr(), raw_parameter.data_ptr())
        self.assertNotEqual(raw_parameter.data_ptr(), teacher_parameter.data_ptr())
        checkpoint['ema']['shadow_params'][0].zero_()
        torch.testing.assert_close(ema_parameter, expected, rtol=0, atol=0)
        raw_before = raw_parameter.clone()
        with torch.no_grad():
            ema_parameter.add_(1)
        torch.testing.assert_close(raw_parameter, raw_before, rtol=0, atol=0)
        self.assertTrue(bool((checkpoint['ema']['shadow_params'][0] == 0).all()))

    def test_missing_ema_requires_explicit_raw_mode(self):
        model = self.uninitialized()
        checkpoint = self.native_checkpoint(model)
        del checkpoint['ema']
        self.assert_rejected_without_mutation(model, checkpoint)
        raw_model = self.uninitialized('raw')
        raw_model.initialize_teacher_checkpoint(checkpoint)
        self.assert_state_equal(raw_model.student_ema.state_dict(), raw_model.backbone.state_dict())
        self.assertIn('raw_backbone', json.dumps(raw_model._didi_student_metadata))

    def test_explicit_raw_mode_ignores_even_malformed_unused_ema(self):
        model = self.uninitialized('raw')
        checkpoint = self.native_checkpoint(model)
        checkpoint['ema'] = {'shadow_params': 'intentionally invalid and unused'}
        model.initialize_teacher_checkpoint(checkpoint)
        self.assert_state_equal(model.student_ema.state_dict(), model.backbone.state_dict())

    def test_wrong_shadow_count_or_container_is_rejected_before_raw_loading(self):
        for corruption in ('missing', 'none', 'dict', 'short', 'long'):
            with self.subTest(corruption=corruption):
                model = self.uninitialized()
                checkpoint = self.native_checkpoint(model)
                shadows = checkpoint['ema']['shadow_params']
                if corruption == 'missing':
                    del checkpoint['ema']['shadow_params']
                elif corruption == 'none':
                    checkpoint['ema']['shadow_params'] = None
                elif corruption == 'dict':
                    checkpoint['ema']['shadow_params'] = {'wrong': shadows[0]}
                elif corruption == 'short':
                    checkpoint['ema']['shadow_params'] = shadows[:-1]
                else:
                    checkpoint['ema']['shadow_params'] = shadows + [shadows[0].clone()]
                self.assert_rejected_without_mutation(model, checkpoint)

    def test_wrong_shapes_and_different_shape_ordering_are_rejected(self):
        for corruption in ('shape', 'order', 'non_tensor'):
            with self.subTest(corruption=corruption):
                model = self.uninitialized()
                checkpoint = self.native_checkpoint(model)
                shadows = checkpoint['ema']['shadow_params']
                if corruption == 'shape':
                    shadows[0] = torch.zeros(1)
                elif corruption == 'order':
                    other = next(index for index, value in enumerate(shadows) if value.shape != shadows[0].shape)
                    shadows[0], shadows[other] = shadows[other], shadows[0]
                else:
                    shadows[0] = shadows[0].tolist()
                self.assert_rejected_without_mutation(model, checkpoint)

    def test_nonfinite_shadow_tensors_are_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            model = self.uninitialized()
            checkpoint = self.native_checkpoint(model)
            checkpoint['ema']['shadow_params'][0].flatten()[0] = value
            self.assert_rejected_without_mutation(model, checkpoint)

    def test_raw_checkpoint_parameter_order_must_match_native_layout(self):
        model = self.uninitialized()
        checkpoint = self.native_checkpoint(model)
        checkpoint['state_dict'] = dict(reversed(list(checkpoint['state_dict'].items())))
        self.assert_rejected_without_mutation(model, checkpoint)

    def test_raw_rotary_style_buffers_are_not_ema_parameters(self):
        model = self.uninitialized()
        # The real native DiT has a rotary buffer in state_dict but not in its
        # EMA parameter list. Preserve that distinction in the toy fixture.
        for module in (model.backbone, model.teacher, model.student_ema, model.discriminator):
            module.register_buffer('rotary_test_buffer', torch.tensor([.1, .2, .3]))
        checkpoint = self.native_checkpoint(model)
        model.initialize_teacher_checkpoint(checkpoint)
        expected = checkpoint['state_dict']['backbone.rotary_test_buffer']
        for module in (model.backbone, model.teacher, model.student_ema, model.discriminator):
            torch.testing.assert_close(module.rotary_test_buffer, expected, rtol=0, atol=0)
        self.assertEqual(model._didi_student_metadata['student_ema']['parameter_count'],
                         len(checkpoint['ema']['shadow_params']))
        self.assertEqual(len(model.backbone.state_dict()), len(checkpoint['ema']['shadow_params']) + 1)

    def test_incompatible_shadow_dtype_or_destination_overflow_is_rejected(self):
        for transform in (lambda value: value.long(), lambda value: value.double().fill_(1e300)):
            model = self.uninitialized()
            checkpoint = self.native_checkpoint(model)
            checkpoint['ema']['shadow_params'][0] = transform(checkpoint['ema']['shadow_params'][0])
            self.assert_rejected_without_mutation(model, checkpoint)

    def test_missing_or_invalid_ema_decay_and_counter_are_rejected(self):
        for key, value in (('decay', None), ('decay', -1), ('decay', float('nan')),
                           ('num_updates', -1), ('num_updates', 1.5), ('num_updates', True)):
            model = self.uninitialized()
            checkpoint = self.native_checkpoint(model)
            checkpoint['ema'][key] = value
            self.assert_rejected_without_mutation(model, checkpoint)
        for key in ('decay', 'num_updates'):
            model = self.uninitialized()
            checkpoint = self.native_checkpoint(model)
            del checkpoint['ema'][key]
            self.assert_rejected_without_mutation(model, checkpoint)

    def test_unsupported_frozen_parameter_or_trainable_noise_layout_is_rejected(self):
        for changed in ('frozen', 'noise'):
            model = self.uninitialized()
            if changed == 'frozen':
                next(model.backbone.parameters()).requires_grad_(False)
            else:
                model.noise.register_parameter('unsupported', torch.nn.Parameter(torch.tensor(1.)))
            checkpoint = self.native_checkpoint(model)
            self.assert_rejected_without_mutation(model, checkpoint)

    def test_step_zero_eval_logits_use_native_ema_but_training_forward_stays_raw(self):
        model = self.uninitialized()
        checkpoint = self.native_checkpoint(model)
        model.initialize_teacher_checkpoint(checkpoint)
        tokens = torch.full((2, 12), model.mask_index, dtype=torch.long)
        tokens[:, :2] = torch.tensor([[1, 2], [2, 3]])
        sigma = torch.ones(2, 1)
        model.eval()
        with torch.no_grad():
            processed_sigma = model._process_sigma(sigma)
            expected_ema = model._process_model_output(
                model.student_ema(tokens, processed_sigma).clone(), tokens, processed_sigma)
            expected_raw = model._process_model_output(
                model.backbone(tokens, processed_sigma).clone(), tokens, processed_sigma)
            with model._didi_evaluation_context():
                actual_ema = model.forward(tokens, sigma)
            actual_training = model.forward(tokens, sigma)
        torch.testing.assert_close(actual_ema, expected_ema, rtol=0, atol=0)
        torch.testing.assert_close(actual_training, expected_raw, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual_ema, actual_training))

    def test_resume_restores_trained_ema_and_provenance_without_reseeding_teacher_ema(self):
        model = self.uninitialized()
        native = self.native_checkpoint(model)
        model.initialize_teacher_checkpoint(native)
        with torch.no_grad():
            for parameter in model.student_ema.parameters():
                parameter.add_(.123)
        model.global_steps = 11
        checkpoint = {'state_dict': copy.deepcopy(model.state_dict())}
        model.on_save_checkpoint(checkpoint)
        metadata = copy.deepcopy(model._didi_student_metadata)
        restored = self.uninitialized()
        # The launcher's fresh native-teacher setup may happen before Lightning
        # restores a training checkpoint; it must not override trained EMA later.
        restored.initialize_teacher_checkpoint(self.native_checkpoint(restored))
        restored.on_load_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint['state_dict'])
        self.assert_state_equal(restored.student_ema.state_dict(), model.student_ema.state_dict())
        self.assertEqual(restored._didi_student_metadata, metadata)
        self.assertEqual(restored.global_steps, 11)
        for parameter, initial_shadow in zip(restored.student_ema.parameters(), native['ema']['shadow_params']):
            self.assertFalse(torch.equal(parameter, initial_shadow))

    def test_raw_vs_native_ema_initialization_does_not_change_training_updates(self):
        trained = []
        for source in ('checkpoint', 'raw'):
            torch.manual_seed(801)
            model = self.uninitialized(source)
            model.initialize_teacher_checkpoint(self.native_checkpoint(model))
            batch = self.helpers.batch()
            loader = torch.utils.data.DataLoader([batch], batch_size=None)
            with tempfile.TemporaryDirectory(prefix='tinygsm-ema-training-') as directory:
                trainer = self.helpers.fixture.trainer(directory, max_steps=2)
                trainer.fit(model, train_dataloaders=loader)
            trained.append({
                'backbone': copy.deepcopy(model.backbone.state_dict()),
                'teacher': copy.deepcopy(model.teacher.state_dict()),
                'discriminator': copy.deepcopy(model.discriminator.state_dict()),
                'student_ema': copy.deepcopy(model.student_ema.state_dict()),
            })
        for name in ('backbone', 'teacher', 'discriminator'):
            self.assert_state_equal(trained[0][name], trained[1][name])
        self.assertTrue(any(not torch.equal(value, trained[1]['student_ema'][key])
                            for key, value in trained[0]['student_ema'].items()))


if __name__ == '__main__':
    unittest.main()
