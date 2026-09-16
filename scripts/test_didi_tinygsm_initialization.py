"""Native TinyGSM EMA initialization regression tests; CPU-only and offline."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L
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

    def uninitialized(self, source=None, *, weights=None):
        config = self.helpers.config()
        if source is not None or weights is not None:
            with open_dict(config):
                config.didi_init = {}
                if source is not None:
                    config.didi_init.tinygsm_student_ema = source
                if weights is not None:
                    config.didi_init.tinygsm_weights = weights
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
        old_ema = copy.deepcopy(model.ema.state_dict()) if model.ema is not None else None
        old_teacher = copy.deepcopy(model._didi_teacher_metadata)
        old_student = copy.deepcopy(model._didi_student_metadata)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            model.initialize_teacher_checkpoint(checkpoint)
        self.assert_state_equal(model.state_dict(), before)
        self.assertEqual(model._didi_teacher_metadata, old_teacher)
        self.assertEqual(model._didi_student_metadata, old_student)
        if old_ema is not None:
            torch.testing.assert_close(model.ema.state_dict(), old_ema, rtol=0, atol=0)

    def test_opt_in_initializes_all_backbones_from_ema_and_preserves_head_and_buffers(self):
        model = self.uninitialized(weights='ema')
        for module in (model.backbone, model.teacher, model.student_ema, model.discriminator):
            module.register_buffer('rotary_test_buffer', torch.tensor([.1, .2, .3]))
        checkpoint = self.native_checkpoint(model)
        untouched = copy.deepcopy(checkpoint)
        head = copy.deepcopy(model.discriminator.output_layer.state_dict())
        # The released discriminator head has spectral-normalization state;
        # copying teacher trunk weights must leave those buffers intact too.
        self.assertTrue(any(key.endswith('weight_u') for key in head))
        old_legacy_ema = copy.deepcopy(model.ema.state_dict())
        model.initialize_teacher_checkpoint(checkpoint)
        expected = {key.removeprefix('backbone.'): value
                    for key, value in checkpoint['state_dict'].items()}
        for (name, _), shadow in zip(model.backbone.named_parameters(),
                                    checkpoint['ema']['shadow_params']):
            expected[name] = shadow
        for module in (model.backbone, model.teacher, model.student_ema):
            self.assert_state_equal(module.state_dict(), expected)
        for key, value in model.discriminator.state_dict().items():
            if not key.startswith('output_layer.'):
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assert_state_equal(model.discriminator.output_layer.state_dict(), head)
        torch.testing.assert_close(checkpoint, untouched, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in model.teacher.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.student_ema.parameters()))
        self.assertEqual(model.ema_beta, .999)
        self.assertEqual(model.ema.decay, old_legacy_ema['decay'])
        self.assertEqual(model.ema.num_updates, old_legacy_ema['num_updates'])
        for shadow, parameter in zip(model.ema.shadow_params, model.backbone.parameters()):
            torch.testing.assert_close(shadow, parameter, rtol=0, atol=0)
            self.assertNotEqual(shadow.data_ptr(), parameter.data_ptr())
        self.assertEqual(model._didi_student_metadata['training_weights'], 'ema')
        self.assertIn('ema.shadow_params', model._didi_teacher_metadata['weights'])

    def test_opt_in_has_independent_parameter_storage_and_does_not_mutate_source(self):
        model = self.uninitialized(weights='ema')
        checkpoint = self.native_checkpoint(model)
        untouched = copy.deepcopy(checkpoint)
        model.initialize_teacher_checkpoint(checkpoint)
        copies = [dict(module.named_parameters()) for module in
                  (model.backbone, model.teacher, model.student_ema, model.discriminator)]
        for (name, parameter), shadow in zip(model.backbone.named_parameters(),
                                            checkpoint['ema']['shadow_params']):
            parameters = [state[name] for state in copies
                          if name in state and not (state is copies[-1]
                                                    and name.startswith('output_layer.'))]
            addresses = [value.data_ptr() for value in parameters] + [shadow.data_ptr()]
            self.assertEqual(len(set(addresses)), len(addresses), name)
            self.assertNotEqual(parameter.data_ptr(), checkpoint['state_dict'][f'backbone.{name}'].data_ptr())
        teacher = copy.deepcopy(model.teacher.state_dict())
        ema = copy.deepcopy(model.student_ema.state_dict())
        disc = copy.deepcopy(model.discriminator.state_dict())
        with torch.no_grad():
            next(model.backbone.parameters()).add_(1)
        self.assert_state_equal(model.teacher.state_dict(), teacher)
        self.assert_state_equal(model.student_ema.state_dict(), ema)
        self.assert_state_equal(model.discriminator.state_dict(), disc)
        torch.testing.assert_close(checkpoint, untouched, rtol=0, atol=0)

    def test_opt_in_rejects_missing_or_malformed_ema_without_altering_any_copy(self):
        for corruption in ('missing', 'count', 'shape', 'nan'):
            with self.subTest(corruption=corruption):
                model = self.uninitialized(weights='ema')
                checkpoint = self.native_checkpoint(model)
                if corruption == 'missing':
                    del checkpoint['ema']
                elif corruption == 'count':
                    checkpoint['ema']['shadow_params'].pop()
                elif corruption == 'shape':
                    checkpoint['ema']['shadow_params'][0] = torch.zeros(1)
                else:
                    checkpoint['ema']['shadow_params'][0].flatten()[0] = float('nan')
                self.assert_rejected_without_mutation(model, checkpoint)

    def test_opt_in_rejects_incompatible_destination_before_loading_any_copy(self):
        for destination in ('teacher', 'student_ema', 'discriminator'):
            with self.subTest(destination=destination):
                model = self.uninitialized(weights='ema')
                getattr(model, destination).register_buffer('unexpected_buffer', torch.ones(1))
                self.assert_rejected_without_mutation(model, self.native_checkpoint(model))

    def test_opt_in_cannot_mix_ema_training_weights_with_raw_evaluation_ema(self):
        with self.assertRaisesRegex(ValueError, 'tinygsm_student_ema|checkpoint'):
            model = self.uninitialized('raw', weights='ema')
            model.initialize_teacher_checkpoint(self.native_checkpoint(model))

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

    def test_ema_initialized_training_updates_student_and_discriminator_but_freezes_teacher(self):
        torch.manual_seed(802)
        model = self.uninitialized(weights='ema')
        model.initialize_teacher_checkpoint(self.native_checkpoint(model))
        before = {name: copy.deepcopy(getattr(model, name).state_dict())
                  for name in ('backbone', 'teacher', 'discriminator', 'student_ema')}
        loader = torch.utils.data.DataLoader([self.helpers.batch()], batch_size=None)
        with tempfile.TemporaryDirectory(prefix='tinygsm-all-ema-training-') as directory:
            trainer = self.helpers.fixture.trainer(directory, max_steps=2)
            trainer.fit(model, train_dataloaders=loader)
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(model.global_steps, 1)
        self.assert_state_equal(model.teacher.state_dict(), before['teacher'])
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher.parameters()))
        for name in ('backbone', 'discriminator', 'student_ema'):
            self.assertTrue(any(not torch.equal(value, before[name][key])
                                for key, value in getattr(model, name).state_dict().items()), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.student_ema.parameters()))
        for value in trainer.logged_metrics.values():
            self.assertTrue(bool(torch.isfinite(torch.as_tensor(value)).all()))

    def test_lightning_resume_restores_all_trained_copies_and_optimizers_despite_new_init_option(self):
        for initial_weights, launch_weights in (('ema', 'raw'), ('raw', 'ema')):
            with self.subTest(initial=initial_weights, launch=launch_weights):
                torch.manual_seed(803)
                model = self.uninitialized(weights=initial_weights)
                model.initialize_teacher_checkpoint(self.native_checkpoint(model))
                loader = torch.utils.data.DataLoader([self.helpers.batch()], batch_size=None)
                with tempfile.TemporaryDirectory(prefix='tinygsm-all-ema-resume-') as directory:
                    trainer = self.helpers.fixture.trainer(directory, max_steps=2)
                    trainer.fit(model, train_dataloaders=loader)
                    path = str(Path(directory) / 'resume.ckpt')
                    trainer.save_checkpoint(path)
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    saved_state = copy.deepcopy(checkpoint['state_dict'])
                    provenance = copy.deepcopy(model._didi_student_metadata)
                    teacher_metadata = copy.deepcopy(model._didi_teacher_metadata)
                    restored = self.uninitialized(weights=launch_weights)
                    # The launcher can initialize fresh copies before Lightning
                    # restores the full training checkpoint. Neither source mode
                    # may overwrite the checkpoint's learned states afterward.
                    restored.initialize_teacher_checkpoint(self.native_checkpoint(restored))
                    self.assertNotEqual(restored._didi_student_metadata['training_weights'],
                                        provenance['training_weights'])
                    test = self
                    observed = []

                    class CheckRestoredState(L.Callback):
                        def on_train_start(self, trainer, pl_module):
                            test.assert_state_equal(pl_module.state_dict(), saved_state)
                            test.assertEqual(pl_module._didi_student_metadata, provenance)
                            test.assertEqual(pl_module._didi_teacher_metadata, teacher_metadata)
                            test.assertEqual(pl_module.global_steps, 1)
                            test.assertEqual(trainer.global_step, 2)
                            torch.testing.assert_close(pl_module.ema.state_dict(),
                                                       checkpoint['ema'], rtol=0, atol=0)
                            torch.testing.assert_close(
                                [optimizer.state_dict() for optimizer in trainer.optimizers],
                                checkpoint['optimizer_states'], rtol=0, atol=0)
                            observed.append(True)

                    continuation = self.helpers.fixture.trainer(directory, max_steps=4)
                    continuation.callbacks.append(CheckRestoredState())
                    continuation.fit(restored, train_dataloaders=loader, ckpt_path=path)
                    self.assertEqual(observed, [True])
                    self.assertEqual(continuation.global_step, 4)
                    self.assertEqual(restored.global_steps, 2)
                    self.assert_state_equal(restored.teacher.state_dict(), model.teacher.state_dict())
                    self.assertEqual(restored._didi_student_metadata, provenance)


if __name__ == '__main__':
    unittest.main()
