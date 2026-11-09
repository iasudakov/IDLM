"""Trajectory collection and initialization use the same selected teacher."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from cdlm import CDLM
from model_assets import (configure_model_weights, pretrained_kwargs,
                          resolve_checkpoint)
from test_cdlm_training import CPUDenoiser, Tokenizer, config as make_config


PUBLISHED_MMD = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'


class TeacherSelectionTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('models.dit.DIT', CPUDenoiser))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = make_config(str(self.directory))
        self.select_teacher(use_ema=False, allow_hub=False)
        self.tokenizer = Tokenizer()
        self.model = CDLM(self.config, self.tokenizer)
        self.raw = {name: torch.full_like(value, (i + 1) * .125)
                    for i, (name, value) in enumerate(self.model.backbone.state_dict().items())}
        self.ema = {name: value + .5 for name, value in self.raw.items()}
        self.student = {name: value + 1. for name, value in self.raw.items()}
        self.teacher_path = self.directory / 'teacher.ckpt'

    def select_teacher(self, *, use_ema, allow_hub=False):
        OmegaConf.update(self.config, 'training.teacher_use_ema', use_ema, force_add=True)
        OmegaConf.update(self.config, 'training.teacher_allow_hub', allow_hub, force_add=True)

    def write_checkpoint(self, path, raw, *, ema=None, ema_only=False):
        checkpoint = {'state_dict': {
            'backbone._orig_mod.' + name: value for name, value in raw.items()}}
        if ema is not None:
            checkpoint['ema'] = {
                'shadow_params': [ema[name] for name, _ in self.model.backbone.named_parameters()],
                'num_updates': 777, 'decay': .9}
        if ema_only:
            checkpoint['checkpoint_weights'] = 'ema'
        torch.save(checkpoint, path)

    def assert_backbone(self, model, expected):
        for name, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)

    def assert_head(self, expected):
        for name, value in self.model.teacher_head.state_dict().items():
            self.assertTrue(torch.equal(value, expected['output_layer.linear.' + name]), name)
        self.model.train()
        self.assertFalse(self.model.teacher_head.training)
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in self.model.teacher_head.parameters()))

    def assert_fresh_ema(self):
        self.assertEqual(self.model.ema.num_updates, 0)
        self.assertEqual(self.model.ema.decay, self.config.training.ema)
        parameters = [p for p in self.model._get_parameters() if p.requires_grad]
        self.assertEqual(len(parameters), len(self.model.ema.shadow_params))
        for parameter, shadow in zip(parameters, self.model.ema.shadow_params):
            self.assertTrue(torch.equal(parameter, shadow))

    def collector(self):
        return CDLM(self.config, self.tokenizer, inference_only=True)

    def test_native_mmd_ema_export_initializes_teacher_student_and_frozen_head(self):
        self.select_teacher(use_ema=True)
        self.write_checkpoint(self.teacher_path, self.ema, ema_only=True)
        collector = self.collector()
        collector.load_teacher_weights(str(self.teacher_path))
        self.assert_backbone(collector, self.ema)
        self.assertIsNone(collector.ema)
        self.assertEqual(collector._evaluation_weights, 'ema')
        self.assertFalse(hasattr(collector, 'teacher_head'))
        self.model.initialize_from_checkpoint(str(self.teacher_path))
        self.assert_backbone(self.model, self.ema)
        self.assert_head(self.ema)
        self.assert_fresh_ema()
        tokens = torch.tensor([[0, 4, 7, 7, 1, 2]])
        output = collector(tokens, torch.zeros(1, 1))
        self.assertEqual(tuple(output.shape), (1, 6, 8))
        self.assertTrue(torch.isfinite(output[..., :7]).all())

    def test_full_checkpoint_ema_is_selected_before_copying_teacher_head(self):
        self.select_teacher(use_ema=True)
        self.write_checkpoint(self.teacher_path, self.raw, ema=self.ema)
        collector = self.collector()
        collector.load_teacher_weights(str(self.teacher_path))
        self.assert_backbone(collector, self.ema)
        self.assertEqual(collector._evaluation_weights, 'ema')
        self.model.initialize_from_checkpoint(str(self.teacher_path))
        self.assert_backbone(self.model, self.ema)
        self.assert_head(self.ema)
        self.assert_fresh_ema()

    def test_explicit_raw_selection_overrides_config_and_keeps_live_teacher(self):
        self.select_teacher(use_ema=True)
        self.write_checkpoint(self.teacher_path, self.raw, ema=self.ema)
        collector = self.collector()
        collector.load_teacher_weights(str(self.teacher_path), use_ema=False)
        self.assert_backbone(collector, self.raw)
        self.assertEqual(collector._evaluation_weights, 'live')
        self.model.initialize_from_checkpoint(str(self.teacher_path), teacher_use_ema=False)
        self.assert_backbone(self.model, self.raw)
        self.assert_head(self.raw)

    def test_raw_selection_rejects_ema_only_teacher_with_configuration_hint(self):
        self.write_checkpoint(self.teacher_path, self.ema, ema_only=True)
        for load in (self.collector().load_teacher_weights,
                     self.model.initialize_from_checkpoint):
            with self.subTest(loader=load.__name__):
                with self.assertRaisesRegex(ValueError, 'training.teacher_use_ema=True'):
                    load(str(self.teacher_path))

    def test_selected_teacher_ema_must_exist_and_match_parameter_shapes(self):
        self.select_teacher(use_ema=True)
        self.write_checkpoint(self.teacher_path, self.raw)
        with self.assertRaisesRegex(ValueError, 'Teacher checkpoint has no EMA'):
            self.collector().load_teacher_weights(str(self.teacher_path))
        checkpoint = torch.load(self.teacher_path, weights_only=False)
        checkpoint['ema'] = {'shadow_params': [torch.zeros(1)]}
        torch.save(checkpoint, self.teacher_path)
        with self.assertRaisesRegex(ValueError, 'EMA shapes'):
            self.model.initialize_from_checkpoint(str(self.teacher_path))

    def test_student_override_retains_selected_teacher_head_and_resets_ema(self):
        self.select_teacher(use_ema=True)
        self.write_checkpoint(self.teacher_path, self.raw, ema=self.ema)
        student_path = self.directory / 'student.ckpt'
        student_ema = {name: value + .25 for name, value in self.student.items()}
        self.write_checkpoint(student_path, self.student, ema=student_ema)
        for use_ema, expected in ((False, self.student), (True, student_ema)):
            with self.subTest(student_use_ema=use_ema):
                self.model.ema.num_updates = 99
                self.model.initialize_from_checkpoint(
                    str(self.teacher_path), student_source=str(student_path),
                    student_use_ema=use_ema)
                self.assert_backbone(self.model, expected)
                self.assert_head(self.ema)
                self.assert_fresh_ema()

    def test_default_mdlm_initialization_remains_raw_with_or_without_saved_ema(self):
        for saved_ema in (None, self.ema):
            with self.subTest(saved_ema=saved_ema is not None):
                self.write_checkpoint(self.teacher_path, self.raw, ema=saved_ema)
                self.model.initialize_from_checkpoint(str(self.teacher_path))
                self.assert_backbone(self.model, self.raw)
                self.assert_head(self.raw)
                self.assert_fresh_ema()
                collector = self.collector()
                collector.load_teacher_weights(str(self.teacher_path))
                self.assert_backbone(collector, self.raw)
                self.assertEqual(collector._evaluation_weights, 'live')

    def test_teacher_backbone_loading_is_strict(self):
        self.write_checkpoint(self.teacher_path, {**self.raw, 'unexpected': torch.zeros(1)})
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            self.collector().load_teacher_weights(str(self.teacher_path))

    def test_explicit_hub_teacher_uses_local_tokenizer_bundle_and_preserves_eval_guard(self):
        self.select_teacher(use_ema=True, allow_hub=True)
        bundle = self.directory / 'offline bundle'
        tokenizer = bundle / 'gpt2'
        tokenizer.mkdir(parents=True)
        (tokenizer / 'tokenizer.json').write_text('{}')
        self.config.model_weights_dir = str(bundle)
        self.config.training.finetune_path = PUBLISHED_MMD
        configure_model_weights(self.config)
        self.assertEqual(self.config.training.finetune_path, PUBLISHED_MMD)
        self.assertEqual(pretrained_kwargs('gpt2', str(bundle), kind='tokenizer'), {
            'pretrained_model_name_or_path': str(tokenizer.resolve()),
            'local_files_only': True})
        self.write_checkpoint(self.teacher_path, self.ema, ema_only=True)
        collector = self.collector()
        with patch('sdtt_checkpoint.hf_hub_download',
                   return_value=str(self.teacher_path)) as download:
            collector.load_teacher_weights(PUBLISHED_MMD)
            self.model.initialize_from_checkpoint(PUBLISHED_MMD)
        self.assertEqual(download.call_count, 2)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs, {
                'repo_id': 'yresearch/MDLM-MMD-TinyGSM', 'filename': 'model.ckpt'})
        self.assert_backbone(collector, self.ema)
        self.assert_head(self.ema)
        with patch('sdtt_checkpoint.hf_hub_download') as download:
            with self.assertRaisesRegex(ValueError, 'local MDLM checkpoint'):
                collector.load_evaluation_weights(PUBLISHED_MMD)
        download.assert_not_called()

    def test_bundle_teacher_hub_requires_opt_in_and_other_assets_remain_local(self):
        bundle = self.directory / 'bundle'
        bundle.mkdir()
        self.config.model_weights_dir = str(bundle)
        with patch('sdtt_checkpoint.hf_hub_download') as download:
            for load in (self.collector().load_teacher_weights,
                         self.model.initialize_from_checkpoint):
                with self.subTest(loader=load.__name__):
                    with self.assertRaisesRegex(ValueError, 'local MDLM checkpoint'):
                        load(PUBLISHED_MMD)
        download.assert_not_called()
        self.select_teacher(use_ema=True, allow_hub=True)
        with self.assertRaisesRegex(FileNotFoundError, 'No model download'):
            pretrained_kwargs('missing-tokenizer', str(bundle), kind='tokenizer')

    def test_only_opted_in_teacher_hub_path_skips_local_normalization(self):
        bundle = self.directory / 'bundle'
        bundle.mkdir()
        self.config.model_weights_dir = str(bundle)
        self.config.training.finetune_path = PUBLISHED_MMD
        for allow_hub in (False, True):
            with self.subTest(teacher_allow_hub=allow_hub):
                self.select_teacher(use_ema=True, allow_hub=allow_hub)
                with patch('model_assets.resolve_checkpoint', wraps=resolve_checkpoint) as resolve:
                    configure_model_weights(self.config)
                normalized = [call.args[0] for call in resolve.call_args_list]
                self.assertEqual(PUBLISHED_MMD in normalized, not allow_hub)


if __name__ == '__main__':
    unittest.main()
