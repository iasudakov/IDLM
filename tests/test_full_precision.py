"""CPU checks for opt-in full-FP32 runtime policy and early failure."""

from contextlib import redirect_stdout
import copy
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import full_precision


def config(**changes):
    result = {'model': {'force_fp32': True}, 'trainer': {'precision': '32-true'},
              'training': {'loss_precision': 'float32'}}
    for section, fields in changes.items():
        result[section].update(fields)
    return result


class FullPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '0'}))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_validation_accepts_fp32_aliases_without_changing_config(self):
        for precision in (32, '32', '32-true'):
            with self.subTest(precision=precision):
                cfg = config(trainer={'precision': precision})
                original = copy.deepcopy(cfg)
                self.assertTrue(full_precision.validate_full_fp32_config(cfg))
                self.assertEqual(cfg, original)

    def test_validation_rejects_conflicting_precision_before_runtime_changes(self):
        for precision in ('bf16-mixed', 'bf16-true', '16-mixed', '64-true', 32.0, None):
            with self.subTest(precision=precision):
                with patch.object(full_precision, 'torch') as runtime:
                    with self.assertRaisesRegex(ValueError, 'trainer.precision=32-true'):
                        full_precision.configure_full_fp32(config(trainer={'precision': precision}))
                    runtime.set_float32_matmul_precision.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'loss_precision=float32'):
            full_precision.validate_full_fp32_config(config(training={'loss_precision': 'bf16'}))

    def test_boolean_flag_is_required(self):
        for flag in ('false', 'true', 0, 1, None):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'must be a boolean'):
                full_precision.validate_full_fp32_config(config(model={'force_fp32': flag}))

    def test_existing_mixed_precision_recipe_is_a_noop(self):
        for cfg in ({}, {'model': {}, 'trainer': {'precision': 'bf16-mixed'}},
                    config(model={'force_fp32': False}, trainer={'precision': 'bf16-mixed'})):
            with self.subTest(cfg=cfg), patch.object(full_precision, 'torch') as runtime:
                self.assertFalse(full_precision.configure_full_fp32(cfg))
                self.assertEqual(runtime.mock_calls, [])

    def test_environment_cannot_force_tf32(self):
        with patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '1'}):
            with self.assertRaisesRegex(ValueError, 'NVIDIA_TF32_OVERRIDE=1'):
                full_precision.configure_full_fp32(config())
            # An unrelated existing mixed-precision run keeps its environment.
            self.assertFalse(full_precision.configure_full_fp32({}))

    def test_legacy_runtime_disables_tf32_and_restores_global_flags(self):
        if hasattr(torch.backends, 'fp32_precision'):
            self.skipTest('The installed PyTorch uses the new precision API.')
        old_precision = torch.get_float32_matmul_precision()
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        cuda_initialized = torch.cuda.is_initialized()
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.allow_tf32 = True
            self.assertTrue(full_precision.configure_full_fp32(config()))
            self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
        finally:
            torch.set_float32_matmul_precision(old_precision)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def test_new_runtime_uses_ieee_api_without_touching_legacy_settings(self):
        matmul = SimpleNamespace(fp32_precision='tf32', allow_tf32='untouched')
        cudnn = SimpleNamespace(fp32_precision='tf32', allow_tf32='untouched',
                                conv=SimpleNamespace(fp32_precision='tf32'),
                                rnn=SimpleNamespace(fp32_precision='tf32'))
        runtime = SimpleNamespace(
            backends=SimpleNamespace(fp32_precision='tf32',
                cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
            set_float32_matmul_precision=Mock())
        with patch.object(full_precision, 'torch', runtime):
            self.assertTrue(full_precision.configure_full_fp32(config()))
        for backend in (runtime.backends, matmul, cudnn, cudnn.conv, cudnn.rnn):
            self.assertEqual(backend.fp32_precision, 'ieee')
        self.assertEqual(matmul.allow_tf32, 'untouched')
        self.assertEqual(cudnn.allow_tf32, 'untouched')
        runtime.set_float32_matmul_precision.assert_not_called()


if __name__ == '__main__':
    unittest.main()
