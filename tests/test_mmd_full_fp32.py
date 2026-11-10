"""Actual small DiT plus MMD objective on CPU, including checkpointed backward."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from mmd_finetune import MMDFinetune
from models.dit import DIT
from test_cdlm_training import Tokenizer
from test_mmd_finetune import config


class FullFP32MMDIntegrationTest(unittest.TestCase):
    def test_original_mdlm_warm_start_and_complete_mmd_backward_stay_fp32(self):
        torch.manual_seed(271)
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            OmegaConf.update(cfg, 'model.force_fp32', True, force_add=True)
            cfg.model.dropout = 0.
            cfg.trainer.precision = '32-true'
            cfg.training.loss_precision = 'float32'
            cfg.mmd.group_size = 8
            cfg.mmd.student_activation_checkpointing = True
            cfg.mmd.student_forward_batch_size = 1
            model = MMDFinetune(cfg, Tokenizer())
            self.assertIsInstance(model.backbone, DIT)
            self.assertIsInstance(model.teacher, DIT)

            # Mimic a trained original MDLM rather than DiT's all-zero output
            # and conditioning gates, so gradients reach attention/embeddings.
            with torch.no_grad():
                for name, parameter in model.backbone.named_parameters():
                    if 'adaLN_modulation' in name or name.startswith('output_layer.linear'):
                        parameter.normal_(std=.07)
            source_state = {name: value.detach().clone()
                            for name, value in model.backbone.state_dict().items()}
            source_ema = [parameter.detach().clone() + .001
                          for parameter in model.backbone.parameters()]
            source = Path(directory) / 'original-mdlm.ckpt'
            torch.save({
                'state_dict': {'backbone.' + name: value for name, value in source_state.items()},
                'ema': {'shadow_params': source_ema, 'decay': .9999, 'num_updates': 10000},
                'global_step': 10000,
            }, source)
            model.initialize_from_checkpoint(str(source), str(source), retain_ema=True)
            for module in (model.backbone, model.teacher):
                for name, value in module.state_dict().items():
                    if value.is_floating_point():
                        self.assertEqual(value.dtype, torch.float32, name)
                    torch.testing.assert_close(value, source_state[name], rtol=0, atol=0)
            self.assertEqual(model.ema.num_updates, 10000)
            for actual, expected in zip(model.ema.shadow_params, source_ema):
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            model.train()
            self.assertFalse(model.teacher.training)
            model._trainer = SimpleNamespace(accumulate_grad_batches=4)
            original = torch.tensor([[0, 4, 5, 6, 1, 2], [0, 3, 6, 4, 1, 2]])
            valid = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
            corrupted = original.clone()
            corrupted[:, 2:] = model.mask_index
            batch = {'input_ids': original, 'attention_mask': valid}
            captured = {'logits': [], 'sampled_logp': [], 'features': []}

            def capture_logits(module, inputs, output):
                captured['logits'].append(output.dtype)

            original_sampler = model._sample_student_outputs

            def capture_samples(*args):
                logp, samples = original_sampler(*args)
                captured['sampled_logp'].append(logp)
                self.assertEqual(samples.dtype, torch.long)
                self.assertEqual(samples.shape, (8, 2, 6))
                return logp, samples

            original_features = model.reward.features

            def capture_features(*args):
                features = original_features(*args)
                captured['features'].append(features.dtype)
                self.assertFalse(features.requires_grad)
                self.assertTrue(torch.isfinite(features).all())
                return features

            handle = model.backbone.register_forward_hook(capture_logits)
            try:
                with patch.object(model, '_sample_t', return_value=torch.full((2,), .5)), \
                        patch.object(model, 'q_xt', return_value=corrupted), \
                        patch.object(model, '_sample_student_outputs', side_effect=capture_samples), \
                        patch.object(model.reward, 'features', side_effect=capture_features), \
                        patch.object(model, 'log'):
                    loss = model.training_step(batch, 0)
                    self.assertEqual(loss.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(loss.requires_grad)
                    loss.backward()
            finally:
                handle.remove()

            self.assertTrue(captured['logits'])
            self.assertTrue(all(dtype == torch.float32 for dtype in captured['logits']))
            self.assertEqual(len(captured['sampled_logp']), 1)
            self.assertEqual(captured['sampled_logp'][0].dtype, torch.float32)
            self.assertTrue(captured['sampled_logp'][0].requires_grad)
            self.assertTrue(torch.isfinite(captured['sampled_logp'][0]).all())
            self.assertTrue(captured['features'])
            self.assertTrue(all(dtype == torch.float32 for dtype in captured['features']))
            self.assertEqual(model._mmd_metric_sum.dtype, torch.float32)
            self.assertTrue(torch.isfinite(model._mmd_metric_sum).all())
            for name, parameter in model.backbone.named_parameters():
                self.assertEqual(parameter.dtype, torch.float32, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(parameter.grad.dtype, torch.float32, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            parameters = dict(model.backbone.named_parameters())
            for name in ('vocab_embed.embedding', 'blocks.0.attn_qkv.weight',
                         'output_layer.linear.weight'):
                self.assertGreater(parameters[name].grad.abs().max().item(), 0, name)
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in model.teacher.parameters()))


if __name__ == '__main__':
    unittest.main()
