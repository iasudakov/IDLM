"""Finite-epoch CDLM reports its final step without a Lightning val context."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import sdtt_eval


class FinalLoggingTest(unittest.TestCase):
    def test_final_frontier_bypasses_epoch_logging_and_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = SimpleNamespace(log_metrics=lambda *args, **kwargs: None)
            model = SimpleNamespace(
                global_step=1248, device=torch.device('cpu'),
                trainer=SimpleNamespace(sanity_checking=False, global_rank=0,
                                        world_size=1, logger=logger, loggers=[logger]),
                config=OmegaConf.create({
                    'data': {'train': 'tiny_gsm'},
                    'eval': {'gsm8k_at_validation': True, 'gsm8k_every_n_validations': 3},
                    'checkpointing': {'save_dir': directory}}),
                _gsm8k_examples=[{'input_ids': [0], 'answer': '#### 1'}],
                _gsm8k_validation_count=1)
            curve = [{'threshold': .9, 'accuracy': .5, 'nfe': 10}]
            with patch.object(sdtt_eval, 'sweep', return_value=curve) as sweep:
                with patch.object(sdtt_eval, 'publish') as publish:
                    sdtt_eval.log_validation_gsm8k(model, final=True)
                    sdtt_eval.log_validation_gsm8k(model, final=True)
            sweep.assert_called_once_with(model, model._gsm8k_examples)
            publish.assert_called_once_with(model, curve, initial=False, direct=True)
            self.assertEqual(model._gsm8k_last_logged_step, 1248)


if __name__ == '__main__':
    unittest.main()
