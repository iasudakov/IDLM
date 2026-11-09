"""Final reports are published to snapshots after the full checkpoint exists."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from cdlm import CDLM


class FinalSnapshotTest(unittest.TestCase):
    def run_final(self, *, evaluation_failure=False):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def save_checkpoint(path):
                Path(path).write_bytes(b'full training checkpoint')
                events.append('checkpoint')

            def evaluate(*args, **kwargs):
                (output / 'pareto_history.json').write_text('final curve')
                events.append('evaluation')
                if evaluation_failure:
                    raise RuntimeError('evaluation failed')

            def snapshot(path, **kwargs):
                self.assertEqual(path, directory)
                self.assertEqual(kwargs['namespace'], 'cdlm_tinygsm')
                self.assertEqual((output / 'checkpoints/last.ckpt').read_bytes(),
                                 b'full training checkpoint')
                self.assertEqual((output / 'pareto_history.json').read_text(), 'final curve')
                events.append('snapshot')

            model = SimpleNamespace(
                config=OmegaConf.create({'checkpointing': {'save_dir': directory,
                    'snapshot': True, 'snapshot_namespace': 'cdlm_tinygsm'}}),
                trainer=SimpleNamespace(save_checkpoint=save_checkpoint), global_step=1248,
                _eval_mode=Mock(side_effect=lambda: events.append('eval mode')),
                _train_mode=Mock(side_effect=lambda: events.append('train mode')))
            with patch('sdtt_eval.log_validation_gsm8k', side_effect=evaluate), \
                 patch('nirvana_snapshot.save', side_effect=snapshot) as publish:
                if evaluation_failure:
                    with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                        CDLM.on_train_end(model)
                else:
                    CDLM.on_train_end(model)
            publish.assert_called_once()
        return events

    def test_final_snapshot_contains_checkpoint_and_latest_evaluation(self):
        self.assertEqual(self.run_final(),
                         ['checkpoint', 'eval mode', 'evaluation', 'train mode', 'snapshot'])

    def test_final_snapshot_is_attempted_even_when_evaluation_fails(self):
        self.assertEqual(self.run_final(evaluation_failure=True)[-2:], ['train mode', 'snapshot'])


if __name__ == '__main__':
    unittest.main()
