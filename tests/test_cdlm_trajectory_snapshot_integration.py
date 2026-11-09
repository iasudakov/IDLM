"""Actual tiny teacher caches survive snapshot round trips and interrupted prep."""

from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cdlm import CDLM
from cdlm_cache import cache_key
import cdlm_trajectories as trajectories
from test_cdlm_training import CPUDenoiser, Original, Tokenizer, config as make_config


class TrajectorySnapshotIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('models.dit.DIT', CPUDenoiser))
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = make_config(str(self.base / 'run'))
        Path(self.cfg.training.finetune_path).parent.mkdir(parents=True)
        self.cfg.algo.trajectory_cache_per_model = True
        self.cfg.checkpointing.snapshot = True
        self.cfg.checkpointing.snapshot_trajectories = True
        self.cfg.checkpointing.trajectory_snapshot_interval_seconds = .01
        self.store = self.base / 'snapshot'
        self.store.mkdir()
        self.dumps = []
        sdk = ModuleType('nirvana_dl')
        sdk.snapshot = SimpleNamespace(
            get_snapshot_path=lambda: str(self.store),
            dump_snapshot=lambda path: self.dumps.append(path))
        self.enterContext(patch.dict(sys.modules, {'nirvana_dl': sdk}))
        self.enterContext(patch.dict('os.environ', {'RANK': '0', 'LOCAL_RANK': '0'}))
        self.tokenizer = Tokenizer()
        self.teacher = CDLM(self.cfg, self.tokenizer, inference_only=True)
        self.raw_state = {'backbone.' + key: value.clone()
                          for key, value in self.teacher.backbone.state_dict().items()}
        torch.save({'state_dict': self.raw_state}, self.cfg.training.finetune_path)
        self.teacher.load_teacher_weights(self.cfg.training.finetune_path)

    def prepare(self):
        return Path(trajectories.prepare_trajectories(
            self.teacher, self.cfg, self.tokenizer, Original()))

    def snapshot_path(self, directory):
        identity = trajectories.read_cache_manifest(directory)['identity']
        return self.store / 'cdlm_trajectories' / cache_key(identity)

    def test_models_are_stored_separately_and_complete_mmd_cache_restores_without_teacher_forward(self):
        mdlm = self.prepare()
        mdlm_snapshot = self.snapshot_path(mdlm)
        original_receipt = (mdlm_snapshot / 'part-00000000.json').read_bytes()

        mmd_checkpoint = self.base / 'published-mmd.ckpt'
        torch.save({'state_dict': {key: value + .25 for key, value in self.raw_state.items()},
                    'checkpoint_weights': 'ema'}, mmd_checkpoint)
        self.cfg.training.finetune_path = str(mmd_checkpoint)
        self.cfg.training.teacher_use_ema = True
        self.teacher = CDLM(self.cfg, self.tokenizer, inference_only=True)
        self.teacher.load_teacher_weights(str(mmd_checkpoint))
        mmd = self.prepare()
        mmd_snapshot = self.snapshot_path(mmd)
        self.assertNotEqual(mdlm_snapshot, mmd_snapshot)
        self.assertEqual(trajectories.read_cache_manifest(mmd_snapshot)['status'], 'complete')
        self.assertEqual((mdlm_snapshot / 'part-00000000.json').read_bytes(), original_receipt)

        shutil.rmtree(self.cfg.algo.trajectory_cache_dir)
        with patch.object(self.teacher, 'forward', side_effect=AssertionError('cache must restore')):
            restored = self.prepare()
        self.assertEqual(restored, mmd)
        self.assertEqual(len(trajectories.CDLMTrajectoryDataset(restored)), 8)
        self.assertEqual((mdlm_snapshot / 'part-00000000.json').read_bytes(), original_receipt)
        self.assertTrue(self.dumps)

    def test_interrupted_collection_restores_confirmed_shards_and_generates_only_missing_work(self):
        real_collect = trajectories.collect_trajectory_batch
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('interrupted collection')
            return real_collect(*args, **kwargs)

        with patch.object(trajectories, 'collect_trajectory_batch', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupted collection'):
                self.prepare()
        from cdlm_cache import cache_directory
        identity = trajectories.build_cache_identity(self.cfg, self.tokenizer, Original())
        directory = cache_directory(self.cfg, identity)
        snapshot = self.snapshot_path(directory)
        self.assertEqual(trajectories.read_cache_manifest(snapshot)['status'], 'building')
        first_hash = (snapshot / 'part-00000000.json').read_bytes()
        self.assertFalse(list(snapshot.glob('*.lock')))
        self.assertFalse(list(snapshot.glob('*.tmp')))
        shutil.rmtree(self.cfg.algo.trajectory_cache_dir)
        with patch.object(trajectories, 'collect_trajectory_batch', wraps=real_collect) as collect:
            restored = self.prepare()
        self.assertEqual(collect.call_count, 3)
        self.assertEqual((restored / 'part-00000000.json').read_bytes(), first_hash)
        self.assertEqual(trajectories.read_cache_manifest(snapshot)['status'], 'complete')
        self.assertEqual(len(trajectories.CDLMTrajectoryDataset(restored)), 8)


if __name__ == '__main__':
    unittest.main()
