"""Check the real Hydra training entry point without loading weights or using GPUs.

Run: python scripts/paper_recompute/test_entrypoint.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

from launch import training_overrides


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class EntrypointTests(unittest.TestCase):
    def test_all_training_configs_from_external_working_directory(self):
        methods = ('baseline', 'attraction', 'regression', 'linear',
                   'sequence_rbf', 'embeddings', 'ar')
        with tempfile.TemporaryDirectory(prefix='paper-hydra-') as tmp:
            assets = {key: str(Path(tmp) / key)
                      for key in ('mdlm', 'ar', 'bundle', 'data', 'cache')}
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                       HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
            env['PYTHONPATH'] = os.pathsep.join([
                str(REPO), str(HERE), env.get('PYTHONPATH', '')])
            for method in methods:
                with self.subTest(method=method):
                    result = subprocess.run([
                        sys.executable, str(HERE / 'paper_train.py'),
                        *training_overrides(method, 2, assets, str(Path(tmp) / 'out')),
                        '--cfg', 'job', '--resolve',
                    ], cwd=tmp, env=env, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cfg = OmegaConf.create(result.stdout)
                    self.assertEqual(cfg.paper_ablation, method)
                    self.assertEqual(cfg.trainer.devices, 2)
                    self.assertEqual(cfg.trainer.max_steps, 8000)
                    self.assertEqual(cfg.adversarial_distill.mmd_group_size, 4)
                    self.assertEqual(cfg.adversarial_distill.mmd_kernel_batch_size, 2)
                    self.assertEqual(cfg.adversarial_distill.accum_batches, 8)
                    self.assertEqual(cfg.data.separator, r'\n')
                    self.assertTrue(cfg.eval.gsm8k_pareto_fp32)
                    self.assertFalse((Path(tmp) / 'out').exists())
                    print(f'{method}: real Hydra entry point OK', flush=True)


if __name__ == '__main__':
    unittest.main()
