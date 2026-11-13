"""Verify the actual confidence launcher argv without GPUs or snapshot access."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class ConfidenceLauncherTest(unittest.TestCase):
    def test_mmd_8000_ema_and_custom_thresholds(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            inputs = Path(temp)
            (inputs / 'tinygsm_mmd_offline_bundle').mkdir()
            data = inputs / 'nirvana_checkpoints/data/gsm8k_test.json'
            data.parent.mkdir(parents=True)
            data.write_text(json.dumps([{}] * 1319))
            result = subprocess.run([
                sys.executable, str(repo / 'scripts/paper_recompute/launch.py'),
                'confidence-mmd', '--source-run', 'test-run', '--step', '8000',
                '--devices', '2', '--dry-run', '--extra',
                '--thresholds', '0.98', '0.99', '0.995',
            ], env=dict(os.environ, INPUT_PATH=str(inputs)),
                capture_output=True, text=True, check=True)
            cmd = shlex.split(result.stdout.strip())
            self.assertEqual(cmd[cmd.index('--suite') + 1], 'confidence')
            self.assertEqual(cmd[cmd.index('--weights') + 1], 'ema')
            self.assertEqual(cmd[cmd.index('--step') + 1], '8000')
            self.assertEqual(cmd[cmd.index('--devices') + 1], '2')
            self.assertEqual(cmd[cmd.index('--thresholds') + 1:], ['0.98', '0.99', '0.995'])
            self.assertNotIn('--samples', cmd)


if __name__ == '__main__':
    unittest.main()
