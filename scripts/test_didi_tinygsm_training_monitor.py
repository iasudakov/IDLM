"""CPU-only tests for exact training counts, resume and source identification."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from didi_repro.tinygsm_training_monitor import TinyGSMTrainingCounters, source_provenance


class TrainingCounterTests(unittest.TestCase):
    def step(self, counters, rows=8, draws=8, student=1, elapsed=2.5):
        counters.observe_step(global_prompt_rows=rows,
                              candidate_trajectories=rows * draws,
                              student_updates=student, discriminator_updates=1,
                              elapsed_seconds=elapsed)

    def test_warmup_short_batch_and_resume_preserve_measured_history(self):
        counters = TinyGSMTrainingCounters()
        self.step(counters, student=0)
        self.step(counters)
        restored = TinyGSMTrainingCounters()
        restored.load_state_dict(json.loads(json.dumps(counters.state_dict())))
        self.step(restored, rows=3, elapsed=.75)
        self.assertEqual(restored.metrics(), {
            'didi/student_updates': 2, 'didi/discriminator_updates': 3,
            'didi/optimizer_updates': 5, 'didi/prompt_rows': 19,
            'didi/candidate_trajectories': 152, 'didi/train_wall_seconds': 5.75})
        self.assertEqual(counters.prompt_rows, 16)

    def test_validation_idle_time_is_never_implicitly_counted(self):
        counters = TinyGSMTrainingCounters()
        self.step(counters, elapsed=.25)
        before = counters.state_dict()
        counters.metrics()
        self.assertEqual(counters.state_dict(), before)
        self.step(counters, elapsed=.5)
        self.assertEqual(counters.train_wall_seconds, .75)

    def test_counts_stay_integer_beyond_float32_precision(self):
        counters = TinyGSMTrainingCounters()
        self.step(counters, rows=2 ** 24 + 1)
        self.assertEqual(counters.metrics()['didi/prompt_rows'], 16777217)
        self.assertIsInstance(counters.metrics()['didi/candidate_trajectories'], int)

    def test_invalid_step_and_restore_do_not_partially_mutate(self):
        counters = TinyGSMTrainingCounters()
        self.step(counters)
        before = counters.state_dict()
        for elapsed in (float('nan'), float('inf'), -1, True):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                self.step(counters, elapsed=elapsed)
            self.assertEqual(counters.state_dict(), before)
        for key, value in (('student_updates', 4), ('prompt_rows', 0),
                           ('version', True), ('candidate_trajectories', -1)):
            invalid = copy.deepcopy(before)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                counters.load_state_dict(invalid)
            self.assertEqual(counters.state_dict(), before)
        for invalid in (None, {}, {'schema': counters.SCHEMA, 'version': 1}):
            with self.assertRaises(ValueError):
                counters.load_state_dict(invalid)
        invalid = TinyGSMTrainingCounters().state_dict()
        invalid['candidate_trajectories'] = 1
        with self.assertRaises(ValueError):
            counters.load_state_dict(invalid)
        self.assertEqual(counters.state_dict(), before)


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'didi_repro/configs/data').mkdir(parents=True)
        (self.root / 'scripts').mkdir()
        (self.root / 'didi_repro/runtime.py').write_text('VALUE = 1\n')
        (self.root / 'didi_repro/configs/data/tiny.yaml').write_text('train: tiny_gsm\n')
        (self.root / 'scripts/train_didi_tinygsm.sh').write_text('python -m didi_repro\n')
        (self.root / 'sandbox_gsm8k.py').write_text('pass\n')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    def test_archive_manifest_is_deterministic_and_excludes_unrelated_files(self):
        first = source_provenance(self.root)
        (self.root / 'credentials.json').write_text('{"secret":"not part of source"}')
        second = source_provenance(self.root)
        self.assertEqual(first, second)
        self.assertFalse(first['git']['available'])
        self.assertNotIn('credentials.json', first['files'])
        self.assertNotIn('not part of source', json.dumps(second))
        self.assertEqual(len(first['files']), 4)
        (self.root / 'didi_repro/new_adapter.py').write_text('VALUE = 2\n')
        self.assertNotEqual(source_provenance(self.root)['sha256'], first['sha256'])

    def test_git_identity_and_dirty_source_are_recorded(self):
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'fixture')
        first = source_provenance(self.root)
        self.assertEqual(first['git'], {'available': True,
                                       'head': self.git('rev-parse', 'HEAD'),
                                       'dirty': False})
        (self.root / 'sandbox_gsm8k.py').write_text('VALUE = 3\n')
        changed = source_provenance(self.root)
        self.assertTrue(changed['git']['dirty'])
        self.assertEqual(changed['git']['head'], first['git']['head'])
        self.assertNotEqual(changed['sha256'], first['sha256'])

    def test_archive_inside_another_checkout_does_not_inherit_parent_sha(self):
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'fixture')
        archive = self.root / 'archive'
        (archive / 'didi_repro').mkdir(parents=True)
        (archive / 'didi_repro/runtime.py').write_text('pass\n')
        self.assertFalse(source_provenance(archive)['git']['available'])


if __name__ == '__main__':
    unittest.main()
