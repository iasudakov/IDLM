"""Checkpoint inventory preflight checks; no model loading or GPU execution."""

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cdlm_checkpoint_eval as evaluator


MDLM_CDLM = '5e0ee1da-8780-42d3-b120-9590e9dc8351'
CDLM_MMD = 'dbb3b25d-4937-4f27-966f-898e696a1f48'
MMD_CDLM = '8cb53b89-96e7-4fe4-ba49-91ed2eb9fb20'
EXPECTED = {
    MDLM_CDLM: ('cdlm_tinygsm_ancestral_sweep',
                [('12-1000.ckpt', 1000), ('last.ckpt', 1248)]),
    CDLM_MMD: ('mmd_from_cdlm_tinygsm',
               [(f'step={step:08d}.ckpt', step) for step in range(500, 7001, 500)]),
    MMD_CDLM: ('cdlm_tinygsm',
               [('12-1000.ckpt', 1000), ('last.ckpt', 1248)]),
}


class CheckpointEvaluationPlanTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(
            tempfile.TemporaryDirectory(prefix='checkpoint sweep, v=2 '))).resolve()
        self.output = self.base / 'new evaluation output, name=three runs'

    def snapshot(self, source, root=None):
        root = root or self.base / ('input snapshot ' + source[:8])
        namespace, files = EXPECTED[source]
        directory = root / namespace / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        for filename, _ in files:
            # Planning validates availability. The worker validates serialized
            # weights; a CPU plan must not deserialize multi-gigabyte models.
            (directory / filename).write_bytes(b'nonempty-checkpoint-placeholder')
        return root

    def source(self, uid, root):
        return {'source_run': uid, 'snapshot_root': str(root)}

    def test_catalog_is_exactly_the_three_selected_runs(self):
        self.assertEqual(set(evaluator.SOURCE_RUNS), set(EXPECTED))
        for uid, (namespace, files) in EXPECTED.items():
            with self.subTest(uid=uid):
                self.assertEqual(evaluator.SOURCE_RUNS[uid]['namespace'], namespace)
                self.assertEqual([tuple(value) for value in
                                  evaluator.SOURCE_RUNS[uid]['checkpoints']], files)

    def test_plan_covers_all_18_checkpoints_with_distinct_outputs(self):
        sources = [self.source(uid, self.snapshot(uid)) for uid in EXPECTED]
        plan = evaluator.build_plan(sources, str(self.output))
        self.assertEqual(plan['schema_version'], 1)
        self.assertEqual(Path(plan['output_dir']), self.output)
        self.assertEqual(plan['snapshot_namespace'], 'cdlm_checkpoint_eval')
        self.assertEqual(len(plan['entries']), 18)
        self.assertEqual(len({row['output_dir'] for row in plan['entries']}), 18)
        for uid, (namespace, expected) in EXPECTED.items():
            rows = [row for row in plan['entries'] if row['source_run'] == uid]
            self.assertEqual([(Path(row['checkpoint']).name, row['step']) for row in rows], expected)
            for row in rows:
                self.assertTrue(Path(row['checkpoint']).is_absolute())
                self.assertTrue(Path(row['checkpoint']).is_file())
                self.assertIn(namespace, Path(row['checkpoint']).parts)
                self.assertTrue(row['family'])
                self.assertIn(self.output, Path(row['output_dir']).parents)
        # This operation only prepares a plan; it must not mutate source state
        # or create checkpoint/output placeholders.
        self.assertFalse(self.output.exists())

    def test_short_run_ids_work_and_unknown_or_ambiguous_ids_fail(self):
        for uid in EXPECTED:
            self.assertEqual(evaluator.resolve_source_run(uid[:8]), uid)
            self.assertEqual(evaluator.resolve_source_run(uid), uid)
        for bad in ('', 'unknown', '7e294cec', 'f7a18dce'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                evaluator.resolve_source_run(bad)
        alias = '5e0ee1da-0000-0000-0000-000000000000'
        with patch.dict(evaluator.SOURCE_RUNS, {alias: evaluator.SOURCE_RUNS[MDLM_CDLM]}):
            with self.assertRaises(ValueError):
                evaluator.resolve_source_run('5e0ee1da')

    def test_inherited_cdlm_checkpoint_is_not_selected_for_mmd(self):
        root = self.snapshot(CDLM_MMD)
        self.snapshot(MDLM_CDLM, root=root)
        other = root / 'mmd_from_cdlm_tinygsm' / 'checkpoints' / 'last.ckpt'
        other.write_bytes(b'inherited-or-unverified-last')
        plan = evaluator.build_plan([self.source(CDLM_MMD[:8], root)], str(self.output))
        self.assertEqual(len(plan['entries']), 14)
        self.assertEqual([row['step'] for row in plan['entries']], list(range(500, 7001, 500)))
        self.assertTrue(all(Path(row['checkpoint']).name.startswith('step=')
                            for row in plan['entries']))

    def test_distinct_namespaces_can_be_selected_from_one_supplied_state(self):
        root = self.snapshot(CDLM_MMD)
        self.snapshot(MDLM_CDLM, root=root)
        plan = evaluator.build_plan([self.source(MDLM_CDLM, root),
                                     self.source(CDLM_MMD, root)], str(self.output))
        self.assertEqual(len(plan['entries']), 16)

    def test_missing_root_and_empty_root_never_fall_back_to_current_directory(self):
        for root in ('', str(self.base / 'absent')):
            with self.subTest(root=root), self.assertRaises((ValueError, FileNotFoundError)):
                evaluator.build_plan([self.source(MDLM_CDLM, root)], str(self.output))

    def test_missing_or_empty_checkpoint_fails_the_entire_plan(self):
        roots = {uid: self.snapshot(uid) for uid in EXPECTED}
        sources = [self.source(uid, root) for uid, root in roots.items()]
        path = roots[CDLM_MMD] / 'mmd_from_cdlm_tinygsm/checkpoints/step=00007000.ckpt'
        path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            evaluator.build_plan(sources, str(self.output), missing_checkpoints='error')
        path.touch()
        with self.assertRaises((ValueError, FileNotFoundError)):
            evaluator.build_plan(sources, str(self.output), missing_checkpoints='error')
        self.assertFalse(self.output.exists())

    def test_partial_snapshot_skips_missing_and_empty_without_stopping_at_first_gap(self):
        root = self.snapshot(CDLM_MMD)
        directory = root / 'mmd_from_cdlm_tinygsm/checkpoints'
        (directory / 'step=00006000.ckpt').unlink()
        (directory / 'step=00007000.ckpt').write_bytes(b'')
        plan = evaluator.build_plan([self.source(CDLM_MMD, root)], str(self.output))
        self.assertEqual(plan['missing_checkpoints'], 'skip')
        self.assertEqual([row['step'] for row in plan['entries']],
                         list(range(500, 6000, 500)) + [6500])
        skipped = plan['skipped_checkpoints']
        self.assertEqual([(row['step'], row['reason']) for row in skipped],
                         [(6000, 'missing'), (7000, 'empty')])
        self.assertTrue(all(row['source_run'] == CDLM_MMD for row in skipped))
        self.assertTrue(all(row['family'] == 'CDLM to MMD' for row in skipped))
        self.assertEqual([Path(row['checkpoint']).name for row in skipped],
                         ['step=00006000.ckpt', 'step=00007000.ckpt'])
        self.assertFalse(self.output.exists())
        self.assertTrue((directory / 'step=00006500.ckpt').is_file())
        self.assertEqual((directory / 'step=00007000.ckpt').stat().st_size, 0)

    def test_each_selected_source_needs_usable_weights_even_if_others_are_present(self):
        first = self.snapshot(MDLM_CDLM)
        second = self.snapshot(CDLM_MMD)
        directory = second / 'mmd_from_cdlm_tinygsm/checkpoints'
        sources = [self.source(MDLM_CDLM, first), self.source(CDLM_MMD, second)]
        for path in directory.glob('*.ckpt'):
            path.unlink()
        # An inherited or unverified last checkpoint is not a substitute for
        # the catalogued MMD exports.
        (directory / 'last.ckpt').write_bytes(b'inherited-last')
        with self.assertRaises(FileNotFoundError):
            evaluator.build_plan(sources, str(self.output))
        for filename, _ in EXPECTED[CDLM_MMD][1]:
            (directory / filename).touch()
        with self.assertRaises(FileNotFoundError):
            evaluator.build_plan(sources, str(self.output))
        self.assertFalse(self.output.exists())

    def test_nonregular_checkpoint_is_not_silently_skipped(self):
        root = self.snapshot(MDLM_CDLM)
        path = root / 'cdlm_tinygsm_ancestral_sweep/checkpoints/last.ckpt'
        path.unlink()
        path.mkdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            evaluator.build_plan([self.source(MDLM_CDLM, root)], str(self.output))

    def test_wrong_snapshot_does_not_fall_back_to_another_namespace(self):
        root = self.snapshot(MDLM_CDLM)
        with self.assertRaises((ValueError, FileNotFoundError)):
            evaluator.build_plan([self.source(MMD_CDLM, root)], str(self.output))

    def test_duplicate_sources_cannot_overwrite_evaluation_results(self):
        root = self.snapshot(MDLM_CDLM)
        with self.assertRaises(ValueError):
            evaluator.build_plan([self.source(MDLM_CDLM, root),
                                  self.source(MDLM_CDLM[:8], root)], str(self.output))

    def test_output_overlap_with_input_state_is_rejected(self):
        root = self.snapshot(MDLM_CDLM)
        for output in (root, root / 'evaluation', root.parent):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                evaluator.build_plan([self.source(MDLM_CDLM, root)], str(output))

    def test_checkpoint_symlink_cannot_escape_snapshot_root(self):
        root = self.snapshot(MDLM_CDLM)
        outside = self.base / 'outside.ckpt'
        outside.write_bytes(b'other-run')
        checkpoint = root / 'cdlm_tinygsm_ancestral_sweep/checkpoints/last.ckpt'
        checkpoint.unlink()
        checkpoint.symlink_to(outside)
        with self.assertRaises(ValueError):
            evaluator.build_plan([self.source(MDLM_CDLM, root)], str(self.output))

    def test_namespace_symlink_cannot_escape_snapshot_root(self):
        external_root = self.snapshot(MDLM_CDLM)
        input_root = self.base / 'second snapshot'
        input_root.mkdir()
        namespace = 'cdlm_tinygsm_ancestral_sweep'
        (input_root / namespace).symlink_to(external_root / namespace, target_is_directory=True)
        with self.assertRaises(ValueError):
            evaluator.build_plan([self.source(MDLM_CDLM, input_root)], str(self.output))


class CheckpointArchiveStagingTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(
            tempfile.TemporaryDirectory(prefix='checkpoint tar, v=2 '))).resolve()
        self.archive = self.base / 'input state.tar'
        self.staging = self.base / 'staging space'
        self.namespace, self.files = EXPECTED[MDLM_CDLM]

    def member(self, filename, prefix='state', payload=b'weights', kind=tarfile.REGTYPE):
        name = f'{prefix}/{self.namespace}/checkpoints/{filename}'
        info = tarfile.TarInfo(name)
        info.type = kind
        info.size = len(payload) if kind == tarfile.REGTYPE else 0
        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            info.linkname = '/outside/checkpoint.ckpt'
        return info, payload

    def write_archive(self, members=None):
        if members is None:
            members = [self.member(name) for name, _ in self.files]
        with tarfile.open(self.archive, 'w') as tar:
            for info, payload in members:
                tar.addfile(info, io.BytesIO(payload) if info.isfile() else None)

    def test_only_selected_checkpoint_payloads_are_extracted(self):
        members = [self.member(name) for name, _ in self.files]
        for name in ('state/cdlm_trajectories/shard_0000.pt',
                     'state/mmd_from_cdlm_tinygsm/checkpoints/step=00000500.ckpt',
                     'state/underdeep_metadata.pt'):
            info = tarfile.TarInfo(name)
            info.size = 8
            members.append((info, b'unneeded'))
        self.write_archive(members)
        with patch.object(tarfile.TarFile, 'extractall', side_effect=AssertionError('Do not extract whole snapshots')):
            destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        self.assertEqual(destination, self.staging / MDLM_CDLM)
        relative_files = {p.relative_to(destination).as_posix()
                          for p in destination.rglob('*') if p.is_file()}
        self.assertEqual(relative_files, {
            f'{self.namespace}/checkpoints/12-1000.ckpt',
            f'{self.namespace}/checkpoints/last.ckpt', 'archive_receipt.json'})
        for filename, _ in self.files:
            self.assertEqual((destination / self.namespace / 'checkpoints' / filename).read_bytes(), b'weights')

    def test_missing_member_fails_before_publishing_any_checkpoint(self):
        self.write_archive([self.member('12-1000.ckpt')])
        with self.assertRaises(FileNotFoundError):
            evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging,
                                    missing_checkpoints='error')
        self.assertFalse((self.staging / MDLM_CDLM / 'archive_receipt.json').exists())
        self.assertFalse(list(self.staging.rglob('*.ckpt')))

    def test_partial_archive_reuses_receipt_but_strict_policy_rejects_missing_member(self):
        self.write_archive([self.member('12-1000.ckpt')])
        destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        self.assertTrue((destination / self.namespace / 'checkpoints/12-1000.ckpt').is_file())
        self.assertFalse((destination / self.namespace / 'checkpoints/last.ckpt').exists())
        with patch.object(evaluator.tarfile, 'open', side_effect=AssertionError('Reuse partial cache')):
            self.assertEqual(evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging), destination)
        with self.assertRaises(FileNotFoundError):
            evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging,
                                    missing_checkpoints='error')
        prepared = evaluator.prepare_sources(
            [{'source_run': MDLM_CDLM, 'snapshot_root': str(self.archive)}], self.staging)
        plan = evaluator.build_plan(prepared, self.base / 'results')
        self.assertEqual([row['step'] for row in plan['entries']], [1000])
        self.assertEqual([row['step'] for row in plan['skipped_checkpoints']], [1248])

    def test_zero_byte_archive_member_is_reported_as_empty_in_plan(self):
        self.write_archive([self.member('12-1000.ckpt'), self.member('last.ckpt', payload=b'')])
        destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        plan = evaluator.build_plan(
            [{'source_run': MDLM_CDLM, 'snapshot_root': str(destination)}], self.base / 'results')
        self.assertEqual([row['step'] for row in plan['entries']], [1000])
        self.assertEqual([(row['step'], row['reason']) for row in plan['skipped_checkpoints']],
                         [(1248, 'empty')])

    def test_archive_without_usable_expected_checkpoints_fails_even_in_skip_mode(self):
        for members in ([], [self.member('last.ckpt', payload=b'')]):
            with self.subTest(members=len(members)):
                self.write_archive(members)
                with self.assertRaises(FileNotFoundError):
                    evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
                self.assertFalse(list(self.staging.rglob('*.ckpt')))

    def test_changed_archive_cannot_reuse_checkpoint_absent_from_new_snapshot(self):
        self.write_archive()
        destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        old_last = destination / self.namespace / 'checkpoints/last.ckpt'
        self.assertTrue(old_last.is_file())
        previous_stat = self.archive.stat()
        self.write_archive([self.member('12-1000.ckpt', payload=b'new-model')])
        os.utime(self.archive, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000))
        destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        self.assertFalse((destination / self.namespace / 'checkpoints/last.ckpt').exists())
        self.assertEqual((destination / self.namespace / 'checkpoints/12-1000.ckpt').read_bytes(),
                         b'new-model')
        plan = evaluator.build_plan(
            [{'source_run': MDLM_CDLM, 'snapshot_root': str(destination)}], self.base / 'results')
        self.assertEqual([row['step'] for row in plan['entries']], [1000])

    def test_duplicate_suffixes_under_different_prefixes_are_rejected(self):
        members = [self.member(name) for name, _ in self.files]
        members.append(self.member('last.ckpt', prefix='another state'))
        self.write_archive(members)
        with self.assertRaises(ValueError):
            evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        self.assertFalse(list(self.staging.rglob('*.ckpt')))

    def test_unsafe_checkpoint_members_are_rejected(self):
        cases = [('absolute', '/', tarfile.REGTYPE),
                 ('traversal', 'state/..', tarfile.REGTYPE),
                 ('symlink', 'state', tarfile.SYMTYPE),
                 ('hardlink', 'state', tarfile.LNKTYPE)]
        for label, prefix, kind in cases:
            with self.subTest(kind=label):
                self.write_archive([self.member('12-1000.ckpt'),
                                    self.member('last.ckpt', prefix=prefix, kind=kind)])
                with self.assertRaises(ValueError):
                    evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        self.assertFalse(list(self.staging.rglob('*.ckpt')))

    def test_matching_receipt_skips_archive_reads_and_missing_file_repairs(self):
        self.write_archive()
        destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        with patch.object(evaluator.tarfile, 'open', side_effect=AssertionError('Matching cache must be reused')):
            self.assertEqual(evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging), destination)
        target = destination / self.namespace / 'checkpoints/last.ckpt'
        target.unlink()
        evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        self.assertEqual(target.read_bytes(), b'weights')

    def test_changed_archive_invalidates_receipt_even_when_sizes_match(self):
        self.write_archive()
        destination = evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        before = json.loads((destination / 'archive_receipt.json').read_text())
        old_stat = self.archive.stat()
        self.write_archive([self.member(name, payload=b'updated') for name, _ in self.files])
        os.utime(self.archive, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
        self.assertEqual(self.archive.stat().st_size, old_stat.st_size)
        evaluator.stage_archive(self.archive, MDLM_CDLM, self.staging)
        after = json.loads((destination / 'archive_receipt.json').read_text())
        self.assertNotEqual(before['archive_identity'], after['archive_identity'])
        for filename, _ in self.files:
            self.assertEqual((destination / self.namespace / 'checkpoints' / filename).read_bytes(), b'updated')


class CheckpointEvaluationConfigTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(
            tempfile.TemporaryDirectory(prefix='eval offline config '))).resolve()
        self.code = self.base / 'code with spaces'
        self.code.mkdir()
        self.output = self.base / 'evaluation, v=1'
        bundle = self.code / 'cache_dir/tinygsm_mmd_offline_bundle'
        tokenizer = bundle / 'tokenizer'
        tokenizer.mkdir(parents=True)
        (tokenizer / 'tokenizer.json').write_text('{}')
        (bundle / 'manifest.json').write_text(json.dumps({'assets': [{
            'repo_id': 'HuggingFaceTB/SmolLM-135M', 'folder': 'tokenizer'}]}))
        self.dataset = self.code / 'data/gsm8k_test.json'
        self.dataset.parent.mkdir()
        self.dataset.write_text(json.dumps([
            {'prompt': f'Question {i}?', 'response_ground_truth': '#### 42'} for i in range(1319)]))

    def config(self, *overrides):
        return evaluator.compose_evaluation_config(
            list(overrides), code_dir=self.code, output_dir=self.output, devices=4)

    def test_offline_composition_resolves_protocol_without_model_loading(self):
        from omegaconf import OmegaConf
        cfg = self.config()
        self.assertEqual(cfg.algo.name, 'cdlm')
        self.assertEqual(cfg.model.length, 512)
        self.assertFalse(cfg.model.force_fp32)
        self.assertEqual(cfg.trainer.precision, 'bf16-mixed')
        self.assertEqual(cfg.trainer.devices, 4)
        self.assertFalse(cfg.eval.disable_ema)
        self.assertEqual(cfg.eval.gsm8k_pareto_temperature, 0.)
        self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds),
                         [.325, .4, .475, .55, .625, .7, .775, .85, .925, .95, .98, .99, .995])
        self.assertEqual(list(cfg.eval.gsm8k_ancestral_steps), [4, 8, 16, 32, 48, 64])
        self.assertEqual(list(cfg.eval.gsm8k_ancestral_seeds), [1, 2, 3])
        self.assertEqual(list(cfg.eval.gsm8k_ancestral_temperatures), [0.1, 1.0])
        self.assertFalse(cfg.eval.gsm8k_checkpoint_confidence)
        self.assertTrue(cfg.eval.gsm8k_ancestral_final_denoise)
        self.assertEqual(cfg.data.separator, r'\n')
        self.assertEqual(Path(cfg.eval.gsm8k_data_path), self.dataset)
        self.assertEqual(cfg.logger._target_, 'underdeep_logger.UnderdeepLogger')
        self.assertEqual(cfg.training.finetune_path, '')
        self.assertFalse(cfg.adversarial_distill.is_distill)
        self.assertNotIn('${', OmegaConf.to_yaml(cfg, resolve=True))

    def test_comparison_breaking_precision_or_seed_overrides_fail(self):
        overrides = ['model.force_fp32=True', 'trainer.precision=32-true',
                     'eval.disable_ema=True', 'eval.gsm8k_ancestral_seeds=[1,1]',
                     'eval.gsm8k_ancestral_seeds=[]', 'eval.gsm8k_pareto_temperature=1',
                     'eval.gsm8k_ancestral_temperatures=[-1]',
                     'eval.gsm8k_ancestral_temperatures=[0.1,0.1]',
                     'eval.gsm8k_ancestral_temperatures=[]',
                     'eval.gsm8k_ancestral_final_denoise=False', 'trainer.devices=1']
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.config(override)

    def test_missing_explicit_dataset_never_falls_back_or_downloads(self):
        with self.assertRaises(FileNotFoundError):
            self.config('eval.gsm8k_data_path=' + json.dumps(str(self.base / 'missing.json')))

    def test_full_evaluation_requires_1319_records_and_rejects_sample_limits(self):
        self.dataset.write_text(json.dumps([{'prompt': 'Q?', 'response_ground_truth': '#### 1'}]))
        with self.assertRaises(ValueError):
            self.config()
        with self.assertRaises(ValueError):
            self.config('eval.gsm8k_num_examples=1')

    def test_launcher_has_valid_bash_syntax(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/eval_cdlm_checkpoints_nirvana.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)


class CheckpointEvaluationOutputTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='eval output '))).resolve()
        self.state = self.base / 'input state'
        self.output = self.base / 'new output'
        self.state.mkdir()

    def test_only_evaluation_namespace_is_restored_into_an_empty_output(self):
        import nirvana_snapshot
        training = self.state / 'cdlm_tinygsm' / 'underdeep_metadata.pt'
        training.parent.mkdir()
        training.write_bytes(b'training-tracker-must-not-be-restored')
        evaluation = self.state / 'cdlm_checkpoint_eval' / MDLM_CDLM
        evaluation.mkdir(parents=True)
        (evaluation / 'underdeep_metadata.pt').write_bytes(b'evaluation-tracker')
        (evaluation / 'result.json').write_text('{"complete": true}')
        sdk = SimpleNamespace(snapshot=SimpleNamespace(get_snapshot_path=lambda: str(self.state)))
        with patch.object(nirvana_snapshot, '_sdk', return_value=sdk):
            self.assertTrue(evaluator.restore_evaluation_output(self.output, 'cdlm_checkpoint_eval'))
        self.assertEqual((self.output / MDLM_CDLM / 'underdeep_metadata.pt').read_bytes(),
                         b'evaluation-tracker')
        self.assertFalse((self.output / 'cdlm_tinygsm').exists())
        self.assertFalse((self.output / 'underdeep_metadata.pt').exists())
        local_result = self.output / MDLM_CDLM / 'result.json'
        local_result.write_text('newer-local-result')
        with patch.object(nirvana_snapshot, '_sdk', side_effect=AssertionError('Do not overwrite nonempty local outputs')):
            self.assertFalse(evaluator.restore_evaluation_output(self.output, 'cdlm_checkpoint_eval'))
        self.assertEqual(local_result.read_text(), 'newer-local-result')
        self.assertEqual(training.read_bytes(), b'training-tracker-must-not-be-restored')

    def test_existing_remote_evaluation_with_failed_restore_raises_clearly(self):
        import nirvana_snapshot
        (self.state / 'cdlm_checkpoint_eval').mkdir()
        sdk = SimpleNamespace(snapshot=SimpleNamespace(get_snapshot_path=lambda: str(self.state)))
        with patch.object(nirvana_snapshot, '_sdk', return_value=sdk), \
                patch.object(nirvana_snapshot, 'restore', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'could not be restored'):
                evaluator.restore_evaluation_output(self.output, 'cdlm_checkpoint_eval')

    def test_summary_preserves_source_step_and_dtype_in_32_rows_per_checkpoint(self):
        plan = {'output_dir': str(self.output), 'entries': []}
        self.output.mkdir()
        thresholds = [.325, .4, .475, .55, .625, .7, .775, .85, .925, .95, .98, .99, .995]
        for uid, step in ((MDLM_CDLM, 1248), (CDLM_MMD, 7000)):
            entry = {'source_run': uid, 'family': evaluator.SOURCE_RUNS[uid]['family'],
                     'step': step, 'output_dir': str(self.output / uid)}
            directory = Path(entry['output_dir'])
            directory.mkdir()
            result = {
                'complete': True, 'identity': {'checkpoint_sha256': uid.replace('-', '') * 2},
                'protocol': {'seed': 1},
                'confidence': {precision: [
                    {'threshold': t, 'accuracy': (20 + i) / 1319,
                     'nfe': 3.25 + i, 'correct': 20 + i, 'count': 1319}
                    for i, t in enumerate(thresholds)] for precision in ('bf16', 'fp32')},
                'ancestral': {'points': [
                    {'steps': budget, 'seeds': [{'seed': 1, 'accuracy': 33 / 1319,
                     'nfe': budget + 1., 'correct': 33, 'count': 1319}]}
                    for budget in (4, 8, 16, 32, 48, 64)]},
            }
            (directory / 'result.json').write_text(json.dumps(result))
            plan['entries'].append(entry)
        output = evaluator.write_summary(plan)
        with output.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 64)
        for entry in plan['entries']:
            selected = [row for row in rows if row['source_run'] == entry['source_run']]
            self.assertEqual(len(selected), 32)
            self.assertEqual({row['checkpoint_step'] for row in selected}, {str(entry['step'])})
            self.assertEqual({row['family'] for row in selected}, {entry['family']})
            self.assertEqual({row['seed'] for row in selected}, {'1'})
            self.assertEqual({row['count'] for row in selected}, {'1319'})
            for precision in ('bf16', 'fp32'):
                confidence = [row for row in selected if row['sampler'] == 'confidence'
                              and row['logit_normalization'] == precision]
                self.assertEqual([float(row['threshold']) for row in confidence], thresholds)
                self.assertAlmostEqual(float(confidence[0]['accuracy']), 20 / 1319)
            ancestral = [row for row in selected if row['sampler'] == 'ancestral']
            self.assertEqual([int(row['sampling_steps']) for row in ancestral], [4, 8, 16, 32, 48, 64])
            self.assertEqual({row['logit_normalization'] for row in ancestral}, {'fp32'})
        previous = output.read_bytes()
        incomplete = Path(plan['entries'][0]['output_dir']) / 'result.json'
        incomplete.write_text('{"complete": false}')
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            evaluator.write_summary(plan)
        self.assertEqual(output.read_bytes(), previous)

    def test_single_connected_sdk_snapshot_plans_without_gpu_and_missing_state_fails_early(self):
        from omegaconf import OmegaConf
        namespace, files = EXPECTED[MDLM_CDLM]
        directory = self.state / namespace / 'checkpoints'
        directory.mkdir(parents=True)
        for filename, _ in files:
            (directory / filename).write_bytes(b'checkpoint')
        config = OmegaConf.create({'checkpointing': {'snapshot': False}})
        clean_env = {key: value for key, value in os.environ.items()
                     if key not in ('CDLM_EVAL_SOURCE_RUN', 'CDLM_EVAL_SNAPSHOT_ROOT')}
        with patch.dict(os.environ, clean_env, clear=True), \
                patch('cdlm_snapshot_init.resolve_snapshot_root', return_value=self.state) as discover, \
                patch.object(evaluator, 'compose_evaluation_config', return_value=config) as compose, \
                patch.object(evaluator, 'restore_evaluation_output', return_value=False), \
                patch.object(evaluator.subprocess, 'run') as worker:
            code = evaluator.main(['--source-run', MDLM_CDLM[:8], '--output-dir', str(self.output), '--plan-only'])
            self.assertEqual(code, 0)
            discover.assert_called_once_with()
            worker.assert_not_called()
            plan = json.loads((self.output / 'plan.json').read_text())
            self.assertEqual({row['source_run'] for row in plan['entries']}, {MDLM_CDLM})
            self.assertEqual([row['step'] for row in plan['entries']], [1000, 1248])
            compose.reset_mock()
            discover.side_effect = FileNotFoundError('Required input state is absent')
            with self.assertRaisesRegex(FileNotFoundError, 'input state is absent'):
                evaluator.main(['--source-run', MDLM_CDLM[:8], '--output-dir', str(self.output)])
            compose.assert_not_called()
            worker.assert_not_called()

    def test_cli_defaults_to_available_checkpoints_and_supports_explicit_strict_policy(self):
        from omegaconf import OmegaConf
        namespace, files = EXPECTED[CDLM_MMD]
        directory = self.state / namespace / 'checkpoints'
        directory.mkdir(parents=True)
        for filename, step in files:
            if step != 6000:
                (directory / filename).write_bytes(b'checkpoint')
        config = OmegaConf.create({'checkpointing': {'snapshot': False}})
        args = ['--source-run', CDLM_MMD[:8], '--snapshot-root', str(self.state),
                '--output-dir', str(self.output), '--plan-only']
        stdout = io.StringIO()
        with patch.object(evaluator, 'compose_evaluation_config', return_value=config) as compose, \
                patch.object(evaluator, 'restore_evaluation_output', return_value=False), \
                patch.object(evaluator.subprocess, 'run') as worker, \
                patch('sys.stdout', stdout):
            self.assertEqual(evaluator.main(args), 0)
            plan = json.loads((self.output / 'plan.json').read_text())
            self.assertEqual(plan['missing_checkpoints'], 'skip')
            self.assertEqual(len(plan['entries']), 13)
            self.assertEqual([row['step'] for row in plan['entries']][-2:], [6500, 7000])
            self.assertEqual([row['step'] for row in plan['skipped_checkpoints']], [6000])
            self.assertIn('step=00006000.ckpt', stdout.getvalue())
            worker.assert_not_called()
            compose.reset_mock()
            with self.assertRaises(FileNotFoundError):
                evaluator.main(args + ['--missing-checkpoints', 'error'])
            compose.assert_not_called()
            worker.assert_not_called()
            for filename, _ in files:
                (directory / filename).unlink(missing_ok=True)
            with self.assertRaises(FileNotFoundError):
                evaluator.main(args)
            compose.assert_not_called()
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
