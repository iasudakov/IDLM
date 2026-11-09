"""Portable source packaging uses committed code and mocked exact MMD bytes."""

import contextlib
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import cdlm_package


class PackageTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.code = self.directory / 'source code'
        self.code.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Package Test')
        self.git('config', 'user.email', 'package@example.invalid')
        (self.code / 'main.py').write_text('print("committed source")\n')
        (self.code / '.gitignore').write_text('.env\ncache_dir/\n*.tar.gz\n')
        script = self.code / 'scripts' / 'train.sh'
        script.parent.mkdir()
        script.write_text('#!/bin/sh\nexit 0\n')
        script.chmod(0o755)
        self.git('add', '.')
        self.git('commit', '-qm', 'Committed source')
        self.commit = self.git('rev-parse', 'HEAD')
        self.weights = self.directory / 'actual MMD.ckpt'
        self.weights.write_bytes(b'exact mocked native MMD EMA checkpoint\x00\xff')
        self.staging = []

    def git(self, *arguments):
        return subprocess.run(['git', '-C', str(self.code), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()

    def resolve(self, source, **kwargs):
        self.assertEqual(source, cdlm_package.PUBLISHED_MMD)
        self.assertIsNone(kwargs['weights_dir'])
        self.assertFalse(kwargs['snapshot_enabled'])
        staging = Path(kwargs['code_dir'])
        self.assertFalse(staging.is_relative_to(self.code))
        self.staging.append(staging)
        target = staging / 'resolved.ckpt'
        target.symlink_to(self.weights)
        return str(target)

    def package(self, output, *, resolver=None):
        with patch('cdlm_package._prepare_checkpoint', side_effect=resolver or self.resolve) as resolve:
            result = cdlm_package.package(output, code_dir=self.code)
        resolve.assert_called_once()
        self.assertTrue(all(not path.exists() for path in self.staging))
        return result

    def test_archive_contains_only_committed_source_and_exact_dereferenced_mmd(self):
        (self.code / '.env').write_text('DO_NOT_PACKAGE=secret\n')
        cache = self.code / 'cache_dir'
        cache.mkdir()
        (cache / 'unrelated.ckpt').write_bytes(b'not the requested model')
        (self.code / 'untracked.py').write_text('not committed\n')
        output = self.code / 'portable bundle.tar.gz'
        output.write_bytes(b'old output must not enter new archive')
        with patch.dict(os.environ, INPUT_PATH=str(self.directory / 'nirvana inputs')):
            with patch('cdlm_package._prepare_checkpoint', side_effect=self.resolve) as resolve:
                actual = cdlm_package.package(output, code_dir=self.code)
        self.assertEqual(actual, output.resolve())
        self.assertEqual(resolve.call_args.kwargs['input_dir'],
                         str(self.directory / 'nirvana inputs'))
        self.assertTrue(all(not path.exists() for path in self.staging))
        with tarfile.open(actual, 'r:gz') as bundle:
            names = bundle.getnames()
            self.assertIn('code/main.py', names)
            self.assertIn('code/.gitignore', names)
            self.assertTrue(bundle.getmember('code/scripts/train.sh').mode & 0o111)
            self.assertTrue(all(name == 'code' or name.startswith('code/') for name in names))
            self.assertNotIn('code/.git', names)
            self.assertNotIn('code/.env', names)
            self.assertNotIn('code/cache_dir', names)
            self.assertNotIn('code/untracked.py', names)
            self.assertNotIn('code/portable bundle.tar.gz', names)
            asset = bundle.getmember(cdlm_package.CHECKPOINT_MEMBER)
            self.assertTrue(asset.isfile())
            self.assertFalse(asset.issym())
            self.assertEqual(bundle.extractfile(asset).read(), self.weights.read_bytes())
            provenance = json.load(bundle.extractfile(cdlm_package.PROVENANCE_MEMBER))
        self.assertEqual(provenance, {
            'schema_version': 1, 'source_uri': cdlm_package.PUBLISHED_MMD,
            'sha256': hashlib.sha256(self.weights.read_bytes()).hexdigest(),
            'size_bytes': self.weights.stat().st_size, 'git_commit': self.commit})
        self.assertFalse(list(output.parent.glob('.portable bundle.tar.gz.*.tmp')))

    def test_same_commit_and_checkpoint_produce_reproducible_archives(self):
        first = self.package(self.directory / 'first.tar.gz')
        second = self.package(self.directory / 'other name.tar.gz')
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_packaged_checkpoint_is_imported_offline_by_real_asset_resolver(self):
        from cdlm_checkpoint_assets import prepare_checkpoint

        def download(**kwargs):
            if kwargs.get('local_files_only'):
                raise FileNotFoundError('No cached checkpoint in the fixture.')
            return str(self.weights)

        output = self.directory / 'portable.tar.gz'
        with patch.dict(os.environ, INPUT_PATH=''), \
             contextlib.redirect_stdout(StringIO()):
            with patch('cdlm_checkpoint_assets.hf_hub_download',
                       side_effect=download) as hub:
                cdlm_package.package(output, code_dir=self.code)
            self.assertTrue(any(not call.kwargs.get('local_files_only')
                                for call in hub.call_args_list))
            extracted = self.directory / 'extracted'
            extracted.mkdir()
            with tarfile.open(output, 'r:gz') as bundle:
                bundle.extractall(extracted, filter='data')
            with patch('cdlm_checkpoint_assets.hf_hub_download',
                       side_effect=AssertionError('Offline import must not call the Hub.')) as hub:
                imported = Path(prepare_checkpoint(
                    cdlm_package.PUBLISHED_MMD, code_dir=extracted / 'code',
                    snapshot_enabled=False))
            hub.assert_not_called()
        self.assertEqual(imported.read_bytes(), self.weights.read_bytes())
        provenance = json.loads((extracted / cdlm_package.PROVENANCE_MEMBER).read_text())
        self.assertEqual(provenance['sha256'], hashlib.sha256(imported.read_bytes()).hexdigest())
        self.assertEqual(provenance['size_bytes'], imported.stat().st_size)

    def test_dirty_tracked_source_fails_before_checkpoint_resolution(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                (self.code / 'main.py').write_text('changed source\n')
                if staged:
                    self.git('add', 'main.py')
                with patch('cdlm_package._prepare_checkpoint') as resolve:
                    with self.assertRaisesRegex(RuntimeError, 'uncommitted changes'):
                        cdlm_package.package(self.directory / 'output.tar.gz', code_dir=self.code)
                resolve.assert_not_called()
                self.git('reset', '--hard', 'HEAD')

    def test_checkpoint_failure_preserves_previous_output_and_cleans_staging(self):
        output = self.directory / 'existing.tar.gz'
        output.write_bytes(b'previous complete artifact')

        def fail(source, **kwargs):
            self.staging.append(Path(kwargs['code_dir']))
            raise RuntimeError('checkpoint unavailable')

        with patch('cdlm_package._prepare_checkpoint', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint unavailable'):
                cdlm_package.package(output, code_dir=self.code)
        self.assertEqual(output.read_bytes(), b'previous complete artifact')
        self.assertTrue(all(not path.exists() for path in self.staging))
        self.assertFalse(list(output.parent.glob('.existing.tar.gz.*.tmp')))

    def test_compression_failure_preserves_previous_output_and_cleans_temporary_files(self):
        output = self.directory / 'existing.tar.gz'
        output.write_bytes(b'previous complete artifact')
        with patch('cdlm_package._prepare_checkpoint', side_effect=self.resolve), \
             patch('cdlm_package.shutil.copyfileobj', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                cdlm_package.package(output, code_dir=self.code)
        self.assertEqual(output.read_bytes(), b'previous complete artifact')
        self.assertTrue(all(not path.exists() for path in self.staging))
        self.assertFalse(list(output.parent.glob('.existing.tar.gz.*.tmp')))

    def test_source_edits_during_resolution_are_rejected(self):
        def modify(source, **kwargs):
            result = self.resolve(source, **kwargs)
            (self.code / 'main.py').write_text('edited during download\n')
            return result

        with self.assertRaisesRegex(RuntimeError, 'uncommitted changes'):
            self.package(self.directory / 'output.tar.gz', resolver=modify)
        self.assertTrue(all(not path.exists() for path in self.staging))

    def test_reserved_committed_asset_cannot_be_shadowed(self):
        asset = self.code / 'cdlm_model_assets' / 'mmd_tinygsm.ckpt'
        asset.parent.mkdir()
        asset.write_bytes(b'committed conflict')
        self.git('add', str(asset.relative_to(self.code)))
        self.git('commit', '-qm', 'Conflicting packaged asset')
        with self.assertRaisesRegex(RuntimeError, 'already contains'):
            self.package(self.directory / 'output.tar.gz')

    def test_output_cannot_overwrite_a_tracked_source_file(self):
        with patch('cdlm_package._prepare_checkpoint') as resolve:
            with self.assertRaisesRegex(ValueError, 'replace tracked source'):
                cdlm_package.package(self.code / 'main.py', code_dir=self.code)
        resolve.assert_not_called()

    def test_cli_calls_package_and_reports_output(self):
        output = self.directory / 'output.tar.gz'
        stream = StringIO()
        with patch('cdlm_package.package', return_value=output) as package, \
             contextlib.redirect_stdout(stream):
            cdlm_package.main(['--output', str(output)])
        package.assert_called_once_with(str(output))
        self.assertEqual(stream.getvalue(), str(output) + '\n')


if __name__ == '__main__':
    unittest.main()
