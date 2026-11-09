"""Checkpoint preflight tests with tiny files and no network/model loading."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cdlm_checkpoint_assets as assets


class CheckpointAssetsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.code, self.inputs, self.store = [self.base / name
                                             for name in ('code', 'inputs', 'snapshot')]
        for path in (self.code, self.inputs, self.store):
            path.mkdir()
        self.dumps = []
        sdk = ModuleType('nirvana_dl')
        sdk.snapshot = SimpleNamespace(get_snapshot_path=lambda: str(self.store),
                                       dump_snapshot=self.dumps.append)
        self.enterContext(patch.dict(sys.modules, {'nirvana_dl': sdk}))
        self.enterContext(patch.dict('os.environ', {}, clear=True))
        self.hub = self.enterContext(patch.object(
            assets, 'hf_hub_download', side_effect=FileNotFoundError('not cached')))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def write(self, path, data=b'exact model bytes\0\xff'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def resolve(self, source=assets.MMD_SOURCE, **kwargs):
        return Path(assets.prepare_checkpoint(source, code_dir=self.code,
                                             input_dir=self.inputs, **kwargs))

    def asset_dir(self, source=assets.MMD_SOURCE):
        key = hashlib.sha256(source.encode()).hexdigest()
        return self.code / 'cache_dir/cdlm_checkpoint_assets' / key

    def snapshot_dir(self, source=assets.MMD_SOURCE):
        return self.store / 'cdlm_assets' / hashlib.sha256(source.encode()).hexdigest()

    def test_explicit_local_paths_are_validated_without_hub_or_snapshot(self):
        original = self.write(self.code / 'checkpoints/source.ckpt')
        self.assertEqual(self.resolve('checkpoints/source.ckpt'), original)
        self.assertEqual(self.resolve(str(original)), original)
        for source in ('missing.ckpt', '', 'empty.ckpt'):
            self.write(self.code / 'empty.ckpt', b'')
            with self.subTest(source=source), self.assertRaisesRegex(
                    FileNotFoundError, 'no other weights'):
                self.resolve(source)
        self.hub.assert_not_called()
        self.assertEqual(self.dumps, [])
        self.assertFalse((self.code / 'cache_dir').exists())

    def test_download_once_preserves_bytes_metadata_and_skips_duplicate_snapshot(self):
        download = self.write(self.base / 'hub-download.ckpt')

        def hub(**kwargs):
            if kwargs.get('local_files_only'):
                raise FileNotFoundError('not cached')
            return str(download)

        self.hub.side_effect = hub
        result = self.resolve()
        self.assertTrue(result.is_absolute())
        self.assertEqual(result.read_bytes(), download.read_bytes())
        metadata = json.loads((result.parent / 'metadata.json').read_text())
        self.assertEqual(metadata['source_uri'], assets.MMD_SOURCE)
        self.assertEqual(metadata['size_bytes'], download.stat().st_size)
        self.assertEqual(metadata['sha256'], hashlib.sha256(download.read_bytes()).hexdigest())
        self.assertEqual(self.dumps, [str(self.store)])
        self.assertEqual((self.snapshot_dir() / 'checkpoint.ckpt').read_bytes(),
                         download.read_bytes())
        self.hub.reset_mock()
        self.assertEqual(self.resolve(), result)
        self.hub.assert_not_called()
        self.assertEqual(len(self.dumps), 1)
        self.assertEqual(list(result.parent.glob('.*')), [])

    def test_relative_local_bundle_path_preserves_exact_subpath_and_code_precedence(self):
        bundle = self.base / 'bundle'
        original = self.write(bundle / 'nested/teacher.ckpt')
        self.assertEqual(self.resolve('nested/teacher.ckpt', weights_dir=bundle), original)
        explicit = self.write(self.code / 'nested/teacher.ckpt', b'explicit code weights')
        self.assertEqual(self.resolve('nested/teacher.ckpt', weights_dir=bundle), explicit)
        with self.assertRaises(FileNotFoundError):
            self.resolve('missing/teacher.ckpt', weights_dir=bundle)
        with self.assertRaises(FileNotFoundError):
            self.resolve(str(self.code / 'absent/teacher.ckpt'), weights_dir=bundle)
        self.hub.assert_not_called()

    def test_download_staging_error_is_not_hidden_as_network_failure(self):
        download = self.write(self.base / 'hub-download.ckpt')

        def hub(**kwargs):
            if kwargs.get('local_files_only'):
                raise FileNotFoundError('not cached')
            return str(download)

        self.hub.side_effect = hub
        with patch.object(assets, '_stage', side_effect=OSError('No space left on device')):
            with self.assertRaisesRegex(OSError, 'No space left on device'):
                self.resolve(snapshot_enabled=False)
        self.assertIn('Downloading requested checkpoint', self.output.getvalue())

    def test_packaged_alias_is_offline_and_provenance_is_checked(self):
        original = self.write(self.code / 'cdlm_model_assets/mmd_tinygsm.ckpt')
        original.with_suffix('.json').write_text(json.dumps(dict(
            source_uri=assets.MMD_SOURCE, size_bytes=original.stat().st_size,
            sha256=hashlib.sha256(original.read_bytes()).hexdigest(), git_commit='abc')))
        with patch.dict('os.environ', {'HF_HUB_OFFLINE': '1'}):
            result = self.resolve(snapshot_enabled=False)
        self.assertEqual(result.read_bytes(), original.read_bytes())
        self.hub.assert_not_called()
        self.assertEqual(self.dumps, [])
        shutil.rmtree(self.asset_dir())
        original.write_bytes(b'changed package')
        with self.assertRaisesRegex(ValueError, 'staged checkpoint provenance'):
            self.resolve()
        self.hub.assert_not_called()

    def test_input_and_checkpoint_aliases_and_environment_input_path(self):
        for alias in (self.inputs / 'mmd_tinygsm.ckpt',
                      self.code / 'checkpoints/tinygsm/mmd_tinygsm.ckpt'):
            with self.subTest(alias=alias):
                self.write(alias)
                with patch.dict('os.environ', {'INPUT_PATH': str(self.inputs)}):
                    result = Path(assets.prepare_checkpoint(
                        assets.MMD_SOURCE, code_dir=self.code, snapshot_enabled=False))
                self.assertEqual(result.read_bytes(), alias.read_bytes())
                alias.unlink()
                shutil.rmtree(self.asset_dir())
        self.hub.assert_not_called()

    def test_generic_hub_source_ignores_mmd_plain_file_aliases(self):
        self.write(self.inputs / 'mmd_tinygsm.ckpt', b'wrong model')
        self.write(self.code / 'cdlm_model_assets/mmd_tinygsm.ckpt', b'wrong model')
        with self.assertRaisesRegex(FileNotFoundError, 'other/repo/nested/model.ckpt'):
            self.resolve('hf://other/repo/nested/model.ckpt', snapshot_enabled=False)
        for call in self.hub.call_args_list:
            self.assertEqual(call.kwargs['repo_id'], 'other/repo')
            self.assertEqual(call.kwargs['filename'], 'nested/model.ckpt')

    def test_staged_hf_cache_and_bundle_cache_are_local_only(self):
        for cache in (self.inputs / 'cdlm_mmd_hf_cache',
                      self.code / 'cache_dir/cdlm_mmd_hf_cache/hub',
                      self.base / 'bundle/hf-cache'):
            with self.subTest(cache=cache):
                cache.mkdir(parents=True)
                cached = self.write(cache / 'cached.ckpt')

                def hub(**kwargs):
                    self.assertTrue(kwargs['local_files_only'])
                    if kwargs.get('cache_dir') == str(cache):
                        return str(cached)
                    raise FileNotFoundError('not this cache')

                self.hub.side_effect = hub
                result = self.resolve(weights_dir=self.base / 'bundle', snapshot_enabled=False)
                self.assertEqual(result.read_bytes(), cached.read_bytes())
                shutil.rmtree(self.asset_dir())
                shutil.rmtree(cache)

    def test_default_hf_cache_uses_local_files_only_before_network(self):
        cached = self.write(self.base / 'default-cache.ckpt')
        self.hub.return_value = str(cached)
        self.hub.side_effect = None
        self.resolve(snapshot_enabled=False)
        self.hub.assert_called_once_with(repo_id='yresearch/MDLM-MMD-TinyGSM',
                                         filename='model.ckpt', local_files_only=True)

    def test_offline_snapshot_restore_precedes_download_and_model_namespaces_are_isolated(self):
        source_file = self.write(self.inputs / 'mmd_tinygsm.ckpt')
        original_bytes = source_file.read_bytes()
        self.resolve()
        source_file.unlink()
        shutil.rmtree(self.asset_dir())

        def hub(**kwargs):
            self.assertTrue(kwargs['local_files_only'], 'network must not be reached')
            raise FileNotFoundError('offline cache miss')

        self.hub.side_effect = hub
        with patch.dict('os.environ', {'HF_HUB_OFFLINE': '1'}):
            result = self.resolve()
        self.assertEqual(result.read_bytes(), original_bytes)
        self.assertEqual(len(self.dumps), 1)
        self.hub.side_effect = FileNotFoundError('offline cache miss')
        with self.assertRaises(FileNotFoundError):
            self.resolve('hf://another/repository/model.ckpt')
        self.assertEqual((self.snapshot_dir() / 'checkpoint.ckpt').read_bytes(), original_bytes)

    def test_verified_local_asset_wins_over_snapshot_and_never_restores(self):
        self.write(self.inputs / 'mmd_tinygsm.ckpt')
        result = self.resolve()
        with patch.object(assets.nirvana_snapshot, 'restore') as restore:
            self.assertEqual(self.resolve(), result)
        restore.assert_not_called()

    def test_corrupt_managed_asset_is_rejected_without_download(self):
        self.write(self.inputs / 'mmd_tinygsm.ckpt')
        result = self.resolve(snapshot_enabled=False)
        result.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt CDLM checkpoint asset'):
            self.resolve()
        self.hub.assert_not_called()

    def test_corrupt_snapshot_is_not_restored_or_replaced_by_download(self):
        staged = self.write(self.inputs / 'mmd_tinygsm.ckpt')
        self.resolve()
        staged.unlink()
        shutil.rmtree(self.asset_dir())
        (self.snapshot_dir() / 'checkpoint.ckpt').write_bytes(b'corrupt snapshot')
        with self.assertRaisesRegex(ValueError, 'Corrupt CDLM checkpoint asset'):
            self.resolve()
        self.assertFalse(self.asset_dir().exists())
        self.assertTrue(all(call.kwargs.get('local_files_only')
                            for call in self.hub.call_args_list))

    def test_incomplete_managed_write_can_be_recovered_from_staged_asset(self):
        staged = self.write(self.inputs / 'mmd_tinygsm.ckpt')
        self.write(self.asset_dir() / 'checkpoint.ckpt', b'interrupted incomplete copy')
        result = self.resolve(snapshot_enabled=False)
        self.assertEqual(result.read_bytes(), staged.read_bytes())
        self.assertTrue((self.asset_dir() / 'metadata.json').is_file())

    def test_missing_asset_error_is_actionable_and_does_not_expose_request_secrets(self):
        self.hub.side_effect = RuntimeError('request failed at https://name:SECRET@proxy')
        with self.assertRaises(FileNotFoundError) as error:
            self.resolve(snapshot_enabled=False)
        message = str(error.exception)
        self.assertIn('before GPU launch', message)
        self.assertIn('Offline workers cannot download', message)
        self.assertIn('mmd_tinygsm.ckpt', message)
        self.assertIn('standard Hugging Face cache', message)
        self.assertNotIn('SECRET', message)

    def test_optional_snapshot_failure_does_not_block_valid_asset(self):
        staged = self.write(self.inputs / 'mmd_tinygsm.ckpt')
        with patch.object(assets.nirvana_snapshot, 'save', side_effect=RuntimeError('SDK failed')):
            result = self.resolve()
        self.assertEqual(result.read_bytes(), staged.read_bytes())

    def test_malformed_hub_uri_rejected_without_io(self):
        for source in ('hf://repo', 'hf:///repo/file', 'hf://org/repo/../file',
                       'hf://org/repo//absolute'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.resolve(source)
        self.hub.assert_not_called()


if __name__ == '__main__':
    unittest.main()
