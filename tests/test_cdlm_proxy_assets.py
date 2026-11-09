"""Checkpoint proxy scope stays outside local assets and Nirvana snapshot IO."""

from contextlib import contextmanager, redirect_stdout
import io
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cdlm_checkpoint_assets as assets
import cdlm_download_proxy as proxy


PROXY_KEYS = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY')
BYPASS_KEYS = ('no_proxy', 'NO_PROXY')
ENV_KEYS = PROXY_KEYS + BYPASS_KEYS + ('ALL_PROXY',)
MMD = 'hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt'


class ProxyAssetIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.code = self.directory / 'code'
        self.code.mkdir()
        self.store = self.directory / 'snapshot'
        self.store.mkdir()
        self.checkpoint = self.directory / 'downloaded.ckpt'
        self.checkpoint.write_bytes(b'byte-exact mocked MMD EMA weights')
        self.events = []
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch.dict(os.environ, {
            'INPUT_PATH': '', 'HF_HUB_OFFLINE': '0',
            'CDLM_DOWNLOAD_PROXY': 'https://name:TOP_SECRET@proxy.invalid:8443',
            'http_proxy': 'inherited-http', 'https_proxy': 'inherited-https',
            'HTTP_PROXY': 'inherited-HTTP', 'HTTPS_PROXY': 'inherited-HTTPS',
            'no_proxy': 'localhost', 'NO_PROXY': 'internal', 'ALL_PROXY': 'original-all'}))
        self.original_environment = self.environment()
        self.proxy_context = self.enterContext(patch.object(
            proxy, 'download_proxy', side_effect=self.download_context))
        self.hub = self.enterContext(patch.object(
            assets, 'hf_hub_download', side_effect=self.download))
        sdk = SimpleNamespace(snapshot=SimpleNamespace(
            get_snapshot_path=self.snapshot_path, dump_snapshot=self.snapshot_dump))
        self.enterContext(patch.object(assets.nirvana_snapshot, '_sdk', return_value=sdk))

    def environment(self):
        return {key: os.environ.get(key) for key in ENV_KEYS}

    def assert_original_environment(self):
        self.assertEqual(self.environment(), self.original_environment)

    @contextmanager
    def download_context(self):
        self.assert_original_environment()
        self.events.append('proxy-enter')
        previous = self.environment()
        try:
            os.environ.update(dict.fromkeys(PROXY_KEYS, 'checkpoint-download-only'))
            os.environ.update(dict.fromkeys(BYPASS_KEYS, ''))
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self.events.append('proxy-exit')

    def download(self, **kwargs):
        self.assertEqual(kwargs['repo_id'], 'yresearch/MDLM-MMD-TinyGSM')
        self.assertEqual(kwargs['filename'], 'model.ckpt')
        if kwargs.get('local_files_only'):
            self.assert_original_environment()
            self.events.append('local-lookup')
            raise FileNotFoundError('Fixture has no HF cache.')
        self.events.append('remote-download')
        for key in PROXY_KEYS:
            self.assertEqual(os.environ[key], 'checkpoint-download-only')
        for key in BYPASS_KEYS:
            self.assertEqual(os.environ[key], '')
        return str(self.checkpoint)

    def snapshot_path(self):
        self.assert_original_environment()
        self.events.append('snapshot-path')
        return str(self.store)

    def snapshot_dump(self, path):
        self.assert_original_environment()
        self.assertEqual(path, str(self.store))
        self.events.append('snapshot-dump')

    def prepare(self, source=MMD, *, snapshot=True):
        return Path(assets.prepare_checkpoint(source, code_dir=self.code,
                                             snapshot_enabled=snapshot))

    def test_remote_miss_uses_proxy_then_restores_environment_before_snapshot_save(self):
        result = self.prepare()
        self.assertEqual(result.read_bytes(), self.checkpoint.read_bytes())
        self.proxy_context.assert_called_once_with()
        self.assertLess(self.events.index('local-lookup'), self.events.index('proxy-enter'))
        self.assertLess(self.events.index('proxy-enter'), self.events.index('remote-download'))
        self.assertLess(self.events.index('remote-download'), self.events.index('proxy-exit'))
        self.assertLess(self.events.index('proxy-exit'), self.events.index('snapshot-dump'))
        self.assert_original_environment()
        self.assertEqual(len(list(self.store.rglob('checkpoint.ckpt'))), 1)

    def test_local_packaged_and_managed_weights_never_enter_proxy_context(self):
        explicit = self.prepare(str(self.checkpoint), snapshot=False)
        self.assertEqual(explicit, self.checkpoint)
        alias = self.code / 'cdlm_model_assets' / 'mmd_tinygsm.ckpt'
        alias.parent.mkdir()
        shutil.copyfile(self.checkpoint, alias)
        staged = self.prepare(snapshot=False)
        alias.unlink()
        managed = self.prepare(snapshot=False)
        self.assertEqual(staged, managed)
        self.assertEqual(managed.read_bytes(), self.checkpoint.read_bytes())
        self.proxy_context.assert_not_called()
        self.hub.assert_not_called()
        self.assert_original_environment()

    def test_standard_hf_cache_hit_never_enters_proxy_context(self):
        self.hub.side_effect = None
        self.hub.return_value = str(self.checkpoint)
        result = self.prepare(snapshot=False)
        self.assertEqual(result.read_bytes(), self.checkpoint.read_bytes())
        self.assertEqual(self.hub.call_count, 1)
        self.assertTrue(self.hub.call_args.kwargs['local_files_only'])
        self.proxy_context.assert_not_called()
        self.assert_original_environment()

    def test_snapshot_restore_happens_without_proxy_registration_or_remote_download(self):
        alias = self.code / 'cdlm_model_assets' / 'mmd_tinygsm.ckpt'
        alias.parent.mkdir()
        shutil.copyfile(self.checkpoint, alias)
        result = self.prepare()
        self.assertIn('snapshot-dump', self.events)
        alias.unlink()
        shutil.rmtree(self.code / 'cache_dir' / 'cdlm_checkpoint_assets')
        self.events.clear()
        restored = self.prepare()
        self.assertEqual(restored, result)
        self.assertEqual(restored.read_bytes(), self.checkpoint.read_bytes())
        self.assertIn('snapshot-path', self.events)
        self.proxy_context.assert_not_called()
        self.assertTrue(all(call.kwargs.get('local_files_only')
                            for call in self.hub.call_args_list))
        self.assert_original_environment()

    def test_failed_hub_download_restores_environment_and_hides_credentials(self):
        def fail(**kwargs):
            if kwargs.get('local_files_only'):
                raise FileNotFoundError('No cached checkpoint.')
            self.assertEqual(os.environ['HTTP_PROXY'], 'checkpoint-download-only')
            raise RuntimeError('https://name:TOP_SECRET@proxy.invalid download failed')

        self.hub.side_effect = fail
        with self.assertRaises(FileNotFoundError) as error:
            self.prepare()
        self.proxy_context.assert_called_once_with()
        self.assert_original_environment()
        self.assertNotIn('snapshot-dump', self.events)
        for secret in ('TOP_SECRET', 'name:', 'proxy.invalid'):
            self.assertNotIn(secret, str(error.exception))
            self.assertNotIn(secret, self.output.getvalue())


if __name__ == '__main__':
    unittest.main()
