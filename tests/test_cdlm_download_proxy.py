"""Proxy registration and environment handling without network requests."""

from contextlib import redirect_stdout
import io
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

import cdlm_download_proxy as proxy


class DownloadProxyTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.factory = self.enterContext(patch.object(proxy.requests, 'Session'))
        self.session = self.factory.return_value.__enter__.return_value
        self.session.get.return_value.status_code = 200
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def test_unset_and_empty_setting_preserve_inherited_environment(self):
        for configured in (None, ''):
            with self.subTest(configured=configured):
                os.environ.clear()
                os.environ.update(http_proxy='inherited', NO_PROXY='localhost')
                if configured is not None:
                    os.environ['CDLM_DOWNLOAD_PROXY'] = configured
                previous = dict(os.environ)
                with proxy.download_proxy():
                    self.assertEqual(dict(os.environ), previous)
                self.assertEqual(dict(os.environ), previous)
        self.factory.assert_not_called()
        self.assertEqual(self.output.getvalue(), '')

    def test_offline_truthy_values_skip_even_invalid_proxy_and_preserve_environment(self):
        for offline in ('1', 'ON', 'yes', 'TrUe', ' YES '):
            with self.subTest(offline=offline):
                os.environ.update(CDLM_DOWNLOAD_PROXY='not a URL', HF_HUB_OFFLINE=offline,
                                  http_proxy='existing', no_proxy='localhost')
                previous = dict(os.environ)
                with proxy.download_proxy():
                    self.assertEqual(dict(os.environ), previous)
                self.assertEqual(dict(os.environ), previous)
        self.factory.assert_not_called()

    def test_default_registers_directly_with_quoted_hostname_then_restores_environment(self):
        os.environ.update(CDLM_DOWNLOAD_PROXY=proxy.DEFAULT_PROXY + '/',
                          http_proxy='old lower', HTTPS_PROXY='old upper',
                          no_proxy='localhost', NO_PROXY='internal', ALL_PROXY='inherited')
        previous = dict(os.environ)

        def register(url, **kwargs):
            self.assertFalse(self.session.trust_env)
            self.assertEqual(dict(os.environ), previous, 'registration precedes mutation')
            self.assertEqual(url, proxy.DEFAULT_PROXY + '/register/job%2Fhost%20name')
            self.assertEqual(kwargs, dict(timeout=(10, 30), allow_redirects=False))
            return MagicMock(status_code=204)

        self.session.get.side_effect = register
        with patch.object(proxy.socket, 'gethostname', return_value='job/host name'):
            with proxy.download_proxy():
                for key in proxy._PROXY_KEYS:
                    self.assertEqual(os.environ[key], proxy.DEFAULT_PROXY)
                for key in proxy._BYPASS_KEYS:
                    self.assertEqual(os.environ[key], '')
                self.assertEqual(os.environ['ALL_PROXY'], 'inherited')
        self.assertEqual(dict(os.environ), previous)
        self.factory.assert_called_once_with()
        self.assertIn('registered', self.output.getvalue())

    def test_custom_proxy_does_not_register_or_log_credentials_and_restores_after_error(self):
        configured = 'https://name:SECRET@proxy.example:8443'
        os.environ.update(CDLM_DOWNLOAD_PROXY=configured, NO_PROXY='existing')
        previous = dict(os.environ)
        with self.assertRaisesRegex(ValueError, 'download failed'):
            with proxy.download_proxy():
                for key in proxy._PROXY_KEYS:
                    self.assertEqual(os.environ[key], configured)
                self.assertEqual(os.environ['NO_PROXY'], '')
                raise ValueError('download failed')
        self.assertEqual(dict(os.environ), previous)
        self.factory.assert_not_called()
        self.assertIn('configured proxy', self.output.getvalue())
        self.assertNotIn('SECRET', self.output.getvalue())
        self.assertNotIn('proxy.example', self.output.getvalue())

    def test_registration_network_failure_is_sanitized_and_does_not_mutate_environment(self):
        os.environ.update(CDLM_DOWNLOAD_PROXY=proxy.DEFAULT_PROXY, http_proxy='existing')
        previous = dict(os.environ)
        self.session.get.side_effect = requests.ConnectionError('https://name:SECRET@proxy')
        with self.assertRaisesRegex(RuntimeError, 'Cannot reach the Nirvana') as error:
            with proxy.download_proxy():
                self.fail('registration failure must not enter context')
        self.assertNotIn('SECRET', str(error.exception))
        self.assertIn('package the checkpoint', str(error.exception))
        self.assertEqual(dict(os.environ), previous)

    def test_registration_requires_2xx_and_never_reads_response_body(self):
        os.environ['CDLM_DOWNLOAD_PROXY'] = proxy.DEFAULT_PROXY
        for status in (199, 302, 403, 500):
            with self.subTest(status=status):
                response = MagicMock(status_code=status)
                self.session.get.return_value = response
                previous = dict(os.environ)
                with self.assertRaisesRegex(RuntimeError, f'HTTP {status}'):
                    with proxy.download_proxy():
                        self.fail('HTTP error must not enter context')
                self.assertEqual(dict(os.environ), previous)
                self.assertNotIn('text', response.__dict__)
                self.assertNotIn('content', response.__dict__)

    def test_invalid_proxy_is_rejected_without_registration_or_credential_leaks(self):
        for value in ('proxy.example:6953', 'socks5://proxy.example:1080',
                      'http://', 'http://bad host:80', 'http://proxy:99999',
                      'http://name:SECRET@proxy/path', 'http://proxy?SECRET',
                      'http://proxy#SECRET', 'http://[invalid'):
            with self.subTest(value=value):
                os.environ['CDLM_DOWNLOAD_PROXY'] = value
                previous = dict(os.environ)
                with self.assertRaisesRegex(RuntimeError, r'HTTP\(S\) proxy URL') as error:
                    with proxy.download_proxy():
                        self.fail('invalid URL must not enter context')
                self.assertNotIn('SECRET', str(error.exception))
                self.assertEqual(dict(os.environ), previous)
        self.factory.assert_not_called()

    def test_nested_custom_contexts_restore_each_previous_environment(self):
        os.environ['CDLM_DOWNLOAD_PROXY'] = 'http://outer.example:80'
        with proxy.download_proxy():
            os.environ['CDLM_DOWNLOAD_PROXY'] = 'http://inner.example:80'
            previous = dict(os.environ)
            with proxy.download_proxy():
                self.assertEqual(os.environ['http_proxy'], 'http://inner.example:80')
            self.assertEqual(dict(os.environ), previous)
        self.assertNotIn('http_proxy', os.environ)
        self.assertNotIn('NO_PROXY', os.environ)


if __name__ == '__main__':
    unittest.main()
