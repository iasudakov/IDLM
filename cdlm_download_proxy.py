"""Temporarily route an explicitly requested Hub download through a proxy."""

from contextlib import contextmanager
import os
import socket
from urllib.parse import quote, urlsplit

import requests


DEFAULT_PROXY = 'http://survival-mode.klg.yp-c.yandex.net:6953'
_PROXY_KEYS = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY')
_BYPASS_KEYS = ('no_proxy', 'NO_PROXY')


@contextmanager
def download_proxy():
    """Use CDLM_DOWNLOAD_PROXY for this download, then restore the environment."""
    proxy = os.environ.get('CDLM_DOWNLOAD_PROXY', '')
    if not proxy or os.environ.get('HF_HUB_OFFLINE', '').strip().upper() in (
            '1', 'ON', 'YES', 'TRUE'):
        yield
        return
    try:
        parsed = urlsplit(proxy)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or any(character.isspace() for character in parsed.hostname)
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('invalid proxy URL')
        parsed.port  # Validate malformed/out-of-range ports without logging the URL.
    except ValueError:
        raise RuntimeError('CDLM_DOWNLOAD_PROXY must be an HTTP(S) proxy URL '
                           'with a hostname and no path, query or fragment.') from None
    proxy = proxy.rstrip('/')
    if proxy == DEFAULT_PROXY:
        try:
            # Registration is direct: inherited proxies, netrc and HF tokens
            # must not be sent to the registration endpoint.
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(
                    proxy + '/register/' + quote(socket.gethostname(), safe=''),
                    timeout=(10, 30), allow_redirects=False)
                status = response.status_code
        except Exception:
            raise RuntimeError('Cannot reach the Nirvana download proxy. Set a working '
                               'CDLM_DOWNLOAD_PROXY or package the checkpoint before launch.') from None
        if not 200 <= status < 300:
            raise RuntimeError(f'Nirvana download proxy registration failed, HTTP {status}. '
                               'Set a working CDLM_DOWNLOAD_PROXY or package the checkpoint.')
        print('[cdlm-assets] Job host registered on the Nirvana download proxy.', flush=True)
    previous = {key: os.environ.get(key) for key in _PROXY_KEYS + _BYPASS_KEYS}
    try:
        os.environ.update(dict.fromkeys(_PROXY_KEYS, proxy))
        os.environ.update(dict.fromkeys(_BYPASS_KEYS, ''))
        print('[cdlm-assets] Using the configured proxy for checkpoint download.', flush=True)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
