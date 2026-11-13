#!/usr/bin/env bash
# Nirvana entry point for MDLM Figures 4, 6 and 14. One invocation per job.
set +x
set -euo pipefail

export PATH="/conda/envs/main/bin:$PATH"
PAPER_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

exec python -u - "$PAPER_REPO_ROOT" "$@" <<'PY_PAPER_BOOTSTRAP'
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler

repo = Path(sys.argv[1])
arguments = sys.argv[2:]
if arguments and not any(a in ('-h', '--help') for a in arguments):
    if os.environ.get('PAPER_SKIP_INSTALL') != '1':
        package_env = os.environ.copy()
        proxy = os.environ.get('PAPER_PROXY', '').rstrip('/')
        if proxy:
            opener = build_opener(ProxyHandler({}))
            with opener.open(proxy + '/register/' + quote(socket.gethostname(), safe=''),
                             timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError('Proxy registration failed')
            for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
                package_env[key] = proxy
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'transformers==4.55.4',
            'ipdb==0.13.13', 'hydra-core==1.3.2', 'lightning==2.2.1',
            'pytorch-lightning==2.2.1',
        ], env=package_env, check=True)

os.execv(sys.executable, [sys.executable, '-u',
    str(repo / 'scripts/paper_recompute/launch.py'), '--repo', str(repo), *arguments])
PY_PAPER_BOOTSTRAP
