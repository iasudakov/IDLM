#!/usr/bin/env bash
set +x
set -euo pipefail
export PATH="/conda/envs/main/bin:$PATH"
CDLM_EXTENSION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python -u - "$CDLM_EXTENSION_ROOT" "$@" <<'PY_CDLM_EXTENSION'
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler

repo = Path(sys.argv[1])
args = sys.argv[2:]
if not args or args[0] not in ('confidence', 'train-mmd'):
    raise SystemExit('Usage: run_cdlm_extension.sh {confidence|train-mmd} [arguments]')
if os.environ.get('CDLM_SKIP_INSTALL') != '1':
    package_env = os.environ.copy()
    proxy = os.environ.get('CDLM_INSTALL_PROXY', '').rstrip('/')
    if proxy:
        with build_opener(ProxyHandler({})).open(
                proxy + '/register/' + quote(socket.gethostname(), safe=''), timeout=30) as response:
            if response.status != 200:
                raise RuntimeError('Proxy registration failed')
        for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
            package_env[key] = proxy
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers==4.55.4',
                    'ipdb==0.13.13', 'hydra-core==1.3.2', 'lightning==2.2.1',
                    'pytorch-lightning==2.2.1'], env=package_env, check=True)
os.chdir(repo)
os.environ['PYTHONPATH'] = os.pathsep.join([str(repo), str(repo.parent), os.environ.get('PYTHONPATH', '')])
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
if args[0] == 'train-mmd':
    cmd = [sys.executable, '-u', '-m', 'cdlm_from_mmd_state', *args[1:]]
else:
    inputs = Path(os.environ.get('INPUT_PATH', repo))
    def existing(paths):
        for p in paths:
            if p.exists():
                return p.resolve()
        raise FileNotFoundError('Missing asset: ' + ', '.join(map(str, paths)))
    bundle = existing([inputs / 'tinygsm_mmd_offline_bundle', repo / 'cache_dir/tinygsm_mmd_offline_bundle'])
    data = existing([inputs / 'nirvana_checkpoints/data/gsm8k_test.json', repo / 'data/gsm8k_test.json'])
    import json
    cmd = [sys.executable, '-u', '-m', 'cdlm_checkpoint_eval',
           'eval.gsm8k_checkpoint_confidence=True', 'eval.gsm8k_checkpoint_ancestral=False',
           'eval.gsm8k_checkpoint_normalizations=[fp32]',
           'model_weights_dir=' + json.dumps(str(bundle)),
           'eval.gsm8k_data_path=' + json.dumps(str(data)),
           '--no-restore-results', *args[1:]]
os.execv(sys.executable, cmd)
PY_CDLM_EXTENSION
