#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  idlm_eval_code_dir="$SOURCE_CODE_PATH/code"
else
  idlm_eval_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd -- "$idlm_eval_code_dir"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec "${PYTHON:-python}" -u -m idlm_state_eval "$@"
