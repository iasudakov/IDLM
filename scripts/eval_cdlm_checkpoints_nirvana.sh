#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CDLM_CODE_DIR:-}" ]]; then
  cdlm_eval_code_dir="$CDLM_CODE_DIR"
elif [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  cdlm_eval_code_dir="$SOURCE_CODE_PATH/code"
else
  cdlm_eval_code_dir="$(dirname -- "${BASH_SOURCE[0]}")/.."
fi
cd -- "$cdlm_eval_code_dir"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec "${PYTHON:-python}" -u -m cdlm_checkpoint_eval "$@"
