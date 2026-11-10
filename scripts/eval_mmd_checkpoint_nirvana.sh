#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MMD_CODE_DIR:-}" ]]; then
  mmd_eval_code_dir="$MMD_CODE_DIR"
elif [[ -n "${CDLM_CODE_DIR:-}" ]]; then
  mmd_eval_code_dir="$CDLM_CODE_DIR"
elif [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  mmd_eval_code_dir="$SOURCE_CODE_PATH/code"
else
  mmd_eval_code_dir="$(dirname -- "${BASH_SOURCE[0]}")/.."
fi
cd -- "$mmd_eval_code_dir"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# The same asset resolver/proxy as MMD -> CDLM training, scoped to checkpoint
# preparation. Keep an explicit empty value (direct access) or offline setting.
# The coordinator disables Hub/dataset networking only for the GPU workers.
export CDLM_DOWNLOAD_PROXY="${CDLM_DOWNLOAD_PROXY-http://survival-mode.klg.yp-c.yandex.net:6953}"
exec "${PYTHON:-python}" -u -m mmd_checkpoint_eval "$@"
