#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MMD_CODE_DIR:-}" ]]; then
  mmd_code_dir="$MMD_CODE_DIR"
elif [[ -n "${CDLM_CODE_DIR:-}" ]]; then
  mmd_code_dir="$CDLM_CODE_DIR"
elif [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  mmd_code_dir="$SOURCE_CODE_PATH/code"
else
  mmd_code_dir="$(dirname -- "${BASH_SOURCE[0]}")/.."
fi
cd -- "$mmd_code_dir"
mmd_code_dir="$PWD"

# Disable driver-level TF32 too, before Python initializes CUDA.
export NVIDIA_TF32_OVERRIDE=0

mmd_hydra_string() {
  local mmd_value="$1"
  mmd_value="${mmd_value//\\/\\\\}"
  mmd_value="${mmd_value//\"/\\\"}"
  printf '"%s"' "$mmd_value"
}

# Original MDLM -> MMD: no CDLM state or trajectory generation is required.
# 4 GPUs x 32 examples x accumulation 4 = 512 examples per optimizer update.
mmd_overrides=(
  +experiment=mmd_tinygsm_fp32
  "data.cache_dir=$(mmd_hydra_string "$mmd_code_dir/cache_dir")"
  "model_weights_dir=$(mmd_hydra_string "$mmd_code_dir/cache_dir/tinygsm_mmd_offline_bundle")"
  "training.finetune_path=$(mmd_hydra_string "$mmd_code_dir/checkpoints/tinygsm/mdlm_notok.ckpt")"
  "checkpointing.save_dir=$(mmd_hydra_string "$mmd_code_dir/mmd_tinygsm_fp32")"
  "hydra.run.dir=$(mmd_hydra_string "$mmd_code_dir/mmd_tinygsm_fp32")"
  logger=underdeep
  logger.experiment=generative-models/tinygsm
  "$@"
)

mmd_asset_file="$(mktemp)"
trap 'rm -f -- "$mmd_asset_file"' EXIT
CDLM_DOWNLOAD_PROXY="${CDLM_DOWNLOAD_PROXY-http://survival-mode.klg.yp-c.yandex.net:6953}" \
"${PYTHON:-python}" -u -m mmd_preflight --output "$mmd_asset_file" "${mmd_overrides[@]}"
mapfile -d '' -t mmd_asset_overrides < "$mmd_asset_file"
rm -f -- "$mmd_asset_file"
trap - EXIT

exec "${PYTHON:-python}" -u -m main "${mmd_overrides[@]}" "${mmd_asset_overrides[@]}" mode=train
