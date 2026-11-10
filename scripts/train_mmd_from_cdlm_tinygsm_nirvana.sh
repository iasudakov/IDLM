#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CDLM_CODE_DIR:-}" ]]; then
  mmd_code_dir="$CDLM_CODE_DIR"
elif [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  mmd_code_dir="$SOURCE_CODE_PATH/code"
else
  mmd_code_dir="$(dirname -- "${BASH_SOURCE[0]}")/.."
fi
cd -- "$mmd_code_dir"
mmd_code_dir="$PWD"

# These are literal Hydra strings, never evaluated as shell code.
mmd_hydra_string() {
  local mmd_value="$1"
  mmd_value="${mmd_value//\\/\\\\}"
  mmd_value="${mmd_value//\"/\\\"}"
  printf '"%s"' "$mmd_value"
}

# Automatic optimization: 32 x 4 GPUs x 4 accumulation = 512 examples/update.
# One training process is started; Lightning launches its DDP workers.
# No CDLM trajectory preparation is needed for representation-space MMD.
mmd_overrides=(
  +experiment=mmd_from_cdlm_tinygsm
  "data.cache_dir=$(mmd_hydra_string "$mmd_code_dir/cache_dir")"
  "model_weights_dir=$(mmd_hydra_string "$mmd_code_dir/cache_dir/tinygsm_mmd_offline_bundle")"
  "training.finetune_path=$(mmd_hydra_string "$mmd_code_dir/checkpoints/tinygsm/mdlm_notok.ckpt")"
  "checkpointing.save_dir=$(mmd_hydra_string "$mmd_code_dir/mmd_from_cdlm_tinygsm")"
  "hydra.run.dir=$(mmd_hydra_string "$mmd_code_dir/mmd_from_cdlm_tinygsm")"
  logger=underdeep
  logger.experiment=generative-models/tinygsm
)
if [[ -n "${CDLM_SNAPSHOT_ROOT:-}" ]]; then
  mmd_overrides+=("mmd.cdlm_snapshot_root=$(mmd_hydra_string "$CDLM_SNAPSHOT_ROOT")")
fi
if [[ -n "${CDLM_SOURCE_NAMESPACE:-}" ]]; then
  mmd_overrides+=("mmd.cdlm_snapshot_namespace=$(mmd_hydra_string "$CDLM_SOURCE_NAMESPACE")")
fi
mmd_overrides+=("$@")

mmd_asset_file="$(mktemp)"
trap 'rm -f -- "$mmd_asset_file"' EXIT
CDLM_DOWNLOAD_PROXY="${CDLM_DOWNLOAD_PROXY-http://survival-mode.klg.yp-c.yandex.net:6953}" \
"${PYTHON:-python}" -u -m mmd_preflight --output "$mmd_asset_file" "${mmd_overrides[@]}"
mapfile -d '' -t mmd_asset_overrides < "$mmd_asset_file"
rm -f -- "$mmd_asset_file"
trap - EXIT

exec "${PYTHON:-python}" -u -m main "${mmd_overrides[@]}" "${mmd_asset_overrides[@]}" mode=train
