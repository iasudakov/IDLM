#!/usr/bin/env bash
# Frozen SDLLM MDLM-1.7B features; the student and policy gradient stay GPT-2.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/owt_mmd_sdllm}"
export FEATURE_SOURCE=sdllm_mdlm
export FEATURE_LAYER="${FEATURE_LAYER:-3}"
EXTRA=()
if [[ -n "${SDLLM_MODEL_DIR:-}" ]]; then
  EXTRA+=("adversarial_distill.mmd_sdllm_path=$SDLLM_MODEL_DIR"
          adversarial_distill.mmd_sdllm_local_files_only=True)
fi
# A native-token feature map scores the whole retokenized sequence. Preserve
# evaluation sample count when changing the GPU count (128 texts per seed).
TRAIN_DEVICES="${TRAIN_DEVICES:-8}"
if ! [[ "$TRAIN_DEVICES" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'TRAIN_DEVICES must be 1, 2, 4, 8 or 16.' >&2
  exit 2
fi
export TRAIN_DEVICES
SAMPLE_BATCHES=$((128 / (8 * TRAIN_DEVICES)))
exec bash "$REPO_DIR/scripts/train_mmd_owt_sdtt_large.sh" \
  adversarial_distill.mmd_score_positions=all \
  adversarial_distill.mmd_noise_level=0.0 \
  adversarial_distill.mmd_reward_type=feature \
  adversarial_distill.mmd_rbf_alpha=null \
  "sampling.num_sample_batches=$SAMPLE_BATCHES" \
  "${EXTRA[@]}" "$@"
