#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/generation_cdlm_tinygsm.sh CHECKPOINT [Hydra overrides...]' >&2
  exit 2
fi
cdlm_checkpoint_path="$1"
shift
if [[ "$cdlm_checkpoint_path" != hf://* ]]; then
  cdlm_checkpoint_path="$(realpath -- "$cdlm_checkpoint_path")"
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -u -m main \
  +experiment=cdlm_tinygsm \
  mode=sample_eval \
  trainer.devices=1 \
  "eval.checkpoint_path=$cdlm_checkpoint_path" \
  eval.generated_samples_path=cdlm_tinygsm_eval.json \
  "$@"
