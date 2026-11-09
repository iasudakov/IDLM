#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/generation_sdtt_tinygsm.sh CHECKPOINT [Hydra overrides...]' >&2
  exit 2
fi
sdtt_checkpoint_path="$1"
shift
if [[ "$sdtt_checkpoint_path" != hf://* ]]; then
  sdtt_checkpoint_path="$(realpath -- "$sdtt_checkpoint_path")"
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -u -m main \
  +experiment=sdtt_tinygsm \
  mode=sample_eval \
  eval.gsm8k_pareto_thresholds=null \
  trainer.devices=1 \
  "eval.checkpoint_path=$sdtt_checkpoint_path" \
  eval.generated_samples_path=sdtt_tinygsm_eval.json \
  "$@"
