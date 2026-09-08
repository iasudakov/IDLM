#!/usr/bin/env bash
# Pinned authors' dedicated inference release; no training/runtime/logger patches.
set -euo pipefail
DIDI_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for didi_override in "$@"; do
  # Keep model/noise/guidance/batching fixed to the inference release. The
  # checkpoint is validated against precisely that architecture and preset.
  case "$didi_override" in
    seed=*|sampling.steps=*|eval.perplexity_batch_size=*) ;;
    *) printf 'Unsupported inference-control override: %s. Allowed: seed, sampling.steps, eval.perplexity_batch_size.\n' "$didi_override" >&2
       exit 1 ;;
  esac
done
export DIDI_AUTHORS_SOURCE=inference
# Original Lightning loading uses pickle. Only evaluate checkpoints you trust.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
exec bash "$DIDI_REPO_ROOT/scripts/eval_didi_authors.sh" "$@"
