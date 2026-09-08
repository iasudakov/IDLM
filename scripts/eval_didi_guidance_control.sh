#!/usr/bin/env bash
# Explicit guidance on/off control through the unchanged inference sampler.
set -euo pipefail
DIDI_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${DIDI_GUIDANCE_CONTROL:-}" in
  on) didi_guidance_start=30.0; didi_guidance_end=40.0; didi_candidates=4 ;;
  off) didi_guidance_start=0.0; didi_guidance_end=0.0; didi_candidates=1 ;;
  *) printf 'Set DIDI_GUIDANCE_CONTROL to on or off.\n' >&2; exit 1 ;;
esac
if [[ "${DIDI_INFERENCE_PRECISION:-upstream}" != upstream ]]; then
  printf 'Guidance controls require original precision: DIDI_INFERENCE_PRECISION=upstream.\n' >&2
  exit 1
fi
for didi_override in "$@"; do
  case "$didi_override" in
    seed=*|sampling.steps=*|eval.perplexity_batch_size=*) ;;
    *) printf 'Unsupported guidance-control override: %s. Allowed: seed, sampling.steps, eval.perplexity_batch_size.\n' "$didi_override" >&2
       exit 1 ;;
  esac
done
export DIDI_AUTHORS_SOURCE=inference
export DIDI_INFERENCE_PRECISION=upstream
# Original Lightning loading uses pickle. Only evaluate checkpoints you trust.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
# Keep predictor=guided, the timestep grid, and terminal denoising from the base
# preset. Off disables both gradient tilting and reranking, not the sampler.
exec bash "$DIDI_REPO_ROOT/scripts/eval_didi_authors.sh" \
  "algo.guidance_scale_start=$didi_guidance_start" \
  "algo.guidance_scale_end=$didi_guidance_end" \
  "algo.num_candidates=$didi_candidates" \
  algo.rerank_steps_ratio=0.5 \
  "$@"
