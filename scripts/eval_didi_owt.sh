#!/usr/bin/env bash
# The official evaluation preset: guided,16 steps,20 samples/batch,2 batches.
set -eo pipefail
DIDI_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIDI_REPO_ROOT"
: "${CHECKPOINT:?set CHECKPOINT to a trained DiDi native .ckpt}"
: "${SAVE_DIR:?set SAVE_DIR to a new evaluation output directory}"
if [ ! -f "$CHECKPOINT" ]; then
  printf 'DiDi checkpoint does not exist: %s\n' "$CHECKPOINT" >&2
  exit 1
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"
SAVE_DIR="$(realpath -m "$SAVE_DIR")"
if [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
exec python -m didi_repro \
  mode=sample_eval \
  "eval.checkpoint_path=$CHECKPOINT" \
  "checkpointing.save_dir=$SAVE_DIR" "hydra.run.dir=$SAVE_DIR" \
  "eval.generated_samples_path=$SAVE_DIR/samples.json" \
  loader.batch_size=20 loader.eval_batch_size=20 \
  sampling.steps=16 sampling.num_sample_batches=2 \
  sampling.predictor=guided sampling.noise_removal=ancestral \
  eval.generate_samples=true \
  "+underdeep.experiment=${EXPERIMENT:-generative-models/owt}" \
  "+underdeep.name=${RUN_NAME:-didi-instruct-eval-guided16}" \
  "$@"
