#!/usr/bin/env bash
# Original DiDi-Instruct training defaults; locations and tracker only differ.
# Usage: TEACHER_CHECKPOINT=/path/mdlm.ckpt OWT_CACHE=/path/cache \
#        SAVE_DIR=/path/owt_didi GPUS=0,1 bash scripts/train_didi_owt.sh
set -eo pipefail

DIDI_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIDI_REPO_ROOT"
: "${TEACHER_CHECKPOINT:?set TEACHER_CHECKPOINT to the original native MDLM .ckpt}"
: "${OWT_CACHE:?set OWT_CACHE to the original tokenized OpenWebText cache}"
: "${SAVE_DIR:?set SAVE_DIR to a new output directory}"
if [ ! -f "$TEACHER_CHECKPOINT" ]; then
  printf 'Teacher checkpoint does not exist: %s\n' "$TEACHER_CHECKPOINT" >&2
  exit 1
fi
for didi_split in openwebtext-train_train openwebtext-valid_validation; do
  if [ ! -d "$OWT_CACHE/${didi_split}_bs1024_wrapped.dat" ]; then
    printf 'Missing upstream OWT split: %s\n' "$OWT_CACHE/${didi_split}_bs1024_wrapped.dat" >&2
    exit 1
  fi
done

# Hydra changes its working directory; preserve input/output paths explicitly.
TEACHER_CHECKPOINT="$(realpath "$TEACHER_CHECKPOINT")"
OWT_CACHE="$(realpath "$OWT_CACHE")"
SAVE_DIR="$(realpath -m "$SAVE_DIR")"
if [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
DIDI_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$DIDI_GPU_COUNT" -lt 1 ]; then
  printf '%s\n' 'DiDi reproduction requires CUDA GPUs and the original environment.' >&2
  exit 1
fi

# Upstream script: one example/GPU, G=8 generated candidates/example,
# global_batch_size=GPU count, no accumulation, validate every1000 batches.
# No MMD learning-rate/batch/precision/EMA settings are imported.
DIDI_COMMAND=(python -m torch.distributed.run --standalone
  --nproc_per_node="$DIDI_GPU_COUNT" --module didi_repro
  "training.finetune_path=$TEACHER_CHECKPOINT"
  "data.cache_dir=$OWT_CACHE"
  "checkpointing.save_dir=$SAVE_DIR" "hydra.run.dir=$SAVE_DIR"
  loader.batch_size=1 "loader.global_batch_size=$DIDI_GPU_COUNT"
  trainer.val_check_interval=1000
  "+underdeep.experiment=${EXPERIMENT:-generative-models/owt}"
  "+underdeep.name=${RUN_NAME:-didi-instruct-owt}"
)
if [ "${DIDI_DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${DIDI_COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${DIDI_COMMAND[@]}" "$@"
