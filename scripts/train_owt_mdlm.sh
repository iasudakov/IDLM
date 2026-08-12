#!/usr/bin/env bash
# Train MDLM on OpenWebText (duo's `train_owt_mdlm.sh` recipe) with underdeep
# tracking and restartable, yrDiT-style checkpointing.
#
# Restartable by design. The run directory is pinned (not hydra's default
# timestamped path), and everything a restart needs keys off it: `last.ckpt`
# under `checkpoints/`, the underdeep run id in `underdeep_metadata.pt`, and --
# on Nirvana -- the snapshot transfers. So re-running this command *continues*
# the run; pass a different RUN_NAME (or OUT_DIR) to start a fresh one.
#
# The tokenized OWT cache is read from DATA_CACHE_DIR; it must hold
# `openwebtext-train_train_bs${MODEL_LENGTH}_wrapped.dat` and
# `openwebtext-valid_validation_bs${MODEL_LENGTH}_wrapped.dat`, else duo
# re-tokenizes OWT into that directory (slow, ~70 GB).
#
#   bash scripts/train_owt_mdlm.sh
#   GPUS=0,1 RUN_NAME=owt-mdlm-debug bash scripts/train_owt_mdlm.sh
#   bash scripts/train_owt_mdlm.sh trainer.max_steps=200000   # extra overrides
#
# The original slurm header is gone: this box has no scheduler, and lightning's
# ddp strategy launches one process per visible GPU itself. To go back to slurm,
# wrap the final command in `srun`.
#
# `sampling.predictor=ancestral_cache` is not a preference: `algo.MDLM`
# asserts it at construction, so the plain `ancestral` default cannot train.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No CUDA devices visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}')." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-owt-mdlm}"
OUT_DIR="${OUT_DIR:-$(pwd)/outputs/${RUN_NAME}}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-/mnt/data/owt}"
UNDERDEEP_EXPERIMENT="${UNDERDEEP_EXPERIMENT:-generative-models/cifar10_mdlm}"

MODEL_LENGTH="${MODEL_LENGTH:-1024}"
# Per-GPU micro-batch; the global batch is held at GLOBAL_BATCH by gradient
# accumulation, so it does not move when GPUS does.
BATCH_SIZE="${BATCH_SIZE:-2}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
# `${eval:...}` in the config would otherwise set this to the core count (255
# here), i.e. 255 loader processes per rank.
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-1000000}"
# Rolling `last.ckpt` cadence, and the rarer step-named archives kept forever.
SAVE_EVERY="${SAVE_EVERY:-500}"
ARCHIVE_EVERY="${ARCHIVE_EVERY:-50000}"

ACCUM=$(( GLOBAL_BATCH / (BATCH_SIZE * NUM_GPUS) ))
# Validation cadence, in optimizer steps. `trainer.val_check_interval` counts
# *micro-batches*, so it has to be multiplied by the accumulation factor to mean
# what it looks like it means: the config's bare 5000 would, at BATCH_SIZE=2
# (64 accumulation steps for a 512 global batch), validate every 78 optimizer
# steps -- and each validation walks the whole OWT validation split and draws two
# 1000-step ancestral samples, so that alone would dominate the wall clock.
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-5000}"
VAL_CHECK_INTERVAL=$(( VAL_EVERY_STEPS * ACCUM ))
# How often `trainer/loss` reaches the tracker, in optimizer steps. Unlike
# val_check_interval, lightning's `log_every_n_steps` already counts optimizer
# steps -- but at BATCH_SIZE=2 one of those is ~4 s of wall clock, so the
# config's 100 leaves the loss curve empty for the first ~7 minutes of a run and
# coarse thereafter. 10 still averages a 5120-sequence window per point.
LOG_EVERY="${LOG_EVERY:-10}"
if [[ $(( BATCH_SIZE * NUM_GPUS * ACCUM )) -ne "${GLOBAL_BATCH}" ]]; then
  # dataloader.get_dataloaders asserts this exactly, so fail here with a message
  # instead of inside the first batch.
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} is not BATCH_SIZE=${BATCH_SIZE} x NUM_GPUS=${NUM_GPUS} x an integer." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "[launch] ${NUM_GPUS} GPUs  global_batch=${GLOBAL_BATCH} (${BATCH_SIZE}/gpu x ${ACCUM} accum)"
echo "[launch] steps=${MAX_STEPS}  val every ${VAL_EVERY_STEPS} steps (${VAL_CHECK_INTERVAL} batches)"
echo "[launch] out=${OUT_DIR}  data=${DATA_CACHE_DIR}"

exec "${PYTHON}" -u -m main \
  mode=train \
  algo=mdlm \
  model=small \
  data=openwebtext-split \
  data.cache_dir="${DATA_CACHE_DIR}" \
  model.length="${MODEL_LENGTH}" \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="${BATCH_SIZE}" \
  loader.global_batch_size="${GLOBAL_BATCH}" \
  loader.num_workers="${NUM_WORKERS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.log_every_n_steps="${LOG_EVERY}" \
  eval.compute_generative_perplexity=False \
  sampling.predictor=ancestral_cache \
  logger=underdeep \
  logger.name="${RUN_NAME}" \
  logger.experiment="${UNDERDEEP_EXPERIMENT}" \
  checkpointing.save_every_n_steps="${SAVE_EVERY}" \
  checkpointing.archive_every_n_steps="${ARCHIVE_EVERY}" \
  hydra.run.dir="${OUT_DIR}" \
  '~wandb' \
  "$@"
