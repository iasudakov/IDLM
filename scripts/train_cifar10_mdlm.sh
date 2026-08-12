#!/usr/bin/env bash
# Train the CIFAR-10 masked-diffusion (MDLM) teacher of D-MMD Sec. 6.1 --
# the model whose FID row Table 1 reports (6.4 at 1024 NFE).
#
# The recipe is DUO's CIFAR-10 MDLM one (the paper publishes no hyperparameter
# table) with the batch and step count rescaled from 128 x 1.5M to 512 x 375k:
# the same ~192M images seen, at a batch that fills 8 A100s. See the docstring
# of train_cifar10_mdlm.py.
#
# Restartable by design. Every ckpt interval the run directory is pushed to the
# Nirvana snapshot path; at startup it is pulled back, so a preempted job
# resumes from its own latest.pt with optimizer, EMA, dataloader epoch and the
# underdeep run id intact. Off Nirvana those transfers are no-ops and the same
# command just trains.
#
#   bash scripts/train_cifar10_mdlm.sh
#   MAX_STEPS=50000 RUN_NAME=short bash scripts/train_cifar10_mdlm.sh
#   GPUS=0,1,2,3 bash scripts/train_cifar10_mdlm.sh
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

RUN_NAME="${RUN_NAME:-cifar10_mdlm_teacher}"
OUT_DIR="${OUT_DIR:-$(pwd)/outputs/${RUN_NAME}}"

# Global batch 512. BATCH_SIZE is per GPU, so the global batch is held fixed
# regardless of how many devices are visible.
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
BATCH_SIZE="${BATCH_SIZE:-$((GLOBAL_BATCH / NUM_GPUS))}"
if [[ $((BATCH_SIZE * NUM_GPUS)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by NUM_GPUS=${NUM_GPUS}." >&2
  echo "Set BATCH_SIZE explicitly, or pick a divisible GLOBAL_BATCH." >&2
  exit 1
fi

MAX_STEPS="${MAX_STEPS:-375000}"
LR="${LR:-4e-4}"
WARMUP="${WARMUP:-5000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
EMA_CKPT_EVERY="${EMA_CKPT_EVERY:-50000}"
UNDERDEEP_EXPERIMENT="${UNDERDEEP_EXPERIMENT:-generative-models/cifar10_mdlm}"

mkdir -p "${OUT_DIR}"
echo "[launch] ${NUM_GPUS} GPUs  global_batch=${GLOBAL_BATCH} (${BATCH_SIZE}/gpu)"
echo "[launch] steps=${MAX_STEPS}  lr=${LR}  out=${OUT_DIR}"

exec torchrun --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT:-29511}" \
  train_cifar10_mdlm.py \
  --out "${OUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --underdeep-experiment "${UNDERDEEP_EXPERIMENT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --warmup "${WARMUP}" \
  --sample-every "${SAMPLE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --ckpt-every "${CKPT_EVERY}" \
  --ema-ckpt-every "${EMA_CKPT_EVERY}" \
  "$@"
