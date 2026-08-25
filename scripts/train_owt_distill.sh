#!/bin/bash
# Portable launcher for the OpenWebText adversarial-distillation run.
#
# Everything machine-specific is an environment variable, so moving to another
# host means setting OWT_CACHE (and usually GPUS), not editing overrides.
#
# Usage:
#   OWT_CACHE=/path/to/owt GPUS=0,1,2,3 bash scripts/train_owt_distill.sh
#
# NB: no `set -u` -- conda's MKL activate hook references an unbound variable.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# --- machine-specific ------------------------------------------------- #

# Directory holding the pre-tokenized OWT splits. `dataloader.get_dataset`
# looks for `<dataset>_<mode>_bs<model.length>_wrapped.dat` under here and
# SILENTLY REGENERATES from the raw HF dataset (~39GB download plus hours of
# tokenization) when the name does not match -- it does not error. So the
# names are checked up front instead.
OWT_CACHE="${OWT_CACHE:?set OWT_CACHE to the directory holding the OWT .dat splits}"
for split in openwebtext-train_train openwebtext-valid_validation; do
  if [ ! -d "${OWT_CACHE}/${split}_bs${LENGTH:-1024}_wrapped.dat" ]; then
    echo "error: missing ${OWT_CACHE}/${split}_bs${LENGTH:-1024}_wrapped.dat" >&2
    echo "       (a run would re-tokenize OWT from scratch rather than fail)" >&2
    exit 1
  fi
done

# Honor an externally-set CUDA_VISIBLE_DEVICES; GPUS overrides it.
if [ -n "${GPUS}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
NUM_GPUS="$(python -c "import os;v=os.environ.get('CUDA_VISIBLE_DEVICES');print(len(v.split(',')) if v else __import__('torch').cuda.device_count())")"

# Needed to unpickle the checkpoints under this torch version.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --- batch size ------------------------------------------------------- #

# The recipe pins accumulate_grad_batches, so the effective batch is
# devices * batch_size * accum -- i.e. it CHANGES WITH THE GPU COUNT unless
# accumulation compensates. Held at EFFECTIVE_BATCH so a 4-GPU host and an
# 8-GPU host train the same model.
BATCH_SIZE="${BATCH_SIZE:-8}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-64}"
ACCUM="$(python -c "import math;print(max(1, math.ceil(${EFFECTIVE_BATCH} / (${NUM_GPUS} * ${BATCH_SIZE}))))")"
echo "devices=${NUM_GPUS} batch_size=${BATCH_SIZE} accum=${ACCUM}" \
     "-> effective $(( NUM_GPUS * BATCH_SIZE * ACCUM ))"

# --- tracker ---------------------------------------------------------- #

RUN_NAME="${RUN_NAME:-idlm-mdlm}"
EXPERIMENT="${EXPERIMENT:-generative-models/idlm}"

python -u -m main \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="${BATCH_SIZE}" \
  loader.num_workers=1 \
  model=small \
  data=openwebtext-split \
  data.cache_dir="${OWT_CACHE}" \
  strategy.find_unused_parameters=True \
  adversarial_distill.is_distill=True \
  algo=mdlm \
  algo.backbone='hf_dit' \
  eval.checkpoint_path='kuleshov-group/mdlm-owt' \
  model.length="${LENGTH:-1024}" \
  eval.compute_generative_perplexity=True \
  trainer.accumulate_grad_batches="${ACCUM}" \
  trainer.gradient_clip_val=null \
  trainer.precision=32 \
  training.loss_precision=float32 \
  sampling.steps=32 \
  sampling.predictor='ancestral_cache' \
  optim.lr="${LR:-1e-6}" \
  adversarial_distill.fake_frequency=1 \
  trainer.val_check_interval=1500 \
  trainer.limit_val_batches=0.01 \
  logger.experiment="${EXPERIMENT}" \
  logger.name="${RUN_NAME}" \
  logger.force_new_run=true \
  "$@"
