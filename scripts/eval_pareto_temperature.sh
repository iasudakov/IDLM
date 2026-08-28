#!/bin/bash
# Sweep one trained checkpoint over sampling temperatures and step budgets, and
# publish the resulting frontiers (one curve per temperature) to underdeep.
#
# Usage:
#   CKPT=/path/to/step=00020000.ckpt GPUS=0 bash scripts/eval_pareto_temperature.sh
#
#   CKPT=""  measures the teacher instead -- the baseline every student curve is
#            read against. Save its JSON and pass it back as REFERENCE to draw
#            it dashed under a student's curves.
#
# NB: no `set -u` -- conda's MKL activate hook references an unbound variable.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT-}"

# GPUS=2 runs one process; GPUS=0,1,2,3,4,5,6,7 runs one rank per GPU under
# torchrun. The ranks SPLIT each point's samples rather than each drawing a
# full set, and the two metrics all-reduce their (sum, weight) pairs before
# dividing -- so the sample count, and the number, are the same either way.
if [ -n "${GPUS}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"

if [ "${NPROC}" -gt 1 ]; then
  # A fixed port collides with any other torchrun on this box; pick one per run.
  LAUNCHER=(torchrun --nproc_per_node="${NPROC}" --nnodes=1
            --master_port="${MASTER_PORT:-$((29500 + RANDOM % 2000))}" -m main)
else
  LAUNCHER=(python -u -m main)
fi

# Needed to unpickle the training checkpoints under this torch version.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --- what to sweep ------------------------------------------------------ #

TEMPS="${TEMPS:-[0.7,0.8,0.9,1.0,2.0]}"
STEPS="${STEPS:-[8,16,32]}"
# Seeded repeats per point, averaged, spread drawn as error bars. Temperatures
# share their seeds, so the curves are a paired comparison -- worth more here
# than raw sample count, because adjacent temperatures differ by less than
# unpaired sampling noise.
SEEDS="${SEEDS:-3}"
BATCHES="${BATCHES:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Sampling cost is seeds * batches * sum(steps) forward passes per temperature,
# divided by the number of ranks.
echo "sweep: temps=${TEMPS} steps=${STEPS} seeds=${SEEDS}" \
     "batches=${BATCHES} x ${BATCH_SIZE} seqs" \
     "= $((BATCHES * BATCH_SIZE)) per point over ${NPROC} gpu(s)"

# --- where the results go ----------------------------------------------- #

# NOT the training run's directory by default. `logger.force_new_run=true`
# rewrites `underdeep_metadata.pt` in `save_dir`, which is the handle the
# training run resumes itself by -- pointing this at a live run's directory
# would cost that run its tracker continuity on the next restart.
if [ -n "${CKPT}" ]; then
  DEFAULT_OUT="$(cd "$(dirname "${CKPT}")" && pwd)/pareto_temp_eval"
else
  DEFAULT_OUT="${REPO_ROOT}/eval_runs/pareto_temp_teacher"
fi
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT}}"
mkdir -p "${OUT_DIR}"

RUN_NAME="${RUN_NAME:-pareto-temp $(basename "${CKPT:-teacher}")}"
EXPERIMENT="${EXPERIMENT:-generative-models/owt}"
# Optional JSON frontier drawn dashed underneath, e.g. the teacher's sweep.
REFERENCE="${REFERENCE:-null}"
# `null` must reach Hydra bare to mean "no reference"; a path must be quoted,
# because checkpoint names contain `=`.
if [ "${REFERENCE}" = "null" ]; then
  REFERENCE_ARG="eval.owt_pareto_reference=null"
else
  REFERENCE_ARG="eval.owt_pareto_reference='${REFERENCE}'"
fi

"${LAUNCHER[@]}" \
  mode=pareto_eval \
  data=openwebtext-split \
  model=small \
  model.length="${LENGTH:-1024}" \
  algo=mdlm \
  algo.backbone='hf_dit' \
  eval.checkpoint_path="${TEACHER:-kuleshov-group/mdlm-owt}" \
  "eval.student_ckpt_path='${CKPT}'" \
  adversarial_distill.is_distill=False \
  eval.compute_generative_perplexity=True \
  eval.perplexity_batch_size="${BATCH_SIZE}" \
  "eval.pareto_temperatures=${TEMPS}" \
  "eval.owt_pareto_steps=${STEPS}" \
  eval.owt_pareto_seeds="${SEEDS}" \
  eval.owt_pareto_num_batches="${BATCHES}" \
  "${REFERENCE_ARG}" \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="${BATCH_SIZE}" \
  loader.num_workers=1 \
  sampling.predictor='ancestral_cache' \
  sampling.noise_removal="${NOISE_REMOVAL:-ancestral}" \
  "checkpointing.save_dir='${OUT_DIR}'" \
  checkpointing.snapshot=false \
  "hydra.run.dir='${OUT_DIR}'" \
  logger.experiment="${EXPERIMENT}" \
  "logger.name='${RUN_NAME}'" \
  logger.force_new_run=true \
  "$@"
