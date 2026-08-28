#!/bin/bash
# Sweep trained checkpoints over sampling temperatures and step budgets, and
# publish the resulting frontiers (one curve per temperature) to underdeep.
#
# Usage:
#   CKPT=/path/to/step=00020000.ckpt GPUS=0 bash scripts/eval_pareto_temperature.sh
#
#   CKPT=""  measures the teacher instead -- the baseline every student curve is
#            read against. Save its JSON and pass it back as REFERENCE to draw
#            it dashed under a student's curves.
#
# Several checkpoints on the same axes, compared per step budget. Point
# CKPT_DIR at the folder and name the steps -- the padding is filled in, so
# 2000 is step=00002000.ckpt:
#   CKPT_DIR=/a/checkpoints CKPTS='[2000,4000,14000]' bash scripts/...
#   CKPT_DIR=/a/checkpoints                           bash scripts/...  # all
#
# Or spell the paths out, which is what CKPT_DIR-less entries are:
#   CKPTS='/a/step=00004000.ckpt,/a/step=00014000.ckpt' bash scripts/...
#   CKPT_GLOB='/a/checkpoints/step=*.ckpt'              bash scripts/...
#
#   The three may be combined, glob first; the glob is sorted, which for
#   zero-padded names is training order. An EMPTY entry (a leading or doubled
#   comma) is the teacher, so `CKPTS='[,2000,14000]'` puts the baseline in the
#   same figure rather than under it as a dashed reference.
#
# NB: no `set -u` -- conda's MKL activate hook references an unbound variable.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT-}"
CKPT_DIR="${CKPT_DIR-}"
CKPT_DIR="${CKPT_DIR%/}"  # so `${CKPT_DIR}/step=...` never doubles the slash

# --- which checkpoints --------------------------------------------------- #

# Resolve one CKPTS entry against CKPT_DIR. Sets RESOLVED instead of echoing it:
# a bad entry has to stop the run, and an `exit` inside `$(...)` would only ever
# kill the subshell.
RESOLVED=''
resolve_ckpt() {
  local entry="$1"
  case "${entry}" in
    '' | */*)
      # Empty is the teacher; anything with a slash is already a path.
      RESOLVED="${entry}" ;;
    *[!0-9]*)
      # A bare filename, e.g. `step=00014000.ckpt`.
      RESOLVED="${CKPT_DIR:+${CKPT_DIR}/}${entry}" ;;
    *)
      # A bare step number -- the short form CKPT_DIR exists for. Padded to the
      # 8 digits PeriodicStepCheckpoint writes; `10#` forces base ten, or printf
      # would read a leading zero as octal and refuse `08000`.
      if [ -z "${CKPT_DIR}" ]; then
        echo "CKPTS entry '${entry}' is a bare step number; set CKPT_DIR" >&2
        exit 1
      fi
      RESOLVED="$(printf '%s/step=%08d.ckpt' "${CKPT_DIR}" "$((10#${entry}))")" ;;
  esac
}

CKPT_LIST=()
# Unquoted on purpose: the word splitting IS the glob expansion, and quoting
# would hand Hydra the pattern itself.
for f in ${CKPT_GLOB-}; do
  [ -e "${f}" ] && CKPT_LIST+=("${f}")
done
if [ -n "${CKPTS-}" ]; then
  # Brackets and spaces tolerated, so CKPTS='[2000, 4000, 7000]' reads the same
  # as CKPTS=2000,4000,7000 -- the bracketed form is the shape TEMPS and STEPS
  # already take, and having one of them differ is a trap.
  ENTRIES="${CKPTS#[}"
  ENTRIES="${ENTRIES%]}"
  IFS=',' read -r -a EXPLICIT <<<"${ENTRIES}"
  for entry in "${EXPLICIT[@]}"; do
    entry="${entry#"${entry%%[![:space:]]*}"}"  # ltrim
    entry="${entry%"${entry##*[![:space:]]}"}"  # rtrim
    resolve_ckpt "${entry}"
    CKPT_LIST+=("${RESOLVED}")
  done
elif [ -z "${CKPT_GLOB-}" ] && [ -n "${CKPT_DIR}" ] && [ -z "${CKPT}" ]; then
  # A folder and nothing else: every checkpoint in it, in training order -- the
  # names are zero-padded, so sorted is chronological.
  for f in "${CKPT_DIR}"/step=*.ckpt; do
    [ -e "${f}" ] && CKPT_LIST+=("${f}")
  done
fi

# Fail now rather than an hour in, three checkpoints down the list.
for f in "${CKPT_LIST[@]}"; do
  if [ -n "${f}" ] && [ ! -e "${f}" ]; then
    echo "no such checkpoint: ${f}" >&2
    exit 1
  fi
done

if [ "${#CKPT_LIST[@]}" -gt 0 ]; then
  # Hydra list syntax, every element single-quoted: checkpoint names contain
  # `=`, which the override grammar otherwise reads as a key/value separator.
  PATHS_ARG="eval.student_ckpt_paths=["
  for i in "${!CKPT_LIST[@]}"; do
    [ "${i}" -gt 0 ] && PATHS_ARG+=","
    PATHS_ARG+="'${CKPT_LIST[$i]}'"
  done
  PATHS_ARG+="]"
  CKPT_ARGS=("${PATHS_ARG}")
  # Naming anchor: the first REAL checkpoint. A leading empty entry is the
  # teacher, and `dirname ''` would name the run after the working directory.
  FIRST_CKPT=""
  for f in "${CKPT_LIST[@]}"; do
    [ -n "${f}" ] && { FIRST_CKPT="${f}"; break; }
  done
else
  CKPT_ARGS=("eval.student_ckpt_path='${CKPT}'")
  FIRST_CKPT="${CKPT}"
fi

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
NUM_CKPT="${#CKPT_LIST[@]}"
[ "${NUM_CKPT}" -eq 0 ] && NUM_CKPT=1
echo "sweep: ${NUM_CKPT} checkpoint(s)" \
     "temps=${TEMPS} steps=${STEPS} seeds=${SEEDS}" \
     "batches=${BATCHES} x ${BATCH_SIZE} seqs" \
     "= $((BATCHES * BATCH_SIZE)) per point over ${NPROC} gpu(s)"
if [ "${#CKPT_LIST[@]}" -gt 0 ]; then
  for f in "${CKPT_LIST[@]}"; do echo "  - ${f:-<teacher, as built>}"; done
else
  echo "  - ${CKPT:-<teacher, as built>}"
fi

# --- where the results go ----------------------------------------------- #

# NOT the training run's directory by default. `logger.force_new_run=true`
# rewrites `underdeep_metadata.pt` in `save_dir`, which is the handle the
# training run resumes itself by -- pointing this at a live run's directory
# would cost that run its tracker continuity on the next restart.
if [ -n "${FIRST_CKPT}" ]; then
  DEFAULT_OUT="$(cd "$(dirname "${FIRST_CKPT}")" && pwd)/pareto_temp_eval"
else
  DEFAULT_OUT="${REPO_ROOT}/eval_runs/pareto_temp_teacher"
fi
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT}}"
mkdir -p "${OUT_DIR}"

if [ "${#CKPT_LIST[@]}" -gt 1 ]; then
  DEFAULT_NAME="pareto-temp ${#CKPT_LIST[@]} ckpts $(basename "$(dirname "${FIRST_CKPT}")")"
else
  DEFAULT_NAME="pareto-temp $(basename "${FIRST_CKPT:-teacher}")"
fi
RUN_NAME="${RUN_NAME:-${DEFAULT_NAME}}"
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
  "${CKPT_ARGS[@]}" \
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
