#!/bin/bash
# Conditional TinyGSM / GSM8K benchmark evaluation for the IDLM-MDLM student.
#
# Sampling + grading follow the S-FLM TinyGSM evaluation protocol
# (https://github.com/jdeschena/s-flm): the question is pinned as a clean
# prefix, the model generates a `simple_math_problem()` program, and the
# program is executed in a restricted sandbox and compared against the
# `#### N` gold answer of the 1319-example GSM8K test set.
#
# Usage:
#   CKPT_PATH=outputs/.../checkpoints/step=00050000.ckpt \
#   DEVICES=4 bash scripts/eval_tinygsm_idlm_mdlm.sh
#
#   # confidence-ranked commits on the same fixed step budget
#   REMASKING=low_confidence_static ... bash scripts/eval_tinygsm_idlm_mdlm.sh
#
#   # adaptive: the threshold is the budget, so THRESHOLDS is what gets swept
#   REMASKING=low_confidence_dynamic ... bash scripts/eval_tinygsm_idlm_mdlm.sh

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the student .ckpt to evaluate}"
DEVICES="${DEVICES:-1}"
NUM_NODES="${NUM_NODES:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
STEPS_LIST="${STEPS_LIST:-4 8 16 32}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
# Which masked positions each reverse step commits: 'random' (upstream),
# 'low_confidence_static', or 'low_confidence_dynamic'. See configs/config.yaml.
REMASKING="${REMASKING:-random}"
# Token draw. 1.0 is the upstream categorical sample; 0.0 is greedy, which is
# what yrOPDLM decodes with. Confidence is read off the raw p_x0 either way.
TEMPERATURE="${TEMPERATURE:-1.0}"
# The budget for low_confidence_dynamic, swept in place of STEPS_LIST. Same grid
# yrOPDLM sweeps (train/final_eval.py).
THRESHOLDS="${THRESHOLDS:-0.85 0.9 0.95}"
# Where the noise-removal step lives. "sflm" (default) folds it into the last
# loop iteration so NFE == steps, matching scripts/sample/tinygsm/mdlm.sh in
# jdeschena/s-flm -- the conditional protocol appendix E.2 of the IDLM paper
# says the authors followed. "idlm" appends a separate denoising call instead,
# so NFE == steps + 1.
STEP_CONVENTION="${STEP_CONVENTION:-sflm}"

# Keep each strategy in its own tree by default: the dirs are named after the
# budget, so a confidence run would otherwise overwrite the baseline's
# results.json with a number that is not comparable to it.
_default_root="${REPO_ROOT}/eval_runs/tinygsm/idlm-mdlm"
if [[ "${REMASKING}" != random ]]; then
  _default_root="${_default_root}-${REMASKING}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${_default_root}}"

cd "${REPO_ROOT}"

# Checkpoints written by a distillation run also carry the frozen `teacher` and
# the `fake_model` critic, so the module must be built with is_distill=True for
# the state_dict to match; the released checkpoints ship the student `backbone`
# alone and need False. Probe the checkpoint instead of making the caller guess.
if [[ -z "${IS_DISTILL:-}" ]]; then
  IS_DISTILL="$(python scripts/probe_is_distill.py "${CKPT_PATH}")"
  echo "Detected adversarial_distill.is_distill=${IS_DISTILL} from ${CKPT_PATH}"
fi

# $1: output subdirectory, the rest: extra hydra overrides carrying the budget.
run_eval() {
  local OUTPUT_DIR="${OUTPUT_ROOT}/$1"; shift
  mkdir -p "${OUTPUT_DIR}"

  python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=gsm8k-test \
    model=small \
    model.length=512 \
    algo=mdlm \
    adversarial_distill.is_distill="${IS_DISTILL}" \
    sampling.predictor=ancestral_cache \
    sampling.noise_removal=ancestral \
    sampling.step_convention="${STEP_CONVENTION}" \
    sampling.remasking_strategy="${REMASKING}" \
    sampling.temperature="${TEMPERATURE}" \
    "$@" \
    loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.devices="${DEVICES}" \
    trainer.num_nodes="${NUM_NODES}" \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    gsm8k.bootstrap_size="${BOOTSTRAP}" \
    hydra.run.dir="${OUTPUT_DIR}"
}

# Sweep whatever is actually the budget for the chosen strategy: the step count
# for the fixed-schedule rules, the threshold for the adaptive one (which
# ignores sampling.steps outright, so it is left off the command line).
if [[ "${REMASKING}" == low_confidence_dynamic ]]; then
  for thr in ${THRESHOLDS}; do
    echo "=== TinyGSM eval: dynamic threshold ${thr} ==="
    run_eval "dyn_${thr}" sampling.dynamic_threshold="${thr}"
  done
else
  for steps in ${STEPS_LIST}; do
    echo "=== TinyGSM eval: ${steps} sampling steps (${REMASKING}) ==="
    run_eval "steps_${steps}" sampling.steps="${steps}"
  done
fi

echo "=== summary ==="
python - "${OUTPUT_ROOT}" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for pattern in ('steps_*', 'dyn_*'):
  for path in sorted(glob.glob(os.path.join(root, pattern, 'results.json'))):
    with open(path) as f:
      r = json.load(f)
    # Label each row by its own budget knob: the step count for the fixed
    # schedules, the threshold for the adaptive one. avg_nfe is the forwards
    # actually spent, and the only axis the two are comparable on.
    if r.get('remasking_strategy') == 'low_confidence_dynamic':
      budget = f"thr={r.get('dynamic_threshold')}"
    else:
      budget = f"steps={r.get('sampling_steps')}"
    rows.append((r.get('avg_nfe'), budget, r['accuracy'],
                 r['accuracy_ci_lo'], r['accuracy_ci_hi'],
                 r['num_correct'], r['num_total']))
rows.sort(key=lambda x: (x[0] is None, x[0]))
print(f"{'NFE':>7}  {'budget':>12}  {'acc %':>7}  {'95% CI':>18}  correct/total")
for nfe, budget, acc, lo, hi, c, n in rows:
  nfe_s = '-' if nfe is None else f'{nfe:.1f}'
  print(f'{nfe_s:>7}  {budget:>12}  {acc*100:>7.2f}  '
        f'[{lo*100:>6.2f}, {hi*100:>6.2f}]  {c}/{n}')
PY
