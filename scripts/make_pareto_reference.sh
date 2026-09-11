#!/bin/bash
# Measure a fixed accuracy-vs-NFE frontier for one checkpoint, to draw as the
# dashed reference line on the in-training Pareto figure (eval.gsm8k_pareto_*).
#
# Run once per reference checkpoint, not during training. Each threshold is a
# full 1319-question sweep, so this costs ~6 standalone evals; they are spread
# across GPUS round-robin.
#
# Usage:
#   CKPT_PATH=checkpoints/tinygsm/mdlm_notok.ckpt GPUS="2 4" \
#   bash scripts/make_pareto_reference.sh
#
# NB: no `set -u` -- conda's MKL activate hook references an unbound variable.
set -eo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the reference .ckpt}"
OUT="${OUT:-${REPO_ROOT}/data/pareto_reference_mdlm_teacher.json}"
LABEL="${LABEL:-Released MDLM teacher}"
THRESHOLDS="${THRESHOLDS:-0.5 0.6 0.7 0.8 0.9 0.95}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEPARATOR="${SEPARATOR:-literal}"
WORK="${WORK:-${REPO_ROOT}/eval_runs/tinygsm/pareto-reference}"

cd "${REPO_ROOT}"
mkdir -p "${WORK}" "$(dirname "${OUT}")"

# Activate first, like the other eval scripts here assume. Checked up front
# because the failure otherwise lands inside six backgrounded jobs at once.
if ! python -c 'import omegaconf, torch' 2>/dev/null; then
  echo "error: run this with the project env active (conda activate mdlm)" >&2
  exit 1
fi

# Fresh, so a stale threshold from an earlier partial run cannot be merged in
# as if this sweep had produced it.
rm -f "${WORK}"/thr_*.json

read -r -a GPU_ARR <<< "${GPUS}"
i=0
pids=()
for thr in ${THRESHOLDS}; do
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  # Round-robin onto the available GPUs, waiting whenever a full wave is in
  # flight so we never oversubscribe one card.
  if [[ $i -ge ${#GPU_ARR[@]} ]]; then
    wait "${pids[$((i - ${#GPU_ARR[@]}))]}"
  fi
  echo "=== threshold ${thr} on GPU ${gpu} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/eval_gsm8k_checkpoint.py \
    "${CKPT_PATH}" \
    --remasking low_confidence_dynamic --threshold "${thr}" \
    --temperature 0.0 --batch-size "${BATCH_SIZE}" \
    --separator "${SEPARATOR}" \
    --out "${WORK}/thr_${thr}.json" > "${WORK}/thr_${thr}.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done
wait

python - "${WORK}" "${OUT}" "${CKPT_PATH}" "${LABEL}" "${THRESHOLDS}" <<'PY'
import glob, json, os, sys
work, out, ckpt, label, wanted = sys.argv[1:6]
wanted = [float(t) for t in wanted.split()]
paths = sorted(glob.glob(os.path.join(work, 'thr_*.json')))
# Refuse to write a partial reference: it would silently become a baseline that
# is missing exactly the thresholds whose runs died.
if len(paths) != len(wanted):
  sys.exit(f'expected {len(wanted)} thresholds, found {len(paths)} results in '
           f'{work} -- check the per-threshold logs; no reference written')
points = []
for path in paths:
  with open(path) as f:
    r = json.load(f)
  points.append({'threshold': r['dynamic_threshold'],
                 'nfe': r['avg_nfe_per_seq'],
                 'nfe_per_batch': r['avg_nfe'],
                 'accuracy': r['accuracy'],
                 'num_total': r['num_total']})
points.sort(key=lambda p: p['nfe'])
with open(out, 'w') as f:
  json.dump({'checkpoint': ckpt, 'label': label, 'points': points}, f, indent=2)
print(f'\nwrote {out}')
for p in points:
  print(f"  thr {p['threshold']:<5} {p['nfe']:6.1f} NFE/seq  "
        f"{p['accuracy'] * 100:5.2f}%  ({p['num_total']} questions)")
PY
