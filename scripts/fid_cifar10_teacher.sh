#!/usr/bin/env bash
# Stage 2 + 3: the downloaded teacher's own FID curve.
#
# This is the baseline the distilled student has to beat, measured on *our*
# teacher rather than copied from the paper (different codebase, and these
# weights were trained class-conditionally with dropout and are sampled here
# through the null class), so it is the honest comparison for Table 1.
#
# The 1024-step point is also the port test: if the pixel layout, vocabulary,
# noise schedule or sampler were wrong, FID would be in the hundreds instead of
# the single digits.
#
#   bash scripts/fid_cifar10_teacher.sh                 # full sweep, 50k samples
#   STEPS_LIST="1024" NUM_SAMPLES=10000 bash scripts/fid_cifar10_teacher.sh
set -euo pipefail

# The repo convention is an already-active conda env (`conda activate mdlm`).
# Override PYTHON to point at one explicitly.
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,3,4,5}"
NUM_DEVICES="$(awk -F',' '{print NF}' <<< "${GPUS}")"
TEACHER="${TEACHER:-$(pwd)/downloads/mdlm/cosine_time_uncond/1500000.safetensors}"
STEPS_LIST="${STEPS_LIST:-1024 128 64 32 16}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-125}"
OUT_DIR="${OUT_DIR:-$(pwd)/outputs/cifar10/teacher_fid}"

mkdir -p "${OUT_DIR}"

for steps in ${STEPS_LIST}; do
  echo "=== teacher FID @ ${steps} steps ==="
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -u -m main \
    mode=fid_eval \
    data=cifar10 \
    model=unet \
    algo=mdlm_dmmd \
    algo.backbone=unet \
    noise=cosine \
    adversarial_distill.is_distill=False \
    adversarial_distill.teacher_ckpt_path="${TEACHER}" \
    loader.batch_size="${BATCH_SIZE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.global_batch_size="$((BATCH_SIZE * NUM_DEVICES))" \
    trainer.devices="${NUM_DEVICES}" \
    loader.num_workers="${NUM_WORKERS:-8}" \
    sampling.num_sample_log=64 \
    sampling.steps="${steps}" \
    sampling.predictor=ancestral_cache \
    eval.fid_num_samples="${NUM_SAMPLES}" \
    eval.fid_report_path="${OUT_DIR}/fid_steps${steps}.json" \
    eval.image_grid_path="${OUT_DIR}/samples_steps${steps}.png" \
    hydra.run.dir="${OUT_DIR}/steps${steps}"
done

echo
echo "=== summary ==="
"${PYTHON}" - "${OUT_DIR}" <<'PY'
import glob, json, os, sys
rows = []
for path in glob.glob(os.path.join(sys.argv[1], 'fid_steps*.json')):
    with open(path) as f:
        d = json.load(f)
    rows.append((d['sampling_steps'], d['fid'], d['num_generated']))
print(f"{'steps':>6} {'FID':>10} {'n':>8}")
for steps, fid, n in sorted(rows):
    print(f'{steps:>6} {fid:>10.3f} {n:>8}')
PY
