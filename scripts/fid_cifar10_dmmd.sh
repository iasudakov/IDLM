#!/usr/bin/env bash
# Student FID sweep, in the shape of D-MMD Table 1 (masked row): 4/8/16/32/64
# steps. Compare against `scripts/fid_cifar10_teacher.sh`, not against the
# paper's absolute numbers -- our teacher is DUO's checkpoint, not the paper's.
#
#   CKPT=outputs/cifar10/cifar10_dmmd/checkpoints/last.ckpt \
#     bash scripts/fid_cifar10_dmmd.sh
set -euo pipefail

# The repo convention is an already-active conda env (`conda activate mdlm`).
# Override PYTHON to point at one explicitly.
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,3,4,5}"
NUM_DEVICES="$(awk -F',' '{print NF}' <<< "${GPUS}")"
CKPT="${CKPT:?set CKPT=/path/to/student.ckpt}"
STEPS_LIST="${STEPS_LIST:-4 8 16 32 64}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-125}"
NOISE_DIM="${NOISE_DIM:-8}"
OUT_DIR="${OUT_DIR:-$(pwd)/outputs/cifar10/student_fid}"

mkdir -p "${OUT_DIR}"

for steps in ${STEPS_LIST}; do
  echo "=== student FID @ ${steps} steps ==="
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -u -m main \
    mode=fid_eval \
    data=cifar10 \
    model=unet \
    algo=mdlm_dmmd \
    algo.backbone=unet \
    noise=cosine \
    adversarial_distill.is_distill=True \
    adversarial_distill.critic_type=diffusion \
    adversarial_distill.noise_dim="${NOISE_DIM}" \
    eval.checkpoint_path="${CKPT}" \
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
