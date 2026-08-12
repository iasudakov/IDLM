#!/usr/bin/env bash
# Download a pretrained CIFAR-10 *masked* diffusion teacher from the DUO
# release (https://huggingface.co/jdeschena/duo2-cifar10).
#
# The `mdlm/` family is masked / absorbing-state diffusion -- the process
# `algo.MDLM_DMMD` distills -- so no teacher pretraining is needed. (The
# `duo/` family in the same repo is uniform-state diffusion; IDLM has no
# uniform D-MMD loss, so those checkpoints are not usable here.)
#
# Variants: mdlm/{cosine,log_linear}_time_{cond,uncond}/{100000,500000,1000000,1500000}
# The default below matches `noise=cosine` + `algo.time_conditioning=False`.
#
#   bash scripts/download_cifar10_teacher.sh
#   VARIANT=mdlm/log_linear_time_uncond STEPS=1000000 bash scripts/download_cifar10_teacher.sh
set -euo pipefail

# The repo convention is an already-active conda env (`conda activate mdlm`).
# Override PYTHON to point at one explicitly.
PYTHON="${PYTHON:-python}"

REPO_ID="${REPO_ID:-jdeschena/duo2-cifar10}"
VARIANT="${VARIANT:-mdlm/cosine_time_uncond}"
STEPS="${STEPS:-1500000}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/downloads}"

mkdir -p "${OUT_DIR}"

"${PYTHON}" - "$REPO_ID" "$VARIANT/$STEPS.safetensors" "$OUT_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, out_dir = sys.argv[1:4]
path = hf_hub_download(repo_id=repo_id, filename=filename,
                       local_dir=out_dir)
print(path)
PY

echo
echo "Teacher at: ${OUT_DIR}/${VARIANT}/${STEPS}.safetensors"
echo "Pass it as: adversarial_distill.teacher_ckpt_path=${OUT_DIR}/${VARIANT}/${STEPS}.safetensors"
