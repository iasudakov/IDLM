#!/bin/bash
# Fetch S-FLM's TinyGSM autoregressive model, for use as the MMD feature map
# (adversarial_distill.mmd_feature_source=ar).
#
# Why this model: it is causal, trained 250k steps on TinyGSM itself, and uses
# the student's exact SmolLM-135M tokenizer -- so its features describe what a
# plausible TinyGSM *program* looks like, which is what the reward is meant to
# measure. Its embedding table carries 49154 rows, covering the pad and mask ids
# the dataloader appends, so no id remapping is needed either.
#
# The strip step is mandatory, not a convenience. The released checkpoint embeds
# a pickled `tokenizers.Tokenizer` written by a newer stack than
# requirements.txt pins, so a plain torch.load dies with
#   "Error while attempting to unpickle Tokenizer: data did not match any
#    variant of untagged enum ModelWrapper"
# before reading a single weight. See scripts/strip_ckpt_tokenizer.py.
#
# NB: no `set -u` -- conda's MKL activate hook references an unbound variable.
set -eo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DEST:-${REPO_ROOT}/checkpoints}"
RAW="${DEST}/tinygsm/ar.ckpt"
OUT="${OUT:-${DEST}/tinygsm/ar_notok.ckpt}"
KEEP_RAW="${KEEP_RAW:-0}"   # 1 to keep the 2.6 GB original after stripping

cd "${REPO_ROOT}"

if ! python -c 'import torch' 2>/dev/null; then
  echo "error: run this with the project env active (conda activate mdlm)" >&2
  exit 1
fi

if [[ -f "${OUT}" ]]; then
  echo "already present: ${OUT}"
  exit 0
fi

if [[ ! -f "${RAW}" ]]; then
  echo "=== downloading jdeschena/s-flm tinygsm/ar.ckpt (2.6 GB) ==="
  # `hf` is the current name; `huggingface-cli` still works but warns.
  if command -v hf >/dev/null 2>&1; then
    hf download jdeschena/s-flm tinygsm/ar.ckpt --local-dir "${DEST}"
  else
    huggingface-cli download jdeschena/s-flm tinygsm/ar.ckpt --local-dir "${DEST}"
  fi
fi

echo "=== stripping the pickled tokenizer ==="
python scripts/strip_ckpt_tokenizer.py "${RAW}" "${OUT}"

if [[ "${KEEP_RAW}" != "1" ]]; then
  rm -f "${RAW}"
  echo "removed the unstripped original (KEEP_RAW=1 to keep it)"
fi

echo
echo "ready: ${OUT}"
echo "use it with: adversarial_distill.mmd_feature_source=ar"
