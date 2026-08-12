#!/usr/bin/env bash
# Train the CIFAR-10 **masked diffusion teacher** from scratch, i.e. the model
# D-MMD Sec. 6.1 distills. Target: FID 6.4 at 1024 denoising steps (Table 1).
#
# Why train one at all, when `scripts/download_cifar10_teacher.sh` exists: the
# released DUO checkpoint is **class-conditional** (trained with
# `class_dropout_p=0.1`), while the paper's teacher is trained
# **unconditionally** -- "we train diffusion models to generate unconditional
# images" (Sec. 6.1). This script trains the paper's setting.
#
# On hyperparameters: the paper specifies the *task* precisely (32x32x3 CIFAR-10
# as 3072 tokens over {0,...,255}, masked diffusion, unconditional) but gives no
# architecture or optimizer table -- there is no hyperparameter appendix. The
# recipe below is therefore DUO's CIFAR-10 MDLM recipe
# (`~/duo/scripts/train_cifar10_mdlm_cosine.sh` plus the settings reported in
# the DUO Chapter II paper, arXiv:2602.21185: 35M-parameter UNet, 1.5M steps,
# global batch 128, lr 2e-4, dropout 0.1, random horizontal flips), which is the
# same backbone on the same data. The single deliberate deviation from DUO is
# `data.num_classes=null` -> no `cond_map` at all, matching the paper.
#
# Restartable by design (Nirvana): every `SNAPSHOT_EVERY` steps the run
# directory is pushed to the Nirvana snapshot path, and at startup it is pulled
# back, so a preempted job resumes from its own `last.ckpt` -- optimizer state,
# EMA, dataloader position and the underdeep run id included. Off Nirvana those
# transfers are no-ops and the same command trains normally.
#
#   bash scripts/train_cifar10_teacher.sh
#   MAX_STEPS=200000 RUN_NAME=teacher_short bash scripts/train_cifar10_teacher.sh
#   GPUS=0,1,2,3 bash scripts/train_cifar10_teacher.sh   # pin devices
set -euo pipefail

# The repo convention is an already-active conda env (`conda activate mdlm`).
# Override PYTHON to point at one explicitly.
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

# Devices: honour an explicit GPUS list, otherwise take whatever the machine (or
# the Nirvana job) already exposes. Nirvana sets CUDA_VISIBLE_DEVICES itself, so
# leaving GPUS unset is the right thing there.
if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
NUM_DEVICES="${NUM_DEVICES:-$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')}"
if [[ "${NUM_DEVICES}" -lt 1 ]]; then
  echo "No CUDA devices visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-unset}')." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-cifar10_teacher}"
# DUO's global batch. Per-device batch is derived so the global batch holds
# regardless of how many GPUs are used.
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
BATCH_SIZE="${BATCH_SIZE:-$((GLOBAL_BATCH / NUM_DEVICES))}"
if [[ $((BATCH_SIZE * NUM_DEVICES)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by NUM_DEVICES=${NUM_DEVICES}." >&2
  echo "Set BATCH_SIZE explicitly, or pick a divisible GLOBAL_BATCH." >&2
  exit 1
fi
MAX_STEPS="${MAX_STEPS:-1500000}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-5000}"
OUT_DIR="${OUT_DIR:-$(pwd)/outputs/cifar10/${RUN_NAME}}"

# Resume state, archival weights, and how much a preemption costs.
#   CKPT_EVERY    -- how often `checkpoints/last.ckpt` (full resume state:
#                    weights + EMA + optimizer + dataloader position, ~570 MB)
#                    is rewritten in place.
#   EMA_EVERY     -- archival `checkpoints/ema_<step>.safetensors` (~143 MB,
#                    never overwritten). These are the intermediate teachers,
#                    in the exact format `adversarial_distill.teacher_ckpt_path`
#                    loads.
#   SNAPSHOT_EVERY-- how often the run dir goes to the Nirvana snapshot path.
#                    A preemption costs at most this many steps. Keep it a
#                    multiple of CKPT_EVERY.
CKPT_EVERY="${CKPT_EVERY:-2500}"
EMA_EVERY="${EMA_EVERY:-100000}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10000}"

mkdir -p "${OUT_DIR}"

# Two deviations from the repo defaults, both required for an *unattended* image
# run rather than optional tuning:
#
#   ~callbacks.checkpoint_monitor -- it monitors `val/gen_ppl`, a text-only
#     metric. The image path logs val/bpd, val/nll, val/ppl, val/fid,
#     val/inception_score, val/sample_entropy; Lightning raises
#     MisconfigurationException on the first validation when the monitored key
#     is absent, so with this callback in place the run dies at step
#     VAL_EVERY. Dropping it also avoids a second ~570 MB `best.ckpt` in every
#     snapshot.
#   save_top_k=0 -- `save_last: True` still rewrites `last.ckpt`, but the
#     per-interval `epoch=..-step=...ckpt` files are not kept. At 570 MB each and
#     a checkpoint every CKPT_EVERY steps, keeping them all fills the disk (and
#     the snapshot) long before 1.5M steps.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "${PYTHON}" -u -m main \
  data=cifar10 \
  data.num_classes=null \
  model=unet \
  algo=mdlm \
  algo.backbone=unet \
  noise=cosine \
  adversarial_distill.is_distill=False \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="${BATCH_SIZE}" \
  loader.global_batch_size="${GLOBAL_BATCH}" \
  loader.num_workers=8 \
  trainer.devices="${NUM_DEVICES}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${VAL_EVERY:-10000}" \
  +trainer.check_val_every_n_epoch=null \
  optim.lr="${LR}" \
  lr_scheduler.num_warmup_steps="${WARMUP}" \
  training.ema=0.9999 \
  sampling.steps=128 \
  sampling.predictor=ancestral_cache \
  sampling.num_sample_batches=1 \
  eval.generate_samples=True \
  eval.compute_fid=True \
  eval.fid_num_samples_val=2000 \
  eval.compute_generative_perplexity=False \
  eval.compute_gm=False \
  eval.pareto_curve=False \
  eval.image_grid_every_n_steps="${IMAGE_EVERY:-1000}" \
  eval.image_grid_num_images="${IMAGE_NUM:-64}" \
  eval.fid_paper_every_n_steps="${FID_PAPER_EVERY:-25000}" \
  eval.fid_paper_num_samples="${FID_PAPER_SAMPLES:-10000}" \
  "eval.fid_paper_steps=[${FID_PAPER_STEPS:-128}]" \
  eval.image_grid_path="${OUT_DIR}/samples.png" \
  logger.name="${RUN_NAME}" \
  '~callbacks.checkpoint_monitor' \
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  checkpointing.save_dir="${OUT_DIR}" \
  checkpointing.ema_export_every_n_steps="${EMA_EVERY}" \
  checkpointing.nirvana_snapshot_every_n_steps="${SNAPSHOT_EVERY}" \
  hydra.run.dir="${OUT_DIR}" \
  "$@"
