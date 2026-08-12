#!/usr/bin/env bash
# Stage 6: D-MMD distillation of the downloaded CIFAR-10 masked teacher into a
# few-step generator (D-MMD arXiv:2603.20155 Sec. 6.1, Table 1 "masked").
#
# Student, frozen teacher and trainable auxiliary critic all start from
# `adversarial_distill.teacher_ckpt_path` (loaded before the critics are
# deep-copied), which is the standard D-MMD/DMD initialisation. That path takes
# either the downloaded `.safetensors` or an `ema_<step>.safetensors` written by
# `train_cifar10_teacher.sh` -- the same format, so an own-trained teacher drops
# straight in.
#
# `~callbacks.checkpoint_monitor` is required, not optional: that callback
# monitors `val/gen_ppl`, which the image path never logs, and Lightning raises
# MisconfigurationException on the first validation when the monitored key is
# missing. Without it the run dies at step 2500.
#
#   bash scripts/train_cifar10_dmmd.sh
#   NOISE_DIM=0 RUN_NAME=cifar10_dmmd_nonoise bash scripts/train_cifar10_dmmd.sh
set -euo pipefail

# The repo convention is an already-active conda env (`conda activate mdlm`).
# Override PYTHON to point at one explicitly.
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,3,4,5}"
NUM_DEVICES="$(awk -F',' '{print NF}' <<< "${GPUS}")"
TEACHER="${TEACHER:-$(pwd)/downloads/mdlm/cosine_time_uncond/1500000.safetensors}"
RUN_NAME="${RUN_NAME:-cifar10_dmmd}"
# Three UNets (student + teacher + auxiliary) at 3072 tokens: keep the
# per-device batch well below DUO's 32.
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
LR="${LR:-1e-5}"
# Table 6: the 2D Gaussian noise pyramid is what separates FID 151 from 22.3 at
# 4 steps for masked image D-MMD. Set 0 to A/B it.
#
# NOISE_SCALE defaults to the repo default 0.0 = zero-init: the student starts
# as an exact copy of the teacher's function and grows its noise usage only if
# the objective pays for it. Watch `noise/proj_norm` and `noise/snr` -- if
# proj_norm stays flat the generator is ignoring the noise and the run is
# equivalent to NOISE_DIM=0. To force the noise on from the start instead, use
# e.g. NOISE_SCALE=0.5 NOISE_WARMUP=1000 (a nonzero scale is required whenever
# a warmup is set).
NOISE_DIM="${NOISE_DIM:-8}"
OUT_DIR="${OUT_DIR:-$(pwd)/outputs/cifar10/${RUN_NAME}}"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -u -m main \
  data=cifar10 \
  model=unet \
  algo=mdlm_dmmd \
  algo.backbone=unet \
  noise=cosine \
  adversarial_distill.is_distill=True \
  adversarial_distill.critic_type=diffusion \
  adversarial_distill.teacher_ckpt_path="${TEACHER}" \
  adversarial_distill.noise_dim="${NOISE_DIM}" \
  adversarial_distill.noise_scale="${NOISE_SCALE:-0.0}" \
  adversarial_distill.noise_warmup_steps="${NOISE_WARMUP:-0}" \
  adversarial_distill.dmmd_mask_sampling=posterior \
  adversarial_distill.top_p=0.85 \
  adversarial_distill.fake_teacher_weight=0.1 \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="${BATCH_SIZE}" \
  loader.global_batch_size="$((BATCH_SIZE * NUM_DEVICES * ACCUM))" \
  trainer.devices="${NUM_DEVICES}" \
  trainer.accumulate_grad_batches="${ACCUM}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval=2500 \
  trainer.gradient_clip_val=null \
  '~callbacks.checkpoint_monitor' \
  strategy.find_unused_parameters=True \
  optim.lr="${LR}" \
  sampling.steps=8 \
  sampling.predictor=ancestral_cache \
  sampling.num_sample_batches=2 \
  eval.compute_fid=True \
  eval.compute_generative_perplexity=False \
  eval.compute_gm=False \
  eval.pareto_curve=False \
  eval.image_grid_path="${OUT_DIR}/samples.png" \
  checkpointing.save_dir="${OUT_DIR}" \
  hydra.run.dir="${OUT_DIR}"
