#!/usr/bin/env bash
# The OWT D-MMD run ("posterior shared noised ... top_p 1.0 noise_scale 0.0"),
# ported to CIFAR-10 with the *time-conditioned* released teacher
# `jdeschena/duo2-cifar10 :: mdlm/cosine_time_cond/1500000.safetensors`.
#
# Requires branch `cifar10-dmmd` (data=cifar10, model=unet, noise=cosine,
# algo.backbone=unet and the image FID / noise-pyramid plumbing all live there).
#
#   bash scripts/download_cifar10_teacher.sh   # VARIANT=mdlm/cosine_time_cond
#   bash scripts/train_cifar10_dmmd_timecond.sh
#
# Knob-for-knob against the OWT command, this changes only:
#
#   data=openwebtext-split          -> data=cifar10          (3072 pixel tokens, vocab 257)
#   model=small                     -> model=unet            (35.76M D3PM UNet; model.length comes from data.length)
#   algo.backbone=hf_dit            -> algo.backbone=unet
#   eval.checkpoint_path=<hub id>   -> adversarial_distill.teacher_ckpt_path=<safetensors>
#   (log-linear noise)              -> noise=cosine          (what mdlm/cosine_* was trained with)
#   (algo.time_conditioning=False)  -> True                  (the *_time_cond variant feeds sigma)
#   eval.compute_generative_perplexity=True -> False, +compute_fid   (no GPT-2 for pixels)
#   noise_dim=768 (DiT hidden)      -> 128 (UNet residual width; see NOISE_DIM)
#   +'~callbacks.checkpoint_monitor'                          (it monitors val/gen_ppl, never logged here)
#   +callbacks.checkpoint_every_n_steps.save_top_k=0          (1.3 GB a pop; the disk has ~46 GB)
#   +trainer.check_val_every_n_epoch=null                     (see below)
#
# That last one is required, not tuning. CIFAR-10's train split is 50 000 images,
# so one epoch is only 50000 / (devices * batch_size) batches -- 1042 at 6 GPUs
# and batch 8. Lightning then rejects `val_check_interval=1500` outright:
#   ValueError: `val_check_interval` (1500) must be less than or equal to the
#   number of the training batches (1042).
# `check_val_every_n_epoch=null` makes `val_check_interval` count *global*
# batches instead of position-within-epoch, which is what the 1500 was already
# doing on OWT (whose epoch is millions of sequences long). The alternative is
# to drop the interval below one epoch's length.
set -euo pipefail

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

GPUS="${GPUS:-0,1,2,3,4,5}"
NUM_DEVICES="$(awk -F',' '{print NF}' <<< "${GPUS}")"

TEACHER="${TEACHER:-${ROOT}/downloads/mdlm/cosine_time_cond/1500000.safetensors}"
RUN_NAME="${RUN_NAME:-cifar10_dmmd_timecond}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/cifar10/${RUN_NAME}}"
# Data directory, picked at runtime rather than assumed -- the same thing the
# `cifar10-mdlm` Nirvana entrypoint does. `cache_dir/cifar10` is tracked in git
# (see .gitignore), so it is present in any checkout and in whatever gets
# uploaded to Nirvana; `~/data/cifar10` is the local fallback. Note this cannot
# live in `configs/data/cifar10.yaml` as `${cwd:}/cache_dir/cifar10`, because
# `hydra.job.chdir: true` makes `${cwd:}` the *run* directory, not the repo.
if [[ -d "${ROOT}/cache_dir/cifar10/cifar-10-batches-py" ]]; then
  CACHE_DIR="${CACHE_DIR:-${ROOT}/cache_dir/cifar10}"
else
  CACHE_DIR="${CACHE_DIR:-$HOME/data/cifar10}"
fi
echo "[data] cache_dir=${CACHE_DIR}"

BATCH_SIZE="${BATCH_SIZE:-8}"
# `adversarial_distill.accum_batches` = global_batch_size / (devices * batch_size),
# i.e. micro-batches accumulated inside the min-max loop before the student and
# the critic each take a step. The OWT run left global_batch_size at its default
# 512 with batch_size=8 on 8 GPUs, so it was running ACCUM=8. ACCUM=1 here makes
# it explicit and cheap; set ACCUM=8 to reproduce the OWT effective batch.
ACCUM="${ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
LR="${LR:-1e-6}"
FAKE_LR="${FAKE_LR:-1e-6}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"

# D-MMD Sec. 3.3 top-p on the *teacher distillation target* (a training-loss
# knob). Not to be confused with `sampling.p_nucleus`, which truncates the
# sampler's own draw and stays at its default 1.0 here.
TOP_P="${TOP_P:-1.0}"

# `ancestral_cache` freezes a position the moment it is revealed. `remask`
# (consistency-style) re-decodes every position each step and renoises, so a
# token committed early from a weak context can be revisited -- which is the
# dominant error source at low step counts. It needs `sampling.noise_removal`
# != 'none' (the default 'ancestral' satisfies that) and it invalidates the
# ELBO, so with it `val/bpd` / `val/nll` / `val/ppl` no longer describe the
# sampler that produced the samples. No extra cost here: `remask` never caches
# `p_x0`, and `algo.time_conditioning=True` already disables that cache.
#   PREDICTOR=remask REMASK_STRATEGY=consistency  # sigma_t = 1 - alpha_s
#   PREDICTOR=remask REMASK_STRATEGY=scaled REMASK_ETA=0.5
PREDICTOR="${PREDICTOR:-ancestral_cache}"
REMASK_STRATEGY="${REMASK_STRATEGY:-consistency}"
REMASK_ETA="${REMASK_ETA:-1.0}"

# The OWT run used noise_dim=768 = the DiT's hidden width, with noise_shared=true
# so it is one Gaussian latent per sample broadcast over positions. The UNet's
# residual stream at `input_conv` is `model.ch`=128 wide, so 128 is the faithful
# port; 768 also runs (the projection is just 768->128).
NOISE_DIM="${NOISE_DIM:-128}"

# Eval sizing. `val/fid` is a cheap tracker, not the paper's number, and it is
# meaningless at the OWT run's limit_val_batches=0.01 (~96 real images). Real
# images come from validation batches, generated ones from
# num_sample_batches * eval_batch_size; eval holds no gradients, so the eval
# batch can be far larger than the training one.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
VAL_FRAC="${VAL_FRAC:-0.25}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-4}"

# Sampler label for the run name. `remask_strategy` is only meaningful when the
# predictor actually is `remask` -- naming a run "consistency" while it samples
# with `ancestral_cache` is exactly the confusion this avoids.
if [[ "${PREDICTOR}" == "remask" ]]; then
  SAMPLER_LABEL="remask/${REMASK_STRATEGY}"
  [[ "${REMASK_STRATEGY}" != "consistency" ]] && SAMPLER_LABEL="${SAMPLER_LABEL} eta ${REMASK_ETA}"
else
  SAMPLER_LABEL="${PREDICTOR}"
fi

# Not an error any more: a missing teacher under `downloads/` is fetched from
# the hub by `algo._resolve_teacher_ckpt_path` at startup, into this same path.
# A TEACHER= pointing elsewhere still fails, with that function's message.
if [[ ! -f "${TEACHER}" ]]; then
  echo "[teacher] ${TEACHER} not present -- main will fetch it from"
  echo "[teacher] https://huggingface.co/jdeschena/duo2-cifar10 on startup."
fi

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -u -m main \
  data=cifar10 \
  data.cache_dir="${CACHE_DIR}" \
  model=unet \
  algo=mdlm_dmmd \
  algo.backbone=unet \
  noise=cosine \
  algo.time_conditioning=True \
  strategy.find_unused_parameters=True \
  adversarial_distill.is_distill=True \
  adversarial_distill.critic_type=diffusion \
  adversarial_distill.teacher_ckpt_path="${TEACHER}" \
  adversarial_distill.dmmd_mask_sampling='posterior' \
  adversarial_distill.top_p="${TOP_P}" \
  adversarial_distill.noise_dim="${NOISE_DIM}" \
  adversarial_distill.noise_scale=0.0 \
  adversarial_distill.noise_lr=1e-2 \
  adversarial_distill.noise_shared=true \
  adversarial_distill.fake_lr="${FAKE_LR}" \
  adversarial_distill.fake_frequency=1 \
  adversarial_distill.generator_num_noised_samples=1 \
  optim.lr="${LR}" \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
  loader.global_batch_size="$((BATCH_SIZE * NUM_DEVICES * ACCUM))" \
  loader.num_workers=1 \
  trainer.devices="${NUM_DEVICES}" \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=32 \
  training.loss_precision=float32 \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval=1500 \
  +trainer.check_val_every_n_epoch=null \
  trainer.limit_val_batches="${VAL_FRAC}" \
  sampling.steps="${SAMPLING_STEPS}" \
  sampling.predictor="${PREDICTOR}" \
  sampling.remask_strategy="${REMASK_STRATEGY}" \
  sampling.remask_eta="${REMASK_ETA}" \
  sampling.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
  eval.noise_diversity=True \
  eval.compute_fid=True \
  eval.compute_generative_perplexity=False \
  eval.compute_gm=False \
  eval.pareto_curve=False \
  eval.image_grid_path="${OUT_DIR}/samples.png" \
  '~callbacks.checkpoint_monitor' \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1500 \
  checkpointing.save_dir="${OUT_DIR}" \
  hydra.run.dir="${OUT_DIR}" \
  logger.name="cifar10 timecond ${SAMPLER_LABEL} posterior shared noised ${NOISE_DIM} lr 1e-2 top_p ${TOP_P} noise_scale 0.0" \
  "$@"
