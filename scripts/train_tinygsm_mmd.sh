#!/bin/bash
# TinyGSM step-distillation driven by the paired AR feature reward ported from
# yrOPDLM's rl-mmd branch (see `mmd_reward.py`).
#
# Data: extract tinygsm_mmd_data.tar into the repo root first. That provides the
# tokenized TinyGSM cache, the GSM8K test set and the teacher checkpoint. Only
# HuggingFaceTB/SmolLM-135M is fetched at runtime -- it is both the student's
# tokenizer and the frozen AR the reward reads features from.
#
#   tar -xf tinygsm_mmd_data.tar
#   DEVICES=4 bash scripts/train_tinygsm_mmd.sh
#
# MODE=mmd     : reward only  (dmd_weight=0) -- no critic, no fake steps
# MODE=dmd     : the published objective, for an A/B baseline
# MODE=both    : reward on top of the published objective
# MODE=teacher : continue the teacher with the ordinary MDLM training loss
# MMD_GROUP=1  : one reward sample per state; MMD_BASELINE selects auto, batch,
#                or none (`auto` resolves to none for a group of one).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/tinygsm/mdlm_notok.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/cache_dir}"
DEVICES="${DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# `loader.global_batch_size` MUST equal devices * batch_size * accumulation:
# `trainer_base._sample_t` draws that many timesteps and splits them across
# ranks and accumulation steps, so any other value hands a micro-batch too few
# and fails deep inside `q_xt` with a broadcast error naming neither.
ACCUM="${ACCUM:-$(( 512 / (DEVICES * BATCH_SIZE) ))}"
GLOBAL_BATCH=$(( DEVICES * BATCH_SIZE * ACCUM ))

MODE="${MODE:-mmd}"
CONTINUE_TEACHER=False
case "${MODE}" in
  mmd)  DMD_W=0.0; MMD_W="${MMD_W:-1.0}" ;;
  dmd)  DMD_W=1.0; MMD_W=0.0 ;;
  both) DMD_W=1.0; MMD_W="${MMD_W:-1e-3}" ;;
  teacher) DMD_W=0.0; MMD_W=0.0; CONTINUE_TEACHER=True ;;
  *) echo "MODE must be mmd|dmd|both|teacher, got ${MODE}" >&2; exit 1 ;;
esac
NAME="${NAME:-idlm-tinygsm-${MODE}}"
OUT="${OUT:-${REPO_ROOT}/outputs/${MODE}}"

cd "${REPO_ROOT}"
echo "[run] mode=${MODE} dmd_weight=${DMD_W} mmd_weight=${MMD_W} | " \
     "${DEVICES} x ${BATCH_SIZE} x ${ACCUM} accum = ${GLOBAL_BATCH} global batch | " \
     "mmd_group=${MMD_GROUP:-4} baseline=${MMD_BASELINE:-auto} | " \
     "high_mask=${HIGH_MASK_FRACTION:-0.0}@${HIGH_MASK_MIN:-0.7}-${HIGH_MASK_MAX:-1.0}"

python -u -m main \
  data="tiny-gsm" \
  data.cache_dir="${CACHE_DIR}" \
  data.wrap=False data.train_on_prompt=False data.train_on_pad=True \
  data.filter_too_long=True \
  model=small model.length=512 algo=mdlm \
  adversarial_distill.is_distill=True \
  adversarial_distill.dmd_weight="${DMD_W}" \
  adversarial_distill.mmd_reward_weight="${MMD_W}" \
  adversarial_distill.high_mask_fraction="${HIGH_MASK_FRACTION:-0.0}" \
  adversarial_distill.high_mask_min="${HIGH_MASK_MIN:-0.7}" \
  adversarial_distill.high_mask_max="${HIGH_MASK_MAX:-1.0}" \
  training.continue_teacher="${CONTINUE_TEACHER}" \
  adversarial_distill.mmd_block_size="${MMD_BLOCK:-64}" \
  adversarial_distill.mmd_group_size="${MMD_GROUP:-4}" \
  adversarial_distill.mmd_baseline="${MMD_BASELINE:-auto}" \
  sampling.predictor=ancestral_cache sampling.steps="${EVAL_STEPS:-64}" \
  sampling.num_sample_batches=2 sampling.num_sample_log=8 \
  loader.batch_size="${BATCH_SIZE}" loader.eval_batch_size=64 \
  loader.global_batch_size="${GLOBAL_BATCH}" loader.num_workers=4 \
  strategy.find_unused_parameters=True \
  training.finetune_path="${CKPT}" \
  training.loss_precision=bf16 optim.lr="${LR:-1e-6}" \
  eval.generate_samples=True eval.compute_generative_perplexity=False \
  eval.validate_before_train=True eval.gsm8k_at_validation=True \
  eval.gsm8k_every_n_validations="${GSM8K_EVERY:-1}" eval.tinygsm_exec=True \
  trainer.precision=bf16 trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=null \
  trainer.devices="${DEVICES}" trainer.max_steps="${MAX_STEPS:-20000}" \
  trainer.val_check_interval="${VAL_EVERY:-1000}" trainer.limit_val_batches=500 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=2000 \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_last=False \
  checkpointing.use_periodic_train_batch_checkpoint=True \
  checkpointing.save_dir="${OUT}" \
  callbacks.checkpoint_monitor.monitor=val/nll \
  logger.experiment="${UD_PROJECT:-generative-models/tinygsm}" \
  logger.name="${NAME}" \
  hydra.run.dir="${OUT}"
