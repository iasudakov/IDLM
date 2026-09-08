#!/usr/bin/env bash
# Conditional TinyGSM adapter; released DiDi training settings, not MMD.
# TEACHER_CHECKPOINT=/path/mdlm_notok.ckpt OWT_CACHE is NOT used here.
# TINYGSM_CACHE=/path/cache_dir GSM8K_DATA=/path/gsm8k_test.json \
# SAVE_DIR=/path/new_run GPUS=0,1,2,3,4,5,6,7 bash scripts/train_didi_tinygsm.sh
# Optional threshold sweeps: append '+eval.gsm8k_pareto_thresholds=[0.5,0.7,0.9]'.
# Images require Matplotlib; +eval.gsm8k_pareto_plot=false keeps numeric curves.
set -euo pipefail

DIDI_TINY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIDI_TINY_ROOT"
: "${TEACHER_CHECKPOINT:?set TEACHER_CHECKPOINT to native TinyGSM mdlm_notok.ckpt}"
: "${TINYGSM_CACHE:?set TINYGSM_CACHE to the existing TinyGSM cache_dir}"
: "${SAVE_DIR:?set SAVE_DIR to a new training output directory}"
GSM8K_DATA="${GSM8K_DATA:-$DIDI_TINY_ROOT/data/gsm8k_test.json}"
GSM8K_EVAL="${GSM8K_EVAL:-true}"
if [[ ! -f "$TEACHER_CHECKPOINT" || ! -d "$TINYGSM_CACHE" ]]; then
  printf '%s\n' 'Missing TinyGSM teacher or cache directory; prepare/copy them first.' >&2
  exit 1
fi
if [[ "$GSM8K_EVAL" == true && ! -f "$GSM8K_DATA" ]]; then
  printf 'Missing GSM8K test JSON: %s (or set GSM8K_EVAL=false).\n' "$GSM8K_DATA" >&2
  exit 1
fi
TEACHER_CHECKPOINT="$(realpath "$TEACHER_CHECKPOINT")"
TINYGSM_CACHE="$(realpath "$TINYGSM_CACHE")"
SAVE_DIR="$(realpath -m "$SAVE_DIR")"
GSM8K_DATA="$(realpath -m "$GSM8K_DATA")"
if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
DIDI_TINY_GPUS="${NPROC_PER_NODE:-}"
if [[ -z "$DIDI_TINY_GPUS" ]]; then
  DIDI_TINY_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
if [[ ! "$DIDI_TINY_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'No CUDA GPUs selected. Set GPUS or NPROC_PER_NODE.' >&2
  exit 1
fi
DIDI_TINY_EVAL_BATCH="${EVAL_BATCH_SIZE:-8}"
if [[ ! "$DIDI_TINY_EVAL_BATCH" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' 'EVAL_BATCH_SIZE must be a positive integer.' >&2
  exit 1
fi

# The original script's max_steps counts individual optimizer updates.
# Do not substitute the paper-informed 10000-iteration/warmup0/EMA.9999 recipe.
DIDI_TINY_COMMAND=(python -u -m torch.distributed.run --standalone
  "--nproc_per_node=$DIDI_TINY_GPUS" --module didi_repro
  mode=train data=tiny-gsm model=small model.length=512
  algo=didi_instruct algo.backbone=dit seed=1
  "data.cache_dir=$TINYGSM_CACHE"
  data.wrap=false data.train_on_prompt=false data.train_on_pad=true
  data.filter_too_long=true
  "training.finetune_path=$TEACHER_CHECKPOINT"
  "checkpointing.save_dir=$SAVE_DIR" "hydra.run.dir=$SAVE_DIR"
  checkpointing.resume_from_ckpt=false
  algo.num_samples_per_prompt=8 algo.tau_mode=beta11
  optim.lr=1e-6 algo.discriminator_lr=1e-6
  algo.lr_scheduler.warmup_steps=1000 algo.ema_beta=0.999
  +didi_training.iterations=null trainer.max_steps=10000
  "trainer.devices=$DIDI_TINY_GPUS"
  loader.batch_size=1 "loader.global_batch_size=$DIDI_TINY_GPUS"
  "loader.eval_batch_size=$DIDI_TINY_EVAL_BATCH"
  "loader.eval_global_batch_size=$((DIDI_TINY_GPUS * DIDI_TINY_EVAL_BATCH))"
  loader.num_workers=4
  trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=0.0
  training.loss_precision=bf16 trainer.precision=bf16
  trainer.val_check_interval=1000 trainer.limit_val_batches=8
  trainer.num_sanity_val_steps=0
  "+didi_eval.before_training=${EVAL_BEFORE_TRAINING:-true}"
  +didi_eval.weights=student_ema
  "+didi_eval.sampler_source=${SAMPLER_SOURCE:-main}"
  "sampling.predictor=${SAMPLER:-guided}" "sampling.steps=${EVAL_STEPS:-16}"
  sampling.noise_removal=ancestral sampling.num_sample_batches=2
  sampling.num_sample_log=8
  eval.generate_samples=true eval.compute_generative_perplexity=false
  +eval.tinygsm_exec=true +eval.tinygsm_exec_timeout=5.0
  "+eval.gsm8k_at_validation=$GSM8K_EVAL"
  "+eval.gsm8k_data_path=$GSM8K_DATA"
  "+eval.gsm8k_every_n_validations=${GSM8K_EVERY:-1}"
  +didi_monitor.gm=false +didi_monitor.gen_ppl=false
  "+underdeep.experiment=${EXPERIMENT:-generative-models/tinygsm}"
  "+underdeep.name=${RUN_NAME:-didi-tinygsm-released-training}"
  +underdeep.force_new_run=true +didi_runtime.strict_environment=true
)
if [[ "${DIDI_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${DIDI_TINY_COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${DIDI_TINY_COMMAND[@]}" "$@"
