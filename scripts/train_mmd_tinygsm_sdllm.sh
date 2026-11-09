#!/usr/bin/env bash
# TinyGSM MDLM student trained with frozen SDLLM features of its answers.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/tinygsm_mmd_sdllm}"
TRAIN_DEVICES="${TRAIN_DEVICES:-4}"
FEATURE_LAYER="${FEATURE_LAYER:-6}"
MDLM_CHECKPOINT="${MDLM_CHECKPOINT:-$REPO_DIR/checkpoints/tinygsm/mdlm_notok.ckpt}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/cache_dir}"
if ! [[ "$TRAIN_DEVICES" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'TRAIN_DEVICES must be 1, 2, 4, 8 or 16.' >&2
  exit 2
fi

# Preserve global batch 512 and the TinyGSM reference update/evaluation budgets.
ACCUM_BATCHES=$((512 / (32 * TRAIN_DEVICES)))
VAL_INTERVAL=$((1000 * ACCUM_BATCHES / 2))
LOG_INTERVAL=$((100 * ACCUM_BATCHES / 2))
CHECKPOINT_INTERVAL=$((2000 * ACCUM_BATCHES / 2))
VAL_BATCHES=$((500 * 8 / TRAIN_DEVICES))
SAMPLE_BATCHES=$((2 * 8 / TRAIN_DEVICES))
ARGS=(
  data=tiny-gsm data.wrap=False data.train_on_prompt=False data.train_on_pad=True
  data.filter_too_long=True "data.cache_dir=$CACHE_DIR"
  model=small model.length=512 algo=mdlm algo.backbone=dit
  strategy.find_unused_parameters=True logger.experiment=generative-models/tinygsm
  adversarial_distill.is_distill=True adversarial_distill.dmd_weight=0.0
  adversarial_distill.mmd_reward_weight=1.0 adversarial_distill.mmd_reward_type=feature
  adversarial_distill.mmd_feature_source=sdllm_mdlm
  adversarial_distill.mmd_noise_level=0.0 adversarial_distill.mmd_score_positions=all
  adversarial_distill.mmd_group_size=4 adversarial_distill.mmd_kernel_source=draws
  adversarial_distill.mmd_kernel_level=token adversarial_distill.mmd_kernel_batch_size=2
  adversarial_distill.mmd_rbf_exclude_same_sentence=True
  adversarial_distill.mmd_rbf_exclude_same_position=True
  adversarial_distill.mmd_kernel=rbf adversarial_distill.mmd_rbf_alpha=null
  "adversarial_distill.mmd_feature_layer=$FEATURE_LAYER"
  adversarial_distill.mmd_forward_batch_size=8
  "training.finetune_path=$MDLM_CHECKPOINT" training.loss_precision=bf16
  optim.lr=1e-5 trainer.precision=bf16
  loader.batch_size=32 loader.eval_batch_size=64 loader.global_batch_size=512 loader.num_workers=4
  "trainer.devices=$TRAIN_DEVICES" "adversarial_distill.accum_batches=$ACCUM_BATCHES"
  trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=null trainer.max_steps=20000
  "trainer.val_check_interval=$VAL_INTERVAL" "trainer.log_every_n_steps=$LOG_INTERVAL"
  "trainer.limit_val_batches=$VAL_BATCHES"
  eval.generate_samples=True eval.compute_generative_perplexity=False
  eval.validate_before_train=True eval.gsm8k_at_validation=True
  eval.gsm8k_every_n_validations=1 eval.tinygsm_exec=True
  'eval.gsm8k_pareto_thresholds=[0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.98]'
  callbacks.checkpoint_monitor.monitor=val/nll
  "callbacks.checkpoint_every_n_steps.every_n_train_steps=$CHECKPOINT_INTERVAL"
  callbacks.checkpoint_every_n_steps.save_top_k=0
  callbacks.checkpoint_every_n_steps.save_last=False
  checkpointing.use_periodic_train_batch_checkpoint=True
  sampling.steps=64 sampling.predictor=ancestral_cache
  "sampling.num_sample_batches=$SAMPLE_BATCHES" sampling.num_sample_log=8
  "checkpointing.save_dir=$OUTPUT_DIR" "hydra.run.dir=$OUTPUT_DIR"
  "logger.name=tinygsm-sdllm-layer$FEATURE_LAYER-g4-b2-auto"
)
if [[ -n "${MODEL_WEIGHTS_DIR:-}" ]]; then
  ARGS+=("model_weights_dir=$MODEL_WEIGHTS_DIR")
fi
if [[ -n "${SDLLM_MODEL_DIR:-}" ]]; then
  ARGS+=("adversarial_distill.mmd_sdllm_path=$SDLLM_MODEL_DIR"
         adversarial_distill.mmd_sdllm_local_files_only=True)
fi
exec "$PYTHON_BIN" -u -m main "${ARGS[@]}" "$@"
