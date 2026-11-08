#!/usr/bin/env bash
# Exact 3daa OWT recipe with an opt-in external SDTT MDLM feature extractor.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/owt_mmd_sdtt_large}"
TRAIN_DEVICES="${TRAIN_DEVICES:-8}"
FEATURE_SOURCE="${FEATURE_SOURCE:-sdtt_mdlm}"
FEATURE_LAYER="${FEATURE_LAYER:-3}"
if ! [[ "$TRAIN_DEVICES" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'TRAIN_DEVICES must be1,2,4,8 or16 to preserve the original evaluation interval.' >&2
  exit 2
fi
# Preserve the original intervals in student updates when GPU count changes.
ACCUM_BATCHES=$((512 / (8 * TRAIN_DEVICES)))
VAL_INTERVAL=$((250 * ACCUM_BATCHES / 8))
LOG_INTERVAL=$((100 * ACCUM_BATCHES / 8))
CHECKPOINT_INTERVAL=$((200000 * ACCUM_BATCHES / 8))
ARGS=(
  data=openwebtext-split model=small model.length=1024 algo=mdlm algo.backbone=hf_dit
  eval.checkpoint_path=kuleshov-group/mdlm-owt strategy.find_unused_parameters=True
  adversarial_distill.is_distill=True adversarial_distill.dmd_weight=0.0
  adversarial_distill.mmd_reward_weight=1.0
  "adversarial_distill.mmd_feature_source=$FEATURE_SOURCE"
  adversarial_distill.mmd_noise_level=0.0 adversarial_distill.mmd_group_size=8
  optim.lr=1e-4 training.loss_precision=bf16 trainer.precision=bf16
  loader.batch_size=8 loader.eval_batch_size=8 loader.global_batch_size=512 loader.num_workers=4
  "trainer.devices=$TRAIN_DEVICES" "adversarial_distill.accum_batches=$ACCUM_BATCHES"
  eval.generate_samples=True eval.compute_generative_perplexity=True
  eval.owt_pareto_temp_every_n_validations=1 'eval.owt_pareto_steps=[8,16,32]'
  'eval.pareto_temperatures=[0.75,0.8,0.85,0.875,0.9,0.925,0.95,0.975,1.0,1.05,1.1,1.2]'
  eval.owt_pareto_seeds=5 eval.owt_gm=True 'eval.eval_step_range=[0,20000]'
  "callbacks.checkpoint_every_n_steps.every_n_train_steps=$CHECKPOINT_INTERVAL"
  callbacks.checkpoint_every_n_steps.save_top_k=0 checkpointing.use_periodic_train_batch_checkpoint=True
  trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=null trainer.max_steps=15000
  "trainer.val_check_interval=$VAL_INTERVAL" "trainer.log_every_n_steps=$LOG_INTERVAL"
  trainer.limit_val_batches=0.01 sampling.steps=32 sampling.predictor=ancestral_cache
  adversarial_distill.mmd_kernel_source=draws adversarial_distill.mmd_kernel_level=token
  adversarial_distill.mmd_kernel_batch_size=2
  adversarial_distill.mmd_rbf_exclude_same_sentence=True adversarial_distill.mmd_rbf_exclude_same_position=True
  adversarial_distill.mmd_kernel=rbf adversarial_distill.mmd_rbf_alpha=2e-5
  "adversarial_distill.mmd_feature_layer=$FEATURE_LAYER" adversarial_distill.mmd_forward_batch_size=8
  "checkpointing.save_dir=$OUTPUT_DIR" "hydra.run.dir=$OUTPUT_DIR"
  "logger.name=paper-owt-mmd-features-$FEATURE_SOURCE-layer$FEATURE_LAYER"
)
if [[ -n "${MODEL_WEIGHTS_DIR:-}" ]]; then
  ARGS+=("model_weights_dir=$MODEL_WEIGHTS_DIR")
fi
if [[ -n "${SDTT_MODEL_DIR:-}" ]]; then
  ARGS+=("adversarial_distill.mmd_sdtt_path=$SDTT_MODEL_DIR"
         adversarial_distill.mmd_sdtt_local_files_only=True)
fi
exec "$PYTHON_BIN" -u -m main "${ARGS[@]}" "$@"
