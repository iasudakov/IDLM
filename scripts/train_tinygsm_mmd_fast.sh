#!/usr/bin/env bash
# Corrected archive objective; existing offline assets, Underdeep and paired eval.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/tinygsm/mdlm_notok.ckpt}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/cache_dir}"
MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-$CACHE_DIR/tinygsm_mmd_offline_bundle}"
OUT="${OUT:-$REPO_ROOT/outputs/mmd_archive_v2}"
DEVICES="${DEVICES:-8}"

exec python -u -m main \
  data=tiny-gsm 'data.separator="\n"' \
  data.wrap=False data.train_on_prompt=False data.train_on_pad=True data.filter_too_long=True \
  model=small model.length=512 algo=mdlm \
  strategy.find_unused_parameters=True \
  logger.experiment=generative-models/tinygsm logger.name='tinygsm mmd archive-v2 fast' \
  detect_anomaly=False \
  adversarial_distill.is_distill=True \
  adversarial_distill.dmd_weight=0.0 adversarial_distill.mmd_reward_weight=1.0 \
  adversarial_distill.mmd_implementation=fast adversarial_distill.mmd_reward_type=feature \
  adversarial_distill.mmd_feature_source=diffusion adversarial_distill.mmd_noise_level=0.0 \
  adversarial_distill.mmd_kernel=rbf adversarial_distill.mmd_kernel_level=token \
  adversarial_distill.mmd_kernel_source=draws adversarial_distill.mmd_kernel_batch_size=2 \
  adversarial_distill.mmd_rbf_exclude_same_sentence=True \
  adversarial_distill.mmd_rbf_exclude_same_position=True \
  adversarial_distill.mmd_rbf_omit_real_real=True \
  adversarial_distill.mmd_group_size=4 adversarial_distill.mmd_average_candidates=False \
  adversarial_distill.mmd_feature_layer=6 adversarial_distill.mmd_rbf_alpha=6e-5 \
  adversarial_distill.mmd_forward_batch_size=8 \
  adversarial_distill.mmd_fast_feature_batch_size=64 \
  training.finetune_path="$CKPT" training.loss_precision=bf16 \
  model_weights_dir="$MODEL_WEIGHTS_DIR" optim.lr=1e-5 \
  loader.batch_size=32 loader.eval_batch_size=64 loader.global_batch_size=512 loader.num_workers=4 \
  eval.generate_samples=True eval.compute_generative_perplexity=False \
  eval.validate_before_train=True eval.gsm8k_at_validation=True \
  eval.gsm8k_every_n_validations=1 eval.tinygsm_exec=True eval.gsm8k_pareto_fp32=True \
  'eval.gsm8k_pareto_thresholds=[0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.98]' \
  callbacks.checkpoint_monitor.monitor=val/nll callbacks.checkpoint_monitor.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_top_k=0 callbacks.checkpoint_every_n_steps.save_last=False \
  checkpointing.use_periodic_train_batch_checkpoint=False \
  checkpointing.save_evaluated_checkpoints=True checkpointing.save_evaluated_from_step=1 \
  checkpointing.resume_from_ckpt=False \
  trainer.precision=bf16 trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=null \
  trainer.devices="$DEVICES" trainer.max_steps=8000 trainer.num_sanity_val_steps=0 \
  'trainer.val_check_interval="${eval:500*${adversarial_distill.accum_batches}}"' \
  trainer.limit_val_batches=500 \
  sampling.steps=64 sampling.predictor=ancestral_cache \
  sampling.num_sample_batches=2 sampling.num_sample_log=8 \
  data.cache_dir="$CACHE_DIR" checkpointing.save_dir="$OUT" hydra.run.dir="$OUT" \
  "$@"
