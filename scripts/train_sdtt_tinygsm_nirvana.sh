#!/usr/bin/env bash
set -euo pipefail

# Match the historical TinyGSM MMD job's code/assets layout and Underdeep
# experiment. SOURCE_CODE_PATH points to the bundle containing code/.
if [[ -n "${SDTT_CODE_DIR:-}" ]]; then
  sdtt_code_dir="$SDTT_CODE_DIR"
elif [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  sdtt_code_dir="$SOURCE_CODE_PATH/code"
else
  sdtt_code_dir="$(dirname -- "${BASH_SOURCE[0]}")/.."
fi
cd -- "$sdtt_code_dir"
sdtt_code_dir="$PWD"

# SDTT uses automatic accumulation: 8 GPUs * 32 examples * 2 = 512.
# Extra Hydra arguments override these defaults, including cluster paths.
exec "${PYTHON:-python}" -u -m main \
  +experiment=sdtt_tinygsm_official \
  data=tiny-gsm \
  data.wrap=False \
  data.train_on_prompt=False \
  data.train_on_pad=True \
  data.filter_too_long=True \
  model=small model.length=512 algo=sdtt \
  strategy.find_unused_parameters=True \
  adversarial_distill.is_distill=False \
  "training.finetune_path=$sdtt_code_dir/checkpoints/tinygsm/mdlm_notok.ckpt" \
  training.loss_precision=float32 \
  loader.batch_size=32 loader.eval_batch_size=64 \
  loader.global_batch_size=512 loader.num_workers=4 \
  optim.lr=1e-5 \
  eval.generate_samples=False eval.compute_generative_perplexity=False \
  eval.validate_before_train=True eval.gsm8k_at_validation=True \
  eval.gsm8k_every_n_validations=1 eval.gsm8k_batch_size=16 \
  eval.gsm8k_pareto_temperature=0.0 \
  'eval.gsm8k_pareto_thresholds=[0.325,0.4,0.475,0.55,0.625,0.7,0.775,0.85,0.925,0.95,0.98,0.99,0.995]' \
  callbacks.checkpoint_monitor.monitor=val/nll \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  callbacks.checkpoint_every_n_steps.save_top_k=-1 \
  callbacks.checkpoint_every_n_steps.save_last=True \
  checkpointing.use_periodic_train_batch_checkpoint=False \
  checkpointing.resume_from_ckpt=False \
  trainer.devices=8 trainer.precision=bf16-mixed \
  trainer.accumulate_grad_batches=2 trainer.gradient_clip_val=null \
  trainer.max_steps=70000 algo.grow_dt_every=10000 \
  trainer.val_check_interval=1000 trainer.limit_val_batches=500 \
  trainer.log_every_n_steps=100 \
  sampling.steps=64 sampling.predictor=ancestral_cache \
  sampling.num_sample_batches=2 sampling.num_sample_log=8 \
  "data.cache_dir=$sdtt_code_dir/cache_dir" \
  "model_weights_dir=$sdtt_code_dir/cache_dir/tinygsm_mmd_offline_bundle" \
  "checkpointing.save_dir=$sdtt_code_dir/sdtt_tinygsm" \
  "hydra.run.dir=$sdtt_code_dir/sdtt_tinygsm" \
  logger=underdeep \
  logger.experiment=generative-models/tinygsm \
  'logger.name=sdtt tinygsm lr1e-5 official 70k' \
  "$@"
