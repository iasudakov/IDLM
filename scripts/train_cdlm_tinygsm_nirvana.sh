#!/usr/bin/env bash
set -euo pipefail

# Use the historical TinyGSM MMD code/checkpoint/tokenizer-bundle layout.
if [[ -n "${CDLM_CODE_DIR:-}" ]]; then
  cdlm_code_dir="$CDLM_CODE_DIR"
elif [[ -n "${SOURCE_CODE_PATH:-}" ]]; then
  cdlm_code_dir="$SOURCE_CODE_PATH/code"
else
  cdlm_code_dir="$(dirname -- "${BASH_SOURCE[0]}")/.."
fi
cd -- "$cdlm_code_dir"
cdlm_code_dir="$PWD"
cdlm_preset="${CDLM_PRESET:-cdlm_tinygsm}"
cdlm_teacher_overrides=()
if [[ "$cdlm_preset" != cdlm_tinygsm_mmd ]]; then
  cdlm_teacher_overrides=("training.finetune_path=$cdlm_code_dir/checkpoints/tinygsm/mdlm_notok.ckpt")
fi

# Four B300 GPUs, 32 examples per GPU, accumulation 4: global batch 512.
# Collection is explicit and completes before training starts. Trailing Hydra
# overrides are shared by both phases; the phase itself is fixed after them.
cdlm_overrides=(
  "+experiment=$cdlm_preset"
  data=tiny-gsm data.wrap=False data.train_on_prompt=False
  data.train_on_pad=True data.filter_too_long=True
  model=small model.length=512 algo=cdlm
  algo.attention_mode=full algo.segment_length=32
  algo.kld_type=forward_kl algo.distill_weight=1.0
  algo.consistency_weight=0.5 algo.denoise_weight=0.1
  algo.denoise_mask_floor=0.001
  'algo.trajectory_temperatures=[0.0,0.5]'
  algo.trajectory_num_examples=20000 algo.trajectory_batch_size=8
  algo.trajectory_shard_size=8 algo.teacher_hidden_dtype=bfloat16
  "algo.trajectory_cache_dir=$cdlm_code_dir/cache_dir/cdlm_tinygsm_trajectories"
  strategy.find_unused_parameters=True adversarial_distill.is_distill=False
  "${cdlm_teacher_overrides[@]}"
  training.loss_precision=float32
  loader.batch_size=32 loader.eval_batch_size=64
  loader.global_batch_size=512 loader.num_workers=4
  optim.lr=1e-5
  eval.generate_samples=False eval.compute_generative_perplexity=False
  eval.validate_before_train=True eval.gsm8k_at_validation=True
  eval.gsm8k_every_n_validations=1 eval.gsm8k_batch_size=16
  eval.gsm8k_greedy_token=True
  eval.gsm8k_pareto_temperature=0.0
  'eval.gsm8k_pareto_thresholds=[0.325,0.4,0.475,0.55,0.625,0.7,0.775,0.85,0.925,0.98]'
  callbacks.checkpoint_monitor.monitor=val/nll
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000
  callbacks.checkpoint_every_n_steps.save_top_k=-1
  callbacks.checkpoint_every_n_steps.save_last=True
  callbacks.checkpoint_every_n_steps.save_on_train_epoch_end=True
  checkpointing.use_periodic_train_batch_checkpoint=False
  checkpointing.resume_from_ckpt=False
  checkpointing.snapshot=True
  checkpointing.snapshot_trajectories=True
  trainer.devices=4 trainer.precision=bf16-mixed
  trainer.accumulate_grad_batches=4 trainer.gradient_clip_val=null
  trainer.val_check_interval=2000 trainer.check_val_every_n_epoch=null
  trainer.limit_val_batches=1000 trainer.log_every_n_steps=100
  sampling.steps=64 sampling.predictor=ancestral_cache
  sampling.num_sample_batches=2 sampling.num_sample_log=8
  "data.cache_dir=$cdlm_code_dir/cache_dir"
  "model_weights_dir=$cdlm_code_dir/cache_dir/tinygsm_mmd_offline_bundle"
  "checkpointing.save_dir=$cdlm_code_dir/cdlm_tinygsm"
  "hydra.run.dir=$cdlm_code_dir/cdlm_tinygsm"
  logger=underdeep logger.experiment=generative-models/tinygsm
  "$@"
)

# Resolve/download each checkpoint once before torchrun starts. Both phases
# receive the same local bytes; diagnostic output is separate from arguments.
cdlm_asset_file="$(mktemp)"
trap 'rm -f -- "$cdlm_asset_file"' EXIT
# Use the proxy from the original Nirvana job only in asset preparation.
# An explicitly empty value opts out; training keeps its original environment.
CDLM_DOWNLOAD_PROXY="${CDLM_DOWNLOAD_PROXY-http://survival-mode.klg.yp-c.yandex.net:6953}" \
"${PYTHON:-python}" -u -m cdlm_preflight --output "$cdlm_asset_file" "${cdlm_overrides[@]}"
mapfile -d '' -t cdlm_asset_overrides < "$cdlm_asset_file"
rm -f -- "$cdlm_asset_file"
trap - EXIT
cdlm_overrides+=("${cdlm_asset_overrides[@]}")

"${PYTHON:-python}" -u -m torch.distributed.run \
  --standalone "--nproc_per_node=${CDLM_PREPARE_GPUS:-4}" \
  -m main "${cdlm_overrides[@]}" mode=prepare_cdlm

exec "${PYTHON:-python}" -u -m main "${cdlm_overrides[@]}" mode=train
