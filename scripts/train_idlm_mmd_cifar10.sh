# Distil a CIFAR-10 MDLM teacher with the PAIRED MMD feature reward.
#
# Ported from the `owt_mmd` branch. The reward replaces (or supplements) the
# D-MMD critic term with a score-estimator objective:
#
#   loss = dmd_weight * (teacher_loss - fake_loss) + REINFORCE(mmd reward)
#
# `mmd_group_size` hard samples are drawn from the student's one-step
# prediction at the same state; each is scored by pooling the FROZEN DIFFUSION
# TEACHER's bottleneck features and comparing them against the features of its
# own clean row `x0`; the group's leave-one-out mean is the baseline. Gradient
# reaches the student only through `log p(sampled token)`, so the feature map
# never has to be differentiable -- which is why a frozen teacher read through a
# forward hook is enough.
#
# WHAT THIS RECIPE PINS, AND WHY
#
#  critic_type=diffusion   Required. It is the only configuration where
#                          `self.teacher` is the frozen diffusion model the
#                          student is distilled from; under `critic_type=ar` the
#                          same attribute holds a causal DiT over pixels.
#  mmd_noise_level=0.0     `[MASK]` is a near-white pixel on CIFAR-10, not an
#                          out-of-band symbol, so masking before the feature
#                          read paints speckle the features then describe.
#  mmd_kernel=linear       The bottleneck gives 16 positions per row, far too
#                          few for the unbiased RBF estimator (see
#                          `mmd_rbf_alpha` / `mmd_kernel_batch_size` in
#                          configs/config.yaml if you want to try it anyway).
#  dmd_weight=0            MMD-only by default: the teacher/fake critic forwards
#  use_fake=False          are skipped entirely, there is no min-max loop and no
#                          fake optimizer, so every step is a generator step.
#                          Set DMD_WEIGHT=1.0 USE_FAKE=True to run both terms.
#
# TWO THINGS TO WATCH
#
#  * MMD_WEIGHT does not transfer from an OWT recipe -- the critic terms here
#    are summed over 3072 positions while the reward is a per-token mean, and
#    the pseudo-Huber constant was tuned for transformer activations. Calibrate
#    against `grad_norm_student` on a short run before committing to a value.
#  * The reward is paired against `x0`, one arbitrary draw from p(x). At the
#    median mask rate the student writes ~70% of the image, so this rewards
#    reproducing that specific image rather than matching the distribution, and
#    it can fight FID. `T_WARMUP_STEPS` caps the draw at `t_warmup_start` and
#    ramps it; that is the intended mitigation. Watch `val/fid` and
#    `student_entropy`, not just `mmd/reward` -- on OpenWebText this reward
#    drove a collapse that its own reward curve reported as progress.
#
# NB: no `set -u` -- conda's MKL activate hook references an unbound variable.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

TEACHER_PATH=${TEACHER_PATH:-'jdeschena/duo2-cifar10:mdlm/cosine_time_cond/1500000.ckpt'}
DATA=${DATA:-${REPO_ROOT}/cache_dir}
MMD_WEIGHT=${MMD_WEIGHT:-1.0}
DMD_WEIGHT=${DMD_WEIGHT:-0}
USE_FAKE=${USE_FAKE:-False}
GROUP_SIZE=${GROUP_SIZE:-4}
T_WARMUP_STEPS=${T_WARMUP_STEPS:-0}
T_WARMUP_START=${T_WARMUP_START:-0.3}
LR=${LR:-1e-6}
RUN_NAME=${RUN_NAME:-"cifar10 mmd paired linear g${GROUP_SIZE} w${MMD_WEIGHT}"}

if [ -n "${GPUS-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
# Needed to unpickle the training checkpoints under this torch version.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -u -m main \
  loader.batch_size=8 \
  loader.eval_batch_size=100 \
  loader.num_workers=1 \
  model=unet \
  data=cifar10 \
  noise=cosine \
  strategy.find_unused_parameters=True \
  adversarial_distill.is_distill=True \
  algo=mdlm_dmmd \
  algo.backbone='unet' \
  algo.time_conditioning=True \
  eval.checkpoint_path="$TEACHER_PATH" \
  model.length=3072 \
  eval.compute_generative_perplexity=False \
  eval.compute_fid=True \
  eval.fid_num_gen_images=1000 \
  eval.fid_num_real_images=50000 \
  eval.compute_perplexity_on_sanity=True \
  callbacks.checkpoint_monitor.monitor=val/fid \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=32 \
  training.loss_precision=float32 \
  sampling.steps=32 \
  sampling.predictor='ancestral_cache' \
  sampling.num_sample_batches=4 \
  adversarial_distill.critic_type=diffusion \
  adversarial_distill.use_fake="$USE_FAKE" \
  adversarial_distill.dmd_weight="$DMD_WEIGHT" \
  adversarial_distill.mmd_reward_weight="$MMD_WEIGHT" \
  adversarial_distill.mmd_kernel=linear \
  adversarial_distill.mmd_noise_level=0.0 \
  adversarial_distill.mmd_group_size="$GROUP_SIZE" \
  adversarial_distill.mmd_baseline=auto \
  adversarial_distill.t_warmup_steps="$T_WARMUP_STEPS" \
  adversarial_distill.t_warmup_start="$T_WARMUP_START" \
  adversarial_distill.generator_num_noised_samples=1 \
  optim.lr="$LR" \
  trainer.val_check_interval=1500 \
  +trainer.check_val_every_n_epoch=null \
  data.cache_dir="$DATA" \
  logger.experiment=generative-models/dmmd-cifar \
  logger.name="$RUN_NAME" \
  "$@"
