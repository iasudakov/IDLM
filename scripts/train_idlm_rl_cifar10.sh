# Policy-gradient (REINFORCE) distillation of a CIFAR-10 MDLM teacher.
#
# The CIFAR-10 counterpart of the OWT `algo=mdlm_rl` recipe, and the RL
# counterpart of `scripts/train_idlm_dmmd_ar_cifar10.sh`. Same reverse KL, same
# AR critics, same student -- only the gradient estimator changes.
#
# WHY on this problem. `mdlm_dmmd` scores a one-shot draw from a *data-derived*
# latent: a sequence the student would never actually emit. The measured
# failure of the D-MMD AR-critic run is the student drifting off-distribution
# while the AR critic scores it there, so training against the sampler's real
# marginal targets exactly that gap. `rl_advantage_norm=std` also makes the
# step size independent of how peaked the critics are, which is the other thing
# that made the pixel-AR critic hard to balance.
#
# REQUIRES `critic_type=ar` -- asserted, not optional. The reward is
# `log p_teacher(x) - log p_fake(x)` on a finished sequence and the diffusion
# critic has no tractable sequence log-likelihood. This recipe therefore only
# works because of `ar_backbone=dit`; see `train_idlm_dmmd_ar_cifar10.sh` for
# what that loader does and why the checkpoint carries its own architecture.
#
# WHAT DOES NOT TRANSFER from the D-MMD recipes -- both are silent if ignored:
#
#  * `optim.lr`. Every RL term is reduced to per-token nats, whereas the
#    inherited D-MMD losses are a plain *sum* over 3072 positions. That is
#    ~1e3x, so the D-MMD scripts' `1e-6` would leave the student frozen. Hence
#    `1e-3` below. Compare `grad_norm_student` across runs, never the losses.
#  * `entropy_weight`. `_rl_record_entropy` normalises the bonus to per-token
#    nats too, so a value tuned for `mdlm_dmmd` (where H is a sum, order 1e3)
#    is ~1e3x too large here. Left at 0 so the first run measures whether the
#    RL estimator collapses at all; if `student_entropy` slides, start at 0.01
#    and calibrate off the logged `entropy_bonus` -- a few percent of
#    |student_loss|.
#
# `fake_lr` DOES transfer: the auxiliary critic keeps its own units.
#
# STEP BUDGET. `rl_num_steps` is both the rollout length and the budget the
# student is trained to decode at, so `sampling.steps` is set to match -- the
# policy is otherwise evaluated under a sampler it was not trained for. At 16
# the resulting FID is NOT comparable to the 64-step D-MMD numbers; it is a
# harder operating point. Raise both together to compare directly, at linear
# cost.
#
# COST. A generator micro-step is `rl_num_steps + 1` no-grad forwards plus
# `rl_grad_steps` with gradient. `rl_rollout_reuse=1` below pays a fresh
# rollout per micro-step, so each optimizer step sees
# `accum_batches * loader.batch_size` = 64 distinct trajectories. Setting
# `rl_rollout_reuse='${adversarial_distill.accum_batches}'` instead amortises
# one rollout across the whole accumulation window -- still *exactly*
# on-policy (neither model steps inside it) and much cheaper, at the price of
# 64 correlated gradients built from only 8 trajectories.
#
# Note `fake_frequency` is cheap here in a way it is not under `mdlm_dmmd`:
# fake steps draw from the rollout FIFO (`rl_buffer_size`) and pay no rollout
# of their own, so a critic that keeps up costs almost nothing.
#
# WATCH, in this order:
#   rl/reward           rising
#   rl/reward_std       NOT collapsing to 0 -- no spread means no advantage and
#                       hence no gradient at all
#   rl/advantage_abs    pinned near 1.41 means the group degenerated to sign
#                       only; raise loader.batch_size or use
#                       rl_advantage_norm=none
#   rl/logp             exactly 0.0 on some steps is expected, not a bug: the
#                       terminal noise-removal step has ~no masked positions
#                       left and is drawn ~1/(k+1) of the time
#   student_entropy     holding up
#   val/fid             the number that decides it

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-32}

AR_PATH=${AR_PATH:-/home/iasudakov/nirvana_checkpoints/checkpoints/last.ckpt}
TEACHER_PATH=${TEACHER_PATH:-'jdeschena/duo2-cifar10:mdlm/cosine_time_cond/1500000.ckpt'}
DATA=${DATA:-/home/iasudakov/IDLM/cache_dir}

python -u -m main \
  loader.global_batch_size=64 \
  loader.batch_size=8 \
  loader.eval_batch_size=100 \
  loader.num_workers=1 \
  model=unet \
  data=cifar10 \
  noise=cosine \
  strategy.find_unused_parameters=True \
  adversarial_distill.is_distill=True \
  algo=mdlm_rl \
  algo.backbone='unet' \
  algo.time_conditioning=True \
  eval.checkpoint_path="$TEACHER_PATH" \
  model.length=3072 \
  eval.compute_generative_perplexity=False \
  eval.compute_fid=True \
  eval.fid_num_gen_images=5000 \
  eval.fid_num_real_images=50000 \
  eval.compute_perplexity_on_sanity=True \
  callbacks.checkpoint_monitor.monitor=val/fid \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=32 \
  training.loss_precision=float32 \
  sampling.steps=16 \
  sampling.predictor='ancestral_cache' \
  adversarial_distill.critic_type=ar \
  adversarial_distill.ar_backbone=dit \
  adversarial_distill.ar_model_path="$AR_PATH" \
  adversarial_distill.rl_num_steps=16 \
  adversarial_distill.rl_grad_steps=1 \
  adversarial_distill.rl_baseline=leave_one_out \
  adversarial_distill.rl_advantage_norm=std \
  adversarial_distill.rl_group_size=0 \
  adversarial_distill.rl_pathwise_weight=0.0 \
  adversarial_distill.rl_buffer_size=8 \
  adversarial_distill.rl_rollout_reuse=1 \
  adversarial_distill.top_p=1.0 \
  adversarial_distill.entropy_weight=0.0 \
  adversarial_distill.fake_warmup_steps=2000 \
  adversarial_distill.fake_frequency=5 \
  optim.lr=1e-3 \
  adversarial_distill.fake_lr=1e-4 \
  adversarial_distill.use_gradient_clipping=True \
  adversarial_distill.gradient_clipping_val=1.0 \
  trainer.val_check_interval=2000 \
  +trainer.check_val_every_n_epoch=null \
  data.cache_dir=$DATA \
  logger.experiment=generative-models/dmmd-cifar \
  logger.name='cifar10 rl ar critic k16 lr 1e-3' \
  "$@"
