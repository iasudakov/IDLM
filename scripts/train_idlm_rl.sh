#!/bin/bash
# Trajectory-level RL distillation (algo=mdlm_rl).
#
# Rolls out the student's own k-step ancestral sampler from the all-mask prior,
# scores the finished sequence with the owt-ar teacher/fake critics
# (reward = log p_teacher(x) - log p_fake(x)) and trains with REINFORCE. Same
# reverse-KL objective as scripts/train_idlm_mdlm.sh, but on samples the
# student would actually emit.
#
# Two things differ from the D-MMD script beyond `algo=`:
#   * optim.lr is 1e3x larger. Every generator-side RL term is reduced to
#     per-token nats, while the D-MMD losses sum over ~model.length positions,
#     so 1e-6 there is ~1e-3 here. Watch `grad_norm_student` (logged by both)
#     rather than the loss value when tuning it.
#   * sampling.steps == adversarial_distill.rl_num_steps, so validation decodes
#     with the step budget the policy was trained at.
#
# Watch: rl/reward rising, rl/fake_nll - rl/teacher_nll shrinking,
# rl/reward_std not collapsing to 0 (no spread => no gradient), and
# student_entropy holding up.

python -u -m main \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=1 \
  model=small \
  data=openwebtext-split \
  strategy.find_unused_parameters=True \
  adversarial_distill.is_distill=True \
  algo=mdlm_rl \
  algo.backbone='hf_dit' \
  eval.checkpoint_path='kuleshov-group/mdlm-owt' \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=32 \
  training.loss_precision=float32 \
  logger.name=rl-mdlm \
  sampling.steps=16 \
  sampling.predictor='ancestral_cache' \
  optim.lr=1e-3 \
  adversarial_distill.fake_lr=1e-4 \
  adversarial_distill.fake_frequency=1 \
  adversarial_distill.critic_type=ar \
  adversarial_distill.rl_num_steps=16 \
  adversarial_distill.rl_grad_steps=1 \
  adversarial_distill.rl_baseline=leave_one_out \
  adversarial_distill.rl_advantage_norm=std \
  adversarial_distill.rl_pathwise_weight=0.0 \
  adversarial_distill.rl_rollout_reuse=1 \
  adversarial_distill.use_gradient_clipping=True \
  adversarial_distill.gradient_clipping_val=1.0 \
  trainer.val_check_interval=1500 \
  trainer.limit_val_batches=0.01 \
  logger.force_new_run=true \
  data.cache_dir='/mnt/data/owt'
