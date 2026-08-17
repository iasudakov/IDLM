#!/bin/bash
# One-step RL distillation (algo=mdlm_rl, adversarial_distill.rl_policy=onestep).
#
# Same reverse-KL objective and the same *samples* as scripts/train_idlm_mdlm.sh
# (one denoiser call on a data-derived latent, x ~ Cat(x_hat_eta(z_t))); only the
# gradient estimator changes. D-MMD differentiates the student's probabilities
# while holding the AR critics' context fixed -- a biased, per-position
# surrogate. Here the same reward (log p_teacher(x) - log p_fake(x)) is
# optimized with REINFORCE, which is unbiased for the sequence-level quantity.
#
# This is the ablation that separates "RL instead of the pathwise gradient" from
# "trajectory samples instead of one-shot ones": run it against
# scripts/train_idlm_rl.sh (rl_policy=trajectory) to attribute the difference.
#
# Cost: ONE student forward per generator micro-step -- the sample and the
# gradient come out of the same call -- plus rl_onestep_group_size critic
# forwards. Cheaper than rl_policy=trajectory by roughly rl_num_steps.
#
# Notable settings:
#   * rl_onestep_group_size=4: the G i.i.d. draws from *one* latent that the
#     baseline is computed over. Grouping per latent is what makes the baseline
#     valid -- across rows of a batch the reward mostly reflects which (x0, t)
#     was drawn. Raise for a lower-variance baseline at G critic forwards each.
#   * optim.lr matches scripts/train_idlm_rl.sh: the generator terms are in
#     per-token nats, ~1e3x smaller than the D-MMD script's summed loss. The
#     one-step surrogate is smaller still by about the average mask rate (only
#     masked positions are actions), so if grad_norm_student looks small this is
#     the first knob to raise.
#   * adversarial_distill.fake_lr transfers from the D-MMD script unchanged --
#     the auxiliary critic keeps the parent's one-shot objective and units here.
#   * rl_pathwise_weight>0 adds the D-MMD term back at the same latent and
#     sample, i.e. literally the non-RL objective, if the pure policy gradient
#     is too noisy early.
#
# Watch: rl/reward rising, rl/fake_nll - rl/teacher_nll shrinking,
# rl/reward_std (the WITHIN-latent spread here) not collapsing to 0 -- no
# spread means no advantage and hence no gradient -- and student_entropy
# holding up.

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
  logger.name=rl-onestep-mdlm \
  sampling.steps=32 \
  sampling.predictor='ancestral_cache' \
  optim.lr=1e-3 \
  adversarial_distill.fake_lr=1e-4 \
  adversarial_distill.fake_frequency=1 \
  adversarial_distill.critic_type=ar \
  adversarial_distill.rl_policy=onestep \
  adversarial_distill.rl_onestep_group_size=4 \
  adversarial_distill.rl_baseline=leave_one_out \
  adversarial_distill.rl_advantage_norm=std \
  adversarial_distill.rl_pathwise_weight=0.0 \
  adversarial_distill.use_gradient_clipping=True \
  adversarial_distill.gradient_clipping_val=1.0 \
  trainer.val_check_interval=1500 \
  trainer.limit_val_batches=0.01 \
  logger.force_new_run=true \
  data.cache_dir='/mnt/data/owt'
