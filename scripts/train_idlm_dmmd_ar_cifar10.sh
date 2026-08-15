# D-MMD distillation of a CIFAR-10 MDLM teacher with an *autoregressive* critic.
#
# The CIFAR-10 analogue of the OWT `critic_type=ar` recipe, and the AR-critic
# counterpart of `scripts/train_idlm_mdlm_cifar10.sh`. Everything about the
# student is identical to that script -- same D3PM UNet initialised from a
# pretrained masked-diffusion checkpoint, same sampler, same metrics. Only the
# critics change:
#
#   adversarial_distill.critic_type=ar    teacher and fake are autoregressive
#                                         models instead of copies of the
#                                         loaded denoiser
#   adversarial_distill.ar_backbone=dit   ... and they are this repo's causal
#                                         DiT (`algo=ar`) rather than the
#                                         published owt-ar text model, which is
#                                         a GPT-2-vocabulary model and has
#                                         nothing to say about pixels
#   adversarial_distill.ar_model_path     the `algo=ar` Lightning checkpoint.
#                                         Its architecture is read out of the
#                                         checkpoint's own
#                                         `hyper_parameters.config`, so
#                                         `model=small` / `model.length=3072` /
#                                         `causal_attention=True` do NOT have to
#                                         be restated here -- and cannot drift
#                                         from what was trained.
#
# Both models are the ones trained on the `duo-cifar10` branch over the same
# 3072 raw-pixel byte tokens, so teacher and student share a vocabulary (256
# pixel values + [MASK]) -- `_load_dit_ar_backbone` asserts this rather than
# letting a mismatch score silently misaligned log-probs.
#
# TEACHER (student initialisation), `eval.checkpoint_path`: a masked-diffusion
# `.ckpt`. Defaults to the published `jdeschena/duo2-cifar10` model that
# `train_idlm_mdlm_cifar10.sh` uses; point `TEACHER_PATH` at your own MDLM run's
# checkpoint to distill from that instead. Both `model=unet noise=cosine
# algo.time_conditioning=True` below describe that checkpoint, so change them
# together with it.
#
# CRITIC, `AR_PATH`: the `algo=ar` checkpoint. `eval.disable_ema` selects raw vs
# EMA weights for it exactly as it does for the student.
#
# What the AR critic changes about the objective, relative to the diffusion one:
#
#  * the critic scores the student's *clean* generated sequence by next-token
#    log-probabilities, so there is no critic noise level and no `w(s)` time
#    weighting -- `adversarial_distill.loss_weighting` and
#    `.dmmd_mask_sampling` (the D-MMD posterior coupling) are diffusion-critic
#    knobs and do nothing here, which is why neither is passed below.
#    `top_p` still applies: it is nucleus-restriction of the *teacher target*
#    (D-MMD Sec. 3.3) and works on any critic's log-probs.
#  * the first pixel is unsupervised. An AR model conditions position `i` on
#    `x_{<i}`, so position 0 carries no next-token term -- 1 of 3072 positions.
#  * the generator gradient is smaller. `optim.lr=1e-6` below is carried over
#    from the diffusion-critic recipe, where every token loss is scaled by the
#    ELBO weight `w(s)` (mean ~7). The AR critic has no time variable and so no
#    such weight, leaving its gradient roughly an order of magnitude smaller at
#    the same lr. If the student barely moves, raise `optim.lr` (and
#    `adversarial_distill.fake_lr` with it) before suspecting the objective.
#  * cost: two extra 3072-token causal DiT forwards per generator step (teacher
#    and fake, both under no_grad) and one forward+backward per fake step.
#
# Nothing needs pre-downloading beyond the two checkpoints: torchvision fetches
# CIFAR-10 into `data.cache_dir` on the first batch, and `eval.checkpoint_path`
# also accepts a `<hf_repo>:<path>` hub spec (see `utils.resolve_checkpoint_path`).
#
# See `scripts/train_idlm_mdlm_cifar10.sh` for why the shared flags are what
# they are -- `+trainer.check_val_every_n_epoch=null`,
# `callbacks.checkpoint_monitor.monitor=val/nll`, the FID sample counts and
# `eval.compute_perplexity_on_sanity=True` (the step-0 pre-distillation
# baseline, when the student is still exactly the loaded teacher).

# Cap CPU threads: this box is shared, and the dataloader / torch intra-op pools
# would otherwise size themselves to every core on the machine.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-32}

AR_PATH=${AR_PATH:-/home/iasudakov/nirvana_checkpoints/checkpoints/last.ckpt}
TEACHER_PATH=${TEACHER_PATH:-'jdeschena/duo2-cifar10:mdlm/cosine_time_cond/1500000.ckpt'}
DATA=${DATA:-/home/iasudakov/IDLM/cache_dir}

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
  callbacks.checkpoint_monitor.monitor=val/nll \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=32 \
  training.loss_precision=float32 \
  sampling.steps=32 \
  sampling.predictor='ancestral_cache' \
  sampling.num_sample_batches=4 \
  adversarial_distill.critic_type=ar \
  adversarial_distill.ar_backbone=dit \
  adversarial_distill.ar_model_path="$AR_PATH" \
  adversarial_distill.top_p=1.0 \
  adversarial_distill.noise_dim=768 \
  adversarial_distill.noise_scale=0.0 \
  adversarial_distill.noise_lr=1e-2 \
  adversarial_distill.noise_shared=true \
  eval.noise_diversity=True \
  optim.lr=1e-6 \
  adversarial_distill.fake_lr=1e-6 \
  adversarial_distill.fake_frequency=1 \
  adversarial_distill.generator_num_noised_samples=1 \
  trainer.val_check_interval=1500 \
  +trainer.check_val_every_n_epoch=null \
  data.cache_dir=$DATA \
  logger.experiment=generative-models/dmmd-cifar \
  logger.name='cifar10 ar critic shared noised 768 lr 1e-2 top_p 1.0 noise_scale 0.0' \
  "$@"
