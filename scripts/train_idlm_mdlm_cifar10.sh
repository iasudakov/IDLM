# D-MMD distillation of the published CIFAR-10 MDLM teacher.
#
# The CIFAR-10 analogue of `train_idlm_mdlm.sh` -- same single `python -m main`
# invocation, same algo (algo=mdlm_dmmd, critic_type=diffusion, so teacher and
# fake critic are copies of the loaded denoiser). Only the domain differs:
#
#   data=cifar10                  raw pixels, 3*32*32 = 3072 byte tokens per
#                                 image; `modality: image` swaps the text
#                                 metrics for the image ones
#   model=unet, algo.backbone=unet    D3PM UNet, the checkpoint's architecture
#   noise=cosine                  the schedule the checkpoint was trained with
#   algo.time_conditioning=True   ... as was its time conditioning ("time_cond")
#   callbacks.checkpoint_monitor.monitor=val/nll
#                                 the default monitors val/gen_ppl, a text
#                                 metric this run does not log, and Lightning
#                                 raises rather than warns on a missing monitor
#                                 key. val/fid also works but is noisy over a
#                                 few batches.
#   +trainer.check_val_every_n_epoch=null
#                                 CIFAR-10 is 50k images, so an epoch is only a
#                                 few hundred batches -- fewer than
#                                 val_check_interval=1500, which Lightning
#                                 rejects outright. With this, val_check_interval
#                                 counts *global* batches across epoch
#                                 boundaries (the OWT semantics) instead of
#                                 batches within one epoch, and stays valid at
#                                 any batch size. The `+` is required because the
#                                 key is absent from configs/config.yaml.
#                                 Alternative: drop val_check_interval below the
#                                 per-epoch batch count (50000 / batch_size).
#
# Nothing needs pre-downloading: `eval.checkpoint_path` is a
# `<hf_repo>:<path>` hub spec that is fetched (and cached) while the model is
# built, before training starts, and torchvision fetches CIFAR-10 into
# `data.cache_dir` on the first batch. A local `.ckpt` path works there too.
#
# Logged at every validation (see `TrainerBase._log_image_metrics`): val/fid
# over 1000 generated images against the cached statistics of the 50k CIFAR-10
# train split, val/fid_num_gen / val/fid_num_real (the counts the estimator's
# bias rides on), val/bpd, val/nll, val/sample_entropy, val/pred_entropy, plus a
# grid of generated images.
#
# `eval.compute_perplexity_on_sanity=True` runs that whole block once during the
# sanity check, BEFORE the first training step, and pushes it to the logger at
# step 0. At that point the student is still exactly the loaded teacher, so it is
# the pre-distillation baseline every later val/fid should be read against.
#
# The generated side is what costs time -- 1000 / eval_batch_size sampling
# batches of sampling.steps denoiser calls per validation -- hence
# eval_batch_size=100 here rather than the training batch size. The real side is
# computed once per process and reused, so 50000 is nearly free.
#
# No `trainer.limit_val_batches`: the OWT recipe caps it because an epoch of that
# validation set is enormous, but CIFAR-10's test split is 10k images, i.e. ~13
# batches per rank here, and one ELBO forward each is negligible beside the
# sampling FID needs. Capping it also breaks at this batch size -- Lightning
# rejects a fraction that rounds to zero batches ("0.01 * 13 < 1") -- and val/nll
# is the checkpoint monitor, so the full split is the better signal anyway.

# Cap CPU threads: this box is shared, and the dataloader / torch intra-op pools
# would otherwise size themselves to every core on the machine.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-32}

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
  eval.checkpoint_path='jdeschena/duo2-cifar10:mdlm/cosine_time_cond/1500000.ckpt' \
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
  adversarial_distill.critic_type=diffusion \
  adversarial_distill.dmmd_mask_sampling='posterior' \
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
  data.cache_dir=$SOURCE_CODE_PATH/code/cache_dir \
  logger.experiment=generative-models/dmmd-cifar \
  logger.name='cifar10 posterior shared noised 768 lr 1e-2 top_p 1.0 noise_scale 0.0'
