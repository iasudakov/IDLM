# Autoregressive CIFAR-10 baseline: the same data, budget and schedule as
# `train_cifar10_mdlm_cosine.sh`, but a *causal* DiT over the 3072 pixel tokens
# (3 x 32 x 32, raster order) instead of the masked-diffusion UNet. `model=unet`
# is not an option here -- it is not causal.
#
# `model.length=3072` is required: `model=small` ships 1024, and for AR the
# sequence is the whole image.
#
# Sampling is one forward per token, i.e. 3072 sequential forwards per batch with
# no KV cache, so the in-loop grid and FID are throttled far below the diffusion
# recipe's settings. `sampling.steps` has no meaning for this algo. Note also
# that `AR.generate_samples` pre-draws Gumbel noise of shape
# (batch, 3072, vocab), which is why `eval.fid_batch_size` is small.
#
# Unconditional, like the MDLM recipe: AR never passes labels to the backbone.
python -u -m main \
    mode=train \
    data=cifar10 \
    data.cache_dir=/home/iasudakov/IDLM/cache_dir/cifar10 \
    data.num_classes=null \
    model=small \
    model.length=3072 \
    algo=ar \
    loader.global_batch_size=128 \
    loader.num_workers=8 \
    trainer.val_check_interval=2500 \
    trainer.max_steps=1_500_000 \
    lr_scheduler.num_warmup_steps=5000 \
    optim.lr=2e-4 \
    eval.generate_samples=False \
    eval.sample_image_num=8 \
    eval.sample_image_every_n_validations=4 \
    eval.fid_num_samples=256 \
    eval.fid_batch_size=32 \
    eval.fid_every_n_validations=8 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    logger=underdeep \
    logger.name=ar_cifar10 \
    logger.experiment=generative-models/cifar10_mdlm \
    '~wandb' \
    hydra.run.dir=./outputs/cifar10/ar_cifar10 \
    "$@"
