# FID / IS over the full 50k CIFAR-10 train split for an `algo=ar` checkpoint.
#
# `mode=fid_eval` generates `eval.fid_eval_num_samples` images, sharded over the
# ranks, against Inception statistics built over the whole un-flipped 50k train
# split. Leaving it unset generates one sample per training image -- the 50k
# that published CIFAR-10 FIDs are quoted at, and the only setting comparable to
# them. `NUM_SAMPLES=5000 bash scripts/fid_cifar10_ar.sh` is the cheap preview:
# same reference, fewer samples, so the score is biased upward but comparable
# across checkpoints evaluated the same way. The torchmetrics states are
# sum-reduced across ranks, so the reported number is over the whole run, not
# per-GPU.
#
# The architecture flags must match training exactly (`model=small`,
# `model.length=3072`, `data.num_classes=null`), otherwise `load_from_checkpoint`
# will not line up with the saved weights. EMA weights are used: the checkpoint
# carries them and `_eval_mode()` copies them in (pass `eval.disable_ema=True`
# for the raw weights instead).
#
# `loader.eval_batch_size` is set explicitly because every rank must generate the
# same number of images: the sample count has to be a multiple of
# `eval_batch_size` x num_gpus. At 125 on 8 GPUs that step is 1000 images, which
# divides 50000 and every round preview size; on 4 GPUs it is 500.
# `sampling.steps` is meaningless here -- `AR.generate_samples` discards it and
# always spends one forward per token.
#
# Cost: AR decoding has no KV cache, so each of the 3072 steps re-runs the whole
# prefix. Measured on an A100-80GB: ~8 s/image, so 50k images over 8 A100s is
# ~14 h and a 5k preview ~1.4 h.
#
# `expandable_segments` is not optional here. Every step allocates a
# differently-shaped activation (the prefix grows by one token), so the default
# caching allocator ends up with a separate segment per shape and *reserves*
# an order of magnitude more than it has live -- 18.6 GB reserved against 1.3 GB
# allocated by step 1024 at batch 32, still climbing. That OOMs long before the
# end of the sequence at any useful batch size. Expandable segments hold the
# reserve to ~1.2x the live set at no measurable cost in speed.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-/home/iasudakov/nirvana_checkpoints/checkpoints/last.ckpt}
DATA=${DATA:-/home/iasudakov/IDLM/cache_dir/cifar10}
OUT=${OUT:-./outputs/cifar10/fid_ar_1_5M_uncond}
NUM_SAMPLES=${NUM_SAMPLES:-null}  # null => the full 50k

python -u -m main \
    mode=fid_eval \
    data=cifar10 \
    data.cache_dir=$DATA \
    data.num_classes=null \
    model=small \
    model.length=3072 \
    algo=ar \
    loader.eval_batch_size=125 \
    loader.num_workers=8 \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    eval.disable_ema=False \
    eval.fid_eval_num_samples=$NUM_SAMPLES \
    '~wandb' \
    hydra.run.dir=$OUT \
    "$@"
