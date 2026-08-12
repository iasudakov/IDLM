#!/usr/bin/env bash
# Nirvana entrypoint for the CIFAR-10 MDLM teacher (D-MMD Sec. 6.1 / Table 1).
#
# This is the standalone `train_cifar10_mdlm.py` trainer, not `python -m main`:
# there is no Hydra here, so the config is the block of assignments below. The
# mapping from the old Hydra flags is in the table at the bottom of this file.
#
# !! NO DOLLAR-BRACE EXPANSIONS ANYWHERE IN THIS FILE, COMMENTS INCLUDED !!
# Nirvana renders `run-command` through FreeMarker before bash ever sees it, and
# FreeMarker claims dollar-brace as its own interpolation syntax. A bash default
# written that way dies at render time with
#   freemarker.core.ParseException: Encountered ":", but was expecting ...
# FreeMarker has no idea what a bash comment is, so one in a comment breaks the
# job just as surely as one in a command.
# Bare $FOO, $(cmd) and $((a / b)) are all invisible to FreeMarker and safe.
# That is why every value here is a plain assignment rather than a defaulted
# override -- edit the config block to change a run.
#
# Preemption: every SNAPSHOT_EVERY steps the run directory is packed to the
# Nirvana snapshot path, and at startup it is unpacked back, so a restarted
# attempt resumes from its own latest.pt -- weights, EMA, optimizer, dataloader
# epoch and the underdeep run id included. Re-running this script is the whole
# restart procedure.

# -u is deliberately absent: an unset optional variable should not kill the job.
set -exo pipefail

# =============== config ===============
MAX_STEPS=375000
GLOBAL_BATCH=512
LR=4e-4
WARMUP=5000
EMA_DECAY=0.9999
NUM_WORKERS=8

SAMPLE_EVERY=1000       # sample grid -> underdeep
EVAL_EVERY=10000        # test BPD + FID
EVAL_NFE=128
EVAL_FID_SAMPLES=2000
CKPT_EVERY=2500         # rewrite latest.pt (bounds what a crash costs)
EMA_CKPT_EVERY=50000    # archival ema_<step>.pt
SNAPSHOT_EVERY=10000    # push to Nirvana snapshot (bounds preemption cost)

UNDERDEEP_EXPERIMENT=generative-models/cifar10_mdlm
RUN_NAME="CIFAR-10 masked teacher (uncond, bs512, lr4e-4)"

echo "=========== env =============="
env > $PWD/envlist
echo "Python 3: $(python3 -V) at $(which python3)"
echo "PWD: $(pwd)"
nvidia-smi

echo "=========== proxies =============="
# Needed for the one-time CIFAR-10 download, the torch-fidelity Inception
# weights that FID uses, and reaching the underdeep API.
export proxy="http://hermes.sas.yp-c.yandex.net:6953"
curl "$proxy"/register/"$(hostname)" -s # Please keep this line
export ftp_proxy=$proxy
export http_proxy=$proxy
export https_proxy=$proxy

echo "=========== paths =============="
PATH="/conda/envs/main/bin:$PATH"
cd $SOURCE_CODE_PATH/code
export PYTHONPATH=$PYTHONPATH:$SOURCE_CODE_PATH
export PYTHONPATH=$PYTHONPATH:$SOURCE_CODE_PATH/code

# torch-fidelity caches the Inception network here, so only the first attempt
# pays the download.
export TORCH_HOME=$HOME/.cache/torch
mkdir -p $TORCH_HOME

# underdeep reads UNDERDEEP_TOKEN first, then ~/.underdeep/token. Point this at
# whichever secret file holds the tracker token. Do NOT use the
# `cat /slot/sandbox/d/secret/*` glob -- it concatenates every secret in the
# directory and silently yields a wrong token once there is more than one.
if [ -f /slot/sandbox/d/secret/underdeep_token ]; then
  export UNDERDEEP_TOKEN=$(cat /slot/sandbox/d/secret/underdeep_token)
fi

# https://github.com/NVIDIA/nccl/issues/929#issuecomment-1650422352
export NCCL_SOCKET_IFNAME=veth0

echo "=========== run =============="
NUM_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())')
BATCH_SIZE=$((GLOBAL_BATCH / NUM_GPUS))   # per GPU

OUT=$SOURCE_CODE_PATH/code/outputs/cifar10_teacher
DATA_DIR=$SOURCE_CODE_PATH/code/cache_dir/cifar10
mkdir -p $OUT
mkdir -p $DATA_DIR

# CIFAR-10 ships inside the code archive (cache_dir/cifar10), so nothing is
# downloaded here. Say so loudly either way: the proxy pulls the 170 MB archive
# at roughly 500 kB/s, and a missing copy means paying that on every fresh
# attempt -- easy to mistake for a hung job.
if [ -d $DATA_DIR/cifar-10-batches-py ]; then
  echo "[data] CIFAR-10 present in $DATA_DIR, no download"
else
  echo "[data] WARNING: $DATA_DIR/cifar-10-batches-py is missing."
  echo "[data] Rank 0 will download 170 MB through the proxy (~6 min), and"
  echo "[data] will do it again on every fresh attempt. To avoid that, ship"
  echo "[data] cache_dir/cifar10/cifar-10-batches-py in the code archive."
fi

echo "[launch] $NUM_GPUS GPUs, global_batch=$GLOBAL_BATCH ($BATCH_SIZE/gpu)"

exec torchrun --nproc_per_node=$NUM_GPUS --master_port=29511 \
  train_cifar10_mdlm.py \
  --out $OUT \
  --data-dir $DATA_DIR \
  --run-name "$RUN_NAME" \
  --underdeep-experiment $UNDERDEEP_EXPERIMENT \
  --batch-size $BATCH_SIZE \
  --max-steps $MAX_STEPS \
  --lr $LR \
  --warmup $WARMUP \
  --ema $EMA_DECAY \
  --num-workers $NUM_WORKERS \
  --sample-every $SAMPLE_EVERY \
  --eval-every $EVAL_EVERY \
  --eval-nfe $EVAL_NFE \
  --eval-fid-samples $EVAL_FID_SAMPLES \
  --ckpt-every $CKPT_EVERY \
  --ema-ckpt-every $EMA_CKPT_EVERY \
  --snapshot-every $SNAPSHOT_EVERY

# ---------------------------------------------------------------------------
# Mapping from the previous Hydra command:
#
#   data.cache_dir=...                        -> --data-dir
#   loader.batch_size=64                      -> --batch-size (GLOBAL/NUM_GPUS)
#   loader.global_batch_size=512              -> GLOBAL_BATCH
#   loader.num_workers=8                      -> --num-workers
#   trainer.max_steps=375000                  -> --max-steps
#   trainer.val_check_interval=10000          -> --eval-every
#   optim.lr=4e-4                             -> --lr
#   lr_scheduler.num_warmup_steps=5000        -> --warmup
#   training.ema=0.9999                       -> --ema
#   sampling.steps=128                        -> --eval-nfe
#   eval.fid_num_samples_val=2000             -> --eval-fid-samples
#   callbacks...every_n_train_steps=2500      -> --ckpt-every
#   checkpointing.ema_export_every_n_steps    -> --ema-ckpt-every
#   checkpointing.nirvana_snapshot_every_...  -> --snapshot-every
#   checkpointing.save_dir / hydra.run.dir /
#     eval.image_grid_path                    -> --out (one flag, not three)
#   logger.name=...                           -> --run-name
#
# Dropped, with reasons:
#   data=cifar10 / model=unet / algo=mdlm / noise=cosine / algo.backbone=unet
#     -- one model and one algorithm in this script; nothing to select.
#   data.num_classes=null / adversarial_distill.is_distill=False
#     -- unconditional plain-likelihood training is the only mode.
#   sampling.predictor=ancestral_cache
#     -- the sampler is plain ancestral. Caching only pays off when consecutive
#        steps reveal nothing, which is rare at these step counts.
#   eval.compute_generative_perplexity / compute_gm / pareto_curve
#     -- text-only metrics.
#   '~callbacks.checkpoint_monitor', callbacks..., hydra.run.dir
#     -- no Lightning callbacks and no Hydra.
#   HF_HOME / HF_DATASETS_CACHE / TRANSFORMERS_CACHE / the HF token
#     -- no HuggingFace model is loaded. FID's Inception comes from
#        torch-fidelity via TORCH_HOME, set above.
#   pip install ipdb
#     -- not imported by this script. models/__init__.py does still pull in
#        models/dit.py, which imports flash_attn, so that must be present in
#        /conda/envs/main (it already is, for `python -m main`).
# ---------------------------------------------------------------------------
