#!/usr/bin/env bash
# Nirvana entrypoint for the CIFAR-10 MDLM teacher (D-MMD Sec. 6.1 / Table 1).
#
# This is the standalone `train_cifar10_mdlm.py` trainer, not `python -m main`:
# there is no Hydra here, so the config is command-line flags. The mapping from
# the old Hydra invocation is in the table at the bottom of this file.
#
# Preemption: every `SNAPSHOT_EVERY` steps the run directory is packed to the
# Nirvana snapshot path, and at startup it is unpacked back, so a restarted
# attempt resumes from its own latest.pt -- weights, EMA, optimizer, dataloader
# epoch and the underdeep run id included. Just re-running this script is the
# whole restart procedure.
set -euxo pipefail   # NB: `set -eux pipefail` (no -o) silently sets a
                     # positional arg named "pipefail" and leaves pipefail OFF.

echo "=========== env =============="
env > "$PWD/envlist"
echo "Python 3: $(python3 -V) at $(which python3)"
echo "PWD: $(pwd)"
nvidia-smi

echo "=========== proxies =============="
# Needed for the one-time CIFAR-10 download, the torch-fidelity Inception
# weights that FID uses, and reaching the underdeep API.
export proxy="http://hermes.sas.yp-c.yandex.net:6953"
curl "$proxy"/register/"$(hostname)" -s # Please keep this line
export ftp_proxy="$proxy"
export http_proxy="$proxy"
export https_proxy="$proxy"

echo "=========== paths =============="
PATH="/conda/envs/main/bin:$PATH"
cd "$SOURCE_CODE_PATH/code"
export PYTHONPATH="$PYTHONPATH:$SOURCE_CODE_PATH:$SOURCE_CODE_PATH/code"

# torch-fidelity caches the Inception network here; keeping it on disk means
# only the first attempt pays the download.
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p "$TORCH_HOME"

# underdeep reads UNDERDEEP_TOKEN first, then ~/.underdeep/token. Point this at
# whichever secret file holds the tracker token -- do NOT reuse the `cat
# /slot/sandbox/d/secret/*` glob from the old script, which concatenates every
# secret in the directory and yields a token that is silently wrong once there
# is more than one file there.
if [[ -z "${UNDERDEEP_TOKEN:-}" && -f /slot/sandbox/d/secret/underdeep_token ]]; then
  export UNDERDEEP_TOKEN="$(< /slot/sandbox/d/secret/underdeep_token)"
fi

# https://github.com/NVIDIA/nccl/issues/929#issuecomment-1650422352
export NCCL_SOCKET_IFNAME=veth0

echo "=========== run =============="
NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
OUT="${OUT:-$SOURCE_CODE_PATH/code/outputs/cifar10_teacher}"
DATA_DIR="${DATA_DIR:-$SOURCE_CODE_PATH/code/cache_dir/cifar10}"
mkdir -p "$OUT" "$DATA_DIR"

# Global batch 512, as in the old config. BATCH_SIZE is per GPU.
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
BATCH_SIZE="${BATCH_SIZE:-$((GLOBAL_BATCH / NUM_GPUS))}"
if [[ $((BATCH_SIZE * NUM_GPUS)) -ne "${GLOBAL_BATCH}" ]]; then
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} not divisible by NUM_GPUS=${NUM_GPUS}" >&2
  exit 1
fi

exec torchrun --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT:-29511}" \
  train_cifar10_mdlm.py \
  --out "$OUT" \
  --data-dir "$DATA_DIR" \
  --run-name "${RUN_NAME:-CIFAR-10 masked teacher (uncond, bs512, lr4e-4)}" \
  --underdeep-experiment "${UNDERDEEP_EXPERIMENT:-generative-models/cifar10_mdlm}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS:-375000}" \
  --lr "${LR:-4e-4}" \
  --warmup "${WARMUP:-5000}" \
  --ema "${EMA:-0.9999}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --sample-every "${SAMPLE_EVERY:-1000}" \
  --eval-every "${EVAL_EVERY:-10000}" \
  --eval-nfe "${EVAL_NFE:-128}" \
  --eval-fid-samples "${EVAL_FID_SAMPLES:-2000}" \
  --ckpt-every "${CKPT_EVERY:-2500}" \
  --ema-ckpt-every "${EMA_CKPT_EVERY:-50000}" \
  --snapshot-every "${SNAPSHOT_EVERY:-10000}"

# ---------------------------------------------------------------------------
# Mapping from the previous Hydra command:
#
#   data.cache_dir=...                        -> --data-dir
#   loader.batch_size=64                      -> --batch-size 64
#   loader.global_batch_size=512              -> batch-size x NUM_GPUS
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
#   checkpointing.save_dir / hydra.run.dir    -> --out (one flag, not three)
#   logger.name=...                           -> --run-name
#
# Dropped, with reasons:
#   data=cifar10 / model=unet / algo=mdlm / noise=cosine / algo.backbone=unet
#     -- there is one model and one algorithm in this script; nothing to select.
#   data.num_classes=null / adversarial_distill.is_distill=False
#     -- unconditional plain-likelihood training is the only mode.
#   sampling.predictor=ancestral_cache
#     -- the sampler is plain ancestral. Caching only pays off when consecutive
#        steps reveal nothing, which is rare at these step counts.
#   eval.compute_generative_perplexity / compute_gm / pareto_curve
#     -- text-only metrics.
#   '~callbacks.checkpoint_monitor', callbacks..., hydra.run.dir
#     -- no Lightning callbacks and no Hydra.
#   HF_HOME / TRANSFORMERS_CACHE / the HF token
#     -- no HuggingFace model is loaded. FID's Inception comes from
#        torch-fidelity via TORCH_HOME, set above.
#   pip install ipdb
#     -- not imported by this script. `models/__init__.py` does still pull in
#        models/dit.py, which imports flash_attn, so that must be present in
#        /conda/envs/main (it already is, for `python -m main`).
# ---------------------------------------------------------------------------
