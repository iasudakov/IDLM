# CIFAR-10 MDLM teacher (D-MMD Sec. 6.1 / Table 1)

Trains the unconditional masked-diffusion model that
[*Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD*](https://arxiv.org/abs/2603.20155)
(Hoogeboom et al., 2026) distills in Sec. 6.1. CIFAR-10 is modelled directly as
**3072 tokens over the 256 pixel values** — no VAE, no inductive bias about the
vocabulary. Target (Table 1, "Masked Teacher", 50K samples vs the *train* split):

| NFE | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| FID | 122.9 | 47.1 | 20.0 | 11.1 | 7.8 | 6.7 | **6.4** |

## Run

```bash
conda activate mdlm

# full recipe, 8 GPUs
bash scripts/train_cifar10_mdlm.sh

# overrides are env vars
MAX_STEPS=50000 RUN_NAME=short GPUS=0,1,2,3 bash scripts/train_cifar10_mdlm.sh

# end-to-end self test: 20 steps, tiny UNet, no underdeep (~10 min, mostly FID)
python train_cifar10_mdlm.py --smoke
```

Then produce the Table 1 row from a finished checkpoint:

```bash
torchrun --nproc_per_node=8 eval_cifar10_fid.py \
    --ckpt outputs/cifar10_mdlm_teacher/checkpoints/latest.pt
```

It prints this run's FID next to the paper's for every NFE and writes
`fid_results.json`.

## Files

| | |
|---|---|
| `train_cifar10_mdlm.py` | everything: data, MDLM loss, sampler, DDP loop, EMA, eval, logging, checkpoints |
| `eval_cifar10_fid.py` | the paper's protocol — 50K samples per NFE against the train split |
| `scripts/train_cifar10_mdlm.sh` | torchrun launcher holding the recipe |
| `models/unet.py` | the D3PM/DUO UNet backbone (reused, unmodified) |
| `nirvana_utils.py` | Nirvana snapshot transfer (reused, unmodified) |

Deliberately standalone: none of the Hydra/Lightning stack (`main.py`, `algo.py`,
`trainer_base.py`, `dataloader.py`) is imported or touched. The ~60 lines of MDLM
maths are re-derived from `trainer_base.py`, and each function cites the
`main`-branch method it mirrors so the two stay comparable.

## Recipe

**These hyperparameters are an assumption.** The paper pins the task down exactly
but publishes no architecture or optimizer table. The recipe is DUO's CIFAR-10
MDLM one — same backbone, same data — with batch and steps rescaled from
128 × 1.5M to 512 × 375k (the same ~192M images seen, at a batch that fills 8
A100s) and the LR sqrt-scaled from 2e-4 accordingly.

| | |
|---|---|
| backbone | UNet 32.9M — `ch=128`, `ch_mult=[1,2,2,2]`, 2 res-blocks/scale, attention at 16×16, dropout 0.1 |
| vocab | 257 = 256 pixel values + `[MASK]`=256 |
| schedule | cosine, `eps=1e-3`, **no time conditioning** (`--time-cond` enables it) |
| conditioning | none — Sec. 6.1 is unconditional |
| optimizer | AdamW, lr 4e-4, wd 0, 5k linear warmup, grad-clip 1.0, bf16 |
| batch / steps | global 512, 375k steps |
| EMA | 0.9999 — all sampling, FID and archival checkpoints use the EMA weights |

Time conditioning is off because MDLM (Sahoo et al. 2024) shows `x_t` already
reveals `t`, and DUO trained its CIFAR-10 masked checkpoints the same way.

## Logging

Scalars and sample grids stream to underdeep (`generative-models/cifar10_mdlm`,
set by the launcher; `--underdeep-experiment` to change, `--no-underdeep` to
disable). An experiment that does not exist yet is created on first use, so any
`project/experiment` name works without a manual setup step:

- every `--log-every` (100) steps: `train/nll_per_token` (the loss that is
  backpropagated), `train/bpd` (the same in bits), `train/lr`,
  `train/grad_norm`, `perf/steps_per_sec`
- every `--sample-every` (1000) steps, on EMA weights: an 8×8
  `samples/nfe128` image grid. Cheap — 64 samples on one rank, no FID
- every `--eval-every` (10k) steps, on EMA weights: `val/bpd` and `fid/nfe128`

`train/bpd` starts around **11** (the truncated-logistic head is not uniform at
init) and must fall below **8.0** — the uniform-over-256-bins baseline — within
the first few dozen steps. Sitting at or above 8.0 means something is wrong.

## Checkpoints and restarts

Follows `~/yrDiT/train_imagenet_accelerator.py`, under `<out>/checkpoints/`:

| file | written | contents |
|---|---|---|
| `latest.pt` | every `--ckpt-every` (2k), rewritten in place | model + EMA + optimizer + step + args |
| `ema_<step>.pt` | every `--ema-ckpt-every` (50k), never overwritten | EMA weights only |
| `metadata.pt` | once | the underdeep run id |
| `fid_ref.pt` | once | cached Inception statistics of the train split |

A restart is just re-running the same command. At startup the Nirvana snapshot is
pulled back into the run directory, `latest.pt` restores model/EMA/optimizer/step,
the dataloader epoch is advanced so the shuffle does not replay, and the underdeep
run is resumed from `metadata.pt` rather than a fresh one being opened. Off
Nirvana the snapshot transfers are no-ops and the same command trains normally.

## Nirvana

`scripts/nirvana_cifar10_mdlm.sh` is the entrypoint. Two constraints that are
easy to trip over:

- **No `${...}` anywhere in that file, comments included.** Nirvana renders
  `run-command` through FreeMarker first, which claims dollar-brace as its own
  interpolation syntax and fails the job at render time. Bare `$FOO`, `$(cmd)`
  and `$((a / b))` are safe.
- **CIFAR-10 ships with the code archive**, in `cache_dir/cifar10/cifar-10-batches-py`.
  It is gitignored (178 MB) but must ride along in whatever gets uploaded.
  Without it, rank 0 pulls 170 MB through the proxy at ~500 kB/s on *every*
  fresh attempt. Stage it with:

  ```bash
  mkdir -p cache_dir/cifar10
  cp -r ~/data/cifar10/cifar-10-batches-py cache_dir/cifar10/
  ```

  The script prints whether the data is present before launching, so a silent
  download does not get mistaken for a hung job.

## Cost note

`torchmetrics`' FID does a 2048×2048 matrix square root on CPU, which takes a few
minutes per call regardless of sample count. It runs once per validation, so keep
`--eval-every` at 10k or above; lower it and FID will start to dominate the run.
