# CIFAR-10 D-MMD (branch `cifar10-dmmd`)

The D-MMD paper's *first* experiment (arXiv:2603.20155, Sec. 6.1, Tables 1 & 4)
is CIFAR-10: a 32x32x3 image is 3072 tokens over a 256-value vocabulary, scored
by FID. This branch makes that experiment runnable with the existing D-MMD logic
(`algo.MDLM_DMMD`) untouched in substance — everything added here is the
surrounding pixel plumbing.

Nothing on the text side changes behaviour: `LogLinear()` is still constructed
exactly as before, `_NoiseConditionedBackbone` is unchanged, and every new code
path is gated on `data.modality=image` or `algo.backbone=unet`.

## The three stages

### 0. Download the teacher (no pretraining needed)

```bash
bash scripts/download_cifar10_teacher.sh
```

This step is optional: a `teacher_ckpt_path` under `downloads/` that is not on
disk is fetched from the hub on first use (`algo._resolve_teacher_ckpt_path`),
into the same place this script would have put it. Run the script to pre-fetch
it, or to pick a variant without launching training.

`jdeschena/duo2-cifar10` (released with the DUO repo) holds two families:
`duo/…` = uniform-state diffusion, and **`mdlm/…` = masked / absorbing-state
diffusion** — the process `MDLM_DMMD` distills. Variants are
`mdlm/{cosine,log_linear}_time_{cond,uncond}/{100000,500000,1000000,1500000}.safetensors`;
the default is `mdlm/cosine_time_uncond/1500000.safetensors`, 35.76M EMA
parameters at 1.5M steps.

That the `mdlm/` family really is masked is corroborated three ways: DUO's own
`scripts/train_cifar10_mdlm_cosine.sh` trains it with `algo=mdlm
algo.backbone=unet noise=cosine data=cifar10`; DUO's `get_tokenizer` adds the
mask token only when `'mdlm' in config.algo.name`, so these weights carry the
257-bin head whose bin 256 *is* `[MASK]`; and IDLM's D-MMD path
(`trainer_base.AbsorbingState`, `mask_index = 256`) is that same process.

**The uniform (DUO) D-MMD arm of the paper (App. D) is out of scope** — the
`duo/` checkpoints exist, but IDLM has no uniform D-MMD loss. `algo=mdlm_rl` is
also out of scope (it needs an autoregressive critic); both error clearly.

**This checkpoint is class-conditional; the paper's teacher is not.** DUO trains
it with `training.class_dropout_p=0.1` and a `LabelEmbedder`, whereas Sec. 6.1
says "we train diffusion models to generate unconditional images". Sampling it
unconditionally runs through the null-class embedding, and sampling it
class-conditionally is not the paper's setting either. Use it to exercise the
pipeline; train the teacher below to reproduce the paper.

### 0b. Train the teacher yourself (the paper's unconditional setting)

```bash
bash scripts/train_cifar10_teacher.sh
```

`data.num_classes=null` drops the `cond_map` entirely — no null class, nothing
conditional in the graph. Everything else is DUO's CIFAR-10 MDLM recipe (35M
UNet, global batch 128, lr 2e-4, 1.5M steps, cosine schedule, EMA 0.9999),
because the paper gives no hyperparameter table and that recipe is the same
backbone on the same data. Measured throughput is ~9 optimizer steps/s and it is
flat from 4 to 6 A100s (the step is per-step-overhead bound, not compute bound),
so 1.5M steps is ~46 h and four GPUs are enough — use `GPUS=` to pin devices and
leave the rest free.

Two overrides in that script are *required* rather than tuning, and both are
hard errors rather than warnings:

* `~callbacks.checkpoint_monitor` — it monitors `val/gen_ppl`, a text-only
  metric. The image path logs `val/bpd`, `val/nll`, `val/ppl`, `val/fid`,
  `val/inception_score`, `val/sample_entropy`, `val/pred_entropy`; Lightning
  raises `MisconfigurationException` on the first validation when the monitored
  key is absent, so with this callback in place the run dies at
  `trainer.val_check_interval`. `scripts/train_cifar10_dmmd.sh` needs the same
  override for the same reason.
* `callbacks.checkpoint_every_n_steps.save_top_k=0` — `save_last: True` still
  rewrites `last.ckpt`, but the per-interval `epoch=..-step=...ckpt` files are
  not kept. They are 572 MB each; keeping all of them fills the disk long before
  1.5M steps.

#### Restartable on a preemptible machine (Nirvana)

`nirvana_utils.py` is vendored from `yt_tools/nirvana_utils.py` (what
`yrDiT/train.py` uses), so the repo is self-contained on the Nirvana image. The
only state a restarted attempt inherits is the *snapshot path*, so the run
directory is pushed there periodically and pulled back at startup. Off Nirvana
`nirvana_dl` is not importable and every transfer is a no-op, so the identical
command trains normally on a local machine.

Three intervals, all env-overridable:

| env | default | what it buys |
|---|---|---|
| `CKPT_EVERY` | 2500 | rewrites `checkpoints/last.ckpt` in place — weights, EMA, optimizer, dataloader position (572 MB) |
| `SNAPSHOT_EVERY` | 10000 | pushes the run dir to the snapshot path. A preemption costs at most this many steps (~18 min). Keep it a multiple of `CKPT_EVERY` |
| `EMA_EVERY` | 100000 | archival `checkpoints/ema_<step>.safetensors` (143 MB, never overwritten) — the intermediate teachers, in exactly the format `adversarial_distill.teacher_ckpt_path` loads |

Restart semantics, all verified by wiping the run directory between two attempts
and checking the resumed step:

* the snapshot is restored *before* `checkpointing.resume_ckpt_path` is resolved,
  otherwise the recovered `last.ckpt` would arrive too late to be used;
* the underdeep run continues rather than forking a new one, because
  `underdeep_metadata.pt` lives in `save_dir` and so travels in the snapshot;
* the periodic push is deferred by one batch on purpose. Lightning reorders
  callbacks to put `ModelCheckpoint` last, so at the end of step N the checkpoint
  for step N is not on disk yet — pushing there ships the *previous* interval's
  checkpoint and silently loses an interval on every restart. This was a real bug
  caught by the restart test: a run that reached step 100 resumed from step 50.

#### Paper-comparable FID during training

`val/fid` is **not** the Table 1 quantity and the two must not be plotted
together: it scores a few hundred generated images against the *validation*
split, while the paper scores 50 000 against the un-augmented *train* split.
Reference split, reference size and sample count all move FID — the same teacher
measured 57.7 at n=2000 and 41.0 at n=10000.

`eval.fid_paper_every_n_steps` runs the Table 1 computation itself (same code
path as `mode=fid_eval`: train split, un-augmented, EMA weights) and logs
`fid_paper/steps<N>` plus its Inception Score:

```
  eval.fid_paper_every_n_steps=25000 \
  eval.fid_paper_num_samples=10000 \
  'eval.fid_paper_steps=[128]' \
```

Cost is `num_samples x steps` denoiser forwards and comes out of the training
budget: ~3 min per evaluation at 10k/128 on 8 GPUs (~40 min over a 375k-step
run, ~4%), versus ~11 min at the full 50k (~18%). 10k is the floor at which FID
still means anything; it is biased *upward* relative to the paper's 50k by a few
points, so treat a 10k curve as an upper bound and run the real sweep offline at
the end against the `ema_<step>.safetensors` exports.

Each entry in `fid_paper_steps` is a separate full evaluation (its own pass over
the real images), so keep the list short during training — one step count is
enough to see the trend — and sweep 1024/512/128/64/32/16 offline.

Two implementation details that were wrong in the first version and are worth not
reintroducing: the metric is written with `logger.log_metrics(..., step=...)`
rather than `pl_module.log(on_step=True)`, because the latter buffers until the
next `log_every_n_steps` flush and so records an occasional metric at the wrong
step and re-emits stale values; and the evaluation resets `ThroughputMonitor`'s
window afterwards, since minutes spent sampling are not training speed and
otherwise drag `trainer/steps_per_sec` and `trainer/eta_hours` down with them.

#### What reaches the tracker

`logger=underdeep` is the default (`logger=tensorboard` still works). Per
validation: `val/fid`, `val/inception_score`, `val/bpd`, `val/nll`, `val/ppl`,
`val/sample_entropy`, `val/pred_entropy`, plus a PNG sample grid via
`log_image`. Per `trainer.log_every_n_steps`: `trainer/loss`, `trainer/lr`,
`trainer/steps_per_sec`, `trainer/eta_hours` — the last two exist because
`trainer/loss` cannot distinguish "training normally" from "training at a third
of the speed on a contended host", which over a multi-day preemptible run is the
difference between finishing and not.

### 1. Teacher baseline

```bash
bash scripts/fid_cifar10_teacher.sh                        # 1024/128/64/32/16 steps
STEPS_LIST=1024 NUM_SAMPLES=10000 bash scripts/fid_cifar10_teacher.sh   # quick
```

The 1024-step point doubles as the port test: if the pixel layout, vocabulary,
noise schedule or sampler were wrong, FID would be in the hundreds rather than
the single digits.

### 2. Distill

```bash
bash scripts/train_cifar10_dmmd.sh
```

### 3. Student sweep

```bash
CKPT=outputs/cifar10/cifar10_dmmd/checkpoints/last.ckpt bash scripts/fid_cifar10_dmmd.sh
```

## Knob → paper section

| Knob | Paper | Note |
|---|---|---|
| `adversarial_distill.critic_type=diffusion` | Sec. 3 | **Required** for images. `ar` is a GPT-2 critic and errors out. |
| `adversarial_distill.teacher_ckpt_path` | Alg. 1 | Student, frozen teacher and auxiliary all start from these weights. Loaded *before* the critics are deep-copied — `training.finetune_path` cannot do this, since Lightning restores after `__init__`. |
| `adversarial_distill.dmmd_mask_sampling` | Alg. 1 | `posterior` / `renoise`. |
| `adversarial_distill.top_p` | Sec. 3.3 | Top-p teacher distillation. |
| `adversarial_distill.fake_soft_target` | Sec. 3.2 | Soft vs. hard auxiliary target. Valid for masked diffusion only. |
| `adversarial_distill.fake_teacher_weight` | Eq. 12 | Auxiliary→teacher anchor. |
| `adversarial_distill.noise_dim` | **Sec. 6.5 / Table 6** | 2D Gaussian noise pyramid into the `input_conv` residual. Table 6: FID **151 → 22.3** at 4 steps for masked image D-MMD. This is the single most important knob here. |
| `noise=cosine` | — | Matches the `mdlm/cosine_*` checkpoints. |
| `eval.fid_num_samples` | Sec. 6.1 | 50 000 = the standard protocol; below ~10 000 FID is biased upward. |

## Paper targets (Table 1, masked)

| steps | 4 | 8 | 16 | 32 | 64 | 128 | 1024 |
|---|---|---|---|---|---|---|---|
| teacher | – | – | 122.9 | 47.1 | 20.0 | 11.1 | 6.4 |
| D-MMD student | 22.3 | 12.7 | 5.3 | 3.8 | 3.5 | – | – |

**These are not the numbers to expect verbatim.** The teacher here is DUO's
checkpoint, not the paper's own: a different codebase, trained
class-conditionally with dropout, and sampled through the null class. Stage 1
measures *our* teacher's curve, and that is the honest baseline for the student.

### Measured baseline (this branch)

`mdlm/cosine_time_uncond/1500000`, unconditional, `ancestral_cache`, FID against
the un-augmented train split.

**Paper protocol, 50 000 generated vs 50 000 train:**

| steps | ours FID | ours IS | paper (Table 1) | ratio |
|---|---|---|---|---|
| 16 | **148.92** | 3.48 | 122.9 | 1.21x |
| 32 | **76.56** | 5.17 | 47.1 | 1.63x |

**Cheaper sweep, 10 000 vs 10 000** (what the student should be compared
against if you evaluate it at 10k):

| steps | 16 | 32 | 64 | 128 | 1024 |
|---|---|---|---|---|---|
| FID | 151.6 | 79.4 | 51.5 | 41.0 | 34.6 |
| IS | 3.47 | 5.19 | 6.03 | 6.58 | 6.86 |

The samples at 1024 steps are clearly recognisable CIFAR-10 objects (see
`outputs/cifar10/teacher_fid/samples_steps1024.png`), and the checkpoint loads
with zero key mismatches, so the pixel layout, vocabulary, schedule and sampler
are right. The offset from Table 1 is still only partly accounted for:

1. **Sample count — measured, and large.** The same teacher at 128 steps scores
   57.7 at n=2 000 and 41.0 at n=10 000. FID is biased upward at small n on
   both the generated and the reference side, and the table above uses a tenth
   of the paper's protocol. Rerun with `NUM_SAMPLES=50000` for the comparable
   number.
2. **The null class — measured, and small.** DUO trains with
   `training.class_dropout_p = 0.1` and its own `_eval_fid` samples *class
   conditionally*, so the obvious hypothesis is that our unconditional sampling
   (D-MMD Sec. 6.1) runs through an under-trained embedding.
   `scripts/test_cifar10_null_class.py` tests exactly that, holding the label
   fixed across each trajectory and changing nothing else: **null 57.7 vs
   class-marginal 53.3 at n=2 000/128 steps, a 1.08x ratio.** So this costs
   ~8%, not the several-fold factor it would need to explain Table 1. Do not
   repeat the class-dropout story as the explanation; it was tested and it is
   not the answer.
3. **A different teacher.** These are DUO's weights from DUO's codebase and
   training run, not the paper's own masked teacher. Unquantified.

## The noise pyramid (`_ImageNoiseConditionedBackbone`)

A masked-diffusion generator is a deterministic function of its masked input,
so moment matching can only be minimised by collapsing the output distribution.
The paper's fix (Sec. 6.5) is an input noise source. For images it is a pyramid:
independent Gaussian noise at 32x32, 16x16, … 1x1, each nearest-upsampled to
32x32, `noise_dim` channels per level, mapped by one zero-initialisable 1x1 conv
onto the `input_conv` output. The pyramid matters because a 32x32 i.i.d. field
can only perturb local texture, while the 1x1 level is a global latent.

The public surface is identical to the text wrapper's, so `noise_scale`,
`noise_warmup_steps`, `noise_hold_rms`, `noise_lr`, the `noise/*` diagnostics
and `mode=noise_diversity_eval` all work unchanged. `noise/snr` keeps meaning
`rms(W)`: `noise_out_scale` is calibrated at construction from the *measured*
`input_conv` output std (`UNet.reference_std`, ≈0.66 for the released teacher),
which is the image analogue of the token embedding's per-entry std.

`noise_scale=0.0` (the repo default, and what `train_cifar10_dmmd.sh` uses)
zero-initialises the projection, so the student starts as an exact copy of the
teacher and learns to use the noise only if the objective pays for it. Watch
`noise/proj_norm`: flat means the generator is ignoring the channel and the run
is equivalent to `noise_dim=0`. To force it on from the start use e.g.
`NOISE_SCALE=0.5 NOISE_WARMUP=1000`; note `noise/snr = 1.0` means the injection
is as large as the entire residual stream, which is a very aggressive setting.
`shared_noise=True` collapses the pyramid to its 1x1 level alone (one global
latent per image, no spatial structure).

## Gotchas

* **`_sigma_from_alphat = -log α` diverges under cosine** as `α → 0` (`t → 1`).
  This is harmless only while `algo.time_conditioning=False` zeroes sigma before
  the backbone reads it — which is the default and what the `*_time_uncond`
  checkpoints expect. If you switch to a `*_time_cond` variant, sigma is fed
  through and you will need a finite sigma at `t → 1`.
* **Memory.** The D-MMD min-max loop holds three UNets (student + teacher +
  auxiliary) over 3072-token sequences at once. DUO trains at 32/GPU; expect to
  come down to ~8. Use `strategy.find_unused_parameters=True`.
* **Environment.** conda `mdlm` (python 3.9, torch 2.2.1, torchvision 0.17.1,
  torchmetrics 1.8.2 + torch-fidelity 0.4.0). No 3.10+ syntax.
* **`val/bpd` is bits/dim.** The NLL metrics stay valid for images — the
  sequence is 3072 pixel tokens, so `bpd` is the standard CIFAR-10 likelihood
  number. Only the reference-LM metrics (gen-ppl, GM) are switched off.
* **`val/fid` ≠ `mode=fid_eval`.** The in-training number is a cheap tracker
  measured against the *validation* split with `eval.fid_num_samples_val` real
  images; `mode=fid_eval` follows the paper's protocol (50 000 samples against
  the *train* split). Compare each only against itself.

## Gotchas, part 2: flags the distillation loop needs

These are not new to this branch, but a CIFAR-10 run trips all of them at once
and each one is a hard error rather than a warning. All are already set in
`scripts/train_cifar10_dmmd.sh`:

* `sampling.predictor=ancestral_cache` — `MDLM_with_AR._validate_configuration`
  rejects plain `ancestral` (too slow). For MDLM `ancestral_cache` is
  numerically equivalent: it only skips the denoiser on a step that reveals
  nothing, where re-running it could not change `p_x0`.
* `trainer.gradient_clip_val=null` — the min-max loop uses manual optimization,
  and Lightning refuses automatic clipping there. Clip with
  `adversarial_distill.use_gradient_clipping` instead.
* `strategy.find_unused_parameters=True` — a generator step touches the student
  but not the auxiliary critic, and vice versa.
* `loader.num_workers` — the default is the machine's **full CPU count** (255
  on this host). Every worker forks a copy of the in-memory CIFAR-10 arrays, so
  across ranks that is >1000 processes and enough resident memory to get a rank
  OOM-killed — which then surfaces as an NCCL watchdog timeout on the *other*
  ranks, pointing away from the real cause. The FID scripts pin it to 8.

## Verification

```bash
python scripts/test_cifar10_unit.py           # stage 1: CPU units, no teacher needed
python scripts/test_cifar10_teacher_load.py   # stage 2: strict load, zero key mismatches
python scripts/test_cifar10_null_class.py     # diagnostic: null-class vs class-marginal FID
```
