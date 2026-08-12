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

`mdlm/cosine_time_uncond/1500000`, unconditional, `ancestral_cache`, **10 000**
samples against 10 000 train images (not the 50 000-sample protocol — these are
biased upward and are the number to compare the student against, not Table 1):

| steps | 16 | 32 | 64 | 128 | 1024 |
|---|---|---|---|---|---|
| FID | 151.6 | 79.4 | 51.5 | 41.0 | 34.6 |
| IS | 3.47 | 5.19 | 6.03 | 6.58 | 6.86 |

The samples at 1024 steps are clearly recognisable CIFAR-10 objects (see
`outputs/cifar10/teacher_fid/samples_steps1024.png`), so the pixel layout,
vocabulary, schedule and sampler are right; the offset from Table 1 is the
checkpoint, not the port. Two known contributors, in order of size:

1. **The null class saw ~10% of training.** DUO trains CIFAR-10 with
   `training.class_dropout_p = 0.1` and its own `_eval_fid` samples *class
   conditionally*, with labels taken from the real batch. Sampling
   unconditionally — which is what D-MMD Sec. 6.1 reports and what this branch
   does — runs through an embedding that was trained on a tenth of the data.
   `scripts/test_cifar10_null_class.py` measures this directly by running the
   same sampler twice, changing only which class embedding it reads.
2. **10 000 samples and a 10 000-image reference.** FID is biased upward at
   both; the standard protocol is 50 000 of each (`NUM_SAMPLES=50000`, ~4.5 h
   at 1024 steps on 5 GPUs).

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
