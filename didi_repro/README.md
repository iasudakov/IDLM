# DiDi-Instruct: pinned upstream training with Underdeep

This is an **isolated reproduction of the released code**, not an MMD objective
or a reimplementation inside `algo.MDLM`. It runs upstream `main.py`, model,
dataloader, training step and sampler from commit
`5d39f84973ad19168210bfb0de61450a946a1cbc` of
[haoyangzheng-ai/didi-instruct](https://github.com/haoyangzheng-ai/didi-instruct).
The 43 files in `vendor/didi_instruct` are byte-identical to that revision.
`UPSTREAM.json` records their SHA-256 hashes; startup verifies every file.
The original MIT license and dependency specification are included. Demo GIFs
and repository metadata are omitted. All adaptations live in `didi_repro`.

## Train from the authors' teacher checkpoint

Use the **native MDLM teacher `.ckpt` linked in the upstream README** (the Google
Drive teacher link under “Prepare the Teacher Model”), or an independently
verified equivalent native checkpoint. This is not the released DiDi student.
Do not substitute `kuleshov-group/mdlm-owt` just because the architecture name
matches: those weights have not been verified equivalent to the authors' teacher.
Teacher `backbone.*` keys and shapes are checked strictly and its file SHA-256 is
recorded. The raw checkpoint weights are used, as in upstream's training loader.

Create the original environment, then install/configure Underdeep there as for
the existing OWT runs:

```bash
conda env create -n didi-upstream -f vendor/didi_instruct/environment.yml
conda activate didi-upstream

TEACHER_CHECKPOINT=/absolute/path/mdlm.ckpt \
OWT_CACHE=/absolute/path/cache_dir \
SAVE_DIR=/absolute/path/owt_didi \
GPUS=0,1 \
RUN_NAME=didi-instruct-owt \
bash scripts/train_didi_owt.sh
```

Use a **new** save directory. `GPUS` chooses visible GPUs; as in the upstream
script, global batch size is the visible GPU count, with one real example per
GPU and eight generated candidates per example. To reproduce a particular
published experiment, its GPU count/effective batch must also match. The exact
original cache names are required:

```text
openwebtext-train_train_bs1024_wrapped.dat/
openwebtext-valid_validation_bs1024_wrapped.dat/
```

The launcher checks cache existence; the runtime refuses corrupt caches rather
than entering upstream's destructive delete/rebuild fallback. It does not
retokenize or download a dataset automatically. Matching filenames alone does
not establish that the cache contents equal the authors' dataset; retain the
same split, tokenizer and preprocessing.

`DIDI_DRY_RUN=1` prints the launch command without training. The equivalent Python
entry is `python -m didi_repro`; the shell uses `torch.distributed.run` just like
upstream's `torchrun`. Location/tracker overrides are passed to upstream Hydra:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 --module didi_repro \
  training.finetune_path=/absolute/path/mdlm.ckpt \
  data.cache_dir=/absolute/path/cache_dir \
  checkpointing.save_dir=/absolute/path/owt_didi \
  hydra.run.dir=/absolute/path/owt_didi \
  loader.batch_size=1 loader.global_batch_size=2 trainer.val_check_interval=1000 \
  +underdeep.experiment=generative-models/owt \
  +underdeep.name=didi-instruct-owt
```

Every additional argument to the launcher is an explicit Hydra override and is
recorded. Overriding original training settings is possible but no longer the
default reproduction recipe. In particular, do not copy MMD accumulation,
learning rates, reward kernels, batch sizes or EMA settings into this command.

## Optional evaluation before training

Append this flag to the training command or launcher:

```bash
+didi_eval.before_training=true
```

This runs the normal validation loop **after checkpoint initialization and before
the first optimizer update**. The existing validation metrics, generated samples,
and enabled Underdeep PPL/entropy/GM diagnostics are logged in the same run at
step0 for a fresh run. Periodic validation during training is unchanged.
The pre-training evaluation uses `trainer.limit_val_batches`, `sampling.*` and
the same `eval.*` / `didi_monitor.*` settings as normal validation. In particular,
the default is1000-step ancestral sampling, not the guided16-step preset.
`trainer.limit_val_batches=0` is rejected when this option is enabled.

Evaluation performs no optimizer updates, restores RNG states and module modes,
and keeps the Underdeep run open for training. When resuming, it evaluates the
resume checkpoint and logs at its saved optimizer step, rather than evaluating
the fresh teacher at step0. The option defaults to **false**.

## Optional warm start from the authors' distilled HF model

Append:

```bash
+didi_init.student_hf=haoyangzheng/didi-instruct-small
```

The default source is **null**, preserving the teacher-initialized reproduction.
With this option enabled, the two safetensors files are fetched from Hugging Face
and cached automatically. The authors' release is pinned by default to revision
`2e9a21a63cc727a1e703370d55de13b842c69076`:

- `model.safetensors` initializes both the trainable student and its EMA copy.
- `discriminator.safetensors` initializes the trained discriminator, including
  its head and spectral-normalization buffers.
- `training.finetune_path` still supplies the **separate frozen MDLM teacher**
  used for KL regularization. The distilled student never replaces this teacher.

Keys and tensor shapes must match the native model before weights are applied.
No HF remote Python code is executed. The resolved revision and SHA-256 of both
weight files are recorded in checkpoint metadata and the Underdeep configuration
table. This is a **warm-start/fine-tuning experiment**, not a fresh reproduction
from the teacher: optimizers, learning-rate schedule, and iteration counters
start anew. All original training hyperparameters otherwise remain unchanged.
Do not combine HF initialization with `checkpointing.resume_from_ckpt=true`;
when resuming an existing run, omit the initialization option instead.

For a downloaded snapshot, supply its directory containing both weight files:

```bash
+didi_init.student_hf=/absolute/path/to/didi-hf-snapshot
```

Optional controls are `+didi_init.revision=<revision>`,
`+didi_init.cache_dir=/absolute/path/hf-cache`, and
`+didi_init.local_files_only=true`. For an ordinary local directory, file hashes
are recorded without claiming a verified HF revision.

Example combining both options on eight GPUs:

```bash
TEACHER_CHECKPOINT=/absolute/path/mdlm.ckpt \
OWT_CACHE=/absolute/path/cache_dir \
SAVE_DIR=/absolute/path/owt_didi_hf_finetune \
GPUS=0,1,2,3,4,5,6,7 \
RUN_NAME=didi-hf-finetune-with-baseline \
bash scripts/train_didi_owt.sh \
  +didi_eval.before_training=true \
  +didi_init.student_hf=haoyangzheng/didi-instruct-small
```

## Optional temperature-sweep Pareto curves

Append these options to the training command or launcher:

```bash
+didi_pareto.every_n_validations=1 \
'+didi_pareto.temperatures=[0.75,0.80,0.85,0.875,0.90,0.925,0.95,0.975,1.00,1.05,1.10,1.20]' \
'+didi_pareto.steps=[8,16,32]' \
+didi_pareto.seeds=5
```

This adds a sweep after each regular validation. Underdeep receives:

- `pareto/temperature_frontier`: PPL versus token-histogram entropy, one curve
  per sampling-step budget, varying temperature along each curve.
- `pareto/temperature_by_checkpoint`: comparison of curves across checkpoints.
- `pareto/temperature_values`: exact means, across-seed sample standard
  deviations, temperatures, budgets, sample counts and training steps.
- `val/gen_ppl_s{steps}_t{temperature}` and `val/entropy_s{steps}_t{temperature}`:
  each operating point tracked over training.

With `+didi_eval.before_training=true`, the sweep also logs initialized-model
curves at step 0, or the saved step when resuming. This baseline does not consume
the periodic-validation cadence. `every_n_validations=0` disables all sweeps
(the default); larger positive values reduce their frequency.

The default sweep sampler is the current `sampling.predictor`. To evaluate with
the released guided sampler without changing ordinary validation sampling, add:

```bash
+didi_pareto.sampler=guided
```

Guided evaluation requires a useful trained discriminator; its initial-teacher
baseline includes a newly initialized discriminator unless using the HF warm
start. Guided budgets include extra discriminator and terminal denoising calls,
so they are not equal-compute budgets with unguided MMD sampling.

Temperature applies only to the denoiser's token probabilities, before guided
tilting/reranking; discriminator scores, guidance schedule and noise schedule
are unchanged. **Temperature other than 1 is an optional evaluation extension,
not an original released DiDi setting.** Temperature 1 takes the original
sampling path exactly. Training equations and ordinary validation are unchanged;
the extra sweep restores RNG states, module modes, sampler and forward method.

Sweeps generate additional samples. By default each GPU generates
`didi_pareto.num_batches=2` batches of `loader.eval_batch_size` sequences per
temperature/budget/seed; `+didi_pareto.batch_size=...` overrides only the sweep's
per-GPU batch size. On eight GPUs with the original eval batch size 4, this is
64 samples per seed/point. The 12-temperature, 3-budget, 5-seed example above
generates **11,520 extra sequences per validation** and can be expensive.
Start with fewer temperatures/seeds while checking the setup. The built-in
temperature list is `[0.75,0.85,0.95,1.0,1.05,1.2]`, with one seed.

Seeds are paired across temperatures and checkpoints. All ranks' samples are
gathered for a separate rank-zero PPL evaluator; it does not reset upstream
validation metrics. GM remains part of the ordinary monitor, not each sweep
point. `didi_monitor.enabled=false` does not disable this separate sweep.

History is saved as `didi_pareto_temp_history.json` in the checkpoint save
directory. Compatible history is reused on resume; different tracker runs or
evaluation protocols are kept separate. Diagnostic failures are reported through
`didi_pareto/failed` instead of publishing incomplete curves. Optional
`+didi_pareto.reference=/absolute/path/reference.json` overlays an existing
OWT-compatible reference curve; `+didi_pareto.keep_samples=2` controls the number
of retained sample texts per point from its first seed. These options apply to training
validation (including pre-training validation), not `mode=sample_eval`.

## Settings preserved from the released code

The authoritative files are the vendored YAMLs and `algo.py`, not this summary.
Constructor defaults that are missing from YAML are also logged as effective
hyperparameters.

| Setting | Released implementation used here |
| --- | --- |
| Model | Native DiT small, 12 blocks, width768, length1024, dropout0.1 |
| Batch and groups | 1 real example/GPU, G=8, no accumulation |
| Student/discriminator LR | Both `1e-6` |
| AdamW | Betas0.9/0.999; student weight decay0.01, discriminator0 |
| Precision | bf16 |
| Training schedule | Two-step generation; independent candidate times, `beta11`, clipped0.02–0.98 |
| Discriminator | Spectral-normalized per-token head; last4 blocks trainable; label smoothing0.1 |
| Reward | Masked log-odds, clip8; group sample standard deviation (`G-1`), advantage clip3 |
| Student loss | PG + `0.05*(0.01*KL_hi + 0.01*KL_lo)` − `0.0005*entropy` |
| Gradient clipping | Discriminator1.0, student0.5 |
| LR schedule | Linear warmup1000, then linear decay, floor`1e-8` |
| Training limit | Original Lightning `max_steps=10000`; normally two optimizer steps/iteration |
| Validation | Every1000 batches, 8 validation batches |
| Validation sampler | Original `ancestral_cache`,1000 steps,4 samples/GPU ×2 batches |
| Student EMA | Updated with0.999; original generator evaluation reads raw `backbone` |

Several original comments disagree with their executable code: `kl_beta` is
actually used; the scheduler is linear rather than cosine; the `unbiased` and
`coupled_traj` YAML flags do not select a different training implementation.
These are **not silently “corrected” into another method**. The original
denoiser-based two-step log-probability formula and independent corruption masks
are retained. Released-code fidelity is not a guarantee that its defaults
reproduce the paper's exact reported numbers.

## Underdeep metrics

Original scalars are retained, including `trainer/loss_discriminator`,
`trainer/loss_student_pg`, `trainer/loss_student_total`,
`trainer/loss_student_entropy_bonus`, both learning rates, discriminator accuracy
and gradient norms. Additional scalars expose `reward/mean`, `reward/std`,
clipping fraction and advantage magnitude. `didi/iteration` distinguishes
training iterations from Lightning's two-optimizer global-step counter.

The monitor captures **only samples already generated by upstream validation**.
After gathering those across ranks it uses the existing OWT diagnostic evaluator
to log `val/gen_ppl`, `val/sample_entropy`, `val/gm`, `val/gen_ppl_gm`,
`val/data_ppl_gm` and the GM auxiliary metrics, plus sample tables. There are no
extra student sampling calls or MMD rewards in this monitor. The separately
opt-in Pareto callback above does generate extra samples. GM is skipped
explicitly if there are too few captured samples/reference rows. Failures are
reported as `didi_monitor/failed=1`, not misrepresented as successful evaluation.

Extra diagnostics preserve Python, NumPy and Torch random states and do not
change training gradients or sampling modes. The reference evaluator is moved
off GPU afterward. Underdeep calls are also wrapped in RNG preservation.
These diagnostics add evaluation time and may need to download GPT-2 Large.
Both diagnostic evaluators disable TorchMetrics' public and cached distributed
sync flags, since only rank zero computes metrics on the gathered samples.

All resolved settings (including strings/lists), teacher hash, optional HF
student/discriminator initialization hashes and revision, effective
constructor defaults, upstream revision and installed versions are recorded in
Underdeep's `didi/config` table and local files:

- `didi_resolved_config.yaml`
- `didi_environment.json`
- `didi_provenance.json` (including every command-line override)
- `didi_monitor_config.json`

Original `val/nll`, `val/ppl`, and `val/bpd` still come from upstream validation.
Training-validation PPL is **not** the paper's guided-sampling result. For that
sampler use the separate evaluation command below.

Optional integration-only flags, without changing the training objective:

```bash
+didi_monitor.gm=false          # omit expensive GM; retain PPL/entropy
+didi_monitor.enabled=false     # omit all additional validation diagnostics
+underdeep.required=false       # allow missing tracking; does not disable an available tracker
```

Underdeep is required by default and initializes only after model/checkpoint
setup succeeds; it must not silently train without logging.

## Official guided evaluation preset

```bash
CHECKPOINT=/absolute/path/owt_didi/checkpoints/last.ckpt \
SAVE_DIR=/absolute/path/didi_guided_eval \
GPUS=0 \
bash scripts/eval_didi_owt.sh
```

This preserves the authors' evaluation-script preset:16 steps, guided sampling,
20 samples/batch and2 batches. Both gradient tilting and four-candidate stochastic
reranking are preserved; guidance is0.2→1.0. The final denoising call and
discriminator evaluations are additional work: “16 steps” is not16 total model
evaluations and should not be presented as equal compute to16-step unguided MMD.
The original evaluator writes `samples.json`; its PPL/entropy and samples are
also published to Underdeep. Evaluation requires a trained discriminator.

For normal training checkpoints, generation still uses raw `backbone` weights,
as released. For the authors' HF-converted exports containing **only**
`student_ema.*`, those tensors are explicitly loaded into the otherwise-missing
backbone; they are not silently ignored.

## Complete repair/integration ledger

The original files remain unchanged. `runtime.py` / `entrypoint.py` apply these
explicit repairs in memory with exact source-match assertions:

1. Retain `_get_tau`'s `alpha_gen` and pass it to corruption. Upstream passes
   `None` and later references an undefined variable; the original crashes.
2. Enable Lightning manual optimization for the already-manual two-optimizer
   loop. Supply the omitted MDLM `loss_type=elbo` runtime attribute.
3. Strictly load the teacher's raw native backbone into student, frozen teacher,
   discriminator backbone and EMA copies. Upstream copies random weights before
   loading only the student. Preserve the newly initialized discriminator head.
4. Correct only the **logged** total loss to include `kl_beta`, matching the loss
   already differentiated upstream. Optimization math is unchanged.
5. Permit input autograd in guided evaluation under Lightning inference mode;
   use `autograd.grad` for the same input gradient without contaminating model
   parameter `.grad` buffers. Reject the original undefined one-step guided
   schedule. Move loaded CPU checkpoints to CUDA for FlashAttention.
6. Save the frozen teacher and DiDi iteration counter and retain Lightning's
   manual optimizer progress. Do not overwrite it from the unused automatic
   loop or silently restore a random teacher. Reject missing requested resume
   checkpoints. Native checkpoint loads explicitly use `weights_only=False`;
   only load checkpoint files you trust.
7. Populate the backbone when loading an official EMA-only evaluation export;
   reject missing generator/discriminator tensors instead of ignoring them.
8. Replace W&B with the existing Underdeep transport plus additional diagnostic
   hooks/provenance. No change to training RNG is introduced by these synchronous
   logging/diagnostic calls.
9. Fail on cache errors instead of executing upstream's delete/rebuild fallback.

The optional `didi_eval.before_training`, `didi_init.student_hf`, and
`didi_pareto` extensions are disabled by default. Their orchestration, HF weight
loading and sweep implementation live in `pretrain_eval.py`,
`initialization.py`, `pareto.py` and `temperature.py`; the vendored code and
original training equations remain unchanged.

Resume restores models, optimizer state and schedule counters. **Bitwise
mid-epoch trajectory replay is not claimed**: upstream's shuffled loader does not
checkpoint its exact data cursor or all RNG state. For strict experimental
comparisons use fresh uninterrupted runs from the recorded teacher/seed.

The startup core-version guard checks the original PyTorch2.5.1,
Lightning2.5.3, Transformers4.55.4, FlashAttention2.8.3 and Python3.12 versions.
Use the complete upstream environment file for the remaining dependencies.
`+didi_runtime.strict_environment=false` is available only as an explicit
non-reproduction development/smoke-test override; the installed versions remain
recorded. Do not report an overridden environment as an exact reproduction.

## Verification

```bash
python scripts/test_didi_runtime.py
python scripts/test_didi_monitor.py
python scripts/test_didi_entrypoint.py
python scripts/test_didi_initialization.py
python scripts/test_didi_pretrain_eval.py
python scripts/test_didi_temperature.py
python scripts/test_didi_pareto.py
python scripts/test_didi_ddp.py  # optional two-rank CPU/Gloo smoke test
torchrun --standalone --nproc_per_node=2 scripts/test_didi_pretrain_eval.py --ddp-smoke
torchrun --standalone --nproc_per_node=2 scripts/test_didi_pretrain_eval.py --ddp-smoke --pareto-smoke
torchrun --standalone --nproc_per_node=2 scripts/test_didi_pareto.py --ddp-smoke
```

These offline tests use tiny CPU networks and mocked tracking, exercise the
actual patched upstream objective, verify frozen teacher/optimizer ownership,
checkpoint restoration, guided-sample equivalence, source/config fidelity and
diagnostic isolation. They do not establish the paper's final model quality.
