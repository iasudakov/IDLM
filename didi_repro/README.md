# DiDi-Instruct: pinned upstream training with Underdeep

This is an **isolated integration of the released DiDi code**, not an MMD
objective or a reimplementation inside `algo.MDLM`. It runs upstream `main.py`, model,
dataloader, training step and (by default) sampler from commit
`5d39f84973ad19168210bfb0de61450a946a1cbc` of
[haoyangzheng-ai/didi-instruct](https://github.com/haoyangzheng-ai/didi-instruct).
The 43 files in `vendor/didi_instruct` are byte-identical to that revision.
`UPSTREAM.json` records their SHA-256 hashes; startup verifies every file.
The original MIT license and dependency specification are included. Demo GIFs
and repository metadata are omitted. All adaptations live in `didi_repro`.

There are now two explicitly different protocols:

- Direct `python -m didi_repro`, without the correction flags below, retains
  the released-code defaults plus the documented runtime/checkpoint repairs.
- The supplied launchers select the corrected paper-reported iteration/LR/EMA
  settings and guided evaluation described below. They do **not** silently
  change the vendored files or make earlier runs equivalent to this protocol.

These corrections address identified code-versus-paper mismatches; they do not
establish that every implementation detail reproduces the paper's results.

## Retain evaluated training checkpoints

The DiDi entrypoint supports retaining every evaluated model from an inclusive
Lightning optimizer step. For example:

```bash
+checkpointing.save_evaluated_checkpoints=true \
+checkpointing.save_evaluated_from_step=2000 \
+checkpointing.snapshot=true \
callbacks.checkpoint_every_n_steps.save_top_k=0 \
callbacks.checkpoint_every_n_steps.save_last=false \
callbacks.checkpoint_monitor.monitor=val/nll \
callbacks.checkpoint_monitor.save_top_k=0 \
callbacks.checkpoint_monitor.save_last=false
```

Use `+` for these new checkpointing fields because this entrypoint composes the
pristine upstream config. The feature is disabled unless explicitly enabled.
Enabling it suppresses the upstream `student_iters_*.ckpt` writes inside the
training step; the callback overrides above disable the separate standard
periodic/best-model savers. Baseline and sanity evaluations do not save.
Every training validation qualifies, even when the optional Pareto sweep runs
less frequently than validation.

Files are retained under
`<checkpointing.save_dir>/checkpoints/eval-step=00002000.ckpt`, etc. They include
the raw student, student EMA, discriminator, frozen teacher, optimizer state,
and existing DiDi resume metadata. With the default discriminator/student
update cadence, DiDi performs two optimizer updates per training batch, so
`trainer.val_check_interval=1000` evaluates at steps 2000, 4000, 6000, and so on.
These labels match the logged Pareto step; DiDi iteration numbers are separate.

With `checkpointing.snapshot=true`, rank zero uploads the entire save directory
through the Nirvana helper after each evaluated checkpoint is fully written.
Outside Nirvana the upload is a no-op, and transfer failures are logged without
aborting training. This option does not add automatic snapshot restoration or
choose a resume checkpoint: DiDi's existing resume path must already be locally
available when resuming. No `last.ckpt` is created by the evaluated saver.

## Standalone control using the dedicated authors' inference release

The authors also publish a **different `inference` branch**, pinned here to
[`eb9f96c78ddcf3c9d601fbde1744eba45238b3c7`](https://github.com/haoyangzheng-ai/didi-instruct/tree/eb9f96c78ddcf3c9d601fbde1744eba45238b3c7).
Its 42 pristine files live separately in `vendor/didi_inference`, verified by
`INFERENCE_UPSTREAM.json`. Training and the default `didi_repro` sampler still
use the main-branch release. The opt-in integrated evaluator below selects the
inference sampler without switching the training implementation.

This inference sampler reads `student_ema` directly, ramps gradient guidance
from **30 to 40 over the first half**, then sets gradient tilting to zero while
retaining four-candidate categorical reranking with `softmax(reward)`. The
direct EMA call also preserves this release's timestep-conditioning path.
Changing main's two guidance overrides alone does not reproduce this sampler.
The original final ancestral denoising call still reads `backbone`; both weight
sets must be present. No sampling/model/metric equations are patched here.

Run from this repository's updated `didi` checkout, in the original environment:

```bash
# Once, if the upstream environment is not already installed:
conda env create -n didi-inference -f vendor/didi_inference/environment.yml
conda activate didi-inference

# Once per new checkpoint path. Downloads the pinned HF generator/discriminator;
# this is an explicit HF format adapter, NOT a downloaded native Drive checkpoint.
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_didi_authors_checkpoint.py \
  --upstream inference \
  --output /absolute/path/didi-authors-inference.ckpt --device cuda:0

GPUS=0 CHECKPOINT=/absolute/path/didi-authors-inference.ckpt \
SAVE_DIR=/absolute/path/didi-inference16-seed1 \
bash scripts/eval_didi_inference.sh seed=1

CUDA_VISIBLE_DEVICES=0 python scripts/score_didi_authors_samples.py \
  --upstream inference \
  --samples /absolute/path/didi-inference16-seed1/samples.json \
  --output /absolute/path/didi-inference16-seed1/all40.json \
  --batch-size 4 --device cuda:0
```

This uses **one GPU, 16 guided steps, 2 batches of 20, no temperature override**.
It neither trains nor logs to Underdeep, and needs no MDLM teacher or OWT dataset.
It refuses existing output directories and overwriting a prepared checkpoint;
reuse an already prepared valid checkpoint or choose a new path.
An original native authors checkpoint can be passed as `CHECKPOINT` instead,
provided its complete backbone/EMA/discriminator states and saved `cuda:0`
device tags pass validation. They are checked separately and never overwritten.
Only load trusted native checkpoints because the upstream loader uses pickle.

The source, checkpoint hash/validation, environment and original console result
are recorded in `authors_eval_environment.json`, `console.log` and Hydra's config
files. The original PPL still scores **32/40** samples; the separate `all40.json`
rescoring covers every saved sample without changing the original output. The
40-sample entropy is identical between these reports. The scorer is offline and
uses GPT-2 Large already cached by the original evaluation.

The shell permits only `seed=...`, `sampling.steps=...` and
`eval.perplexity_batch_size=...` overrides. Use a fresh SAVE_DIR for 8/32-step or
additional-seed controls. If overriding metric batching, provide matching count
assumptions to the rescorer's `--original-perplexity-batch-size` option.
Keep `DIDI_STRICT_ENVIRONMENT` at its default `1` for the controlled comparison;
setting it to `0` labels a mismatched-environment diagnostic, not reproduction.
The core guard is not a complete Conda/system-build check. Likewise, preserving
upstream `use_float64=true` is not a claim all internal model computation is
float64; the original DiT contains bf16 autocast.

First compare the 16-step result with the paper's approximately PPL 38.188 /
entropy 5.21. Matching is **not guaranteed** before running the experiment.
Existing commands continue to use the main sampler unless they explicitly set
`+didi_eval.sampler_source=inference`. That option and its source-aware
temperature extension are described next; the standalone control remains
available to isolate the unchanged authors' entry point.

Offline checks: `python scripts/test_didi_inference.py` and
`python scripts/test_didi_authors_checkpoint.py`. These validate source hashes,
release selection, checkpoint safety, and tiny-CPU sampler control flow, not
the paper's numerical results on the full model.

## Eight-GPU inference-release Pareto sweep with Underdeep

Use the updated `didi` checkout on the other machine. The new option is
`+didi_eval.sampler_source=inference`, with `sampling.predictor=guided` and
`+didi_eval.weights=student_ema`. It selects the two sampling methods from the
verified inference release and its defaults (guidance 30→40, four candidates,
reranking over the final half). Explicit `algo.guidance_scale_start/end`,
`algo.num_candidates`, and `algo.rerank_steps_ratio` overrides are respected
and recorded; non-default values are a changed experiment.

```bash
python -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 --module didi_repro \
  mode=train data=openwebtext \
  data.cache_dir="$SOURCE_CODE_PATH/code/cache_dir" \
  training.finetune_path="$SOURCE_CODE_PATH/code/cache_dir/mdlm.ckpt" \
  checkpointing.save_dir="$SOURCE_CODE_PATH/code/owt_didi_inference_pareto" \
  hydra.run.dir="$SOURCE_CODE_PATH/code/owt_didi_inference_pareto" \
  checkpointing.resume_from_ckpt=false \
  model=small model.length=1024 \
  algo=didi_instruct algo.backbone=dit \
  +didi_init.student_hf=haoyangzheng/didi-instruct-small \
  +didi_init.revision=2e9a21a63cc727a1e703370d55de13b842c69076 \
  +didi_eval.before_training=true \
  +didi_eval.weights=student_ema \
  +didi_eval.sampler_source=inference \
  +didi_training.iterations=null \
  trainer.max_steps=0 trainer.devices=8 \
  trainer.limit_val_batches=1 trainer.num_sanity_val_steps=0 \
  trainer.accumulate_grad_batches=1 \
  training.loss_precision=bf16 trainer.precision=bf16 \
  loader.batch_size=1 loader.global_batch_size=8 \
  loader.eval_batch_size=5 loader.eval_global_batch_size=40 \
  loader.num_workers=4 \
  sampling.predictor=guided sampling.steps=16 \
  sampling.noise_removal=ancestral sampling.num_sample_batches=1 \
  eval.generate_samples=false eval.perplexity_batch_size=4 \
  +didi_monitor.gm=false +didi_monitor.gen_ppl=false \
  +didi_pareto.every_n_validations=1 \
  '+didi_pareto.steps=[8,16,32]' \
  '+didi_pareto.temperatures=[0.75,0.80,0.85,0.875,0.90,0.925,0.95,0.975,1.00,1.05,1.10,1.20]' \
  +didi_pareto.seeds=5 +didi_pareto.num_batches=1 \
  +underdeep.experiment=generative-models/owt \
  +underdeep.name=didi-inference-authors-pareto-8-16-32 \
  +underdeep.force_new_run=true \
  'callbacks=[]' +trainer.enable_checkpointing=false \
  +didi_runtime.strict_environment=true
```

Activate the original environment first (see above). To use a different
environment for a diagnostic only, change the last flag to
`+didi_runtime.strict_environment=false`; that is **not** a strict paper
reproduction. The source/config/HF checkpoint hashes are logged, but a matching
environment and sampler do not guarantee matching the reported numbers.
Underdeep and Matplotlib must also be installed for the added tracking/plots;
the standalone authors' environment alone may not include those integrations.

This keeps the existing train-entry-point initialization and OWT cache checks,
so the native teacher file and tokenized cache must still be present. The HF
generator initializes both raw and EMA student copies, and the HF discriminator
is loaded; the frozen teacher is not used to generate the Pareto samples.
`trainer.max_steps=0` performs no optimizer updates. The baseline sweep is at
step 0, uses 8 GPUs × 5 sequences × 1 batch = **40 sequences per seed/point**, and
scores all 40 with perplexity batches of 4. Across 5 seeds and 36 points this is
7,200 sequences. This is the same plot template and aggregation as the existing
MMD-style Pareto logging, not the literal one-GPU authors' RNG/batching protocol.

At **T=1** the temperature context is an exact no-op. For other temperatures,
the integration divides the direct EMA raw logits by T before guidance and
applies the existing SUBS-aware temperature transform to terminal ancestral
cleanup. It does not temperature-scale discriminator reranking rewards or
change the source's raw-logit support/timestep path. Non-unit temperature sweeps
are an explicit extension to the released sampler, not a paper preset.

The inference release uses EMA for its guided steps and the **raw backbone** for
ancestral cleanup. This routing is preserved even if a native checkpoint's two
copies differ; validation NLL still uses `didi_eval.weights`. The sole sampling
repair replaces `reward.backward(); grad = input.grad` with the same input
gradient from `autograd.grad`, so evaluation does not fill discriminator `.grad`
buffers. The outer context enables input autograd under Lightning inference
mode and restores module modes afterward. No vendor file is changed.

`pareto/protocol`, `didi/config`, local provenance and new checkpoints record
the sampler source/revision separately from the main training revision, weight
routing, and temperature transform. Histories from different sampler sources
are not merged. Tests in `test_didi_inference_sampler.py`,
`test_didi_integrated_inference.py`, and `test_didi_inference_temperature.py` exercise
literal-source equivalence, configuration, temperature routing, and the
evaluation-only lifecycle on tiny CPU models; they do not validate full-model
paper scores.

## Standalone control using the authors' main-branch evaluator

To isolate evaluation from this integration, `scripts/eval_didi_authors.sh`
runs **unmodified `vendor/didi_instruct/main.py`**, with the sampling/model/batch
settings from the authors' `scripts/eval-didi-instruct.sh`. It does not install
the `didi_repro` runtime repairs, train a model, read the OWT dataset, or create
an Underdeep run. It replaces only the cluster setup and output/checkpoint paths.
Use one GPU, not `torchrun`. Activate the authors' environment first.

Prepare the released HF weights once, then evaluate into a new directory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_didi_authors_checkpoint.py \
  --output /absolute/path/didi-authors.ckpt --device cuda:0

GPUS=0 CHECKPOINT=/absolute/path/didi-authors.ckpt \
SAVE_DIR=/absolute/path/didi-authors-eval \
bash scripts/eval_didi_authors.sh

CUDA_VISIBLE_DEVICES=0 python scripts/score_didi_authors_samples.py \
  --samples /absolute/path/didi-authors-eval/samples.json \
  --output /absolute/path/didi-authors-eval/all40.json \
  --batch-size 4 --device cuda:0
```

The checkpoint preparer pins the authors' HF revision and verifies all generator
and discriminator keys/shapes against the pristine native architecture. It puts
the same generator tensors under both `backbone.*` and `student_ema.*`. This is
necessary because the original HF converter writes only `student_ema.*`, whereas
the released sampling forward reads `backbone.*`. No weight values are changed.
It also serializes tensors on logical `cuda:0`: the original standalone loader
relies on the checkpoint device and does not explicitly move a CPU export to CUDA.
The native MDLM teacher is **not needed** for this evaluation-only control.

The original evaluator generates 2 × 20 sequences at 16 guided steps, but its
default perplexity batch size 8 drops each batch's final 4 sequences. Consequently
its printed PPL scores **32 of the 40 generated sequences**. The final command
above separately scores all saved sequences with the original metric function,
without regenerating samples or overwriting the original result. The scorer uses
an already cached GPT-2 Large snapshot; the original evaluation downloads it if
needed. This remainder issue does not affect the integration's training monitor,
which already scores every captured sample.

The launcher records the checkpoint hash, dependency versions, console output,
Hydra configuration and original samples/results. It checks core upstream package
versions by default; that is **not** a claim that every Conda build/system library
matches `environment.yml`. An explicitly labelled mismatched-environment diagnostic
can use `DIDI_STRICT_ENVIRONMENT=0`; it is not a strict reproduction. Neither this
control nor using the original script guarantees the paper's numerical results:
the released guidance schedule and precision implementation still differ from
the paper's stated protocol. Preserve them for the first control.

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
SAVE_DIR=/absolute/path/owt_didi_corrected \
GPUS=0,1 \
RUN_NAME=didi-instruct-owt-corrected \
bash scripts/train_didi_owt.sh
```

Use a **new run and new save directory** for the corrected protocol; do not
resume an old run under different iteration/schedule/EMA settings. The runtime
rejects changing the recorded training protocol when resuming.

`GPUS` chooses visible GPUs; as in the upstream
script, global batch size is the visible GPU count, with one real example per
GPU and eight generated candidates per example. The paper's table entry
“batch size 8” is ambiguous relative to the released code's candidate group
size `G=8`; this launcher does not reinterpret it as eight real examples/GPU.
An eight-GPU run is **not established equivalent to the paper's single-H100
experiment**. For its 40-sample generated validation, the corrected training
launcher requires a GPU count dividing 40. One GPU uses 20 samples × 2 batches;
with multiple GPUs, each GPU generates one batch of `40 / GPU_count` samples.
`loader.eval_global_batch_size=40` is explicit. The standalone evaluation script
always uses 20 samples × 2 batches in one process. The exact original cache names
are required:

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
upstream's `torchrun`. See the launcher for its complete, hardware-dependent
evaluation batch allocation. A two-GPU corrected command is:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 --module didi_repro \
  training.finetune_path=/absolute/path/mdlm.ckpt \
  data.cache_dir=/absolute/path/cache_dir \
  checkpointing.save_dir=/absolute/path/owt_didi_corrected \
  hydra.run.dir=/absolute/path/owt_didi_corrected \
  loader.batch_size=1 loader.global_batch_size=2 trainer.val_check_interval=1000 \
  +didi_training.iterations=10000 algo.lr_scheduler.warmup_steps=0 \
  algo.ema_beta=0.9999 +didi_eval.weights=student_ema \
  sampling.predictor=guided sampling.steps=16 \
  loader.eval_batch_size=20 loader.eval_global_batch_size=40 sampling.num_sample_batches=1 \
  +underdeep.experiment=generative-models/owt \
  +underdeep.name=didi-instruct-owt-corrected
```

Every additional argument to the launcher is an explicit Hydra override and is
recorded. Do not copy MMD accumulation, learning rates, reward kernels, batch
sizes or gradient-clipping settings into this command.

## Corrected iteration, learning-rate and EMA protocol

The following corrections are opt-in at the Python entry point and enabled by
the updated training launcher:

| Setting | Direct entry with legacy defaults | Corrected launcher |
| --- | --- | --- |
| Stop condition | `trainer.max_steps=10000`, normally 5000 DiDi iterations | `+didi_training.iterations=10000`, normally 20000 optimizer steps |
| LR horizon | Released scheduler uses the optimizer-step limit as an iteration horizon | 10000 DiDi iterations |
| LR warmup | 1000 iterations | `algo.lr_scheduler.warmup_steps=0` |
| Student EMA decay | `algo.ema_beta=0.999` | `algo.ema_beta=0.9999` |
| Evaluation generator | Raw trainable `backbone` | `+didi_eval.weights=student_ema` |
| Regular generated samples | 1000-step ancestral, 4/GPU × 2 batches | 16-step guided, 40 samples globally |

One DiDi iteration always updates the discriminator and, with the default
cadence, also updates the student. Lightning counts both optimizer updates.
`didi_training.iterations` computes the optimizer-step budget from the requested
iteration count and actual discriminator-warmup/student-update cadence; do not
manually double a step limit instead. The learning-rate schedule has a separate
iteration horizon. With no warmup it starts at the original `1e-6` learning rate
and decays linearly. The released code's **`1e-8` LR floor is retained**; the
paper does not specify that floor.

The new EMA selection routes evaluation forwards to `student_ema`; it does not
copy EMA tensors over trainable student parameters. Training still uses the raw
student, maintains the separate frozen MDLM teacher, and updates the student EMA
after student optimizer steps. `training.ema` is the separate legacy MDLM EMA
setting, not the DiDi EMA controlled by `algo.ema_beta`.

Leaving `didi_training.iterations=null` and `didi_eval.weights=backbone` preserves
the direct entry point's legacy counting/evaluation behavior. A corrected run
must start fresh rather than continuing a legacy checkpoint with changed
training semantics. Resuming an unchanged recorded protocol is supported, with
the data/RNG replay limitations documented below.

The evaluation-weight choice can be changed independently when evaluating or
resuming a compatible training protocol; provenance records which generator was
used. It does not reconstruct a different EMA history for an old checkpoint.

The corrections above do not resolve every paper/code discrepancy. The original
PG/KL objective and normalization, candidate grouping, discriminator update
rules, dataset preprocessing and tokenizer remain as implemented upstream.
Dependency versions, actual teacher/cache contents, GPU count and effective
batch must also be controlled. Report this as the documented corrected protocol,
not proof of an exact reproduction of the reported model quality.

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
the same `eval.*` / `didi_monitor.*` settings as normal validation, including
`didi_eval.weights`. Unmodified direct-entry defaults use 1000-step ancestral
sampling with the raw backbone; the corrected launcher uses guided16 and EMA.
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
start anew. This option itself does not change the selected training protocol:
the updated launcher still applies its corrected iteration/schedule/EMA settings.
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
sampling equations exactly, using the selected `didi_eval.weights` generator.
The sweep itself leaves training equations and ordinary validation unchanged;
the extra sweep restores RNG states, module modes, sampler and forward method.

Sweeps generate additional samples. By default each GPU generates
`didi_pareto.num_batches=2` batches of `loader.eval_batch_size` sequences per
temperature/budget/seed; `+didi_pareto.batch_size=...` overrides only the sweep's
per-GPU batch size. For example, explicitly setting
`+didi_pareto.batch_size=4` on eight GPUs gives 64 samples per seed/point.
With that batch override, the 12-temperature, 3-budget, 5-seed example above
generates **11,520 extra sequences per validation** and can be expensive.
Without it, sweep counts follow the resolved evaluation batch size; the sweep's
two-batch default is independent of the launcher's regular 40-sample allocation.
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

## Released-code defaults retained by the direct entry point

The authoritative files are the vendored YAMLs and `algo.py`, not this summary.
Constructor defaults that are missing from YAML are also logged as effective
hyperparameters.

These defaults are retained for comparability with earlier runs; the corrected
launcher explicitly overrides the schedule, EMA and evaluation rows listed above.

| Setting | Released implementation default |
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
Except for the explicitly selected corrections above, these are **not silently
“corrected” into another method**. The original
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

`val/nll`, `val/ppl`, and `val/bpd` still use upstream validation equations, with
the configured raw/EMA evaluation generator. Legacy ancestral validation PPL is
**not** the paper's guided-sampling result. The corrected launcher selects guided
sampling, but matching that choice alone does not establish identical reported
results. The separate evaluation command below uses the same guided preset.

Optional integration-only flags, without changing the training objective:

```bash
+didi_monitor.gm=false          # omit expensive GM; retain PPL/entropy
+didi_monitor.enabled=false     # omit all additional validation diagnostics
+underdeep.required=false       # allow missing tracking; does not disable an available tracker
```

Underdeep is required by default and initializes only after model/checkpoint
setup succeeds; it must not silently train without logging.

## Guided evaluation with corrected EMA selection

```bash
CHECKPOINT=/absolute/path/owt_didi_corrected/checkpoints/last.ckpt \
SAVE_DIR=/absolute/path/didi_guided_eval \
GPUS=0 \
bash scripts/eval_didi_owt.sh
```

This uses the authors' evaluation-script sampling preset:16 steps, guided
sampling, 20 samples/batch and2 batches, with corrected
`+didi_eval.weights=student_ema`. Both gradient tilting and four-candidate stochastic
reranking are preserved; guidance is0.2→1.0. The final denoising call and
discriminator evaluations are additional work: “16 steps” is not16 total model
evaluations and should not be presented as equal compute to16-step unguided MMD.
The original evaluator writes `samples.json`; its PPL/entropy and samples are
also published to Underdeep. Evaluation requires a trained discriminator.

Direct entry with `didi_eval.weights=backbone` retains the released raw-weight
evaluation. The corrected launcher evaluates `student_ema` instead. The authors'
HF-converted exports contain `student_ema.*`; those weights are loaded explicitly
and can also populate the otherwise-missing backbone for legacy evaluation.

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
10. When explicitly selected, interpret `didi_training.iterations` as DiDi
    iterations, derive the Lightning optimizer-step budget from update cadence,
    and use the iteration count as the LR horizon. The corrected launcher sets
    zero warmup and `algo.ema_beta=0.9999`; direct-entry defaults stay unchanged.
11. When `didi_eval.weights=student_ema` is selected, use EMA evaluation forwards
    without swapping trainable parameters; record the selected evaluation and
    training protocols and reject incompatible training resumes. The inference
    release's terminal ancestral cleanup is the documented raw-backbone exception.
12. When `didi_eval.sampler_source=inference` is selected, load the two guided
    sampling methods from the separately pinned inference release, using the
    same input-autograd repair from item 5. Its guidance defaults, direct EMA
    timestep path, categorical reranking and raw ancestral cleanup are preserved.
    Temperature interception covers both direct EMA and processed cleanup calls.
13. When `checkpointing.save_evaluated_checkpoints` is enabled, suppress native
    in-training checkpoint writes and append a validation-end saver. It retains
    full models from the requested optimizer step and optionally uploads the
    completed run directory to Nirvana snapshots. Training updates are unchanged.

The optional `didi_eval.before_training`, `didi_init.student_hf`, and
`didi_pareto` extensions are disabled by default. Their orchestration, HF weight
loading and sweep implementation live in `pretrain_eval.py`,
`initialization.py`, `pareto.py` and `temperature.py`; the vendored code and
original training objective equations remain unchanged. Separately,
`didi_training.iterations` and `didi_eval.weights` select the corrected protocol
above; those flags are now supplied by the shell launchers.

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
python scripts/test_didi_inference_sampler.py
python scripts/test_didi_inference_temperature.py
python scripts/test_didi_integrated_inference.py
python scripts/test_didi_ddp.py  # optional two-rank CPU/Gloo smoke test
torchrun --standalone --nproc_per_node=2 scripts/test_didi_pretrain_eval.py --ddp-smoke
torchrun --standalone --nproc_per_node=2 scripts/test_didi_pretrain_eval.py --ddp-smoke --pareto-smoke
torchrun --standalone --nproc_per_node=2 scripts/test_didi_pareto.py --ddp-smoke
torchrun --standalone --nproc_per_node=2 scripts/test_didi_integrated_inference.py --ddp-smoke
```

These offline tests use tiny CPU networks and mocked tracking, exercise the
actual patched upstream objective, verify frozen teacher/optimizer ownership,
checkpoint restoration, guided-sample equivalence, source/config fidelity and
diagnostic isolation. They do not establish the paper's final model quality.
