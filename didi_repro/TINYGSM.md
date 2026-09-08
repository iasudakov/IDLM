# Conditional DiDi on TinyGSM

This branch adapts the repaired, pinned DiDi training code to TinyGSM using
`tinygsm-upstream-repro` commit `a3bfe2b3c00ce5ab02db98608a5a517a2e841a25`
as the data/checkpoint/evaluation reference. It is a **new task adaptation**,
not a claim that the DiDi authors published or reproduced TinyGSM results.
Both DiDi vendor source trees remain byte-identical to their manifests.

## What stays the same and what changes

The two-step student trajectory, discriminator update, group-normalized
reward, timestep weighting, policy objective, KL/entropy coefficients, EMA
update and optimizer normalization remain DiDi's released implementation.
No MMD kernel, feature model, `1/G` change, or adaptive training-mask schedule
is introduced. The optional confidence-threshold sampler below is evaluation
only; it does not replace either of DiDi's training trajectory steps.

The TinyGSM-specific behavior follows the reference:

- SmolLM-135M tokenizer; length 512; native small DiT checkpoint.
- `[BOS] question separator program [EOS]`, padded to length 512.
- The separator is **literal backslash-n**, YAML `'\n'`, not a newline.
  Its cache suffix is `sep5c6e` (a real newline would be `sep0a`).
- `attention_mask` is the **loss/generation mask**, not an attention mask
  passed to the transformer. The question remains visible as context.
- Question tokens stay fixed during both student steps, real/fake corruption,
  validation NLL, and conditional sampling. Direct policy/KL/entropy/critic
  losses exclude the question; attention gradients through context remain.
- `train_on_pad=true` generates and scores padding, matching the reference
  training mask. DiDi's reward is averaged over corrupted target positions;
  MMD's optional EOS-cropped feature reward is not imported.
- Evaluation receives only each question prefix and generates its entire
  remaining suffix. It never uses the gold answer length or padding boundary.
- Guided candidate rewards exclude the prompt. Both main-release and
  inference-release guided samplers, plus ancestral sampling, are supported.

For SmolLM the tokenizer has 49,153 entries after adding PAD (`49152`);
DiDi appends a distinct MASK (`49153`), giving model vocabulary 49,154.
The reference and vendored DiT have identical 131 state-dict keys and shapes
at this vocabulary, and strict meta-device state loading was verified.
Actual teacher loading still checks every backbone tensor strictly.

## Training recipe

The launcher follows `vendor/didi_instruct/scripts/distill-didi-instruct-owt.sh`
for training hyperparameters, rather than the paper-informed OWT launcher:

| Setting | TinyGSM DiDi launcher |
| --- | --- |
| Real examples per GPU | 1 |
| Candidates per example | 8 |
| Student / discriminator learning rate | `1e-6` / `1e-6` |
| Timestep distribution | `beta11` |
| LR warmup | 1,000 DiDi iterations |
| Student EMA | `0.999` |
| Stop condition | 10,000 individual optimizer updates |
| DiDi iterations at the default two updates each | approximately 5,000 |
| Validation cadence | every 1,000 training batches |

`didi_training.iterations=null` retains the original scheduler's mixed-unit
behavior: its LR horizon is 10,000 while the iteration counter reaches about
5,000. This is intentional released-code fidelity, not a corrected schedule.
The existing runtime/checkpoint repairs remain enabled. Gradient accumulation
is not supported by the released manual-optimization loop; keep it at 1.

The **evaluation choices are task-adaptation choices**: main-release guided
sampling, 16 denoising steps, student EMA, guidance `0.2` to `1.0`, four
candidates, conditional validation and GSM8K execution accuracy. These are
not claimed to be tuned TinyGSM hyperparameters. To use the dedicated
inference-release sampler instead, set `SAMPLER_SOURCE=inference`; its default
guidance becomes 30 to 40. To disable guidance and use unguided DiDi sampling,
set `SAMPLER=ancestral_cache EVAL_STEPS=64` with the default main source.
This retains DiDi's separate final cleanup; the same step number is not
equivalent to the reference's S-FLM step/NFE convention.

## Run on eight GPUs

Use the authors' environment (`vendor/didi_instruct/environment.yml`) with
Hydra/OmegaConf and your working Underdeep SDK/credentials. The strict guard
checks core dependency versions; it is not a complete environment lock check.
No GPT-2 evaluator/model/tokenizer is downloaded for TinyGSM metrics.

Run from this checkout's `code` directory. Supply the existing TinyGSM assets
used by the reference branch; the OWT `mdlm.ckpt` and authors' OWT HF student
are **not** substitutes for the TinyGSM teacher.

```bash
conda activate didi-authors

TEACHER_CHECKPOINT="$SOURCE_CODE_PATH/code/checkpoints/tinygsm/mdlm_notok.ckpt" \
TINYGSM_CACHE="$SOURCE_CODE_PATH/code/cache_dir" \
GSM8K_DATA="$SOURCE_CODE_PATH/code/data/gsm8k_test.json" \
SAVE_DIR="$SOURCE_CODE_PATH/code/tinygsm_didi" \
GPUS=0,1,2,3,4,5,6,7 \
bash scripts/train_didi_tinygsm.sh
```

The script launches `python -u -m torch.distributed.run --module didi_repro`.
For the full expanded command without starting it, add
`DIDI_DRY_RUN=1 NPROC_PER_NODE=8` to the environment. Hydra overrides can be
appended to the script. For example:

```bash
# Add to the launcher command above to evaluate multiple fixed budgets:
bash scripts/train_didi_tinygsm.sh '+eval.gsm8k_steps=[8,16,32]'
```

That example uses the same environment assignments as the complete command.
The primary 16-step score and optional budget scores are logged every
validation. Set `+eval.gsm8k_plot=true` to add an image (requires Matplotlib);
tables and scalar trends work without it. The horizontal axis is **denoising
steps**, not measured NFE: guidance, candidates and final cleanup add work.
Those fixed-budget points are separate from the adaptive-threshold frontier
described below.

### Threshold sweeps: accuracy versus NFE

To add the same adaptive evaluation rule used by `tinygsm-upstream-repro`,
append these overrides to the training command (or the launcher):

```bash
  '+eval.gsm8k_pareto_thresholds=[0.5,0.6,0.7,0.8,0.9,0.95]' \
  +eval.gsm8k_pareto_temperature=0.0 \
  +eval.gsm8k_pareto_plot=true
```

Any explicit list of distinct values in `[0,1]` is accepted, including the denser threshold list
from the TinyGSM MMD runs. `null` or `[]` disables this sweep; the default is
disabled. `eval.gsm8k_at_validation=true` must also be enabled, as it already
is in the supplied training command. Matplotlib is required for images;
`+eval.gsm8k_pareto_plot=false` keeps scalar curves, tables and on-disk history
without it. Missing plotting dependencies fail before training model loading.

The sampler keeps the question fixed, proposes tokens, and commits every
remaining position whose **raw model probability is strictly greater than
the threshold**. If none qualify, it commits that row's highest-confidence
position. Temperature zero makes proposals greedy; at nonzero temperature,
temperature affects proposals, not the probability used as confidence.
EOS is not an early stop: the whole answer/padding suffix is generated,
matching the reference. This is **unguided** adaptive decoding of the selected
student/EMA weights, not discriminator-guided DiDi inference.
Nonfinite probabilities or a selected MASK proposal fail explicitly instead
of publishing incomplete answers; lower the temperature if that occurs.

`sampling.steps` does not constrain the threshold sweep. The default loop
cap is `model.length`; an optional `+eval.gsm8k_pareto_max_steps=N` caps its
threshold iterations and adds a counted final completion forward if needed.
There is no extra cleanup call once all masks are gone. Noise time is inferred
from the remaining MASK fraction as in the reference adapter.

The curve's horizontal axis follows the reference: mean **active NFE per
sequence**, counting forwards until each sequence is complete, including a
forced final completion if applicable. The implementation also reports batch
forward calls separately. Finished rows can remain in a batched forward while
another row is unfinished, so mean active NFE is not a direct GPU-compute or
wall-time measurement. Exact floating-point NFE values are kept in tables,
history and images; tracker scalar curves use the reference `NFE * 100`
integer axis and explicitly account for colliding bins.

The sweep runs at the existing GSM8K cadence and after the primary/fixed-step
evaluations. It preserves training RNG, gradients, model modes and sampling
settings. The main guided 16-step score remains `val/gsm8k_acc`; threshold
results use separate accuracy/NFE trend metrics and Pareto tables/figures.
The image is logged as `pareto/frontier`, and exact points as
`pareto/gsm8k_thresholds`. Local history is saved as
`didi_tinygsm_pareto_history.json` in the checkpoint output directory.
Histories are isolated by run identity and restored for matching resumed runs.
An optional `+eval.gsm8k_pareto_reference=/path/reference.json` adds the
reference frontier in the same point format as TinyGSM MMD.

Each threshold evaluates the benchmark again. A long threshold list can make
initial and periodic validation substantially slower; it does not indicate
that training is stuck. Use fewer thresholds or a small
`+eval.gsm8k_num_examples=32` for a smoke test, not a full benchmark score.

Pre-training evaluation is enabled by the script. Set
`EVAL_BEFORE_TRAINING=false` to start updating immediately. The full benchmark
contains 1,319 questions and can take substantial time; add
`+eval.gsm8k_num_examples=32` for a smoke test, not a comparable full score.

For a fresh **teacher-initialization evaluation only**, use the same launcher
with `trainer.max_steps=0` and pre-training evaluation enabled. This shortcut
does not load a trained student. To resume a trained student,
append `checkpointing.resume_from_ckpt=true` and
`checkpointing.resume_ckpt_path=/absolute/path/to/tinygsm-didi.ckpt`.
Resume rejects checkpoints missing or disagreeing on TinyGSM task/tokenizer/
separator/mask/length metadata. Keep the saved run's training budget and
schedule: changing its `max_steps` to zero also changes the LR protocol and
is rejected. Pre-training validation can evaluate the resumed weights before
the remaining training updates. The native MDLM teacher is still required.

Checkpoints retain the released save cadence (including internal saves every
200 DiDi iterations); plan disk space accordingly. No training job is started
by config inspection or dry-run mode.

## Data preparation and cache safety

Training reads the existing separator-specific reference cache, for example:

```text
cache_dir/tiny_gsm_bs512_answer_only_filtered_train_on_pad_HuggingFaceTB__SmolLM-135M_sep5c6e/
  train/
  validation/
```

Legacy caches without `prompt_len` are supported only when their binary,
contiguous answer-only loss mask unambiguously identifies the boundary.
All-zero masks, ambiguous prompt boundaries and inconsistent padding are
rejected. Legacy caches require the original validation split settings
(`val_ratio=0.01`, `val_seed=42`); changing those flags does not re-split data.
Training never deletes, downloads or regenerates dataset caches.

If needed, explicitly prepare a **new cache directory** from local records:

```bash
python scripts/prepare_didi_tinygsm.py \
  --cache-dir /absolute/path/new_cache_dir \
  --data-path /absolute/path/tinygsm_records.json
```

Use `--allow-download` instead of `--data-path` to explicitly download
`TinyGSM/TinyGSM`. Existing target caches, even incomplete ones, are never
overwritten. The tokenizer can download normally; use
`--tokenizer-local-files-only` for offline preparation. Prepared caches record
their preprocessing/split metadata. Training can consume a transferred cache
without access to the original source JSON.

GSM8K uses the reference cached benchmark or a local JSON with `prompt` and
`response_ground_truth` fields. Questions are sharded across ranks without
padding/duplicate benchmark rows, and rank zero runs the unchanged reference
grader and publishes global counts.

## Metrics and verification

Underdeep receives existing DiDi loss/reward/gradient/LR metrics plus:

- Conditional `val/nll`, `val/ppl` and `val/bpd` (answer-mask likelihood).
- `val/sample_entropy` over generated answer positions, prompt excluded.
- `val/answer_exact_match`, and optional execution `val/answer_acc` with
  `val/answer_gradable` (ungradable reference programs are excluded).
- `val/gsm8k_acc`, correct/count totals, and per-budget `_s8`, `_s16`, etc.
- Optional threshold accuracy/NFE trends, `pareto/frontier` images and exact
  `pareto/gsm8k_thresholds` tables.
- Question/generated/reference sample tables and task/sampler/grader provenance.

Execution grading runs generated Python with the reference restricted
builtins/timeout. **It is not an OS security sandbox.** The launcher enables
it for this trusted reproduction workflow; disable it with
`++eval.tinygsm_exec=false ++eval.gsm8k_at_validation=false` when inappropriate
(the double `+` replaces integration options already added by the launcher).
If threshold sweeps were enabled, also set `++eval.gsm8k_pareto_thresholds=null`.
Evaluation restores RNG, parameter gradient references and module modes.
The OWT monitor is provenance-only on TinyGSM, so it cannot overwrite answer
metrics with unconditional samples or prompt-inclusive entropy.

CPU tests (run each in its own process; no GPU/download/tracker required):

```bash
python scripts/test_didi_tinygsm_data.py
python scripts/test_didi_tinygsm_runtime.py
python scripts/test_didi_tinygsm_eval.py
python scripts/test_didi_tinygsm_entrypoint.py
python scripts/test_didi_tinygsm_ddp.py
python scripts/test_didi_tinygsm_adaptive.py
python scripts/test_didi_tinygsm_pareto.py
```

They cover conditional masks and direct gradients, two-optimizer updates,
frozen teacher, unequal prompt lengths, ancestral and both guided samplers,
checkpoint save/resume, real entrypoint pre-training evaluation, reference
cache/tokenizer behavior, and two-rank CPU evaluation/training. These checks
do **not** establish full-scale training quality or GPU memory requirements.
