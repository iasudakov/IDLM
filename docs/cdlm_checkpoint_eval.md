# Evaluate the three TinyGSM CDLM chains

This evaluates the **available checkpoints from an expected catalog of up to 18**,
without training or trajectory generation. Training logs describe the expected
catalog; a connected snapshot can contain fewer files.

The default checkpoint evaluation now runs **ancestral only**, at temperatures
**0.1 and 1.0**, with generation seeds **1, 2, 3** for each temperature and budget.
Confidence sampling and ancestral temperature 0 are disabled. The launch command
below already selects these defaults after updating the Nirvana source-code input.

| Source run | Chain | Expected optimizer steps |
| --- | --- | --- |
| `5e0ee1da-8780-42d3-b120-9590e9dc8351` | MDLM → CDLM | 1000, 1248 (`last.ckpt`) |
| `dbb3b25d-4937-4f27-966f-898e696a1f48` | CDLM → MMD | 500, 1000, …, 7000 |
| `8cb53b89-96e7-4fe4-ba49-91ed2eb9fb20` | MMD → CDLM | 1000, 1248 (`last.ckpt`) |

The older `f7a18dce`, `7e294cec` and unavailable `669290a9` snapshots are excluded.
The two CDLM sources both contain files named `last.ckpt`; their source run IDs
and snapshot namespaces remain separate throughout evaluation.

## Three Nirvana launches, one input state each

Create three evaluation launches. For each, connect the matching training run's
**output state** to the ordinary Nirvana **snapshot/state input**. The launcher
uses `nirvana_dl.snapshot.get_snapshot_path()` to read the unpacked state.
No manual checkpoint paths, extraction or merging of snapshots is required.

Use the same environment and offline assets as the training jobs. The evaluator
needs the tokenizer bundle `tinygsm_mmd_offline_bundle` and `gsm8k_test.json`;
it does not need the original MDLM teacher or tokenized TinyGSM training data.
After the usual environment setup, use one of these commands in each block.
The first block receives `5e0ee1da`'s state:

```bash
CDLM_EVAL_SOURCE_RUN=5e0ee1da \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" --devices 1
```

The second block receives `dbb3b25d`'s state:

```bash
CDLM_EVAL_SOURCE_RUN=dbb3b25d \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" --devices 1
```

The third block receives `8cb53b89`'s state:

```bash
CDLM_EVAL_SOURCE_RUN=8cb53b89 \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" --devices 1
```

These commands use one GPU per Nirvana block. Request one GPU in the block's
resources as well. Use `--devices 4` for four GPUs; omitting `--devices` also
defaults to four. Use this CLI option rather than `trainer.devices=1`, since it
sets both the worker count and the resolved model configuration. Batch size
remains 16 per GPU, and all 1,319 questions are evaluated on either setup.

Keep the existing tokenizer bundle under `code/cache_dir/tinygsm_mmd_offline_bundle` and GSM8K JSON under
`code/data/gsm8k_test.json`, or provide `model_weights_dir` and
`eval.gsm8k_data_path` overrides as in the complete example below.
`--snapshot-root /path/to/state` bypasses SDK snapshot discovery.

The [complete environment setup example](../examples/nirvana_cdlm_checkpoint_eval_run_command.sh)
includes the same pinned packages and reads the offline assets directly from
their input paths. Only change its `CDLM_EVAL_SOURCE_RUN` value for each block.
No Hugging Face downloads are attempted. Available checkpoint paths and offline
assets are checked before starting GPU workers. `--plan-only` writes the plan
and checks inputs without starting evaluation.

### Partial snapshots

By default, missing or zero-byte catalog files are skipped with a warning. The
launcher prints the selected steps and every skipped path before GPU workers
start. Both lists are stored in `plan.json` (`entries` and `skipped_checkpoints`)
and published with the evaluation results. For example, a missing
`step=00006000.ckpt` does not prevent evaluation of steps 5500, 6500 or 7000 if
those files are present. No missing checkpoint is downloaded or reconstructed.
This works for the ordinary unpacked Nirvana state and archive inputs.

The existing three launch commands already use this behavior. To make the
policy explicit:

```bash
CDLM_EVAL_SOURCE_RUN=dbb3b25d \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" \
  --missing-checkpoints skip
```

Use `--missing-checkpoints error` to require the complete catalog. Both policies
fail before GPU initialization if any selected source has no usable expected
checkpoint, the state is absent, or a checkpoint path escapes its namespace.
Inherited CDLM weights and unverified `last.ckpt` files are never substituted
for missing MMD weights. Nonempty files still undergo strict weight loading and
saved-step validation; corrupt weights are errors rather than silently skipped.

## Common evaluation protocol

- One or four GPUs, one checkpoint evaluated at a time; all 1,319 GSM8K questions
  are partitioned across ranks exactly as in distributed training validation.
  With `--devices 1`, the single worker evaluates every question.
- EMA weights in every case. Full CDLM checkpoints supply their saved EMA;
  MMD weights-only exports already contain EMA and are not averaged again.
- Small full-attention MDLM, length 512, the original literal `\n` separator,
  batch size 16 per GPU and the same execution-based grading.
- **Confidence (disabled by default):** temperature 0, thresholds
  `[0.325,0.4,0.475,0.55,0.625,0.7,0.775,0.85,0.925,0.95,0.98,0.99,0.995]`.
  If enabled, run twice with the **same BF16 transformer forward**, changing only logit
  normalization between BF16 and FP32. Both accuracy and actual NFE are recorded.
- **Ancestral:** temperatures **0.1 and 1.0**, budgets `[4,8,16,32,48,64]`,
  generation seeds **1, 2, 3** at each temperature and budget, BF16 forward and
  **FP32 normalization**. Token proposals use the temperature-scaled categorical
  distribution, including the separate final denoise; position selection remains
  ancestral Bernoulli unmasking. Actual cached NFE is measured. Results retain
  per-seed counts plus mean and sample SD across the three seeds. Rank RNG seeds
  use `seed + 104729 * rank`; each budget/temperature restarts that seeded stream.

This gives 36 evaluations of all 1,319 questions per checkpoint (2 temperatures
times 6 budgets times 3 seeds). The settings can be overridden explicitly:

```bash
CDLM_EVAL_SOURCE_RUN=5e0ee1da \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" --devices 1 \
  'eval.gsm8k_ancestral_temperatures=[0.1,1.0]' \
  'eval.gsm8k_ancestral_seeds=[1,2,3]' \
  eval.gsm8k_checkpoint_confidence=False
```

To include temperature 0 again, add it to the temperatures list. To enable
confidence again, set `eval.gsm8k_checkpoint_confidence=True`. The separate
`eval_mmd_checkpoint_nirvana.sh` launcher keeps its previous confidence and
single-seed T=0 defaults; these changes are scoped to the CDLM checkpoint sweep.

This is not full-FP32 model inference. Original training and validation metrics
are preserved; the reevaluation uses separate Underdeep runs and output files.

## Outputs and restart

The output root defaults to `code/cdlm_checkpoint_eval`; change it with
`--output-dir`. There is one new Underdeep evaluation run per source training
run, with metrics indexed by the saved optimizer step. The two confidence
normalizations are labeled separately. Source run, checkpoint hash, data hash,
EMA selection and evaluation settings are recorded alongside the results.

Results and evaluation tracker metadata are published to the separate Nirvana
snapshot namespace `cdlm_checkpoint_eval` after completed checkpoints. This does
not modify the input model files or copy trajectory data into the evaluation
output directory. When restarting from an evaluation state, the launcher
restores this evaluation namespace into an empty local output directory. A
completed matching report is reused; partial checkpoint evaluations are repeated.
Changed checkpoint contents or evaluation settings must not silently reuse old results.

For this temperature/seed protocol, start new reports when the connected state
contains evaluations from the old T=0 or confidence sweep:

```bash
CDLM_EVAL_SOURCE_RUN=5e0ee1da \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" \
  --devices 1 \
  --output-dir "$SOURCE_CODE_PATH/code/cdlm_checkpoint_eval_t01_t1_s123" \
  --no-restore-results
```

When resuming this same new sweep from its evaluation state on a fresh worker,
omit `--no-restore-results`. Reuse requires every requested temperature, budget
and seed, and all 1,319 examples. Partial checkpoint results are rerun.

Changing from four GPUs to one changes the protocol identity. If the connected
state contains previous evaluation results, start fresh reports with:

```bash
CDLM_EVAL_SOURCE_RUN=5e0ee1da \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh" \
  --devices 1 \
  --output-dir "$SOURCE_CODE_PATH/code/cdlm_checkpoint_eval_1gpu" \
  --no-restore-results
```

Keep the matching source run and its model state. These options prevent reuse
of evaluation reports; the same checkpoint files are still read from the state.
For direct comparisons, use the same GPU count across models: ancestral RNG
streams and batch assignments depend on the rank count, so seed 1 does not
guarantee identical generations between one and four GPUs. Underdeep logging,
plots, JSON/CSV results and snapshot publication remain enabled on one GPU.

The extended 13-threshold grid changes the evaluation protocol. Results from
the older ten-threshold sweep are not silently treated as complete. To rerun
from a state containing those reports, use a new local output directory and
`--no-restore-results`, for example `--output-dir "$SOURCE_CODE_PATH/code/cdlm_checkpoint_eval_13" --no-restore-results`.
The same model weights can be reused; no retraining is required.

Each checkpoint has a `result.json`; `summary.csv` combines the confidence and
ancestral results for the launch, with an explicit `temperature` column.
Confidence results are empty when disabled; no confidence decoding or plots run.
When enabled, confidence metrics carry `_bf16norm` and
`_fp32norm` suffixes, with images `pareto/frontier_bf16norm` and
`pareto/frontier_fp32norm`. Ancestral curves and metrics carry `_temp0p1` or
`_temp1p0` suffixes; images are `ancestral/accuracy_vs_steps_temp0p1` and
`ancestral/accuracy_vs_steps_temp1p0`. Per-seed and mean/SD scalars stay separate.
The optional T=0 curve retains the old unsuffixed ancestral names.

GPU execution is intended for the user's Nirvana machine; local verification
uses CPU fixtures and mocked model/tracker interfaces.
