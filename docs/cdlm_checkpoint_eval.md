# Evaluate the three TinyGSM CDLM chains

This evaluates **18 saved checkpoints**, without training or trajectory generation:

| Source run | Chain | Saved optimizer steps |
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
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh"
```

The second block receives `dbb3b25d`'s state:

```bash
CDLM_EVAL_SOURCE_RUN=dbb3b25d \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh"
```

The third block receives `8cb53b89`'s state:

```bash
CDLM_EVAL_SOURCE_RUN=8cb53b89 \
bash "$SOURCE_CODE_PATH/code/scripts/eval_cdlm_checkpoints_nirvana.sh"
```

Keep the existing tokenizer bundle under `code/cache_dir/tinygsm_mmd_offline_bundle` and GSM8K JSON under
`code/data/gsm8k_test.json`, or provide `model_weights_dir` and
`eval.gsm8k_data_path` overrides as in the complete example below.
`--snapshot-root /path/to/state` bypasses SDK snapshot discovery.

The [complete environment setup example](../examples/nirvana_cdlm_checkpoint_eval_run_command.sh)
includes the same pinned packages and reads the offline assets directly from
their input paths. Only change its `CDLM_EVAL_SOURCE_RUN` value for each block.
No Hugging Face downloads are attempted. All required checkpoint paths and
offline assets are checked before starting GPU workers. `--plan-only` writes
the plan and checks inputs without starting evaluation.

## Common evaluation protocol

- Four GPUs, one checkpoint evaluated at a time; all 1,319 GSM8K questions are
  partitioned across ranks exactly as in distributed training validation.
- EMA weights in every case. Full CDLM checkpoints supply their saved EMA;
  MMD weights-only exports already contain EMA and are not averaged again.
- Small full-attention MDLM, length 512, the original literal `\n` separator,
  batch size 16 per GPU and the same execution-based grading.
- **Confidence:** temperature 0, thresholds
  `[0.325,0.4,0.475,0.55,0.625,0.7,0.775,0.85,0.925,0.98]`.
  Run twice with the **same BF16 transformer forward**, changing only logit
  normalization between BF16 and FP32. Both accuracy and actual NFE are recorded.
- **Ancestral:** temperature 0, budgets `[4,8,16,32,48,64]`, one generation
  seed **1**, BF16 forward and **FP32 normalization**. A separate final denoise
  is included and actual cached NFE is measured. A one-seed result has no
  across-seed standard deviation. Rank RNG seeds use `1 + 104729 * rank`.

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

Each checkpoint has a `result.json`; `summary.csv` combines the confidence and
ancestral results for the launch. Confidence metrics carry `_bf16norm` and
`_fp32norm` suffixes, with images `pareto/frontier_bf16norm` and
`pareto/frontier_fp32norm`. Ancestral results retain `ancestral/accuracy_vs_steps`
and the existing `val/gsm8k_*_ancestral_steps*` metric names.

GPU execution is intended for the user's Nirvana machine; local verification
uses CPU fixtures and mocked model/tracker interfaces.
