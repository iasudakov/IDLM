# Evaluate the original TinyGSM MMD checkpoint

This is one evaluation launch on one or four GPUs, using the same inference worker as
the MDLM → CDLM, CDLM → MMD and MMD → CDLM checkpoint comparisons. It does not
train, collect trajectories or require a CDLM training state.

By default, it evaluates the published
[MDLM-MMD TinyGSM checkpoint](https://huggingface.co/yresearch/MDLM-MMD-TinyGSM/blob/main/model.ckpt)
linked in the project README:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/eval_mmd_checkpoint_nirvana.sh" \
  --devices 1 \
  'logger.name=mdlm to mmd published checkpoint dual norm ancestral'
```

Use the existing environment and offline tokenizer/GSM8K assets. If those are
still in the Nirvana inputs rather than the source code directory:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/eval_mmd_checkpoint_nirvana.sh" \
  --devices 1 \
  "model_weights_dir=$INPUT_PATH/tinygsm_mmd_offline_bundle" \
  "eval.gsm8k_data_path=$INPUT_PATH/nirvana_checkpoints/data/gsm8k_test.json" \
  'logger.name=mdlm to mmd published checkpoint dual norm ancestral'
```

The [complete run-command](../examples/nirvana_mmd_checkpoint_eval_run_command.sh)
includes the environment setup for one GPU. Request one GPU in the Nirvana block's
resources as well. `--devices 1` starts one worker and sets the matching model
configuration; use it rather than a `trainer.devices=1` override. `--devices 4`
uses four workers, and omitting the option also defaults to four. Batch size is
16 per GPU; all 1,319 questions are evaluated in both cases. There is no gradient
accumulation or optimizer.

## Weights and offline assets

The launcher resolves the requested checkpoint once, before `torchrun` starts.
It reuses the existing checkpoint resolver, which checks prepared local assets,
the Hugging Face cache and `cdlm_assets/` in a connected snapshot before attempting
a download. It can reuse the MMD model asset from a previous MMD → CDLM launch.
The default Nirvana download proxy is the same as the training launcher;
`CDLM_DOWNLOAD_PROXY` overrides it, and an explicit empty value uses direct
access. Existing `HF_HUB_OFFLINE=1` is respected: download then requires removing
that setting or providing the weights locally. GPU evaluation workers always
run with Hub and datasets offline.

An explicitly requested local file never falls back to a different model:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/eval_mmd_checkpoint_nirvana.sh" \
  --devices 1 \
  --checkpoint "$INPUT_PATH/mmd_tinygsm.ckpt" \
  'logger.name=mdlm to mmd local checkpoint dual norm ancestral'
```

`MMD_EVAL_CHECKPOINT` is the environment equivalent of `--checkpoint`. The saved
`global_step` is read from the checkpoint; it is not inferred from its filename
or assumed to be 7000. Published EMA-only exports are used directly. Full
checkpoints must supply EMA state. Missing/empty weights, missing step metadata,
or absent EMA are errors before GPU launch; raw weights are not substituted.
The worker still checks the exact model architecture and EMA compatibility.

The tokenizer bundle and GSM8K JSON remain local assets. Neither is downloaded
by this launcher. The original MDLM teacher and tokenized training set are not
needed. Use `--plan-only` for CPU checkpoint/asset checks without inference.

## Evaluation and reporting

- Same BF16 transformer forward and EMA weights for both confidence variants.
- Confidence T=0, evaluated once with BF16 normalization and once with FP32
  normalization. The common 13-threshold grid for all model checkpoint evaluations
  is `[0.325,0.4,0.475,0.55,0.625,0.7,0.775,0.85,0.925,0.95,0.98,0.99,0.995]`.
- Ancestral T=0, FP32 normalization, budgets `[4,8,16,32,48,64]`, seed 1,
  with a separate final denoise. Actual NFE is recorded; one seed has no sample SD.
- Full 1,319-question GSM8K execution evaluation with the same prompts, sharding,
  batch size and scoring as the CDLM checkpoint comparisons.

The launcher creates a separate Underdeep run in `generative-models/tinygsm`.
Confidence metrics end in `_bf16norm` / `_fp32norm`; the plots are
`pareto/frontier_bf16norm` and `pareto/frontier_fp32norm`. Ancestral uses
`ancestral/accuracy_vs_steps` and the existing `val/gsm8k_*_ancestral_steps*`
metrics. Results use the saved checkpoint step on the tracker axis.

The default output is `code/mmd_checkpoint_eval`, with `plan.json`, the resolved
config, protocol and checkpoint hashes, a checkpoint `result.json`, and
`summary.csv` containing 32 rows (26 confidence points plus six ancestral points).
Results and tracker metadata are published to the separate `mmd_checkpoint_eval`
Nirvana snapshot namespace. Reconnecting that state restores completed matching
results; different checkpoint bytes or protocol settings cannot silently reuse
old results. Change `--output-dir` to select another local output directory.

GPU count is part of the protocol identity. To evaluate on one GPU from a state
containing earlier four-GPU evaluation reports, use a fresh output directory and
disable report restoration:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/eval_mmd_checkpoint_nirvana.sh" \
  --devices 1 \
  --output-dir "$SOURCE_CODE_PATH/code/mmd_checkpoint_eval_1gpu" \
  --no-restore-results \
  'logger.name=mdlm to mmd published checkpoint 1gpu dual norm ancestral'
```

Prepared weights in the connected state remain reusable. Underdeep logging,
plots and snapshot publication stay enabled. Use the same GPU count for all
models being compared: changing ranks changes ancestral RNG streams and batch
assignments, so the same seed need not produce the same generations.

The expanded threshold grid is a new evaluation protocol. Existing results from
the earlier ten-threshold grid remain valid for those thresholds, but cannot be
treated as a completed 13-threshold sweep. When evaluating again with a previous
evaluation state connected, use `--no-restore-results` and a fresh output directory
to create new results rather than attempting to resume the old protocol.
