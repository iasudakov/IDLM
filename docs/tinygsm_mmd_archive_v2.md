# Corrected archive MMD objective

This branch imports `dlm-mmd-fast (1).zip`. The two objective modules are copied
verbatim from its `Untitled/mmd/` directory; `mmd/archive_source.json` records
the ZIP and source hashes. The only differences between that nested directory
and the archive root are the two feature-forward autocast fixes. They keep
rotary phases in FP32 outside the backbone's own BF16 region, in both fast and
reference extraction. `mmd/ARCHIVE_LICENSE` retains the source license.

The integration retains this repository's trainer, offline bundles, Underdeep,
manual accumulation, EMA and paired GSM8K validation. It does not replace them
with the standalone archive's training and evaluation harness. In particular,
checkpoints retain full live/EMA/training state rather than exporting EMA-only
weights. The archive's objective sources and their tests are unchanged.

## Run

Stage the same data, `mdlm_notok.ckpt` and offline tokenizer bundle as before.
The launcher does not install dependencies or download assets.

```bash
DEVICES=4 OUT="$PWD/outputs/mmd_archive_l9_b2_g4_sigma90" \
bash scripts/train_tinygsm_mmd_fast.sh \
  adversarial_distill.mmd_feature_layer=9 \
  adversarial_distill.mmd_kernel_batch_size=2 \
  adversarial_distill.mmd_group_size=4 \
  adversarial_distill.mmd_rbf_alpha=6.17283950617284e-5
```

Defaults: fast, layer 6 (zero based), B=2, G=4, alpha=6e-5 (sigma about 91.29),
sum over candidates, LR=1e-5, global batch 512, up to 8000 optimizer updates.
Layers 0-11 are available in the small MDLM. Sigma uses alpha=1/(2*sigma^2).
Override `trainer.max_steps` if a different training budget is needed.

`DEVICES`, `CKPT`, `CACHE_DIR`, `MODEL_WEIGHTS_DIR` and `OUT` can be set in the
environment. Additional Hydra arguments are applied last. Each independent
experiment should use a distinct `OUT` and `logger.name`.

| Parameter | Meaning |
| --- | --- |
| `adversarial_distill.mmd_implementation=fast` | Corrected archive fast objective |
| `adversarial_distill.mmd_implementation=reference` | Corrected archive reference objective |
| `adversarial_distill.mmd_kernel_batch_size` | B >= 2 independent sampled sequences per candidate, for the same prompt |
| `adversarial_distill.mmd_group_size` | G >= 1 candidates in the REINFORCE group |
| `adversarial_distill.mmd_average_candidates` | False preserves the archive TinyGSM sum over G; True divides by G |
| `adversarial_distill.mmd_forward_batch_size` | Student forward chunks and reference extractor batch (default 8) |
| `adversarial_distill.mmd_fast_feature_batch_size` | Fast frozen-extractor batch (default 64) |
| `adversarial_distill.mmd_fast_scored_head` | Restrict the student vocabulary projection to scored positions |

`legacy` is the global default outside this launcher and preserves the old
code path. `fast` and `reference` now both mean the archive cross-draw
objective, unlike the deleted branch's historical rows/B=1 implementation.

The archive requires B >= 2, token RBF, a fixed positive alpha, a single
feature layer, clean frozen native MDLM features, and a within-group
leave-one-out baseline (no baseline for G=1). It omits the real-real constant,
excludes same-sequence fake pairs, and optionally excludes matching positions
across different fake sequences (enabled by default). Kernel arithmetic is
FP32. Rewards, the negative REINFORCE loss and the paper surrogate follow the
archive, including its candidate/draw normalization. Fast uses gathered log
probabilities directly; reference retains the archive's exp/clamp/log path.
Their exact equivalence is tested at normal finite probabilities; the two
source paths can differ when that clamp is active or due to BF16 rounding.

This is a different estimator from the old paper run's rows/B=1 recipe. Its
training trajectory is not a bitwise reproduction of that run. Fast and
reference also consume different random streams unless draws are fixed.

## Evaluation and saving

Validation runs every 500 optimizer updates: 1000 train batches on eight GPUs,
2000 on four GPUs, 4000 on two GPUs. The interval follows manual accumulation
through Hydra.
The evaluated checkpoint callback saves after each completed validation, with
names such as `checkpoints/eval-step=00000500.ckpt`. It retains live weights,
EMA and optimizer state, and invokes the existing Nirvana snapshot helper.
For archive runs, checkpoint saving preserves Lightning's manual-loop counters;
the legacy hook would otherwise overwrite them with the unused automatic-loop
counters. This path is covered by a save/load/resume integration check.
Passing `trainer.val_check_interval=2000` still works and is in train batches.

To skip all evaluations before optimizer step 5500:

```bash
DEVICES=2 bash scripts/train_tinygsm_mmd_fast.sh eval.start_step=5500
```

`eval.start_step` defaults to 0 (previous behavior). A positive value skips
the explicit initial evaluation, Lightning's sanity check, and complete
training validation loops, including NLL, sample generation and both GSM8K
precision sweeps. With the default cadence, evaluations then run at steps
5500, 6000, 6500, etc.; evaluated checkpoints also start at 5500. The threshold
uses the restored optimizer step when resuming. It does not shift the cadence:
for example, a threshold of 5501 first evaluates at 6000. A threshold above
the training budget means no evaluations or evaluated checkpoints in that run.
Independent periodic checkpoint policies and standalone evaluation commands
are unaffected. `eval.validate_before_train=False` alone only disables the
initial evaluation; use `eval.start_step` to delay periodic validation too.

The existing confidence sweep at T=0 runs in its original precision and again
with FP32 prediction normalization (`_fp32norm`). Both backbone forwards keep
their BF16 region. This does not make the whole transformer FP32 or duplicate
the separate fixed-budget ancestral evaluation. An initial evaluation is
also enabled. The tests below use only synthetic short training steps, with no
external datasets, downloads or tracker runs.

## Checks

```bash
CUDA_VISIBLE_DEVICES='' python -m tests.test_objective_fast
CUDA_VISIBLE_DEVICES='' python scripts/test_mmd_archive_integration.py
CUDA_VISIBLE_DEVICES='' python scripts/test_gsm8k_fp32_pareto.py
CUDA_VISIBLE_DEVICES='' python scripts/test_evaluated_checkpoint.py
CUDA_VISIBLE_DEVICES='' python scripts/test_validation_schedule.py
```

The archive test suite checks B/G combinations, per-candidate rewards,
gradients, empty/singleton masks, deduplication, scored heads and the paper
surrogate. Integration checks use real tiny DiT modules and synthetic tokens;
they also check the autocast fix, accumulation, checkpoint load and validation.
