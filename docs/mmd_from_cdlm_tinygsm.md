# MMD finetuning from a completed CDLM snapshot

This recipe starts the MMD student from the latest saved CDLM checkpoint and
keeps the **original MDLM** as its frozen feature extractor. It uses the actual
TinyGSM MMD run underlying the paper's Figure 3, `3e0ac6b1-7a8f-43c3-af58-d9e21b886da7`.
It creates a new training stage and a new Underdeep run. It does not generate or
read CDLM trajectories.

## Nirvana command

Keep the same `nirvana_checkpoints` and `tinygsm_mmd_offline_bundle` inputs and
environment setup used for CDLM. Supply the updated code and the previous
CDLM training `state` through the block's snapshot input. The launcher reads
the unpacked directory returned by `nirvana_dl.snapshot.get_snapshot_path()`.
For a snapshot containing the namespace used in the preceding example:

```bash
CDLM_SOURCE_NAMESPACE=cdlm_tinygsm_ancestral_sweep \
bash "$SOURCE_CODE_PATH/code/scripts/train_mmd_from_cdlm_tinygsm_nirvana.sh" \
  'logger.name=cdlm to mmd tinygsm 4xB300 paper recipe'
```

Use `CDLM_SOURCE_NAMESPACE=cdlm_tinygsm` for the original default CDLM namespace.
Omit that variable to autodetect a single CDLM training namespace. Multiple
CDLM runs require an explicit choice; steps from different runs are not compared.
If state is staged as a separate, already unpacked input directory, set
`CDLM_SNAPSHOT_ROOT=/absolute/path/to/unpacked/state` as well. This bypasses the
SDK's snapshot discovery. A trajectory-only snapshot cannot supply trained CDLM
weights. The [complete setup example](../examples/nirvana_mmd_from_cdlm_run_command.sh)
includes the existing environment and input preparation.

Before launching training workers, the preflight inspects `.ckpt` files in the
selected namespace's `checkpoints/` directory. It selects the greatest saved
`global_step`, preferring `last.ckpt` only on ties. Unreadable files are reported
and skipped. File names and modification times do not determine training age.
The chosen source, step, SHA-256 and independent feature checkpoint are recorded
in `mmd_initialization.json` and embedded in subsequent full MMD checkpoints.

The default student starts from **raw CDLM weights and its saved EMA history**,
following the historical MMD warm-start policy. The optimizer, scheduler,
training step and sampler start fresh. CDLM's teacher head is discarded. EMA
updates and EMA validation continue for the MMD student only. For an explicitly
different warm start from CDLM's EMA weights, use both
`training.student_init_use_ema=True mmd.retain_student_ema=False`.

## Training and validation

| Setting | Value |
| --- | --- |
| Model and data | Small full-attention MDLM, length 512, TinyGSM answer-only loss, PAD corruption |
| Effective batch | 4 GPUs × 32 examples × accumulation 4 = 512 |
| Student forward | Chunks of 8, activation checkpointing |
| Frozen feature source | Raw `checkpoints/tinygsm/mdlm_notok.ckpt` |
| Feature layer / noise | Block 6 (zero-based), clean input, noise 0 |
| Kernel | Historical paired-token RBF, alpha `6e-5` |
| Group | 8 candidates, 1 sequence each; sum over candidates |
| Feature forward batch | 8 |
| Optimizer | AdamW, LR `1e-5`, betas `.9/.999`, epsilon `1e-8`, no weight decay or clipping |
| Warmup / budget | 2500 / 7000 optimizer updates |
| Precision | BF16 mixed, FP32 feature-kernel arithmetic |
| EMA | `0.9999`; inherit CDLM EMA shadows and update counter |
| Validation | Every 2000 microbatches = 500 optimizer updates |

The paper selected optimizer step **7000** from a launch with a 20,000-update
cap. This recipe defaults to 7000 new MMD updates. To reproduce the old maximum
budget instead, append `trainer.max_steps=20000`. The public `dlm-mmd` launcher
currently uses a different two-sequence estimator; this recipe deliberately
uses the historical Figure 3 estimator, adapted from IDLM
`tinygsm-upstream-repro` revision `14594e9` (`mmd_reward.py`, `mmd_rbf.py`).

The loss uses a leave-one-out group baseline. Kernel self-pairs within each
sequence are excluded, cross-pairs use all selected positions, and negative
finite-sample MMD² estimates are retained. Only sampled-token log probabilities
carry gradients. Normalization is over the full logical microbatch, not each
physical forward chunk. The feature network stays frozen throughout training.
Four versus eight GPUs can change RNG streams and numerical results; this is
a matched training recipe, not a claim of bitwise reproduction.

Underdeep keeps the same `generative-models/tinygsm` experiment, with loss,
reward, advantage, learning rate, NLL and confidence-validation metrics.
Confidence decoding uses the reduced 10-threshold grid and temperature 0 on
all 1,319 GSM8K questions. **Ancestral validation is disabled for this stage.**
The previous CDLM and SDTT presets retain their existing evaluation settings.

## Checkpoints and snapshots

Every **1000 optimizer updates**, a numbered checkpoint contains **only the
student's EMA inference weights**, with the minimal metadata needed to load
them. It excludes optimizer/scheduler state, CDLM trajectories and the frozen
feature network. All numbered files are retained and published to the separate
snapshot namespace `mmd_from_cdlm_tinygsm`:

```text
mmd_from_cdlm_tinygsm/
  mmd_initialization.json
  checkpoints/
    step=00001000.ckpt
    step=00002000.ckpt
    ...
    step=00007000.ckpt
    last.ckpt
```

The final **`last.ckpt` is a full training checkpoint**, saved before final
evaluation and published through the checkpoint I/O wrapper. It includes the
live student, EMA, frozen feature network, optimizer, scheduler and loop state.
The final evaluation history and tracker metadata are published afterward.
Periodic EMA exports are for evaluation or a new finetuning stage; full training
resume requires `last.ckpt`. Snapshot SDK failures are reported and local
checkpoint files remain available.

To resume the MMD stage from its full checkpoint, reconnect its snapshot and run:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/train_mmd_from_cdlm_tinygsm_nirvana.sh" \
  checkpointing.resume_from_ckpt=True \
  trainer.max_steps=20000
```

On full MMD resume, the preflight skips CDLM selection and original feature asset
preparation: both models are inside the MMD checkpoint. Data and tokenizers are
still required. Full resume reuses the saved Underdeep run unless
`logger.force_new_run=True` is supplied. Missing full checkpoints cause an error.

## Verification

CPU checks cover snapshot selection, fresh-stage isolation, frozen features,
EMA selection, checkpoint export, snapshot publication, toy optimizer updates
and full resume. Dense kernel and gradient references check the historical MMD
objective, including chunked student forwards. Real TinyGSM training and GPU
evaluation are left to the Nirvana launch.
