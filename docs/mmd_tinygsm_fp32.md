# TinyGSM MMD with the complete model in FP32

This is an FP32 variant of the TinyGSM MMD recipe behind Figure 3, run
`3e0ac6b1-7a8f-43c3-af58-d9e21b886da7`. Both the student and the frozen feature
model start from the original `checkpoints/tinygsm/mdlm_notok.ckpt`. The student
retains that checkpoint's EMA history and starts a fresh optimizer and scheduler.

After the usual Nirvana data/checkpoint setup, run:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/train_mmd_tinygsm_fp32_nirvana.sh" \
  'logger.name=mdlm to mmd tinygsm full fp32 4xB300 g8'
```

Use the same `nirvana_checkpoints` and `tinygsm_mmd_offline_bundle` inputs as the
other TinyGSM launches. A CDLM snapshot and CDLM trajectories are not required.
The [complete setup example](../examples/nirvana_mmd_tinygsm_fp32_run_command.sh)
includes dependency installation and input placement. Preflight resolves the
original MDLM once, records its SHA-256 for both roles in
`mmd_initialization.json`, and fails before training workers if it is missing.

| Setting | Value |
| --- | --- |
| Model/data | Small full-attention MDLM, length 512, TinyGSM answer-only, train on PAD |
| Effective batch | 4 GPUs × batch 32 × accumulation 4 = 512 |
| Group | G=8, one sequence per candidate, summed historical group objective |
| Features | Original raw MDLM, block 6, clean input |
| Kernel | Historical paired-token RBF, alpha `6e-5` |
| Physical student/feature batch | 8, student activation checkpointing enabled |
| Optimizer | AdamW, LR `1e-5`, betas `.9/.999`, epsilon `1e-8`, weight decay 0 |
| Schedule | 2500 warmup updates, 7000 updates total |
| EMA | `0.9999`, original MDLM history retained |
| Validation | Every 500 optimizer updates, thirteen confidence thresholds, T=0 |
| Periodic export | Student EMA weights only, every 500 optimizer updates, saved to snapshot |
| Output/snapshot namespace | `mmd_tinygsm_fp32` |

G=8 matches the paper run. The separate CDLM-to-MMD preset continues to use its
requested G=2. The selected paper checkpoint was step 7000; to run the original
20,000-update launch budget, append `trainer.max_steps=20000`.

## What full FP32 means here

The preset sets `trainer.precision=32-true`, `model.force_fp32=True`, and
`training.loss_precision=float32`. DiT disables its internal BF16 autocast for
the complete forward, including timestep embeddings, attention, MLPs and output
projection. RoPE uses FP32 PyTorch operations and full attention uses the math
backend of scaled dot-product attention. Logit normalization, probabilities,
log probabilities, the MMD kernel and gradient computations use FP32 as well.
The frozen feature model follows the same FP32 forward path.

Runtime initialization disables TF32 through the applicable PyTorch precision
API, and the launcher sets `NVIDIA_TF32_OVERRIDE=0` before Python starts. A
conflicting mixed-precision trainer setting fails during preflight. Startup
prints `[precision] Full FP32: IEEE matmul and TF32 disabled.`

This changes the numerical protocol of the mixed-precision paper run, including
its feature extraction and evaluation. It is an explicit precision experiment,
not a claim of identical results. Full FP32/math attention can require more GPU
memory and time. The default batch is configured for four B300 GPUs; no full GPU
training or peak-memory benchmark was run locally.

## Logging, saved weights and evaluation

Underdeep uses `generative-models/tinygsm`, the existing MMD loss/reward metrics,
and GSM8K accuracy/NFE before training and every 500 optimizer updates. Ancestral
validation is disabled. The new output directory and snapshot namespace keep
this run separate from CDLM-to-MMD.

Every 500 optimizer updates, `checkpoints/step=00000500.ckpt`,
`step=00001000.ckpt`, etc. contain the student's FP32 EMA weights plus metadata.
At the end, `checkpoints/last.ckpt` contains the full training state. The
[existing snapshot/resume policy](mmd_from_cdlm_tinygsm.md#checkpoints-and-snapshots)
applies. To resume this FP32 run from its full `last.ckpt`, reconnect its snapshot
and append `checkpointing.resume_from_ckpt=True` to the same launcher.

For standalone evaluation of the saved weights, retain the
`+experiment=mmd_tinygsm_fp32` preset so that evaluation uses the same complete
FP32 forward. Existing mixed-precision presets keep their previous execution
path by default. CPU tests exercise the actual small DiT forward/backward,
checkpoint compatibility, the FP32 runtime controls and preflight/launcher.
