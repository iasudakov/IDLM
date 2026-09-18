# Save every evaluated training checkpoint

Add these overrides to the underlying `python -m main` training command:

```bash
checkpointing.save_evaluated_checkpoints=True \
checkpointing.save_evaluated_from_step=1000
```

After each validation during training, this retains a full checkpoint under
`<checkpointing.save_dir>/checkpoints/eval-step=00001000.ckpt`, and similarly
for later evaluated steps. The threshold is inclusive and uses Lightning's
optimizer/global step, the same counter used by the Pareto series. It does not
count training batches. If no validation occurs exactly at the threshold, the
first later validation is saved. This does not change the validation cadence.

All validation passes qualify, including NLL-only passes outside an OWT
generative-evaluation window. Sanity checks and standalone evaluation runs are
excluded. There is no final save unless a validation occurs there. No metric
ranking or automatic Pareto-best selection is applied; every qualifying model
is retained.

If you use `eval.eval_step_range` or `eval.eval_at_steps`, also remove the
default generative-perplexity monitor with `'~callbacks.checkpoint_monitor'`,
or set `callbacks.checkpoint_monitor.monitor=val/nll`. The existing validation
window check rejects a `val/gen_ppl` monitor even when its saves are disabled.

Files include model parameters and the existing optimizer, scheduler and EMA
checkpoint state. Saving happens after the module restores live parameters
following EMA evaluation, so raw model weights and EMA shadows remain distinct.
The existing EMA-aware evaluation loaders should be used to reproduce the
EMA-based metrics. This feature does not repair the training loop's existing
limitations around exact resumption of manual distillation.

The callback persists its last saved optimizer step to suppress duplicate
validation saves after resume. If a new run/callback encounters an existing
filename, it adds `-v1`, `-v2`, etc., preserving the existing file. All DDP ranks
participate in checkpoint saving, while rank zero chooses the filename and
writes the file. Filename versioning assumes one training job writes to a
directory at a time. With OWT snapshot persistence enabled, completed saves are
also pushed through the existing Nirvana snapshot helper.

This policy is independent of the periodic and metric-best checkpoint
callbacks, which keep their current settings. To use only validation-based
files and avoid earlier periodic/best saves, also set:

```bash
checkpointing.use_periodic_train_batch_checkpoint=False \
callbacks.checkpoint_every_n_steps.save_top_k=0 \
callbacks.checkpoint_every_n_steps.save_last=False \
'~callbacks.checkpoint_monitor'
```

The new callback does not create `last.ckpt`. To resume from one of its files,
set `checkpointing.resume_ckpt_path` to the chosen `eval-step=...ckpt` path.
