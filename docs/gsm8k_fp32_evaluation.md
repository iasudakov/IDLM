# Additional FP32 prediction evaluation

On `tinygsm-upstream-repro`, both IDLM (DMD) and IDLM-REINFORCE use the same
GSM8K validation threshold sweep. Each enabled sweep now also runs with FP32
prediction normalization. Existing training commands keep their original
metrics and automatically add the second curve.

To make the setting explicit, append this Hydra override to the existing
training command:

```bash
++eval.gsm8k_pareto_fp32=true
```

Use `++eval.gsm8k_pareto_fp32=false` to keep only the original sweep.
`eval.gsm8k_at_validation` and a nonempty `eval.gsm8k_pareto_thresholds` must
still be enabled. The additional pass follows the original validation cadence,
dataset, per-rank batch size, temperature, threshold grid and weight/EMA choice.
This approximately doubles the cost of the threshold sweep.

The backbone forward keeps its configured precision (BF16 in the paper runs).
Only raw logits are converted to FP32, before mask suppression and SUBS
logsumexp normalization. Probabilities and confidence decisions then use those
FP32 predictions, matching the MMD FP32-normalization comparison. This does
not enable FP32 training or change the fixed-budget/ancestral, greedy-token,
NLL or standalone `gsm8k_eval` paths.

Original tracker keys are preserved. The extra pass writes:

- `val/gsm8k_acc_thr{tag}_fp32norm` and `val/gsm8k_nfe_thr{tag}_fp32norm`;
- `pareto/gsm8k_step_{training_step}_fp32norm` (x axis is NFE multiplied by 100);
- `pareto/frontier_fp32norm`;
- `pareto_history_fp32norm.json` under `checkpointing.save_dir`.

The extra sweep replays the original sweep's starting Python, NumPy, CPU and
active CUDA RNG states, then restores the post-original state. Different
precisions may take different numbers of decoding steps, so this pairs initial
RNG states, not every subsequent stochastic draw. The normalization override
is restored on success and failure; training gradients and EMA logic are
unchanged.

CPU regression checks:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/test_eval_prediction_precision.py
CUDA_VISIBLE_DEVICES='' python scripts/test_gsm8k_fp32_pareto.py
```
