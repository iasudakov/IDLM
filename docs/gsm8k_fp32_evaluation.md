# Additional FP32 prediction evaluation

On `didi-tinygsk`, TinyGSM DiDi-Instruct validation now adds an FP32-normalized
pass to every enabled GSM8K confidence-threshold sweep. Existing commands keep
their original metrics and automatically add the second curve. To make the
setting explicit, append this Hydra override to the training command:

```bash
++eval.gsm8k_pareto_fp32=true
```

Use `++eval.gsm8k_pareto_fp32=false` to keep only the original sweep.
`eval.gsm8k_at_validation` and a nonempty `eval.gsm8k_pareto_thresholds` must
still be enabled. The second pass follows the original validation cadence,
dataset, per-rank batch size, temperature, threshold grid, maximum-step cap
and configured backbone/student-EMA choice. It approximately doubles the
threshold-sweep evaluation cost.

The backbone forward keeps its configured precision (BF16 in the paper run).
Only raw logits are converted to FP32, before mask suppression and SUBS
logsumexp normalization. Probabilities and confidence decisions then use those
FP32 predictions, matching the MMD FP32-normalization comparison. Training,
NLL, conditional validation and the fixed-budget/guided sampler are unchanged.
The adaptive sweep continues to use standard SUBS predictions without the
discriminator or guided inference sampler. Pinned upstream sources are unchanged.

Original tracker keys are preserved. The extra pass writes:

- `val/gsm8k_acc_thr{tag}_fp32norm` and `val/gsm8k_nfe_thr{tag}_fp32norm`;
- `pareto/gsm8k_step_{training_step}_fp32norm` (x axis is NFE multiplied by 100);
- `pareto/gsm8k_thresholds_fp32norm`, `didi/tinygsm_pareto_protocol_fp32norm`
  and `pareto/frontier_fp32norm`;
- `didi_tinygsm_pareto_history_fp32norm.json` under `checkpointing.save_dir`.

History and table retain exact mean per-sequence NFE, including points omitted
from approximate scalar series when integer NFE*100 coordinates collide.

Both sweeps begin with the same Python, NumPy, CPU and active CUDA RNG states.
Different decoding lengths may consume different subsequent draws. The outer
validation callback preserves training RNGs, module modes and gradients; the
temporary normalizer is restored even if generation fails on one rank.

CPU regression checks:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/test_eval_prediction_precision.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/test_didi_tinygsm_eval.py
CUDA_VISIBLE_DEVICES='' python scripts/test_didi_tinygsm_pareto.py
CUDA_VISIBLE_DEVICES='' python scripts/test_didi_tinygsm_adaptive.py
```
