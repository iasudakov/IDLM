MDLM: recompute Figures 4, 6 and 14
=================================

Nirvana source code: branch feat/tinygsm-mmd-archive-v2 including this suite.
The suite extends the corrected archive implementation from e6a0c22.
Do not use tinygsm-upstream-repro for these new training commands: it lacks the corrected archive objective.

Use the tracked launcher for one job at a time:
  bash "$SOURCE_CODE_PATH/code/scripts/run_paper_recompute.sh" JOB --devices 2

The launcher installs the pinned Python dependencies in the same interpreter,
then runs the code in scripts/paper_recompute. No embedded code is required.
For an already prepared local environment, set PAPER_SKIP_INSTALL=1.

Resources: 2 GPUs by default. Pass --devices 4 or --devices 8 for training;
manual accumulation and checkpoint cadence adapt to keep global batch 512.
Changing the GPU count does not promise identical RNG/numerical trajectories.

Inputs for every job:
  - nirvana_checkpoints (data/gsm8k_test.json, checkpoints/tinygsm/mdlm_notok.ckpt;
    training also needs the prepared cache_dir/*_sep5c6e/train and validation)
  - tinygsm_mmd_offline_bundle (local SmolLM and gpt2 tokenizer assets)
For train-ar also attach ar_notok.ckpt, unless already present in
nirvana_checkpoints/checkpoints/tinygsm/ar_notok.ckpt.

Existing moved code/cache_dir, code/data and code/checkpoints locations are
also accepted. Inputs are not moved. HF downloads are disabled during jobs.
pip uses the environment's package access. If needed, set PAPER_PROXY to
http://proxy-host:port in the run-command environment;
the bootstrap registers the host and uses it only for package installation.
PAPER_SKIP_INSTALL=1 skips installation when the required environment exists.

Figures 4 and 6: fresh G=4 training
---------------------------------
Run each of these in a separate fresh job WITHOUT a previous state input:

  train-attraction
  train-regression
  train-linear
  train-sequence_rbf
  train-embeddings
  train-ar

Optional control, useful for an independent repeat of the default setup:
  train-baseline

All use the corrected archive's fast student head, sampling, leave-one-out
baseline and candidate/draw loss normalization, G=4, B=2, global batch 512,
lr=1e-5, layer 6 (except embeddings), and 8000 saved global optimizer steps.
New variants use chunked reference feature extraction with the same fast
student loss. Their feature scorer can be slower than the fused baseline.
They do not silently switch to the old legacy training objective.

Variant definitions:
  baseline:       original new token RBF, alpha=6e-5, layer 6.
  attraction:     same real-generated term; remove generated-generated term.
  regression:     negative aligned feature MSE, averaged over B draws/tokens/
                  hidden coordinates; common archive loss normalization.
  linear:         negative squared distance between pooled feature means
                  (linear_l2, matching the type of the old linear ablation).
  sequence_rbf:   pool each sequence, alpha=2.9e-4; cross-draw off-diagonal
                  generated-generated term, omit the real-real constant.
  embeddings:     same token RBF, pre-attention embedding layer, alpha=6e-3.
  ar:             same token RBF and alpha=6e-5, native causal AR extractor,
                  clean inputs and layer 6.

These are NEW experiments aligned to the current G4/B2 setup, not a recovery
of the old G8/B1 trajectories. In particular B is now 2 in the linear and
sequence-RBF variants as well. Historical run provenance is supplied separately from this public code.

Evaluation: BF16 backbone forward with FP32 prediction normalization, as in
the current updated curves. This is NOT a full-FP32 backbone forward.
Both the original BF16-normalization sweep and the added _fp32norm sweep run;
use only _fp32norm for the new article figures.
Validation starts at 5500 and repeats every 500 global steps through 8000.
The threshold grid extends through 0.99 and 0.995 to cover the expensive end.

Weights are saved every 500 global steps, including steps before 5500. We use
one periodic checkpoint stream to avoid duplicate multi-GB checkpoint files.
At two GPUs accumulation=8, cadence=4000 train batches. Consequently the
filename step=00064000.ckpt contains global_step=8000. The evaluator checks
the saved global_step, not the filename. Save state for every new run.

Figure 14: ancestral T=0 pass@k
-----------------------------
32 independently seeded generations per question, all 1319 questions,
budgets 4/8/16/32, k=1/2/4/8/16/32. The estimator is
  mean_questions[1 - C(n-c,k)/C(n,k)], n=32.
Greedy token choice at T=0 still has random reveal decisions. No confidence
sampling is substituted. One model requires 168832 generated answers.
All predictions, correctness flags, actual NFE, configuration and hashes are
saved, and new metrics go to a separate Underdeep evaluation run.

  passk-mdlm
    Original mdlm_notok.ckpt, saved EMA, distillation iteration 0.
    Does not need a state input.

  passk-mmd --source-run RUN_UUID --step 8000
    Connect the source run state and pass its UUID explicitly.
    Requires a saved 8000 checkpoint and its EMA.

  passk-idlm --source-run RUN_UUID --step 5000
    Connect the source run state and pass its UUID explicitly.
    Explicitly uses saved 5000 RAW student weights.

  passk-idlm-reinforce --source-run RUN_UUID --step 5000
    Connect the source run state and pass its UUID explicitly.
    Explicitly uses saved 5000 RAW student weights.

The example commands use step 5000 explicitly. For a different saved
checkpoint pass --step with its true global optimizer step. The evaluator
rejects missing steps rather than silently choosing a nearby checkpoint.
Default target steps are 8000 (MMD), 4500 (IDLM) and 5500 (IDLM-REINFORCE).
Only use a default when that checkpoint actually exists in the supplied state.

RAW is explicit for the two IDLM baselines because their historical training
validation can skip EMA.copy_to when the student is frozen. An EMA evaluation
would be a separate comparison. MMD/MDLM commands explicitly use saved EMA.
No newly evaluated points are merged with old results.

Preflight and direct usage
-------------------------
Append --plan-only to the launcher command to inspect the resolved training
configuration or exact checkpoint selection without using a GPU. The default
starts the requested training/evaluation. --dry-run only prints the command.

Direct usage from the repository root (dependencies already installed):
  python scripts/paper_recompute/launch.py train-attraction --devices 2
  python scripts/paper_recompute/launch.py passk-mmd --source-run RUN_UUID --devices 2 --step 8000

For a newly trained ablation checkpoint (example):
  PYTHONPATH="$PWD" python scripts/paper_recompute/paper_eval.py --suite confidence \
    --checkpoint /path/to/step=00064000.ckpt \
    --config /path/to/run/.hydra/config.yaml \
    --step 8000 --weights ema --devices 2 \
    --model-weights-dir /path/to/tinygsm_mmd_offline_bundle \
    --data-path /path/to/gsm8k_test.json

If running outside Nirvana add --no-snapshot. --no-underdeep disables tracker
publication. Output paths default to unique new directories. On Nirvana the
evaluation outputs are copied into the output state without overwriting the
source run directory.

Verification performed
----------------------
CPU tests passed: reward equations including empty masks; fast/reference
losses and student gradients for all 7 variants; RAW/EMA student-only loading;
exact checkpoint-step selection; pass@k against combinatorial calculations.
AR adapter CPU tests use a causal toy backbone because native AR rotary
kernels require CUDA. No full GPU training or benchmark was run here.

Run the CPU checks from the repository root:
  PYTHONPATH="$PWD" python scripts/paper_recompute/test_paper.py
