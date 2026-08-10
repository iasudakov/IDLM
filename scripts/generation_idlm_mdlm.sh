# Sweeps denoising steps and reports the D-MMD paper's evaluation for each
# (arXiv:2603.20155, Table 2): 1024 unconditional tokens scored by the GPT-2
# Gradient Moment (lower = closer to the data), alongside generative perplexity
# and sample entropy.
#
# GM is the metric to read. Generative perplexity is gameable -- a collapsed,
# repetitive sampler scores a *better* ppl than real text (measured with this
# reference model: ppl 1.6 vs 25.9) while GM correctly rejects it.
#
# Two things GM needs that generative perplexity did not:
#   * real data to compare against, hence `data.cache_dir` -- the config default
#     points at a stale sandbox path, and plain sample_eval never loaded data, so
#     this override is new here;
#   * at least 2 paired (generated, data) batches. The estimator averages the
#     M(M-1) cross-pairs, so M = min(gm_num_batches, num_sample_batches) is what
#     its variance rides on; M = 10 below, i.e. all 80 generated sequences.
# GM values only compare across runs with the same eval_batch_size, model.length
# and M.

noise_removal=ancestral
outdir=${OUTDIR:-$(pwd)/gm_sweep}
mkdir -p "$outdir"

for steps in 4 8 16 32; do
  echo "Running with steps = $steps"

  python main.py \
    mode=sample_eval \
    loader.batch_size=2 \
    loader.eval_batch_size=8 \
    data=openwebtext-split \
    data.cache_dir=/mnt/data/owt \
    algo=mdlm \
    algo.backbone='hf_dit' \
    eval.checkpoint_path="kekchpek/idlm-mdlm" \
    sampling.steps="$steps" \
    sampling.num_sample_batches=10 \
    sampling.noise_removal=$noise_removal \
    +wandb.offline=true \
    sampling.predictor='ancestral_cache' \
    eval.compute_gm=True \
    eval.gm_num_batches=10 \
    eval.gm_eval_model_name_or_path=gpt2 \
    eval.generated_samples_path="$outdir/samples_steps${steps}.json" \
    adversarial_distill.is_distill=True \
    sampling.predictor=ancestral_cache \
    logger.force_new_run=true
done

# The sweep in the paper's Table 2 layout, so it is readable without scrolling
# back through each run's output.
python - "$outdir" <<'PY'
import glob, json, os, re, sys
rows = []
for path in glob.glob(os.path.join(sys.argv[1], 'samples_steps*.json')):
  with open(path) as f:
    report = json.load(f)
  rows.append((int(re.search(r'steps(\d+)', path).group(1)),
               report.get('gpt2_gm'), report['generative_ppl'],
               report['entropy'], report.get('gm_num_pairs')))
print(f'\n{"NFE":>5}  {"GPT-2 GM":>10}  {"gen ppl":>10}  {"entropy":>8}  {"M":>3}')
for nfe, gm, ppl, entropy, pairs in sorted(rows):
  gm = 'n/a' if gm is None else f'{gm:.4f}'
  print(f'{nfe:>5}  {gm:>10}  {ppl:>10.2f}  {entropy:>8.3f}  {str(pairs):>3}')
PY
