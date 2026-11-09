# Frozen SDLLM MDLM-1.7B features for TinyGSM and OWT MMD

Select `adversarial_distill.mmd_feature_source=sdllm_mdlm` to use
[nvidia/SDLLM-MDLM-1.7B-Base](https://huggingface.co/nvidia/SDLLM-MDLM-1.7B-Base)
as the frozen feature model. TinyGSM uses the original small MDLM checkpoint
and SmolLM tokenizer; OWT uses `kuleshov-group/mdlm-owt` and GPT-2. The inherited recipe and
archived paper-run provenance are described in [mdlm_large_features.md](mdlm_large_features.md).

## TinyGSM launch

```bash
pip install tiktoken==0.9.0
MDLM_CHECKPOINT=/path/to/tinygsm/mdlm_notok.ckpt \
TRAIN_DEVICES=4 bash scripts/train_mmd_tinygsm_sdllm.sh
```

`MDLM_CHECKPOINT` supplies the existing **small TinyGSM student** initialization;
its default is `checkpoints/tinygsm/mdlm_notok.ckpt` inside this repository.
SDLLM feature weights are downloaded automatically as described below.
`MODEL_WEIGHTS_DIR=/path/to/tinygsm_mmd_offline_bundle` optionally resolves
existing TinyGSM assets; `CACHE_DIR` sets the dataset cache.

This launcher defaults to layer 6 (zero-based in SDLLM), G=4, B=2,
cross-sentence/same-position exclusions, automatic RBF alpha, LR=1e-5,
source length 512, and GSM8K greedy/Pareto evaluation. `FEATURE_LAYER` and
trailing Hydra arguments override these choices. The pretrained SDLLM layer
is a separate feature space, so the old TinyGSM alpha 6e-5 is not copied.

On 4 GPUs the local batch is 32, accumulation is 4, and global batch is 512.
The raw validation/log/checkpoint intervals are 2000/200/4000. On 8 GPUs,
accumulation is 2 and those intervals are 1000/100/2000. Validation and sample
batch counts also scale to preserve the number of evaluated examples.

The archived OWT source did not contain conditional GSM8K validation, so this
option also adds an opt-in TinyGSM evaluation hook. It runs fixed-budget
random-unmask/greedy-token decoding (64 scheduled steps with S-FLM final-step
counting) and the 27 confidence thresholds in the launcher. It logs
`val/gsm8k_acc_greedy`, per-threshold accuracy/NFE, the Underdeep Pareto curve,
and `pareto_history.json`. Counts are pooled over rank-strided test shards;
the GSM8K test-set size does not change with GPU count. Generation uses the
student/EMA weights; the SDLLM feature model is only used by the training reward.
The conditional hook avoids unconditional OWT text generation on TinyGSM.

GSM8K data is read from `eval.gsm8k_data_path`, the offline bundle's
`data/gsm8k_test.json`, or the dataset cache. Missing test data is downloaded
automatically. Grading follows the existing S-FLM function-execution protocol
in a separate process. Evaluation saves/restores the model-device Torch RNG.

TinyGSM must use `data.wrap=False`, `data.train_on_prompt=False`,
`mmd_score_positions=all`, and clean features (`mmd_noise_level=0`). Here
**all means all reference-answer content**, including its EOS, not the prompt
or padding. The existing content mask fixes the answer window for every draw.
Prompt and answer are decoded/tokenized separately, concatenated, and fed to
SDLLM together. MMD selects only the resulting native answer features, while
the question contributes through bidirectional context. BPE merges across the
prompt/answer boundary are intentionally prevented; no byte-offset projection
is used. Same-position exclusion is relative to the native answer start.

Suffix padding outside the reference answer is removed before the teacher
forward. A generated PAD *inside* the reference answer window is kept as
ordinary text; a generated EOS does not shorten that window. This matches the
existing estimator's use of a fixed reference span. Student gradients are
still applied only to the originally written source tokens.

```bash
TRAIN_DEVICES=4 bash scripts/train_mmd_tinygsm_sdllm.sh --cfg job --resolve
```

## OWT launch

Install the additional tokenizer dependency in the training environment:

```bash
pip install tiktoken==0.9.0
TRAIN_DEVICES=4 bash scripts/train_mmd_owt_sdllm.sh
```

The launcher defaults to layer 3 (zero-based), G=8, B=2, clean token-RBF features,
cross-sentence and same-position exclusions, and **automatic bandwidth**
`mmd_rbf_alpha=null`. The old model's alpha 2e-5 is not assumed appropriate in
the new feature space. Set `FEATURE_LAYER=13` or pass explicit Hydra overrides
after the script name to choose another layer/bandwidth. Layers 0–27 and −1
(embeddings) are supported, including the existing weighted layer-list syntax.

By default the launcher uses 8 GPUs. `TRAIN_DEVICES=4` keeps local batch 8 and
global batch 512, changes manual accumulation 8→16, and doubles raw validation,
logging and checkpoint intervals. Evaluation remains 128 texts per seed.
This preserves the update and evaluation budgets; floating-point reductions
and random-number partitioning can differ across GPU counts.
`MODEL_WEIGHTS_DIR=/path/to/owt_mmd_offline_bundle` optionally supplies the
original student/tokenizer/perplexity-evaluator bundle.

Before the first training batch, missing files are automatically downloaded
from immutable Hub revision `4007e0b2a23465a31adf282d843a61172932af2a`:

- `config.yaml`
- `tokenizer/tiktokenizer.json`
- `model.ckpt` (the native Lightning checkpoint, approximately 26.5 GB)

The normal Hugging Face cache and its file locks are used, including across
DDP workers. Set `HF_HOME`/`HF_HUB_CACHE` to change the cache location. An
explicit `SDLLM_MODEL_DIR=/path/to/release` uses those same three local files
and fails if any is missing. Download failures stop startup before training
updates. A config-only check does not download or initialize the model:

```bash
TRAIN_DEVICES=4 bash scripts/train_mmd_owt_sdllm.sh --cfg job --resolve
```

## Native-token MMD

Source and feature tokenizers differ. Each clean source sequence is decoded
as text and encoded with SDLLM's released tokenizer. Actual source BOS/EOS
boundaries are preserved as native BOS/EOS. Ordinary text that resembles a
special-token spelling is kept as ordinary text. The source tokenizer's normal Unicode
decoding semantics apply, including replacement of invalid UTF-8 fragments.
No byte-offset mapping or projection onto source tokens is performed.

MMD operates on SDLLM's own token features. Sequences can have different native
lengths: real/fake kernel sums use their own valid-token pair counts. With
cross-sentence exclusion, independently sampled sentences receive equal weight
after each pair's normalization. The fake/fake term excludes pairs within the
same draw; optional same-position exclusion removes equal **native ordinal**
indices between draws. As in the original cross-sentence estimator, the entire
real/real term is omitted and signed estimates are retained. Equal-length
inputs recover the original cross-sentence formula.

This option requires `mmd_reward_type=feature`, `mmd_kernel=rbf`,
`mmd_kernel_level=token`, `mmd_kernel_source=draws`,
`mmd_score_positions=all` and `mmd_noise_level=0`. The launcher sets these.
For OWT all native positions are scored. TinyGSM instead accepts a contiguous
reference-answer mask and drops suffix padding as described above. Source
MASK IDs within the teacher input, context overflow, and scored native spans
shorter than two tokens fail explicitly. There is no silent truncation.

The existing REINFORCE sampling, group baseline and source-token log-probability
gradient are used. Scoring all native positions changes the reward compared
with scoring only newly written source positions. Enabling SDLLM intentionally
changes the feature map and objective; it is not a seed-equivalent reproduction
of the archived diffusion-feature experiment.

## Loader and computation

The released tokenizer reserves 1000 special IDs (BOS 1 / EOS 2 / MASK 3); ordinary
byte-BPE ranks are shifted by 1000. Vocabulary size is inferred from the actual
checkpoint embedding and validated against the tokenizer, rather than using
the paper's rounded 128k value. The audited release configuration has 28 blocks,
hidden size 2176, 17 heads and context 2048.

The loader strictly checks every backbone key, shape and FP32 dtype. It reads
native `state_dict.backbone.*` weights; EMA and optimizer weights are not
substituted. The native checkpoint uses Python pickle, so local overrides
must be trusted artifacts. No downloaded Python model code is executed.
Memory mapping avoids materializing the checkpoint's optimizer state in RAM.

The local archived DiT has the released model's parameter layout and block
equations. Its PyTorch SDPA backend can differ numerically from the release's
FlexAttention. The existing PyTorch environment is retained. The model stays
frozen/in eval mode and outside the student's optimizer/checkpoint state.
Forwards stop after the deepest requested layer and skip the vocabulary head.
Only equal-length native sequences are batched together, avoiding padding in
bidirectional attention. `mmd_forward_batch_size` caps the physical batch;
bandwidth and reward aggregation still use the original logical batch.

Native length bucketing can produce small feature-model batches. The current
scoring loop also repeats clean-feature extraction for group candidates and
uses separate blocked kernels for each sentence pair. Automatic bandwidth
samples up to 512 real features and consumes RNG. Production speed has not
been measured; this option does not claim the old feature path's throughput.

## Verification

```bash
OMP_NUM_THREADS=1 python -B scripts/test_sdllm_features.py
OMP_NUM_THREADS=1 python -B scripts/test_sdllm_reward.py
OMP_NUM_THREADS=1 python -B scripts/test_sdllm_feature_routing.py
OMP_NUM_THREADS=1 python -B scripts/test_sdllm_tinygsm.py
OMP_NUM_THREADS=1 python -B scripts/test_tinygsm_validation.py
OMP_NUM_THREADS=1 python -B scripts/test_tinygsm_validation_hooks.py
```

CPU fixtures cover strict native-checkpoint loading, tokenizer conversion,
post-block features against a full tiny-DiT forward, variable-length batching,
frozen/eval behavior, RNG preservation, reward formulas and REINFORCE backward.
All 51 new CPU tests passed (24 loader/end-to-end, 16 reward, 11 routing/startup).
An additional 14 TinyGSM tests cover the real SmolLM tokenizer, prompt context,
answer-only features, padding/EOS handling, and the actual training-loss call.
All 17 conditional-evaluation tests and 8 validation-hook tests also passed.
The greedy sampler matched the archived 14594e methods exactly in 18 seeded
CPU cases, including forward inputs, NFE and the final RNG state. DDP count
aggregation was exercised with simulated uneven/empty rank shards.
Existing SDTT/original-feature regression tests also passed; their opt-in CUDA
test was skipped. Both launchers resolved successfully for 4 and 8 GPUs.

The full released 1.7B checkpoint and production/DDP training have not been
executed in this environment: Hub downloads time out and GPUs are occupied.
Tiny-model checks qualify the adapter and estimator, not the actual released
weight values or production memory/throughput.
