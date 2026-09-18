# [IDLM: Inverse-distilled Diffusion Language Models](https://david-cripto.github.io/idlm-project-page/)

<p align="center">
  <strong>By</strong><br>
  <a href="https://david-cripto.github.io/Bio-page/">David Li</a>*,
  <a href="https://scholar.google.com/citations?hl=en&amp;oi=ao&amp;user=UaRTbNoAAAAJ">Nikita Gushchin</a>*,
  <a href="https://scholar.google.com/citations?hl=en&amp;oi=ao&amp;user=h0oeoJkAAAAJ">Dmitry Abulkhanov</a>,
  <a href="https://scholar.google.com/citations?hl=en&amp;oi=ao&amp;user=_XE1LvQAAAAJ">Eric Moulines</a>,
  <a href="https://scholar.google.com/citations?hl=en&amp;oi=ao&amp;user=5kMqBQEAAAAJ">Ivan Oseledets</a>,
  <a href="https://scholar.google.com/citations?hl=en&amp;oi=ao&amp;user=BqDhGJQAAAAJ">Maxim Panov</a>,
  <a href="https://akorotin.netlify.app/">Alexander Korotin</a>
</p>


<p align="center">
  <a href="https://david-cripto.github.io/idlm-project-page/">
    <img src="https://img.shields.io/badge/Bio-Page-blue.svg" alt="Bio Page">
  </a>

  <a href="https://david-cripto.github.io/Bio-page/blog/idlm/">
    <img src="https://img.shields.io/badge/Blog-Post-2ea44f.svg" alt="Blog Post">
  </a>
  
  <a href="https://arxiv.org/abs/2602.19066">
    <img src="https://img.shields.io/badge/arXiv-2602.19066-b31b1b.svg" alt="arXiv">
  </a>
  
  <a href="https://huggingface.co/kekchpek/models">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc4d.svg" alt="Hugging Face Models">
  </a>

  <a href="https://www.youtube.com/watch?v=RZ6_huata1Y&amp;feature=youtu.be">
    <img src="https://img.shields.io/badge/YouTube-Recording-red?logo=youtube&amp;logoColor=white" alt="YouTube Recording">
  </a>
  
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
</p>

<div align="center">
  <img src="assets/teaser.png" width="70%">
</div>


## What is IDLM?

Diffusion Language Models can generate high-quality text, but their iterative reverse-diffusion sampling makes inference slow. **IDLM** speeds them up by distilling a pretrained many-step diffusion language model into a **few-step generator**.

Instead of simply matching every teacher step, IDLM uses an [Inverse Distillation](https://arxiv.org/abs/2502.01362) view for discrete token spaces.The paper reports **4×–64× fewer inference steps** while preserving the teacher model’s generation quality.

---

## Repository layout

```text
IDLM/
├── configs/                 # Hydra configs: data, model, algo, strategy, callbacks, etc.
│   ├── algo/                # ar, mdlm, duo, duo_base, d3pm, sedd
│   ├── data/                # OpenWebText and TinyGSM/GSM8K configs
│   │   ├── tiny-gsm.yaml    # TinyGSM training/evaluation data config
│   │   └── gsm8k-test.yaml  # GSM8K/TinyGSM benchmark evaluation config
│   ├── model/               # tiny / small / medium model configs
│   ├── noise/               # diffusion noise schedules
│   └── config.yaml          # main experiment config
├── integral/                # precomputed tokenizer / integration assets, including SmolLM TinyGSM support
├── models/                  # DiT backbone, EMA utilities, attention tests
├── scripts/                 # training and generation recipes
├── algo.py                  # model families and IDLM distillation logic
├── dataloader.py            # tokenizers, OpenWebText/TinyGSM datasets, dataloaders
├── main.py                  # Hydra + Lightning entry point
├── metrics.py               # perplexity, entropy, BPD, NLL metrics
├── trainer_base.py          # shared training / sampling base classes
├── utils.py                 # logging and helper utilities
├── requirements.txt         # environment note / dependency list
└── LICENSE
```

---

## Getting Started
<a name="getting_started"></a>

### Use existing model weights without downloading

Add this override to your OWT training or evaluation command:

```bash
model_weights_dir="$SOURCE_CODE_PATH/code/cache_dir/owt_mmd_offline_bundle"
```

You can also use `/home/iasudakov/owt_mmd_offline_bundle`. Keep the usual model
IDs in the command: `eval.checkpoint_path=kuleshov-group/mdlm-owt`,
`adversarial_distill.mmd_ar_model=kuleshov-group/owt-ar`,
`data.tokenizer_name_or_path=gpt2`, and
`eval.gen_ppl_eval_model_name_or_path=gpt2-large`. They resolve to the bundled
MDLM, AR, training tokenizer, and evaluation assets. Perplexity, temperature
sweeps, and Gradient Moment share the local evaluation model.

With the option set, model/tokenizer loaders use local files only and report
missing assets instead of downloading them. Leaving it unset preserves normal
loading. The resolver supports the bundle's `manifest.json`, pinned `hf-cache`
snapshots, and folders named by model ID or basename. Explicit local paths also
work; native checkpoint paths can be relative to the bundle. Existing native
checkpoint paths relative to the launch directory take precedence.

The bundle option changes where model assets are read. Logical tokenizer IDs,
dataset cache keys, training, and sampling are preserved. Configure the dataset
cache separately with `data.cache_dir`.

### 1. Clone the repository

```bash
git clone https://github.com/David-cripto/IDLM.git
cd IDLM
```

### 2. Create an environment

To get started, create a conda environment containing the required dependencies.

```bash
conda create -n idlm python=3.12
conda activate idlm
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1
```

---
## Checkpoints
<a name="checkpoints"></a>

* **IDLM-MDLM**. Trained on OpenWebText:
  * [Huggingface](https://huggingface.co/kekchpek/idlm-mdlm)🤗.
* **IDLM-MDLM**. Trained on TinyGSM:
  * [Huggingface](https://huggingface.co/kekchpek/idlm-mdlm-tynigsm)🤗.
* **IDLM-Duo**. Trained on OpenWebText:
  * [Huggingface](https://huggingface.co/kekchpek/idlm-duo)🤗.
* **IDLM-Duo**. Trained on TinyGSM:
  * [Huggingface](https://huggingface.co/kekchpek/idlm-duo-tynigsm)🤗.
* **IDLM-DCD**. Trained on OpenWebText:
  * [Huggingface](https://huggingface.co/kekchpek/idlm-dcd)🤗.

---

## Train IDLM

This section provides reference training entry points for unconditional OpenWebText distillation and conditional TinyGSM distillation. The scripts are written as Hydra override recipes, update the dataset cache and checkpoint paths before launching a run.

### Unconditional OpenWebText Distillation

The OpenWebText recipes train IDLM students for unconditional language generation. Before executing the scripts, configure the `cache_dir` parameter in `configs/data/openwebtext-split.yaml` to specify the desired output path.

#### MDLM teacher → IDLM-MDLM student

```bash
bash scripts/train_idlm_mdlm.sh
```

For the IDLM ratio reward, add these overrides to an OWT reward-distillation
command:

```bash
adversarial_distill.mmd_reward_type=idlm \
adversarial_distill.idlm_reward_batch_size=8
```

`idlm_reward_batch_size` limits generated sequences per teacher/fake scoring
forward; it defaults to 8. It leaves `loader.batch_size`, `mmd_group_size`,
gradient accumulation, shared corruption and the group baseline unchanged.
Hard-token rewards use token lookups instead of dense one-hot targets. Lower
this positive integer if reward scoring needs less memory; this adds critic
forwards. It does not control fake-critic training or feature-reward batches.

For OWT MMD distillation, the portable launcher can bias the generator toward
the heavily masked states that matter most at low NFE. This keeps the original
schedule for half of the rows and draws the other half from 70--100% expected
masking:

```bash
OWT_CACHE=/path/to/owt \
HIGH_MASK_FRACTION=0.5 HIGH_MASK_MIN=0.7 HIGH_MASK_MAX=1.0 \
  bash scripts/train_owt_distill.sh
```

The default fraction is zero, so existing runs are unchanged. MMD runs log the
realized whole-sequence fraction as `mmd/generator_mask_fraction`.

An alternative is the [BD3-LMs clipped masking schedule](https://arxiv.org/abs/2503.09573).
`MASK_SCHEDULE=clipped` samples every generator row's masking probability
uniformly inside a fixed interval, preserving stratification across devices
and accumulation steps:

```bash
OWT_CACHE=/path/to/owt MASK_SCHEDULE=clipped MASK_MIN=0.45 MASK_MAX=0.95 \
  bash scripts/train_owt_distill.sh
```

Use `MASK_SCHEDULE=adaptive` to select the interval during validation:

```bash
OWT_CACHE=/path/to/owt MASK_SCHEDULE=adaptive MASK_SEARCH_BATCHES=4 \
  bash scripts/train_owt_distill.sh
```

These options also apply to existing MMD/DMD combination commands. Direct
Hydra equivalents are `adversarial_distill.mask_schedule.mode`, `.min`, `.max`,
and `.search_batches`. Override `.search_intervals` with a list of `[min,max]`
pairs to change the candidates. Adaptive mode starts from `.min`/`.max`, then
chooses the lowest pooled **per-sequence student denoising-loss variance**
over the first few validation batches, with one extra forward per candidate
per batch. This is a BD3-LMs-inspired proxy, not measured MMD gradient variance;
it does not assume that larger masking rates are always better. Sanity checks,
step-zero validation, and standalone evaluation do not update the schedule.
The selected bounds and proxy are logged under `mask_schedule/*` and restored
from checkpoints when the schedule settings match.

The default `original` mode preserves the existing sampler and high-mask
mixture. For `clipped`/`adaptive`, use `HIGH_MASK_FRACTION=0` and disable
`adversarial_distill.generator_force_full_mask`. Only the corruption used to
produce student samples changes, including samples used to train the fake
critic; teacher/fake loss corruption, evaluation NLL, and generation remain
unchanged. Bounds refer to **expected** masking probabilities: mask patterns
are still Bernoulli draws, without blockwise rejection resampling. The
log-linear schedule caps a requested maximum of 1 at 0.999 before drawing.

For sentence-level MMD with the AR critic, add these overrides to an existing
OWT MMD training command to use the **last token's hidden state** instead of
the mean over scored (usually masked) positions:

```bash
adversarial_distill.mmd_reward_type=feature \
adversarial_distill.mmd_feature_source=ar \
adversarial_distill.mmd_kernel_level=sentence \
adversarial_distill.mmd_sentence_pooling=last_token
```

On packed OWT this reads position `model.length - 1`, even if it was not
masked; it does not read the last masked token or stop at an internal EOS.
The critic still processes the full sequence, so a contextual AR layer's final
hidden state can represent changes earlier in the sequence. Layer `-1` is only
the input embedding and cannot summarize the prefix; the default middle block
is contextual.

Keep your existing kernel/source/draw-count settings. This option works with
both `rows` and `draws` under their existing kernel restrictions (for example,
`pseudo_huber` still requires `draws`). Estimators and loss normalization are
unchanged: sentence `draws` still uses the unbiased fake-pair correction for
K > 1 and the biased fallback for K = 1. `mmd_score_positions` does not move
the last-token readout, and only masked/written positions carry the REINFORCE
gradient. Omit the new option or set `mmd_sentence_pooling=mean` to retain the
previous behavior. Diffusion features and token-level kernels do not support
`last_token`.

To reduce model-forward memory while preserving a batch-32 feature-reward
objective, keep the logical loader batch and add a separate forward limit:

```bash
loader.batch_size=32 loader.global_batch_size=512 \
adversarial_distill.mmd_forward_batch_size=8
```

This runs the student and frozen feature model on at most 8 sequences per
forward. Student activations are checkpointed and recomputed during backward;
only sampled token IDs and their differentiable log probabilities are retained
between chunks. Captured frozen features are concatenated before reward
calculation. Thus masked-token normalization, automatic RBF bandwidth, K-row
groups, G-candidate baselines, and optional batch baselines still use the
original 32 examples. Groups may cross a forward-chunk boundary. Feature
tensors and kernel workspaces still use the logical batch; this option does
not promise that all memory scales with the forward limit.

Keep `loader.batch_size`, `loader.global_batch_size`, device count, K, G and
other training settings fixed when comparing forward limits. With 8 GPUs and
global batch 512, logical batch 32 still requires 2 accumulation steps in the
existing manual training loop. Validation and scheduler timing also stay the
same. Changing `loader.batch_size` to 8 instead still changes the objective.

The default `null` preserves the existing path. A positive integer enables
this option for MDLM with `dmd_weight=0`, nonzero `mmd_reward_weight`, and
`mmd_reward_type=feature` or `feature_mse`; DMD and IDLM-ratio runs reject it.
The loss and gradients agree for the same examples, corruptions and sampled
tokens, up to floating-point rounding. RNG ordering, dropout draws and GPU
kernels can differ, so identical seeds do not guarantee identical trajectories.
The memory saving trades extra student-forward computation during backward.

To replace MMD with **position-aligned masked-token feature MSE**, add:

```bash
adversarial_distill.mmd_reward_type=feature_mse \
adversarial_distill.mmd_score_positions=written
```

For each generated sequence and selected layer, the reward is
`-mean_{t in masked content, d}((h_fake[t,d] - h_real[t,d])**2)`.
Squared differences are computed before averaging tokens; swapping token
features cannot cancel the error just because their pooled means match.
The existing feature source, layer selection/weights and diffusion feature
noise remain in use. Set `mmd_noise_level=0.0` for clean diffusion features.
The feature model stays frozen, and the student learns through the existing
sample-group REINFORCE baseline, weighted by `mmd_reward_weight`.

This mode requires `mmd_score_positions=written`. MMD kernel, bandwidth,
same-sentence exclusion and sentence-pooling settings are ignored. With
`mmd_kernel_source=rows`, each sequence has its own aligned reference and
`mmd_kernel_batch_size` is ignored. With `mmd_kernel_source=draws`, each
candidate retains K draws and its reward averages their individual MSEs;
the surrogate omits MMD's extra division by K, keeping the expected MSE
gradient scale independent of K. Empty masks contribute zero; a single
masked token is valid. Distances accumulate in FP32. Existing `feature` and
`idlm` rewards are unchanged. MSE has a different reward scale from MMD,
so its learning rate or reward weight may need separate tuning.

To use pseudo-Huber distance inside an RBF kernel, add these overrides to an
existing feature-MMD command:

```bash
adversarial_distill.mmd_reward_type=feature \
adversarial_distill.mmd_kernel=rbf \
adversarial_distill.mmd_rbf_distance=pseudo_huber \
adversarial_distill.mmd_rbf_huber_delta=0.001 \
adversarial_distill.mmd_rbf_alpha=null
```

The kernel is `k(x,y) = exp(-alpha * D(x,y))`, where
`D(x,y) = sum_d(sqrt((x_d - y_d)^2 + delta^2) - delta)`.
This sums feature coordinates, with no extra division by feature dimension or
factor of delta. Positive `delta` (default `0.001`) controls the transition
from quadratic behavior for small differences to linear behavior for large
ones, reducing the influence of large coordinate differences.

Leave `mmd_rbf_alpha=null` to use `alpha = 1 / (2 * median(D))` estimated from
the selected real features. Recalibrate an explicit alpha when changing the
distance. The default distance remains `squared_euclidean`.

This works with AR or diffusion features and the existing valid combinations
of `rows`/`draws` and token/sentence kernels. It requires an active feature
reward (`mmd_reward_weight > 0`); DMD-only and IDLM-ratio rewards do not use
these options. `mmd_kernel=pseudo_huber` remains the separate additive kernel.
On CUDA with Triton available, this option automatically uses the fused
distance and gradient reductions in `mmd_laplace_cuda.py`, adapted from the
provided L1 implementation to the pseudo-Huber formula above. Coordinate
intermediates stay inside the kernel instead of allocating pair-by-feature
tensors in GPU memory. The same path accelerates automatic bandwidth
estimation; no additional configuration is needed.

CPU, FP64, and installations without Triton retain the tiled PyTorch fallback.
Triton is imported lazily and compiles the CUDA kernels on their first use.
Keep `mmd_rbf_block_size=0` to choose tiles for the active backend. Existing
distance scaling, bandwidth selection, masks, and MMD estimators are preserved,
up to floating-point reduction differences.

To omit only the real-real term from an RBF feature reward, add:

```bash
adversarial_distill.mmd_kernel=rbf \
adversarial_distill.mmd_rbf_omit_real_real=True
```

The reward becomes `2 * RF - FF`. This skips the reference-reference kernel
sums and leaves fake-fake pairs, real-fake pairs, their normalization, and
existing activity masks unchanged. The default is `False`. The option works
with token/sentence features, rows/draws, and squared-Euclidean or pseudo-Huber
RBF distance. Token-level draws support K=1 and still require at least two
scored tokens, as before. Sentence-level draws drop the fixed real self-term
of 1, retaining the existing fake self-term fallback when K=1.

Keep `mmd_rbf_exclude_same_sentence=False` to change only the real-real term.
When that separate option is `True`, it already omits real-real and excludes
within-sentence fake-fake pairs; enabling both flags produces the same result.
Real features are still needed for RF and for automatic bandwidth estimation,
which still computes real-feature distances. The remaining signed statistic
is not the full MMD squared.

This requires `mmd_reward_type=feature` and `mmd_kernel=rbf`; the kernel-free
`feature_mse` reward ignores it. Student sampling, K/G scaling, and baselines
are unchanged. A reference term shared by every candidate cancels under the
group baseline, but per-candidate feature noise or automatic bandwidth
subsampling can make those terms differ; equal training updates are not
guaranteed merely by removing the term.

For active feature rewards (`mmd_reward_type=feature`), token RBF can exclude
pairs of tokens from the same generated sentence:

```bash
adversarial_distill.mmd_kernel=rbf \
adversarial_distill.mmd_kernel_source=draws \
adversarial_distill.mmd_kernel_level=token \
adversarial_distill.mmd_kernel_batch_size=4 \
adversarial_distill.mmd_rbf_exclude_same_sentence=True
```

The flag defaults to `False` and requires the combination above with at least
two draws (`K >= 2`). With `K=4`, it removes the four diagonal sentence blocks
and keeps the twelve ordered cross-sentence blocks. Each retained block still
compares **all token-position pairs**, including equal position indices;
sentence features are not mean-pooled first.

For `n` selected positions shared by all K draws, the statistic is
`FF_between - 2 * RF`. The generated-generated sum uses the denominator
`K * (K - 1) * n^2`; the reference-generated sum includes all pairs and uses
`K * n^2`. The reward is the negative of this statistic. One selected position
is sufficient; rows with none receive zero reward.

The reference-reference term is omitted, so the raw statistic is not the
full MMD squared. Negative values are retained. This option works with both RBF distances,
`squared_euclidean` and `pseudo_huber`.

To use cosine similarity as the MMD kernel, change only the kernel in your
feature-reward command (keep its existing source, level, K, G and pooling):

```bash
adversarial_distill.mmd_reward_type=feature \
adversarial_distill.mmd_kernel=cosine
```

This computes MMD with `k(x,y) = dot(normalize(x), normalize(y))`, using L2
normalization in float32. At token level each token vector is normalized
individually; at sentence level the existing mean/last-token readout happens
first, and that sentence vector is normalized. It is **not** cosine similarity
between two final pooled means and does not apply pseudo-Huber. It supports
both AR and diffusion features (the `last_token` readout remains AR-only).

Self-pair removal and activity rules follow RBF: token comparisons need at
least two scored vectors on each side; sentence `rows` needs K >= 2. Sentence
`draws` treats the observed real vector as fixed, removes fake self-pairs at
K > 1, and uses a biased squared distance between normalized vectors at K = 1
(`2 * (1 - cosine)` for nonzero unit vectors). Zero vectors are handled with
epsilon-safe normalization and their actual diagonal kernel values. Negative
finite-sample MMD estimates are retained. No bandwidth is needed;
`mmd_rbf_alpha` and `mmd_rbf_block_size` do not affect this kernel.

Cosine is a linear kernel on unit features, so its population MMD only matches
their **mean**, unlike RBF's nonlinear comparison. Removing feature magnitude
therefore also changes the objective's sensitivity; an RBF kernel applied to
unit-normalized features would be a different experiment (not enabled by this
option). Existing `linear`, `rbf`, and `pseudo_huber` behavior and the REINFORCE
`G*K`/token loss normalization are unchanged. Cosine uses a dot-product sum,
not an extra `1/hidden_dim` average; reward scales need not match the prior run.
Test it as a fresh run rather than resuming and mixing kernel histories.

#### DUO teacher → IDLM-Duo student

```bash
bash scripts/train_idlm_duo.sh
```

#### DCD teacher → IDLM-DCD student

```bash
bash scripts/train_idlm_dcd.sh
```

### Conditional TinyGSM Distillation

The TinyGSM recipes train IDLM students for conditional mathematical reasoning on TinyGSM-style question-answer examples. Before executing the scripts, configure `cache_dir` in `configs/data/tiny-gsm.yaml` and replace the `training.finetune_path` placeholder in each script with the corresponding pretrained teacher checkpoint. Our TinyGSM distillation runs use the teacher checkpoints from the [S-FLM repository](https://github.com/jdeschena/s-flm): [MDLM](https://huggingface.co/jdeschena/s-flm/blob/main/tinygsm/mdlm.ckpt) for IDLM-MDLM and [Duo](https://huggingface.co/jdeschena/s-flm/blob/main/tinygsm/duo.ckpt) for IDLM-Duo.

#### MDLM teacher → IDLM-MDLM TinyGSM student

```bash
bash scripts/train_idlm_mdlm_tynigsm.sh
```

#### DUO teacher → IDLM-Duo TinyGSM student

```bash
bash scripts/train_idlm_duo_tynigsm.sh
```

---

## Evaluation

This section separates unconditional OpenWebText generation from conditional TinyGSM benchmark evaluation.

### Unconditional OpenWebText Evaluation

The generation scripts sweep over 4, 8, 16, and 32 sampling steps.

Before running them, set `eval.generated_samples_path` to a real JSON output path.

#### IDLM-MDLM checkpoint

```bash
mkdir -p samples

python -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=mdlm \
  algo.backbone=hf_dit \
  eval.checkpoint_path=kekchpek/idlm-mdlm \
  sampling.steps=16 \
  sampling.num_sample_batches=10 \
  sampling.predictor=ancestral_cache \
  sampling.noise_removal=ancestral \
  +wandb.offline=true \
  eval.generated_samples_path=samples/idlm_mdlm_16steps.json
```

#### IDLM-Duo checkpoint

```bash
mkdir -p samples

python -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo \
  algo.backbone=hf_dit \
  eval.checkpoint_path=kekchpek/idlm-duo \
  sampling.steps=16 \
  sampling.num_sample_batches=10 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  eval.generated_samples_path=samples/idlm_duo_16steps.json
```

#### IDLM-DCD checkpoint

```bash
mkdir -p samples

python -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo \
  algo.backbone=hf_dit \
  eval.checkpoint_path=kekchpek/idlm-dcd \
  sampling.steps=4 \
  sampling.num_sample_batches=10 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  eval.generated_samples_path=samples/idlm_duo_4steps.json
```

#### Run the provided scripts

```bash
bash scripts/generation_idlm_mdlm.sh
bash scripts/generation_idlm_duo.sh
bash scripts/generation_idlm_dcd.sh
```

Generated sample files contain:

```json
{
  "generative_ppl": 0.0,
  "entropy": 0.0,
  "generated_seqs": []
}
```

### Conditional TinyGSM Evaluation

We release the TinyGSM IDLM checkpoints on Hugging Face: [IDLM-MDLM TinyGSM](https://huggingface.co/kekchpek/idlm-mdlm-tynigsm) and [IDLM-Duo TinyGSM](https://huggingface.co/kekchpek/idlm-duo-tynigsm). To evaluate these conditional models on the TinyGSM benchmark, use the `.ckpt` files from the Hugging Face repositories together with the TinyGSM evaluation code from the [S-FLM repository](https://github.com/jdeschena/s-flm).

---

## Outputs

By default, Hydra writes experiment outputs under:

```text
outputs/<dataset>/<date>/<time>/
```

TensorBoard logs are written under:

```text
tb_logs/
```

Checkpoints are written according to the checkpointing config in `configs/config.yaml`.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{li2026idlm,
  title={IDLM: Inverse-distilled Diffusion Language Models},
  author={Li, David and Gushchin, Nikita and Abulkhanov, Dmitry and Moulines, Eric and Oseledets, Ivan and Panov, Maxim and Korotin, Alexander},
  journal={arXiv preprint arXiv:2602.19066},
  year={2026}
}
```

---

## Acknowledgements

Our codebase is inspired by recent Discrete Diffusion Models projects. Namely, [MDLM](https://github.com/kuleshov-group/mdlm), [Duo](https://github.com/s-sahoo/duo), and [S-FLM](https://github.com/jdeschena/s-flm).

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
