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

### Use a local TinyGSM model bundle

Add these overrides to your TinyGSM training command:

```bash
model_weights_dir="$SOURCE_CODE_PATH/code/cache_dir/tinygsm_mmd_offline_bundle" \
adversarial_distill.mmd_ar_model=checkpoints/tinygsm/ar_notok.ckpt
```

You can also use `/home/iasudakov/tinygsm_mmd_offline_bundle` directly.
Keep `data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M` and point
`training.finetune_path` to your local MDLM checkpoint. With `model_weights_dir`
set, model and tokenizer loaders use local files and report missing assets
without downloading them. Leaving it unset preserves the existing loading
behavior. Explicit local paths still work; native checkpoint paths can also be
relative to the bundle. Tokenizer IDs and dataset cache names stay unchanged.

The example bundle includes the AR checkpoint and the training and evaluation
tokenizers. It does not include the MDLM checkpoint or GPT-2 Large model weights,
so use `eval.compute_generative_perplexity=False` with this bundle. GSM8K
validation uses the bundle's `data/gsm8k_test.json` if the configured local file
is missing. Configure the TinyGSM training dataset cache separately.

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

To set different student and fake-model base learning rates, add
`optim.lr=1e-5 optim.fake_lr=1e-6` to an MDLM or Duo distillation command.
When omitted, `optim.fake_lr` follows `optim.lr`, including for older checkpoint
configs without this field. Both optimizers keep the configured learning-rate
schedule, each using its own base LR. This also applies to IDLM-ratio rewards;
feature-only MMD does not train the fake model.

#### MDLM teacher → IDLM-MDLM TinyGSM student

```bash
bash scripts/train_idlm_mdlm_tynigsm.sh
```

To use the same fixed clipped generator schedule as OWT, add these overrides
to a TinyGSM MDLM distillation command (DMD, MMD, or their combination):

```bash
adversarial_distill.mask_schedule.mode=clipped \
adversarial_distill.mask_schedule.min=0.45 \
adversarial_distill.mask_schedule.max=0.95
```

This remaps the original stratified draws into 45--95% expected masking
probability, without drawing new times. It applies to generator inputs in
both student and fake-critic training. With `data.train_on_prompt=False`,
prompts remain visible; padding follows `data.train_on_pad`. The subsequent
teacher/fake loss corruption, validation NLL and GSM8K evaluation retain their
existing schedules. Actual mask counts remain Bernoulli draws.

The default `mask_schedule.mode=original` preserves existing runs, including
the high-mask mixture below. For `clipped`, keep
`adversarial_distill.high_mask_fraction=0.0` (the default); replace the
`high_mask_fraction=1.0` workaround if you previously used it. Fixed clipping
requires distillation, so it cannot be combined with teacher continuation.

To bias distillation toward the heavily masked states that matter most at low
NFE, enable the optional high-mask mixture. This example keeps the original
uniform schedule for half of the rows and draws the other half from 70--100%
expected masking:

```bash
HIGH_MASK_FRACTION=0.5 HIGH_MASK_MIN=0.7 HIGH_MASK_MAX=1.0 \
  MODE=mmd bash scripts/train_tinygsm_mmd.sh
```

The default fraction is zero, so existing commands are unchanged. During MMD
training, `mmd/generator_mask_fraction` reports the realized fraction over real
answer tokens (padding and the prompt are excluded).

To use one reward sample per state and cut the reward critic cost, select the
fallback baseline explicitly:

```bash
MMD_GROUP=1 MMD_BASELINE=batch MODE=mmd bash scripts/train_tinygsm_mmd.sh
```

`MMD_BASELINE=auto` resolves to `none` for a group of one. `batch` instead uses
the other active sequences on the same device as a leave-one-out baseline; it
removes the reward offset but is noisier because TinyGSM rows have different
questions and masking fractions.

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
The coordinatewise distance is computed in tiles to limit memory use, but
requires more work than the matrix-multiplication squared-distance path.

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

#### Continue training the TinyGSM teacher

Use the same data, checkpoint, validation, and logging setup while optimizing
the ordinary MDLM denoising objective instead of DMD/MMD distillation:

```bash
MODE=teacher bash scripts/train_tinygsm_mmd.sh
```

For a custom Hydra command, add `training.continue_teacher=True`. It takes
precedence over `adversarial_distill.is_distill=True`, which makes it safe to
append to an existing shared `COMMON` string. The checkpoint at
`training.finetune_path` initializes the trainable backbone and EMA; no frozen
teacher or fake model is constructed. If the input is itself an MDLM
distillation checkpoint, its frozen `teacher.*` weights are selected rather
than its student `backbone.*` weights.

The mode also sets `trainer.accumulate_grad_batches` to the value implied by
`loader.global_batch_size`. This matters when reusing a distillation command:
that path accumulates internally and commonly pins Lightning's value to `1`,
whereas ordinary teacher training uses Lightning's automatic accumulation.

This is a new training run: optimizer, scheduler, and global step start from
zero. To resume an interrupted teacher-continuation run exactly, use
`checkpointing.resume_ckpt_path` as usual.

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

We release the TinyGSM IDLM checkpoints on Hugging Face: [IDLM-MDLM TinyGSM](https://huggingface.co/kekchpek/idlm-mdlm-tynigsm) and [IDLM-Duo TinyGSM](https://huggingface.co/kekchpek/idlm-duo-tynigsm).

The benchmark evaluation is available in-tree as `mode=gsm8k_eval`. It follows the
TinyGSM evaluation protocol of the [S-FLM repository](https://github.com/jdeschena/s-flm):
the question is pinned as a clean prefix, the model generates a `simple_math_problem()`
program over the remaining positions, and that program is executed in a restricted
sandbox (`sandbox_gsm8k.py`) and compared against the `#### N` gold answer of the
1319-example GSM8K test set in `data/gsm8k_test.json`.

```bash
CKPT_PATH=outputs/tiny_gsm/<date>/<time>/checkpoints/step=00050000.ckpt \
DEVICES=4 \
bash scripts/eval_tinygsm_idlm_mdlm.sh
```

The script sweeps 4, 8, 16, and 32 sampling steps and prints an accuracy table with
bootstrapped 95% confidence intervals. Per-step results (including every generated
program) are written to `eval_runs/tinygsm/idlm-mdlm/steps_<N>/results.json`.
`adversarial_distill.is_distill` is probed from the checkpoint itself
(`scripts/probe_is_distill.py`), because a checkpoint produced by a distillation run
also stores the frozen `teacher` and the `fake_model` critic while a released student
checkpoint ships `backbone` alone; override it with `IS_DISTILL` if needed.

Useful environment overrides: `STEPS_LIST`, `EVAL_BATCH_SIZE`, `OUTPUT_ROOT`,
`BOOTSTRAP`, `NUM_NODES`, `IS_DISTILL`, `STEP_CONVENTION`. To re-grade an existing
generation file without resampling, pass `gsm8k.input_file=<results.json>`.

`STEP_CONVENTION` selects where the noise-removal step lives, which is the one place
this repository's `generate_samples` and the S-FLM sampler differ:

| Value | Behaviour | NFE |
| --- | --- | --- |
| `sflm` (default here) | Denoising folded into the final loop iteration (`alpha_s = 1` at `timesteps[-2]`), matching `scripts/sample/tinygsm/mdlm.sh` in S-FLM | `steps` |
| `idlm` | A separate denoising call appended after the loop at `t = eps` | `steps + 1` |

Appendix E.2 of the paper states the conditional protocol follows Deschenaux &
Gulcehre, so `sflm` is the setting that reproduces Table 1. Everything else already
matches S-FLM: `use_float64`, `noise_removal=ancestral`, `p_nucleus=1.0`, the
`linspace(1, eps, steps + 1)` grid with `eps=1e-5`, prefix pinning at init and after
every step, batch size 32 and seed 1. The global default in `configs/config.yaml`
stays `idlm` so unconditional `mode=sample_eval` is unaffected.

#### The same benchmark during training

`eval.gsm8k_at_validation` runs these exact 1319 questions at every validation and
logs the result as `val/gsm8k_acc`, so a training curve and a standalone benchmark
number are directly comparable. It uses `eval.gsm8k_step_convention` (default `sflm`,
matching the script) rather than the global `sampling.step_convention`; set
`eval.gsm8k_steps` and `eval.gsm8k_num_examples` to trade fidelity for wall-clock.

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
