# MDLM-L features for the archived OWT MMD training code

For the newer native-token SDLLM feature option, see
[SDLLM MDLM-1.7B features](sdllm_features.md).

This worktree starts from the exact frozen source of Underdeep run
[3daa3e6f-1c54-4854-8bda-0b33dfd3257f](https://deep.yandex-team.ru/projects/generative-models/experiments/owt/run/3daa3e6f-1c54-4854-8bda-0b33dfd3257f).
Its checkpoint label3218 matches all36 points of Figure2/Table2. The paper's
nominal3250 and the later public reproduction are distinct. The restored source
is commit634176b; the122 archived files, hashes and launch overrides are in
`provenance/owt_3daa`. The original `IDLM` checkout and `dlm-mmd` worktree are
untouched. Existing IDLM shell scripts/README in this branch are inherited
documentation; use the MMD launcher below for this experiment.

## Enable the independent feature model

The new switch is `adversarial_distill.mmd_feature_source=sdtt_mdlm`. Existing
`ar` and `diffusion` routes and all their defaults remain unchanged. The
student and diffusion teacher remain `kuleshov-group/mdlm-owt`; only the
frozen MMD feature map is replaced. Sampling, group8, draws2, token-RBF,
alpha2e-5, exclusions, loss, baseline and accumulation are unchanged.

The declared source is [SDTT's undistilled Large teacher](https://huggingface.co/jdeschena/sdtt):
`jdeschena/sdtt`, revision `teacher_400k_large_step_400000`, files
`config.json` and `model.safetensors`. It has863,158,098 parameters,
24 blocks, hidden1536,16 heads, GPT-2 vocabulary, context1024 and400k OWT
pretraining updates. No SDTT package, training checkpoint paths from its
config, or remote Python model code is imported.

```bash
bash scripts/train_mmd_owt_sdtt_large.sh
```

At training startup, before the first training batch, the model automatically
downloads missing `config.json` and `model.safetensors` from the declared
teacher revision. No manual download or `SDTT_MODEL_DIR` is required. Files
are kept in the standard Hugging Face cache (`HF_HOME`/`HF_HUB_CACHE` can
change its location); repeat launches reuse cached weights. DDP processes
share the Hub's cache file locks. Download failures stop the run before
training updates. Existing `ar`/`diffusion` and disabled-reward paths do not
prepare or download SDTT. Config-only runs (`--cfg job`) do not download it.

`SDTT_MODEL_DIR` remains an explicit local-only override and fails if either
file is missing. Optional manual preparation is also available through
`python scripts/download_sdtt_teacher.py --output-dir /path/to/sdtt-large`;
that helper records file SHA256 hashes in `sdtt_provenance.json`.
Pin `mmd_sdtt_revision` to an immutable Hub commit for a repeatable online
run. The loader pins weights to the config's resolved commit when available
and records that commit and the config hash.

The supplied launcher recreates the frozen recipe. Defaults are8 GPUs,
local batch8, accumulation8, global batch512, eval5seeds/128examples each.
`TRAIN_DEVICES=4` changes accumulation to16 and doubles raw validation and
logging intervals; this preserves counts, not bitwise identity across GPU
partitions. `MODEL_WEIGHTS_DIR=/path/to/owt_mmd_offline_bundle` optionally
resolves the original student, tokenizer and PPL evaluator offline.

```bash
TRAIN_DEVICES=4 FEATURE_LAYER=11 \
  bash scripts/train_mmd_owt_sdtt_large.sh
```

`FEATURE_LAYER` is zero-based in the Large model:0–23,−1 for embeddings;
there is no automatic depth remapping or feature normalization. The launcher
keeps the original absolute layer3 by default. Existing layer lists and
weights also work via final Hydra overrides. Do not assume the original
alpha/layer is optimal after changing the feature space.

Set `FEATURE_SOURCE=diffusion` to run the original feature route using the
same launcher. Extra command-line Hydra overrides are passed last. Run a
config-only check with `bash scripts/train_mmd_owt_sdtt_large.sh --cfg job`;
this does not start training or initialize the logger.

## Loading and feature semantics

The loader validates the complete GPT-2 base vocabulary mapping, BOS/EOS,
mask50257 and the original added `[PAD]` alias. Other tokenizers, including
TinyGSM's tokenizer, fail explicitly. Inputs are not retokenized.

Weights are loaded with a complete strict key/shape check into the archived
DiT, accepting raw keys or uniform `backbone.`/compiled prefixes. Only an
FP32 export is accepted; no partial load, implicit casting, EMA reconstruction
or guessed fallback weights are used. The model is frozen and stays in
eval mode. The teacher is time-agnostic, so it receives zero timestep
conditioning while the existing upstream physical feature-mask RNG remains
unchanged. Initialization preserves Torch RNG. Feature forwards stop after
the deepest requested post-block feature and omit the unused vocabulary head.

The archived DiT uses SDPA and separate rotary Q/K operations. SDTT's own
implementation uses packed FlashAttention/rotary operations. Parameters are
structurally compatible; bitwise equivalence to the upstream kernel backend
is not asserted. Enabling Large features intentionally changes the reward.

## Validation and remaining limitation

```bash
python scripts/test_sdtt_features.py
python scripts/test_sdtt_feature_routing.py
```

Tests use actual archived tiny-DiT fixtures with randomized nonzero weights,
local safetensors, exact feature comparisons, frozen/eval behavior, tokenizer
and malformed-checkpoint rejection, and original-route seed regression.
The latter compares samples, rewards, loss, gradients and RNG with634176b.
26 CPU loader tests and7 routing/regression tests passed. An opt-in tiny-DiT
CUDA/BF16 smoke check also passed on H100: features for embeddings/blocks0/1
matched the archived zero-timestep forward exactly, with CPU/CUDA RNG preserved.
The complete launcher parsed successfully in config-only mode for8 and4GPUs.

```bash
MMD_SDTT_TEST_CUDA=1 python scripts/test_sdtt_features.py \
  SDTTFeatureLoaderTests.test_cuda_bf16_features_match_and_preserve_rng
```

During implementation, the real SDTT Hub revision could not be fetched from
this environment (connection timeouts). Its actual config/header/file hash,
exported weight values, full863M forward and production/DDP training remain
unverified. The adapter fails on incompatibility; it does not substitute
random weights. Validate the downloaded release before launching the full run.
