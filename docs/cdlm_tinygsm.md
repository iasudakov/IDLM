# CDLM objectives for IDLM MDLM on TinyGSM

This branch implements the three CDLM training objectives on the project's existing MDLM backbone and TinyGSM setup. The method comes from [Kim et al., CDLM: Consistency Diffusion Language Models for Faster Sampling](https://arxiv.org/abs/2511.19269) and the [official implementation at commit `708800ab`](https://github.com/SqueezeAILab/CDLM/tree/708800ab0df03e2c284ab2cb358e42652a5e85a5). Upstream code is MIT licensed; its notice is preserved in [third_party/CDLM_LICENSE](../third_party/CDLM_LICENSE).

The default keeps full bidirectional MDLM attention and does not use KV caching. It adapts CDLM's objectives and trajectory supervision to the same backbone used by MMD. The released CDLM also trains block-causal attention for cached decoding; that architectural change and its latency results are outside this default comparison. Label results as **CDLM objectives on MDLM, full attention**. A trajectory window groups reveal steps for training; it does not change the attention mask.

The comparison reference is MMD run `3e0ac6b1-7a8f-43c3-af58-d9e21b886da7`. [The CDLM provenance record](../provenance/cdlm_tinygsm_reference.json) identifies the exact source files and adaptations; [the inherited MMD record](../provenance/sdtt_tinygsm_reference.json) contains the original launch and tracker evidence. This branch starts from SDTT branch commit `708e77c1691cdb4e147f9bffa05747133676a235`.

## Objective

The frozen original MDLM teacher produces trajectories offline. It is never promoted from the student. From each saved trajectory, training selects an earlier partially masked state and a later window endpoint, then optimizes:

`loss = distillation + 0.5 × self_consistency + 0.1 × ground_truth_denoising`

- **Distillation:** teacher-to-student forward cross entropy on tokens revealed between the two states. The teacher distribution is reconstructed from the frozen vocabulary projection and the saved head inputs at each token's reveal time. Omitting teacher entropy leaves the forward-KL gradient unchanged.
- **Self consistency:** the current student's detached later-state distribution supervises its earlier-state prediction at positions that remain masked in both states. It is a separate student consistency target, not another copy of the original MDLM teacher.
- **Ground truth:** randomly mask the true answer/EOS/padding span and apply inverse-mask-probability-weighted token cross entropy. Draw `p = 0.001 + 0.999 × Uniform(0,1)` and scale masked token losses by `1 / (p + 1e-8)`.

These terms follow [the released loss implementation](https://github.com/SqueezeAILab/CDLM/blob/708800ab0df03e2c284ab2cb358e42652a5e85a5/CDLM-train/utils/loss.py). All probability normalization and losses use float32. The weights follow [the released LLaDA configuration](https://github.com/SqueezeAILab/CDLM/blob/708800ab0df03e2c284ab2cb358e42652a5e85a5/CDLM-train/config/llada.yaml).

Thirty-two-reveal windows preserve a nonempty still-masked span while later windows remain unfinished. Always pairing a state with the fully clean sequence would remove the positions needed for self consistency. The final window may naturally have no still-masked positions; that term then contributes zero for that sample.

## Prepare trajectories, then train

Use the existing project environment and supply the same TinyGSM MDLM checkpoint as the MMD run. Preparation and optimization are separate stages so the fixed teacher's generation cost is visible and its data can be reused.

The target is four B300 GPUs. Preserve the working Blackwell-compatible PyTorch/CUDA/FlashAttention environment; the complete run-command does not reinstall the older Torch/CUDA versions listed in the inherited repository requirements. GPU execution on B300 has not been tested in this code-only task.

```bash
bash scripts/prepare_cdlm_tinygsm.sh \
  training.finetune_path=/absolute/path/to/checkpoints/tinygsm/mdlm_notok.ckpt

bash scripts/train_cdlm_tinygsm.sh \
  training.finetune_path=/absolute/path/to/checkpoints/tinygsm/mdlm_notok.ckpt \
  hydra.run.dir=/absolute/path/to/runs/cdlm_tinygsm
```

If the teacher path is omitted, the preset uses the public `hf://jdeschena/s-flm/tinygsm/mdlm.ckpt` release. Use the original run's local checkpoint for checkpoint-level matching. No separately pretrained CDLM weights are required: preparation loads the existing MDLM, and optimization needs its frozen output projection plus the generated cache. With `model_weights_dir` configured, supply a local checkpoint path; the offline bundle does not trigger a Hub fallback.

The MMD initialization uses the published
[yresearch/MDLM-MMD-TinyGSM checkpoint](https://huggingface.co/yresearch/MDLM-MMD-TinyGSM),
`model.ckpt`, linked from the MMD project's README. Select the
`cdlm_tinygsm_mmd` preset to load it automatically from Hugging Face or the
standard HF cache. Keep `training.finetune_path` pointing to the original
MDLM teacher. For the existing Nirvana layout:

```bash
CDLM_PRESET=cdlm_tinygsm_mmd \
bash "$SOURCE_CODE_PATH/code/scripts/train_cdlm_tinygsm_nirvana.sh" \
  'logger.name=mmd to cdlm tinygsm 4xB300 16 epochs'
```

This preset sets
`training.student_init_path=hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt` and
`training.student_init_use_ema=True`. The verified cached release contains
EMA-only weights at MMD step 7000, so EMA selection is required. An explicit
student `hf://` source is allowed alongside `model_weights_dir`; local
teacher/tokenizer loading keeps the bundle policy. Preparation and full resume
do not fetch the external student checkpoint; fresh student initialization
performs the Hub lookup.

To prefetch the checkpoint before spending time collecting trajectories, run
this in the same environment on a machine with network access:

```bash
python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id="yresearch/MDLM-MMD-TinyGSM", filename="model.ckpt"))'
```

An offline Nirvana worker needs this asset staged in its HF cache. Prepare a
portable cache on a connected machine:

```bash
python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="yresearch/MDLM-MMD-TinyGSM", filename="model.ckpt", cache_dir="cdlm_mmd_hf_cache")'
```

Pass that entire `cdlm_mmd_hf_cache` directory as a Nirvana input, then add this
after creating `HF_HUB_CACHE` in the run-command:

```bash
cp -a "$INPUT_PATH/cdlm_mmd_hf_cache/." "$HF_HUB_CACHE/"
export HF_HUB_OFFLINE=1
```

The same preset then reads the staged cache without network requests. Global
Hugging Face offline settings are respected; selecting a Hub URI does not
provide network access on a disconnected worker.

For a different local MMD checkpoint, override `training.student_init_path`.
`student_init_use_ema=False` (the base CDLM default) imports live `backbone.*` weights;
`True` imports the checkpoint's EMA weights. An EMA-only export requires the
explicit `True` setting. Architecture and vocabulary must match the current
MDLM; incompatible checkpoints fail strict loading. Native MMD checkpoints
may include teacher, fake-model and optimizer tensors: only the student
backbone is imported.

The original MDLM output head and cached teacher trajectories remain fixed.
Student initialization does not change the cache fingerprint, so a matching
prepared cache is reused. CDLM starts a fresh optimizer/scheduler and fresh
student EMA from the imported weights, with its normal 16-epoch budget.
Label this run **MMD to CDLM** to record its starting point. Leaving
`student_init_path=null` keeps direct MDLM initialization. Full checkpoint
resume takes precedence and does not reopen the MMD initialization file.

The preparation default selects 20,000 TinyGSM training prompts and produces one trajectory at each temperature, zero and 0.5: 40,000 trajectories in total. Each teacher call reveals the globally most confident eligible token; unlike the released collector, it does not restrict selection to contiguous token blocks. Prompts stay visible, and answer/EOS/padding positions are generated. Only training-split examples are used for collection. Preparation is substantial work and must be counted separately when reporting training cost.

The cache stores the monotone reveal order, generated tokens, ground truth, and bf16 inputs to the frozen linear vocabulary head. It reconstructs intermediate states and teacher logits when training, instead of storing a full vocabulary tensor or a complete token/mask sequence at every step. At 40,000 trajectories, length 512, and hidden width 768, head inputs alone occupy about 29.3 GiB by shape; prompt rows are retained as zeros. Allow additional storage for token records, metadata, and filesystem overhead. This is an estimate, not a measured cache size.

`algo.trajectory_cache_dir` selects the cache. Its default is `cache_dir/cdlm_tinygsm_trajectories` under the checkout. Preparation runs on four processes by default; `CDLM_PREPARE_GPUS` selects the process count. `algo.trajectory_batch_size=8` sets the collection microbatch and `algo.trajectory_shard_size=8` sets eight trajectories per file. The full default creates 5,000 tensor shards plus metadata.

The cache format is schema 2. The manifest fingerprints the original checkpoint's bytes, tokenizer implementation/vocabulary, model config, source training data, selected indices, seed, temperatures, and rollout settings, including hidden-state dtype. `algo.teacher_hidden_dtype` supports only `bfloat16` or its `bf16` alias. Repeating preparation resumes completed matching shards after verifying their metadata/checksums; new shards are published atomically. Training requires the manifest to be complete and match the inputs. Earlier schemas are rejected and need a new cache. Changing teacher, tokenizer, separator, prompt selection, sequence length, or fingerprinted collection settings requires a different cache. Relocating identical files does not change content identity.

The default training budget is 16 epochs over the saved trajectories, matching the released author's epoch count. The loader uses 39,936 of the 40,000 trajectories per epoch so all updates contain the full global batch of 512; the remaining 64 are omitted. This gives 78 optimizer updates per epoch, or 1,248 updates across 16 epochs. The default follows CDLM's 5% warmup, calculated at runtime from the estimated optimizer budget: 62 updates for this dataset. The configured `lr_scheduler.num_warmup_steps=0` is a placeholder replaced by that calculation.

The optional `cdlm_tinygsm_matched` preset instead runs 20,000 optimizer updates:

```bash
CDLM_PRESET=cdlm_tinygsm_matched bash scripts/train_cdlm_tinygsm.sh \
  training.finetune_path=/absolute/path/to/checkpoints/tinygsm/mdlm_notok.ckpt
```

The matched-update preset retains MMD's fixed 2,500-update warmup with `algo.warmup_ratio=null`. These budgets are distinct; an epoch count should not be presented as the same computation as a fixed update budget. If a reduced source dataset is supplied, counts and updates change accordingly.

## Matched setup and deliberate differences

| Setting | This adaptation |
| --- | --- |
| Backbone and tokenizer | IDLM small MDLM, 12 layers, width 768; `HuggingFaceTB/SmolLM-135M` |
| Data | `TinyGSM/TinyGSM`, filtered to length 512; no wrapping |
| Split | Existing 1% holdout, split seed 42 |
| Input | BOS + question + literal backslash-n separator + code + EOS + padding |
| Corruption and loss | Answer, EOS, padding; exclude prompt |
| Attention | Full bidirectional; no KV cache |
| Default hardware | Four devices, 32 examples per device per microbatch |
| Accumulation | Four microbatches; effective global batch 512 |
| Optimizer | AdamW LR `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0 |
| Schedule | Default: constant after 5% warmup, 62 updates for the default cache; matched 20k: 2,500 warmup updates |
| EMA | `0.9999`, used for evaluation |
| Precision | bf16 mixed model forwards; float32 losses |
| Validation | Before training, then every 500 optimizer updates at default batch |

The reference MMD run used eight GPUs with manual accumulation two; four GPUs with accumulation four preserve its global batch. Changing devices or microbatch size requires preserving that product and the desired validation cadence. The inherited TinyGSM separator is two characters, backslash and `n`; it is not a newline.

The release uses LoRA, a smaller global batch, AdamW betas `(0.9, 0.95)`, weight decay `0.01`, and no EMA. This adaptation uses full backbone finetuning and the MMD optimizer/EMA settings so the baseline compares the CDLM objectives under the existing project setup. The default preserves CDLM's 5% warmup; the optional matched-update preset preserves MMD's 2,500-update warmup. It also uses TinyGSM in place of the release's model-specific training mixture. The auxiliary ground-truth term uses the original TinyGSM Python answer and normalizes over that sample's eligible answer/EOS/padding span; the release uses a fixed generation length of 256. See [upstream train.py](https://github.com/SqueezeAILab/CDLM/blob/708800ab0df03e2c284ab2cb358e42652a5e85a5/CDLM-train/train.py) for the author settings.

## Nirvana layout and logging

After the same environment and input staging used for the MMD job, the Nirvana run-command field can invoke the external launcher:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/train_cdlm_tinygsm_nirvana.sh"
```

This wrapper prepares trajectories and then trains, using the code bundle's original `checkpoints/tinygsm/mdlm_notok.ckpt`, `cache_dir`, and `cache_dir/tinygsm_mmd_offline_bundle`. Outputs go to a separate `cdlm_tinygsm` directory. The tracker experiment stays `generative-models/tinygsm`. Keep shell arrays and parameter expansions inside the external script; Nirvana interprets some brace expressions in its run-command field before Bash starts.

Preparation reports progress from rank zero every 30 seconds, starting before
data and teacher loading. Console messages use the `[cdlm-prepare]` prefix.
When Underdeep is configured, a separate run named `<logger.name> [prepare]`
records completed/total trajectories and shards, progress, elapsed time, and
the rate of newly completed trajectories. Its tracker metadata lives under
`<checkpointing.save_dir>/cdlm_prepare/`, so repeated preparation can resume
that run. The subsequent training run keeps the original name and metadata.
The monitor reads committed shard receipts without loading tensors or adding
distributed collectives; tracker requests run in a background thread.

An older job can spend its entire preparation phase without creating an
Underdeep run. Its console landmark is `Collecting original-teacher CDLM
trajectories; student training is a separate phase.` The cache manifest stays
`building` until all workers finish; each `part-*.json` receipt represents a
committed shard. At the default settings, 5,000 shards contain 40,000
trajectories. `Starting Training.` marks the transition to the student run.
If that message appears without a tracker run, inspect the console for
`[underdeep] disabled (setup failed: ...)`.

The model bundle supplies tokenizer/model assets, while the original TinyGSM
data must be present separately under `data.cache_dir` for offline preparation.
Both the untagged DatasetDict cache and the older matching-separator cache are
supported. For this preset the latter directory is
`tiny_gsm_bs512_answer_only_filtered_train_on_pad_HuggingFaceTB__SmolLM-135M_sep5c6e`,
containing `train/` and `validation/`. These splits are loaded unchanged; the
generic `tiny_gsm_train_bs512_unwrapped.dat` file is not required in that case.

[examples/nirvana_cdlm_run_command.sh](../examples/nirvana_cdlm_run_command.sh) contains the full setup, input staging, and launcher invocation for the Nirvana run-command field. It uses the same environment and input names as the MMD job and invokes the external script for both CDLM phases.

`PYTHON=/absolute/path/to/python` selects an environment. Trailing Hydra overrides change the run settings. Pass the same teacher/data/cache settings to preparation and training. Training-only and preparation-only wrappers are available for scheduling those stages separately.

## GSM8K comparison

```bash
bash scripts/generation_cdlm_tinygsm.sh \
  /absolute/path/to/runs/cdlm_tinygsm/checkpoints/last.ckpt \
  eval.generated_samples_path=/absolute/path/to/results/cdlm_tinygsm_eval.json
```

Evaluation reuses the MMD/SDTT confidence-threshold decoder and execution scorer. Training validation uses 10 thresholds: `0.325, 0.4, 0.475, 0.55, 0.625, 0.7, 0.775, 0.85, 0.925, 0.98`. The standalone generation script retains all 27 historical thresholds, `0.325, 0.35, …, 0.95, 0.98`. Both use greedy token proposals at temperature zero; `eval.gsm8k_pareto_thresholds=[...]` overrides either sweep. Decoding completes all 512 positions, including padding after the answer. NFE is the mean active denoiser calls per example, including a capped final removal if needed; accuracy uses the same execution-based math grader and five-second timeout.

CDLM also evaluates a separate fixed-budget **ancestral sampler at temperature
zero**, enabled by `eval.gsm8k_greedy_token=True`. It follows the neighboring
TinyGSM S-FLM protocol: choose each token by `argmax`, while randomly revealing
masked positions with probability `(alpha_s-alpha_t)/(1-alpha_t)`. Prompt tokens
remain fixed. The grid spans time 1 to `1e-5`; the last scheduled transition
sets `alpha_s=1` to finish all masks without an extra denoiser call. With
`sampling.predictor=ancestral_cache`, predictions are reused while the full
batch state is unchanged. NFE counts actual batch forward calls and assigns
that count to each batch member, matching the reference implementation.

The default budget is `sampling.steps=64`. Set `eval.gsm8k_steps=32`, for example,
to change only this evaluation budget. The confidence sweep keeps its own
threshold-dependent stopping rule. Both evaluations use the same GSM8K
examples, execution scorer, and validation EMA weights, with independent
seeded RNG scopes. Initial validation, periodic validation, final reporting,
and the standalone evaluation script all include the ancestral result.
Set `eval.gsm8k_greedy_token=False` to skip it.

EMA is enabled by default; `eval.disable_ema=True` selects live student weights. `eval.gsm8k_data_path` can point to the existing GSM8K test JSON. Full GSM8K remains the default; limits are only for explicit smoke checks. `eval.generate_samples=False` does not disable the GSM8K validation hook.

Underdeep retains `val/gsm8k_acc_thr*`, `val/gsm8k_nfe_thr*`, the `pareto/*` frontier, `val/samples`, and `pareto_history.json`. The ancestral temperature-zero result is logged separately as `val/gsm8k_acc_greedy` and `val/gsm8k_nfe_greedy`, and saved under `greedy` in `pareto_history.json` and the standalone evaluation JSON. The historical temperature-one `val/gsm8k_acc` and held-out `val/answer_*` diagnostics remain separate and are not implemented here. Claims about released CDLM block-causal KV-cache speedups require a separate architectural experiment; this run measures the full-attention MDLM frontier.

## Resume and validation

Full checkpoints are saved every 1,000 optimizer updates, with `checkpoints/last.ckpt` also refreshed at epoch boundaries so the final short interval is retained. Use the same preset, trajectory cache, and output directory when restoring a full Lightning checkpoint. Set `checkpointing.resume_from_ckpt=True`, `checkpointing.resume_ckpt_path=/absolute/path/to/checkpoints/last.ckpt`, and the original `hydra.run.dir`. Keep the teacher and trajectory manifest consistent. A student-only checkpoint is for evaluation or explicit initialization, not a full optimizer resume.

The CDLM presets and the Nirvana launcher enable `checkpointing.snapshot=True`.
After each completed checkpoint write, the run directory is published through
the same Nirvana snapshot APIs used by IDLM:
`nirvana_dl.snapshot.get_snapshot_path()` and `dump_snapshot()`. Its default
namespace is `cdlm_tinygsm`, so the snapshot contains
`cdlm_tinygsm/checkpoints/` together with Underdeep metadata, the saved config,
and evaluation history. `checkpointing.snapshot_namespace` changes that
relative namespace. `checkpointing.snapshot=False` keeps local checkpoints
without publishing them.

The active Lightning checkpoint-I/O backend is wrapped after Trainer creation.
It first finishes the full checkpoint write, then copies and publishes the run
directory on global rank zero. This covers periodic callbacks, epoch-end
`last.ckpt`, optional best-model callbacks, and the final module save. Each
snapshot file is copied to a temporary destination and renamed after the copy
finishes. The final checkpoint is published before final GSM8K evaluation;
another publication afterward retains the final JSON/frontier too.

Fresh runs keep `checkpointing.resume_from_ckpt=False`. For a restart using the
published MMD initialization preset, use:

```bash
CDLM_PRESET=cdlm_tinygsm_mmd \
bash "$SOURCE_CODE_PATH/code/scripts/train_cdlm_tinygsm_nirvana.sh" \
  checkpointing.resume_from_ckpt=True
```

If the requested local resume checkpoint is absent, the snapshot namespace is
restored into the current `checkpointing.save_dir` before declaring that path
missing or initializing the training tracker. The default resume path selects
`checkpoints/last.ckpt`. An existing local checkpoint takes priority, so an
older snapshot cannot overwrite progress whose upload failed. Restoration is
only requested for an explicit resume; a fresh run does not import old snapshot
weights. Successful uploads print `[snapshot] pushed ...`.

Snapshotting this run directory does not include the separate
`algo.trajectory_cache_dir`, original teacher weights, tokenizer or input
datasets. Keep the original inputs available and preserve or regenerate the
same trajectory cache before resuming. Full resume restores student, frozen
teacher head, EMA, optimizer, scheduler and training loops; it does not need the
external MMD initialization checkpoint. Outside Nirvana, or on a snapshot SDK
failure, a diagnostic is printed and local checkpoint saving continues.

An explicit CDLM resume request with a missing checkpoint raises an error;
it does not fall back to a fresh MMD or MDLM initialization.

At training end, the final `last.ckpt` is saved with live student weights and separate EMA state **before** final EMA evaluation and tracker publication. A final frontier is then evaluated if that optimizer step was not already reported, so a 16-epoch run ending between periodic validations still has an end-of-training comparison.

Run the CDLM CPU checks with:

```bash
python -m unittest discover -s tests -p 'test_cdlm*.py' -v
```

CPU initialization was also verified with the cached public S-FLM TinyGSM MDLM checkpoint and the production DiT: strict loading matched all 131 backbone state tensors, the frozen vocabulary-head weight and bias matched the raw student exactly, and all 130 fresh EMA shadows matched its trainable parameters. This check used offline tokenizer/checkpoint reads, with no forward call or optimizer update.

No GPU preparation, training, or quality evaluation was launched for this code-only task. CPU checks establish local loading, numerical, and interface behavior; they do not establish distributed GPU behavior, pretrained quality, cache size, or runtime. Report offline collection and training costs separately when the experiment is run on the target machine.
