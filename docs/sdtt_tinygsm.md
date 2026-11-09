# SDTT on TinyGSM with the IDLM MDLM backbone

This branch adds full-sequence SDTT to IDLM. It uses the existing time-unconditional MDLM, TinyGSM tokenization, prompt masking, and GSM8K accuracy/NFE comparison protocol. There is no block decomposition. The implementation follows [the official SDTT source at commit `1150985e`](https://github.com/jdeschena/sdtt/blob/1150985e90b8f2d5749e4469d5154eff9ec922c4/src/sdtt/core/distill/multi_round_sdtt.py), adapted to the conditional TinyGSM task. The branch starts from IDLM commit `831fbac67fb753d86066f756faa06c63ff9b6dee`.

Only code and CPU checks were requested. No GPU training or quality evaluation was launched, and no trained SDTT checkpoint or measured accuracy is included.

## Training

Use the project's existing Python environment and dependencies. The default launcher selects the official SDTT round cadence: seven rounds of 10,000 student optimizer updates, for 70,000 updates total. A fresh run initializes student and frozen teacher from raw MDLM backbone weights, resets student EMA, and starts a fresh optimizer.

```bash
bash scripts/train_sdtt_tinygsm.sh \
  training.finetune_path=/absolute/path/to/tinygsm/mdlm_notok.ckpt \
  hydra.run.dir=/absolute/path/to/runs/sdtt-tinygsm
```

If `training.finetune_path` is omitted, the preset downloads `hf://jdeschena/s-flm/tinygsm/mdlm.ckpt`. Use the original checkpoint from the MMD run when available: the public release is an explicit fallback, and checkpoint identity is not established by this implementation. `PYTHON=/path/to/python` selects a Python executable. Extra Hydra arguments passed to the launcher override the preset.

For the historical Nirvana MMD launch layout, use the explicit cluster launcher:

```bash
bash "$SOURCE_CODE_PATH/code/scripts/train_sdtt_tinygsm_nirvana.sh"
```

It uses `$SOURCE_CODE_PATH/code/checkpoints/tinygsm/mdlm_notok.ckpt`, the existing `cache_dir/tinygsm_mmd_offline_bundle`, and the same Underdeep experiment, `generative-models/tinygsm`, with run name `sdtt tinygsm lr1e-5 official 70k`. Checkpoints, Hydra output, and `pareto_history.json` go to `$SOURCE_CODE_PATH/code/sdtt_tinygsm`. `SDTT_CODE_DIR=/absolute/path/to/code` overrides the code directory; when both directory variables are unset, it uses this checkout. The launcher lists the data, batching, validation, sampler, and official SDTT schedule overrides explicitly. Trailing Hydra overrides still win. For four GPUs with the same effective batch and validation cadence, append `trainer.devices=4 trainer.accumulate_grad_batches=4 trainer.val_check_interval=2000`.

The SDTT GSM8K validation hook publishes the same accuracy/NFE curves independently of `eval.generate_samples`; the cluster launcher leaves this flag false to avoid an additional unconditional sample-generation path. Execution scoring and all reference confidence thresholds remain enabled.

`model_weights_dir` supplies model/tokenizer assets; training data must also be
present under `data.cache_dir` on a machine without network access. The loader
accepts both the original TinyGSM DatasetDict cache name and older MMD caches
with a separator suffix. For this preset the latter directory is
`tiny_gsm_bs512_answer_only_filtered_train_on_pad_HuggingFaceTB__SmolLM-135M_sep5c6e`,
containing both `train/` and `validation/`. It reuses their tokens and split
membership directly. The generic `tiny_gsm_train_bs512_unwrapped.dat` cache
may be absent when this DatasetDict exists. If neither format exists, supply
the original cache from the MMD job, or set `data.data_path` to an existing
TinyGSM question/code JSON to preprocess locally. GSM8K evaluation separately
needs `data/gsm8k_test.json` or an explicit `eval.gsm8k_data_path`.

The default assumes eight CUDA devices, 32 examples per device per microbatch, and two microbatches per optimizer update: `8 × 32 × 2 = 512` examples. If changing hardware, preserve `loader.global_batch_size=512`; accumulation is derived automatically. For example:

```bash
bash scripts/train_sdtt_tinygsm.sh \
  training.finetune_path=/absolute/path/to/mdlm_notok.ckpt \
  trainer.devices=1 loader.batch_size=8 \
  trainer.val_check_interval=32000
```

Here accumulation becomes 64. Setting the validation interval to `500 × accumulation` preserves the default cadence of one validation per 500 optimizer updates. The code verifies that the actual global batch equals the requested batch before training starts.

| Round | Completed optimizer updates | Teacher grid | Student grid |
| --- | --- | --- | --- |
| 1 | 0–9,999 | 1,024 | 512 |
| 2 | 10,000–19,999 | 512 | 256 |
| 3 | 20,000–29,999 | 256 | 128 |
| 4 | 30,000–39,999 | 128 | 64 |
| 5 | 40,000–49,999 | 64 | 32 |
| 6 | 50,000–59,999 | 32 | 16 |
| 7 | 60,000–69,999 | 16 | 8 |

Each target uses two frozen-teacher denoising calls and a single student prediction. The loss is reverse KL, `KL(student || teacher)`, averaged over initially masked answer/padding positions. The teacher records a position's distribution when it first reveals that position; positions still masked after the rollout use the final teacher distribution. Teacher promotion copies the live student at round boundaries; `algo.use_ema_on_growth=False` follows the selected official default. The student grid stops shrinking at eight steps if the training budget is extended.

For a shorter comparison run, the optional 20,000-update preset uses 2,500 updates per compression round and spends the remaining budget at the final eight-step grid:

```bash
SDTT_PRESET=sdtt_tinygsm bash scripts/train_sdtt_tinygsm.sh \
  training.finetune_path=/absolute/path/to/mdlm_notok.ckpt
```

This shorter preset changes the official round cadence. The default 70,000-update run preserves that cadence and uses a larger training budget than the MMD launch configuration.

## Data and comparison protocol

The reference is MMD run `3e0ac6b1-7a8f-43c3-af58-d9e21b886da7`. [The provenance record](../provenance/sdtt_tinygsm_reference.json) includes its launch overrides, cached logged numeric parameters, source hashes, and limitations.

| Setting | SDTT TinyGSM presets |
| --- | --- |
| Dataset | `TinyGSM/TinyGSM`, training split |
| Tokenizer | `HuggingFaceTB/SmolLM-135M` |
| Backbone | IDLM `small` DiT: 12 layers, width 768, 12 attention heads |
| Sequence length | 512 |
| Example format | BOS + stripped question + separator + stripped Python code + EOS + padding |
| Separator | Literal backslash followed by `n`, codepoints `[92, 110]` |
| Split | 1% held out, split seed 42 |
| Filtering | Drop sequences longer than 512; `wrap=False` |
| Corruption/loss span | Answer, EOS, and padding; prompt stays visible |
| Mask layout | Independent random full-sequence masking |
| Optimizer | AdamW, LR `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0 |
| LR schedule | Constant after 2,500 warmup optimizer updates |
| EMA | `0.9999` |
| Precision | bf16 mixed model execution, float32 distributions and reverse KL |
| Gradient clipping | Disabled |
| GSM8K evaluation | Before training and each validation; greedy confidence sweep |
| Default validation cadence | Every 1,000 microbatches = 500 optimizer updates at default hardware/batch |

The separator is intentionally preserved from IDLM's TinyGSM config. It is two text characters, not a newline. The exact historical run string is inferred from that source: the numeric tracker cache does not preserve strings, and retrieval of the reference run's frozen source returned 404. If changing the separator, use a fresh `data.cache_dir`; the inherited dataset cache name does not include the separator.

The reference logs explicitly record `adversarial_distill.accum_batches=2`, `loader.global_batch_size=512`, eight devices, and per-device batch 32. Its separate `trainer.accumulate_grad_batches=1` controlled Lightning while the MMD loop accumulated manually. SDTT uses automatic student optimization with Lightning accumulation two. Both therefore target 512 original examples per student update. Reference timing records map raw tracker step 999 to 500 student updates and 13,999 to 7,000; compare normalized student updates when plotting training curves.

The reference used `trainer.precision=bf16` and `training.loss_precision=bf16`. This implementation keeps SDTT probability and KL calculations in float32 for stable distillation, with bf16 model execution. The MMD feature kernel, feature layer, candidate groups, and critic are replaced by the SDTT teacher objective. These changes and the selected 70,000-update training budget prevent a claim of identical training computation or wall time.

The diffusion-time floor stays at IDLM's `training.sampling_eps=1e-3`, compared with `1e-5` in the official SDTT config. Time sampling also retains IDLM's independent jitter within antithetic strata; the official SDTT implementation uses a shared random offset. Both produce uniform marginal diffusion times. These are deliberate choices to retain the MMD setup.

## Evaluation

Run the same GSM8K threshold sweep on any saved student checkpoint:

```bash
bash scripts/generation_sdtt_tinygsm.sh \
  /absolute/path/to/runs/sdtt-tinygsm/checkpoints/last.ckpt \
  eval.generated_samples_path=/absolute/path/to/results/sdtt_tinygsm_eval.json
```

Evaluation uses EMA weights by default. Add `eval.disable_ema=True` for live student weights, including when a checkpoint does not contain EMA. To use an existing GSM8K test JSON, add `eval.gsm8k_data_path=/absolute/path/to/gsm8k_test.json`. Its records must contain `prompt` and `response_ground_truth` strings. Otherwise the evaluator reuses a cached file or downloads the `openai/gsm8k` main test split.

The sampler is the MMD comparison's `low_confidence_dynamic` decoder with temperature zero. Thresholds are `0.325, 0.35, …, 0.95, 0.98`. Each forward commits masked tokens whose confidence is strictly greater than the threshold; an unfinished row with no accepted token commits its most confident token. EOS does not stop decoding: the remaining answer/padding positions are completed. The default cap is 512 iterations, with a counted final noise-removal call if needed. `sampling.steps=64` is inherited metadata and does not limit this threshold decoder.

Accuracy uses the project's S-FLM execution-based math scoring, with a five-second timeout. NFE is the mean number of active denoiser calls per example, including the capped final call. The JSON also records total batch forwards, checkpoint weight choice, the dataset hash, separator, seed, sequence length, and thresholds. Underdeep uses the reference experiment `generative-models/tinygsm` and exact threshold keys `val/gsm8k_acc_thr*` and `val/gsm8k_nfe_thr*`; the earlier `pareto/*` aliases remain available. `pareto/gsm8k_step_*`, `pareto/frontier`, and `pareto_history.json` retain the accuracy/NFE frontier, including prior checkpoints. `val/samples` contains up to eight already-generated examples near threshold 0.9, without extra model calls. `student_loss` aliases the SDTT reverse KL; `trainer/lr`, `train/sdtt_kl`, `train/sdtt_round`, and `train/sdtt_student_steps` describe training. Held-out `val/nll`, `val/bpd`, and `val/ppl` remain MDLM ELBO diagnostics.

The historical run also recorded separate fixed-64-step GSM8K and held-out answer-execution diagnostics (`val/gsm8k_acc`, `val/gsm8k_acc_greedy`, `val/gsm8k_nfe`, `val/answer_*`). Those separate evaluations are not implemented here and are not substituted with threshold-sweep scores. The matched comparison is the confidence-threshold accuracy/NFE frontier. MMD-specific rewards have no SDTT equivalent.

## Checkpoints and resume

The new preset retains full Lightning checkpoints every 1,000 optimizer updates and writes `checkpoints/last.ckpt`. This matches the reference's cadence of 2,000 microbatches at accumulation two. Checkpoints retain student, frozen teacher, compression round, EMA, optimizer, scheduler, and Lightning loop state.

Use the same preset, schedule, output directory, and checkpoint when resuming:

```bash
bash scripts/train_sdtt_tinygsm.sh \
  checkpointing.resume_from_ckpt=True \
  checkpointing.resume_ckpt_path=/absolute/path/to/runs/sdtt-tinygsm/checkpoints/last.ckpt \
  hydra.run.dir=/absolute/path/to/runs/sdtt-tinygsm
```

On a resume, the full SDTT state is restored rather than reinitializing from `training.finetune_path`. Keep the original round cadence and global batch. Lightning restores its loop and optimizer state; this implementation does not promise bitwise replay of dataloader worker RNG across interruption.

## Validation status

The 39 CPU tests cover the SDTT target/objective and schedule, checkpoint loading/EMA selection, offline model assets, cluster launch command, conditional evaluator, scoring interface, and exact Underdeep metric/image/table names. Loading the actual S-FLM TinyGSM checkpoint into the production DiT was also checked on CPU: strict loading passes, all 130 student/teacher parameter tensors match, and the vocabulary/embedding shape is correct. Full GPU training, multi-device execution, and pretrained accuracy/NFE results remain to be run on the target machine. Neither comparison quality nor runtime is inferred from CPU checks.
