#!/usr/bin/env bash
# Original standalone evaluation preset, without the didi_repro runtime/logger.
set -euo pipefail
DIDI_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${CHECKPOINT:?set CHECKPOINT to the prepared native DiDi .ckpt}"
: "${SAVE_DIR:?set SAVE_DIR to a NEW evaluation output directory}"
CHECKPOINT="$(realpath "$CHECKPOINT")"
SAVE_DIR="$(realpath -m "$SAVE_DIR")"
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HUB_DISABLE_TELEMETRY=1
cd "$DIDI_REPO_ROOT"
for didi_override in "$@"; do
  didi_key="${didi_override%%=*}"
  didi_key="${didi_key#++}"
  didi_key="${didi_key#+}"
  didi_key="${didi_key#\~}"
  case "$didi_key" in
    mode|checkpointing*|hydra*|eval.checkpoint_path|eval.generated_samples_path|eval)
      printf 'Protected evaluation/path override: %s. Use CHECKPOINT and SAVE_DIR.\n' "$didi_override" >&2
      exit 1
      ;;
  esac
done

# Validate before generation, but never install the repaired algorithm or logger.
python - "$CHECKPOINT" "$SAVE_DIR" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from didi_repro.entrypoint import environment_record, validate_environment
from didi_repro.runtime import verify_upstream

checkpoint, directory = map(Path, sys.argv[1:])
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
if directory.exists():
    raise FileExistsError(f'Use a new SAVE_DIR; will not overwrite {directory}')
if torch.cuda.device_count() != 1:
    raise ValueError('The original standalone evaluator needs exactly one visible GPU.')
manifest = verify_upstream()
record = environment_record()
strict = os.environ.get('DIDI_STRICT_ENVIRONMENT', '1') != '0'
try:
    validate_environment(SimpleNamespace(
        didi_runtime=SimpleNamespace(strict_environment=strict)), record)
except RuntimeError as error:
    raise RuntimeError(str(error).replace(
        '+didi_runtime.strict_environment=false', 'DIDI_STRICT_ENVIRONMENT=0')) from None
digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
directory.mkdir(parents=True, exist_ok=False)
(directory / 'authors_eval_environment.json').write_text(json.dumps({
    'environment': record,
    'installed_packages': {d.metadata['Name']: d.version
                           for d in importlib.metadata.distributions()},
    'core_environment_guard_strict': strict,
    'upstream_revision': manifest['commit'],
    'checkpoint': str(checkpoint),
    'checkpoint_sha256': digest.hexdigest(),
    'entrypoint': 'vendor/didi_instruct/main.py; no runtime patches',
    'note': 'Core version check is not a complete environment.yml build match.',
}, indent=2) + '\n')
PY

cd "$DIDI_REPO_ROOT/vendor/didi_instruct"
# These sampling/model/batching flags match scripts/eval-didi-instruct.sh.
# Path changes only. Keep default perplexity_batch_size=8 for the literal result;
# score_didi_authors_samples.py can score all saved samples separately afterward.
python -u main.py \
  mode=sample_eval \
  loader.batch_size=20 loader.eval_batch_size=20 \
  model=small algo=didi_instruct \
  "eval.checkpoint_path=$CHECKPOINT" \
  sampling.steps=16 sampling.num_sample_batches=2 \
  eval.generate_samples=true \
  sampling.predictor=guided sampling.noise_removal=ancestral \
  "checkpointing.save_dir=$SAVE_DIR" "hydra.run.dir=$SAVE_DIR" \
  "eval.generated_samples_path=$SAVE_DIR/samples.json" \
  "$@" 2>&1 | tee "$SAVE_DIR/console.log"
