#!/usr/bin/env bash
# Explicit diagnostic, NOT the untouched authors' inference control.
set -euo pipefail
DIDI_REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export DIDI_INFERENCE_PRECISION=float64
exec bash "$DIDI_REPO_ROOT/scripts/eval_didi_inference.sh" "$@"
