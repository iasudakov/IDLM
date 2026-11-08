#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Prepare trajectories first with prepare_cdlm_tinygsm.sh. Missing caches fail
# explicitly instead of silently launching a preprocessing job during training.
exec "${PYTHON:-python}" -u -m main \
  "+experiment=${CDLM_PRESET:-cdlm_tinygsm}" \
  "$@"
