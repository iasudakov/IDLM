#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# PYTHON selects an existing environment; trailing Hydra overrides win.
# Default: official 70k-update cadence. SDTT_PRESET=sdtt_tinygsm selects
# the optional matched 20k-update budget.
exec "${PYTHON:-python}" -u -m main \
  "+experiment=${SDTT_PRESET:-sdtt_tinygsm_official}" \
  logger.name=sdtt-tinygsm \
  "$@"
