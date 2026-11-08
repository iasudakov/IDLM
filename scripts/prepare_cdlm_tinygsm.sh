#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Each torchrun process collects one rank's shard on its local GPU. Mode is
# fixed after user overrides so this entry point only prepares the cache.
exec "${PYTHON:-python}" -u -m torch.distributed.run \
  --standalone "--nproc_per_node=${CDLM_PREPARE_GPUS:-4}" \
  -m main \
  "+experiment=${CDLM_PRESET:-cdlm_tinygsm}" \
  "$@" \
  mode=prepare_cdlm
