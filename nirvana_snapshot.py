"""Persist the run directory across preemptions, via Nirvana's snapshot store.

A preemptible cluster job can be killed and restarted at any point, with a fresh
container each time: whatever was written to local disk is gone. Nirvana's
answer is a snapshot store that survives restarts, and two calls that move a
directory in and out of it. `yrDiT/train_imagenet_accelerator.py` uses exactly
this pair -- restore once at startup, push after every checkpoint write.

`yt_tools` only exists inside the Nirvana job image, so every entry point here
degrades to a no-op locally rather than failing. That is not merely convenient:
the local case is the common one during development, and a snapshot layer that
raised on a laptop would be disabled and then forgotten by the time it mattered.

The restore is rank-zero-only. Lightning's `subprocess_script` launcher runs the
main process through `_train` and into `trainer.fit`, which is what spawns the
other ranks -- so rank zero's restore has already completed before any child
process reaches this code. Children are recognised by `LOCAL_RANK`, which the
launcher sets for them and not for the parent.
"""

import os


def _rank():
  """Best-effort rank before a Trainer exists to ask."""
  for key in ('LOCAL_RANK', 'RANK', 'SLURM_PROCID'):
    value = os.environ.get(key)
    if value is not None:
      try:
        return int(value)
      except ValueError:
        return 0
  return 0


def available():
  """Whether the Nirvana helpers can be imported at all."""
  try:
    from yt_tools.nirvana_utils import (  # noqa: F401
      copy_out_to_snapshot, copy_snapshot_to_out)
  except Exception:
    return False
  return True


def restore(path, enabled=True):
  """Pull a previous run's directory out of the snapshot store into `path`.

  Must run BEFORE the resume-checkpoint existence check: on a restarted job the
  local disk is empty, and this call is what puts `last.ckpt` back on it. Check
  first and the run silently starts from scratch with a full snapshot sitting
  unused.

  Returns True when a restore actually ran.
  """
  if not enabled:
    return False
  if _rank() != 0:
    return False
  try:
    from yt_tools.nirvana_utils import copy_snapshot_to_out
  except Exception:
    print('[snapshot] yt_tools unavailable; not restoring (expected off '
          'Nirvana)', flush=True)
    return False
  try:
    os.makedirs(path, exist_ok=True)
    copy_snapshot_to_out(path)
    print(f'[snapshot] restored into {path}', flush=True)
    return True
  except Exception as e:  # pragma: no cover - cluster-side helper
    # A failed restore means starting over, which is bad but recoverable. A
    # raise here means the job dies on startup every time it is preempted,
    # which is worse.
    print(f'[snapshot] restore failed, continuing without it: {e}', flush=True)
    return False


def save(path, enabled=True, reason=''):
  """Push `path` into the snapshot store, so a restart can restore it.

  Called after a checkpoint is written. Failure is reported and swallowed: the
  checkpoint is already safely on local disk, and losing the push costs the
  *next* preemption some progress rather than this run its life.
  """
  if not enabled:
    return False
  if _rank() != 0:
    return False
  try:
    from yt_tools.nirvana_utils import copy_out_to_snapshot
  except Exception:
    return False
  try:
    copy_out_to_snapshot(path)
    suffix = f' ({reason})' if reason else ''
    print(f'[snapshot] pushed {path}{suffix}', flush=True)
    return True
  except Exception as e:  # pragma: no cover - cluster-side helper
    print(f'[snapshot] push failed, continuing: {e}', flush=True)
    return False
