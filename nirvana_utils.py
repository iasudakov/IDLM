"""Nirvana state transfer for preemptible jobs.

Vendored from ``yt_tools/nirvana_utils.py`` (the module ``yrDiT/train.py`` uses)
so this repo is self-contained on the Nirvana image: the training script must not
depend on a sibling checkout being present.

The contract Nirvana offers a preempted job is a single directory, the *snapshot
path*. Whatever a job puts there is packed into a ``state`` archive and unpacked
into the snapshot path of the job that restarts it; everything else about the
container is gone. Restartable training therefore has exactly two obligations:

1. at startup, copy the snapshot path *into* the local output dir, so the
   checkpoints of the preempted attempt reappear where the trainer looks for
   them (:func:`copy_snapshot_to_out`);
2. periodically, copy the local output dir *out* to the snapshot path and dump it
   (:func:`copy_out_to_snapshot`), which bounds how much progress a preemption
   can destroy to one snapshot interval.

Off Nirvana ``nirvana_dl`` is not importable and every function here is a no-op,
so the same script runs unchanged on a local machine.
"""

import os
import shutil

try:  # only present inside a Nirvana PythonDL job
  import nirvana_dl
except ImportError:  # pragma: no cover - the local case
  nirvana_dl = None


def nirvana_available():
  """True when running inside Nirvana, i.e. when the transfers do something."""
  return nirvana_dl is not None


def _copy_tree(src, dst):
  """Merge-copy `src` into an existing `dst`, overwriting per file.

  The reference implementation uses ``distutils.dir_util.copy_tree``, which is
  removed in Python 3.12; ``shutil.copytree(dirs_exist_ok=True)`` has the same
  merge semantics on 3.8+. Try the modern one first and keep distutils as the
  fallback so this works on whichever interpreter the Nirvana image ships.
  """
  try:
    shutil.copytree(src, dst, dirs_exist_ok=True)
  except TypeError:  # pragma: no cover - python < 3.8
    from distutils.dir_util import copy_tree
    copy_tree(src, dst)


def copy_snapshot_to_out(out):
  """Restore the preempted attempt's state from the snapshot path into `out`.

  Call this *before* the trainer decides whether a checkpoint to resume from
  exists, otherwise the restored `last.ckpt` arrives too late to be used.
  """
  if nirvana_dl is None:
    print('[nirvana] not a Nirvana job; nothing to restore.', flush=True)
    return
  snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
  print(f'[nirvana] copy previous state from {snapshot_path} to {out}',
        flush=True)
  os.makedirs(out, exist_ok=True)
  _copy_tree(snapshot_path, out)


def copy_out_to_snapshot(out, dump=True):
  """Publish `out` to the snapshot path so a restart can pick it up.

  Args:
    out: local directory holding the run state (checkpoints, logs, the
      underdeep resume handle).
    dump: also pack the snapshot path into the ``state`` archive that Nirvana
      hands to the next attempt. Without this the copy is invisible to the
      restart, so it defaults on, exactly as in the reference.
  """
  if nirvana_dl is None:
    print('[nirvana] not a Nirvana job; checkpoint not transferred.',
          flush=True)
    return
  snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
  print(f'[nirvana] copy {out} to the snapshot path: {snapshot_path}',
        flush=True)
  _copy_tree(out, snapshot_path)
  if dump:
    nirvana_dl.snapshot.dump_snapshot(snapshot_path)


def copy_logs_to_logs_path(out):
  """Publish `out` to the Nirvana logs path (visible in the job's UI)."""
  if nirvana_dl is None:
    return
  logs_path = nirvana_dl.logs_path()
  print(f'[nirvana] copy {out} to the logs path: {logs_path}', flush=True)
  _copy_tree(out, logs_path)
