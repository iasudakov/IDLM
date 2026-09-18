"""Optional evaluated checkpoints for the separate DiDi training entrypoint."""

import os


def make_evaluated_checkpoint(config):
  """Retain complete evaluated models, labelled by Lightning optimizer step."""
  checkpointing = config.checkpointing
  if not checkpointing.get('save_evaluated_checkpoints', False):
    return None
  from evaluated_checkpoint import EvaluatedCheckpoint
  directory = os.fspath(checkpointing.save_dir)
  return EvaluatedCheckpoint(
    os.path.join(directory, 'checkpoints'),
    start_step=checkpointing.get('save_evaluated_from_step', 0),
    snapshot_dir=directory, snapshot=checkpointing.get('snapshot', False))
