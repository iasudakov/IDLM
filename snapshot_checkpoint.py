"""Publish completed checkpoints through the existing Lightning I/O backend."""

import os

from lightning.fabric.plugins.io import CheckpointIO

import nirvana_snapshot


class SnapshotCheckpointIO(CheckpointIO):
    """Keep backend serialization intact and snapshot only after it returns.

    Lightning's strategy invokes checkpoint I/O on global rank zero. The
    snapshot helper also checks rank, and export failures never turn a complete
    local checkpoint into a distributed save failure.
    """

    def __init__(self, checkpoint_io, snapshot_dir, enabled=True, namespace=None):
        self.checkpoint_io = checkpoint_io
        self.snapshot_dir = os.fspath(snapshot_dir)
        self.enabled = enabled
        self.namespace = namespace

    def save_checkpoint(self, checkpoint, path, *args, **kwargs):
        result = self.checkpoint_io.save_checkpoint(checkpoint, path, *args, **kwargs)
        if self.enabled:
            try:
                nirvana_snapshot.save(
                    self.snapshot_dir, enabled=self.enabled,
                    reason=f'checkpoint written: {os.fspath(path)}', namespace=self.namespace)
            except Exception as error:
                # Keep every DDP rank able to reach Trainer.save_checkpoint's
                # barrier even if an optional snapshot exporter malfunctions.
                print(f'[snapshot] checkpoint export failed, continuing: {error}', flush=True)
        return result

    def load_checkpoint(self, *args, **kwargs):
        # Omitting map_location must preserve the actual backend's default,
        # notably TorchCheckpointIO's CPU remapping rather than None.
        return self.checkpoint_io.load_checkpoint(*args, **kwargs)

    def remove_checkpoint(self, *args, **kwargs):
        return self.checkpoint_io.remove_checkpoint(*args, **kwargs)

    def teardown(self, *args, **kwargs):
        return self.checkpoint_io.teardown(*args, **kwargs)
