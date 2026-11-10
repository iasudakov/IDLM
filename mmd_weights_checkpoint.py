"""Keep periodic student-only MMD weights alongside the full final checkpoint."""

import copy
import os
from pathlib import Path
import tempfile

from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
import torch

import nirvana_snapshot


def student_weights(module, *, global_step, epoch):
    """Build an inference checkpoint without swapping live/EMA parameters.

    EMA tracks ``_get_parameters()`` order, which need not be the same as a
    backbone state dict. Match shadows by parameter identity, retaining actual
    backbone buffers and any explicitly frozen backbone parameters.
    """
    backbone = module.backbone
    state = backbone.state_dict()
    parameters = dict(backbone.named_parameters(remove_duplicate=False))
    ema = getattr(module, 'ema', None)
    shadows = {}
    if ema is not None:
        tracked = [parameter for parameter in module._get_parameters() if parameter.requires_grad]
        if len(tracked) != len(ema.shadow_params):
            raise ValueError('Student EMA parameter count does not match the tracked model parameters.')
        for parameter, shadow in zip(tracked, ema.shadow_params):
            if parameter.shape != shadow.shape:
                raise ValueError('Student EMA parameter shape does not match the tracked model parameter.')
            shadows[id(parameter)] = shadow
    exported = {}
    for name, value in state.items():
        parameter = parameters.get(name)
        if ema is not None and parameter is not None and parameter.requires_grad:
            if id(parameter) not in shadows:
                raise ValueError(f'Student EMA is missing backbone parameter {name}.')
            value = shadows[id(parameter)]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Only tensor backbone state is supported, got {name}.')
        exported['backbone.' + name] = value.detach().cpu().clone()
    config = module.config
    if OmegaConf.is_config(config):
        # Keep interpolation strings as recorded, without resolving secrets or
        # requiring a live Hydra runtime during an inference-only export.
        config = OmegaConf.to_container(config, resolve=False)
    else:
        config = copy.deepcopy(config)
    return {
        'state_dict': exported,
        'checkpoint_weights': 'ema' if ema is not None else 'raw',
        'global_step': int(global_step),
        'epoch': int(epoch),
        'hyper_parameters': {'config': config},
    }


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.',
                                              suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class MMDWeightsCheckpoint(Callback):
    """Export every N completed optimizer steps; retain all numbered exports.

    Every rank enters one broadcast at an export boundary. Only global rank zero
    builds/writes weights and publishes the snapshot. A failed local write raises
    on every rank; optional snapshot service failures remain best effort.
    """

    def __init__(self, save_dir, every_n_train_steps=500, snapshot=True,
                 namespace='mmd_from_cdlm_tinygsm'):
        if type(every_n_train_steps) is not int or every_n_train_steps <= 0:
            raise ValueError('every_n_train_steps must be a positive integer.')
        self.save_dir = Path(save_dir)
        self.every_n_train_steps = every_n_train_steps
        self.snapshot = bool(snapshot)
        self.namespace = namespace
        self.last_saved_step = 0

    @property
    def state_key(self):
        # New Nirvana containers have different absolute paths. The cadence,
        # rather than that path, identifies this callback across full resumes.
        return self._generate_state_key(every_n_train_steps=self.every_n_train_steps)

    def state_dict(self):
        return {'last_saved_step': self.last_saved_step}

    def load_state_dict(self, state_dict):
        step = state_dict.get('last_saved_step', 0)
        if type(step) is not int or step < 0:
            raise ValueError('Invalid periodic MMD weights callback step.')
        self.last_saved_step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if (step <= self.last_saved_step or step <= 0
                or step % self.every_n_train_steps != 0):
            return
        error = None
        message = None
        if trainer.is_global_zero:
            try:
                checkpoint = student_weights(pl_module, global_step=step,
                                             epoch=trainer.current_epoch)
                path = self.save_dir / 'checkpoints' / f'step={step:08d}.ckpt'
                _atomic_save(checkpoint, path)
                print(f'[mmd-weights] Saved {checkpoint["checkpoint_weights"]} student '
                      f'weights at step={step}: {path}', flush=True)
            except Exception as caught:
                error = caught
                message = f'{type(caught).__name__}: {caught}'
            if error is None:
                # This helper catches SDK/copy/publication failures itself; also
                # keep an unexpected optional-export exception out of DDP's path.
                try:
                    nirvana_snapshot.save(
                        str(self.save_dir), enabled=self.snapshot, namespace=self.namespace,
                        reason=f'MMD student weights step={step}')
                except Exception as caught:
                    print(f'[mmd-weights] Snapshot export failed, continuing: {caught}', flush=True)
        message = trainer.strategy.broadcast(message, src=0)
        if message is not None:
            raise RuntimeError(f'Periodic MMD student weights could not be saved: {message}') from error
        self.last_saved_step = step
