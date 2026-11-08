"""Read the S-FLM/IDLM checkpoint without restoring serialized tokenizers."""

import os
import pickle

from hydra.utils import to_absolute_path
from huggingface_hub import hf_hub_download
import torch


class _TokenizerPlaceholder:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] == 'tokenizers':
            return _TokenizerPlaceholder
        return super().find_class(module, name)


class _CheckpointPickle:
    Unpickler = _Unpickler
    load = pickle.load


def read_checkpoint(source):
    if source.startswith('hf://'):
        parts = source[5:].split('/', 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError('Expected hf://owner/repository/checkpoint-file.')
        path = hf_hub_download(repo_id='/'.join(parts[:2]), filename=parts[2])
    else:
        path = to_absolute_path(os.path.expanduser(source))
    return torch.load(path, map_location='cpu', weights_only=False,
                      pickle_module=_CheckpointPickle)


def backbone_state(checkpoint):
    state = checkpoint.get('state_dict', checkpoint)
    selected = {key[len('backbone.'):].replace('_orig_mod.', ''): value
                for key, value in state.items() if key.startswith('backbone.')}
    if not selected:
        raise ValueError('Expected an MDLM checkpoint with backbone.* tensors.')
    return selected


def apply_ema(model, checkpoint):
    if checkpoint.get('checkpoint_weights') == 'ema':
        return
    if 'ema' not in checkpoint:
        raise ValueError('Checkpoint has no EMA; set eval.disable_ema=true for live weights.')
    params = [p for p in model._get_parameters() if p.requires_grad]
    shadows = checkpoint['ema']['shadow_params']
    if len(params) != len(shadows) or any(p.shape != s.shape for p, s in zip(params, shadows)):
        raise ValueError('Checkpoint EMA shapes do not match the student.')
    with torch.no_grad():
        for parameter, shadow in zip(params, shadows):
            parameter.copy_(shadow)
