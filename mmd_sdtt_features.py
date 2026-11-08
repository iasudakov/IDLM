"""Frozen SDTT MDLM teachers as token-aligned MMD feature maps.

No SDTT Python dependency or remote model code is executed. The released
config describes the architecture; weights must match this repository's DiT
strictly. By default the Hub revision selects the undistilled 863M teacher.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_MODEL = 'jdeschena/sdtt'
DEFAULT_REVISION = 'teacher_400k_large_step_400000'
GPT2_VOCAB_SIZE = 50257
MDLM_VOCAB_SIZE = GPT2_VOCAB_SIZE + 1
# SHA256 of canonical GPT-2 token->ID JSON (sort_keys, UTF-8, compact separators).
# Checked against the locally cached official GPT2TokenizerFast vocabulary.
GPT2_BASE_VOCAB_SHA256 = '179cad62d906b7217f1c9431ece06e7a78a7721f9580960147a6c1ea0a53fc65'


def validate_tokenizer(tokenizer, mask_index):
  """Accept GPT-2, including the historical [PAD]/mask-ID alias, only."""
  if mask_index != GPT2_VOCAB_SIZE or tokenizer.vocab_size != GPT2_VOCAB_SIZE:
    raise ValueError('SDTT MDLM features require GPT-2 base vocab50257 and '
                     'mask_index50257; TinyGSM/other vocabularies are incompatible')
  vocab = tokenizer.get_vocab()
  base = {token: index for token, index in vocab.items() if index < GPT2_VOCAB_SIZE}
  digest = hashlib.sha256(json.dumps(
    base, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
  if len(base) != GPT2_VOCAB_SIZE or digest != GPT2_BASE_VOCAB_SHA256:
    raise ValueError('SDTT MDLM features require the original GPT-2 token-to-ID mapping')
  extras = {token: index for token, index in vocab.items() if index >= GPT2_VOCAB_SIZE}
  if extras not in ({}, {'[PAD]': GPT2_VOCAB_SIZE}):
    raise ValueError('SDTT MDLM features allow only the historical added [PAD] '
                     'alias at50257; token IDs are never remapped')
  if tokenizer.bos_token_id != 50256 or tokenizer.eos_token_id != 50256:
    raise ValueError('SDTT MDLM features require GPT-2 BOS/EOS ID50256')


def _checkpoint_files(model_id, revision, local_path, local_files_only):
  if local_path is not None:
    root = Path(local_path).expanduser()
    if not root.is_dir():
      raise FileNotFoundError(f'SDTT feature checkpoint directory is missing: {root}')
    files = tuple(root / name for name in ('config.json', 'model.safetensors'))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
      raise FileNotFoundError('Incomplete local SDTT feature checkpoint; no Hub '
                              f'fallback will be attempted. Missing: {missing}')
    return files
  if not model_id or not revision:
    raise ValueError('SDTT feature checkpoint needs an explicit model ID and revision')
  from huggingface_hub import hf_hub_download
  kwargs = dict(repo_id=model_id, revision=revision,
                local_files_only=bool(local_files_only))
  config_path = Path(hf_hub_download(filename='config.json', **kwargs))
  # Pin the second request to the same immutable Hub snapshot when available.
  if config_path.parent.parent.name == 'snapshots' and re.fullmatch(r'[0-9a-f]{40}', config_path.parent.name):
    kwargs['revision'] = config_path.parent.name
  weights_path = Path(hf_hub_download(filename='model.safetensors', **kwargs))
  return config_path, weights_path


def _architecture(config):
  """Read primitives, never instantiate/resolve Hydra values from the artifact."""
  if not isinstance(config, dict) or config.get('time_conditioning') is not False:
    raise ValueError('Expected an SDTT MDLM config with time_conditioning=False')
  if config.get('tokenizer', {}).get('name') != 'gpt2':
    raise ValueError('SDTT feature checkpoint must declare tokenizer.name=gpt2')
  if config.get('algo', {}).get('causal_attention', False) is not False:
    raise ValueError('SDTT MDLM feature checkpoint cannot request causal attention')
  model = config.get('model', {})
  if model.get('type') != 'ddit-orig':
    raise ValueError('Expected the SDTT ddit-orig architecture')
  result = {}
  for name in ('hidden_size', 'n_blocks', 'n_heads', 'cond_dim', 'length'):
    value = model.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
      raise ValueError(f'SDTT model.{name} must be a positive integer')
    result[name] = value
  if result['hidden_size'] % result['n_heads'] or (result['hidden_size'] // result['n_heads']) % 2:
    raise ValueError('SDTT head dimension must be an even integer')
  dropout = model.get('dropout', .1)
  if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
    raise ValueError('SDTT model.dropout must be in [0,1)')
  result['dropout'] = float(dropout)
  result['scale_by_sigma'] = bool(model.get('scale_by_sigma', True))
  # SDTT's stale model.causal field is ignored by its ddit-orig implementation.
  # MDLM is bidirectional and has AdaLN, unlike the causal AR variant here.
  return {'model': result, 'algo': {'causal_attention': False}}


def _strip_prefixes(state):
  """Normalize only uniform wrapper prefixes; do not filter unknown tensors."""
  for prefix in ('_orig_mod.', 'backbone.', '_orig_mod.'):
    if state and all(key.startswith(prefix) for key in state):
      state = {key[len(prefix):]: tensor for key, tensor in state.items()}
  return state


class SdttFeatureBackbone(nn.Module):
  """Zero-timestep, frozen DiT prefix with the original post-block hook paths."""
  def __init__(self, backbone, feature_layer=None):
    super().__init__()
    self.backbone = backbone
    self.config = backbone.config
    self.vocab_size = backbone.vocab_size
    from mmd_reward import resolve_feature_layers
    _, indices = resolve_feature_layers(backbone, feature_layer)
    self.last_feature_layer = max(indices)
    self.requires_grad_(False)
    self.train(False)

  @property
  def blocks(self):
    return self.backbone.blocks

  @property
  def vocab_embed(self):
    return self.backbone.vocab_embed

  def train(self, mode=True):
    # Protect dropout when a caller applies .train() recursively.
    return super().train(False)

  @torch.no_grad()
  def forward(self, tokens, sigma=None):
    del sigma  # SDTT's teacher is time-agnostic; physical masking stays upstream.
    if tokens.ndim != 2 or tokens.shape[1] > self.config.model.length:
      raise ValueError(f'SDTT features expect (batch,length), length <= '
                       f'{self.config.model.length}; got {tuple(tokens.shape)}')
    self.backbone.eval()
    x = self.backbone.vocab_embed(tokens)
    if self.last_feature_layer == -1:
      return x
    zero_sigma = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.float32)
    conditioning = F.silu(self.backbone.sigma_map(zero_sigma))
    rotary = self.backbone.rotary_emb(x)
    # This is the exact archived DiT prefix. The unused vocabulary head and
    # blocks deeper than the deepest requested feature do not run.
    with torch.cuda.amp.autocast(enabled=tokens.is_cuda, dtype=torch.bfloat16):
      for block in self.backbone.blocks[:self.last_feature_layer + 1]:
        x = block(x, rotary, c=conditioning)
    return x


def load_sdtt_feature_backbone(model_id=DEFAULT_MODEL, device='cpu', *,
                               tokenizer, mask_index, revision=DEFAULT_REVISION,
                               local_path=None, local_files_only=False,
                               feature_layer=None):
  """Load full, strictly compatible teacher weights without changing global RNG.

  A local directory must contain config.json and model.safetensors from the
  teacher revision. Offline mode never falls back to an online model loader.
  No optimizer/EMA state is inferred or synthesized from this weight export.
  """
  validate_tokenizer(tokenizer, mask_index)
  config_path, weights_path = _checkpoint_files(
    model_id, revision, local_path, local_files_only)
  with config_path.open(encoding='utf-8') as stream:
    architecture = _architecture(json.load(stream))
  from models.dit import DIT
  # Meta construction avoids a second 863M allocation and cannot consume CPU
  # or device random numbers. fork_rng also guards future constructor changes.
  with torch.random.fork_rng(devices=[]), torch.device('meta'):
    backbone = DIT(architecture, MDLM_VOCAB_SIZE)
  from safetensors.torch import load_file
  state = _strip_prefixes(load_file(str(weights_path), device='cpu'))
  expected = backbone.state_dict()
  missing, unexpected = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
  shapes = [(key, tuple(state[key].shape), tuple(expected[key].shape))
            for key in expected.keys() & state.keys() if state[key].shape != expected[key].shape]
  if missing or unexpected or shapes:
    raise ValueError('SDTT feature checkpoint is incompatible; no partial load: '
                     f'missing={missing[:8]}, unexpected={unexpected[:8]}, '
                     f'wrong_shapes={shapes[:8]}')
  if any(not tensor.is_floating_point() or tensor.dtype != torch.float32 for tensor in state.values()):
    raise ValueError('SDTT teacher export must contain float32 model tensors; '
                     'weights are not silently cast')
  backbone.load_state_dict(state, strict=True, assign=True)
  wrapper = SdttFeatureBackbone(backbone, feature_layer=feature_layer)
  wrapper.to(device)
  wrapper.checkpoint_info = {
    'model_id': model_id, 'revision': revision,
    'local_path': str(local_path) if local_path is not None else None,
    'parameters': sum(p.numel() for p in backbone.parameters()),
    'hidden_size': architecture['model']['hidden_size'],
    'blocks': architecture['model']['n_blocks'],
    'last_feature_layer': wrapper.last_feature_layer,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'hub_commit': (config_path.parent.name if config_path.parent.parent.name == 'snapshots'
                   and re.fullmatch(r'[0-9a-f]{40}', config_path.parent.name) else None),
  }
  print('[mmd-sdtt] ' + json.dumps(wrapper.checkpoint_info, sort_keys=True), flush=True)
  return wrapper
