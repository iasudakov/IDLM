"""Frozen SDLLM MDLM features on the teacher's native token sequence.

The published Lightning checkpoint is a native pickle artifact. Loading it uses
``torch.load(weights_only=False)`` and therefore requires trusting the supplied
checkpoint; the default Hub revision is immutable. No downloaded Python code is
imported. The audited release DiT has the same parameters and block equations as
this repository's DIT. Our SDPA attention backend can differ numerically from the
release's FlexAttention backend, while avoiding a PyTorch upgrade.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_MODEL = 'nvidia/SDLLM-MDLM-1.7B-Base'
DEFAULT_REVISION = '4007e0b2a23465a31adf282d843a61172932af2a'
SPECIAL_SLOTS = 1000
SMOLLM_BASE_VOCAB_SIZE = 49152
SMOLLM_BASE_VOCAB_SHA256 = '7fc3b488e0654b519ba97fb5d30821d8172edb8c67ad24e2afc62de18d89dbc5'
# The exact PATTERN_TIKTOKEN_V2 used by the released CustomTikTokenizer.
PATTERN_TIKTOKEN_V2 = (
  r'[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+'
  r'|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*'
  r'|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+')


def validate_source_tokenizer(tokenizer, mask_index):
  """Identify the original OWT/GPT-2 or TinyGSM/SmolLM source vocabulary.

  TinyGSM's wrapper includes the added PAD in ``vocab_size``; the diffusion
  MASK is the next ID and is not a text token. Neither feature route accepts
  another same-sized but differently indexed vocabulary.
  """
  if tokenizer.vocab_size == 50257:
    from mmd_sdtt_features import validate_tokenizer
    validate_tokenizer(tokenizer, mask_index)
    return 'gpt2'
  if (tokenizer.vocab_size != SMOLLM_BASE_VOCAB_SIZE + 1
      or mask_index != SMOLLM_BASE_VOCAB_SIZE + 1):
    raise ValueError('SDLLM source tokenizer must be canonical OWT/GPT-2 or '
                     'TinyGSM/SmolLM with wrapped vocab49153 and mask49153')
  vocab = tokenizer.get_vocab()
  base = {token: index for token, index in vocab.items()
          if index < SMOLLM_BASE_VOCAB_SIZE}
  digest = hashlib.sha256(json.dumps(
    base, sort_keys=True, ensure_ascii=False,
    separators=(',', ':')).encode()).hexdigest()
  if len(base) != SMOLLM_BASE_VOCAB_SIZE or digest != SMOLLM_BASE_VOCAB_SHA256:
    raise ValueError('SDLLM TinyGSM features require the original '
                     'SmolLM-135M token-to-ID mapping')
  extras = {token: index for token, index in vocab.items()
            if index >= SMOLLM_BASE_VOCAB_SIZE}
  if (extras != {'[PAD]': SMOLLM_BASE_VOCAB_SIZE}
      or tokenizer.pad_token_id != SMOLLM_BASE_VOCAB_SIZE):
    raise ValueError('SDLLM TinyGSM features require the added [PAD] at49152')
  if tokenizer.bos_token_id != 0 or tokenizer.eos_token_id != 0:
    raise ValueError('SDLLM TinyGSM features require SmolLM BOS/EOS ID0')
  return 'tinygsm'


def _checkpoint_files(model_id, revision, local_path, local_files_only):
  names = ('config.yaml', 'tokenizer/tiktokenizer.json', 'model.ckpt')
  if local_path is not None:
    root = Path(local_path).expanduser()
    files = tuple(root / name for name in names)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
      raise FileNotFoundError('Incomplete local SDLLM feature checkpoint; no Hub '
                              f'fallback will be attempted. Missing: {missing}')
    return files, None
  if not model_id or not revision:
    raise ValueError('SDLLM features need an explicit model ID and revision')
  from huggingface_hub import hf_hub_download
  print(f'[mmd-sdllm] Resolving {model_id}@{revision} in the Hugging Face cache'
        + (' (offline)' if local_files_only else '; missing files will be downloaded'),
        flush=True)
  kwargs = dict(repo_id=model_id, revision=revision,
                local_files_only=bool(local_files_only))
  config = Path(hf_hub_download(filename=names[0], **kwargs))
  commit = None
  if (config.parent.parent.name == 'snapshots'
      and re.fullmatch(r'[0-9a-f]{40}', config.parent.name)):
    commit = config.parent.name
    kwargs['revision'] = commit
  tokenizer = Path(hf_hub_download(filename=names[1], **kwargs))
  weights = Path(hf_hub_download(filename=names[2], **kwargs))
  return (config, tokenizer, weights), commit


def _architecture(config):
  if not isinstance(config, dict):
    raise ValueError('SDLLM config.yaml must contain a mapping')
  algo, model = config.get('algo', {}), config.get('model', {})
  required = {'name': 'mdlm', 'causal_attention': False,
              'adaLN': True, 'time_conditioning': False}
  if (not isinstance(algo, dict) or algo.get('name') != 'mdlm'
      or any(algo.get(key) is not value for key, value in required.items()
             if key != 'name')):
    raise ValueError('Expected a bidirectional SDLLM MDLM with adaLN=True and '
                     'time_conditioning=False')
  if algo.get('backbone') not in ('dit', 'dit_legacy') or model.get('type') != 'ddit':
    raise ValueError('SDLLM features support the released dit/dit_legacy ddit backbone')
  if model.get('tie_word_embeddings', False) is not False:
    raise ValueError('The released SDLLM backbone has untied vocabulary embeddings')
  result = {}
  for name in ('hidden_size', 'n_blocks', 'n_heads', 'cond_dim', 'length'):
    value = model.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
      raise ValueError(f'SDLLM model.{name} must be a positive integer')
    result[name] = value
  if result['hidden_size'] % result['n_heads'] or (result['hidden_size'] // result['n_heads']) % 2:
    raise ValueError('SDLLM head dimension must be an even integer')
  dropout = model.get('dropout', 0)
  if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
    raise ValueError('SDLLM model.dropout must be in [0,1)')
  result['dropout'] = float(dropout)
  result['scale_by_sigma'] = bool(model.get('scale_by_sigma', True))
  return {'model': result, 'algo': {'causal_attention': False}}


class _UnusedCheckpointTokenizer:
  """Discard the release's pickled tokenizer; load the real one from JSON."""

  def __setstate__(self, state):
    pass


class _SdllmCheckpointUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    # These two verified globals belong to hyper_parameters.tokenizer in the
    # native release, not state_dict. Do not import the release's dataloader or
    # rebuild its nested BPE engine; the real feature tokenizer comes from JSON.
    if (module, name) in {('dataloader', 'CustomTikTokenizer'),
                          ('tiktoken.core', 'Encoding')}:
      return _UnusedCheckpointTokenizer
    return super().find_class(module, name)


class _SdllmCheckpointPickle:
  Unpickler = _SdllmCheckpointUnpickler
  load = pickle.load


def _native_state(weights_path):
  # mmap keeps the much larger optimizer state from occupying resident CPU RAM.
  # The standard native Lightning format is trusted explicitly, as in sdllm.py.
  checkpoint = torch.load(str(weights_path), map_location='cpu', mmap=True,
                          weights_only=False, pickle_module=_SdllmCheckpointPickle)
  if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
    raise ValueError('SDLLM model.ckpt must contain a native Lightning state_dict')
  raw = checkpoint['state_dict']
  if not all(isinstance(key, str) for key in raw):
    raise ValueError('SDLLM state_dict keys must be strings')
  state = {key[len('backbone.'):]: value for key, value in raw.items()
           if key.startswith('backbone.')}
  if not state:
    raise ValueError('SDLLM state_dict has no backbone.* model tensors')
  if all(key.startswith('_orig_mod.') for key in state):
    state = {key[len('_orig_mod.'):]: value for key, value in state.items()}
  if not all(isinstance(value, torch.Tensor) for value in state.values()):
    raise ValueError('SDLLM backbone state must contain tensors only')
  # EMA and optimizer entries are intentionally not substituted: released
  # config.yaml has training.ema=-1, eval.disable_ema=True.
  return state


def _native_tokenizer(path, vocab_size):
  try:
    import tiktoken
  except ImportError as error:
    raise ImportError('SDLLM features require tiktoken>=0.9; install '
                      '`pip install "tiktoken>=0.9"`') from error
  if not isinstance(vocab_size, int) or vocab_size < SPECIAL_SLOTS + 256:
    raise ValueError('SDLLM vocabulary must include 1000 special slots and all 256 bytes')
  with Path(path).open(encoding='utf-8') as stream:
    records = json.load(stream)
  ordinary_count = vocab_size - SPECIAL_SLOTS
  if not isinstance(records, list) or len(records) < ordinary_count:
    raise ValueError('SDLLM tokenizer has too few ordinary ranks for the checkpoint vocabulary')
  ranks = {}
  for i, record in enumerate(records[:ordinary_count]):
    if (not isinstance(record, dict)
        or set(record) != {'rank', 'token_bytes', 'token_str'}
        or type(record['rank']) is not int or record['rank'] != i):
      raise ValueError(f'Invalid SDLLM tokenizer rank record {i}')
    try:
      token = base64.b64decode(record['token_bytes'], validate=True)
    except (ValueError, TypeError) as error:
      raise ValueError(f'Invalid base64 bytes for SDLLM rank {i}') from error
    if not token or (i < 256 and token != bytes([i])) or token in ranks:
      raise ValueError(f'Invalid/duplicate byte token at SDLLM rank {i}')
    ranks[token] = i + SPECIAL_SLOTS
  specials = {token: i for i, token in enumerate(
    ['<unk>', '<s>', '</s>', '<mask>']
    + [f'<SPECIAL_{i}>' for i in range(4, SPECIAL_SLOTS)])}
  encoding = tiktoken.Encoding(name='sdllm-native', pat_str=PATTERN_TIKTOKEN_V2,
                               mergeable_ranks=ranks, special_tokens=specials)
  if encoding.n_vocab != vocab_size:
    raise ValueError('SDLLM tokenizer IDs do not cover the checkpoint vocabulary exactly')
  return encoding


class FrozenSdllmModel(nn.Module):
  """A frozen DiT plus byte-BPE tokenizer, without source-token projection."""
  bos_token_id, eos_token_id, mask_token_id = 1, 2, 3

  def __init__(self, backbone, native_tokenizer, *, source_tokenizer=None,
               source_mask_index=None, feature_layer=None, source_family=None):
    super().__init__()
    self.backbone = backbone
    self.config = backbone.config
    self.vocab_size = backbone.vocab_size
    self.native_tokenizer = self.tokenizer = self.tiktoken = native_tokenizer
    # This object has already been validated before construction.
    self._source_tokenizer = source_tokenizer
    self._source_mask_index = source_mask_index
    self.source_family = (source_family if source_family is not None else
                          validate_source_tokenizer(source_tokenizer, source_mask_index)
                          if source_tokenizer is not None else None)
    from mmd_reward import resolve_feature_layers
    _, indices = resolve_feature_layers(self, feature_layer)
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
    return super().train(False)

  def _source_family(self, source_tokenizer, source_mask_index):
    if (source_tokenizer is self._source_tokenizer
        and source_mask_index == self._source_mask_index
        and self.source_family is not None):
      return self.source_family
    return validate_source_tokenizer(source_tokenizer, source_mask_index)

  def retokenize(self, tokens, source_tokenizer, source_mask_index):
    """Decode clean GPT-2 segments, then tokenize with the released byte BPE.

    GPT-2 BOS/EOS IDs delimit segments; their literal spelling inside ordinary
    text is not special. GPT-2 decoding retains its normal invalid-UTF8
    replacement semantics. No teacher feature is aligned back to GPT-2 tokens.
    """
    if self._source_family(source_tokenizer, source_mask_index) != 'gpt2':
      raise ValueError('TinyGSM SDLLM features require retokenize_conditioned '
                       'with the reference answer mask to remove padding')
    if isinstance(tokens, torch.Tensor):
      if tokens.ndim != 2 or tokens.dtype not in (
          torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('SDLLM retokenize expects an integer (batch,length) tensor')
      rows = tokens.detach().cpu().tolist()
    else:
      rows = list(tokens)
    result = []
    boundaries = {source_tokenizer.bos_token_id, source_tokenizer.eos_token_id}
    for row_index, row in enumerate(rows):
      row = list(row)
      if any(type(token) is not int for token in row):
        raise ValueError('SDLLM source token IDs must be integers')
      if source_mask_index in row:
        raise ValueError('SDLLM native features require mmd_noise_level=0 and clean '
                         'source sequences; a source MASK/PAD token was encountered')
      if any(token < 0 or token >= source_tokenizer.vocab_size for token in row):
        raise ValueError('SDLLM source token IDs are outside the GPT-2 vocabulary')
      converted, segment = [], []

      def flush_segment():
        if segment:
          text = source_tokenizer.decode(
            segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
          converted.extend(self.native_tokenizer.encode_ordinary(text))
          segment.clear()

      for position, token in enumerate(row):
        if token in boundaries:
          flush_segment()
          converted.append(self.bos_token_id if position == 0 else self.eos_token_id)
        else:
          segment.append(token)
      flush_segment()
      if not converted:
        raise ValueError(f'SDLLM retokenization produced an empty sequence at row {row_index}')
      if len(converted) > self.config.model.length:
        raise ValueError(f'SDLLM sequence {row_index} has {len(converted)} native '
                         f'tokens, exceeding context {self.config.model.length}; '
                         'the reward never silently truncates text')
      result.append(converted)
    return result

  def retokenize_conditioned(self, tokens, source_tokenizer, source_mask_index,
                            selected):
    """Retokenize TinyGSM prompt and answer with an explicit native boundary.

    ``selected`` is the contiguous reference-answer window, including its EOS.
    Its start/end are reused for every generated draw. The teacher sees the
    prompt plus this complete answer window; all suffix padding is discarded.
    Generated EOS tokens inside the window do not truncate it. Prefix and answer
    are tokenized separately, so their native boundary needs no byte alignment.
    Return native token rows and the first native answer index for each row.
    """
    if self._source_family(source_tokenizer, source_mask_index) != 'tinygsm':
      raise ValueError('Conditional SDLLM retokenization requires TinyGSM/SmolLM')
    if isinstance(tokens, torch.Tensor):
      if tokens.ndim != 2 or tokens.dtype not in (
          torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('SDLLM retokenize expects an integer (batch,length) tensor')
      rows = tokens.detach().cpu().tolist()
    else:
      rows = [list(row) for row in tokens]
    if isinstance(selected, torch.Tensor):
      if selected.ndim != 2 or selected.dtype != torch.bool:
        raise ValueError('TinyGSM SDLLM answer mask must be a boolean (batch,length) tensor')
      masks = selected.detach().cpu().tolist()
    else:
      masks = [list(row) for row in selected]
    if len(rows) != len(masks) or any(len(row) != len(mask)
                                    for row, mask in zip(rows, masks)):
      raise ValueError('TinyGSM SDLLM answer mask must match the source token shape')

    boundaries = {source_tokenizer.bos_token_id, source_tokenizer.eos_token_id}

    def convert_segment(ids, *, prompt):
      converted, segment = [], []

      def flush():
        if segment:
          text = source_tokenizer.decode(
            segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
          converted.extend(self.native_tokenizer.encode_ordinary(text))
          segment.clear()

      for position, token in enumerate(ids):
        if token in boundaries:
          flush()
          converted.append(self.bos_token_id if prompt and position == 0
                           else self.eos_token_id)
        else:
          # Added PAD sampled inside the answer is literal model output.
          # Only suffix padding outside the reference window is discarded.
          segment.append(token)
      flush()
      return converted

    # TinyGSM's vocab_size delegates to len(HF tokenizer), whose added-token
    # accounting can be expensive. Read it once per call, not for every token.
    # Refresh on the next call so this does not introduce a persistent cache.
    source_vocab_size = source_tokenizer.vocab_size
    native_rows, answer_offsets = [], []
    for row_index, (row, mask) in enumerate(zip(rows, masks)):
      if any(type(flag) is not bool for flag in mask):
        raise ValueError('TinyGSM SDLLM answer mask must contain booleans')
      positions = [position for position, flag in enumerate(mask) if flag]
      if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError('TinyGSM SDLLM needs one nonempty contiguous reference-answer window')
      start, stop = positions[0], positions[-1] + 1
      # Inspect only prompt+answer: ignored suffix slots may contain PAD or MASK.
      active = row[:stop]
      if any(type(token) is not int for token in active):
        raise ValueError('SDLLM source token IDs must be integers')
      if source_mask_index in active:
        raise ValueError('TinyGSM SDLLM features require clean source prompt and '
                         'answer tokens; a source MASK was encountered')
      if any(token < 0 or token >= source_vocab_size for token in active):
        raise ValueError('SDLLM source token IDs are outside the TinyGSM vocabulary')
      prompt_ids = convert_segment(active[:start], prompt=True)
      answer_ids = convert_segment(active[start:], prompt=False)
      if not answer_ids:
        raise ValueError(f'TinyGSM SDLLM answer {row_index} retokenized to an empty sequence')
      combined = prompt_ids + answer_ids
      if len(combined) > self.config.model.length:
        raise ValueError(f'SDLLM sequence {row_index} has {len(combined)} native '
                         f'tokens, exceeding context {self.config.model.length}; '
                         'the reward never silently truncates text')
      native_rows.append(combined)
      answer_offsets.append(len(prompt_ids))
    return native_rows, answer_offsets

  @torch.no_grad()
  def features(self, token_lists, layers, forward_batch_size=None):
    """Return all native positions, batching only sequences of equal length."""
    from mmd_reward import resolve_feature_layers
    _, indices = resolve_feature_layers(self, layers)
    if forward_batch_size is not None and (
        type(forward_batch_size) is not int or forward_batch_size <= 0):
      raise ValueError('SDLLM forward_batch_size must be a positive integer or None')
    rows = [list(row) for row in token_lists]
    output = {index: [None] * len(rows) for index in indices}
    buckets = defaultdict(list)
    for position, row in enumerate(rows):
      if not row or len(row) > self.config.model.length:
        raise ValueError('SDLLM native sequence length must be in '
                         f'[1,{self.config.model.length}]')
      if any(type(token) is not int or token < 0 or token >= self.vocab_size for token in row):
        raise ValueError('SDLLM native token IDs are outside the teacher vocabulary')
      if self.mask_token_id in row:
        raise ValueError('SDLLM native MMD features expect clean sequences without MASK tokens')
      buckets[len(row)].append(position)
    self.backbone.eval()
    device = self.backbone.vocab_embed.embedding.device
    deepest, requested = max(indices), set(indices)
    for positions in buckets.values():
      chunk_size = forward_batch_size or len(positions)
      for start in range(0, len(positions), chunk_size):
        chunk = positions[start:start + chunk_size]
        tokens = torch.tensor([rows[i] for i in chunk], dtype=torch.long, device=device)
        x = self.backbone.vocab_embed(tokens)

        def save(layer):
          # Each entry retains its entire native sequence and GPU device.
          for source, values in zip(chunk, x.unbind(0)):
            output[layer][source] = values.detach()

        if -1 in requested:
          save(-1)
        if deepest < 0:
          continue
        sigma = torch.zeros(len(chunk), dtype=torch.float32, device=device)
        conditioning = F.silu(self.backbone.sigma_map(sigma))
        rotary = self.backbone.rotary_emb(x)
        with torch.cuda.amp.autocast(enabled=tokens.is_cuda, dtype=torch.bfloat16):
          for layer, block in enumerate(self.backbone.blocks[:deepest + 1]):
            x = block(x, rotary, c=conditioning)
            if layer in requested:
              save(layer)
    return output


def load_sdllm_feature_model(model_id=DEFAULT_MODEL, device='cpu', *,
                             tokenizer, mask_index, revision=DEFAULT_REVISION,
                             local_path=None, local_files_only=False,
                             feature_layer=None):
  """Load the pinned native release, using raw backbone weights rather than EMA.

  ``local_path`` must contain config.yaml, tokenizer/tiktokenizer.json and
  model.ckpt. An incomplete explicit directory never falls back to the Hub.
  Standard Lightning pickle loading is enabled only for this explicit feature
  source; use trusted checkpoints. No automatic remote Python import is used.
  """
  source_family = validate_source_tokenizer(tokenizer, mask_index)
  paths, commit = _checkpoint_files(model_id, revision, local_path, local_files_only)
  config_path, tokenizer_path, weights_path = paths
  import yaml
  with config_path.open(encoding='utf-8') as stream:
    architecture = _architecture(yaml.safe_load(stream))
  state = _native_state(weights_path)
  embedding = state.get('vocab_embed.embedding')
  if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
    raise ValueError('SDLLM checkpoint lacks a two-dimensional vocabulary embedding')
  vocab_size = embedding.shape[0]
  if embedding.shape[1] != architecture['model']['hidden_size']:
    raise ValueError('SDLLM embedding dimension disagrees with config.yaml')
  native_tokenizer = _native_tokenizer(tokenizer_path, vocab_size)
  from models.dit import DIT
  with torch.random.fork_rng(devices=[]), torch.device('meta'):
    backbone = DIT(architecture, vocab_size)
  expected = backbone.state_dict()
  missing = sorted(expected.keys() - state.keys())
  unexpected = sorted(state.keys() - expected.keys())
  shapes = [(key, tuple(state[key].shape), tuple(expected[key].shape))
            for key in expected.keys() & state.keys() if state[key].shape != expected[key].shape]
  if missing or unexpected or shapes:
    raise ValueError('SDLLM feature checkpoint is incompatible; no partial load: '
                     f'missing={missing[:8]}, unexpected={unexpected[:8]}, '
                     f'wrong_shapes={shapes[:8]}')
  if any(not tensor.is_floating_point() or tensor.dtype != torch.float32 for tensor in state.values()):
    raise ValueError('SDLLM native backbone tensors must be float32; '
                     'checkpoint weights are not silently cast')
  backbone.load_state_dict(state, strict=True, assign=True)
  wrapper = FrozenSdllmModel(backbone, native_tokenizer, source_tokenizer=tokenizer,
                            source_mask_index=mask_index, feature_layer=feature_layer,
                            source_family=source_family)
  wrapper.to(device)
  wrapper.checkpoint_info = {
    'model_id': model_id, 'revision': revision, 'hub_commit': commit,
    'local_path': str(local_path) if local_path is not None else None,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    'parameters': sum(p.numel() for p in backbone.parameters()),
    'hidden_size': architecture['model']['hidden_size'],
    'blocks': architecture['model']['n_blocks'],
    'context': architecture['model']['length'], 'vocab_size': vocab_size,
    'last_feature_layer': wrapper.last_feature_layer,
    'weights': 'native state_dict.backbone; EMA disabled',
    'attention_backend': 'local torch SDPA (release: FlexAttention)',
    'source_tokenizer': source_family,
    'token_positions': 'native SDLLM positions; no source-token projection',
  }
  print('[mmd-sdllm] ' + json.dumps(wrapper.checkpoint_info, sort_keys=True), flush=True)
  return wrapper
