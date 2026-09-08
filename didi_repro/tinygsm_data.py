"""TinyGSM data adapter, matching the conditional MMD reference protocol.

``attention_mask`` is the *loss/generation* mask here, not a transformer
attention mask: zero on the prompt and one on the answer (and optionally
padding). Training only reads an existing, separator-specific cache. Dataset
preparation is a separate, explicit operation and never deletes a cache.
"""

import json
import os
from pathlib import Path

import torch


class VocabSizeTokenizerWrapper:
  """Include added PAD in vocab_size before DiDi appends its MASK token."""

  def __init__(self, tokenizer):
    object.__setattr__(self, '_wrapped_tokenizer', tokenizer)

  @property
  def vocab_size(self):
    return len(self._wrapped_tokenizer)

  def __len__(self):
    return len(self._wrapped_tokenizer)

  def __call__(self, *args, **kwargs):
    return self._wrapped_tokenizer(*args, **kwargs)

  def __getattr__(self, name):
    if name == '_wrapped_tokenizer':
      raise AttributeError(name)
    return getattr(self._wrapped_tokenizer, name)

  def __setattr__(self, name, value):
    if name == '_wrapped_tokenizer':
      object.__setattr__(self, name, value)
    else:
      setattr(self._wrapped_tokenizer, name, value)


def get_tokenizer(config):
  """Use the reference SmolLM vocabulary, including its distinct PAD/MASK."""
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(
    config.data.tokenizer_name_or_path,
    local_files_only=bool(config.data.get('tokenizer_local_files_only', False)),
    cache_dir=config.data.get('tokenizer_cache_dir', None))
  if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.cls_token or tokenizer.eos_token
  if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.bos_token is None or tokenizer.eos_token is None:
    raise ValueError('TinyGSM tokenizer requires BOS and EOS tokens.')
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  if getattr(tokenizer, 'mask_token_id', None) in {
      tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}:
    tokenizer.mask_token = None
  if tokenizer.pad_token_id in {tokenizer.bos_token_id, tokenizer.eos_token_id}:
    raise ValueError('TinyGSM requires a distinct padding token.')
  return VocabSizeTokenizerWrapper(tokenizer)


def separator_tag(separator):
  return 'sep' + ''.join(f'{value:02x}' for value in separator.encode('utf-8'))


def _validate_config(config):
  data = config.data
  if data.get('train', 'tiny_gsm') != 'tiny_gsm':
    raise ValueError('The TinyGSM adapter requires data.train=tiny_gsm.')
  if data.get('wrap', False) or data.get('train_on_prompt', False):
    raise ValueError('Conditional DiDi requires data.wrap=false and '
                     'data.train_on_prompt=false.')
  if data.get('streaming', False):
    raise ValueError('TinyGSM uses a precomputed, non-streaming dataset.')
  if not isinstance(data.separator, str) or not data.separator:
    raise ValueError('TinyGSM requires an explicit nonempty question separator.')
  if int(config.model.length) < 2:
    raise ValueError('TinyGSM model.length must leave space for prompt and answer.')
  for name in ('train_ratio', 'valid_ratio'):
    if float(data.get(name, 1.0)) != 1.0:
      raise ValueError(f'TinyGSM does not silently apply data.{name}; prepare '
                       'an explicit dataset subset instead.')


def tinygsm_cache_path(config):
  """Use the exact separator-aware cache name from tinygsm-upstream-repro."""
  _validate_config(config)
  data = config.data
  filtered = '_filtered' if data.get('filter_too_long', True) else ''
  padding = '_train_on_pad' if data.get('train_on_pad', True) else ''
  tokenizer_tag = str(data.tokenizer_name_or_path).replace('/', '__')
  name = (f'tiny_gsm_bs{int(config.model.length)}_answer_only{filtered}{padding}'
          f'_{tokenizer_tag}_{separator_tag(data.separator)}')
  return Path(data.cache_dir) / name


def _cache_metadata(config, tokenizer=None):
  result = {
    'format': 'tinygsm-upstream-repro/answer-only-v1',
    'length': int(config.model.length),
    'separator': config.data.separator,
    'tokenizer': str(config.data.tokenizer_name_or_path),
    'train_on_pad': bool(config.data.get('train_on_pad', True)),
    'filter_too_long': bool(config.data.get('filter_too_long', True)),
    'val_ratio': float(config.data.get('val_ratio', 0.01)),
    'val_seed': int(config.data.get('val_seed', 42)),
    # Preparation source identity is recorded without re-reading a potentially
    # huge local JSON on every training rank. Training always reads the cache;
    # changing data_path does not retokenize or update an existing dataset.
    'source': ({'local_json': str(Path(config.data.data_path).resolve())}
               if config.data.get('data_path') else {'hub': 'TinyGSM/TinyGSM'}),
  }
  if tokenizer is not None:
    result.update(bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id,
                  pad_id=tokenizer.pad_token_id, tokenizer_size=len(tokenizer))
  return result


class ConditionalDataset(torch.utils.data.Dataset):
  """Validate masks on access without copying an eleven-million-row cache."""

  def __init__(self, dataset, config, tokenizer=None):
    self.dataset = dataset
    self.length = int(config.model.length)
    self.train_on_pad = bool(config.data.get('train_on_pad', True))
    self.pad_id = getattr(tokenizer, 'pad_token_id', None)
    self.vocab_size = len(tokenizer) if tokenizer is not None else None

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    row = self.dataset[int(index)]
    raw_ids = torch.as_tensor(row['input_ids'])
    raw_mask = torch.as_tensor(row['attention_mask'])
    if raw_ids.ndim != 1 or raw_mask.ndim != 1 \
        or raw_ids.numel() != self.length or raw_mask.numel() != self.length:
      raise ValueError(f'TinyGSM row {index}: input_ids and attention_mask must '
                       f'both have model.length={self.length} entries.')
    if not bool(((raw_mask == 0) | (raw_mask == 1)).all()):
      raise ValueError(f'TinyGSM row {index}: attention_mask must be binary.')
    ids = raw_ids.to(dtype=torch.long)
    if not bool((raw_ids == ids).all()) or bool((ids < 0).any()):
      raise ValueError(f'TinyGSM row {index}: input_ids must be nonnegative integers.')
    if self.vocab_size is not None and bool((ids >= self.vocab_size).any()):
      raise ValueError(f'TinyGSM row {index}: token ID exceeds tokenizer vocabulary; '
                       'check the tokenizer and teacher checkpoint.')
    mask = raw_mask.to(dtype=torch.long)
    positions = mask.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
      raise ValueError(f'TinyGSM row {index}: no answer positions; cannot derive '
                       'a conditional prompt from an all-zero loss mask.')
    first, last = int(positions[0]), int(positions[-1])
    if first <= 0 or int(positions.numel()) != last - first + 1:
      raise ValueError(f'TinyGSM row {index}: expected a nonempty prompt followed '
                       'by one contiguous answer span in attention_mask.')
    supplied = row.get('prompt_len', first)
    if torch.as_tensor(supplied).numel() != 1 or float(supplied) != first:
      raise ValueError(f'TinyGSM row {index}: prompt_len disagrees with loss mask.')
    if self.train_on_pad and last != self.length - 1:
      raise ValueError(f'TinyGSM row {index}: data.train_on_pad=true requires '
                       'the loss mask to reach the end of the sequence.')
    if not self.train_on_pad and self.pad_id is not None \
        and bool((ids[last + 1:] != self.pad_id).any()):
      raise ValueError(f'TinyGSM row {index}: inactive suffix contains non-padding '
                       'tokens; its conditional generation semantics are ambiguous.')
    return {'input_ids': ids, 'attention_mask': mask,
            'prompt_len': torch.tensor(first, dtype=torch.long)}


def _load_cached(config, tokenizer=None):
  import datasets
  path = tinygsm_cache_path(config)
  if not path.is_dir():
    raise FileNotFoundError(f'Missing TinyGSM cache: {path}. Prepare it once '
                            'with didi_repro.tinygsm_data.prepare_cache or the '
                            'TinyGSM reference preparation script. Training never '
                            'downloads, deletes, or regenerates dataset caches.')
  metadata_path = path / 'didi_tinygsm_metadata.json'
  if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text())
    for key, expected in _cache_metadata(config, tokenizer).items():
      if key == 'source' and config.data.get('data_path') is None:
        # A transferred prepared cache is sufficient for training; its source
        # JSON need not exist at the original machine's absolute path.
        continue
      if metadata.get(key) != expected:
        raise ValueError(f'TinyGSM cache metadata mismatch for {key}: '
                         f'{metadata.get(key)!r} != {expected!r}. Cache preserved.')
  else:
    if (float(config.data.get('val_ratio', 0.01)) != 0.01
        or int(config.data.get('val_seed', 42)) != 42):
      raise ValueError('A legacy TinyGSM cache has no split metadata and only '
                       'supports the reference data.val_ratio=0.01 and '
                       'data.val_seed=42. Prepare a fresh cache in another '
                       'data.cache_dir for a different split; cache preserved.')
    if config.data.get('data_path') is not None:
      raise ValueError('A legacy TinyGSM cache cannot verify data.data_path. '
                       'Training reads the resolved cache, not that JSON. '
                       'Clear data.data_path to use the reference cache, or '
                       'prepare a fresh cache in another data.cache_dir from '
                       'the requested source; cache preserved.')
  result = datasets.load_from_disk(str(path))
  for split in ('train', 'validation'):
    if split not in result or len(result[split]) == 0:
      raise ValueError(f'TinyGSM cache requires a nonempty {split} split: {path}')
    if not {'input_ids', 'attention_mask'}.issubset(result[split].column_names):
      raise ValueError(f'TinyGSM cache {split} lacks input_ids/attention_mask: {path}')
    # Legacy reference caches omit prompt_len. Per-row validation derives it
    # only when the answer-only loss mask unambiguously specifies the boundary.
    ConditionalDataset(result[split], config, tokenizer)[0]
  return result


def check_cached_data(config):
  """Read-only preflight, compatible with the OWT entrypoint's check hook."""
  _load_cached(config)


def configured_world_size(config):
  """Count only configured trainer devices, never all visible physical GPUs."""
  devices = config.trainer.get('devices', 1)
  if isinstance(devices, int):
    count = devices
  elif isinstance(devices, str):
    if devices in {'auto', '-1'}:
      count = int(os.environ.get('LOCAL_WORLD_SIZE', '0'))
    elif ',' in devices:
      count = len([value for value in devices.split(',') if value.strip()])
    else:
      count = int(devices)
  else:
    count = len(devices)
  if count <= 0:
    raise ValueError('Set trainer.devices explicitly for TinyGSM batch-size '
                     'validation (or launch under torchrun with LOCAL_WORLD_SIZE).')
  return count * int(config.trainer.get('num_nodes', 1))


def get_dataloaders(config, tokenizer, skip_train=False, skip_valid=False,
                    valid_seed=None):
  """Return CPU loaders; Lightning supplies distributed sampling as usual."""
  world = configured_world_size(config)
  if not skip_train:
    expected = (int(config.loader.batch_size) * world
                * int(config.trainer.get('accumulate_grad_batches', 1)))
    if int(config.loader.global_batch_size) != expected:
      raise ValueError(f'loader.global_batch_size must equal batch_size * '
                       f'configured world size * accumulation = {expected}.')
  if not skip_valid and config.loader.get('eval_global_batch_size') is not None:
    expected = int(config.loader.eval_batch_size) * world
    if int(config.loader.eval_global_batch_size) != expected:
      raise ValueError(f'loader.eval_global_batch_size must equal eval_batch_size '
                       f'* configured world size = {expected}.')
  dataset = _load_cached(config, tokenizer)
  workers = int(config.loader.get('num_workers', 0))
  if workers < 0:
    raise ValueError('loader.num_workers must be nonnegative.')
  common = {'num_workers': workers,
            'pin_memory': bool(config.loader.get('pin_memory', True))}
  train_loader = valid_loader = None
  if not skip_train:
    train_loader = torch.utils.data.DataLoader(
      ConditionalDataset(dataset['train'], config, tokenizer),
      batch_size=int(config.loader.batch_size), shuffle=True,
      persistent_workers=workers > 0, **common)
    train_loader.tokenizer = tokenizer
  if not skip_valid:
    generator = (None if valid_seed is None else
                 torch.Generator().manual_seed(int(valid_seed)))
    valid_loader = torch.utils.data.DataLoader(
      ConditionalDataset(dataset['validation'], config, tokenizer),
      batch_size=int(config.loader.eval_batch_size),
      shuffle=valid_seed is not None, generator=generator, **common)
    valid_loader.tokenizer = tokenizer
  return train_loader, valid_loader


def get_gsm8k_test_dataset(config, tokenizer):
  """Read the reference cache or tokenize an explicitly provided local JSON.

  Unlike training preparation, the small benchmark may be tokenized in memory
  on each rank. This never races to write a shared cache or downloads a dataset.
  Rows preserve ``response_ground_truth`` for the S-FLM execution grader.
  """
  import datasets
  _validate_config(config)
  tokenizer_tag = str(config.data.tokenizer_name_or_path).replace('/', '__')
  path = Path(config.data.cache_dir) / (
    f'gsm8k_test_{tokenizer_tag}_{separator_tag(config.data.separator)}_with_text')
  if path.is_dir():
    dataset = datasets.load_from_disk(str(path))
  else:
    source = config.get('eval', {}).get('gsm8k_data_path') \
      or config.data.get('data_path')
    if not source or not Path(source).is_file():
      raise FileNotFoundError(f'Missing GSM8K cache {path}. Set '
                              'eval.gsm8k_data_path to the local GSM8K test JSON.')
    records = json.loads(Path(source).read_text())
    if not isinstance(records, list) or not records:
      raise ValueError('GSM8K JSON must contain a nonempty list of records.')
    separator = tokenizer(config.data.separator, add_special_tokens=False).input_ids
    encoded = []
    for row in records:
      if not isinstance(row.get('prompt'), str) \
          or not isinstance(row.get('response_ground_truth'), str):
        raise ValueError('GSM8K records require prompt and response_ground_truth strings.')
      prompt = [tokenizer.bos_token_id] + tokenizer(
        row['prompt'].strip(), add_special_tokens=False).input_ids + separator
      encoded.append({**row, 'input_ids': prompt, 'answer': tokenizer(
        row['response_ground_truth'].strip(), add_special_tokens=False).input_ids})
    dataset = datasets.Dataset.from_list(encoded)
  if len(dataset) == 0 or not {'input_ids', 'response_ground_truth'}.issubset(
      dataset.column_names):
    raise ValueError('GSM8K dataset must contain prompt input_ids and '
                     'response_ground_truth, not only answer tokens.')
  for index, row in enumerate(dataset):
    if not 0 < len(row['input_ids']) < int(config.model.length):
      raise ValueError(f'GSM8K prompt {index} leaves no answer space at '
                       f'model.length={config.model.length}.')
    if not isinstance(row['response_ground_truth'], str):
      raise ValueError(f'GSM8K reference answer {index} is not a string.')
  return dataset


def prepare_cache(config, tokenizer, *, allow_download=False):
  """Explicit single-process preparation using the reference encoding/split.

  Existing targets, including incomplete ones, are never overwritten. A local
  ``data.data_path`` JSON requires no network. TinyGSM/TinyGSM download is only
  allowed with the separate explicit ``allow_download=True`` argument.
  """
  import datasets
  target = tinygsm_cache_path(config)
  if target.exists():
    raise FileExistsError(f'Refusing to overwrite TinyGSM cache: {target}')
  source = config.data.get('data_path')
  if source:
    records = json.loads(Path(source).read_text())
    if not isinstance(records, list) or not records:
      raise ValueError('TinyGSM JSON must contain a nonempty list of records.')
    normalized = []
    for row in records:
      question = row.get('question') or row.get('prompt')
      answer = row.get('code') or row.get('response_ground_truth') or row.get('answer')
      if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError('TinyGSM local records require question/code or '
                         'prompt/response_ground_truth strings.')
      normalized.append({'question': question, 'code': answer})
    dataset = datasets.Dataset.from_list(normalized)
  elif allow_download:
    dataset = datasets.load_dataset('TinyGSM/TinyGSM', split='train',
                                    cache_dir=str(config.data.cache_dir))
  else:
    raise FileNotFoundError('No TinyGSM source selected. Set data.data_path to '
                            'a local JSON or explicitly allow_download=True.')
  separator = tokenizer(config.data.separator, add_special_tokens=False).input_ids
  length = int(config.model.length)
  workers = int(config.loader.get('num_workers', 0)) or None

  def tokenize(row):
    question = tokenizer(row['question'].strip(), add_special_tokens=False).input_ids
    answer = tokenizer(row['code'].strip(), add_special_tokens=False).input_ids
    prompt = [tokenizer.bos_token_id] + question + separator
    return {'input_ids': prompt + answer + [tokenizer.eos_token_id],
            'prompt_len': len(prompt)}

  tokenized = dataset.map(tokenize, remove_columns=dataset.column_names,
                          num_proc=workers, desc='Tokenizing TinyGSM')
  if config.data.get('filter_too_long', True):
    tokenized = tokenized.filter(lambda row: len(row['input_ids']) <= length,
                                 num_proc=workers, desc='Filtering TinyGSM')

  def pad(row):
    ids, prompt_length = row['input_ids'], row['prompt_len']
    size = len(ids)
    if prompt_length >= length:
      raise ValueError('TinyGSM prompt leaves no answer space; enable '
                       'data.filter_too_long=true or increase model.length.')
    ids = (ids[:length - 1] + [tokenizer.eos_token_id] if size >= length else
           ids + [tokenizer.pad_token_id] * (length - size))
    end = length if config.data.get('train_on_pad', True) else min(size, length)
    mask = [0] * prompt_length + [1] * (end - prompt_length) + [0] * (length - end)
    return {'input_ids': ids, 'attention_mask': mask}

  tokenized = tokenized.map(pad, num_proc=workers, desc='Padding TinyGSM')
  split = tokenized.train_test_split(
    test_size=float(config.data.get('val_ratio', 0.01)),
    seed=int(config.data.get('val_seed', 42)))
  result = datasets.DatasetDict({'train': split['train'], 'validation': split['test']})
  for partition in result.values():
    if not len(partition):
      raise ValueError('Prepared TinyGSM train/validation splits must both be nonempty.')
    ConditionalDataset(partition, config, tokenizer)[0]
  result.save_to_disk(str(target))
  (target / 'didi_tinygsm_metadata.json').write_text(
    json.dumps(_cache_metadata(config, tokenizer), indent=2) + '\n')
  return target
