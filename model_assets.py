"""Resolve optional model bundles without consulting the Hugging Face Hub.

Keep logical model/tokenizer names in the experiment configuration: changing
them can change vocabulary handling, dataset cache keys, or metric behavior.
Only the argument passed to a loader is replaced with a local path.
"""

import json
import os
from pathlib import Path


def _bundle_root(weights_dir):
  root = Path(weights_dir).expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError(f'model_weights_dir is not a directory: {root}')
  return root


def _check_pretrained(path, kind):
  if not path.is_dir():
    return 'model/tokenizer directory is missing'
  if kind == 'tokenizer':
    if any((path / name).is_file() for name in (
        'tokenizer.json', 'vocab.json', 'vocab.txt', 'tokenizer.model',
        'spiece.model', 'sentencepiece.bpe.model')):
      return None
    return 'tokenizer files are missing'
  if not (path / 'config.json').is_file():
    return 'config.json is missing'
  if any((path / name).is_file() for name in (
      'model.safetensors', 'pytorch_model.bin')):
    return None
  for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
    index_path = path / name
    if index_path.is_file():
      index = json.loads(index_path.read_text())
      shards = set(index.get('weight_map', {}).values())
      missing = sorted(shard for shard in shards if not (path / shard).is_file())
      if shards and not missing:
        return None
      return f'weight shards are missing: {missing or name}'
  return 'model weights are missing (this may be a tokenizer-only bundle)'


def _cache_candidates(root, repo_id, revision=None):
  for cache_root in (root / 'hf-cache', root):
    repo = cache_root / ('models--' + repo_id.replace('/', '--'))
    if revision:
      yield repo / 'snapshots' / revision
    else:
      ref = repo / 'refs' / 'main'
      if ref.is_file():
        yield repo / 'snapshots' / ref.read_text().strip()


def resolve_pretrained_path(name_or_path, weights_dir=None, *, kind='model'):
  """Resolve a directory, manifest entry, or bundled HF cache snapshot.

  Unset ``weights_dir`` preserves the existing online loader behavior. With a
  bundle, a missing asset raises here; the caller must not fall back to the Hub.
  Explicit existing paths take precedence over entries in the bundle.
  """
  if not weights_dir:
    return name_or_path
  if kind not in ('model', 'tokenizer'):
    raise ValueError(f'Unknown model asset kind: {kind}')
  root = _bundle_root(weights_dir)
  name = os.fspath(name_or_path) if name_or_path is not None else ''
  if not name:
    raise FileNotFoundError(
      f'No {kind} name/path configured for model_weights_dir={root}')

  explicit = Path(name).expanduser()
  if explicit.exists() or explicit.is_absolute():
    candidates = [explicit.resolve()]
  else:
    candidates = []
    manifest_path = root / 'manifest.json'
    if manifest_path.is_file():
      manifest = json.loads(manifest_path.read_text())
      for asset in manifest.get('assets', []):
        aliases = (asset.get('repo_id'), asset.get('canonical_repo_id'))
        if name not in aliases:
          continue
        if asset.get('folder'):
          candidates.append(root / asset['folder'])
        for repo_id in dict.fromkeys(alias for alias in aliases if alias):
          candidates.extend(_cache_candidates(root, repo_id, asset.get('revision')))
    candidates.extend([root / name, root / Path(name).name])
    candidates.extend(_cache_candidates(root, name))

  checked = []
  for path in dict.fromkeys(candidates):
    path = path.resolve()
    problem = _check_pretrained(path, kind)
    if problem is None:
      return str(path)
    checked.append(f'{path}: {problem}')
  raise FileNotFoundError(
    f'Local {kind} {name!r} is unavailable in model_weights_dir={root}. '
    'No model download will be attempted. Copy the required files into the '
    'bundle or specify an existing local path. Checked:\n  '
    + '\n  '.join(checked))


def pretrained_kwargs(name_or_path, weights_dir=None, *, kind='model'):
  """Arguments for Transformers ``from_pretrained`` with optional local-only IO."""
  kwargs = {'pretrained_model_name_or_path': resolve_pretrained_path(
    name_or_path, weights_dir, kind=kind)}
  if weights_dir:
    kwargs['local_files_only'] = True
  return kwargs


def resolve_checkpoint(path, weights_dir=None, *, base_dir=None, required=True):
  """Resolve native checkpoint paths, preferring explicit existing files.

  Relative paths are checked against ``base_dir`` (the launch directory) and
  then the bundle. Absolute paths are always treated as explicit overrides.
  Empty paths continue to mean no checkpoint, as in the existing config.
  ``required=False`` normalizes existing paths without rejecting unused fields.
  """
  if not weights_dir or not path:
    return path
  root = _bundle_root(weights_dir)
  requested = Path(path).expanduser()
  explicit = requested if requested.is_absolute() else Path(base_dir or os.getcwd()) / requested
  candidates = [explicit]
  if not requested.is_absolute():
    candidates.append(root / requested)
  for candidate in candidates:
    if candidate.is_file():
      return str(candidate.resolve())
  if not required:
    return path
  raise FileNotFoundError(
    f'Local checkpoint {str(path)!r} is missing. Checked: '
    + ', '.join(str(p) for p in candidates)
    + '. Supply an existing checkpoint path; no download will be attempted.')


def configure_model_weights(config, *, base_dir=None):
  """Normalize offline paths before Hydra launches any model or worker."""
  weights_dir = config.get('model_weights_dir')
  if not weights_dir:
    return
  base = Path(base_dir or os.getcwd())
  requested = Path(weights_dir).expanduser()
  config.model_weights_dir = str(_bundle_root(
    requested if requested.is_absolute() else base / requested))
  native_keys = [('training', 'finetune_path'), ('training', 'student_init_path'),
                 ('eval', 'student_ckpt_path')]
  if 'hf' not in config.algo.backbone:
    native_keys.append(('eval', 'checkpoint_path'))
  ar_path = config.adversarial_distill.get('mmd_ar_model', '')
  if str(ar_path).endswith('.ckpt'):
    native_keys.append(('adversarial_distill', 'mmd_ar_model'))
  for section, key in native_keys:
    path = config[section].get(key)
    if path:
      config[section][key] = resolve_checkpoint(
        path, config.model_weights_dir, base_dir=base, required=False)
  if config.eval.get('student_ckpt_paths'):
    config.eval.student_ckpt_paths = [
      resolve_checkpoint(path, config.model_weights_dir,
                         base_dir=base, required=False)
      for path in config.eval.student_ckpt_paths]
