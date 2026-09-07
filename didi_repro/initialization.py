"""Optional warm start from safetensors; never replace the separate MDLM teacher.

This is continued training of a released DiDi student, not reproduction from
the original teacher-only initialization. Only two weight files are fetched;
neither Hugging Face model code nor serialized optimizer state is executed.
"""

import hashlib
from pathlib import Path
import re


AUTHORS_STUDENT_REPO = 'haoyangzheng/didi-instruct-small'
AUTHORS_STUDENT_REVISION = '2e9a21a63cc727a1e703370d55de13b842c69076'
WEIGHT_FILES = ('model.safetensors', 'discriminator.safetensors')


def _snapshot_revision(path):
  """Read the immutable revision from a standard HF cache snapshot path."""
  path = Path(path).absolute()
  for parent in (path, *path.parents):
    if parent.parent.name == 'snapshots' and re.fullmatch(r'[0-9a-f]{40}', parent.name):
      return parent.name
  return None


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _resolve_files(options):
  source = str(options['student_hf'])
  requested = options.get('revision')
  local_path = Path(source).expanduser()
  if local_path.is_dir():
    # Do not resolve individual symlinks: HF weights point out of snapshots/
    # into blobs/, whereas their containing directory records the revision.
    directory = local_path.absolute()
    files = {name: directory / name for name in WEIGHT_FILES}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
      raise FileNotFoundError(f'Incomplete local DiDi snapshot: {missing}')
    revision = _snapshot_revision(directory)
    if requested is not None and revision is not None and str(requested) != revision:
      raise ValueError('Local DiDi snapshot revision does not match didi_init.revision')
    return files, {'source': str(directory), 'repo_id': None,
                   'requested_revision': requested, 'revision': revision,
                   'local_files_only': True}
  if local_path.exists() or source.startswith(('/', './', '../', '~')):
    raise FileNotFoundError(f'DiDi student_hf must be a snapshot directory: {source}')
  if requested is None:
    if source != AUTHORS_STUDENT_REPO:
      raise ValueError('A non-author HF repository requires explicit didi_init.revision')
    requested = AUTHORS_STUDENT_REVISION
  requested = str(requested)
  if not requested:
    raise ValueError('didi_init.revision must not be empty')

  from huggingface_hub import hf_hub_download
  kwargs = {'repo_id': source, 'cache_dir': options.get('cache_dir'),
            'local_files_only': bool(options.get('local_files_only', False))}
  generator = Path(hf_hub_download(filename=WEIGHT_FILES[0], revision=requested, **kwargs))
  revision = _snapshot_revision(generator)
  if revision is None:
    raise RuntimeError('HF download did not resolve to an immutable snapshot revision')
  if re.fullmatch(r'[0-9a-f]{40}', requested) and revision != requested:
    raise RuntimeError('Downloaded DiDi generator revision does not match the requested commit')
  # Resolve aliases only once. Both files must come from exactly the same
  # commit even if an upstream branch changes between the two downloads.
  discriminator = Path(hf_hub_download(
    filename=WEIGHT_FILES[1], revision=revision, **kwargs))
  if _snapshot_revision(discriminator) != revision:
    raise RuntimeError('DiDi generator and discriminator resolved to different revisions')
  return dict(zip(WEIGHT_FILES, (generator, discriminator))), {
    'source': source, 'repo_id': source, 'requested_revision': requested,
    'revision': revision, 'local_files_only': kwargs['local_files_only']}


def _validate_state(module, state, label):
  expected = module.state_dict()
  missing = sorted(set(expected) - set(state))
  unexpected = sorted(set(state) - set(expected))
  mismatched = [f'{name}: {tuple(state[name].shape)} != {tuple(expected[name].shape)}'
                for name in sorted(set(expected) & set(state))
                if state[name].shape != expected[name].shape]
  if missing or unexpected or mismatched:
    raise ValueError(f'Incompatible DiDi {label} weights: missing={missing}, '
                     f'unexpected={unexpected}, shapes={mismatched}')


def initialize_student_from_hf(model, config):
  """Load generator + discriminator safely before creating fresh optimizers.

  ``student_hf=None`` leaves the original teacher-initialization path untouched.
  ``student_hf`` otherwise names a HF repo or a local directory containing both
  safetensors files. Remote revisions are immutable in recorded provenance;
  ordinary local directories without HF snapshot metadata are identified by
  their file hashes instead of claiming an unverifiable repository revision.
  All three destination states are validated before any model tensor changes.
  """
  options = config.get('didi_init', {})
  if options.get('student_hf') is None:
    return None
  if config.checkpointing.resume_from_ckpt:
    raise ValueError('Fresh HF student initialization cannot be combined with checkpoint resume')
  if config.mode != 'train':
    raise ValueError('HF student initialization is a training warm start, not checkpoint evaluation')
  if getattr(model, '_didi_teacher_metadata', None) is None:
    raise ValueError('Initialize the separate native MDLM teacher before the HF student')
  from safetensors.torch import load_file
  from .monitor import preserve_rng_state
  with preserve_rng_state():
    files, metadata = _resolve_files(options)
    generator = load_file(str(files[WEIGHT_FILES[0]]), device='cpu')
    discriminator = load_file(str(files[WEIGHT_FILES[1]]), device='cpu')
    _validate_state(model.backbone, generator, 'generator')
    _validate_state(model.student_ema, generator, 'student EMA')
    _validate_state(model.discriminator, discriminator, 'discriminator')
    # Hash before loading so provenance I/O failures cannot leave half-applied
    # initialization. The separate teacher is never a destination here.
    metadata['files'] = {
      name: {'path': str(path), 'sha256': _sha256(path)} for name, path in files.items()}
    metadata.update({'kind': 'pretrained_didi_warm_start',
                     'generator_destinations': ['backbone', 'student_ema'],
                     'discriminator': 'complete state including spectral normalization buffers',
                     'teacher': 'separate native MDLM checkpoint; unchanged',
                     'optimizers': 'fresh; no optimizer state imported'})
    model.backbone.load_state_dict(generator, strict=True)
    model.student_ema.load_state_dict(generator, strict=True)
    model.discriminator.load_state_dict(discriminator, strict=True)
    if model.ema is not None:
      model.ema.shadow_params = [parameter.detach().clone()
                                for parameter in model._get_parameters()
                                if parameter.requires_grad]
      model.ema.collected_params = []
      if model.ema.num_updates is not None:
        model.ema.num_updates = 0
    model._didi_student_metadata = metadata
    return metadata
