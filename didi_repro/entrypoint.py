"""Location/logger adapters for the original DiDi-Instruct entry point."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from . import UPSTREAM_REVISION
from .runtime import UPSTREAM_ROOT, load_upstream, repaired_function


def prepare_config(config):
  """Add integration settings only; never overwrite an upstream setting."""
  from omegaconf import OmegaConf, open_dict
  additions = {
    'underdeep': {'experiment': 'generative-models/owt', 'name': 'didi-instruct-owt',
                  'force_new_run': False, 'required': True},
    'didi_monitor': {'enabled': True, 'gen_ppl': True, 'gm': True,
                     'gm_batch_size': 4, 'max_reference_samples': 64},
    'didi_runtime': {'strict_environment': True},
  }
  with open_dict(config):
    for key, value in additions.items():
      config[key] = OmegaConf.merge(value, config.get(key, {}))
  return config


def environment_record():
  result = {'python': platform.python_version()}
  for name in ('torch', 'lightning', 'pytorch-lightning', 'transformers',
               'flash-attn', 'numpy', 'datasets', 'hydra-core', 'omegaconf', 'underdeep'):
    try:
      result[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      result[name] = None
  try:
    import torch
    result['torch_cuda'] = torch.version.cuda
    result['cudnn'] = torch.backends.cudnn.version()
    result['devices'] = [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())]
  except Exception as error:
    result['device_query_error'] = str(error)
  return result


def validate_environment(config, record):
  # The core versions from the pinned environment.yml. The complete lockfile
  # is shipped unchanged; this guard does not claim to check every dependency.
  expected = {'torch': '2.5.1', 'lightning': '2.5.3',
              'pytorch-lightning': '2.5.3', 'flash-attn': '2.8.3',
              'transformers': '4.55.4'}
  differences = [f'{key}: installed {record.get(key)}, upstream {value}'
                 for key, value in expected.items()
                 if str(record.get(key)).split('+')[0] != value]
  if not record['python'].startswith('3.12.'):
    differences.append(f'python: installed {record["python"]}, upstream 3.12')
  if differences:
    message = ('DiDi dependency mismatch: ' + '; '.join(differences)
               + '. Use vendor/didi_instruct/environment.yml. '
               'For a non-reproduction smoke test only, explicitly set '
               '+didi_runtime.strict_environment=false.')
    if config.didi_runtime.strict_environment:
      raise RuntimeError(message)
    print(f'[didi] WARNING: {message}', flush=True)


def _check_cached_data(config):
  """Avoid upstream's destructive cache-rebuild fallback on shared OWT data."""
  import datasets
  for split, mode, eos in ((config.data.train, 'train', config.data.insert_train_eos),
                           (config.data.valid, 'validation', config.data.insert_valid_eos)):
    tag = 'wrapped' if config.data.wrap else 'unwrapped'
    eos_tag = '' if eos else '_eosFalse'
    filename = f'{split}_{mode}_bs{config.model.length}_{tag}{eos_tag}.dat'
    path = Path(config.data.cache_dir) / filename
    if not path.is_dir():
      raise FileNotFoundError(f'Missing original tokenized split: {path}. '
                              'Prepare the upstream OWT cache first; no automatic '
                              'download, deletion, or regeneration is performed here.')
    # A failed load is raised before upstream could call shutil.rmtree.
    dataset = datasets.load_from_disk(str(path))
    if len(dataset) == 0 or 'input_ids' not in dataset.column_names:
      raise ValueError(f'Invalid or empty upstream tokenized dataset: {path}')


def _write_provenance(config, record):
  from omegaconf import OmegaConf
  if int(os.environ.get('RANK', 0)) != 0:
    return
  directory = Path(config.checkpointing.save_dir)
  directory.mkdir(parents=True, exist_ok=True)
  # Runtime output files are part of the requested training workflow.
  OmegaConf.save(config, directory / 'didi_resolved_config.yaml', resolve=True)
  (directory / 'didi_environment.json').write_text(json.dumps(record, indent=2) + '\n')
  from hydra.core.hydra_config import HydraConfig
  overrides = list(HydraConfig.get().overrides.task) if HydraConfig.initialized() else []
  (directory / 'didi_provenance.json').write_text(json.dumps({
    'upstream_revision': UPSTREAM_REVISION,
    'overrides': overrides,
    'evaluation_weights': 'backbone (original generate_samples)',
    'runtime_fix_ledger': 'didi_repro/README.md',
  }, indent=2) + '\n')


def initialize_underdeep_logger(logger, config):
  if config.underdeep.required and int(os.environ.get('RANK', 0)) == 0:
    if logger.experiment is None:
      raise RuntimeError('Underdeep could not initialize. Configure its credentials '
                         'or explicitly use +underdeep.required=false for offline testing.')


def make_underdeep_logger(config, initialize=True):
  from .tracking import DiDiUnderdeepLogger
  from .monitor import preserve_rng_state
  with preserve_rng_state():
    logger = DiDiUnderdeepLogger(
      experiment=str(config.underdeep.experiment), name=str(config.underdeep.name),
      save_dir=str(config.checkpointing.save_dir),
      resume=bool(config.checkpointing.resume_from_ckpt),
      force_new_run=bool(config.underdeep.force_new_run))
  # The shared logger normally degrades to a no-op. A reproduction explicitly
  # requesting Underdeep must not silently run for hours without a tracker.
  if initialize:
    initialize_underdeep_logger(logger, config)
  return logger


def install_entrypoint(upstream):
  if getattr(upstream, '_didi_entrypoint_installed', False):
    return
  from .monitor import attach_monitor
  upstream.make_underdeep_logger = make_underdeep_logger
  upstream.initialize_underdeep_logger = initialize_underdeep_logger
  upstream.attach_didi_monitor = lambda model, config: attach_monitor(
    model, config, upstream_revision=UPSTREAM_REVISION)
  patched_train = repaired_function(upstream._train, (
    ("wandb_logger = None\n  if config.get('wandb', None) is not None:\n"
     "    wandb_logger = L.pytorch.loggers.WandbLogger(\n"
     "      config=omegaconf.OmegaConf.to_object(config),\n"
     "      **config.wandb)",
     'wandb_logger = make_underdeep_logger(config, initialize=False)'),
    ("checkpoint = torch.load(f, map_location='cpu')",
     "checkpoint = torch.load(f, map_location='cpu', weights_only=False)"),
    ("model.load_state_dict(checkpoint['state_dict'], strict=False)",
     'model.initialize_teacher_checkpoint(checkpoint, config.training.finetune_path)'),
    ('  # Print the number of trainable parameters',
     '  initialize_underdeep_logger(wandb_logger, config)\n'
     '  monitor = attach_didi_monitor(model, config)\n'
     '  if monitor is not None:\n'
     '    callbacks.append(monitor)\n\n'
     '  # Print the number of trainable parameters'),
  ))

  def train(diffusion_model, config, logger, tokenizer):
    if not Path(config.training.finetune_path).is_file():
      raise FileNotFoundError('Set training.finetune_path to the original native '
                              'teacher .ckpt; the strict reproduction never starts '
                              'with random weights or an unverified HF substitute.')
    if (config.checkpointing.resume_from_ckpt
        and not Path(config.checkpointing.resume_ckpt_path).is_file()):
      raise FileNotFoundError('Requested resume checkpoint does not exist; '
                              'refusing to append a fresh teacher restart to an old run.')
    _check_cached_data(config)
    return patched_train(diffusion_model, config, logger, tokenizer)

  upstream._train = train
  # Published native checkpoints contain serialized metadata and, on torch
  # >=2.6, require an explicit trusted-checkpoint loading choice.
  upstream._load_from_checkpoint = repaired_function(upstream._load_from_checkpoint, (
    ('strict=False)', "strict=False, weights_only=False).to('cuda')"),))

  # A shuffle/index-cache or distributed-barrier failure after preflight must
  # never reach upstream's broad except -> rmtree -> regenerate fallback.
  upstream.dataloader.get_dataset = repaired_function(upstream.dataloader.get_dataset, (
    ('  if utils.fsspec_exists(_path):\n    try:',
     '  if True:  # Reproduction accepts existing caches only.\n    try:'),
    ('    except Exception as e:\n      LOGGER.warning',
     "    except Exception as e:\n      raise RuntimeError(f'DiDi cache load failed; preserved {_path}') from e\n      LOGGER.warning"),
  ))

  original_sample_eval = upstream._generate_samples

  def sample_eval(diffusion_model, config, logger, tokenizer):
    # Original sampler, original metrics, original output JSON. Only publish
    # its already-computed results to the requested tracker afterward.
    original_sample_eval(diffusion_model, config, logger, tokenizer)
    from .monitor import config_rows, preserve_rng_state
    with preserve_rng_state():
      tracker = make_underdeep_logger(config)
      output = json.loads(Path(config.eval.generated_samples_path).read_text())
      tracker.log_metrics({'val/gen_ppl': output['generative_ppl'],
                           'val/sample_entropy': output['entropy']}, step=0)
      tracker.log_table('didi/config', columns=['Parameter', 'JSON value'],
                        data=config_rows(config, UPSTREAM_REVISION), step=0)
      tracker.log_table('val/samples', columns=['Generated Samples'],
                        data=[[text] for text in output['generated_seqs'][
                          :config.sampling.num_sample_log]], step=0)
      tracker.finalize('success')

  upstream._generate_samples = sample_eval
  upstream._didi_entrypoint_installed = True


def run():
  import hydra
  upstream = load_upstream()
  install_entrypoint(upstream)

  @hydra.main(version_base=None, config_path=str(UPSTREAM_ROOT / 'configs'), config_name='config')
  def main(config):
    prepare_config(config)
    record = environment_record()
    validate_environment(config, record)
    _write_provenance(config, record)
    # Calls the actual original entry body, not IDLM's algorithm dispatch.
    upstream.main.__wrapped__(config)

  main()
