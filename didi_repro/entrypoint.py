"""Location/logger adapters for the original DiDi-Instruct entry point."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from . import UPSTREAM_REVISION
from .protocol import optimizer_step_budget, training_iterations, training_protocol
from .runtime import UPSTREAM_ROOT, load_upstream, repaired_function
from .sampling_protocol import configure_sampler_source, sampling_protocol


def prepare_config(config, overrides=None):
  """Keep upstream defaults unless an explicit integration option opts in."""
  from omegaconf import OmegaConf, open_dict
  additions = {
    'underdeep': {'experiment': 'generative-models/owt', 'name': 'didi-instruct-owt',
                  'force_new_run': False, 'required': True},
    'didi_monitor': {'enabled': True, 'gen_ppl': True, 'gm': True,
                     'gm_batch_size': 4, 'max_reference_samples': 64},
    'didi_eval': {'before_training': False, 'weights': 'backbone', 'sampler_source': 'main'},
    'didi_training': {'iterations': None},
    'didi_pareto': {'every_n_validations': 0,
                    'temperatures': [0.75, 0.85, 0.95, 1.0, 1.05, 1.2],
                    'steps': [8, 16, 32], 'seeds': 1, 'num_batches': 2,
                    'batch_size': None, 'keep_samples': 2,
                    'reference': None, 'sampler': None},
    'didi_init': {'student_hf': None, 'revision': None, 'cache_dir': None,
                  'local_files_only': False},
    'didi_runtime': {'strict_environment': True},
  }
  with open_dict(config):
    for key, value in additions.items():
      config[key] = OmegaConf.merge(value, config.get(key, {}))
  configure_training_budget(config, overrides=overrides)
  configure_sampler_source(config, overrides=overrides)
  return config


def configure_training_budget(config, overrides=None):
  """Resolve an opt-in iteration horizon without hiding conflicting CLI flags."""
  from hydra.core.hydra_config import HydraConfig
  from omegaconf import open_dict
  weights = config.get('didi_eval', {}).get('weights', 'backbone')
  if weights not in ('backbone', 'student_ema'):
    raise ValueError('didi_eval.weights must be backbone or student_ema.')
  iterations = training_iterations(config)
  if iterations is None or config.get('mode', 'train') != 'train':
    return
  budget = optimizer_step_budget(config, iterations)
  configured = config.trainer.max_steps
  if overrides is None:
    overrides = (HydraConfig.get().overrides.task if HydraConfig.initialized() else [])
  explicit = any(str(item).split('=', 1)[0].lstrip('+') == 'trainer.max_steps'
                 for item in overrides)
  if configured != budget and (explicit or configured != 10000):
    raise ValueError(f'didi_training.iterations={iterations} requires '
                     f'trainer.max_steps={budget} individual optimizer updates, '
                     f'but trainer.max_steps={configured} was explicitly configured. '
                     'Remove that override or set it to the derived budget.')
  with open_dict(config):
    config.trainer.max_steps = budget


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
    'evaluation_weights': config.get('didi_eval', {}).get('weights', 'backbone'),
    'sampling_protocol': sampling_protocol(config),
    'training_protocol': training_protocol(config),
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
  from .initialization import initialize_student_from_hf
  from .pretrain_eval import maybe_evaluate_before_training
  from .pareto import attach_pareto, validate_options
  upstream.make_underdeep_logger = make_underdeep_logger
  upstream.initialize_underdeep_logger = initialize_underdeep_logger
  upstream.initialize_didi_student = initialize_student_from_hf
  upstream.maybe_evaluate_didi_before_training = maybe_evaluate_before_training
  upstream.attach_didi_monitor = lambda model, config: attach_monitor(
    model, config, upstream_revision=UPSTREAM_REVISION)
  upstream.attach_didi_pareto = attach_pareto
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
     '  initialize_didi_student(model, config)\n'
     '  initialize_underdeep_logger(wandb_logger, config)\n'
     '  monitor = attach_didi_monitor(model, config)\n'
     '  if monitor is not None:\n'
     '    callbacks.append(monitor)\n'
     '  pareto = attach_didi_pareto(model, config)\n'
     '  if pareto is not None:\n'
     '    callbacks.append(pareto)\n\n'
     '  # Print the number of trainable parameters'),
    ('  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)',
     '  maybe_evaluate_didi_before_training(trainer, model, valid_ds, ckpt_path=ckpt_path)\n'
     '  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)'),
  ))

  def train(diffusion_model, config, logger, tokenizer):
    configure_training_budget(config)
    validate_options(config)
    if (config.get('didi_init', {}).get('student_hf')
        and config.checkpointing.resume_from_ckpt):
      raise ValueError('HF student initialization is a fresh warm start, not a resume. '
                       'Clear didi_init.student_hf when resuming a training checkpoint.')
    if (config.get('didi_eval', {}).get('before_training', False)
        and config.trainer.limit_val_batches == 0):
      raise ValueError('didi_eval.before_training requires trainer.limit_val_batches > 0.')
    if not Path(config.training.finetune_path).is_file():
      raise FileNotFoundError('Set training.finetune_path to the original native '
                              'teacher .ckpt; the strict reproduction never starts '
                              'with a random teacher. didi_init.student_hf initializes '
                              'the student/discriminator only, not this frozen teacher.')
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
