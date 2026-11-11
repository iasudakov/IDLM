"""Opt-in TinyGSM integration; the pinned DiDi vendor trees stay untouched."""

from pathlib import Path


REFERENCE_REVISION = 'a3bfe2b3c00ce5ab02db98608a5a517a2e841a25'


def is_tinygsm(config):
  return config.get('data', {}).get('train') == 'tiny_gsm'


def register_configs():
  """Expose ``data=tiny-gsm`` without editing the upstream Hydra configs."""
  from hydra.core.config_store import ConfigStore
  from omegaconf import OmegaConf
  path = Path(__file__).parent / 'configs' / 'data' / 'tiny-gsm.yaml'
  ConfigStore.instance().store(
    group='data', name='tiny-gsm', node=OmegaConf.load(path),
    provider='didi-tinygsm-adaptation')


def prepare_tinygsm_config(config):
  """Add conditional metrics and reject incompatible unconditional pathways."""
  if not is_tinygsm(config):
    return
  from omegaconf import open_dict
  from .tinygsm_initialization import student_ema_source, training_weights_source
  with open_dict(config.didi_init):
    if 'tinygsm_weights' not in config.didi_init:
      config.didi_init.tinygsm_weights = 'raw'
    if 'tinygsm_student_ema' not in config.didi_init:
      config.didi_init.tinygsm_student_ema = 'checkpoint'
  student_ema_source(config)
  training_weights_source(config)
  defaults = {
    'tinygsm_exec': False, 'tinygsm_exec_timeout': 5.0,
    'gsm8k_at_validation': False, 'gsm8k_every_n_validations': 1,
    'gsm8k_num_examples': None, 'gsm8k_batch_size': None,
    'gsm8k_data_path': None, 'gsm8k_steps': None, 'gsm8k_plot': False,
    'gsm8k_pareto_thresholds': None, 'gsm8k_pareto_temperature': 0.0,
    'gsm8k_pareto_max_steps': None, 'gsm8k_pareto_reference': None,
    'gsm8k_pareto_plot': True, 'gsm8k_pareto_fp32': True,
  }
  with open_dict(config.eval):
    for key, value in defaults.items():
      if key not in config.eval:
        config.eval[key] = value
  if config.mode != 'train':
    raise ValueError('TinyGSM uses mode=train with conditional validation. '
                     'For fresh teacher evaluation only, set trainer.max_steps=0 '
                     'and +didi_eval.before_training=true. Resuming a trained '
                     'student requires preserving its saved training budget.')
  if config.algo.name != 'didi_instruct' or config.algo.backbone != 'dit':
    raise ValueError('TinyGSM DiDi requires algo=didi_instruct algo.backbone=dit.')
  if config.data.valid != 'tiny_gsm':
    raise ValueError('TinyGSM training validation uses data.valid=tiny_gsm; '
                     'enable eval.gsm8k_at_validation for the separate test benchmark.')
  if config.data.wrap or config.data.train_on_prompt:
    raise ValueError('Conditional TinyGSM requires data.wrap=false and '
                     'data.train_on_prompt=false.')
  if config.didi_init.student_hf is not None:
    raise ValueError('The released OWT DiDi HF student has a different vocabulary. '
                     'Start TinyGSM from its native MDLM checkpoint, or resume '
                     'a native DiDi TinyGSM training checkpoint.')
  if config.eval.compute_generative_perplexity:
    raise ValueError('TinyGSM uses answer/GSM8K accuracy, not OWT generative perplexity; '
                     'set eval.compute_generative_perplexity=false.')
  if config.didi_monitor.gm or config.didi_monitor.gen_ppl:
    raise ValueError('TinyGSM disables OWT monitor metrics; use '
                     '+didi_monitor.gm=false +didi_monitor.gen_ppl=false.')
  if config.didi_pareto.every_n_validations:
    raise ValueError('OWT didi_pareto is unconditional. For TinyGSM use '
                     '+eval.gsm8k_steps=[8,16,32] or '
                     '+eval.gsm8k_pareto_thresholds=[0.5,0.7,0.9] with '
                     '+eval.gsm8k_at_validation=true instead.')
  if config.eval.gsm8k_pareto_thresholds is not None:
    # Validate before allocating/checkpoint-loading the training models or
    # opening a tracker. This does not load data or run inference.
    from .tinygsm_eval import threshold_options
    threshold_options(config)
    if config.eval.gsm8k_pareto_thresholds and config.eval.gsm8k_pareto_plot:
      try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
      except ImportError as error:
        raise RuntimeError('TinyGSM threshold Pareto images require Matplotlib. '
                           'Install it in the training environment, or set '
                           '+eval.gsm8k_pareto_plot=false to keep numeric '
                           'curves, tables and history without images.') from error


def install_tinygsm(upstream):
  """Select adapters in this TinyGSM process, never on the OWT entry path."""
  if getattr(upstream, '_didi_tinygsm_installed', False):
    return upstream.algo.DiDiInstruct
  from . import tinygsm_data
  from .runtime import repaired_function
  from .tinygsm_runtime import install_tinygsm_runtime
  upstream.dataloader.get_tokenizer = tinygsm_data.get_tokenizer
  upstream.dataloader.get_dataloaders = tinygsm_data.get_dataloaders

  # Upstream eagerly downloads GPT-2's tokenizer even when generative PPL is
  # disabled. Conditional evaluation only uses the existing NLL/entropy states.
  metrics_module = upstream.algo.trainer_base.metrics
  original = metrics_module.Metrics
  initializer = repaired_function(original.__init__, ((
    "self.tokenizer = transformers.AutoTokenizer.\\\n"
    "    from_pretrained(gen_ppl_eval_model_name_or_path)\n"
    "  if self.tokenizer.pad_token is None:\n"
    "    self.tokenizer.pad_token = self.tokenizer.eos_token\n"
    "    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id",
    'self.tokenizer = None'),))

  class TinyGSMNLLMetrics(original):
    __init__ = initializer

    def _eval_retokenize(self, *args, **kwargs):
      raise RuntimeError('Generative PPL is disabled for TinyGSM conditional evaluation.')

  TinyGSMNLLMetrics.__module__ = __name__
  TinyGSMNLLMetrics.__qualname__ = 'TinyGSMNLLMetrics'
  globals()['TinyGSMNLLMetrics'] = TinyGSMNLLMetrics
  metrics_module.Metrics = TinyGSMNLLMetrics
  result = install_tinygsm_runtime(upstream)
  upstream._didi_tinygsm_installed = True
  return result


def adaptation_record(config):
  if not is_tinygsm(config):
    return None
  from .tinygsm_initialization import training_weights_source
  from .tinygsm_runtime import TRAINING_CORRECTIONS
  weights = training_weights_source(config)
  checkpoint_source = 'EMA parameters' if weights == 'ema' else 'raw backbone'
  return {
    'dataset': 'tiny_gsm',
    'reference_branch': 'tinygsm-upstream-repro',
    'reference_revision': REFERENCE_REVISION,
    'kind': 'conditional TinyGSM adaptation, not an authors-published TinyGSM reproduction',
    'conditioning': 'fixed prompt; target-only corruption, policy, KL, entropy and discriminator reward',
    'train_on_pad': bool(config.data.train_on_pad),
    'separator': config.data.separator,
    'training_settings': 'released DiDi hyperparameters with explicit TinyGSM training corrections',
    'training_corrections': dict(TRAINING_CORRECTIONS),
    'checkpoint': f'native TinyGSM MDLM {checkpoint_source} initializes student, teacher and discriminator trunk',
    'training_weights_initialization': weights,
    'initialization_scope': 'requested fresh initialization; saved provenance is authoritative on resume',
    'training_weights_resume': 'saved DiDi model tensors and initialization provenance take precedence',
    'student_ema_initialization': config.get('didi_init', {}).get('tinygsm_student_ema', 'checkpoint'),
    'student_ema_resume': 'saved DiDi student_ema tensors and initialization provenance take precedence',
    'evaluation': 'prefix-only generation and reference TinyGSM/S-FLM program grader',
  }
