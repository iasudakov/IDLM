"""Explicit evaluation-source selection; training always uses the main release."""

from . import UPSTREAM_REVISION
from .standalone import INFERENCE_REVISION, ROOT, verify_source


def sampler_source(config):
  source = config.get('didi_eval', {}).get('sampler_source', 'main')
  if source not in ('main', 'inference'):
    raise ValueError('didi_eval.sampler_source must be main or inference.')
  return source


def validate_sampler_source(config):
  """Reject combinations that would silently bypass the selected sampler."""
  source = sampler_source(config)
  if source == 'main':
    return source
  if config.sampling.predictor != 'guided':
    raise ValueError('didi_eval.sampler_source=inference requires sampling.predictor=guided.')
  if config.get('didi_eval', {}).get('weights', 'backbone') != 'student_ema':
    raise ValueError('didi_eval.sampler_source=inference requires didi_eval.weights=student_ema '
                     '(the official ancestral cleanup still uses raw backbone weights).')
  if config.get('didi_pareto', {}).get('sampler', None) not in (None, 'guided'):
    raise ValueError('didi_eval.sampler_source=inference requires didi_pareto.sampler=null or guided.')
  return source


def configure_sampler_source(config, overrides=None):
  """Opt into inference defaults without erasing explicit guidance overrides."""
  source = validate_sampler_source(config)
  if source == 'main':
    return source
  from hydra.core.hydra_config import HydraConfig
  from omegaconf import OmegaConf, open_dict

  directory, _ = verify_source('inference')
  defaults = OmegaConf.load(directory / 'configs/algo/didi_instruct.yaml')
  main_defaults = OmegaConf.load(ROOT / 'vendor/didi_instruct/configs/algo/didi_instruct.yaml')
  if overrides is None:
    overrides = HydraConfig.get().overrides.task if HydraConfig.initialized() else []
  explicit = {str(item).split('=', 1)[0].lstrip('+') for item in overrides}
  with open_dict(config.algo):
    for name in ('guidance_scale_start', 'guidance_scale_end',
                 'num_candidates', 'rerank_steps_ratio'):
      # Also preserve non-default programmatic settings, for callers that do
      # not have Hydra's override list. Repeated preparation is idempotent.
      if (f'algo.{name}' not in explicit
          and config.algo.get(name, main_defaults[name]) == main_defaults[name]):
        config.algo[name] = defaults[name]
  return source


def sampling_protocol(config):
  """Source/weight/temperature semantics for local and tracker provenance."""
  source = sampler_source(config)
  weights = config.get('didi_eval', {}).get('weights', 'backbone')
  return {
    'sampler_source': source,
    'sampler_revision': INFERENCE_REVISION if source == 'inference' else UPSTREAM_REVISION,
    'guided_weights': 'student_ema' if source == 'inference' else weights,
    'ancestral_cleanup_weights': 'backbone' if source == 'inference' else weights,
    'temperature_transform': ('inference-raw-EMA-logits-and-SUBS-cleanup-temperature-v1'
                              if source == 'inference'
                              else 'masked-SUBS-logprob-temperature-v1'),
    'guidance_schedule': ('phase-local gradient tilting followed by categorical reranking'
                          if source == 'inference' else 'main-release guided schedule'),
    'reranking_distribution': ('softmax(reward)' if source == 'inference'
                               else 'softmax(reward * guidance_scale)'),
  }
