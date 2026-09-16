"""Explicit EMA initialization for fresh native MDLM distillation runs."""

import copy

import torch


def finetune_weights(config):
  choice = config.training.get('finetune_weights', 'raw')
  if choice not in ('raw', 'ema'):
    raise ValueError('training.finetune_weights must be raw or ema.')
  return choice


def validate_finetune_weights(config):
  if finetune_weights(config) == 'raw' or config.mode != 'train':
    return
  if (config.algo.name != 'mdlm' or config.algo.backbone != 'dit'
      or not config.adversarial_distill.is_distill):
    raise ValueError(
      'training.finetune_weights=ema requires native MDLM distillation '
      '(algo=mdlm, algo.backbone=dit, adversarial_distill.is_distill=True).')
  if config.training.get('continue_teacher', False):
    raise ValueError(
      'training.finetune_weights=ema cannot be combined with '
      'training.continue_teacher=True.')
  if not config.training.finetune_path:
    raise ValueError(
      'training.finetune_weights=ema requires training.finetune_path.')
  if config.training.ema <= 0:
    raise ValueError(
      'training.finetune_weights=ema requires training.ema > 0 '
      'to restore the checkpoint EMA.')


@torch.no_grad()
def initialize_backbones_from_ema(model):
  """Use saved EMA parameters for every diffusion model copy.

  Call only after a warm-start load, never from a checkpoint loading hook:
  those hooks also run when resuming training or evaluating saved students.
  The EMA parameter order is the native backbone's named-parameter order.
  Buffers are not averaged and retain their checkpoint values.
  """
  ema = getattr(model, 'ema', None)
  if ema is None:
    raise ValueError('EMA initialization requires a loaded checkpoint EMA.')
  reference = list(model.backbone.named_parameters())
  shadows = ema.shadow_params
  if len(shadows) != len(reference):
    raise ValueError(
      f'Checkpoint EMA has {len(shadows)} parameters; '
      f'the backbone requires {len(reference)}.')
  targets = []
  # Do not filter requires_grad: the teacher and, on fake-model steps, the
  # student are frozen, but must still receive every EMA parameter.
  for name in ('backbone', 'teacher', 'fake_model'):
    module = getattr(model, name, None)
    if module is None:
      continue
    parameters = list(module.named_parameters())
    if [key for key, _ in parameters] != [key for key, _ in reference]:
      raise ValueError(f'{name} parameter order differs from the EMA backbone.')
    for (key, parameter), shadow in zip(parameters, shadows):
      if not torch.is_tensor(shadow) or shadow.shape != parameter.shape:
        raise ValueError(f'Checkpoint EMA shape mismatch for {name}.{key}.')
    targets.append(parameters)

  # Validate every destination before changing any parameter. Copies leave the
  # EMA shadows independent of the trainable models and the frozen teacher.
  for parameters in targets:
    for (_, parameter), shadow in zip(parameters, shadows):
      parameter.copy_(shadow)
  if getattr(model, 'ema_fake', None) is not None:
    # Its starting model is now the same EMA teacher. Keep the saved decay and
    # update counter, and do not share shadow tensors with the student EMA.
    model.ema_fake.load_state_dict(copy.deepcopy(ema.state_dict()))


def apply_finetune_weights(model, config, *, resuming=False):
  if finetune_weights(config) == 'raw' or resuming:
    return False
  initialize_backbones_from_ema(model)
  return True
