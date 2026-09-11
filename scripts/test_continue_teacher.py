"""CPU-only checks for the teacher-continuation mode.

Run with:

    python scripts/test_continue_teacher.py
"""
import collections
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import main  # noqa: E402
import algo  # noqa: E402
import models  # noqa: E402


def config(continue_teacher, is_distill=True, finetune_path='teacher.ckpt'):
  return OmegaConf.create({
    'mode': 'train',
    'training': {
      'continue_teacher': continue_teacher,
      'finetune_path': finetune_path,
    },
    'adversarial_distill': {
      'is_distill': is_distill,
      'accum_batches': 2,
    },
    'trainer': {'accumulate_grad_batches': 1},
  })


distill = config(False)
main._configure_training_mode(distill)
assert distill.adversarial_distill.is_distill is True
assert distill.trainer.accumulate_grad_batches == 1

teacher = config(True)
main._configure_training_mode(teacher)
assert teacher.adversarial_distill.is_distill is False
assert teacher.trainer.accumulate_grad_batches == 2

already_standard = config(True, is_distill=False)
main._configure_training_mode(already_standard)
assert already_standard.adversarial_distill.is_distill is False

missing = config(True, finetune_path='')
try:
  main._configure_training_mode(missing)
except ValueError as error:
  assert 'finetune_path' in str(error)
else:
  raise AssertionError('teacher continuation accepted no teacher checkpoint')

eval_config = config(True)
eval_config.mode = 'sample_eval'
main._configure_training_mode(eval_config)
assert eval_config.adversarial_distill.is_distill is True

# Loading a distillation checkpoint in this mode must select its frozen teacher,
# not the student backbone or the student's EMA.
model = algo.MDLM.__new__(algo.MDLM)
torch.nn.Module.__init__(model)
model.config = config(True, is_distill=False)
model.config.training.ema = 0.9
model.backbone = torch.nn.Linear(2, 2)
model.ema = models.ema.ExponentialMovingAverage(
  model.backbone.parameters(), decay=0.9)
model.is_adversarial_distill = False

student_weight = torch.full_like(model.backbone.weight, 1.0)
student_bias = torch.full_like(model.backbone.bias, 2.0)
teacher_weight = torch.full_like(model.backbone.weight, 3.0)
teacher_bias = torch.full_like(model.backbone.bias, 4.0)
checkpoint = {
  'state_dict': collections.OrderedDict({
    'backbone.weight': student_weight,
    'backbone.bias': student_bias,
    'teacher.weight': teacher_weight,
    'teacher.bias': teacher_bias,
    'fake_model.weight': torch.full_like(student_weight, 5.0),
    'fake_model.bias': torch.full_like(student_bias, 6.0),
  }),
  'ema': {
    'decay': 0.9,
    'num_updates': 10,
    'shadow_params': [student_weight.clone(), student_bias.clone()],
  },
  'loops': {
    'fit_loop': {
      'epoch_progress': {'current': {'completed': 0}},
      'epoch_loop.batch_progress': {'current': {'completed': 0}},
    },
  },
}
model.on_load_checkpoint(checkpoint)
assert set(checkpoint['state_dict']) == {'backbone.weight', 'backbone.bias'}
assert torch.equal(checkpoint['state_dict']['backbone.weight'], teacher_weight)
assert torch.equal(checkpoint['state_dict']['backbone.bias'], teacher_bias)
assert torch.equal(model.ema.shadow_params[0], teacher_weight)
assert torch.equal(model.ema.shadow_params[1], teacher_bias)
assert model.ema.num_updates == 0

print('6/6 passed')
