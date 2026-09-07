"""Offline safetensors fixtures for optional pretrained DiDi warm starts."""

import copy
import hashlib
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf, open_dict
from safetensors.torch import save_file
import torch

import test_didi_runtime as runtime_fixture
from didi_repro.initialization import (
  AUTHORS_STUDENT_REPO, AUTHORS_STUDENT_REVISION, initialize_student_from_hf)


class InitializationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    runtime_fixture.DiDiRuntimeTests.setUpClass()

  def setUp(self):
    self.fixture = runtime_fixture.DiDiRuntimeTests()
    self.model = self.fixture.model()
    self.fixture.initialize(self.model)
    self.config = self.model.config
    self.directory = tempfile.TemporaryDirectory(prefix='didi-hf-init-')
    self.addCleanup(self.directory.cleanup)
    self.snapshot = Path(self.directory.name) / 'snapshots' / AUTHORS_STUDENT_REVISION
    self.snapshot.mkdir(parents=True)
    self.generator = {key: tensor.detach().clone() + .27
                      for key, tensor in self.model.backbone.state_dict().items()}
    self.discriminator = {key: tensor.detach().clone() + .43
                          for key, tensor in self.model.discriminator.state_dict().items()}
    self.write_weights()
    with open_dict(self.config):
      self.config.didi_init = {
        'student_hf': str(self.snapshot), 'revision': None,
        'cache_dir': None, 'local_files_only': False}

  def write_weights(self):
    save_file(self.generator, str(self.snapshot / 'model.safetensors'))
    save_file(self.discriminator, str(self.snapshot / 'discriminator.safetensors'))

  def assert_unchanged(self, expected):
    self.fixture.assert_same(expected, self.model.state_dict())
    self.assertIsNone(self.model._didi_student_metadata)

  def test_full_warm_start_preserves_teacher_and_optimizer_ownership(self):
    teacher = copy.deepcopy(self.model.teacher.state_dict())
    teacher_metadata = copy.deepcopy(self.model._didi_teacher_metadata)
    parameter_ids = {name: id(value) for name, value in self.model.named_parameters()}
    trainable = {name: value.requires_grad for name, value in self.model.named_parameters()}
    settings = OmegaConf.to_container(self.config, resolve=True)
    self.model.ema.num_updates = 17
    self.model.ema.collected_params = [torch.ones(1)]
    metadata = initialize_student_from_hf(self.model, self.config)
    self.fixture.assert_same(self.model.backbone.state_dict(), self.generator)
    self.fixture.assert_same(self.model.student_ema.state_dict(), self.generator)
    self.fixture.assert_same(self.model.discriminator.state_dict(), self.discriminator)
    self.fixture.assert_same(self.model.teacher.state_dict(), teacher)
    self.assertEqual(self.model._didi_teacher_metadata, teacher_metadata)
    self.assertEqual(parameter_ids, {name: id(value) for name, value in self.model.named_parameters()})
    self.assertEqual(trainable, {name: value.requires_grad for name, value in self.model.named_parameters()})
    self.assertTrue(any(key.endswith('weight_u') for key in self.discriminator))
    self.assertTrue(any(key.endswith('weight_v') for key in self.discriminator))
    self.assertEqual(OmegaConf.to_container(self.config, resolve=True), settings)
    for shadow, parameter in zip(self.model.ema.shadow_params, self.model._get_parameters()):
      torch.testing.assert_close(shadow, parameter, rtol=0, atol=0)
    self.assertEqual(self.model.ema.num_updates, 0)
    self.assertEqual(self.model.ema.collected_params, [])
    self.assertEqual(metadata['revision'], AUTHORS_STUDENT_REVISION)
    self.assertEqual(metadata['optimizers'], 'fresh; no optimizer state imported')
    for name in ('model.safetensors', 'discriminator.safetensors'):
      expected = hashlib.sha256((self.snapshot / name).read_bytes()).hexdigest()
      self.assertEqual(metadata['files'][name]['sha256'], expected)
    for optimizer in self.model.configure_optimizers():
      self.assertEqual(len(optimizer.state), 0)

  def test_none_source_is_noop_even_for_resume(self):
    self.config.didi_init.student_hf = None
    self.config.checkpointing.resume_from_ckpt = True
    before = copy.deepcopy(self.model.state_dict())
    with mock.patch('huggingface_hub.hf_hub_download') as download:
      self.assertIsNone(initialize_student_from_hf(self.model, self.config))
      download.assert_not_called()
    self.assert_unchanged(before)

  def test_resume_and_missing_teacher_fail_before_any_download(self):
    before = copy.deepcopy(self.model.state_dict())
    with mock.patch('huggingface_hub.hf_hub_download') as download:
      self.config.checkpointing.resume_from_ckpt = True
      with self.assertRaisesRegex(ValueError, 'cannot be combined'):
        initialize_student_from_hf(self.model, self.config)
      self.config.checkpointing.resume_from_ckpt = False
      self.model._didi_teacher_metadata = None
      with self.assertRaisesRegex(ValueError, 'separate native MDLM teacher'):
        initialize_student_from_hf(self.model, self.config)
      download.assert_not_called()
    self.assert_unchanged(before)

  def test_missing_file_cannot_partially_initialize_generator(self):
    before = copy.deepcopy(self.model.state_dict())
    (self.snapshot / 'discriminator.safetensors').unlink()
    with self.assertRaisesRegex(FileNotFoundError, 'Incomplete local'):
      initialize_student_from_hf(self.model, self.config)
    self.assert_unchanged(before)

  def test_missing_unexpected_or_wrong_shape_state_is_atomic(self):
    before = copy.deepcopy(self.model.state_dict())
    original = copy.deepcopy(self.discriminator)
    first_key = next(iter(original))
    for error in ('missing', 'unexpected', 'shape', 'spectral_buffer'):
      with self.subTest(error=error):
        self.discriminator = copy.deepcopy(original)
        if error == 'missing':
          self.discriminator.pop(first_key)
        elif error == 'unexpected':
          self.discriminator['unrecognized.weight'] = torch.zeros(1)
        elif error == 'shape':
          self.discriminator[first_key] = torch.zeros(1)
        else:
          self.discriminator.pop(next(key for key in original if key.endswith('weight_u')))
        self.write_weights()
        with self.assertRaisesRegex(ValueError, 'Incompatible DiDi discriminator'):
          initialize_student_from_hf(self.model, self.config)
        self.assert_unchanged(before)

  def test_partial_generator_is_rejected_before_discriminator_changes(self):
    before = copy.deepcopy(self.model.state_dict())
    self.generator.pop(next(iter(self.generator)))
    self.write_weights()
    with self.assertRaisesRegex(ValueError, 'Incompatible DiDi generator'):
      initialize_student_from_hf(self.model, self.config)
    self.assert_unchanged(before)

  def test_metadata_round_trip_and_old_checkpoint_compatibility(self):
    metadata = initialize_student_from_hf(self.model, self.config)
    checkpoint = {'state_dict': copy.deepcopy(self.model.state_dict())}
    self.model.on_save_checkpoint(checkpoint)
    self.assertEqual(checkpoint['didi_repro']['student_initialization'], metadata)
    restored = self.fixture.model()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'])
    self.assertEqual(restored._didi_student_metadata, metadata)
    self.fixture.assert_same(restored.teacher.state_dict(), self.model.teacher.state_dict())
    del checkpoint['didi_repro']['student_initialization']
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'])
    self.assertIsNone(restored._didi_student_metadata)

  def test_authors_defaults_fetch_only_two_weight_files_at_pinned_revision(self):
    self.config.didi_init.student_hf = AUTHORS_STUDENT_REPO
    self.config.didi_init.local_files_only = True
    self.config.didi_init.cache_dir = '/offline/mock-cache'
    def download(**kwargs):
      self.assertEqual(kwargs['repo_id'], AUTHORS_STUDENT_REPO)
      self.assertEqual(kwargs['revision'], AUTHORS_STUDENT_REVISION)
      self.assertTrue(kwargs['local_files_only'])
      self.assertEqual(kwargs['cache_dir'], '/offline/mock-cache')
      return str(self.snapshot / kwargs['filename'])
    with mock.patch('huggingface_hub.hf_hub_download', side_effect=download) as fetch:
      metadata = initialize_student_from_hf(self.model, self.config)
    self.assertEqual([call.kwargs['filename'] for call in fetch.call_args_list],
                     ['model.safetensors', 'discriminator.safetensors'])
    self.assertEqual(metadata['repo_id'], AUTHORS_STUDENT_REPO)
    self.assertEqual(metadata['revision'], AUTHORS_STUDENT_REVISION)

  def test_other_repository_requires_explicit_revision_and_pins_resolved_alias(self):
    self.config.didi_init.student_hf = 'another-owner/native-didi'
    with mock.patch('huggingface_hub.hf_hub_download') as fetch:
      with self.assertRaisesRegex(ValueError, 'requires explicit'):
        initialize_student_from_hf(self.model, self.config)
      fetch.assert_not_called()
    self.config.didi_init.revision = 'release-tag'
    with mock.patch('huggingface_hub.hf_hub_download',
                    side_effect=lambda **kwargs: str(self.snapshot / kwargs['filename'])) as fetch:
      metadata = initialize_student_from_hf(self.model, self.config)
    self.assertEqual(fetch.call_args_list[0].kwargs['revision'], 'release-tag')
    self.assertEqual(fetch.call_args_list[1].kwargs['revision'], AUTHORS_STUDENT_REVISION)
    self.assertEqual(metadata['requested_revision'], 'release-tag')
    self.assertEqual(metadata['revision'], AUTHORS_STUDENT_REVISION)

  def test_mismatched_download_commits_are_rejected(self):
    before = copy.deepcopy(self.model.state_dict())
    self.config.didi_init.student_hf = AUTHORS_STUDENT_REPO
    other = self.snapshot.parent / ('a' * 40) / 'discriminator.safetensors'
    with mock.patch('huggingface_hub.hf_hub_download',
                    side_effect=[str(self.snapshot / 'model.safetensors'), str(other)]):
      with self.assertRaisesRegex(RuntimeError, 'different revisions'):
        initialize_student_from_hf(self.model, self.config)
    self.assert_unchanged(before)

  def test_warm_start_does_not_advance_training_rng(self):
    self.config.didi_init.student_hf = AUTHORS_STUDENT_REPO
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    def download(**kwargs):
      random.random()
      np.random.rand()
      torch.rand(4)
      return str(self.snapshot / kwargs['filename'])
    with mock.patch('huggingface_hub.hf_hub_download', side_effect=download):
      initialize_student_from_hf(self.model, self.config)
    self.assertEqual(random.getstate(), before[0])
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))


if __name__ == '__main__':
  torch.set_num_threads(1)
  unittest.main(verbosity=2)
