"""CPU-only tests for the original-evaluation checkpoint format adapter."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import torch
from torch import nn

import prepare_didi_authors_checkpoint as prepare


class CheckpointPreparationTests(unittest.TestCase):
    def setUp(self):
        self.backbone = nn.Linear(3, 4)
        self.discriminator = nn.utils.spectral_norm(nn.Linear(3, 1))
        self.model = SimpleNamespace(
            backbone=self.backbone, student_ema=copy.deepcopy(self.backbone),
            discriminator=self.discriminator)
        self.generator = {name: value.detach().clone() + .125
                          for name, value in self.backbone.state_dict().items()}
        self.disc = {name: value.detach().clone() - .125
                     for name, value in self.discriminator.state_dict().items()}

    def test_mapping_preserves_values_dtypes_and_spectral_buffers(self):
        before = copy.deepcopy(self.model.backbone.state_dict())
        state = prepare.mapped_state(self.model, self.generator, self.disc, 'cpu')
        self.assertEqual(len(state), 2 * len(self.generator) + len(self.disc))
        for name, tensor in self.generator.items():
            raw, ema = state[f'backbone.{name}'], state[f'student_ema.{name}']
            self.assertIs(raw, ema)
            self.assertEqual(raw.dtype, tensor.dtype)
            torch.testing.assert_close(raw, tensor, rtol=0, atol=0)
        for name, tensor in self.disc.items():
            torch.testing.assert_close(state[f'discriminator.{name}'], tensor, rtol=0, atol=0)
        self.assertIn('discriminator.weight_u', state)
        self.assertIn('discriminator.weight_v', state)
        for name, tensor in before.items():
            torch.testing.assert_close(self.model.backbone.state_dict()[name], tensor, rtol=0, atol=0)
        self.assertFalse(any(name.startswith('teacher.') for name in state))

    def test_rejects_incomplete_wrong_shape_and_unexpected_states(self):
        for variant in ('missing', 'unexpected', 'shape', 'spectral_buffer'):
            with self.subTest(variant=variant):
                disc = dict(self.disc)
                if variant == 'missing':
                    disc.pop('weight_orig')
                elif variant == 'unexpected':
                    disc['extra'] = torch.zeros(1)
                elif variant == 'shape':
                    disc['weight_orig'] = torch.zeros(1)
                else:
                    disc.pop('weight_v')
                with self.assertRaisesRegex(ValueError, 'Incompatible DiDi discriminator'):
                    prepare.mapped_state(self.model, self.generator, disc, 'cpu')

    def test_ema_destination_is_validated_too(self):
        self.model.student_ema = nn.Linear(3, 5)
        with self.assertRaisesRegex(ValueError, 'Incompatible DiDi student EMA'):
            prepare.mapped_state(self.model, self.generator, self.disc, 'cpu')

    def test_cpu_checkpoint_roundtrip_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='didi-authors-checkpoint-') as folder:
            target = Path(folder) / 'prepared.ckpt'
            state = prepare.mapped_state(self.model, self.generator, self.disc, 'cpu')
            output, sidecar = prepare.write_checkpoint(
                target, state, {'saved_tensor_device': 'cpu', 'source': 'fixture'}, '2.5.3')
            checkpoint = torch.load(output, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['pytorch-lightning_version'], '2.5.3')
            self.assertEqual(checkpoint['epoch'], 0)
            self.assertEqual(checkpoint['global_step'], 0)
            self.assertNotIn('optimizer_states', checkpoint)
            restored = checkpoint['state_dict']
            for name, tensor in state.items():
                torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)
            self.assertEqual(restored['backbone.weight'].data_ptr(),
                             restored['student_ema.weight'].data_ptr())
            record = json.loads(sidecar.read_text())
            self.assertEqual(record['checkpoint']['sha256'], prepare._sha256(output))
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare.write_checkpoint(output, state, {}, '2.5.3')
            self.assertEqual(output.read_bytes(), original)

    def test_existing_sidecar_prevents_checkpoint_creation(self):
        with tempfile.TemporaryDirectory(prefix='didi-authors-checkpoint-') as folder:
            target = Path(folder) / 'prepared.ckpt'
            sidecar = target.with_name(target.name + '.json')
            sidecar.touch()
            with self.assertRaises(FileExistsError):
                prepare.output_paths(target)
            self.assertFalse(target.exists())

    def test_dangling_output_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory(prefix='didi-authors-checkpoint-') as folder:
            target = Path(folder) / 'prepared.ckpt'
            destination = Path(folder) / 'missing.ckpt'
            target.symlink_to(destination)
            with self.assertRaises(FileExistsError):
                prepare.output_paths(target)
            self.assertFalse(destination.exists())

    def test_cuda_requires_single_visible_device_without_allocating(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(
            is_available=mock.Mock(return_value=True), device_count=mock.Mock(return_value=1)),
            device=lambda name: name)
        with mock.patch.dict(prepare.os.environ, {'WORLD_SIZE': '1'}):
            self.assertEqual(prepare.saved_device('cuda', fake), 'cuda:0')
            fake.cuda.device_count.return_value = 2
            with self.assertRaisesRegex(ValueError, 'exactly one visible GPU'):
                prepare.saved_device('cuda:0', fake)
            fake.cuda.is_available.return_value = False
            with self.assertRaisesRegex(ValueError, 'exactly one visible GPU'):
                prepare.saved_device('cuda:0', fake)
            self.assertEqual(prepare.saved_device('cpu', fake), 'cpu')
            with self.assertRaisesRegex(ValueError, 'Use cuda:0'):
                prepare.saved_device('cuda:1', fake)
        with mock.patch.dict(prepare.os.environ, {'WORLD_SIZE': '8'}):
            with self.assertRaisesRegex(ValueError, 'one process'):
                prepare.saved_device('cpu', fake)

    def test_cli_defaults_to_authors_and_cuda(self):
        options = prepare.parser().parse_args(['--output', '/tmp/unused-didi-test.ckpt'])
        self.assertEqual(options.student_hf, prepare.AUTHORS_STUDENT_REPO)
        self.assertIsNone(options.revision)  # Resolver supplies pinned author commit.
        self.assertEqual(options.device, 'cuda:0')

    def test_existing_output_fails_before_model_or_download(self):
        with tempfile.TemporaryDirectory(prefix='didi-authors-checkpoint-') as folder:
            target = Path(folder) / 'prepared.ckpt'
            target.touch()
            with mock.patch.object(prepare, '_resolve_files') as resolve, \
                    mock.patch.object(prepare, 'pristine_model') as model:
                with self.assertRaises(FileExistsError):
                    prepare.main(['--output', str(target), '--device', 'cpu'])
                resolve.assert_not_called()
                model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
