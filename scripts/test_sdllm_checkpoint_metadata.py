"""Regression checks for unused tokenizer objects in native SDLLM checkpoints.

Only small synthetic CPU checkpoints are read. The metadata deliberately
references the release's dataloader.CustomTikTokenizer without importing the
release's Python code. Network calls and GPU initialization are unnecessary.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import mmd_sdllm_features
import test_sdllm_feature_routing as routing


def write_checkpoint_with_metadata(path, module_name, class_name, state):
    """Save a real pickle global, then remove its original implementation."""
    module = ModuleType(module_name)
    exec(f'class {class_name}:\n    pass\n', module.__dict__)
    metadata = getattr(module, class_name)()
    metadata.tokenizer_path = '/unused/release/tokenizer/tiktokenizer.json'
    metadata.vocab_size = 131072
    metadata.nested = {'special_tokens': {'bos': 1, 'eos': 2, 'mask': 3}}
    with mock.patch.dict(sys.modules, {module_name: module}):
        torch.save({
            'state_dict': state,
            'hyper_parameters': {'tokenizer': metadata},
            'optimizer_states': [{'state': {0: {'step': torch.tensor(8.)}}}],
        }, path)
    return ModuleType(module_name)


class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='sdllm-metadata-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'model.ckpt'
        # A non-contiguous tensor helps detect accidental reconstruction/casts.
        self.state = {
            'backbone.vocab_embed.embedding': torch.arange(24, dtype=torch.float32).reshape(6, 4),
            'backbone.blocks.0.weight': torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
            'backbone.rotary_emb.inv_freq': torch.tensor([1., .01]),
            'irrelevant.weight': torch.tensor([999.]),
        }
        guard = mock.patch.object(socket.socket, 'connect',
                                  side_effect=AssertionError('Network access attempted'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_unused_release_tokenizer_metadata_preserves_raw_backbone_and_rng(self):
        unavailable = write_checkpoint_with_metadata(
            self.path, 'dataloader', 'CustomTikTokenizer', self.state)
        before = torch.get_rng_state().clone()
        with mock.patch.dict(sys.modules, {'dataloader': unavailable}):
            actual = mmd_sdllm_features._native_state(self.path)
            self.assertFalse(hasattr(unavailable, 'CustomTikTokenizer'))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        expected = {name.removeprefix('backbone.'): tensor
                    for name, tensor in self.state.items() if name.startswith('backbone.')}
        self.assertEqual(set(actual), set(expected))
        for name, tensor in expected.items():
            self.assertEqual(actual[name].dtype, tensor.dtype)
            self.assertEqual(actual[name].stride(), tensor.stride())
            torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)

    def test_unrelated_missing_global_is_not_silently_discarded(self):
        # Neither another class in dataloader nor the same name in another
        # module is the explicitly supported release metadata type.
        for module_name, class_name in (
                ('dataloader', 'UnrelatedMetadata'),
                ('sdllm_missing_fixture', 'CustomTikTokenizer')):
            with self.subTest(module=module_name, name=class_name):
                unavailable = write_checkpoint_with_metadata(
                    self.path, module_name, class_name, self.state)
                with mock.patch.dict(sys.modules, {module_name: unavailable}):
                    with self.assertRaises(AttributeError):
                        mmd_sdllm_features._native_state(self.path)

    def test_nested_release_encoding_is_not_imported_or_restored(self):
        # Match the verified release's nested NEWOBJ/BUILD pickle structure.
        # Its original Encoding.__setstate__ would rebuild a native BPE engine;
        # that unused metadata must not import or execute either implementation.
        release = ModuleType('dataloader')
        core = ModuleType('tiktoken.core')
        package = ModuleType('tiktoken')
        package.__path__ = []
        package.core = core
        exec('class CustomTikTokenizer:\n'
             '    def __setstate__(self, state):\n'
             '        raise AssertionError("release tokenizer restored")\n',
             release.__dict__)
        exec('class Encoding:\n'
             '    def __setstate__(self, state):\n'
             '        raise AssertionError("native encoding rebuilt")\n',
             core.__dict__)
        metadata = release.CustomTikTokenizer()
        metadata.tokenizer = core.Encoding()
        metadata.tokenizer._pat_str = 'unused pattern'
        metadata.tokenizer._mergeable_ranks = {b'a': 1000, b'b': 1001}
        modules = {'dataloader': release, 'tiktoken': package, 'tiktoken.core': core}
        with mock.patch.dict(sys.modules, modules):
            torch.save({'state_dict': self.state,
                        'hyper_parameters': {'tokenizer': metadata}}, self.path)
            original_import = __import__

            def forbid_release_import(name, *args, **kwargs):
                if name in ('dataloader', 'tiktoken', 'tiktoken.core'):
                    raise AssertionError(f'Unused checkpoint module imported: {name}')
                return original_import(name, *args, **kwargs)

            before = torch.get_rng_state().clone()
            with mock.patch('builtins.__import__', side_effect=forbid_release_import):
                actual = mmd_sdllm_features._native_state(self.path)
            self.assertIs(release.CustomTikTokenizer, type(metadata))
            self.assertIs(core.Encoding, type(metadata.tokenizer))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name, value in self.state.items():
            if name.startswith('backbone.'):
                torch.testing.assert_close(actual[name.removeprefix('backbone.')],
                                           value, rtol=0, atol=0)

    def test_plain_tensor_checkpoint_still_loads_without_metadata(self):
        torch.save({'state_dict': self.state}, self.path)
        actual = mmd_sdllm_features._native_state(self.path)
        torch.testing.assert_close(actual['vocab_embed.embedding'],
                                   self.state['backbone.vocab_embed.embedding'],
                                   rtol=0, atol=0)

    def test_invalid_backbone_payload_is_still_rejected(self):
        unavailable = write_checkpoint_with_metadata(
            self.path, 'dataloader', 'CustomTikTokenizer',
            {'backbone.invalid': 'not a tensor'})
        with mock.patch.dict(sys.modules, {'dataloader': unavailable}):
            with self.assertRaisesRegex(ValueError, 'tensors only'):
                mmd_sdllm_features._native_state(self.path)


class CriticPropertyDiagnosticsTests(unittest.TestCase):
    def test_external_initialization_attribute_error_preserves_original_cause(self):
        # Exercise normal nn.Module attribute access, not descriptor.fget:
        # only normal access reproduces the misleading missing-property error.
        for failing_stage in ('loader', 'critic'):
            with self.subTest(stage=failing_stage):
                model = routing.make_model()
                original = AttributeError('missing release metadata fixture')
                with routing.external_stubs() as (_, loader, _, constructor):
                    (loader if failing_stage == 'loader' else constructor).side_effect = original
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(RuntimeError) as caught:
                            _ = model._mmd_critic
                self.assertIs(caught.exception.__cause__, original)
                self.assertIsNone(getattr(model, '_mmd_critic_cached', None))


if __name__ == '__main__':
    unittest.main(verbosity=2)
