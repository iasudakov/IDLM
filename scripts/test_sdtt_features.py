"""CPU-only SDTT feature-loader checks with a tiny real DiT checkpoint.

No network, GPU, large weights, or SDTT installation is needed. Run with:
  python scripts/test_sdtt_features.py
The ordinary GPT-2 tokenizer vocabulary must already be in the local HF cache.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dit import DIT
from mmd_sdtt_features import load_sdtt_feature_backbone
import mmd_reward


class VocabularyOverride:
    """Keep actual cached tokenizer behavior except a deliberately bad mapping."""

    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def get_vocab(self):
        return self.vocab


class SDTTFeatureLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained('gpt2-large', local_files_only=True)
        cls.model_config = {
            'name': 'tiny-test', 'type': 'ddit-orig', 'hidden_size': 16,
            'cond_dim': 8, 'length': 16, 'n_blocks': 2, 'n_heads': 2,
            'scale_by_sigma': True, 'dropout': .1, 'tie_word_embeddings': False,
            'vocab_lookup': True,
        }
        cls.config = {'model': copy.deepcopy(cls.model_config),
                      'tokenizer': {'name': 'gpt2'}, 'time_conditioning': False}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4107)
            cls.reference = DIT({'model': cls.model_config,
                                 'algo': {'causal_attention': False}}, vocab_size=50258).eval()
            # Default DiT gates/head are zero; nonzero values ensure that this
            # test actually detects incorrect timestep conditioning or blocks.
            with torch.no_grad():
                for name, param in cls.reference.named_parameters():
                    if 'adaLN_modulation' in name or name.startswith('output_layer.linear'):
                        param.normal_(mean=0., std=.025)
        cls.state = {'backbone.' + name: tensor.detach().clone().contiguous()
                     for name, tensor in cls.reference.state_dict().items()}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sdtt feature test ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.asset = self.root / 'feature model'
        self.asset.mkdir()
        self.write_config(self.config)
        self.write_weights(self.state)
        # Explicit local loading may not silently fall back to the Hub.
        network_guard = mock.patch('socket.socket.connect', side_effect=AssertionError('network access attempted'))
        network_guard.start()
        self.addCleanup(network_guard.stop)

    def write_config(self, config):
        (self.asset / 'config.json').write_text(json.dumps(config))

    def write_weights(self, state):
        save_file(state, str(self.asset / 'model.safetensors'))

    def load(self, **kwargs):
        arguments = dict(tokenizer=self.tokenizer, mask_index=50257,
                         revision='teacher_400k_large_step_400000',
                         local_path=str(self.asset), local_files_only=True,
                         feature_layer=1)
        arguments.update(kwargs)
        return load_sdtt_feature_backbone('jdeschena/sdtt', 'cpu', **arguments)

    def test_valid_checkpoint_is_frozen_in_eval_and_load_preserves_rng(self):
        state = torch.get_rng_state().clone()
        loaded = self.load()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(loaded.training)
        self.assertTrue(all(not module.training for module in loaded.modules()))
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_features_match_actual_dit_zero_conditioning_and_preserve_rng(self):
        loaded = self.load(feature_layer=[-1, 0, 1])
        tokens = torch.tensor([[3, 15, 50257, 50256], [50256, 19, 1, 8]])
        with torch.no_grad(), mmd_reward.capture_layer_hidden(self.reference, [-1, 0, 1]) as (expected, _):
            self.reference(tokens, torch.zeros(2))
        with torch.no_grad(), mmd_reward.capture_layer_hidden(self.reference, 1) as (time_dependent, _):
            self.reference(tokens, torch.ones(2))
        self.assertFalse(torch.equal(time_dependent[1], expected[1]),
                         'fixture must be sensitive to incorrect nonzero timestep conditioning')
        rng = torch.get_rng_state().clone()
        with mmd_reward.capture_layer_hidden(loaded, [-1, 0, 1]) as (actual, _):
            output = loaded(tokens, torch.tensor([.3, 7.]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertFalse(output.requires_grad)
        for layer in (-1, 0, 1):
            self.assertTrue(torch.equal(actual[layer], expected[layer]), f'layer {layer}')
        self.assertTrue(torch.equal(output, expected[1]))

    @unittest.skipUnless(os.environ.get('MMD_SDTT_TEST_CUDA') == '1',
                         'opt-in GPU smoke: set MMD_SDTT_TEST_CUDA=1')
    def test_cuda_bf16_features_match_and_preserve_rng(self):
        """Tiny real CUDA DiT only; the default suite remains CPU-only."""
        self.assertTrue(torch.cuda.is_available(), 'CUDA smoke requested but CUDA is unavailable')
        device = torch.device('cuda:0')
        # A private CPU generator creates reproducible physically masked input
        # and nonzero sigma without touching the global CPU/CUDA generators.
        generator = torch.Generator(device='cpu').manual_seed(891)
        tokens = torch.randint(0, 50257, (2, 8), generator=generator)
        tokens[0, 2], tokens[1, 5] = 50257, 50257
        tokens = tokens.to(device)
        sigma = (.2 + torch.rand(2, generator=generator)).to(device)
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()

        loaded = self.load(feature_layer=[-1, 0, 1]).to(device)
        reference = copy.deepcopy(self.reference).requires_grad_(False).eval().to(device)
        # Rotary caches are plain attributes in the archived model; remove a
        # possible CPU cache inherited from earlier CPU test methods.
        reference.rotary_emb.seq_len_cached = None
        reference.rotary_emb.cos_cached = None
        reference.rotary_emb.sin_cached = None
        with torch.no_grad(), mmd_reward.capture_layer_hidden(reference, [-1, 0, 1]) as (expected, _):
            reference(tokens, torch.zeros(2, device=device))
        with mmd_reward.capture_layer_hidden(loaded, [-1, 0, 1]) as (actual, _):
            output = loaded(tokens, sigma)
        torch.cuda.synchronize(device)

        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
        self.assertFalse(output.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        self.assertTrue(all(not module.training for module in loaded.modules()))
        for layer in (-1, 0, 1):
            self.assertTrue(torch.equal(actual[layer], expected[layer]), f'CUDA BF16 layer {layer}')
        self.assertTrue(torch.equal(output, expected[1]))

    def test_parent_train_cannot_reactivate_teacher_dropout(self):
        loaded = self.load()
        parent = torch.nn.Module()
        parent.feature_model = loaded
        parent.train()
        self.assertFalse(loaded.training)
        self.assertTrue(all(not block.training for block in loaded.modules()))

    def test_feature_prefix_does_not_execute_later_blocks_or_vocabulary_head(self):
        loaded = self.load(feature_layer=0)
        with mock.patch.object(loaded.backbone.blocks[1], 'forward', side_effect=AssertionError('unused later block')):
            with mock.patch.object(loaded.backbone.output_layer, 'forward', side_effect=AssertionError('unused vocabulary head')):
                result = loaded(torch.tensor([[1, 2, 3]]), torch.ones(1))
        self.assertEqual(tuple(result.shape), (1, 3, 16))

    def test_embedding_only_features_skip_all_transformer_blocks(self):
        loaded = self.load(feature_layer=-1)
        tokens = torch.tensor([[1, 2, 50257]])
        with mock.patch.object(loaded.backbone.blocks[0], 'forward', side_effect=AssertionError('embedding-only path used blocks')):
            result = loaded(tokens, torch.ones(1))
        self.assertTrue(torch.equal(result, loaded.backbone.vocab_embed(tokens)))

    def test_actual_reward_plumbing_accepts_features_without_changing_kernel(self):
        loaded = self.load(feature_layer=[0, 1])
        token_rows = torch.tensor([[1, 2, 3, 4], [5, 7, 11, 19]])
        draws = torch.stack((token_rows.roll(1, dims=1), token_rows.roll(2, dims=1)))
        selected = torch.ones_like(token_rows, dtype=torch.bool)
        outputs = []
        start = torch.get_rng_state().clone()
        for core in (self.reference, loaded):
            critic = mmd_reward.FrozenBackboneFeatures(core, 50257, layer_index=[0, 1],
                                                       layer_weights=[.25, .75], noise_level=0.)
            critic.kernel = 'rbf'
            critic.rbf_alpha = 2e-5
            critic.rbf_exclude_same_sentence = True
            critic.rbf_exclude_same_position = True
            torch.set_rng_state(start)
            reward, active, per_layer = critic.multi_draw_reward(
                token_rows, draws, selected, 50256, return_per_layer=True)
            outputs.append((reward, active, per_layer, torch.get_rng_state().clone()))
        for pos in (0, 1, 3):
            self.assertTrue(torch.equal(outputs[0][pos], outputs[1][pos]))
        self.assertTrue(torch.isfinite(outputs[1][0]).all())
        self.assertEqual(tuple(outputs[1][0].shape), (2,))
        for layer in (0, 1):
            self.assertTrue(torch.equal(outputs[0][2][layer], outputs[1][2][layer]))

    def test_requested_feature_layer_is_validated(self):
        for layer in (-2, 2, [], [0, 0]):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                self.load(feature_layer=layer)

    def test_context_limit_is_enforced(self):
        loaded = self.load()
        with self.assertRaises(ValueError):
            loaded(torch.zeros(1, 17, dtype=torch.long), torch.zeros(1))

    def test_padding_alias_for_mask_id_is_allowed(self):
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.assertEqual(tokenizer.pad_token_id, 50257)
        self.assertIsNotNone(self.load(tokenizer=tokenizer))

    def test_wrong_student_mask_id_is_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(mask_index=49153)

    def test_same_vocab_size_with_permuted_ids_is_rejected(self):
        vocab = self.tokenizer.get_vocab().copy()
        first, second = self.tokenizer.convert_ids_to_tokens([0, 1])
        vocab[first], vocab[second] = vocab[second], vocab[first]
        tokenizer = VocabularyOverride(self.tokenizer, vocab)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(tokenizer=tokenizer)

    def test_unrelated_added_token_is_rejected(self):
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_special_tokens({'pad_token': '[UNRELATED]'})
        with self.assertRaises(ValueError):
            self.load(tokenizer=tokenizer)

    def test_noise_conditioning_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['time_conditioning'] = True
        self.write_config(config)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_causal_attention_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['algo'] = {'causal_attention': True}
        self.write_config(config)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_unknown_backbone_type_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['model']['type'] = 'not-a-dit'
        self.write_config(config)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_wrong_checkpoint_tokenizer_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['tokenizer']['name'] = 'bert-base-uncased'
        self.write_config(config)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_missing_checkpoint_key_is_not_partially_loaded(self):
        state = dict(self.state)
        state.pop('backbone.blocks.1.attn_qkv.weight')
        self.write_weights(state)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_wrong_weight_shape_is_not_partially_loaded(self):
        state = dict(self.state)
        state['backbone.blocks.0.attn_qkv.weight'] = torch.zeros(3, 3)
        self.write_weights(state)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_unsupported_tensor_dtype_is_rejected_without_casting(self):
        state = dict(self.state)
        state['backbone.blocks.0.attn_qkv.weight'] = state['backbone.blocks.0.attn_qkv.weight'].to(torch.bfloat16)
        self.write_weights(state)
        with self.assertRaises(ValueError):
            self.load()

    def test_compiled_backbone_uniform_prefix_is_supported(self):
        state = {'backbone._orig_mod.' + key[len('backbone.'):]: value
                 for key, value in self.state.items()}
        self.write_weights(state)
        loaded = self.load()
        self.assertTrue(torch.equal(loaded.backbone.vocab_embed.embedding,
                                    self.reference.vocab_embed.embedding))

    def test_failed_weight_validation_also_preserves_rng(self):
        state = dict(self.state)
        state.pop('backbone.blocks.0.attn_qkv.weight')
        self.write_weights(state)
        rng = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            self.load()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_corrupt_safetensors_file_is_rejected(self):
        (self.asset / 'model.safetensors').write_bytes(b'not a safetensors checkpoint')
        from safetensors import SafetensorError
        with self.assertRaises((SafetensorError, ValueError)):
            self.load()

    def test_unexpected_backbone_key_is_not_silently_ignored(self):
        state = dict(self.state)
        state['backbone.unexpected.weight'] = torch.zeros(1)
        self.write_weights(state)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load()

    def test_missing_weights_fail_locally(self):
        (self.asset / 'model.safetensors').unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_nonexistent_local_directory_never_uses_hub(self):
        with mock.patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('Hub fallback attempted')) as hub:
            with self.assertRaises(FileNotFoundError):
                self.load(local_path=str(self.root / 'missing'), local_files_only=False)
            hub.assert_not_called()

    def test_remote_loader_passes_model_revision_and_offline_flag(self):
        def file_for(filename, **kwargs):
            return str(self.asset / filename)

        with mock.patch('huggingface_hub.hf_hub_download', side_effect=file_for) as hub:
            result = load_sdtt_feature_backbone('test/teacher', 'cpu', tokenizer=self.tokenizer,
                                                mask_index=50257, revision='immutable-test-revision',
                                                local_files_only=True, feature_layer=1)
        self.assertEqual(hub.call_count, 2)
        for filename in ('config.json', 'model.safetensors'):
            hub.assert_any_call(filename=filename, repo_id='test/teacher',
                                revision='immutable-test-revision', local_files_only=True)
        self.assertEqual(result.checkpoint_info['revision'], 'immutable-test-revision')


if __name__ == '__main__':
    unittest.main(verbosity=2)
