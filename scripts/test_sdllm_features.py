"""Offline SDLLM feature-loader checks using a real tiny DiT checkpoint.

Run with the repository environment and tiktoken available. The canonical
GPT-2 vocabulary is read from the existing local Hugging Face cache. No model
weights are downloaded and no GPU is used.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

import torch
from transformers import AutoTokenizer
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dit import DIT
import mmd_reward
from mmd_sdllm_features import load_sdllm_feature_model


MODEL_ID = 'nvidia/SDLLM-MDLM-1.7B-Base'
REVISION = '4007e0b2a23465a31adf282d843a61172932af2a'
VOCAB = 1256  # 1000 reserved slots plus a complete 256-byte BPE vocabulary.


class SDLLMFeatureLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained('gpt2-large', local_files_only=True)
        cls.model_config = {
            'name': 'tiny-native-checkpoint', 'type': 'ddit',
            'hidden_size': 16, 'n_blocks': 2, 'n_heads': 2,
            'cond_dim': 8, 'length': 64, 'dropout': .1,
            'scale_by_sigma': True, 'tie_word_embeddings': False,
        }
        cls.config = {
            'algo': {'name': 'mdlm', 'backbone': 'dit_legacy', 'adaLN': True,
                     'causal_attention': False, 'time_conditioning': False},
            'model': copy.deepcopy(cls.model_config),
            'data': {'train': 'nvidia',
                     'tokenizer_name_or_path': 'tokenizer/tiktokenizer.json'},
            'eval': {'disable_ema': True}, 'training': {'ema': -1},
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9473)
            cls.reference = DIT({'model': cls.model_config,
                                 'algo': {'causal_attention': False}}, VOCAB).eval()
            with torch.no_grad():
                # Zero-init gates/head would make several incorrect forwards
                # observationally identical; use a genuinely nontrivial model.
                for name, parameter in cls.reference.named_parameters():
                    if 'adaLN_modulation' in name or name.startswith('output_layer.linear'):
                        parameter.normal_(mean=0., std=.04)
        cls.native_state = {
            'backbone.' + name: value.detach().clone()
            for name, value in cls.reference.state_dict().items()}
        cls.byte_records = [
            {'rank': byte, 'token_bytes': base64.b64encode(bytes([byte])).decode('ascii'),
             'token_str': bytes([byte]).decode('utf-8', errors='replace')}
            for byte in range(256)]

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='sdllm feature test ')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.asset = self.root / 'native release'
        (self.asset / 'tokenizer').mkdir(parents=True)
        self.write_config(self.config)
        self.write_tokenizer(self.byte_records)
        self.write_checkpoint(self.native_state)
        network_guard = mock.patch.object(socket.socket, 'connect',
                                          side_effect=AssertionError('network access attempted'))
        network_guard.start()
        self.addCleanup(network_guard.stop)

    def write_config(self, value):
        (self.asset / 'config.yaml').write_text(yaml.safe_dump(value))

    def write_tokenizer(self, records):
        (self.asset / 'tokenizer/tiktokenizer.json').write_text(json.dumps(records))

    def write_checkpoint(self, state, **metadata):
        torch.save({'state_dict': state, **metadata}, self.asset / 'model.ckpt')

    def load(self, **overrides):
        kwargs = dict(tokenizer=self.tokenizer, mask_index=50257,
                      revision=REVISION, local_path=str(self.asset),
                      local_files_only=True, feature_layer=1)
        kwargs.update(overrides)
        return load_sdllm_feature_model(MODEL_ID, 'cpu', **kwargs)

    @staticmethod
    def byte_tokens(text):
        return [1000 + value for value in text.encode('utf-8')]

    def source_ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def expected_features(self, tokens, layers):
        ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad(), mmd_reward.capture_layer_hidden(self.reference, layers) as (features, _):
            self.reference(ids, torch.zeros(1))
        return {layer: value[0].detach().clone() for layer, value in features.items()}

    def test_native_weights_load_strictly_and_stay_frozen_without_rng_change(self):
        before = torch.get_rng_state().clone()
        loaded = self.load()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(loaded.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.backbone.parameters()))
        self.assertEqual(set(loaded.backbone.state_dict()), set(self.reference.state_dict()))
        for key, value in loaded.backbone.state_dict().items():
            torch.testing.assert_close(value, self.reference.state_dict()[key], rtol=0, atol=0)

    def test_actual_post_block_features_match_full_archived_dit(self):
        loaded = self.load(feature_layer=[-1, 0, 1])
        token_lists = [[1, 1104, 1105, 2], [1, 1100, 1111, 1109, 1103, 2]]
        expected = [self.expected_features(row, [-1, 0, 1]) for row in token_lists]
        rng = torch.get_rng_state().clone()
        actual = loaded.features(token_lists, [-1, 0, 1])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(actual), {-1, 0, 1})
        for layer in (-1, 0, 1):
            self.assertEqual(len(actual[layer]), len(token_lists))
            for row, oracle in zip(actual[layer], expected):
                torch.testing.assert_close(row, oracle[layer], rtol=0, atol=0)
                self.assertFalse(row.requires_grad)

    def test_variable_lengths_and_physical_forward_chunks_match_individual_rows(self):
        loaded = self.load(feature_layer=[0, 1])
        rows = [[1, 1100, 2], [1, 1104, 1105, 1106, 2],
                [1, 1111, 1112, 2], [1, 1120, 2],
                [1, 1121, 1122, 1123, 2]]
        individual = [loaded.features([row], [0, 1]) for row in rows]
        for batch_size in (None, 1, 2, 3, 10):
            with self.subTest(batch_size=batch_size):
                actual = loaded.features(rows, [0, 1], forward_batch_size=batch_size)
                for layer in (0, 1):
                    for position, row in enumerate(rows):
                        self.assertEqual(tuple(actual[layer][position].shape), (len(row), 16))
                        torch.testing.assert_close(actual[layer][position], individual[position][layer][0],
                                                   rtol=1e-6, atol=2e-7)

    def test_real_loader_native_reward_and_score_estimator_backpropagate_only_to_policy(self):
        """Real retokenization, teacher features, ragged kernels and policy loss."""
        from mmd_sdllm_reward import FrozenSdllmFeatures

        loaded = self.load(feature_layer=[0, 1])
        critic = FrozenSdllmFeatures(
            loaded, self.tokenizer, 50257, layer_index=[0, 1],
            layer_weights=[.25, .75], noise_level=0.)
        critic.kernel = 'rbf'
        critic.kernel_level = 'token'
        critic.kernel_batch_size = 2
        critic.rbf_exclude_same_sentence = True
        critic.rbf_exclude_same_position = True
        critic.rbf_alpha = None
        critic.forward_batch_size = 2

        def one(text):
            ids = self.source_ids(text)
            self.assertEqual(len(ids), 1, f'fixture must be one source token: {text!r}')
            return ids[0]

        clean = torch.tensor([[50256, one('hello'), one(' world'), 50256],
                              [50256, one('hi'), one(' there'), 50256]])
        written = torch.tensor([[False, True, True, False],
                                [False, True, False, False]])
        scored = torch.ones_like(written)
        samples = clean.unsqueeze(0).repeat(4, 1, 1)  # G=2 candidates, K=2 draws.
        samples[1, :, 1] = one('hi')
        samples[2, :, 1] = one(' world')
        samples[2, 0, 2] = one('!')
        samples[3, :, 1] = one('x')
        samples[3, 0, 2] = one('!')
        self.assertTrue(torch.equal(samples[:, ~written],
                                    clean.unsqueeze(0).expand_as(samples)[:, ~written]))
        native = loaded.retokenize(
            torch.cat([clean, samples.reshape(-1, clean.shape[1])]), self.tokenizer, 50257)
        self.assertGreater(len({len(row) for row in native}), 1,
                           'fixture must exercise genuinely ragged native sequences')

        reward, active, breakdown = critic.multi_draw_reward(
            clean, samples[:2], scored, 50256, return_per_layer=True)
        self.assertEqual(tuple(reward.shape), (2,))
        self.assertTrue(bool(active.all()))
        self.assertTrue(bool(torch.isfinite(reward).all()))
        self.assertEqual(set(breakdown), {0, 1})

        generator = torch.Generator().manual_seed(8081)
        logits = torch.randn(2, 4, 50258, generator=generator).requires_grad_()
        log_probabilities = logits.log_softmax(-1)
        sampled_logp = log_probabilities.gather(
            2, samples.permute(1, 2, 0)).permute(2, 0, 1)
        loss, stats = mmd_reward.mmd_reinforce_loss_from_samples(
            sampled_logp, clean, written, samples, critic,
            group_size=2, eos_id=50256, baseline='group', reward_weight=.7,
            scored=scored, kernel_source='draws', kernel_draws=2)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(stats['mmd/advantage_absmean'].item(), 0.)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(logits.grad[written].abs().max().item(), 0.)
        self.assertEqual(logits.grad[~written].abs().max().item(), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in loaded.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))

    def test_prefix_features_do_not_execute_later_blocks_or_vocabulary_head(self):
        loaded = self.load(feature_layer=0)
        with mock.patch.object(loaded.backbone.blocks[1], 'forward',
                               side_effect=AssertionError('unused later block executed')):
            with mock.patch.object(loaded.backbone.output_layer, 'forward',
                                   side_effect=AssertionError('unused vocabulary head executed')):
                actual = loaded.features([[1, 1110, 2]], [0])
        self.assertEqual(tuple(actual[0][0].shape), (3, 16))

    def test_embedding_features_do_not_execute_transformer_blocks(self):
        loaded = self.load(feature_layer=-1)
        ids = [1, 1101, 2]
        with mock.patch.object(loaded.backbone.blocks[0], 'forward',
                               side_effect=AssertionError('embedding mode executed a block')):
            actual = loaded.features([ids], [-1])[-1][0]
        oracle = loaded.backbone.vocab_embed(torch.tensor([ids]))[0]
        torch.testing.assert_close(actual, oracle, rtol=0, atol=0)

    def test_feature_calls_keep_teacher_eval_and_leave_no_temporary_hooks(self):
        loaded = self.load()
        if callable(getattr(loaded, 'train', None)):
            loaded.train()
        loaded.features([[1, 1100, 2]], [1])
        self.assertTrue(all(not module.training for module in loaded.backbone.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.backbone.parameters()))
        self.assertTrue(all(not block._forward_hooks for block in loaded.backbone.blocks))

    def test_retokenizes_whole_unicode_segments_without_per_token_replacement(self):
        loaded = self.load()
        # Each isolated GPT-2 piece of this sequence decodes badly, but their
        # concatenation is valid UTF-8. This catches token-by-token decoding.
        text = 'Я🙂'
        ids = self.source_ids(text)
        self.assertEqual(ids, [140, 107, 8582, 25081])
        self.assertNotEqual(''.join(self.tokenizer.decode([x]) for x in ids), text)
        rng = torch.get_rng_state().clone()
        result = loaded.retokenize(torch.tensor([ids]), self.tokenizer, 50257)
        self.assertEqual(result, [self.byte_tokens(text)])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_whitespace_and_combining_marks_are_not_cleaned_or_normalized(self):
        loaded = self.load()
        for text in ('  hello', 'a\n\nb', 'café', 'e\u0301', '\t x  '):
            with self.subTest(text=text):
                ids = self.source_ids(text)
                result = loaded.retokenize(torch.tensor([ids]), self.tokenizer, 50257)
                self.assertEqual(result, [self.byte_tokens(text)])

    def test_student_bos_and_interior_eos_map_to_distinct_native_special_ids(self):
        loaded = self.load()
        first, second = self.source_ids('hello'), self.source_ids(' world')
        row = [50256, *first, 50256, *second, 50256]
        result = loaded.retokenize(torch.tensor([row]), self.tokenizer, 50257)
        self.assertEqual(result, [[1, *self.byte_tokens('hello'), 2,
                                   *self.byte_tokens(' world'), 2]])

    def test_literal_teacher_special_spellings_are_encoded_as_ordinary_text(self):
        loaded = self.load()
        for text in ('<mask>', '<s>', '</s>', '<SPECIAL_10>'):
            with self.subTest(text=text):
                ids = self.source_ids(text)
                result = loaded.retokenize(torch.tensor([ids]), self.tokenizer, 50257)
                self.assertEqual(result, [self.byte_tokens(text)])
                self.assertTrue(all(token >= 1000 for token in result[0]))

    def test_source_mask_and_unknown_token_ids_fail_explicitly(self):
        loaded = self.load()
        for ids in ([50257], [-1], [50258]):
            with self.subTest(ids=ids), self.assertRaises((ValueError, IndexError)):
                loaded.retokenize(torch.tensor([ids]), self.tokenizer, 50257)

    def test_context_overflow_is_rejected_instead_of_truncating(self):
        loaded = self.load()
        with self.assertRaises(ValueError):
            loaded.features([[1000] * 65], [1])

    def test_invalid_feature_layers_are_rejected(self):
        for layer in (-2, 2, [], [0, 0]):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                self.load(feature_layer=layer)

    def test_student_wrong_tokenizer_or_mask_is_rejected_before_weight_loading(self):
        with self.assertRaises(ValueError):
            self.load(mask_index=49153)
        wrong = copy.deepcopy(self.tokenizer)
        vocab = wrong.get_vocab()
        first, second = wrong.convert_ids_to_tokens([0, 1])
        vocab[first], vocab[second] = vocab[second], vocab[first]
        class BadMapping:
            def __getattr__(self, key):
                return getattr(wrong, key)
            def get_vocab(self):
                return vocab
        with self.assertRaises(ValueError):
            self.load(tokenizer=BadMapping())

    def test_native_weights_are_used_without_implicitly_selecting_ema(self):
        wrong_ema = {'shadow_params': [torch.full_like(parameter, -123)
                                       for parameter in self.reference.parameters()], 'decay': .9999}
        self.write_checkpoint(self.native_state, ema=wrong_ema)
        loaded = self.load()
        for key, actual in loaded.backbone.state_dict().items():
            torch.testing.assert_close(actual, self.reference.state_dict()[key], rtol=0, atol=0)

    def test_nonbackbone_training_state_does_not_replace_native_weights(self):
        state = dict(self.native_state)
        state['metrics.training_seen'] = torch.tensor(128)
        self.write_checkpoint(state, optimizer_states=[{'state': {}}], global_step=123)
        loaded = self.load()
        torch.testing.assert_close(loaded.backbone.vocab_embed.embedding,
                                   self.reference.vocab_embed.embedding, rtol=0, atol=0)

    def test_missing_and_unexpected_backbone_keys_are_not_partially_loaded(self):
        for change in ('missing', 'unexpected'):
            with self.subTest(change=change):
                state = dict(self.native_state)
                if change == 'missing':
                    state.pop('backbone.blocks.1.attn_qkv.weight')
                else:
                    state['backbone.unexpected.weight'] = torch.zeros(1)
                self.write_checkpoint(state)
                rng = torch.get_rng_state().clone()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.load()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_wrong_weight_shape_and_nonfloating_dtype_are_rejected(self):
        for tensor in (torch.zeros(3, 3), torch.zeros_like(
                self.native_state['backbone.blocks.0.attn_qkv.weight'], dtype=torch.int32)):
            state = dict(self.native_state)
            state['backbone.blocks.0.attn_qkv.weight'] = tensor
            self.write_checkpoint(state)
            with self.assertRaises((ValueError, RuntimeError)):
                self.load()

    def test_incomplete_corrupt_or_reordered_native_byte_vocabulary_is_rejected(self):
        bad_inputs = [self.byte_records[:-1]]
        for kind in ('rank', 'byte', 'base64'):
            records = copy.deepcopy(self.byte_records)
            if kind == 'rank':
                records[20]['rank'] = 21
            elif kind == 'byte':
                records[20]['token_bytes'] = records[21]['token_bytes']
            else:
                records[20]['token_bytes'] = '*invalid-base64*'
            bad_inputs.append(records)
        for index, records in enumerate(bad_inputs):
            with self.subTest(case=index):
                self.write_tokenizer(records)
                with self.assertRaises(ValueError):
                    self.load()

    def test_native_feature_tokens_cannot_contain_mask_or_out_of_range_ids(self):
        loaded = self.load()
        for tokens in ([1, 3, 2], [VOCAB], [-1], []):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                loaded.features([tokens], [1])

    def test_wrong_model_family_or_attention_configuration_is_rejected(self):
        for key, value in (('name', 'duo'), ('causal_attention', True),
                           ('time_conditioning', True), ('adaLN', False)):
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config['algo'][key] = value
                self.write_config(config)
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_local_assets_never_fall_back_to_hub(self):
        with mock.patch('huggingface_hub.hf_hub_download',
                        side_effect=AssertionError('unexpected Hub fallback')):
            for name in ('model.ckpt', 'config.yaml', 'tokenizer/tiktokenizer.json'):
                path = self.asset / name
                original = path.read_bytes()
                path.unlink()
                with self.subTest(asset=name), self.assertRaises(FileNotFoundError):
                    self.load(local_files_only=False)
                path.write_bytes(original)

    def test_remote_resolution_pins_all_files_to_the_same_hub_commit(self):
        snapshot = self.root / 'models--nvidia--SDLLM' / 'snapshots' / REVISION
        (snapshot / 'tokenizer').mkdir(parents=True)
        for name in ('config.yaml', 'model.ckpt', 'tokenizer/tiktokenizer.json'):
            (snapshot / name).write_bytes((self.asset / name).read_bytes())
        def file_for(filename, **kwargs):
            return str(snapshot / filename)
        with mock.patch('huggingface_hub.hf_hub_download', side_effect=file_for) as hub:
            loaded = load_sdllm_feature_model(MODEL_ID, 'cpu', tokenizer=self.tokenizer,
                                              mask_index=50257, revision='main',
                                              local_files_only=True, feature_layer=1)
        self.assertEqual(hub.call_count, 3)
        for call in hub.call_args_list:
            self.assertEqual(call.kwargs['repo_id'], MODEL_ID)
            self.assertTrue(call.kwargs['local_files_only'])
        self.assertEqual(hub.call_args_list[0].kwargs['revision'], 'main')
        for call in hub.call_args_list[1:]:
            self.assertEqual(call.kwargs['revision'], REVISION)
        self.assertIsNotNone(loaded)


if __name__ == '__main__':
    unittest.main(verbosity=2)
