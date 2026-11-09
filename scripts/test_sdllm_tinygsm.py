"""CPU integration checks for TinyGSM rewards from native SDLLM features.

Uses the cached canonical SmolLM tokenizer and an actual tiny DiT/native
checkpoint. No model downloads, Lightning initialization, or GPUs are needed.
"""
from __future__ import annotations

import ast
import base64
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import mmd_reward
import mmd_sdllm_features
from mmd_sdllm_reward import FrozenSdllmFeatures
import test_sdllm_features as loader_fixtures
import test_sdllm_feature_routing as routing_fixtures
from test_sdllm_reward import dense_cross_sentence


PAD, MASK = 49152, 49153


class SourceTokenizer:
    """The production VocabSizeTokenizerWrapper's relevant public contract."""
    def __init__(self, tokenizer):
        self.backend = tokenizer

    @property
    def vocab_size(self):
        return len(self.backend)

    def __len__(self):
        return len(self.backend)

    def __getattr__(self, name):
        return getattr(self.backend, name)


class TinyGsmSdllmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        loader_fixtures.SDLLMFeatureLoaderTests.setUpClass()
        tokenizer = AutoTokenizer.from_pretrained(
            'HuggingFaceTB/SmolLM-135M', local_files_only=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        cls.source = SourceTokenizer(tokenizer)
        assert cls.source.vocab_size == MASK
        assert cls.source.pad_token_id == PAD
        assert cls.source.bos_token_id == cls.source.eos_token_id == 0

    def setUp(self):
        # Reuse the real tiny checkpoint fixture without inheriting and
        # rerunning the entire independent loader test suite.
        self.fixture = loader_fixtures.SDLLMFeatureLoaderTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def load(self, layers=(0, 1)):
        return self.fixture.load(tokenizer=self.source, mask_index=MASK,
                                 feature_layer=list(layers))

    def encode(self, text):
        return self.source.encode(text, add_special_tokens=False)

    @staticmethod
    def native_text(text):
        # The fixture has exactly the 256 byte ranks, shifted by 1000.
        return [1000 + value for value in text.encode('utf-8')]

    def batch(self, examples, padding=3):
        rows, starts, stops = [], [], []
        for prompt, answer in examples:
            prefix = [0] + self.encode(prompt)
            row = prefix + self.encode(answer) + [0]
            rows.append(row)
            starts.append(len(prefix))
            stops.append(len(row))
        width = max(stops) + padding
        tokens = torch.tensor([row + [PAD] * (width - len(row)) for row in rows])
        selected = torch.zeros_like(tokens, dtype=torch.bool)
        for i, (start, stop) in enumerate(zip(starts, stops)):
            selected[i, start:stop] = True
        return tokens, selected, starts, stops

    def critic(self, loaded=None, layers=(0, 1), alpha=.07):
        loaded = self.load(layers) if loaded is None else loaded
        critic = FrozenSdllmFeatures(loaded, self.source, MASK,
                                     layer_index=list(layers), noise_level=0.)
        critic.kernel = 'rbf'
        critic.kernel_level = 'token'
        critic.kernel_batch_size = 2
        critic.rbf_alpha = alpha
        critic.rbf_exclude_same_sentence = True
        critic.rbf_exclude_same_position = True
        critic.forward_batch_size = 2
        return critic

    def test_canonical_wrapped_smol_vocabulary_is_accepted_and_checked(self):
        self.assertEqual(mmd_sdllm_features.validate_source_tokenizer(
            self.source, MASK), 'tinygsm')
        for mask in (MASK - 1, MASK + 1):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                mmd_sdllm_features.validate_source_tokenizer(self.source, mask)
        vocab = self.source.get_vocab().copy()
        ordinary = [key for key, value in vocab.items() if 100 <= value < 200]
        left, right = ordinary[:2]
        vocab[left], vocab[right] = vocab[right], vocab[left]
        with mock.patch.object(self.source, 'get_vocab', return_value=vocab):
            with self.assertRaises(ValueError):
                mmd_sdllm_features.validate_source_tokenizer(self.source, MASK)
        with mock.patch.object(self.source, 'bos_token_id', 1):
            with self.assertRaises(ValueError):
                mmd_sdllm_features.validate_source_tokenizer(self.source, MASK)

    def test_actual_factory_checks_conditional_data_flags_before_loading(self):
        for changes in ({'wrap': True}, {'train_on_prompt': True}):
            model = routing_fixtures.make_model()
            model.config.data = {'train': 'tiny_gsm', 'wrap': False,
                                 'train_on_prompt': False, **changes}
            with self.subTest(changes=changes):
                with routing_fixtures.reject_sdllm_import(), self.assertRaises(ValueError):
                    routing_fixtures.legacy_routing.construct(model)
        model = routing_fixtures.make_model()
        model.config.data = {'train': 'tiny_gsm', 'wrap': False, 'train_on_prompt': False}
        model.tokenizer, model.mask_index = self.source, MASK
        loaded = self.load()
        with mock.patch.object(mmd_sdllm_features, 'load_sdllm_feature_model',
                               return_value=loaded) as loader:
            actual = routing_fixtures.legacy_routing.construct(model)
        self.assertEqual(loader.call_count, 1)
        self.assertIs(actual.denoiser, loaded)
        self.assertEqual(actual.source_family, 'tinygsm')

    def test_retokenized_boundary_preserves_whole_unicode_and_source_specials(self):
        prompt, answer = 'e\u0301\t\n', 'Я🙂'
        tokens, selected, _, _ = self.batch([(prompt, answer)])
        loaded = self.load()
        self.assertEqual(loaded.source_family, 'tinygsm')
        native, offsets = loaded.retokenize_conditioned(
            tokens, self.source, MASK, selected)
        prefix = [1] + self.native_text(prompt)
        suffix = self.native_text(answer) + [2]
        self.assertEqual(native, [prefix + suffix])
        self.assertEqual(offsets, [len(prefix)])
        self.assertEqual(native[0][offsets[0]:], suffix)

    def test_prompt_and_answer_tokenization_have_an_explicit_boundary(self):
        # Build real native BPE merges crossing the prompt/answer boundary:
        # encode("hello") is one token, but encode("hel") + encode("lo")
        # must retain a usable answer offset without byte-offset projection.
        additions = ['he', 'hel', 'hell', 'hello']
        records = self.fixture.byte_records + [
            {'rank': 256 + i, 'token_bytes': base64.b64encode(text.encode()).decode(),
             'token_str': text} for i, text in enumerate(additions)]
        self.fixture.write_tokenizer(records)
        with torch.random.fork_rng(devices=[]):
            backbone = loader_fixtures.DIT(
                {'model': self.fixture.model_config, 'algo': {'causal_attention': False}},
                1256 + len(additions))
        self.fixture.write_checkpoint({
            'backbone.' + key: value for key, value in backbone.state_dict().items()})
        tokens, selected, _, _ = self.batch([('hel', 'lo')])
        loaded = self.load()
        self.assertEqual(loaded.native_tokenizer.encode_ordinary('hello'), [1259])
        native, offsets = loaded.retokenize_conditioned(
            tokens, self.source, MASK, selected)
        self.assertEqual(native[0], [1, 1257] + self.native_text('lo') + [2])
        self.assertEqual(offsets, [2])
        self.assertEqual(native[0][offsets[0]:], self.native_text('lo') + [2])

    def test_candidate_eos_and_generated_pad_inside_answer_are_not_truncation(self):
        tokens, selected, starts, stops = self.batch([('Q?\n', 'hello world')])
        loaded = self.load()
        start, stop = starts[0], stops[0]
        self.assertGreaterEqual(stop - start, 3)
        candidate = tokens.clone()
        candidate[0, start] = 0
        candidate[0, start + 1] = PAD
        native, offsets = loaded.retokenize_conditioned(
            candidate, self.source, MASK, selected)
        self.assertEqual(native[0][offsets[0]:],
                         [2] + self.native_text('[PAD]') + [2])

    def test_trailing_padding_is_removed_before_token_validation(self):
        tokens, selected, _, stops = self.batch([('Q?\n', 'print(1)')])
        loaded = self.load()
        expected = loaded.retokenize_conditioned(tokens, self.source, MASK, selected)
        dirty_suffix = tokens.clone()
        dirty_suffix[0, stops[0]:] = torch.tensor([MASK, -123, 999999])
        self.assertEqual(loaded.retokenize_conditioned(
            dirty_suffix, self.source, MASK, selected), expected)
        with self.assertRaises(ValueError):
            loaded.retokenize(tokens, self.source, MASK)

    def test_partial_holey_or_empty_answer_windows_are_rejected(self):
        tokens, selected, starts, _ = self.batch([('Q?\n', 'print(123)')])
        loaded = self.load()
        masks = [torch.zeros_like(selected), selected[:, :-1], selected.to(torch.long)]
        hole = selected.clone()
        hole[0, starts[0] + 1] = False
        masks.append(hole)
        for invalid in masks:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    loaded.retokenize_conditioned(tokens, self.source, MASK, invalid)

    def test_source_mask_and_invalid_ids_inside_kept_context_are_rejected(self):
        tokens, selected, starts, _ = self.batch([('Q?\n', 'print(1)')])
        loaded = self.load()
        for position in (1, starts[0]):
            for value in (MASK, -1, MASK + 5):
                candidate = tokens.clone()
                candidate[0, position] = value
                with self.subTest(position=position, value=value):
                    with self.assertRaises(ValueError):
                        loaded.retokenize_conditioned(candidate, self.source, MASK, selected)

    def test_native_context_overflow_is_not_silently_truncated(self):
        tokens, selected, _, _ = self.batch([('question ' * 9, 'print(1)')])
        loaded = self.load()
        with self.assertRaisesRegex(ValueError, 'context|exceed'):
            loaded.retokenize_conditioned(tokens, self.source, MASK, selected)

    def test_answer_features_match_full_dit_forward_with_prompt_context(self):
        examples = [('Q?\n', 'print(1)'), ('Longer?\n', 'return 12')]
        tokens, selected, _, _ = self.batch(examples)
        loaded = self.load(layers=(-1, 0, 1))
        critic = self.critic(loaded=loaded, layers=(-1, 0, 1))
        actual = critic.features(tokens, selected=selected)
        for row, (prompt, answer) in enumerate(examples):
            prefix = [1] + self.native_text(prompt)
            answer_ids = self.native_text(answer) + [2]
            full = self.fixture.expected_features(prefix + answer_ids, [-1, 0, 1])
            for layer in (-1, 0, 1):
                self.assertEqual(tuple(actual[layer][row].shape), (len(answer_ids), 16))
                torch.testing.assert_close(actual[layer][row], full[layer][len(prefix):],
                                           atol=2e-7, rtol=1e-6)

    def test_prompt_changes_context_features_but_is_not_a_kernel_point(self):
        tokens, selected, _, _ = self.batch([('Q?\n', 'print(1)'), ('X?\n', 'print(1)')])
        critic = self.critic(layers=(-1, 1))
        features = critic.features(tokens, selected=selected)
        expected_length = len(self.native_text('print(1)')) + 1
        self.assertEqual(len(features[1][0]), expected_length)
        self.assertEqual(len(features[1][1]), expected_length)
        torch.testing.assert_close(features[-1][0], features[-1][1], atol=0, rtol=0)
        self.assertGreater((features[1][0] - features[1][1]).abs().max().item(), 1e-8)

    def test_padding_and_physical_chunk_size_do_not_change_answer_rewards(self):
        examples = [('Q?\n', 'hello world'), ('Question?\n', 'print(1)')]
        tokens, selected, starts, _ = self.batch(examples)
        draws = tokens.unsqueeze(0).repeat(2, 1, 1)
        draws[0, 0, starts[0]] = self.encode('x')[0]
        draws[1, 0, starts[0]] = self.encode('world')[0]
        draws[1, 1, starts[1]] = self.encode('code')[0]
        critic = self.critic()
        expected, active = critic.multi_draw_reward(tokens, draws, selected, eos_id=0)
        self.assertTrue(bool(active.all()))
        extended_tokens = torch.nn.functional.pad(tokens, (0, 7), value=PAD)
        extended_draws = torch.nn.functional.pad(draws, (0, 7), value=PAD)
        extended_selected = torch.nn.functional.pad(selected, (0, 7), value=False)
        for chunk in (1, 2, 10):
            critic.forward_batch_size = chunk
            actual, actual_active = critic.multi_draw_reward(
                extended_tokens, extended_draws, extended_selected, eos_id=0)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
            self.assertTrue(torch.equal(actual_active, active))

    def test_native_answer_reward_matches_independent_full_forward_dense_oracle(self):
        tokens, selected, starts, _ = self.batch([('Q?\n', 'hello world')])
        loaded = self.load(layers=(1,))
        critic = self.critic(loaded=loaded, layers=(1,))
        draws = tokens.unsqueeze(0).repeat(2, 1, 1)
        draws[0, 0, starts[0]] = self.encode('x')[0]
        draws[1, 0, starts[0]] = self.encode('code')[0]

        def oracle(source_row):
            start, stop = starts[0], int(selected[0].nonzero()[-1]) + 1
            prompt_text = self.source.decode(source_row[1:start].tolist(),
                                             clean_up_tokenization_spaces=False)
            answer_text = self.source.decode(source_row[start:stop - 1].tolist(),
                                             clean_up_tokenization_spaces=False)
            prefix = [1] + self.native_text(prompt_text)
            native = prefix + self.native_text(answer_text) + [2]
            return self.fixture.expected_features(native, [1])[1][len(prefix):]

        expected = -dense_cross_sentence(oracle(tokens[0]),
                                         [oracle(draw[0]) for draw in draws],
                                         critic.rbf_alpha, exclude_position=True)
        actual, active = critic.multi_draw_reward(tokens, draws, selected, eos_id=0)
        torch.testing.assert_close(actual[0].double(), expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(bool(active[0]))

    def test_actual_policy_backward_keeps_prompt_padding_and_unwritten_positions_zero(self):
        tokens, scored, starts, _ = self.batch([('Q?\n', 'hello world'), ('X?\n', 'hello world')])
        written = torch.zeros_like(scored)
        written[0, starts[0]:starts[0] + 2] = True
        written[1, starts[1]] = True
        samples = tokens.unsqueeze(0).repeat(4, 1, 1)  # G=2, K=2.
        for index, replacement in enumerate(('x', 'world', 'code', 'y')):
            samples[index, :, starts[0]] = self.encode(replacement)[0]
        samples[2:, 0, starts[0] + 1] = self.encode('!')[0]
        critic = self.critic(alpha=None)
        self.assertTrue(torch.equal(samples[:, ~written],
                                    tokens.unsqueeze(0).expand_as(samples)[:, ~written]))
        logits = torch.randn(*tokens.shape, MASK + 1,
                             generator=torch.Generator().manual_seed(731)).requires_grad_()
        sampled_logp = logits.log_softmax(-1).gather(
            2, samples.permute(1, 2, 0)).permute(2, 0, 1)
        # Exercise the real MDLM mask handoff as well as the reward/loss.
        # TinyGSM's train_on_pad=True mask includes the padded suffix; the
        # original content-mask method must remove it before native features.
        tree = ast.parse((ROOT / 'algo.py').read_text())
        original = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'MDLM')
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_mmd_content_mask', '_mmd_term')]
        cls = ast.ClassDef(name='ConditionalMmdHarness', bases=[], keywords=[],
                           body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        namespace = {'torch': torch, 'F': torch.nn.functional}
        exec(compile(module, str(ROOT / 'algo.py'), 'exec'), namespace)
        model = namespace['ConditionalMmdHarness']()
        model.tokenizer, model._mmd_critic = self.source, critic
        model.mask_index = MASK
        model.config = OmegaConf.create({'adversarial_distill': {
            'mmd_score_positions': 'all', 'mmd_group_size': 2,
            'mmd_reward_weight': .7, 'mmd_baseline': 'group',
            'mmd_kernel_source': 'draws', 'mmd_kernel_batch_size': 2,
            'mmd_reward_type': 'feature', 'mmd_sentence_pooling': 'mean'}})
        logged = {}
        model.log_dict = lambda stats, **kwargs: logged.update(stats)
        valid_tokens = torch.zeros_like(scored)
        for row, start in enumerate(starts):
            valid_tokens[row, start:] = True
        self.assertTrue(torch.equal(model._mmd_content_mask(tokens, valid_tokens), scored))
        noisy = torch.where(written, torch.full_like(tokens, MASK), tokens)
        with mock.patch.object(critic, 'multi_draw_reward', wraps=critic.multi_draw_reward) as calls:
            loss = model._mmd_term(tokens, valid_tokens, None, noisy,
                                   samples=samples, sampled_logp=sampled_logp)
        self.assertEqual(calls.call_count, 2)
        for call in calls.call_args_list:
            self.assertTrue(torch.equal(call.args[2], scored))
        stats = logged
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(stats['mmd/advantage_absmean'].item(), 0.)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(logits.grad[written].abs().max().item(), 0.)
        self.assertEqual(logits.grad[~written].abs().max().item(), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in critic.denoiser.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in critic.denoiser.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
