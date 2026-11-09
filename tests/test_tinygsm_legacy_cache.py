"""Reuse the original separator-tagged TinyGSM DatasetDict while offline."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import datasets
from omegaconf import OmegaConf
import torch

import dataloader


class OfflineRawData(RuntimeError):
    pass


class LegacyTinyGSMCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tinygsm legacy cache ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = OmegaConf.create({
            'data': {
                'train': 'tiny_gsm',
                'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M',
                'cache_dir': str(self.root),
                'separator': r'\n',
                'train_on_prompt': False,
                'filter_too_long': True,
                'wrap': False,
                'train_on_pad': True,
            },
            'model': {'length': 512},
        })
        # These names are the artifacts produced by the historical IDLM job;
        # sep5c6e identifies its literal backslash + n separator.
        self.current = self.root / (
            'tiny_gsm_bs512_answer_only_filtered_train_on_pad_'
            'HuggingFaceTB__SmolLM-135M')
        self.legacy = Path(str(self.current) + '_sep5c6e')
        self.tokenizer = object()  # A cache hit must not invoke tokenization.
        self.raw_load = self.enterContext(patch(
            'dataloader.datasets.load_dataset',
            side_effect=OfflineRawData('Raw TinyGSM is unavailable offline.')))

    @staticmethod
    def fixture(marker=17):
        def row(value):
            return {'input_ids': [0, value] + [1] * 510,
                    'attention_mask': [0, 0] + [1] * 510,
                    'prompt_len': 2}
        return datasets.DatasetDict({
            'train': datasets.Dataset.from_list([row(marker), row(marker + 1)]),
            'validation': datasets.Dataset.from_list([row(marker + 2)]),
        })

    def assert_fixture(self, actual, expected):
        self.assertEqual(set(actual), {'train', 'validation'})
        for split in expected:
            self.assertEqual(actual[split].to_dict(), expected[split].to_dict())

    def test_matching_separator_tag_loads_both_cached_splits_without_hub(self):
        expected = self.fixture()
        expected.save_to_disk(str(self.legacy))
        actual = dataloader.get_tiny_gsm_dataset(self.config, self.tokenizer)
        self.assert_fixture(actual, expected)
        self.raw_load.assert_not_called()

    def test_other_separator_does_not_reuse_literal_backslash_n_cache(self):
        self.fixture().save_to_disk(str(self.legacy))
        self.config.data.separator = '\n'
        with patch('dataloader.datasets.load_from_disk') as cached_load:
            with self.assertRaises(OfflineRawData):
                dataloader.get_tiny_gsm_dataset(self.config, self.tokenizer)
        cached_load.assert_not_called()
        self.raw_load.assert_called_once_with(
            'TinyGSM/TinyGSM', split='train', cache_dir=str(self.root))

    def test_incomplete_tagged_cache_is_not_treated_as_a_datasetdict(self):
        self.fixture()['train'].save_to_disk(str(self.legacy / 'train'))
        with patch('dataloader.datasets.load_from_disk') as cached_load:
            with self.assertRaises(OfflineRawData):
                dataloader.get_tiny_gsm_dataset(self.config, self.tokenizer)
        cached_load.assert_not_called()
        self.raw_load.assert_called_once()

    def test_current_untagged_cache_still_loads_without_hub(self):
        expected = self.fixture(marker=31)
        expected.save_to_disk(str(self.current))
        actual = dataloader.get_tiny_gsm_dataset(self.config, self.tokenizer)
        self.assert_fixture(actual, expected)
        self.raw_load.assert_not_called()

    def test_current_cache_keeps_priority_if_both_names_exist(self):
        expected = self.fixture(marker=41)
        expected.save_to_disk(str(self.current))
        self.fixture(marker=51).save_to_disk(str(self.legacy))
        actual = dataloader.get_tiny_gsm_dataset(self.config, self.tokenizer)
        self.assert_fixture(actual, expected)
        self.raw_load.assert_not_called()

    def test_get_dataset_restores_train_and_validation_as_unchanged_tensors(self):
        expected = self.fixture(marker=61)
        expected.save_to_disk(str(self.legacy))
        for split in ['train', 'validation']:
            with self.subTest(split=split), contextlib.redirect_stdout(io.StringIO()):
                actual = dataloader.get_dataset(
                    'tiny_gsm', self.tokenizer, wrap=False, mode=split,
                    cache_dir=str(self.root), block_size=512, config=self.config)
                self.assertEqual(len(actual), len(expected[split]))
                for index, row in enumerate(expected[split]):
                    for column in ['input_ids', 'attention_mask', 'prompt_len']:
                        value = actual[index][column]
                        self.assertIsInstance(value, torch.Tensor)
                        self.assertTrue(torch.equal(value, torch.tensor(row[column])))
        self.raw_load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
