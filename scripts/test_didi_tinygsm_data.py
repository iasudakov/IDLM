"""CPU-only TinyGSM cache/tokenizer/conditional-mask contract tests."""

import copy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import datasets
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from didi_repro import tinygsm_data as data


class FakeTokenizer:
  bos_token = '<eos>'
  eos_token = '<eos>'
  cls_token = None
  sep_token = None
  bos_token_id = 0
  eos_token_id = 0
  vocab_size = 256

  def __init__(self, with_pad=True):
    self.pad_token = '[PAD]' if with_pad else None
    self.pad_token_id = 256 if with_pad else None
    self.mask_token = '<eos>'

  @property
  def mask_token_id(self):
    return 0 if self.mask_token is not None else None

  def __len__(self):
    return 257 if self.pad_token is not None else 256

  def add_special_tokens(self, mapping):
    self.pad_token = mapping['pad_token']
    self.pad_token_id = 256

  def __call__(self, text, add_special_tokens=False):
    assert not add_special_tokens
    return types.SimpleNamespace(input_ids=[ord(character) for character in text])


def configuration(directory):
  return OmegaConf.create({
    'data': {'train': 'tiny_gsm', 'valid': 'tiny_gsm',
             'cache_dir': str(directory), 'data_path': None,
             'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M',
             'separator': '\\n', 'train_on_prompt': False,
             'train_on_pad': True, 'filter_too_long': True,
             'streaming': False, 'wrap': False, 'val_ratio': 0.01, 'val_seed': 42},
    'model': {'length': 12},
    'loader': {'batch_size': 1, 'global_batch_size': 1,
               'eval_batch_size': 1, 'eval_global_batch_size': 1,
               'num_workers': 0, 'pin_memory': False},
    'trainer': {'devices': 1, 'num_nodes': 1, 'accumulate_grad_batches': 1},
    'eval': {'gsm8k_data_path': None},
  })


def example(*, prompt_len=True, train_on_pad=True):
  row = {'input_ids': [0, 113, 92, 110, 49, 0] + [256] * 6,
         'attention_mask': [0] * 4 + [1] * (8 if train_on_pad else 2)
                           + ([] if train_on_pad else [0] * 6)}
  if prompt_len:
    row['prompt_len'] = 4
  return row


class TinyGSMDataTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.config = configuration(self.directory.name)
    self.tokenizer = FakeTokenizer()

  def save_cache(self, rows=None):
    rows = [example()] if rows is None else rows
    path = data.tinygsm_cache_path(self.config)
    ds = datasets.Dataset.from_list(rows)
    datasets.DatasetDict({'train': ds, 'validation': ds}).save_to_disk(str(path))
    return path

  def wrap(self, row):
    return data.ConditionalDataset([row], self.config, self.tokenizer)[0]

  def test_reference_separator_cache_key(self):
    self.assertEqual(data.tinygsm_cache_path(self.config).name,
      'tiny_gsm_bs12_answer_only_filtered_train_on_pad_'
      'HuggingFaceTB__SmolLM-135M_sep5c6e')
    self.config.data.separator = '\n'
    self.assertTrue(data.tinygsm_cache_path(self.config).name.endswith('_sep0a'))

  def test_prompt_and_padding_config_rejected(self):
    for key in ('wrap', 'train_on_prompt', 'streaming'):
      with self.subTest(key=key):
        cfg = copy.deepcopy(self.config)
        cfg.data[key] = True
        with self.assertRaises(ValueError):
          data.tinygsm_cache_path(cfg)

  def test_wrapper_accounts_for_added_pad_and_clears_mask_alias(self):
    fake = FakeTokenizer(with_pad=False)
    self.config.data.tokenizer_local_files_only = True
    with mock.patch('transformers.AutoTokenizer.from_pretrained', return_value=fake) as load:
      tokenizer = data.get_tokenizer(self.config)
    self.assertEqual(len(tokenizer), 257)
    self.assertEqual(tokenizer.vocab_size, 257)
    self.assertEqual(tokenizer.pad_token_id, 256)
    self.assertIsNone(tokenizer.mask_token)
    self.assertTrue(load.call_args.kwargs['local_files_only'])
    restored = pickle.loads(pickle.dumps(tokenizer))
    self.assertEqual(restored.vocab_size, 257)
    restored.bos_token = 'changed'
    self.assertEqual(restored._wrapped_tokenizer.bos_token, 'changed')

  def test_tokenizer_download_is_standard_hf_default(self):
    with mock.patch('transformers.AutoTokenizer.from_pretrained',
                    return_value=self.tokenizer) as load:
      data.get_tokenizer(self.config)
    self.assertFalse(load.call_args.kwargs['local_files_only'])

  def test_legacy_prompt_length_is_derived(self):
    row = self.wrap(example(prompt_len=False))
    self.assertEqual(row['prompt_len'].item(), 4)
    self.assertEqual(row['attention_mask'].dtype, torch.long)

  def test_ambiguous_allzero_allone_and_noncontiguous_masks_rejected(self):
    for mask in ([0] * 12, [1] * 12, [0, 1, 0] + [1] * 9):
      with self.subTest(mask=mask):
        row = example(prompt_len=False)
        row['attention_mask'] = mask
        with self.assertRaises(ValueError):
          self.wrap(row)

  def test_fractional_masks_rejected(self):
    row = example()
    row['attention_mask'][4] = 0.5
    with self.assertRaisesRegex(ValueError, 'binary'):
      self.wrap(row)

  def test_fractional_and_oov_token_ids_rejected(self):
    for value in (1.5, -1, 257):
      row = example()
      row['input_ids'][4] = value
      with self.subTest(value=value), self.assertRaises(ValueError):
        self.wrap(row)

  def test_prompt_length_mismatch_rejected(self):
    row = example()
    row['prompt_len'] = 3
    with self.assertRaisesRegex(ValueError, 'prompt_len'):
      self.wrap(row)

  def test_train_on_pad_requires_active_suffix(self):
    with self.assertRaisesRegex(ValueError, 'reach the end'):
      self.wrap(example(train_on_pad=False))

  def test_train_on_pad_false_accepts_only_padding_suffix(self):
    self.config.data.train_on_pad = False
    row = example(train_on_pad=False)
    self.assertEqual(self.wrap(row)['attention_mask'].sum().item(), 2)
    row['input_ids'][-1] = 45
    with self.assertRaisesRegex(ValueError, 'non-padding'):
      self.wrap(row)

  def test_wrong_sequence_length_rejected(self):
    row = example()
    row['input_ids'].pop()
    with self.assertRaisesRegex(ValueError, 'model.length'):
      self.wrap(row)

  def test_missing_cache_does_not_download_or_create(self):
    with mock.patch.object(datasets, 'load_dataset') as download:
      with self.assertRaises(FileNotFoundError):
        data.check_cached_data(self.config)
      download.assert_not_called()
    self.assertEqual(list(Path(self.directory.name).iterdir()), [])

  def test_preflight_accepts_legacy_cache_without_prompt_length(self):
    self.save_cache([example(prompt_len=False)])
    data.check_cached_data(self.config)

  def test_legacy_cache_rejects_unverifiable_split_overrides(self):
    path = self.save_cache()
    for key, value in (('val_ratio', 0.25), ('val_seed', 43)):
      with self.subTest(key=key):
        cfg = copy.deepcopy(self.config)
        cfg.data[key] = value
        with self.assertRaisesRegex(ValueError, 'no split metadata'):
          data.check_cached_data(cfg)
    self.assertTrue((path / 'dataset_dict.json').is_file())

  def test_legacy_cache_does_not_silently_ignore_custom_source(self):
    self.save_cache()
    self.config.data.data_path = '/some/other/tinygsm.json'
    with self.assertRaisesRegex(ValueError, 'cannot verify data.data_path'):
      data.check_cached_data(self.config)

  def test_bad_cache_preserved(self):
    row = example()
    row['attention_mask'] = [0] * 12
    path = self.save_cache([row])
    with self.assertRaises(ValueError):
      data.check_cached_data(self.config)
    self.assertTrue((path / 'dataset_dict.json').is_file())

  def test_separator_mismatch_never_reuses_other_cache(self):
    self.save_cache()
    self.config.data.separator = '\n'
    with self.assertRaises(FileNotFoundError):
      data.check_cached_data(self.config)

  def test_zero_worker_loader(self):
    self.save_cache()
    train, valid = data.get_dataloaders(self.config, self.tokenizer)
    self.assertFalse(train.persistent_workers)
    self.assertEqual(next(iter(train))['prompt_len'].tolist(), [4])
    self.assertEqual(next(iter(valid))['input_ids'].shape, (1, 12))
    self.assertIs(train.tokenizer, self.tokenizer)

  def test_configured_devices_not_visible_gpu_count(self):
    self.save_cache()
    self.config.trainer.devices = [2]
    with mock.patch('torch.cuda.device_count', side_effect=AssertionError('queried GPU')):
      self.assertEqual(data.configured_world_size(self.config), 1)
      data.get_dataloaders(self.config, self.tokenizer)
    self.config.trainer.devices = [0, 2]
    self.assertEqual(data.configured_world_size(self.config), 2)
    with self.assertRaisesRegex(ValueError, 'global_batch_size'):
      data.get_dataloaders(self.config, self.tokenizer)

  def test_batch_size_mismatch_rejected(self):
    self.config.loader.eval_global_batch_size = 2
    with self.assertRaisesRegex(ValueError, 'eval_global_batch_size'):
      data.get_dataloaders(self.config, self.tokenizer)

  def test_auto_devices_needs_launch_world_size(self):
    self.config.trainer.devices = 'auto'
    with mock.patch.dict('os.environ', {'LOCAL_WORLD_SIZE': '0'}):
      with self.assertRaises(ValueError):
        data.configured_world_size(self.config)
    with mock.patch.dict('os.environ', {'LOCAL_WORLD_SIZE': '2'}):
      self.assertEqual(data.configured_world_size(self.config), 2)

  def test_local_preparation_matches_reference_encoding_and_split(self):
    source = Path(self.directory.name) / 'input.json'
    source.write_text(json.dumps([{'question': ' q ', 'code': ' 1 '}] * 8))
    self.config.data.data_path = str(source)
    self.config.data.val_ratio = 0.25
    with mock.patch.object(datasets, 'load_dataset') as download:
      target = data.prepare_cache(self.config, self.tokenizer)
      download.assert_not_called()
    ds = datasets.load_from_disk(str(target))
    self.assertEqual(len(ds['train']), 6)
    self.assertEqual(len(ds['validation']), 2)
    self.assertEqual(ds['train'][0], example())
    data.check_cached_data(self.config)
    with self.assertRaises(FileExistsError):
      data.prepare_cache(self.config, self.tokenizer)
    self.config.data.data_path = None
    data.check_cached_data(self.config)
    self.config.data.data_path = str(source.with_name('different.json'))
    with self.assertRaisesRegex(ValueError, 'metadata mismatch for source'):
      data.check_cached_data(self.config)

  def test_preparation_does_not_implicitly_download(self):
    with mock.patch.object(datasets, 'load_dataset') as download:
      with self.assertRaises(FileNotFoundError):
        data.prepare_cache(self.config, self.tokenizer)
      download.assert_not_called()

  def test_new_cache_metadata_mismatch_rejected(self):
    path = self.save_cache()
    metadata = data._cache_metadata(self.config)
    metadata['separator'] = '\n'
    (path / 'didi_tinygsm_metadata.json').write_text(json.dumps(metadata))
    with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
      data.check_cached_data(self.config)

  def test_local_gsm8k_tokenization_keeps_grader_text_without_shared_write(self):
    path = Path(self.directory.name) / 'gsm8k.json'
    path.write_text(json.dumps([{'prompt': ' q ', 'response_ground_truth': '#### 1'}]))
    self.config.eval.gsm8k_data_path = str(path)
    ds = data.get_gsm8k_test_dataset(self.config, self.tokenizer)
    self.assertEqual(ds[0]['input_ids'], [0, 113, 92, 110])
    self.assertEqual(ds[0]['response_ground_truth'], '#### 1')
    self.assertEqual(list(Path(self.directory.name).iterdir()), [path])

  def test_missing_gsm8k_does_not_download(self):
    with mock.patch.object(datasets, 'load_dataset') as download:
      with self.assertRaises(FileNotFoundError):
        data.get_gsm8k_test_dataset(self.config, self.tokenizer)
      download.assert_not_called()

  def test_gsm8k_prompt_must_leave_answer_space(self):
    path = Path(self.directory.name) / 'gsm8k.json'
    path.write_text(json.dumps([{'prompt': 'q' * 20, 'response_ground_truth': '#### 1'}]))
    self.config.eval.gsm8k_data_path = str(path)
    with self.assertRaisesRegex(ValueError, 'no answer space'):
      data.get_gsm8k_test_dataset(self.config, self.tokenizer)


if __name__ == '__main__':
  unittest.main()
