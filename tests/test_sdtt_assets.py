"""Reuse the MMD offline tokenizer bundle without changing dataset cache IDs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

import dataloader
from model_assets import pretrained_kwargs
from sdtt import SDTT


class Tokenizer:
    bos_token, bos_token_id = '<bos>', 0
    eos_token, eos_token_id = '<eos>', 1
    pad_token, pad_token_id = '<pad>', 2
    mask_token, mask_token_id = None, None

    def __len__(self):
        return 8


class AssetsTest(unittest.TestCase):
    def test_existing_manifest_and_logical_tokenizer_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = root / 'smollm'
            tokenizer.mkdir()
            (tokenizer / 'tokenizer.json').write_text('{}')
            (root / 'manifest.json').write_text(json.dumps({'assets': [{
                'repo_id': 'HuggingFaceTB/SmolLM-135M', 'folder': 'smollm'}]}))
            cfg = OmegaConf.create({'model_weights_dir': str(root), 'data': {
                'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}})
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()) as load:
                result = dataloader.get_tokenizer(cfg)
            load.assert_called_once_with(pretrained_model_name_or_path=str(tokenizer.resolve()),
                                         local_files_only=True)
            self.assertEqual(cfg.data.tokenizer_name_or_path, 'HuggingFaceTB/SmolLM-135M')
            self.assertEqual(result.vocab_size, 8)

    def test_missing_bundled_tokenizer_rejected_without_hub_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                pretrained_kwargs('HuggingFaceTB/SmolLM-135M', directory, kind='tokenizer')

    def test_offline_checkpoint_does_not_fall_back_to_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            holder = type('Holder', (), {'config': OmegaConf.create({'model_weights_dir': directory})})()
            with self.assertRaisesRegex(ValueError, 'local MDLM checkpoint'):
                SDTT._checkpoint_source(holder, 'hf://jdeschena/s-flm/tinygsm/mdlm.ckpt')
            with self.assertRaises(FileNotFoundError):
                SDTT._checkpoint_source(holder, str(Path(directory) / 'missing.ckpt'))
            checkpoint = Path(directory) / 'mdlm_notok.ckpt'
            checkpoint.touch()
            self.assertEqual(SDTT._checkpoint_source(holder, str(checkpoint)), str(checkpoint.resolve()))


if __name__ == '__main__':
    unittest.main()
