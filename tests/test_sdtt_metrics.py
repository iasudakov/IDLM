"""GSM8K construction must not download an unrelated GPT-2 tokenizer."""

import unittest
from unittest.mock import patch

import torch

from metrics import Metrics


class LazyTokenizerTest(unittest.TestCase):
    def test_download_only_when_generative_retokenization_is_requested(self):
        class Tokenizer:
            pad_token = None
            eos_token = '<eos>'
            eos_token_id = 1

            def __call__(self, text, **kwargs):
                return {'input_ids': torch.tensor([[1]]),
                        'attention_mask': torch.tensor([[1]])}

        with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()) as load:
            metrics = Metrics(gen_ppl_eval_model_name_or_path='gpt2-large')
            metrics.reset()
            load.assert_not_called()
            metrics._eval_retokenize(['text'], 512, 'cpu')
            metrics._eval_retokenize(['other'], 512, 'cpu')
            load.assert_called_once_with('gpt2-large')
            self.assertEqual(metrics.tokenizer.pad_token, '<eos>')


if __name__ == '__main__':
    unittest.main()
