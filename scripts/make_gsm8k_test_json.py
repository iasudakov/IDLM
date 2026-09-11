"""Write the GSM8K test split as the JSON `data=gsm8k-test` expects.

`configs/data/gsm8k-test.yaml` ships with an empty `data_path` because the
reference protocol (S-FLM, which the README's "Conditional TinyGSM Evaluation"
section points at) reads a local file that is not redistributed with either
repo. Its format is two fields per record:

  {"prompt": <the question>, "response_ground_truth": <the reference answer>}

which is exactly the raw `openai/gsm8k` test split -- no instruction wrapper,
no few-shot prefix. That matters: a TinyGSM model is trained on
`question \\n <python program>`, so the question alone *is* the prompt it was
taught to complete, and adding a template would put the model off-distribution
and quietly depress the accuracy. The reference answer keeps its trailing
`#### <number>` line, which is what `sandbox_gsm8k._extract_gold_answer` reads.

Usage:
  python scripts/make_gsm8k_test_json.py --out data/gsm8k_test.json
"""

import argparse
import json
import os

import datasets


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out', default='data/gsm8k_test.json',
                      help='where to write the JSON')
  parser.add_argument('--split', default='test',
                      help="'test' (1319 records) or 'train'")
  parser.add_argument('--cache-dir', default=None,
                      help='HuggingFace datasets cache')
  args = parser.parse_args()

  ds = datasets.load_dataset(
    'openai/gsm8k', 'main', split=args.split, cache_dir=args.cache_dir)
  records = [{'prompt': q, 'response_ground_truth': a}
             for q, a in zip(ds['question'], ds['answer'])]

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(records, f, indent=2)
  print(f'Wrote {len(records)} GSM8K {args.split} records to {args.out}')


if __name__ == '__main__':
  main()
