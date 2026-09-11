"""Build the tokenized TinyGSM cache before training, in one process.

`_train` calls `dataloader.get_dataloaders` on *every* DDP rank, so a run
whose cache does not exist yet has every rank tokenize 11.8M examples
simultaneously and then race to `save_to_disk` the same directory. Building it
once up front avoids that, and means a tokenization failure costs CPU minutes
instead of a GPU allocation.

The separator matters and is not the config default -- see
`dataloader._separator_tag` and TINYGSM_RESULTS.md. Distilling from an S-FLM
teacher requires the literal two-character `\\n` it was trained with; a
checkpoint trained inside this repo wants a real newline. Pass `--separator`
accordingly; the cache is keyed on it, so both can coexist.

  python scripts/prepare_tinygsm.py --cache-dir cache_dir --separator literal
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omegaconf
import torch
from hydra import compose, initialize_config_dir

import utils  # noqa: E402

utils.register_omegaconf_resolvers()

import dataloader  # noqa: E402  (must follow resolver registration)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cache-dir', default=os.path.join(REPO, 'cache_dir'))
  parser.add_argument('--separator', default='literal',
                      choices=['literal', 'newline'],
                      help="'literal' = backslash-n (S-FLM checkpoints), "
                           "'newline' = a real newline")
  parser.add_argument('--length', type=int, default=512)
  parser.add_argument('--num-proc', type=int, default=8)
  args = parser.parse_args()

  with initialize_config_dir(config_dir=os.path.join(REPO, 'configs'),
                             version_base=None):
    config = compose(config_name='config', overrides=[
      'data=tiny-gsm',
      f'data.cache_dir={args.cache_dir}',
      'data.wrap=False',
      'data.train_on_prompt=False',
      'data.train_on_pad=True',
      'data.filter_too_long=True',
      f'model.length={args.length}',
      f'loader.num_workers={args.num_proc}',
      'trainer.devices=1',
    ])
  # Set outside the override list: Hydra's override grammar cannot express a
  # bare newline at all, and reads a backslash escape rather than passing it
  # through, so neither value survives the command line intact.
  config.data.separator = '\\n' if args.separator == 'literal' else '\n'

  tokenizer = dataloader.get_tokenizer(config)
  print(f'separator={config.data.separator!r} -> ids '
        f'{tokenizer(config.data.separator, add_special_tokens=False).input_ids}')

  dataset = dataloader.get_tiny_gsm_dataset(config, tokenizer)
  for split, part in dataset.items():
    print(f'{split}: {len(part)} rows, columns {part.column_names}')

  tag = dataloader._separator_tag(config.data.separator)
  print(f'\ncache tag: {tag}  (under {args.cache_dir})')


if __name__ == '__main__':
  main()
