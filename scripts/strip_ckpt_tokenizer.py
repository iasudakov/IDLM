"""Rewrite a checkpoint without its pickled tokenizer.

Lightning stores `__init__`'s arguments under `hyper_parameters`, and this
repo's modules take `tokenizer=`, so every checkpoint embeds a pickled
`tokenizers.Tokenizer`. That object is serialized by the Rust library and
refuses to load into an older `tokenizers` than wrote it, raising

    Error while attempting to unpickle Tokenizer: data did not match any
    variant of untagged enum ModelWrapper

before a single weight is read. The released TinyGSM teachers were written by
a newer stack than `requirements.txt` pins (transformers==4.38.2 =>
tokenizers 0.15.2), so in the repo's *own* documented environment they are
unreadable.

The stored tokenizer is dead weight: `load_from_checkpoint` is always called
with an explicit `tokenizer=`, which overrides the hyper-parameter. So drop
just that key and re-save. Nothing else is touched -- weights, EMA, optimizer
state and config all round-trip untouched -- which means the result can be fed
to unmodified upstream code.

  python scripts/strip_ckpt_tokenizer.py in.ckpt out.ckpt
"""

import argparse
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _StubTokenizer:
  def __init__(self, *args, **kwargs):
    pass

  def __setstate__(self, state):
    del state


class _Unpickler(pickle.Unpickler):
  def find_class(self, module, name):
    if module.split('.')[0] == 'tokenizers' and name == 'Tokenizer':
      return _StubTokenizer
    return super().find_class(module, name)


class _pickle_module:
  Unpickler = _Unpickler
  load = pickle.load


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('src')
  parser.add_argument('dst')
  args = parser.parse_args()

  checkpoint = torch.load(args.src, map_location='cpu',
                          pickle_module=_pickle_module)
  hparams = checkpoint.get('hyper_parameters', {})
  removed = hparams.pop('tokenizer', None)
  print(f'hyper_parameters: {sorted(hparams)} '
        f'(tokenizer {"removed" if removed is not None else "absent"})')
  print(f'state_dict tensors: {len(checkpoint["state_dict"])}')
  torch.save(checkpoint, args.dst)
  print(f'wrote {args.dst} ({os.path.getsize(args.dst) / 1e9:.2f} GB)')


if __name__ == '__main__':
  main()
