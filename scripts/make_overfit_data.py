"""Build a 2-example OWT cache, for the factorization sanity check.

A one-step masked-diffusion model trained on exactly two sequences A and B
cannot emit either one cleanly: its prediction factorises over positions, so
the best it can do from a fully masked start is the per-position mixture
`0.5*A_i + 0.5*B_i`, and sampling that gives a token-wise chimera of the two.
That is the factorisation error with nothing else mixed in, and a generator
with a working per-sample input noise channel should be able to escape it by
mapping the noise to A or to B.

`get_dataset` loads `{cache_dir}/{name}_{mode}_bs{L}_wrapped.dat` whenever it
exists, so writing that path is enough -- no training code changes.

Usage:
  python scripts/make_overfit_data.py --out /tmp/overfit_cache [--rows 1024]
Then train with `data.cache_dir=/tmp/overfit_cache`.
"""
import argparse
import os

import datasets
import torch

SRC = ('/extra_disk_1/mnt_source/data/owt/'
       'openwebtext-valid_validation_bs1024_wrapped.dat')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='/tmp/overfit_cache')
  ap.add_argument('--rows', type=int, default=1024,
                  help='total rows; the two examples are repeated to fill them')
  ap.add_argument('--length', type=int, default=1024)
  ap.add_argument('--src', default=SRC)
  ap.add_argument('--pick', type=int, nargs=2, default=(0, 1),
                  help='which two source rows to use')
  args = ap.parse_args()

  src = datasets.load_from_disk(args.src)
  a, b = (src[i]['input_ids'][:args.length] for i in args.pick)
  assert a != b, 'the two examples must differ'
  overlap = sum(int(x == y) for x, y in zip(a, b)) / len(a)
  print(f'example A/B agree on {overlap:.1%} of positions '
        f'(a chimera of the two scores about half way between)')

  ids, mask = [], []
  for i in range(args.rows):
    ids.append(a if i % 2 == 0 else b)
    mask.append([1] * args.length)
  ds = datasets.Dataset.from_dict({'input_ids': ids, 'attention_mask': mask})

  os.makedirs(args.out, exist_ok=True)
  for name in (f'openwebtext-train_train_bs{args.length}_wrapped.dat',
               f'openwebtext-valid_validation_bs{args.length}_wrapped.dat'):
    path = os.path.join(args.out, name)
    ds.save_to_disk(path)
    print(f'wrote {path}  ({len(ds)} rows, 2 distinct)')

  torch.save({'A': a, 'B': b}, os.path.join(args.out, 'examples.pt'))
  print(f"reference pair saved to {os.path.join(args.out, 'examples.pt')}")


if __name__ == '__main__':
  main()
