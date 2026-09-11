"""Grade a `mode=gsm8k_eval` run from its shards, finished or not.

`_sample_gsm8k` rewrites each rank's `rank_XXXX.json` every `gsm8k.flush_every`
batches (atomically), so a long sweep can be checked while it is still running
instead of only at the end. Point this at the run's `gsm8k.output_dir`:

  python scripts/gsm8k_partial.py eval_runs/tinygsm/sflm-mdlm-teacher_1024steps

The accuracy printed is over the completions written *so far*. It is an
unbiased estimate of the final number -- test examples are handed to ranks
round-robin, so a prefix of a shard is an arbitrary subset rather than, say,
all the short questions -- but it is noisier than the final one, and the
printed band says how much: +/- 1.96 standard errors of a binomial proportion.
Do not quote it as the result; wait for `complete` to be true on every rank,
at which point the run has written its own `results.json` anyway.
"""

import argparse
import glob
import json
import math
import os
import sys
import warnings

# `sandbox_gsm8k._extract_code` compiles the candidate program to find its
# valid prefix, and that compile is outside the sandbox's own warning filter,
# so model-written code that is merely *odd* (`5(x)`, a stray float call)
# prints a SyntaxWarning per example. Nothing is wrong -- those programs are
# then graded as failures, which is correct -- but a thousand of them buries
# the accuracy line.
warnings.simplefilter('ignore', SyntaxWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sandbox_gsm8k


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('output_dir', help="the run's gsm8k.output_dir")
  parser.add_argument('--timeout', type=float, default=5.0,
                      help='seconds allowed per generated program')
  parser.add_argument('--show', type=int, default=0,
                      help='print this many graded completions')
  args = parser.parse_args()

  shards = sorted(glob.glob(os.path.join(args.output_dir, 'rank_*.json')))
  if not shards:
    raise SystemExit(f'no rank_*.json in {args.output_dir} yet')

  records, done, expected = [], 0, 0
  for path in shards:
    with open(path) as f:
      shard = json.load(f)
    records.extend(shard['records'])
    expected += shard.get('num_expected', 0)
    done += bool(shard.get('complete', False))
    print(f"{os.path.basename(path)}: {len(shard['records'])}"
          f"/{shard.get('num_expected', '?')}"
          f"{' (complete)' if shard.get('complete') else ''}")

  correct = sum(
    sandbox_gsm8k.evaluate_samples(r['response'],
                                   r['response_ground_truth'],
                                   timeout_s=args.timeout)
    for r in records)
  n = len(records)
  acc = correct / n
  # Binomial standard error; the band is 1.96 of them.
  se = math.sqrt(acc * (1 - acc) / n) if n else 0.0
  coverage = f'{n}/{expected}' if expected else str(n)
  print(f'\n{done}/{len(shards)} ranks complete, {coverage} graded')
  print(f'accuracy so far: {acc * 100:.2f}% '
        f'(+/- {1.96 * se * 100:.2f}) [{correct}/{n}]')

  for record in records[: args.show]:
    graded = sandbox_gsm8k.evaluate_samples(
      record['response'], record['response_ground_truth'], args.timeout)
    print('=' * 70)
    print('CORRECT' if graded else 'WRONG')
    print('Q:', record['prompt'][:300])
    print('A:', record['response'][:600])


if __name__ == '__main__':
  main()
