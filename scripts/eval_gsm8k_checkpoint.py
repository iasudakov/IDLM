"""Score a checkpoint on the full GSM8K test set, standalone.

The in-training benchmark (`eval.gsm8k_at_validation`) answers "is this run
improving"; this answers "what does this checkpoint score", which is what a
released artifact needs. Same 1319 questions, same prefix-pinned generation
and the same vendored S-FLM grader, so the number is directly comparable both
to the in-training curve and to published tables.

  python scripts/eval_gsm8k_checkpoint.py CKPT --steps 64

The separator must be the one the checkpoint was trained with -- see
`dataloader._separator_tag`. Both the S-FLM teachers and the released IDLM
students use the literal two-character form, which is the default here.
"""

import argparse
import json
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omegaconf
import torch
from hydra import compose, initialize_config_dir

import algo  # noqa: E402
import confidence_sampling  # noqa: E402
import dataloader  # noqa: E402
import sandbox_gsm8k  # noqa: E402
import utils  # noqa: E402

utils.register_omegaconf_resolvers()

# `_extract_code` compiles candidates to find their valid prefix, and
# odd-but-parseable model output prints a SyntaxWarning per example.
warnings.simplefilter('ignore', SyntaxWarning)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('checkpoint')
  parser.add_argument('--steps', type=int, default=64,
                      help='denoising steps (the paper reports 32/64/128)')
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--limit', type=int, default=None,
                      help='score only the first N questions (default: all)')
  parser.add_argument('--separator', default='literal',
                      choices=['literal', 'newline'])
  parser.add_argument('--data-path',
                      default=os.path.join(REPO, 'data/gsm8k_test.json'))
  parser.add_argument('--cache-dir', default=os.path.join(REPO, 'cache_dir'))
  parser.add_argument('--out', default=None, help='write results JSON here')
  parser.add_argument('--timeout', type=float, default=5.0)
  parser.add_argument('--disable-ema', action='store_true',
                      help='score the raw weights instead of the EMA copy')
  parser.add_argument('--remasking', default='random',
                      choices=list(confidence_sampling.STRATEGIES),
                      help='which masked positions each step commits; the '
                           'low_confidence_* rules rank by model confidence')
  parser.add_argument('--threshold', type=float, default=0.9,
                      help='budget for low_confidence_dynamic. That rule '
                           'ignores --steps entirely: the threshold decides '
                           'how many positions commit per step, so the forward '
                           'count is an outcome and is reported below.')
  parser.add_argument('--temperature', type=float, default=1.0,
                      help='token draw; 0 is greedy. Only meaningful for the '
                           'low_confidence_* rules.')
  args = parser.parse_args()

  with initialize_config_dir(config_dir=os.path.join(REPO, 'configs'),
                             version_base=None):
    config = compose(config_name='config', overrides=[
      'data=gsm8k-test',
      f'data.cache_dir={args.cache_dir}',
      f'data.data_path={args.data_path}',
      'model=small',
      'model.length=512',
      'algo=mdlm',
      'sampling.predictor=ancestral_cache',
      'sampling.noise_removal=ancestral',
      f'loader.eval_batch_size={args.batch_size}',
      f'sampling.remasking_strategy={args.remasking}',
      f'sampling.dynamic_threshold={args.threshold}',
      f'sampling.temperature={args.temperature}',
      'trainer.devices=1',
    ])
  # Set outside the override list: Lightning's default checkpoint names embed
  # `=` (step=00068000.ckpt) and Hydra's override grammar cannot parse a second
  # one in the value.
  config.eval.checkpoint_path = args.checkpoint

  # A checkpoint saved mid-distillation carries the critics (`teacher.*`,
  # `fake_model.*`) alongside the student, and those submodules only exist when
  # `is_distill` is set -- otherwise the load fails on unexpected keys. Detect
  # it from the file rather than making the caller remember which kind of
  # checkpoint they have. Only the student is ever sampled from either way.
  probe = torch.load(args.checkpoint, map_location='cpu')['state_dict']
  is_distill = any(k.startswith(('teacher.', 'fake_model.')) for k in probe)
  del probe
  config.adversarial_distill.is_distill = is_distill
  if is_distill:
    print('checkpoint carries distillation critics; loading with '
          'is_distill=True (only the student is sampled)')
  config.data.separator = '\\n' if args.separator == 'literal' else '\n'
  omegaconf.OmegaConf.resolve(config)

  tokenizer = dataloader.get_tokenizer(config)
  model = algo.MDLM.load_from_checkpoint(
    args.checkpoint, tokenizer=tokenizer, config=config,
    weights_only=False).to('cuda').eval()
  if args.disable_ema:
    model.ema = None
  n_ema = None if model.ema is None else len(model.ema.shadow_params)
  n_par = sum(1 for _ in model._get_parameters())
  print(f'EMA: {"disabled" if model.ema is None else "enabled"}; '
        f'shadow_params={n_ema} model_params={n_par}')
  if model.ema is not None and n_ema != n_par:
    print('!! EMA/parameter count MISMATCH -- copy_to would misalign weights')
  model._eval_mode()

  dataset = dataloader.get_gsm8k_test_dataset(config, tokenizer)
  total = len(dataset) if not args.limit else min(args.limit, len(dataset))
  adaptive = args.remasking in confidence_sampling.ADAPTIVE_STRATEGIES
  budget = (f'threshold {args.threshold} (steps ignored)' if adaptive
            else f'{args.steps} steps')
  print(f'{os.path.basename(args.checkpoint)}: {total} questions, '
        f'{args.remasking}, {budget}, separator={args.separator}', flush=True)

  records = []
  nfes = []
  per_seq = []
  for start in range(0, total, args.batch_size):
    chunk = list(range(start, min(start + args.batch_size, total)))
    prompts = [torch.tensor(dataset[i]['input_ids']) for i in chunk]
    lengths = torch.tensor([len(p) for p in prompts], device='cuda')
    padded = torch.zeros(len(chunk), int(lengths.max()),
                         dtype=torch.long, device='cuda')
    for i, ids in enumerate(prompts):
      padded[i, :len(ids)] = ids.to('cuda')
    with torch.no_grad():
      samples, metadata = model.generate_samples(
        num_samples=len(chunk), num_steps=args.steps,
        prefix_tokens=padded, prefix_lengths=lengths,
        return_metadata=True)
    nfes.append(metadata['nfe'])
    per_seq.extend(metadata['nfe_per_row'].cpu().tolist())
    for i, idx in enumerate(chunk):
      generated = samples[i, int(lengths[i]):].cpu().tolist()
      records.append({
        'prompt': dataset[idx]['prompt'],
        'response_ground_truth': dataset[idx]['response_ground_truth'],
        'response': tokenizer.decode(generated, skip_special_tokens=True)})
    del samples
    print(f'  {len(records)}/{total}', end='\r', flush=True)

  correct = sum(
    sandbox_gsm8k.evaluate_samples(r['response'],
                                   r['response_ground_truth'],
                                   timeout_s=args.timeout)
    for r in records)
  accuracy = correct / len(records)
  # Binomial standard error; +/- 1.96 of them is the 95% band.
  se = math.sqrt(accuracy * (1 - accuracy) / len(records))
  # The forwards actually spent. Under an adaptive rule this, not --steps, is
  # the cost, and it is the axis two strategies can be compared on.
  # Per batch is the wall-clock cost, but it is a *max* over the batch's rows
  # (the loop runs until the slowest finishes) and so grows with --batch-size.
  # Per sequence is batch-size independent, and is what the Pareto curves plot.
  avg_nfe = sum(nfes) / max(len(nfes), 1)
  avg_nfe_seq = sum(per_seq) / max(len(per_seq), 1)
  print(f'\n>>> {budget}: {accuracy * 100:.2f}% '
        f'(+/- {1.96 * se * 100:.2f}) [{correct}/{len(records)}] '
        f'at {avg_nfe:.1f} NFE/batch, {avg_nfe_seq:.1f} NFE/seq '
        f'[{args.remasking}]')

  if args.out:
    with open(args.out, 'w') as f:
      json.dump({'checkpoint': args.checkpoint, 'steps': args.steps,
                 'separator': args.separator, 'accuracy': accuracy,
                 'remasking_strategy': args.remasking,
                 'dynamic_threshold': args.threshold if adaptive else None,
                 'temperature': args.temperature,
                 'avg_nfe': avg_nfe,
                 'avg_nfe_per_seq': avg_nfe_seq,
                 'num_correct': correct, 'num_total': len(records),
                 'records': records}, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
