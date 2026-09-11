"""Is the frozen diffusion teacher actually a good MMD feature map?

`scripts/test_mmd_ar_features.py` checks the reward *arithmetic* against a stub
backbone whose features are a lookup table -- i.e. a pure bag-of-embeddings. It
therefore cannot distinguish a feature map that understands a TinyGSM solution
from one that only counts matching token ids. This script asks the question the
stub cannot: given the *real* teacher weights, does
`pseudo_huber(pooled_gen - pooled_clean)` rank answers the way a useful reward
must?

Three probes, all on real TinyGSM validation rows, all using the exact pooling
and masking `mmd_reward.FrozenBackboneFeatures` uses:

  RETRIEVAL  For each row, splice every other row's answer into its answer span
    and score all of them. Top-1 = the fraction of rows whose own answer wins.
    Chance is 1/B. A feature map that cannot clear chance here cannot rank the
    `mmd_group_size` samples REINFORCE asks it to rank.

  LADDER     Reward under corruptions of known severity. The two that matter:
    * `shuffle` permutes the answer's tokens inside the row -- *identical bag of
      tokens*, destroyed order and syntax. A bag-of-embeddings feature map
      scores this exactly 0. How close to `rand100` it gets is how much of the
      feature is structure rather than token identity.
    * `digit_swap` changes only digit tokens. That is the failure mode that
      actually costs GSM8K accuracy, and it is a handful of tokens, so a reward
      that cannot see it is not rewarding correctness.

  EMBED-BASELINE  Every probe is also run at layer -1, the raw `vocab_embed`
    output. That is the bag-of-embeddings reward. Any block whose numbers merely
    match layer -1 is contributing nothing beyond token overlap, however good
    its absolute retrieval score looks.

The sweep covers every block and several `mmd_noise_level`s, and splits the
pooled positions three ways, because the default pooling mixes two populations
with completely different semantics: at a position masked in *both* rows the
tokens are equal and the feature can only differ through bidirectional
attention (`maskonly`), while at an unmasked one the rows differ in the token
itself (`keptonly`). At the default noise_level=0.1, 90% of the pooled signal is
`keptonly` -- the regime where the denoiser's readout is untrained.

    python scripts/probe_mmd_features.py --ckpt checkpoints/tinygsm/mdlm_notok.ckpt
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import utils  # noqa: E402

utils.register_omegaconf_resolvers()

import algo  # noqa: E402
import dataloader as dataloader_mod  # noqa: E402
import mmd_reward  # noqa: E402

POOLS = ('all', 'keptonly', 'maskonly')


def build_config(ckpt, batch_size, extra):
  import hydra
  overrides = [
    'data=tiny-gsm', 'data.wrap=False', 'data.train_on_prompt=False',
    'data.train_on_pad=True', 'data.filter_too_long=True',
    'model=small', 'model.length=512', 'algo=mdlm',
    f'data.cache_dir={ROOT}/cache_dir',
    f'loader.batch_size={batch_size}',
    f'loader.eval_batch_size={batch_size}',
    'loader.num_workers=2', 'trainer.devices=1',
    'adversarial_distill.is_distill=False',
    'sampling.predictor=ancestral_cache',
    f'training.finetune_path={ckpt}',
  ] + list(extra)
  with hydra.initialize_config_dir(
      config_dir=os.path.join(ROOT, 'configs'), version_base=None):
    return hydra.compose(config_name='config', overrides=overrides)


class AllLayerFeatures:
  """Pooled features at every block plus the embedding, in one forward.

  Layer -1 is `vocab_embed`'s output: the bag-of-embeddings reference every
  other layer has to beat. Only pooled `(batch, hidden)` tensors survive the
  call, so sweeping 13 layers costs one activation set, not 13.
  """

  def __init__(self, backbone):
    self.backbone = backbone
    self.n_blocks = len(backbone.blocks)

  @torch.no_grad()
  def pooled(self, tokens, masks):
    """`{layer: {pool: (batch, hidden)}}` for each named position mask."""
    captured = {}

    def make_hook(index):
      def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured[index] = hidden.detach().float()
      return hook

    handles = [self.backbone.vocab_embed.register_forward_hook(make_hook(-1))]
    for i, block in enumerate(self.backbone.blocks):
      handles.append(block.register_forward_hook(make_hook(i)))
    sigma = torch.zeros(tokens.shape[0], device=tokens.device)
    try:
      self.backbone(tokens, sigma)
    finally:
      for handle in handles:
        handle.remove()

    out = {}
    for layer, hidden in captured.items():
      out[layer] = {}
      for name, mask in masks.items():
        weight = mask.unsqueeze(-1).float()
        denom = mask.sum(1).clamp_min(1).float().unsqueeze(-1)
        out[layer][name] = (hidden * weight).sum(1) / denom
    return out


def content_mask(x0, valid_tokens, tokenizer):
  """`algo.MDLM._mmd_content_mask`, standalone: answer content before EOS."""
  content = valid_tokens.bool()
  if tokenizer.pad_token_id is not None:
    content = content & x0.ne(tokenizer.pad_token_id)
  if tokenizer.eos_token_id is not None:
    is_eos = x0.eq(tokenizer.eos_token_id) & content
    seen = torch.nn.functional.pad(is_eos.cumsum(1)[:, :-1], (1, 0))
    content = content & seen.eq(0)
  return content


def digit_token_ids(tokenizer, vocab_size, device):
  """Ids whose surface form contains a digit -- the numeric surface of TinyGSM."""
  ids = [i for i in range(min(vocab_size, len(tokenizer)))
         if any(c.isdigit() for c in tokenizer.convert_ids_to_tokens(i) or '')]
  return torch.tensor(ids, device=device)


def make_corruptions(x0, selected, unigram, digits, generator):
  """Generated rows of graded, *known* severity. Equal to x0 off `selected`."""
  device = x0.device
  batch, length = x0.shape

  def draw(shape):
    idx = torch.multinomial(unigram, int(torch.tensor(shape).prod()),
                            replacement=True, generator=generator)
    return idx.view(shape).to(device)

  def replace_fraction(frac):
    hit = (torch.rand(x0.shape, generator=generator).to(device) < frac) & selected
    return torch.where(hit, draw(x0.shape), x0)

  cases = {'identical': x0.clone()}
  for frac in (0.1, 0.5, 1.0):
    cases[f'rand{int(frac * 100)}'] = replace_fraction(frac)

  # Same multiset of tokens, permuted: isolates order/structure from identity.
  shuffled = x0.clone()
  for row in range(batch):
    pos = selected[row].nonzero(as_tuple=True)[0]
    if pos.numel() > 1:
      perm = pos[torch.randperm(pos.numel(), generator=generator).to(device)]
      shuffled[row, pos] = x0[row, perm]
  cases['shuffle'] = shuffled

  # Another problem's answer: fluent, well-formed, and wrong.
  other = x0.clone()
  for row in range(batch):
    src = (row + 1) % batch
    dst = selected[row].nonzero(as_tuple=True)[0]
    donor = selected[src].nonzero(as_tuple=True)[0]
    if dst.numel() and donor.numel():
      take = donor[torch.arange(dst.numel(), device=device) % donor.numel()]
      other[row, dst] = x0[src, take]
  cases['other_answer'] = other

  # Digits only: fluent, well-formed, and arithmetically wrong -- the failure
  # mode that actually costs GSM8K accuracy.
  is_digit = torch.isin(x0, digits) & selected
  swap = digits[torch.randint(len(digits), x0.shape, generator=generator)
                .to(device)]
  cases['digit_swap'] = torch.where(is_digit, swap, x0)

  # The same *number* of edits, placed on non-digit tokens and drawn from the
  # corpus unigram. Paired with `digit_swap` this is the one contrast that
  # isolates meaning from edit count: both rows are the same lexical distance
  # from the clean answer, but only `digit_swap` is still a valid-looking
  # program. A reward that scores them equally is counting edits.
  budget = is_digit.sum(1)
  nondigit = selected & ~is_digit
  hit = torch.zeros_like(selected)
  for row in range(batch):
    pos = nondigit[row].nonzero(as_tuple=True)[0]
    take = int(min(int(budget[row]), pos.numel()))
    if take:
      pick = pos[torch.randperm(pos.numel(), generator=generator)[:take].to(device)]
      hit[row, pick] = True
  cases['rand_matched'] = torch.where(hit, draw(x0.shape), x0)
  return cases


def apply_noise(clean, gen, selected, mask_index, noise_level, generator):
  """`FrozenBackboneFeatures`'s masking: one mask, both rows, scored span only."""
  if noise_level <= 0:
    drop = torch.zeros_like(selected)
  else:
    keep = torch.rand(clean.shape, generator=generator).to(clean.device)
    drop = (keep < noise_level) & selected
  return (torch.where(drop, mask_index, clean),
          torch.where(drop, mask_index, gen), drop)


def reward_from(pooled_clean, pooled_gen):
  return mmd_reward.pseudo_huber_reward(pooled_gen - pooled_clean)


def mask_audit(extractor, x0, selected, mask_index, unigram, digits,
               generator, layer, pool, noise_levels, group_size, realizations):
  """How much of the REINFORCE advantage is the drop mask rather than the sample?

  `FrozenBackboneFeatures.paired_reward` draws a fresh `keep` on every call, and
  `mmd_reinforce_loss` calls it once per group member -- so the leave-one-out
  baseline compares rewards computed under *different* masking realizations.
  Whatever that costs is pure estimator variance: it is not a property of the
  samples being ranked.

  Scores `group_size` distinct corruptions under `realizations` independent
  masks, then compares three advantage estimates against the mask-averaged
  truth: `fresh` (what the code does today), `shared` (one mask for the whole
  group), and the noiseless `noise_level=0` case.
  """
  device = x0.device
  # Group members stand in for the multinomial draws at one state: same clean
  # row, independently wrong in a third of the answer.
  members = []
  for _ in range(group_size):
    hit = (torch.rand(x0.shape, generator=generator).to(device) < 0.3) & selected
    draw = torch.multinomial(unigram, x0.numel(), replacement=True,
                             generator=generator).view(x0.shape).to(device)
    members.append(torch.where(hit, draw, x0))

  print(f'\n=== drop-mask variance audit (layer {layer}, pool {pool}, '
        f'group={group_size}, {realizations} mask realizations) ===')
  print(f'{"noise":>6} {"signal":>10} {"maskvar":>10} {"SNR":>8} '
        f'{"corr_fresh":>11} {"corr_shared":>12} {"sign_fresh":>11}')
  for noise in noise_levels:
    rewards = []
    for _ in range(realizations):
      _, _, drop = apply_noise(x0, x0, selected, mask_index, noise, generator)
      masks = {pool: {'all': selected, 'keptonly': selected & ~drop,
                      'maskonly': drop}[pool]}
      clean = extractor.pooled(torch.where(drop, mask_index, x0), masks)
      row = [reward_from(clean[layer][pool],
                         extractor.pooled(
                           torch.where(drop, mask_index, m), masks)[layer][pool])
             for m in members]
      rewards.append(torch.stack(row, -1))          # (batch, group)
    stacked = torch.stack(rewards)                  # (real, batch, group)

    truth = stacked.mean(0)
    signal = truth.var(-1).mean().item()            # between-sample
    maskvar = stacked.var(0).mean().item()          # within-sample, mask only
    adv_true = mmd_reward.group_advantages(truth)
    adv_fresh = mmd_reward.group_advantages(
      torch.stack([stacked[g % realizations, :, g]
                   for g in range(group_size)], -1))
    adv_shared = mmd_reward.group_advantages(stacked[0])

    def corr(a, b):
      a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
      return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()

    sign = (torch.sign(adv_fresh) == torch.sign(adv_true)).float().mean().item()
    snr = signal / maskvar if maskvar > 0 else float('inf')
    print(f'{noise:6.2f} {signal:10.3e} {maskvar:10.3e} {snr:8.2f} '
          f'{corr(adv_fresh, adv_true):11.3f} {corr(adv_shared, adv_true):12.3f} '
          f'{sign:11.3f}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ckpt', default='checkpoints/tinygsm/mdlm_notok.ckpt')
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--mask-ratio', type=float, default=1.0,
                      help='fraction of answer content treated as generated')
  parser.add_argument('--noise-levels', default='0.0,0.1,0.3')
  parser.add_argument('--layers', default='',
                      help='comma-separated; default -1 and every block')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--override', nargs='*', default=[])
  parser.add_argument('--feature-source', default='diffusion',
                      choices=('diffusion', 'ar'),
                      help="'ar' profiles the S-FLM TinyGSM AR checkpoint "
                           'instead of the frozen diffusion teacher')
  parser.add_argument('--ar-ckpt', default='checkpoints/tinygsm/ar_notok.ckpt')
  parser.add_argument('--mask-audit', action='store_true',
                      help='measure drop-mask variance instead of the ladder')
  parser.add_argument('--audit-layer', type=int, default=6)
  parser.add_argument('--audit-pool', default='all', choices=POOLS)
  parser.add_argument('--group-size', type=int, default=8)
  parser.add_argument('--realizations', type=int, default=8)
  args = parser.parse_args()

  ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
  config = build_config(ckpt, args.batch_size, args.override)
  device = torch.device('cuda:0')
  torch.manual_seed(args.seed)

  tokenizer = dataloader_mod.get_tokenizer(config)
  model = algo.MDLM.load_from_checkpoint(
    ckpt, tokenizer=tokenizer, config=config, weights_only=False)
  model = model.to(device).eval()
  utils.freeze_model(model)
  if args.feature_source == 'ar':
    # The same loader `algo._mmd_critic` uses, so the blocks profiled here are
    # the blocks the reward will read. Causal, unlike the diffusion teacher:
    # a corruption at position j can only move features at positions >= j.
    ar_ckpt = (args.ar_ckpt if os.path.isabs(args.ar_ckpt)
               else os.path.join(ROOT, args.ar_ckpt))
    backbone = mmd_reward.load_ar_backbone(ar_ckpt, config, tokenizer, device)
    print(f'profiling AR checkpoint {ar_ckpt}')
  else:
    backbone = model.backbone
    print('profiling the frozen diffusion teacher')
  extractor = AllLayerFeatures(backbone)

  _, valid_ds = dataloader_mod.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=args.seed)
  batch = next(iter(valid_ds))
  x0 = batch['input_ids'][:args.batch_size].to(device)
  valid_tokens = batch['attention_mask'][:args.batch_size].to(device)

  content = content_mask(x0, valid_tokens, tokenizer)
  generator = torch.Generator(device='cpu').manual_seed(args.seed)
  if args.mask_ratio >= 1.0:
    selected = content
  else:
    selected = content & (torch.rand(x0.shape, generator=generator)
                          .to(device) < args.mask_ratio)
  keep_rows = selected.sum(1) >= 8
  x0, selected, content = x0[keep_rows], selected[keep_rows], content[keep_rows]
  batch_size = x0.shape[0]

  unigram = torch.bincount(x0[content].cpu(), minlength=model.vocab_size).float()
  unigram = unigram.clamp_min(0) + 1e-9
  digits = digit_token_ids(tokenizer, model.vocab_size, device)

  layers = ([int(v) for v in args.layers.split(',')] if args.layers
            else [-1] + list(range(extractor.n_blocks)))
  noise_levels = [float(v) for v in args.noise_levels.split(',')]

  print(f'rows={batch_size} blocks={extractor.n_blocks} '
        f'mask_ratio={args.mask_ratio} '
        f'selected/row={selected.sum(1).float().mean():.0f} '
        f'content/row={content.sum(1).float().mean():.0f} '
        f'digit-ids={len(digits)}', flush=True)

  if args.mask_audit:
    mask_audit(extractor, x0, selected, model.mask_index, unigram, digits,
               generator, args.audit_layer, args.audit_pool, noise_levels,
               args.group_size, args.realizations)
    return

  cases = make_corruptions(x0, selected, unigram, digits, generator)

  for noise in noise_levels:
    clean_xt, _, drop = apply_noise(x0, x0, selected, model.mask_index,
                                    noise, generator)
    masks = {'all': selected,
             'keptonly': selected & ~drop,
             'maskonly': drop}
    pools = [p for p in POOLS if masks[p].sum() > 0]
    clean_feats = extractor.pooled(clean_xt, masks)

    # --- ladder -------------------------------------------------------------
    ladder, edits = {}, {}
    for name, gen in cases.items():
      gen_xt = torch.where(drop, model.mask_index, gen)
      gen_feats = extractor.pooled(gen_xt, masks)
      ladder[name] = {
        (layer, pool): reward_from(clean_feats[layer][pool],
                                   gen_feats[layer][pool]).mean().item()
        for layer in layers for pool in pools}
      # Edits the *feature map can see*: a position masked in both rows is
      # identical whatever the corruption wrote there.
      edits[name] = float(((gen != x0) & selected & ~drop).float()
                          .sum(1).mean())

    # --- retrieval ----------------------------------------------------------
    # Row i's frame with row j's answer spliced in, for every (i, j).
    scores = {(layer, pool): [] for layer in layers for pool in pools}
    order = torch.arange(batch_size, device=device)
    for shift in range(batch_size):
      src = (order + shift) % batch_size
      spliced = x0.clone()
      for row in range(batch_size):
        dst = selected[row].nonzero(as_tuple=True)[0]
        donor = selected[int(src[row])].nonzero(as_tuple=True)[0]
        take = donor[torch.arange(dst.numel(), device=device) % donor.numel()]
        spliced[row, dst] = x0[int(src[row]), take]
      feats = extractor.pooled(torch.where(drop, model.mask_index, spliced),
                               masks)
      for layer in layers:
        for pool in pools:
          scores[(layer, pool)].append(
            reward_from(clean_feats[layer][pool], feats[layer][pool]))
    # `(shift, row)`; shift 0 splices the row's own answer back in.
    top1 = {key: (torch.stack(vals).argmax(0) == 0).float().mean().item()
            for key, vals in scores.items()}

    print(f'\n=== noise_level={noise}  (dropped {drop.sum().item()} of '
          f'{selected.sum().item()} scored positions) ===')
    header = (f'{"layer":>5} {"pool":>9} {"top1":>6} ' +
              ' '.join(f'{n:>11}' for n in cases))
    print(header)
    print('-' * len(header))
    for layer in layers:
      for pool in pools:
        row = [f'{ladder[n][(layer, pool)]:11.5f}' for n in cases]
        tag = 'embed' if layer == -1 else str(layer)
        print(f'{tag:>5} {pool:>9} {top1[(layer, pool)]:6.2f} ' + ' '.join(row))
    print(f'chance top1 = {1.0 / batch_size:.2f}')

    print('visible edits/row: ' +
          '  '.join(f'{n}={edits[n]:.0f}' for n in cases))

    print('\n  structure = reward(shuffle)/reward(rand100)   0 = pure token '
          'overlap, 1 = fully order-sensitive')
    print('  semantic  = reward(digit_swap)/reward(rand_matched)   same edit '
          'count, 1.00 = the reward is only counting edits')
    print('  per-edit  = reward(rand100) / visible edits, for scale')
    for layer in layers:
      for pool in pools:
        r100 = ladder['rand100'][(layer, pool)]
        matched = ladder['rand_matched'][(layer, pool)]
        struct = ladder['shuffle'][(layer, pool)] / r100 if r100 else float('nan')
        sem = (ladder['digit_swap'][(layer, pool)] / matched if matched
               else float('nan'))
        per = r100 / max(edits['rand100'], 1.0)
        tag = 'embed' if layer == -1 else str(layer)
        print(f'{tag:>5} {pool:>9} structure={struct:6.3f} '
              f'semantic={sem:6.3f} per-edit={per:9.5f}')


if __name__ == '__main__':
  main()
