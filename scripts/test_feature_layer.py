"""`mmd_feature_layer` resolution on both OWT feature sources, incl. -1.

-1 is the token embeddings: a pure lookup with no attention, so its pooled
features carry the token multiset and nothing about order. Run on a GPU --
mdlm-owt's rotary embedding is a Triton kernel with no CPU backend.
"""
import os
import sys

import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward as M

ok = []
def check(n, c): ok.append((n, bool(c))); print(f"{'PASS' if c else 'FAIL'}  {n}")

dit = transformers.AutoModelForMaskedLM.from_pretrained(
    'kuleshov-group/mdlm-owt', trust_remote_code=True).to('cuda').eval()
ar = M.load_owt_ar_backbone('kuleshov-group/owt-ar', 'cuda')

for name, core in (('hf_dit (mdlm-owt)', dit), ('owt-ar adapter', ar)):
    n_blocks = len(M.ar_decoder_layers(core))
    emb = M.embedding_module(core)
    check(f'{name}: embedding_module found ({type(emb).__name__})', emb is not None)
    mods, idx = M.resolve_feature_layers(core, -1)
    check(f'{name}: -1 resolves to the embedding', mods[0] is emb and idx == [-1])
    mods, idx = M.resolve_feature_layers(core, None)
    check(f'{name}: None still resolves to the middle block ({n_blocks//2})',
          idx == [n_blocks // 2])
    mods, idx = M.resolve_feature_layers(core, [-1, 0, n_blocks - 1])
    check(f'{name}: a list may mix -1 with real blocks', idx == [-1, 0, n_blocks - 1])
    for bad in (-2, n_blocks):
        try:
            M.resolve_feature_layers(core, bad)
            check(f'{name}: {bad} rejected', False)
        except ValueError as e:
            check(f'{name}: {bad} rejected', '-1' in str(e))

# -1 must actually yield rank-3 features through the real hook path
x = torch.randint(0, 50257, (2, 16), device='cuda')
with M.capture_layer_hidden(dit, -1) as (feats, idx):
    dit(x, torch.zeros(2, device='cuda'))
h = feats[-1]
check(f'hook on -1 captures rank-3 features {tuple(h.shape)}', h.ndim == 3)

# bag-of-embeddings floor: a pure lookup has no context, so permuting the
# tokens must leave the POOLED feature identical.
sel = torch.ones(2, 16, dtype=torch.bool, device='cuda')
perm = x[:, torch.randperm(16)]
with M.capture_layer_hidden(dit, -1) as (f2, _):
    dit(perm, torch.zeros(2, device='cuda'))
d_emb, _ = M.sequence_pooled_delta(h, f2[-1], sel)
with M.capture_layer_hidden(dit, 6) as (f3, _):
    dit(x, torch.zeros(2, device='cuda'))
with M.capture_layer_hidden(dit, 6) as (f4, _):
    dit(perm, torch.zeros(2, device='cuda'))
d_blk, _ = M.sequence_pooled_delta(f3[6], f4[6], sel)
check(f'-1 is order-blind: pooled delta under a permutation = {d_emb.abs().max():.2e}',
      float(d_emb.abs().max()) < 1e-4)
check(f'block 6 is NOT order-blind ({d_blk.abs().max():.3f})',
      float(d_blk.abs().max()) > 1e-2)

print(f"\n{sum(v for _, v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _, v in ok) else 1)
