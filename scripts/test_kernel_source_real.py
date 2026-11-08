"""`multi_draw_reward` on the real feature map. Needs a GPU."""
import sys, torch, transformers
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mmd_reward as M
ok=[]
def chk(n,c): ok.append((n,bool(c))); print(f"{'PASS' if c else 'FAIL'}  {n}")

dev='cuda'; B, L, K = 4, 256, 4
torch.manual_seed(0)
model = transformers.AutoModelForMaskedLM.from_pretrained(
    'kuleshov-group/mdlm-owt', trust_remote_code=True).to(dev).eval()
V = model.config.vocab_size
critic = M.FrozenBackboneFeatures(model, V-1, layer_index=None, noise_level=0.1)
critic.kernel = 'rbf'; critic.rbf_block_size = 0; critic.rbf_alpha = None

x0 = torch.randint(0, V-1, (B, L), device=dev)
sel = torch.zeros(B, L, dtype=torch.bool, device=dev); sel[:, 32:160] = True

def draws_from(corrupt_frac):
    d = []
    for _ in range(K):
        hit = (torch.rand(B, L, device=dev) < corrupt_frac) & sel
        rnd = torch.randint(0, V-1, (B, L), device=dev)
        d.append(torch.where(hit, rnd, x0))
    return torch.stack(d)

with torch.no_grad():
    perfect = torch.stack([x0]*K)
    r_perf, a_perf = critic.multi_draw_reward(x0, perfect, sel, None)
    r_mild, _ = critic.multi_draw_reward(x0, draws_from(0.5), sel, None)
    r_bad,  _ = critic.multi_draw_reward(x0, draws_from(0.9), sel, None)
    r_near, _ = critic.multi_draw_reward(x0, draws_from(0.15), sel, None)

chk(f'shape is per row {tuple(r_perf.shape)}', r_perf.shape == (B,))
chk(f'all rows active', bool(a_perf.all()))
chk(f'perfect draws preserve positive reward ({r_perf.mean():.4f})',
    float(r_perf.mean()) > 0)
chk(f'reward is monotone where it is live '
    f'(perfect {r_perf.mean():.4f} > 50% {r_mild.mean():.4f} > 90% {r_bad.mean():.4f})',
    float(r_perf.mean()) > float(r_mild.mean()) > float(r_bad.mean()))
# Negative estimates near the reference retain positive rewards instead of
# saturating at zero. Exact analytic cases are covered by test_kernel_source.py.
chk(f'near-correct draws preserve positive reward ({r_near.mean():.4f})',
    float(r_near.mean()) > 0)

# linear kernel must also work through this path
critic.kernel = 'linear'
with torch.no_grad():
    l_perf, _ = critic.multi_draw_reward(x0, perfect, sel, None)
    l_bad, _ = critic.multi_draw_reward(x0, draws_from(0.9), sel, None)
chk(f'linear: perfect {l_perf.mean():.4f} > bad {l_bad.mean():.4f}',
    float(l_perf.mean()) > float(l_bad.mean()))

# Unscored tokens still provide context through attention. The scoring mask
# excludes their feature vectors from MMD; it does not remove their context.
# A row with no scored positions, however, must be inactive and score zero.
critic.kernel = 'rbf'
d = draws_from(0.3)
partly_active = sel.clone(); partly_active[0] = False
with torch.no_grad():
    masked_reward, active = critic.multi_draw_reward(x0, d, partly_active, None)
chk('empty scoring mask gives zero reward and an inactive row',
    float(masked_reward[0]) == 0.0 and active.tolist() == [False, True, True, True]
    and bool(torch.isfinite(masked_reward).all()))

print(f"\n{sum(v for _,v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _,v in ok) else 1)
