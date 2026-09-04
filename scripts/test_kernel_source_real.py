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
chk(f'perfect draws score ~0 ({r_perf.mean():.4f})', float(r_perf.mean()) > -1e-3)
chk(f'reward is monotone where it is live '
    f'(perfect {r_perf.mean():.4f} > 50% {r_mild.mean():.4f} > 90% {r_bad.mean():.4f})',
    float(r_perf.mean()) > float(r_mild.mean()) > float(r_bad.mean()))
# Documented property, not a defect: `-mmd2.clamp_min(0)` returns exactly 0
# once the unbiased estimate goes negative, which it does whenever the draws
# are close to the reference. The raw statistic stays ordered below that point
# (-0.0038 perfect .. +0.0067 at 90% on this probe); the clamp is what flattens
# it. So the rbf reward carries no gradient in the near-converged regime.
chk(f'reward saturates at 0 for near-correct draws ({r_near.mean():.4f})',
    abs(float(r_near.mean())) < 1e-6)

# linear kernel must also work through this path
critic.kernel = 'linear'
with torch.no_grad():
    l_perf, _ = critic.multi_draw_reward(x0, perfect, sel, None)
    l_bad, _ = critic.multi_draw_reward(x0, draws_from(0.9), sel, None)
chk(f'linear: perfect {l_perf.mean():.4f} > bad {l_bad.mean():.4f}',
    float(l_perf.mean()) > float(l_bad.mean()))

# unscored positions must not matter
critic.kernel = 'rbf'
d = draws_from(0.3)
d2 = d.clone(); d2[:, :, :32] = torch.randint(0, V-1, (K, B, 32), device=dev)
with torch.no_grad():
    torch.manual_seed(7); ra, _ = critic.multi_draw_reward(x0, d, sel, None)
    torch.manual_seed(7); rb, _ = critic.multi_draw_reward(x0, d2, sel, None)
chk(f'unscored positions ignored (max |d| {(ra-rb).abs().max():.2e})',
    float((ra-rb).abs().max()) < 1e-4)

print(f"\n{sum(v for _,v in ok)}/{len(ok)} passed")
sys.exit(0 if all(v for _,v in ok) else 1)
