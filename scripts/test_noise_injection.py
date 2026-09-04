"""Offline checks for the generator input-noise conditioning (D-MMD Sec. 6.5).

Run: `python scripts/test_noise_injection.py` -- no GPU, no checkpoint download.

Covers the wrapper (`algo._NoiseConditionedBackbone`) and the MDLM-side wiring:
optimizer param groups, the checkpoint key mirroring, and the embedding-std
lookup. The stand-in backbone mirrors `kuleshov-group/mdlm-owt`'s shape -- a
`vocab_embed` module holding an `embedding` Parameter -- and its per-entry std
(0.309), which is the number the injection is calibrated against.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import omegaconf
import torch

import algo

ok = []


def check(name, cond):
  ok.append((name, bool(cond)))
  print(f"{'PASS' if cond else 'FAIL'}  {name}")


class FakeEmbed(torch.nn.Module):
  """Stands in for `EmbeddingLayer`: an `embedding` Parameter (vocab, dim)."""

  def __init__(self, vocab, dim, std=0.309):
    super().__init__()
    self.embedding = torch.nn.Parameter(torch.randn(vocab, dim) * std)

  def forward(self, x):
    return self.embedding[x]


class FakeBackbone(torch.nn.Module):
  def __init__(self, vocab=64, dim=32):
    super().__init__()
    self.vocab_embed = FakeEmbed(vocab, dim)
    self.out = torch.nn.Linear(dim, vocab)

  def forward(self, xt, sigma=None):
    return self.out(self.vocab_embed(xt))


def wrap(backbone, **kw):
  return algo._wrap_backbone_with_noise(
    backbone, kw.pop('noise_dim', 8), 32, **kw)


torch.manual_seed(0)
xt = torch.randint(0, 64, (4, 16))

# --- the injection is calibrated to the real embedding --------------------
base = FakeBackbone()
check('_embedding_std reads the real embedding (~0.309)',
      abs(algo._embedding_std(base.vocab_embed) - 0.309) < 0.02)
check('_embedding_std falls back to 0.02',
      algo._embedding_std(torch.nn.Identity()) == 0.02)

# --- zero-init is bit-identical to the unwrapped backbone -----------------
ref = base(xt).clone()
check('zero-init forward is bit-identical to unwrapped',
      torch.equal(wrap(copy.deepcopy(base), noise_scale=0.0)(xt), ref))

# --- noise actually perturbs, and is redrawn each pass --------------------
w1 = wrap(copy.deepcopy(base), noise_scale=1.0)
a, b = w1(xt), w1(xt)
check('noise_scale>0 changes the output', not torch.equal(a, ref))
check('noise is redrawn per forward', not torch.equal(a, b))
check('the draw is released after the forward', w1._noise is None)
check('shared_noise broadcasts one vector per sample',
      wrap(copy.deepcopy(base), noise_scale=1.0, shared_noise=True)
      (xt).shape == ref.shape)

# --- the warmup ramp ------------------------------------------------------
wr = wrap(copy.deepcopy(base), noise_scale=1.0, warmup_steps=100)
wr.set_noise_step(0)
check('gain 0 at step 0', wr.noise_gain == 0.0)
check('gain 0 forward is bit-identical to unwrapped', torch.equal(wr(xt), ref))
wr.set_noise_step(50)
check('gain 0.5 halfway', abs(wr.noise_gain - 0.5) < 1e-9)
wr.set_noise_step(100)
check('gain 1.0 at warmup end', wr.noise_gain == 1.0)
check('warmed-up forward differs', not torch.equal(wr(xt), ref))
wr.set_noise_step(10_000)
check('gain clamps at 1.0', wr.noise_gain == 1.0)
check('no ramp => gain 1 even without set_noise_step',
      wrap(copy.deepcopy(base), noise_scale=1.0).noise_gain == 1.0)

# --- hold_rms pins the magnitude, leaving only the direction learnable ----
hr = wrap(copy.deepcopy(base), noise_scale=0.4, hold_rms=True)
with torch.no_grad():
  hr.noise_proj.weight.mul_(37.0)
check('hold_rms pins rms(effective_weight) to noise_scale',
      abs(hr.effective_weight().pow(2).mean().sqrt().item() - 0.4) < 1e-5)

try:
  wrap(copy.deepcopy(base), noise_scale=0.0, warmup_steps=100)
  check('warmup with noise_scale=0 raises', False)
except AssertionError:
  check('warmup with noise_scale=0 raises', True)

# --- zero-init still receives gradient (the residual-branch property) -----
g = wrap(copy.deepcopy(base), noise_scale=0.0)
g(xt).square().mean().backward()
check('zero-init noise_proj gets a non-zero gradient',
      g.noise_proj.weight.grad is not None
      and g.noise_proj.weight.grad.abs().sum().item() > 0)

# --- the output rescaling makes snr == rms(W) ----------------------------
s = wrap(copy.deepcopy(base), noise_scale=0.6)
w = s.effective_weight().detach()
inject_std = float(s.noise_out_scale) * w.norm() / (w.shape[0] ** 0.5)
snr = inject_std / algo._embedding_std(s.backbone.vocab_embed)
check('snr ~= rms(W)',
      abs(snr.item() - w.pow(2).mean().sqrt().item()) < 0.05)

# --- state_dict hygiene ---------------------------------------------------
keys = set(wrap(copy.deepcopy(base)).state_dict())
check('noise_out_scale is not in state_dict', 'noise_out_scale' not in keys)
check('noise_proj.weight is in state_dict', 'noise_proj.weight' in keys)
check('inner backbone keys keep a backbone. prefix',
      any(k.startswith('backbone.') for k in keys))


# ==== MDLM-side wiring =====================================================

def cfg(noise_lr=None):
  return omegaconf.OmegaConf.create(
    {'optim': {'lr': 1e-6}, 'adversarial_distill': {'noise_lr': noise_lr}})


class FakeGroups:
  _student_param_groups = algo.MDLM._student_param_groups


plain = FakeBackbone()
wrapped = wrap(FakeBackbone(), noise_scale=0.5)

f = FakeGroups(); f.backbone = plain; f.config = cfg(None)
check('unwrapped, no noise_lr => one group', len(f._student_param_groups()) == 1)
f = FakeGroups(); f.backbone = wrapped; f.config = cfg(None)
check('wrapped, no noise_lr => one group', len(f._student_param_groups()) == 1)
f = FakeGroups(); f.backbone = plain; f.config = cfg(1e-3)
check('unwrapped, noise_lr set => still one group',
      len(f._student_param_groups()) == 1)

f = FakeGroups(); f.backbone = wrapped; f.config = cfg(1e-3)
groups = f._student_param_groups()
check('wrapped + noise_lr => two groups', len(groups) == 2)
check('group lrs are (optim.lr, noise_lr)',
      groups[0]['lr'] == 1e-6 and groups[1]['lr'] == 1e-3)
check('noise group holds exactly noise_proj params',
      len(groups[1]['params']) == len(list(wrapped.noise_proj.parameters())))
check('every backbone param is covered exactly once',
      len(groups[0]['params']) + len(groups[1]['params'])
      == len(list(wrapped.parameters())))
ids = [id(p) for group in groups for p in group['params']]
check('no parameter appears twice', len(ids) == len(set(ids)))
torch.optim.AdamW(groups, lr=1e-6)
check('AdamW accepts the groups', True)


def mirror(state_dict, student_is_wrapped):
  """The block from `MDLM.on_load_checkpoint`, exercised in isolation."""
  import collections
  state_dict = collections.OrderedDict(state_dict)
  prefix = 'backbone.backbone.' if student_is_wrapped else 'backbone.'
  for key, value in list(state_dict.items()):
    if key.startswith(prefix):
      suffix = key[len(prefix):]
      state_dict.setdefault(f'teacher.{suffix}', value)
      state_dict.setdefault(f'fake_model.{suffix}', value)
  return state_dict


def migrate_plain_to_wrapped(state_dict, noise_state):
  """Checkpoint namespace migration from `MDLM.on_load_checkpoint`."""
  import collections
  state_dict = collections.OrderedDict(state_dict)
  has_noise_state = any(
    key.startswith('backbone.noise_proj.') for key in state_dict)
  plain_keys = [key for key in state_dict if key.startswith('backbone.')]
  if plain_keys and not has_noise_state:
    for key in plain_keys:
      suffix = key[len('backbone.'):]
      state_dict[f'backbone.backbone.{suffix}'] = state_dict.pop(key)
    for name, value in noise_state.items():
      state_dict[f'backbone.noise_proj.{name}'] = value
  return state_dict


# The critics are UNWRAPPED, so mirroring the student's keys has to strip the
# wrapper's extra level and drop the projection, or strict resume rejects both.
critic = FakeBackbone()
critic_keys = set(critic.state_dict())
student_sd = {f'backbone.{k}': v for k, v in wrapped.state_dict().items()}
check('wrapped student state_dict has the doubled prefix',
      any(k.startswith('backbone.backbone.') for k in student_sd))

mirrored = mirror(student_sd, student_is_wrapped=True)
teacher_keys = {k[len('teacher.'):] for k in mirrored if k.startswith('teacher.')}
fake_keys = {k[len('fake_model.'):] for k in mirrored
             if k.startswith('fake_model.')}
check('teacher gets exactly the unwrapped critic keys', teacher_keys == critic_keys)
check('fake_model gets exactly the unwrapped critic keys', fake_keys == critic_keys)
check('no noise_proj leaks into the critics',
      not any('noise_proj' in k for k in teacher_keys | fake_keys))
critic.load_state_dict({k: mirrored[f'teacher.{k}'] for k in teacher_keys},
                       strict=True)
check('mirrored keys load into the unwrapped critic strictly', True)

plain_sd = {f'backbone.{k}': v for k, v in plain.state_dict().items()}
check('unwrapped path still mirrors exactly the critic keys',
      {k[len('teacher.'):] for k in mirror(plain_sd, False)
       if k.startswith('teacher.')} == critic_keys)

buggy = {k[len('teacher.'):] for k in mirror(student_sd, False)
         if k.startswith('teacher.')}
check('regression guard: the un-stripped prefix produces wrong keys',
      buggy != critic_keys and 'noise_proj.weight' in buggy)

# HF MDLM's plain module already contains its own `.backbone`, so its keys have
# two backbone components before the noise wrapper is introduced. The
# projection key, not prefix depth, is what distinguishes a resume from a
# plain-teacher finetune.
hf_plain = {'backbone.backbone.vocab_embed.embedding': torch.ones(2)}
hf_migrated = migrate_plain_to_wrapped(
  hf_plain, wrapped.noise_proj.state_dict())
check('nested HF plain key gains exactly one wrapper level',
      'backbone.backbone.backbone.vocab_embed.embedding' in hf_migrated)
check('plain migration initializes noise projection',
      'backbone.noise_proj.weight' in hf_migrated)
check('noise-conditioned resume is not migrated twice',
      set(migrate_plain_to_wrapped(hf_migrated,
                                   wrapped.noise_proj.state_dict()))
      == set(hf_migrated))


class FakeLM(torch.nn.Module):
  """`_student_embed_std` must not cache the embedding MODULE on `self`.

  Assigning a submodule registers it, so the embedding is emitted a second
  time in `state_dict` and resume dies on an unexpected key.
  """

  _student_embed_std = algo.MDLM._student_embed_std

  def __init__(self, backbone):
    super().__init__()
    self.backbone = backbone
    self._embed_module_name = None


for label, bb in (('unwrapped', FakeBackbone()),
                  ('wrapped', wrap(FakeBackbone(), noise_scale=0.5))):
  lm = FakeLM(bb)
  before = set(lm.state_dict())
  std = lm._student_embed_std()
  check(f'{label}: _student_embed_std adds no state_dict key',
        before == set(lm.state_dict()))
  check(f'{label}: reports the real embedding std', abs(std - 0.309) < 0.03)
  with torch.no_grad():
    embed = (bb.backbone.vocab_embed if hasattr(bb, 'noise_proj')
             else bb.vocab_embed)
    embed.embedding.mul_(2.0)
  check(f'{label}: re-reads the embedding after it drifts',
        abs(lm._student_embed_std() - 2 * std) < 1e-4)

print()
failed = [name for name, passed in ok if not passed]
print(f'{len(ok) - len(failed)}/{len(ok)} passed')
if failed:
  print('FAILED:', failed)
  sys.exit(1)
