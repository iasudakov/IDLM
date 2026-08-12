"""Stage-1 CPU unit tests for the CIFAR-10 D-MMD branch (README_cifar10.md).

Runs without a GPU and without the downloaded teacher, so it is the fast gate
before anything expensive:

  python scripts/test_cifar10_unit.py

Checks, in order:
  1. dataset -> 3072 int64 tokens in [0, 255]
  2. `batch_decode` round-trips an image to uint8 (B, 3, 32, 32) bit-exactly
  3. `UNet(x, sigma)` -> (B, 3072, 257) finite logits, and `mask_index == 256`
  4. `Cosine` matches DUO's schedule values
  5. the image noise wrapper reports `noise/snr ~= rms(W)` and is a
     bit-identical no-op at gain 0
"""
import os
import sys

import numpy as np
import omegaconf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algo  # noqa: E402
import dataloader  # noqa: E402
import models  # noqa: E402
import trainer_base  # noqa: E402

CACHE_DIR = os.path.expanduser(
  os.environ.get('CIFAR10_CACHE_DIR', '~/data/cifar10'))


def _config():
  base = omegaconf.OmegaConf.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'configs', 'config.yaml'))
  data = omegaconf.OmegaConf.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'configs', 'data', 'cifar10.yaml'))
  model = omegaconf.OmegaConf.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'configs', 'model', 'unet.yaml'))
  cfg = omegaconf.OmegaConf.create({'data': data, 'model': model,
                                    'algo': {'name': 'mdlm_dmmd'}})
  del base
  cfg.data.cache_dir = CACHE_DIR
  # Resolve the interpolations that reference the `data` group.
  cfg.model.size = cfg.data.size
  cfg.model.length = cfg.data.length
  cfg.model.time_embed_dim = cfg.model.ch
  cfg.model.cond_dim = cfg.model.ch
  cfg.model.hidden_size = cfg.model.ch
  return cfg


def test_dataset_and_tokenizer(cfg):
  tokenizer = dataloader.get_tokenizer(cfg)
  assert tokenizer.vocab_size == 257, tokenizer.vocab_size
  assert tokenizer.mask_token_id == 256, tokenizer.mask_token_id
  assert tokenizer.bos_token_id is None and tokenizer.eos_token_id is None

  ds = dataloader.get_dataset(
    'cifar10', tokenizer, wrap=False, mode='train',
    cache_dir=cfg.data.cache_dir, augment=False)
  item = ds[0]
  ids = item['input_ids']
  assert ids.shape == (3072,), ids.shape
  assert ids.dtype == torch.int64, ids.dtype
  assert int(ids.min()) >= 0 and int(ids.max()) <= 255, (
    int(ids.min()), int(ids.max()))
  assert item['attention_mask'].shape == (3072,)
  print(f'  [1] dataset: {ids.shape} {ids.dtype} '
        f'in [{int(ids.min())}, {int(ids.max())}]  OK')

  # Round trip: (c h w) flat -> (3, 32, 32) uint8 must equal the raw PIL array
  # transposed the same way, i.e. no channel/row transposition crept in.
  batch = torch.stack([ds[i]['input_ids'] for i in range(4)])
  decoded = tokenizer.batch_decode(batch)
  assert decoded.shape == (4, 3, 32, 32), decoded.shape
  assert decoded.dtype == torch.uint8, decoded.dtype
  raw = np.array(ds._dataset[0][0])  # (32, 32, 3) HWC uint8
  assert np.array_equal(decoded[0].numpy(), raw.transpose(2, 0, 1)), (
    'batch_decode does not round-trip the raw image')
  print('  [2] batch_decode round-trip is bit-exact  OK')
  return tokenizer


def test_unet(cfg, tokenizer):
  vocab_size = tokenizer.vocab_size
  net = models.unet.UNet(cfg, vocab_size=vocab_size)
  # eval(): the ResBlocks carry dropout, so two forwards in train mode differ
  # for reasons that have nothing to do with what is being tested here.
  net.eval()
  n_params = sum(p.numel() for p in net.parameters())
  x = torch.randint(0, 257, (2, 3072))
  sigma = torch.zeros(2)
  with torch.no_grad():
    out = net(x, sigma)
  assert out.shape == (2, 3072, 257), out.shape
  assert torch.isfinite(out).all()
  assert net.cond_map.embedding_table.weight.shape == (11, 512), (
    net.cond_map.embedding_table.weight.shape)
  print(f'  [3] UNet: {n_params/1e6:.2f}M params, out {tuple(out.shape)}, '
        f'cond_map {tuple(net.cond_map.embedding_table.weight.shape)}  OK')

  # The null-class default must be a real conditioning vector, not a no-op.
  with torch.no_grad():
    explicit = net(x, sigma,
                   class_cond=torch.full((2,), 10, dtype=torch.long))
  assert torch.allclose(out, explicit), (
    'class_cond=None must equal the explicit null class')
  print('  [3] class_cond=None == explicit null class (index 10)  OK')
  return net


def test_mask_index(cfg, tokenizer):
  # `AbsorbingState.__init__` reads the mask token off the tokenizer; only
  # replicate that arithmetic here (constructing the LightningModule would pull
  # in the whole trainer).
  vocab_size = tokenizer.vocab_size
  assert getattr(tokenizer, 'mask_token', None) is not None
  mask_index = tokenizer.mask_token_id
  assert mask_index == 256, mask_index
  assert vocab_size == 257, vocab_size
  print(f'  [3] mask_index == {mask_index}, vocab_size == {vocab_size}  OK')


def test_cosine():
  cosine = trainer_base.Cosine(1e-3)
  t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
  dalpha_t, alpha_t = cosine(t)
  # Reference values recomputed from DUO's formula.
  eps, half_pi = 1e-3, np.pi / 2
  tt = (1 - eps) * t.numpy()
  ref_alpha = 1 - np.cos(half_pi * (1 - tt))
  ref_dalpha = -np.sin(half_pi * (1 - tt)) * half_pi
  assert np.allclose(alpha_t.numpy(), ref_alpha, atol=1e-6)
  assert np.allclose(dalpha_t.numpy(), ref_dalpha, atol=1e-6)
  # alpha(0) == 1 (clean data), alpha(1) ~ 0 (pure mask), monotone decreasing.
  assert abs(float(alpha_t[0]) - 1.0) < 1e-5
  assert float(alpha_t[-1]) < 2e-3
  assert (alpha_t[1:] < alpha_t[:-1]).all()
  # Inverse.
  assert np.allclose(cosine.get_t_for_alpha(alpha_t).numpy(), tt, atol=1e-5)
  # log-linear stays exactly as it was.
  loglin = trainer_base.LogLinear()
  d, a = loglin(t)
  assert np.allclose(a.numpy(), 1 - (1 - 1e-3) * t.numpy())
  assert d == -(1 - 1e-3)
  print('  [4] Cosine matches DUO; LogLinear unchanged  OK')


def test_noise_wrapper(cfg, net):
  x = torch.randint(0, 257, (2, 3072))
  sigma = torch.zeros(2)
  with torch.no_grad():
    base_out = net(x, sigma)

  # gain 0 (warmup not started) must be a bit-identical no-op.
  torch.manual_seed(0)
  wrapped = algo._wrap_backbone_with_noise(
    net, noise_dim=8, default_hidden=cfg.model.hidden_size,
    noise_scale=1.0, warmup_steps=100)
  assert isinstance(wrapped, algo._ImageNoiseConditionedBackbone)
  assert wrapped.resolutions == [32, 16, 8, 4, 2, 1], wrapped.resolutions
  assert wrapped.noise_proj.weight.shape == (128, 48, 1, 1), (
    wrapped.noise_proj.weight.shape)
  wrapped.set_noise_step(0)
  assert wrapped.noise_gain == 0.0
  with torch.no_grad():
    zero_out = wrapped(x, sigma)
  assert torch.equal(base_out, zero_out), (
    'gain 0 must be bit-identical to the unwrapped backbone')
  print('  [5] noise wrapper is a bit-identical no-op at gain 0  OK')

  # snr == rms(W): measure the injected vector's std directly against
  # reference_std and compare with what the diagnostics would report.
  wrapped.set_noise_step(100)
  assert wrapped.noise_gain == 1.0
  w = wrapped.effective_weight().detach()
  rms_w = float(w.pow(2).mean().sqrt())
  out_scale = float(wrapped.noise_out_scale)
  inject_std = out_scale * float(w.norm()) / (w.shape[0] ** 0.5)
  reference_std = float(net.reference_std)
  snr = inject_std / reference_std
  assert abs(snr - rms_w) / rms_w < 1e-4, (snr, rms_w)
  print(f'  [5] noise/snr = {snr:.4f} ~= rms(W) = {rms_w:.4f} '
        f'(reference_std = {reference_std:.4f})  OK')

  # Empirical check that the injection really lands at `reference_std * rms(W)`.
  torch.manual_seed(0)
  z = wrapped._draw_noise(4, torch.device('cpu'))
  assert z.shape == (4, 48, 32, 32), z.shape
  assert abs(float(z.std()) - 1.0) < 0.05, float(z.std())
  with torch.no_grad():
    injected = out_scale * torch.nn.functional.conv2d(
      z, w, wrapped.noise_proj.bias)
  measured = float(injected.std()) / reference_std
  assert abs(measured - rms_w) / rms_w < 0.1, (measured, rms_w)
  print(f'  [5] measured injection ratio {measured:.4f} '
        f'(predicted {rms_w:.4f})  OK')

  # `shared_noise` collapses to the single global level.
  shared = algo._wrap_backbone_with_noise(
    net, noise_dim=8, default_hidden=cfg.model.hidden_size,
    shared_noise=True, noise_scale=0.0)
  assert shared.resolutions == [1], shared.resolutions
  torch.manual_seed(0)
  zs = shared._draw_noise(2, torch.device('cpu'))
  assert zs.shape == (2, 8, 32, 32)
  assert torch.allclose(zs[:, :, 0, 0], zs[:, :, 17, 29]), (
    'shared noise must be constant over space')
  print('  [5] shared_noise = one global vector per sample  OK')

  # `noise_stream` must be reproducible and independent of the global RNG.
  torch.manual_seed(123)
  with wrapped.noise_stream(7):
    a = wrapped._draw_noise(2, torch.device('cpu'))
  torch.manual_seed(999)
  with wrapped.noise_stream(7):
    b = wrapped._draw_noise(2, torch.device('cpu'))
  assert torch.equal(a, b), 'noise_stream(seed) must replay identically'
  print('  [5] noise_stream(seed) replays identically  OK')


def main():
  torch.manual_seed(0)
  cfg = _config()
  print('Stage 1: CPU unit tests')
  tokenizer = test_dataset_and_tokenizer(cfg)
  test_mask_index(cfg, tokenizer)
  net = test_unet(cfg, tokenizer)
  test_cosine()
  test_noise_wrapper(cfg, net)
  print('Stage 1: all checks passed.')


if __name__ == '__main__':
  main()
