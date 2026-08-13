"""Stage-2 gate: the downloaded teacher must load into IDLM's UNet strictly.

This is the real port test. Asserting **zero** missing and unexpected keys is
what pins `models/unet.py`'s module names to DUO's; any rename there shows up
here rather than as a mysteriously bad FID later.

  python scripts/test_cifar10_teacher_load.py [path/to/teacher.safetensors]
"""
import os
import sys

import omegaconf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algo  # noqa: E402
import models  # noqa: E402

DEFAULT_CKPT = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  'downloads', 'mdlm', 'cosine_time_uncond', '1500000.safetensors')


def build_unet(vocab_size=257):
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  data = omegaconf.OmegaConf.load(
    os.path.join(root, 'configs', 'data', 'cifar10.yaml'))
  model = omegaconf.OmegaConf.load(
    os.path.join(root, 'configs', 'model', 'unet.yaml'))
  cfg = omegaconf.OmegaConf.create({'data': data, 'model': model})
  cfg.model.size = cfg.data.size
  cfg.model.length = cfg.data.length
  cfg.model.time_embed_dim = cfg.model.ch
  cfg.model.cond_dim = cfg.model.ch
  cfg.model.hidden_size = cfg.model.ch
  return cfg, models.unet.UNet(cfg, vocab_size=vocab_size)


def main():
  # Fetches from the hub if it is not on disk, exactly as a training run would
  # (`algo._resolve_teacher_ckpt_path`), so this gate needs no separate
  # download step. A path outside `downloads/` still has to exist.
  path = algo._resolve_teacher_ckpt_path(
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT)

  import safetensors.torch
  from safetensors import safe_open
  with safe_open(path, 'pt') as f:
    meta = f.metadata()
  print('checkpoint metadata:', meta)

  _, net = build_unet()
  n_params = sum(p.numel() for p in net.parameters())
  own = dict(net.state_dict())
  ckpt = safetensors.torch.load_file(path, device='cpu')

  missing = sorted(k for k in own if k not in ckpt)
  unexpected = sorted(k for k in ckpt if k not in own)
  mismatched = sorted(
    (k, tuple(own[k].shape), tuple(ckpt[k].shape))
    for k in own if k in ckpt and own[k].shape != ckpt[k].shape)
  print(f'params: {n_params/1e6:.2f}M | state_dict keys: {len(own)} | '
        f'checkpoint tensors: {len(ckpt)}')
  assert not missing, f'missing keys: {missing[:10]}'
  assert not unexpected, f'unexpected keys: {unexpected[:10]}'
  assert not mismatched, f'shape mismatches: {mismatched[:10]}'
  print('  zero missing / unexpected keys, no shape mismatches  OK')

  algo._load_pretrained_backbone(net, path)
  assert net.cond_map.embedding_table.weight.shape == (11, 512)
  print('  cond_map is [11, 512] = 10 classes + null class  OK')

  # A forward from the all-mask prior must be finite and, since nothing is
  # observed, close to the pixel *marginal*: near-max entropy, mean around the
  # CIFAR-10 mean pixel (~121).
  net.eval()
  x = torch.full((2, 3072), 256, dtype=torch.long)
  with torch.no_grad():
    out = net(x, torch.zeros(2))
  assert out.shape == (2, 3072, 257), out.shape
  assert torch.isfinite(out).all()
  probs = out.softmax(-1)
  entropy = float(-(probs * out.log_softmax(-1)).sum(-1).mean())
  mean_pixel = float(out.argmax(-1).float().mean())
  print(f'  all-mask forward: entropy {entropy:.3f} nats '
        f'(max {torch.tensor(257.).log():.3f}), mean argmax pixel '
        f'{mean_pixel:.1f}  OK')
  assert 3.0 < entropy < 5.55, entropy
  assert 60 < mean_pixel < 190, mean_pixel
  print(f'  reference_std (noise calibration): {net.reference_std:.4f}')
  print('Stage 2 (load): all checks passed. Now run '
        'scripts/fid_cifar10_teacher.sh with STEPS_LIST=1024.')


if __name__ == '__main__':
  main()
