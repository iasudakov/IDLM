"""Diagnostic: how much of the teacher's FID is the *null class*, not the port?

The released `jdeschena/duo2-cifar10` weights were trained class-conditionally
with `training.class_dropout_p = 0.1`, so the null-class embedding (index 10)
saw only ~10% of training, and DUO's own `_eval_fid` samples *class
conditionally* with labels taken from the real batch. This branch samples
unconditionally (D-MMD Sec. 6.1), i.e. through that under-trained embedding.

This script runs both arms back to back at identical sample count, step count
and seed, changing **only** which class embedding the backbone reads:

  * `null`  -- index `num_classes`, what `UNet.forward` defaults to.
  * `random`-- a class drawn uniformly per sample, which approximates the class
               marginal, i.e. what DUO measures.

If `random` is much better, the gap to the paper's Table 1 is the checkpoint's
conditioning, not the pixel layout / vocabulary / schedule / sampler. If both
are equally bad, something in the port is wrong.

Absolute FIDs here are inflated by the small sample count -- only the
*difference* between the two arms is the signal.

  python scripts/test_cifar10_null_class.py [--n 2000] [--steps 128]
"""
import argparse
import os
import sys

import omegaconf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algo  # noqa: E402
import dataloader  # noqa: E402
import metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(
  ROOT, 'downloads', 'mdlm', 'cosine_time_uncond', '1500000.safetensors')

# `configs/config.yaml` interpolates through resolvers that `main.py` registers
# at import; this script composes the config by hand, so it must register them
# too.
for _name, _fn in (('cwd', os.getcwd),
                   ('device_count', torch.cuda.device_count),
                   ('eval', eval),
                   ('div_up', lambda x, y: (x + y - 1) // y)):
  omegaconf.OmegaConf.register_new_resolver(_name, _fn, replace=True)


def build_config(steps, ckpt):
  config = omegaconf.OmegaConf.load(os.path.join(ROOT, 'configs', 'config.yaml'))
  for group, name in (('data', 'cifar10'), ('model', 'unet'),
                      ('noise', 'cosine'), ('algo', 'mdlm_dmmd')):
    config[group] = omegaconf.OmegaConf.load(
      os.path.join(ROOT, 'configs', group, f'{name}.yaml'))
  config.algo.backbone = 'unet'
  config.adversarial_distill.is_distill = False
  config.adversarial_distill.teacher_ckpt_path = ckpt
  config.sampling.predictor = 'ancestral_cache'
  config.sampling.steps = steps
  config.loader.batch_size = 100
  config.loader.eval_batch_size = 100
  config.loader.global_batch_size = 100
  config.loader.num_workers = 4
  config.trainer.devices = 1
  config.eval.compute_fid = False
  config.eval.generate_samples = False
  # `hydra.*` interpolates through resolvers only Hydra itself provides
  # (`${now:...}`), and nothing here reads it.
  config.pop('hydra', None)
  return omegaconf.OmegaConf.create(
    omegaconf.OmegaConf.to_container(config, resolve=True))


def run_arm(model, tokenizer, device, num_samples, batch_size, real_images,
            arm):
  """FID for one conditioning arm. `arm` is 'null' or 'random'."""
  unet = getattr(model.backbone, 'backbone', model.backbone)
  num_classes = unet.num_classes
  original_forward = unet.forward

  def patched(x, sigma=None, class_cond=None, weights=None):
    if class_cond is None:
      if arm == 'random':
        class_cond = torch.randint(0, num_classes, (x.shape[0],),
                                   device=x.device)
      else:
        class_cond = torch.full((x.shape[0],), num_classes,
                                dtype=torch.long, device=x.device)
    return original_forward(x, sigma, class_cond, weights)

  unet.forward = patched
  try:
    fid = metrics.build_fid(device=device)
    # Chunked: Inception runs the whole tensor as one batch, and a few thousand
    # images at once is several GiB of activations.
    for i in range(0, real_images.shape[0], batch_size):
      fid.update(real_images[i:i + batch_size], real=True)
    torch.manual_seed(0)
    for _ in range(num_samples // batch_size):
      samples = model.generate_samples(num_samples=batch_size,
                                       num_steps=model.config.sampling.steps)
      fid.update(tokenizer.batch_decode(samples).to(device), real=False)
    value = float(fid.compute())
    is_mean, _ = fid.compute_inception_score()
    return value, float(is_mean)
  finally:
    unet.forward = original_forward


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', type=int, default=2000)
  parser.add_argument('--steps', type=int, default=128)
  parser.add_argument('--ckpt', default=DEFAULT_CKPT)
  args = parser.parse_args()

  device = torch.device('cuda')
  config = build_config(args.steps, args.ckpt)
  tokenizer = dataloader.get_tokenizer(config)
  model = algo.MDLM_DMMD(config, tokenizer=tokenizer).to(device)
  model._eval_mode()

  # One shared set of real images so both arms are scored against identical
  # reference statistics.
  real_set = dataloader.get_dataset(
    'cifar10', tokenizer, wrap=False, mode='train',
    cache_dir=config.data.cache_dir, augment=False)
  loader = torch.utils.data.DataLoader(
    real_set, batch_size=args.n, shuffle=False, num_workers=2)
  real_images = tokenizer.batch_decode(
    next(iter(loader))['input_ids']).to(device)
  print(f'real reference: {tuple(real_images.shape)}')

  print(f'\n{args.n} samples @ {args.steps} steps, identical seed:\n')
  print(f"{'arm':>8} {'FID':>10} {'IS':>8}")
  results = {}
  for arm in ('null', 'random'):
    fid, inception = run_arm(model, tokenizer, device, args.n,
                             config.loader.eval_batch_size, real_images, arm)
    results[arm] = fid
    print(f'{arm:>8} {fid:>10.2f} {inception:>8.2f}', flush=True)

  ratio = results['null'] / max(results['random'], 1e-9)
  print(f'\nnull / random = {ratio:.2f}x')
  if ratio > 1.3:
    print('=> The unconditional (null-class) arm is materially worse. The gap '
          "to the paper's Table 1 is this checkpoint's conditioning "
          '(class_dropout_p=0.1, so the null embedding saw ~10% of training), '
          'not the pixel layout / vocabulary / schedule / sampler.')
  else:
    print('=> Both arms are comparable, so the null class does NOT explain the '
          'gap. Look elsewhere (pixel layout, vocabulary, schedule, sampler).')


if __name__ == '__main__':
  main()
