"""Train an unconditional masked diffusion (MDLM) teacher on CIFAR-10 pixels.

This reproduces the "Masked Teacher" row of Table 1 in *Beyond Single Tokens:
Distilling Discrete Diffusion Models via Discrete MMD* (Hoogeboom et al., 2026,
arXiv:2603.20155), Sec. 6.1: CIFAR-10 modelled directly as 3072 tokens over the
256 pixel values, no VAE and no inductive bias about the vocabulary. FID is
measured with 50K samples against the *train* split:

    NFE   |   16     32     64    128    256    512   1024
    FID   |  122.9   47.1   20.0   11.1    7.8    6.7    6.4

Run `eval_cifar10_fid.py` on a finished checkpoint to produce that row; this
script only tracks a cheap single-NFE FID while training so the run can be
watched converge.

HYPERPARAMETERS ARE AN ASSUMPTION. The paper pins the task down exactly but
publishes no architecture or optimizer table. The recipe here is DUO's CIFAR-10
MDLM recipe -- the same 33M-parameter D3PM UNet on the same data -- with the
batch/steps rescaled from 128 x 1.5M to 512 x 375k, i.e. the same ~192M images
seen, at a batch that actually occupies 8 A100s. The learning rate is
sqrt-scaled from DUO's 2e-4 accordingly.

Deliberately standalone: it does not import the repo's Hydra/Lightning stack
(`main.py`, `algo.py`, `trainer_base.py`, `dataloader.py`), which is built for
text diffusion. It borrows exactly two things -- `models/unet.py` (the backbone)
and `models/ema.py` -- and re-derives the ~60 lines of MDLM maths from
`trainer_base.py` so the two stay comparable. Every formula below cites the
`main`-branch function it mirrors.

Checkpointing follows `~/yrDiT/train_imagenet_accelerator.py`: a single
rewritten `latest.pt` holding model+EMA+optimizer+step, archival `ema_*.pt`
snapshots, a `metadata.pt` holding the underdeep run id, and Nirvana snapshot
transfer on both ends so a preempted job resumes itself.

  # 8 GPUs, the full recipe
  torchrun --nproc_per_node=8 train_cifar10_mdlm.py --out outputs/cifar10_teacher

  # end-to-end self test, 1 GPU, ~1 minute, no underdeep
  python train_cifar10_mdlm.py --smoke
"""
import argparse
import math
import os
import time

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP

import nirvana_utils
from models.ema import ExponentialMovingAverage
from models.unet import UNet

# CIFAR-10 as tokens: 3 * 32 * 32 positions over the 256 pixel values, plus the
# absorbing [MASK] state the forward process sends them to. `MASK` is the last
# bin of the UNet's truncated-logistic head, matching DUO's masked checkpoints.
IMAGE_SIZE = 32
LENGTH = 3 * IMAGE_SIZE * IMAGE_SIZE  # 3072
MASK = 256
VOCAB = 257
LN2 = math.log(2)
NEG_INF = -1e6


# --------------------------------------------------------------------------- #
# Distributed plumbing
# --------------------------------------------------------------------------- #

def setup_distributed():
  """Return (rank, world_size, device). Works with and without torchrun."""
  if 'RANK' in os.environ:
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
  else:
    rank, world_size, local_rank = 0, 1, 0
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
  else:
    device = torch.device('cpu')
  return rank, world_size, device


def barrier():
  if dist.is_initialized():
    dist.barrier()


def is_main(rank):
  return rank == 0


def all_gather_uint8(x):
  """Gather equally-shaped uint8 tensors from every rank into one on all ranks."""
  if not dist.is_initialized():
    return x
  parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
  dist.all_gather(parts, x)
  return torch.cat(parts, dim=0)


def all_reduce_mean(value, device):
  """Mean of a python float across ranks."""
  if not dist.is_initialized():
    return value
  t = torch.tensor([value], dtype=torch.float64, device=device)
  dist.all_reduce(t, op=dist.ReduceOp.SUM)
  return (t / dist.get_world_size()).item()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class DiscreteCIFAR10(torch.utils.data.Dataset):
  """CIFAR-10 flattened to `(c h w)` int64 pixel tokens in [0, 255].

  `augment` is separated from `train` because the FID reference pass has to read
  the *un-flipped* train images while training wants the random horizontal flip.
  """

  def __init__(self, cache_dir, train, augment=None):
    import torchvision
    self._dataset = torchvision.datasets.CIFAR10(
      root=os.path.expanduser(cache_dir), train=train, download=True)
    self.augment = train if augment is None else augment

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, index):
    img, _ = self._dataset[index]
    img = np.array(img)  # (H, W, C) uint8
    if self.augment and np.random.rand() < 0.5:
      img = img[:, ::-1]
    img = rearrange(np.ascontiguousarray(img), 'h w c -> (c h w)')
    return torch.from_numpy(img).long()


def build_loader(dataset, batch_size, rank, world_size, shuffle,
                 num_workers, drop_last):
  sampler = torch.utils.data.DistributedSampler(
    dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
    drop_last=drop_last) if world_size > 1 else None
  loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=sampler,
    shuffle=(sampler is None and shuffle), num_workers=num_workers,
    pin_memory=True, drop_last=drop_last, persistent_workers=num_workers > 0)
  return loader, sampler


def infinite(loader, sampler, start_epoch=0):
  """Yield batches forever, advancing the sampler epoch so restarts don't replay."""
  epoch = start_epoch
  while True:
    if sampler is not None:
      sampler.set_epoch(epoch)
    for batch in loader:
      yield batch
    epoch += 1


# --------------------------------------------------------------------------- #
# MDLM: noise schedule, forward process, loss, sampler
# --------------------------------------------------------------------------- #

def cosine_alpha(t, eps=1e-3):
  """Cosine alpha-schedule, mirroring `trainer_base.Cosine` (DUO's).

  Returns `(dalpha_t, alpha_t)` with `alpha_t` falling 1 -> ~0 over t in [0, 1].
  NOTE the `(1 - eps)` time rescale is deliberately *not* propagated into the
  derivative, exactly as in DUO and in this repo's `Cosine.forward`; it is a
  constant 0.999 factor on the loss weight and keeping it makes the two
  implementations comparable term for term.
  """
  half_pi = math.pi / 2
  t = (1 - eps) * t
  alpha_t = 1 - torch.cos(half_pi * (1 - t))
  dalpha_t = -torch.sin(half_pi * (1 - t)) * half_pi
  return dalpha_t, alpha_t


def sample_t(batch_size, device, sampling_eps=1e-3, antithetic=True):
  """t ~ U(sampling_eps, 1), stratified over the batch.

  Mirrors `Diffusion._sample_t`, stratifying within the local batch rather than
  the global one -- the variance reduction is what matters, not which ranks the
  strata land on.
  """
  u = torch.rand(batch_size, device=device)
  if antithetic:
    offset = torch.arange(batch_size, device=device) / batch_size
    u = (u / batch_size + offset) % 1
  return (1 - sampling_eps) * u + sampling_eps


def q_xt(x0, alpha_t, generator=None):
  """Forward process: mask each token independently w.p. `1 - alpha_t`.

  Mirrors `AbsorbingState.q_xt`. `alpha_t` is `(B, 1)`.
  """
  u = torch.rand(x0.shape, device=x0.device, generator=generator)
  return torch.where(u < 1 - alpha_t, MASK, x0)


def log_p_x0(model, xt, sigma):
  """SUBS-parameterized log p_theta(x0 | x_t), as `(B, L, VOCAB)` float32.

  Mirrors `MDLM_with_AR._process_model_output`, minus its final step. That
  method also pins already-revealed positions to a one-hot. We skip it because
  both consumers make it a no-op: the loss multiplies those positions by the
  masked indicator, and the sampler carries them over with a copy flag. Zeroing
  out the MASK bin is the part that matters -- the model must never predict the
  absorbing state as clean data.

  The backbone runs in bf16 and the log-softmax in fp32: the weight `w(t)`
  reaches ~1e3 near t=0, so the normalization wants the headroom.
  """
  with torch.autocast('cuda', dtype=torch.bfloat16, enabled=xt.is_cuda):
    logits = model(xt, sigma=sigma)
  logits = logits.float()
  # Additive, out-of-place: same effect as the repo's `+= neg_infinity` without
  # an in-place write into a tensor autograd still holds.
  bias = torch.zeros(VOCAB, device=logits.device, dtype=logits.dtype)
  bias[MASK] = NEG_INF
  logits = logits + bias
  return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def mdlm_loss(model, x0, sampling_eps, time_conditioning, antithetic=True):
  """Continuous-time MDLM NELBO, in nats per token.

  Mirrors `Diffusion.nll` + `MDLM_with_AR.nll_per_token` + `TrainerBase._loss`:
  the per-position weight is `w(t) = -alpha'_t / (1 - alpha_t)` and the
  reduction is a mean over all `B * L` positions (revealed positions contribute
  zero), which is the normalization the repo's `optim.lr` was tuned against.

  Sanity check: 8.0 bpd is the uniform-over-256-bins baseline. The truncated
  logistic head is not uniform at init, so the curve starts *above* it -- ~11
  bpd measured -- and the thing to watch is that it drops below 8.0 within the
  first few dozen steps. Sitting at or above 8.0 means the model is doing no
  better than chance and the masking or the `w(t)` weight is wrong.
  """
  t = sample_t(x0.shape[0], x0.device, sampling_eps, antithetic)
  dalpha_t, alpha_t = cosine_alpha(t)
  alpha_t = alpha_t.unsqueeze(-1)  # (B, 1)

  xt = q_xt(x0, alpha_t)
  sigma = -torch.log(alpha_t.squeeze(-1)) if time_conditioning else None
  log_p = log_p_x0(model, xt, sigma)

  nll = -torch.gather(log_p, -1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
  weight = (-dalpha_t / (1 - alpha_t.squeeze(-1))).unsqueeze(-1)  # (B, 1)
  return (weight * nll * (xt == MASK)).sum(-1).mean() / LENGTH


@torch.no_grad()
def sample(model, num_samples, nfe, device, time_conditioning,
           eps=1e-5, generator=None):
  """Ancestral sampler from the all-mask prior. Returns `(B, L)` int64 tokens.

  Mirrors `Diffusion.generate_samples` + `AbsorbingState._ancestral_update`.
  `nfe` counts *denoiser calls*, which is what the paper's tables index: the
  grid contributes `nfe - 1` of them and the terminal noise-removal step (which
  fills whatever is still masked at t=eps) contributes the last one.
  """
  assert nfe >= 1
  num_steps = nfe - 1
  x = torch.full((num_samples, LENGTH), MASK, dtype=torch.long, device=device)
  timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

  def step(t_scalar, alpha_s):
    nonlocal x
    t = t_scalar * torch.ones(x.shape[0], 1, device=device)
    _, alpha_t = cosine_alpha(t)
    sigma = -torch.log(alpha_t.squeeze(-1)) if time_conditioning else None
    p = log_p_x0(model, x, sigma).exp()

    q = p * (alpha_s - alpha_t).unsqueeze(-1)
    q[..., MASK] = (1 - alpha_s).expand_as(q[..., MASK])
    # Gumbel-max categorical draw, as in `trainer_base.sample_categorical`.
    noise = torch.rand(q.shape, device=device, generator=generator)
    proposal = (q / (1e-10 - (noise + 1e-10).log())).argmax(-1)
    x = torch.where(x == MASK, proposal, x)

  for i in range(num_steps):
    _, alpha_s = cosine_alpha(timesteps[i + 1] * torch.ones(
      x.shape[0], 1, device=device))
    step(timesteps[i], alpha_s)
  # Noise removal: alpha_s = 1 leaves no mass on [MASK], so nothing stays masked.
  step(timesteps[-1], torch.ones(x.shape[0], 1, device=device))
  return x


def tokens_to_images(x):
  """`(B, 3072)` tokens -> `(B, 3, 32, 32)` uint8, the layout FID/underdeep want."""
  return rearrange(x.clamp(0, 255).to(torch.uint8), 'b (c h w) -> b c h w',
                   c=3, h=IMAGE_SIZE, w=IMAGE_SIZE)


def image_grid(images, ncol=8):
  """`(B, 3, H, W)` uint8 -> a single `(H*nrow, W*ncol, 3)` HWC uint8 array."""
  images = images[:ncol * ncol]
  nrow = math.ceil(images.shape[0] / ncol)
  pad = nrow * ncol - images.shape[0]
  if pad:
    images = torch.cat([images, torch.zeros_like(images[:pad])], dim=0)
  grid = rearrange(images.cpu(), '(r c) ch h w -> (r h) (c w) ch',
                   r=nrow, c=ncol)
  return grid.numpy()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(ch=128, dropout=0.1, time_conditioning=False):
  """The D3PM/DUO UNet from `models/unet.py`, unconditional, 257-bin head."""
  cfg = omegaconf.OmegaConf.create({
    'model': {
      'name': 'unet',
      'ch': ch,
      'num_res_blocks': 2,
      'num_scales': 4,
      'ch_mult': [1, 2, 2, 2],
      'input_channels': 3,
      'scale_count_to_put_attn': 1,  # self-attention at the 16x16 scale
      'dropout': dropout,
      'skip_rescale': True,
      'time_conditioning': time_conditioning,
      'time_scale_factor': 1000,
      'time_embed_dim': ch,
      'fix_logistic': False,
      'size': IMAGE_SIZE * IMAGE_SIZE,
      'length': LENGTH,
    },
    # No class conditioning at all: Sec. 6.1 is unconditional generation, so the
    # UNet is built without a `cond_map` rather than with a null-class row.
    'data': {'num_classes': None},
  })
  return UNet(cfg, vocab_size=VOCAB)


# --------------------------------------------------------------------------- #
# FID
# --------------------------------------------------------------------------- #

class FID:
  """torchmetrics FID whose real-side statistics are computed once and cached.

  The reference is the full un-flipped CIFAR-10 *train* split, which is what the
  paper compares against. Building it costs an Inception pass over 50K images,
  so it is cached to `<out>/fid_ref.pt` and reloaded on every restart.
  """

  def __init__(self, device, cache_path, cache_dir, num_ref=None):
    from torchmetrics.image.fid import FrechetInceptionDistance
    self.metric = FrechetInceptionDistance(normalize=False).to(device)
    self.device = device
    self._load_or_build_reference(cache_path, cache_dir, num_ref)

  def _load_or_build_reference(self, cache_path, cache_dir, num_ref):
    if cache_path is not None and os.path.exists(cache_path):
      state = torch.load(cache_path, map_location=self.device)
      self.metric.real_features_sum = state['sum'].to(self.device)
      self.metric.real_features_cov_sum = state['cov_sum'].to(self.device)
      self.metric.real_features_num_samples = state['n'].to(self.device)
      print(f'[fid] reference loaded from {cache_path} '
            f'({int(state["n"].item())} images)', flush=True)
      return

    print('[fid] building reference statistics over the CIFAR-10 train split '
          '(un-flipped); this happens once', flush=True)
    dataset = DiscreteCIFAR10(cache_dir, train=True, augment=False)
    if num_ref is not None:
      dataset = torch.utils.data.Subset(dataset, range(num_ref))
    loader = torch.utils.data.DataLoader(dataset, batch_size=500,
                                         num_workers=4)
    for batch in loader:
      self.metric.update(tokens_to_images(batch).to(self.device), real=True)
    if cache_path is not None:
      torch.save({'sum': self.metric.real_features_sum.cpu(),
                  'cov_sum': self.metric.real_features_cov_sum.cpu(),
                  'n': self.metric.real_features_num_samples.cpu()},
                 cache_path)
      print(f'[fid] reference cached to {cache_path}', flush=True)

  @staticmethod
  def _mean_cov(feat_sum, cov_sum, n):
    """(mu, Sigma) from the sums torchmetrics accumulates. Same algebra as its
    `FrechetInceptionDistance.compute`, in float64."""
    feat_sum, cov_sum = feat_sum.double(), cov_sum.double()
    n = n.double()
    mu = (feat_sum / n).unsqueeze(0)
    sigma = (cov_sum - n * mu.t().mm(mu)) / (n - 1)
    return mu.squeeze(0), sigma

  @staticmethod
  def _frechet(mu1, sigma1, mu2, sigma2):
    """||mu1-mu2||^2 + tr(S1) + tr(S2) - 2 tr((S1 S2)^1/2).

    torchmetrics evaluates the last term as `eigvals(S1 @ S2).sqrt().sum()` --
    a *general* (non-symmetric) eigendecomposition of a 2048x2048 matrix in
    complex arithmetic, which is the single slowest step of an evaluation and
    costs minutes regardless of how many samples were drawn.

    S1 S2 is similar to the symmetric PSD matrix S1^(1/2) S2 S1^(1/2), so the
    two have identical eigenvalues and the trace can be had from `eigh` /
    `eigvalsh` instead: real arithmetic, and a far smaller constant than
    `geev`. The value is the same; only the route to it changes.
    """
    w, v = torch.linalg.eigh(sigma1)
    sigma1_half = (v * w.clamp_min(0).sqrt()) @ v.t()
    m = sigma1_half @ sigma2 @ sigma1_half
    tr_covmean = torch.linalg.eigvalsh(m).clamp_min(0).sqrt().sum()
    return ((mu1 - mu2) ** 2).sum() + sigma1.trace() + sigma2.trace() \
        - 2 * tr_covmean

  def score(self, images):
    """FID of `(N, 3, 32, 32)` uint8 generated images against the reference.

    torchmetrics is used for the Inception features (the part worth reusing)
    but not for the final distance -- see `_frechet`.
    """
    self.metric.fake_features_sum.zero_()
    self.metric.fake_features_cov_sum.zero_()
    self.metric.fake_features_num_samples.zero_()
    for i in range(0, images.shape[0], 500):
      self.metric.update(images[i:i + 500].to(self.device), real=False)
    mu_real, cov_real = self._mean_cov(self.metric.real_features_sum,
                                       self.metric.real_features_cov_sum,
                                       self.metric.real_features_num_samples)
    mu_fake, cov_fake = self._mean_cov(self.metric.fake_features_sum,
                                       self.metric.fake_features_cov_sum,
                                       self.metric.fake_features_num_samples)
    return self._frechet(mu_real, cov_real, mu_fake, cov_fake).item()


@torch.no_grad()
def generate_for_fid(model, total, nfe, batch_size, device, time_conditioning,
                     rank, world_size, log_every=None):
  """Sample `total` images sharded over the ranks; every rank returns all of them."""
  per_rank = math.ceil(total / world_size)
  per_rank = math.ceil(per_rank / batch_size) * batch_size
  chunks = []
  for i in range(0, per_rank, batch_size):
    x = sample(model, batch_size, nfe, device, time_conditioning)
    chunks.append(tokens_to_images(x))
    if log_every and is_main(rank) and (i // batch_size) % log_every == 0:
      print(f'[fid] nfe={nfe} {i + batch_size}/{per_rank} per rank', flush=True)
  local = torch.cat(chunks, dim=0)
  return all_gather_uint8(local)[:total]


# --------------------------------------------------------------------------- #
# underdeep
# --------------------------------------------------------------------------- #

class Tracker:
  """Thin underdeep wrapper: resumes the previous run, never crashes training.

  Mirrors the setup in `~/yrDiT/train_imagenet_accelerator.py` and in this
  repo's `underdeep_logger.py`, with one deliberate difference: yrDiT clears
  `run_uid = None` immediately after reading it, so every restart opens a fresh
  run. Here the id read from `metadata.pt` is actually used, which is the whole
  point of persisting it.
  """

  def __init__(self, experiment, name, meta_path, enabled=True):
    self.run = None
    self.meta_path = meta_path
    if not enabled:
      return
    try:
      import underdeep as U
      client = self._client(U, experiment)
      run_uid = None
      if os.path.exists(meta_path):
        run_uid = torch.load(meta_path, weights_only=False).get('run_uid')
      if run_uid:
        self.run = client.resume_from(run_uid)
        print(f'[underdeep] resumed run {run_uid}', flush=True)
      else:
        self.run = client.init_run(name=name)
        torch.save({'run_uid': self.run.uid}, meta_path)
        print(f'[underdeep] new run {self.run.uid} in {experiment}', flush=True)
    except Exception as e:
      print(f'[underdeep] disabled ({e})', flush=True)

  @staticmethod
  def _client(U, experiment):
    """Client for `project/experiment`, creating the experiment if it is new.

    `U.Client(experiment=...)` resolves the experiment eagerly and raises "There
    is no experiment with uid/code=..." for one that does not exist yet, so
    pointing `--underdeep-experiment` at a fresh name would otherwise just
    silently disable logging. Resolve the project first, ensure the experiment
    (`exist_ok=True`, so concurrent ranks and restarts are fine), then switch.
    """
    project, sep, code = experiment.partition('/')
    if not sep:  # bare experiment code: let underdeep resolve it as before
      return U.Client(experiment=experiment)
    client = U.Client(project=project)
    client.experiments.add(code=code, project_code=project, exist_ok=True)
    client.change_experiment(code)
    return client

  def log(self, metrics, step):
    if self.run is None:
      return
    try:
      self.run.log(data=metrics, step=step)
    except Exception as e:
      print(f'[underdeep] scalar log failed: {e}', flush=True)

  def log_image(self, title, array, step):
    if self.run is None:
      return
    try:
      self.run.log_image(title=title, array=array, step=step)
    except Exception as e:
      print(f'[underdeep] image log failed: {e}', flush=True)

  def finish(self):
    if self.run is None:
      return
    try:
      self.run.finish()
    except Exception as e:
      print(f'[underdeep] finish failed: {e}', flush=True)


# --------------------------------------------------------------------------- #
# Checkpointing (yrDiT style)
# --------------------------------------------------------------------------- #

def save_checkpoint(path, model, ema, opt, step, args):
  # Re-created rather than assumed: on Nirvana the run directory is copied
  # around underneath us, and a missing parent here would throw away the run.
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save({'model': model.state_dict(),
              'ema': ema.state_dict(),
              'opt': opt.state_dict(),
              'step': step,
              'args': vars(args)}, path)
  print(f'[ckpt] saved {path} at step {step}', flush=True)


def load_checkpoint(path, model, ema, opt, device):
  """Restore model/EMA/optimizer in place. Returns the step to resume at."""
  if not os.path.isfile(path):
    return 0
  ckpt = torch.load(path, map_location=device, weights_only=False)
  model.load_state_dict(ckpt['model'])
  ema.load_state_dict(ckpt['ema'])
  ema.shadow_params = [p.to(device) for p in ema.shadow_params]
  if opt is not None and 'opt' in ckpt:
    opt.load_state_dict(ckpt['opt'])
  step = ckpt.get('step', 0)
  print(f'[ckpt] resumed {path} at step {step}', flush=True)
  return step


# --------------------------------------------------------------------------- #
# Evaluation during training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validation_bpd(model, loader, device, sampling_eps, time_conditioning,
                   max_batches=20):
  """Mean test-set NELBO in bits per dimension.

  The `t` draw and the masking are reseeded to a fixed value so successive
  validations differ only by the model, then the global RNG state is put back so
  the training stream is not perturbed by having evaluated.
  """
  cpu_state = torch.get_rng_state()
  cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  torch.manual_seed(0)
  try:
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
      if i >= max_batches:
        break
      total += mdlm_loss(model, batch.to(device), sampling_eps,
                         time_conditioning, antithetic=True).item()
      count += 1
  finally:
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
      torch.cuda.set_rng_state_all(cuda_state)
  return total / max(count, 1) / LN2


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--out', default='outputs/cifar10_teacher',
                 help='run directory: checkpoints, FID cache, underdeep handle')
  p.add_argument('--data-dir', default='~/data/cifar10')

  # Recipe (see the module docstring on where these come from).
  p.add_argument('--batch-size', type=int, default=64,
                 help='per GPU; global batch is this times the world size')
  p.add_argument('--max-steps', type=int, default=375_000)
  p.add_argument('--lr', type=float, default=4e-4)
  p.add_argument('--warmup', type=int, default=5_000)
  p.add_argument('--grad-clip', type=float, default=1.0)
  p.add_argument('--ema', type=float, default=0.9999)
  p.add_argument('--dropout', type=float, default=0.1)
  p.add_argument('--ch', type=int, default=128)
  p.add_argument('--sampling-eps', type=float, default=1e-3,
                 help='floor on the training time draw; w(t) diverges at t=0')
  p.add_argument('--time-cond', action='store_true',
                 help='condition the UNet on sigma = -log(alpha_t). Off by '
                      'default: MDLM (Sahoo et al. 2024) shows x_t already '
                      'reveals t, and DUO trained its CIFAR-10 masked '
                      'checkpoints time-unconditionally.')
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--num-workers', type=int, default=4)

  # Logging / eval cadence.
  p.add_argument('--log-every', type=int, default=100)
  p.add_argument('--sample-every', type=int, default=1_000,
                 help='log a sample grid this often. Cheap: --sample-grid '
                      'samples on one rank, no FID and no test-set pass.')
  p.add_argument('--sample-grid', type=int, default=64,
                 help='images per grid (rendered 8 per row)')
  p.add_argument('--eval-every', type=int, default=10_000,
                 help='the expensive pass: test BPD + FID')
  p.add_argument('--eval-nfe', type=int, default=128)
  p.add_argument('--eval-fid-samples', type=int, default=2_000,
                 help='samples per in-loop FID. This is a trend metric, and '
                      'sampling dominates its cost, so it is kept small; the '
                      'Table 1 number comes from eval_cifar10_fid.py at 50K. '
                      'FID is biased upward at low sample counts, so compare '
                      'these only against each other, never against the paper.')
  p.add_argument('--eval-sample-batch', type=int, default=125)
  p.add_argument('--ckpt-every', type=int, default=2_000,
                 help='rewrite latest.pt this often')
  p.add_argument('--ema-ckpt-every', type=int, default=50_000,
                 help='archive ema_<step>.pt this often (never overwritten)')
  p.add_argument('--snapshot-every', type=int, default=10_000,
                 help='push the run directory to the Nirvana snapshot path '
                      'this often. No-op off Nirvana. Keep it a multiple of '
                      '--ckpt-every so the snapshot holds a fresh latest.pt.')

  p.add_argument('--underdeep-experiment',
                 default='generative-models/cifar10_mdlm',
                 help='project/experiment; created if it does not exist yet')
  p.add_argument('--run-name', default='cifar10_mdlm_teacher')
  p.add_argument('--no-underdeep', action='store_true')

  p.add_argument('--smoke', action='store_true',
                 help='tiny end-to-end self test: 20 steps, small UNet, a few '
                      'samples, a checkpoint round-trip. ~1 minute on 1 GPU.')
  args = p.parse_args()

  if args.smoke:
    args.ch = 32
    args.batch_size = 8
    args.max_steps = 20
    args.warmup = 5
    args.log_every = 5
    args.sample_every = 10
    args.sample_grid = 16
    args.eval_every = 10
    args.eval_nfe = 8
    args.eval_fid_samples = 64
    args.eval_sample_batch = 32
    args.ckpt_every = 10
    args.ema_ckpt_every = 20
    args.snapshot_every = 20
    args.num_workers = 2
    args.no_underdeep = True
    args.out = args.out if args.out != 'outputs/cifar10_teacher' else 'outputs/smoke'
  return args


def main():
  args = parse_args()
  rank, world_size, device = setup_distributed()
  torch.manual_seed(args.seed + rank)
  np.random.seed(args.seed + rank)

  out_dir = os.path.abspath(args.out)
  ckpt_dir = os.path.join(out_dir, 'checkpoints')
  # Pull the preempted attempt's state back before anything looks for it.
  if is_main(rank):
    os.makedirs(ckpt_dir, exist_ok=True)
    nirvana_utils.copy_snapshot_to_out(out_dir)
  barrier()
  os.makedirs(ckpt_dir, exist_ok=True)

  # -- model / optimizer / EMA ---------------------------------------------- #
  model = build_model(args.ch, args.dropout, args.time_cond).to(device)
  num_params = sum(p.numel() for p in model.parameters())
  ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
  ema.move_shadow_params_to_device(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                          betas=(0.9, 0.999), eps=1e-8)

  start_step = load_checkpoint(os.path.join(ckpt_dir, 'latest.pt'),
                               model, ema, opt, device)
  ddp_model = DDP(model, device_ids=[device.index]) if world_size > 1 else model

  # -- data ------------------------------------------------------------------ #
  # Rank 0 fetches CIFAR-10 first and everyone else waits, so the 170 MB
  # archive is downloaded once rather than once per rank. Without this the
  # ranks race: they share the bandwidth (so each crawls), and they extract the
  # same tar into the same directory concurrently, which can corrupt it.
  # Constructing the dataset is what triggers torchvision's download, and doing
  # it twice on rank 0 just re-reads an already-present pickle.
  if is_main(rank) and world_size > 1:
    DiscreteCIFAR10(args.data_dir, train=True)
    DiscreteCIFAR10(args.data_dir, train=False)
  barrier()
  train_set = DiscreteCIFAR10(args.data_dir, train=True)
  test_set = DiscreteCIFAR10(args.data_dir, train=False)
  train_loader, train_sampler = build_loader(
    train_set, args.batch_size, rank, world_size, shuffle=True,
    num_workers=args.num_workers, drop_last=True)
  test_loader, _ = build_loader(
    test_set, args.batch_size, rank, world_size, shuffle=False,
    num_workers=0, drop_last=False)
  steps_per_epoch = max(len(train_loader), 1)
  batches = infinite(train_loader, train_sampler,
                     start_epoch=start_step // steps_per_epoch)

  # -- tracking / FID -------------------------------------------------------- #
  tracker = Tracker(args.underdeep_experiment, args.run_name,
                    os.path.join(ckpt_dir, 'metadata.pt'),
                    enabled=is_main(rank) and not args.no_underdeep)
  fid = FID(device, os.path.join(out_dir, 'fid_ref.pt'), args.data_dir,
            num_ref=512 if args.smoke else None) if is_main(rank) else None

  if is_main(rank):
    global_batch = args.batch_size * world_size
    print(f'[run] out={out_dir}\n'
          f'[run] params={num_params / 1e6:.1f}M  world_size={world_size}  '
          f'global_batch={global_batch}  lr={args.lr}  '
          f'steps={start_step}->{args.max_steps}\n'
          f'[run] time_conditioning={args.time_cond}  ema={args.ema}',
          flush=True)
    tracker.log({'params_M': num_params / 1e6,
                 'global_batch': global_batch, 'lr': args.lr}, step=start_step)

  # -- train loop ------------------------------------------------------------ #
  model.train()
  step = start_step
  running, running_n = 0.0, 0
  t_last = time.time()

  while step < args.max_steps:
    x0 = next(batches).to(device, non_blocking=True)

    lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
    for group in opt.param_groups:
      group['lr'] = lr

    loss = mdlm_loss(ddp_model, x0, args.sampling_eps, args.time_cond)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
    opt.step()
    ema.update(model.parameters())
    step += 1

    running += loss.item()
    running_n += 1

    if step % args.log_every == 0:
      mean_loss = all_reduce_mean(running / max(running_n, 1), device)
      running, running_n = 0.0, 0
      if is_main(rank):
        elapsed = time.time() - t_last
        sps = args.log_every / max(elapsed, 1e-9)
        metrics = {'train/nll_per_token': mean_loss,
                   'train/bpd': mean_loss / LN2,
                   'train/lr': lr,
                   'train/grad_norm': grad_norm.item(),
                   'perf/steps_per_sec': sps}
        tracker.log(metrics, step=step)
        print(f'step {step:>7d}  bpd {mean_loss / LN2:.4f}  '
              f'lr {lr:.2e}  gnorm {grad_norm.item():.2f}  '
              f'{sps:.2f} it/s', flush=True)
      t_last = time.time()

    # -- sample grid: cheap and frequent, so the images track the loss curve -- #
    # Only `--sample-grid` samples on rank 0, no FID, no test-set pass. Every
    # rank swaps to the EMA weights and back even though only rank 0 draws, so
    # no rank ever runs a training forward on parameters the others do not have.
    if step % args.sample_every == 0 or step == args.max_steps:
      ema.store(model.parameters())
      ema.copy_to(model.parameters())
      model.eval()
      if is_main(rank):
        x = sample(model, args.sample_grid, args.eval_nfe, device,
                   args.time_cond)
        tracker.log_image(f'samples/nfe{args.eval_nfe}',
                          image_grid(tokens_to_images(x)), step=step)
      ema.restore(model.parameters())
      model.train()
      barrier()
      t_last = time.time()

    # -- periodic evaluation: test BPD and FID -- all on EMA weights --------- #
    if step % args.eval_every == 0 or step == args.max_steps:
      ema.store(model.parameters())
      ema.copy_to(model.parameters())
      model.eval()

      bpd = all_reduce_mean(
        validation_bpd(model, test_loader, device, args.sampling_eps,
                       args.time_cond), device)
      images = generate_for_fid(model, args.eval_fid_samples, args.eval_nfe,
                                args.eval_sample_batch, device, args.time_cond,
                                rank, world_size)
      if is_main(rank):
        score = fid.score(images)
        tracker.log({'val/bpd': bpd, f'fid/nfe{args.eval_nfe}': score},
                    step=step)
        print(f'[eval] step {step}  val/bpd {bpd:.4f}  '
              f'fid@{args.eval_nfe} {score:.2f} '
              f'({args.eval_fid_samples} samples)', flush=True)

      ema.restore(model.parameters())
      model.train()
      t_last = time.time()  # don't bill the eval to the training throughput

    # -- checkpointing ------------------------------------------------------- #
    # Only rank 0 writes, and only then does everyone wait -- a barrier on every
    # step would serialize training behind the slowest rank for no reason.
    save_now = step % args.ckpt_every == 0 or step == args.max_steps
    archive_now = step % args.ema_ckpt_every == 0
    # Pushing to the Nirvana snapshot path archives the whole run directory, so
    # it is deliberately rarer than writing latest.pt: `--ckpt-every` bounds
    # what a crash costs, `--snapshot-every` bounds what a *preemption* costs.
    snapshot_now = step % args.snapshot_every == 0 or step == args.max_steps
    if save_now or archive_now or snapshot_now:
      if is_main(rank):
        if save_now:
          save_checkpoint(os.path.join(ckpt_dir, 'latest.pt'), model, ema, opt,
                          step, args)
        if archive_now:
          torch.save(ema.state_dict(),
                     os.path.join(ckpt_dir, f'ema_{step:07d}.pt'))
        if snapshot_now:
          nirvana_utils.copy_out_to_snapshot(out_dir)
      barrier()

  if is_main(rank):
    print(f'[run] done at step {step}', flush=True)
    tracker.finish()
  if dist.is_initialized():
    dist.destroy_process_group()


if __name__ == '__main__':
  main()
