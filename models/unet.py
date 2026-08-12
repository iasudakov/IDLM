"""D3PM-style UNet over 32x32x3 pixels, ported from the DUO repo
(`~/duo/models/unet.py`, itself from https://github.com/yang-song/score_sde_pytorch
and https://github.com/hojonathanho/diffusion).

Every module name is kept **byte-identical** to DUO's so the released CIFAR-10
checkpoints (`jdeschena/duo2-cifar10`, family `mdlm/...`) load with
`strict=True`. Do not rename `input_conv`, `temb_modules`,
`downsampling_modules`, `middle_modules`, `upsampling_modules`,
`output_modules` or `cond_map`.

Deltas from DUO's file, all of them forced by this repo rather than by taste:

* `_truncated_logistic_output` builds `bin_centers` on the tensor's own device
  instead of a hardcoded `'cuda'`, so the CPU unit tests can run the backbone.
* `_logistic_output_res` is out-of-place. `main.py` enables
  `torch.autograd.set_detect_anomaly(True)`, which trips on DUO's in-place slice
  assignment into a tensor that autograd still needs.
* `forward` defaults `class_cond` to the **null class** (`num_classes`) whenever
  a `cond_map` exists and nothing is passed. The released checkpoints were
  trained class-conditionally with classifier-free dropout, so their null-class
  embedding is a *learned, generally non-zero* vector -- passing `None` (DUO's
  behaviour, which skips the conditioning term entirely) is a different function
  and is not what unconditional sampling means for these weights.
* `reference_std` exposes the scale of the residual stream that
  `_ImageNoiseConditionedBackbone` injects into, the image analogue of the token
  embedding's per-entry std (see `algo._embedding_std`).

The output head is a truncated logistic over `vocab_size` bins (Austin et al.,
arXiv:2107.03006). The bin grid is derived from `vocab_size` at runtime, which
is why one architecture serves both the 256-bin uniform models and the 257-bin
masked ones whose last bin *is* `[MASK]`.
"""
import math

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dit import LabelEmbedder


# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


# Code modified from https://github.com/yang-song/score_sde_pytorch
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


class NiN(nn.Module):
  def __init__(self, in_ch, out_ch, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_ch, out_ch)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(out_ch), requires_grad=True)

  def forward(self, x,  # ["batch", "in_ch", "H", "W"]
    ):
    x = x.permute(0, 2, 3, 1)
    # x (batch, H, W, in_ch)
    y = torch.einsum('bhwi,ik->bhwk', x, self.W) + self.b
    # y (batch, H, W, out_ch)
    return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
  """Channel-wise self-attention block."""
  def __init__(self, channels, skip_rescale=True):
    super().__init__()
    self.skip_rescale = skip_rescale
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels//4, 32),
        num_channels=channels, eps=1e-6)
    self.NIN_0 = NiN(channels, channels)
    self.NIN_1 = NiN(channels, channels)
    self.NIN_2 = NiN(channels, channels)
    self.NIN_3 = NiN(channels, channels, init_scale=0.)

  def forward(self, x,  # ["batch", "channels", "H", "W"]
    ):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)

    if self.skip_rescale:
      return (x + h) / np.sqrt(2.)
    else:
      return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.1, skip_rescale=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.skip_rescale = skip_rescale

        self.act = nn.functional.silu
        self.groupnorm0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32),
            num_channels=in_ch, eps=1e-6
        )
        self.conv0 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1
        )

        if temb_dim is not None:
            self.dense0 = nn.Linear(temb_dim, out_ch)
            nn.init.zeros_(self.dense0.bias)

        self.groupnorm1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32),
            num_channels=out_ch, eps=1e-6
        )
        self.dropout0 = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1
        )
        if out_ch != in_ch:
            self.nin = NiN(in_ch, out_ch)

    def forward(self, x,  # ["batch", "in_ch", "H", "W"]
                temb=None,  # ["batch", "temb_dim"]
        ):
        assert x.shape[1] == self.in_ch

        h = self.groupnorm0(x)
        h = self.act(h)
        h = self.conv0(h)

        if temb is not None:
            # Out-of-place: `h` is the output of a conv autograd still needs, and
            # `main.py` runs with anomaly detection on.
            h = h + self.dense0(self.act(temb))[:, :, None, None]

        h = self.groupnorm1(h)
        h = self.act(h)
        h = self.dropout0(h)
        h = self.conv1(h)
        if h.shape[1] != self.in_ch:
            x = self.nin(x)

        assert x.shape == h.shape

        if self.skip_rescale:
            return (x + h) / np.sqrt(2.)
        else:
            return x + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
            stride=2, padding=0)

    def forward(self, x,  # ["batch", "ch", "inH", "inW"]
        ):
        B, C, H, W = x.shape
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x = self.conv(x)

        assert x.shape == (B, C, H // 2, W // 2)
        return x


class Upsample(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

  def forward(self, x,  # ["batch", "ch", "inH", "inW"]
    ):
    B, C, H, W = x.shape
    h = F.interpolate(x, (H*2, W*2), mode='nearest')
    h = self.conv(h)

    assert h.shape == (B, C, H*2, W*2)
    return h


class UNet(nn.Module):
    def __init__(self, config, vocab_size=None):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        assert config.model.name == 'unet'
        self.ch = config.model.ch
        self.num_res_blocks = config.model.num_res_blocks
        self.num_scales = config.model.num_scales
        self.ch_mult = config.model.ch_mult
        assert self.num_scales == len(self.ch_mult)
        self.input_channels = config.model.input_channels
        self.output_channels = 2 * config.model.input_channels
        self.scale_count_to_put_attn = config.model.scale_count_to_put_attn
        # Tuple of min and max value of the input so it can be rescaled to
        # [-1, 1]. Derived from `vocab_size` (not `config.model.data_min_max`)
        # so the masked 257-bin head and the uniform 256-bin head agree with
        # their own bin grids; see `_truncated_logistic_output`.
        self.data_min_max = [0, vocab_size]
        self.dropout = config.model.dropout
        self.skip_rescale = config.model.skip_rescale
        self.time_conditioning = config.model.time_conditioning  # Whether to add in time embeddings
        self.time_scale_factor = config.model.time_scale_factor  # scale to make the range of times be 0 to 1000
        self.time_embed_dim = config.model.time_embed_dim
        self.vocab_size = vocab_size

        self.size = config.model.size
        self.length = config.model.length

        # truncated logistic
        self.fix_logistic = config.model.fix_logistic

        self.act = nn.functional.silu

        if self.time_conditioning:
            self.temb_modules = []
            self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules = nn.ModuleList(self.temb_modules)

        self.expanded_time_dim = 4 * self.time_embed_dim if self.time_conditioning else None

        self.input_conv = nn.Conv2d(
            in_channels=self.input_channels, out_channels=self.ch,
            kernel_size=3, padding=1
        )

        h_cs = [self.ch]
        in_ch = self.ch

        # Downsampling
        self.downsampling_modules = []

        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.downsampling_modules.append(
                    ResBlock(in_ch, out_ch, temb_dim=self.expanded_time_dim,
                        dropout=self.dropout, skip_rescale=self.skip_rescale)
                )
                in_ch = out_ch
                h_cs.append(in_ch)
                if scale_count == self.scale_count_to_put_attn:
                    self.downsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )

            if scale_count != self.num_scales - 1:
                self.downsampling_modules.append(Downsample(in_ch))
                h_cs.append(in_ch)

        self.downsampling_modules = nn.ModuleList(self.downsampling_modules)

        # Middle
        self.middle_modules = []

        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                dropout=self.dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            AttnBlock(in_ch, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                dropout=self.dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules = nn.ModuleList(self.middle_modules)

        # Upsampling
        self.upsampling_modules = []

        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks+1):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.upsampling_modules.append(
                    ResBlock(in_ch + h_cs.pop(),
                        out_ch,
                        temb_dim=self.expanded_time_dim,
                        dropout=self.dropout,
                        skip_rescale=self.skip_rescale
                    )
                )
                in_ch = out_ch

                if scale_count == self.scale_count_to_put_attn:
                    self.upsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )
            if scale_count != 0:
                self.upsampling_modules.append(Upsample(in_ch))

        self.upsampling_modules = nn.ModuleList(self.upsampling_modules)

        assert len(h_cs) == 0

        # output
        self.output_modules = []

        self.output_modules.append(
            nn.GroupNorm(min(in_ch//4, 32), in_ch, eps=1e-6)
        )

        self.output_modules.append(
           nn.Conv2d(in_ch, self.output_channels, kernel_size=3, padding=1)
        )
        self.output_modules = nn.ModuleList(self.output_modules)

        self.num_classes = config.data.get('num_classes', None)
        if self.num_classes is not None:
            # `LabelEmbedder` allocates `num_classes + 1` rows; the extra one is
            # the classifier-free-guidance null class, and index `num_classes`
            # *is* "unconditional" for the released checkpoints.
            self.cond_map = LabelEmbedder(
                self.num_classes,
                self.time_embed_dim*4)
        else:
            self.cond_map = None

    @property
    def reference_std(self):
        """Std of the residual stream `_ImageNoiseConditionedBackbone` adds to.

        The image analogue of the token embedding's per-entry std (see
        `algo._embedding_std`): the scale the injected noise competes with, so
        that `noise/snr` keeps reading as `rms(W)`. Measured from the *current*
        `input_conv` weights on a fixed synthetic batch drawn uniformly from
        `[-1, 1]` -- the range `_center_data` maps real pixels into -- so it
        tracks the weights the way the text version tracks the embedding, and
        is deterministic across ranks and restarts.
        """
        weight = self.input_conv.weight
        generator = torch.Generator(device='cpu')
        generator.manual_seed(0)
        probe = (torch.rand(8, self.input_channels, 8, 8,
                            generator=generator, dtype=torch.float32) * 2. - 1.)
        probe = probe.to(device=weight.device)
        with torch.no_grad():
            out = F.conv2d(probe, weight.float(),
                           self.input_conv.bias.float(), padding=1)
        return float(out.std())

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0])  # [0, 1]
        return 2 * out - 1  # to put it in [-1, 1]

    def _time_embedding(self, timesteps):
        if self.time_conditioning:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None

        return temb

    def _do_input_conv(self, h):
        h = self.input_conv(h)
        hs = [h]
        return h, hs

    def _do_downsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                h = self.downsampling_modules[m_idx](h, temb)
                m_idx += 1
                if scale_count == self.scale_count_to_put_attn:
                    h = self.downsampling_modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if scale_count != self.num_scales - 1:
                h = self.downsampling_modules[m_idx](h)
                hs.append(h)
                m_idx += 1

        assert m_idx == len(self.downsampling_modules)

        return h, hs

    def _do_middle(self, h, temb):
        m_idx = 0
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1
        h = self.middle_modules[m_idx](h)
        m_idx += 1
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1

        assert m_idx == len(self.middle_modules)

        return h

    def _do_upsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks+1):
                h = self.upsampling_modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

                if scale_count == self.scale_count_to_put_attn:
                    h = self.upsampling_modules[m_idx](h)
                    m_idx += 1

            if scale_count != 0:
                h = self.upsampling_modules[m_idx](h)
                m_idx += 1

        assert len(hs) == 0
        assert m_idx == len(self.upsampling_modules)

        return h

    def _do_output(self, h):

        h = self.output_modules[0](h)
        h = self.act(h)
        h = self.output_modules[1](h)

        return h

    def _logistic_output_res(self,
        h,  # ["B", "twoC", "H", "W"]
        centered_x_in,  # ["B", "C", "H", "W"]
    ):
        # DUO writes `h[:, 0:C] = tanh(...)` in place. `main.py` enables
        # `torch.autograd.set_detect_anomaly(True)`, which flags that as a
        # version-counter bump on a tensor the backward pass of `output_modules`
        # still needs. Concatenating instead is numerically identical.
        B, twoC, H, W = h.shape
        C = twoC//2
        mu = torch.tanh(centered_x_in + h[:, 0:C, :, :])
        return torch.cat([mu, h[:, C:, :, :]], dim=1)

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
            Compute log (exp(a) - exp(b)) for (b<a)
            From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b-a) + eps)

    def _truncated_logistic_output(self, net_out):
        B, D = net_out.shape[0], self.length
        C = 3
        S = self.vocab_size

        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf
        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        inv_scale = torch.exp(- (log_scale - 2))

        bin_width = 2. / S
        # DUO hardcodes `device='cuda'` here; take the device from the tensor so
        # the backbone also runs on CPU (unit tests) and on non-default GPUs.
        bin_centers = torch.linspace(start=-1. + bin_width/2,
            end=1. - bin_width/2,
            steps=S,
            device=net_out.device).view(1, 1, 1, 1, S)

        sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(-sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf)
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B, D, S)

        return logits

    def forward(self,
        x,  # ["B", "D"] flattened (c h w) pixel tokens
        sigma=None,
        class_cond=None,
        weights=None
    ):
        assert weights is None
        img_size = int(np.sqrt(self.size))

        h = rearrange(x, "b (c h w) -> b c h w", h=img_size,
                      w=img_size, c=3)
        h = self._center_data(h)
        centered_x_in = h

        temb = self._time_embedding(sigma)
        if class_cond is None and self.cond_map is not None:
            # Unconditional generation with a classifier-free-dropout
            # checkpoint = the *null* class, whose embedding is learned and
            # generally non-zero. Skipping `cond_map` entirely (DUO's behaviour
            # for `class_cond=None`) would evaluate a different function than
            # the one the checkpoint was trained to be unconditional under.
            class_cond = torch.full((x.shape[0],), self.num_classes,
                                    dtype=torch.long, device=x.device)
        if class_cond is not None:
            if self.cond_map is None:
                raise ValueError("Conditioning variable provided, "
                                "but Model was not initialized "
                                "with condition embedding layer.")
            else:
                assert class_cond.shape == (x.shape[0],)
                cond = self.cond_map(class_cond)
                # `temb` is None when the backbone was built without time
                # embeddings (`model.time_conditioning=False`); the class
                # embedding is then the whole conditioning vector.
                temb = cond if temb is None else temb + cond

        h, hs = self._do_input_conv(h)

        h, hs = self._do_downsampling(h, hs, temb)

        h = self._do_middle(h, temb)

        h = self._do_upsampling(h, hs, temb)

        h = self._do_output(h)

        # h (B, 2*C, H, W)
        h = self._logistic_output_res(h, centered_x_in)
        h = self._truncated_logistic_output(h)  # (B, D, S)

        return h
