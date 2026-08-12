import math
import os
import typing

import torch
import torch.nn.functional as F
import torchmetrics
import transformers

LOG2 = math.log(2)


@torch.no_grad()
def entropy_sum(dist, mask=None, log_input=False,
                max_chunk_elems=1 << 24):
  """Summed Shannon entropy (nats) of a batch of categorical distributions.

  Returns ``(total, count)``: the sum of ``-sum_v p(v) log p(v)`` over the
  counted positions as a 0-dim tensor on `dist`'s device (None when nothing
  was counted), and how many positions that was. Kept as a sum rather than a
  mean so callers can pool across calls with different position counts.

  Args:
    dist: ``(..., V)`` categorical distribution(s) from the model.
    mask: optional bool tensor broadcastable to ``dist.shape[:-1]``; only
      those positions are counted. For masked diffusion pass
      ``xt == mask_index`` -- `_process_model_output` pins the already-
      revealed positions to a one-hot, whose entropy is 0 by construction,
      so including them measures how full the sequence is rather than how
      confident the model is.
    log_input: `dist` holds log-probabilities rather than probabilities.
    max_chunk_elems: cap on the temporary allocated per chunk. The full
      ``(B, L, V)`` tensor is ~1.6 GB at B=8, L=1024, V=50257, so the
      reduction walks it in row blocks instead of materializing a
      same-shape temporary. Masked-out rows are dropped before the math,
      so a half-masked latent costs about half as much.
  """
  vocab_size = dist.shape[-1]
  flat = dist.reshape(-1, vocab_size)
  keep = None if mask is None else mask.reshape(-1)
  rows_per_chunk = max(1, int(max_chunk_elems) // max(vocab_size, 1))
  total = None
  count = 0
  for lo in range(0, flat.shape[0], rows_per_chunk):
    chunk = flat[lo: lo + rows_per_chunk]
    if keep is not None:
      keep_chunk = keep[lo: lo + rows_per_chunk]
      if not bool(keep_chunk.any()):
        continue
      chunk = chunk[keep_chunk]
    chunk = chunk.detach().to(torch.float32)
    if log_input:
      # -sum p log p read off log-probs directly. `_process_model_output`
      # writes a large finite `neg_infinity` rather than -inf, but clamp
      # anyway so a true -inf gives 0 * -inf = 0 instead of NaN.
      h = - (chunk.exp() * chunk.clamp_min(-1e30)).sum(-1)
    else:
      h = torch.special.entr(chunk).sum(-1)
    chunk_total = h.sum()
    total = chunk_total if total is None else total + chunk_total
    count += h.numel()
  return total, count


class NLL(torchmetrics.aggregation.MeanMetric):
  def update(self,
             value:typing.Union[float, torch.Tensor],
             weight:typing.Union[float, torch.Tensor]=1.0) -> None:
    """Update state with data.

    Args:
      value: Either a float or tensor containing data.
        Additional tensor dimensions will be flattened
      weight: Either a float or tensor containing weights
        for calculating the average. Shape of weight should
        be able to broadcast with the shape of `value`.
        Default to `1.0` corresponding to simple harmonic
        average.
    """
    # broadcast weight to value shape
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=self.dtype,
                                device=self.device)
    if (weight is not None
        and not isinstance(weight, torch.Tensor)):
      weight = torch.as_tensor(weight,
                               dtype=self.dtype,
                               device=self.device)
    weight = torch.broadcast_to(weight, value.shape)
    value, weight = self._cast_and_nan_check_input(value,
                                                   weight)

    if value.numel() == 0:
      return
    self.mean_value += value.sum()
    self.weight += weight.sum()


class BPD(NLL):
  def compute(self) -> torch.Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> torch.Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class GradientMoment:
  """GPT-2 Gradient Moment (GM), D-MMD paper Eq. 13/14.

  ``GM = || E_g[grad_theta log p_LLM(x)] - E_q[grad_theta log p_LLM(x)] ||^2``,
  a distribution-distance metric where ``g`` = generated samples, ``q`` = data
  and ``p_LLM`` is a frozen reference LM (GPT-2). GM = 0 iff the two sample sets
  are indistinguishable to the reference model's loss-gradient; larger means
  further from the data. Unlike generative perplexity it is not gamed by mode
  collapse (a collapsed generator scores *low* ppl but *high* GM).

  Unbiased U-statistic estimator over M paired (gen, data) batches. With
  ``d_m = grad(gen_m) - grad(data_m)`` (reference-model parameter gradients of
  the mean-per-sequence log-likelihood):

      GM ~= ( ||sum_m d_m||^2 - sum_m ||d_m||^2 ) / ( M (M-1) )   (needs M >= 2)

  because ``E[d_i . d_j] = ||E_g - E_q||^2`` for independent i != j. Only the
  running sum ``S = sum_m d_m`` (one gradient-shaped structure) and the scalar
  ``Q = sum_m ||d_m||^2`` are kept. The sign of the gradient cancels in ``d_m``.
  """

  def __init__(self, model_name_or_path='gpt2') -> None:
    self.model_name_or_path = model_name_or_path
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
      model_name_or_path)
    self.model.eval()  # disable dropout; parameters still require grad
    for p in self.model.parameters():
      p.requires_grad_(True)
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_name_or_path)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    self.device = 'cpu'
    self._S = None   # dict: param name -> running sum of d_m
    self._Q = 0.0    # scalar: sum_m ||d_m||^2
    self._M = 0      # number of accumulated batches

  def to(self, *args, **kwargs):
    self.model = self.model.to(*args, **kwargs)
    for a in args:
      if isinstance(a, (str, torch.device)):
        self.device = a
    if 'device' in kwargs and kwargs['device'] is not None:
      self.device = kwargs['device']
    return self

  def reset(self) -> None:
    self._S = None
    self._Q = 0.0
    self._M = 0

  def _tokenize(self, texts, max_length, device):
    enc = self.tokenizer(
      texts, return_tensors='pt', return_token_type_ids=False,
      return_attention_mask=True, truncation=True, padding=True,
      max_length=max_length)
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)

  def _grad(self, texts, max_length, device):
    """Reference-model parameter gradient of the mean-per-sequence
    log-likelihood on ``texts``. Returns ``{param_name: grad tensor}``."""
    input_ids, attn_mask = self._tokenize(texts, max_length, device)
    self.model.zero_grad(set_to_none=True)
    logits = self.model(input_ids, attention_mask=attn_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    tok_mask = attn_mask[:, 1:].to(logits.dtype)
    nll = F.cross_entropy(
      logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
      reduction='none').reshape(targets.shape)
    # gradient of the mean-per-token NLL (the standard LM loss; length-robust so
    # a generated batch with early EOS isn't scored differently just for being
    # shorter). Sign is irrelevant -- it cancels in d_m = grad(gen) - grad(data).
    loss = (nll * tok_mask).sum() / tok_mask.sum().clamp_min(1.0)
    loss.backward()
    return {n: p.grad.detach().clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None}

  @torch.enable_grad()
  def accumulate(self, gen_texts, data_texts, max_length, device) -> None:
    """Add one paired (generated, data) batch to the GM estimate."""
    self.to(device)
    try:
      a = self._grad(gen_texts, max_length, device)
      b = self._grad(data_texts, max_length, device)
    except RuntimeError as e:
      m = str(e).lower()
      # Under Lightning validation's default inference_mode, the loss has no
      # grad_fn so backward fails with a "does not require grad" / inference
      # message. Surface an actionable hint instead of the cryptic error.
      if ('inference' in m or 'does not require grad' in m
          or 'grad_fn' in m):
        raise RuntimeError(
          'GradientMoment needs gradients but the reference-model forward could '
          'not build an autograd graph -- Lightning validation runs under '
          'inference_mode by default. Re-run with `trainer.inference_mode=false` '
          '(required whenever eval.compute_gm=true).') from e
      raise
    q_m = 0.0
    if self._S is None:
      self._S = {}
      for n in a:
        d = a[n] - b[n]
        self._S[n] = d
        q_m += (d * d).sum().item()
    else:
      for n in a:
        d = a[n] - b[n]
        self._S[n] += d
        q_m += (d * d).sum().item()
    self._Q += q_m
    self._M += 1

  def compute(self) -> float:
    if self._M < 2:
      return 0.0
    s_norm_sq = sum((v * v).sum().item() for v in self._S.values())
    return (s_norm_sq - self._Q) / (self._M * (self._M - 1))


def build_fid(device=None):
  """torchmetrics FID + Inception Score, or `None` when unavailable.

  Both take **uint8** `(B, 3, H, W)` tensors -- exactly what
  `dataloader.RawPixelsVisionTokenizer.batch_decode` returns -- hence
  `normalize=False`. Returned as a pair so callers keep one handle.

  Requires `torch-fidelity` (torchmetrics' image backend). Missing it is not
  fatal: everything except FID still runs, and the caller logs the reason.
  """
  try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
  except Exception as e:  # pragma: no cover - environment dependent
    print(f'[metrics] FID unavailable ({e}); install torch-fidelity.',
          flush=True)
    return None
  fid = _FID(FrechetInceptionDistance(normalize=False),
             InceptionScore(normalize=False))
  return fid.to(device) if device is not None else fid


class _FID:
  """Thin holder pairing FID with Inception Score over uint8 images.

  Kept out of `MetricCollection` because both metrics carry a frozen Inception
  network; they must not be dtype-cast to float64 with the NLL collection, and
  their `compute()` is a DDP collective that has to be called on every rank.
  """

  def __init__(self, fid, inception_score):
    self.fid = fid
    self.inception_score = inception_score

  def to(self, *args, **kwargs):
    self.fid = self.fid.to(*args, **kwargs)
    self.inception_score = self.inception_score.to(*args, **kwargs)
    return self

  def update(self, images, real):
    """`images`: uint8 `(B, 3, H, W)`. `real=False` also feeds Inception Score."""
    assert images.dtype == torch.uint8, images.dtype
    assert images.ndim == 4 and images.shape[1] == 3, images.shape
    self.fid.update(images, real=real)
    if not real:
      self.inception_score.update(images)

  def compute(self):
    return self.fid.compute()

  def compute_inception_score(self):
    """(mean, std) of the Inception Score over the generated images."""
    return self.inception_score.compute()

  def reset(self):
    self.fid.reset()
    self.inception_score.reset()


class Metrics:
  def __init__(self, gen_ppl_eval_model_name_or_path=None,
               eval_ppl_batch_size=None,
               compute_gm=False,
               gm_eval_model_name_or_path='gpt2',
               modality='text',
               compute_fid=False) -> None:
    metrics = torchmetrics.MetricCollection({
        'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
    metrics.set_dtype(torch.float64)
    self.modality = modality
    self.train_nlls = metrics.clone(prefix='train/')
    self.train_aux = BPD()
    self.valid_nlls = metrics.clone(prefix='val/')
    self.valid_aux = BPD()
    self.gen_ppl = Perplexity()
    self.sample_entropy = torchmetrics.aggregation.MeanMetric()
    self.pred_entropy = torchmetrics.aggregation.MeanMetric()
    self.eval_ppl_batch_size = eval_ppl_batch_size
    # NLL / BPD / Perplexity stay valid for images: the sequence is 3072 pixel
    # tokens, so `val/bpd` *is* bits-per-dim, the standard likelihood number for
    # CIFAR-10. Only the reference-LM metrics below are text-only.
    is_image = modality == 'image'
    self.gm = (GradientMoment(gm_eval_model_name_or_path)
               if (compute_gm and not is_image) else None)
    self.fid = build_fid() if (compute_fid and is_image) else None
    self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
    if is_image:
      # No GPT-2 tokenizer download, and no text metrics: generative perplexity
      # and GM both score samples with a *language* model.
      self.tokenizer = None
    else:
      self.tokenizer = transformers.AutoTokenizer.\
        from_pretrained(gen_ppl_eval_model_name_or_path)
      if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  def to(self, *args, **kwargs):
    self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
    self.sample_entropy = self.sample_entropy.to(*args, **kwargs)
    self.pred_entropy = self.pred_entropy.to(*args, **kwargs)
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.train_aux = self.train_aux.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.valid_aux = self.valid_aux.to(*args, **kwargs)
    if self.gm is not None:
      self.gm.to(*args, **kwargs)
    if self.fid is not None:
      self.fid.to(*args, **kwargs)

  def reset(self):
    self.gen_ppl.reset()
    self.sample_entropy.reset()
    self.pred_entropy.reset()
    self.train_nlls.reset()
    self.train_aux.reset()
    self.valid_nlls.reset()
    self.valid_aux.reset()
    if self.gm is not None:
      self.gm.reset()
    if self.fid is not None:
      self.fid.reset()

  def update_train(self, nll, aux_loss, num_tokens):
    self.train_nlls.update(nll, num_tokens)
    self.train_aux.update(aux_loss, num_tokens)

  def update_valid(self, nll, aux_loss, num_tokens):
    self.valid_nlls.update(nll, num_tokens)
    self.valid_aux.update(aux_loss, num_tokens)


  @torch.no_grad()
  def _eval_retokenize(self, text_samples, max_length,
                       device):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.tokenizer(text_samples,
                             **tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(device)
      samples = samples.to(device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def record_entropy(self, tokens):
    for sample in tokens:
      _, counts = torch.unique(
        sample, return_counts=True, sorted=False)
      entropy = torch.special.entr(
        counts.float() / counts.sum()).sum().item()
      self.sample_entropy.update(entropy)

  @torch.no_grad()
  def record_pred_entropy(self, dist, mask=None, log_input=False,
                          max_chunk_elems=1 << 24):
    """Mean Shannon entropy (nats) of the *model's own* predictive
    distribution: ``-sum_v p_theta(v) log p_theta(v)``, averaged over
    positions. Successive calls accumulate a *position-weighted* mean, so
    over a sampling trajectory the early, mostly-masked steps -- where the
    model has the most positions to predict -- dominate.

    This is the sibling of `record_entropy`, which measures the unigram
    entropy of the token *counts* in a finished sample. That one can look
    healthy while every individual conditional has collapsed to a spike;
    this one reads the spike directly, so it is the number that moves when
    distillation / nucleus truncation sharpens the denoiser. Range is
    ``[0, log V]`` (``log 50257 ~= 10.82`` for GPT-2 BPE); low = confident /
    near-deterministic conditionals, high = diffuse.

    Args:
      dist: ``(..., V)`` categorical distribution from the model.
      mask: optional bool tensor broadcastable to ``dist.shape[:-1]``;
        only those positions are counted. For masked diffusion pass
        ``xt == mask_index`` -- `_process_model_output` pins the
        already-revealed positions to a one-hot, whose entropy is 0 by
        construction and would otherwise dominate the average as the
        sampler fills the sequence in.
      log_input: `dist` holds log-probabilities rather than probabilities.
      max_chunk_elems: see `entropy_sum`.
    """
    total, count = entropy_sum(
      dist, mask=mask, log_input=log_input,
      max_chunk_elems=max_chunk_elems)
    if count:
      self.pred_entropy.update(total / count, count)

  @torch.no_grad()
  def record_generative_perplexity(
    self,
    text_samples: typing.List[str],
    max_length: int,
    retokenize: bool = True,
    device='cuda') -> None:
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self._eval_retokenize(
         text_samples, max_length=max_length, device=device)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(device)
      eval_context_size = samples.shape[-1]
    batch_size = min(self.eval_ppl_batch_size,
                     samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(_samples,
                                                 _attn_mask):
        logits = eval_model(sample_chunk,
                            attention_mask=attn_mask_chunk)
        logits = logits[0].transpose(-1, -2)
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (
          sample_chunk
          == self.tokenizer.eos_token_id).cumsum(-1) == 1
        token_mask = sample_chunk != self.tokenizer.eos_token_id
        valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
        self.gen_ppl.update(nlls * valid_tokens, valid_tokens)
