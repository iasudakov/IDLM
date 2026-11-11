import gc
import itertools
import json
import os
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

import confidence_sampling
from eval_precision import capture_rng_state, fp32_prediction_normalization, replay_rng_state
import dataloader
import metrics
import models
from model_assets import pretrained_kwargs
import utils
from ipdb import set_trace as debug


# `pareto/gsm8k_step_*` plots accuracy against realized NFE, but a tracker
# indexes a series by an integer step. Mean NFE is a float -- an adaptive
# sampler's whole point is that it lands between whole budgets -- so it is
# carried at this fixed scale: an x of 567 reads as 5.67 NFE. Two decimals
# separates thresholds whose budgets sit hundredths apart, which the dense
# grids at the cheap end of `eval.gsm8k_pareto_thresholds` do.
_NFE_X_SCALE = 100


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  prior_loss: torch.FloatTensor
  num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - t 
    dalpha_t = - (1 - self.eps)
    return dalpha_t, alpha_t


def sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer,
    vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    if hasattr(self.config.algo, 'ignore_bos'):
      self.ignore_bos = config.algo.ignore_bos
    else:
      self.ignore_bos = False
    if hasattr(self.config.algo, 'loss_type'):
      self.loss_type = config.algo.loss_type
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = self.config.sampling.predictor
    # Which still-masked positions each reverse step commits. 'random' is
    # upstream's independent Bernoulli draw; the low_confidence_* rules rank by
    # the model's confidence instead (see confidence_sampling). Read with .get so
    # a checkpoint whose saved hparams predate these keys still loads.
    self.remasking_strategy = confidence_sampling.validate_strategy(
      self.config.sampling.get('remasking_strategy', 'random'))
    self.dynamic_threshold = float(
      self.config.sampling.get('dynamic_threshold', 0.9))
    self.sampling_temperature = float(
      self.config.sampling.get('temperature', 1.0))
    # Checked here rather than in `_validate_configuration`: MDLM overrides that
    # without calling super (algo.MDLM), so a check placed there would never run
    # for the algo these strategies are aimed at.
    if self.remasking_strategy != 'random':
      # The commit set is chosen inside `_ancestral_update`; the analytic and
      # denoiser updates resample the whole sequence and have no such decision.
      assert self.sampler in {'ancestral', 'ancestral_cache'}, (
        f'sampling.remasking_strategy={self.remasking_strategy!r} needs '
        f"predictor 'ancestral' or 'ancestral_cache', got {self.sampler!r}.")
    if self.sampling_temperature < 0:
      raise ValueError('sampling.temperature must be >= 0, got '
                       f'{self.sampling_temperature}')
    if self.sampling_temperature != 1.0:
      # The token is chosen inside `_ancestral_update`. The analytic and
      # denoiser updates resample from a score, never from `p_x0`, so a
      # temperature set against them would be silently discarded -- and a
      # "greedy" number that was in fact sampled at temperature 1 is worse than
      # an error.
      assert self.sampler in {'ancestral', 'ancestral_cache'}, (
        f'sampling.temperature={self.sampling_temperature} needs predictor '
        f"'ancestral' or 'ancestral_cache', got {self.sampler!r}.")
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        **pretrained_kwargs(
          config.eval.checkpoint_path,
          getattr(config, 'model_weights_dir', None)),
        trust_remote_code=True)

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    # Noise Schedule
    self.noise = LogLinear()

    self.metrics = metrics.Metrics(
      gen_ppl_eval_model_name_or_path=\
        self.config.eval.gen_ppl_eval_model_name_or_path,
      eval_ppl_batch_size=\
        self.config.eval.perplexity_batch_size,
      model_weights_dir=getattr(config, 'model_weights_dir', None))

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _validate_configuration(self):
    assert self.config.algo.backbone in {'dit', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def q_xt(self, x, alpha_t):
    raise NotImplementedError

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(),
                           self.noise.parameters())

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed']
    # is 1 iteration behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps,
    # not the number of local steps, so we don't multiply with
    # self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    raise NotImplementedError

  def forward(self, xt, sigma):
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output = self.backbone(xt, sigma)
    return self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)

  def on_train_epoch_start(self):
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    current_accumulation_step = (
      batch_idx % self.trainer.accumulate_grad_batches)
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'],
                        current_accumulation_step,
                        train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_train_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)

  @property
  def _conditional_eval(self):
    """Whether validation-time generation is prompt-conditioned.

    On a Q -> A corpus an unconditional sample answers no question, so
    metrics computed from one say nothing about the task. Auto-detected
    from the dataset; `eval.conditional_samples` overrides.
    """
    override = self.config.eval.get('conditional_samples', None)
    if override is not None:
      return bool(override)
    return getattr(self.config.data, 'train', None) in (
      'tiny_gsm', 'gsm8k_test')

  def _prompt_lengths(self, batch):
    """Per-row prompt length, i.e. where the model's answer starts."""
    if 'prompt_len' in batch:
      lengths = batch['prompt_len'].to(torch.long)
    else:
      lengths = batch['attention_mask'].to(torch.long).argmax(dim=-1)
    assert int(lengths.min()) > 0, (
      'conditional validation needs a prompt span, but a row reports '
      'length 0 -- with data.train_on_prompt=True the loss mask cannot '
      'locate the boundary. Set eval.conditional_samples=False.')
    return lengths.to(self.device)

  @torch.no_grad()
  def _conditional_val_batch(self, batch, num_steps=None):
    """Answer one buffered validation batch's questions."""
    input_ids = batch['input_ids'].to(self.device)
    lengths = self._prompt_lengths(batch)
    width = int(lengths.max())
    samples = self.generate_samples(
      num_samples=input_ids.shape[0], num_steps=num_steps,
      prefix_tokens=input_ids[:, :width], prefix_lengths=lengths)
    answer_tokens, answer_texts, gold_texts, question_texts = [], [], [], []
    for row in range(samples.shape[0]):
      start = int(lengths[row])
      generated = samples[row, start:]
      answer_tokens.append(generated)
      answer_texts.append(self.tokenizer.decode(
        generated.tolist(), skip_special_tokens=True))
      gold_texts.append(self.tokenizer.decode(
        input_ids[row, start:].tolist(), skip_special_tokens=True))
      question_texts.append(self.tokenizer.decode(
        input_ids[row, :start].tolist(), skip_special_tokens=True))
    return (samples, answer_tokens, answer_texts, gold_texts,
            question_texts)

  def _log_conditional_accuracy(self, cond_rows):
    """val/answer_exact_match always; val/answer_acc behind eval.tinygsm_exec.

    The accuracy path executes the generated *and* the reference program in
    `sandbox_gsm8k` and compares the numbers, so an alternative correct
    solution counts. Rank zero only: the sandbox's timeout uses
    `signal.setitimer`, which is process-wide.
    """
    exact = sum(g.strip() == gold.strip() for _, g, gold in cond_rows)
    self.log('val/answer_exact_match',
             torch.tensor(exact / len(cond_rows), device=self.device),
             on_epoch=True, on_step=False, sync_dist=True)
    if not self.config.eval.get('tinygsm_exec', False):
      return
    if self.trainer.global_rank != 0:
      return
    import sandbox_gsm8k
    timeout = float(self.config.eval.get('tinygsm_exec_timeout', 5.0))
    correct = gradable = 0
    for _, generated, gold in cond_rows:
      ok, can = sandbox_gsm8k.evaluate_tinygsm_sample(
        generated, gold, timeout_s=timeout)
      correct += int(ok)
      gradable += int(can)
    self.log('val/answer_gradable',
             torch.tensor(gradable / len(cond_rows), device=self.device),
             on_epoch=True, on_step=False, sync_dist=False,
             rank_zero_only=True)
    if gradable:
      self.log('val/answer_acc',
               torch.tensor(correct / gradable, device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)

  def _gsm8k_benchmark_dataset(self):
    """GSM8K test set, tokenized to match this run's prompt format."""
    if getattr(self, '_gsm8k_cached', None) is None:
      import omegaconf
      cfg = omegaconf.OmegaConf.create(
        omegaconf.OmegaConf.to_container(self.config, resolve=True))
      cfg.data.valid = 'gsm8k_test'
      cfg.data.data_path = self.config.eval.gsm8k_data_path
      weights_dir = getattr(self.config, 'model_weights_dir', None)
      if weights_dir and not os.path.isfile(cfg.data.data_path):
        bundled_path = os.path.join(weights_dir, 'data', 'gsm8k_test.json')
        if os.path.isfile(bundled_path):
          cfg.data.data_path = bundled_path
      cfg.data.separator = self.config.data.separator
      self._gsm8k_cached = dataloader.get_gsm8k_test_dataset(
        cfg, self.tokenizer)
    return self._gsm8k_cached

  @torch.no_grad()
  def _log_gsm8k_benchmark(self):
    """The real GSM8K benchmark mid-training, as val/gsm8k_acc.

    Same 1319 questions, same prefix-pinned generation and the same
    vendored S-FLM grader as a standalone benchmark run, so a training
    curve and a published number mean the same thing. Sharded round-robin
    across ranks, gathered to rank zero for grading.
    """
    config = self.config.eval
    if not config.get('gsm8k_at_validation', False):
      return
    if self.trainer.sanity_checking:
      return
    every = int(config.get('gsm8k_every_n_validations', 1) or 1)
    self._gsm8k_val_count = getattr(self, '_gsm8k_val_count', 0) + 1
    if (self._gsm8k_val_count - 1) % every:
      return

    dataset = self._gsm8k_benchmark_dataset()
    limit = config.get('gsm8k_num_examples', None)
    total = len(dataset) if not limit else min(int(limit), len(dataset))
    steps = config.get('gsm8k_steps', None) or self.config.sampling.steps
    rank = self.trainer.global_rank
    world = max(1, self.trainer.world_size)
    indices = list(range(rank, total, world))
    batch_size = int(config.get('gsm8k_batch_size', None)
                     or self.config.loader.eval_batch_size)

    rows, nfes, per_seq = self._gsm8k_generate(
      dataset, indices, batch_size, steps,
      config.get('gsm8k_step_convention', 'sflm'))

    # Every rank has to reach the Pareto sweep: it runs its own shard and its
    # own collectives. Only the grading and logging below are rank-zero.
    if rank == 0:
      correct, accuracy = self._gsm8k_grade(rows)
      # Report the forwards actually spent, not `steps`: under an adaptive
      # remasking strategy the threshold is the budget and `steps` is ignored,
      # so it is the NFE that makes two runs comparable.
      avg_nfe = sum(nfes) / max(len(nfes), 1)
      print(f'[gsm8k] step {self.global_step}: {accuracy * 100:.2f}% '
            f'({correct}/{len(rows)}) at {avg_nfe:.1f} NFE '
            f'({sum(per_seq) / max(len(per_seq), 1):.1f} per sequence) '
            f'[{self.remasking_strategy}]', flush=True)
      self.log('val/gsm8k_acc',
               torch.tensor(accuracy, device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)
      self.log('val/gsm8k_nfe',
               torch.tensor(avg_nfe, device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)

    self._log_gsm8k_greedy(dataset, indices, batch_size, steps,
                           config.get('gsm8k_step_convention', 'sflm'))
    self._log_gsm8k_pareto(dataset, indices, batch_size)

  @torch.no_grad()
  def _log_gsm8k_greedy(self, dataset, indices, batch_size, steps,
                        step_convention):
    """The same benchmark again, with the token choice made greedily.

    Identical unmasking schedule and step budget as `val/gsm8k_acc`; the only
    difference is that each newly-unmasked position takes the argmax of `p_x0`
    instead of a sample from it. Worth its own curve because the two can rank
    checkpoints differently -- greedy tends to start far higher and flatten,
    while the sampled score keeps climbing, so reading only one of them can
    misjudge whether a run is still improving.
    """
    if not self.config.eval.get('gsm8k_greedy_token', True):
      return
    if self.sampling_temperature == 0.0:
      # The configured sampler is already greedy: `val/gsm8k_acc` *is* this
      # number, and running it twice would only spend the budget again.
      return

    saved = self.sampling_temperature
    self.sampling_temperature = 0.0
    try:
      rows, nfes, per_seq = self._gsm8k_generate(
        dataset, indices, batch_size, steps, step_convention)
    finally:
      self.sampling_temperature = saved

    if self.trainer.global_rank != 0:
      return
    correct, accuracy = self._gsm8k_grade(rows)
    avg_nfe = sum(nfes) / max(len(nfes), 1)
    print(f'[gsm8k-greedy] step {self.global_step}: {accuracy * 100:.2f}% '
          f'({correct}/{len(rows)}) at {avg_nfe:.1f} NFE '
          f'({sum(per_seq) / max(len(per_seq), 1):.1f} per sequence) '
          f'[{self.remasking_strategy}, T=0]', flush=True)
    self.log('val/gsm8k_acc_greedy',
             torch.tensor(accuracy, device=self.device),
             on_epoch=True, on_step=False, sync_dist=False,
             rank_zero_only=True)

  def _gsm8k_generate(self, dataset, indices, batch_size, steps,
                      step_convention):
    """Answer this rank's shard, then gather every rank's answers.

    Returns `(rows, nfes, nfes_per_seq)` -- decoded (response, gold) pairs, the
    forwards each *batch* cost, and the forwards each *sequence* cost. All three
    are the whole run's after the gather, so only rank zero's copy is complete
    enough to grade; the others get it too and simply drop it.
    """
    rows, nfes, per_seq = [], [], []
    for start in range(0, len(indices), batch_size):
      chunk = indices[start: start + batch_size]
      prompts = [torch.tensor(dataset[i]['input_ids']) for i in chunk]
      assert max(len(p) for p in prompts) < self.num_tokens, (
        f'a GSM8K prompt is at least model.length={self.num_tokens} long, '
        'leaving no room for an answer.')
      lengths = torch.tensor([len(p) for p in prompts], device=self.device)
      padded = torch.zeros(len(chunk), int(lengths.max()),
                           dtype=torch.long, device=self.device)
      for i, ids in enumerate(prompts):
        padded[i, :len(ids)] = ids.to(self.device)
      samples, metadata = self.generate_samples(
        num_samples=len(chunk), num_steps=steps,
        prefix_tokens=padded, prefix_lengths=lengths,
        return_metadata=True,
        step_convention=step_convention)
      nfes.append(metadata['nfe'])
      per_seq.extend(metadata['nfe_per_row'].cpu().tolist())
      for i, idx in enumerate(chunk):
        generated = samples[i, int(lengths[i]):].cpu().tolist()
        rows.append((self.tokenizer.decode(generated,
                                           skip_special_tokens=True),
                     dataset[idx]['response_ground_truth']))
      del samples, padded, prompts
    torch.cuda.empty_cache()

    world = max(1, self.trainer.world_size)
    if world > 1 and torch.distributed.is_available() \
        and torch.distributed.is_initialized():
      gathered = [None] * world
      torch.distributed.all_gather_object(gathered, (rows, nfes, per_seq))
      rows = [r for shard, _, _ in gathered for r in shard]
      nfes = [n for _, shard, _ in gathered for n in shard]
      per_seq = [n for _, _, shard in gathered for n in shard]
    return rows, nfes, per_seq

  @torch.no_grad()
  def _log_gsm8k_pareto(self, dataset, indices, batch_size):
    """Keep the original sweep and optionally add FP32-normalized predictions."""
    if not self.config.eval.get('gsm8k_pareto_thresholds', None):
      return
    fp32 = self.config.eval.get('gsm8k_pareto_fp32', True)
    if not isinstance(fp32, bool):
      raise ValueError('eval.gsm8k_pareto_fp32 must be a boolean')
    initial_rng = capture_rng_state() if fp32 else None
    self._run_gsm8k_pareto(dataset, indices, batch_size)
    if fp32:
      # Paired starting RNGs, with no extra effect on subsequent training.
      with replay_rng_state(initial_rng), fp32_prediction_normalization(self):
        self._run_gsm8k_pareto(dataset, indices, batch_size, suffix='_fp32norm')

  def _run_gsm8k_pareto(self, dataset, indices, batch_size, suffix=''):
    """Sweep the adaptive sampler's threshold and publish the frontier.

    A single `val/gsm8k_acc` cannot describe `low_confidence_dynamic`, because
    that rule has no fixed step budget -- its threshold is the budget and the
    NFE is an outcome. The honest summary is a curve, so score every threshold
    in `eval.gsm8k_pareto_thresholds` and log accuracy against realized NFE.

    Thresholds are swept in ascending order because NFE rises with them, and the
    per-checkpoint scalar series uses NFE as its step: underdeep, like most
    trackers, wants that axis monotonic.
    """
    config = self.config.eval
    thresholds = config.get('gsm8k_pareto_thresholds', None)
    if not thresholds:
      return
    thresholds = sorted(float(t) for t in thresholds)

    # The sweep overrides the sampler for its duration. Restore in `finally` --
    # otherwise the next validation's `val/gsm8k_acc` would silently be measured
    # with the last threshold's settings rather than the configured ones.
    saved = (self.remasking_strategy, self.dynamic_threshold,
             self.sampling_temperature)
    self.remasking_strategy = 'low_confidence_dynamic'
    self.sampling_temperature = float(
      config.get('gsm8k_pareto_temperature', 0.0))
    curve = []
    try:
      for threshold in thresholds:
        self.dynamic_threshold = threshold
        # `steps`/`step_convention` are ignored by the adaptive sampler; passed
        # only because the signature wants them.
        rows, _, per_seq = self._gsm8k_generate(
          dataset, indices, batch_size, self.config.sampling.steps, 'sflm')
        if self.trainer.global_rank != 0:
          continue
        correct, accuracy = self._gsm8k_grade(rows)
        nfe = sum(per_seq) / max(len(per_seq), 1)
        curve.append({'threshold': threshold, 'nfe': nfe,
                      'accuracy': accuracy})
        print(f'[gsm8k-pareto{suffix}] step {self.global_step} thr {threshold}: '
              f'{accuracy * 100:.2f}% ({correct}/{len(rows)}) '
              f'at {nfe:.1f} NFE/seq', flush=True)
    finally:
      (self.remasking_strategy, self.dynamic_threshold,
       self.sampling_temperature) = saved

    if self.trainer.global_rank != 0 or not curve:
      return
    self._publish_pareto(curve, suffix=suffix)

  def _publish_pareto(self, curve, suffix=''):
    """Log one checkpoint's frontier: scalar series, trends, and the figure."""
    logger = self.trainer.logger
    for point in curve:
      # Per-threshold trends against the *training* step, so each threshold also
      # reads as an ordinary curve over the run.
      tag = f"{point['threshold']:g}".replace('.', '')
      self.log(f'val/gsm8k_acc_thr{tag}{suffix}',
               torch.tensor(point['accuracy'], device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)
      self.log(f'val/gsm8k_nfe_thr{tag}{suffix}',
               torch.tensor(point['nfe'], device=self.device),
               on_epoch=True, on_step=False, sync_dist=False,
               rank_zero_only=True)

    if logger is None:
      return
    # The frontier itself, as a series whose x-axis is NFE rather than the
    # training step. `self.log` cannot express that -- it always stamps the
    # trainer's step -- so go through the logger's own `log_metrics`, which
    # passes `step` straight to the tracker. One series per checkpoint, since a
    # single graph cannot hold two curves.
    #
    # NFE is carried at `_NFE_X_SCALE`: the mean is a float, and rounding it to
    # a whole forward pass merged thresholds whose budgets differ by less than
    # one -- two points at the same x, where the later silently overwrote the
    # earlier. Read an x of 567 as 5.67 NFE.
    series = f'pareto/gsm8k_step_{self.global_step}{suffix}'
    for point in curve:
      try:
        logger.log_metrics({series: point['accuracy']},
                           step=int(round(point['nfe'] * _NFE_X_SCALE)))
      except Exception as e:  # pragma: no cover - tracker best-effort
        print(f'[gsm8k-pareto] scalar log failed: {e}', flush=True)
        break

    history_path = os.path.join(self.config.checkpointing.save_dir,
                                f'pareto_history{suffix}.json')
    run_uid = getattr(logger, 'version', '') or ''
    history_attr = f'_pareto_history{suffix}'
    history = getattr(self, history_attr, None)
    if history is None:
      # A *resumed* run resumes the same underdeep run, so the figure has to
      # pick up the curves the earlier process drew. A fresh run reusing the
      # same save_dir (logger.force_new_run, the default) must not -- it would
      # graft the previous run's frontiers onto this one's plot. The run uid
      # tells the two cases apart.
      history = {}
      if os.path.exists(history_path):
        try:
          with open(history_path) as f:
            stored = json.load(f)
          if stored.get('run_uid') == run_uid:
            history = {int(k): v for k, v in stored['curves'].items()}
          else:
            print('[gsm8k-pareto] ignoring history from a different run '
                  f"({stored.get('run_uid')!r} != {run_uid!r})", flush=True)
        except Exception as e:  # pragma: no cover
          print(f'[gsm8k-pareto] history read failed: {e}', flush=True)
      setattr(self, history_attr, history)
    history[int(self.global_step)] = curve
    try:
      os.makedirs(self.config.checkpointing.save_dir, exist_ok=True)
      with open(history_path, 'w') as f:
        json.dump({'run_uid': run_uid, 'curves': history},
                  f, indent=2)
    except Exception as e:  # pragma: no cover
      print(f'[gsm8k-pareto] history write failed: {e}', flush=True)

    if not hasattr(logger, 'log_image'):
      return
    try:
      import pareto_plot
      image = pareto_plot.render(
        history,
        reference=pareto_plot.load_reference(
          self.config.eval.get('gsm8k_pareto_reference', None)))
      logger.log_image(f'pareto/frontier{suffix}', image, step=self.global_step)
    except Exception as e:  # pragma: no cover - plotting is best-effort
      print(f'[gsm8k-pareto] figure failed: {e}', flush=True)

  def _gsm8k_grade(self, rows):
    """Execute each generated program and score it. Rank zero only."""
    import sandbox_gsm8k
    timeout = float(self.config.eval.get('tinygsm_exec_timeout', 5.0))
    correct = sum(
      sandbox_gsm8k.evaluate_samples(response, gold, timeout_s=timeout)
      for response, gold in rows)
    return correct, correct / max(len(rows), 1)

  def on_validation_epoch_start(self):
    self.metrics.reset()
    self._eval_mode()
    self._cond_batches = []
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    del batch_idx
    if (self._conditional_eval
        and self.config.eval.generate_samples
        and len(self._cond_batches)
        < self.config.sampling.num_sample_batches):
      self._cond_batches.append(
        {k: batch[k].clone()
         for k in ('input_ids', 'attention_mask', 'prompt_len')
         if k in batch})
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    return losses.loss

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      samples, text_samples = None, None
      cond_rows = []
      for _sb in range(
        self.config.sampling.num_sample_batches):
        if self._conditional_eval:
          # Answer a real validation question instead of sampling from
          # the prior.
          if _sb >= len(self._cond_batches):
            break
          (samples, answer_tokens, text_samples, gold_texts,
           question_texts) = self._conditional_val_batch(
             self._cond_batches[_sb])
          cond_rows.extend(zip(question_texts, text_samples, gold_texts))
          # Ragged on purpose: each row's answer starts at its own
          # prompt boundary, and the prompt must not dilute the metric.
          self.metrics.record_entropy(answer_tokens)
        else:
          samples = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size)

          self.metrics.record_entropy(samples)
          # Decode the samples to be re-tokenized by eval model
          text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.metrics.record_generative_perplexity(
            text_samples, self.num_tokens, self.device)
      if cond_rows:
        self._log_conditional_accuracy(cond_rows)
      if text_samples is not None:
        if self.trainer.global_rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Conditional runs show the question and reference answer
          # beside the completion; a table of bare programs is unreadable.
          if cond_rows:
            self.trainer.logger.log_table(
              key='val/samples',
              columns=['Question', 'Generated', 'Ground truth'],
              data=[list(r) for r in
                    cond_rows[: self.config.sampling.num_sample_log]],
              step=self.global_step)
          else:
            text_samples = text_samples[
              : self.config.sampling.num_sample_log]
            self.trainer.logger.log_table(
              key='val/samples',
              columns=['Generated Samples'],
              data=[[s] for s in text_samples],
              step=self.global_step)
        if self.config.eval.compute_generative_perplexity:
          self.log('val/gen_ppl',
                   self.metrics.gen_ppl.compute(),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          self.log('val/sample_entropy',
                   self.metrics.sample_entropy.compute(),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
    self._log_gsm8k_benchmark()
    self._train_mode()
    # Validation generation allocates far more than a training step, and with
    # `eval.validate_before_train` it runs first -- without this the first
    # training step meets a GPU that is nominally free but fully reserved by
    # the allocator.
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def generate_samples(self, num_samples, num_steps, eps,
                       prefix_tokens=None, prefix_lengths=None):
    raise NotImplementedError

  @staticmethod
  def _project_prefix(x, prefix_tokens, prefix_lengths):
    """Pin each row's prompt back onto the latent, in place.

    Conditional generation on a Q -> A corpus is unconditional sampling with
    the question held fixed. Rows have different prompt lengths, so
    `prefix_tokens` is right-padded and `prefix_lengths` says how much of each
    row is real.
    """
    if prefix_tokens is None:
      return x
    width = prefix_tokens.shape[1]
    positions = torch.arange(width, device=x.device)
    keep = positions[None, :] < prefix_lengths[:, None]
    x[:, :width] = torch.where(keep, prefix_tokens, x[:, :width])
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    self._eval_mode()
    samples = self.generate_samples(
      num_samples=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self._train_mode()
    return samples

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    raise NotImplementedError

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    (input_tokens, output_tokens,
     valid_tokens) = self._process_model_input(
       x0, valid_tokens)
    loss = self.nll(input_tokens, output_tokens,
                    current_accumulation_step, train_mode,
                    valid_tokens=valid_tokens)
    assert loss.ndim == 2
    if self.ignore_bos:
      loss[:, 1:] = loss[:, 1:]
      valid_tokens[:, 1:] = valid_tokens[:, 1:]

    nlls = (loss * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    token_nll = nlls / num_tokens

    return Loss(loss=token_nll,
                nlls=nlls,
                prior_loss=0.0,
                num_tokens=num_tokens)


class Diffusion(TrainerBase):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.noise_removal in {
      'none', 'ancestral', 'greedy'}
    assert self.loss_type in {'elbo', 'low_var'}
    if self.config.sampling.noise_removal == 'greedy':
      assert self.sampler != 'analytic'
      assert self.parameterization in {'mean', 'subs'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      if self.config.adversarial_distill.is_distill:
        t = t.chunk(self.accum_batches)[accum_step]
      else:
        t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype,
                     device=self.device)
    sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
    model_output_t0 = self.forward(x0, sigma_t0)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)


  def nll_per_token(self, model_output, xt, x0, alpha_t,
                    dalpha_t, low_var):
    raise NotImplementedError

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, valid_tokens=valid_tokens)
    log_x_theta = self.forward(xt, sigma=sigma)
    utils.print_nans(log_x_theta, 'model_output')
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

  def _get_score(self, **kwargs):
    del kwargs
    raise NotImplementedError

  def _denoiser_update(self, x, t):
    raise NotImplementedError

  def _analytic_update(self, x, t, dt):
    raise NotImplementedError

  def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
    raise NotImplementedError

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5, prefix_tokens=None,
                       prefix_lengths=None,
                       return_metadata=False,
                       step_convention=None):
    """Generate samples from the model.

    With `prefix_tokens` / `prefix_lengths` this is *conditional*
    generation: those tokens are written into the initial latent and
    re-pinned after every update, so the model completes the prompt
    instead of sampling a whole sequence from the prior.
    """
    if self.remasking_strategy in confidence_sampling.ADAPTIVE_STRATEGIES:
      # Its threshold, not `num_steps`, is the budget -- see the docstring of
      # `_adaptive_generate_samples`.
      return self._adaptive_generate_samples(
        num_samples=num_samples, eps=eps, prefix_tokens=prefix_tokens,
        prefix_lengths=prefix_lengths, return_metadata=return_metadata)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens)
    x = self._project_prefix(x, prefix_tokens, prefix_lengths)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    nfe = 0

    noise_removal = self.config.sampling.noise_removal
    # Where the noise-removal step lives. 'idlm' runs `num_steps` reverse
    # transitions and then appends a separate denoising call, so
    # NFE == num_steps + 1. 'sflm' folds denoising into the final loop
    # iteration -- matching jdeschena/s-flm, whose conditional TinyGSM
    # protocol appendix E.2 of the paper says we follow -- so NFE == num_steps.
    if step_convention is None:
      step_convention = self.config.sampling.get('step_convention', 'idlm')
    fold_final = (step_convention == 'sflm')

    for i in range(num_steps):
      is_final = (fold_final and i == num_steps - 1
                  and noise_removal != 'none')
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if is_final and noise_removal == 'greedy':
        nfe += 1
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
      elif is_final and self.sampler == 'analytic':
        nfe += 1
        x = self._denoiser_update(x=x, t=t)
      elif self.sampler == 'ancestral':
        nfe += 1
        _, x = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=None,
          noise_removal_step=is_final)
      elif self.sampler == 'ancestral_cache':
        nfe += int(p_x0_cache is None)
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=p_x0_cache,
          noise_removal_step=is_final)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        nfe += 1
        x = self._analytic_update(x=x,t=t, dt=dt)
      x = self._project_prefix(x, prefix_tokens, prefix_lengths)

    if not fold_final:
      t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                      device=self.device)
      if noise_removal == 'ancestral':
        if self.sampler == 'analytic':
          nfe += 1
          x = self._denoiser_update(x=x, t=t0)
        else:
          nfe += int(p_x0_cache is None)
          _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                   p_x0=p_x0_cache,
                                   noise_removal_step=True)
      elif noise_removal == 'greedy':
        nfe += 1
        sigma = self._sigma_from_alphat(self.noise(t0)[1])
        x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
      x = self._project_prefix(x, prefix_tokens, prefix_lengths)
    if return_metadata:
      # A fixed grid spends the same forwards on every row, so per-row cost is
      # just `nfe`. Reported anyway so callers need not care which sampler ran.
      return x, {'nfe': nfe,
                 'nfe_per_row': torch.full((x.shape[0],), float(nfe),
                                           device=self.device)}
    return x

  @torch.no_grad()
  def _adaptive_generate_samples(self, num_samples, eps=1e-5,
                                 prefix_tokens=None, prefix_lengths=None,
                                 return_metadata=False):
    """Threshold-driven generation, for the adaptive remasking strategies.

    `sampling.steps` is deliberately NOT read here. Under an adaptive rule the
    threshold is the budget and the step count is an outcome: each step commits
    however many positions clear `sampling.dynamic_threshold`, so an easy
    sequence finishes in a handful of forwards and a hard one takes more. This
    mirrors yrOPDLM, which sweeps the threshold (0.85/0.9/0.95, tagged `dyn<b>`)
    while pinning its nominal step count to the worst case, and reports the
    realized forwards -- `train/online_eval.py::_budget_grid` and the
    `low_confidence_dynamic` branch of `_eval_one_budget`.

    `sampling.max_steps` only caps the loop; `null` gives it the one-commit-per-
    step worst case, `model.length`, which `select_threshold`'s forced-argmax
    fallback guarantees is never reached.
    """
    x = self.prior_sample(num_samples, self.num_tokens)
    x = self._project_prefix(x, prefix_tokens, prefix_lengths)
    max_steps = int(self.config.sampling.get('max_steps', None)
                    or self.num_tokens)
    nfe = 0
    steps = 0
    # Forwards each *row* needed, i.e. the step it stopped having masks on. The
    # batch as a whole runs until its slowest row finishes, so `nfe` is a max
    # over rows and grows with the batch size; this is the mean-able,
    # batch-size-independent cost, and the x-axis a Pareto curve wants.
    nfe_per_row = torch.zeros(x.shape[0], device=self.device)

    for _ in range(max_steps):
      unfinished = (x == self.mask_index).any(dim=-1)
      if not bool(unfinished.any()):
        break
      nfe_per_row += unfinished.to(nfe_per_row.dtype)
      # There is no timestep grid to walk any more, so read the time off the
      # latent itself: alpha_t is the fraction of the sequence already decided.
      # It reaches the model only as `sigma`, which MDLM's subs parameterisation
      # discards outright (algo.MDLM._process_model_output), and for a
      # time-conditioned absorbing model it is the honest value.
      t = self._alpha_to_t(1 - (x == self.mask_index).to(
        torch.float32).mean(dim=-1, keepdim=True), eps=eps)
      _, x_next = self._ancestral_update(
        x=x, t=t, dt=None, p_x0=None, noise_removal_step=False)
      nfe += 1
      steps += 1
      x = self._project_prefix(x_next, prefix_tokens, prefix_lengths)

    stragglers = (x == self.mask_index).any(dim=-1)
    if bool(stragglers.any()):
      # Only reachable if `max_steps` was set below the worst case by hand.
      t0 = eps * torch.ones(x.shape[0], 1, device=self.device)
      _, x = self._ancestral_update(x=x, t=t0, dt=None, p_x0=None,
                                    noise_removal_step=True)
      nfe += 1
      nfe_per_row += stragglers.to(nfe_per_row.dtype)
      x = self._project_prefix(x, prefix_tokens, prefix_lengths)

    if return_metadata:
      return x, {'nfe': nfe, 'steps': steps, 'nfe_per_row': nfe_per_row}
    return x

  def _alpha_to_t(self, alpha_t, eps=1e-5):
    """Invert the noise schedule: the `t` whose alpha is `alpha_t`.

    Only correct for the linear `LogLinear` schedule this repo uses, which is
    also the only one `noise` implements.
    """
    t = (1 - alpha_t) / (1 - self.noise.eps)
    return t.clamp(eps, 1.0)

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, stride_length, num_strides, dt=0.001):
    # TODO(subham): Test this method after refactoring.
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self.prior_sample(n_samples, self.num_tokens)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    # TODO(subham): Test this method after refactoring.
    self._eval_mode()
    (sampling_steps, samples,
     sequence_lengths) = self._semi_ar_sampler(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    self._train_mode()
    return sampling_steps, samples, sequence_lengths


class AbsorbingState(Diffusion):
  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do 
    # vocab_size = len(tokenizer), so that we account
    # for the special tokens added in dataloader.py.
    # But we use tokenizer.vocab_size so as to to be
    # consistent with the prior checkpoints.
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    self.subs_masking = config.algo.subs_masking
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    assert not (self.parameterization == 'mean'
                and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'mean'

  def q_xt(self, x, alpha_t, valid_tokens=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      alpha_t: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < 1 - alpha_t
    if valid_tokens is not None:
      move_indices = move_indices & valid_tokens.bool()
    xt = torch.where(move_indices, self.mask_index, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    elif dt is None:
      # Only the adaptive strategies omit `dt` mid-decode: their commit set
      # comes from a confidence threshold rather than from the schedule's
      # per-step budget, so there is no `s` to evaluate.
      assert self.remasking_strategy in \
        confidence_sampling.ADAPTIVE_STRATEGIES
      alpha_s = None
    else:
      _, alpha_s = self.noise(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      p_x0 = self.forward(
        x, self._sigma_from_alphat(alpha_t)).exp()

    if self.remasking_strategy != 'random':
      # `p_x0` is returned unchanged either way, so the caching contract in
      # `generate_samples` is identical: a step that commits nothing leaves `x`
      # untouched and the cached distribution is reused instead of a forward.
      return p_x0, self._confidence_commit(
        x=x, p_x0=p_x0, alpha_s=alpha_s, alpha_t=alpha_t,
        noise_removal_step=noise_removal_step)

    if self.sampling_temperature != 1.0:
      return p_x0, self._ancestral_greedy_token(x, p_x0, alpha_s, alpha_t)

    q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    _x = sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ancestral_greedy_token(self, x, p_x0, alpha_s, alpha_t):
    """Ancestral unmasking, but the token is chosen at `sampling.temperature`.

    The `q_xs` draw above makes two decisions in one categorical: it puts mass
    `1 - alpha_s` on "stay masked" and `p_x0[v] * (alpha_s - alpha_t)` on each
    token v. Fused like that, the token can only ever come out at temperature 1.
    Greedy-token ancestral decoding needs them separated, so do exactly the same
    two decisions explicitly -- unmask with the conditional probability implied
    by those masses,

        P(unmask | masked at t) = (alpha_s - alpha_t) / (1 - alpha_t),

    then write whichever token `_propose_tokens` picks. At temperature 1 this is
    the same distribution, which is why the fused form above is kept as the
    default: it is also the same *draw*, so every existing number reproduces
    bit-for-bit rather than merely in expectation.
    """
    x0, _ = self._propose_tokens(p_x0)
    unmask_prob = ((alpha_s - alpha_t)
                   / (1 - alpha_t).clamp_min(1e-12)).clamp(0, 1)
    unmask = torch.rand(x.shape, device=x.device) < unmask_prob
    return torch.where((x == self.mask_index) & unmask, x0, x)

  def _propose_tokens(self, p_x0):
    """The token each position would receive, and how sure the model is of it.

    `sampling.temperature` shapes only the draw. The confidence is always read off
    the raw `p_x0`, which is what makes the ranking meaningful under greedy
    decoding -- the tempered distribution is one-hot there, so every confidence
    would be 1.0 and the whole sequence would commit in a single step. Same split
    as yrOPDLM's `_compute_probs` (probs_sampling vs probs_raw).
    """
    temperature = self.sampling_temperature
    if temperature == 0:
      x0 = p_x0.argmax(dim=-1)
    elif temperature == 1:
      x0 = sample_categorical(p_x0)
    else:
      x0 = sample_categorical(
        torch.softmax(p_x0.clamp_min(1e-30).log() / temperature, dim=-1))
    confidence = p_x0.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    return x0, confidence.float()

  def _confidence_commit(self, x, p_x0, alpha_s, alpha_t,
                         noise_removal_step):
    """Commit the positions the model is most confident about.

    Replaces the independent Bernoulli unmasking of `_ancestral_update` with
    yrOPDLM's confidence-ranked rules; see confidence_sampling for the two
    budgets. Returns the updated `x`.
    """
    candidates = x == self.mask_index
    x0, confidence = self._propose_tokens(p_x0)
    confidence = torch.where(
      candidates, confidence,
      torch.full_like(confidence, -float('inf')))

    if noise_removal_step:
      # alpha_s == 1: the random path drops every mask here too, and doing the
      # same is what stops an adaptive decode from emitting a MASK token.
      transfer = candidates
    elif self.remasking_strategy == 'low_confidence_static':
      frac = ((alpha_s - alpha_t)
              / (1 - alpha_t).clamp_min(1e-12)).clamp(0, 1)
      k = confidence_sampling.commit_counts(
        candidates.sum(dim=-1), frac.squeeze(-1))
      transfer = confidence_sampling.select_top_k(confidence, candidates, k)
    else:
      transfer = confidence_sampling.select_threshold(
        confidence, candidates, self.dynamic_threshold)
    return torch.where(transfer, x0, x)

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_alphat(self.noise(t)[1])
    sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self._get_score(x, sigma_t)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_alphat(self.noise(t)[1])
    score = self._get_score(x, sigma)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge


class UniformState(Diffusion):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.time_conditioning
    assert self.parameterization == 'mean'
    # A uniform-state latent has no absorbing mask, so there is no set of
    # "positions still to be decided" for a confidence rule to rank.
    assert self.remasking_strategy == 'random', (
      'sampling.remasking_strategy is an absorbing-state (masked diffusion) '
      f'decoder; got {self.remasking_strategy!r} on a uniform-state model.')
    # Same reason: its ancestral update resamples every position from a
    # transition kernel, so there is no per-position token choice to temper.
    assert self.sampling_temperature == 1.0, (
      'sampling.temperature is an absorbing-state decoding knob; got '
      f'{self.sampling_temperature} on a uniform-state model.')
    if self.config.algo.name != 'distillation':
      assert self.T == 0

  def q_xt(self, x, alpha_t, valid_tokens=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      move_chance: float torch.Tensor with shape
        (batch_size, 1).
    """
    move_indices = torch.rand(
      *x.shape, device=x.device) < 1 - alpha_t
    if valid_tokens is not None:
      move_indices = move_indices & valid_tokens.bool()
    uniform_tensor = torch.randint(
      0, self.vocab_size, x.shape, device=x.device)
    xt = torch.where(move_indices, uniform_tensor, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return torch.randint(
      0, self.vocab_size, batch_dims, dtype=torch.int64,
      device=self.device)
