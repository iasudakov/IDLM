"""SDTT for IDLM's MDLM backbone with conditional TinyGSM masking."""

import copy

import torch

import models.ema
import trainer_base
from sdtt_checkpoint import apply_ema, backbone_state, read_checkpoint
from sdtt_objective import collect_teacher_targets, masked_reverse_kl, round_schedule


class SDTT(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer, *, inference_only=False):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        if config.adversarial_distill.is_distill:
            raise ValueError('SDTT uses automatic student optimization; disable adversarial_distill.is_distill.')
        if self.time_conditioning or self.parameterization != 'subs' or self.T != 0:
            raise ValueError('This SDTT baseline requires time-unconditional continuous MDLM (subs).')
        if config.training.get('mask_layout', 'random') != 'random':
            raise ValueError('SDTT uses full-sequence independent random masking.')
        cfg = config.algo
        round_schedule(cfg.orig_num_sampling_steps, cfg.num_distill_steps,
                       cfg.min_num_sampling_steps, cfg.grow_dt_every, 0)
        self.register_buffer('_sdtt_round', torch.tensor(0, dtype=torch.long))
        if not inference_only:
            self.teacher = copy.deepcopy(self.backbone).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self

    def _process_model_output(self, model_output, xt, sigma):
        # MDLM SUBS parameterization, identical for teacher and student.
        logits = model_output.float()
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        visible = xt.ne(self.mask_index)
        logits[visible] = self.neg_infinity
        logits[visible, xt[visible]] = 0
        return logits

    def forward_teacher(self, xt, sigma):
        return self._process_model_output(
            self.teacher(xt, self._process_sigma(sigma)), xt, sigma)

    def initialize_from_checkpoint(self, source):
        checkpoint = read_checkpoint(source)
        state = backbone_state(checkpoint)
        self.backbone.load_state_dict(state, strict=True)
        self.teacher.load_state_dict(state, strict=True)
        # Official SDTT initializes both from raw teacher weights and resets
        # student EMA. Pretraining optimizer/loop state must not be resumed.
        if self.ema is not None:
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema)

    def load_evaluation_weights(self, source, use_ema=True):
        checkpoint = read_checkpoint(source)
        self.backbone.load_state_dict(backbone_state(checkpoint), strict=True)
        if use_ema:
            apply_ema(self, checkpoint)
        elif checkpoint.get('checkpoint_weights') == 'ema':
            raise ValueError('This checkpoint contains only EMA weights.')
        # Weights are applied once; avoid replacing them with fresh EMA shadows.
        self.ema = None
        self._evaluation_weights = 'ema' if use_ema else 'live'

    def _schedule(self, step=None):
        cfg = self.config.algo
        return round_schedule(cfg.orig_num_sampling_steps, cfg.num_distill_steps,
                              cfg.min_num_sampling_steps, cfg.grow_dt_every,
                              int(self.global_step) if step is None else step)

    @torch.no_grad()
    def _promote_teacher(self, step):
        stage, _, _ = self._schedule(step)
        saved = int(self._sdtt_round)
        if stage < saved or stage > saved + 1:
            raise ValueError('SDTT schedule differs from the resumed checkpoint.')
        if stage == saved:
            return
        if self.config.algo.use_ema_on_growth:
            if self.ema is None:
                raise ValueError('use_ema_on_growth requires training.ema > 0.')
            self.ema.copy_to(self._get_parameters())
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema)
            self.ema.move_shadow_params_to_device(self.device)
        self.teacher.load_state_dict(self.backbone.state_dict(), strict=True)
        self.teacher.eval()
        self._sdtt_round.fill_(stage)

    def on_train_start(self):
        # Check the actual global batch rather than silently treating a loader
        # metadata value as the number of examples per optimizer update.
        effective = (self.config.loader.batch_size * self.trainer.world_size
                     * self.trainer.accumulate_grad_batches)
        if effective != self.config.loader.global_batch_size:
            raise ValueError(f'loader.global_batch_size must equal actual batch {effective}.')
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Use Lightning's normal distributed sampler and checkpoint loop. IDLM's
        # legacy loader replacement always enables persistent_workers, even at 0.
        if self.config.eval.get('validate_before_train', False) and self.global_step == 0:
            from sdtt_eval import log_validation_gsm8k
            self._eval_mode()
            try:
                log_validation_gsm8k(self, initial=True)
            finally:
                self._train_mode()

    def on_train_batch_start(self, batch, batch_idx):
        self._promote_teacher(int(self.global_step))

    def training_step(self, batch, batch_idx):
        x0, valid = batch['input_ids'], batch['attention_mask'].bool()
        accum = batch_idx % self.trainer.accumulate_grad_batches
        t = self._sample_t(x0.shape[0], accum)
        _, alpha = self.noise(t.unsqueeze(-1))
        xt = self.q_xt(x0, alpha, valid_tokens=valid)
        _, teacher_grid, student_grid = self._schedule()
        targets = collect_teacher_targets(
            xt, t, forward=self.forward_teacher, noise=self.noise,
            mask_index=self.mask_index, vocab_size=self.vocab_size,
            dt=(1 - self.sampling_eps) / teacher_grid,
            num_steps=self.config.algo.num_distill_steps,
            sampling_eps=self.sampling_eps,
            sample_categorical=trainer_base.sample_categorical)
        predictions = self.forward(xt, self._sigma_from_alphat(alpha))
        loss = masked_reverse_kl(predictions, targets, xt.eq(self.mask_index) & valid)
        self.log('train/sdtt_kl', loss, on_step=True, sync_dist=True)
        self.log('train/sdtt_round', int(self._sdtt_round), on_step=True)
        self.log('train/sdtt_student_steps', student_grid, on_step=True)
        return loss

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        # Held-out MDLM ELBO diagnostic; never used as the SDTT objective.
        log_prob = log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        return log_prob * dalpha_t / (1 - alpha_t)

    def on_validation_epoch_end(self):
        # TrainerBase has already put EMA weights into the backbone.
        from sdtt_eval import log_validation_gsm8k
        try:
            log_validation_gsm8k(self)
        finally:
            # The inherited hook logs the held-out ELBO and restores live
            # student parameters exactly once, after all EMA evaluation.
            super().on_validation_epoch_end()
