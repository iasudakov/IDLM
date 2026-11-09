"""Full-attention CDLM adaptation for the IDLM MDLM TinyGSM checkpoint."""

import copy
from pathlib import Path

import hydra.utils
import models.ema
from model_assets import resolve_checkpoint
from sdtt_checkpoint import apply_ema, backbone_state, read_checkpoint
import torch
import trainer_base

from cdlm_objective import (corrupt_ground_truth, denoising_loss,
                            distribution_loss, trajectory_pair)


class CDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer, *, inference_only=False):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        if config.adversarial_distill.is_distill:
            raise ValueError('CDLM uses automatic student optimization, not adversarial distillation.')
        if self.time_conditioning or self.parameterization != 'subs' or self.T != 0:
            raise ValueError('CDLM TinyGSM requires time-unconditional continuous MDLM/SUBS.')
        if config.algo.attention_mode != 'full' or config.algo.causal_attention:
            raise ValueError('This adaptation preserves MDLM full bidirectional attention.')
        if config.algo.kld_type not in {'forward_kl', 'reverse_kl'}:
            raise ValueError('Unknown CDLM kld_type.')
        if type(config.algo.segment_length) is not int or config.algo.segment_length <= 0:
            raise ValueError('segment_length must be positive.')
        for name in ('distill_weight', 'consistency_weight', 'denoise_weight'):
            if config.algo[name] < 0:
                raise ValueError('CDLM loss weights must be nonnegative.')
        self._cache_fingerprint = None
        if not inference_only:
            # Training only needs the frozen output projection: teacher hidden
            # states were saved before this exact layer during preprocessing.
            self.teacher_head = copy.deepcopy(self.backbone.output_layer.linear)
            self.teacher_head.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher_head'):
            self.teacher_head.eval()
        return self

    def _process_model_output(self, model_output, xt, sigma):
        logits = model_output.float()
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        visible = xt.ne(self.mask_index)
        logits[visible] = self.neg_infinity
        logits[visible, xt[visible]] = 0
        return logits

    def _checkpoint_source(self, source):
        bundle = self.config.get('model_weights_dir')
        if bundle and source.startswith('hf://'):
            raise ValueError('With model_weights_dir, supply a local MDLM checkpoint path.')
        return resolve_checkpoint(source, bundle)

    def _student_checkpoint_source(self, source):
        # An explicit student Hub source is an opt-in download/cache lookup.
        # The original teacher and evaluation assets keep their bundle policy.
        if source.startswith('hf://'):
            return source
        return self._checkpoint_source(source)

    def initialize_from_checkpoint(self, source, student_source=None, student_use_ema=False):
        """Keep the original teacher head while optionally warm-starting the student.

        Only backbone weights are imported from the optional student checkpoint;
        optimizer, scheduler, loop state and its old EMA history are discarded.
        Full training resumes bypass this initializer in the main entry point.
        """
        checkpoint = read_checkpoint(self._checkpoint_source(source))
        self.backbone.load_state_dict(backbone_state(checkpoint), strict=True)
        self.teacher_head.load_state_dict(self.backbone.output_layer.linear.state_dict(), strict=True)
        del checkpoint
        if student_source:
            checkpoint = read_checkpoint(self._student_checkpoint_source(student_source))
            if not student_use_ema and checkpoint.get('checkpoint_weights') == 'ema':
                raise ValueError('Student initialization checkpoint contains only EMA weights; '
                                 'set training.student_init_use_ema=True to use them.')
            self.backbone.load_state_dict(backbone_state(checkpoint), strict=True)
            if student_use_ema:
                # _get_parameters contains only student/noise parameters; the
                # frozen original teacher_head must retain its own vocabulary map.
                apply_ema(self, checkpoint)
        if self.ema:
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema)

    def load_evaluation_weights(self, source, use_ema=True):
        checkpoint = read_checkpoint(self._checkpoint_source(source))
        self.backbone.load_state_dict(backbone_state(checkpoint), strict=True)
        if use_ema:
            apply_ema(self, checkpoint)
        elif checkpoint.get('checkpoint_weights') == 'ema':
            raise ValueError('This checkpoint contains only EMA weights.')
        self.ema = None
        self._evaluation_weights = 'ema' if use_ema else 'live'

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['cdlm_cache_fingerprint'] = self._cache_fingerprint

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get('cdlm_cache_fingerprint')
        if self._cache_fingerprint is not None and saved != self._cache_fingerprint:
            raise ValueError('The resumed CDLM checkpoint belongs to a different trajectory cache.')
        super().on_load_checkpoint(checkpoint)

    def on_train_start(self):
        effective = (self.config.loader.batch_size * self.trainer.world_size
                     * self.trainer.accumulate_grad_batches)
        if effective != self.config.loader.global_batch_size:
            raise ValueError(f'loader.global_batch_size must equal actual batch {effective}.')
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        if self.config.eval.get('validate_before_train', False) and self.global_step == 0:
            from sdtt_eval import log_validation_gsm8k
            self._eval_mode()
            try:
                log_validation_gsm8k(self, initial=True)
            finally:
                self._train_mode()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(), lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps, weight_decay=self.config.optim.weight_decay)
        ratio = self.config.algo.get('warmup_ratio')
        if ratio is not None:
            if not 0 <= ratio <= 1:
                raise ValueError('algo.warmup_ratio must be in [0, 1] or null.')
            self.config.lr_scheduler.num_warmup_steps = int(
                self.trainer.estimated_stepping_batches * ratio)
        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step',
                              'monitor': 'val/loss', 'name': 'trainer/lr'}]

    def training_step(self, batch, batch_idx):
        original, valid = batch['input_ids'], batch['attention_mask'].bool()
        earlier, later, new, still, _, _ = trajectory_pair(
            batch['final_tokens'], batch['reveal_order'], valid, self.mask_index,
            self.config.algo.segment_length)
        # MDLM ignores sigma, but its existing forward interface expects [B,1].
        sigma = torch.zeros(original.shape[0], 1, device=original.device)
        predictions = self.forward(earlier, sigma)
        with torch.no_grad():
            endpoint = self.forward(later, sigma)
            # Only reconstruct the newly finalized targets, avoiding a second
            # complete [B,L,V] teacher buffer in memory.
            hidden = batch['teacher_hidden'][new].to(self.teacher_head.weight.dtype)
            target_logits = self.teacher_head(hidden).float()
            target_logits[:, self.mask_index] += self.neg_infinity
        distill = distribution_loss(predictions[new], target_logits, self.config.algo.kld_type)
        consistency = distribution_loss(predictions[still], endpoint[still],
                                        self.config.algo.kld_type, target_is_log_probs=True)
        del endpoint, target_logits
        corrupted, selected, probabilities = corrupt_ground_truth(
            original, valid, self.mask_index, self.config.algo.denoise_mask_floor)
        ground_truth = self.forward(corrupted, sigma)
        denoise = denoising_loss(ground_truth, original, selected, probabilities, valid)
        cfg = self.config.algo
        loss = (cfg.distill_weight * distill + cfg.consistency_weight * consistency
                + cfg.denoise_weight * denoise)
        for name, value in [('student_loss', loss), ('train/cdlm_loss', loss),
                            ('train/cdlm_distill', distill),
                            ('train/cdlm_consistency', consistency),
                            ('train/cdlm_denoise', denoise)]:
            self.log(name, value, on_step=True, sync_dist=True)
        return loss

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        log_prob = log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        return log_prob * dalpha_t / (1 - alpha_t)

    def on_validation_epoch_end(self):
        from sdtt_eval import log_validation_gsm8k
        try:
            log_validation_gsm8k(self)
        finally:
            super().on_validation_epoch_end()

    def on_train_end(self):
        output = Path(self.config.checkpointing.save_dir) / 'checkpoints' / 'last.ckpt'
        output.parent.mkdir(parents=True, exist_ok=True)
        # Save live weights and separate EMA state before final reporting, so
        # an evaluator or tracker failure cannot discard the final updates.
        self.trainer.save_checkpoint(str(output))
        # Finite 16-epoch training can finish between periodic validations and
        # saves. Publish the final EMA frontier before the tracker is finalized.
        from sdtt_eval import log_validation_gsm8k
        self._eval_mode()
        try:
            log_validation_gsm8k(self, final=True)
        finally:
            self._train_mode()
            if self.config.checkpointing.get('snapshot', False):
                import nirvana_snapshot
                # The CheckpointIO already published last.ckpt before evaluation;
                # now retain the final JSON/frontier and tracker metadata too.
                nirvana_snapshot.save(
                    str(self.config.checkpointing.save_dir), enabled=True,
                    namespace=self.config.checkpointing.get('snapshot_namespace'),
                    reason=f'final evaluation step={self.global_step}')
