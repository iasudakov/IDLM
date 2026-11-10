"""TinyGSM representation-MMD after CDLM, with a separate MDLM feature model.

The optimizer objective is the historical TinyGSM MMD recipe. The lifecycle
keeps the CDLM branch's GSM8K/Underdeep reports and full Nirvana checkpoints.
"""

import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import dataloader
import models.ema
from model_assets import resolve_checkpoint
from mmd_objective import FeatureReward, reinforce_loss_from_samples
from sdtt_checkpoint import backbone_state, read_checkpoint
import trainer_base


def answer_content_mask(tokens, valid_tokens, tokenizer):
    """Keep answer tokens through the first eligible EOS, excluding padding."""
    selected = valid_tokens.bool()
    if tokenizer.pad_token_id is not None:
        selected = selected & tokens.ne(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        eos = tokens.eq(tokenizer.eos_token_id) & selected
        previous_eos = F.pad(eos.cumsum(-1)[:, :-1], (1, 0))
        selected = selected & previous_eos.eq(0)
    return selected


class MMDFinetune(trainer_base.AbsorbingState):
    """Train the selected CDLM student using frozen original-MDLM features."""

    def __init__(self, config, tokenizer, *, inference_only=False):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        if config.adversarial_distill.is_distill:
            raise ValueError('MMD finetuning uses automatic optimization; disable adversarial distillation.')
        if (config.data.train != 'tiny_gsm' or self.time_conditioning
                or self.parameterization != 'subs' or self.T != 0):
            raise ValueError('This MMD recipe requires time-unconditional continuous TinyGSM MDLM/SUBS.')
        if config.algo.causal_attention or self.sampler != 'ancestral_cache':
            raise ValueError('MMD finetuning requires full MDLM attention and ancestral_cache sampling.')
        if inference_only:
            return
        self.teacher = copy.deepcopy(self.backbone).requires_grad_(False).eval()
        cfg = config.mmd
        physical_batch = cfg.get('student_forward_batch_size')
        if physical_batch is not None and (type(physical_batch) is not int or physical_batch < 1):
            raise ValueError('mmd.student_forward_batch_size must be a positive integer or null.')
        if type(cfg.get('student_activation_checkpointing', False)) is not bool:
            raise ValueError('mmd.student_activation_checkpointing must be a boolean.')
        self.reward = FeatureReward(
            self.teacher, layer=cfg.feature_layer, kernel=cfg.kernel,
            alpha=cfg.rbf_alpha, block_size=cfg.kernel_block_size,
            feature_batch_size=cfg.feature_batch_size)
        self._mmd_metric_sum = None
        self._mmd_metric_step = 0
        self._mmd_initialization = None

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self

    def _process_model_output(self, model_output, xt, sigma):
        # Preserve the historical MDLM output dtype (BF16 for the local DiT).
        # Only the detached representation-kernel arithmetic uses FP32.
        logits = model_output
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        visible = xt.ne(self.mask_index)
        logits[visible] = self.neg_infinity
        logits[visible, xt[visible]] = 0
        return logits

    def _checkpoint_source(self, source):
        source = str(source)
        bundle = self.config.get('model_weights_dir')
        if bundle and source.startswith('hf://'):
            raise ValueError('With model_weights_dir, prepare a local MMD initialization checkpoint first.')
        return resolve_checkpoint(source, bundle)

    @staticmethod
    def _load_selected_backbone(module, checkpoint, *, use_ema, label):
        if checkpoint.get('checkpoint_weights') == 'ema' and not use_ema:
            raise ValueError(f'{label} checkpoint contains only EMA weights; explicitly select EMA.')
        module.load_state_dict(backbone_state(checkpoint), strict=True)
        if not use_ema or checkpoint.get('checkpoint_weights') == 'ema':
            return
        ema = checkpoint.get('ema')
        if ema is None:
            raise ValueError(f'{label} checkpoint has no EMA weights.')
        parameters = list(module.parameters())
        shadows = ema.get('shadow_params', [])
        if len(parameters) != len(shadows) or any(p.shape != s.shape for p, s in zip(parameters, shadows)):
            raise ValueError(f'{label} checkpoint EMA shapes do not match its backbone.')
        with torch.no_grad():
            for parameter, shadow in zip(parameters, shadows):
                parameter.copy_(shadow)

    def initialize_from_checkpoint(self, student_source, feature_source, *,
                                   student_use_ema=False, retain_ema=True,
                                   feature_use_ema=False):
        """Load only model weights; optimizer, sampler and loops always start fresh.

        The paper warm start uses raw student weights and retains the saved
        student's EMA history. Selecting EMA student weights requires
        ``retain_ema=False`` and creates new EMA history at those weights.
        Frozen features always come from the independently selected source.
        """
        if student_use_ema and retain_ema:
            raise ValueError('student_use_ema=True requires retain_ema=False: EMA-selected starts reset EMA history.')
        student = read_checkpoint(self._checkpoint_source(student_source))
        self._load_selected_backbone(self.backbone, student, use_ema=student_use_ema, label='Student')
        feature = read_checkpoint(self._checkpoint_source(feature_source))
        self._load_selected_backbone(self.teacher, feature, use_ema=feature_use_ema, label='Feature model')
        self.teacher.requires_grad_(False).eval()
        if self.ema is not None:
            if retain_ema:
                state = student.get('ema')
                parameters = [p for p in self._get_parameters() if p.requires_grad]
                shadows = state.get('shadow_params', []) if state is not None else []
                if state is None or len(parameters) != len(shadows):
                    raise ValueError('Retaining EMA requires a student checkpoint with matching EMA parameters.')
                if any(p.shape != s.shape for p, s in zip(parameters, shadows)):
                    raise ValueError('Student checkpoint EMA shapes do not match the model.')
                decay, updates = state.get('decay'), state.get('num_updates')
                if (not isinstance(decay, (int, float)) or not math.isfinite(decay)
                        or not 0 <= decay <= 1 or
                        (updates is not None and (type(updates) is not int or updates < 0))):
                    raise ValueError('Student checkpoint EMA decay or update counter is invalid.')
                self.ema.load_state_dict(copy.deepcopy(state))
            else:
                self.ema = models.ema.ExponentialMovingAverage(
                    self._get_parameters(), decay=self.config.training.ema)
        self._mmd_initialization = {
            'student_source': str(student_source), 'feature_source': str(feature_source),
            'student_use_ema': bool(student_use_ema), 'retain_ema': bool(retain_ema),
            'feature_use_ema': bool(feature_use_ema)}
        manifest = self.config.mmd.get('initialization_manifest')
        if manifest:
            provenance = json.loads(Path(manifest).read_text())
            if not isinstance(provenance, dict):
                raise ValueError('MMD initialization manifest must be a JSON object.')
            self._mmd_initialization['manifest'] = provenance

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['mmd_finetune_schema'] = 1
        checkpoint['mmd_initialization'] = copy.deepcopy(self._mmd_initialization)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('mmd_finetune_schema') != 1:
            raise ValueError('MMD resume requires a full MMD-finetune checkpoint. '
                             'Use student initialization for a CDLM checkpoint.')
        super().on_load_checkpoint(checkpoint)
        self._mmd_initialization = copy.deepcopy(checkpoint.get('mmd_initialization'))

    def load_evaluation_weights(self, source, use_ema=True):
        checkpoint = read_checkpoint(self._checkpoint_source(source))
        self._load_selected_backbone(self.backbone, checkpoint, use_ema=use_ema, label='Evaluation')
        self.ema = None
        self._evaluation_weights = 'ema' if use_ema else 'live'

    def on_train_start(self):
        effective = (self.config.loader.batch_size * self.trainer.world_size
                     * self.trainer.accumulate_grad_batches)
        if effective != self.config.loader.global_batch_size:
            raise ValueError(f'loader.global_batch_size must equal actual batch {effective}.')
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Mirror the native MDLM fault-tolerant sampler, including genuine MMD
        # resume offsets. Unlike the old base implementation, allow 0 workers.
        connector = self.trainer._accelerator_connector
        distributed = connector.use_distributed_sampler and connector.is_distributed
        sampler_cls = (dataloader.FaultTolerantDistributedSampler if distributed
                       else dataloader.RandomFaultTolerantSampler)
        loaders = []
        for loader in self.trainer.fit_loop._combined_loader.flattened:
            kwargs = {'shuffle': loader.sampler.shuffle} if hasattr(loader.sampler, 'shuffle') else {}
            sampler = sampler_cls(loader.dataset, **kwargs)
            if (distributed and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                sampler.load_state_dict({'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.config.loader.batch_size})
            loaders.append(torch.utils.data.DataLoader(
                loader.dataset, batch_size=self.config.loader.batch_size,
                num_workers=self.config.loader.num_workers,
                pin_memory=self.config.loader.pin_memory, sampler=sampler,
                shuffle=False, persistent_workers=self.config.loader.num_workers > 0))
        self.trainer.fit_loop._combined_loader.flattened = loaders
        self._mmd_metric_sum = None
        self._mmd_metric_step = self.global_step
        if self.config.eval.get('validate_before_train', False) and self.global_step == 0:
            from sdtt_eval import log_validation_gsm8k
            self._eval_mode()
            try:
                log_validation_gsm8k(self, initial=True)
            finally:
                self._train_mode()

    def _sample_student_outputs(self, corrupted, sigma, original, selected):
        """Retain only grouped sample IDs/log-probabilities from physical forwards.

        This mirrors the paper run's physical batches of eight and non-reentrant
        activation checkpointing. Reassemble the complete loader batch before
        scoring rewards or normalizing the objective; chunks are not minibatches.
        """
        cfg = self.config.mmd
        group_size = cfg.group_size
        if type(group_size) is not int or group_size < 1:
            raise ValueError('mmd.group_size must be a positive integer.')
        physical_batch = cfg.get('student_forward_batch_size') or original.shape[0]
        use_checkpoint = cfg.get('student_activation_checkpointing', False)

        def sample_chunk(noisy, noise, clean, written):
            probabilities = self(noisy, noise).exp()
            with torch.no_grad():
                sampled = torch.multinomial(
                    probabilities.detach().reshape(-1, probabilities.shape[-1]).float(),
                    group_size, replacement=True)
                samples = sampled.T.reshape(group_size, *clean.shape)
                samples = torch.where(written.unsqueeze(0), samples, clean.unsqueeze(0))
            logp = probabilities.clamp_min(1e-12).log()
            sampled_logp = logp.gather(2, samples.permute(1, 2, 0)).permute(2, 0, 1)
            return sampled_logp, samples

        logp_chunks, sample_chunks = [], []
        for start in range(0, original.shape[0], physical_batch):
            selection = slice(start, start + physical_batch)
            inputs = (corrupted[selection], sigma[selection], original[selection], selected[selection])
            if use_checkpoint:
                logp, samples = activation_checkpoint(
                    sample_chunk, *inputs, use_reentrant=False, preserve_rng_state=True)
            else:
                logp, samples = sample_chunk(*inputs)
            logp_chunks.append(logp)
            sample_chunks.append(samples)
        return torch.cat(logp_chunks, dim=1), torch.cat(sample_chunks, dim=1)

    def training_step(self, batch, batch_idx):
        cfg = self.config.mmd
        original, valid = batch['input_ids'], batch['attention_mask'].bool()
        t = self._sample_t(original.shape[0], batch_idx % self.trainer.accumulate_grad_batches)
        _, alpha = self.noise(t)
        alpha = alpha.unsqueeze(-1)
        corrupted = self.q_xt(original, alpha, valid_tokens=valid)
        selected = corrupted.eq(self.mask_index) & answer_content_mask(original, valid, self.tokenizer)
        sampled_logp, samples = self._sample_student_outputs(
            corrupted, self._sigma_from_alphat(alpha), original, selected)
        loss, stats = reinforce_loss_from_samples(
            sampled_logp, original, selected, samples, self.reward,
            group_size=cfg.group_size, draws_per_candidate=cfg.draws_per_candidate,
            average_candidates=cfg.average_candidates)
        self.log('student_loss', loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/mmd_loss', loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        # Pool sums/counts over actual accumulation and DDP ranks. Reward metrics
        # count only examples with a defined reward; advantages include zeros.
        batch_count = stats['reward_count'].new_tensor(original.shape[0])
        counts = torch.stack([stats['reward_count'], batch_count, batch_count]).float()
        values = torch.stack([stats['mmd/reward'], stats['mmd/advantage_absmean'],
                              stats['mmd/reward_spread']]).detach().float()
        totals = torch.cat([values * counts, counts])
        self._mmd_metric_sum = totals if self._mmd_metric_sum is None else self._mmd_metric_sum + totals
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step == self._mmd_metric_step:
            return
        totals = self.trainer.strategy.reduce(self._mmd_metric_sum, reduce_op='sum')
        values = totals[:3] / totals[3:].clamp_min(1)
        self.log_dict({'mmd/reward': values[0], 'mmd/advantage_absmean': values[1],
                       'mmd/reward_spread': values[2]}, on_step=True, on_epoch=False, sync_dist=False)
        self._mmd_metric_sum = None
        self._mmd_metric_step = self.global_step

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        return log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1) * dalpha_t / (1 - alpha_t)

    def on_validation_epoch_end(self):
        from sdtt_eval import log_validation_gsm8k
        try:
            log_validation_gsm8k(self)
        finally:
            super().on_validation_epoch_end()

    def on_train_end(self):
        output = Path(self.config.checkpointing.save_dir) / 'checkpoints' / 'last.ckpt'
        output.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(output))
        from sdtt_eval import log_validation_gsm8k
        self._eval_mode()
        try:
            log_validation_gsm8k(self, final=True)
        finally:
            self._train_mode()
            if self.config.checkpointing.get('snapshot', False):
                import nirvana_snapshot
                nirvana_snapshot.save(str(self.config.checkpointing.save_dir), enabled=True,
                    namespace=self.config.checkpointing.get('snapshot_namespace'),
                    reason=f'final evaluation step={self.global_step}')


# Short entry-point name, matching the repository's other algorithm classes.
MMD = MMDFinetune
