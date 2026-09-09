"""Prompt-conditioned TinyGSM extension of the repaired, released DiDi recipe.

This is a task adaptation, not a claim that DiDi's paper used TinyGSM. The
two-step trajectory, optimizer updates, rewards, KL/entropy terms, schedules,
and normalization stay in the released implementation. Only the positions
allowed to change or contribute to its existing masks are restricted.
"""

from contextlib import contextmanager
import inspect

import torch

from .runtime import TRAINING_REPAIRS, repaired_function


def install_tinygsm_runtime(upstream_main):
    """Install after ``load_upstream()``, and only for TinyGSM entry points."""
    base = upstream_main.algo.DiDiInstruct
    if getattr(base, '_didi_tinygsm_installed', False):
        return base
    if not getattr(base, '_didi_runtime_installed', False):
        raise RuntimeError('Install the repaired DiDi runtime before TinyGSM.')
    # Empty Bernoulli masks occur naturally on short conditional responses.
    # The released mean over an empty clipped-fraction diagnostic is NaN;
    # leave the actual reward/advantage/objective equations unchanged.
    original = base.__bases__[0]
    rewards = repaired_function(inspect.unwrap(original._get_rewards), ((
        'reward_fraction_clipped = is_clipped[mask_fake].mean()',
        'reward_fraction_clipped = is_clipped[mask_fake].sum() / mask_fake.sum().clamp_min(1)'),))
    training = repaired_function(original.training_step, TRAINING_REPAIRS + ((
        'disc_acc = (real_preds.sum() + fake_preds.sum()) / (real_logits.numel() + fake_logits.numel())',
        'disc_acc = ((real_preds.squeeze(-1) * mask_real).sum() + '
        '(fake_preds.squeeze(-1) * mask_fake).sum()) / '
        '(mask_real.sum() + mask_fake.sum()).clamp_min(1)'),))

    class TinyGSMDiDi(base):
        _didi_tinygsm_installed = True
        _didi_training_step_impl = training

        def __init__(self, config, tokenizer):
            if config.data.get('train') != 'tiny_gsm':
                raise ValueError('TinyGSMDiDi requires data=tiny-gsm.')
            if config.data.get('wrap', False):
                raise ValueError('Conditional TinyGSM requires data.wrap=false.')
            if config.data.get('train_on_prompt', False):
                raise ValueError('Conditional TinyGSM requires data.train_on_prompt=false.')
            if config.sampling.predictor not in ('ancestral', 'ancestral_cache', 'guided'):
                raise ValueError('TinyGSM DiDi supports ancestral, ancestral_cache or guided sampling.')
            self._didi_target_state = None
            super().__init__(config, tokenizer)
            self._didi_effective_hyperparameters['conditional_task'] = {
                'dataset': 'tiny_gsm',
                'training_target': 'attention_mask (answer and optionally padding)',
                'train_on_pad': bool(config.data.get('train_on_pad', True)),
                'generation_target': 'entire suffix after the supplied prompt',
                'reward_positions': 'corrupted training-target positions',
                'rerank_positions': 'entire generated suffix, excluding prompt',
            }

        def _conditional_checkpoint_metadata(self):
            """Persist task semantics independently of the optimizer protocol.

            Dataset/cache/output paths are intentionally absent so checkpoints
            can move between machines. Tokenization and masking semantics may
            not change silently on resume or evaluation.
            """
            return {
                'schema': 'didi-tinygsm-conditional',
                'version': 1,
                'data.train': str(self.config.data.train),
                'data.tokenizer_name_or_path': str(self.config.data.tokenizer_name_or_path),
                'data.separator': str(self.config.data.get('separator', '\n')),
                'data.train_on_prompt': bool(self.config.data.get('train_on_prompt', False)),
                'data.train_on_pad': bool(self.config.data.get('train_on_pad', True)),
                'data.wrap': bool(self.config.data.get('wrap', False)),
                'model.length': int(self.num_tokens),
                'target_mask': 'contiguous-answer-and-optional-padding-v1',
                'generation_target': 'entire-suffix-after-prompt-v1',
                'special_token_ids': {
                    'bos': getattr(self.tokenizer, 'bos_token_id', None),
                    'eos': getattr(self.tokenizer, 'eos_token_id', None),
                    'pad': getattr(self.tokenizer, 'pad_token_id', None),
                    'mask': int(self.mask_index),
                },
            }

        def initialize_teacher_checkpoint(self, checkpoint, source_path=None):
            # Validate the entire native EMA before the common initializer can
            # alter any model copy. Keep raw training/teacher/discriminator
            # weights; only the evaluation EMA gets the saved MDLM average.
            from .tinygsm_initialization import prepare_student_ema
            ema_state, metadata = prepare_student_ema(self, checkpoint)
            super().initialize_teacher_checkpoint(checkpoint, source_path)
            self.student_ema.load_state_dict(ema_state, strict=True)
            metadata['checkpoint'] = dict(self._didi_teacher_metadata)
            self._didi_student_metadata = metadata

        def on_save_checkpoint(self, checkpoint):
            super().on_save_checkpoint(checkpoint)
            checkpoint['didi_tinygsm'] = self._conditional_checkpoint_metadata()

        def on_load_checkpoint(self, checkpoint):
            expected = self._conditional_checkpoint_metadata()
            actual = checkpoint.get('didi_tinygsm')
            if actual is None:
                raise ValueError('TinyGSM DiDi checkpoint is missing conditional-task metadata. '
                                 'An OWT DiDi checkpoint cannot resume a TinyGSM run. '
                                 'For a fresh run initialize the native TinyGSM MDLM teacher '
                                 'through training.finetune_path instead.')
            if actual != expected:
                if isinstance(actual, dict):
                    differing = sorted(key for key in set(actual) | set(expected)
                                       if actual.get(key) != expected.get(key))
                else:
                    differing = ['metadata schema']
                raise ValueError('TinyGSM checkpoint conditional-task mismatch: '
                                 + ', '.join(differing)
                                 + '. Preserve its tokenizer, separator, sequence length '
                                 'and prompt/padding mask settings.')
            return super().on_load_checkpoint(checkpoint)

        def _validate_targets(self, tokens, mask):
            if tokens.ndim != 2 or mask.shape != tokens.shape:
                raise ValueError('TinyGSM input_ids and attention_mask must have equal [B,L] shapes.')
            if tokens.shape[1] != self.num_tokens:
                raise ValueError('TinyGSM input width must equal model.length.')
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError('TinyGSM attention_mask must be a binary loss/target mask.')
            target = mask.bool()
            if not bool(target.any(dim=1).all()) or bool(target[:, 0].any()):
                raise ValueError('Every TinyGSM row needs a fixed prompt and at least one target token.')
            # The prefix must be fixed, then one contiguous answer/padding span.
            changes = target[:, 1:].to(torch.int8) - target[:, :-1].to(torch.int8)
            if bool(((changes == 1).sum(dim=1) != 1).any()):
                raise ValueError('TinyGSM target masks must contain one contiguous answer span.')
            if bool((tokens[~target] == self.mask_index).any()):
                raise ValueError('Fixed TinyGSM prompt/padding cannot contain MASK tokens.')
            return target

        @contextmanager
        def _target_context(self, tokens, target):
            previous = self._didi_target_state
            self._didi_target_state = (tokens, target)
            try:
                yield
            finally:
                self._didi_target_state = previous

        def _target_state(self, shape):
            if self._didi_target_state is None:
                raise RuntimeError('TinyGSM diffusion needs an explicit prompt/target context.')
            tokens, target = self._didi_target_state
            if tuple(shape) != tuple(tokens.shape):
                raise ValueError('TinyGSM target context has the wrong batch or sequence shape.')
            return tokens, target

        def _restore_fixed(self, tokens):
            fixed, target = self._target_state(tokens.shape)
            return torch.where(target, tokens, fixed)

        def training_step(self, batch, batch_idx):
            tokens = batch['input_ids']
            target = self._validate_targets(tokens, batch['attention_mask'])
            if 'prompt_len' in batch:
                lengths = target.long().argmax(dim=1)
                expected = torch.as_tensor(batch['prompt_len'], device=tokens.device).reshape(-1)
                if expected.shape != lengths.shape or not torch.equal(expected, lengths):
                    raise ValueError('TinyGSM prompt_len disagrees with its answer-only loss mask.')
            group = self.num_samples_per_prompt
            with self._target_context(tokens.repeat_interleave(group, dim=0),
                                      target.repeat_interleave(group, dim=0)):
                return super().training_step(batch, batch_idx)

        def _get_masked_input(self, batch_size):
            tokens = super()._get_masked_input(batch_size)
            return self._restore_fixed(tokens)

        def _corrupt_with_mask_ratio(self, x0, alpha_t):
            # Draw precisely the upstream noise, then disallow prompt/padding
            # corruption. Do not force a masked token on short answers: that
            # would alter the released Bernoulli corruption distribution.
            zt, mask = super()._corrupt_with_mask_ratio(x0, alpha_t)
            _, target = self._target_state(x0.shape)
            return self._restore_fixed(zt), mask & target

        def _get_rewards(self, *args, **kwargs):
            result = rewards(self, *args, **kwargs)
            final, mean, std, clipped, advantage = result
            # Same diagnostics as the repaired OWT runtime.
            self.log_dict({
                'reward/mean': mean,
                'reward/std': std,
                'reward/clipped_fraction': clipped,
                'reward/advantage_abs_mean': advantage.abs().mean(),
                'reward/advantage_std': advantage.std(),
                'reward/normalized_abs_mean': final.abs().mean(),
            }, on_step=True, on_epoch=False, sync_dist=True)
            return result

        def _loss(self, x0, valid_tokens, current_accumulation_step=None,
                  train_mode=False):
            # Validation NLL otherwise masks the loss but still corrupts the
            # question, which is not conditional answer likelihood.
            target = self._validate_targets(x0, valid_tokens)
            with self._target_context(x0, target):
                return super()._loss(x0, valid_tokens, current_accumulation_step, train_mode)

        def q_xt(self, x, alpha_t):
            return self._restore_fixed(super().q_xt(x, alpha_t))

        def prior_sample(self, *batch_dims):
            return self._restore_fixed(super().prior_sample(*batch_dims))

        def _denoiser_update(self, x, t):
            return self._restore_fixed(super()._denoiser_update(x, t))

        def _conditional_discriminator_output(self, module, inputs, output):
            del module, inputs
            # Both released guided samplers sum ALL discriminator outputs in
            # candidate reranking. Prompt hidden states depend on the answer,
            # so their logits are not a harmless candidate-independent offset.
            # Zero only those direct rewards, preserving context attention and
            # all original sum/temperature/candidate-selection equations.
            if self._didi_target_state is None:
                raise RuntimeError('Conditional discriminator scoring needs a target context.')
            _, target = self._didi_target_state
            if output.ndim != 3 or output.shape[1:] != (target.shape[1], 1):
                raise ValueError('Expected per-token DiDi discriminator logits [B,L,1].')
            if output.shape[0] % target.shape[0]:
                raise ValueError('Unexpected conditional discriminator candidate batch size.')
            target = target.repeat_interleave(output.shape[0] // target.shape[0], dim=0)
            return output * target.unsqueeze(-1).to(output.dtype)

        def generate_samples(self, num_samples, num_steps=None, eps=1e-5,
                             prefix_tokens=None, prefix_lengths=None):
            if prefix_tokens is None or prefix_lengths is None:
                raise ValueError('TinyGSM generation requires prefix_tokens and prefix_lengths; '
                                 'unconditional OWT sampling is not a TinyGSM evaluation.')
            if self._didi_in_training_step:
                raise RuntimeError('TinyGSM evaluation cannot run inside a training step.')
            # Clone outside inference mode; guided evaluation needs ordinary
            # tensors even when called from a Lightning inference-mode callback.
            with torch.inference_mode(False):
                prefixes = torch.as_tensor(prefix_tokens, device=self.device).clone()
                lengths = torch.as_tensor(prefix_lengths, device=self.device).clone()
                if prefixes.ndim != 2 or prefixes.shape[0] != num_samples:
                    raise ValueError('prefix_tokens must have shape [num_samples, prefix_width].')
                if lengths.ndim != 1 or lengths.shape[0] != num_samples:
                    raise ValueError('prefix_lengths must have shape [num_samples].')
                if lengths.is_floating_point() or prefixes.is_floating_point():
                    raise ValueError('Prefix token IDs and lengths must be integers.')
                if (bool((lengths <= 0).any()) or bool((lengths >= self.num_tokens).any())
                        or bool((lengths > prefixes.shape[1]).any())):
                    raise ValueError('Each prefix must fit and leave at least one generated position.')
                target = torch.arange(self.num_tokens, device=self.device)[None, :] >= lengths[:, None]
                fixed = torch.full((num_samples, self.num_tokens), self.mask_index,
                                   dtype=torch.long, device=self.device)
                width = min(prefixes.shape[1], self.num_tokens)
                fixed[:, :width] = prefixes[:, :width].long()
                # Discard every supplied token beyond its own prefix length:
                # neither answer values nor gold answer/padding boundaries may
                # leak into conditional generation.
                fixed.masked_fill_(target, self.mask_index)
                if bool(((fixed[~target] < 0) | (fixed[~target] >= self.vocab_size)).any()):
                    raise ValueError('Prefix contains token IDs outside the model vocabulary.')
                if bool((fixed[~target] == self.mask_index).any()):
                    raise ValueError('Fixed prefixes cannot contain MASK tokens.')
                with self._target_context(fixed, target):
                    hook = self.discriminator.register_forward_hook(self._conditional_discriminator_output)
                    try:
                        samples = super().generate_samples(num_samples, num_steps, eps)
                        return self._restore_fixed(samples)
                    finally:
                        hook.remove()

    TinyGSMDiDi.__name__ = 'TinyGSMDiDi'
    TinyGSMDiDi.__qualname__ = 'TinyGSMDiDi'
    globals()['TinyGSMDiDi'] = TinyGSMDiDi
    upstream_main.algo.DiDiInstruct = TinyGSMDiDi
    return TinyGSMDiDi
