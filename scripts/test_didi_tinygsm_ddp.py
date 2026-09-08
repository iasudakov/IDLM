"""Two-rank CPU/Gloo TinyGSM smoke; no GPU, downloads, or external logger.

Run separately in the DiDi environment:
  python scripts/test_didi_tinygsm_ddp.py

Uses the tiny CPU fixture network but the real conditional training adapter,
released optimizer-step budget semantics, distributed reward normalization,
and both released guided samplers. All checkpoints are temporary.
"""

import copy
from datetime import timedelta
import tempfile
import types

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import open_dict
import torch
from torch.utils.data import DataLoader, Dataset

import test_didi_tinygsm_runtime as fixtures


class UnevenPrefixDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        tokens = ((torch.arange(12) + index) % 7).long()
        prefix = 2 + index % 4
        target = torch.arange(12) >= prefix
        return {'input_ids': tokens, 'attention_mask': target.long(),
                'prompt_len': torch.tensor(prefix)}


class CheckConditionalDistributedState(L.Callback):
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.batch_checks = 0

    def on_fit_start(self, trainer, pl_module):
        assert trainer.world_size == 2
        self.original_corrupt = pl_module._corrupt_with_mask_ratio
        self.original_backward = pl_module._get_backward

        def corrupt(instance, x0, alpha_t):
            del instance
            result = self.original_corrupt(x0, alpha_t)
            assert torch.equal(result[0][~self.target], self.fixed[~self.target])
            assert not result[1][~self.target].any()
            self.corruption_masks.append(result[1].clone())
            return result

        def backward(instance, x_prev, t_hi, t_lo):
            del instance
            result = self.original_backward(x_prev, t_hi, t_lo)
            assert torch.equal(x_prev[~self.target], self.fixed[~self.target])
            assert torch.equal(result[0][~self.target], self.fixed[~self.target])
            assert not result[3][~self.target].any()
            self.backward_checks += 1
            return result

        def discriminator_hook(module, inputs, output):
            del module
            assert torch.equal(inputs[0][~self.target], self.fixed[~self.target])
            if output.requires_grad:
                output.retain_grad()
                self.discriminator_outputs.append(output)

        pl_module._corrupt_with_mask_ratio = types.MethodType(corrupt, pl_module)
        pl_module._get_backward = types.MethodType(backward, pl_module)
        self.discriminator_hook = pl_module.discriminator.register_forward_hook(discriminator_hook)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, batch_idx
        group = pl_module.num_samples_per_prompt
        self.fixed = batch['input_ids'].repeat_interleave(group, dim=0)
        self.target = batch['attention_mask'].bool().repeat_interleave(group, dim=0)
        self.corruption_masks = []
        self.discriminator_outputs = []
        self.backward_checks = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        assert self.backward_checks == 2
        assert len(self.corruption_masks) == 2
        assert len(self.discriminator_outputs) == 2
        for output in self.discriminator_outputs:
            assert output.grad is not None
            assert (output.grad.squeeze(-1)[~self.target] == 0).all()
        fake_mask, real_mask = self.corruption_masks
        real_logits, fake_logits = self.discriminator_outputs
        expected_accuracy = (
            ((real_logits.detach().squeeze(-1) > 0) * real_mask).sum()
            + ((fake_logits.detach().squeeze(-1) < 0) * fake_mask).sum()
        ).float() / (real_mask.sum() + fake_mask.sum()).clamp_min(1)
        torch.distributed.all_reduce(expected_accuracy)
        expected_accuracy /= trainer.world_size
        torch.testing.assert_close(
            trainer.logged_metrics['monitor/discriminator_acc'], expected_accuracy,
            rtol=1e-6, atol=1e-7)
        assert pl_module._didi_target_state is None
        assert all(torch.isfinite(torch.as_tensor(value)).all()
                   for value in trainer.logged_metrics.values())
        self.batch_checks += 1

    def on_train_end(self, trainer, pl_module):
        self.discriminator_hook.remove()
        pl_module._corrupt_with_mask_ratio = self.original_corrupt
        pl_module._get_backward = self.original_backward
        assert trainer.global_step == 4, trainer.global_step
        assert pl_module.global_steps == 2, pl_module.global_steps
        assert self.batch_checks == 2
        assert pl_module._didi_iterations is None
        assert any(not torch.equal(value, self.student[key])
                   for key, value in pl_module.backbone.state_dict().items())
        for key, value in pl_module.teacher.state_dict().items():
            torch.testing.assert_close(value, self.teacher[key], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in pl_module.teacher.parameters())
        assert all(parameter.grad is None for parameter in pl_module.student_ema.parameters())
        assert 'reward/mean' in trainer.logged_metrics

        for module in (pl_module.backbone, pl_module.student_ema):
            flattened = torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])
            gathered = [torch.empty_like(flattened) for _ in range(2)]
            torch.distributed.all_gather(gathered, flattened)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)

        # Known questions only; supplied junk beyond the row's own prefix is
        # ignored, rather than revealing gold answer/padding boundaries.
        prefixes = torch.tensor([[1, 2, 3, 6, 6], [2, 3, 4, 5, 6]])
        lengths = torch.tensor([3, 5])
        before_disc_grads = [None if parameter.grad is None else parameter.grad.clone()
                             for parameter in pl_module.discriminator.parameters()]
        for source in ('main', 'inference'):
            pl_module._didi_sampler_source = source
            pl_module.sampler = 'guided'
            with torch.inference_mode():
                samples = pl_module.generate_samples(
                    2, num_steps=4, prefix_tokens=prefixes, prefix_lengths=lengths)
            assert samples.shape == (2, 12)
            for row, length in enumerate(lengths):
                assert torch.equal(samples[row, :length], prefixes[row, :length])
            assert pl_module._didi_target_state is None
            assert not pl_module.discriminator._forward_hooks
        for parameter, before in zip(pl_module.discriminator.parameters(), before_disc_grads):
            if before is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
        torch.distributed.barrier()
        print(f'Rank {trainer.global_rank}: conditional training, target metrics, '
              'frozen teacher, synchronized student/EMA and guided sampling passed.', flush=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(73)
    fixtures.TinyGSMRuntimeTests.setUpClass()
    helpers = fixtures.TinyGSMRuntimeTests()
    config = helpers.config(source='inference', sampler='guided')
    with open_dict(config):
        config.didi_training = {'iterations': None}
    config.trainer.max_steps = 4
    config.trainer.devices = 2
    config.loader.batch_size = 2
    config.loader.global_batch_size = 4
    model = helpers.fixture.model(config)
    helpers.fixture.initialize(model)
    checker = CheckConditionalDistributedState(
        copy.deepcopy(model.teacher.state_dict()), copy.deepcopy(model.backbone.state_dict()))
    try:
        with tempfile.TemporaryDirectory(prefix='didi-tinygsm-ddp-test-') as directory:
            model.config.checkpointing.save_dir = directory
            trainer = L.Trainer(
                accelerator='cpu', devices=2, precision='32-true', max_steps=4,
                strategy=DDPStrategy(find_unused_parameters=True, start_method='fork',
                                     process_group_backend='gloo', timeout=timedelta(seconds=60)),
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0,
                default_root_dir=directory, gradient_clip_val=0.0, callbacks=[checker],
            )
            trainer.fit(model, train_dataloaders=DataLoader(UnevenPrefixDataset(), batch_size=2))
    finally:
        fixtures.TinyGSMRuntimeTests.tearDownClass()
    print('CPU/Gloo two-rank conditional TinyGSM DiDi smoke passed.', flush=True)


if __name__ == '__main__':
    main()
