"""Optional two-rank CPU/Gloo smoke test; no GPU, downloads, or tracker.

Run separately: conda run -n mdlm python scripts/test_didi_ddp.py
This exercises the released manual two-optimizer schedule and distributed group
reward normalization with the same tiny network as test_didi_runtime.py.
"""

import copy
from datetime import timedelta
import tempfile

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import open_dict
import torch
from torch.utils.data import DataLoader

from test_didi_runtime import DiDiRuntimeTests, ToyDataset


class CheckDistributedState(L.Callback):
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student

    def on_train_end(self, trainer, pl_module):
        assert trainer.world_size == 2
        assert trainer.global_step == 4, trainer.global_step
        assert pl_module.global_steps == 2, pl_module.global_steps
        assert any(not torch.equal(value, self.student[key])
                   for key, value in pl_module.backbone.state_dict().items())
        for key, value in pl_module.teacher.state_dict().items():
            torch.testing.assert_close(value, self.teacher[key], rtol=0, atol=0)
        assert all(p.grad is None for p in pl_module.teacher.parameters())
        assert all(p.grad is None for p in pl_module.student_ema.parameters())
        assert 'reward/mean' in trainer.logged_metrics
        assert all(torch.isfinite(torch.as_tensor(value)).all()
                   for value in trainer.logged_metrics.values())
        parameters = torch.cat([p.detach().reshape(-1) for p in pl_module.parameters()
                                if p.requires_grad])
        gathered = [torch.empty_like(parameters) for _ in range(2)]
        torch.distributed.all_gather(gathered, parameters)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)

        # Exercise monitoring's sample capture and rank-zero gather/barrier
        # around the original guided sampler, without invoking a reference LM.
        from didi_repro.monitor import attach_monitor
        with open_dict(pl_module.config):
            pl_module.config.didi_monitor = {
                'enabled': True, 'gen_ppl': False, 'gm': False,
                'max_reference_samples': 8, 'gm_batch_size': 1,
            }
        pl_module.sampler = 'guided'
        monitor = attach_monitor(pl_module, pl_module.config)
        monitor.on_validation_start(trainer, pl_module)
        with torch.inference_mode():
            samples = pl_module.generate_samples(2, num_steps=4)
        assert samples.shape == (2, 12)
        assert len(monitor.samples) == 1
        monitor.on_validation_end(trainer, pl_module)
        assert not monitor.samples
        print(f'Rank {trainer.global_rank}: two updates and guided-monitor gather passed.', flush=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(73)
    DiDiRuntimeTests.setUpClass()
    helpers = DiDiRuntimeTests()
    model = helpers.model()
    helpers.initialize(model)
    checker = CheckDistributedState(copy.deepcopy(model.teacher.state_dict()),
                                    copy.deepcopy(model.backbone.state_dict()))
    with tempfile.TemporaryDirectory(prefix='didi-ddp-test-') as directory:
        model.config.checkpointing.save_dir = directory
        trainer = L.Trainer(
            accelerator='cpu', devices=2, precision='32-true', max_steps=4,
            strategy=DDPStrategy(find_unused_parameters=True, start_method='fork',
                                 process_group_backend='gloo', timeout=timedelta(seconds=60)),
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0,
            default_root_dir=directory, gradient_clip_val=0.0, callbacks=[checker],
        )
        trainer.fit(model, train_dataloaders=DataLoader(ToyDataset(), batch_size=1))
    print('CPU/Gloo two-rank DiDi smoke passed.', flush=True)


if __name__ == '__main__':
    main()
