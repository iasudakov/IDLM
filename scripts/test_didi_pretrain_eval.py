"""CPU-only pre-training validation lifecycle tests using the upstream loss.

    conda run -n mdlm python scripts/test_didi_pretrain_eval.py

The DiDi objective and validation metrics are real upstream code. Only the
network and tokenizer are tiny local substitutes; no remote tracker/model is
contacted and no GPU is used.
"""

import copy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader

from didi_repro.monitor import attach_monitor
from didi_repro.pretrain_eval import maybe_evaluate_before_training
from didi_repro.runtime import UPSTREAM_ROOT, load_upstream
from underdeep_logger import UnderdeepLogger
from test_didi_runtime import ToyBackbone, ToyDataset, ToyTokenizer


class MemoryRun:
    uid = 'local-test-only'

    def __init__(self):
        self.metrics = []
        self.finish_calls = 0

    def log(self, values, step=None):
        if self.finish_calls:
            raise AssertionError('cannot log to a finalized run')
        self.metrics.append((dict(values), step))

    def finish(self):
        self.finish_calls += 1


class MemoryLogger(UnderdeepLogger):
    """Exercise actual Underdeep finalize semantics without its network SDK."""

    def __init__(self, directory):
        super().__init__(save_dir=directory)
        self._initialized = True
        self._run = MemoryRun()
        self.tables = []

    def log_table(self, key, columns, data, step=None):
        self.tables.append((key, data, step))


class OrderCallback(L.Callback):
    def __init__(self):
        self.events = []
        self.validation_weights = []

    def on_validation_start(self, trainer, pl_module):
        self.events.append(('validation', trainer.global_step))
        self.validation_weights.append(copy.deepcopy(pl_module.backbone.state_dict()))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.events.append(('training', trainer.global_step))


class TestPretrainEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.upstream = load_upstream()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='didi-pretrain-test-')
        self.addCleanup(self.directory.cleanup)

    def config(self, enabled=True):
        with hydra.initialize_config_dir(version_base=None,
                                        config_dir=str(UPSTREAM_ROOT / 'configs')):
            cfg = hydra.compose(config_name='config')
        cfg.model.hidden_size = 8
        cfg.model.length = 12
        cfg.loader.num_workers = 0
        cfg.loader.eval_batch_size = 2
        cfg.trainer.devices = 1
        cfg.trainer.accelerator = 'cpu'
        cfg.trainer.max_steps = 4
        cfg.trainer.limit_val_batches = 1
        cfg.optim.lr = .01
        cfg.algo.discriminator_lr = .01
        cfg.algo.lr_scheduler.warmup_steps = 1
        cfg.algo.save_after_n_steps = 1000000
        cfg.algo.print_every = 0
        cfg.sampling.steps = 2
        cfg.sampling.num_sample_batches = 1
        cfg.sampling.noise_removal = 'none'
        cfg.checkpointing.save_dir = self.directory.name
        with open_dict(cfg):
            cfg.didi_eval = {'before_training': enabled}
            cfg.didi_monitor = {'enabled': True, 'gen_ppl': False, 'gm': False}
        return cfg

    def model(self, cfg=None):
        cfg = cfg or self.config()
        reference_tokenizer = SimpleNamespace(pad_token='pad', pad_token_id=0)
        base = self.upstream.algo.trainer_base
        with mock.patch.object(base.models.dit, 'DIT', ToyBackbone), \
             mock.patch.object(base.metrics.transformers.AutoTokenizer, 'from_pretrained',
                               return_value=reference_tokenizer):
            model = self.upstream.algo.DiDiInstruct(cfg, ToyTokenizer())
        teacher = {f'backbone.{name}': tensor.detach().clone() + .03
                   for name, tensor in model.backbone.state_dict().items()}
        model.initialize_teacher_checkpoint({'state_dict': teacher})
        return model

    def trainer(self, model, max_steps=2, limit_val_batches=1):
        logger = MemoryLogger(self.directory.name)
        order = OrderCallback()
        callback = attach_monitor(model, model.config, upstream_revision='test')
        trainer = L.Trainer(
            accelerator='cpu', devices=1, precision='32-true', max_steps=max_steps,
            logger=logger, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=limit_val_batches,
            num_sanity_val_steps=0, val_check_interval=8, log_every_n_steps=1,
            default_root_dir=self.directory.name, gradient_clip_val=0.0,
            callbacks=[order, callback],
        )
        return trainer, logger, order

    def loader(self):
        return DataLoader(ToyDataset(), batch_size=1, shuffle=False)

    def assert_same_state(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_disabled_is_exact_noop(self):
        model = SimpleNamespace(config={'didi_eval': {'before_training': False}})
        trainer = mock.Mock()
        self.assertIsNone(maybe_evaluate_before_training(trainer, model, None))
        trainer.validate.assert_not_called()

    def test_zero_validation_limit_and_missing_loader_rejected(self):
        model = self.model()
        trainer, _, _ = self.trainer(model, limit_val_batches=0)
        with self.assertRaisesRegex(ValueError, 'limit_val_batches'):
            maybe_evaluate_before_training(trainer, model, self.loader())
        trainer.limit_val_batches = 1
        with self.assertRaisesRegex(ValueError, 'validation dataloader'):
            maybe_evaluate_before_training(trainer, model, None)

    def test_original_validation_precedes_update_and_keeps_run_open(self):
        torch.manual_seed(24)
        model = self.model()
        trainer, logger, order = self.trainer(model)
        state = copy.deepcopy(model.state_dict())
        flags = [module.training for module in model.modules()]
        rng = torch.get_rng_state()
        result = maybe_evaluate_before_training(trainer, model, self.loader())
        self.assertIsInstance(result, list)
        self.assertEqual(order.events, [('validation', 0)])
        self.assert_same_state(model.state_dict(), state)
        self.assertEqual([module.training for module in model.modules()], flags)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(logger._run.finish_calls, 0)
        self.assertFalse(logger.defer_finalize)
        self.assertFalse(list(Path(self.directory.name).glob('*.ckpt')))
        values = [(data, step) for data, step in logger._run.metrics if 'val/nll' in data]
        self.assertTrue(values)
        self.assertEqual(values[-1][1], 0)
        self.assertTrue(any('val/sample_entropy' in data for data, _ in logger._run.metrics))
        trainer.fit(model, train_dataloaders=self.loader())
        self.assertEqual(order.events[1], ('training', 0))
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(logger._run.finish_calls, 1)

    def test_pre_evaluation_keeps_subsequent_cpu_training_trajectory(self):
        states = []
        for enabled in (False, True):
            random.seed(25)
            np.random.seed(25)
            torch.manual_seed(25)
            model = self.model(self.config(enabled=enabled))
            trainer, _, _ = self.trainer(model)
            maybe_evaluate_before_training(trainer, model, self.loader())
            trainer.fit(model, train_dataloaders=self.loader())
            states.append(copy.deepcopy(model.state_dict()))
        self.assert_same_state(states[0], states[1])

    def test_resume_evaluates_restored_weights_at_checkpoint_step(self):
        model = self.model()
        trainer, _, _ = self.trainer(model)
        trainer.fit(model, train_dataloaders=self.loader())
        checkpoint = str(Path(self.directory.name) / 'resume.ckpt')
        trainer.save_checkpoint(checkpoint)
        saved = copy.deepcopy(model.backbone.state_dict())
        resumed = self.model()
        continuation, logger, order = self.trainer(resumed, max_steps=4)
        maybe_evaluate_before_training(continuation, resumed, self.loader(), ckpt_path=checkpoint)
        self.assert_same_state(saved, order.validation_weights[0])
        values = [(data, step) for data, step in logger._run.metrics if 'val/nll' in data]
        self.assertTrue(values)
        self.assertEqual(values[-1][1], 2)
        self.assertEqual(logger._run.finish_calls, 0)
        self.assertEqual(resumed.global_steps, 1)
        continuation.fit(resumed, train_dataloaders=self.loader(), ckpt_path=checkpoint)
        self.assertEqual(continuation.global_step, 4)
        self.assertEqual(resumed.global_steps, 2)
        self.assertEqual(logger._run.finish_calls, 1)

    def test_exception_restores_rng_modes_and_temporary_methods(self):
        model = self.model()
        trainer, logger, _ = self.trainer(model)
        original_save = trainer.save_checkpoint
        original_load = model.on_load_checkpoint
        original_log = logger.log_metrics
        flags = [module.training for module in model.modules()]
        torch_state, python_state, numpy_state = (
            torch.get_rng_state(), random.getstate(), np.random.get_state())

        def failed_validation(*args, **kwargs):
            model.eval()
            torch.rand(3)
            random.random()
            np.random.random()
            raise RuntimeError('simulated validation failure')

        with mock.patch.object(trainer, 'validate', side_effect=failed_validation):
            with self.assertRaisesRegex(RuntimeError, 'simulated validation failure'):
                maybe_evaluate_before_training(trainer, model, self.loader())
        self.assertFalse(logger.defer_finalize)
        self.assertEqual(trainer.save_checkpoint, original_save)
        self.assertEqual(model.on_load_checkpoint, original_load)
        self.assertEqual(logger.log_metrics, original_log)
        self.assertEqual([module.training for module in model.modules()], flags)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertEqual(logger._run.finish_calls, 1)

    def test_validation_checkpoint_attempt_fails_without_writing(self):
        model = self.model()
        trainer, logger, _ = self.trainer(model)
        target = Path(self.directory.name) / 'forbidden.ckpt'

        def validation(*args, **kwargs):
            trainer.save_checkpoint(str(target))

        with mock.patch.object(trainer, 'validate', side_effect=validation):
            with self.assertRaisesRegex(RuntimeError, 'must not write checkpoints'):
                maybe_evaluate_before_training(trainer, model, self.loader())
        self.assertFalse(target.exists())
        self.assertEqual(logger._run.finish_calls, 1)

    def test_lazy_cuda_rng_is_captured_on_rank_local_device_without_gpu_use(self):
        model = self.model()
        trainer, _, _ = self.trainer(model)
        initialized = {'value': False}
        fake_state = torch.zeros(8, dtype=torch.uint8)

        def initialize_generator(device):
            initialized['value'] = True
            return fake_state

        with mock.patch.object(type(trainer.strategy), 'root_device',
                               new_callable=mock.PropertyMock,
                               return_value=torch.device('cuda:1')), \
             mock.patch.object(torch.cuda, 'is_initialized', side_effect=lambda: initialized['value']), \
             mock.patch.object(torch.cuda, 'device') as select_device, \
             mock.patch.object(torch.cuda, 'get_rng_state', side_effect=initialize_generator) as get_rng, \
             mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[fake_state]) as get_all, \
             mock.patch.object(torch.cuda, 'set_rng_state_all') as restore_all, \
             mock.patch.object(trainer, 'validate', return_value=[]):
            maybe_evaluate_before_training(trainer, model, self.loader())
        select_device.assert_called_once_with(torch.device('cuda:1'))
        get_rng.assert_called_once_with(torch.device('cuda:1'))
        get_all.assert_called_once()
        restore_all.assert_called_once_with([fake_state])


def ddp_smoke():
    """Opt-in: torchrun --standalone --nproc_per_node=2 this_file --ddp-smoke."""
    from lightning.pytorch.strategies import DDPStrategy

    TestPretrainEvaluation.setUpClass()
    case = TestPretrainEvaluation()
    case.setUp()
    try:
        torch.manual_seed(30)
        cfg = case.config()
        cfg.trainer.devices = 2
        model = case.model(cfg)
        logger = MemoryLogger(case.directory.name)
        order = OrderCallback()
        capture = attach_monitor(model, cfg)
        trainer = L.Trainer(
            accelerator='cpu', devices=2, precision='32-true', max_steps=2,
            strategy=DDPStrategy(process_group_backend='gloo', find_unused_parameters=True),
            logger=logger, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=1, num_sanity_val_steps=0,
            val_check_interval=4, log_every_n_steps=1, default_root_dir=case.directory.name,
            gradient_clip_val=0.0, callbacks=[order, capture],
        )
        before = copy.deepcopy(model.state_dict())
        maybe_evaluate_before_training(trainer, model, case.loader())
        case.assert_same_state(before, model.state_dict())
        assert order.events == [('validation', 0)]
        assert logger._run.finish_calls == 0
        trainer.fit(model, train_dataloaders=case.loader(), val_dataloaders=case.loader())
        assert trainer.global_step == 2
        assert order.events[1] == ('training', 0)
        if trainer.is_global_zero:
            assert logger._run.finish_calls == 1
            assert any(data.get('didi_monitor/sample_count') == 4 for data, _ in logger._run.metrics)
            print('PASS: two-rank CPU validate -> fit; gathered samples; one logger finalization')
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        case.doCleanups()


if __name__ == '__main__':
    if '--ddp-smoke' in sys.argv:
        ddp_smoke()
    else:
        unittest.main()
