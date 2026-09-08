"""CPU-only pre-training validation lifecycle tests using the upstream loss.

    conda run -n mdlm python scripts/test_didi_pretrain_eval.py

The DiDi objective and validation metrics are real upstream code. Only the
network and tokenizer are tiny local substitutes; no remote tracker/model is
contacted and no GPU is used.
"""

import copy
import json
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
from didi_repro.pareto import attach_pareto
from didi_repro.pretrain_eval import maybe_evaluate_before_training
from didi_repro.runtime import UPSTREAM_ROOT, load_upstream
from underdeep_logger import UnderdeepLogger
from test_didi_runtime import ToyBackbone, ToyDataset, ToyTokenizer
from test_didi_monitor import Evaluator


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
        self.evaluation_contexts = []

    def on_validation_start(self, trainer, pl_module):
        self.events.append(('validation', trainer.global_step))
        self.validation_weights.append(copy.deepcopy(pl_module.backbone.state_dict()))
        self.evaluation_contexts.append(copy.deepcopy(
            getattr(pl_module, '_didi_evaluation_step', None)))

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
        callbacks = [order, callback]
        pareto = attach_pareto(model, model.config)
        if pareto is not None:
            pareto.metrics_factory = Evaluator
            callbacks.append(pareto)
            # Capture image calls locally, never instantiate tracker SDK media.
            logger.log_image = mock.Mock()
        trainer = L.Trainer(
            accelerator='cpu', devices=1, precision='32-true', max_steps=max_steps,
            logger=logger, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=limit_val_batches,
            num_sanity_val_steps=0, val_check_interval=8, log_every_n_steps=1,
            default_root_dir=self.directory.name, gradient_clip_val=0.0,
            callbacks=callbacks,
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
        self.assertEqual(order.evaluation_contexts, [
            {'value': 0, 'checkpoint_loaded': True}])
        self.assertFalse(hasattr(model, '_didi_evaluation_step'))
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

    def _assert_pretrain_metric_axes(self, evaluation_step):
        """Cost-axis curves coexist with train-step scalars in one logger call."""
        model = self.model()
        trainer, logger, _ = self.trainer(model)
        logger.log_image = mock.Mock()
        original_log_metrics = logger.log_metrics
        original_log_table = logger.log_table
        original_log_image = logger.log_image
        checkpoint = None
        checkpoint_path = None
        if evaluation_step:
            model.global_steps = evaluation_step // 2
            checkpoint = {'global_step': evaluation_step,
                          'state_dict': copy.deepcopy(model.state_dict())}
            model.on_save_checkpoint(checkpoint)
            checkpoint_path = 'memory-only-resume.ckpt'
        curve = f'pareto/gsm8k_step_{evaluation_step}'
        mixed = {curve: .33, 'val/gsm8k_acc_thr05': .33,
                 'val/gen_ppl_s8_t1': 41., 'didi_pareto/samples_per_seed': 40}

        def validate(*args, **kwargs):
            self.assertEqual(kwargs['ckpt_path'], checkpoint_path)
            if checkpoint is not None:
                model.on_load_checkpoint(checkpoint)
                model.load_state_dict(checkpoint['state_dict'])
            self.assertEqual(model._didi_evaluation_step['value'], evaluation_step)
            logger.log_metrics({'val/nll': 4.2}, step=999)
            # Two pure-cost calls must remain distinct points on the NFE axis,
            # even when the outer validation phase has trainer.global_step=0.
            logger.log_metrics({curve: .11}, step=137)
            logger.log_metrics({curve: .22}, step=245)
            logger.log_metrics(mixed, step=389)
            # Existing OWT Pareto telemetry uses the checkpoint/train axis,
            # not TinyGSM's cost-series prefix, and must remain unchanged.
            logger.log_metrics({'val/gen_ppl_s16_t075': 39.,
                                'val/entropy_s16_t075': 5.2,
                                'didi_pareto/failed': 0}, step=1001)
            logger.log_metrics({'pareto/gsm8k_steps': 2,
                                'pareto/gsm8k_stepx_17': 3}, step=1002)
            logger.log_table(key='pareto/protocol', columns=['key'], data=[['value']], step=777)
            logger.log_image(key='pareto/frontier', image='memory-only', step=778)
            return [{'val/nll': 4.2}]

        with mock.patch.object(trainer, 'validate', side_effect=validate):
            result = maybe_evaluate_before_training(
                trainer, model, self.loader(), ckpt_path=checkpoint_path)
        self.assertEqual(result, [{'val/nll': 4.2}])
        cost_calls = [(metrics, step) for metrics, step in logger._run.metrics if curve in metrics]
        self.assertEqual(cost_calls, [({curve: .11}, 137), ({curve: .22}, 245), ({curve: .33}, 389)])
        regular_calls = [(metrics, step) for metrics, step in logger._run.metrics if curve not in metrics]
        self.assertTrue(regular_calls)
        self.assertTrue(all(step == evaluation_step for _, step in regular_calls))
        self.assertIn(({'val/gsm8k_acc_thr05': .33, 'val/gen_ppl_s8_t1': 41.,
                        'didi_pareto/samples_per_seed': 40}, evaluation_step), regular_calls)
        self.assertIn(({'val/gen_ppl_s16_t075': 39., 'val/entropy_s16_t075': 5.2,
                        'didi_pareto/failed': 0}, evaluation_step), regular_calls)
        self.assertIn(({'pareto/gsm8k_steps': 2, 'pareto/gsm8k_stepx_17': 3}, evaluation_step), regular_calls)
        self.assertEqual(mixed, {curve: .33, 'val/gsm8k_acc_thr05': .33,
                                 'val/gen_ppl_s8_t1': 41., 'didi_pareto/samples_per_seed': 40})
        self.assertTrue(all(step == evaluation_step for _, _, step in logger.tables))
        original_log_image.assert_called_once_with(
            key='pareto/frontier', image='memory-only', step=evaluation_step)
        self.assertEqual(logger.log_metrics, original_log_metrics)
        self.assertEqual(logger.log_table, original_log_table)
        self.assertIs(logger.log_image, original_log_image)
        self.assertFalse(hasattr(model, '_didi_evaluation_step'))
        self.assertEqual(logger._run.finish_calls, 0)
        # The temporary adapter must not leak into subsequent fit logging.
        logger.log_metrics({'val/post_training': 7}, step=1007)
        self.assertEqual(logger._run.metrics[-1], ({'val/post_training': 7}, 1007))

    def test_fresh_pretrain_validation_preserves_tinygsm_nfe_axis_and_owt_steps(self):
        self._assert_pretrain_metric_axes(evaluation_step=0)

    def test_resume_pretrain_validation_preserves_tinygsm_nfe_axis_and_owt_steps(self):
        self._assert_pretrain_metric_axes(evaluation_step=26)

    def test_resume_evaluates_restored_weights_at_checkpoint_step(self):
        model = self.model()
        trainer, _, _ = self.trainer(model)
        trainer.fit(model, train_dataloaders=self.loader())
        checkpoint = str(Path(self.directory.name) / 'resume.ckpt')
        trainer.save_checkpoint(checkpoint)
        saved = copy.deepcopy(model.backbone.state_dict())
        config = self.config()
        with open_dict(config):
            config.didi_pareto = {
                'every_n_validations': 3, 'temperatures': [1.0],
                'steps': [2], 'seeds': 1, 'num_batches': 1, 'batch_size': 2,
            }
        resumed = self.model(config)
        continuation, logger, order = self.trainer(resumed, max_steps=4)
        maybe_evaluate_before_training(continuation, resumed, self.loader(), ckpt_path=checkpoint)
        self.assert_same_state(saved, order.validation_weights[0])
        self.assertEqual(order.evaluation_contexts, [
            {'value': 2, 'checkpoint_loaded': True}])
        self.assertFalse(hasattr(resumed, '_didi_evaluation_step'))
        history = json.loads((Path(self.directory.name)
                              / 'didi_pareto_temp_history.json').read_text())
        self.assertEqual(set(history['curves']), {'2'})
        values = [(data, step) for data, step in logger._run.metrics if 'val/nll' in data]
        self.assertTrue(values)
        self.assertEqual(values[-1][1], 2)
        self.assertEqual(logger._run.finish_calls, 0)
        self.assertEqual(resumed.global_steps, 1)
        continuation.fit(resumed, train_dataloaders=self.loader(), ckpt_path=checkpoint)
        self.assertEqual(continuation.global_step, 4)
        self.assertEqual(resumed.global_steps, 2)
        self.assertEqual(logger._run.finish_calls, 1)

    def test_pareto_baseline_keeps_training_trajectory_and_separate_history(self):
        states = []
        for enabled in (False, True):
            random.seed(51)
            np.random.seed(51)
            torch.manual_seed(51)
            config = self.config()
            with open_dict(config):
                config.didi_pareto = {
                    'every_n_validations': 3 if enabled else 0,
                    'temperatures': [0.75, 1.0], 'steps': [2], 'seeds': 1,
                    'num_batches': 1, 'batch_size': 2,
                }
            model = self.model(config)
            trainer, logger, _ = self.trainer(model)
            initial = copy.deepcopy(model.state_dict())
            maybe_evaluate_before_training(trainer, model, self.loader())
            self.assert_same_state(model.state_dict(), initial)
            if enabled:
                history = json.loads((Path(self.directory.name)
                                      / 'didi_pareto_temp_history.json').read_text())
                self.assertEqual(set(history['curves']), {'0'})
                self.assertEqual(set(history['curves']['0']), {'0.75', '1'})
                pareto = next(callback for callback in trainer.callbacks
                              if hasattr(callback, 'validations_seen'))
                self.assertEqual(pareto.validations_seen, 0)
                failures = [values['didi_pareto/failed'] for values, _ in logger._run.metrics
                            if 'didi_pareto/failed' in values]
                self.assertEqual(failures, [0])
                logger.log_image.assert_called()
            trainer.fit(model, train_dataloaders=self.loader())
            states.append(copy.deepcopy(model.state_dict()))
        self.assert_same_state(*states)

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
            self.assertEqual(model._didi_evaluation_step['value'], 0)
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
        self.assertFalse(hasattr(model, '_didi_evaluation_step'))
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


def ddp_smoke(pareto_enabled=False):
    """Opt-in: torchrun --standalone --nproc_per_node=2 this_file --ddp-smoke."""
    from lightning.pytorch.strategies import DDPStrategy

    TestPretrainEvaluation.setUpClass()
    case = TestPretrainEvaluation()
    case.setUp()
    try:
        torch.manual_seed(30)
        cfg = case.config()
        cfg.trainer.devices = 2
        if pareto_enabled:
            from test_didi_pareto import uniform_root_factory
            cfg.didi_monitor.gen_ppl = True
            with open_dict(cfg):
                cfg.didi_pareto = {
                    'every_n_validations': 1, 'temperatures': [0.75, 1.0],
                    'steps': [2], 'seeds': 2, 'num_batches': 1, 'batch_size': 2,
                    'sampler': 'guided',
                }
        model = case.model(cfg)
        logger = MemoryLogger(case.directory.name)
        order = OrderCallback()
        capture = attach_monitor(model, cfg)
        callbacks = [order, capture]
        if pareto_enabled:
            capture.metrics_factory = uniform_root_factory
            pareto = attach_pareto(model, cfg)
            pareto.metrics_factory = uniform_root_factory
            callbacks.append(pareto)
            logger.log_image = mock.Mock()
        trainer = L.Trainer(
            accelerator='cpu', devices=2, precision='32-true', max_steps=2,
            strategy=DDPStrategy(process_group_backend='gloo', find_unused_parameters=True),
            logger=logger, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=1, num_sanity_val_steps=0,
            val_check_interval=4, log_every_n_steps=1, default_root_dir=case.directory.name,
            gradient_clip_val=0.0, callbacks=callbacks,
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
            if pareto_enabled:
                history = json.loads((Path(case.directory.name)
                                      / 'didi_pareto_temp_history.json').read_text())
                assert history['protocol']['samples_per_seed'] == 4
                assert set(history['curves']) == {'0'}
                assert any(data.get('didi_pareto/failed') == 0 for data, _ in logger._run.metrics)
                assert any(abs(data.get('val/gen_ppl', 0) - 9.0) < 1e-5
                           for data, _ in logger._run.metrics)
                print('PASS: baseline guided temperature sweep; global counts; CPU DDP -> fit')
            print('PASS: two-rank CPU validate -> fit; gathered samples; one logger finalization')
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        case.doCleanups()


if __name__ == '__main__':
    if '--ddp-smoke' in sys.argv:
        ddp_smoke(pareto_enabled='--pareto-smoke' in sys.argv)
    else:
        unittest.main()
