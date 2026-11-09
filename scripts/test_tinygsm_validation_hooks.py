"""Exercise actual TrainerBase hooks with a tiny EMA model and offline stubs.

The AST-extracted methods are the production hooks. Model assets, datasets,
Lightning setup and the conditional evaluator itself are not constructed here.
"""
import ast
from contextlib import contextmanager, redirect_stdout
import io
import itertools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


class Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class Metric:
    mean_value = 0
    weight = 0

    def compute(self):
        return torch.tensor(1.25)


class ValidMetrics(dict):
    @property
    def nll(self):
        return self['val/nll']


class TinyEMA:
    def __init__(self):
        self.saved = None
        self.stores = 0
        self.restores = 0
        self.moved = []

    def move_shadow_params_to_device(self, device):
        self.moved.append(device)

    def store(self, parameters):
        if self.saved is not None:
            raise AssertionError('EMA entered twice before restoring training weights')
        self.saved = [parameter.detach().clone() for parameter in parameters]
        self.stores += 1

    def copy_to(self, parameters):
        with torch.no_grad():
            for parameter in parameters:
                parameter.fill_(7.)

    def restore(self, parameters):
        if self.saved is None:
            raise AssertionError('EMA restored without a matching eval entry')
        with torch.no_grad():
            for parameter, saved in zip(parameters, self.saved):
                parameter.copy_(saved)
        self.saved = None
        self.restores += 1


def hook_class():
    tree = ast.parse((ROOT / 'trainer_base.py').read_text())
    base = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'TrainerBase')
    names = {'_get_parameters', '_eval_mode', '_train_mode', 'on_train_start',
             'on_validation_epoch_start', 'on_validation_epoch_end',
             '_generative_eval_active'}
    methods = [node for node in base.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    probe = ast.parse('class HookProbe:\n    pass').body[0]
    probe.body = methods
    sampler = mock.Mock(side_effect=lambda dataset, **kwargs:
                        SimpleNamespace(dataset=dataset, **kwargs))
    loader = mock.Mock(side_effect=lambda dataset, **kwargs:
                       SimpleNamespace(dataset=dataset, **kwargs))
    namespace = {
        'itertools': itertools,
        'torch': SimpleNamespace(utils=SimpleNamespace(
            data=SimpleNamespace(DataLoader=loader))),
        'dataloader': SimpleNamespace(RandomFaultTolerantSampler=sampler,
                                     FaultTolerantDistributedSampler=sampler),
        'metrics': SimpleNamespace(GRADIENT_MOMENT_ENABLED=False),
    }
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    exec(compile(module, 'trainer_base.py', 'exec'), namespace)
    return namespace['HookProbe'], loader


@contextmanager
def callback_stub(callback):
    module = ModuleType('tinygsm_validation')
    module.log_validation_gsm8k = callback
    with mock.patch.dict(sys.modules, {'tinygsm_validation': module}):
        yield


@contextmanager
def reject_conditional_import():
    original = __import__
    def guarded(name, *args, **kwargs):
        if name == 'tinygsm_validation':
            raise AssertionError('inactive/legacy route imported the GSM evaluator')
        return original(name, *args, **kwargs)
    with mock.patch('builtins.__import__', side_effect=guarded):
        yield


class TinyGSMValidationHookTests(unittest.TestCase):
    def make(self, *, dataset='tiny_gsm', step=0, **eval_overrides):
        cls, loader = hook_class()
        model = cls()
        settings = dict(gsm8k_at_validation=True, validate_before_train=True,
                        generate_samples=False, compute_perplexity_on_sanity=False,
                        compute_generative_perplexity=False,
                        eval_step_range=None, eval_at_steps=None)
        settings.update(eval_overrides)
        model.config = Config(
            data=Config(train=dataset), eval=Config(**settings),
            loader=Config(batch_size=2, eval_batch_size=2, num_workers=1,
                          pin_memory=False),
            sampling=Config(num_sample_batches=1, num_sample_log=2))
        with torch.random.fork_rng(devices=[]):
            model.backbone = torch.nn.Linear(1, 1, bias=False)
            model.noise = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.backbone.weight.fill_(2.)
            model.noise.weight.fill_(3.)
        model.ema = TinyEMA()
        model.device = torch.device('cpu')
        model.global_step = step
        model.num_tokens = 3
        model.fast_forward_epochs = model.fast_forward_batches = None
        original_loader = SimpleNamespace(dataset=[0, 1],
                                          sampler=SimpleNamespace(shuffle=True))
        model.trainer = SimpleNamespace(
            sanity_checking=False, global_rank=0, state=SimpleNamespace(fn='fit'),
            logger=SimpleNamespace(log_table=mock.Mock()),
            _accelerator_connector=SimpleNamespace(
                use_distributed_sampler=False, is_distributed=False),
            fit_loop=SimpleNamespace(_combined_loader=SimpleNamespace(
                flattened=[original_loader])))
        model.metrics = SimpleNamespace(
            valid_nlls=ValidMetrics({'val/nll': Metric()}), reset=mock.Mock(),
            record_entropy=mock.Mock(), record_generative_perplexity=mock.Mock(),
            gen_ppl=Metric(), sample_entropy=Metric())
        model.log = mock.Mock()
        model.tokenizer = SimpleNamespace(batch_decode=lambda _: ['first', 'second'])
        model.generate_samples = mock.Mock(return_value=torch.ones((2, 3), dtype=torch.long))
        model._log_gradient_moment = mock.Mock()
        model._log_owt_pareto = mock.Mock()
        model._log_owt_pareto_temperature = mock.Mock()
        return model, loader, original_loader

    def assert_eval_weights(self, model):
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.noise.training)
        self.assertEqual(model.backbone.weight.item(), 7.)
        self.assertEqual(model.noise.weight.item(), 7.)

    def assert_training_weights(self, model):
        self.assertTrue(model.backbone.training)
        self.assertTrue(model.noise.training)
        self.assertEqual(model.backbone.weight.item(), 2.)
        self.assertEqual(model.noise.weight.item(), 3.)
        self.assertEqual(model.ema.stores, model.ema.restores)

    def test_validation_uses_existing_ema_and_restores_training_weights(self):
        model, _, _ = self.make(step=1000)
        callback = mock.Mock(side_effect=lambda actual:
                             self.assert_eval_weights(actual))
        with callback_stub(callback):
            model.on_validation_epoch_start()
            self.assert_eval_weights(model)
            model.on_validation_epoch_end()
        callback.assert_called_once_with(model)
        self.assertEqual(model.ema.stores, 1)
        self.assert_training_weights(model)
        model.log.assert_called_once_with(name='val/nll', value=mock.ANY,
                                          on_step=False, on_epoch=True, sync_dist=True)
        model.generate_samples.assert_not_called()
        model._log_owt_pareto.assert_not_called()

    def test_validation_callback_error_still_restores_training_weights(self):
        model, _, _ = self.make()
        def fail(actual):
            self.assert_eval_weights(actual)
            raise RuntimeError('conditional evaluation failed')
        with callback_stub(fail):
            model.on_validation_epoch_start()
            with self.assertRaisesRegex(RuntimeError, 'conditional evaluation failed'):
                model.on_validation_epoch_end()
        self.assert_training_weights(model)

    def test_validation_import_error_still_restores_training_weights(self):
        model, _, _ = self.make()
        original = __import__
        def fail_import(name, *args, **kwargs):
            if name == 'tinygsm_validation':
                raise ImportError('conditional evaluator dependency unavailable')
            return original(name, *args, **kwargs)
        model.on_validation_epoch_start()
        with mock.patch('builtins.__import__', side_effect=fail_import):
            with self.assertRaisesRegex(ImportError, 'dependency unavailable'):
                model.on_validation_epoch_end()
        self.assert_training_weights(model)

    def test_eval_window_skips_conditional_import_but_logs_nll_and_restores(self):
        model, _, _ = self.make(step=99, eval_step_range=[100, 200])
        with reject_conditional_import(), redirect_stdout(io.StringIO()):
            model.on_validation_epoch_start()
            self.assertFalse(model._gen_eval_active)
            model.on_validation_epoch_end()
        self.assert_training_weights(model)
        self.assertEqual(model.log.call_count, 1)
        model._log_owt_pareto.assert_not_called()

    def test_owt_keeps_sampling_perplexity_and_pareto_without_conditional_import(self):
        model, _, _ = self.make(dataset='openwebtext', generate_samples=True,
                               gsm8k_at_validation=False,
                               compute_generative_perplexity=True)
        with reject_conditional_import():
            model.on_validation_epoch_start()
            model.on_validation_epoch_end()
        self.assert_training_weights(model)
        model.generate_samples.assert_called_once_with(num_samples=2)
        model.metrics.record_entropy.assert_called_once()
        model.metrics.record_generative_perplexity.assert_called_once_with(
            ['first', 'second'], 3, model.device)
        model.trainer.logger.log_table.assert_called_once()
        model._log_gradient_moment.assert_called_once_with([])
        model._log_owt_pareto.assert_called_once_with()
        model._log_owt_pareto_temperature.assert_called_once_with()
        self.assertEqual(model.log.call_count, 3)

    def test_initial_validation_runs_after_loader_rebuild_and_uses_ema(self):
        model, loader, original_loader = self.make()
        def initial(actual, *, initial):
            self.assertTrue(initial)
            self.assert_eval_weights(actual)
            self.assertIsNot(actual.trainer.fit_loop._combined_loader.flattened[0],
                             original_loader)
        callback = mock.Mock(side_effect=initial)
        with callback_stub(callback):
            model.on_train_start()
        callback.assert_called_once_with(model, initial=True)
        self.assert_training_weights(model)
        self.assertEqual(model.ema.stores, 1)
        self.assertEqual(model.ema.moved, [model.device])
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs['batch_size'], 2)
        self.assertTrue(loader.call_args.kwargs['persistent_workers'])

    def test_initial_validation_requires_tinygsm_flag_and_step_zero(self):
        cases = [dict(dataset='openwebtext'), dict(step=1),
                 dict(gsm8k_at_validation=False), dict(validate_before_train=False)]
        for settings in cases:
            with self.subTest(settings=settings):
                model, loader, _ = self.make(**settings)
                with reject_conditional_import():
                    model.on_train_start()
                loader.assert_called_once()
                self.assertEqual(model.ema.stores, 0)
                self.assert_training_weights(model)

    def test_initial_callback_error_still_restores_training_weights(self):
        model, _, _ = self.make()
        def fail(actual, *, initial):
            self.assertTrue(initial)
            self.assert_eval_weights(actual)
            raise RuntimeError('initial conditional evaluation failed')
        with callback_stub(fail), self.assertRaisesRegex(RuntimeError, 'initial conditional'):
            model.on_train_start()
        self.assert_training_weights(model)


if __name__ == '__main__':
    unittest.main(verbosity=2)
