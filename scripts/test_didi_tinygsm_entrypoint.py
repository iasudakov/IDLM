"""CPU integration tests for the TinyGSM command and repaired training entry."""

import copy
from contextlib import chdir
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
from omegaconf import OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader

from didi_repro.entrypoint import install_entrypoint, prepare_config
from didi_repro.monitor import attach_monitor, DiDiProvenanceMonitor
from didi_repro.protocol import training_protocol
from didi_repro.runtime import load_upstream, UPSTREAM_ROOT, verify_upstream
from didi_repro.tinygsm import (
    adaptation_record, install_tinygsm, register_configs)
from test_didi_runtime import ToyBackbone, ToyTokenizer


class Tokenizer(ToyTokenizer):
    bos_token_id = 0
    eos_token_id = 7

    def decode(self, row, **kwargs):
        return ' '.join(str(int(value)) for value in row)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]


class MemoryLogger(L.pytorch.loggers.Logger):
    def __init__(self):
        super().__init__()
        self.values, self.tables = [], []

    @property
    def name(self):
        return 'local-test'

    @property
    def version(self):
        return 'local-test'

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        self.values.append((step, dict(metrics)))

    def log_table(self, key, columns, data, step=None):
        self.tables.append((key, columns, data, step))


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        register_configs()
        cls.upstream = load_upstream()
        install_entrypoint(cls.upstream)
        cls.model_class = install_tinygsm(cls.upstream)

    def config(self, overrides=()):
        with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
            return hydra.compose(config_name='config', overrides=[
                'data=tiny-gsm', 'model.length=512', 'trainer.devices=8', *overrides])

    def test_upstream_training_defaults_survive_tinygsm_selection(self):
        with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
            original = hydra.compose(config_name='config')
        cfg = prepare_config(self.config())
        self.assertEqual(cfg.algo, original.algo)
        self.assertEqual(cfg.optim, original.optim)
        self.assertEqual(training_protocol(cfg)['optimizer_step_budget'], 10000)
        self.assertEqual(training_protocol(cfg)['lr_warmup_steps'], 1000)
        self.assertEqual(training_protocol(cfg)['ema_beta'], .999)
        self.assertIsNone(cfg.didi_training.iterations)
        self.assertEqual(cfg.data.separator, r'\n')
        self.assertFalse(cfg.didi_monitor.gm)
        self.assertFalse(cfg.didi_monitor.gen_ppl)
        self.assertEqual(cfg.underdeep.experiment, 'generative-models/tinygsm')
        self.assertIsNotNone(adaptation_record(cfg))
        prepare_config(original)
        self.assertTrue(original.didi_monitor.gm)
        self.assertEqual(original.underdeep.experiment, 'generative-models/owt')
        self.assertIsNone(adaptation_record(original))

    def test_rejects_incompatible_owt_and_unconditional_options(self):
        for override in ('+didi_init.student_hf=haoyangzheng/didi-instruct-small',
                         '+didi_pareto.every_n_validations=1',
                         '+didi_monitor.gen_ppl=true', 'data.train_on_prompt=true',
                         'data.wrap=true', 'mode=sample_eval',
                         'eval.compute_generative_perplexity=true'):
            with self.subTest(override=override), self.assertRaises(ValueError):
                prepare_config(self.config([override]))

    def test_unused_gpt2_tokenizer_is_not_loaded(self):
        with mock.patch('transformers.AutoTokenizer.from_pretrained',
                        side_effect=AssertionError('unexpected evaluator download')):
            metrics = self.upstream.algo.trainer_base.metrics.Metrics(
                gen_ppl_eval_model_name_or_path='gpt2-large', eval_ppl_batch_size=8)
        self.assertIsNone(metrics.tokenizer)

    def test_threshold_config_keeps_training_and_guided_protocol_unchanged(self):
        cfg = prepare_config(self.config([
            'sampling.predictor=guided', '+didi_eval.weights=student_ema',
            '+eval.gsm8k_at_validation=true',
            '+eval.gsm8k_pareto_thresholds=[0.325,0.5,0.98]',
            '+eval.gsm8k_pareto_plot=false']))
        self.assertEqual(list(cfg.eval.gsm8k_pareto_thresholds), [.325, .5, .98])
        self.assertEqual(cfg.eval.gsm8k_pareto_temperature, 0.)
        self.assertIsNone(cfg.eval.gsm8k_pareto_max_steps)
        self.assertEqual(cfg.sampling.predictor, 'guided')
        self.assertEqual(cfg.algo.tau_mode, 'beta11')
        self.assertEqual(training_protocol(cfg)['optimizer_step_budget'], 10000)
        self.assertEqual(training_protocol(cfg)['lr_warmup_steps'], 1000)
        self.assertEqual(training_protocol(cfg)['ema_beta'], .999)

    def test_threshold_image_dependency_fails_before_training(self):
        cfg = self.config(['+eval.gsm8k_at_validation=true',
                           '+eval.gsm8k_pareto_thresholds=[0.5]',
                           '+eval.gsm8k_pareto_plot=true'])
        with mock.patch.dict(sys.modules, {'matplotlib.backends.backend_agg': None}), \
                self.assertRaisesRegex(RuntimeError, 'require Matplotlib'):
            prepare_config(cfg)

    def test_monitor_is_provenance_only_and_does_not_capture_generation(self):
        cfg = prepare_config(self.config())
        model = mock.Mock()
        original = model.generate_samples
        monitor = attach_monitor(model, cfg)
        self.assertIsInstance(monitor, DiDiProvenanceMonitor)
        self.assertIs(model.generate_samples, original)
        monitor.on_validation_start(mock.Mock(sanity_checking=False), model)
        self.assertFalse(monitor.capturing)

    def test_launcher_dry_run_matches_released_training(self):
        with tempfile.TemporaryDirectory(prefix='didi-tiny-launcher-') as directory:
            path = Path(directory)
            teacher = path / 'mdlm_notok.ckpt'
            teacher.touch()
            gsm = path / 'gsm8k.json'
            gsm.write_text('[]')
            environment = dict(os.environ, TEACHER_CHECKPOINT=str(teacher),
                               TINYGSM_CACHE=str(path), SAVE_DIR=str(path / 'run'),
                               GSM8K_DATA=str(gsm), NPROC_PER_NODE='8', DIDI_DRY_RUN='1')
            result = subprocess.run(['bash', str(ROOT / 'scripts/train_didi_tinygsm.sh')],
                                    env=environment, check=True, capture_output=True, text=True)
            command = shlex.split(result.stdout)
            args = command[command.index('didi_repro') + 1:]
            with hydra.initialize_config_dir(version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
                cfg = hydra.compose(config_name='config', overrides=args)
            prepare_config(cfg, overrides=args)
            self.assertEqual(cfg.trainer.max_steps, 10000)
            self.assertEqual(cfg.algo.tau_mode, 'beta11')
            self.assertEqual(cfg.algo.ema_beta, .999)
            self.assertEqual(cfg.loader.global_batch_size, 8)
            self.assertEqual(cfg.loader.eval_global_batch_size, 64)
            self.assertIsNone(cfg.didi_init.student_hf)
            self.assertTrue(cfg.eval.gsm8k_at_validation)
            self.assertEqual(cfg.sampling.predictor, 'guided')
            self.assertEqual(cfg.model.length, 512)

    def test_actual_entrypoint_pretrain_eval_two_optimizer_updates_and_metrics(self):
        with tempfile.TemporaryDirectory(prefix='didi-tiny-entry-') as directory:
            cfg = self.config([
                'trainer.devices=1', 'trainer.accelerator=cpu', 'trainer.precision=32-true',
                'trainer.max_steps=4', 'trainer.val_check_interval=1',
                'trainer.log_every_n_steps=1',
                'trainer.limit_val_batches=1', 'model.length=12', 'model.hidden_size=8',
                'loader.num_workers=0', 'loader.batch_size=2', 'loader.global_batch_size=2',
                'loader.eval_batch_size=2', 'loader.eval_global_batch_size=2',
                'algo.num_samples_per_prompt=2', 'algo.lr_scheduler.warmup_steps=1',
                'algo.print_every=0', 'sampling.steps=2', 'sampling.num_sample_batches=1',
                '+didi_eval.before_training=true', '+didi_eval.weights=student_ema',
                '+eval.gsm8k_at_validation=true',
                '+eval.gsm8k_pareto_thresholds=[0.0,0.9]',
                '+eval.gsm8k_pareto_plot=false',
                '+underdeep.required=false', 'callbacks=[]',
                '+trainer.enable_checkpointing=false', '+trainer.enable_model_summary=false',
                '+trainer.enable_progress_bar=false'])
            cfg.checkpointing.save_dir = directory
            cfg.training.finetune_path = str(Path(directory) / 'teacher.ckpt')
            with open_dict(cfg):
                cfg.strategy = {'_target_': 'lightning.pytorch.strategies.SingleDeviceStrategy',
                                'device': 'cpu'}
            prepare_config(cfg)
            tokenizer, logger = Tokenizer(), MemoryLogger()
            rows = []
            for index in range(4):
                ids = (torch.arange(12) + index) % 8
                mask = torch.ones_like(ids)
                mask[:2 + index % 2] = 0
                rows.append({'input_ids': ids, 'attention_mask': mask,
                             'prompt_len': torch.tensor(2 + index % 2)})
            loader = DataLoader(rows, batch_size=2)
            loader.tokenizer = tokenizer
            models = []

            def factory(config, tokenizer):
                model = self.model_class(config, tokenizer)
                models.append(model)
                return model

            with mock.patch.object(self.upstream.algo.trainer_base.models.dit, 'DIT', ToyBackbone), \
                    mock.patch('transformers.AutoTokenizer.from_pretrained',
                               side_effect=AssertionError('unexpected download')):
                initial = self.model_class(cfg, tokenizer)
                teacher = copy.deepcopy(initial.backbone.state_dict())
                torch.save({'state_dict': {f'backbone.{key}': value for key, value in teacher.items()}},
                           cfg.training.finetune_path)
                patched_train = next(
                    cell.cell_contents for cell in self.upstream._train.__closure__
                    if getattr(cell.cell_contents, '__name__', None) == '_train')
                with chdir(directory), \
                        mock.patch.object(self.upstream.dataloader, 'get_dataloaders',
                                       return_value=(loader, loader)), \
                        mock.patch.dict(patched_train.__globals__,
                                        {'make_underdeep_logger': lambda *a, **kw: logger}), \
                        mock.patch('didi_repro.tinygsm_eval._benchmark_dataset', return_value=[
                            {'input_ids': [1, 2], 'response_ground_truth': '#### 1'},
                            {'input_ids': [2, 3, 4], 'response_ground_truth': '#### 2'}]), \
                        mock.patch('didi_repro.entrypoint._check_cached_data'):
                    self.upstream._train(factory, cfg, logging.getLogger(__name__), tokenizer)
            trained = models[-1]
            self.assertEqual(trained.trainer.global_step, 4)
            self.assertEqual(trained.global_steps, 2)
            for key, value in teacher.items():
                torch.testing.assert_close(trained.teacher.state_dict()[key], value, rtol=0, atol=0)
            logged = {key for _, values in logger.values for key in values}
            self.assertIn('val/nll', logged)
            self.assertIn('val/answer_exact_match', logged)
            self.assertIn('reward/mean', logged)
            self.assertIn('val/gsm8k_acc', logged)
            initial_curve = [(step, values) for step, values in logger.values
                             if 'pareto/gsm8k_step_0' in values]
            self.assertTrue(initial_curve)
            # Initial evaluation's curves keep their NFE*100 x-axis; the
            # generic pretrain wrapper must not collapse every point to zero.
            self.assertTrue(all(step >= 100 for step, _ in initial_curve))
            self.assertTrue(any(key.startswith('val/gsm8k_nfe_thr') for key in logged))
            samples = [table for table in logger.tables if table[0] == 'val/samples']
            self.assertTrue(samples)
            self.assertTrue(all(len(table[1]) == 3 for table in samples))
            self.assertTrue(any(table[0] == 'didi/config' for table in logger.tables))
            verify_upstream()


if __name__ == '__main__':
    unittest.main()
