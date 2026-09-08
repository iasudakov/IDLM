"""CPU-only checks for official inference inside distributed DiDi evaluation.

    python scripts/test_didi_integrated_inference.py
    torchrun --standalone --nproc_per_node=2 \
        scripts/test_didi_integrated_inference.py --ddp-smoke

The command-composition tests retain the real 8-GPU/12-temperature/5-seed
settings. Runtime tests substitute only a tiny CPU network/tokenizer and local
reference LM; they never contact HF or Underdeep and never train a model.
"""

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import MethodType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import hydra
import lightning as L
from omegaconf import open_dict
import torch

from didi_repro.entrypoint import prepare_config
from didi_repro.initialization import AUTHORS_STUDENT_REVISION
from didi_repro.monitor import attach_monitor
from didi_repro.pareto import attach_pareto, validate_options
from didi_repro.pretrain_eval import maybe_evaluate_before_training
from didi_repro.runtime import UPSTREAM_ROOT
from didi_repro.standalone import INFERENCE_REVISION
from test_didi_pretrain_eval import MemoryLogger, OrderCallback
import test_didi_pretrain_eval as pretrain_fixture
from test_didi_pareto import uniform_root_factory


TEMPERATURES = [.75, .80, .85, .875, .90, .925, .95, .975, 1.0, 1.05, 1.10, 1.20]


def evaluation_overrides():
    """The user's 8-GPU evaluation-only command, with harmless fixture paths."""
    return [
        'mode=train', 'data=openwebtext', 'data.cache_dir=/unused/owt-cache',
        'checkpointing.save_dir=/unused/didi-inference-pareto',
        'checkpointing.resume_from_ckpt=false',
        'hydra.run.dir=/unused/didi-inference-pareto',
        'model=small', 'model.length=1024', 'algo=didi_instruct', 'algo.backbone=dit',
        'training.finetune_path=/unused/mdlm.ckpt',
        '+didi_init.student_hf=haoyangzheng/didi-instruct-small',
        f'+didi_init.revision={AUTHORS_STUDENT_REVISION}',
        '+didi_eval.before_training=true', '+didi_eval.weights=student_ema',
        '+didi_eval.sampler_source=inference', '+didi_training.iterations=null',
        'algo.num_samples_per_prompt=8', 'optim.lr=1e-6', 'algo.discriminator_lr=1e-6',
        'training.loss_precision=bf16', 'trainer.precision=bf16',
        'trainer.devices=8', 'trainer.num_nodes=1',
        'loader.batch_size=1', 'loader.global_batch_size=8',
        'loader.eval_batch_size=5', 'loader.eval_global_batch_size=40',
        'loader.num_workers=4', 'trainer.accumulate_grad_batches=1',
        'trainer.gradient_clip_val=0.0', 'trainer.max_steps=0',
        'callbacks=[]', '+trainer.enable_checkpointing=false',
        'trainer.limit_val_batches=1', 'trainer.num_sanity_val_steps=0',
        'eval.generate_samples=false', 'eval.perplexity_batch_size=4',
        '+didi_monitor.enabled=true',
        '+didi_monitor.gen_ppl=false', '+didi_monitor.gm=false',
        'sampling.steps=16', 'sampling.predictor=guided',
        'sampling.noise_removal=ancestral', 'sampling.num_sample_batches=1',
        '+didi_pareto.every_n_validations=1',
        '+didi_pareto.temperatures=[0.75,0.80,0.85,0.875,0.90,0.925,0.95,0.975,1.00,1.05,1.10,1.20]',
        '+didi_pareto.steps=[8,16,32]', '+didi_pareto.seeds=5',
        '+didi_pareto.num_batches=1',
        '+underdeep.experiment=generative-models/owt',
        '+underdeep.force_new_run=true', '+underdeep.name=hf-didi-official-inference-pareto',
        '+didi_runtime.strict_environment=false',
    ]


def compose(overrides):
    with hydra.initialize_config_dir(
            version_base=None, config_dir=str(UPSTREAM_ROOT / 'configs')):
        config = hydra.compose(config_name='config', overrides=overrides)
    return prepare_config(config, overrides=overrides)


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pretrain_fixture.TestPretrainEvaluation.setUpClass()

    def test_exact_eight_gpu_command_composes_without_training_budget(self):
        config = compose(evaluation_overrides())
        validate_options(config)
        self.assertEqual(config.mode, 'train')  # Existing prevalidation-only route.
        self.assertEqual(config.trainer.max_steps, 0)
        self.assertFalse(config.trainer.enable_checkpointing)
        self.assertFalse(config.get('callbacks'))
        self.assertFalse(config.checkpointing.resume_from_ckpt)
        self.assertIsNone(config.didi_training.iterations)
        self.assertEqual(config.trainer.devices, 8)
        self.assertEqual(config.didi_eval.sampler_source, 'inference')
        self.assertEqual(config.didi_eval.weights, 'student_ema')
        self.assertTrue(config.didi_eval.before_training)
        self.assertFalse(config.eval.generate_samples)
        self.assertFalse(config.didi_monitor.gen_ppl)
        self.assertFalse(config.didi_monitor.gm)
        self.assertEqual(config.algo.guidance_scale_start, 30)
        self.assertEqual(config.algo.guidance_scale_end, 40)
        self.assertEqual(config.algo.num_candidates, 4)
        self.assertEqual(config.algo.rerank_steps_ratio, .5)
        self.assertEqual(list(config.didi_pareto.temperatures), TEMPERATURES)
        self.assertEqual(list(config.didi_pareto.steps), [8, 16, 32])
        global_samples = (config.loader.eval_batch_size * config.trainer.devices
                          * config.didi_pareto.num_batches)
        self.assertEqual(global_samples, 40)
        self.assertEqual(global_samples * config.didi_pareto.seeds, 200)
        self.assertEqual(global_samples * config.didi_pareto.seeds
                         * len(config.didi_pareto.steps)
                         * len(config.didi_pareto.temperatures), 7200)
        self.assertEqual(config.didi_init.student_hf, 'haoyangzheng/didi-instruct-small')
        self.assertEqual(config.didi_init.revision, AUTHORS_STUDENT_REVISION)
        self.assertEqual(config.eval.perplexity_batch_size, 4)
        # The existing route still requires teacher and data preflight. This
        # composition deliberately does not call either or connect a logger.
        self.assertEqual(config.training.finetune_path, '/unused/mdlm.ckpt')
        self.assertEqual(config.data.cache_dir, '/unused/owt-cache')

    def test_main_defaults_are_unchanged(self):
        config = compose(['trainer.devices=1'])
        self.assertEqual(config.didi_eval.sampler_source, 'main')
        self.assertEqual(config.algo.guidance_scale_start, .2)
        self.assertEqual(config.algo.guidance_scale_end, 1)
        self.assertEqual(config.trainer.max_steps, 10000)
        self.assertEqual(config.sampling.predictor, 'ancestral_cache')

    def test_explicit_guidance_overrides_survive_even_when_equal_to_old_defaults(self):
        overrides = evaluation_overrides() + [
            'algo.guidance_scale_start=0.2', 'algo.guidance_scale_end=1.0',
            'algo.num_candidates=3', 'algo.rerank_steps_ratio=0.25']
        config = compose(overrides)
        prepare_config(config, overrides=overrides)  # Repeated preparation is safe.
        self.assertEqual(config.algo.guidance_scale_start, .2)
        self.assertEqual(config.algo.guidance_scale_end, 1)
        self.assertEqual(config.algo.num_candidates, 3)
        self.assertEqual(config.algo.rerank_steps_ratio, .25)

    def test_inference_defaults_are_idempotent(self):
        overrides = evaluation_overrides()
        config = compose(overrides)
        prepare_config(config, overrides=overrides)
        self.assertEqual(config.algo.guidance_scale_start, 30)
        self.assertEqual(config.algo.guidance_scale_end, 40)
        self.assertEqual(config.trainer.max_steps, 0)

    def test_nondefault_programmatic_guidance_is_preserved(self):
        config = compose(evaluation_overrides())
        config.algo.guidance_scale_start = 17
        config.algo.guidance_scale_end = 23
        prepare_config(config, overrides=[])
        self.assertEqual(config.algo.guidance_scale_start, 17)
        self.assertEqual(config.algo.guidance_scale_end, 23)

    def test_pareto_cannot_override_official_source_to_ancestral(self):
        with self.assertRaises(ValueError):
            compose(evaluation_overrides() + ['+didi_pareto.sampler=ancestral'])

    def test_invalid_source_weights_and_sampler_are_rejected(self):
        replacements = (
            ('+didi_eval.sampler_source=inference', '+didi_eval.sampler_source=bogus'),
            ('+didi_eval.weights=student_ema', '+didi_eval.weights=backbone'),
            ('sampling.predictor=guided', 'sampling.predictor=ancestral_cache'),
        )
        for old, new in replacements:
            with self.subTest(override=new), self.assertRaises(ValueError):
                compose([new if argument == old else argument
                         for argument in evaluation_overrides()])

    def test_zero_steps_cannot_be_combined_with_positive_iteration_budget(self):
        overrides = [argument.replace('+didi_training.iterations=null',
                                      '+didi_training.iterations=10000')
                     for argument in evaluation_overrides()]
        with self.assertRaises(ValueError):
            compose(overrides)


def inference_fixture(case, world=1):
    config = case.config()
    config.trainer.devices = world
    config.trainer.max_steps = 0
    config.eval.generate_samples = False
    config.eval.perplexity_batch_size = 3  # Deliberate tail tests all-sample scoring.
    config.sampling.predictor = 'guided'
    config.sampling.noise_removal = 'ancestral'
    with open_dict(config):
        config.didi_training = {'iterations': None}
        config.didi_eval = {'before_training': True, 'weights': 'student_ema',
                            'sampler_source': 'inference'}
        config.didi_pareto = {
            'every_n_validations': 1, 'temperatures': [1.0], 'steps': [8, 16, 32],
            'seeds': 2, 'num_batches': 1, 'batch_size': 2, 'keep_samples': 1,
        }
    prepare_config(config, overrides=[])
    model = case.model(config)
    return model


def zero_update_lifecycle(case, world=1):
    """Actual original sampler/Pareto lifecycle; tiny CPU substitutes only."""
    from lightning.pytorch.strategies import DDPStrategy

    torch.manual_seed(814)
    model = inference_fixture(case, world)
    logger = MemoryLogger(case.directory.name)
    logger.log_image = mock.Mock()
    order = OrderCallback()
    monitor = attach_monitor(model, model.config)
    pareto = attach_pareto(model, model.config)
    scored_batch_sizes = []

    def local_evaluator(**kwargs):
        evaluator = uniform_root_factory(**kwargs)
        original_score = evaluator.record_generative_perplexity

        def score(texts, **score_kwargs):
            scored_batch_sizes.append(len(texts))
            return original_score(texts, **score_kwargs)

        evaluator.record_generative_perplexity = score
        return evaluator

    pareto.metrics_factory = local_evaluator
    options = {}
    if world > 1:
        options['strategy'] = DDPStrategy(
            process_group_backend='gloo', find_unused_parameters=True,
            timeout=timedelta(seconds=90))
    trainer = L.Trainer(
        accelerator='cpu', devices=world, precision='32-true', max_steps=0,
        logger=logger, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, limit_val_batches=1, num_sanity_val_steps=0,
        val_check_interval=4, log_every_n_steps=1,
        default_root_dir=case.directory.name, gradient_clip_val=0.,
        callbacks=[order, monitor, pareto], **options)
    calls = {'training_step': 0, 'on_train_start': 0}

    def no_training(self, batch, batch_idx):
        calls['training_step'] += 1
        raise AssertionError('Evaluation-only command performed training!')

    def no_train_start(self):
        calls['on_train_start'] += 1
        raise AssertionError('Evaluation-only command entered on_train_start!')

    model.training_step = MethodType(no_training, model)
    model.on_train_start = MethodType(no_train_start, model)
    before = copy.deepcopy(model.state_dict())
    before_modes = [module.training for module in model.modules()]
    raw_calls, ema_calls, generations = [], [], []
    raw_hook = model.backbone.register_forward_pre_hook(lambda *args: raw_calls.append(1))
    ema_hook = model.student_ema.register_forward_pre_hook(lambda *args: ema_calls.append(1))
    original_generate = model.generate_samples

    def capture(*args, **kwargs):
        generations.append((kwargs['num_samples'], kwargs['num_steps']))
        return original_generate(*args, **kwargs)

    try:
        with mock.patch.object(model, 'generate_samples', side_effect=capture), \
                mock.patch.object(trainer, 'save_checkpoint',
                                  side_effect=AssertionError('Unexpected checkpoint write')) as save:
            maybe_evaluate_before_training(trainer, model, case.loader())
            case.assert_same_state(before, model.state_dict())
            assert before_modes == [module.training for module in model.modules()]
            trainer.fit(model, train_dataloaders=case.loader(), val_dataloaders=case.loader())
            assert save.call_count == 0
    finally:
        raw_hook.remove()
        ema_hook.remove()
    case.assert_same_state(before, model.state_dict())
    assert calls == {'training_step': 0, 'on_train_start': 0}, calls
    assert trainer.global_step == model.global_steps == 0
    assert order.events == [('validation', 0)], order.events
    assert len(trainer.optimizers) == 2
    assert all(len(optimizer.state) == 0 for optimizer in trainer.optimizers)
    assert model._didi_sampler_source == 'inference'
    assert len(generations) == 6, generations  # 3 budgets × 2 repeats × 1 batch.
    assert sum(batch for batch, _ in generations) == 12
    assert len(raw_calls) == 6, raw_calls  # One original raw terminal-denoise per batch.
    assert len(ema_calls) >= sum(steps for _, steps in generations), len(ema_calls)
    assert not list(Path(case.directory.name).rglob('*.ckpt'))
    if trainer.is_global_zero:
        assert logger._run.finish_calls == 1
        assert logger.log_image.call_count >= 1
        history = json.loads((Path(case.directory.name) / 'didi_pareto_temp_history.json').read_text())
        assert set(history['curves']) == {'0'}
        assert history['protocol']['world_size'] == world
        assert history['protocol']['samples_per_seed'] == 2 * world
        assert history['protocol']['weights'] == 'student_ema'
        assert history['protocol']['sampler'] == 'guided'
        assert history['protocol']['sampler_source'] == 'inference'
        assert history['protocol']['sampler_revision'] == INFERENCE_REVISION
        assert history['protocol']['guided_weights'] == 'student_ema'
        assert history['protocol']['ancestral_cleanup_weights'] == 'backbone'
        protocol_rows = next(rows for key, rows, step in logger.tables
                             if key == 'pareto/protocol' and step == 0)
        logged_protocol = {name: json.loads(value) for name, value in protocol_rows}
        assert logged_protocol['sampler_source'] == 'inference'
        assert logged_protocol['sampler_revision'] == INFERENCE_REVISION
        assert logged_protocol['ancestral_cleanup_weights'] == 'backbone'
        assert set(history['curves']['0']) == {'1'}
        points = history['curves']['0']['1']
        assert len(points) == 3
        assert {point['steps'] for point in points} == {8, 16, 32}
        assert all(point['total_samples'] == 4 * world for point in points)
        assert all(point['seeds'] == 2 for point in points)
        assert all(abs(point['gen_ppl'] - 9.) < 1e-5 for point in points)
        expected_chunks = [min(3, 2 * world - offset) for offset in range(0, 2 * world, 3)]
        assert scored_batch_sizes == expected_chunks * 6, scored_batch_sizes
        assert sum(scored_batch_sizes) == 12 * world
        failures = [data['didi_pareto/failed'] for data, _ in logger._run.metrics
                    if 'didi_pareto/failed' in data]
        assert failures == [0], failures
        print(f'PASS: official inference baseline, {world} CPU rank(s), '
              f'{12 * world} total samples, zero training/checkpoint calls, '
              'unchanged tensors, complete Pareto metrics and logger finalization')


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pretrain_fixture.TestPretrainEvaluation.setUpClass()

    def test_official_inference_baseline_then_zero_step_fit(self):
        case = pretrain_fixture.TestPretrainEvaluation()
        case.setUp()
        try:
            zero_update_lifecycle(case)
        finally:
            case.doCleanups()


def ddp_smoke():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != 2:
        raise ValueError('Use torchrun --standalone --nproc_per_node=2 for the CPU smoke test.')
    pretrain_fixture.TestPretrainEvaluation.setUpClass()
    case = pretrain_fixture.TestPretrainEvaluation()
    case.setUp()
    try:
        zero_update_lifecycle(case, world=world)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        case.doCleanups()


if __name__ == '__main__':
    if '--ddp-smoke' in sys.argv:
        ddp_smoke()
    else:
        unittest.main()
