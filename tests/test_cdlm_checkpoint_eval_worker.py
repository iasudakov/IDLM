"""CPU checks for the common inference adapter and immutable evaluation reports."""

import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import cdlm_checkpoint_eval_worker as worker


class TinyModel:
    def __init__(self):
        self.backbone = torch.nn.Linear(2, 2)
        self.ema = object()
        self.evaluation = False

    def _get_parameters(self):
        return self.backbone.parameters()

    def eval(self):
        self.evaluation = True
        self.backbone.eval()
        return self


def checkpoint(model, *, ema_only=False, step=1000):
    result = {'global_step': step, 'epoch': 12,
              'state_dict': {f'backbone.{key}': torch.full_like(value, 2.)
                             for key, value in model.backbone.state_dict().items()}}
    if ema_only:
        result['checkpoint_weights'] = 'ema'
    else:
        # Teacher tensors must be ignored; they are not part of the student EMA.
        result['state_dict']['teacher_head.weight'] = torch.zeros(100, 100)
        result['ema'] = {'shadow_params': [torch.full_like(p, 3.)
                                           for p in model.backbone.parameters()]}
    return result


def _distributed_error(rank, rendezvous, queue):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        try:
            worker._sync_error(ValueError('rank-one-load-error') if rank else None, 'Checkpoint load')
        except RuntimeError as error:
            queue.put((rank, str(error)))
        else:
            queue.put((rank, 'NO ERROR'))
    finally:
        dist.destroy_process_group()


class NormalizationTest(unittest.TestCase):
    def test_same_bf16_logits_reproduce_both_confidence_normalizations(self):
        logits = torch.tensor([8., 5.5, -20., -20., -30.], dtype=torch.bfloat16)
        logits = logits.repeat(1, 3, 1)
        xt = torch.tensor([[0, 4, 4]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            bf16 = worker.normalize_output(logits.clone(), xt, 4, -1e6, 'bf16')
            fp32 = worker.normalize_output(logits.clone(), xt, 4, -1e6, 'fp32')
        self.assertEqual(bf16.dtype, torch.bfloat16)
        self.assertEqual(fp32.dtype, torch.float32)
        self.assertGreater(float(bf16.exp()[0, 1, 0]), .925)
        self.assertLess(float(fp32.exp()[0, 1, 0]), .925)
        self.assertAlmostEqual(float(fp32.exp()[0, 1].sum()), 1., places=6)
        self.assertGreater(float(bf16.exp()[0, 1].sum()), 1.)
        for output in (bf16, fp32):
            self.assertEqual(float(output[0, 0, 0]), 0.)
            self.assertEqual(float(output.exp()[0, 0, 1:].sum()), 0.)
            self.assertEqual(float(output.exp()[0, 1, 4]), 0.)

    def test_accidental_full_fp32_forward_is_rejected(self):
        logits, xt = torch.zeros(1, 2, 3), torch.tensor([[2, 2]])
        with self.assertRaisesRegex(ValueError, 'Expected BF16'):
            worker.normalize_output(logits, xt, 2, -1e6, 'fp32')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            worker.normalize_output(logits.bfloat16(), xt, 2, -1e6, 'other')


class WeightLoadingTest(unittest.TestCase):
    def test_full_checkpoint_uses_saved_ema_and_ignores_teacher(self):
        model = TinyModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.ckpt'
            torch.save(checkpoint(model), path)
            meta = worker.load_weights(model, path, 1000)
        for parameter in model.backbone.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 3.)))
        self.assertEqual(meta['checkpoint_format'], 'full_checkpoint_with_ema')
        self.assertIsNone(model.ema)
        self.assertEqual(model._checkpoint_step, 1000)
        self.assertTrue(model.evaluation)

    def test_ema_only_export_does_not_require_or_reapply_ema(self):
        model = TinyModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step=00000500.ckpt'
            torch.save(checkpoint(model, ema_only=True, step=500), path)
            meta = worker.load_weights(model, path, 500)
        for parameter in model.backbone.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 2.)))
        self.assertEqual(meta['checkpoint_format'], 'ema_weights_only')
        self.assertEqual(model._evaluation_weights, 'ema')

    def test_step_mismatch_and_missing_ema_fail(self):
        model = TinyModel()
        before = copy.deepcopy(model.backbone.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.ckpt'
            torch.save(checkpoint(model), path)
            with self.assertRaisesRegex(ValueError, 'global_step'):
                worker.load_weights(model, path, 1248)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.backbone.state_dict()[key]))
            data = checkpoint(model)
            del data['ema']
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, 'no EMA'):
                worker.load_weights(model, path, 1000)


class PlanAndResumeTest(unittest.TestCase):
    def test_plan_rejects_input_output_overlap_and_duplicate_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint_path = root / 'weights.ckpt'
            checkpoint_path.touch()
            plan = {'schema_version': 1, 'output_dir': str(root / 'output'),
                    'snapshot_namespace': 'cdlm_checkpoint_eval', 'entries': [
                        {'source_run': 'source-one', 'family': 'CDLM', 'step': 1000,
                         'checkpoint': str(checkpoint_path),
                         'output_dir': str(root / 'output/source-one/step=1000')}]}
            self.assertEqual(worker.validate_plan(plan), root / 'output')
            duplicate = copy.deepcopy(plan)
            duplicate['entries'].append(copy.deepcopy(plan['entries'][0]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                worker.validate_plan(duplicate)
            overlap = copy.deepcopy(plan)
            overlap['output_dir'] = str(root)
            with self.assertRaisesRegex(ValueError, 'Input checkpoint'):
                worker.validate_plan(overlap)
            escape = copy.deepcopy(plan)
            escape['entries'][0]['output_dir'] = str(root / 'outside')
            with self.assertRaisesRegex(ValueError, 'inside'):
                worker.validate_plan(escape)

    def test_resume_checks_identity_and_complete_grid(self):
        identity = worker.report_identity({'source_run': 'source', 'family': 'CDLM', 'step': 1248},
                                           'checkpoint-hash', {'protocol': 'same'})
        report = {'identity': identity, 'complete': True,
                  'confidence': {norm: [{'threshold': t, 'count': 1319}
                                        for t in worker.CONFIDENCE_THRESHOLDS]
                                 for norm in worker.NORMALIZATIONS},
                  'ancestral': {'complete': True, 'points': [
                      {'steps': steps, 'seeds': [{'seed': 1, 'count': 1319}]}
                      for steps in worker.ANCESTRAL_STEPS]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            worker.write_json(path, report)
            self.assertEqual(worker.reusable_report(path, identity), report)
            other = {**identity, 'checkpoint_sha256': 'different'}
            with self.assertRaisesRegex(ValueError, 'different checkpoint or protocol'):
                worker.reusable_report(path, other)
            report['complete'] = False
            worker.write_json(path, report)
            self.assertIsNone(worker.reusable_report(path, identity))
            report['complete'] = True
            report['confidence']['bf16'].pop()
            worker.write_json(path, report)
            with self.assertRaisesRegex(ValueError, 'incomplete confidence'):
                worker.reusable_report(path, identity)

    def test_normalization_protocols_have_different_identity(self):
        entry = {'source_run': 'source', 'family': 'CDLM', 'step': 1248}
        a = worker.report_identity(entry, 'hash', {'normalization': 'fp32'})
        b = worker.report_identity(entry, 'hash', {'normalization': 'bf16'})
        self.assertNotEqual(a, b)

    def test_rank_local_failure_reaches_all_ranks(self):
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [context.Process(target=_distributed_error,
                                          args=(rank, str(Path(directory) / 'rendezvous'), queue))
                         for rank in range(2)]
            for process in processes:
                process.start()
            try:
                results = [queue.get(timeout=30) for _ in processes]
            finally:
                for process in processes:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join()
            self.assertEqual(sorted(rank for rank, _ in results), [0, 1])
            for _, message in results:
                self.assertIn('rank-one-load-error', message)
            self.assertEqual([process.exitcode for process in processes], [0, 0])


class PipelineTest(unittest.TestCase):
    def test_partial_report_reruns_and_complete_report_skips_model_work(self):
        class Model(TinyModel):
            def __init__(self, config, tokenizer, inference_only):
                super().__init__()
                self.config = config

            def to(self, device):
                return self

        class Tracker:
            def __init__(self, plan, config):
                self.points = []

            def provenance(self, *args): pass
            def confidence(self, *args): self.points.append(args[1:])
            def ancestral_point(self, *args): pass
            def ancestral_result(self, *args): pass
            def completed(self, *args): pass
            def republish(self, *args): pass
            def finish(self): pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = root / 'bundle'
            bundle.mkdir()
            data = root / 'gsm8k.json'
            data.write_text('[]')
            checkpoint_path = root / 'last.ckpt'
            torch.save(checkpoint(TinyModel()), checkpoint_path)
            output = root / 'output'
            entry = {'source_run': 'source', 'family': 'CDLM', 'step': 1000,
                     'checkpoint': str(checkpoint_path), 'output_dir': str(output / 'source/step=1000')}
            plan = {'schema_version': 1, 'output_dir': str(output),
                    'snapshot_namespace': 'eval', 'entries': [entry]}
            config = OmegaConf.create({
                'model': {'length': 512, 'force_fp32': False},
                'trainer': {'devices': 1, 'precision': 'bf16-mixed'},
                'algo': {'name': 'cdlm', 'causal_attention': False, 'time_conditioning': False},
                'data': {'train': 'tiny_gsm', 'separator': r'\n', 'tokenizer_name_or_path': 'test'},
                'sampling': {'predictor': 'ancestral_cache'}, 'seed': 1,
                'eval': {'disable_ema': False, 'gsm8k_data_path': str(data),
                         'gsm8k_num_examples': None, 'gsm8k_pareto_temperature': 0.,
                         'gsm8k_pareto_thresholds': worker.CONFIDENCE_THRESHOLDS,
                         'gsm8k_ancestral_steps': worker.ANCESTRAL_STEPS,
                         'gsm8k_ancestral_seeds': [1], 'gsm8k_ancestral_final_denoise': True,
                         'gsm8k_max_steps': None, 'gsm8k_batch_size': 16,
                         'tinygsm_exec_timeout': 5.},
                'model_weights_dir': str(bundle), 'checkpointing': {'snapshot': True},
            })
            fail = [True]
            calls = []

            def confidence(model, examples):
                threshold = float(config.eval.gsm8k_pareto_thresholds[0])
                if fail[0] and model.normalization == 'fp32' and threshold == .55:
                    raise RuntimeError('interrupted at second normalization')
                calls.append((model.normalization, threshold))
                model._gsm8k_sample_records = []
                return [{'threshold': threshold, 'accuracy': .5, 'correct': 659,
                         'count': 1319, 'nfe': 5.}]

            def ancestral(model, examples, *, distributed, on_point, cfg):
                self.assertEqual(model.normalization, 'fp32')
                points = []
                for steps in worker.ANCESTRAL_STEPS:
                    point = {'steps': steps, 'accuracy_mean': .5, 'accuracy_std': None,
                             'nfe_mean': steps + 1., 'nfe_std': None, 'num_seeds': 1,
                             'seeds': [{'seed': 1, 'count': 1319, 'correct': 659,
                                        'accuracy': .5, 'nfe': steps + 1.}]}
                    points.append(point)
                    on_point(point)
                return {'points': points, 'protocol': {'final_denoise': True}}

            with ExitStack() as stack:
                def patched(target, **kwargs):
                    return stack.enter_context(patch(target, **kwargs))
                patched('torch.distributed.get_rank', return_value=0)
                patched('torch.distributed.get_world_size', return_value=1)
                patched('torch.distributed.all_gather_object',
                        side_effect=lambda dest, value: dest.__setitem__(0, value))
                patched('torch.distributed.broadcast_object_list', return_value=None)
                patched('torch.cuda.current_device', return_value=0)
                patched('torch.cuda.get_device_name', return_value='CPU-test-device')
                patched('dataloader.get_tokenizer', return_value=SimpleNamespace(get_vocab=lambda: {'a': 0}))
                patched('model_assets.configure_model_weights', return_value=None)
                patched('sdtt_eval.load_examples', return_value=[{}] * 1319)
                patched('sdtt_eval.sweep', side_effect=confidence)
                patched('cdlm_ancestral_sweep.sweep', side_effect=ancestral)
                stack.enter_context(patch.object(worker, '_model_type', return_value=Model))
                stack.enter_context(patch.object(worker, 'Publisher', Tracker))
                snapshots = stack.enter_context(patch.object(worker, '_publish_snapshot', return_value=True))
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    worker.run(plan, config)
                import json
                result_path = Path(entry['output_dir']) / 'result.json'
                partial = json.loads(result_path.read_text())
                self.assertFalse(partial['complete'])
                self.assertEqual(len(partial['confidence']['bf16']), 13)
                self.assertEqual(len(partial['confidence']['fp32']), 3)
                snapshots.assert_not_called()
                fail[0] = False
                calls.clear()
                worker.run(plan, config)
                complete = json.loads(result_path.read_text())
                self.assertTrue(complete['complete'])
                self.assertEqual(len(calls), 26)
                self.assertEqual(len(complete['ancestral']['points']), 6)
                self.assertEqual(snapshots.call_count, 2)
                calls.clear()
                with patch.object(worker, 'load_weights', side_effect=AssertionError('should skip loading')):
                    worker.run(plan, config)
                self.assertEqual(calls, [])
                self.assertEqual(snapshots.call_count, 4)


if __name__ == '__main__':
    unittest.main()
