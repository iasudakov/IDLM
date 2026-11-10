"""CPU checks for published/local MMD evaluation; no real Hub or GPU calls."""

from contextlib import contextmanager, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import cdlm_checkpoint_assets as assets
import cdlm_checkpoint_eval as common
import cdlm_checkpoint_eval_worker as worker
import mmd_checkpoint_eval as evaluator


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def payload(*, step=2300, ema_only=True):
    result = {
        'global_step': step, 'epoch': 3,
        'state_dict': {'backbone.weight': torch.full((2, 2), 2.),
                       'backbone.bias': torch.full((2,), 2.)},
    }
    if ema_only:
        result['checkpoint_weights'] = 'ema'
    else:
        result['state_dict']['teacher_head.weight'] = torch.zeros(10, 10)
        result['ema'] = {'shadow_params': [torch.full((2, 2), 3.), torch.full((2,), 3.)]}
    return result


class TinyModel:
    def __init__(self):
        self.backbone = torch.nn.Linear(2, 2)
        self.ema = object()

    def _get_parameters(self):
        return self.backbone.parameters()

    def eval(self):
        self.backbone.eval()
        return self


class Fixture(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='MMD eval, v=1 '))).resolve()
        self.code = self.base / 'code with spaces'
        self.code.mkdir()
        self.weights = self.base / 'weights step=2300.ckpt'
        self.output = self.code / 'mmd_checkpoint_eval'
        self.write()

    def write(self, value=None):
        torch.save(payload() if value is None else value, self.weights)
        return self.weights

    def offline_assets(self):
        bundle = self.code / 'cache_dir/tinygsm_mmd_offline_bundle'
        tokenizer = bundle / 'tokenizer'
        tokenizer.mkdir(parents=True)
        (tokenizer / 'tokenizer.json').write_text('{}')
        (bundle / 'manifest.json').write_text(json.dumps({'assets': [{
            'repo_id': 'HuggingFaceTB/SmolLM-135M', 'folder': 'tokenizer'}]}))
        data = self.code / 'data/gsm8k_test.json'
        data.parent.mkdir()
        data.write_text(json.dumps([{'prompt': f'Question {i}?', 'response_ground_truth': '#### 42'}
                                    for i in range(1319)]))


class CheckpointInspectionTest(Fixture):
    def test_export_metadata_and_worker_use_saved_step_and_ema_bytes(self):
        before = self.weights.read_bytes()
        info = evaluator.inspect_checkpoint(self.weights)
        self.assertEqual(info['global_step'], 2300)
        self.assertEqual(info['epoch'], 3)
        self.assertEqual(info['checkpoint_format'], 'ema_weights_only')
        self.assertEqual(info['size_bytes'], len(before))
        self.assertEqual(info['sha256'], hashlib.sha256(before).hexdigest())
        model = TinyModel()
        worker.load_weights(model, self.weights, info['global_step'])
        for parameter in model.backbone.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 2.)))
        self.assertIsNone(model.ema)
        self.assertEqual(self.weights.read_bytes(), before)

    def test_full_checkpoint_ema_ignores_frozen_teacher_tensors(self):
        self.write(payload(ema_only=False))
        info = evaluator.inspect_checkpoint(self.weights)
        self.assertEqual(info['checkpoint_format'], 'full_checkpoint_with_ema')
        model = TinyModel()
        worker.load_weights(model, self.weights, info['global_step'])
        for parameter in model.backbone.parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 3.)))

    def test_invalid_steps_never_fall_back_to_an_assumed_paper_step(self):
        for value in (None, True, -1, 7.0, '7000'):
            with self.subTest(value=value):
                data = payload(step=value)
                if value is None:
                    del data['global_step']
                self.write(data)
                with self.assertRaises(ValueError):
                    evaluator.inspect_checkpoint(self.weights)
        self.write(payload(step=0))
        self.assertEqual(evaluator.inspect_checkpoint(self.weights)['global_step'], 0)

    def test_missing_empty_unreadable_and_non_model_payloads_fail(self):
        self.weights.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            evaluator.inspect_checkpoint(self.weights)
        for data in (b'', b'not a torch checkpoint'):
            with self.subTest(data=data):
                self.weights.write_bytes(data)
                with self.assertRaises((FileNotFoundError, ValueError)):
                    evaluator.inspect_checkpoint(self.weights)
        for data in ([], {'global_step': 1, 'state_dict': {}},
                     {'global_step': 1, 'state_dict': {'backbone.weight': 'wrong'},
                      'checkpoint_weights': 'ema'}):
            with self.subTest(payload=type(data).__name__):
                self.write(data)
                with self.assertRaises(ValueError):
                    evaluator.inspect_checkpoint(self.weights)

    def test_raw_only_and_malformed_ema_fail_before_gpu(self):
        for ema in (None, {}, {'shadow_params': []}, {'shadow_params': ['not a tensor']},
                    {'shadow_params': [torch.ones(5, 5), torch.ones(2)]}):
            with self.subTest(ema=str(ema)):
                data = payload(ema_only=False)
                if ema is None:
                    del data['ema']
                else:
                    data['ema'] = ema
                self.write(data)
                with self.assertRaises(ValueError):
                    evaluator.inspect_checkpoint(self.weights)


class SinglePlanTest(Fixture):
    def test_published_checkpoint_plan_uses_common_worker_without_training_state(self):
        before = self.weights.read_bytes()
        plan = evaluator.build_single_plan(self.weights, assets.MMD_SOURCE, self.output)
        self.assertEqual(plan['schema_version'], 1)
        self.assertEqual(plan['snapshot_namespace'], 'mmd_checkpoint_eval')
        self.assertEqual(len(plan['entries']), 1)
        entry = plan['entries'][0]
        self.assertEqual(entry['source_run'], 'hf-yresearch-mdlm-mmd-tinygsm')
        self.assertEqual(entry['family'], 'MDLM to MMD')
        self.assertEqual(entry['step'], 2300)
        self.assertEqual(Path(entry['checkpoint']), self.weights)
        self.assertEqual(worker.validate_plan(plan), self.output)
        self.assertIn(assets.MMD_SOURCE, json.dumps(plan))
        self.assertFalse(self.output.exists())
        self.assertEqual(self.weights.read_bytes(), before)

    def test_custom_source_ids_are_stable_safe_and_distinct(self):
        first = evaluator.build_single_plan(self.weights, str(self.weights), self.output)
        again = evaluator.build_single_plan(self.weights, str(self.weights), self.output)
        other = evaluator.build_single_plan(self.weights, 'hf://owner/model/other.ckpt', self.output)
        a, b, c = [plan['entries'][0]['source_run'] for plan in (first, again, other)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.startswith('mmd-checkpoint-'))
        self.assertNotIn('/', a)
        worker.validate_plan(first)

    def test_output_overlap_and_unsafe_namespaces_are_rejected(self):
        for output in (self.base, self.weights):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                evaluator.build_single_plan(self.weights, assets.MMD_SOURCE, output)
        for namespace in ('', '../training', '/absolute', 'eval/../training'):
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                evaluator.build_single_plan(self.weights, assets.MMD_SOURCE, self.output,
                                            snapshot_namespace=namespace)


class SingleCheckpointCLITest(Fixture):
    def setUp(self):
        super().setUp()
        self.offline_assets()
        self.enterContext(working_directory(self.code))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_default_download_resolution_happens_once_then_workers_are_offline(self):
        observed = []

        def prepare(source, **kwargs):
            observed.append((source, kwargs, dict(os.environ)))
            return str(self.weights)

        with patch.object(evaluator, 'prepare_checkpoint', side_effect=prepare) as resolver, \
                patch.object(evaluator, 'execute_plan', return_value=0) as execute:
            self.assertEqual(evaluator.main(['checkpointing.snapshot=False']), 0)
        resolver.assert_called_once()
        source, kwargs, preparation_env = observed[0]
        self.assertEqual(source, assets.MMD_SOURCE)
        self.assertNotEqual(preparation_env.get('HF_HUB_OFFLINE'), '1')
        self.assertEqual(Path(kwargs['code_dir']), self.code)
        plan, config = execute.call_args.args[:2]
        self.assertEqual(plan['entries'][0]['step'], 2300)
        self.assertEqual(plan['snapshot_namespace'], 'mmd_checkpoint_eval')
        self.assertEqual(config.checkpointing.snapshot_namespace, 'mmd_checkpoint_eval')
        self.assertEqual(config.trainer.devices, 4)
        self.assertEqual(config.trainer.precision, 'bf16-mixed')
        self.assertFalse(config.model.force_fp32)
        self.assertFalse(config.eval.disable_ema)
        self.assertEqual(list(config.eval.gsm8k_pareto_thresholds),
                         [.325, .4, .475, .55, .625, .7, .775, .85, .925, .95, .98, .99, .995])
        self.assertEqual(list(config.eval.gsm8k_ancestral_steps), [4, 8, 16, 32, 48, 64])
        self.assertEqual(list(config.eval.gsm8k_ancestral_seeds), [1])
        self.assertTrue(config.eval.gsm8k_ancestral_final_denoise)
        self.assertEqual(config.eval.gsm8k_batch_size, 16)
        self.assertEqual(config.training.finetune_path, '')
        self.assertEqual(config.algo.name, 'cdlm')  # Common inference adapter only.
        self.assertEqual(execute.call_args.kwargs['devices'], 4)
        child = execute.call_args.kwargs['worker_env']
        self.assertEqual(child['HF_HUB_OFFLINE'], '1')
        self.assertEqual(child['HF_DATASETS_OFFLINE'], '1')

    def test_explicit_local_checkpoint_uses_no_hub_or_training_snapshot(self):
        with patch.object(assets, 'hf_hub_download', side_effect=AssertionError('No Hub request')), \
                patch.object(evaluator, 'execute_plan', return_value=0) as execute:
            self.assertEqual(evaluator.main(['--checkpoint', str(self.weights),
                                            '--no-restore-results', 'checkpointing.snapshot=False']), 0)
        entry = execute.call_args.args[0]['entries'][0]
        self.assertEqual(Path(entry['checkpoint']), self.weights)
        self.assertFalse(execute.call_args.kwargs['restore_results'])

    def test_checkpoint_env_and_inherited_offline_setting_are_respected(self):
        with patch.dict(os.environ, {'MMD_EVAL_CHECKPOINT': str(self.weights), 'HF_HUB_OFFLINE': '1'}), \
                patch.object(evaluator, 'prepare_checkpoint', return_value=str(self.weights)) as prepare, \
                patch.object(evaluator, 'execute_plan', return_value=0):
            self.assertEqual(evaluator.main(['checkpointing.snapshot=False']), 0)
            self.assertEqual(prepare.call_args.args[0], str(self.weights))
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')

    def test_invalid_checkpoint_or_offline_assets_fail_before_launch(self):
        broken = payload()
        del broken['global_step']
        self.write(broken)
        with patch.object(evaluator, 'prepare_checkpoint', return_value=str(self.weights)), \
                patch.object(evaluator, 'execute_plan') as execute:
            with self.assertRaises(ValueError):
                evaluator.main(['checkpointing.snapshot=False'])
            execute.assert_not_called()
        (self.code / 'data/gsm8k_test.json').unlink()
        with patch.object(evaluator, 'prepare_checkpoint') as prepare, \
                patch.object(evaluator, 'execute_plan') as execute:
            with self.assertRaises(FileNotFoundError):
                evaluator.main(['checkpointing.snapshot=False'])
            prepare.assert_not_called()
            execute.assert_not_called()

    def test_empty_output_argument_or_environment_fails_before_asset_preparation(self):
        for args, env in ((['--output-dir', ''], {}), ([], {'MMD_EVAL_OUTPUT_DIR': ''})):
            with self.subTest(args=args, env=env), patch.dict(os.environ, env), \
                    patch.object(evaluator, 'compose_evaluation_config') as compose, \
                    patch.object(evaluator, 'prepare_checkpoint') as prepare, \
                    patch.object(evaluator, 'execute_plan') as execute:
                with self.assertRaises(ValueError):
                    evaluator.main(args)
                compose.assert_not_called()
                prepare.assert_not_called()
                execute.assert_not_called()

    def test_plan_only_writes_single_checkpoint_plan_without_gpu_subprocess(self):
        with patch.object(evaluator, 'prepare_checkpoint', return_value=str(self.weights)), \
                patch.object(common, 'restore_evaluation_output', return_value=False), \
                patch.object(common.subprocess, 'run', side_effect=AssertionError('No GPU process')):
            self.assertEqual(evaluator.main(['--plan-only', 'checkpointing.snapshot=False']), 0)
        plan = json.loads((self.output / 'plan.json').read_text())
        self.assertEqual([row['step'] for row in plan['entries']], [2300])
        self.assertTrue((self.output / 'evaluation_config.yaml').is_file())

    def test_launch_passes_offline_environment_and_exports_all_protocol_points(self):
        calls = []

        def fake_worker(command, *, check, env):
            calls.append((command, check, env))
            plan_path = Path(command[command.index('--plan') + 1])
            plan = json.loads(plan_path.read_text())
            entry = plan['entries'][0]
            target = Path(entry['output_dir'])
            target.mkdir(parents=True)
            point = {'accuracy': 2 / 1319, 'nfe': 8., 'correct': 2, 'count': 1319}
            report = {
                'complete': True,
                'identity': {'checkpoint_sha256': plan['checkpoint_metadata']['sha256']},
                'protocol': {'seed': 1},
                'confidence': {norm: [{**point, 'threshold': threshold}
                                     for threshold in worker.CONFIDENCE_THRESHOLDS]
                               for norm in worker.NORMALIZATIONS},
                'ancestral': {'points': [{'steps': steps, 'seeds': [{**point, 'seed': 1}]}
                                         for steps in worker.ANCESTRAL_STEPS]},
            }
            (target / 'result.json').write_text(json.dumps(report))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(evaluator, 'prepare_checkpoint', return_value=str(self.weights)), \
                patch.object(common, 'restore_evaluation_output', return_value=False), \
                patch.object(common.subprocess, 'run', side_effect=fake_worker):
            self.assertEqual(evaluator.main(['checkpointing.snapshot=False']), 0)
        self.assertEqual(len(calls), 1)
        command, check, child = calls[0]
        self.assertIn('--nproc_per_node=4', command)
        self.assertIn('cdlm_checkpoint_eval_worker', command)
        self.assertFalse(check)
        self.assertEqual(child['HF_HUB_OFFLINE'], '1')
        self.assertEqual(child['HF_DATASETS_OFFLINE'], '1')
        self.assertNotIn('HF_HUB_OFFLINE', os.environ)
        with (self.output / 'summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 32)
        self.assertEqual({row['checkpoint_step'] for row in rows}, {'2300'})
        confidence = [row for row in rows if row['sampler'] == 'confidence']
        self.assertEqual(len(confidence), 26)
        self.assertEqual({row['logit_normalization'] for row in confidence}, {'bf16', 'fp32'})
        self.assertTrue({'0.95', '0.99', '0.995'} <= {row['threshold'] for row in confidence})


class LauncherTest(unittest.TestCase):
    def test_nirvana_launcher_has_valid_bash_syntax(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/eval_mmd_checkpoint_nirvana.sh'
        subprocess.run(['bash', '-n', str(script)], check=True)

    def test_launcher_preserves_arguments_and_proxy_override_without_forcing_offline(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/eval_mmd_checkpoint_nirvana.sh'
        with tempfile.TemporaryDirectory(prefix='mmd launcher ') as directory:
            root = Path(directory).resolve()
            capture, stub = root / 'capture.json', root / 'python stub'
            stub.write_text('#!' + sys.executable + '\n'
                            'import json, os, sys\n'
                            'from pathlib import Path\n'
                            'Path(os.environ["CAPTURE"]).write_text(json.dumps({\n'
                            ' "argv": sys.argv[1:], "cwd": os.getcwd(),\n'
                            ' "proxy": os.environ.get("CDLM_DOWNLOAD_PROXY"),\n'
                            ' "offline": os.environ.get("HF_HUB_OFFLINE")}))\n')
            stub.chmod(0o755)
            arguments = ['--checkpoint', 'weights with spaces = 7000.ckpt',
                         'logger.name=literal $(not_a_command)']
            for configured, expected in ((None, 'http://survival-mode.klg.yp-c.yandex.net:6953'),
                                         ('', ''), ('http://user-proxy:1234', 'http://user-proxy:1234')):
                with self.subTest(proxy=configured):
                    env = dict(os.environ, PYTHON=str(stub), MMD_CODE_DIR=str(root), CAPTURE=str(capture))
                    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'CDLM_DOWNLOAD_PROXY'):
                        env.pop(key, None)
                    if configured is not None:
                        env['CDLM_DOWNLOAD_PROXY'] = configured
                    subprocess.run(['bash', str(script), *arguments], env=env, check=True)
                    observed = json.loads(capture.read_text())
                    self.assertEqual(observed['argv'], ['-u', '-m', 'mmd_checkpoint_eval', *arguments])
                    self.assertEqual(observed['cwd'], str(root))
                    self.assertEqual(observed['proxy'], expected)
                    self.assertIsNone(observed['offline'])


if __name__ == '__main__':
    unittest.main()
