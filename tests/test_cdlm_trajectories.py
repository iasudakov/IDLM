"""CPU tests for actual CDLM collection, provenance, sharding and restart."""

from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from cdlm_trajectories import (
    CDLMTrajectoryDataset, SCHEMA_VERSION, build_cache_identity, collect_trajectory_batch,
    prepare_trajectories, read_cache_manifest,
)


class Tokenizer:
    vocab_size = 6
    mask_token_id = None
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def get_vocab(self):
        return {str(i): i for i in range(self.vocab_size)}


class Teacher(torch.nn.Module):
    """Context-dependent model exposing exactly MDLM's final linear contract."""

    mask_index = 6

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.embedding = torch.nn.Embedding(7, 4)
        self.backbone.output_layer = torch.nn.Module()
        self.backbone.output_layer.linear = torch.nn.Linear(4, 7)
        with torch.no_grad():
            self.backbone.embedding.weight.copy_(torch.arange(28).reshape(7, 4) % 5 - 2)
            self.backbone.output_layer.linear.weight.copy_(
                torch.arange(28).reshape(7, 4).float() / 20 - .5)
            self.backbone.output_layer.linear.bias.copy_(torch.arange(7).float() / 20)
        self.history = []

    def forward(self, xt, sigma):
        hidden = self.backbone.embedding(xt)
        hidden = hidden + hidden.sum(dim=1, keepdim=True)
        logits = self.backbone.output_layer.linear(hidden).float()
        logits[..., self.mask_index] = -1e6
        result = logits.log_softmax(dim=-1)
        self.history.append({'ids': xt.detach().clone(), 'hidden': hidden.detach().clone(),
                             'log_probs': result.detach().clone()})
        return result


def inputs():
    return (torch.tensor([[0, 4, 3, 1, 2, 2], [0, 5, 4, 3, 1, 2]]),
            torch.tensor([[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0]], dtype=torch.bool))


def configuration(directory):
    return OmegaConf.create({
        'seed': 17,
        'training': {'finetune_path': str(directory / 'teacher.ckpt')},
        'model': {'hidden_size': 4, 'length': 6, 'n_blocks': 1},
        'data': {'cache_dir': str(directory), 'tokenizer_name_or_path': 'fixture-tokenizer',
                 'separator': '\\n', 'train_on_prompt': False, 'train_on_pad': True,
                 'wrap': False, 'filter_too_long': True, 'val_ratio': .01, 'val_seed': 42},
        'algo': {'trajectory_cache_dir': str(directory / 'cache'),
                 'trajectory_num_examples': 3, 'trajectory_batch_size': 2,
                 'trajectory_shard_size': 2, 'trajectory_temperatures': [0.0, 0.5]},
    })


def data():
    ids, valid = inputs()
    return [{'input_ids': ids[i % 2].clone(), 'attention_mask': valid[i % 2].clone()}
            for i in range(4)]


def distributed_preparation_worker(rank, directory, failure_stage):
    """Two real Gloo peers must both return the same preparation failure."""
    torch.set_num_threads(1)
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    directory = Path(directory)
    try:
        torch.distributed.init_process_group(
            'gloo', init_method='file://' + str(directory / 'rendezvous'),
            rank=rank, world_size=2, timeout=timedelta(seconds=15))
        cfg = configuration(directory)
        teacher = Teacher()
        if failure_stage == 'identity' and rank == 0:
            cfg.training.finetune_path = str(directory / 'missing-teacher.ckpt')
        if failure_stage == 'shard' and rank == 1:
            def broken_forward(*args, **kwargs):
                raise RuntimeError('injected rank-one teacher failure')
            teacher.forward = broken_forward
        cache_dir = prepare_trajectories(teacher, cfg, Tokenizer(), data())
        status = {'ok': True, 'cache': cache_dir}
    except Exception as error:
        status = {'ok': False, 'error': str(error)}
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    (directory / f'rank-{rank}.json').write_text(json.dumps(status))


class TrajectoryCollectionTests(unittest.TestCase):
    def test_monotone_global_reveals_hidden_reconstruction_and_prompt_protection(self):
        teacher = Teacher().train()
        ids, valid = inputs()
        result = collect_trajectory_batch(teacher, ids, valid,
                                           temperatures=[0.0, 0.5], seeds=[9, 41])
        self.assertTrue(teacher.training)
        self.assertEqual(len(teacher.history), 3)
        self.assertEqual(result['teacher_hidden'].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result['final_tokens'][~valid], ids[~valid]))
        self.assertTrue(result['reveal_order'][~valid].eq(-1).all())
        self.assertFalse(result['final_tokens'].eq(teacher.mask_index).any())
        head = teacher.backbone.output_layer.linear
        for row in range(len(ids)):
            count = int(valid[row].sum())
            self.assertEqual(sorted(result['reveal_order'][row, valid[row]].tolist()), list(range(count)))
            for step in range(count):
                position = result['reveal_order'][row].eq(step).nonzero().item()
                recorded = teacher.history[step]
                self.assertTrue(torch.equal(result['teacher_hidden'][row, position].float(),
                                            recorded['hidden'][row, position]))
                logits = head(result['teacher_hidden'][row, position].float())
                logits[teacher.mask_index] = -1e6
                torch.testing.assert_close(logits.log_softmax(-1),
                                           recorded['log_probs'][row, position])
                # Reconstruct the stored trajectory from ranks and final tokens.
                state = torch.where(valid[row], teacher.mask_index, ids[row])
                already_revealed = valid[row] & result['reveal_order'][row].lt(step)
                state[already_revealed] = result['final_tokens'][row, already_revealed]
                self.assertTrue(torch.equal(state, recorded['ids'][row]))
        self.assertEqual(len(head._forward_pre_hooks), 0)

    def test_individual_seeds_are_independent_of_microbatch_composition(self):
        ids, valid = inputs()
        grouped = collect_trajectory_batch(Teacher(), ids, valid,
                                            temperatures=[.5, .5], seeds=[42, 73])
        separate = [collect_trajectory_batch(Teacher(), ids[i:i + 1], valid[i:i + 1],
                                             temperatures=[.5], seeds=[seed])
                    for i, seed in enumerate([42, 73])]
        for key, values in grouped.items():
            self.assertTrue(torch.equal(values, torch.cat([row[key] for row in separate])))

    def test_eos_does_not_end_extraction(self):
        teacher = Teacher()
        with torch.no_grad():
            teacher.backbone.output_layer.linear.bias[1] = 1000
        ids, valid = inputs()
        result = collect_trajectory_batch(teacher, ids, valid,
                                           temperatures=[0, 0], seeds=[1, 2])
        self.assertEqual(len(teacher.history), 3)
        self.assertTrue(result['final_tokens'][valid].eq(1).all())
        self.assertTrue(result['reveal_order'][valid].ge(0).all())

    def test_gumbel_sampling_retains_high_uniform_tail(self):
        teacher = Teacher()
        with torch.no_grad():
            head = teacher.backbone.output_layer.linear
            head.weight.zero_()
            head.bias.fill_(-100)
            head.bias[3] = 0
            head.bias[4] = -15
        ids, _ = inputs()
        valid = torch.zeros_like(ids[:1], dtype=torch.bool)
        valid[0, 2] = True

        def uniforms(size, **kwargs):
            value = torch.full(size, .5, device=kwargs['device'], dtype=kwargs['dtype'])
            value[..., 4] = torch.nextafter(torch.tensor(1.), torch.tensor(0.))
            value[..., 2] = 0
            return value

        with patch('cdlm_trajectories.torch.rand', side_effect=uniforms):
            result = collect_trajectory_batch(teacher, ids[:1], valid,
                                               temperatures=[1.0], seeds=[7])
        # The real float32 upper tail overcomes the 15-logit deficit. Clipping
        # U to 1-1e-6 incorrectly suppresses that valid rare-token proposal.
        self.assertEqual(int(result['final_tokens'][0, 2]), 4)

    def test_invalid_inputs_and_hook_cleanup_on_forward_error(self):
        ids, valid = inputs()
        teacher = Teacher()
        with self.assertRaisesRegex(ValueError, 'eligible'):
            collect_trajectory_batch(teacher, ids, torch.zeros_like(valid),
                                       temperatures=[0, 0], seeds=[1, 2])
        with patch.object(teacher, 'forward', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                collect_trajectory_batch(teacher, ids, valid,
                                           temperatures=[0, 0], seeds=[1, 2])
        self.assertEqual(len(teacher.backbone.output_layer.linear._forward_pre_hooks), 0)


class TrajectoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        (self.directory / 'teacher.ckpt').write_bytes(b'original teacher checkpoint fixture')
        self.config = configuration(self.directory)
        self.tokenizer = Tokenizer()

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_identity_and_single_shard_residency(self):
        identity = build_cache_identity(self.config, self.tokenizer, data())
        cache_dir = prepare_trajectories(Teacher(), self.config, self.tokenizer, data())
        manifest = read_cache_manifest(cache_dir)
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['identity'], identity)
        cached = CDLMTrajectoryDataset(cache_dir, expected_fingerprint=identity)
        self.assertEqual(len(cached), 6)
        entries = [cached[i] for i in range(len(cached))]
        self.assertEqual(len({row['example_index'] for row in entries}), 3)
        self.assertEqual([row['temperature'] for row in entries], [0., .5] * 3)
        for entry in entries:
            source = data()[entry['example_index']]
            self.assertTrue(torch.equal(entry['input_ids'], source['input_ids']))
            self.assertTrue(torch.equal(entry['attention_mask'], source['attention_mask']))
        self.assertEqual(len(cached._resident_items), 2)
        first = cached[0]
        first['input_ids'].fill_(99)
        self.assertFalse(cached[0]['input_ids'].eq(99).any())
        self.assertTrue(torch.equal(cached[-1]['final_tokens'], entries[-1]['final_tokens']))
        with self.assertRaises(IndexError):
            cached[6]

    def test_resume_skips_finished_shards_and_recovers_missing_receipt(self):
        import cdlm_trajectories
        real = cdlm_trajectories.collect_trajectory_batch
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated stop')
            return real(*args, **kwargs)

        with patch('cdlm_trajectories.collect_trajectory_batch', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'simulated stop'):
                prepare_trajectories(Teacher(), self.config, self.tokenizer, data())
        cache_dir = Path(self.config.algo.trajectory_cache_dir)
        first_shard = cache_dir / 'part-00000000.pt'
        first_stat = first_shard.stat().st_mtime_ns
        self.assertEqual(read_cache_manifest(cache_dir)['status'], 'building')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            CDLMTrajectoryDataset(cache_dir)
        # Simulate stopping between tensor rename and receipt rename.
        first_shard.with_suffix('.json').unlink()
        with patch('cdlm_trajectories.collect_trajectory_batch', wraps=real) as collect:
            prepare_trajectories(Teacher(), self.config, self.tokenizer, data())
            self.assertEqual(collect.call_count, 2)
        self.assertEqual(first_shard.stat().st_mtime_ns, first_stat)
        self.assertEqual(read_cache_manifest(cache_dir)['status'], 'complete')
        with patch('cdlm_trajectories.collect_trajectory_batch', side_effect=AssertionError('should skip')):
            prepare_trajectories(Teacher(), self.config, self.tokenizer, data())

    def test_explicit_rank_partition_and_completion(self):
        cache_dir = prepare_trajectories(Teacher(), self.config, self.tokenizer, data(),
                                         rank=0, world_size=2)
        self.assertEqual(read_cache_manifest(cache_dir)['status'], 'building')
        self.assertEqual(len(list(Path(cache_dir).glob('*.pt'))), 2)
        prepare_trajectories(Teacher(), self.config, self.tokenizer, data(), rank=1, world_size=2)
        self.assertEqual(read_cache_manifest(cache_dir)['status'], 'complete')
        self.assertEqual(len(CDLMTrajectoryDataset(cache_dir)), 6)

    def test_teacher_content_and_data_and_config_mismatches_are_rejected(self):
        cache_dir = prepare_trajectories(Teacher(), self.config, self.tokenizer, data())
        identity = build_cache_identity(self.config, self.tokenizer, data())
        relocated = self.directory / 'identical.ckpt'
        relocated.write_bytes((self.directory / 'teacher.ckpt').read_bytes())
        self.config.training.finetune_path = str(relocated)
        self.assertEqual(build_cache_identity(self.config, self.tokenizer, data()), identity)
        relocated.write_bytes(b'a different teacher')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            prepare_trajectories(Teacher(), self.config, self.tokenizer, data())
        self.config.training.finetune_path = str(self.directory / 'teacher.ckpt')
        altered = data()
        altered[0]['input_ids'][2] = 5
        changed = build_cache_identity(self.config, self.tokenizer, altered)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            CDLMTrajectoryDataset(cache_dir, changed)
        self.config.seed += 1
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            prepare_trajectories(Teacher(), self.config, self.tokenizer, data())

    def test_hidden_dtype_is_validated_and_recorded_in_new_schema(self):
        identity = build_cache_identity(self.config, self.tokenizer, data())
        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertEqual(identity['parameters']['rollout']['hidden_dtype'], 'bfloat16')
        self.assertIn('open_uniform_v2', identity['parameters']['rollout']['rng'])
        self.config.algo.teacher_hidden_dtype = 'bf16'
        self.assertEqual(build_cache_identity(self.config, self.tokenizer, data()), identity)
        self.config.algo.teacher_hidden_dtype = 'float32'
        with self.assertRaisesRegex(ValueError, 'only bfloat16'):
            prepare_trajectories(Teacher(), self.config, self.tokenizer, data())

    def test_corrupted_shard_and_forged_manifest_are_rejected(self):
        cache_dir = Path(prepare_trajectories(Teacher(), self.config, self.tokenizer, data()))
        with open(cache_dir / 'part-00000000.pt', 'ab') as target:
            target.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            CDLMTrajectoryDataset(cache_dir)[0]
        path = cache_dir / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['identity']['parameters']['teacher_checkpoint_sha256'] = 'changed'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'identity'):
            read_cache_manifest(cache_dir)


@unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                     'Gloo distributed backend is unavailable')
class DistributedPreparationTests(unittest.TestCase):
    def _run_workers(self, failure_stage):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'teacher.ckpt').write_bytes(b'original teacher checkpoint fixture')
            torch.multiprocessing.spawn(distributed_preparation_worker,
                                        args=(str(directory), failure_stage), nprocs=2, join=True)
            statuses = [json.loads((directory / f'rank-{rank}.json').read_text()) for rank in range(2)]
            if failure_stage is None:
                self.assertTrue(all(row['ok'] for row in statuses), statuses)
                self.assertEqual(read_cache_manifest(directory / 'cache')['status'], 'complete')
            else:
                self.assertFalse(any(row['ok'] for row in statuses), statuses)
                self.assertEqual(statuses[0]['error'], statuses[1]['error'])
                expected = ('cache identity construction failed (rank 0:' if failure_stage == 'identity'
                            else 'trajectory shard generation failed (rank 1:')
                self.assertIn(expected, statuses[0]['error'])
                if failure_stage == 'shard':
                    self.assertIn('injected rank-one teacher failure', statuses[0]['error'])
                    self.assertEqual(read_cache_manifest(directory / 'cache')['status'], 'building')

    def test_rank_zero_identity_error_reaches_all_peers(self):
        self._run_workers('identity')

    def test_shard_error_reaches_all_peers_before_final_barrier(self):
        self._run_workers('shard')

    def test_successful_collective_preparation_completes_cache(self):
        self._run_workers(None)


if __name__ == '__main__':
    unittest.main()
