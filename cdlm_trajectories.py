"""Offline, full-attention CDLM trajectories for the TinyGSM MDLM teacher.

The reveal rule and hidden-state buffer follow SqueezeAILab/CDLM at commit
708800ab0df03e2c284ab2cb358e42652a5e85a5, CDLM-train/utils/trajectory.py
(MIT). Unlike its block-wise rollout, this adaptation chooses globally among
eligible answer tokens. A rank and final token per position encode every
intermediate state without storing a quadratic token-trajectory tensor.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 2
SOURCE_COMMIT = '708800ab0df03e2c284ab2cb358e42652a5e85a5'


def _plain(value):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json_bytes(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(source):
    if source.startswith('hf://'):
        from huggingface_hub import hf_hub_download
        parts = source[5:].split('/', 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError('Expected hf://owner/repository/checkpoint-file.')
        return Path(hf_hub_download(repo_id='/'.join(parts[:2]), filename=parts[2]))
    path = os.path.expanduser(source)
    try:
        from hydra.utils import to_absolute_path
        path = to_absolute_path(path)
    except ImportError:
        path = os.path.abspath(path)
    return Path(path)


def _settings(config, dataset_length):
    cfg = config.algo
    requested = cfg.get('trajectory_num_examples', 20000)
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError('algo.trajectory_num_examples must be positive or null.')
    if dataset_length <= 0:
        raise ValueError('Cannot prepare trajectories from an empty training dataset.')
    count = dataset_length if requested is None else min(requested, dataset_length)
    batch_size = cfg.get('trajectory_batch_size', 8)
    shard_size = cfg.get('trajectory_shard_size', 8)
    if any(type(x) is not int or x <= 0 for x in (batch_size, shard_size)):
        raise ValueError('Trajectory batch and shard sizes must be positive integers.')
    temperatures = [float(t) for t in cfg.get('trajectory_temperatures', [0.0, 0.5])]
    if (not temperatures or len(set(temperatures)) != len(temperatures)
            or any(not math.isfinite(t) or t < 0 for t in temperatures)):
        raise ValueError('Trajectory temperatures must be distinct finite nonnegative values.')
    hidden_dtype = str(cfg.get('teacher_hidden_dtype', 'bfloat16'))
    if hidden_dtype not in ('bfloat16', 'bf16'):
        raise ValueError('algo.teacher_hidden_dtype supports only bfloat16 (bf16).')
    return {'requested_examples': requested, 'num_examples': count,
            'batch_size': batch_size, 'shard_size': shard_size,
            'temperatures': temperatures, 'seed': int(config.seed),
            'hidden_dtype': 'bfloat16'}


def _selected_indices(length, count, seed):
    # random.sample(range(...)) needs memory proportional to the selected
    # subset, rather than materializing a permutation of the full TinyGSM set.
    return random.Random(seed).sample(range(length), count)


def _tokenizer_identity(config, tokenizer):
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None and hasattr(backend, 'to_str'):
        implementation = backend.to_str().encode('utf-8')
    elif hasattr(tokenizer, 'get_vocab'):
        implementation = _json_bytes(tokenizer.get_vocab())
    else:
        raise ValueError('Tokenizer must expose a backend serialization or vocabulary.')
    return {'name': config.data.tokenizer_name_or_path,
            'implementation_sha256': hashlib.sha256(implementation).hexdigest(),
            'vocab_size': int(tokenizer.vocab_size),
            'special_ids': {name: getattr(tokenizer, name, None) for name in (
                'bos_token_id', 'eos_token_id', 'pad_token_id', 'mask_token_id')}}


def build_cache_identity(config, tokenizer, original_train_dataset):
    """Fingerprint actual teacher bytes, tokenization, selected data and rollout.

    Checkpoint and cache path strings are deliberately excluded: relocating an
    identical checkpoint or completed cache is valid. The full checkpoint is
    hashed once per call; distributed preparation computes this only on rank 0.
    """
    settings = _settings(config, len(original_train_dataset))
    indices = _selected_indices(len(original_train_dataset), settings['num_examples'],
                                settings['seed'])
    dataset_fingerprint = getattr(original_train_dataset, '_fingerprint', None)
    if dataset_fingerprint is None:
        # Small/local datasets need a content identity too. Hugging Face datasets
        # supply their fingerprint and avoid this scan of the original corpus.
        digest = hashlib.sha256()
        for row in original_train_dataset:
            digest.update(_json_bytes({name: torch.as_tensor(row[name]).tolist()
                                       for name in ('input_ids', 'attention_mask')}))
        dataset_fingerprint = digest.hexdigest()
    source = str(config.training.finetune_path)
    if not source or source == 'None':
        raise ValueError('training.finetune_path must identify the original MDLM teacher.')
    data_keys = ('separator', 'wrap', 'train_on_prompt', 'train_on_pad',
                 'filter_too_long', 'val_ratio', 'val_seed')
    parameters = {
        'schema_version': SCHEMA_VERSION, 'source_commit': SOURCE_COMMIT,
        'teacher_checkpoint_sha256': _file_sha256(_checkpoint_path(source)),
        'teacher_weights': 'raw', 'teacher_model': _plain(config.model),
        'teacher_time_conditioning': False, 'teacher_attention': 'full',
        'tokenizer': _tokenizer_identity(config, tokenizer),
        'dataset': {'fingerprint': str(dataset_fingerprint),
                    'length': len(original_train_dataset),
                    'selected_indices_sha256': hashlib.sha256(_json_bytes(indices)).hexdigest(),
                    'format': {key: _plain(config.data.get(key)) for key in data_keys}},
        'rollout': {**settings, 'selection': 'global_highest_confidence_one_token',
                    'hidden_state': 'backbone.output_layer.linear.input',
                    'rng': 'sha256_example_temperature_open_uniform_v2',
                    'early_eos_stopping': False},
    }
    return {'fingerprint': hashlib.sha256(_json_bytes(parameters)).hexdigest(),
            'parameters': parameters}


def _trajectory_seed(seed, example_index, temperature):
    raw = _json_bytes([int(seed), int(example_index), float(temperature)])
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)


@torch.no_grad()
def collect_trajectory_batch(model, input_ids, attention_mask, *, temperatures, seeds):
    """Reveal all eligible positions, preserving every prompt/ineligible token.

    The pre-hook records the exact input to the frozen output linear layer,
    including final normalization and AdaLN. Log probabilities already supplied
    by MDLM are used for proposals and their untempered confidence.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('Expected matching [batch, length] input and eligibility tensors.')
    batch, length = input_ids.shape
    if len(temperatures) != batch or len(seeds) != batch:
        raise ValueError('One temperature and seed are required per example.')
    if length > torch.iinfo(torch.int16).max:
        raise ValueError('Compact reveal ranks support sequence lengths below 32768.')
    valid = attention_mask.bool()
    if not bool(valid.any(dim=-1).all()):
        raise ValueError('Every trajectory needs at least one eligible answer token.')
    if any(not math.isfinite(float(t)) or float(t) < 0 for t in temperatures):
        raise ValueError('Temperatures must be finite and nonnegative.')
    if bool(input_ids.eq(model.mask_index).any()):
        raise ValueError('Ground-truth input must not contain the diffusion mask token.')
    device = input_ids.device
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    ranks = torch.full_like(input_ids, -1, dtype=torch.int16)
    current = torch.where(valid, model.mask_index, input_ids).clone()
    hidden_size = model.backbone.output_layer.linear.in_features
    hidden_buffer = torch.zeros((batch, length, hidden_size), device=device,
                                dtype=torch.bfloat16)
    captured = []

    def capture(_module, args):
        captured[:] = [args[0].detach()]

    hook = model.backbone.output_layer.linear.register_forward_pre_hook(capture)
    was_training = model.training
    model.eval()
    try:
        for step in range(int(valid.sum(dim=-1).max())):
            captured.clear()
            log_probs = model.forward(current, torch.zeros(batch, 1, device=device)).float()
            if not captured or captured[0].shape != (batch, length, hidden_size):
                raise ValueError('Teacher did not expose the expected pre-output hidden states.')
            if log_probs.shape[:2] != input_ids.shape or model.mask_index >= log_probs.shape[-1]:
                raise ValueError('Teacher output shape/vocabulary differs from the token input.')
            # Exclude the mask even if a caller returns raw rather than SUBS
            # probabilities. No generated mask may stall a monotone trajectory.
            log_probs[..., model.mask_index] = -torch.inf
            for row in range(batch):
                candidates = current[row].eq(model.mask_index) & valid[row]
                if not bool(candidates.any()):
                    continue
                temperature = float(temperatures[row])
                if temperature > 0:
                    uniform = torch.rand(log_probs[row].shape, device=device,
                                         generator=generators[row], dtype=torch.float32)
                    # torch.rand is already strictly below 1. Only exclude its
                    # possible zero; clipping either tail changes Gumbel-Max.
                    uniform.clamp_min_(torch.finfo(uniform.dtype).tiny)
                    proposals = (log_probs[row] / temperature
                                 - (-uniform.log()).log()).argmax(dim=-1)
                else:
                    proposals = log_probs[row].argmax(dim=-1)
                # exp is monotonic, so log confidence produces the same argmax
                # without constructing another [length, vocabulary] tensor.
                confidence = log_probs[row].gather(-1, proposals[:, None]).squeeze(-1)
                confidence = confidence.masked_fill(~candidates, -torch.inf)
                position = confidence.argmax()
                if not bool(torch.isfinite(confidence[position])):
                    raise ValueError('Teacher supplied no finite non-mask token prediction.')
                current[row, position] = proposals[position]
                ranks[row, position] = step
                hidden_buffer[row, position] = captured[0][row, position].to(torch.bfloat16)
    finally:
        hook.remove()
        model.train(was_training)
    if bool((valid & current.eq(model.mask_index)).any()):
        raise RuntimeError('Teacher rollout left eligible positions masked.')
    return {'final_tokens': current.detach().cpu(), 'reveal_order': ranks.cpu(),
            'teacher_hidden': hidden_buffer.cpu()}


@contextmanager
def _file_lock(path):
    import fcntl
    with open(path, 'a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as target:
            target.write(_json_bytes(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, 'rb') as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_cache_manifest(cache_dir):
    with open(Path(cache_dir) / 'manifest.json', encoding='utf-8') as source:
        manifest = json.load(source)
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported CDLM trajectory cache schema.')
    if manifest.get('identity', {}).get('fingerprint') != hashlib.sha256(
            _json_bytes(manifest['identity']['parameters'])).hexdigest():
        raise ValueError('CDLM cache manifest identity is inconsistent.')
    return manifest


def _shard_descriptor(shard_index, total, size):
    start = shard_index * size
    return {'file': f'part-{shard_index:08d}.pt', 'start': start,
            'count': min(size, total - start)}


def _valid_shard(directory, descriptor, fingerprint, *, verify_hash=True):
    path = directory / descriptor['file']
    metadata_path = path.with_suffix('.json')
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        with open(metadata_path, encoding='utf-8') as source:
            metadata = json.load(source)
        if (any(metadata.get(key) != value for key, value in descriptor.items())
                or metadata.get('fingerprint') != fingerprint):
            raise ValueError(f'CDLM trajectory shard metadata mismatch: {path}.')
        if verify_hash and _file_sha256(path) != metadata.get('sha256'):
            raise ValueError(f'CDLM trajectory shard checksum mismatch: {path}.')
        return metadata
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f'Unreadable CDLM trajectory shard metadata: {metadata_path}.') from error


def _validate_payload(payload, descriptor, fingerprint):
    if (payload.get('fingerprint') != fingerprint
            or payload.get('start') != descriptor['start']
            or len(payload.get('items', [])) != descriptor['count']):
        raise ValueError('CDLM trajectory shard payload differs from its manifest.')


def _distributed_context(rank, world_size):
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if rank is None:
        rank = torch.distributed.get_rank() if initialized else 0
    if world_size is None:
        world_size = torch.distributed.get_world_size() if initialized else 1
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError('Expected 0 <= rank < positive world_size.')
    return rank, world_size, initialized


@contextmanager
def _collective_failure_boundary(initialized, stage):
    """Share local exceptions before peers enter the next data collective.

    Rank-zero identity construction and unequal per-rank shard loops can fail
    independently. Exchanging their status first prevents peers waiting in a
    broadcast/barrier that the failing rank will never enter.
    """
    local_error = None
    try:
        yield
    except Exception as error:
        if not initialized:
            raise
        local_error = error
    if initialized:
        failures = [None] * torch.distributed.get_world_size()
        message = None if local_error is None else f'{type(local_error).__name__}: {local_error}'
        torch.distributed.all_gather_object(failures, message)
        failed = [(rank, failure) for rank, failure in enumerate(failures) if failure is not None]
        if failed:
            detail = '; '.join(f'rank {rank}: {failure}' for rank, failure in failed)
            raise RuntimeError(f'CDLM {stage} failed ({detail}).') from local_error


def _prepare_local_shards(model, train_dataset, settings, selected, directory,
                          descriptors, fingerprint, rank, world_size):
    device = next(model.parameters()).device
    for shard_index in range(rank, len(descriptors), world_size):
        descriptor = descriptors[shard_index]
        path = directory / descriptor['file']
        with _file_lock(path.with_suffix('.lock')):
            if _valid_shard(directory, descriptor, fingerprint) is not None:
                continue
            if path.exists():
                # A process can stop after atomic tensor rename and before the
                # tiny metadata rename. Recover that completed payload directly.
                payload = torch.load(path, map_location='cpu', weights_only=True)
                _validate_payload(payload, descriptor, fingerprint)
            else:
                items = []
                stop = descriptor['start'] + descriptor['count']
                for begin in range(descriptor['start'], stop, settings['batch_size']):
                    jobs = []
                    rows = []
                    for job in range(begin, min(begin + settings['batch_size'], stop)):
                        index = selected[job // len(settings['temperatures'])]
                        temperature = settings['temperatures'][job % len(settings['temperatures'])]
                        jobs.append((index, temperature))
                        rows.append(train_dataset[index])
                    ids = torch.stack([torch.as_tensor(row['input_ids'], dtype=torch.long)
                                       for row in rows]).to(device)
                    eligible = torch.stack([torch.as_tensor(row['attention_mask'], dtype=torch.bool)
                                            for row in rows]).to(device)
                    output = collect_trajectory_batch(
                        model, ids, eligible, temperatures=[job[1] for job in jobs],
                        seeds=[_trajectory_seed(settings['seed'], *job) for job in jobs])
                    for row, (index, temperature) in enumerate(jobs):
                        items.append({'input_ids': ids[row].cpu(),
                                      'attention_mask': eligible[row].cpu(),
                                      **{key: value[row].clone() for key, value in output.items()},
                                      'example_index': index, 'temperature': temperature})
                payload = {'fingerprint': fingerprint, 'start': descriptor['start'], 'items': items}
                _atomic_torch(path, payload)
            metadata = {**descriptor, 'fingerprint': fingerprint, 'sha256': _file_sha256(path)}
            _atomic_json(path.with_suffix('.json'), metadata)


def prepare_trajectories(model, config, tokenizer, train_dataset, *, rank=None, world_size=None):
    """Prepare atomic resumable shards; initialized DDP ranks partition shards.

    Explicit rank/world_size also permit independently launched workers sharing
    a filesystem. In that case, callers synchronize externally before training;
    the last completing worker marks the manifest complete. Initialized ranks
    exchange errors at each phase, including before broadcast and final barrier.
    """
    rank, world_size, initialized = _distributed_context(rank, world_size)
    with _collective_failure_boundary(initialized, 'preparation setup'):
        if getattr(model, '_evaluation_weights', 'live') == 'ema':
            raise ValueError('CDLM offline trajectories require raw original teacher weights, not EMA.')
        settings = _settings(config, len(train_dataset))
    shared = [None]
    with _collective_failure_boundary(initialized, 'cache identity construction'):
        if rank == 0 or not initialized:
            shared[0] = build_cache_identity(config, tokenizer, train_dataset)
    if initialized:
        torch.distributed.broadcast_object_list(shared, src=0)
    identity = shared[0]
    fingerprint = identity['fingerprint']
    with _collective_failure_boundary(initialized, 'cache initialization'):
        default_dir = Path(str(config.data.cache_dir)) / 'cdlm_trajectories'
        directory = Path(str(config.algo.get('trajectory_cache_dir') or default_dir)).expanduser()
        if not directory.is_absolute():
            try:
                from hydra.utils import to_absolute_path
                directory = Path(to_absolute_path(str(directory)))
            except ImportError:
                directory = directory.absolute()
        directory.mkdir(parents=True, exist_ok=True)
        total = settings['num_examples'] * len(settings['temperatures'])
        descriptors = [_shard_descriptor(i, total, settings['shard_size'])
                       for i in range(math.ceil(total / settings['shard_size']))]
        with _file_lock(directory / 'manifest.lock'):
            if (directory / 'manifest.json').exists():
                manifest = read_cache_manifest(directory)
                if manifest['identity']['fingerprint'] != fingerprint:
                    raise ValueError('CDLM trajectory cache configuration/teacher/data mismatch; '
                                     'select a different algo.trajectory_cache_dir.')
            else:
                manifest = {'schema_version': SCHEMA_VERSION, 'identity': identity,
                            'status': 'building', 'num_trajectories': total, 'shards': descriptors}
                _atomic_json(directory / 'manifest.json', manifest)
        selected = _selected_indices(len(train_dataset), settings['num_examples'], settings['seed'])
    # The boundary surrounds the entire local loop: ranks may own different
    # numbers of shards and therefore must not synchronize once per shard.
    with _collective_failure_boundary(initialized, 'trajectory shard generation'):
        _prepare_local_shards(model, train_dataset, settings, selected, directory,
                              descriptors, fingerprint, rank, world_size)
    if initialized:
        torch.distributed.barrier()
    with _collective_failure_boundary(initialized, 'cache finalization'):
        # Every completion is atomic, so independent workers may inspect
        # receipts while another is still writing its final shard.
        with _file_lock(directory / 'manifest.lock'):
            receipts = [_valid_shard(directory, descriptor, fingerprint, verify_hash=False)
                        for descriptor in descriptors]
            manifest = read_cache_manifest(directory)
            if all(receipt is not None for receipt in receipts):
                manifest['status'] = 'complete'
                manifest['shards'] = [{key: receipt[key] for key in ('file', 'start', 'count', 'sha256')}
                                      for receipt in receipts]
            else:
                manifest['status'] = 'building'
            _atomic_json(directory / 'manifest.json', manifest)
    return str(directory)


class CDLMTrajectoryDataset(Dataset):
    """Read completed trajectory shards with one resident shard per worker.

    A manifest fingerprint can be supplied to reject stale source/configuration.
    Shard integrity is checked on load; serialized tensors are loaded with
    weights_only=True and no model/tokenizer deserialization is involved.
    """

    def __init__(self, cache_dir, expected_fingerprint=None):
        self.directory = Path(cache_dir)
        self.manifest = read_cache_manifest(self.directory)
        if self.manifest.get('status') != 'complete':
            raise ValueError('CDLM trajectory cache is incomplete; finish mode=prepare_cdlm first.')
        expected = (expected_fingerprint.get('fingerprint')
                    if isinstance(expected_fingerprint, dict) else expected_fingerprint)
        self.fingerprint = self.manifest['identity']['fingerprint']
        if expected is not None and expected != self.fingerprint:
            raise ValueError('CDLM trajectory cache fingerprint does not match current inputs.')
        self.shards = self.manifest['shards']
        self._starts = [shard['start'] for shard in self.shards]
        self._resident_index = None
        self._resident_items = None
        expected_start = 0
        for shard in self.shards:
            if (shard['start'] != expected_start or shard['count'] <= 0
                    or Path(shard['file']).name != shard['file']):
                raise ValueError('Invalid CDLM trajectory shard layout.')
            expected_start += shard['count']
        if expected_start != self.manifest['num_trajectories']:
            raise ValueError('CDLM cache manifest has an inconsistent trajectory count.')

    def __len__(self):
        return self.manifest['num_trajectories']

    def __getitem__(self, index):
        import bisect
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._starts, index) - 1
        if shard_index != self._resident_index:
            shard = self.shards[shard_index]
            receipt = _valid_shard(self.directory, {key: shard[key] for key in ('file', 'start', 'count')},
                                   self.fingerprint)
            if receipt is None or receipt['sha256'] != shard['sha256']:
                raise ValueError('CDLM trajectory shard is missing or differs from the completed manifest.')
            payload = torch.load(self.directory / shard['file'], map_location='cpu', weights_only=True)
            _validate_payload(payload, shard, self.fingerprint)
            self._resident_items = payload['items']
            self._resident_index = shard_index
        row = self._resident_items[index - self.shards[shard_index]['start']]
        return {key: value.clone() if torch.is_tensor(value) else value
                for key, value in row.items()}
