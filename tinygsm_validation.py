"""Conditional GSM8K evaluation for the native-token SDLLM MMD option.

Adapted from IDLM-sdtt-tinygsm/sdtt_eval.py, itself based on
dlm-mmd/mmd/tinygsm_validation.py (commit ebb2a6b). The default threshold grid
follows reference run 3e0ac6b1-7a8f-43c3-af58-d9e21b886da7. Fixed-grid greedy
sampling follows the S-FLM convention in archived IDLM TinyGSM source 14594e:
random ancestral unmasking, argmax token proposals, final removal inside the
requested step budget, and actual cache-aware forward counts.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import warnings

from filelock import FileLock
from hydra.utils import to_absolute_path
import torch
import torch.distributed as dist

THRESHOLDS = [i / 1000 for i in range(325, 951, 25)] + [.98]


def _rng_devices(model):
    """Fork only the model's CUDA generator; CPU checks must not open GPUs."""
    device = torch.device(model.device)
    if device.type != 'cuda':
        return []
    return [torch.cuda.current_device() if device.index is None else device.index]


def _seed_model_rng(model, seed):
    torch.random.default_generator.manual_seed(seed)
    device = torch.device(model.device)
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def load_examples(path, tokenizer, separator, length, limit=None):
    """Use the paper's BOS + stripped question + literal separator prompt."""
    records = json.loads(Path(path).expanduser().read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('GSM8K data must be a nonempty JSON list.')
    separator_ids = tokenizer(separator, add_special_tokens=False).input_ids
    examples = []
    for index, record in enumerate(records[:limit]):
        question, answer = record.get('prompt'), record.get('response_ground_truth')
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f'Example {index} requires prompt and response_ground_truth strings.')
        tokens = [tokenizer.bos_token_id] + tokenizer(
            question.strip(), add_special_tokens=False).input_ids + separator_ids
        if len(tokens) >= length:
            raise ValueError(f'Example {index} leaves no answer space at sequence length {length}.')
        examples.append({'input_ids': tokens, 'answer': answer, 'question': question.strip()})
    return examples


@torch.no_grad()
def generate_samples(model, prompts, threshold, temperature=0., max_steps=None):
    """Paper-run low_confidence_dynamic sampler, including the capped fallback.

    Confidence uses untempered probabilities, the comparison is strictly `>`,
    and a stalled row commits its most confident token. EOS does not stop a row:
    the reference finishes all 512 positions, including padding after the answer.
    """
    if not prompts or any(not 0 < len(p) < model.num_tokens for p in prompts):
        raise ValueError('Each prompt must leave at least one answer position.')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0, 1].')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Temperature must be finite and nonnegative.')
    if max_steps is not None and max_steps <= 0:
        raise ValueError('max_steps must be positive or None.')
    x = model.prior_sample(len(prompts), model.num_tokens)
    for row, prompt in enumerate(prompts):
        prefix = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
        if bool((prefix == model.mask_index).any()):
            raise ValueError('Prompt cannot contain the diffusion mask token.')
        x[row, :len(prefix)] = prefix
    per_row = torch.zeros(len(prompts), dtype=torch.long, device=model.device)
    nfe = 0
    cap = max_steps or model.num_tokens
    for step in range(cap + 1):
        candidates = x == model.mask_index
        unfinished = candidates.any(dim=-1)
        if not bool(unfinished.any()):
            break
        per_row += unfinished.long()
        nfe += 1
        if step == cap:
            t = torch.full((len(prompts), 1), 1e-5, device=model.device)
        else:
            # Invert the original linear schedule from the current mask fraction.
            alpha = 1 - candidates.float().mean(dim=-1, keepdim=True)
            t = ((1 - alpha) / (1 - model.noise.eps)).clamp(1e-5, 1.)
        probs = model.forward(x, model._sigma_from_alphat(model.noise(t)[1])).exp()
        if temperature == 0:
            proposed = probs.argmax(dim=-1)
        elif temperature == 1:
            from trainer_base import sample_categorical
            proposed = sample_categorical(probs)
        else:
            from trainer_base import sample_categorical
            proposed = sample_categorical(torch.softmax(
                probs.clamp_min(1e-30).log() / temperature, dim=-1))
        confidence = probs.gather(-1, proposed.unsqueeze(-1)).squeeze(-1).float()
        confidence = confidence.masked_fill(~candidates, -float('inf'))
        if step == cap:
            transfer = candidates
        else:
            transfer = (confidence > threshold) & candidates
            stalled = unfinished & ~transfer.any(dim=-1)
            forced = torch.zeros_like(transfer)
            forced.scatter_(-1, confidence.argmax(dim=-1, keepdim=True), True)
            transfer = torch.where(stalled.unsqueeze(-1), forced, transfer)
        x = torch.where(transfer, proposed, x)
    if bool((x == model.mask_index).any()):
        raise ValueError('Decoder returned masked tokens after final noise removal.')
    return x, {'nfe': nfe, 'nfe_per_row': per_row}


@torch.no_grad()
def generate_greedy_samples(model, prompts, num_steps=None, eps=1e-5):
    """S-FLM fixed-budget random unmasking with greedy token proposals.

    Unmasking remains stochastic. This differs from the confidence-threshold
    sampler above, which chooses positions by confidence. The final scheduled
    iteration removes all remaining masks, so no extra denoising call is added.
    """
    if not prompts or any(not 0 < len(p) < model.num_tokens for p in prompts):
        raise ValueError('Each prompt must leave at least one answer position.')
    if num_steps is None:
        num_steps = model.config.eval.get('gsm8k_steps') or model.config.sampling.steps
    if type(num_steps) is not int or num_steps <= 0:
        raise ValueError('GSM8K greedy steps must be a positive integer.')
    if not math.isfinite(eps) or not 0 < eps < 1:
        raise ValueError('GSM8K greedy sampling epsilon must be in (0, 1).')
    predictor = model.config.sampling.get('predictor', 'ancestral_cache')
    if predictor not in ('ancestral', 'ancestral_cache'):
        raise ValueError('GSM8K greedy evaluation requires an ancestral predictor.')
    x = model.prior_sample(len(prompts), model.num_tokens)
    prefixes = []
    for row, prompt in enumerate(prompts):
        prefix = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
        if bool((prefix == model.mask_index).any()):
            raise ValueError('Prompt cannot contain the diffusion mask token.')
        prefixes.append(prefix)
        x[row, :len(prefix)] = prefix
    times = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    cached, nfe = None, 0
    for step in range(num_steps):
        t = times[step] * torch.ones(x.shape[0], 1, device=model.device)
        _, alpha_t = model.noise(t)
        alpha_s = torch.ones_like(alpha_t) if step == num_steps - 1 else model.noise(t - dt)[1]
        if cached is None:
            cached = model.forward(x, model._sigma_from_alphat(alpha_t)).exp()
            nfe += 1
        proposed = cached.argmax(dim=-1)
        probability = ((alpha_s - alpha_t) / (1 - alpha_t).clamp_min(1e-12)).clamp(0, 1)
        unmask = torch.rand(x.shape, device=x.device) < probability
        next_x = torch.where((x == model.mask_index) & unmask, proposed, x)
        if (predictor != 'ancestral_cache' or not torch.allclose(next_x, x)
                or getattr(model, 'time_conditioning', False)):
            cached = None
        x = next_x
        for row, prefix in enumerate(prefixes):
            x[row, :len(prefix)] = prefix
    if bool((x == model.mask_index).any()):
        raise ValueError('Greedy decoder returned masks after final removal.')
    return x, {'nfe': nfe,
               'nfe_per_row': torch.full((len(prompts),), nfe, dtype=torch.long,
                                         device=model.device)}


def grade_sample(sample, answer, timeout_s=5.):
    """Run reference scoring outside the model process with an outer timeout."""
    worker = Path(__file__).resolve().with_name('tinygsm_sandbox_gsm8k.py')
    request = json.dumps({'sample': sample, 'answer': answer, 'timeout_s': timeout_s})
    try:
        result = subprocess.run([sys.executable, '-I', str(worker)], input=request,
                                text=True, capture_output=True, timeout=timeout_s + 5,
                                check=True)
    except subprocess.TimeoutExpired:
        return False
    # Infrastructure failures should stop evaluation, not silently lower accuracy.
    value = json.loads(result.stdout)
    if not isinstance(value, bool):
        raise ValueError('Execution worker returned a non-boolean grade.')
    return value


def resolve_data_path(config):
    """Reuse or download GSM8K test data; an explicit path never falls back."""
    explicit = config.eval.get('gsm8k_data_path')
    if explicit:
        path = Path(to_absolute_path(str(Path(explicit).expanduser())))
        if not path.is_file():
            raise FileNotFoundError(f'GSM8K evaluation data not found: {path}')
        return path
    candidates = []
    if config.get('model_weights_dir'):
        candidates.append(Path(config.model_weights_dir) / 'data' / 'gsm8k_test.json')
    if config.data.get('cache_dir'):
        candidates.append(Path(config.data.cache_dir) / 'gsm8k_test.json')
    candidates.append(Path(__file__).resolve().parent / 'data' / 'gsm8k_test.json')
    for candidate in candidates:
        path = Path(to_absolute_path(str(candidate.expanduser())))
        if path.is_file():
            return path
    cache_dir = config.data.get('cache_dir') or str(Path(__file__).resolve().parent / 'data')
    path = Path(to_absolute_path(str(Path(cache_dir).expanduser()))) / 'gsm8k_test.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    # DDP ranks can reach the first evaluation together. Publish only complete JSON.
    with FileLock(str(path) + '.lock'):
        if not path.is_file():
            print(f'Preparing GSM8K test data at {path}', flush=True)
            import datasets
            samples = datasets.load_dataset(
                'openai/gsm8k', 'main', split='test', cache_dir=str(path.parent))
            records = [{'prompt': row['question'], 'response_ground_truth': row['answer']}
                       for row in samples]
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(records, indent=2), encoding='utf-8')
            temporary.replace(path)
    return path


def _raise_if_failed(error):
    """Reach the same error boundary on every rank before reducing metrics."""
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, None if error is None else repr(error))
        if any(item is not None for item in errors):
            raise RuntimeError(f'GSM8K evaluation failed across ranks: {errors}') from error
    elif error is not None:
        raise error


def _settings(cfg, thresholds):
    if not cfg.get('tinygsm_exec', False):
        raise ValueError('GSM8K function-execution accuracy requires eval.tinygsm_exec=True.')
    thresholds = sorted(set(float(t) for t in thresholds))
    batch_size = int(cfg.get('gsm8k_batch_size', 16))
    workers = int(cfg.get('gsm8k_grading_workers', 4))
    cap = cfg.get('gsm8k_max_steps')
    temperature = float(cfg.get('gsm8k_pareto_temperature', 0.))
    timeout = float(cfg.get('tinygsm_exec_timeout', 5.))
    if min(batch_size, workers) <= 0 or (cap is not None and cap <= 0):
        raise ValueError('GSM8K batch size, grading workers and cap must be positive.')
    if not thresholds or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('GSM8K thresholds must be finite and in [0, 1].')
    if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('GSM8K temperature must be nonnegative and timeout positive; both finite.')
    return thresholds, batch_size, workers, cap, temperature, timeout


def _evaluate_threshold(model, examples, threshold, pool, batch_size, temperature, cap, timeout,
                        sample_records=None, max_sample_records=0):
    correct = count = total_nfe = batch_nfe = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [example['input_ids'] for example in batch]
        samples, metadata = generate_samples(model, prompts, threshold, temperature, cap)
        texts = [model.tokenizer.decode(sample[len(prompt):].tolist(), skip_special_tokens=True)
                 for sample, prompt in zip(samples.cpu(), prompts)]
        pairs = zip(texts, [example['answer'] for example in batch])
        grades = list(pool.map(lambda pair: grade_sample(*pair, timeout_s=timeout), pairs))
        correct += sum(grades)
        # Reuse decoded completions and grades; logging must not add model calls.
        if sample_records is not None and len(sample_records) < max_sample_records:
            nfes = metadata['nfe_per_row'].cpu().tolist()
            for example, text, grade, nfe in zip(batch, texts, grades, nfes):
                if len(sample_records) >= max_sample_records:
                    break
                sample_records.append({
                    'question': example.get('question', ''), 'generated': text,
                    'reference': example['answer'], 'grade': bool(grade),
                    'nfe': int(nfe), 'threshold': float(threshold)})
        count += len(batch)
        total_nfe += int(metadata['nfe_per_row'].sum())
        batch_nfe += metadata['nfe']
    return correct, count, total_nfe, batch_nfe


@torch.no_grad()
def evaluate(model, config):
    """Callable standalone threshold sweep; this helper is not a main.py mode."""
    cfg = config.eval
    settings = _settings(cfg, cfg.get('gsm8k_pareto_thresholds') or THRESHOLDS)
    thresholds, batch_size, workers, cap, temperature, timeout = settings
    limit = cfg.get('gsm8k_num_examples')
    if limit is not None and limit <= 0:
        raise ValueError('eval.gsm8k_num_examples must be positive or null.')
    path = resolve_data_path(config)
    examples = load_examples(path, model.tokenizer, model.config.data.separator, model.num_tokens, limit)
    results = []
    payload = {
        'checkpoint': cfg.checkpoint_path, 'weights': getattr(model, '_evaluation_weights',
                           'ema' if getattr(model, 'ema', None) is not None and not cfg.disable_ema else 'live'),
        'data_path': str(path), 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'sequence_length': model.num_tokens, 'separator': model.config.data.separator,
        'batch_size': batch_size, 'temperature': temperature, 'seed': config.seed,
        'max_steps': cap or model.num_tokens,
        'sampler': 'low_confidence_dynamic', 'scoring': 'S-FLM simple_math_problem execution',
        'timeout_seconds': timeout,
        'nfe_definition': 'mean active forwards per example, including capped final removal',
        'results': results,
    }
    output = Path(to_absolute_path(str(Path(cfg.generated_samples_path).expanduser())))
    output.parent.mkdir(parents=True, exist_ok=True)
    _seed_model_rng(model, config.seed)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for threshold in thresholds:
            correct, count, total_nfe, batch_nfe = _evaluate_threshold(
                model, examples, threshold, pool, batch_size, temperature, cap, timeout)
            point = {'threshold': threshold, 'accuracy': correct / count,
                     'correct': correct, 'count': count,
                     'nfe': total_nfe / count, 'total_batch_forwards': batch_nfe}
            results.append(point)
            temporary = output.with_suffix(output.suffix + '.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(output)
            print(f'threshold={threshold:g}: {correct}/{count} '
                  f'({100 * point["accuracy"]:.2f}%) at {point["nfe"]:.2f} NFE/example', flush=True)
    print(f'Results saved to {output}', flush=True)
    return results


@torch.no_grad()
def sweep(model, examples):
    """Pool counts across rank-strided shards, including ranks with no examples."""
    cfg = model.config.eval
    error = None
    try:
        if not examples:
            raise ValueError('GSM8K examples must be nonempty.')
        settings = _settings(cfg, cfg.get('gsm8k_pareto_thresholds', THRESHOLDS))
        thresholds, batch_size, workers, cap, temperature, timeout = settings
    except Exception as caught:
        error = caught
    _raise_if_failed(error)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else model.trainer.global_rank
    world_size = dist.get_world_size() if distributed else getattr(model.trainer, 'world_size', 1)
    local_examples = examples[rank::world_size]
    devices = _rng_devices(model)
    curve = []
    sample_threshold = min(thresholds, key=lambda value: abs(value - .9))
    sample_limit = max(0, int(model.config.sampling.get('num_sample_log', 8)))
    model._gsm8k_sample_records = []
    with torch.random.fork_rng(devices=devices), ThreadPoolExecutor(max_workers=workers) as pool:
        for threshold in thresholds:
            _seed_model_rng(model, int(model.config.get('seed', 1)) + 104729 * rank)
            error = None
            try:
                correct, count, total_nfe, _ = _evaluate_threshold(
                    model, local_examples, threshold, pool, batch_size, temperature, cap, timeout,
                    sample_records=model._gsm8k_sample_records if rank == 0 and threshold == sample_threshold else None,
                    max_sample_records=sample_limit)
            except Exception as caught:
                error = caught
            _raise_if_failed(error)
            totals = torch.tensor([correct, count, total_nfe], dtype=torch.float64, device=model.device)
            if distributed:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            correct, count, total_nfe = [int(value) for value in totals.tolist()]
            point = {'threshold': threshold, 'accuracy': correct / count,
                     'nfe': total_nfe / count, 'correct': correct, 'count': count}
            curve.append(point)
            if rank == 0:
                print(f'[gsm8k-pareto] step {model.global_step} thr {threshold:g}: '
                      f'{100 * point["accuracy"]:.2f}% ({correct}/{count}) '
                      f'at {point["nfe"]:.2f} NFE/example', flush=True)
    return curve


@torch.no_grad()
def sweep_greedy(model, examples):
    """Grade one fixed-grid greedy run and pool per-example counts across ranks."""
    cfg = model.config.eval
    error = None
    try:
        if not examples:
            raise ValueError('GSM8K examples must be nonempty.')
        _, batch_size, workers, _, _, timeout = _settings(cfg, [0.])
        steps = cfg.get('gsm8k_steps') or model.config.sampling.steps
        if type(steps) is not int or steps <= 0:
            raise ValueError('GSM8K greedy steps must be a positive integer.')
    except Exception as caught:
        error = caught
    _raise_if_failed(error)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else model.trainer.global_rank
    world = dist.get_world_size() if distributed else getattr(model.trainer, 'world_size', 1)
    local = examples[rank::world]
    correct = count = total_nfe = 0
    with torch.random.fork_rng(devices=_rng_devices(model)), ThreadPoolExecutor(max_workers=workers) as pool:
        _seed_model_rng(model, int(model.config.get('seed', 1)) + 104729 * rank)
        error = None
        try:
            for start in range(0, len(local), batch_size):
                chunk = local[start:start + batch_size]
                prompts = [row['input_ids'] for row in chunk]
                samples, metadata = generate_greedy_samples(model, prompts, num_steps=steps)
                requests = []
                for row, example in enumerate(chunk):
                    text = model.tokenizer.decode(samples[row, len(prompts[row]):].cpu().tolist(),
                                                  skip_special_tokens=True)
                    requests.append((text, example['answer']))
                correct += sum(pool.map(lambda pair: grade_sample(pair[0], pair[1], timeout), requests))
                count += len(chunk)
                total_nfe += int(metadata['nfe_per_row'].sum().item())
        except Exception as caught:
            error = caught
        _raise_if_failed(error)
    totals = torch.tensor([correct, count, total_nfe], dtype=torch.float64, device=model.device)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    correct, count, total_nfe = [int(value) for value in totals.tolist()]
    point = {'accuracy': correct / count, 'correct': correct, 'count': count,
             'nfe': total_nfe / count, 'steps': steps,
             'sampler': 'random', 'temperature': 0.}
    if rank == 0:
        print(f'[gsm8k-greedy] step {model.global_step}: {100 * point["accuracy"]:.2f}% '
              f'({correct}/{count}) at {point["nfe"]:.2f} NFE/seq [random, T=0]', flush=True)
    return point


def log_validation_gsm8k(model, initial=False):
    """Called on every rank with EMA weights and evaluation mode already active."""
    cfg = model.config.eval
    every = int(cfg.get('gsm8k_every_n_validations', 1))
    if (model.config.data.train != 'tiny_gsm' or model.trainer.sanity_checking
            or not cfg.get('gsm8k_at_validation', False) or every <= 0
            or (initial and (not cfg.get('validate_before_train', False) or model.global_step != 0))):
        return
    count = getattr(model, '_gsm8k_validation_count', 0)
    model._gsm8k_validation_count = count + 1
    if count % every:
        return
    devices = _rng_devices(model)
    with torch.random.fork_rng(devices=devices):
        error = None
        try:
            if getattr(model, '_gsm8k_examples', None) is None:
                limit = cfg.get('gsm8k_num_examples')
                if limit is not None and limit <= 0:
                    raise ValueError('eval.gsm8k_num_examples must be positive or null.')
                path = resolve_data_path(model.config)
                model._gsm8k_examples = load_examples(
                    path, model.tokenizer, model.config.data.separator, model.num_tokens, limit)
                model._gsm8k_protocol = {
                    'data_path': str(path), 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'examples': len(model._gsm8k_examples), 'sequence_length': model.num_tokens,
                    'separator': model.config.data.separator,
                    'weights': 'ema' if getattr(model, 'ema', None) is not None else 'live',
                    'batch_size': cfg.get('gsm8k_batch_size', 16),
                    'world_size': getattr(model.trainer, 'world_size', 1),
                    'seed': model.config.get('seed', 1), 'rank_seed_offset': 104729,
                    'sampler': 'low_confidence_dynamic', 'temperature': cfg.get('gsm8k_pareto_temperature', 0.),
                    'max_steps': cfg.get('gsm8k_max_steps') or model.num_tokens,
                    'timeout_seconds': cfg.get('tinygsm_exec_timeout', 5.),
                    'nfe_definition': 'mean active forwards per example, including capped final removal'}
        except Exception as caught:
            error = caught
        _raise_if_failed(error)
        model._gsm8k_greedy_result = (sweep_greedy(model, model._gsm8k_examples)
                                     if cfg.get('gsm8k_greedy_token', True) else None)
        curve = sweep(model, model._gsm8k_examples)
        error = None
        try:
            publish(model, curve, initial=initial)
        except Exception as caught:
            error = caught
        _raise_if_failed(error)


def publish(model, curve, initial=False):
    """Preserve the reference metric names, native frontier, image and samples."""
    logger = model.trainer.logger
    metrics = {}
    greedy = getattr(model, '_gsm8k_greedy_result', None)
    if greedy is not None:
        for name, key in [('val/gsm8k_acc_greedy', 'accuracy'), ('val/gsm8k_nfe_greedy', 'nfe')]:
            metrics[name] = greedy[key]
            if not initial:
                model.log(name, torch.tensor(greedy[key], device=model.device, dtype=torch.float32),
                          on_epoch=True, on_step=False, sync_dist=False, rank_zero_only=True)
    for point in curve:
        tag = f'{point["threshold"]:g}'.replace('.', '')
        for suffix, key in [('acc', 'accuracy'), ('nfe', 'nfe')]:
            # The reference run used val/*; retain the first SDTT aliases too.
            for prefix in ('val', 'pareto'):
                name = f'{prefix}/gsm8k_{suffix}_thr{tag}'
                metrics[name] = point[key]
                if not initial:
                    model.log(name, torch.tensor(point[key], device=model.device, dtype=torch.float32),
                              on_epoch=True, on_step=False, sync_dist=False, rank_zero_only=True)
    if model.trainer.global_rank != 0:
        return
    step = int(model.global_step)
    if initial and logger is not None:
        for destination in model.trainer.loggers:
            destination.log_metrics(metrics, step=step)
    path = Path(model.config.checkpointing.save_dir) / 'pareto_history.json'
    if getattr(model, '_gsm8k_pareto_history', None) is None:
        history = {}
        if step > 0 and model.trainer.ckpt_path is not None and path.is_file():
            try:
                stored = json.loads(path.read_text())
                # A new run may reuse a directory, or resume an earlier checkpoint.
                history = {int(k): v for k, v in stored['curves'].items() if int(k) <= step}
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.warn(f'Could not restore GSM8K Pareto history: {error}')
        model._gsm8k_pareto_history = history
    history = model._gsm8k_pareto_history
    history[step] = curve
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'curves': history, 'greedy': greedy,
                                     'protocol': getattr(model, '_gsm8k_protocol', {})}, indent=2) + '\n')
    temporary.replace(path)
    if logger is None:
        return
    for logger in model.trainer.loggers:
        for point in sorted(curve, key=lambda p: p['threshold']):
            logger.log_metrics({f'pareto/gsm8k_step_{step}': point['accuracy']},
                               step=int(round(point['nfe'] * 100)))

        rows = getattr(model, '_gsm8k_sample_records', [])
        if rows and hasattr(logger, 'log_table'):
            logger.log_table('val/samples',
                             ['question', 'generated', 'reference', 'grade', 'NFE', 'threshold'],
                             [[row['question'], row['generated'], row['reference'], row['grade'],
                               row['nfe'], row['threshold']] for row in rows], step=step)
        if hasattr(logger, 'log_image'):
            try:
                logger.log_image('pareto/frontier', render_frontier(history, step), step=step)
            except Exception as error:
                warnings.warn(f'Could not render GSM8K Pareto frontier: {error}')


def render_frontier(history, current_step):
    """Render actual prior and current accuracy/NFE curves without interpolation."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5), dpi=120, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    older_label = True
    for step, curve in sorted(history.items(), key=lambda item: int(item[0])):
        ordered = sorted(curve, key=lambda point: point['nfe'])
        current = int(step) == int(current_step)
        label = f'Step {step}' if current else ('Earlier checkpoints' if older_label else None)
        axes.plot([point['nfe'] for point in ordered],
                  [100 * point['accuracy'] for point in ordered],
                  color='#1763a6' if current else '#888888',
                  alpha=1. if current else .3,
                  linewidth=2.2 if current else 1.,
                  marker='o' if current else None, markersize=3, label=label)
        if not current:
            older_label = False
    axes.set(xlabel='Mean denoiser calls per example (NFE)',
             ylabel='GSM8K execution accuracy (%)', title='TinyGSM confidence-threshold frontier',
             ylim=(0, 100))
    axes.grid(alpha=.2)
    if history:
        axes.legend()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
