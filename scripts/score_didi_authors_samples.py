#!/usr/bin/env python3
"""Offline-only, all-sample rescoring with the pristine authors' PPL metric.

This script never generates text, starts training, or contacts Underdeep. It
preserves the original samples JSON and writes a separate, exclusive-create
report. The original entropy is retained, not reconstructed from decoded text.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / 'vendor' / 'didi_instruct' / 'metrics.py'


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--samples', required=True, type=Path)
    result.add_argument('--output', required=True, type=Path)
    result.add_argument('--batch-size', type=positive_int, default=4)
    result.add_argument('--device', default='cuda')
    result.add_argument('--model', default='gpt2-large',
                        help='Already-cached HF repository or local snapshot directory.')
    result.add_argument('--revision', help='Cached HF revision; never downloaded.')
    result.add_argument('--cache-dir', type=Path)
    result.add_argument('--max-length', type=positive_int, default=1024)
    result.add_argument('--original-generation-batch-size', type=positive_int, default=20)
    result.add_argument('--original-perplexity-batch-size', type=positive_int, default=8)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_samples(path):
    with path.open(encoding='utf-8') as stream:
        original = json.load(stream)
    texts = original.get('generated_seqs') if isinstance(original, dict) else None
    if not isinstance(texts, list) or not texts or not all(isinstance(x, str) for x in texts):
        raise ValueError('Samples JSON must contain a nonempty generated_seqs list of strings.')
    for key in ('generative_ppl', 'entropy'):
        value = original.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Samples JSON requires a finite numeric {key}.')
    return original, texts


def load_pristine_metrics():
    manifest = json.loads((REPO_ROOT / 'didi_repro' / 'UPSTREAM.json').read_text())
    actual_hash = sha256(METRICS_PATH)
    if actual_hash != manifest['files']['metrics.py']:
        raise RuntimeError('Vendored metrics.py does not match the pristine upstream manifest.')
    name = '_didi_authors_pristine_rescore_metrics'
    spec = importlib.util.spec_from_file_location(name, METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module.Metrics, {'commit': manifest['commit'],
                            'metrics_path': str(METRICS_PATH),
                            'metrics_sha256': actual_hash}


def resolve_cached_model(source, revision=None, cache_dir=None):
    local = Path(source).expanduser()
    if local.is_dir():
        directory = local.absolute()
    elif local.exists() or source.startswith(('/', './', '../', '~')):
        raise FileNotFoundError(f'Evaluator snapshot directory does not exist: {source}')
    else:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(
            repo_id=source, revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=True))
    resolved_revision = (directory.name if directory.parent.name == 'snapshots'
                         and re.fullmatch(r'[0-9a-f]{40}', directory.name) else None)
    if (revision is not None and re.fullmatch(r'[0-9a-f]{40}', revision)
            and resolved_revision is not None and revision != resolved_revision):
        raise ValueError('Evaluator snapshot does not match the requested immutable revision.')
    config_path = directory / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Incomplete cached evaluator: missing {config_path}')
    return directory, {'source': source, 'requested_revision': revision,
                       'resolved_revision': resolved_revision,
                       'snapshot_path': str(directory),
                       'config_sha256': sha256(config_path),
                       'local_files_only': True}


def inferred_original_count(count, generation_batch_size, perplexity_batch_size):
    """Original metric drops the remainder separately in each generation batch."""
    total = 0
    for start in range(0, count, generation_batch_size):
        size = min(generation_batch_size, count - start)
        batch = min(size, perplexity_batch_size)
        total += size // batch * batch
    return total


def score_all(evaluator, texts, batch_size, max_length, device):
    """Keep original internal batching; score any final remainder separately.

    One call for the divisible prefix avoids loading the frozen evaluator once
    per batch, while the second call includes the tail the original loop drops.
    No metric equations or model-loading functions are patched.
    """
    evaluator.gen_ppl.reset()
    full = len(texts) // batch_size * batch_size
    chunks = ([texts[:full]] if full else []) + ([texts[full:]] if full < len(texts) else [])
    processed = 0
    for chunk in chunks:
        evaluator.record_generative_perplexity(
            chunk, max_length=max_length, retokenize=True, device=device)
        processed += len(chunk)
        print(f'[authors-rescore] Scored {processed}/{len(texts)} samples.', flush=True)
    result = float(evaluator.gen_ppl.compute())
    if not math.isfinite(result):
        raise RuntimeError('All-sample perplexity is not finite; no report written.')
    return result


def environment(torch, device):
    result = {'python': platform.python_version(), 'executable': sys.executable,
              'torch_cuda': torch.version.cuda,
              'default_dtype': str(torch.get_default_dtype()),
              'cuda_autocast_enabled': torch.is_autocast_enabled(),
              'device': str(device)}
    for name in ('torch', 'transformers', 'torchmetrics', 'huggingface-hub', 'numpy'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    if device.type == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(device)
        result['allow_tf32_matmul'] = torch.backends.cuda.matmul.allow_tf32
        result['allow_tf32_cudnn'] = torch.backends.cudnn.allow_tf32
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    samples_path, output_path = args.samples.expanduser().resolve(), args.output.expanduser().resolve()
    if samples_path == output_path or output_path.exists():
        raise FileExistsError('Choose a new output path; original samples and existing reports are never overwritten.')
    original, texts = read_samples(samples_path)
    # Set before importing Transformers/HF Hub; every actual model load also uses
    # a local snapshot path, so no remote code or network request is necessary.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    Metrics, upstream = load_pristine_metrics()
    snapshot, model_metadata = resolve_cached_model(args.model, args.revision, args.cache_dir)
    device = torch.device(args.device)
    evaluator = Metrics(gen_ppl_eval_model_name_or_path=str(snapshot),
                        eval_ppl_batch_size=args.batch_size)
    evaluator.to(device)
    print(f'[authors-rescore] Original PPL={original["generative_ppl"]}; '
          f'rescoring all {len(texts)} saved samples.', flush=True)
    ppl = score_all(evaluator, texts, args.batch_size, args.max_length, device)
    inferred_count = inferred_original_count(
        len(texts), args.original_generation_batch_size, args.original_perplexity_batch_size)
    report = {
        'samples_path': str(samples_path), 'samples_sha256': sha256(samples_path),
        'original': {'generative_ppl': original['generative_ppl'], 'entropy': original['entropy'],
                     'generated_sample_count': len(texts),
                     'inferred_scored_sample_count': inferred_count,
                     'count_assumptions': {'generation_batch_size': args.original_generation_batch_size,
                                           'perplexity_batch_size': args.original_perplexity_batch_size}},
        'all_samples': {'generative_ppl': ppl, 'entropy': original['entropy'],
                        'entropy_source': 'copied unchanged from original token-based evaluation',
                        'scored_sample_count': len(texts),
                        'scored_token_weight': float(evaluator.gen_ppl.weight)},
        'scoring': {'max_length': args.max_length, 'batch_size': args.batch_size,
                    'metric_equations': 'unmodified vendored metrics.py',
                    'model_dtype_policy': 'unmodified upstream from_pretrained; no dtype override',
                    'accumulator_dtype': str(evaluator.gen_ppl.dtype),
                    'model': model_metadata},
        'upstream': upstream, 'environment': environment(torch, device),
        'script_sha256': sha256(Path(__file__)),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    print(f'[authors-rescore] Original PPL ({inferred_count} inferred scored samples): '
          f'{original["generative_ppl"]}; all-{len(texts)} PPL: {ppl}; '
          f'entropy unchanged: {original["entropy"]}\nReport: {output_path}', flush=True)


if __name__ == '__main__':
    main()
