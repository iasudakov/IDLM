#!/usr/bin/env python3
"""Internal entry point: original inference main with one explicit diagnostic.

Launch via eval_didi_precision.sh so source/checkpoint/environment preflight is
not bypassed. This does not install the integrated training/runtime patches.
"""

from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from didi_repro.inference_precision import load_precision_sampler
from didi_repro.standalone import verify_source


def main():
    report = Path(os.environ['DIDI_PRECISION_REPORT']).resolve()
    provenance_path = report.parent / 'authors_eval_environment.json'
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get('precision_diagnostic') or {}).get('diagnostic') != 'float64_sampling_arithmetic':
        raise RuntimeError('Run via eval_didi_precision.sh with completed diagnostic preflight.')
    if report.exists():
        raise FileExistsError(report)
    directory, _ = verify_source('inference')
    sys.path.insert(0, str(directory))
    upstream_main = importlib.import_module('main')
    upstream_algo = importlib.import_module('algo')
    if Path(upstream_main.__file__).resolve() != directory / 'main.py':
        raise RuntimeError('Conflicting main module; use a fresh standalone process.')
    if Path(upstream_algo.__file__).resolve() != directory / 'algo.py':
        raise RuntimeError('Conflicting algo module; use a fresh standalone process.')

    observations = Counter()

    def observe(stage, tensor):
        # Only inspect metadata: no RNG draws or GPU reductions/synchronization.
        observations[(stage, str(tensor.dtype), tuple(tensor.shape))] += 1

    function, metadata = load_precision_sampler(upstream_algo, observe)
    if metadata != provenance['precision_diagnostic']:
        raise RuntimeError('Diagnostic adapter changed after preflight.')
    owner = upstream_algo.DiDiInstruct
    original = owner.reward_guided_ancestral_step
    owner.reward_guided_ancestral_step = function
    print('[precision-diagnostic] float64 sampling arithmetic ONLY; '
          'original model forwards, gradient calculation, cleanup and scorer.', flush=True)
    success = False
    try:
        # Imported @hydra.main would otherwise search pkg://configs rather than
        # the script-relative directory used by `python main.py`. Select the
        # exact original config directory explicitly, without config overrides.
        sys.argv.insert(1, '--config-path=' + str(directory / 'configs'))
        upstream_main.main()
        success = True
    finally:
        owner.reward_guided_ancestral_step = original
        import torch
        with report.open('x') as stream:
            json.dump({
                'completed': success,
                'precision': metadata,
                'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
                'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
                'observed_tensors': [
                    {'stage': stage, 'dtype': dtype, 'shape': shape, 'calls': count}
                    for (stage, dtype, shape), count in sorted(observations.items())],
            }, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
