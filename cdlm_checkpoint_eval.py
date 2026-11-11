"""Plan a common evaluation of the three selected TinyGSM CDLM chains.

The parent process inventories available checkpoints and checks offline assets
before torchrun. Missing catalog checkpoints are reported and skipped by default.
Only selected checkpoint files are staged from archive inputs; trajectories are
never extracted. Actual model evaluation lives in cdlm_checkpoint_eval_worker.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


SOURCE_RUNS = {
    '5e0ee1da-8780-42d3-b120-9590e9dc8351': {
        'family': 'MDLM to CDLM', 'namespace': 'cdlm_tinygsm_ancestral_sweep',
        'input_name': 'cdlm_state',
        'checkpoints': [('12-1000.ckpt', 1000), ('last.ckpt', 1248)],
    },
    'dbb3b25d-4937-4f27-966f-898e696a1f48': {
        'family': 'CDLM to MMD', 'namespace': 'mmd_from_cdlm_tinygsm',
        'input_name': 'cdlm_mmd_state',
        'checkpoints': [(f'step={step:08d}.ckpt', step) for step in range(500, 7001, 500)],
    },
    '8cb53b89-96e7-4fe4-ba49-91ed2eb9fb20': {
        'family': 'MMD to CDLM', 'namespace': 'cdlm_tinygsm',
        'input_name': 'mmd_cdlm_state',
        'checkpoints': [('12-1000.ckpt', 1000), ('last.ckpt', 1248)],
    },
}


def resolve_source_run(value):
    value = str(value).strip()
    matches = [uid for uid in SOURCE_RUNS if value and uid.startswith(value)]
    if len(matches) != 1:
        raise ValueError(f'Unknown or ambiguous source run {value!r}; use one of {list(SOURCE_RUNS)}')
    return matches[0]


def _within(path, root):
    return path == root or root in path.parents


def _validate_missing_policy(value):
    if value not in ('skip', 'error'):
        raise ValueError(f'Unknown missing checkpoint policy: {value!r}; use skip or error.')


def build_plan(sources, output_dir, *, missing_checkpoints='skip'):
    """Inventory requested sources without reading tensors or writing files."""
    _validate_missing_policy(missing_checkpoints)
    if not sources:
        raise ValueError('At least one source snapshot is required.')
    if not str(output_dir).strip():
        raise ValueError('An output directory is required.')
    output = Path(output_dir).expanduser().resolve()
    entries, skipped, seen = [], [], set()
    for source in sources:
        uid = resolve_source_run(source['source_run'])
        if uid in seen:
            raise ValueError(f'Duplicate source run: {uid}')
        seen.add(uid)
        value = source.get('snapshot_root')
        if value is None or not str(value).strip():
            raise ValueError(f'Snapshot path is empty for {uid}.')
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'Unpacked snapshot directory not found: {root}')
        if _within(output, root) or _within(root, output):
            raise ValueError('Evaluation output and input snapshots must not overlap.')
        spec = SOURCE_RUNS[uid]
        directory = (root / spec['namespace'] / 'checkpoints').resolve()
        if not _within(directory, root):
            raise ValueError(f'Checkpoint namespace escapes input snapshot: {directory}')
        source_count = 0
        for filename, step in spec['checkpoints']:
            path = (directory / filename).resolve()
            if not _within(path, directory):
                raise ValueError(f'Checkpoint link escapes its namespace: {path}')
            if path.exists() and not path.is_file():
                raise ValueError(f'Checkpoint path is not a regular file: {path}')
            reason = 'missing' if not path.is_file() else ('empty' if path.stat().st_size == 0 else None)
            if reason is not None:
                if missing_checkpoints == 'error':
                    raise FileNotFoundError(f'Required checkpoint missing or empty: {path}')
                skipped.append({'source_run': uid, 'family': spec['family'],
                                'checkpoint': str(path), 'step': step, 'reason': reason})
                continue
            entries.append({'source_run': uid, 'family': spec['family'],
                            'checkpoint': str(path), 'step': step,
                            'output_dir': str(output / uid / f'step={step:08d}')})
            source_count += 1
        if source_count == 0:
            raise FileNotFoundError(f'No usable expected checkpoints for source {uid} in {directory}. '
                                    'Check that the matching training snapshot is connected; '
                                    'no other namespace or inherited last.ckpt will be used.')
    return {'schema_version': 1, 'output_dir': str(output),
            'snapshot_namespace': 'cdlm_checkpoint_eval', 'entries': entries,
            'missing_checkpoints': missing_checkpoints, 'skipped_checkpoints': skipped}


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stage_archive(archive, uid, staging_root, *, missing_checkpoints='skip'):
    """Extract available catalog checkpoint members, never arbitrary tar paths."""
    _validate_missing_policy(missing_checkpoints)
    archive = Path(archive).expanduser().resolve()
    destination = (Path(staging_root) / uid).resolve()
    spec = SOURCE_RUNS[uid]
    expected = {f"{spec['namespace']}/checkpoints/{filename}"
                for filename, _ in spec['checkpoints']}
    stat = archive.stat()
    identity = {'archive': str(archive), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    receipt_path = destination / 'archive_receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        files = receipt.get('files', {})
        usable = {name for name, size in files.items() if size > 0}
        if (receipt.get('archive_identity') == identity and set(files) <= expected and usable
                and (missing_checkpoints == 'skip' or usable == expected)):
            if all(_within((destination / name).resolve(), destination)
                   and (destination / name).is_file() and (destination / name).stat().st_size == size
                   for name, size in files.items()) and not any(
                       (destination / name).exists() for name in expected - set(files)):
                return destination
    print(f'[checkpoint-eval] Indexing archive {archive}; looking for {len(expected)} catalog checkpoints.', flush=True)
    with tarfile.open(archive, mode='r:*') as tar:
        selected = {}
        for member in tar:
            parts = PurePosixPath(member.name).parts
            # Prefixes used by Nirvana tar writers may differ; suffixes must match exactly.
            suffix = '/'.join(parts[-3:])
            if suffix not in expected:
                continue
            if member.name.startswith('/') or '..' in parts or not member.isfile() or member.size < 0:
                raise ValueError(f'Invalid checkpoint archive member: {member.name}')
            if suffix in selected:
                raise ValueError(f'Ambiguous duplicate checkpoint in archive: {suffix}')
            selected[suffix] = member
        usable = {name for name, member in selected.items() if member.size > 0}
        missing = expected - usable
        if missing and missing_checkpoints == 'error':
            raise FileNotFoundError(f'Archive is missing required checkpoints: {sorted(missing)}')
        if not usable:
            raise FileNotFoundError(f'No usable expected checkpoints for source {uid} in archive {archive}.')
        # An archive can be replaced by an earlier/partial snapshot at the same
        # input path. Never let old staged weights fill gaps in the new state.
        for name in expected - set(selected):
            target = destination / name
            if not _within(target.resolve(), destination):
                raise ValueError('Staged checkpoint path escapes the staging directory.')
            target.unlink(missing_ok=True)
        sizes = {}
        for name, member in sorted(selected.items()):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not _within(target.resolve(), destination):
                raise ValueError('Staged checkpoint path escapes the staging directory.')
            fd, temporary = tempfile.mkstemp(prefix='.' + target.name, dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as stream, tar.extractfile(member) as source:
                    shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
                if Path(temporary).stat().st_size != member.size:
                    raise IOError(f'Incomplete checkpoint extracted: {name}')
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            sizes[name] = member.size
    after = archive.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Snapshot archive changed during extraction.')
    _atomic_json(receipt_path, {'archive_identity': identity, 'files': sizes})
    return destination


def prepare_sources(sources, staging_root, *, missing_checkpoints='skip'):
    _validate_missing_policy(missing_checkpoints)
    prepared = []
    for source in sources:
        uid = resolve_source_run(source['source_run'])
        value = source.get('snapshot_root')
        if value is None or not str(value).strip():
            raise ValueError(f'Snapshot path is empty for {uid}.')
        path = Path(value).expanduser().resolve()
        if path.is_file():
            path = stage_archive(path, uid, staging_root, missing_checkpoints=missing_checkpoints)
        elif not path.is_dir():
            raise FileNotFoundError(f'Snapshot input not found: {path}')
        prepared.append({'source_run': uid, 'snapshot_root': str(path)})
    return prepared


def compose_evaluation_config(overrides, *, code_dir, output_dir, devices):
    """Compose and resolve offline inference settings without importing the model."""
    from cdlm_preflight import compose_config
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    defaults = ['+experiment=cdlm_checkpoint_eval', f'trainer.devices={devices}',
                'checkpointing.save_dir=' + json.dumps(str(output_dir)),
                'hydra.run.dir=' + json.dumps(str(output_dir)),
                'data.cache_dir=' + json.dumps(str(code_dir / 'cache_dir')),
                'model_weights_dir=' + json.dumps(str(code_dir / 'cache_dir/tinygsm_mmd_offline_bundle'))]
    config = compose_config(defaults + list(overrides), code_dir=code_dir)
    hydra_config, previous = HydraConfig.instance(), HydraConfig.instance().cfg
    try:
        hydra_config.set_config(config)
        plain = OmegaConf.to_container(config, resolve=False)
        plain.pop('hydra', None)
        config = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(plain), resolve=True))
    finally:
        hydra_config.cfg = previous
    if (config.algo.name != 'cdlm' or config.model.get('force_fp32', False)
            or config.trainer.precision != 'bf16-mixed' or config.eval.disable_ema
            or config.data.train != 'tiny_gsm' or config.model.length != 512
            or config.adversarial_distill.is_distill or config.trainer.devices != devices):
        raise ValueError('Sweep requires the TinyGSM CDLM inference adapter, length512, BF16 mixed forward and EMA.')
    if config.eval.gsm8k_pareto_temperature != 0:
        raise ValueError('This sweep uses confidence T=0 when confidence sampling is enabled.')
    if not config.eval.gsm8k_greedy_token or not config.eval.gsm8k_ancestral_final_denoise:
        raise ValueError('Ancestral evaluation with separate final denoise must be enabled.')
    if config.logger.get('_target_') != 'underdeep_logger.UnderdeepLogger':
        raise ValueError('This sweep publishes to Underdeep; keep logger=underdeep.')
    # Resolve local-only tokenizer assets now, before GPU workers are created.
    import model_assets
    model_assets.configure_model_weights(config, base_dir=code_dir)
    if not config.model_weights_dir:
        raise ValueError('Provide the existing offline tokenizer bundle via model_weights_dir.')
    model_assets.resolve_pretrained_path(config.data.tokenizer_name_or_path,
                                        config.model_weights_dir, kind='tokenizer')
    explicit = config.eval.get('gsm8k_data_path')
    candidates = ([Path(explicit).expanduser()] if explicit else
                  [Path(config.data.cache_dir) / 'gsm8k_test.json', code_dir / 'data/gsm8k_test.json'])
    candidates = [path if path.is_absolute() else code_dir / path for path in candidates]
    data_path = next((path.resolve() for path in candidates if path.is_file()), None)
    if data_path is None:
        raise FileNotFoundError('Local GSM8K JSON missing; set eval.gsm8k_data_path. No dataset download is attempted.')
    records = json.loads(data_path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('GSM8K JSON must be a nonempty list.')
    if config.eval.gsm8k_num_examples is None and len(records) != 1319:
        raise ValueError(f'Expected all 1319 GSM8K test examples, found {len(records)}.')
    config.eval.gsm8k_data_path = str(data_path)
    from cdlm_checkpoint_eval_worker import validate_config
    validate_config(config)
    return config


def restore_evaluation_output(output, namespace):
    """Restore only evaluation results, never the source training tracker."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        return False
    import nirvana_snapshot
    if not nirvana_snapshot.available():
        return False
    value = nirvana_snapshot._sdk().snapshot.get_snapshot_path()
    if value is None or not str(value).strip():
        return False
    root = Path(value).expanduser().resolve()
    if not (root / namespace).is_dir():
        return False
    if not nirvana_snapshot.restore(output, namespace=namespace):
        raise RuntimeError('Saved evaluation results exist but could not be restored from snapshot.')
    return True


def write_summary(plan):
    """Export comparable rows after a successful evaluation (including resumed reports)."""
    import csv
    rows = []
    for entry in plan['entries']:
        path = Path(entry['output_dir']) / 'result.json'
        result = json.loads(path.read_text())
        if result.get('complete') is not True:
            raise ValueError(f'Evaluation did not complete: {path}')
        base = {'source_run': entry['source_run'], 'family': entry['family'],
                'checkpoint_step': entry['step'],
                'checkpoint_sha256': result['identity']['checkpoint_sha256']}
        for precision, points in result['confidence'].items():
            for point in points:
                rows.append({**base, 'sampler': 'confidence', 'logit_normalization': precision,
                             'temperature': result['protocol'].get('confidence', {}).get('temperature', 0.),
                             'threshold': point['threshold'], 'sampling_steps': '',
                             'seed': result['protocol']['seed'], 'accuracy': point['accuracy'],
                             'nfe': point['nfe'], 'correct': point['correct'], 'count': point['count']})
        for point in result['ancestral']['points']:
            for seed in point['seeds']:
                rows.append({**base, 'sampler': 'ancestral', 'logit_normalization': 'fp32',
                             'temperature': point.get('temperature', 0.),
                             'threshold': '', 'sampling_steps': point['steps'],
                             'seed': seed['seed'], 'accuracy': seed['accuracy'],
                             'nfe': seed['nfe'], 'correct': seed['correct'], 'count': seed['count']})
    path = Path(plan['output_dir']) / 'summary.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def execute_plan(plan, config, *, devices, plan_only=False, restore_results=True,
                 worker_env=None):
    """Publish a checked plan and run the common evaluation/reporting lifecycle."""
    from cdlm_checkpoint_eval_worker import validate_plan
    from omegaconf import OmegaConf
    output = validate_plan(plan)
    if type(devices) is not int or devices <= 0:
        raise ValueError('Evaluation devices must be a positive integer.')
    if restore_results:
        restore_evaluation_output(output, plan['snapshot_namespace'])
    output.mkdir(parents=True, exist_ok=True)
    plan['devices'] = devices
    _atomic_json(output / 'plan.json', plan)
    OmegaConf.save(config, output / 'evaluation_config.yaml')
    if plan_only:
        print(f'[checkpoint-eval] Plan only; no GPU work: {output / "plan.json"}', flush=True)
        return 0
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={devices}', '-m', 'cdlm_checkpoint_eval_worker',
               '--plan', str(output / 'plan.json'), '--config', str(output / 'evaluation_config.yaml')]
    kwargs = {} if worker_env is None else {'env': worker_env}
    result = subprocess.run(command, check=False, **kwargs)
    if result.returncode == 0:
        summary = write_summary(plan)
        print(f'[checkpoint-eval] Combined results: {summary}', flush=True)
        if config.checkpointing.snapshot:
            import nirvana_snapshot
            if not nirvana_snapshot.save(output, namespace=plan['snapshot_namespace'], reason='evaluation CSV summary'):
                raise RuntimeError('Summary is available locally but could not be published to snapshot.')
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default=os.environ.get('CDLM_EVAL_SOURCE_RUN'))
    parser.add_argument('--snapshot-root', default=os.environ.get('CDLM_EVAL_SNAPSHOT_ROOT'))
    parser.add_argument('--sources-json', help='JSON list of source_run/snapshot_root objects; paths can be unpacked directories or tar archives.')
    parser.add_argument('--input-root', default=os.environ.get('INPUT_PATH'), help='Directory containing cdlm_state, cdlm_mmd_state and mmd_cdlm_state inputs.')
    parser.add_argument('--output-dir', default=os.environ.get('CDLM_EVAL_OUTPUT_DIR', str(Path.cwd() / 'cdlm_checkpoint_eval')))
    parser.add_argument('--devices', type=int, default=4,
                        help='Number of GPU workers (default: 4); use --devices 1 for one GPU.')
    parser.add_argument('--missing-checkpoints', choices=('skip', 'error'), default='skip',
                        help='Skip and report absent/empty catalog checkpoints (default), or require the complete catalog.')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--no-restore-results', action='store_true', help='Do not restore prior evaluation outputs from the connected snapshot.')
    parser.add_argument('--list-sources', action='store_true')
    options, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    if options.list_sources:
        print(json.dumps(SOURCE_RUNS, ensure_ascii=False, indent=2))
        return 0
    if options.devices <= 0 or any('=' not in value or value.startswith('--') for value in overrides):
        raise ValueError('Use a positive --devices value and Hydra key=value overrides.')
    if options.sources_json:
        if options.source_run or options.snapshot_root:
            raise ValueError('--sources-json cannot be combined with a single source selection.')
        sources = json.loads(Path(options.sources_json).expanduser().read_text())
        if not isinstance(sources, list):
            raise ValueError('--sources-json must contain a list.')
    elif options.source_run:
        root = options.snapshot_root
        if root is None:
            from cdlm_snapshot_init import resolve_snapshot_root
            root = str(resolve_snapshot_root())
        sources = [{'source_run': options.source_run, 'snapshot_root': root}]
    else:
        if options.snapshot_root:
            raise ValueError('--snapshot-root requires --source-run.')
        if not options.input_root:
            raise ValueError('Connect three named snapshot inputs, or provide --sources-json / --source-run.')
        sources = [{'source_run': uid, 'snapshot_root': str(Path(options.input_root) / spec['input_name'])}
                   for uid, spec in SOURCE_RUNS.items()]
    output = Path(options.output_dir).expanduser().resolve()
    staging = output.parent / ('.' + output.name + '-weights')
    prepared = prepare_sources(sources, staging, missing_checkpoints=options.missing_checkpoints)
    plan = build_plan(prepared, output, missing_checkpoints=options.missing_checkpoints)
    for source in prepared:
        uid = source['source_run']
        steps = [entry['step'] for entry in plan['entries'] if entry['source_run'] == uid]
        print(f'[checkpoint-eval] {uid}: selected {len(steps)} available checkpoints at steps {steps}.', flush=True)
    for item in plan['skipped_checkpoints']:
        print(f'[checkpoint-eval] WARNING: Skipping {item["reason"]} checkpoint '
              f'(step={item["step"]}): {item["checkpoint"]}', flush=True)
    config = compose_evaluation_config(overrides, code_dir=Path.cwd().resolve(),
                                       output_dir=output, devices=options.devices)
    print(f'[checkpoint-eval] Verified {len(plan["entries"])} checkpoints from {len(prepared)} source runs; '
          f'skipped {len(plan["skipped_checkpoints"])} absent/empty checkpoints.', flush=True)
    return execute_plan(plan, config, devices=options.devices, plan_only=options.plan_only,
                        restore_results=not options.no_restore_results)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, tarfile.TarError) as error:
        print(f'[checkpoint-eval] Preflight failed: {error}', file=sys.stderr)
        raise SystemExit(2)
