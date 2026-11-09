"""Resolve CDLM checkpoint assets once, before starting distributed workers."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, read_write

from cdlm_checkpoint_assets import prepare_checkpoint


def compose_config(overrides, code_dir=None):
    """Resolve only asset fields; composing must not import the training stack."""
    code_dir = Path(code_dir or Path.cwd()).resolve()
    if not OmegaConf.has_resolver('cwd'):
        OmegaConf.register_new_resolver('cwd', os.getcwd)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent / 'configs'),
                               version_base=None):
        config = compose(config_name='config', overrides=list(overrides),
                         return_hydra_config=True)
    config.hydra.runtime.cwd = str(code_dir)
    hydra_config = HydraConfig.instance()
    previous = hydra_config.cfg
    try:
        hydra_config.set_config(config)
        # Asset paths may refer to the launch directory or to hydra.run.dir.
        output_dir = Path(str(config.hydra.run.dir)).expanduser()
        if not output_dir.is_absolute():
            output_dir = code_dir / output_dir
        with read_write(config.hydra.runtime):
            config.hydra.runtime.output_dir = str(output_dir)
        hydra_config.set_config(config)
        for key in ('training.finetune_path', 'training.student_init_path',
                    'training.teacher_allow_hub', 'model_weights_dir',
                    'checkpointing.snapshot', 'checkpointing.resume_from_ckpt'):
            value = OmegaConf.select(config, key)
            OmegaConf.update(config, key, value)
        return config
    finally:
        hydra_config.cfg = previous


def prepare_overrides(config, *, code_dir=None):
    """Preserve weight-selection flags and emit only resolved checkpoint paths."""
    if config.algo.name != 'cdlm':
        raise ValueError('CDLM checkpoint preflight requires algo=cdlm.')
    code_dir = Path(code_dir or Path.cwd()).resolve()
    teacher = config.training.get('finetune_path')
    if not teacher:
        raise ValueError('training.finetune_path must identify the CDLM trajectory teacher.')
    weights_dir = config.get('model_weights_dir')
    if weights_dir:
        weights_dir = Path(str(weights_dir)).expanduser()
        if not weights_dir.is_absolute():
            weights_dir = code_dir / weights_dir
    if (weights_dir and str(teacher).startswith('hf://')
            and not config.training.get('teacher_allow_hub', False)):
        raise ValueError('With model_weights_dir, supply a local teacher checkpoint or '
                         'explicitly set training.teacher_allow_hub=True.')
    resolved = {}

    def resolve(source):
        source = str(source)
        if source not in resolved:
            resolved[source] = str(prepare_checkpoint(
                source, code_dir=code_dir, input_dir=os.environ.get('INPUT_PATH'),
                weights_dir=weights_dir,
                snapshot_enabled=bool(config.checkpointing.get('snapshot', False))))
        return resolved[source]

    values = {'training.finetune_path': resolve(teacher)}
    student = config.training.get('student_init_path')
    if student and not config.checkpointing.get('resume_from_ckpt', False):
        values['training.student_init_path'] = resolve(student)
    # These are Hydra arguments, not shell code; quoted values also protect
    # spaces, commas and '=' in local filenames. The launcher never uses eval.
    return [f'{key}={json.dumps(value, ensure_ascii=False)}'
            for key, value in values.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='Write NUL-separated Hydra path overrides to this file.')
    options, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    output = Path(options.output).expanduser()
    try:
        output.unlink(missing_ok=True)
        print('[cdlm-assets] Preparing checkpoint assets before GPU workers.', flush=True)
        config = compose_config(overrides)
        values = prepare_overrides(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.' + output.name, dir=output.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(('\0'.join(values) + '\0').encode('utf-8'))
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print('[cdlm-assets] Checkpoints ready; starting trajectory preparation.', flush=True)
        return 0
    except Exception as error:
        print(f'[cdlm-assets] Preflight failed before GPU launch: {error}', file=sys.stderr,
              flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
