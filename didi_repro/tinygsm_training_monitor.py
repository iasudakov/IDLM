"""Serializable TinyGSM training counts and adapter source identification.

Timing is elapsed host wall time supplied by the training wrapper, with no
CUDA synchronization. It excludes validation and time outside that wrapper.
"""

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def _seconds(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError('elapsed_seconds must be finite and nonnegative.')
    return float(value)


class TinyGSMTrainingCounters:
    """Count completed training work; call once after optimizer updates.

    Row and trajectory counts are global exposures, including repeated rows
    across epochs. They are not unique examples or GPU-time/FLOP estimates.
    The caller supplies actual global batch sizes, so a short batch is counted
    correctly. Checkpoint loading requires complete measured history.
    """

    SCHEMA = 'didi-tinygsm-training-counters'
    VERSION = 1
    _COUNTS = ('student_updates', 'discriminator_updates', 'prompt_rows',
               'candidate_trajectories')

    def __init__(self):
        for name in self._COUNTS:
            setattr(self, name, 0)
        self.train_wall_seconds = 0.0

    def observe_step(self, *, global_prompt_rows, candidate_trajectories,
                     student_updates, discriminator_updates, elapsed_seconds):
        """Validate all inputs before atomically adding one completed step."""
        rows = _integer(global_prompt_rows, 'global_prompt_rows', 1)
        trajectories = _integer(candidate_trajectories, 'candidate_trajectories', 1)
        student = _integer(student_updates, 'student_updates')
        discriminator = _integer(discriminator_updates, 'discriminator_updates')
        elapsed = _seconds(elapsed_seconds)
        if student not in (0, 1) or discriminator != 1:
            raise ValueError('A completed DiDi step has one discriminator update '
                             'and zero or one student updates.')
        if trajectories < rows or trajectories % rows:
            raise ValueError('candidate_trajectories must be an integer number '
                             'of draws per global prompt row.')
        total_seconds = self.train_wall_seconds + elapsed
        if not math.isfinite(total_seconds):
            raise ValueError('Accumulated training wall time is not finite.')
        self.student_updates += student
        self.discriminator_updates += discriminator
        self.prompt_rows += rows
        self.candidate_trajectories += trajectories
        self.train_wall_seconds = total_seconds

    def metrics(self):
        """Return exact Python counts; use float64 tensors for tracker logging."""
        values = {f'didi/{name}': getattr(self, name) for name in self._COUNTS}
        values['didi/optimizer_updates'] = self.student_updates + self.discriminator_updates
        values['didi/train_wall_seconds'] = self.train_wall_seconds
        return values

    def state_dict(self):
        return {'schema': self.SCHEMA, 'version': self.VERSION,
                **{name: getattr(self, name) for name in self._COUNTS},
                'train_wall_seconds': self.train_wall_seconds}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get('schema') != self.SCHEMA
                or type(state.get('version')) is not int
                or state['version'] != self.VERSION):
            raise ValueError('Missing or unsupported TinyGSM training-counter history.')
        if set(state) != set(self.state_dict()):
            raise ValueError('TinyGSM training-counter history is incomplete or has unknown fields.')
        counts = {name: _integer(state[name], name) for name in self._COUNTS}
        elapsed = _seconds(state['train_wall_seconds'])
        if counts['student_updates'] > counts['discriminator_updates']:
            raise ValueError('Student updates cannot exceed completed DiDi steps.')
        if counts['candidate_trajectories'] < counts['prompt_rows']:
            raise ValueError('Trajectory count cannot be smaller than prompt-row count.')
        if counts['prompt_rows'] < counts['discriminator_updates']:
            raise ValueError('Each completed DiDi step must contain a prompt row.')
        if counts['discriminator_updates'] == 0 and (any(counts.values()) or elapsed):
            raise ValueError('Nonempty training history requires a completed DiDi step.')
        for name, value in counts.items():
            setattr(self, name, value)
        self.train_wall_seconds = elapsed


def source_provenance(repo_root):
    """Hash adapter source, including untracked code and git-less archives.

    Only explicitly selected source/config files are hashed. File contents,
    environment variables and Git remote URLs are never recorded.
    """
    root = Path(repo_root).resolve()
    adapter = root / 'didi_repro'
    if not adapter.is_dir():
        raise ValueError(f'Missing DiDi adapter directory: {adapter}')
    paths = set(adapter.rglob('*.py'))
    config_root = adapter / 'configs'
    if config_root.is_dir():
        for pattern in ('*.yaml', '*.yml', '*.json'):
            paths.update(config_root.rglob(pattern))
    for relative in ('didi_repro/UPSTREAM.json', 'scripts/train_didi_tinygsm.sh',
                     'sandbox_gsm8k.py', 'underdeep_logger.py'):
        path = root / relative
        if path.is_file():
            paths.add(path)
    manifest = {}
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    git = {'available': False, 'head': None, 'dirty': None}
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel', 'HEAD'],
                              capture_output=True, text=True, timeout=10, check=True)
        top, revision = head.stdout.strip().splitlines()
        if Path(top).resolve() != root:
            raise ValueError('Archive directory is not the Git worktree root.')
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                                 '--untracked-files=normal'], capture_output=True,
                                text=True, timeout=10, check=True)
        git = {'available': True, 'head': revision,
               'dirty': bool(status.stdout.strip())}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {'schema': 'didi-tinygsm-adapter-source', 'version': 1,
            'git': git, 'files': manifest,
            'sha256': hashlib.sha256(encoded).hexdigest()}
