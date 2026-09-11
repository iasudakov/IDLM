"""Lightning logger backed by the `underdeep` experiment tracker.

This mirrors the underdeep setup used in ``Halton-MaskGIT/main.py`` (create or
resume a run from on-disk metadata, log hyper-parameters once, stream scalar
metrics, and finish the run at teardown), but exposes it through the
``lightning.pytorch`` Logger interface so it can be dropped into the IDLM Hydra
``logger`` config in place of ``TensorBoardLogger``.

Every underdeep call is wrapped so a missing or broken ``underdeep`` install
degrades to a no-op instead of crashing training -- exactly as the reference
implementation does.
"""

import os

import torch
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

try:  # optional flatten helpers; fall back to a tiny local flatten if absent
  from lightning.fabric.utilities.logger import (
    _convert_params, _flatten_dict)
except Exception:  # pragma: no cover - very old lightning
  def _convert_params(params):
    return dict(params) if params is not None else {}

  def _flatten_dict(params, delimiter='/'):
    return dict(params)


_META_FILENAME = 'underdeep_metadata.pt'


class UnderdeepLogger(Logger):
  """Stream Lightning metrics to an underdeep run.

  Args:
    experiment: underdeep experiment path, e.g. ``generative-models/idlm``.
    name: run name used when a fresh run is created.
    save_dir: directory in which ``underdeep_metadata.pt`` (the resume handle)
      lives. Kept alongside the checkpoints so resuming training resumes the
      same run.
    resume: when True and metadata exists, resume that run instead of creating
      a new one.
    force_new_run: always create a fresh run, ignoring any existing metadata.
  """

  def __init__(self, experiment='generative-models/idlm', name='idlm',
               save_dir='.', resume=True, force_new_run=False):
    super().__init__()
    self._experiment_path = experiment
    self._name = name
    self._save_dir = os.fspath(save_dir)
    self._resume = resume
    self._force_new_run = force_new_run
    self._run = None
    self._initialized = False
    self._finished = False
    # Lightning calls `finalize` at the end of *every* trainer entry point,
    # `trainer.validate` included -- so a baseline validation before
    # `trainer.fit` (see `eval.validate_before_train`) would close the run
    # before training logs a single scalar, and every later write fails with
    # "Can't log data to finished run". Callers that run a phase before
    # training set this to defer the close; see `main._train`.
    self.defer_finalize = False

  # -- Lightning Logger interface -------------------------------------- #

  @property
  def name(self):
    return self._name

  @property
  def version(self):
    run = self.experiment
    uid = getattr(run, 'uid', None)
    return uid if isinstance(uid, str) else ''

  @property
  def save_dir(self):
    return self._save_dir

  @property
  def experiment(self):
    """The underlying underdeep run (created/resumed lazily on rank zero)."""
    if not self._initialized:
      self._initialized = True
      self._run = self._setup_run()
    return self._run

  @rank_zero_only
  def log_hyperparams(self, params, *args, **kwargs):
    run = self.experiment
    if run is None:
      return
    try:
      params = _flatten_dict(_convert_params(params))
      scalars = {'params/{}'.format(k): v for k, v in params.items()
                 if isinstance(v, (int, float, bool))}
      if scalars:
        run.log(scalars, step=0)
    except Exception as e:  # pragma: no cover - tracker best-effort
      print('[underdeep] hyperparam log failed: {}'.format(e), flush=True)

  @rank_zero_only
  def log_metrics(self, metrics, step=None):
    run = self.experiment
    if run is None:
      return
    scalars = {}
    for key, value in metrics.items():
      if key == 'epoch':
        continue
      if hasattr(value, 'item'):
        value = value.item()
      scalars[key] = value
    if not scalars:
      return
    try:
      run.log(scalars, step=step)
    except Exception as e:  # pragma: no cover - tracker best-effort
      print('[underdeep] scalar log failed: {}'.format(e), flush=True)

  @rank_zero_only
  def log_table(self, key, columns, data, step=None):
    """Log one table (e.g. a batch of generated samples) to the run.

    Lightning has no table hook, so `trainer_base` reaches this the way
    it reaches `log_image`: behind a `hasattr(logger, 'log_table')`
    guard, which a logger without table support (e.g.
    ``TensorBoardLogger``) simply fails, skipping the call. Until this
    existed the underdeep logger was one of those -- every sample table
    the validation loop built was silently dropped.

    Args:
      key: table title. As with `log_image`, repeated calls under the
        same title become a step-indexed series, so pass a *stable* key
        and let `step` carry the training step rather than baking the
        step into the title -- otherwise each validation lands as its
        own unrelated table.
      columns: column names.
      data: rows, each an iterable as long as `columns`.
      step: training step the table belongs to.
    """
    run = self.experiment
    if run is None or not data:
      return
    try:
      from underdeep.runs.data_types import UTable
      rows = [dict(zip(columns, row)) for row in data]
      run.log({key: UTable(value=rows)}, step=step)
    except Exception as e:  # pragma: no cover - tracker best-effort
      print('[underdeep] table log failed: {}'.format(e), flush=True)

  @rank_zero_only
  def log_image(self, key, image, step=None):
    """Log one image to the run, versioned by `step` like a scalar.

    Lightning has no image hook of its own, so callers reach this the way
    ``trainer_base`` already reaches ``log_table``: behind a
    ``hasattr(logger, 'log_image')`` guard, which a logger without image
    support (e.g. ``TensorBoardLogger``) simply fails, skipping the call.

    Args:
      key: image title; repeated calls under the same title become a
        step-indexed series in the tracker, so a curve re-rendered every
        validation can be scrubbed through.
      image: ``numpy`` HWC array, ``PIL`` image, raw bytes, or a path to a
        file on disk -- whatever ``underdeep.runs.data_types.UImage`` takes.
      step: training step the image belongs to.
    """
    run = self.experiment
    if run is None or image is None:
      return
    try:
      if isinstance(image, str):
        run.log_image(title=key, file_path=image, step=step)
      else:
        run.log_image(title=key, array=image, step=step)
    except Exception as e:  # pragma: no cover - tracker best-effort
      print('[underdeep] image log failed: {}'.format(e), flush=True)

  @rank_zero_only
  def finalize(self, status):
    if self.defer_finalize:
      return
    if self._run is None or self._finished:
      return
    self._finished = True
    try:
      self._run.finish()
      print('[underdeep] run finished ({})'.format(status), flush=True)
    except Exception as e:  # pragma: no cover - tracker best-effort
      print('[underdeep] finish failed: {}'.format(e), flush=True)

  # -- internal -------------------------------------------------------- #

  def _meta_path(self):
    return os.path.join(self._save_dir, _META_FILENAME)

  def _setup_run(self):
    """Create or resume the underdeep run. Rank-zero only, never raises."""
    if getattr(rank_zero_only, 'rank', 0) != 0:
      return None
    try:
      import underdeep as U
      client = U.Client(experiment=self._experiment_path)
      meta = self._meta_path()
      if self._resume and not self._force_new_run and os.path.exists(meta):
        run_uid = torch.load(meta, weights_only=False).get('run_uid')
        run = client.resume_from(run_uid)
        print('[underdeep] resumed run {}'.format(run_uid), flush=True)
      else:
        run = client.init_run(name=self._name)
        os.makedirs(self._save_dir, exist_ok=True)
        torch.save({'run_uid': run.uid}, meta)
        why = 'forced new run' if self._force_new_run else 'no prior metadata'
        print("[underdeep] init run '{}' uid={} exp={} ({})".format(
          self._name, run.uid, self._experiment_path, why), flush=True)
      return run
    except Exception as e:
      print('[underdeep] disabled (setup failed: {})'.format(e), flush=True)
      return None
