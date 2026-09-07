"""Optional original validation before the first training optimizer update.

This is opt-in: it changes evaluation cost, never training hyperparameters or
the upstream sampler. On resume, validation loads the requested checkpoint and
the following ``fit(..., ckpt_path=...)`` remains responsible for restoring the
optimizer and training-loop state.
"""

from contextlib import contextmanager, ExitStack
import functools

import torch

from .monitor import preserve_rng_state


@contextmanager
def _temporary_attribute(obj, name, value):
    existed = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def maybe_evaluate_before_training(trainer, model, valid_loader, ckpt_path=None):
    """Run full validation before fit when ``didi_eval.before_training=true``.

    Returns Lightning's validation results, or None when disabled. The same
    callback collection, validation limit, and sampler are used as during fit.
    Underdeep is kept open for the subsequent fit. All module modes and random
    states are restored even if validation fails. No checkpoint is written.

    Lightning's standalone validation does not restore the fit-loop global
    step. A temporary logging adapter therefore labels resumed evaluation with
    the checkpoint's actual global_step, read by its existing load hook, rather
    than silently publishing resumed weights at step zero.
    """
    if not model.config.get('didi_eval', {}).get('before_training', False):
        return None
    if trainer.limit_val_batches == 0:
        raise ValueError('didi_eval.before_training requires trainer.limit_val_batches > 0.')
    if valid_loader is None:
        raise ValueError('didi_eval.before_training requires a validation dataloader.')

    evaluation_step = {'value': 0, 'checkpoint_loaded': ckpt_path is None}
    modes = [(module, module.training) for module in model.modules()]
    loggers = list(getattr(trainer, 'loggers', []) or [])
    if not loggers and getattr(trainer, 'logger', None):
        loggers = [trainer.logger]
    # Distinct logger objects only; Lightning may expose the same one twice.
    loggers = list({id(logger): logger for logger in loggers}.values())

    original_load = model.on_load_checkpoint

    @functools.wraps(original_load)
    def load_checkpoint(checkpoint):
        original_load(checkpoint)
        if 'global_step' not in checkpoint:
            raise ValueError('Pre-training resume evaluation needs checkpoint global_step metadata.')
        evaluation_step['value'] = int(checkpoint['global_step'])
        evaluation_step['checkpoint_loaded'] = True

    def no_checkpoint_writes(*args, **kwargs):
        raise RuntimeError('Pre-training validation must not write checkpoints.')

    def remap_metric_step(original):
        @functools.wraps(original)
        def logged(metrics, step=None, **kwargs):
            return original(metrics, step=evaluation_step['value'], **kwargs)
        return logged

    def remap_media_step(original):
        @functools.wraps(original)
        def logged(*args, **kwargs):
            kwargs['step'] = evaluation_step['value']
            return original(*args, **kwargs)
        return logged

    with preserve_rng_state(), ExitStack() as stack:
        try:
            root_device = trainer.strategy.root_device
            if root_device.type == 'cuda' and not torch.cuda.is_initialized():
                # Validation can be the first CUDA use for a CPU-initialized
                # native model. Capture its seeded GPU generator before any
                # sampling; the outer guard alone skipped uninitialized CUDA.
                # Select the rank-local device before initializing its context.
                with torch.cuda.device(root_device):
                    torch.cuda.get_rng_state(root_device)
                stack.enter_context(preserve_rng_state())
            stack.enter_context(_temporary_attribute(model, 'on_load_checkpoint', load_checkpoint))
            # Callbacks that write their own history must use the same saved
            # checkpoint step as the logger adapters, not validate()'s zero.
            stack.enter_context(_temporary_attribute(
                model, '_didi_evaluation_step', evaluation_step))
            stack.enter_context(_temporary_attribute(trainer, 'save_checkpoint', no_checkpoint_writes))
            for logger in loggers:
                stack.enter_context(_temporary_attribute(logger, 'defer_finalize', True))
                stack.enter_context(_temporary_attribute(
                    logger, 'log_metrics', remap_metric_step(logger.log_metrics)))
                for name in ('log_table', 'log_image'):
                    if hasattr(logger, name):
                        stack.enter_context(_temporary_attribute(
                            logger, name, remap_media_step(getattr(logger, name))))
            results = trainer.validate(model, dataloaders=valid_loader,
                                       ckpt_path=ckpt_path, verbose=False)
            if not evaluation_step['checkpoint_loaded']:
                raise RuntimeError('Requested pre-training checkpoint was not loaded; refusing mislabeled evaluation.')
            if trainer.is_global_zero:
                for logger in loggers:
                    if hasattr(logger, 'log_table'):
                        logger.log_table(
                            key='didi/pretrain_eval', columns=['Property', 'Value'],
                            data=[['phase', 'before training'],
                                  ['checkpoint', str(ckpt_path or 'fresh initialized model')],
                                  ['global_step', str(evaluation_step['value'])],
                                  ['didi_iteration', str(getattr(model, 'global_steps', 0))],
                                  ['weights', getattr(model, '_didi_evaluation_weights', 'backbone')],
                                  ['sampler', str(model.config.sampling.predictor)],
                                  ['sampling_steps', str(model.config.sampling.steps)]],
                            step=evaluation_step['value'])
            return results
        except BaseException:
            # Lightning may have attempted finalize('failed') while closing
            # was deliberately deferred. Restore that flag first, then close
            # the failed run; tracker errors must not hide the real failure.
            stack.close()
            for logger in loggers:
                try:
                    logger.finalize('failed')
                except Exception as error:
                    print(f'[didi-pretrain-eval] failed-run finalize failed: {error}', flush=True)
            raise
        finally:
            # Set each flag directly: recursively calling train() would erase
            # mixed modes such as an eval-only teacher and a training student.
            for module, training in modes:
                module.training = training
