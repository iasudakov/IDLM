"""CPU/offline publication tests: run isolation, resume, exact values, atomic IO."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from didi_repro import tinygsm_pareto as pareto


class Logger:
  def __init__(self, version='run-a'):
    self.version = version
    self.metrics = []
    self.tables = []
    self.images = []

  def log_metrics(self, metrics, step=None):
    self.metrics.append((copy.deepcopy(metrics), step))

  def log_table(self, key, columns, data, step=None):
    self.tables.append((key, columns, copy.deepcopy(data), step))

  def log_image(self, key, image, step=None):
    self.images.append((key, image, step))


def curve():
  return [
    {'threshold': .35, 'nfe': 4.125, 'accuracy': .5, 'correct': 5, 'count': 10,
     'total_batch_forwards': 31, 'generation_batches': 2, 'answer_entropy': 2.3},
    {'threshold': .325, 'nfe': 3.2, 'accuracy': .4, 'correct': 4, 'count': 10,
     'total_batch_forwards': 23, 'generation_batches': 2, 'answer_entropy': 2.4},
  ]


class ParetoPublisherTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.model = SimpleNamespace(config=SimpleNamespace(
      checkpointing=SimpleNamespace(save_dir=self.directory.name)))
    self.logger = Logger()
    self.trainer = SimpleNamespace(logger=self.logger, loggers=[self.logger],
                                    global_step=100, is_global_zero=True, global_rank=0)
    self.path = Path(self.directory.name) / pareto.HISTORY_FILENAME

  def publish(self, points=None, **kwargs):
    return pareto.publish_threshold_pareto(self.trainer, self.model,
                                           curve() if points is None else points, **kwargs)

  def stored(self):
    return json.loads(self.path.read_text())

  def scalar_points(self):
    return [(next(iter(values.values())), step) for values, step in self.logger.metrics
            if next(iter(values)).startswith('pareto/gsm8k_step_')]

  def test_reference_trends_and_exact_table(self):
    result = self.publish()
    trends, step = self.logger.metrics[0]
    self.assertEqual(step, 100)
    self.assertEqual(trends['val/gsm8k_acc_thr0325'], .4)
    self.assertEqual(trends['val/gsm8k_nfe_thr035'], 4.125)
    table = next(item for item in self.logger.tables if item[0] == 'pareto/gsm8k_thresholds')
    self.assertEqual(table[3], 100)
    self.assertEqual([row[1] for row in table[2]], [3.2, 4.125])
    exact = [json.loads(row[-1]) for row in table[2]]
    self.assertEqual(exact, result['curve'])
    self.assertEqual(exact[0]['total_batch_forwards'], 23)
    self.assertEqual(exact[0]['generation_batches'], 2)
    self.assertEqual(exact[0]['answer_entropy'], 2.4)

  def test_fp32_keeps_original_history_metrics_tables_and_images(self):
    with mock.patch.object(pareto, 'render', return_value='image'):
      self.publish(plot=True)
      original = self.path.read_bytes()
      old_metrics = len(self.logger.metrics)
      changed = curve()
      changed[0]['nfe'] = 5.0
      self.publish(changed, plot=True, suffix='_fp32norm')
    self.assertEqual(self.path.read_bytes(), original)
    fp32_path = self.path.with_name(self.path.stem + '_fp32norm.json')
    stored = json.loads(fp32_path.read_text())
    self.assertEqual(stored['runs']['run-a']['curves']['100'][1]['nfe'], 5.)
    self.assertTrue(all(key.endswith('_fp32norm')
                        for metrics, _ in self.logger.metrics[old_metrics:] for key in metrics))
    self.assertEqual([item[0] for item in self.logger.tables],
                     ['pareto/gsm8k_thresholds', 'didi/tinygsm_pareto_protocol',
                      'pareto/gsm8k_thresholds_fp32norm', 'didi/tinygsm_pareto_protocol_fp32norm'])
    self.assertEqual([item[0] for item in self.logger.images], ['pareto/frontier', 'pareto/frontier_fp32norm'])

  def test_scalar_axis_is_nfe_times_100_not_training_step(self):
    self.publish()
    self.assertEqual(self.scalar_points(), [(.4, 320), (.5, 412)])

  def test_scalar_axis_sorted_by_nfe_not_threshold(self):
    points = curve()
    points[0]['nfe'], points[1]['nfe'] = 1.0, 4.0
    self.publish(points)
    self.assertEqual(self.scalar_points(), [(.5, 100), (.4, 400)])

  def test_arbitrary_precision_threshold_tags_do_not_collide(self):
    points = curve()
    points[0]['threshold'], points[1]['threshold'] = .1234567891234567, .1234567891234568
    self.assertNotEqual(pareto.threshold_tag(points[0]['threshold']),
                        pareto.threshold_tag(points[1]['threshold']))
    self.publish(points)
    metrics = self.logger.metrics[0][0]
    self.assertEqual(sum(key.startswith('val/gsm8k_acc_thr') for key in metrics), 2)
    self.assertNotEqual(pareto.threshold_tag(1e-7), pareto.threshold_tag(1e-8))
    self.assertEqual(pareto.threshold_tag(1.0), '1')

  def test_colliding_scalar_buckets_never_overwrite_points(self):
    for nfe in (3.2, 3.201):
      with self.subTest(nfe=nfe):
        self.logger.metrics.clear()
        points = curve()
        points[0]['nfe'] = nfe
        result = self.publish(points)
        self.assertEqual(self.scalar_points(), [])
        self.assertEqual(result['protocol']['scalar_collision_buckets'], [320])
        self.assertEqual(len(result['curve']), 2)
        self.assertEqual(len(self.stored()['runs']['run-a']['curves']['100']), 2)
        self.assertEqual(self.logger.metrics[0][0]['didi_tinygsm_pareto/scalar_collision_points'], 2)

  def test_duplicate_thresholds_rejected_without_writing(self):
    points = curve()
    points[0]['threshold'] = points[1]['threshold']
    with self.assertRaisesRegex(ValueError, 'only once'):
      self.publish(points)
    self.assertFalse(self.path.exists())
    self.assertEqual(self.logger.metrics, [])

  def test_nonfinite_metrics_rejected_before_writing(self):
    for key, value in (('nfe', math.nan), ('accuracy', math.inf),
                       ('answer_entropy', math.nan), ('nfe', -1)):
      points = curve()
      points[0][key] = value
      with self.subTest(key=key, value=value), self.assertRaises(ValueError):
        self.publish(points)
    self.assertFalse(self.path.exists())

  def test_resumed_run_accumulates_checkpoint_history(self):
    self.publish()
    self.trainer.global_step = 200
    self.model = SimpleNamespace(config=self.model.config)
    result = self.publish()
    self.assertEqual(set(result['history']), {'100', '200'})
    self.assertEqual(set(self.stored()['runs']), {'run-a'})

  def test_resumed_pretraining_uses_checkpoint_context(self):
    self.trainer.global_step = 0
    self.model._didi_evaluation_step = {'value': 6000}
    result = self.publish()
    self.assertEqual(result['training_step'], 6000)
    self.assertEqual(set(result['history']), {'6000'})
    self.assertEqual(self.logger.metrics[0][1], 6000)
    self.assertEqual(self.scalar_points(), [(.4, 320), (.5, 412)])
    self.assertIn('pareto/gsm8k_step_6000', self.logger.metrics[1][0])

  def test_repeated_same_checkpoint_replaces_its_curve(self):
    self.publish()
    replacement = curve()
    replacement[0]['accuracy'] = .6
    result = self.publish(replacement)
    self.assertEqual(set(result['history']), {'100'})
    self.assertEqual(result['history']['100'][1]['accuracy'], .6)

  def test_new_run_keeps_old_data_but_never_mixes_plot_histories(self):
    self.publish()
    self.logger.version = 'run-b'
    self.trainer.global_step = 0
    with mock.patch.object(pareto, 'render', return_value='image') as render:
      result = self.publish(plot=True)
    self.assertEqual(result['run_uid'], 'run-b')
    self.assertEqual(set(result['history']), {'0'})
    self.assertEqual(set(self.stored()['runs']), {'run-a', 'run-b'})
    self.assertEqual(set(render.call_args.args[0]), {'0'})
    self.assertEqual(self.logger.images, [('pareto/frontier', 'image', 0)])

  def test_untracked_process_does_not_resume_another_unknown_run(self):
    self.logger.version = ''
    first = self.publish()['run_uid']
    self.trainer.global_step = 200
    self.assertEqual(self.publish()['run_uid'], first)
    self.model = SimpleNamespace(config=self.model.config)
    second = self.publish()['run_uid']
    self.assertNotEqual(first, second)
    self.assertEqual(set(self.stored()['runs']), {first, second})

  def test_atomic_write_failure_preserves_existing_history(self):
    self.publish()
    previous = self.path.read_bytes()
    self.trainer.global_step = 200
    with mock.patch.object(pareto.os, 'replace', side_effect=OSError('disk failure')):
      with self.assertRaisesRegex(OSError, 'disk failure'):
        self.publish()
    self.assertEqual(self.path.read_bytes(), previous)
    self.assertEqual(list(Path(self.directory.name).iterdir()), [self.path])

  def test_corrupt_existing_history_is_not_overwritten(self):
    self.path.write_text('not JSON')
    with self.assertRaisesRegex(RuntimeError, 'existing file preserved'):
      self.publish()
    self.assertEqual(self.path.read_text(), 'not JSON')
    self.assertEqual(self.logger.metrics, [])

  def test_history_saved_before_tracker_failure(self):
    with mock.patch.object(self.logger, 'log_metrics', side_effect=RuntimeError('tracker failed')):
      with self.assertRaisesRegex(RuntimeError, 'tracker failed'):
        self.publish()
    self.assertIn('100', self.stored()['runs']['run-a']['curves'])

  def test_disabled_plot_does_not_import_or_render_matplotlib(self):
    with mock.patch.object(pareto, 'render', side_effect=AssertionError('rendered')):
      self.publish(plot=False)
    self.assertEqual(self.logger.images, [])

  def test_missing_optional_matplotlib_keeps_tables_and_history(self):
    with mock.patch.object(pareto, 'render', side_effect=ImportError('no matplotlib')):
      self.publish(plot=True)
    self.assertTrue(self.path.is_file())
    self.assertEqual(len(self.logger.tables), 2)
    self.assertEqual(self.logger.images, [])

  def test_optional_reference_passed_to_renderer(self):
    reference = {'points': [{'threshold': .5, 'nfe': 6.3, 'accuracy': .75}]}
    with mock.patch.object(pareto, 'render', return_value='pixels') as render:
      self.publish(reference=reference, plot=True)
    self.assertEqual(render.call_args.kwargs['reference'], reference['points'])
    self.assertEqual(self.logger.images[0], ('pareto/frontier', 'pixels', 100))
    self.assertIsNone(pareto.load_reference('/nonexistent/teacher-reference.json'))

  def test_duplicate_logger_objects_are_only_published_once(self):
    self.trainer.loggers = [self.logger, self.logger]
    self.publish()
    self.assertEqual(len(self.logger.metrics), 3)
    self.assertEqual(len(self.logger.tables), 2)

  def test_no_logger_still_persists_numeric_history(self):
    self.trainer.logger, self.trainer.loggers = None, []
    result = self.publish(plot=True)
    self.assertTrue(result['run_uid'].startswith('untracked-'))
    self.assertTrue(self.path.is_file())

  def test_nonzero_rank_rejected(self):
    self.trainer.is_global_zero = False
    with self.assertRaisesRegex(RuntimeError, 'rank zero'):
      self.publish()
    self.assertFalse(self.path.exists())

  def test_input_curve_not_mutated(self):
    points = curve()
    before = copy.deepcopy(points)
    self.publish(points)
    self.assertEqual(points, before)


if __name__ == '__main__':
  unittest.main()
