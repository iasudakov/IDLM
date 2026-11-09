"""CPU-only checks of explicit training/budget axes and sample-SD publishing."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
from matplotlib.figure import Figure

import cdlm_ancestral_logging as logging


def point(steps=8, *, deviation=.1):
    return {'steps': steps, 'accuracy_mean': .5, 'accuracy_std': deviation,
            'nfe_mean': steps + 1., 'nfe_std': .25 if deviation is not None else None,
            'num_seeds': 5 if deviation is not None else 1,
            'seeds': [{'seed': 17, 'correct': 4, 'count': 10, 'accuracy': .4,
                       'nfe': steps + .75, 'total_batch_forwards': 20},
                      *([{'seed': 23, 'correct': 6, 'count': 10, 'accuracy': .6,
                          'nfe': steps + 1.25, 'total_batch_forwards': 24},
                         {'seed': 29, 'correct': 5, 'count': 10, 'accuracy': .5,
                          'nfe': steps + 1., 'total_batch_forwards': 22},
                         {'seed': 31, 'correct': 4, 'count': 10, 'accuracy': .4,
                          'nfe': steps + .75, 'total_batch_forwards': 20},
                         {'seed': 37, 'correct': 6, 'count': 10, 'accuracy': .6,
                          'nfe': steps + 1.25, 'total_batch_forwards': 24}]
                        if deviation is not None else [])]}


class Logger:
    def __init__(self):
        self.metrics, self.tables, self.images = [], [], []

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))

    def log_table(self, key, columns, data, step=None):
        self.tables.append((key, columns, data, step))

    def log_image(self, key, image, step=None):
        self.images.append((key, image, step))


def model(*loggers):
    return SimpleNamespace(global_step=500,
        trainer=SimpleNamespace(global_rank=0, loggers=list(loggers)),
        log=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('Use direct logger calls, without Lightning step remapping.')))


class AncestralLoggingTest(unittest.TestCase):
    def test_flat_metrics_include_sample_std_and_each_seed_without_greedy_aliases(self):
        metrics = logging.point_metrics(point())
        self.assertEqual(metrics['val/gsm8k_acc_ancestral_steps8'], .5)
        self.assertEqual(metrics['val/gsm8k_acc_ancestral_std_steps8'], .1)
        self.assertEqual(metrics['val/gsm8k_nfe_ancestral_steps8'], 9.)
        self.assertEqual(metrics['val/gsm8k_nfe_ancestral_std_steps8'], .25)
        self.assertEqual(metrics['val/gsm8k_acc_ancestral_steps8_seed17'], .4)
        self.assertEqual(metrics['val/gsm8k_acc_ancestral_steps8_seed23'], .6)
        self.assertEqual(metrics['val/gsm8k_nfe_ancestral_steps8_seed17'], 8.75)
        self.assertTrue(all('greedy' not in key for key in metrics))

    def test_point_is_immediately_logged_at_exact_training_step_and_configured_budget(self):
        destination = Logger()
        protocol = {'steps': [4, 8, 16, 32, 48, 64], 'final_removal': 'extra forward'}
        before = copy.deepcopy(protocol)
        logging.log_point(model(destination), point(), protocol)
        scalars, native = destination.metrics
        self.assertEqual(scalars[1], 500)
        self.assertEqual(native, ({'ancestral/gsm8k_step_500': .5,
                                  'ancestral/gsm8k_std_step_500': .1}, 8))
        self.assertEqual(protocol, before)

    def test_undefined_one_seed_std_is_omitted_while_zero_std_is_logged(self):
        single = point(deviation=None)
        metrics = logging.point_metrics(single)
        self.assertFalse(any('_std_' in key for key in metrics))
        destination = Logger()
        logging.log_point(model(destination), single, {})
        self.assertEqual(destination.metrics[1], ({'ancestral/gsm8k_step_500': .5}, 8))
        self.assertEqual(logging.point_metrics(point(deviation=0.))[
            'val/gsm8k_acc_ancestral_std_steps8'], 0.)

    def test_nonzero_rank_never_accesses_loggers_step_or_rendering(self):
        class Trainer:
            global_rank = 1

            @property
            def loggers(self):
                raise AssertionError('Nonzero ranks must not touch loggers.')

        class OtherRank:
            trainer = Trainer()

            @property
            def global_step(self):
                raise AssertionError('Nonzero rank must return before publication work.')

        result = {'points': [point()], 'protocol': {}}
        before = copy.deepcopy(result)
        with patch.object(logging, 'render_ancestral', side_effect=AssertionError('No render.')):
            logging.log_point(OtherRank(), result['points'][0], result['protocol'])
            logging.publish_result(OtherRank(), result)
        self.assertEqual(result, before)

    def test_missing_loggers_skip_rendering_and_ignore_single_logger_alias(self):
        destination = Logger()
        unused = SimpleNamespace(global_step=500,
                                 trainer=SimpleNamespace(global_rank=0, logger=destination))
        with patch.object(logging, 'render_ancestral', side_effect=AssertionError('No render.')):
            logging.log_point(unused, point(), {})
            logging.publish_result(unused, {'points': [point()], 'protocol': {}})
        self.assertFalse(destination.metrics or destination.tables or destination.images)

    def test_result_table_and_image_are_discoverable_without_duplicate_scalars(self):
        first, second = Logger(), Logger()
        result = {'points': [point(16), point(4)], 'protocol': {'final_removal_extra_forward': True}}
        before = copy.deepcopy(result)
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        with patch.object(logging, 'render_ancestral', return_value=image) as render:
            logging.publish_result(model(first, second), result)
        render.assert_called_once_with([result['points'][1], result['points'][0]], step=500)
        for destination in (first, second):
            self.assertFalse(destination.metrics)
            key, columns, rows, step = destination.tables[0]
            self.assertEqual((key, step), ('ancestral/results', 500))
            self.assertEqual(columns, ['steps', 'seed', 'correct', 'count', 'accuracy', 'NFE'])
            self.assertEqual(rows[0], [4, 17, 4, 10, .4, 4.75])
            self.assertEqual([row[0] for row in rows], [4] * 5 + [16] * 5)
            self.assertEqual(destination.images[0][0], 'ancestral/accuracy_vs_steps')
            self.assertIs(destination.images[0][1], image)
            self.assertEqual(destination.images[0][2], 500)
        self.assertEqual(result, before)

    def test_optional_presentation_errors_do_not_abort_validation(self):
        result = {'points': [point()], 'protocol': {}}
        first, second = Logger(), Logger()
        with patch.object(logging, 'render_ancestral', side_effect=RuntimeError('plot unavailable')) as render:
            with self.assertWarnsRegex(UserWarning, 'Could not render ancestral GSM8K accuracy plot'):
                logging.publish_result(model(first, second), result)
        render.assert_called_once()
        self.assertEqual(len(first.tables), 1)
        self.assertEqual(len(second.tables), 1)
        self.assertFalse(first.images or second.images)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for method in ('log_table', 'log_image'):
            with self.subTest(failing_logger_method=method):
                failing, healthy = Logger(), Logger()
                with patch.object(failing, method, side_effect=RuntimeError('tracker unavailable')), \
                     patch.object(logging, 'render_ancestral', return_value=image):
                    with self.assertWarnsRegex(UserWarning, 'Could not publish ancestral GSM8K'):
                        logging.publish_result(model(failing, healthy), result)
                self.assertEqual(len(healthy.tables), 1)
                self.assertEqual(len(healthy.images), 1)

    def render(self, points):
        figures = []

        def capture(*args, **kwargs):
            figure = Figure(*args, **kwargs)
            figures.append(figure)
            return figure

        with patch('matplotlib.figure.Figure', side_effect=capture):
            image = logging.render_ancestral(points, step=500)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.dtype, np.uint8)
        return figures[0]

    def test_render_uses_configured_log2_ticks_and_percentage_sample_sd_errorbars(self):
        budgets = [4, 8, 16, 32, 48, 64]
        points = [point(budget, deviation=.02) for budget in reversed(budgets)]
        before = copy.deepcopy(points)
        figure = self.render(points)
        axes = figure.axes[0]
        self.assertEqual(axes.get_xscale(), 'log')
        self.assertEqual(axes.xaxis.get_transform().base, 2)
        np.testing.assert_array_equal(axes.get_xticks(), budgets)
        np.testing.assert_array_equal(axes.lines[0].get_xdata(), budgets)
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), [50.] * len(budgets))
        np.testing.assert_allclose(axes.get_ylim(), [0., 57.2])  # SD and margin, without forcing 100%.
        segments = axes.containers[0].lines[2][0].get_segments()
        for budget, segment in zip(budgets, segments):
            np.testing.assert_allclose(segment, [[budget, 48.], [budget, 52.]])
        self.assertIn('Configured', axes.get_xlabel())
        self.assertIn('(%)', axes.get_ylabel())
        self.assertIn('500', axes.get_title())
        self.assertIn('5 seeds', figure.texts[0].get_text())
        self.assertIn('ddof=1', ' '.join(text.get_text() for text in axes.get_legend().get_texts()))
        table = figure.axes[1].tables[0]
        texts = [cell.get_text().get_text() for cell in table.get_celld().values()]
        self.assertIn('65.00', texts)  # Actual NFE differs from configured budget 64.
        self.assertIn('Mean actual NFE', texts)
        self.assertEqual(points, before)

    def test_one_seed_render_has_no_defined_errorbar_and_marks_nfe_std_unavailable(self):
        figure = self.render([point(deviation=None)])
        self.assertFalse(figure.axes[0].containers)
        table = figure.axes[1].tables[0]
        self.assertIn('n/a', [cell.get_text().get_text()
                              for cell in table.get_celld().values()])
        self.assertIn('single seed', figure.texts[0].get_text())


if __name__ == '__main__':
    unittest.main()
