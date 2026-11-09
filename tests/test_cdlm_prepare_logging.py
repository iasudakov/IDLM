"""Preparation reports its lifecycle without running a GPU teacher."""

from contextlib import nullcontext
import os
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

import main


class PreparationLoggingTest(unittest.TestCase):
    def run_preparation(self, failure=None, student_source='/not-mounted-on-collector/mmd.ckpt',
                        teacher_source='original-mdlm.ckpt', teacher_use_ema=False):
        events = []
        config = OmegaConf.create({
            'algo': {'name': 'cdlm'}, 'data': {'train': 'tiny_gsm'},
            'training': {'finetune_path': teacher_source,
                         'teacher_use_ema': teacher_use_ema,
                         'student_init_path': student_source,
                         'student_init_use_ema': True},
        })
        progress = Mock()
        progress.start.side_effect = lambda: events.append('progress started')
        progress.close.side_effect = lambda status: events.append(status)
        loader = Mock()
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model

        def load_data(*args, **kwargs):
            events.append('data loading')
            return loader, None

        def collect(*args, **kwargs):
            events.append('collecting')
            if failure:
                raise failure

        with patch.dict(os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0'), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.autocast', return_value=nullcontext()), \
             patch('main.dataloader.get_dataloaders', side_effect=load_data), \
             patch('cdlm_trajectories.prepare_trajectories', side_effect=collect), \
             patch('cdlm_progress.PreparationProgress', return_value=progress) as factory:
            if failure:
                with self.assertRaisesRegex(RuntimeError, str(failure)):
                    main._prepare_cdlm(Mock(return_value=model), config, Mock(), object())
            else:
                main._prepare_cdlm(Mock(return_value=model), config, Mock(), object())
            factory.assert_called_once_with(config, rank=0)
            model.load_teacher_weights.assert_called_once_with(teacher_source, use_ema=teacher_use_ema)
            model.load_evaluation_weights.assert_not_called()
            model.initialize_from_checkpoint.assert_not_called()
        return events, progress

    def test_mmd_teacher_is_selected_for_collection(self):
        events, progress = self.run_preparation(
            teacher_source='hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt', teacher_use_ema=True)
        self.assertEqual(events, ['progress started', 'data loading', 'collecting', 'complete'])
        progress.close.assert_called_once_with('complete')

    def test_reporting_starts_before_loading_and_closes_after_collection(self):
        events, progress = self.run_preparation()
        self.assertEqual(events, ['progress started', 'data loading', 'collecting', 'complete'])
        progress.close.assert_called_once_with('complete')

    def test_hub_student_initialization_never_downloads_during_teacher_preparation(self):
        with patch('sdtt_checkpoint.hf_hub_download') as download:
            events, progress = self.run_preparation(
                student_source='hf://yresearch/MDLM-MMD-TinyGSM/model.ckpt')
        download.assert_not_called()
        self.assertEqual(events, ['progress started', 'data loading', 'collecting', 'complete'])
        progress.close.assert_called_once_with('complete')

    def test_teacher_failure_is_preserved_and_reported(self):
        events, progress = self.run_preparation(RuntimeError('teacher collection failed'))
        self.assertEqual(events[-1], 'failed')
        progress.close.assert_called_once_with('failed')


if __name__ == '__main__':
    unittest.main()
