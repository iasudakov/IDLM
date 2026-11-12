"""CPU tests for checkpoint identity, exact selection and curve extension."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import idlm_state_eval as ev

class StateEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'checkpoints').mkdir()
        torch.save({'run_uid': ev.SOURCE_RUN}, self.root / 'underdeep_metadata.pt')
        for name, step in [('best.ckpt', 500), ('step=00040000.ckpt', 5000)]:
            torch.save({'global_step':step, 'ema':{'shadow_params':[torch.zeros(2)]},
                        'state_dict':{'weight':torch.ones(2)}}, self.root / 'checkpoints' / name)

    def test_saved_step_does_not_come_from_filename(self):
        records=ev.inventory_checkpoints(self.root)
        chosen=ev.select_checkpoint(records,5000)
        self.assertTrue(chosen['path'].endswith('step=00040000.ckpt'))
        self.assertEqual(chosen['global_step'],5000)
        with self.assertRaisesRegex(ValueError,'Available saved steps:'):
            ev.select_checkpoint(records,5500)
        with self.assertRaises(ValueError):ev.select_checkpoint(records,40000)

    def test_source_identity_required(self):
        self.assertEqual(ev.find_source_directory(self.root,ev.SOURCE_RUN),self.root)
        with self.assertRaises(ValueError):ev.find_source_directory(self.root,'different-run')

    def test_duplicate_step_and_missing_ema_rejected(self):
        records=ev.inventory_checkpoints(self.root)
        chosen=ev.select_checkpoint(records,5000)
        with self.assertRaises(ValueError):ev.select_checkpoint(records+[chosen],5000)
        chosen['has_ema']=False
        with self.assertRaisesRegex(ValueError,'no EMA'):ev.select_checkpoint([chosen],5000)

    def test_no_cross_step_or_cross_run_reference(self):
        doc={'source_run_uid':ev.SOURCE_RUN,'normalization':'fp32',
             'curves':{'5000':[{'threshold':.98,'nfe':40.,'accuracy':.4}]}}
        self.assertEqual(ev.reference_points(doc,ev.SOURCE_RUN,5500),[])
        with self.assertRaises(ValueError):ev.reference_points(doc,'another-run',5000)
        doc['normalization']='bf16'
        with self.assertRaises(ValueError):ev.reference_points(doc,ev.SOURCE_RUN,5000)

    def test_measured_overlap_replaces_only_same_threshold(self):
        old=[{'threshold':.9,'nfe':20.,'accuracy':.3},
             {'threshold':.98,'nfe':40.,'accuracy':.4}]
        new=[{'threshold':.98,'nfe':41.,'accuracy':.41},
             {'threshold':.99,'nfe':60.,'accuracy':.45},
             {'threshold':.995,'nfe':80.,'accuracy':.46}]
        merged=ev.merge_points(old,new)
        self.assertEqual([r['threshold'] for r in merged],[.9,.98,.99,.995])
        self.assertEqual(merged[1]['accuracy'],.41)
        self.assertEqual(merged[0]['origin'],'source_training_eval')
        self.assertEqual(merged[-1]['origin'],'state_reevaluation')
        self.assertNotIn('origin',old[0])

if __name__=='__main__':unittest.main()
