"""CPU checks: reward equations, archive fast/reference gradients, preflight."""
import copy
import math
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
# Native DiT contains CUDA autocast regions even with CPU inputs. Keep this
# standalone CPU suite from lazily initializing CUDA inside checkpoint().
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
try:
    import ipdb
except ImportError:
    import pdb
    sys.modules['ipdb']=pdb
import algo
import utils
import paper_objective as objective
from launch import training_overrides
from paper_eval import pass_at_k, choose_checkpoint, load_student

REPO=Path(algo.__file__).parent
objective.install()


def model_for(method,implementation='fast'):
    utils.register_omegaconf_resolvers()
    assets=dict(mdlm='/tmp/mdlm.ckpt',ar='/tmp/ar.ckpt',bundle='/tmp/bundle',data='/tmp/gsm8k.json',cache='/tmp/cache')
    overrides=training_overrides(method,2,assets,'/tmp/test-run')
    overrides += ['model.hidden_size=16','model.cond_dim=8','model.n_blocks=7',
                  'model.n_heads=2','model.length=6','model.dropout=0',
                  'loader.batch_size=2','loader.global_batch_size=4',
                  'adversarial_distill.accum_batches=1',
                  'adversarial_distill.mmd_forward_batch_size=2',
                  'adversarial_distill.mmd_rbf_alpha=0.2',
                  f'adversarial_distill.mmd_implementation={implementation}']
    with initialize_config_dir(config_dir=str(REPO/'configs'),version_base=None):
        cfg=compose(config_name='config',overrides=overrides)
    assert cfg.data.separator == r'\n'
    OmegaConf.resolve(cfg)
    tokenizer=SimpleNamespace(vocab_size=13,mask_token='[MASK]',mask_token_id=12,eos_token_id=1,pad_token_id=0)
    torch.manual_seed(7)
    with mock.patch('metrics.Metrics',return_value=SimpleNamespace()):
        m=algo.MDLM(cfg,tokenizer)
    torch.nn.init.normal_(m.backbone.output_layer.linear.weight,std=.1)
    return m


def deterministic_draws(probs,num_samples,**kwargs):
    return probs.argsort(-1,descending=True)[:,torch.arange(num_samples)%probs.shape[-1]]


class Tests(unittest.TestCase):
    def test_passk_exact_combinations(self):
        for n in (1,4,32):
            for c in range(n+1):
                for k in range(1,n+1):
                    expected=1-(math.comb(n-c,k)/math.comb(n,k) if n-c>=k else 0)
                    self.assertAlmostEqual(pass_at_k(n,c,k),expected,places=12)
        with self.assertRaises(ValueError):pass_at_k(4,1,8)

    def test_reward_equations(self):
        real=torch.tensor([[[0.],[2.],[100.]]])
        fake=torch.tensor([[[[1.],[3.],[1000.]],[[2.],[4.],[-500.]]]])
        mask=torch.tensor([[True,True,False]])
        for method,expected in [('regression',-2.5),('linear',-2.25)]:
            reward,active=objective.ablation_reward(real,fake,mask,method,.2)
            self.assertAlmostEqual(reward.item(),expected,places=6)
            self.assertTrue(active.item())
        cross=sum(math.exp(-.2*(x-y)**2) for x in (0,2) for y in (1,3,2,4))
        reward,_=objective.ablation_reward(real,fake,mask,'attraction',.2)
        self.assertAlmostEqual(reward.item(),2*cross/8,places=6)
        reward,_=objective.ablation_reward(real,fake,mask,'sequence_rbf',.2)
        self.assertAlmostEqual(reward.item(),math.exp(-.2)+math.exp(-.8)-math.exp(-.2),places=6)
        for method in objective.METHODS:
            result=objective.ablation_reward(real,fake,torch.zeros_like(mask),method,.2)
            self.assertEqual(result[0].item(),0)
            self.assertFalse(result[1].item())

    def test_all_variants_fast_reference_loss_and_gradients(self):
        torch.set_num_threads(1)
        x0=torch.tensor([[2,3,4,5,6,1],[3,2,6,5,4,1]])
        selected=torch.tensor([[False,True,True,True,False,False],[False,False,True,True,True,False]])
        xt=torch.where(selected,12,x0)
        sigma=torch.zeros(2,1)
        for method in sorted(objective.METHODS):
            with self.subTest(method=method):
                fast=model_for(method,'fast')
                reference=model_for(method,'reference')
                def ar_model(*args):
                    class CausalBlock(torch.nn.Module):
                        def __init__(self):
                            super().__init__();self.linear=torch.nn.Linear(16,16)
                        def forward(self,x):return self.linear(x.cumsum(1)).tanh()
                    class CausalToy(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.vocab_embed=torch.nn.Embedding(13,16)
                            self.blocks=torch.nn.ModuleList([CausalBlock() for _ in range(7)])
                        def forward(self,tokens,sigma):
                            x=self.vocab_embed(tokens)
                            for block in self.blocks:x=block(x)
                            return x
                    return CausalToy()
                # Native causal DiT uses CUDA-only rotary kernels. Test the AR
                # adapter with a causal CPU backbone, without a weight download.
                torch.manual_seed(91)
                with mock.patch('mmd_reward.load_ar_backbone',side_effect=ar_model):
                    a=objective.PaperObjective(fast)
                torch.manual_seed(91)
                with mock.patch('mmd_reward.load_ar_backbone',side_effect=ar_model):
                    b=objective.PaperObjective(reference)
                with mock.patch.object(torch,'multinomial',side_effect=deterministic_draws):
                    la,_=a.loss(fast,xt,sigma,x0,selected)
                    lb,_=b.loss(reference,xt,sigma,x0,selected)
                la.backward();lb.backward()
                torch.testing.assert_close(la,lb,rtol=2e-4,atol=2e-6)
                for p,q in zip(fast.backbone.parameters(),reference.backbone.parameters()):
                    torch.testing.assert_close(p.grad,q.grad,rtol=1e-3,atol=1e-5)
                self.assertTrue(torch.isfinite(la))

    def test_checkpoint_step_not_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir()
            p=root/'checkpoints/step=00008000.ckpt'
            torch.save({'global_step':1000,'ema':{}},p)
            self.assertEqual(choose_checkpoint(root,1000)[0],p)
            with self.assertRaisesRegex(ValueError,'No saved global_step=8000'):
                choose_checkpoint(root,8000)

    def test_load_student_raw_and_ema_without_training_modules(self):
        m=model_for('baseline')
        cfg=copy.deepcopy(m.config)
        cfg.adversarial_distill.is_distill=False
        cfg.adversarial_distill.mmd_implementation='legacy'
        cfg.training.ema=0
        shadows=[p.detach().clone()+.125 for p in m._get_parameters()]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'student.ckpt'
            torch.save({'state_dict':m.state_dict(),'ema':{'shadow_params':shadows},'global_step':8000},path)
            with mock.patch('metrics.Metrics',return_value=SimpleNamespace()), mock.patch('dataloader.get_tokenizer',return_value=m.tokenizer):
                raw=load_student(cfg,path,'raw')
                ema=load_student(cfg,path,'ema')
            for p,q,s in zip(m._get_parameters(),raw._get_parameters(),shadows):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            for p,s in zip(ema._get_parameters(),shadows):
                torch.testing.assert_close(p,s,rtol=0,atol=0)
            self.assertFalse(hasattr(ema,'teacher'))


if __name__=='__main__':unittest.main(verbosity=2)
