"""Run one paper job from Nirvana or a local checkout. --dry-run prints argv."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

HERE=Path(__file__).resolve().parent


def existing(paths,directory=False):
    for p in paths:
        if p.is_dir() if directory else p.is_file():return p.resolve()
    raise FileNotFoundError('Missing asset: '+', '.join(map(str,paths)))


def training_overrides(method,devices,assets,out,stop=None):
    if devices not in (1,2,4,8):raise ValueError('Use 1,2,4 or 8 GPUs')
    common=[
        'data=tiny-gsm',r'data.separator="\n"',
        'data.wrap=False','data.train_on_prompt=False','data.train_on_pad=True','data.filter_too_long=True',
        'model=small','model.length=512','algo=mdlm','strategy.find_unused_parameters=True',
        'logger.experiment=generative-models/tinygsm',
        f'logger.name=paper G4 {method} step{stop or 8000} FP32norm',
        'detect_anomaly=False',f'+paper_ablation={method}',
        'adversarial_distill.is_distill=True','adversarial_distill.dmd_weight=0.0',
        'adversarial_distill.mmd_reward_weight=1.0','adversarial_distill.mmd_implementation=fast',
        'adversarial_distill.mmd_reward_type='+('feature_mse' if method=='regression' else 'feature'),
        'adversarial_distill.mmd_feature_source='+('ar' if method=='ar' else 'diffusion'),
        'adversarial_distill.mmd_noise_level=0.0',
        'adversarial_distill.mmd_kernel='+('linear_l2' if method=='linear' else 'rbf'),
        'adversarial_distill.mmd_kernel_level='+('sentence' if method in ('linear','sequence_rbf') else 'token'),
        'adversarial_distill.mmd_kernel_source=draws','adversarial_distill.mmd_kernel_batch_size=2',
        'adversarial_distill.mmd_rbf_exclude_same_sentence=True',
        'adversarial_distill.mmd_rbf_exclude_same_position=True',
        'adversarial_distill.mmd_rbf_omit_real_real=True',
        'adversarial_distill.mmd_rbf_omit_fake_fake='+str(method=='attraction'),
        'adversarial_distill.mmd_group_size=4','adversarial_distill.mmd_average_candidates=False',
        'adversarial_distill.mmd_feature_layer='+('-1' if method=='embeddings' else '6'),
        'adversarial_distill.mmd_rbf_alpha='+('0.006' if method=='embeddings' else '0.00029' if method=='sequence_rbf' else '0.00006'),
        'adversarial_distill.mmd_forward_batch_size=8','adversarial_distill.mmd_fast_feature_batch_size=64',
        f'adversarial_distill.accum_batches={512//(devices*32)}',
        f'training.finetune_path={assets["mdlm"]}','training.loss_precision=bf16',
        f'model_weights_dir={assets["bundle"]}','optim.lr=1e-5',
        'loader.batch_size=32','loader.eval_batch_size=64','loader.global_batch_size=512','loader.num_workers=4',
        'eval.generate_samples=True','eval.compute_generative_perplexity=False',
        'eval.validate_before_train=False','eval.gsm8k_at_validation=True','eval.gsm8k_every_n_validations=1',
        'eval.tinygsm_exec=True','eval.gsm8k_pareto_fp32=True','eval.gsm8k_pareto_temperature=0.0',
        'eval.start_step=5500',f'eval.gsm8k_data_path={assets["data"]}',
        'eval.gsm8k_pareto_thresholds=[0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.98,0.99,0.995]',
        'callbacks.checkpoint_monitor.monitor=val/nll','callbacks.checkpoint_monitor.save_top_k=0',
        'callbacks.checkpoint_every_n_steps.save_top_k=0','callbacks.checkpoint_every_n_steps.save_last=False',
        'checkpointing.use_periodic_train_batch_checkpoint=True',
        f'callbacks.checkpoint_every_n_steps.every_n_train_steps={500*512//(devices*32)}',
        'checkpointing.save_evaluated_checkpoints=False',
        'checkpointing.resume_from_ckpt=False',
        'trainer.precision=bf16','trainer.accumulate_grad_batches=1','trainer.gradient_clip_val=null',
        f'trainer.devices={devices}',f'trainer.max_steps={stop or 8000}','trainer.num_sanity_val_steps=0',
        f'trainer.val_check_interval={500*512//(devices*32)}','trainer.limit_val_batches=500',
        'sampling.steps=64','sampling.predictor=ancestral_cache',
        'sampling.num_sample_batches=2','sampling.num_sample_log=8',
        f'data.cache_dir={assets["cache"]}',f'checkpointing.save_dir={out}',f'hydra.run.dir={out}',
    ]
    if method=='ar':common.append(f'adversarial_distill.mmd_ar_model={assets["ar"]}')
    return common


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',choices=['train-baseline','train-attraction','train-regression',
        'train-linear','train-sequence_rbf','train-embeddings','train-ar',
        'passk-mdlm','passk-mmd','passk-idlm','passk-idlm-reinforce','ancestral-mmd','confidence-mmd'])
    parser.add_argument('--repo')
    parser.add_argument('--devices',type=int,default=2)
    parser.add_argument('--step',type=int)
    parser.add_argument('--source-run',help='Run UUID whose state is connected for evaluation')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    a=parser.parse_args()
    if a.job in ('passk-mmd','passk-idlm','passk-idlm-reinforce','ancestral-mmd','confidence-mmd') and not a.source_run:
        parser.error('--source-run is required for a student checkpoint evaluation')
    repo=Path(a.repo).resolve() if a.repo else HERE.parents[1]
    if not (repo/'mmd/objective_fast.py').is_file():
        raise FileNotFoundError('Use feat/tinygsm-mmd-archive-v2 (e6a0c22 or later) as SOURCE_CODE_PATH/code')
    os.chdir(repo)
    os.environ['PYTHONPATH']=os.pathsep.join([str(repo),str(repo.parent),str(HERE),os.environ.get('PYTHONPATH','')])
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1'
    inputs=Path(os.environ.get('INPUT_PATH',repo))
    bundle=existing([inputs/'tinygsm_mmd_offline_bundle',repo/'cache_dir/tinygsm_mmd_offline_bundle'],True)
    data=existing([inputs/'nirvana_checkpoints/data/gsm8k_test.json',repo/'data/gsm8k_test.json',bundle/'data/gsm8k_test.json'])
    assets={'bundle':bundle,'data':data}
    if a.job.startswith('train-') or a.job=='passk-mdlm':
        assets['mdlm']=existing([inputs/'nirvana_checkpoints/checkpoints/tinygsm/mdlm_notok.ckpt',repo/'checkpoints/tinygsm/mdlm_notok.ckpt'])
    if a.job.startswith('train-'):
        assets['cache']=existing([inputs/'nirvana_checkpoints/cache_dir',repo/'cache_dir'],True)
        # Explicit separator uses the existing _sep5c6e cache; no aliases/mv.
        cache=assets['cache']/'tiny_gsm_bs512_answer_only_filtered_train_on_pad_HuggingFaceTB__SmolLM-135M_sep5c6e'
        if not (cache/'train').is_dir() or not (cache/'validation').is_dir():
            raise FileNotFoundError('The separator-aware TinyGSM train/validation cache is missing: '+str(cache))
        if a.job=='train-ar':
            assets['ar']=existing([inputs/'ar_notok.ckpt',inputs/'nirvana_checkpoints/checkpoints/tinygsm/ar_notok.ckpt',repo/'checkpoints/tinygsm/ar_notok.ckpt'])
        out=repo/'outputs/paper_recompute'/(a.job+'-'+uuid.uuid4().hex[:10])
        cmd=[sys.executable,'-u',str(HERE/'paper_train.py'),
             *training_overrides(a.job[6:],a.devices,assets,out,a.step),*a.extra]
        if a.plan_only:cmd+=['--cfg','job','--resolve']
    else:
        suite=('confidence' if a.job=='confidence-mmd' else
               'ancestral' if a.job=='ancestral-mmd' else 'passk')
        cmd=[sys.executable,'-u',str(HERE/'paper_eval.py'),'--suite',suite,
             '--devices',str(a.devices),'--model-weights-dir',str(bundle),'--data-path',str(data)]
        if suite=='ancestral':
            cmd+=['--budgets','4','8','16','32','48','64','--seeds','1','2','3']
        elif suite=='passk':
            cmd+=['--samples','32','--budgets','4','8','16','32','--ks','1','2','4','8','16','32']
        if a.job=='passk-mdlm':
            cmd+=['--checkpoint',str(assets['mdlm']),'--base-teacher','--step','0','--weights','ema']
        else:
            # Source IDs are supplied at launch, never embedded in public code.
            # Require the requested saved step; do not substitute nearby weights.
            presets={'passk-mmd':(8000,'ema'),
                     'confidence-mmd':(8000,'ema'),
                     'ancestral-mmd':(8000,'ema'),
                     'passk-idlm':(4500,'raw'),
                     'passk-idlm-reinforce':(5500,'raw')}
            step,weights=presets[a.job]
            cmd+=['--source-run',a.source_run,'--step',str(a.step or step),'--weights',weights]
        cmd+=['--name',a.job+' FP32norm'+(' ancestral T0' if suite=='passk' else '')]
        if a.plan_only:cmd+=['--plan-only']
        cmd+=a.extra
    print(shlex.join(cmd),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True)


if __name__=='__main__':main()
