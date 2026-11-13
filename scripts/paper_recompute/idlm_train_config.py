"""Fresh IDLM/IDLM-REINFORCE runs with paired raw/EMA validation."""


def training_overrides(method, devices, assets, out, stop=None):
    if method not in ('idlm', 'idlm-reinforce') or devices not in (1, 2, 4, 8):
        raise ValueError('Expected IDLM or IDLM-REINFORCE on 1/2/4/8 GPUs')
    reinforce = method == 'idlm-reinforce'
    accum = 512 // (32 * devices)
    return [
        'mode=train', 'seed=1', 'data=tiny-gsm', r'data.separator="\n"',
        'data.wrap=False', 'data.train_on_prompt=False', 'data.train_on_pad=True',
        'data.filter_too_long=True', 'model=small', 'model.length=512', 'algo=mdlm',
        'strategy.find_unused_parameters=True',
        'logger.experiment=generative-models/tinygsm',
        f'logger.name={method} raw EMA confidence passk', 'logger.force_new_run=True',
        'adversarial_distill.is_distill=True', 'adversarial_distill.mmd_implementation=legacy',
        f'adversarial_distill.dmd_weight={0.0 if reinforce else 1.0}',
        f'adversarial_distill.mmd_reward_weight={1.0 if reinforce else 0.0}',
        'adversarial_distill.mmd_reward_type=' + ('idlm' if reinforce else 'feature'),
        'adversarial_distill.mmd_group_size=4', 'adversarial_distill.fake_frequency=1',
        f'adversarial_distill.accum_batches={accum}',
        f'training.finetune_path={assets["mdlm"]}', 'training.finetune_weights=raw',
        'training.loss_precision=bf16', 'training.ema=0.9999',
        f'model_weights_dir={assets["bundle"]}', 'optim.lr=' + ('1e-5' if reinforce else '1e-6'),
        'loader.batch_size=32', 'loader.eval_batch_size=64', 'loader.global_batch_size=512',
        'loader.num_workers=4', 'eval.generate_samples=True', 'eval.compute_generative_perplexity=False',
        'eval.validate_before_train=True', 'eval.start_step=0', 'eval.gsm8k_at_validation=True',
        'eval.gsm8k_every_n_validations=1', 'eval.gsm8k_num_examples=null', 'eval.gsm8k_batch_size=16',
        'eval.tinygsm_exec=True', f'eval.gsm8k_data_path={assets["data"]}',
        'eval.gsm8k_weights=[raw,ema]', 'eval.gsm8k_passk=True', 'eval.gsm8k_passk_samples=32',
        'eval.gsm8k_passk_budgets=[4,8,16,32]', 'eval.gsm8k_passk_ks=[1,2,4,8,16,32]',
        'eval.gsm8k_passk_temperature=0.0', 'eval.gsm8k_passk_seed=1',
        'eval.gsm8k_pareto_fp32=True', 'eval.gsm8k_pareto_temperature=0.0',
        'eval.gsm8k_pareto_thresholds=[0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,0.98,0.99,0.995,0.999]',
        'callbacks.checkpoint_monitor.monitor=val/nll', 'callbacks.checkpoint_monitor.save_top_k=0',
        'callbacks.checkpoint_every_n_steps.save_top_k=0', 'callbacks.checkpoint_every_n_steps.save_last=False',
        'checkpointing.use_periodic_train_batch_checkpoint=False',
        f'callbacks.checkpoint_every_n_steps.every_n_train_steps={500*accum}',
        # One live+EMA checkpoint after each validation, together with its JSON.
        'checkpointing.save_evaluated_checkpoints=True', 'checkpointing.save_evaluated_from_step=0',
        'checkpointing.resume_from_ckpt=False', 'trainer.precision=bf16',
        'trainer.accumulate_grad_batches=1', 'trainer.gradient_clip_val=null',
        f'trainer.devices={devices}', f'trainer.max_steps={stop or 20000}',
        f'trainer.val_check_interval={500*accum}', 'trainer.limit_val_batches=500',
        'trainer.num_sanity_val_steps=0', 'sampling.steps=64', 'sampling.predictor=ancestral_cache',
        'sampling.noise_removal=ancestral', 'sampling.num_sample_batches=2', 'sampling.num_sample_log=8',
        f'data.cache_dir={assets["cache"]}', f'checkpointing.save_dir={out}', f'hydra.run.dir={out}',
    ]
