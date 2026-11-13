"""Paper ablations on the corrected archive trainer and REINFORCE estimator.

Loaded explicitly by paper_train.py. No installed repository file is modified.
The fast student head, candidate sampler, leave-one-out baseline and loss
normalization are the archive's. Nonbaseline feature scoring uses chunked
reference forwards; it can be slower than the fused baseline feature scorer.
"""
import torch
from omegaconf import OmegaConf
import mmd_archive
from mmd.objective import FeatureReward, feature_reward, _kernel_sum
from mmd.objective_fast import ScoredHead

METHODS = {'baseline', 'attraction', 'regression', 'linear', 'sequence_rbf',
           'embeddings', 'ar'}
_original_validate = mmd_archive.validate_config
_OriginalObjective = mmd_archive.ArchiveObjective


def validate_config(config):
    method = config.get('paper_ablation', 'baseline')
    if method not in METHODS:
        raise ValueError('Unknown paper_ablation: ' + str(method))
    if method == 'baseline':
        return _original_validate(config)
    cfg = config.adversarial_distill
    expected = {
        'mmd_feature_source': 'ar' if method == 'ar' else 'diffusion',
        'mmd_feature_layer': -1 if method == 'embeddings' else 6,
        'mmd_kernel': 'linear_l2' if method == 'linear' else 'rbf',
        'mmd_kernel_level': 'sentence' if method in ('linear', 'sequence_rbf') else 'token',
        'mmd_reward_type': 'feature_mse' if method == 'regression' else 'feature',
        'mmd_rbf_omit_fake_fake': method == 'attraction',
    }
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'{method} requires {key}={value!r}')
    # Validate all unchanged archive constraints using the baseline view.
    canonical = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    for key, value in dict(mmd_feature_source='diffusion', mmd_feature_layer=6,
                           mmd_kernel='rbf', mmd_kernel_level='token',
                           mmd_reward_type='feature', mmd_rbf_omit_fake_fake=False).items():
        canonical.adversarial_distill[key] = value
    _original_validate(canonical)


def ablation_reward(real, fake, selected, method, alpha, block_size=0,
                    exclude_same_position=True, return_paper=False):
    """real N,L,D; fake N,B,L,D. Never pool across different prompts."""
    if method in ('baseline', 'ar', 'embeddings'):
        return feature_reward(real, fake, selected, alpha=alpha,
                              block_size=block_size,
                              exclude_same_position=exclude_same_position,
                              return_paper=return_paper)
    if real.ndim != 3 or fake.ndim != 4 or fake.shape[0] != real.shape[0] or fake.shape[2:] != real.shape[1:]:
        raise ValueError('Expected real N,L,D and fake N,B,L,D')
    real, fake, mask = real.detach().float(), fake.detach().float(), selected.bool()
    count = mask.sum(-1)
    draws = fake.shape[1]
    if draws < 2:
        raise ValueError('The aligned suite uses B >= 2')
    with torch.autocast(device_type=real.device.type, enabled=False):
        if method == 'attraction':
            size = block_size or max(64, min(real.shape[1], 128))
            cross = sum(_kernel_sum(real, fake[:, b], mask.float(), alpha, size)
                        for b in range(draws))
            reward = 2 * cross / (draws * count.square()).clamp_min(1)
            active = count >= (2 if exclude_same_position else 1)
        elif method == 'regression':
            error = (fake - real[:, None]).square().mean(-1)
            reward = -(error * mask[:, None]).sum((1, 2)) / (draws * count).clamp_min(1)
            active = count > 0
        else:
            real_mean = (real * mask[..., None]).sum(1) / count[:, None].clamp_min(1)
            fake_mean = (fake * mask[:, None, :, None]).sum(2) / count[:, None, None].clamp_min(1)
            if method == 'linear':
                # The article's linear_l2 variant: squared distance of means.
                reward = -(fake_mean.mean(1) - real_mean).square().mean(-1)
            elif method == 'sequence_rbf':
                cross = (-alpha * (fake_mean - real_mean[:, None]).square().sum(-1)).exp().mean(1)
                ff = (-alpha * (fake_mean[:, :, None] - fake_mean[:, None, :]).square().sum(-1)).exp()
                off_diagonal = ~torch.eye(draws, dtype=torch.bool, device=real.device)
                reward = 2 * cross - ff[:, off_diagonal].mean(1)
            else:
                raise ValueError(method)
            active = count > 0
    reward = torch.where(active, reward, torch.zeros_like(reward))
    return (reward, active, -reward) if return_paper else (reward, active)


class AblationReward(FeatureReward):
    def __init__(self, backbone, method, layer, **kwargs):
        # The embedding layer is read before any attention block.
        super().__init__(backbone, layer=max(layer, 0), **kwargs)
        if layer == -1:
            self.block = getattr(backbone, 'backbone', backbone).vocab_embed
        self.method = method

    @torch.no_grad()
    def __call__(self, x0, draws, selected, *, return_paper=False):
        torch.rand(x0.shape, device=x0.device)  # same critic RNG convention
        n, length = x0.shape
        hidden = self.features(torch.cat([x0, draws.reshape(-1, length)]))
        real = hidden[:n]
        fake = hidden[n:].reshape(draws.shape[0], n, length, -1).transpose(0, 1)
        return ablation_reward(real, fake, selected, self.method, self.alpha,
                               self.block_size, self.exclude_same_position, return_paper)

    @torch.no_grad()
    def score_candidates(self, x0, samples, selected, group_size, *, return_paper=False):
        # Adapter for unchanged reinforce_loss_fast. B stays the number of
        # independent answers per candidate, G the number of candidates.
        if len(samples) % group_size:
            raise ValueError('Candidate array must contain exactly G*B draws')
        b = len(samples) // group_size
        values = [self(x0, samples[g*b:(g+1)*b], selected, return_paper=return_paper)
                  for g in range(group_size)]
        rewards = torch.stack([v[0] for v in values], -1)
        active = values[0][1]
        if any(not torch.equal(active, v[1]) for v in values):
            raise ValueError('Active row masks differ between candidates')
        if return_paper:
            return rewards, active, torch.stack([v[2] for v in values], -1)
        return rewards, active


class PaperObjective(_OriginalObjective):
    def __init__(self, model):
        validate_config(model.config)
        method = model.config.get('paper_ablation', 'baseline')
        self.paper_method = method
        if method == 'baseline':
            super().__init__(model)
            return
        cfg = model.config.adversarial_distill
        backbone = model.teacher
        if method == 'ar':
            from mmd_reward import load_ar_backbone
            backbone = load_ar_backbone(str(cfg.mmd_ar_model), model.config,
                                        model.tokenizer, model.device)
        self.reward = AblationReward(
            backbone, method, cfg.mmd_feature_layer, alpha=cfg.mmd_rbf_alpha,
            block_size=cfg.mmd_rbf_block_size,
            feature_batch_size=model._mmd_forward_batch_size(),
            exclude_same_position=cfg.mmd_rbf_exclude_same_position)
        self.fast = cfg.mmd_implementation == 'fast'
        self.head = ScoredHead(model.backbone) if self.fast and cfg.mmd_fast_scored_head else None

    def loss(self, *args, **kwargs):
        loss, stats = super().loss(*args, **kwargs)
        if self.paper_method != 'baseline':
            stats['mmd/ablation_surrogate'] = stats.pop('mmd/paper_surrogate')
        return loss, stats


def install():
    mmd_archive.validate_config = validate_config
    mmd_archive.ArchiveObjective = PaperObjective
