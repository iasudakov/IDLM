"""The three CDLM objectives, adapted to variable-length TinyGSM answers.

Based on SqueezeAILab/CDLM at 708800ab0df03e2c284ab2cb358e42652a5e85a5,
CDLM-train/utils/loss.py. See third_party/CDLM_LICENSE.
"""

import torch
import torch.nn.functional as F


def trajectory_pair(final_tokens, reveal_order, valid_tokens, mask_index,
                    segment_length, *, starts=None):
    """Reconstruct an earlier state and the end of its reveal-order segment.

    Segments describe progress along the trajectory, not attention blocks.
    Completing the entire answer for every pair would erase the positions on
    which CDLM's consistency regularizer is defined.
    """
    if type(segment_length) is not int or segment_length <= 0:
        raise ValueError('segment_length must be a positive integer.')
    valid = valid_tokens.bool()
    lengths = valid.sum(-1)
    if bool((lengths <= 0).any()):
        raise ValueError('Each trajectory needs at least one answer position.')
    if starts is None:
        starts = (torch.rand(len(lengths), device=final_tokens.device) * lengths).long()
    if starts.shape != lengths.shape or bool(((starts < 0) | (starts >= lengths)).any()):
        raise ValueError('Each start index must be in [0, answer_length).')
    ends = torch.minimum((starts // segment_length + 1) * segment_length, lengths)
    earlier_mask = valid & (reveal_order >= starts[:, None])
    later_mask = valid & (reveal_order >= ends[:, None])
    earlier = torch.where(earlier_mask, mask_index, final_tokens)
    later = torch.where(later_mask, mask_index, final_tokens)
    newly_revealed = earlier_mask & ~later_mask
    still_masked = earlier_mask & later_mask
    return earlier, later, newly_revealed, still_masked, starts, ends


def corrupt_ground_truth(tokens, valid_tokens, mask_index, floor=.001):
    """Author GT corruption: p=floor+(1-floor)*U, at least one answer mask."""
    if not 0 < floor < 1:
        raise ValueError('denoise_mask_floor must be in (0, 1).')
    valid = valid_tokens.bool()
    if bool((valid.sum(-1) == 0).any()):
        raise ValueError('Each example needs an answer loss span.')
    probabilities = floor + (1 - floor) * torch.rand(tokens.shape[0], device=tokens.device)
    selected = (torch.rand(tokens.shape, device=tokens.device) < probabilities[:, None]) & valid
    empty = ~selected.any(-1)
    if bool(empty.any()):
        scores = torch.rand(tokens.shape, device=tokens.device).masked_fill(~valid, -1)
        forced = F.one_hot(scores.argmax(-1), num_classes=tokens.shape[1]).bool()
        selected |= empty[:, None] & forced
    return torch.where(selected, mask_index, tokens), selected, probabilities


def distribution_loss(student_log_probs, target, kind='forward_kl', *, target_is_log_probs=False):
    """Released CDLM forward KL omits the constant target entropy (soft CE).

    For consistency, the later student prediction is always stop-gradient.
    Teacher logits and all probability reductions are evaluated in float32.
    """
    if student_log_probs.shape != target.shape:
        raise ValueError('Student and target distributions must have matching shapes.')
    if student_log_probs.numel() == 0:
        return student_log_probs.sum() * 0
    student = student_log_probs.float()
    target = target.detach().float()
    target_logs = target if target_is_log_probs else F.log_softmax(target, dim=-1)
    if kind == 'forward_kl':
        return -(target_logs.exp() * student).sum(-1).mean()
    if kind == 'reverse_kl':
        return (student.exp() * (student - target_logs)).sum(-1).mean()
    raise ValueError('kld_type must be forward_kl or reverse_kl.')


def denoising_loss(log_probs, original_tokens, selected, probabilities, valid_tokens):
    """Inverse-p GT denoising, normalized by each example's answer span."""
    token_loss = -log_probs.float().gather(-1, original_tokens.unsqueeze(-1)).squeeze(-1)
    scaled = token_loss / (probabilities.float()[:, None] + 1e-8)
    numerator = torch.where(selected, scaled, torch.zeros_like(scaled)).sum(-1)
    denominator = valid_tokens.sum(-1).float()
    if bool((denominator <= 0).any()):
        raise ValueError('Answer lengths must be positive.')
    return (numerator / denominator).mean()
