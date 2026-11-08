"""Full-sequence SDTT targets and reverse KL, adapted from jdeschena/sdtt.

Reference: 1150985e90b8f2d5749e4469d5154eff9ec922c4,
src/sdtt/core/distill/multi_round_sdtt.py. No block decomposition is used.
"""

import math

import torch
import torch.nn.functional as F


def round_schedule(original_steps, teacher_steps, minimum_steps, grow_every, step):
    """Return the round, teacher grid size and effective student grid size.

    Saturate at the last complete round rather than exiting the process when
    the requested training budget exceeds the compression schedule.
    """
    values = (original_steps, teacher_steps, minimum_steps, grow_every)
    if any(type(x) is not int or x <= 0 for x in values) or teacher_steps < 2:
        raise ValueError('SDTT grid sizes/cadence must be positive integers; teacher_steps >= 2.')
    if original_steps < teacher_steps * minimum_steps:
        raise ValueError('Initial teacher grid must support at least one distillation round.')
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer.')
    grid, last_round = original_steps, 0
    while grid // teacher_steps >= minimum_steps:
        if grid % teacher_steps:
            raise ValueError('Each SDTT round must divide its teacher grid exactly.')
        if grid // teacher_steps // teacher_steps < minimum_steps:
            break
        grid //= teacher_steps
        last_round += 1
    current = min(step // grow_every, last_round)
    grid = original_steps // teacher_steps ** current
    return current, grid, grid // teacher_steps


@torch.no_grad()
def collect_teacher_targets(xt, t_start, *, forward, noise, mask_index,
                            vocab_size, dt, num_steps, sampling_eps,
                            sample_categorical):
    """Collect each position's distribution when the teacher first reveals it.

    Remaining masked positions use the last teacher prediction. The forward
    time interpolation deliberately follows the released implementation,
    including its endpoint t_start - num_steps * dt (not (num_steps - 1)*dt).
    Initially visible positions never enter the distillation objective.
    """
    if num_steps < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('At least two teacher steps and positive finite dt are required.')
    space = torch.linspace(1, 0, num_steps, device=xt.device).double()[:, None]
    start = t_start.reshape(1, -1).double()
    times = (start * space + (1 - space) * (start - dt * num_steps))
    times = times.clamp_min(sampling_eps)
    targets = torch.zeros((*xt.shape, vocab_size), device=xt.device, dtype=torch.float32)
    current = xt.clone()
    for time in times:
        time = time.float().unsqueeze(-1)
        _, alpha_t = noise(time)
        # Clamping the clean endpoint prevents negative mask mass near t=0.
        # Those negative entries can never win the released exponential race;
        # zero has the same sampled outcome and is a valid categorical weight.
        _, alpha_s = noise((time - dt).clamp_min(0))
        log_probs = forward(current, -alpha_t.log()).float()
        posterior = log_probs.exp() * (alpha_s - alpha_t).unsqueeze(-1)
        posterior[..., mask_index] = (1 - alpha_s)
        proposed = sample_categorical(posterior)
        following = torch.where(current.eq(mask_index), proposed, current)
        revealed = current.ne(following)
        targets[revealed] = log_probs[revealed]
        current = following
    still_masked = current.eq(mask_index)
    targets[still_masked] = log_probs[still_masked]
    return targets


def masked_reverse_kl(student_log_probs, teacher_log_probs, mask):
    """KL(student || teacher), averaged over initially masked answer positions.

    Float32 is used even under bf16 mixed precision. An empty local mask gives
    a differentiable zero, so every DDP rank can complete backward.
    """
    if not bool(mask.any()):
        return student_log_probs.sum() * 0
    student = student_log_probs[mask].float()
    teacher = teacher_log_probs[mask].float()
    return F.kl_div(teacher, student, log_target=True, reduction='batchmean')
