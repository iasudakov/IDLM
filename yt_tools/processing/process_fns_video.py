
import io
import random
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
from torchvision import transforms
from torchvision.transforms.functional import crop

from yt_tools.encoding.process_fns import NumpyEncoding

from .frame_samplers import ErrorMode, RandomSequenceSampler, SampleMethod, UniformFrameSampler
from .process_fns import _STR_TO_DTYPE, decode_embeddings_pipeline

try:
    import decord
except ImportError:
    decord = None


def sample_cond_drop_mask(drop_proba: float) -> torch.Tensor:
    # false - text condition will be dropped
    # true - text condition will be kept
    return torch.as_tensor(random.random() > drop_proba, dtype=torch.bool)


def video_frames_reader(
    video_bytes: Any,
    num_frames: int,
    frame_step: int,
    max_frame_id: Optional[int] = None,
    sample_method: str = "uniform",
    error_mode: str = "sample_first",
    dtype: str = "float32",
    frame_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[int]]:
    if decord is None:
        raise ImportError("Module decord is not installed.")

    decord.bridge.set_bridge("torch")

    byte_stream = io.BytesIO(video_bytes)
    vr = decord.VideoReader(byte_stream)
    total_num_frames = len(vr)

    if frame_ids is None:
        sample_method = SampleMethod(sample_method)
        if sample_method is SampleMethod.UNIFORM:
            sampler = UniformFrameSampler(num_frames=num_frames, frame_step=frame_step, error_mode=error_mode)
        elif sample_method is SampleMethod.RANDOM:
            assert max_frame_id is not None, "Specify max_frame_id, if want to use random sampler"
            sampler = RandomSequenceSampler(
                num_frames=num_frames, max_frame_id=max_frame_id, frame_step=frame_step, error_mode=error_mode
            )
        else:
            raise NotImplementedError(f"{sample_method=} is not supported.")

        frame_ids = sampler(total_num_frames)

    # (bs, num_ch, h, w)
    frames = vr.get_batch(frame_ids).permute(0, 3, 1, 2)
    frames = frames.to(dtype=_STR_TO_DTYPE[dtype])
    # todo (atselousov: 2023.10.17): move this operation to cuda
    frames.div_(127.5).sub_(1)

    return frames, frame_ids


def uniform_latent_frames_sampler(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    sampler = UniformFrameSampler(num_frames=num_frames, frame_step=1, error_mode=ErrorMode.RAISE_ERROR)
    frame_ids = sampler(frames.shape[0])

    return frames[frame_ids]


def uniform_video_vae_sampler(
    video_mean: torch.Tensor,
    video_std: torch.Tensor,
    video_orig: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    sampler = UniformFrameSampler(num_frames=num_frames, frame_step=1, error_mode=ErrorMode.RAISE_ERROR)

    frame_ids = sampler(video_std.shape[0])

    return video_mean[frame_ids], video_std[frame_ids], video_orig[frame_ids]


def sample_video_vae_latents(
    video_mean: torch.Tensor,
    video_std: torch.Tensor,
) -> torch.Tensor:
    return video_mean + torch.randn_like(video_mean).mul_(video_std)


def conditional_video_frames_reader(
    num_frames: int, cond_mask: List[int], use_aligned_condition_frames: bool = False, **kwargs
) -> Tuple[torch.Tensor, List[int]]:
    assert all([val in {0, 1} for val in cond_mask]), "All values in mask should be 0 or 1"
    if not use_aligned_condition_frames:
        num_frames += sum(cond_mask)
    return video_frames_reader(num_frames=num_frames, **kwargs)


def normalize_video_frame_ids(frame_ids: List[int], dtype: str = "float32") -> torch.Tensor:
    frame_ids = torch.as_tensor(frame_ids, dtype=_STR_TO_DTYPE[dtype])
    frame_ids -= frame_ids[0].item()

    return frame_ids


def get_video_frame_watermark_masks(
    frames: torch.Tensor,
    wt_height: int = 10,
    wt_width: int = 60,
) -> torch.Tensor:
    bs, _, height, width = frames.shape
    masks = torch.ones(bs, 1, height, width, dtype=torch.float32)

    # hardcode for WebVid dataset
    masks[
        :, :, int(0.6 * height) : int(0.6 * height) + wt_height, width // 2 - wt_width // 2 : width // 2 + wt_width // 2
    ] = 0

    return masks


def crop_video_frames(
    frames: torch.Tensor,
    side_size: int = 64,
    center_crop: bool = True,
    masks: Optional[torch.Tensor] = None,
    top_y: Optional[int] = None,
    top_x: Optional[int] = None,
    scale_coef: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = frames.shape[2:]

    if center_crop:
        top_y = height // 2 - side_size // 2
        top_x = width // 2 - side_size // 2

        crop_method = transforms.CenterCrop(side_size)
    else:
        top_y = top_y or random.randint(0, height - side_size)
        top_x = top_x or random.randint(0, width - side_size)
        if scale_coef:
            top_y = int(scale_coef * top_y)
            top_x = int(scale_coef * top_x)

        crop_method = partial(
            crop,
            top=top_y,
            left=top_x,
            height=side_size,
            width=side_size,
        )

    frames = crop_method(frames)
    masks = crop_method(masks) if masks is not None else masks

    return frames, masks, height, width, top_y, top_x


def get_size_condition(
    side_size: int,
    height: int,
    width: int,
    top_y: int,
    top_x: int,
    dtype: str = "float32",
) -> torch.Tensor:
    size_condition = torch.as_tensor([height, width, top_y, top_x], dtype=_STR_TO_DTYPE[dtype])
    size_condition /= side_size

    return size_condition


def frame_conditioning(
    frames: torch.Tensor,
    frame_position: torch.Tensor,
    cond_mask: Optional[List[int]] = None,
    frame_masks: Optional[torch.Tensor] = None,
    use_aligned_condition_frames: bool = False,
    dtype: str = "float32",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Prepare condition frames based on the cond_mask value. Other arguments can also be modified.

    Parameters:
        frames: Tensor of shape [num_cond_frames, img_channels, h, w]
        frame_position: torch.Tensor of frames_ids diff starting from first frame
        cond_mask: vector of {0, 1} described should frame be used for conditioning
        frame_masks: mask for computing loss not on all pixels
        use_aligned_condition_frames: Flag of usage masking from https://arxiv.org/pdf/2304.08818.pdf
        dtype: Returning tensors dtype
    Returns:
        frames: frames with removed condition frames
        frame_position: frame_position with removed condition frames
        frame_masks: frame_masks with removed condition frames
        condition_frames: Tensor of shape [num_final_frames, cond_channels, h, w] with cond mask
    """
    assert all([val in {0, 1} for val in cond_mask]), "All values in mask should be 0 or 1"
    if cond_mask is None or sum(cond_mask) == 0:
        return frames, frame_masks, None, frame_position

    assert (
        len(frame_position) == len(cond_mask) == len(frames)
    ), "number of frames should be equal to length of condition mask"

    mask = torch.as_tensor(cond_mask, dtype=torch.bool)
    condition_frames, non_empty_condition_mask = prepare_condition_frames(
        input_condition_frames=frames[mask],
        cond_mask=cond_mask,
        use_aligned_condition_frames=use_aligned_condition_frames,
        dtype=dtype,
    )

    if not use_aligned_condition_frames:
        # Removed masked frames from input
        frames = frames[~mask]
        frame_position = frame_position[~mask]
        if frame_masks is not None:
            frame_masks = frame_masks[~mask]

    return frames, frame_masks, condition_frames, non_empty_condition_mask, frame_position


def prepare_condition_frames(
    input_condition_frames: torch.Tensor,
    cond_mask: List[int],
    use_aligned_condition_frames: bool,
    dtype: str = "float32",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    prepare condition frames for video extrapolation/interpolation models
    Parameters:
        frames: Tensor of shape [num_cond_frames, img_channels, h, w]
        cond_mask: vector of {0, 1} described should frame be used for conditioning
        use_aligned_condition_frames: Flag of usage masking from https://arxiv.org/pdf/2304.08818.pdf
        dtype: Returning tensors dtype
    Returns:
        condition_frames: Tensor of shape [num_final_frames, cond_channels, h, w] with cond mask
        non_empty_condition_mask: tensor of shape [num_final_frames, ], where true means that condition frames are set
    """
    assert all([val in {0, 1} for val in cond_mask]), "All values in mask should be 0 or 1"
    assert len(input_condition_frames) == sum(cond_mask)

    num_final_frames = len(cond_mask)
    if not use_aligned_condition_frames:
        # Remove masked frames
        num_final_frames -= sum(cond_mask)

    _, channels, h, w = input_condition_frames.shape
    mask = torch.as_tensor(cond_mask, dtype=torch.bool, device=input_condition_frames.device)

    if use_aligned_condition_frames:
        condition_frames = torch.zeros(
            (num_final_frames, channels + 1, h, w), dtype=_STR_TO_DTYPE[dtype], device=input_condition_frames.device
        )
        condition_frames[mask, :channels] = torch.as_tensor(input_condition_frames, dtype=_STR_TO_DTYPE[dtype])
        condition_frames[mask, -1] = 1.0
        non_empty_condition_mask = mask
    else:
        condition_frames = input_condition_frames.reshape(-1, h, w).unsqueeze(0)
        condition_frames = condition_frames.repeat_interleave(num_final_frames, axis=0)
        non_empty_condition_mask = torch.ones(num_final_frames, dtype=torch.bool, device=input_condition_frames.device)

    return condition_frames, non_empty_condition_mask


def uniform_window_sampler(
    *,
    num_windows: int,
    encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES,
    dtype: str = "float32",
    quantized: bool = False,
    sample_condition: bool = False,
    **windows: Mapping[str, Any],
) -> Tuple[torch.tensor, List[int]]:
    windows = {key: val for (key, val) in windows.items() if val}
    window_keys = list(windows.keys())
    window_keys.sort(key=lambda x: int(x.split("_", 1)[0]))

    sampler = UniformFrameSampler(num_frames=num_windows, frame_step=1, error_mode=ErrorMode.RAISE_ERROR)

    frames = [windows[window_keys[i]] for i in sampler(len(window_keys))]
    frames = [
        decode_embeddings_pipeline(frame_buf, encoding=encoding, dtype=dtype, quantized=quantized)[0]
        for frame_buf in frames
    ]

    # frames[0].shape = (num_ch, num_frames, h, w)
    frames = torch.cat(frames, dim=1)

    return frames
