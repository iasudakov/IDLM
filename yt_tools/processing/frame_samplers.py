import random
from enum import Enum
from typing import List, Union

import torch


class SampleMethod(str, Enum):
    UNIFORM = "uniform"
    RANDOM = "random"


# StrEnum cannot be used because of old python
class ErrorMode(str, Enum):
    SAMPLE_FIRST = "sample_first"
    SAMPLE_UNIFORM = "sample_uniform"
    RAISE_ERROR = "raise_error"


class UniformFrameSampler:
    def __init__(
        self,
        num_frames: int,
        frame_step: int,
        error_mode: Union[ErrorMode, str] = ErrorMode.SAMPLE_FIRST,
    ) -> None:
        self._num_frames = num_frames
        self._step = frame_step
        self._length = num_frames * frame_step
        self._error_mode = ErrorMode(error_mode)

    def __call__(self, total_num_frames: int) -> List[int]:
        max_offset = total_num_frames - self._length

        if max_offset >= 0:
            start = random.randint(0, max_offset)
            return list(range(start, start + self._length, self._step))

        if self._error_mode is ErrorMode.SAMPLE_FIRST:
            return self._num_frames * [0]

        if self._error_mode is ErrorMode.SAMPLE_UNIFORM:
            return torch.linspace(0, total_num_frames - 1, self._num_frames).round().long().tolist()

        raise RuntimeError("Video too short!")


class RandomSequenceSampler:
    def __init__(
        self,
        num_frames: int,
        max_frame_id: int = 144,
        frame_step: int = 2,
        error_mode: Union[ErrorMode, str] = ErrorMode.SAMPLE_UNIFORM,
    ) -> None:
        self._num_frames = num_frames
        self._max_frame_id = max_frame_id
        self._error_mode = ErrorMode(error_mode)

        self._frame_grid_pool = list(range(0, max_frame_id + 1, frame_step))

        assert len(self._frame_grid_pool) > self._num_frames

    def __call__(self, total_num_frames: int) -> List[int]:
        max_offset = total_num_frames - self._max_frame_id - 1

        if max_offset >= 0:
            ids = random.sample(self._frame_grid_pool, self._num_frames)
            ids.sort()

            return [max_offset + i for i in ids]

        if self._error_mode is ErrorMode.SAMPLE_FIRST:
            return self._num_frames * [0]

        if self._error_mode is ErrorMode.SAMPLE_UNIFORM:
            return torch.linspace(0, total_num_frames - 1, self._num_frames).round().long().tolist()

        raise RuntimeError("Video too short!")
