from enum import Enum
import io
from typing import Union, List, Any
import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from typing import Optional
from torchvision.utils import make_grid
import math
import string
import re


class NumpyEncoding(Enum):
    BYTES = 0
    BUFFER = 1


def to_numpy(val: torch.Tensor, dtype: Union[str, np.dtype] = None) -> NDArray:
    return val.cpu().numpy().astype(dtype)


def tensor2img(tensor: torch.Tensor) -> torch.Tensor:
    "Img tensor in [-1, 1] -> [0, 255] uint8"
    return ((tensor.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)


def image_tensor2bytes(
    image: torch.Tensor, 
    normalize: bool = True, 
    img_ext: str = ".jpg",
) -> bytes:
    """
    image: C x H x W
    """
    if normalize:
        image = tensor2img(image)
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, image = cv2.imencode(img_ext, image)
    return image.tobytes()


def numpy2bytes(array: NDArray, encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES) -> bytes:
    if isinstance(encoding, str):
        encoding = NumpyEncoding[encoding]

    if encoding == NumpyEncoding.BYTES:
        return array.tobytes()
    elif encoding == NumpyEncoding.BUFFER:
        buffer = io.BytesIO()
        np.save(buffer, array)

        return buffer.getvalue()
    else:
        raise NotImplementedError()


def text_embedding2bytes(
    text_embedding: torch.Tensor,
    text_mask: torch.Tensor,
    dtype=np.float16,
    encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES,
) -> bytes:
    """
    text_embedding: L x D
    text_mask: L
    """
    assert text_mask.shape[0] == text_embedding.shape[0]
    real_sequence_len = text_mask.sum(dim=0).long().item()
    return tensor2numpy2bytes(text_embedding[:real_sequence_len, :], dtype, encoding)


def string2bytes(string: str) -> bytes:
    return string.encode("utf-8")


def tensor2list(arr: torch.Tensor) -> List[Any]:
    """
    yt does not support numpy dtypes, so we need to use builtin ones like float/int
    """
    return arr.tolist()


def tensor2numpy2bytes(
    arr: torch.Tensor,
    dtype: Union[str, np.dtype] = np.float32,
    encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES,
) -> bytes:
    if isinstance(dtype, str):
        dtype = eval(dtype)
    arr_np = to_numpy(arr, dtype=dtype)
    return numpy2bytes(arr_np, encoding)


def identity(data: bytes) -> bytes:
    return data


def image_tensor2grid(image: torch.Tensor, normalize: bool = True, ncol: Optional[int] = None) -> torch.Tensor:
    """
    Converts a batch of image tensors into a grid of images.

    Args:
        image (torch.Tensor): A batch of image tensors of shape (B, C, H, W),
            where B is the batch size, C is the number of channels, H is the height,
            and W is the width.
        normalize (bool, optional): If True, normalizes the input tensor
            to a valid image range, i.e., [0, 255]. Default is True.
        ncol (Optional[int], optional): The number of columns in the resulting image grid.
            If not specified, the grid will have a square layout with ncol=sqrt(B).

    Returns:
        torch.Tensor: A tensor representing the grid of images, with shape (C, H', W'),
            where H' and W' are the height and width of the grid, respectively.
    """
    if normalize:
        image = tensor2img(image)

    if ncol is None:
        ncol = round(math.sqrt(image.shape[0]))

    grid = make_grid(image, nrow=ncol, value_range=[0, 255])
    return grid


def clean_punctutation(query: str) -> str:
    for punct in string.punctuation:
        query = query.replace(punct, " ")
    return query


def remove_multiple_spaces(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def chop_and_add_dots(query: str, max_len: int = 128) -> str:
    if len(query) > max_len:
        query = query[: max_len - 3] + "..."
    return query


def clean_query(query: str, max_len: int = 128) -> str:
    query = clean_punctutation(query)
    query = remove_multiple_spaces(query)
    query = chop_and_add_dots(query, max_len=max_len)
    return query


def query_list2grid(queries: List[str], ncol: Optional[int] = None, max_len: int = 128) -> str:
    """
    Converts a list of query strings into a formatted grid table in Markdown.

    Args:
        queries (List[str]): A list of query strings to be formatted as a grid table.
        ncol (Optional[int], optional): The number of columns in the resulting grid table.
            If not specified, the table will have a square layout with ncol=sqrt(len(queries)).
        max_len (int, optional): The maximum length of each query string after cleaning.
            Default is 128.

    Returns:
        str: A string representing the grid table in Markdown format.
    """
    queries = [clean_query(query, max_len=max_len) for query in queries]
    if ncol is None:
        ncol = round(math.sqrt(len(queries)))
    nrow = int(np.ceil(len(queries) / ncol))
    queries = queries + [""] * (nrow * ncol - len(queries))

    table = []
    for i in range(nrow):
        row = []
        for j in range(ncol):
            row.append(queries[i * ncol + j])
        table.append(row)

    table = [f"| {' | '.join(row)} |" for row in table]
    table.insert(1, f"| {' | '.join(['---'] * ncol)} |")

    return "\n".join(table)
