import base64
import io
import random
from typing import Any, List, Union, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf, ListConfig
import yt
import yt.yson as yson
from yt_tools.encoding.process_fns import NumpyEncoding
from yt_tools.processing.frame_samplers import UniformFrameSampler
from yt_tools.utils import instantiate_from_config
from numpy.typing import NDArray
from PIL import Image

try:
    import decord
except ImportError:
    decord = None

_STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def to_tensor(val: Union[float, int, NDArray], dtype: Union[str, torch.dtype] = None) -> torch.Tensor:
    if isinstance(dtype, str):
        dtype = _STR_TO_DTYPE[dtype]
    return torch.as_tensor(val, dtype=dtype)


def multiple_to_tensor(*args, dtype: Union[str, torch.dtype] = None) -> Tuple[torch.Tensor, ...]:
    if isinstance(dtype, str):
        dtype = _STR_TO_DTYPE[dtype]
    return tuple(torch.as_tensor(val, dtype=dtype) for val in args)


def normalize_image(val):
    return val / 127.5 - 1.0


def image_to_tensor(
    image: Union[float, int, NDArray], dtype: Union[str, torch.dtype] = None, normalize: bool = True
) -> torch.Tensor:
    image = np.transpose(image, [2, 0, 1])
    image = to_tensor(image, dtype=dtype)
    if normalize:
        image = normalize_image(image)
    return image


def string_decode(string: str):
    if string is None:
        string=b""
    return string.decode("utf-8")


def identity(data: bytes) -> bytes:
    return data


def stack(dtype: np.dtype = None, *args):
    return np.array(args, dtype=dtype)


def decode_image_from_bytes(
    image_bytes,
    dtype: np.dtype = np.uint8,
    use_base64: bool = False,
    transpose_hwc2chw: bool = True,
    convert_bgr2rgb=True,
) -> np.array:
    image_bytes = yson.get_bytes(image_bytes)
    if use_base64:
        image_bytes = base64.b64decode(image_bytes)

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=dtype), cv2.IMREAD_ANYCOLOR)
    
    if convert_bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if transpose_hwc2chw:
        image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_torchvision_images(images: torch.Tensor, dtype: str = "float32") -> torch.Tensor:
    images = images.to(dtype=_STR_TO_DTYPE[dtype])
    images.div_(127.5).sub_(1.0)

    return images


def decode_mask_from_bytes(image_bytes, dtype=np.uint8):
    image_bytes = yson.get_bytes(image_bytes)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=dtype), cv2.IMREAD_GRAYSCALE)

    if len(image.shape) == 2:
        image = image[:, :, None]

    image = np.transpose(image, [2, 0, 1])
    image = image.astype(np.float32) / 255.0

    return image


def crop_image(
    img: np.ndarray,
    mask: np.ndarray,
) -> List[torch.FloatTensor]:
    h, w, _ = img.shape
    crop = min(h, w)
    slice_h, slice_w = slice((h - crop) // 2, (h + crop) // 2), slice((w - crop) // 2, (w + crop) // 2)
    img = img[slice_h, slice_w, :]
    mask = mask[slice_h, slice_w]
    return img, mask


def pad_image(
    img: np.ndarray,
    mask: np.ndarray,
) -> List[torch.FloatTensor]:
    h, w, _ = img.shape
    ext = max(h, w)
    pad_h, pad_w = ((ext - h) // 2,), ((ext - w) // 2,)
    img = np.pad(img, (pad_h * 2, pad_w * 2, (0, 0)), "edge")
    mask = np.pad(mask, (pad_h * 2, pad_w * 2), "constant", constant_values=((0, 0), (0, 0)))
    return img, mask


def decode_latent_nparray_from_bytes(image_bytes, dtype: np.dtype = np.float32):
    arr_bytes = yt.yson.get_bytes(image_bytes)
    arr = np.frombuffer(arr_bytes, dtype=dtype)
    arr = np.reshape(arr, (4, 64, 64))
    return to_tensor(arr, dtype=torch.float32)


def bytes2torch(arr_bytes: Any, dtype: Union[str, torch.dtype] = torch.float32) -> torch.Tensor:
    if isinstance(dtype, str):
        dtype = _STR_TO_DTYPE[dtype]

    arr_bytes = yson.get_bytes(arr_bytes)
    decoded = np.load(io.BytesIO(arr_bytes))
    return torch.as_tensor(decoded, dtype=dtype)


def decode_nparray_from_bytes(
    arr_bytes,
    dims: Optional[Tuple[int, ...]],
    dtype: Optional[torch.dtype],
    encoding: NumpyEncoding,
) -> Union[torch.Tensor, NDArray]:
    if encoding == NumpyEncoding.BYTES:
        assert (
            dims is not None and dtype is not None
        ), "dims and dtype should be provided for byte-encoded numpy arrays."

    arr_bytes = yson.get_bytes(arr_bytes)
    if encoding == NumpyEncoding.BYTES:
        decoded = torch.frombuffer(arr_bytes, dtype=dtype)
        decoded = decoded.reshape(*dims)
    elif encoding == NumpyEncoding.BUFFER:
        decoded = np.load(io.BytesIO(arr_bytes))
    else:
        raise NotImplementedError()
    return decoded


def decode_quantized_embeddings(
    arr_bytes: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    decoded_map = np.load(io.BytesIO(arr_bytes))
    decoded = decoded_map["ids"] * decoded_map["scales"]
    return to_tensor(decoded, dtype=dtype)


def decode_embeddings_pipeline(
    arr_bytes: Any,
    text_embedding_dim: Optional[int] = 4096,  # TODO: get rid of this default
    encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES,
    dtype: str = "float32",
    target_seq_len: Optional[int] = None,
    quantized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = _STR_TO_DTYPE[dtype]
    # TODO: @mathemage remove hardcoded float16 for text embeds
    # here dtype is *out* dtype
    if not quantized:
        embedding_dim = (-1, text_embedding_dim) if text_embedding_dim is not None else None
        decoded = decode_numpy_pipeline(
            arr_bytes, embedding_dim, encoding=encoding, dtype=torch.float16, out_dtype=dtype
        )
    else:
        decoded = decode_quantized_embeddings(arr_bytes, dtype)
    text_mask = torch.ones(decoded.shape[0], dtype=torch.bool)
    if target_seq_len is not None:
        if text_mask.shape[0] < target_seq_len:
            decoded = torch.nn.functional.pad(decoded, (0, 0, 0, target_seq_len - text_mask.shape[0]))
            text_mask = torch.nn.functional.pad(text_mask, (0, target_seq_len - text_mask.shape[0]))
        else:
            decoded = decoded[:target_seq_len]
            text_mask = text_mask[:target_seq_len]
    return decoded, text_mask


def decode_numpy_pipeline(
    arr_bytes,
    dims: Optional[Tuple[int, ...]] = (3, 64, 64),
    dtype: Optional[Union[torch.dtype, str]] = torch.float16,
    encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES,
    out_dtype: Union[torch.dtype, str] = torch.float32,
):
    """
    If encoding=NumpyEncoding.BUFFER then dims and dtype are loaded from the encoded data.
    If encoding=NumpyEncoding.BYTES then dims and dtype should be provided explicitly.
    """
    if isinstance(encoding, str):
        encoding = NumpyEncoding[encoding]

    if isinstance(dtype, str):
        dtype = _STR_TO_DTYPE[dtype]

    decoded = decode_nparray_from_bytes(arr_bytes, encoding=encoding, dtype=dtype, dims=dims)
    return to_tensor(decoded, dtype=out_dtype)


def decode_pos_mask_pipeline(arr_bytes):
    decoded = decode_nparray_from_bytes(arr_bytes, encoding=NumpyEncoding.BYTES, dims=(-1,), dtype=torch.bool)
    return to_tensor(decoded, dtype=torch.float32)


def prepare_continuous_pipeline(*args):
    stacked = stack(*args, dtype=np.float32)
    return to_tensor(stacked, dtype=torch.float32)


def prepare_discrete_pipeline(val):
    val = np.array([val], dtype=np.int32)
    return to_tensor(val, dtype=torch.int32)


def decode_continuous_from_bytes(arr_bytes):
    decoded = decode_nparray_from_bytes(arr_bytes)
    return to_tensor(decoded, dtype=torch.float32)


def get_dummy_tensor(
    size: Union[ListConfig, List[int]],
    fill_value: Any,
    dtype: Union[torch.dtype, str],
    squeeze: bool = False,
) -> torch.Tensor:
    if isinstance(size, ListConfig):
        size = OmegaConf.to_container(size)

    if isinstance(dtype, str):
        dtype = _STR_TO_DTYPE[dtype]

    tensor = torch.full(size, fill_value, dtype=dtype)
    if squeeze:
        tensor = tensor.squeeze_()

    return tensor


# todo (atselousov: 2023.12.06): this function is depricated and will be removed
def get_dummy_float32_tensor(
    size: Union[ListConfig, List[int]],
    fill_value: Any,
    squeeze: bool = False,
) -> torch.Tensor:
    return get_dummy_tensor(size=size, fill_value=fill_value, dtype=torch.float32, squeeze=squeeze)


# todo (atselousov: 2023.12.06): this function is depricated and will be removed
def get_dummy_bool_tensor(
    size: Union[ListConfig, List[int]],
    fill_value: Any,
    squeeze: bool = False,
) -> torch.Tensor:
    return get_dummy_tensor(size=size, fill_value=fill_value, dtype=torch.bool, squeeze=squeeze)


def get_empty_string(*args, **kwargs) -> str:
    return ""


# Soon deprecated

def decode_image_pipeline(image_bytes, convert_bgr2rgb=True, size=None):
    image = decode_image_from_bytes(image_bytes, transpose_hwc2chw=False, convert_bgr2rgb=convert_bgr2rgb)
    image = np.transpose(image, [2, 0, 1])
    image = to_tensor(image, dtype=torch.float32)
    
    if size is not None:
        size = [int(s) for s in size.split(',')]
        image = torch.nn.functional.interpolate(
            image[None], size=size, mode='bilinear'
        )[0]
        
    return normalize_image(image)


def decode_image_pipeline_with_augmentations(
    image_bytes, aug_pipeline_config=None
):
    image = decode_image_from_bytes(image_bytes, transpose_hwc2chw=False)
    if aug_pipeline_config is not None:
        aug_pipeline = instantiate_from_config(
            aug_pipeline_config
        )
        image = aug_pipeline(image)
    image = np.transpose(image, [2, 0, 1])
    image = to_tensor(image, dtype=torch.float32)
    return normalize_image(image)


def np_chw_to_pil(img: np.ndarray):
    # if channel-first (C,H,W), convert to (H,W,C)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] != img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))
    return Image.fromarray(img)


# ImageNet specific preprocessing
# ---------------------------------------------------------------------------------------------------------

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def imagenet_model_preprocessing(model_type, img_size):
    if model_type == 'jit' or model_type == 'rae':
        transform = transforms.Compose([
            transforms.Lambda(np_chw_to_pil),
            transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor() if model_type == 'jit' else transforms.ToTensor()
        ])
    elif model_type == 'sit':
        transform = transforms.Compose([
            transforms.Lambda(np_chw_to_pil),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    return transform


def imagenet_preprocessing(
    image_bytes,
    dtype: np.dtype = np.uint8,
    use_base64: bool = False,
    transpose_hwc2chw: bool = True,
    convert_bgr2rgb=True,
    img_size = 256,
    model_type = 'jit'
) -> torch.Tensor:
    image = decode_image_from_bytes(image_bytes, transpose_hwc2chw=transpose_hwc2chw, convert_bgr2rgb=convert_bgr2rgb)
    transform = imagenet_model_preprocessing(model_type, img_size)
    image = transform(image)
    return image
# ---------------------------------------------------------------------------------------------------------


def np_center_crop_arr(np_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*np_image.shape[:2]) >= 2 * image_size:
        h, w = np_image.shape[:2]
        np_image = cv2.resize(np_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    h, w = np_image.shape[:2]
    scale = image_size / min(h, w)
    arr = cv2.resize(np_image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)

    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def efficient_imagenet_model_preprocessing(model_type, img_size):
    if model_type == 'jit' or model_type == 'rae':
        transform = transforms.Compose([
            transforms.Lambda(lambda np_image: np_center_crop_arr(np_image, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor() if model_type == 'jit' else transforms.ToTensor()
        ])
    elif model_type == 'sit':
        transform = transforms.Compose([
            transforms.Lambda(lambda np_image: np_center_crop_arr(np_image, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    return transform


def efficient_imagenet_preprocessing(
    image_bytes,
    dtype: np.dtype = np.uint8,
    use_base64: bool = False,
    transpose_hwc2chw: bool = False,
    convert_bgr2rgb=True,
    img_size = 256,
    model_type = 'jit'
) -> torch.Tensor:
    image = decode_image_from_bytes(image_bytes, transpose_hwc2chw=transpose_hwc2chw, convert_bgr2rgb=convert_bgr2rgb)
    transform = efficient_imagenet_model_preprocessing(model_type, img_size)
    image = transform(image)
    return image
