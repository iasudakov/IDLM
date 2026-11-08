from typing import Optional, Dict, Any
import functools

from yt_tools.utils import instantiate_from_config, call_with_remap
from yt_tools.processing.process_fns import *


class Preprocessor:
    def __init__(self, preprocesses: Optional[List] = None):
        self.preprocesses = preprocesses if preprocesses is not None else []

    @staticmethod
    def _process(
        data: Dict[str, Any],
        in_map: Union[List[str], Dict[str, str]],
        out_map: Optional[Union[str, List[str]]] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        function = functools.partial(instantiate_from_config, kwargs)
        result = call_with_remap(function=function, data=data, in_map=in_map, out_map=out_map, overwrite=overwrite)
        return result

    @staticmethod
    def _dict_maybe_decode_utf8(data: Dict[Union[str, bytes], Any]) -> Dict[str, Any]:
        decoded_data = {}
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            decoded_data[key] = value
        return decoded_data

    def __call__(self, row):
        data_dict = self._dict_maybe_decode_utf8(row)
        for preprocess in self.preprocesses:
            data_dict.update(self._process(data_dict, **preprocess))
        return data_dict
