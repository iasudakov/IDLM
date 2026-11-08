
from typing import Dict, Any, Union, Optional, List, Callable
from omegaconf import ListConfig, DictConfig
import importlib
import functools
import re
import yt_tools.dist_util as dist


LOG_FN_BLACKLIST = [
    "yt_tools.processing.*",
    "yt_tools.encoding.*",
    "torch.randn_like",
]


def instantiate_from_config(config: Dict, *args, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", dict())
    target = config["target"]
    if re.search("|".join(LOG_FN_BLACKLIST), target) is None:
        dist.print0(f"Instantiating {config['target']} with params {params}")
    return get_obj_from_str(target)(*args, **params, **kwargs)


def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def getattr_nested(obj, dot_separated_names):
    return functools.reduce(getattr, [obj] + dot_separated_names.split("."))


def call_with_remap(
    function: Callable,
    data: Dict[str, Any],
    in_map: Union[List[str], Dict[str, str]],
    out_map: Optional[Union[str, List[str]]] = None,
    overwrite: bool = False,
):
    if isinstance(in_map, list) or isinstance(in_map, ListConfig):
        # get and remap list inputs
        inputs = [data[key] if isinstance(key, str) else key for key in in_map]
        in_cols = in_map
        # process data
        result = function(*inputs)
    elif isinstance(in_map, dict) or isinstance(in_map, DictConfig):
        # get and remap dict inputs
        inputs = {
            key: data[val] if isinstance(val, str) else val
            for key, val in in_map.items()
        }
        in_cols = in_map.values()
        # process data
        result = function(**inputs)
    else:
        raise TypeError(
            f"Type of in_map has to be either dict for keyword arguments or list. "
            f"for positional arguments, but got {type(in_map)}."
        )

    # remap outputs
    if out_map is not None:
        if isinstance(result, tuple):
            assert isinstance(out_map, list) or isinstance(
                out_map, ListConfig
            ), f"Function returns multiple objects, out_map has to be list."
            assert len(result) == len(
                out_map
            ), f" Amount of result items and out_map items must be the same, but {len(result)} and {len(out_map)}."
            # check that out_map names is not in data
            for name in out_map:
                assert (
                    overwrite or name not in data or name in in_cols or name == "_"
                ), f"Result key {name} is going to overwrite correspondent name in input data."
            result = {key: val for key, val in zip(out_map, result) if key != "_"}
        else:
            assert isinstance(
                out_map, str
            ), f"Function doesn't return multiple objects, out_map has to be single str."
            if out_map != "_":
                assert (
                    overwrite or out_map not in data or out_map in in_cols
                ), f"Result key {out_map} is going to overwrite correspondent name in input data."
                result = {out_map: result}
    else:
        # if out_map is not specified - use the same name as input
        assert not isinstance(
            result, tuple
        ), f"Can't infer output name, please, specify out_map."
        result = {list(in_map.values())[0]: result}
    return result