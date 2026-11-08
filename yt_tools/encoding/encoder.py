from typing import List
from yt_tools.processing.processor import Preprocessor
from typing import Dict, Any, List


def is_iterable(value: Any) -> bool:
    try:
        iter(value)
        return True
    except TypeError:
        return False


def dict_of_lists_to_list_of_dicts(d: Dict[str, Any], drop_non_iterable: bool = True) -> List[Dict[str, Any]]:
    """
    Convert a dictionary of lists to a list of dictionaries.
    If `drop_non_iterable` is True, only keep iterable values to avoid crashing
    for fields such as 0-dimensional tensors.
    """
    if drop_non_iterable:
        d = {k: v for k, v in d.items() if is_iterable(v)}

    return [dict(zip(d, t)) for t in zip(*d.values())]


class Encoder(Preprocessor):
    def __call__(self, rows: Dict[str, List[Any]]) -> List[Dict[str, bytes]]:
        result = []
        list_of_rows = dict_of_lists_to_list_of_dicts(rows)
        for row in list_of_rows:
            processed_row = {}
            for preprocess in self.preprocesses:
                processed_row.update(self._process(row, **preprocess))
            result.append(processed_row)
        return result
