from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


class TTorchDict(dict):
    def to(self, device: torch.device, float_dtype: torch.dtype = torch.float32) -> "TTorchDict":
        cast_dict = dict()

        for key, value in self.items():
            if not hasattr(value, "to"):
                cast_dict[key] = value
            elif isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                cast_dict[key] = value.to(device=device, dtype=float_dtype)
            else:
                cast_dict[key] = value.to(device=device)


        return TTorchDict(cast_dict)
    

class Collate:
    def __init__(
        self,
        stackable: Optional[List] = None,
        non_stackable: Optional[List] = None,
        names2cat: Optional[List] = None,
        names2repeat: Optional[Dict[str, int]] = None,
    ) -> None:

        self.stackable = stackable or []
        self.non_stackable = non_stackable or []
        self.names2cat = names2cat or []
        self.names2repeat = names2repeat or dict()

    def __call__(self, rows: List[Dict[str, Any]]):
        result = TTorchDict()
        for name in self.stackable:
            result[name] = torch.stack([row[name] for row in rows], dim=0)

        for name in self.names2cat:
            result[name] = torch.cat([row[name] for row in rows], dim=0)

        for name in self.non_stackable:
            if isinstance(rows[0][name], torch.Tensor) and name != 'image':
                result[name] = pad_sequence(
                    [row[name] for row in rows], batch_first=True, padding_value=0
                )
            else:
                result[name] = [row[name] for row in rows]

        for name in result:
            if name not in self.names2repeat:
                continue

            if isinstance(result[name], torch.Tensor):
                # this logic is more difficult than torch.expand, but does not cause extra memory allocation
                tensor = result[name]
                tensor_shape = tensor.shape

                tensor = tensor.view(tensor_shape[0], 1, *tensor_shape[1:])
                tensor_list = self.names2repeat[name] * [tensor]

                tensor = torch.cat(tensor_list, dim=1)
                result[name] = tensor.view(-1, *tensor_shape[1:])
            elif isinstance(result[name], list):
                result[name] = sum((self.names2repeat[name] * [item] for item in result[name]), [])
            else:
                raise NotImplementedError(f"Repeat logic is not implemented for {type(result[name])}.")

        return result

