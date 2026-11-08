import gc
import random
from copy import deepcopy
from math import ceil
from typing import Any, Dict, List, Optional

import torch
from yt_tools.processing.collate import Collate
from yt_tools.processing.processor import Preprocessor
from yt_tools.yt_iterable_dataset import IterableYTDataset
import yt_tools.dist_util as dist
from yt_tools.utils import instantiate_from_config


class IterableDataloader:
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int,
        name: str,
        cycle: bool = True,
        skip_rows: int = 0,
        preprocess: Optional[List] = None,
        stackable: Optional[List] = None,
        non_stackable: Optional[List] = None,
        names2cat: Optional[List] = None,
        names2repeat: Optional[Dict[str, int]] = None
    ):
        self.name = name
        self.dataset_config = dataset
        self.dataset = None
        self.batch_size = batch_size
        self.cycle = cycle
        self.decode_fn = Preprocessor(preprocess)
        self.collate_fn = Collate(
            stackable=stackable,
            non_stackable=non_stackable,
            names2cat=names2cat,
            names2repeat=names2repeat,
        )
        self.reset(skip_rows)

    def _get_dataloader(self, skip_rows: int):
        dataset = instantiate_from_config(
            self.dataset_config,
            row_processor=self.decode_fn,
            skip_rows=skip_rows,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=dataset._drop_last,
            pin_memory=True
        )
        return dataset, iter(dataloader), len(dataloader)

    def reset(self, skip_rows: int = 0):
        self.dataset, self.dataloader, self._len = self._get_dataloader(skip_rows)

    def cleanup(self):
        if hasattr(self, "dataset"):
            del self.dataset
        if hasattr(self, "dataloader"):
            del self.dataloader

        gc.collect()

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                batch = next(self.dataloader)
            except StopIteration:
                if self.cycle:
                    self.cleanup()
                    self.reset()
                    batch = next(self.dataloader)
                    dist.barrier()
                else:
                    raise StopIteration

            return batch


class WeightedIterableDataloader:
    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        batch_size: int,
        name: str,
        cycle: bool = True,
        skip_rows: int = 0,
        preprocess: Optional[List] = None,
        stackable: Optional[List] = None,
        non_stackable: Optional[List] = None,
        names2cat: Optional[List] = None,
        names2repeat: Optional[Dict[str, int]] = None,
    ):
        weights = torch.as_tensor(
            [ds.pop("weight") for ds in datasets], dtype=torch.float64
        )
        weights /= weights.sum()

        always_sample = [ds.pop("always_sample", False) for ds in datasets]
        always_sample_count = always_sample.count(True)
        if always_sample_count > 1:
            raise ValueError(
                f"There can always be examples from max one table, "
                f"but {always_sample_count} tables were specified."
            )

        self.always_sample_index = None
        if always_sample_count:
            self.always_sample_index = always_sample.index(True)
        else:
            dist.print0(
                "Make sure that you always sample from a table with condition. "
                "Otherwise DDP errors are possible."
            )

        bss = [int(batch_size * w) for w in weights.tolist()]
        bss_diff = batch_size - sum(bss, 0)
        bss_change = ceil(bss_diff / len(bss))

        for i in range(len(bss)):
            bs_delta = min(bss_diff, bss_change)
            bss[i] += bs_delta
            bss_diff -= bs_delta

        assert (
            sum(bss, 0) == batch_size
        ), f"Check batch_size logic, {batch_size} and {sum(bss, 0)}."

        self.name = name

        self.cycle = cycle
        self.total_batch_size = batch_size

        self.batch_sizes = bss
        self.weights = weights.tolist()

        self.collate_fn = Collate(
            stackable=stackable,
            non_stackable=non_stackable,
            names2cat=names2cat,
            names2repeat=names2repeat,
        )
        self.preprocess = preprocess
        self.dataset_configs = deepcopy(datasets)

        self.need_reset = True
        self.reset(skip_rows)

    def cleanup(self) -> None:
        if hasattr(self, "dataset_iterators"):
            del self.dataset_iterators

        gc.collect()

    def reset(self, skip_rows: int = 0) -> None:
        if not self.need_reset:
            return

        self.cleanup()

        datasets = []
        for ds_id, bs in enumerate(self.batch_sizes):
            # skip_rows are approximated
            datasets.append(
                self._reset_ith_dataset(
                    ds_id,
                    (skip_rows // self.total_batch_size) * bs,
                )
            )

        self.dataset_lens = [len(ds) for ds in datasets]
        self.dataset_iterators = [iter(ds) for ds in datasets]

        self.need_reset = False

    def _reset_ith_dataset(self, ds_id: int, skip_rows: int = 0) -> IterableYTDataset:
        dist.print0(f"Weighted loader {self.name}: dataset {ds_id} will be reseted.")
        dataset = instantiate_from_config(
            self.dataset_configs[ds_id]["dataset"],
            row_processor=Preprocessor(
                self.dataset_configs[ds_id].get("preprocess", self.preprocess)
            ),
            skip_rows=skip_rows,
        )

        return dataset

    def __len__(self):
        # length approximation
        return min(ln // bs for ln, bs in zip(self.dataset_lens, self.batch_sizes))

    def __iter__(self):
        self.reset()

        return self

    def __next__(self):
        self.need_reset = True

        batch = []

        ds_ids = random.choices(range(len(self.dataset_iterators)), weights=self.weights, k=self.total_batch_size)

        if self.always_sample_index is not None and self.always_sample_index not in ds_ids:
            ds_ids[0] = self.always_sample_index

        for ds_id in ds_ids:
            try:
                item = next(self.dataset_iterators[ds_id])
            except StopIteration as e:
                if self.cycle:
                    self.dataset_iterators[ds_id] = iter(self._reset_ith_dataset(ds_id))
                    item = next(self.dataset_iterators[ds_id])
                    gc.collect()
                else:
                    self.cleanup()
                    raise e

            batch.append(item)

        return self.collate_fn(batch)
