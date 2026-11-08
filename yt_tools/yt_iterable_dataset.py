from typing import Optional, Callable
import math
import time
from datetime import timedelta
import ytreader
import yt.wrapper as yt

from torch.utils.data import IterableDataset
import yt_tools.dist_util as dist
import torch.utils.data as torch_data

from .parallel_proc import iterate_with_preproc
from .yt_utils import (
    get_yt_token,
    get_modification_time,
    get_current_time,
    get_table_schema,
)


def get_local_num_workers():
    num_local = 1
    local_worker_info = torch_data.get_worker_info()
    if local_worker_info is not None:
        num_local = local_worker_info.num_workers
    return num_local


def get_local_worker_id():
    id_local = 0
    local_worker_info = torch_data.get_worker_info()
    if local_worker_info is not None:
        id_local = local_worker_info.id
    return id_local


def get_dist_num_workers():
    num_dist = 1
    if dist.is_initialized():
        num_dist = dist.num_workers()
    return num_dist


def get_dist_worker_id():
    id_dist = 0
    if dist.is_initialized():
        id_dist = dist.worker_idx()
    return id_dist


def _make_reader(
    table: str,
    num_workers: int,
    worker_idx: int,
    drop_last: bool = True,
    yt_num_readers: int = 16,
    yt_cache_size: int = 1024,
    read_from_tail: Optional[int] = None,
    num_retries: int = 1,
    timeout: Optional[int] = None,
    table_freshness: Optional[int] = None,
    yt_proxy: str = "hahn",
):
    assert num_retries >= 1

    if num_workers > 1 and not drop_last:
        dist.print0(
            """WARNING: drop_last is False with num_workers > 1.
            This may cause ddp to hang while training.
            Use with care and only for inference (when synchronization is not required)
            """
        )

    err_msg = "No error message"
    for retry in range(num_retries):
        if retry > 0 and timeout is not None:
            dist.print0(f"Retry in {timeout} sec.")
            time.sleep(timeout)

        try:
            reader = ytreader.YTTableParallelReader(
                yt_proxy, table, yt_cache_size, yt_num_readers
            )
        except RuntimeError as e:
            err_msg = str(e)
            dist.print0(f"Caught exception when creating table {table}: {err_msg}.")
            reader = None
            continue

        if read_from_tail is not None:
            assert (
                read_from_tail >= num_workers * yt_num_readers
            ), f"Table should have at least {num_workers * yt_num_readers} rows, but read_from_tail is {read_from_tail}."

            if table_freshness is not None:
                client = yt.YtClient(proxy=yt_proxy, token=get_yt_token())
                modification_time = get_modification_time(table, client)
                current_time = get_current_time(client)
                if current_time - modification_time > timedelta(
                    seconds=table_freshness
                ):
                    err_msg = f"Table {table} is older than {table_freshness} sec."
                    dist.print0(err_msg)
                    reader = None
                    continue

            reader = reader.make_subset_reader(
                max(0, reader.num_rows - read_from_tail), reader.num_rows
            )
            break

    assert reader is not None, f"Failed to load table {table}: {err_msg}"

    if callable(reader.num_rows):
        num_rows = reader.num_rows()
    else:
        num_rows = reader.num_rows

    if num_workers > 1:
        num_rows_per_worker = int(math.floor(num_rows / num_workers))
        first_row_idx = worker_idx * num_rows_per_worker
        # for training we dont want chunks of different size
        # but for inference in DDP setting we may loose a small number of rows
        # fix the size of the last chunk in this case
        if drop_last:
            last_row_idx = first_row_idx + num_rows_per_worker
        else:
            if worker_idx == num_workers - 1:
                last_row_idx = num_rows
            else:
                last_row_idx = first_row_idx + num_rows_per_worker
        return reader.make_subset_reader(first_row_idx, last_row_idx)
    return reader


class IterableYTDataset(IterableDataset):
    def __init__(
        self,
        table: str,
        row_processor: Callable,
        num_readers: int = 8,
        chunk_distributed: bool = True,
        skip_rows: int = 0,
        buf_size: int = 2048,
        preproc_workers: int = 8,
        drop_last: bool = True,
        read_from_tail: Optional[int] = None,
        num_retries: int = 1,
        timeout: Optional[int] = None,
        table_freshness: Optional[int] = None,
        yt_proxy="hahn",
    ):
        self._table = table
        self._row_processor = row_processor
        self._num_readers = num_readers
        self._reader = None
        self._chunk_distributed = chunk_distributed
        self._buf_size = buf_size
        self._preproc_workers = preproc_workers
        self._drop_last = drop_last

        global_num_workers, global_worker_idx = (
            get_dist_num_workers(),
            get_dist_worker_id(),
        )
        local_num_workers, local_worker_idx = (
            get_local_num_workers(),
            get_local_worker_id(),
        )
        if self._chunk_distributed:
            num_workers = global_num_workers * local_num_workers
            worker_idx = local_num_workers * global_worker_idx + local_worker_idx
        else:
            num_workers = local_num_workers
            worker_idx = local_worker_idx

        self._reader = _make_reader(
            self._table,
            num_workers,
            worker_idx,
            yt_num_readers=max(1, self._num_readers // get_local_num_workers()),
            drop_last=drop_last,
            read_from_tail=read_from_tail,
            num_retries=num_retries,
            timeout=timeout,
            table_freshness=table_freshness,
            yt_proxy=yt_proxy,
        )
        schema = get_table_schema(
            table, yt.YtClient(proxy=yt_proxy, token=get_yt_token())
        )
        dist.print0(f"Creating YT Reader for table : {table} with schema: {schema}")
        dist.print0(
            f"DATASET CHUNK #{worker_idx} REPORTING FOR DUTY WITH {self._reader.num_rows} rows."
        )

        if read_from_tail is not None:
            dist.print0(
                "Always set skip_rows to 0 whenever read_from_tail is specified."
            )
            skip_rows = 0

        self.skip_rows = skip_rows % len(self)
        dist.print0(f"Skip {self.skip_rows} rows.")

    def __len__(self):
        if callable(self._reader.num_rows):
            num_rows = self._reader.num_rows()
        else:
            num_rows = self._reader.num_rows
        return num_rows

    def __iter__(self):
        self._reader.reset_to_row(self.skip_rows)
        yield from iterate_with_preproc(
            iter(self._reader),
            fn=self._row_processor,
            num_workers=self._preproc_workers,
            buf_size=self._buf_size,
        )
