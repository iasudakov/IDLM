import time
import threading
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from queue import Queue
import yt_tools.dist_util as dist
from .yt_utils import make_client
from typing import List, Dict, Any, Callable, Union
from .encoding.encoder import Encoder


def total_size(o, handlers: dict = dict(), verbose: bool = False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


class ConfigurableYTLogger:
    def __init__(
        self,
        table: str,
        preprocess: Union[Callable[[Dict[str, Any]], List[Dict[str, bytes]]], List],
        interval: int = 100,
        yt_proxy: str = "hahn",
        run_upload_in_background: bool = True,
        queue_max_size: int = 1000,
        sleep_after_write_sec: float = 1,
        sleep_on_empty_buf: float = 0.5,
    ):
        self.yt = make_client(yt_proxy=yt_proxy)
        self.table = self.make_table(table)
        self._buf_rows = Queue(maxsize=queue_max_size)
        if callable(preprocess):
            self.process_batch = preprocess
        else:
            self.process_batch = Encoder(preprocesses=preprocess)
        self.interval = interval
        self.run_upload_in_background = run_upload_in_background
        self.sleep_on_empty_buf = sleep_on_empty_buf
        self.sleep_after_write_sec = sleep_after_write_sec
        if self.run_upload_in_background:
            self.thread = threading.Thread(target=self.worker, daemon=True)
            self.thread.start()

    def make_table(self, table: str) -> str:
        table = self.yt.TablePath(table, append=True)
        if dist.get_rank() == 0:
            self.yt.create_table(table, recursive=True, ignore_existing=True)
        dist.barrier()
        return table

    def _get_top_batch_and_process(self) -> List[Dict[str, bytes]]:
        rows = self._buf_rows.get()
        return self.process_batch(rows)

    def _write_rows_and_log(self, rows: List[Dict[str, bytes]]):
        dist.print0(
            f"Uploading {len(rows)} rows | {total_size(rows)/(1024*1024):.2f} MB"
        )
        self.yt.write_table(
            self.table, rows, table_writer=dict(max_row_weight=(128 * 1024 * 1024))
        )

    def write(self, row: Dict[str, List[Any]]):
        self._buf_rows.put(row)
        if not self.run_upload_in_background:
            rows = self._get_top_batch_and_process()
            self._write_rows_and_log(rows)
            self._buf_rows.task_done()

    def worker(self):
        dist.print0("Starting upload thread.")
        start = time.time()
        total_uploaded = 0
        counter = 0
        self.buf = list()

        while True:
            if not self._buf_rows.empty():
                rows = self._get_top_batch_and_process()
                self.buf.extend(rows)
                if counter % self.interval == 0:
                    self._write_rows_and_log(self.buf)
                    total_uploaded += len(self.buf)
                    self.buf = list()
                    dist.print0(
                        f"Uploading {(total_uploaded * dist.num_workers())/(time.time() - start):.2f} / s | {self._buf_rows.qsize() * len(rows)} in buf"
                    )
                    time.sleep(self.sleep_after_write_sec)

                self._buf_rows.task_done()
                counter += 1
            else:
                time.sleep(self.sleep_on_empty_buf)

    def finalize(self):
        if self.run_upload_in_background:
            self._buf_rows.join()
            self.dump_remaining()

    def dump_remaining(self):
        self.yt.write_table(
            self.table,
            self.buf,
            table_writer=dict(max_row_weight=(128 * 1024 * 1024)),
        )

    def __del__(self):
        if self.run_upload_in_background:
            self.finalize()
