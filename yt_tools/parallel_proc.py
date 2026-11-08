from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque

# from typing import Iterable, Iterator, Optional, TypeVar, Callable


# Tinp = TypeVar('Tinp')
# Tout = TypeVar('Tout')


def iterate_with_preproc(data, fn, num_workers, buf_size):
    if buf_size is None:
        buf_size = num_workers * 2
    assert buf_size > 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        data = iter(data)
        buf = deque([], buf_size)

        for x in data:
            buf.append(pool.submit(fn, x))
            if len(buf) == buf_size:
                yield buf.popleft().result()

        while buf:
            yield buf.popleft().result()
