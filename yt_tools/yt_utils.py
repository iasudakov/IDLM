import os

import yt.wrapper as yt
from datetime import datetime, timedelta


def get_yt_token():
    return os.environ.get("YT_TOKEN", None)


def get_modification_time(path, client):
    modification_time = yt.get(path=path + "/@modification_time", client=client)
    modification_time = datetime.strptime(modification_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    return modification_time


def get_current_time(client):
    with yt.TempTable(client=client) as tmp_table:
        current_time = yt.get(path=tmp_table + "/@modification_time", client=client)
        current_time = datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    return current_time


def make_client(
    yt_proxy: str = "hahn", max_thread_count: int = 4, enable: bool = True
) -> yt.YtClient:
    config = {
        "read_retries": {"enable": enable},
        "allow_receive_token_by_current_ssh_session": True,
        "table_writer": {"desired_chunk_size": 1024 * 1024 * 500},
        "concatenate_retries": {
            "enable": enable,
            "total_timeout": timedelta(minutes=128),
        },
        "write_retries": {"enable": enable, "count": 30},
    }
    if max_thread_count > 1:
        config["read_parallel"] = {
            "enable": True,
            "max_thread_count": max_thread_count,
        }
        config["write_parallel"] = {
            "enable": True,
            "max_thread_count": max_thread_count,
            "unordered": True,
        }
    return yt.YtClient(proxy=yt_proxy, config=config, token=get_yt_token())


def get_table_schema(path, client):
    schema = yt.get(path=path + "/@schema", client=client)
    return schema
