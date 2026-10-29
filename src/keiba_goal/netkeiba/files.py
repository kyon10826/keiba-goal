"""netkeiba 層の書き込み: 一意な一時ファイルに書いて rename (中断で壊れたファイルを残さない) と、作り直しのロック。

作り直し (build)・馬マスタの更新・combine は data/raw/build.lock の fcntl ロックで直列化する。窓 1〜3 を別々の
プロセスで動かしても、card_horses.parquet や まとめたファイルを古い内容で上書きしないため。
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

BUILD_LOCK = "build.lock"
_held: dict[Path, int] = {}   # このプロセスが持っているロック (入れ子の取得は数えるだけ)


def _tmp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return Path(tmp)


def save_bytes(path: Path, data: bytes) -> None:
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = _tmp_path(path)
    try:
        df.write_parquet(tmp, compression="zstd")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_json(obj: object, path: Path) -> None:
    save_bytes(path, json.dumps(obj, ensure_ascii=False, indent=1, default=str).encode("utf-8"))


@contextmanager
def build_lock(raw_dir: Path) -> Iterator[None]:
    """作り直しの排他ロック。ほかのプロセスが持っていれば待つ (待つ旨を表示)。同じプロセスの入れ子は素通り。"""
    path = (raw_dir / BUILD_LOCK).resolve()
    if _held.get(path):
        _held[path] += 1
        try:
            yield
        finally:
            _held[path] -= 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ほかのプロセスが作り直し中 ({path})。終わるまで待つ", file=sys.stderr, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        _held[path] = 1
        try:
            yield
        finally:
            _held.pop(path, None)
            fcntl.flock(lock, fcntl.LOCK_UN)
