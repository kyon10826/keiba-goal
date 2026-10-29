"""特徴量ストアのファイル入出力 (原子的な書き込み・メタデータ・読み込みの再試行・履歴の固定)。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

META_KEY = b"keiba_goal"
RETRIES = 5
RETRY_SEC = 3.0


def write_parquet(df: pl.DataFrame, path: Path, meta: dict[str, Any] | None = None) -> None:
    """一時ファイルに書いてから置き換える。meta は parquet のスキーマのメタデータ (META_KEY) に JSON で入れる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df.to_arrow()
    if meta is not None:
        table = table.replace_schema_metadata({META_KEY: json.dumps(meta, ensure_ascii=False).encode()})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def read_meta(path: Path) -> dict[str, Any] | None:
    """write_parquet の meta。ファイルもメタデータも無ければ None。"""
    if not path.exists():
        return None
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    return json.loads(raw) if raw else None


def read_parquet_retry(path: Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """データ層が同じファイルを置き換えている最中に読んでも失敗しないよう、数回やり直す。"""
    for attempt in range(RETRIES):
        try:
            return pl.read_parquet(path, columns=list(columns) if columns is not None else None)
        except FileNotFoundError:
            raise
        except (OSError, pl.exceptions.PolarsError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_SEC)
    raise AssertionError("unreachable")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin_file(src: Path, dst: Path) -> Path:
    """src をハードリンク (できなければコピー) で dst に固定する。

    データ層は一時ファイル + rename で置き換えるので、リンク先の中身は 1 回の作成のあいだ変わらない。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and os.path.samefile(src, dst):
        return dst
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    # 同じ inode へのリンク同士の rename は何もしない (POSIX) ので、tmp が残ることがある
    tmp.unlink(missing_ok=True)
    return dst
