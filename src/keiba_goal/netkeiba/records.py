"""record_data の行 (コース表・馬マスタの初期値・ゲートの比較・窓 2/3 の race_id 用)。

読み込みは P1 の keiba_goal.data.record をそのまま使い、history.parquet と同じ表現 (HISTORY_DTYPES) にする。
自前の CSV 読み込みは持たない: 引用符やゼロ埋めの扱いが history とずれると、馬マスタの父・母やゲートの比較が
history の値と食い違うため。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl

from keiba_goal.data.record import load_record_file, select_record_files
from keiba_goal.data.schema import HISTORY_DTYPES

_YEAR = re.compile(r"^record_data_(\d{4})")


def record_frames(record_dir: Path, first_year: int | None = None, last_year: int | None = None
                  ) -> Iterator[pl.DataFrame]:
    """読むべきファイル (年次 + 最新スナップショット) ごとに HISTORY_DTYPES の表。範囲外の年のファイルは読まない。"""
    for path in select_record_files(record_dir):
        m = _YEAR.match(path.name)
        if m is None:
            raise ValueError(f"record_data のファイル名から年が分からない: {path.name}")
        year = int(m.group(1))
        if (first_year is None or year >= first_year) and (last_year is None or year <= last_year):
            yield load_record_file(path)[0]


def read_record_range(record_dir: Path, start: int, end: int, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """日付 (YYYYMMDD) が [start, end] の行。columns を省くと history の全列。"""
    cols = list(dict.fromkeys(["race_id", *(columns or HISTORY_DTYPES)]))
    parts = [df.filter(pl.col("date").is_between(start, end)).select(cols)
             for df in record_frames(record_dir, start // 10000, end // 10000)]
    if not parts:
        return pl.DataFrame(schema={c: HISTORY_DTYPES[c] for c in cols})
    return pl.concat(parts)
