"""履歴表 history.parquet = records.parquet + netkeiba から変換した行 (nk_rows.parquet, 無ければ省略)。

同じ race_id が両方にあれば record の行を採る。並びは (date, race_key, horse_num)、型は schema.HISTORY_DTYPES。
netkeiba の行は record の行と同じ規則で検査する (check_nk_rows)。年ごとに読み書きするので、全体をメモリに載せない。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from keiba_goal.data.record import SORT_KEYS, add_derived, check_race_id, write_parquet_parts
from keiba_goal.data.schema import DERIVED_DTYPES, HISTORY_DTYPES, RECORD_COLUMNS

log = logging.getLogger(__name__)

NK_SOURCE = "netkeiba"
# race_id と 47 列から決まる派生列 (trainer_id と source は netkeiba の行だけが持つ値)
RECOMPUTED_COLUMNS: tuple[str, ...] = tuple(c for c in DERIVED_DTYPES if c not in ("trainer_id", "source"))


def conform(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """HISTORY_DTYPES の列だけを、その順・型で。列が欠けていれば例外。"""
    missing = [c for c in HISTORY_DTYPES if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: history の列が足りません {missing}")
    return df.select(pl.col(c).cast(t, strict=True) for c, t in HISTORY_DTYPES.items())


def check_nk_rows(nk: pl.DataFrame, name: str) -> None:
    """netkeiba 由来の行を検査する。どれか破れていれば例外。

    source がすべて 'netkeiba'、race_id が一意でその分解が各列と一致 (record.check_race_id)、
    派生列 (date, race_key, nk_race_id, venue_code, 芝ダ等, class_ord, starters) が record.add_derived で
    47 列から作り直した値と一致すること (1 レースの行がそろっていないと starters が合わない)。
    """
    bad_source = nk.filter(pl.col("source").ne_missing(NK_SOURCE))["source"].unique().to_list()
    if bad_source:
        raise ValueError(f"{name}: source が '{NK_SOURCE}' でない行があります {bad_source}")
    check_race_id(nk, name)
    expected = add_derived(nk.select(RECORD_COLUMNS))
    bad = {c: n for c in RECOMPUTED_COLUMNS if (n := int((~expected[c].eq_missing(nk[c])).sum()))}
    if bad:
        raise ValueError(f"{name}: 派生列が race_id と 47 列から作った値と一致しない行があります {bad}")


@dataclass(frozen=True)
class HistorySummary:
    rows: int
    record_rows: int
    nk_rows_added: int
    nk_rows_replaced_by_record: int
    nk_rows_in_record_races: int
    min_date: int
    max_date: int


def _years(lf: pl.LazyFrame) -> set[int]:
    return set(lf.select((pl.col("date") // 10_000).unique()).collect().to_series().to_list())


def build_history(records_path: Path, nk_path: Path | None, out_path: Path) -> HistorySummary:
    rec = pl.scan_parquet(records_path)
    nk = pl.scan_parquet(nk_path) if nk_path is not None and nk_path.exists() else None
    years = _years(rec) | (_years(nk) if nk is not None else set())
    counts = {"record": 0, "added": 0, "replaced": 0, "mixed": 0}
    dates: list[int] = []

    def parts() -> Iterator[pl.DataFrame]:
        for year in sorted(years):
            in_year = (pl.col("date") // 10_000) == year
            part = conform(rec.filter(in_year).collect(), records_path.name)
            counts["record"] += part.height
            if nk is not None:
                extra = conform(nk.filter(in_year).collect(), nk_path.name)
                check_nk_rows(extra, nk_path.name)
                kept = extra.join(part.select("race_id"), on="race_id", how="anti")
                counts["replaced"] += extra.height - kept.height
                counts["mixed"] += kept.join(part.select("race_key").unique(), on="race_key", how="semi").height
                counts["added"] += kept.height
                part = pl.concat([part, kept])
            part = part.sort(SORT_KEYS)
            if part.height:
                dates.extend((int(part["date"][0]), int(part["date"][-1])))
            yield part

    rows = write_parquet_parts(parts(), out_path)
    if counts["mixed"]:
        log.warning("record にあるレースへ netkeiba の行が %d 行追加されました (馬番の不一致の疑い)", counts["mixed"])
    return HistorySummary(
        rows=rows,
        record_rows=counts["record"],
        nk_rows_added=counts["added"],
        nk_rows_replaced_by_record=counts["replaced"],
        nk_rows_in_record_races=counts["mixed"],
        min_date=min(dates),
        max_date=max(dates),
    )
