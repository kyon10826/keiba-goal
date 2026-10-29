"""(場, track_code, 距離) → コーナー回数 の表 (record_data の全期間から)。障害は corner_num が欠損。

1 つのキーに 2 つ以上の値があれば作らずに止める (design §3.1: 矛盾 0)。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from keiba_goal.netkeiba.files import save_parquet
from keiba_goal.netkeiba.records import record_frames

KEY = ["venue_code", "track_code", "dist"]
SCHEMA: dict[str, pl.DataType] = {
    "venue_code": pl.Int8, "place": pl.Utf8, "track_code": pl.Int16, "dist": pl.Int16, "corner_num": pl.Int8,
    "first_year": pl.Int16, "last_year": pl.Int16, "n_races": pl.Int32,
}


class CourseConflict(ValueError):
    pass


def build_course_table(record_dir: Path) -> pl.DataFrame:
    parts = []
    for df in record_frames(record_dir):
        races = df.select("race_key", "venue_code", "place", "track_code", "corner_num", "dist",
                          year=(pl.col("date") // 10000).cast(pl.Int16)).unique("race_key")
        parts.append(races.group_by("venue_code", "place", "track_code", "corner_num", "dist", "year").agg(
            n_races=pl.len()))
    table = (pl.concat(parts)
             .group_by(KEY).agg(
                 place=pl.col("place").first(),
                 corner_nums=pl.col("corner_num").unique(),
                 first_year=pl.col("year").min(),
                 last_year=pl.col("year").max(),
                 n_races=pl.col("n_races").sum().cast(pl.Int32)))
    conflicts = table.filter(pl.col("corner_nums").list.len() > 1)
    if conflicts.height:
        raise CourseConflict(f"コーナー回数が一意でないコース {conflicts.height} 件: {conflicts.head(5).to_dicts()}")
    table = table.with_columns(corner_num=pl.col("corner_nums").list.first()).drop("corner_nums")
    return table.select(list(SCHEMA)).cast(SCHEMA).sort(KEY)


def load_or_build(path: Path, record_dir: Path) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    table = build_course_table(record_dir)
    save_parquet(table, path)
    return table


def lookup(table: pl.DataFrame, venue_code: int, track_code: int, dist: int) -> dict | None:
    hit = table.filter((pl.col("venue_code") == venue_code) & (pl.col("track_code") == track_code)
                       & (pl.col("dist") == dist))
    return hit.row(0, named=True) if hit.height else None


def track_codes(table: pl.DataFrame, venue_code: int, dist: int, codes: set[int]) -> list[int]:
    """その場・距離で使われたことのある track_code のうち codes に含まれるもの (新しい順)。"""
    hit = table.filter((pl.col("venue_code") == venue_code) & (pl.col("dist") == dist)
                       & pl.col("track_code").is_in(sorted(codes)))
    return hit.sort("last_year", descending=True)["track_code"].to_list()


if __name__ == "__main__":
    from keiba_goal.config import load_config

    cfg = load_config()
    out = cfg.paths.interim / "course_table.parquet"
    t = build_course_table(cfg.paths.record_dir)
    save_parquet(t, out)
    print(f"{out}: {t.height} コース (平地 {t.filter(pl.col('track_code') < 50).height})、矛盾 0")
