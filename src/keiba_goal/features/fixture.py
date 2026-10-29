"""テスト用の小さな履歴 fixture (tests/fixtures/features/) を data/interim から作り直す。

    .venv/bin/python -m keiba_goal.features.fixture [--config path]

中身: 2022-01-01〜04-30 の中山・中京・阪神の全レースの全行 + それらに出た馬の 04-30 までの全行 (他場・過去年。
前走が欠けないように) と、2022 年の行のレースの ts_snap 行 (ts_snap/2022.parquet、列はデータ層のまま)。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from keiba_goal.config import REPO_ROOT, load_config
from keiba_goal.features.io import read_parquet_retry, write_parquet

VENUES: tuple[int, ...] = (6, 7, 9)
START, END = 20220101, 20220430
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "features"


def build_fixture(history_path: Path, ts_snap_dir: Path, out_dir: Path) -> tuple[int, int]:
    hist = pl.scan_parquet(history_path)
    period = pl.col("date").is_between(START, END)
    horses = hist.filter(period & pl.col("venue_code").is_in(VENUES)).select("id").unique()
    rows = hist.filter(pl.col("date") <= END).join(horses, on="id", how="semi")
    races = hist.filter(period & pl.col("venue_code").is_in(VENUES)).select("race_key").unique()
    fixture = (
        pl.concat([rows, hist.join(races, on="race_key", how="semi")])
        .unique("race_id")
        .sort(["date", "race_key", "horse_num"])
        .collect()
    )
    keys = fixture.filter((pl.col("date") // 10_000) == 2022).select("race_key").unique()
    snaps = read_parquet_retry(ts_snap_dir / "2022.parquet").join(keys, on="race_key", how="semi")
    write_parquet(fixture, out_dir / "history.parquet")
    write_parquet(snaps.sort(["race_key", "x", "horse_num"]), out_dir / "ts_snap" / "2022.parquet")
    return fixture.height, snaps.height


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keiba_goal.features.fixture", description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    n_rows, n_snaps = build_fixture(cfg.paths.interim / "history.parquet", cfg.paths.interim / "ts_snap", FIXTURE_DIR)
    print(f"[fixture] history {n_rows:,} 行, ts_snap {n_snaps:,} 行 → {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
