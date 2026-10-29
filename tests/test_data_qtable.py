import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from keiba_goal.data.qtable import (
    BAND_EDGES,
    build_q_by_year,
    build_q_tables,
    estimate_bands,
    hit_samples,
    load_q_table,
    merge_small_bands,
    raw_bands,
)


def _paid(df: pl.DataFrame) -> pl.DataFrame:
    """出走頭数の規則の支払着数 (paid 列があればその値)。"""
    by_starters = pl.when(pl.col("starters") >= 8).then(3).when(pl.col("starters") >= 5).then(2).otherwise(0)
    paid = pl.col("paid") if "paid" in df.columns else by_starters
    return df.group_by("race_key").agg(paid.first().alias("n_paid_places")).select(
        pl.col("race_key").cast(pl.Int64), pl.col("n_paid_places").cast(pl.Int8)
    )


def _frames(horses: list[dict]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """horses: race_key, horse_num, rank, starters, error_code, win(T-6), place_lo(T-6), win_final, place_lo_final
    (paid = 支払着数を出走頭数の規則と変えるときだけ)。"""
    df = pl.DataFrame(horses)
    records = df.select(
        pl.col("race_key").cast(pl.Int64), pl.col("horse_num").cast(pl.Int8), pl.col("rank").cast(pl.Int16),
        pl.col("error_code").cast(pl.Int8),
    )
    snap = df.select(
        pl.col("race_key").cast(pl.Int64), pl.col("horse_num").cast(pl.Int8), pl.lit(6, dtype=pl.Int16).alias("x"),
        pl.col("win").cast(pl.Float32), pl.col("place_lo").cast(pl.Float32),
    )
    final = df.select(
        pl.col("race_key").cast(pl.Int64), pl.col("horse_num").cast(pl.Int8),
        pl.col("win_final").cast(pl.Float32), pl.col("place_lo_final").cast(pl.Float32),
    )
    return records, snap, final, _paid(df)


def _horse(race_key, horse_num, rank, starters, win=4.0, lo=1.5, win_final=3.6, lo_final=1.4, error_code=0, paid=None):
    h = dict(race_key=race_key, horse_num=horse_num, rank=rank, starters=starters, error_code=error_code,
             win=win, place_lo=lo, win_final=win_final, place_lo_final=lo_final)
    return h if paid is None else {**h, "paid": paid}


def test_merge_small_bands():
    assert merge_small_bands([500, 300, 150, 100, 50, 10], 200) == [[0], [1], [2, 3, 4, 5]]
    assert merge_small_bands([500, 300, 250, 200, 220, 210], 200) == [[0], [1], [2], [3], [4], [5]]
    assert merge_small_bands([10, 500, 500, 500, 500, 500], 200) == [[0, 1], [2], [3], [4], [5]]
    assert merge_small_bands([300, 100, 400, 90, 500, 600], 200) == [[0, 1], [2, 3], [4], [5]]
    assert merge_small_bands([1, 2, 3, 4, 5, 6], 200) == [[0, 1, 2, 3, 4, 5]]


def test_estimate_bands_quantile_merge_and_clip():
    rng = np.random.default_rng(0)
    sizes = [1000, 1000, 300, 300, 300, 50]
    odds = np.concatenate([rng.uniform(lo, hi, n) for lo, hi, n in zip(BAND_EDGES[:-1], BAND_EDGES[1:], sizes)] + [[0.5, 150.0]])
    ratio = np.concatenate([
        rng.uniform(0.8, 1.2, 1000), rng.uniform(1.05, 1.3, 1000), rng.uniform(0.7, 0.9, 900),
        rng.uniform(0.6, 0.9, 50), [0.1, 0.1],
    ])
    bands = estimate_bands(odds, ratio, 0.10, 200)
    assert [(b.lo, b.hi) for b in bands] == [(1, 3), (3, 5), (5, 10), (10, 20), (20, 100)]   # [50,100) は 50 件 → 隣と結合
    assert bands[0].q == pytest.approx(np.quantile(ratio[:1000], 0.10)) and bands[0].n == 1000
    assert bands[1].q == 1.0 and bands[1].q_raw > 1.0           # q ≤ 1 に切る
    assert bands[-1].n == 350
    assert sum(b.n for b in bands) == 2950                       # 帯の外 (0.5, 150) は使わない


def test_estimate_bands_does_not_force_monotonicity():
    odds = np.concatenate([np.full(300, 2.0), np.full(300, 7.0), np.full(300, 70.0)])
    ratio = np.concatenate([np.full(300, 0.9), np.full(300, 0.7), np.full(300, 0.95)])
    q = [b.q for b in estimate_bands(odds, ratio, 0.10, 200)]
    assert q == pytest.approx([0.9, 0.7, 0.95])   # U 字のまま


def test_estimate_bands_without_samples_raises():
    with pytest.raises(ValueError):
        estimate_bands(np.array([150.0]), np.array([0.9]), 0.10, 200)


def test_hit_samples_rules():
    horses = [
        # A: 10 頭立て、単独 1 着 → 単勝 1 件、3 着内 → 複勝 3 件
        _horse(1, 1, 1, 10, win=4.0, win_final=3.0), _horse(1, 2, 2, 10), _horse(1, 3, 3, 10), _horse(1, 4, 4, 10),
        _horse(1, 5, 0, 10, error_code=1),
        # B: 1 着同着 → 単勝は除く、複勝は 1,1,3 着の 3 件 (8 頭立て)
        _horse(2, 1, 1, 8), _horse(2, 2, 1, 8), _horse(2, 3, 3, 8), _horse(2, 4, 4, 8),
        # C: 6 頭立て → 複勝は 2 着内だけ
        _horse(3, 1, 1, 6), _horse(3, 2, 2, 6), _horse(3, 3, 3, 6),
        # D: 4 頭立て → 複勝なし (単勝は単独 1 着)
        _horse(4, 1, 1, 4), _horse(4, 2, 2, 4),
        # E: 判断時の単勝 100 倍以上は単複とも除く・判断時オッズ欠損も除く
        _horse(5, 1, 1, 12, win=120.0, lo=12.0), _horse(5, 2, 2, 12, win=float("nan")), _horse(5, 3, 3, 12),
    ]
    s = hit_samples(*_frames(horses))
    win = s.filter(pl.col("bet_type") == "win")
    place = s.filter(pl.col("bet_type") == "place")
    assert win.height == 3                                    # A, C, D (B は同着、E は 100 倍以上)
    assert sorted(win["ratio"].to_list()) == pytest.approx([0.75, 0.9, 0.9])   # A は 3.0 / 4.0
    assert place.height == 3 + 3 + 2 + 0 + 1                  # A, B, C, D, E (E は 3 着の 1 頭だけ)
    assert place["ratio"].to_list() == pytest.approx([1.4 / 1.5] * 9)


def test_place_hits_follow_paid_places_not_starters():
    # 8 頭登録・発売開始後の取消で 7 頭出走 (202309040202 型): 支払着数 3 なので 3 着も複勝の標本
    horses = [_horse(6, h, h, 7, paid=3) for h in range(1, 8)] + [_horse(6, 8, 0, 7, error_code=1, paid=3)]
    s = hit_samples(*_frames(horses))
    assert s.filter(pl.col("bet_type") == "place").height == 3


def test_races_without_paid_places_raise():
    records, snap, final, paid = _frames([_horse(1, 1, 1, 10), _horse(2, 1, 1, 10)])
    with pytest.raises(ValueError, match="paid_places"):
        hit_samples(records, snap, final, paid.filter(pl.col("race_key") == 1))


def _write_window(interim: Path, years: list[int], n_races: int = 40) -> None:
    rng = np.random.default_rng(1)
    recs, paids = [], []
    for year in years:
        horses = []
        for r in range(n_races):
            rk = int(f"{year}0105060101{r % 12 + 1:02d}") + r // 12 * 10**4
            for h in range(1, 11):
                win = float(rng.choice([2.0, 4.0, 8.0, 15.0, 30.0, 70.0]))
                horses.append(_horse(rk, h, h, 10, win=win, lo=win / 3 + 1, win_final=win * 0.9, lo_final=(win / 3 + 1) * 0.95))
        records, snap, final, paid = _frames(horses)
        (interim / "ts_snap").mkdir(parents=True, exist_ok=True)
        (interim / "ts_final").mkdir(parents=True, exist_ok=True)
        snap.write_parquet(interim / "ts_snap" / f"{year}.parquet")
        final.write_parquet(interim / "ts_final" / f"{year}.parquet")
        recs.append(records.with_columns((pl.col("race_key") // 10**8).cast(pl.Int32).alias("date")))
        paids.append(paid)
    pl.concat(recs).write_parquet(interim / "records.parquet")
    pl.concat(paids).write_parquet(interim / "paid_places.parquet")


def test_build_and_load_q_tables(tmp_path: Path):
    _write_window(tmp_path, [2015, 2016, 2017])
    tables = build_q_tables(tmp_path, {"bt": [2015, 2016], "prod": [2015, 2017]}, 0.10, 20)
    assert set(tables) == {"bt", "prod"}
    data = json.loads((tmp_path / "q_tables" / "bt.json").read_text(encoding="utf-8"))
    assert data["window"] == [2015, 2016] and data["quantile"] == 0.10 and data["decision_x"] == 6
    assert data["band_edges"] == list(BAND_EDGES)
    datetime.fromisoformat(data["created_at"])
    assert all({"lo", "hi", "n", "q", "q_raw"} <= set(b) for b in data["win"] + data["place"])
    assert sum(b["n"] for b in data["win"]) == 2 * 40          # 各レースの単独 1 着
    assert sum(b["n"] for b in data["place"]) == 2 * 40 * 3

    t = load_q_table("bt", [2023], q_dir=tmp_path / "q_tables")
    assert t.window == (2015, 2016)
    assert t.q("win", 4.0) == pytest.approx(0.9)
    assert t.q("place", 2.0) == pytest.approx(0.95)
    assert t.q("win", 150.0) is None and t.q("win", 0.9) is None


@pytest.mark.parametrize("name, years, ok", [
    ("bt", [2017], True), ("bt", [2016, 2017], False), ("bt", [2015], False),
    ("bt", [2014], False), ("bt", [2014, 2023], False),          # 評価年より後の年 (未来) で推定した表も拒否
    ("prod", [2023, 2024], True), ("prod", [2017], False), ("prod", [], True),
])
def test_load_refuses_tables_not_ending_before_evaluated_years(tmp_path: Path, name: str, years: list[int], ok: bool):
    _write_window(tmp_path, [2015, 2016, 2017], n_races=12)
    build_q_tables(tmp_path, {"bt": [2015, 2016], "prod": [2015, 2017]}, 0.10, 5)
    if ok:
        load_q_table(name, years, q_dir=tmp_path / "q_tables")
    else:
        with pytest.raises(ValueError, match="評価年"):
            load_q_table(name, years, q_dir=tmp_path / "q_tables")


def test_missing_year_in_window_raises(tmp_path: Path):
    _write_window(tmp_path, [2015], n_races=12)
    with pytest.raises(FileNotFoundError, match="2016"):
        build_q_tables(tmp_path, {"bt": [2015, 2016]}, 0.10, 5)


def test_q_by_year_keeps_every_band_unmerged(tmp_path: Path):
    _write_window(tmp_path, [2015, 2016])
    out = build_q_by_year(tmp_path, [2015, 2016], 0.10)
    data = json.loads((tmp_path / "q_tables" / "by_year.json").read_text(encoding="utf-8"))
    assert set(data["years"]) == {"2015", "2016"} and data["quantile"] == 0.10
    bands = out[2015]["win"]
    assert [(b["lo"], b["hi"]) for b in bands] == list(zip(BAND_EDGES[:-1], BAND_EDGES[1:]))
    assert sum(b["n"] for b in bands) == 40 and sum(b["n"] for b in out[2015]["place"]) == 40 * 3
    assert all(b["n"] > 0 and b["q"] == pytest.approx(0.9) for b in bands)   # n < 200 の帯も結合しない
    empty = raw_bands(np.array([2.0, 150.0]), np.array([0.9, 0.5]), 0.10)
    assert [b["n"] for b in empty] == [1, 0, 0, 0, 0, 0] and empty[1]["q"] is None and empty[1]["q_raw"] is None


def test_missing_paid_places_table_raises(tmp_path: Path):
    _write_window(tmp_path, [2015], n_races=12)
    (tmp_path / "paid_places.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="paid_places"):
        build_q_tables(tmp_path, {"bt": [2015, 2015]}, 0.10, 5)
