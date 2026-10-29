"""オッズ下落率表 q (design §3.2): 判断時 (T-6) のオッズから確定オッズまでの比の下側分位。

券が当たったときにだけ意味がある値なので、当たった馬に限って推定する。
- 単勝: 確定単勝 / T-6 単勝、単独 1 着の馬 (1 着同着のレースは除く)
- 複勝: 確定複勝 Lo / T-6 複勝 Lo、支払対象の馬 (出走して 1 ≤ 着順 ≤ paid_places の支払着数)。複勝の払戻は確定 Lo
  以上なので控えめ側
帯は判断時オッズ (複勝は Lo)。件数の少ない帯は隣と結合し、q ≤ 1 に切る。帯の間で単調にはしない (実測は U 字)。
推定期間 (評価年の直前 3 年) を表のメタデータに書き、評価年より前に終わらない表 (評価年と重なる・評価年より後の年を
含む) は load_q_table が拒否する。q は年々下がる (締切直前の売れ方が強まる) ので、年ごとの q も by_year.json に書く。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from keiba_goal.config import load_config
from keiba_goal.data.paid_places import load_paid_places
from keiba_goal.data.schema import NOT_STARTED_CODES

log = logging.getLogger(__name__)

BAND_EDGES: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DECISION_X = 6
MAX_DECISION_WIN_ODDS = 100.0   # 判断時の単勝オッズが 100 倍以上の馬は単複とも候補外 (design §9.3)
BET_TYPES = ("win", "place")


def hit_samples(records: pl.DataFrame, snap: pl.DataFrame, final: pl.DataFrame, paid: pl.DataFrame) -> pl.DataFrame:
    """当たった券の (bet_type, odds_dec, ratio)。

    records: race_key, horse_num, rank, error_code / snap: ts_snap (x 列を含む) / final: ts_final /
    paid: paid_places (race_key, n_paid_places)。支払着数の無いレースがあれば例外。
    判断時の単勝オッズが無い・100 倍以上の馬は単複とも除く (買い目の候補と同じ母集団にする)。
    """
    n_first = records.group_by("race_key").agg((pl.col("rank") == 1).sum().alias("n_first"))
    started = records.filter(~pl.col("error_code").is_in(sorted(NOT_STARTED_CODES)))
    no_paid = started.join(paid, on="race_key", how="anti")["race_key"].unique()
    if no_paid.len():
        raise ValueError(f"支払着数 (paid_places) の無いレースがあります: {no_paid.head(5).to_list()}")
    d = (
        started.join(n_first, on="race_key")
        .join(paid.select("race_key", "n_paid_places"), on="race_key")
        .join(snap.filter(pl.col("x") == DECISION_X), on=["race_key", "horse_num"], how="inner")
        .join(final, on=["race_key", "horse_num"], how="inner")
        .filter(pl.col("win").is_finite() & (pl.col("win") < MAX_DECISION_WIN_ODDS))
    )
    win = d.filter((pl.col("rank") == 1) & (pl.col("n_first") == 1) & pl.col("win_final").is_finite()).select(
        pl.lit("win").alias("bet_type"),
        pl.col("win").alias("odds_dec"),
        (pl.col("win_final") / pl.col("win")).alias("ratio"),
    )
    place = d.filter(
        (pl.col("rank") >= 1)
        & (pl.col("rank") <= pl.col("n_paid_places"))
        & pl.col("place_lo").is_finite()
        & pl.col("place_lo_final").is_finite()
    ).select(
        pl.lit("place").alias("bet_type"),
        pl.col("place_lo").alias("odds_dec"),
        (pl.col("place_lo_final") / pl.col("place_lo")).alias("ratio"),
    )
    return pl.concat([win, place]).with_columns(pl.col("odds_dec", "ratio").cast(pl.Float64))


def merge_small_bands(counts: Sequence[int], min_n: int) -> list[list[int]]:
    """件数 < min_n の帯を隣の帯と結合する。最も件数の少ない帯から、隣のうち件数の少ない方と結合。"""
    groups = [[i] for i in range(len(counts))]
    n = [int(c) for c in counts]
    while len(groups) > 1 and min(n) < min_n:
        i = min(range(len(n)), key=lambda j: (n[j], -j))
        if i == 0:
            nb = 1
        elif i == len(n) - 1:
            nb = i - 1
        else:
            nb = i - 1 if n[i - 1] <= n[i + 1] else i + 1
        lo, hi = sorted((i, nb))
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
        n[lo:hi + 1] = [n[lo] + n[hi]]
    return groups


@dataclass(frozen=True)
class QBand:
    lo: float
    hi: float
    n: int
    q: float
    q_raw: float


def estimate_bands(
    odds_dec: np.ndarray, ratio: np.ndarray, quantile: float, min_n: int, edges: Sequence[float] = BAND_EDGES
) -> list[QBand]:
    """帯ごとの ratio の分位 (numpy の線形補間)。帯の外 (< 1 または ≥ 100) は使わない。"""
    band = np.searchsorted(np.asarray(edges), odds_dec, side="right") - 1
    ok = (band >= 0) & (band < len(edges) - 1) & np.isfinite(ratio)
    band, ratio = band[ok], ratio[ok]
    if band.size == 0:
        raise ValueError("下落率を推定できる標本がありません")
    groups = merge_small_bands([int((band == b).sum()) for b in range(len(edges) - 1)], min_n)
    out = []
    for g in groups:
        r = ratio[np.isin(band, g)]
        if r.size < min_n:
            log.warning("帯 [%s, %s) の標本が %d 件しかありません (全帯を結合済み)", edges[g[0]], edges[g[-1] + 1], r.size)
        q_raw = float(np.quantile(r, quantile))
        out.append(QBand(float(edges[g[0]]), float(edges[g[-1] + 1]), int(r.size), min(q_raw, 1.0), q_raw))
    return out


@dataclass(frozen=True)
class QTable:
    name: str
    window: tuple[int, int]
    quantile: float
    bands: dict[str, tuple[QBand, ...]]

    def q(self, bet_type: str, odds: float) -> float | None:
        """判断時オッズ odds (複勝は Lo) の帯の q。帯の外 (100 倍以上など) は None。"""
        for b in self.bands[bet_type]:
            if b.lo <= odds < b.hi:
                return b.q
        return None


def estimate_q_table(
    samples: pl.DataFrame, name: str, window: tuple[int, int], quantile: float, min_n: int
) -> QTable:
    bands = {}
    for bt in BET_TYPES:
        s = samples.filter(pl.col("bet_type") == bt)
        bands[bt] = tuple(estimate_bands(s["odds_dec"].to_numpy(), s["ratio"].to_numpy(), quantile, min_n))
    return QTable(name=name, window=window, quantile=quantile, bands=bands)


def q_table_to_json(table: QTable, min_n: int) -> dict:
    return {
        "name": table.name,
        "window": list(table.window),
        "quantile": table.quantile,
        "decision_x": DECISION_X,
        "band_edges": list(BAND_EDGES),
        "min_band_samples": min_n,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "population": {
            "win": f"単独 1 着の馬、確定単勝 / T-{DECISION_X} 単勝、T-{DECISION_X} 単勝 < {MAX_DECISION_WIN_ODDS:g}",
            "place": f"支払対象の馬 (着順 ≤ paid_places の支払着数)、確定複勝 Lo / T-{DECISION_X} 複勝 Lo、"
                     f"T-{DECISION_X} 単勝 < {MAX_DECISION_WIN_ODDS:g}",
        },
        **{bt: [vars(b) for b in table.bands[bt]] for bt in BET_TYPES},
    }


def year_samples(interim: Path, year: int, paid: pl.DataFrame) -> pl.DataFrame:
    """1 年分の ts_snap (x = 6)・ts_final・records・paid_places から当たり券の標本を作る。"""
    snap_path = interim / "ts_snap" / f"{year}.parquet"
    final_path = interim / "ts_final" / f"{year}.parquet"
    if not (snap_path.exists() and final_path.exists()):
        raise FileNotFoundError(f"{year} 年の ts_snap / ts_final がありません (先に --step ts)")
    snap = pl.scan_parquet(snap_path).filter(pl.col("x") == DECISION_X).collect()
    final = pl.read_parquet(final_path)
    records = (
        pl.scan_parquet(interim / "records.parquet")
        .filter((pl.col("date") // 10_000) == year)
        .select("race_key", "horse_num", "rank", "error_code")
        .collect()
    )
    return hit_samples(records, snap, final, paid)


def load_window_samples(interim: Path, window: tuple[int, int], paid: pl.DataFrame) -> pl.DataFrame:
    """推定期間の各年の当たり券の標本。"""
    return pl.concat([year_samples(interim, y, paid) for y in range(window[0], window[1] + 1)])


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def build_q_tables(
    interim: Path, windows: dict[str, Sequence[int]], quantile: float, min_n: int
) -> dict[str, QTable]:
    """windows の各期間で表を推定し、interim/q_tables/{name}.json に書く。支払着数は interim/paid_places.parquet。"""
    out_dir = interim / "q_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    paid = load_paid_places(interim / "paid_places.parquet")
    tables = {}
    for name, (start, end) in windows.items():
        window = (int(start), int(end))
        table = estimate_q_table(load_window_samples(interim, window, paid), name, window, quantile, min_n)
        _write_json(out_dir / f"{name}.json", q_table_to_json(table, min_n))
        tables[name] = table
    return tables


def raw_bands(odds_dec: np.ndarray, ratio: np.ndarray, quantile: float) -> list[dict]:
    """帯を結合しない (lo, hi, n, q, q_raw)。標本の無い帯は q = q_raw = None。"""
    edges = np.asarray(BAND_EDGES)
    band = np.searchsorted(edges, odds_dec, side="right") - 1
    out = []
    for b in range(len(BAND_EDGES) - 1):
        r = ratio[(band == b) & np.isfinite(ratio)]
        q_raw = float(np.quantile(r, quantile)) if r.size else None
        out.append({"lo": BAND_EDGES[b], "hi": BAND_EDGES[b + 1], "n": int(r.size),
                    "q": None if q_raw is None else min(q_raw, 1.0), "q_raw": q_raw})
    return out


def build_q_by_year(interim: Path, years: Iterable[int], quantile: float) -> dict[int, dict[str, list[dict]]]:
    """年ごと・券種ごと・帯ごと (結合しない) の q を interim/q_tables/by_year.json に書く (下落率の推移の確認用)。"""
    paid = load_paid_places(interim / "paid_places.parquet")
    out: dict[int, dict[str, list[dict]]] = {}
    for year in years:
        s = year_samples(interim, year, paid)
        out[int(year)] = {
            bt: raw_bands(s.filter(pl.col("bet_type") == bt)["odds_dec"].to_numpy(),
                          s.filter(pl.col("bet_type") == bt)["ratio"].to_numpy(), quantile)
            for bt in BET_TYPES
        }
    out_dir = interim / "q_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "by_year.json", {
        "quantile": quantile,
        "decision_x": DECISION_X,
        "band_edges": list(BAND_EDGES),
        "note": "帯は結合しない。n の少ない帯の q は不安定。推定期間の表は {backtest,production}.json",
        "years": {str(y): v for y, v in out.items()},
    })
    return out


def load_q_table(name: str, evaluated_years: Iterable[int], q_dir: Path | None = None) -> QTable:
    """q 表を読む。推定期間が評価年のどれかより前に終わっていない表は拒否する。

    評価年と重なる表は評価年の確定オッズを、評価年より後の年を含む表は未来の確定オッズを使っているため。
    evaluated_years が空 (本番・ライブ) なら期間を問わない。
    """
    if q_dir is None:
        q_dir = load_config().paths.interim / "q_tables"
    data = json.loads((q_dir / f"{name}.json").read_text(encoding="utf-8"))
    start, end = (int(y) for y in data["window"])
    not_before = sorted({int(y) for y in evaluated_years if int(y) <= end})
    if not_before:
        raise ValueError(f"q 表 {name} の推定期間 {start}-{end} が評価年 {not_before} より前に終わっていません")
    bands = {
        bt: tuple(QBand(float(b["lo"]), float(b["hi"]), int(b["n"]), float(b["q"]), float(b["q_raw"])) for b in data[bt])
        for bt in BET_TYPES
    }
    for bt, bs in bands.items():
        if not bs or any(not (0 < b.q <= 1) for b in bs):
            raise ValueError(f"q 表 {name} の {bt} の値が不正です")
    return QTable(name=data["name"], window=(start, end), quantile=float(data["quantile"]), bands=bands)
