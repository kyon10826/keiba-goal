"""as-of 検査 (design §12-4): 日付 D の特徴量が「D より後の行」「D の行のレース後の列」と、段階 A では「D の当日
変わりうる列」に依らないこと。

check_day は D の行の特徴量を 4 通りの履歴で作って比べる (NaN 同士は一致、許容誤差 1e-6):
  (a) そのまま (D の年の全行を作る batch と同じ)
  (b) date > D の行を消し、D の行のレース後の列 (schema.POST_RACE_COLUMNS) を null にした履歴 (D の行は残す)
  (c) date ≥ D の行のレース後の列を、それらの行の間で列ごとに乱数で並べ替えた履歴
  (d) D の偶数番のレースの行の当日変わりうる列 (registry.STAGE_B_COLUMNS)・scratched・スナップショットのオッズを
      別の値にした履歴: 段階 A の特徴量は D の全行で、段階 B も含む全特徴量は奇数番のレースの行で (a) と一致すること
      (段階 A は朝に作ってキャッシュする。段階 B はレースごとに判断時刻に作るので、他のレースの当日の値を使わない)
(b)(c)(d) は D の行だけを作る (ライブと同じ)。判断時刻の取消の印 scratched は加工より前に履歴から作る
(ライブでは出馬表から渡す値で、レース後の error_code ではない)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl

from keiba_goal.data.schema import POST_RACE_COLUMNS
from keiba_goal.features.decision import with_live_columns
from keiba_goal.features.exprs import GOING, WEATHER
from keiba_goal.features.pipeline import build_store, checked_groups
from keiba_goal.features.registry import OUTPUT_KEYS, SNAPSHOT_VALUES, STAGE_B_COLUMNS, Group

ATOL = 1e-6


def _rotate(labels: Sequence[str]) -> dict[str, str]:
    return {a: b for a, b in zip(labels, [*labels[1:], labels[0]])}


def _scrambled(seed: int) -> dict[str, pl.Expr]:
    """STAGE_B_COLUMNS の各列と scratched を必ず (か、ほぼ必ず) 別の値にする式。"""
    return {
        "jockey_id": pl.col("jockey_id").shuffle(seed=seed),
        "basis_weight": pl.col("basis_weight") + 1.5,
        "weight": pl.col("weight") + 6.0,
        "inc_dec": pl.col("inc_dec") + 4.0,
        "blinker": pl.when(pl.col("blinker") == "B").then(None).otherwise(pl.lit("B")),
        "state": pl.col("state").replace(_rotate(list(GOING))),
        "weather": pl.col("weather").replace(_rotate(list(WEATHER))),
        "horse_N": pl.col("horse_N") + 1,
        "scratched": ~pl.col("scratched"),
    }


def scramble_same_day(rows: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """rows (scratched 付き) の当日変わりうる列と scratched を別の値にする (型はそのまま)。"""
    exprs = _scrambled(seed)
    if missing := sorted(set(STAGE_B_COLUMNS) - set(exprs)):
        raise NotImplementedError(f"STAGE_B_COLUMNS の {missing} の変え方が無い (asof._scrambled に足す)")
    return rows.with_columns(e.cast(rows.schema[c]).alias(c) for c, e in exprs.items())


def even_races(d: int) -> pl.Expr:
    return (pl.col("date") == d) & (pl.col("race_key") % 100 % 2 == 0)


def scramble_from(history: pl.DataFrame, snapshots: pl.DataFrame | None, d: int, seed: int = 0
                  ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """(d) の履歴とスナップショット: D の偶数番のレースの当日の列・scratched とオッズ (×1.25) を変える。"""
    hit = even_races(d)
    out = pl.concat([history.filter(~hit), scramble_same_day(history.filter(hit), seed)])
    out = out.sort(["date", "race_key", "horse_num"])
    if snapshots is None:
        return out, None
    keys = history.filter(hit).select("race_key").unique()
    in_hit = pl.col("race_key").is_in(keys["race_key"].implode())
    snaps = snapshots.with_columns(
        pl.when(in_hit).then(pl.col(v) * 1.25).otherwise(pl.col(v)).cast(snapshots.schema[v]).alias(v)
        for v in SNAPSHOT_VALUES
    )
    return out, snaps


def post_race_columns(history: pl.DataFrame) -> list[str]:
    return sorted(c for c in POST_RACE_COLUMNS if c in history.columns)


def truncate_at(history: pl.DataFrame, d: int) -> pl.DataFrame:
    """date > d の行を消し、date == d の行のレース後の列を null にする。"""
    kept = history.filter(pl.col("date") <= d)
    return kept.with_columns(pl.when(pl.col("date") < d).then(pl.col(c)).alias(c) for c in post_race_columns(history))


def perturb_from(history: pl.DataFrame, d: int, seed: int = 0) -> pl.DataFrame:
    """date ≥ d の行のレース後の列を、それらの行の間で列ごとに並べ替える (型と値の範囲は保つ)。"""
    future = history.filter(pl.col("date") >= d)
    past = history.filter(pl.col("date") < d)
    shuffled = future.with_columns(
        future[c].sample(fraction=1.0, shuffle=True, seed=seed + i).alias(c)
        for i, c in enumerate(post_race_columns(history))
    )
    return pl.concat([past, shuffled]).sort(["date", "race_key", "horse_num"])


def diff_counts(a: pl.DataFrame, b: pl.DataFrame, columns: Sequence[str], atol: float = ATOL) -> dict[str, int]:
    """列ごとの一致しない行数 (片方にしか無い行は全列で不一致)。NaN 同士は一致。"""
    j = a.select("race_id", *columns).join(b.select("race_id", *columns), on="race_id", how="full",
                                           suffix="_b", coalesce=True)
    out = {}
    for c in columns:
        x, y = pl.col(c), pl.col(f"{c}_b")
        same = (x.is_nan() & y.is_nan()) | ((x - y).abs() <= atol)
        out[c] = int(j.select((~same.fill_null(False)).sum()).item())
    return out


@dataclass
class AsofResult:
    date: int
    rows: int
    truncated: dict[str, int] = field(default_factory=dict)
    perturbed: dict[str, int] = field(default_factory=dict)
    same_day: dict[str, int] = field(default_factory=dict)

    def failures(self, features: Sequence[str] | None = None) -> dict[str, dict[str, int]]:
        keep = set(features) if features is not None else None
        pick = lambda d: {k: v for k, v in d.items() if v and (keep is None or k in keep)}  # noqa: E731
        checks = (("truncated", self.truncated), ("perturbed", self.perturbed), ("same_day", self.same_day))
        return {name: bad for name, d in checks if (bad := pick(d))}


def check_day(
    history: pl.DataFrame,
    snapshots: pl.DataFrame | None,
    d: int,
    *,
    field_open: pl.DataFrame | None = None,
    groups: Sequence[Group] | None = None,
    seed: int = 0,
    full: pl.DataFrame | None = None,
) -> AsofResult:
    """D の行の特徴量を (a)〜(d) で作って比べる。full は (a) を作り済みなら渡す (D の年の build_store の出力)。"""
    groups = checked_groups(groups)
    hist = with_live_columns(history.lazy()).collect()
    year = d // 10_000
    kw = {"field_open": field_open, "groups": groups}
    if full is None:
        full = build_store(hist, snapshots, years=(year,), **kw)
    day = full.filter(pl.col("date") == d)
    if day.height == 0:
        raise ValueError(f"{d} の行がありません")
    features = [c for c in day.columns if c not in OUTPUT_KEYS]
    stage_a = {s.name for g in groups if g.stage == "A" for s in g.specs}
    truncated = build_store(truncate_at(hist, d), snapshots, years=(year,), dates=(d,), **kw)
    perturbed = build_store(perturb_from(hist, d, seed), snapshots, years=(year,), dates=(d,), **kw)
    same_day = build_store(*scramble_from(hist, snapshots, d, seed), years=(year,), dates=(d,), **kw)
    odd = ~even_races(d)
    return AsofResult(
        date=d,
        rows=day.height,
        truncated=diff_counts(day, truncated, features),
        perturbed=diff_counts(day, perturbed, features),
        same_day=diff_counts(day, same_day, [c for c in features if c in stage_a])
        | diff_counts(day.filter(odd), same_day.filter(odd), [c for c in features if c not in stage_a]),
    )
