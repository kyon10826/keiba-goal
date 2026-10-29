"""1〜5 走前 (D より前の出走、取消・除外を除く) の成績 (段階 A)。「騎手が今回と同じか」は段階 B の horse_day。"""

from __future__ import annotations

import polars as pl

from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import DNF_CODES, days_between, f32, going, positive, ratio
from keiba_goal.features.groups.speed import FIG_SOURCES, started_with_figures
from keiba_goal.features.registry import Group, spec_maker

NAME = "past_runs"
S = spec_maker(NAME, "A")
K = 5
CORNERS = ("corner1_rank", "corner2_rank", "corner3_rank", "corner4_rank")
VALUE_SOURCES: tuple[str, ...] = (
    "rank", "starters", "time_diff", "last_3F_time", "last_3F_rank", *CORNERS, "PCI", "last_3F_time_diff", "pop",
    "win_odds", "dist", "surface", "class_ord", "state", "error_code",
)
# (名前, 種類, 説明)。値の式は _values()。
VALUES: tuple[tuple[str, str, str], ...] = (
    ("rank", "num", "着順 (着順なしは NaN)"),
    ("rank_ratio", "num", "着順 / 出走頭数"),
    ("time_diff", "num", "着差タイム (秒)"),
    ("last3f", "num", "上がり 3F (秒)"),
    ("last3f_rank_ratio", "num", "上がり順位 / 出走頭数"),
    ("early_pos", "num", "最初の通過順位 / 出走頭数 (先行度)"),
    ("late_pos", "num", "4 角の通過順位 / 出走頭数"),
    ("pci", "num", "PCI"),
    ("last3f_diff", "num", "-3F 差"),
    ("pop", "num", "確定人気 (そのレースの)"),
    ("log_odds", "num", "log 確定単勝オッズ (そのレースの)"),
    ("dist", "num", "距離"),
    ("surface", "cat", "芝ダ"),
    ("class_ord", "num", "クラス序数"),
    ("going", "num", "馬場 0-3"),
    ("starters", "num", "出走頭数"),
    ("days", "num", "その走からの日数"),
    ("speed", "num", "スピード指数"),
    ("dnf", "flag", "競走中止・失格・落馬再騎乗"),
)
HIST = (*VALUE_SOURCES, "date", *FIG_SOURCES)
SPECS = tuple(
    S(f"p{k}_{v}", kind, ["date"] if v == "days" else [], HIST, doc=f"{k} 走前の{text}")
    for k in range(1, K + 1)
    for v, kind, text in VALUES
)


def _values() -> list[pl.Expr]:
    starters = pl.col("starters")
    return [
        positive("rank").alias("rank"),
        ratio(positive("rank"), starters).alias("rank_ratio"),
        pl.col("time_diff").cast(pl.Float32).alias("time_diff"),
        pl.col("last_3F_time").cast(pl.Float32).alias("last3f"),
        ratio(positive("last_3F_rank"), starters).alias("last3f_rank_ratio"),
        ratio(pl.coalesce(*CORNERS), starters).alias("early_pos"),
        ratio(pl.col("corner4_rank"), starters).alias("late_pos"),
        pl.col("PCI").cast(pl.Float32).alias("pci"),
        pl.col("last_3F_time_diff").cast(pl.Float32).alias("last3f_diff"),
        positive("pop").alias("pop"),
        positive("win_odds").log().alias("log_odds"),
        pl.col("dist").cast(pl.Float32).alias("dist"),
        pl.col("surface").cast(pl.Float32).alias("surface"),
        pl.when(pl.col("class_ord") >= 0).then(pl.col("class_ord")).cast(pl.Float32).alias("class_ord"),
        going("state").alias("going"),
        starters.cast(pl.Float32).alias("starters"),
        pl.col("speed_fig").alias("speed"),
        pl.col("error_code").is_in(list(DNF_CODES)).cast(pl.Float32).alias("dnf"),
    ]


def build(ctx: Ctx) -> pl.DataFrame:
    names = [v for v, _, _ in VALUES if v != "days"]
    ev = (
        started_with_figures(ctx, *VALUE_SOURCES)
        .select("id", "date", "race_key", lags.SEQ, *_values())
        .select("id", "date", "race_key", lags.SEQ, *(pl.col(v).cast(pl.Float32) for v in names))
        .collect()
        .lazy()
    )
    rows = lags.prior_count(ctx.target(), ev).select("race_id", "id", "date", lags.N_PRIOR).collect()
    # 1 走ずつ作って横に並べる (5 走分を 1 つの式で結合するとメモリが倍以上になる)
    parts = [rows.select("race_id")]
    for k in range(1, K + 1):
        part = lags.at_prior(rows.lazy(), ev, k, {**{v: f"p{k}_{v}" for v in names}, "date": "_d"})
        part = part.with_columns(days_between("date", "_d").alias(f"p{k}_days"))
        parts.append(part.select(f32(pl.col(f"p{k}_{v}"), f"p{k}_{v}") for v, _, _ in VALUES).collect())
    return parts[0].hstack([col for part in parts[1:] for col in part.get_columns()])


GROUP = Group(NAME, "A", SPECS, build, requires=("speed",))
