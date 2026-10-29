"""スピード指数 (段階 A)。

行ごとの指数 speed_fig = (基準平均 − 走破タイム) / 基準標準偏差 (大きいほど速い)。基準は (場, 芝ダ, 距離, 馬場) ごとに
その行の年より前の 3 暦年の完走馬 (着順 ≥ 1、タイムあり) から。標本が 30 未満なら場をまたいだ (芝ダ, 距離, 馬場)、
それも 30 未満なら NaN。行自身のタイムを使うので、補助出力 speed_fig はその行より後の行の特徴量にだけ使う。
"""

from __future__ import annotations

import polars as pl

from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import f32, going, started
from keiba_goal.features.registry import Group, Helper, spec_maker

NAME = "speed"
S = spec_maker(NAME, "A")
BASELINE_YEARS = 3
MIN_SAMPLES = 30
VENUE_KEYS: tuple[str, ...] = ("venue_code", "surface", "dist", "going")
POOLED_KEYS: tuple[str, ...] = ("surface", "dist", "going")
FIG_SOURCES: tuple[str, ...] = ("venue_code", "surface", "dist", "state", "time", "rank")
HIST = ("error_code", "surface", *FIG_SOURCES)

SPECS = (
    S("sp_last", "num", [], HIST, doc="前走のスピード指数"),
    S("sp_mean3", "num", [], HIST, doc="直近 3 走の平均"),
    S("sp_max3", "num", [], HIST, doc="直近 3 走の最大"),
    S("sp_mean5", "num", [], HIST, doc="直近 5 走の平均"),
    S("sp_max5", "num", [], HIST, doc="直近 5 走の最大"),
    S("sp_same_surface_last", "num", ["surface"], HIST, doc="今回と同じ芝ダの直近の出走の指数"),
)


def _baseline(finishers: pl.LazyFrame, keys: tuple[str, ...], suffix: str) -> pl.LazyFrame:
    """年 Y の行の基準 = Y−3〜Y−1 年の完走馬の (平均, 標準偏差, 件数)。"""
    per_year = finishers.group_by("_year", *keys).agg(
        pl.len().cast(pl.Int64).alias("_n"),
        pl.col("_t").sum().alias("_s"),
        (pl.col("_t") ** 2).sum().alias("_ss"),
    )
    shifted = pl.concat([per_year.with_columns(pl.col("_year") + lag) for lag in range(1, BASELINE_YEARS + 1)])
    agg = shifted.group_by("_year", *keys).agg(pl.col("_n").sum(), pl.col("_s").sum(), pl.col("_ss").sum())
    n = pl.col("_n")
    var = (pl.col("_ss") - pl.col("_s") ** 2 / n) / (n - 1)
    return agg.select(
        "_year", *keys,
        n.alias(f"_n{suffix}"),
        (pl.col("_s") / n).alias(f"_mean{suffix}"),
        pl.when(n > 1).then(var.clip(lower_bound=0).sqrt()).alias(f"_std{suffix}"),
    )


def speed_figures(ctx: Ctx) -> pl.DataFrame:
    rows = ctx.history(*FIG_SOURCES).with_columns(
        (pl.col("date") // 10_000).alias("_year"), going("state").alias("going"), pl.col("time").cast(pl.Float64).alias("_t")
    )
    finishers = rows.filter((pl.col("rank") >= 1) & pl.col("_t").is_not_null() & pl.col("going").is_not_null())
    out = rows.join(_baseline(finishers, VENUE_KEYS, "1"), on=["_year", *VENUE_KEYS], how="left").join(
        _baseline(finishers, POOLED_KEYS, "2"), on=["_year", *POOLED_KEYS], how="left"
    )
    use1 = pl.col("_n1") >= MIN_SAMPLES
    use2 = pl.col("_n2") >= MIN_SAMPLES
    mean = pl.when(use1).then(pl.col("_mean1")).when(use2).then(pl.col("_mean2"))
    std = pl.when(use1).then(pl.col("_std1")).when(use2).then(pl.col("_std2"))
    fig = pl.when(std > 0).then((mean - pl.col("_t")) / std)
    return out.select("race_id", fig.cast(pl.Float32).alias("speed_fig")).collect()


SPEED_FIG = Helper(
    "speed_fig", speed_figures, FIG_SOURCES, own_post_race=True,
    doc="行ごとのスピード指数 (race_id, speed_fig)。行自身のタイムを使う",
)


def started_with_figures(ctx: Ctx, *cols: str) -> pl.LazyFrame:
    """出走した行 (D より前の過去走の候補) にスピード指数を付けた出来事の表 (lags.events)。"""
    ev = ctx.history("error_code", *cols).filter(started()).join(ctx.helper("speed_fig"), on="race_id", how="left")
    return lags.events(ev.with_columns(pl.col("speed_fig").fill_nan(None)))


def build(ctx: Ctx) -> pl.DataFrame:
    ev = started_with_figures(ctx, "surface")
    fig = pl.col("speed_fig")
    ev = ev.with_columns(
        *(fig.rolling_mean(w, min_samples=1).over("id").alias(f"_mean{w}") for w in (3, 5)),
        *(fig.rolling_max(w, min_samples=1).over("id").alias(f"_max{w}") for w in (3, 5)),
    )
    rows = lags.prior_count(ctx.target("surface"), ev)
    rows = lags.at_prior(rows, ev, 1, {"speed_fig": "sp_last", "_mean3": "sp_mean3", "_max3": "sp_max3",
                                       "_mean5": "sp_mean5", "_max5": "sp_max5"})
    by_surface = lags.events(ev.drop(lags.SEQ), by=("id", "surface"))
    same = lags.prior_count(rows.select("race_id", "id", "date", "surface"), by_surface, by=("id", "surface"))
    same = lags.at_prior(same, by_surface, 1, {"speed_fig": "sp_same_surface_last"}, by=("id", "surface"))
    out = rows.join(same.select("race_id", "sp_same_surface_last"), on="race_id", how="left")
    return out.select("race_id", *(f32(pl.col(s.name), s.name) for s in SPECS)).collect()


GROUP = Group(NAME, "A", SPECS, build, helpers=(SPEED_FIG,))
