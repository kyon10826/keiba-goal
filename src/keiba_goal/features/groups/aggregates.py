"""直近 3/5/10 走と通算 (D より前の出走) の集計 (段階 A)。

出走数は horse_static.hs_n_starts、直近 3/5 走のスピード指数の平均・最大は speed (sp_mean3 など) にあるので
ここでは作らない (同じ列を 2 回作らない)。
"""

from __future__ import annotations

import polars as pl

from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import f32, positive, ratio
from keiba_goal.features.groups.speed import FIG_SOURCES, started_with_figures
from keiba_goal.features.registry import Group, spec_maker

NAME = "aggregates"
S = spec_maker(NAME, "A")
WINDOWS: tuple[int | None, ...] = (3, 5, 10, None)   # None = 通算
SPEED_WINDOWS: tuple[int | None, ...] = (10, None)
VALUE_SOURCES: tuple[str, ...] = ("rank", "starters", "time_diff", "last_3F_rank", "pop", "prize", "error_code")
HIST = (*VALUE_SOURCES, "date", *FIG_SOURCES)
# (統計名, 元の値, 集計, 説明)
STATS: tuple[tuple[str, str, str, str], ...] = (
    ("rank_ratio_mean", "rr", "mean", "着順/頭数の平均"),
    ("win_rate", "win", "mean", "勝率"),
    ("top3_rate", "top3", "mean", "3 着内率"),
    ("time_diff_mean", "td", "mean", "着差タイムの平均"),
    ("time_diff_min", "td", "min", "着差タイムの最小"),
    ("last3f_rank_ratio_mean", "l3r", "mean", "上がり順位/頭数の平均"),
    ("pop_mean", "pop", "mean", "人気の平均"),
    ("prize_sum", "prize", "sum", "本賞金の合計 (万円)"),
)
SPEED_STATS: tuple[tuple[str, str, str, str], ...] = (
    ("speed_mean", "sp", "mean", "スピード指数の平均"),
    ("speed_max", "sp", "max", "スピード指数の最大"),
)


def window_name(w: int | None) -> str:
    return "all" if w is None else f"l{w}"


def _plan() -> list[tuple[str, str, str, int | None, str]]:
    plan = [(f"ag_{window_name(w)}_{s}", v, agg, w, text) for w in WINDOWS for s, v, agg, text in STATS]
    plan += [(f"ag_{window_name(w)}_{s}", v, agg, w, text) for w in SPEED_WINDOWS for s, v, agg, text in SPEED_STATS]
    return plan


PLAN = _plan()
SPECS = tuple(
    S(name, "num", [], HIST, doc=f"{'通算' if w is None else f'直近 {w} 走'}の{text}") for name, _, _, w, text in PLAN
)


def _window(v: str, agg: str, w: int | None) -> pl.Expr:
    """出来事 (時刻順) ごとに、その出来事を含む直近 w 件 (None は全件) の集計。null は数えない。"""
    c = pl.col(v)
    if w is not None:
        rolled = {"mean": c.rolling_mean, "min": c.rolling_min, "max": c.rolling_max, "sum": c.rolling_sum}[agg]
        return rolled(w, min_samples=1).over("id")
    if agg in ("mean", "sum"):
        total = c.fill_null(0).cum_sum().over("id")
        count = c.is_not_null().cast(pl.Int64).cum_sum().over("id")
        return pl.when(count > 0).then(total / count if agg == "mean" else total)
    fill = float("inf") if agg == "min" else float("-inf")
    best = (c.fill_null(fill).cum_min() if agg == "min" else c.fill_null(fill).cum_max()).over("id")
    return pl.when(best.is_finite()).then(best)


def build(ctx: Ctx) -> pl.DataFrame:
    ev = started_with_figures(ctx, *VALUE_SOURCES).with_columns(
        ratio(positive("rank"), pl.col("starters")).alias("rr"),
        (pl.col("rank") == 1).cast(pl.Float32).alias("win"),
        pl.col("rank").is_between(1, 3).cast(pl.Float32).alias("top3"),
        pl.col("time_diff").cast(pl.Float32).fill_nan(None).alias("td"),
        ratio(positive("last_3F_rank"), pl.col("starters")).alias("l3r"),
        positive("pop").alias("pop"),
        pl.col("prize").cast(pl.Float32).alias("prize"),
        pl.col("speed_fig").alias("sp"),
    )
    ev = ev.with_columns(_window(v, agg, w).cast(pl.Float32).alias(name) for name, v, agg, w, _ in PLAN)
    rows = lags.at_prior(lags.prior_count(ctx.target(), ev), ev, 1, {name: name for name, *_ in PLAN})
    return rows.select("race_id", *(f32(pl.col(s.name), s.name) for s in SPECS)).collect()


GROUP = Group(NAME, "A", SPECS, build, requires=("speed",))
