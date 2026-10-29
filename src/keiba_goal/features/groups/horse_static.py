"""馬・枠の静的な属性と出走歴の長さ (段階 A)。"""

from __future__ import annotations

import polars as pl

from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import SEX, coded, days_between, f32, started
from keiba_goal.features.registry import Group, spec_maker

NAME = "horse_static"
S = spec_maker(NAME, "A")

SPECS = (
    S("hs_waku", "num", ["waku_num"], doc="枠番"),
    S("hs_horse_num", "num", ["horse_num"], doc="馬番"),
    S("hs_sex", "cat", ["sex"], doc="牡 1 / 牝 2 / セ 3"),
    S("hs_age", "num", ["age"], doc="馬齢"),
    S("hs_career_days", "num", ["date"], ["date", "error_code"], doc="初出走からの日数 (D より前の出走が無ければ NaN)"),
    S("hs_n_starts", "num", ["date"], ["date", "error_code"], doc="D より前の出走数 (取消・除外を除く)"),
)


def build(ctx: Ctx) -> pl.DataFrame:
    ev = ctx.history("error_code").filter(started())
    first = ev.group_by("id").agg(pl.col("date").min().alias("_first"))
    rows = lags.prior_count(ctx.target("waku_num", "sex", "age"), ev).join(first, on="id", how="left")
    n = pl.col(lags.N_PRIOR)
    return rows.select(
        "race_id",
        f32(pl.col("waku_num"), "hs_waku"),
        f32(pl.col("horse_num"), "hs_horse_num"),
        f32(coded("sex", SEX), "hs_sex"),
        f32(pl.col("age"), "hs_age"),
        f32(pl.when(n > 0).then(days_between("date", "_first")), "hs_career_days"),
        f32(n, "hs_n_starts"),
    ).collect()


GROUP = Group(NAME, "A", SPECS, build)
