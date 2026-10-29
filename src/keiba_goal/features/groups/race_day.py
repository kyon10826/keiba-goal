"""判断時刻に分かるレース条件 (段階 B): 馬場・天気・field_dec の頭数・判断時の複勝の支払着数。"""

from __future__ import annotations

import polars as pl

from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import WEATHER, coded, f32, going, n_paid_rule
from keiba_goal.features.registry import Group, spec_maker

NAME = "race_day"
S = spec_maker(NAME, "B")

SPECS = (
    S("rd_going", "num", ["state"], doc="馬場 良 0 / 稍 1 / 重 2 / 不 3"),
    S("rd_weather", "cat", ["weather"], doc="天気 晴 1 / 曇 2 / 小雨 3 / 雨 4 / 小雪 5 / 雪 6"),
    S("rd_field_size", "num", ["field_dec_size"], doc="判断時の頭数 (field_dec)"),
    S("rd_n_paid_places", "num", ["field_open"], doc="判断時の複勝の支払着数: 発売開始時の頭数 ≥ 8 → 3、5-7 → 2、≤ 4 → 0"),
    S("rd_field_fallback", "flag", ["field_dec_fallback"], doc="T-6 が無く field_dec を取消の印で決めたレース"),
)


def build(ctx: Ctx) -> pl.DataFrame:
    return (
        ctx.target("state", "weather", "field_dec_size", "field_open", "field_dec_fallback")
        .select(
            "race_id",
            f32(going("state"), "rd_going"),
            f32(coded("weather", WEATHER), "rd_weather"),
            f32(pl.col("field_dec_size"), "rd_field_size"),
            f32(n_paid_rule(pl.col("field_open")), "rd_n_paid_places"),
            f32(pl.col("field_dec_fallback"), "rd_field_fallback"),
        )
        .collect()
    )


GROUP = Group(NAME, "B", SPECS, build)
