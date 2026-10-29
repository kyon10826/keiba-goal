"""レース条件のうち前日に決まっているもの (段階 A)。馬場・天気・頭数は段階 B の race_day。"""

from __future__ import annotations

import polars as pl

from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import f32
from keiba_goal.features.registry import Group, spec_maker
from keiba_goal.features.vocab import COURSE_KEYS, course_vocab

NAME = "race_static"
S = spec_maker(NAME, "A")
# 距離帯 (SMILE): S ≤ 1300 < M < 1900 ≤ I ≤ 2100 < L ≤ 2700 < E
DIST_BANDS: tuple[int, ...] = (1300, 1899, 2100, 2700)

SPECS = (
    S("rs_venue", "cat", ["venue_code"], doc="場コード 1-10"),
    S("rs_surface", "cat", ["surface"], doc="1 芝 / 2 ダート / 3 障害"),
    S("rs_direction", "cat", ["direction"], doc="1 右 / 2 左 / 3 直線 / 0 障害"),
    S("rs_inout", "cat", ["course_inout"], doc="内外 (JV トラックコードから)"),
    S("rs_dist", "num", ["dist"], doc="距離 (m)"),
    S("rs_dist_band", "cat", ["dist"], doc="距離帯 SMILE 1-5"),
    S("rs_class_ord", "num", ["class_ord"], doc="クラス序数 0-7 (対象外 -1 は NaN)"),
    S("rs_age_code", "cat", ["age_code"], doc="競走種別コード"),
    S("rs_weight_code", "cat", ["weight_code"], doc="重量種別 1-4 (0 不明は NaN)"),
    S("rs_race_num", "num", ["race_num"], doc="R 番号"),
    S("rs_month", "num", ["date"], doc="月"),
    S("rs_course", "cat", ["venue_code", "surface", "dist"], doc="場×芝ダ×距離の固定語彙 (2003 年以前) のコード、語彙外は NaN"),
    S("rs_corner_num", "num", ["corner_num"], doc="コーナー数"),
)


def dist_band(col: str = "dist") -> pl.Expr:
    d = pl.col(col)
    expr = pl.when(d.is_null()).then(None)
    for i, upper in enumerate(DIST_BANDS, start=1):
        expr = expr.when(d <= upper).then(i)
    return expr.otherwise(len(DIST_BANDS) + 1)


def build(ctx: Ctx) -> pl.DataFrame:
    rows = ctx.target(
        "venue_code", "surface", "direction", "course_inout", "dist", "class_ord", "age_code", "weight_code",
        "race_num", "corner_num",
    )
    return (
        rows.join(course_vocab().lazy(), on=list(COURSE_KEYS), how="left")
        .select(
            "race_id",
            f32(pl.col("venue_code"), "rs_venue"),
            f32(pl.col("surface"), "rs_surface"),
            f32(pl.col("direction"), "rs_direction"),
            f32(pl.col("course_inout"), "rs_inout"),
            f32(pl.col("dist"), "rs_dist"),
            f32(dist_band(), "rs_dist_band"),
            f32(pl.when(pl.col("class_ord") >= 0).then(pl.col("class_ord")), "rs_class_ord"),
            f32(pl.col("age_code"), "rs_age_code"),
            f32(pl.when(pl.col("weight_code") > 0).then(pl.col("weight_code")), "rs_weight_code"),
            f32(pl.col("race_num"), "rs_race_num"),
            f32(pl.col("date") // 100 % 100, "rs_month"),
            f32(pl.col("course_code"), "rs_course"),
            f32(pl.col("corner_num"), "rs_corner_num"),
        )
        .collect()
    )


GROUP = Group(NAME, "A", SPECS, build)
