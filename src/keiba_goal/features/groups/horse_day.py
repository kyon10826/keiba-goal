"""判断時刻の馬の情報 (段階 B): 斤量・馬体重・ブリンカー・馬番/頭数と、前走以前との比較 (騎手が同じか など)。

前走以前の値 (通算平均体重・前走の斤量・1〜5 走前の騎手) は補助出力 horse_prev (履歴だけから作る) で受け取り、
当日の出馬表の値と比べる。騎手の乗り替わり (変化) は hd_same_jockey_1 なので、他のグループで作り直さない。

ブリンカーの列は 'B' = 着用、null = 着用なし (schema の約束: record は空欄、netkeiba の行は出馬表 shutuba_past に
B の印が無い馬)。出馬表の無い netkeiba の行 (着用が不明) は history に入らない: 窓 1 は出馬表の無いレースを
捨て (netkeiba/rebuild.py)、出馬表を取らない窓 2/3 の行は同じ race_id の record の行が優先される (data/history.py)。
'B'・null・'' 以外の値は NaN (約束が変わったら黙って 0 にしない)。
"""

from __future__ import annotations

import polars as pl

from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import f32, ratio, started
from keiba_goal.features.registry import Group, Helper, spec_maker

NAME = "horse_day"
S = spec_maker(NAME, "B")
K_JOCKEY = 5
PREV_SOURCES: tuple[str, ...] = ("error_code", "weight", "basis_weight", "jockey_id")
PREV_JOCKEY = tuple(f"prev_jockey_{k}" for k in range(1, K_JOCKEY + 1))

SPECS = (
    S("hd_basis_weight", "num", ["basis_weight"], doc="斤量"),
    S("hd_basis_weight_dev", "num", ["basis_weight", "in_field_dec"], doc="斤量 − field_dec の平均 (field_dec 外は NaN)"),
    S("hd_horse_num_ratio", "num", ["horse_num", "field_dec_size"], doc="馬番 / field_dec の頭数"),
    S("hd_blinker", "flag", ["blinker"], doc="ブリンカー着用 ('B' → 1、null・空 → 0、それ以外 → NaN)"),
    S("hd_weight", "num", ["weight"], doc="馬体重"),
    S("hd_inc_dec", "num", ["inc_dec"], doc="馬体重の増減"),
    S("hd_weight_change_rate", "num", ["inc_dec", "weight"], doc="増減 / 馬体重"),
    S("hd_weight_vs_past_mean", "num", ["weight"], ["error_code", "weight"], doc="馬体重 / D より前の出走の平均体重"),
    S("hd_basis_weight_change", "num", ["basis_weight"], ["error_code", "basis_weight"], doc="斤量 − 前走の斤量"),
    *(
        S(f"hd_same_jockey_{k}", "flag", [], ["error_code", "jockey_id"], doc=f"{k} 走前と同じ騎手", keys=["jockey_id"])
        for k in range(1, K_JOCKEY + 1)
    ),
)


def horse_prev(ctx: Ctx) -> pl.DataFrame:
    """全行について、その日より前の出走の通算平均体重・前走の斤量・1〜5 走前の騎手。"""
    w = pl.col("weight").fill_nan(None)
    ev = lags.events(ctx.history(*PREV_SOURCES).filter(started())).with_columns(
        w.fill_null(0).cum_sum().over("id").alias("_wsum"),
        w.is_not_null().cast(pl.Int64).cum_sum().over("id").alias("_wcnt"),
    ).with_columns(pl.when(pl.col("_wcnt") > 0).then(pl.col("_wsum") / pl.col("_wcnt")).alias("_wmean"))
    rows = lags.prior_count(ctx.history(), ev)
    rows = lags.at_prior(rows, ev, 1, {"_wmean": "prev_weight_mean", "basis_weight": "prev_basis_weight"})
    for k in range(1, K_JOCKEY + 1):
        rows = lags.at_prior(rows, ev, k, {"jockey_id": PREV_JOCKEY[k - 1]})
    return rows.select(
        "race_id",
        pl.col("prev_weight_mean").cast(pl.Float32),
        pl.col("prev_basis_weight").cast(pl.Float32),
        *PREV_JOCKEY,
    ).collect()


HORSE_PREV = Helper(
    "horse_prev", horse_prev, PREV_SOURCES, own_post_race=False,
    doc="行ごとの D より前の通算平均体重・前走斤量・1〜5 走前の騎手 (その行の特徴量に使ってよい)",
)


def blinker_flag(col: str = "blinker") -> pl.Expr:
    b = pl.col(col)
    return pl.when(b == "B").then(1.0).when(b.is_null() | (b == "")).then(0.0)


def build(ctx: Ctx) -> pl.DataFrame:
    rows = ctx.target(
        "basis_weight", "in_field_dec", "field_dec_size", "blinker", "weight", "inc_dec", "jockey_id"
    ).join(ctx.helper("horse_prev"), on="race_id", how="left")
    bw = pl.col("basis_weight")
    member = pl.col("in_field_dec")
    field_mean = pl.when(member).then(bw).mean().over("race_key")
    jockey = pl.col("jockey_id")
    return rows.select(
        "race_id",
        f32(bw, "hd_basis_weight"),
        f32(pl.when(member).then(bw - field_mean), "hd_basis_weight_dev"),
        f32(ratio(pl.col("horse_num"), pl.col("field_dec_size")), "hd_horse_num_ratio"),
        f32(blinker_flag(), "hd_blinker"),
        f32(pl.col("weight"), "hd_weight"),
        f32(pl.col("inc_dec"), "hd_inc_dec"),
        f32(ratio(pl.col("inc_dec"), pl.col("weight")), "hd_weight_change_rate"),
        f32(ratio(pl.col("weight"), pl.col("prev_weight_mean")), "hd_weight_vs_past_mean"),
        f32(bw - pl.col("prev_basis_weight"), "hd_basis_weight_change"),
        *(
            f32(pl.when(jockey.is_not_null() & pl.col(p).is_not_null()).then(jockey == pl.col(p)), f"hd_same_jockey_{k}")
            for k, p in enumerate(PREV_JOCKEY, start=1)
        ),
    ).collect()


GROUP = Group(NAME, "B", SPECS, build, helpers=(HORSE_PREV,))
