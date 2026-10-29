"""目的変数と学習対象の印 (design §5 v2.1)。レース後の列から作る (特徴量には使わない)。

y_win   = 着順 1 (同着は両方 1)。出走していない馬は null。
y_place = 着順 ≤ 支払着数。支払着数は paid_places.parquet (データ層 §3.4) の n_paid_places、表が無い・表に無い
          レースは出走頭数の規則 (≥ 8 → 3、5-7 → 2、≤ 4 → 0)。支払着数 0 (複勝なし)・不明 (null) のレースは null。
started  = error_code ∉ {1,2,3}。eligible = 平地 (surface != 3)・出走・出走頭数 ≥ 5・class_ord ≥ 0。
noise_race = 競走中止 (4) か落馬再騎乗 (6) を含むレース (学習時にレースごと除く)。
"""

from __future__ import annotations

import polars as pl

from keiba_goal.data.schema import NOISE_RACE_CODES
from keiba_goal.features.exprs import n_paid_rule, started
from keiba_goal.features.registry import LABEL_NAMES, OUTPUT_KEYS

SOURCE_COLUMNS: tuple[str, ...] = (*OUTPUT_KEYS, "rank", "error_code", "starters", "surface", "class_ord")
STARTERS_RULE = "starters_rule"
MIN_STARTERS = 5
LABEL_COLUMNS: tuple[str, ...] = (*OUTPUT_KEYS, *LABEL_NAMES)   # 名前は特徴量に使えない (registry.RESERVED_NAMES)


def build_labels(rows: pl.DataFrame, frame: pl.DataFrame, paid_places: pl.DataFrame | None) -> pl.DataFrame:
    """rows: 履歴の行 (SOURCE_COLUMNS)。frame: 同じ行の decision_frame (field_dec の列)。

    paid_places: (race_key, n_paid_places, source) の表か None (None なら全レースを出走頭数の規則で)。
    """
    df = rows.select(SOURCE_COLUMNS).join(
        frame.select("race_id", "in_field_dec", "field_dec_size", "field_dec_fallback"), on="race_id", how="left"
    )
    rule = n_paid_rule(pl.col("starters"))
    if paid_places is None:
        df = df.with_columns(rule.alias("n_paid_places"), pl.lit(STARTERS_RULE).alias("n_paid_source"))
    else:
        pp = paid_places.select(
            "race_key", pl.col("n_paid_places").cast(pl.Int8).alias("_pp"), pl.col("source").alias("_pp_source")
        )
        if pp["race_key"].is_duplicated().any():
            raise ValueError("paid_places に race_key の重複があります")
        df = df.join(pp.with_columns(pl.lit(True).alias("_in_pp")), on="race_key", how="left").with_columns(
            pl.when(pl.col("_in_pp")).then(pl.col("_pp")).otherwise(rule).alias("n_paid_places"),
            pl.when(pl.col("_in_pp")).then(pl.col("_pp_source")).otherwise(pl.lit(STARTERS_RULE)).alias("n_paid_source"),
        ).drop("_pp", "_pp_source", "_in_pp")
    run = started()
    rank = pl.col("rank")
    n_paid = pl.col("n_paid_places")
    return df.with_columns(
        pl.when(run).then(rank == 1).cast(pl.Int8).alias("y_win"),
        pl.when(run & (n_paid > 0)).then((rank >= 1) & (rank <= n_paid)).cast(pl.Int8).alias("y_place"),
        run.alias("started"),
        ((pl.col("surface") != 3) & run & (pl.col("starters") >= MIN_STARTERS) & (pl.col("class_ord") >= 0))
        .fill_null(False).alias("eligible"),
        pl.col("error_code").is_in(sorted(NOISE_RACE_CODES)).any().over("race_key").alias("noise_race"),
    ).select(LABEL_COLUMNS)
