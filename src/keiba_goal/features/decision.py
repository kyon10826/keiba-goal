"""段階 B の対象行の枠 (判断時刻の情報): 取消の印・判断スナップショット・field_dec・発売開始時の頭数 (design §4.1)。

batch とライブで同じ関数を通す。batch では次の 2 つを履歴から近似し、ライブでは出馬表から渡す:
- scratched: 判断時刻までに取消と分かっている馬。batch は error_code ∈ {1,2} (出走取消・発走除外)。
- field_open: 発売開始時に売られていた頭数 (複勝の支払着数を決める)。batch は paid_places.field_open、無ければ
  T-120/60/30/6 のどれかで単勝オッズが付いた馬の数、それも無ければ field_dec の頭数。
"""

from __future__ import annotations

import polars as pl

from keiba_goal.features.exprs import SCRATCHED_CODES
from keiba_goal.features.registry import DECISION_X, SNAPSHOT_COLUMNS, SNAPSHOT_VALUES, SNAPSHOT_X

SNAPSHOT_SOURCE_COLUMNS: tuple[str, ...] = ("race_key", "horse_num", "x", *SNAPSHOT_VALUES)
FIELD_OPEN_SOURCES: tuple[str, ...] = ("paid_places", "snapshots", "field_dec")


def with_live_columns(history: pl.LazyFrame) -> pl.LazyFrame:
    """scratched が無い (または null の) 行は error_code から作る。ライブの出馬表の行は渡された値を保つ。"""
    derived = pl.col("error_code").is_in(list(SCRATCHED_CODES))
    if "scratched" in history.collect_schema():
        return history.with_columns(pl.coalesce(pl.col("scratched"), derived).alias("scratched"))
    return history.with_columns(derived.alias("scratched"))


def pivot_snapshots(snaps: pl.DataFrame) -> pl.DataFrame:
    """ts_snap 形式 (race_key, horse_num, x, win, place_lo, place_hi, …) → 1 馬 1 行の win_120 … place_hi_6。

    診断列 (vote_share・snap_time・staleness_min) は読まない。欠損は NaN。
    """
    s = snaps.select(SNAPSHOT_SOURCE_COLUMNS)
    if s.select(pl.struct("race_key", "horse_num", "x").is_duplicated().any()).item():
        raise ValueError("スナップショットに (race_key, horse_num, x) の重複があります")
    out = s.select("race_key", "horse_num").unique()
    for x in SNAPSHOT_X:
        part = s.filter(pl.col("x") == x).select(
            "race_key", "horse_num", *(pl.col(v).cast(pl.Float32).alias(f"{v}_{x}") for v in SNAPSHOT_VALUES)
        )
        out = out.join(part, on=["race_key", "horse_num"], how="left")
    return out.with_columns(pl.col(c).fill_null(float("nan")) for c in SNAPSHOT_COLUMNS)


def decision_frame(
    rows: pl.DataFrame,
    snapshots: pl.DataFrame | None,
    field_open: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """rows (対象行: キー + 発走前の列 + scratched) に判断時刻の列 (registry.LIVE_COLUMNS) を付ける。

    field_dec = 判断スナップショット (T-6) で単勝オッズが付いている馬。そのレースに T-6 が 1 頭も無ければ
    scratched でない馬 (field_dec_fallback = True)。field_open は (race_key, field_open) の表 (無い・null は代替)。
    """
    if rows["scratched"].null_count():
        raise ValueError("scratched が null の行があります (ライブは出馬表の取消の印を渡す)")
    if snapshots is not None and snapshots.height:
        df = rows.join(pivot_snapshots(snapshots), on=["race_key", "horse_num"], how="left")
        df = df.with_columns(pl.col(c).fill_null(float("nan")) for c in SNAPSHOT_COLUMNS)
    else:
        df = rows.with_columns(pl.lit(float("nan"), dtype=pl.Float32).alias(c) for c in SNAPSHOT_COLUMNS)
    finite = pl.col(f"win_{DECISION_X}").is_finite() & (pl.col(f"win_{DECISION_X}") > 0)
    race_has = finite.any().over("race_key")
    df = df.with_columns(
        pl.when(race_has).then(finite).otherwise(~pl.col("scratched")).alias("in_field_dec"),
        (~race_has).alias("field_dec_fallback"),
    ).with_columns(pl.col("in_field_dec").sum().over("race_key").cast(pl.Int16).alias("field_dec_size"))
    any_odds = pl.any_horizontal(pl.col(f"win_{x}").is_finite() & (pl.col(f"win_{x}") > 0) for x in SNAPSHOT_X)
    proxy = pl.when(any_odds.any().over("race_key")).then(any_odds.sum().over("race_key"))
    table = pl.lit(None, dtype=pl.Int16)
    if field_open is not None:
        fo = field_open.select("race_key", pl.col("field_open").cast(pl.Int16).alias("_fo"))
        if fo["race_key"].is_duplicated().any():
            raise ValueError("field_open の表に race_key の重複があります")
        df = df.join(fo, on="race_key", how="left")
        table = pl.col("_fo")
    df = df.with_columns(
        pl.coalesce(table, proxy.cast(pl.Int16), pl.col("field_dec_size")).alias("field_open"),
        pl.when(table.is_not_null()).then(pl.lit(FIELD_OPEN_SOURCES[0]))
        .when(proxy.is_not_null()).then(pl.lit(FIELD_OPEN_SOURCES[1]))
        .otherwise(pl.lit(FIELD_OPEN_SOURCES[2])).alias("field_open_source"),
    )
    return df.drop("_fo", strict=False)
