"""「対象日 D より前の出来事」を引く道具。同じ日の行は数えない (ライブで再現できる D−1 時点の状態)。

events(...) で出来事 (例: 馬の出走) に by ごとの通し番号 _seq を付け、prior_count(...) で対象行ごとに
「date < D の出来事の数」_n を数える。k 個前の出来事は _seq = _n − k、直前までの累積は _seq = _n − 1 の行。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

SEQ = "_seq"
N_PRIOR = "_n"


def events(lf: pl.LazyFrame, by: Sequence[str] = ("id",)) -> pl.LazyFrame:
    """(by, date, race_key) の順に並べ、by ごとの通し番号 _seq (0 始まり, Int64) を付ける。"""
    return lf.sort([*by, "date", "race_key"]).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64).over(list(by)).alias(SEQ)
    )


def prior_count(targets: pl.LazyFrame, events_lf: pl.LazyFrame, by: Sequence[str] = ("id",)) -> pl.LazyFrame:
    """targets の各行に _n = 同じ by の出来事のうち date が対象行の date より前のものの数 (Int64) を付ける。"""
    by = list(by)
    days = (
        events_lf.group_by([*by, "date"])
        .agg(pl.len().cast(pl.Int64).alias("_c"))
        .sort([*by, "date"])
        .with_columns(pl.col("_c").cum_sum().over(by).alias(N_PRIOR))
        .select(*by, "date", N_PRIOR)
        .sort("date")
    )
    return (
        targets.sort("date", maintain_order=True)
        .join_asof(days, on="date", by=by, strategy="backward", allow_exact_matches=False, check_sortedness=False)
        .with_columns(pl.col(N_PRIOR).fill_null(0))
    )


def at_prior(
    targets: pl.LazyFrame,
    events_lf: pl.LazyFrame,
    k: int,
    columns: Mapping[str, str],
    by: Sequence[str] = ("id",),
) -> pl.LazyFrame:
    """k 個前 (k = 1 が直前) の出来事の列を付ける。columns = {出来事の列: 付ける名前}。無ければ null。"""
    by = list(by)
    right = events_lf.select(*by, SEQ, *columns).rename(dict(columns))
    return (
        targets.with_columns((pl.col(N_PRIOR) - k).alias("_at"))
        .join(right, left_on=[*by, "_at"], right_on=[*by, SEQ], how="left", maintain_order="left")
        .drop("_at")
    )
