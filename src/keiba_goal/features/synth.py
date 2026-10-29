"""テスト用の小さな履歴を手で書くための道具 (schema.HISTORY_DTYPES の列・型・順にそろえる)。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from keiba_goal.data.schema import CODE_TO_PLACE, HISTORY_DTYPES, NOT_STARTED_CODES

REQUIRED: tuple[str, ...] = ("date", "venue_code", "race_num", "horse_num", "id")


def history_frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """rows の各行は REQUIRED が必須。race_id・race_key・nk_race_id・month・day・place は作り、times/daily は既定 1、
    error_code は既定 0、starters は与えなければ同じレースの出走 (error_code ∉ {1,2,3}) の数。ほかの列は null。"""
    recs = []
    for r in rows:
        if missing := [c for c in REQUIRED if c not in r]:
            raise ValueError(f"列 {missing} が必要です: {dict(r)}")
        rec = {"times": 1, "daily": 1, "error_code": 0, "source": "record", **r}
        rec["race_id"] = int(
            f"{rec['date']:08d}{rec['venue_code']:02d}{rec['times']:02d}{rec['daily']:02d}{rec['race_num']:02d}"
            f"{rec['horse_num']:02d}"
        )
        recs.append(rec)
    df = pl.DataFrame(recs, infer_schema_length=None)
    rid = pl.col("race_id")
    df = df.with_columns(
        (rid // 100).alias("race_key"),
        ((rid // 10**14).cast(pl.Utf8) + (rid // 100 % 10**8).cast(pl.Utf8).str.zfill(8)).alias("nk_race_id"),
        (pl.col("date") // 100 % 100).alias("month"),
        (pl.col("date") % 100).alias("day"),
        (pl.col("date") // 10_000 % 100).alias("year"),
        pl.col("venue_code").replace_strict(CODE_TO_PLACE, return_dtype=pl.Utf8).alias("place"),
    )
    if "starters" not in df.columns:
        started = ~pl.col("error_code").is_in(sorted(NOT_STARTED_CODES))
        df = df.with_columns(started.sum().over("race_key").alias("starters"))
    out = df.select(
        pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c) for c, t in HISTORY_DTYPES.items()
    )
    return out.sort(["date", "race_key", "horse_num"])
