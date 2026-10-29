"""グループ間で共有する列の式 (コード化・比・日付)。値の無いところは null (最後に NaN へそろえる)。"""

from __future__ import annotations

import polars as pl

from keiba_goal.data.schema import NOT_STARTED_CODES

# 語彙は固定 (学習期間に依らない)。語彙外の値は null。
GOING: dict[str, int] = {"良": 0, "稍": 1, "重": 2, "不": 3}
WEATHER: dict[str, int] = {"晴": 1, "曇": 2, "小雨": 3, "雨": 4, "小雪": 5, "雪": 6}
SEX: dict[str, int] = {"牡": 1, "牝": 2, "セ": 3}
DNF_CODES: tuple[int, ...] = (4, 5, 6)          # 競走中止・失格・落馬再騎乗
SCRATCHED_CODES: tuple[int, ...] = (1, 2)       # 判断時刻までに分かる取消 (出走取消・発走除外)


def started() -> pl.Expr:
    """出走した行 (error_code ∉ {1,2,3})。error_code が null の行 (当日の出馬表) は null。"""
    return ~pl.col("error_code").is_in(sorted(NOT_STARTED_CODES))


def coded(col: str, mapping: dict[str, int]) -> pl.Expr:
    return pl.col(col).replace_strict(mapping, default=None, return_dtype=pl.Float32)


def going(col: str = "state") -> pl.Expr:
    return coded(col, GOING)


def positive(col: str) -> pl.Expr:
    """0 以下 (着順なし・人気なし・オッズなし) を null に。"""
    return pl.when(pl.col(col) > 0).then(pl.col(col).cast(pl.Float32))


def ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num.cast(pl.Float32) / den.cast(pl.Float32))


def ymd(col: str = "date") -> pl.Expr:
    """YYYYMMDD 整数 → Date。"""
    c = pl.col(col)
    return pl.date(c // 10_000, c // 100 % 100, c % 100)


def days_between(later: str, earlier: str) -> pl.Expr:
    return (ymd(later) - ymd(earlier)).dt.total_days().cast(pl.Float32)


def n_paid_rule(n: pl.Expr) -> pl.Expr:
    """頭数 → 複勝の支払着数 (design §3.4): ≥ 8 → 3、5-7 → 2、≤ 4 → 0 (複勝なし)。null は null。"""
    return pl.when(n.is_null()).then(None).when(n >= 8).then(3).when(n >= 5).then(2).otherwise(0).cast(pl.Int8)


def f32(expr: pl.Expr, name: str) -> pl.Expr:
    """特徴量の列: Float32、値の無いところは NaN。"""
    return expr.cast(pl.Float32).fill_null(float("nan")).alias(name)
