"""複勝の支払着数の表 paid_places.parquet (design §3.4・§5)。ラベル・q 表・バックテストが同じ表を使う。

支払着数は「発売開始時の頭数」で決まる (8 頭登録・発売開始後に 1 頭取消で 7 頭出走の 202309040202 は 3 着まで払戻)。
レースごとに次の順で決める:
1. 払戻表 (P2 の crawl combine が作る payouts.parquet) にそのレースの n_paid_places (null は不明) があればその値
2. JD のあるレースは field_open (ts_race: 締切前の候補行のどれかで単勝オッズが付いた馬の数。発売開始前の取消馬は
   一度も値が付かない) から
3. どちらも無ければ実際の出走頭数 starters から
2・3 は 8 頭以上なら 3、5〜7 頭なら 2、4 頭以下は 0 (複勝の発売なし)。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from keiba_goal.data.record import write_parquet_atomic

PAID_PLACES_SCHEMA: dict[str, pl.DataType] = {
    "race_key": pl.Int64,
    "n_paid_places": pl.Int8,
    "field_open": pl.Int16,
    "starters": pl.Int16,
    "source": pl.Utf8,
}
SOURCES: tuple[str, ...] = ("payout", "jd_field_open", "starters")


def places_for_field(n: pl.Expr) -> pl.Expr:
    """頭数 → 支払着数 (8 頭以上 3、5〜7 頭 2、4 頭以下 0)。"""
    return pl.when(n >= 8).then(3).when(n >= 5).then(2).otherwise(0).cast(pl.Int8)


def payout_places(payouts: pl.DataFrame) -> pl.DataFrame:
    """払戻表 (race_key, n_paid_places の行) → レースごとの (race_key, n_paid_places)。null (不明) だけのレースは除く。

    1 レースの行で値が食い違っていれば例外。
    """
    per_race = (
        payouts.filter(pl.col("n_paid_places").is_not_null())
        .group_by("race_key")
        .agg(pl.col("n_paid_places").unique().alias("values"))
    )
    bad = per_race.filter(pl.col("values").list.len() > 1)
    if bad.height:
        raise ValueError(f"払戻表の n_paid_places が 1 レースの中で食い違っています: {bad.head(5).to_dicts()}")
    return per_race.select(pl.col("race_key"), pl.col("values").list.first().cast(pl.Int8).alias("n_paid_places"))


def build_paid_places(races: pl.DataFrame, field_open: pl.DataFrame, payouts: pl.DataFrame | None) -> pl.DataFrame:
    """races: (race_key, starters) 1 レース 1 行 / field_open: ts_race の (race_key, field_open) / payouts: payout_places。"""
    if races["race_key"].n_unique() != races.height:
        raise ValueError("races の race_key が重複しています")
    paid = payouts if payouts is not None else pl.DataFrame(schema={"race_key": pl.Int64, "n_paid_places": pl.Int8})
    d = (
        races.select("race_key", "starters")
        .join(field_open.select("race_key", "field_open"), on="race_key", how="left")
        .join(paid.rename({"n_paid_places": "paid"}), on="race_key", how="left")
    )
    has_field = pl.col("field_open").is_not_null()
    out = d.select(
        pl.col("race_key"),
        pl.when(pl.col("paid").is_not_null()).then(pl.col("paid"))
        .when(has_field).then(places_for_field(pl.col("field_open")))
        .otherwise(places_for_field(pl.col("starters"))).alias("n_paid_places"),
        pl.col("field_open"),
        pl.col("starters"),
        pl.when(pl.col("paid").is_not_null()).then(pl.lit("payout"))
        .when(has_field).then(pl.lit("jd_field_open"))
        .otherwise(pl.lit("starters")).alias("source"),
    )
    return out.cast(PAID_PLACES_SCHEMA).sort("race_key")


@dataclass(frozen=True)
class PaidPlacesSummary:
    races: int
    by_source: dict[str, int]
    changed_vs_starters: dict[tuple[int, str], int]   # (年, source) → 出走頭数の規則と支払着数が違うレース数
    field_below_starters: int                          # field_open < starters (あればデータの異常)
    payout_vs_field_open: tuple[int, int]              # 払戻表と field_open の両方があるレースの (一致, 不一致)
    payout_races_not_in_history: int
    examples_changed: list[dict]


def load_field_open(ts_race_dir: Path, years: Iterable[int]) -> pl.DataFrame:
    """ts_race/{year}.parquet の (race_key, field_open)。年のファイルが無ければ例外。"""
    parts = []
    for year in years:
        path = ts_race_dir / f"{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} がありません (先に --step ts)")
        parts.append(pl.read_parquet(path, columns=["race_key", "field_open"]))
    return pl.concat(parts) if parts else pl.DataFrame(schema={"race_key": pl.Int64, "field_open": pl.Int16})


def summarize(out: pl.DataFrame, payouts: pl.DataFrame | None) -> PaidPlacesSummary:
    d = out.with_columns(
        places_for_field(pl.col("starters")).alias("by_starters"),
        places_for_field(pl.col("field_open")).alias("by_field"),
        (pl.col("race_key") // 10**12).alias("year"),
    )
    changed = d.filter(pl.col("n_paid_places") != pl.col("by_starters"))
    per_year = changed.group_by("year", "source").len().sort("year", "source")
    both = d.filter((pl.col("source") == "payout") & pl.col("field_open").is_not_null())
    agree = int((both["n_paid_places"] == both["by_field"]).sum())
    return PaidPlacesSummary(
        races=out.height,
        by_source={s: int((out["source"] == s).sum()) for s in SOURCES},
        changed_vs_starters={(int(y), s): int(n) for y, s, n in per_year.iter_rows()},
        field_below_starters=d.filter(pl.col("field_open") < pl.col("starters")).height,
        payout_vs_field_open=(agree, both.height - agree),
        payout_races_not_in_history=0 if payouts is None else payouts.join(out, on="race_key", how="anti").height,
        examples_changed=changed.select(list(PAID_PLACES_SCHEMA)).head(5).to_dicts(),
    )


def write_paid_places(
    history_path: Path, ts_race_dir: Path, ts_years: Iterable[int], payouts_path: Path | None, out_path: Path
) -> PaidPlacesSummary:
    """history の全レースの支払着数を out_path に書く。payouts_path が None なら払戻表なし (無いファイルは例外)。"""
    races = pl.scan_parquet(history_path).group_by("race_key").agg(pl.col("starters").first()).collect()
    field_open = load_field_open(ts_race_dir, ts_years)
    payouts = None
    if payouts_path is not None:
        payouts = payout_places(pl.read_parquet(payouts_path, columns=["race_key", "n_paid_places"]))
    out = build_paid_places(races, field_open, payouts)
    write_parquet_atomic(out, out_path)
    return summarize(out, payouts)


def load_paid_places(path: Path) -> pl.DataFrame:
    """paid_places.parquet を読む。無ければ例外 (出走頭数の規則で黙って代用しない)。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません (先に --step paid_places)")
    return pl.read_parquet(path)
