"""ライブの入口: 当日 D の出馬表の行から、batch と同じグループ関数で D のレースの特徴量を作る。

朝 (prepare_morning) に段階 A (前日までの履歴 + 出馬表の静的な列) と全グループの補助出力を作っておき、各レースの
判断時刻 (decide) には段階 B (オッズ・馬体重・騎手・斤量・馬場/天気・レース内相対) だけを最新の出馬表で作る。
build_rows はその 2 つを続けて呼ぶ基準の実装 (batch とのパリティ検査の基準)。

card_rows: 当日の出馬表 (netkeiba.convert.convert_card の行 = schema.HISTORY_DTYPES の列、レース後の列は null) +
    scratched (Boolean: 判断時刻までに取消・除外と分かっている馬。出馬表の取消の印)。
snapshots: ts_snap 形式 (race_key, horse_num, x ∈ {120,60,30,6}, win, place_lo, place_hi) の判断時刻までのオッズ。
field_open: (race_key, field_open) = 発売開始時の頭数 (design §4.1: 出馬表の登録頭数 − 前日発売の開始前に取消だった馬)。
    None ならスナップショットでオッズの付いた馬の数 (batch で paid_places が無いときと同じ代替)。

朝の状態を判断時刻に使ってよいのは、段階 A と補助出力が D の行の当日変わりうる列 (registry.STAGE_B_COLUMNS) に
依らないから (as-of 検査の (d) と、朝と判断時刻で出馬表が違う場合のパリティ検査 test_features_live.py が確かめる)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from keiba_goal.data.schema import HISTORY_DTYPES, POST_RACE_COLUMNS
from keiba_goal.features.decision import decision_frame, with_live_columns
from keiba_goal.features.pipeline import HelperCache, checked_groups, frame_columns, stage_a, stage_b
from keiba_goal.features.registry import Group

COLUMNS: tuple[str, ...] = (*HISTORY_DTYPES, "scratched")


def card_frame(card_rows: pl.DataFrame) -> pl.DataFrame:
    """出馬表の行を検査して HISTORY_DTYPES + scratched にそろえる。"""
    if "scratched" not in card_rows.columns or card_rows["scratched"].null_count():
        raise ValueError("card_rows には scratched (取消・除外と分かっている馬) が全行に必要です")
    post = [c for c in POST_RACE_COLUMNS if c in card_rows.columns and card_rows[c].null_count() != card_rows.height]
    if post:
        raise ValueError(f"card_rows にレース後の列の値があります {post} (出馬表の行はレース後の列を null にする)")
    missing = [c for c in HISTORY_DTYPES if c not in card_rows.columns and c not in POST_RACE_COLUMNS]
    if missing:
        raise ValueError(f"card_rows に発走前の列がありません {missing}")
    return card_rows.select(
        *(pl.col(c).cast(t) if c in card_rows.columns else pl.lit(None, dtype=t).alias(c) for c, t in HISTORY_DTYPES.items()),
        pl.col("scratched").cast(pl.Boolean),
    )


def card_date(card: pl.DataFrame) -> int:
    dates = card["date"].unique().to_list()
    if len(dates) != 1:
        raise ValueError(f"card_rows は 1 日分にする ({sorted(dates)})")
    return int(dates[0])


@dataclass(frozen=True)
class Morning:
    """朝の状態: D の行の段階 A の特徴量と、全グループの補助出力 (D より前の履歴 + 朝の出馬表から作ったもの)。"""

    date: int
    groups: tuple[Group, ...]
    history: pl.LazyFrame
    helpers: HelperCache
    stage_a: Mapping[str, pl.DataFrame]
    rows: pl.DataFrame          # 朝の出馬表の (race_id, race_key)

    @property
    def year(self) -> int:
        return self.date // 10_000


def prepare_morning(history_before_d: pl.DataFrame | pl.LazyFrame, card_rows: pl.DataFrame, *,
                    groups: Sequence[Group] | None = None) -> Morning:
    """朝の出馬表 (その日の全レース) から段階 A と補助出力を作る。履歴は D より前の行だけであること。"""
    card = card_frame(card_rows)
    d = card_date(card)
    hist = with_live_columns(history_before_d.lazy()).select(COLUMNS)
    last = hist.select(pl.col("date").max()).collect().item()
    if last is not None and last >= d:
        raise ValueError(f"history_before_d に {d} 以降の行があります (最終日 {last})")
    combined = pl.concat([hist, card.lazy()])
    groups = checked_groups(groups)
    helpers = HelperCache(groups, combined)
    helpers.build_all()
    a_outputs = stage_a(combined, groups, helpers, (d // 10_000,), (d,))
    return Morning(d, groups, combined, helpers, a_outputs, card.select("race_id", "race_key"))


def decide(morning: Morning, card_rows: pl.DataFrame, snapshots: pl.DataFrame | None,
           field_open: pl.DataFrame | None = None) -> pl.DataFrame:
    """判断時刻: 最新の出馬表 card_rows (朝の出馬表のレースのうち 1 つ以上、各レースは全馬) の行の全特徴量。"""
    card = card_frame(card_rows)
    if (d := card_date(card)) != morning.date:
        raise ValueError(f"card_rows の日付 {d} が朝の状態の日付 {morning.date} と違う")
    if extra := card.join(morning.rows, on="race_id", how="anti").height:
        raise ValueError(f"card_rows に朝の出馬表に無い行が {extra} 行あります")
    races = morning.rows.join(card.select("race_key").unique(), on="race_key", how="semi")
    if races.height != card.height:
        raise ValueError("card_rows のレースは全馬の行を渡す (レース内の相対値・頭数がレースの全馬で決まるため)")
    rows = card.select(frame_columns(morning.groups)).sort(["date", "race_key", "horse_num"])
    frame = decision_frame(rows, snapshots, field_open)
    return stage_b(morning.history, morning.groups, morning.helpers, frame, morning.stage_a, morning.year)


def build_rows(
    history_before_d: pl.DataFrame | pl.LazyFrame,
    card_rows: pl.DataFrame,
    snapshots: pl.DataFrame | None,
    field_open: pl.DataFrame | None = None,
    *,
    groups: Sequence[Group] | None = None,
) -> pl.DataFrame:
    """D (card_rows の日付、1 日だけ) の行の特徴量 (OUTPUT_KEYS + 全特徴量)。履歴は D より前の行だけであること。"""
    return decide(prepare_morning(history_before_d, card_rows, groups=groups), card_rows, snapshots, field_open)
