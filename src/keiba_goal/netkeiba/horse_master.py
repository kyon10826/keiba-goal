"""馬マスタ: 馬 ID → 馬名・性・生年・父・母・出典 (design §3.3)。

record_data で初期化し (出典 'record')、shutuba_past の馬で更新する。record_data に無い馬は 'netkeiba'。
父・母は record の値を優先する (JV は外国産馬の親を外国名で持つ。例 'Hat Trick' と netkeiba の 'ハットトリック')。
shutuba_past の母父は特徴量に使わない参考列 (broodmare_sire_ref)。

マスタは毎回「record_data の初期値 + これまでに読んだ出馬表の馬 (card_horses)」から作り直す。保存済みの
マスタを更新し続けると、record の読み込みが変わったとき (引用符の扱いなど) に古い名前が残るため。

出馬表だけの経路 (ライブ、convert.convert_card) の父・母は parent_names(マスタ, card_horses) で決める: マスタに
いる馬 (出典を問わない) はマスタの名前、いない外国産馬は作り直しと同じ別名 (learn_aliases) を当てる。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from keiba_goal.netkeiba.convert import ParentNames, jv_parent_name
from keiba_goal.netkeiba.parse_shutuba import CardPage
from keiba_goal.netkeiba.records import record_frames

SCHEMA: dict[str, pl.DataType] = {
    "horse_id": pl.Int64, "horse": pl.Utf8, "sex": pl.Utf8, "birth_year": pl.Int16, "father": pl.Utf8,
    "mother": pl.Utf8, "broodmare_sire_ref": pl.Utf8, "foreign_bred": pl.Boolean, "first_date": pl.Int32,
    "last_date": pl.Int32, "source": pl.Utf8,
}
_LATEST = ("horse", "sex", "father", "mother")
SEEN_SCHEMA: dict[str, pl.DataType] = {
    "horse_id": pl.Int64, "horse": pl.Utf8, "sex": pl.Utf8, "father": pl.Utf8, "mother": pl.Utf8,
    "broodmare_sire": pl.Utf8, "foreign_bred": pl.Boolean, "first_date": pl.Int32, "last_date": pl.Int32,
}


def _collapse(df: pl.DataFrame) -> pl.DataFrame:
    """(horse_id, date, 各列) → 馬ごとに最新の値と初出・最終日。"""
    return (df.sort("last_date").group_by("horse_id").agg(
        *[pl.col(c).drop_nulls().last() for c in _LATEST],
        first_date=pl.col("first_date").min(), last_date=pl.col("last_date").max()))


def seed_from_record(record_dir: Path, before: int | None = None) -> pl.DataFrame:
    """record_data の馬。before (YYYYMMDD) を渡すとその日より前の行だけ使う (ゲートで答えを見ないため)。"""
    parts = []
    for df in record_frames(record_dir, last_year=None if before is None else before // 10000):
        if before is not None:
            df = df.filter(pl.col("date") < before)
        parts.append(_collapse(df.select(pl.col("id").alias("horse_id"), *_LATEST,
                                         first_date=pl.col("date"), last_date=pl.col("date"))))
    empty = pl.DataFrame(schema={c: SCHEMA[c] for c in ("horse_id", *_LATEST, "first_date", "last_date")})
    m = _collapse(pl.concat(parts)) if parts else empty
    return m.with_columns(birth_year=(pl.col("horse_id") // 1_000_000).cast(pl.Int16),
                          broodmare_sire_ref=pl.lit(None, pl.Utf8), foreign_bred=pl.lit(None, pl.Boolean),
                          source=pl.lit("record")).select(list(SCHEMA)).cast(SCHEMA).sort("horse_id")


def card_horses(date: int, card: CardPage) -> pl.DataFrame:
    """出馬表 1 枚から馬マスタの更新に要る列だけ (開催日 YYYYMMDD 付き)。"""
    return card.horses.select(
        "horse_id", "horse", "sex",
        *[pl.col(c).map_elements(jv_parent_name, return_dtype=pl.Utf8) for c in ("father", "mother", "broodmare_sire")],
        "foreign_bred",
    ).with_columns(first_date=pl.lit(date, pl.Int32), last_date=pl.lit(date, pl.Int32)).cast(SEEN_SCHEMA)


def merge_seen(*frames: pl.DataFrame | None) -> pl.DataFrame:
    """card_horses() の表を縦に並べ、同じ (馬, 開催日) は後の表の行を残す。"""
    parts = [f.select(list(SEEN_SCHEMA)).cast(SEEN_SCHEMA) for f in frames if f is not None and f.height]
    if not parts:
        return pl.DataFrame(schema=SEEN_SCHEMA)
    return pl.concat(parts).unique(["horse_id", "first_date"], keep="last", maintain_order=True)


def load_seen(path: Path) -> pl.DataFrame:
    return merge_seen(pl.read_parquet(path)) if path.exists() else pl.DataFrame(schema=SEEN_SCHEMA)


def update_from_cards(master: pl.DataFrame, cards: Iterable[tuple[int, CardPage]]) -> pl.DataFrame:
    """出馬表 (shutuba_past) の馬で更新する。cards は (開催日 YYYYMMDD, CardPage)。"""
    return update_from_seen(master, merge_seen(*(card_horses(date, card) for date, card in cards)))


def update_from_seen(master: pl.DataFrame, seen: pl.DataFrame) -> pl.DataFrame:
    """card_horses() を縦に並べた表で更新する。

    外国産馬の親は JV が外国名で持つ (netkeiba は日本で走った馬などをカナで出す)。record にいる外国産馬から
    「カナ名 → record の名前」を学び、record にいない外国産馬の父・母に当てる。
    """
    if seen.height == 0:
        return master
    seen = apply_aliases(seen, learn_aliases(master, seen), _record_ids(master))
    extra = seen.sort("last_date").group_by("horse_id").agg(
        bms=pl.col("broodmare_sire").drop_nulls().last(), foreign=pl.col("foreign_bred").drop_nulls().last())
    nk = _collapse(seen.drop("broodmare_sire", "foreign_bred")).join(extra, on="horse_id")
    j = master.join(nk, on="horse_id", how="full", suffix="_nk", coalesce=True)
    newer = pl.col("last_date_nk").is_not_null() & (pl.col("last_date").is_null()
                                                    | (pl.col("last_date_nk") >= pl.col("last_date")))
    out = j.select(
        "horse_id",
        *[pl.when(newer).then(pl.col(f"{c}_nk")).otherwise(pl.col(c)).fill_null(pl.col(f"{c}_nk")).alias(c)
          for c in ("horse", "sex")],
        # record の父・母を優先し、無い馬だけ netkeiba の値で埋める
        *[pl.coalesce(pl.col(c), pl.col(f"{c}_nk")).alias(c) for c in ("father", "mother")],
        pl.coalesce(pl.col("bms"), pl.col("broodmare_sire_ref")).alias("broodmare_sire_ref"),
        pl.coalesce(pl.col("foreign"), pl.col("foreign_bred")).alias("foreign_bred"),
        pl.min_horizontal("first_date", "first_date_nk").alias("first_date"),
        pl.max_horizontal("last_date", "last_date_nk").alias("last_date"),
        pl.col("source").fill_null("netkeiba"),
        birth_year=(pl.col("horse_id") // 1_000_000).cast(pl.Int16),
    )
    return out.select(list(SCHEMA)).cast(SCHEMA).sort("horse_id")


def _record_ids(master: pl.DataFrame) -> pl.Series:
    return master.filter(pl.col("source") == "record")["horse_id"]


def learn_aliases(master: pl.DataFrame, seen: pl.DataFrame) -> dict[str, dict[str, str]]:
    """外国産馬の親の名前: 出馬表のカナ → record の名前 ('father' / 'mother')。

    record にいる外国産馬の出馬表の名前と record の名前が違う組から学ぶ。1 つのカナに record の名前が 2 つ以上
    対応すれば使わない。
    """
    known = master.filter(pl.col("source") == "record").select("horse_id", rec_f="father", rec_m="mother")
    pairs = seen.filter(pl.col("foreign_bred")).join(known, on="horse_id")
    out: dict[str, dict[str, str]] = {}
    for col, rec in (("father", "rec_f"), ("mother", "rec_m")):
        diff = pairs.filter(pl.col(col).is_not_null() & pl.col(rec).is_not_null() & (pl.col(col) != pl.col(rec)))
        alias = diff.group_by(col).agg(pl.col(rec).unique()).filter(pl.col(rec).list.len() == 1)
        out[col] = dict(zip(alias[col].to_list(), alias[rec].list.first().to_list()))
    return out


def apply_aliases(seen: pl.DataFrame, aliases: dict[str, dict[str, str]], record_ids: pl.Series) -> pl.DataFrame:
    """record にいない外国産馬の父・母を別名で置き換える。"""
    newcomer = pl.col("foreign_bred") & ~pl.col("horse_id").is_in(record_ids.implode())
    return seen.with_columns(pl.when(newcomer).then(pl.col(c).replace(m)).otherwise(pl.col(c)).alias(c)
                             for c, m in aliases.items() if m)


def parents_map(master: pl.DataFrame) -> dict[int, tuple[str | None, str | None]]:
    """作り直しの変換器に渡す父・母 (record 由来の馬だけ。行の父・母は最後に馬マスタの名前にそろえる)。"""
    rec = master.filter(pl.col("source") == "record")
    return {r[0]: (r[1], r[2]) for r in rec.select("horse_id", "father", "mother").iter_rows()}


def parent_names(master: pl.DataFrame, seen: pl.DataFrame) -> ParentNames:
    """出馬表だけの経路 (convert_card) の父・母の決め方。master は保存した馬マスタ、seen は card_horses。

    作り直しで行の父・母になるのは、マスタにいる馬はマスタの名前、新しい外国産馬は learn_aliases の別名。
    """
    known = {r[0]: (r[1], r[2]) for r in master.select("horse_id", "father", "mother").iter_rows()}
    return ParentNames(known, learn_aliases(master, seen))
