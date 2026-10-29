"""キャッシュから窓の出力を作り直す (design §3.3 の作り直し・§3.4 の払戻)。

- 使うのは manifest で完全なページだけ。今のパーサでも完全か確かめ (行数 = 登録頭数、単勝・複勝の払戻)、使えない
  ページは build_errors_{窓}.json に (nk_race_id, date, stage, error) で載せる (そのレースの行・払戻・発走時刻は
  出さない)。P6 は払戻の無いレースがある日を丸ごと除く。
- n_paid_places は払戻表で払われた着数 (2 か 3。同着はタイブレーク)。複勝が売られたはず (出走 5 頭以上) なのに
  複勝の払戻が無いときは null (0 にしない)。0 は出走 4 頭以下 (複勝の発売なし)。
- 発走時刻の source は 'result' (結果ページ)。combine は開催日一覧の発走時刻 (post_times_racelist.parquet、
  source='race_list') と合わせ、同じレースは 'result' を優先する。
- 書き込み (窓ごとの出力・card_horses・馬マスタ・まとめ) は files.build_lock の中で行う。一時ファイルの名前は一意。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from keiba_goal.config import Config
from keiba_goal.data.record import write_parquet_parts
from keiba_goal.data.schema import HISTORY_DTYPES
from keiba_goal.netkeiba import course_table, files, horse_master
from keiba_goal.netkeiba.convert import convert_race
from keiba_goal.netkeiba.http import Fetcher, Target
from keiba_goal.netkeiba.parse_result import ResultPage, parse_result
from keiba_goal.netkeiba.parse_shutuba import CardPage, parse_shutuba_past
from keiba_goal.netkeiba.validate import error_text, result_error

CARD_HORSES = "card_horses.parquet"   # data/raw/ 以下。読んだ出馬表の馬 (馬マスタの作り直し用)
RACES_SCHEMA = {"nk_race_id": pl.Utf8, "date": pl.Int32, "horse_N": pl.Int16}
CARD_PARENTS_SCHEMA = {"nk_race_id": pl.Utf8, "horse_num": pl.Int8, "horse_id": pl.Int64, "father": pl.Utf8,
                       "mother": pl.Utf8, "broodmare_sire": pl.Utf8}
PAYOUT_SCHEMA: dict[str, pl.DataType] = {
    "race_key": pl.Int64, "nk_race_id": pl.Utf8, "bet_type": pl.Utf8, "horse_num": pl.Int8, "combo": pl.Utf8,
    "numbers": pl.List(pl.Int8), "payout_per_100": pl.Int32, "popularity": pl.Int16, "n_paid_places": pl.Int8,
    "source": pl.Utf8, "fetched_at": pl.Datetime("ms", "Asia/Tokyo"),
}
POST_SCHEMA: dict[str, pl.DataType] = {
    "race_key": pl.Int64, "nk_race_id": pl.Utf8, "date": pl.Int32, "post_time": pl.Utf8,
    "post_datetime": pl.Datetime("ms"), "source": pl.Utf8,
}
POST_SOURCE_PRIORITY = {"result": 0, "race_list": 1}   # 小さいほど優先 (結果ページ > 開催日の一覧)
# (窓ごとのファイルの接頭辞, まとめたファイル, 一意のキー, 型)
COMBINED: tuple[tuple[str, str, tuple[str, ...], dict[str, pl.DataType]], ...] = (
    ("nk_rows", "nk_rows.parquet", ("race_id",), HISTORY_DTYPES),
    ("nk_payouts", "payouts.parquet", ("race_key", "bet_type", "combo"), PAYOUT_SCHEMA),
    ("post_times", "post_times.parquet", ("race_key",), POST_SCHEMA),
)


class CombineError(ValueError):
    pass


def paid_places(result: ResultPage) -> int | None:
    """複勝の支払着数: 払戻のある馬 = 着順 k 以内の馬 となる k (2 か 3)。

    複勝の払戻が無いとき: 出走 4 頭以下なら 0 (発売なし)、5 頭以上なら None (読めていない)。合う k が無ければ None。
    2 着同着 (1・2・2 着に払戻) や 3 頭の 1 着同着では k = 2 と 3 の両方が合う。そのときは登録頭数 8 頭以上なら 3
    (JRA は発売時の出走頭数 8 頭以上で 3 着払い。発売後の取消では減らない: 202309040202 は 8 頭登録・7 頭出走で 3 着払い)。
    """
    paid = set(result.payouts.filter(pl.col("bet_type") == "place")["numbers"].list.first().to_list())
    if not paid:
        return 0 if result.starters <= 4 else None
    ranks = dict(result.rows.select("horse_num", "rank").iter_rows())
    fits = [k for k in (2, 3) if paid == {h for h, r in ranks.items() if 1 <= r <= k}]
    if len(fits) == 2:
        return 3 if result.rows.height >= 8 else 2
    return fits[0] if fits else None


def _race_key(result: ResultPage) -> int:
    return int(f"{result.race.date:08d}{result.race.nk_race_id[4:12]}")


def payout_frame(result: ResultPage, fetched_at: datetime) -> pl.DataFrame:
    single = pl.col("bet_type").is_in(["win", "place"])
    return result.payouts.with_columns(
        race_key=pl.lit(_race_key(result), pl.Int64), nk_race_id=pl.lit(result.race.nk_race_id),
        horse_num=pl.when(single).then(pl.col("numbers").list.first()),
        n_paid_places=pl.lit(paid_places(result), pl.Int8), source=pl.lit("netkeiba"),
        fetched_at=pl.lit(fetched_at)).select(list(PAYOUT_SCHEMA)).cast(PAYOUT_SCHEMA)


def post_time_frame(result: ResultPage) -> pl.DataFrame:
    race = result.race
    dt = datetime.strptime(f"{race.date} {race.post_time}", "%Y%m%d %H:%M")
    return pl.DataFrame([{"race_key": _race_key(result), "nk_race_id": race.nk_race_id, "date": race.date,
                          "post_time": race.post_time, "post_datetime": dt, "source": "result"}], schema=POST_SCHEMA)


class _Chunks:
    """小さな表をためて一定数ごとに縦に結合する (窓 3 の約 6.6 万レースでもメモリを抑える)。"""

    def __init__(self, schema: dict[str, pl.DataType], every: int = 500):
        self.schema, self.every, self.parts, self.pending = schema, every, [], []

    def add(self, df: pl.DataFrame) -> None:
        self.pending.append(df)
        if len(self.pending) >= self.every:
            self.parts.append(pl.concat(self.pending, rechunk=True))
            self.pending = []

    def frame(self) -> pl.DataFrame:
        parts = self.parts + ([pl.concat(self.pending)] if self.pending else [])
        return pl.concat(parts, rechunk=True) if parts else pl.DataFrame(schema=self.schema)


@dataclass
class BuildResult:
    rows: pl.DataFrame
    payouts: pl.DataFrame
    post_times: pl.DataFrame
    card_parents: pl.DataFrame
    seen: pl.DataFrame                 # 出馬表の馬 (horse_master.SEEN_SCHEMA)
    known_horses: frozenset[int]       # 父・母を馬マスタ (record) から採った馬
    errors: list[dict] = field(default_factory=list)


def _usable_card(fetcher: Fetcher, rid: str) -> tuple[CardPage | None, str | None]:
    page = fetcher.cached(Target("shutuba_past", rid))
    if page is None:
        return None, "出馬表がキャッシュに無い (未取得、またはファイルが消えた・壊れた) ので行を作らない"
    if not page.complete:
        return None, f"出馬表が manifest で不完全 ({page.error or '未検査'}) なので行を作らない"
    try:
        return parse_shutuba_past(page.text, rid), None
    except ValueError as e:
        return None, error_text(e)


def _usable_result(fetcher: Fetcher, rid: str, expected: int | None) -> tuple[ResultPage | None, datetime | None,
                                                                             str | None]:
    page = fetcher.cached(Target("result", rid))
    if page is None:
        return None, None, "キャッシュに無い (未取得、またはファイルが消えた・壊れた)"
    if not page.complete:
        return None, None, f"manifest で不完全 ({page.error or '未検査'})"
    try:
        result = parse_result(page.text, rid)
    except ValueError as e:
        return None, None, error_text(e)
    error = result_error(result, expected)
    if error is not None:
        return None, None, f"今のパーサで不完全 ({error})"
    return result, page.fetched_at, None


def build(cfg: Config, fetcher: Fetcher, pages: tuple[str, ...], races: pl.DataFrame, seed: pl.DataFrame,
          other_seen: pl.DataFrame | None = None) -> BuildResult:
    """キャッシュから窓の行・払戻・発走時刻を作る (書き込みはしない。manifest も読むだけ)。

    seed: record_data 由来の馬マスタ。ここにいる馬の父・母は record の名前、いない馬は出馬表の名前
    (外国産馬は seed から学んだ別名に直す)。other_seen: ほかの窓の出馬表の馬 (別名の学習と名前をマスタに揃える用)。
    """
    courses = course_table.load_or_build(cfg.paths.interim / "course_table.parquet", cfg.paths.record_dir)
    parents = horse_master.parents_map(seed)
    need_card = "shutuba_past" in pages
    first_seen = None if need_card else dict(seed.select("horse_id", "first_date").iter_rows())
    rows, pays, posts = _Chunks(HISTORY_DTYPES), _Chunks(PAYOUT_SCHEMA), _Chunks(POST_SCHEMA)
    seen, card_par = _Chunks(horse_master.SEEN_SCHEMA), _Chunks(CARD_PARENTS_SCHEMA)
    errors: list[dict] = []

    def fail(r: dict, stage: str, e: ValueError | str) -> None:
        errors.append({"nk_race_id": r["nk_race_id"], "date": r["date"], "stage": stage,
                       "error": e if isinstance(e, str) else error_text(e)})

    for r in races.iter_rows(named=True):
        rid = r["nk_race_id"]
        card, card_error = _usable_card(fetcher, rid) if need_card else (None, None)
        expected = r["horse_N"] if r["horse_N"] is not None else (card.horses.height if card is not None else None)
        result, fetched_at, error = _usable_result(fetcher, rid, expected)
        if result is None:
            fail(r, "result", error)
            continue
        try:
            pay, post = payout_frame(result, fetched_at), post_time_frame(result)
        except ValueError as e:
            fail(r, "result", e)
            continue
        pays.add(pay)
        posts.add(post)
        if need_card:
            if card is None:
                # 父・母・ブリンカー・本賞金が埋まらない行は履歴に入れない
                fail(r, "shutuba_past", card_error)
                continue
            seen.add(horse_master.card_horses(result.race.date, card))
            card_par.add(card.horses.select(pl.lit(rid).alias("nk_race_id"), *list(CARD_PARENTS_SCHEMA)[1:]))
        try:
            rows.add(convert_race(result, card, courses, parents, first_seen))
        except ValueError as e:
            fail(r, "convert", e)
    seen_df = seen.frame()
    out_rows = rows.frame()
    if seen_df.height:
        # 行の父・母を馬マスタと同じにする (record にいる馬は record の名前、新しい外国産馬は学んだ別名)
        master = horse_master.update_from_seen(seed, horse_master.merge_seen(other_seen, seen_df))
        names = master.select(pl.col("horse_id").alias("id"), father_m="father", mother_m="mother")
        out_rows = out_rows.join(names, on="id", how="left").with_columns(
            father=pl.coalesce("father_m", "father"), mother=pl.coalesce("mother_m", "mother")
        ).select(list(HISTORY_DTYPES))
    return BuildResult(out_rows, pays.frame(), posts.frame(), card_par.frame(), seen_df, frozenset(parents), errors)


def write_outputs(cfg: Config, window: str, b: BuildResult) -> None:
    out = cfg.paths.interim
    files.save_parquet(b.rows.sort("race_id"), out / f"nk_rows_{window}.parquet")
    files.save_parquet(b.payouts.sort("race_key", "bet_type", "combo"), out / f"nk_payouts_{window}.parquet")
    files.save_parquet(b.post_times.sort("race_key"), out / f"post_times_{window}.parquet")
    files.save_json(b.errors, cfg.paths.raw / "reports" / f"build_errors_{window}.json")


def update_master(cfg: Config, start: int, end: int, seed: pl.DataFrame, b: BuildResult) -> pl.DataFrame:
    """読んだ出馬表の馬を保存し、馬マスタを record + 全部の出馬表の馬から作り直して保存する。

    card_horses は書く直前に読み直す (呼ぶ側が build_lock を持つので、ほかの窓の作り直しと混ざらない)。
    """
    path = cfg.paths.raw / CARD_HORSES
    seen_prev = horse_master.load_seen(path)
    seen_all = horse_master.merge_seen(seen_prev.filter(~pl.col("first_date").is_between(start, end)), b.seen)
    files.save_parquet(seen_all, path)
    master = horse_master.update_from_seen(seed, seen_all)
    files.save_parquet(master, cfg.paths.interim / "horse_master.parquet")
    return master


def _combined_year(lf: pl.LazyFrame, year: int, target: str, key: list[str], by_source: bool) -> pl.DataFrame:
    """1 年分。by_source なら同じキーは出どころの優先順 (POST_SOURCE_PRIORITY) で 1 行にする。"""
    df = lf.filter(pl.col("race_key") // 10**12 == year).collect()
    check = key
    if by_source:
        unknown = sorted(set(df["source"].unique().to_list()) - set(POST_SOURCE_PRIORITY), key=str)
        if unknown:
            raise CombineError(f"{target}: 発走時刻の source が {sorted(POST_SOURCE_PRIORITY)} でない: {unknown}")
        df = df.with_columns(_prio=pl.col("source").replace_strict(POST_SOURCE_PRIORITY, return_dtype=pl.Int8))
        check = [*key, "_prio"]
    dup = df.filter(pl.struct(check).is_duplicated())
    if dup.height:
        ex = dup.group_by(check).agg(pl.col("_file").unique()).head(3).to_dicts()
        raise CombineError(f"{target}: 同じキーの行が {dup.height} 行、別々のファイルにある (窓が重なっている?) 例 {ex}")
    if by_source:
        df = df.sort("_prio").unique(key, keep="first", maintain_order=True).drop("_prio")
    return df.drop("_file").sort(key)


def combine(cfg: Config) -> dict[str, int]:
    """窓ごとの出力を P1 が読む 1 つのファイルにまとめる (data/interim/{nk_rows,payouts,post_times}.parquet)。

    - 発走時刻は結果ページ (post_times_{窓}) と開催日の一覧 (post_times_racelist) を合わせ、同じレースは結果ページを採る。
    - それ以外で同じキーの行が 2 つのファイルにあれば止める (窓の重なり。黙ってどちらかを選ばない)。
    """
    out = cfg.paths.interim
    written: dict[str, int] = {}
    with files.build_lock(cfg.paths.raw):
        for prefix, target, key, schema in COMBINED:
            paths = sorted(out.glob(f"{prefix}_*.parquet"))
            if not paths:
                continue
            lf = pl.concat([pl.scan_parquet(p).select(list(schema)).cast(schema).with_columns(_file=pl.lit(p.name))
                            for p in paths])
            years = sorted(lf.select((pl.col("race_key") // 10**12).unique()).collect().to_series().to_list())
            parts = (_combined_year(lf, y, target, list(key), prefix == "post_times") for y in years)
            if years:
                written[target] = write_parquet_parts(parts, out / target)
            else:
                files.save_parquet(pl.DataFrame(schema=schema), out / target)
                written[target] = 0
    return written


def post_time_sources(cfg: Config) -> dict[str, int]:
    path = cfg.paths.interim / "post_times.parquet"
    if not path.exists():
        return {}
    df = pl.read_parquet(path, columns=["source"])
    return dict(df.group_by("source").len().sort("source").iter_rows())
