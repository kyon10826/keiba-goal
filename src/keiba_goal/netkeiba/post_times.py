"""開催日のレース一覧 (race_list_sub) から発走時刻を集める (design §3.2 の T-x 抽出に使う)。

    python -m keiba_goal.netkeiba.post_times [--start 2007-07-01] [--end 2025-09-15] [--limit N] [--dry-run]

- 対象日は record_data にレースがある日だけ。1 日 1 GET (全プロセス共通の 2.5 秒間隔のレート制限を通る)。
- 一覧のレースは 12 桁 race_id で、record_data の**同じ日**のレースと突き合わせ、一致したものだけ出力する。
  - 日付の違う一致は使わない: 2008-02-09 京都 1・2R は JD では 2/9 10:11 に締め切られているのに、一覧は
    代替開催日 2/11 の番組に同じ回日のレースとして載せている (時刻は合っても日付が違う)。
  - 一覧にだけあるレース (中止で record に無い。例 2008-02-03 京都 4R、別の日の record にある例) と、
    record にだけあるその日のレースは報告だけする。差がある日も「完全」(complete=1、error に差の内容) と
    し取り直さない。
- 中身が空の一覧 (空の div.RaceList_Box だけ) は「netkeiba に一覧が無い日」(2007-07 下旬より前) として扱う。
  キャッシュがあれば取り直さない。網羅が始まった後の日に空の一覧が 5 日続いたら、ブロックやメンテナンスを
  疑って止める。容器そのものが無いページ (メンテナンス・エラー) は不完全とし、取り直す。
- 1 つの URL の失敗 (FetchError) はその日を不完全として記録して次へ進む。3 日続けて失敗するか、
  FetchBlocked (連続失敗の上限) なら止める。
- --offline はネットワークに触れず、キャッシュだけで判定と出力をやり直す。
- 出力: data/interim/post_times_racelist.parquet (crawl.POST_SCHEMA、source='race_list') と
  data/raw/reports/post_times_racelist.json。crawl combine がほかの post_times_*.parquet と一緒に
  post_times.parquet にまとめる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from keiba_goal.config import Config, load_config
from keiba_goal.data.record import write_parquet_atomic
from keiba_goal.netkeiba.common import ParseError, has_class, parse_doc
from keiba_goal.netkeiba.http import FetchBlocked, FetchError, Fetcher, Page, Target
from keiba_goal.netkeiba.parse_race_list import PARSER_VERSION, parse_race_list
from keiba_goal.netkeiba.records import read_record_range

OUTPUT = "post_times_racelist.parquet"
REPORT = "post_times_racelist.json"
DEFAULT_START = "2007-07-01"   # これより前の一覧は空 (2007-07-01 の一覧も空、2007-12-22 は 36 レース)
MAX_EMPTY_AFTER_COVERAGE = 5
MAX_FAILED_DAYS = 3
POST_SCHEMA: dict[str, pl.DataType] = {
    "race_key": pl.Int64, "nk_race_id": pl.Utf8, "date": pl.Int32, "post_time": pl.Utf8,
    "post_datetime": pl.Datetime("ms"), "source": pl.Utf8,
}


class Stop(RuntimeError):
    """クロールを止める (理由を表示して終了コード 2)。"""


@dataclass
class DayResult:
    date: str
    status: str              # complete | partial | empty | broken | failed | uncached
    detail: str | None = None
    rows: pl.DataFrame | None = None


@dataclass
class Summary:
    days: int = 0
    gets: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    first_covered: str | None = None

    def add(self, r: DayResult) -> None:
        self.by_status[r.status] = self.by_status.get(r.status, 0) + 1
        ex = self.examples.setdefault(r.status, [])
        if r.status != "complete" and (len(ex) < 20 or r.status == "partial"):
            ex.append(f"{r.date}: {r.detail}")
        if r.status in ("complete", "partial") and (self.first_covered is None or r.date < self.first_covered):
            self.first_covered = r.date


def _ymd(s: str) -> int:
    return int(s.replace("-", ""))


@dataclass(frozen=True)
class RecordRaces:
    days: dict[str, set[str]]       # record の日付 → その日の nk_race_id
    race_key: dict[str, int]        # nk_race_id → race_key (順延でも record の元の日付のまま)


def record_races(cfg: Config, start: int, end: int) -> RecordRaces:
    """record_data にレースがある日と、nk_race_id → race_key。順延を拾えるよう ID の表は前後 14 日広く読む。"""
    def shift(ymd: int, days: int) -> int:
        return int((datetime.strptime(str(ymd), "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d"))

    lo, hi = shift(start, -14), shift(end, 14)
    rec = read_record_range(cfg.paths.record_dir, lo, hi, ["nk_race_id", "race_key", "date"]).unique(["race_key"])
    if rec["nk_race_id"].n_unique() != rec.height:
        raise Stop("record_data の nk_race_id が一意でない")
    inside = rec.filter(pl.col("date").is_between(start, end))
    days = inside.group_by("date").agg(pl.col("nk_race_id")).sort("date")
    return RecordRaces({str(d): set(ids) for d, ids in days.iter_rows()},
                       dict(zip(rec["nk_race_id"].to_list(), rec["race_key"].to_list())))


def is_empty_container(text: str) -> bool:
    """中身が空のレース一覧 (#RaceTopRace の div.RaceList_Box に項目が 1 つも無い)。"""
    try:
        doc = parse_doc(text)
    except ParseError:
        return False
    boxes = doc.xpath(f'//*[@id="RaceTopRace"]//div[{has_class("RaceList_Box")}]')
    items = doc.xpath(f'//li[{has_class("RaceList_DataItem")}]')
    return bool(boxes) and not items


def judge(page: Page, date: str, rr: RecordRaces) -> DayResult:
    """1 日分の一覧を record_data のレースと race_id で突き合わせる。"""
    expected = rr.days.get(date, set())
    if not page.ok:
        return DayResult(date, "broken", f"http {page.status}")
    text = page.text
    if is_empty_container(text):
        return DayResult(date, "empty", f"一覧が空 / record {len(expected)} レース")
    try:
        df = parse_race_list(text, date)
    except ParseError as e:
        return DayResult(date, "broken", str(e))
    if df.is_empty():
        return DayResult(date, "broken", "レース一覧の容器が無い (メンテナンス・エラーページの疑い)")
    matched = df.filter(pl.col("nk_race_id").is_in(list(expected)))
    rest = set(df["nk_race_id"]) - expected
    other_day = sorted(r for r in rest if r in rr.race_key)
    list_only = sorted(r for r in rest if r not in rr.race_key)
    record_only = sorted(expected - set(df["nk_race_id"]))
    not_result = matched.filter(pl.col("link_page") != "result")["nk_race_id"].to_list()
    if not_result:
        return DayResult(date, "broken", f"結果ページへのリンクでないレース {not_result[:5]}")
    rows = matched.select(
        race_key=pl.col("nk_race_id").replace_strict(rr.race_key, return_dtype=pl.Int64),
        nk_race_id=pl.col("nk_race_id"),
        date=pl.lit(int(date), pl.Int32),
        post_time=pl.col("post_time"),
        post_datetime=(pl.lit(date) + pl.col("post_time")).str.strptime(pl.Datetime("ms"), "%Y%m%d%H:%M"),
        source=pl.lit("race_list"),
    ).cast(POST_SCHEMA)
    if list_only or record_only or other_day:
        return DayResult(date, "partial", f"一覧のみ (record に無い) {list_only[:5]} / record では別の日 {other_day[:5]}"
                                          f" / record のみ (一覧に無い) {record_only[:5]}", rows)
    return DayResult(date, "complete", None, rows)


def _mark(fetcher: Fetcher, target: Target, r: DayResult) -> None:
    if fetcher.offline or fetcher.manifest.get(target.url) is None:
        return
    complete = r.status in ("complete", "partial", "empty")
    fetcher.mark(target, complete=complete, parser_version=PARSER_VERSION, error=None if r.status == "complete" else r.detail)


def _needs_fetch(page: Page | None, date: str, rr: RecordRaces) -> bool:
    """キャッシュが無い・HTTP エラー・一覧として読めない (容器が無い) ときだけ取り直す。"""
    return page is None or judge(page, date, rr).status == "broken"


def run(cfg: Config, start: int, end: int, *, limit: int | None, dry_run: bool, offline: bool = False,
        fetcher: Fetcher | None = None) -> Summary:
    rr = record_races(cfg, start, end)
    days = rr.days
    s = Summary(days=len(days))
    if dry_run:
        f = fetcher or Fetcher.from_config(cfg, offline=True)
        todo = [d for d in days if _needs_fetch(f.cached(Target("race_list", d)), d, rr)]
        print(f"対象 {len(days):,} 日、キャッシュで足りる {len(days) - len(todo):,} 日、取得予定 {len(todo):,} GET "
              f"(約 {len(todo) * 3.0 / 3600:.1f} 時間)")
        return s
    fetcher = fetcher or Fetcher.from_config(cfg, offline=offline)
    results: list[DayResult] = []
    empty_after_cov = failed_days = 0
    t0 = time.time()
    for i, date in enumerate(days, 1):
        target = Target("race_list", date)
        page = fetcher.cached(target)
        refetch = _needs_fetch(page, date, rr)
        if refetch and fetcher.offline:
            r = DayResult(date, "uncached", "オフライン: キャッシュが無いか読めない")
            results.append(r)
            s.add(r)
            continue
        if refetch:
            if limit is not None and s.gets >= limit:
                break
            try:
                page = fetcher.get(target)
                s.gets = fetcher.network_requests
                failed_days = 0
            except FetchBlocked as e:
                raise Stop(f"連続失敗の上限で停止: {e}") from e
            except FetchError as e:
                s.gets = fetcher.network_requests
                failed_days += 1
                r = DayResult(date, "failed", str(e))
                results.append(r)
                s.add(r)
                if failed_days >= MAX_FAILED_DAYS:
                    raise Stop(f"{failed_days} 日続けて取得に失敗したので停止 (最後: {e})") from e
                continue
        r = judge(page, date, rr)
        _mark(fetcher, target, r)
        results.append(r)
        s.add(r)
        if r.status == "empty" and s.first_covered is not None:
            empty_after_cov += 1
            if empty_after_cov >= MAX_EMPTY_AFTER_COVERAGE:
                raise Stop(f"網羅の開始 ({s.first_covered}) 後に空の一覧が {empty_after_cov} 日続いた (最後: {date})。"
                           "ブロックかメンテナンスを疑って停止")
        elif r.status in ("complete", "partial"):
            empty_after_cov = 0
        if r.status == "broken" and refetch:
            raise Stop(f"{date}: 一覧として読めないページ ({r.detail})。ブロックかメンテナンスを疑って停止")
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i:,}/{len(days):,} 日 GET {s.gets:,} 経過 {el / 60:.1f} 分 {s.by_status}", flush=True)
    write_outputs(cfg, results, s)
    return s


def write_outputs(cfg: Config, results: list[DayResult], s: Summary) -> None:
    frames = [r.rows for r in results if r.rows is not None]
    df = pl.concat(frames) if frames else pl.DataFrame(schema=POST_SCHEMA)
    if df["race_key"].n_unique() != df.height:
        raise Stop("race_key が重複した (同じ日の突き合わせなので起きないはず。一覧の解析を確認)")
    bad_day = df.filter((pl.col("race_key") // 10**8).cast(pl.Int32) != pl.col("date"))
    if bad_day.height:
        raise Stop(f"race_key の日付と一覧の日付が違う行がある: {bad_day.head(3).to_dicts()}")
    write_parquet_atomic(df.sort("race_key"), cfg.paths.interim / OUTPUT)
    rep = cfg.paths.raw / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / REPORT).write_text(json.dumps({
        "days": s.days, "gets": s.gets, "by_status": s.by_status, "first_covered": s.first_covered,
        "races": df.height, "examples": s.examples,
    }, ensure_ascii=False, indent=1))
    print(f"発走時刻 {df.height:,} レース → {cfg.paths.interim / OUTPUT}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default="2025-09-15")
    ap.add_argument("--limit", type=int, default=None, help="この実行の GET 数の上限")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true", help="キャッシュだけで判定と出力をやり直す (GET しない)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        s = run(cfg, _ymd(a.start), _ymd(a.end), limit=a.limit, dry_run=a.dry_run, offline=a.offline)
    except Stop as e:
        print(f"停止: {e}", file=sys.stderr)
        return 2
    if not a.dry_run:
        print(f"対象 {s.days:,} 日、GET {s.gets:,}、内訳 {s.by_status}、網羅の開始 {s.first_covered}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
