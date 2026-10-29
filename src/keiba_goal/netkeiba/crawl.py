"""netkeiba の補完クロール (design §3.3 の窓)。

    python -m keiba_goal.netkeiba.crawl --window gate|1|2|3 [--limit N] [--dry-run] [--parse-only] [--no-rebuild]
    python -m keiba_goal.netkeiba.crawl combine
    python -m keiba_goal.netkeiba.crawl check-shutuba-past-history

1. 取得: manifest で完全・今のパーサの版・キャッシュのファイルが無事なページは飛ばす。それ以外はまずキャッシュの
   写しを今のパーサで検査し直し (ネットワークなし)、通れば完全にする。通らなければ GET する (不完全なページは TTL を
   過ぎてから。ファイルが消えた・壊れたページはすぐ)。--limit N はこの実行で扱うレース数の上限 (0 ならネットワークに
   触れない)。
   - 1 つの URL が再試行の上限まで失敗したら、manifest に不完全 (最後の HTTP 状態と理由) として記録して次へ進む。
   - 止めるのは、リクエストの失敗が URL をまたいで max_consecutive_failures 回続いたとき (FetchBlocked) と、200 だが
     検査に通らないページが max_consecutive_invalid_pages 件続いたとき (ソフトブロック・メンテナンス・レイアウト
     変更)。どちらも終了コード 2 で、作り直しはしない。
2. 作り直し (rebuild): キャッシュの完全なページから data/interim/{nk_rows,nk_payouts,post_times}_{窓}.parquet と
   馬マスタを作る。作り直しは毎回キャッシュの全ページを解析し直す (窓 3 の約 6.6 万レースで約 1 時間) ので、
   再開を重ねる取得は --no-rebuild (1 だけ) で進め、ページがそろってから 1 回 --parse-only (2 だけ。manifest にも
   ネットワークにも触れない) で作り直す。--dry-run は取得も書き込みもせず計画だけ表示する。
3. まとめ (combine): 窓ごとの出力と開催日一覧の発走時刻を P1 が読む data/interim/{nk_rows,payouts,post_times}.parquet
   にまとめる。窓 1・2・3 の作り直しの最後に自動で行う (ゲートは行わないが、まとめるときはゲートの出力も含める)。
作り直し・馬マスタの更新・まとめは data/raw/build.lock で直列化する。
窓 1 は一覧の取りこぼしの疑いを data/raw/reports/suspicious_days_1.json に書く。ゲートの窓は record_data と全列を
比べた結果を表示し、data/raw/reports/gate_report.json に保存する。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

import polars as pl

from keiba_goal.config import Config, load_config
from keiba_goal.netkeiba import files, gate, horse_master, rebuild
from keiba_goal.netkeiba.http import JST, FetchBlocked, FetchError, Fetcher, Page, Target
from keiba_goal.netkeiba.parse_race_list import SCHEMA as LIST_SCHEMA
from keiba_goal.netkeiba.parse_shutuba import parse_shutuba_past
from keiba_goal.netkeiba.race_days import ListDay, read_race_list, suspicious_days
from keiba_goal.netkeiba.records import read_record_range
from keiba_goal.netkeiba.validate import card_entries, done_page, error_text, mark, page_error

SECONDS_PER_GET = 3.0   # 間隔 2.5 秒 + 応答 (ETA の目安)
RACES_PER_CALENDAR_DAY = 3456 / 365   # JRA は年 3,456 レース前後 (窓 1 の見積もり用)
GATE = "gate"
RACES_SCHEMA = rebuild.RACES_SCHEMA


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    end: int
    pages: tuple[str, ...]
    ids: str            # 'record' | 'race_list'


def jst_today() -> date:
    return datetime.now(JST).date()


def _ymd(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def load_window(cfg: Config, name: str, today: date) -> Window:
    spec = (cfg.section("netkeiba").get("windows") or {}).get(name)
    if spec is None:
        raise SystemExit(f"config の netkeiba.windows に窓 {name!r} が無い")
    end = _ymd(today - timedelta(days=1)) if spec["end"] == "today" else int(spec["end"].replace("-", ""))
    return Window(name, int(spec["start"].replace("-", "")), end, tuple(spec["pages"]), spec["ids"])


def record_races(cfg: Config, w: Window) -> pl.DataFrame:
    """record_data の行から窓のレース (12 桁 ID・日付・登録頭数)。"""
    rec = read_record_range(cfg.paths.record_dir, w.start, w.end, ["race_key", "nk_race_id", "date", "horse_N"])
    races = rec.group_by("race_key").agg(pl.col("nk_race_id", "date", "horse_N").first())
    return races.sort("race_key").select(list(RACES_SCHEMA)).cast(RACES_SCHEMA)


def dates_between(start: int, end: int) -> list[str]:
    d0 = datetime.strptime(str(start), "%Y%m%d").date()
    d1 = datetime.strptime(str(end), "%Y%m%d").date()
    return [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]


class CrawlStopped(RuntimeError):
    """200 だが検査に通らないページが続いた (ソフトブロック・メンテナンス・レイアウト変更の疑い)。"""


@dataclass
class Crawl:
    """1 回の取得: 要るときだけ GET し、失敗は記録して進み、検査に通らない 200 が続いたら止める。"""

    fetcher: Fetcher
    invalid_streak: int = 0
    failed: list[str] = field(default_factory=list)

    def page(self, target: Target, expected_entries: int | None = None) -> Page | None:
        """ページを用意する。取得に失敗したら None (manifest に不完全として記録済み)。"""
        f = self.fetcher
        page = done_page(f, target)
        if page is not None:
            return page
        cached = f.cached(target)
        if cached is not None and cached.ok and mark(f, cached, expected_entries)[0]:
            return replace(cached, complete=True, error=None)   # 今のパーサで通った: ネットワークに触れない
        try:
            page = f.get(target)
        except FetchBlocked:
            raise
        except FetchError as e:
            f.record_failure(target, e)
            self.failed.append(target.url)
            print(f"  {target.kind} {target.key}: 取得できない ({e.reason or e})。不完全と記録して次へ", file=sys.stderr)
            return None
        if page.from_cache:
            return page   # TTL 内の不完全なページ (まだ取り直さない)
        ok, error = mark(f, page, expected_entries)
        if not ok:
            print(f"  {target.kind} {target.key}: 不完全 ({error})", file=sys.stderr)
        if page.ok:
            self.invalid_streak = 0 if ok else self.invalid_streak + 1
            if self.invalid_streak >= f.settings.max_consecutive_invalid_pages:
                raise CrawlStopped(f"200 だが検査に通らないページが {self.invalid_streak} 件続いた (最後: {target.url}: "
                                   f"{error})。ソフトブロック・メンテナンス・レイアウト変更を疑って停止")
        return page


def needs_get(fetcher: Fetcher, target: Target, expected_entries: int | None = None) -> bool:
    """GET が要るか (dry-run の見積もり。manifest に書かない)。キャッシュの写しが今のパーサで完全なら要らない。"""
    if done_page(fetcher, target) is not None:
        return False
    cached = fetcher.cached(target)
    return cached is None or not cached.ok or page_error(cached, expected_entries) is not None


@dataclass(frozen=True)
class RaceLists:
    races: pl.DataFrame        # RACES_SCHEMA: 結果ページへのリンクのあるレース
    lists: pl.DataFrame        # 読めた一覧の全行 (parse_race_list.SCHEMA)
    days: list[ListDay]


def _read_day(day: str, page: Page | None, missing: str) -> tuple[ListDay, pl.DataFrame | None]:
    if page is None:
        return ListDay(day, "unreadable", 0, missing), None
    if not page.ok:
        return ListDay(day, "unreadable", 0, page.error or f"http {page.status}"), None
    try:
        df = read_race_list(page.text, day)
    except ValueError as e:
        return ListDay(day, "unreadable", 0, error_text(e)), None
    error = None if page.complete else page.error
    return ListDay(day, "empty" if df.is_empty() else "races", df.height, error), df


def listed_races(fetcher: Fetcher, w: Window, crawl: Crawl | None = None, want_new: int | None = None) -> RaceLists:
    """窓 1: 日付ごとの一覧から race_id を列挙する。crawl を渡すと要るときだけ一覧を取得し、無ければキャッシュだけ使う。

    want_new: 取得が要るレースがこの数に達したら一覧の取得をやめる (--limit 用)。
    結果ページへのリンクでないレース (過去日なら中止の疑い) は取得しない (suspicious_days に載る)。
    """
    found: list[pl.DataFrame] = []
    days: list[ListDay] = []
    n_new = 0
    for d in dates_between(w.start, w.end):
        if want_new is not None and n_new >= want_new:
            break
        target = Target("race_list", d)
        page = crawl.page(target) if crawl is not None else fetcher.cached(target)
        day, df = _read_day(d, page, "取得に失敗 (manifest に記録)" if crawl is not None else "キャッシュに無い")
        days.append(day)
        if df is None:
            continue
        found.append(df)
        if want_new is not None:
            linked = df.filter(pl.col("link_page") == "result")["nk_race_id"]
            n_new += sum(any(done_page(fetcher, Target(k, rid)) is None for k in w.pages) for rid in linked)
    lists = pl.concat(found) if found else pl.DataFrame(schema=LIST_SCHEMA)
    races = (lists.filter(pl.col("link_page") == "result")
             .select("nk_race_id", "date", pl.lit(None, pl.Int16).alias("horse_N")).cast(RACES_SCHEMA)
             # 同じレースが 2 つの日付の一覧に出たら最初の日付だけ残す (変換は結果ページの日付を使う)
             .unique("nk_race_id", keep="first", maintain_order=True))
    return RaceLists(races, lists, days)


def fetch_races(crawl: Crawl, races: pl.DataFrame, pages: tuple[str, ...], limit: int | None) -> int:
    """完全でないページを用意する。扱ったレース数を返す。"""
    f = crawl.fetcher
    todo = [r for r in races.iter_rows(named=True)
            if any(done_page(f, Target(k, r["nk_race_id"])) is None for k in pages)]
    if limit is not None:
        todo = todo[:limit]
    # 出馬表を先に: record_data が無い窓 1 は出馬表の頭数 (取消を含む) で結果の行数を確かめる
    order = sorted(pages, key=lambda k: k != "shutuba_past")
    t0 = time.monotonic()
    for i, r in enumerate(todo, start=1):
        rid = r["nk_race_id"]
        for kind in order:
            expected = None
            if kind == "result":
                expected = r["horse_N"] if r["horse_N"] is not None else card_entries(f, rid)
            crawl.page(Target(kind, rid), expected)
        if i % 10 == 0 or i == len(todo):
            el = time.monotonic() - t0
            eta = el / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] {rid}  経過 {el / 60:.1f} 分  残り約 {eta / 60:.1f} 分", flush=True)
    return len(todo)


def dry_run(fetcher: Fetcher, w: Window, races: pl.DataFrame, list_gets: int | None) -> None:
    need = {k: sum(needs_get(fetcher, Target(k, r["nk_race_id"]), r["horse_N"] if k == "result" else None)
                   for r in races.iter_rows(named=True)) for k in w.pages}
    gets = sum(need.values()) + (list_gets or 0)
    print(f"窓 {w.name}: {w.start}〜{w.end}、ページ {list(w.pages)}")
    if list_gets is not None:
        unlisted = len(dates_between(w.start, w.end)) - races["date"].n_unique()
        guess = round(unlisted * RACES_PER_CALENDAR_DAY)
        gets += guess * len(w.pages)
        print(f"  一覧を取得する日数: {list_gets} (列挙済みのレース {races.height}、未列挙の日から約 {guess} レースの見込み)")
    else:
        print(f"  レース数: {races.height}")
    print(f"  GET が要るページ: {need}  → 約 {gets} GET、約 {gets * SECONDS_PER_GET / 3600:.1f} 時間")


def check_history(cfg: Config) -> int:
    rid = str(cfg.section("netkeiba").get("history_check_race_id", "202509040311"))
    fetcher = Fetcher.from_config(cfg)
    page = fetcher.get(Target("shutuba_past", rid))
    mark(fetcher, page)
    card = parse_shutuba_past(page.text, rid)
    h = card.horses
    rec = read_record_range(cfg.paths.record_dir, int(rid[:4] + "0101"), int(rid[:4] + "1231"),
                            ["nk_race_id", "horse_num", "blinker", "weight"])
    rec = rec.filter(pl.col("nk_race_id") == rid)
    j = h.join(rec.select("horse_num", pl.col("blinker").alias("bl_rec"), pl.col("weight").alias("w_rec")),
               on="horse_num", how="left")
    n_marks = h["blinker"].drop_nulls().len()
    n_weights = h["weight"].drop_nulls().len()
    bl_ok = int(j.select(pl.col("blinker").eq_missing(pl.col("bl_rec")).sum()).item())
    w_ok = int(j.select(((pl.col("weight") - pl.col("w_rec")).abs() < 0.5).sum()).item())
    print(f"shutuba_past {rid} ({'キャッシュ' if page.from_cache else '取得'}, {page.fetched_at:%Y-%m-%d}):"
          f" {h.height} 頭、ブリンカー印 {n_marks}、馬体重 {n_weights}/{h.height}")
    print(f"  record_data との一致: ブリンカー {bl_ok}/{h.height}、馬体重 {w_ok}/{h.height}")
    same_day = n_weights == h.height and bl_ok == h.height and w_ok == h.height
    print("  → レース当日の内容を返している" if same_day else "  → レース当日の内容ではない (父・母は血統ページ、前走ブリンカーは NaN に)")
    return 0 if same_day else 1


def write_suspicious(cfg: Config, w: Window, lists: RaceLists) -> dict:
    rep = {"window": w.name, "start": w.start, "end": w.end, **suspicious_days(lists.days, lists.lists)}
    files.save_json(rep, cfg.paths.raw / "reports" / f"suspicious_days_{w.name}.json")
    n = {k: len(rep[k]) for k in ("empty_weekend_or_holiday", "unreadable_days", "daily_gaps", "race_num_gaps",
                                  "races_not_linked_to_result")}
    print(f"一覧の検査 ({rep['by_status']}): 疑わしいもの {n} → reports/suspicious_days_{w.name}.json")
    return rep


def rebuild_window(cfg: Config, w: Window, races: pl.DataFrame) -> None:
    """キャッシュから窓の出力・馬マスタを作り直す (ゲートは比較の報告、窓 1〜3 はまとめまで)。"""
    cache = Fetcher.from_config(cfg, offline=True)
    with files.build_lock(cfg.paths.raw):
        seed = horse_master.seed_from_record(cfg.paths.record_dir)
        if w.name == GATE:
            # ゲートは答え (窓の record 行) を見ない: 窓より前の record だけで作った馬マスタで変換する
            b = rebuild.build(cfg, cache, w.pages, races,
                              horse_master.seed_from_record(cfg.paths.record_dir, before=w.start))
        else:
            others = horse_master.load_seen(cfg.paths.raw / rebuild.CARD_HORSES).filter(
                ~pl.col("first_date").is_between(w.start, w.end))
            b = rebuild.build(cfg, cache, w.pages, races, seed, others)
        rebuild.write_outputs(cfg, w.name, b)
        master = rebuild.update_master(cfg, w.start, w.end, seed, b)
        print(f"出力: nk_rows_{w.name} {b.rows.height} 行 / nk_payouts_{w.name} {b.payouts.height} 行 / "
              f"post_times_{w.name} {b.post_times.height} レース、作り直しのエラー {len(b.errors)} 件"
              f" (reports/build_errors_{w.name}.json)、馬マスタ {master.height} 頭")
        if w.name == GATE:
            gate.print_gate(gate.gate_report(cfg, w.start, w.end, b))
        else:
            print(f"まとめ: {rebuild.combine(cfg)} 発走時刻の出どころ {rebuild.post_time_sources(cfg)}")


def run(cfg: Config, name: str, limit: int | None, dry: bool, parse_only: bool, no_rebuild: bool = False) -> int:
    w = load_window(cfg, name, jst_today())
    online = not (dry or parse_only or limit == 0)
    fetcher = Fetcher.from_config(cfg, offline=not online)
    crawl = Crawl(fetcher) if online else None
    try:
        if w.ids == "record":
            races = record_races(cfg, w)
        else:
            races = listed_races(fetcher, w, crawl, want_new=limit).races
        if dry:
            list_gets = None if w.ids == "record" else sum(
                needs_get(fetcher, Target("race_list", d)) for d in dates_between(w.start, w.end))
            dry_run(fetcher, w, races, list_gets)
            return 0
        if crawl is not None:
            n = fetch_races(crawl, races, w.pages, limit)
            print(f"取得: {n} レース、ネットワーク {fetcher.network_requests} GET (manifest の累計 "
                  f"{fetcher.manifest.request_count()})")
            if crawl.failed:
                print(f"  取得できなかった URL {len(crawl.failed)} 件 (manifest に不完全として記録。TTL 後に取り直す)"
                      f" 例 {crawl.failed[:3]}", file=sys.stderr)
    except FetchBlocked as e:
        print(f"停止: {e}", file=sys.stderr)
        return 2
    except CrawlStopped as e:
        print(f"停止: {e}", file=sys.stderr)
        return 2
    if w.ids == "race_list":
        lists = listed_races(Fetcher.from_config(cfg, offline=True), w)
        write_suspicious(cfg, w, lists)
        races = lists.races
    if no_rebuild:
        print("作り直しは省略 (--no-rebuild)。ページがそろったら --parse-only で作り直す")
        return 0
    rebuild_window(cfg, w, races)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keiba_goal.netkeiba.crawl", description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", default="run", choices=["run", "combine", "check-shutuba-past-history"])
    ap.add_argument("--window", choices=["gate", "1", "2", "3"])
    ap.add_argument("--limit", type=int, help="この実行で扱うレース数の上限")
    ap.add_argument("--dry-run", action="store_true", help="取得も書き込みもせず計画だけ表示")
    ap.add_argument("--parse-only", action="store_true",
                    help="ネットワークに触れず manifest にも書かずに、キャッシュの完全なページから作り直す")
    ap.add_argument("--no-rebuild", action="store_true", help="取得だけ (作り直し・まとめはしない)")
    ap.add_argument("--config", help="設定ファイル (既定 config/default.yaml)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "check-shutuba-past-history":
        return check_history(cfg)
    if args.command == "combine":
        print(f"まとめ: {rebuild.combine(cfg)} 発走時刻の出どころ {rebuild.post_time_sources(cfg)}")
        return 0
    if args.window is None:
        ap.error("--window が必要")
    if args.parse_only and args.no_rebuild:
        ap.error("--parse-only と --no-rebuild は同時に使えない")
    return run(cfg, args.window, args.limit, args.dry_run, args.parse_only, args.no_rebuild)


if __name__ == "__main__":
    sys.exit(main())
