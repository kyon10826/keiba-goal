"""窓 1 の開催日の一覧 (race_id の列挙) と、一覧の取りこぼしの検査 (design §3.3)。

- 一覧の読み方: 空の容器 (post_times.is_empty_container) だけが「レースの無い日」。容器も項目も無い 200 のページは
  ParseError (不完全として TTL 後に取り直す)。
- 疑わしい日 (data/raw/reports/suspicious_days_{窓}.json): 一覧が空の土日・祝日、一覧を読めなかった日、
  (年, 場, 回) ごとの日目の飛び、(日, 場) ごとの R 番号の飛び、結果ページへのリンクでないレース (中止の疑い)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import polars as pl

from keiba_goal.netkeiba.parse_race_list import SCHEMA, parse_race_list
from keiba_goal.netkeiba.post_times import is_empty_container

WEEKDAYS = "月火水木金土日"
# 五輪の年は海の日・スポーツの日・山の日が移動した
MOVED_HOLIDAYS = {2020: (date(2020, 7, 23), date(2020, 7, 24), date(2020, 8, 10)),
                  2021: (date(2021, 7, 22), date(2021, 7, 23), date(2021, 8, 8))}
ENTHRONEMENT_2019 = (date(2019, 5, 1), date(2019, 10, 22))


def read_race_list(text: str, day: str) -> pl.DataFrame:
    """一覧ページ → JRA のレース。空の容器はレースの無い日 (空の表)、それ以外で項目が無ければ ParseError。"""
    if is_empty_container(text):
        return pl.DataFrame(schema=SCHEMA)
    return parse_race_list(text, day)


def _nth_monday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7 + 7 * (n - 1))


def _equinox(year: int, base: float) -> int:
    # 1980〜2099 年の春分・秋分の日の近似式
    return int(base + 0.242194 * (year - 1980) - (year - 1980) // 4)


def jp_holidays(year: int) -> set[date]:
    """日本の祝日・振替休日・国民の休日 (2007 年以降の祝日法)。"""
    if not 2007 <= year <= 2099:
        raise ValueError(f"祝日の計算は 2007〜2099 年だけ: {year}")
    base = {date(year, 1, 1), _nth_monday(year, 1, 2), date(year, 2, 11), date(year, 3, _equinox(year, 20.8431)),
            date(year, 4, 29), date(year, 5, 3), date(year, 5, 4), date(year, 5, 5), _nth_monday(year, 9, 3),
            date(year, 9, _equinox(year, 23.2488)), date(year, 11, 3), date(year, 11, 23)}
    if year <= 2018:
        base.add(date(year, 12, 23))
    elif year >= 2020:
        base.add(date(year, 2, 23))
    if year == 2019:
        base.update(ENTHRONEMENT_2019)
    if year in MOVED_HOLIDAYS:
        base.update(MOVED_HOLIDAYS[year])
    else:
        base.update({_nth_monday(year, 7, 3), _nth_monday(year, 10, 2)})
        if year >= 2016:
            base.add(date(year, 8, 11))
    one = timedelta(days=1)
    kokumin = {d + one for d in base if d + 2 * one in base and d + one not in base}
    furikae = set()
    for d in base:
        if d.weekday() == 6:
            n = d + one
            while n in base:
                n += one
            furikae.add(n)
    return base | kokumin | furikae


@dataclass(frozen=True)
class ListDay:
    date: str                  # YYYYMMDD
    status: str                # races (JRA のレースあり) / empty (空の一覧) / unreadable (取得・解析できない)
    n_races: int
    error: str | None = None   # manifest の error や解析の失敗 (races でも「結果へのリンクでない」など)


def suspicious_days(days: list[ListDay], lists: pl.DataFrame) -> dict:
    """一覧の取りこぼしの疑い。lists は読めた一覧の全行 (parse_race_list の列)。"""
    holidays = set().union(*(jp_holidays(y) for y in {int(d.date[:4]) for d in days}))
    empty_rest_days, unreadable = [], []
    for d in days:
        dt = datetime.strptime(d.date, "%Y%m%d").date()
        if d.status == "unreadable":
            unreadable.append({"date": d.date, "weekday": WEEKDAYS[dt.weekday()], "error": d.error})
        elif d.status == "empty":
            holiday = dt in holidays
            if dt.weekday() >= 5 or holiday:
                empty_rest_days.append({"date": d.date, "weekday": WEEKDAYS[dt.weekday()], "holiday": holiday})
    ids = lists.select("date", "venue", "race_num", "link_page", "nk_race_id",
                       year=pl.col("nk_race_id").str.slice(0, 4).cast(pl.Int16),
                       times=pl.col("nk_race_id").str.slice(6, 2).cast(pl.Int16),
                       daily=pl.col("nk_race_id").str.slice(8, 2).cast(pl.Int16))
    daily_gaps = []
    for (year, venue, times), g in ids.group_by("year", "venue", "times", maintain_order=True):
        seen = sorted(set(g["daily"].to_list()))
        missing = sorted(set(range(seen[0], seen[-1] + 1)) - set(seen))
        if missing:
            daily_gaps.append({"year": year, "venue": venue, "times": times, "missing_daily": missing,
                               "daily_seen": seen})
    race_gaps = []
    for (day, venue), g in ids.group_by("date", "venue", maintain_order=True):
        nums = sorted(g["race_num"].to_list())
        if nums != list(range(1, nums[-1] + 1)):
            race_gaps.append({"date": day, "venue": venue, "race_nums": nums})
    not_result = ids.filter(pl.col("link_page").ne_missing("result")).select("date", "nk_race_id", "link_page")
    return {
        "days": len(days),
        "by_status": {s: sum(d.status == s for d in days) for s in ("races", "empty", "unreadable")},
        "empty_weekend_or_holiday": empty_rest_days,
        "unreadable_days": unreadable,
        "daily_gaps": daily_gaps,
        "race_num_gaps": race_gaps,
        "races_not_linked_to_result": not_result.to_dicts(),
        "days_with_errors": [asdict(d) for d in days if d.status == "races" and d.error],
    }
