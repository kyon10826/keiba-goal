"""race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD (開催日のレース一覧) の解析。

race_id はリンクから正規表現 `race_id=(\\d{12})` で取り、場コード 01-10 (JRA) だけ残す。
項目 (li.RaceList_DataItem) が 1 つも無いページは ParseError。レースの無い日の一覧は「空の容器」
(post_times.is_empty_container) なので、呼ぶ側が先にそれを判定する (race_days.read_race_list)。容器も項目も無い
200 のページ (メンテナンス・エラー・ブロック) をレースの無い日と取り違えないため。
過去日の一覧は全レースが結果ページ (result.html) へのリンクになっていれば「完全」。
"""

from __future__ import annotations

import re

import polars as pl

from keiba_goal.data.schema import CODE_TO_PLACE
from keiba_goal.netkeiba.common import ParseError, first, has_class, norm, parse_doc, text_of

PARSER_VERSION = "race_list-2"
_RACE_ID = re.compile(r"race_id=(\d{12})")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
_PAGE_KIND = re.compile(r"/race/(\w+)\.html")

SCHEMA: dict[str, pl.DataType] = {
    "nk_race_id": pl.Utf8, "date": pl.Int32, "venue_code": pl.Int8, "venue": pl.Utf8, "race_num": pl.Int8,
    "race_name": pl.Utf8, "post_time": pl.Utf8, "n_entries": pl.Int16, "course": pl.Utf8, "link_page": pl.Utf8,
}


def parse_race_list(text: str, date: str) -> pl.DataFrame:
    """1 開催日のレース一覧 (JRA の場だけ)。項目が 1 つも無いページは ParseError。"""
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date は YYYYMMDD: {date!r}")
    doc = parse_doc(text)
    items = doc.xpath(f'//li[{has_class("RaceList_DataItem")}]')
    if not items:
        raise ParseError(f"{date}: レース一覧の項目が無い (空の一覧の容器でもない: メンテナンス・エラーページの疑い)")
    recs = []
    for li in items:
        hrefs = li.xpath(".//a/@href")
        main = next((h for h in hrefs if _RACE_ID.search(h) and "movie" not in h), None)
        if main is None:
            raise ParseError(f"{date}: race_id のリンクが無い項目がある")
        rid = _RACE_ID.search(main).group(1)
        venue_code = int(rid[4:6])
        if venue_code not in CODE_TO_PLACE:
            continue
        num = re.sub(r"\D", "", text_of(first(li, f'.//div[{has_class("Race_Num")}]')))
        post = norm(text_of(first(li, f'.//span[{has_class("RaceList_Itemtime")}]')))
        tm = _TIME.match(post)
        if not num or tm is None:
            raise ParseError(f"{date}: {rid} の R 番号・発走時刻を解釈できない: {num!r} {post!r}")
        if int(num) != int(rid[10:12]):
            raise ParseError(f"{date}: {rid} の R 番号 {num} が race_id と合わない")
        heads = re.sub(r"\D", "", text_of(first(li, f'.//span[{has_class("RaceList_Itemnumber")}]')))
        page = _PAGE_KIND.search(main)
        recs.append({
            "nk_race_id": rid,
            "date": int(date),
            "venue_code": venue_code,
            "venue": CODE_TO_PLACE[venue_code],
            "race_num": int(num),
            "race_name": text_of(first(li, f'.//span[{has_class("ItemTitle")}]')),
            "post_time": f"{int(tm.group(1)):02d}:{tm.group(2)}",
            "n_entries": int(heads) if heads else None,
            "course": text_of(first(li, f'.//span[{has_class("RaceList_ItemLong")}]')) or None,
            "link_page": page.group(1) if page else None,
        })
    df = pl.DataFrame(recs, schema=SCHEMA)
    if df["nk_race_id"].n_unique() != df.height:
        raise ParseError(f"{date}: race_id が重複")
    return df.sort("venue_code", "race_num")


def is_complete_past_list(df: pl.DataFrame) -> bool:
    """過去日の一覧: 全レースのリンクが結果ページ。空の表 (空の容器の日、JRA のレースが無い日) も完全。"""
    return df.is_empty() or bool((df["link_page"] == "result").all())
