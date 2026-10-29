"""保存ページの「完全」の判定 (design §3.3)。取得 (crawl) と作り直し (rebuild) が同じ規則を使う。

- 結果ページ: 解析でき、単勝・複勝 (出走 5 頭以上) の払戻があり、行数 = 登録頭数 (分かるとき)。
- 出馬表 (shutuba_past): 解析できる (父・母・騎手・調教師・斤量の欠けは parse_shutuba が ParseError にする)。
- 開催日の一覧: 空の容器 (レースの無い日) か、全レースが結果ページへのリンク。容器も項目も無いページは不完全。
"""

from __future__ import annotations

from keiba_goal.netkeiba.common import ParseError
from keiba_goal.netkeiba.http import Fetcher, Page, Target
from keiba_goal.netkeiba.parse_race_list import PARSER_VERSION as LIST_VERSION
from keiba_goal.netkeiba.parse_race_list import is_complete_past_list
from keiba_goal.netkeiba.parse_result import PARSER_VERSION as RESULT_VERSION
from keiba_goal.netkeiba.parse_result import ResultPage, parse_result
from keiba_goal.netkeiba.parse_shutuba import PARSER_VERSION as CARD_VERSION
from keiba_goal.netkeiba.parse_shutuba import parse_shutuba_past
from keiba_goal.netkeiba.race_days import read_race_list

VERSIONS = {"result": RESULT_VERSION, "shutuba_past": CARD_VERSION, "race_list": LIST_VERSION}


def error_text(e: ValueError) -> str:
    return str(e) if isinstance(e, ParseError) else f"{type(e).__name__}: {e}"


def result_error(result: ResultPage, expected_entries: int | None) -> str | None:
    error = result.completeness()
    if error is None and expected_entries is not None and result.rows.height != expected_entries:
        error = f"行数 {result.rows.height} ≠ 登録頭数 {expected_entries}"
    return error


def page_error(page: Page, expected_entries: int | None = None) -> str | None:
    """完全なら None、そうでなければ理由 (manifest の error に入れる)。ネットワーク・manifest に触れない。"""
    t = page.target
    if not page.ok:
        return page.error or (f"http {page.status}" if page.status else "取得できていない")
    try:
        if t.kind == "result":
            return result_error(parse_result(page.text, t.key), expected_entries)
        if t.kind == "shutuba_past":
            parse_shutuba_past(page.text, t.key)
            return None
        if t.kind == "race_list":
            return None if is_complete_past_list(read_race_list(page.text, t.key)) else "結果へのリンクでない"
    except ValueError as e:
        # ParseError のほか int()/float() が想定外の値で失敗しても、そのページの失敗にとどめる
        return error_text(e)
    raise TypeError(f"検査できない kind: {t.kind}")


def mark(fetcher: Fetcher, page: Page, expected_entries: int | None = None) -> tuple[bool, str | None]:
    """page_error で判定し、manifest に complete・今のパーサの版・理由を付ける。"""
    error = page_error(page, expected_entries)
    if fetcher.manifest.get(page.target.url) is not None:
        fetcher.mark(page.target, complete=error is None, parser_version=VERSIONS[page.target.kind], error=error)
    return error is None, error


def done_page(fetcher: Fetcher, target: Target) -> Page | None:
    """取り直しも検査し直しも要らないページ: manifest で完全、今のパーサの版、キャッシュのファイルが無事。

    ファイルが消えた・壊れた (Fetcher.cached が None) ページは None (取り直す)。
    """
    rec = fetcher.manifest.get(target.url)
    if rec is None or not rec.complete or rec.parser_version != VERSIONS[target.kind]:
        return None
    return fetcher.cached(target)


def card_entries(fetcher: Fetcher, nk_race_id: str) -> int | None:
    """キャッシュの出馬表の頭数 (取消を含む登録頭数)。無い・読めなければ None。"""
    page = fetcher.cached(Target("shutuba_past", nk_race_id))
    if page is None or not page.ok:
        return None
    try:
        return parse_shutuba_past(page.text, nk_race_id).horses.height
    except ValueError:
        return None
