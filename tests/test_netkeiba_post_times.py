"""開催日一覧からの発走時刻 (netkeiba.post_times): 空の一覧・壊れたページ・同じ日だけの突き合わせ。"""

from __future__ import annotations

import gzip
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from keiba_goal.netkeiba.http import JST, Page, Target
from keiba_goal.netkeiba.post_times import RecordRaces, is_empty_container, judge

FX = Path(__file__).parent / "fixtures" / "netkeiba"


def _page(name: str | None, date: str, status: int = 200, body: bytes | None = None) -> Page:
    content = body if body is not None else gzip.decompress((FX / name).read_bytes())
    return Page(Target("race_list", date), status, content, datetime(2026, 10, 29, tzinfo=JST), True, None)


def _ids_20080203() -> list[str]:
    # 2008-02-03 の一覧: 東京 1 回 4 日・京都 2 回 2 日・小倉 1 回 6 日。京都 4R は中止で record に無い
    return ([f"20080501040{r:1d}" if r < 10 else f"2008050104{r:02d}" for r in range(1, 13)]
            + [f"20080802020{r:1d}" if r < 10 else f"2008080202{r:02d}" for r in range(1, 13) if r != 4])


def test_empty_container_is_a_day_without_list():
    p = _page("race_list_20040105_empty.html.gz", "20040105")
    assert is_empty_container(p.text)
    r = judge(p, "20040105", RecordRaces({"20040105": {"200406010101"}}, {"200406010101": 2004010506010101}))
    assert r.status == "empty" and r.rows is None


def test_page_without_container_is_broken_not_empty():
    body = "<html><body>ただいまメンテナンス中です</body></html>".encode()
    p = _page(None, "20080203", body=body)
    assert not is_empty_container(p.text)
    assert judge(p, "20080203", RecordRaces({}, {})).status == "broken"
    assert judge(_page(None, "20080203", status=500, body=b""), "20080203", RecordRaces({}, {})).status == "broken"


def test_same_day_match_and_report_of_differences():
    p = _page("race_list_20080203.html.gz", "20080203")
    listed = set(pl.Series(_ids_20080203()))
    full = {nk for nk in listed}
    # record にあるのは京都 4R を除いた日。東京は省略して「record のみ」と「一覧のみ」の両方を作る
    kyoto = {nk for nk in full if nk[4:6] == "08"}
    other_day = "200808020204"   # 京都 4R が別の日の record にあることにする
    rr = RecordRaces({"20080203": kyoto | {"200808020299"}},
                     {**{nk: int("20080203" + nk[4:]) for nk in kyoto}, "200808020299": 2008020308020299,
                      other_day: 2008021008020204})
    r = judge(p, "20080203", rr)
    assert r.status == "partial"
    assert "200808020204" in r.detail and "200808020299" in r.detail
    assert set(r.rows["nk_race_id"]) == kyoto - set()            # 同じ日の record にあるレースだけ
    assert (r.rows["race_key"] // 10**8 == 20080203).all()
    assert r.rows["post_time"].str.contains(r"^\d{2}:\d{2}$").all()
    assert r.rows["post_datetime"].dt.date().cast(pl.Utf8).str.replace_all("-", "").cast(pl.Int32).eq(20080203).all()


def test_complete_when_lists_agree():
    p = _page("race_list_20080203.html.gz", "20080203")
    from keiba_goal.netkeiba.parse_race_list import parse_race_list
    df = parse_race_list(p.text, "20080203")
    # 中止の京都 4R は結果ページでなく出馬表へのリンク (record に無いので突き合わせの対象外)
    assert df.filter(pl.col("link_page") != "result")["nk_race_id"].to_list() == ["200808020204"]
    ids = set(df.filter(pl.col("link_page") == "result")["nk_race_id"])
    rr = RecordRaces({"20080203": ids}, {nk: int("20080203" + nk[4:]) for nk in ids})
    r = judge(p, "20080203", rr)
    assert r.status == "partial" and r.rows.height == len(ids)   # 中止レースは「一覧のみ」として報告
    assert "200808020204" in r.detail
    # record にある (=実施された) レースが結果ページへのリンクでなければ壊れたページ扱い
    rr_bad = RecordRaces({"20080203": ids | {"200808020204"}},
                         {nk: int("20080203" + nk[4:]) for nk in ids | {"200808020204"}})
    assert judge(p, "20080203", rr_bad).status == "broken"


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_page_is_broken(bad):
    assert judge(_page(None, "20080203", body=bad.encode()), "20080203", RecordRaces({}, {})).status == "broken"
