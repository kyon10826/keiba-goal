"""窓 1 の一覧の検査: 祝日の暦、空の土日・祝日、日目・R 番号の飛び、結果へのリンクでないレース。"""

from datetime import date

import polars as pl
import pytest

from keiba_goal.netkeiba.parse_race_list import SCHEMA
from keiba_goal.netkeiba.race_days import ListDay, jp_holidays, suspicious_days


def test_holidays_include_substitute_and_sandwiched_days():
    h26 = jp_holidays(2026)
    assert {date(2026, 5, 6), date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23), date(2026, 10, 12)} <= h26
    assert date(2026, 9, 22) in h26 and date(2026, 9, 24) not in h26   # 敬老の日と秋分の日に挟まれた国民の休日
    h25 = jp_holidays(2025)
    assert {date(2025, 2, 24), date(2025, 5, 6), date(2025, 9, 15), date(2025, 10, 13), date(2025, 11, 24)} <= h25
    assert len(h25) == 19 and len(h26) == 18
    h19 = jp_holidays(2019)   # 即位の年: 4/30・5/2 は国民の休日、12/23 は休日でない
    assert {date(2019, 4, 30), date(2019, 5, 1), date(2019, 5, 2), date(2019, 10, 22)} <= h19
    assert date(2019, 12, 23) not in h19
    assert {date(2021, 7, 22), date(2021, 7, 23), date(2021, 8, 9)} <= jp_holidays(2021)   # 五輪で移動
    with pytest.raises(ValueError):
        jp_holidays(2006)


def _lists(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """(日付, nk_race_id, リンク先) → parse_race_list の列。"""
    from keiba_goal.data.schema import CODE_TO_PLACE

    return pl.DataFrame([{"nk_race_id": rid, "date": int(d), "venue_code": int(rid[4:6]),
                          "venue": CODE_TO_PLACE[int(rid[4:6])], "race_num": int(rid[10:12]), "race_name": "",
                          "post_time": "10:00", "n_entries": 10, "course": None, "link_page": link}
                         for d, rid, link in rows], schema=SCHEMA)


def test_suspicious_days_report_gaps_empty_rest_days_and_unread_days():
    days = [ListDay("20250920", "races", 24), ListDay("20250921", "races", 23),
            ListDay("20250922", "empty", 0), ListDay("20250923", "empty", 0),     # 月 (平日) / 火 (秋分の日)
            ListDay("20250927", "unreadable", 0, "http 500 ×3"), ListDay("20250928", "empty", 0),
            ListDay("20251004", "races", 12, "結果へのリンクでない")]
    rows = ([("20250920", f"2025060401{r:02d}", "result") for r in range(1, 13)]      # 4 回中山 1 日目
            + [("20250921", f"2025060402{r:02d}", "result") for r in range(1, 13) if r != 5]   # 5R が抜けた
            + [("20251004", f"2025060404{r:02d}", "result" if r != 7 else "shutuba") for r in range(1, 13)])
    rep = suspicious_days(days, _lists(rows))
    assert rep["by_status"] == {"races": 3, "empty": 3, "unreadable": 1}
    assert rep["empty_weekend_or_holiday"] == [{"date": "20250923", "weekday": "火", "holiday": True},
                                               {"date": "20250928", "weekday": "日", "holiday": False}]
    assert rep["unreadable_days"] == [{"date": "20250927", "weekday": "土", "error": "http 500 ×3"}]
    assert rep["daily_gaps"] == [{"year": 2025, "venue": "中山", "times": 4, "missing_daily": [3],
                                  "daily_seen": [1, 2, 4]}]
    assert rep["race_num_gaps"] == [{"date": 20250921, "venue": "中山", "race_nums": [1, 2, 3, 4, 6, 7, 8, 9, 10,
                                                                                     11, 12]}]
    assert rep["races_not_linked_to_result"] == [{"date": 20251004, "nk_race_id": "202506040407",
                                                  "link_page": "shutuba"}]
    assert rep["days_with_errors"] == [{"date": "20251004", "status": "races", "n_races": 12,
                                        "error": "結果へのリンクでない"}]


def test_meetings_of_the_same_number_in_different_years_are_not_merged():
    rows = ([("20250920", f"2025060401{r:02d}", "result") for r in range(1, 13)]
            + [("20260919", f"2026060403{r:02d}", "result") for r in range(1, 13)])
    assert suspicious_days([ListDay("20250920", "races", 12), ListDay("20260919", "races", 12)],
                           _lists(rows))["daily_gaps"] == []
