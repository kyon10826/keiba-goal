import gzip
import math
import re
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from keiba_goal.netkeiba.common import ParseError
from keiba_goal.netkeiba.http import decode
from keiba_goal.netkeiba.parse_odds import parse_odds
from keiba_goal.netkeiba.parse_race_list import SCHEMA as RACE_LIST_SCHEMA
from keiba_goal.netkeiba.parse_race_list import is_complete_past_list, parse_race_list
from keiba_goal.netkeiba.parse_result import (
    N_CELLS,
    parse_body_weight,
    parse_rank,
    parse_result,
    parse_time,
    strip_diary_tags,
)
from keiba_goal.netkeiba.parse_shutuba import parse_shutuba, parse_shutuba_past
from keiba_goal.netkeiba.race_days import read_race_list

FX = Path(__file__).parent / "fixtures" / "netkeiba"


def page(kind: str, key: str) -> str:
    raw = gzip.decompress((FX / f"{kind}_{key}.html.gz").read_bytes())
    return decode(raw, "https://db.netkeiba.com/" if kind == "result" else "https://race.netkeiba.com/")


def result(key: str):
    return parse_result(page("result", key), key)


def card(key: str):
    return parse_shutuba_past(page("shutuba_past", key), key)


# --- race_list_sub ---------------------------------------------------------------------------

def test_race_list_ids_times_and_completeness():
    df = parse_race_list(page("race_list", "20260919"), "20260919")
    assert df.height == 24
    assert set(df["venue"]) == {"中山", "阪神"}
    assert df["nk_race_id"].str.contains(r"^\d{12}$").all()
    first = df.row(0, named=True)
    assert (first["nk_race_id"], first["race_num"], first["post_time"], first["n_entries"]) == (
        "202606040501", 1, "10:00", 9)
    jump = df.filter(pl.col("nk_race_id") == "202609040504").row(0, named=True)
    assert jump["race_name"] == "阪神ジャンプ" and jump["course"] is None
    assert is_complete_past_list(df)
    live = df.with_columns(link_page=pl.lit("shutuba"))
    assert not is_complete_past_list(live)


def test_race_list_drops_non_jra_venues():
    html = ('<ul><li class="RaceList_DataItem"><a href="../race/result.html?race_id=202644050101">'
            '<div class="Race_Num">1R</div><span class="RaceList_Itemtime">10:00</span></a></li></ul>')
    assert parse_race_list(html, "20260919").is_empty()  # 場コード 44 (地方) は捨てる


def test_only_the_verified_empty_container_is_a_day_without_races():
    empty = gzip.decompress((FX / "race_list_20040105_empty.html.gz").read_bytes()).decode()
    df = read_race_list(empty, "20040105")   # 本物の空の一覧 (#RaceTopRace の div.RaceList_Box に項目なし)
    assert df.is_empty() and df.schema == pl.Schema(RACE_LIST_SCHEMA) and is_complete_past_list(df)
    with pytest.raises(ParseError, match="項目が無い"):
        parse_race_list(empty, "20040105")   # 容器の判定は read_race_list の役目
    for page_200 in ("<html><body>ただいまメンテナンス中です</body></html>",
                     "<html><body><div class='RaceList_Body'></div></body></html>",   # 容器の id も箱も無い
                     "<html><body>Access denied</body></html>"):
        with pytest.raises(ParseError, match="項目が無い"):
            read_race_list(page_200, "20260920")
    assert read_race_list(page("race_list", "20260919"), "20260919").height == 24


# --- odds API ---------------------------------------------------------------------------------

def test_odds_result_status_win_and_place():
    o = parse_odds((FX / "odds_result_202606040512.json").read_text(), "202606040512")
    assert o.status == "result" and not o.usable_for_features
    assert o.official_datetime == datetime(2026, 9, 19, 16, 36, 22) and o.update_count == 0
    assert o.win.height == 16 and o.place.height == 16
    w1 = o.win.row(0, named=True)
    assert w1["horse_num"] == 1 and w1["win_odds"] == pytest.approx(4.1) and w1["win_pop"] == 1
    p5 = o.place.filter(pl.col("horse_num") == 5).row(0, named=True)
    assert (p5["place_lo"], p5["place_hi"], p5["place_pop"]) == (pytest.approx(1.5), pytest.approx(2.0), 1)


def test_odds_middle_non_numeric_values_become_nan():
    o = parse_odds((FX / "odds_middle_synthetic.json").read_text(), "202606040512")
    assert o.status == "middle" and o.usable_for_features and o.update_count == 12
    h7 = o.win.filter(pl.col("horse_num") == 7).row(0, named=True)
    assert math.isnan(h7["win_odds"]) and h7["win_pop"] is None
    p3 = o.place.filter(pl.col("horse_num") == 3).row(0, named=True)
    assert p3["place_lo"] == pytest.approx(14.1) and math.isnan(p3["place_hi"])


def test_odds_yoso_has_no_place_and_is_not_usable():
    o = parse_odds((FX / "odds_yoso_synthetic.json").read_text(), "202606040512")
    assert o.status == "yoso" and not o.usable_for_features
    assert o.place.is_empty() and o.win.height == 16 and o.win["win_pop"].null_count() == 16


def test_odds_rejects_unknown_status_and_missing_win():
    with pytest.raises(ParseError, match="status"):
        parse_odds((FX / "odds_bad_status_synthetic.json").read_text(), "202606040512")
    with pytest.raises(ParseError, match="単勝"):
        parse_odds('{"status":"middle","data":{"odds":{}}}', "202606040512")
    with pytest.raises(ParseError, match="JSON"):
        parse_odds("<html>", "202606040512")


# --- shutuba_past / shutuba -------------------------------------------------------------------

def test_shutuba_past_race_data():
    r = card("202306010110").race
    assert (r.post_time, r.surface, r.dist, r.course_tokens) == ("15:01", "芝", 1600, ("右", "外", "C"))
    assert (r.weather, r.going_turf, r.going_dirt) == ("晴", "良", None)
    assert (r.times, r.venue, r.daily) == (1, "中山", 1)
    assert (r.age_text, r.class_text, r.weight_type, r.grade) == ("3歳", "オープン", "別定", "L")
    assert r.prizes == (2000.0, 800.0, 500.0, 300.0, 200.0) and r.n_entries == 13


def test_shutuba_past_obstacle_has_two_goings():
    r = card("202306010208").race
    assert (r.surface, r.dist, r.course_tokens) == ("障", 3200, ("芝", "ダート"))
    assert (r.going_turf, r.going_dirt, r.age_text, r.grade) == ("良", "良", "障害4歳以上", "OP")


def test_shutuba_past_horses_pedigree_blinker_weight():
    c = card("202606040512")
    h = c.horses
    assert h.height == 16 and h["horse_num"].to_list() == list(range(1, 17))
    assert h["blinker"].drop_nulls().len() == 6
    v = h.filter(pl.col("horse_num") == 2).row(0, named=True)
    assert (v["horse_id"], v["horse"], v["blinker"]) == (2023103320, "ヴェネチアンキング", "B")
    assert (v["father"], v["mother"], v["broodmare_sire"]) == ("マジェスティックウォリアー", "アルデンテ", "エンパイアメーカー")
    assert (v["sex"], v["age"], v["coat"], v["basis_weight"]) == ("牡", 3, "鹿", 53.0)
    assert (v["jockey_id"], v["trainer_id"], v["trainer_area"]) == ("01207", "01023", "美浦")
    assert (v["weight"], v["inc_dec"], v["n_past_runs"], v["n_past_starts"]) == (522.0, 12.0, 5, 4)
    past = c.past_runs.filter(pl.col("horse_id") == 2023103320).sort("k")
    assert past.row(0, named=True)["rank_text"] == "取" and past.row(0, named=True)["past_nk_race_id"] == "202606040112"


def test_shutuba_past_prev_weight_unknown_foreign_runs_and_scratch():
    g1 = card("202305010811").horses.filter(pl.col("horse_num") == 2).row(0, named=True)
    assert g1["prev_weight_unknown"] and g1["inc_dec"] is None and g1["weight"] == 470.0
    assert g1["n_past_runs"] == 4  # 海外のレース ID (英数字) も読む
    scratched = card("202306010106").horses.filter(pl.col("scratched"))
    assert scratched["horse_num"].to_list() == [6] and scratched["weight"].to_list() == [None]
    debut = card("202307010204").horses
    assert debut["n_past_runs"].sum() == 0


def test_shutuba_fallback_page():
    c = parse_shutuba(page("shutuba", "202606040512"), "202606040512")
    assert c.source == "shutuba" and c.horses.height == 16
    h1 = c.horses.row(0, named=True)
    assert (h1["horse_id"], h1["jockey_id"], h1["trainer_id"], h1["weight"], h1["inc_dec"]) == (
        2023100359, "05386", "01112", 476.0, 4.0)
    assert h1["father"] is None and h1["blinker"] is None
    assert c.race.weather == "雨" and c.race.going_dirt == "良" and c.race.class_text == "1勝クラス"


def test_card_parser_rejects_missing_race_data():
    with pytest.raises(ParseError, match="RaceData"):
        parse_shutuba_past("<html><body><h1 class='RaceName'>x</h1></body></html>", "202606040512")


# --- db result page ---------------------------------------------------------------------------

def test_diary_snap_cut_must_be_stripped():
    raw = page("result", "202606040512")
    assert raw.count("<diary_snap_cut>") == 54
    from lxml import html
    direct = html.fromstring(raw).xpath('//table[contains(@class,"race_table_01")]//tr[2]/td')
    assert len(direct) == 13  # タグを残すと直下のセルは 25 中 13
    stripped = html.fromstring(strip_diary_tags(raw)).xpath('//table[contains(@class,"race_table_01")]//tr[2]/td')
    assert len(stripped) == N_CELLS
    res = result("202606040512")
    top = res.rows.filter(pl.col("horse_num") == 5).row(0, named=True)
    assert (top["passing"], top["last_3F_time"], top["prize"], top["trainer_id"]) == ("7-7", 36.4, 820.0, "01102")


def test_result_requires_25_cells_and_required_columns():
    raw = strip_diary_tags(page("result", "202606040512"))
    one_less = re.sub(r"<td[^>]*>[^<]*</td>\s*</tr>", "</tr>", raw, count=2)
    with pytest.raises(ParseError, match="セル"):
        parse_result(one_less, "202606040512")
    no_corner = raw.replace(">通過<", ">通過順<")
    with pytest.raises(ParseError, match="通過"):
        parse_result(no_corner, "202606040512")


def test_result_race_info_post_time_and_conditions():
    r = result("202304010810").race
    assert (r.date, r.venue, r.times, r.daily, r.race_num) == (20230521, "新潟", 1, 8, 10)
    assert (r.surface, r.course_detail, r.dist, r.weather, r.goings) == ("ダ", "左", 1200, "晴", {"ダート": "良"})
    assert r.post_time == "14:40" and r.conditions == "4歳以上2勝クラス (混)[指](定量)" and r.grade == "2勝"
    assert result("202309040202").race.course_detail == "右 外"
    assert result("202306010208").race.course_detail == "芝 ダート"
    assert result("202305010811").race.grade == "GI"


def test_result_rank_tokens():
    assert parse_rank("1") == (1, 0) and parse_rank("3(降)") == (3, 7)
    assert parse_rank("取") == (0, 1) and parse_rank("除") == (0, 3) and parse_rank("中") == (0, 4)
    assert parse_rank("失") == (0, 5)
    with pytest.raises(ParseError):
        parse_rank("再")
    assert parse_time("1:11.5") == pytest.approx(71.5) and parse_time("59.8") == pytest.approx(59.8)
    assert parse_time("") is None
    assert parse_body_weight("496(0)") == (496.0, 0.0) and parse_body_weight("460(+2)") == (460.0, 2.0)
    assert parse_body_weight("計不") == (None, None)


def test_result_edge_case_rows():
    scratch = result("202306010106").rows.filter(pl.col("error_code") == 1).row(0, named=True)
    assert (scratch["horse_num"], scratch["rank"], scratch["weight"], scratch["win_odds"], scratch["time"]) == (
        6, 0, None, None, None)
    excl = result("202307010204").rows.filter(pl.col("error_code") == 3).row(0, named=True)
    assert excl["weight"] == 450.0 and excl["win_odds"] is None
    dnf = result("202306010103").rows.filter(pl.col("error_code") == 4).row(0, named=True)
    assert (dnf["passing"], dnf["win_odds"], dnf["pop"], dnf["time"]) == ("8-7-6-7", pytest.approx(4.7), 3, None)
    demoted = result("202304040405").rows.filter(pl.col("error_code") == 7).row(0, named=True)
    assert (demoted["horse_num"], demoted["rank"]) == (7, 3)


def test_payouts_dead_heats_and_bet_types():
    p = result("202304010810").payouts  # 1 着同着
    win = p.filter(pl.col("bet_type") == "win")
    assert sorted(win["combo"].to_list()) == ["14", "6"] and sorted(win["payout_per_100"].to_list()) == [300, 360]
    assert p.filter(pl.col("bet_type") == "exacta").height == 2
    assert p.filter(pl.col("bet_type") == "trifecta")["payout_per_100"].to_list() == [18810, 19210]
    p3 = result("202307020512").payouts  # 3 着同着
    assert p3.filter(pl.col("bet_type") == "place").height == 4 and p3.filter(pl.col("bet_type") == "wide").height == 5
    small = result("202307010311").payouts  # 7 頭: 枠連なし、複勝 2 頭
    assert "bracket_quinella" not in small["bet_type"].to_list()
    assert small.filter(pl.col("bet_type") == "place").height == 2
    assert set(p["bet_type"]) == {"win", "place", "bracket_quinella", "quinella", "wide", "exacta", "trio", "trifecta"}


def test_result_completeness():
    assert result("202306010106").completeness() is None
    res = result("202306010106")
    no_place = type(res)(res.race, res.rows, res.payouts.filter(pl.col("bet_type") != "place"))
    assert no_place.completeness() == "複勝の払戻が無い"
    no_win = type(res)(res.race, res.rows, res.payouts.filter(pl.col("bet_type") != "win"))
    assert no_win.completeness() == "単勝の払戻が無い"


# --- 欠けた値・数字でない値は ParseError (黙って null/0 にしない) ------------------------------------

def test_card_parser_rejects_runners_without_pedigree_jockey_trainer_or_weight():
    html = page("shutuba_past", "202509040311")
    assert card("202509040311").horses.select("father", "mother", "jockey_id", "trainer_id", "basis_weight"
                                              ).null_count().sum_horizontal().item() == 0
    mutations = {
        "father": html.replace('class="Horse01', 'class="HorseX1'),
        "mother": html.replace('class="Horse03', 'class="HorseX3'),
        "jockey_id": re.sub(r"/jockey/(result/recent/)?", "/jky/", html),
        "trainer_id": re.sub(r"/trainer/(result/recent/)?", "/trn/", html),
        "basis_weight": re.sub(r"(<br\s*/>\s*)<span>[0-9.]+</span>", r"\1", html),
    }
    for field, mutated in mutations.items():
        with pytest.raises(ParseError, match=field):
            parse_shutuba_past(mutated, "202509040311")


def test_scratched_horses_may_lack_jockey_and_carry_the_scratch_kind():
    html = page("shutuba_past", "202306010106")
    row = re.search(r'<tr class="HorseList Cancel">.*?</tr>', html, re.S).group(0)
    no_jockey = html.replace(row, re.sub(r"/jockey/(result/recent/)?", "/jky/", row))
    h = parse_shutuba_past(no_jockey, "202306010106").horses
    six = h.filter(pl.col("horse_num") == 6).row(0, named=True)
    assert six["scratched"] and six["scratch"] == "取消" and six["jockey_id"] is None
    assert h.filter(pl.col("horse_num") != 6)["scratch"].null_count() == h.height - 1
    excluded = card("202304010810").horses.filter(pl.col("horse_num") == 3).row(0, named=True)
    assert (excluded["scratch"], excluded["blinker"]) == ("除外", "B")


def test_card_weight_999_means_not_weighed():
    html = page("shutuba_past", "202307010204")
    first_weight = re.search(r'(<div class="Weight[^"]*">)\s*(\d{3})kg', html)
    mutated = html.replace(first_weight.group(0), f"{first_weight.group(1)}999kg", 1)
    before = card("202307010204").horses.row(0, named=True)
    after = parse_shutuba_past(mutated, "202307010204").horses.row(0, named=True)
    assert before["weight"] is not None and (after["weight"], after["inc_dec"]) == (None, None)


def test_result_non_numeric_payout_or_number_is_a_parse_error():
    raw = page("result", "202306010106")
    assert "14,150" in raw
    with pytest.raises(ParseError, match="払戻"):
        parse_result(raw.replace("14,150", "特払"), "202306010106")
    from keiba_goal.netkeiba.common import to_float, to_int

    assert to_int("1,234", "x") == 1234 and to_float("56.0", "x") == 56.0
    for bad in ("", "①", "1.5", "--"):
        with pytest.raises(ParseError):
            to_int(bad, "枠番")


def test_odds_malformed_entries_are_parse_errors():
    for odds in ('{"1": {"01": ["2.1"]}}', '{"1": {"x1": ["2.1", "0", "1"]}}', '{"1": {"01": "2.1"}}',
                 '{"1": {"01": ["2.1", "0", "1"]}, "2": ["a"]}'):
        with pytest.raises(ParseError):
            parse_odds(f'{{"status": "middle", "data": {{"odds": {odds}}}}}', "202606040512")
    with pytest.raises(ParseError, match="オブジェクト"):
        parse_odds("[1, 2]", "202606040512")
