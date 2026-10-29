import gzip
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from keiba_goal.config import load_config
from keiba_goal.data.schema import HISTORY_DTYPES, SENTINEL_ZERO_COLUMNS
from keiba_goal.netkeiba import horse_master
from keiba_goal.netkeiba.convert import (
    ConvertError,
    ParentNames,
    age_code,
    class_code,
    convert_card,
    convert_race,
    corner_ranks,
    flat_track_code,
    obstacle_track_code,
    weight_code,
)
from keiba_goal.netkeiba.gate import compare, dtype_problems, win_payout_check
from keiba_goal.netkeiba.http import decode
from keiba_goal.netkeiba.parse_result import ResultPage, parse_result
from keiba_goal.netkeiba.parse_shutuba import parse_shutuba_past
from keiba_goal.netkeiba.rebuild import paid_places

FX = Path(__file__).parent / "fixtures" / "netkeiba"
FIXTURE_RACES = ("202304010810", "202307020512", "202306010106", "202307010204", "202306010103", "202304040405",
                 "202307010311", "202309040202", "202306010208", "202305010811", "202306010110", "202509040311")

# 障害・平地の各コースを含む小さなコース表 (record_data から作る本物の表の部分集合)
_COURSE_KEYS = [  # (場, 場名, track_code, 距離, コーナー数)
    (4, "新潟", 23, 1200, 2), (4, "新潟", 23, 1800, 4), (7, "中京", 23, 1200, 2), (7, "中京", 23, 1800, 4),
    (7, "中京", 11, 1600, 3), (6, "中山", 17, 2000, 4), (6, "中山", 24, 1800, 4), (6, "中山", 18, 1600, 3),
    (6, "中山", 52, 3200, None), (5, "東京", 23, 1600, 2), (9, "阪神", 18, 1600, 2), (9, "阪神", 17, 2000, 4),
    (9, "阪神", 52, 2970, None), (8, "京都", 17, 2000, 4), (5, "東京", 11, 2400, 4),
]
COURSES = pl.DataFrame(
    [(*k, 2001, 2025, 100) for k in _COURSE_KEYS], orient="row",
    schema={"venue_code": pl.Int8, "place": pl.Utf8, "track_code": pl.Int16, "dist": pl.Int16,
            "corner_num": pl.Int8, "first_year": pl.Int16, "last_year": pl.Int16, "n_races": pl.Int32},
)


def load_result(key: str):
    return parse_result(decode(gzip.decompress((FX / f"result_{key}.html.gz").read_bytes()),
                               "https://db.netkeiba.com/"), key)


def load(key: str):
    card = parse_shutuba_past(decode(gzip.decompress((FX / f"shutuba_past_{key}.html.gz").read_bytes())), key)
    return load_result(key), card


def rows(key: str, **kw) -> pl.DataFrame:
    res, card = load(key)
    return convert_race(res, card, COURSES, **kw)


def horse(df: pl.DataFrame, num: int) -> dict:
    return df.filter(pl.col("horse_num") == num).row(0, named=True)


def test_output_has_exact_history_columns_and_dtypes():
    df = rows("202304010810")
    assert dtype_problems(df) == []
    assert list(df.schema) == list(HISTORY_DTYPES)
    assert df["race_id"].to_list() == sorted(df["race_id"].to_list())
    h = horse(df, 6)
    assert h["race_id"] == 202305210401081006 and h["race_key"] == 2023052104010810
    assert (h["year"], h["month"], h["day"], h["times"], h["daily"], h["race_num"]) == (23, 5, 21, 1, 8, 10)
    assert (h["place"], h["venue_code"], h["nk_race_id"], h["source"]) == ("新潟", 4, "202304010810", "netkeiba")
    assert h["leg"] is None and h["trainer_id"] == "01139" and h["jockey_id"] == "01168"


def test_dead_heat_first():
    df = rows("202304010810")
    for num in (6, 14):
        h = horse(df, num)
        assert h["rank"] == 1 and h["time_diff"] == 0.0 and h["prize"] == 1085.0  # (1550 + 620) / 2
    assert horse(df, 9)["time_diff"] == pytest.approx(0.2) and horse(df, 9)["prize"] == 390.0
    h6 = horse(df, 6)
    assert (h6["Ave_3F"], h6["PCI"], h6["last_3F_time_diff"], h6["last_3F_rank"]) == (
        pytest.approx(34.40), pytest.approx(42.7), pytest.approx(1.1), 4)
    assert (h6["class_code"], h6["track_code"], h6["corner_num"], h6["age_code"], h6["weight_code"]) == (43, 23, 2, 14, 4)
    assert h6["starters"] == 14 and h6["horse_N"] == 15


def test_dead_heat_third_and_winner_margin():
    df = rows("202307020512")
    for num in (1, 5):
        assert horse(df, num)["rank"] == 3 and horse(df, num)["prize"] == 160.0  # (200 + 120) / 2
        assert horse(df, num)["time_diff"] == pytest.approx(0.5)
    w = horse(df, 6)
    assert w["rank"] == 1 and w["time_diff"] == pytest.approx(-0.3) and w["state"] == "不" and w["weather"] == "雨"


def test_scratched_horse():
    df = rows("202306010106")
    h = horse(df, 6)
    # record の読み込みと同じく 0 埋めの欠損 (タイム・通過順・上がり順位) は null (schema.null_sentinels)
    assert (h["error_code"], h["rank"], h["time"], h["time_diff"]) == (1, 0, None, None)
    assert (h["weight"], h["inc_dec"], h["win_odds"], h["pop"], h["blinker"]) == (None, None, None, None, None)
    assert [h[f"corner{i}_rank"] for i in range(1, 5)] == [None, None, None, None]
    assert (h["last_3F_time"], h["last_3F_rank"], h["Ave_3F"], h["prize"]) == (None, None, None, 0.0)
    assert h["starters"] == 9 and h["horse_N"] == 10 and h["weight_code"] == 3 and h["age_code"] == 12


def test_excluded_debut_horse():
    h = horse(rows("202307010204"), 12)
    assert (h["error_code"], h["rank"], h["weight"], h["inc_dec"], h["win_odds"]) == (3, 0, 450.0, None, None)
    assert h["class_code"] == 15


def test_did_not_finish_and_debut_weight_change():
    df = rows("202306010103")
    h = horse(df, 13)
    assert (h["error_code"], h["rank"], h["time"], h["time_diff"]) == (4, 0, None, None)
    assert [h[f"corner{i}_rank"] for i in range(1, 5)] == [8, 7, 6, 7]
    assert (h["win_odds"], h["pop"], h["last_3F_time"], h["last_3F_rank"]) == (pytest.approx(4.7), 3, None, None)
    # 新馬戦: 過去に「出走」していない馬の増減は欠損 (取消だけの過去走がある 16 番も)
    assert df["inc_dec"].null_count() == df.height
    assert horse(df, 16)["rank"] == 1


def test_demoted_horse_keeps_official_rank():
    df = rows("202304040405")
    h = horse(df, 7)
    assert (h["error_code"], h["rank"], h["prize"], h["time_diff"]) == (7, 3, 180.0, pytest.approx(0.5))
    assert horse(df, 8)["rank"] == 2 and horse(df, 8)["prize"] == 290.0
    assert horse(df, 7)["last_3F_rank"] == horse(df, 8)["last_3F_rank"] == 4  # 同タイムは min 順位
    assert horse(df, 15)["time_diff"] == pytest.approx(-0.5)


def test_small_field_corner_padding_and_prize_without_added_money():
    df = rows("202307010311")
    h = horse(df, 2)
    assert [h[f"corner{i}_rank"] for i in range(1, 5)] == [None, 5, 5, 5]   # 3 コーナーのレース: 1 角は無い
    assert h["prize"] == 4000.0  # 結果ページは付加賞込みの 4,030.8
    assert (h["class_code"], h["weight_code"], h["track_code"], h["corner_num"]) == (163, 2, 11, 3)
    assert h["starters"] == 7


def test_paid_places_follow_the_payout_table():
    assert paid_places(load("202307010311")[0]) == 2   # 7 頭立て
    assert paid_places(load("202309040202")[0]) == 3   # 8 頭登録・1 頭取消 (7 頭出走) でも 3 着まで払戻
    assert paid_places(load("202307020512")[0]) == 3   # 3 着同着 (複勝 4 頭)
    assert paid_places(load_result("202408010206")) == 3   # 9 頭 (中止 1)


def test_paid_places_is_null_when_place_should_have_been_sold_but_was_not_read():
    res = load_result("202309040202")   # 7 頭出走
    no_place = ResultPage(res.race, res.rows, res.payouts.filter(pl.col("bet_type") != "place"))
    assert paid_places(no_place) is None and no_place.completeness() == "複勝の払戻が無い"
    four = res.rows.filter(pl.col("rank").is_between(1, 4))   # 出走 4 頭以下は複勝の発売なし
    assert paid_places(ResultPage(res.race, four, no_place.payouts)) == 0


def test_paid_places_with_a_dead_heat_for_second():
    jc = load_result("202405050812")   # 2024 ジャパンC: 2 着同着 (7・10 番)、複勝は 1・2・2 着の 3 頭
    assert jc.rows.filter(pl.col("rank") == 2)["horse_num"].sort().to_list() == [7, 10]
    assert jc.payouts.filter(pl.col("bet_type") == "place")["combo"].sort().to_list() == ["10", "3", "7"]
    assert paid_places(jc) == 3   # {1, 2 着} = {3 着以内} なので登録 14 頭で決める
    # 同じ形で 7 頭立てなら 2 着払い
    small = jc.rows.filter(pl.col("horse_num").is_in([1, 2, 3, 4, 7, 10, 14]))
    assert paid_places(ResultPage(jc.race, small, jc.payouts)) == 2


def test_obstacle_race():
    df = rows("202306010208")
    h = horse(df, 1)
    assert (h["track_code"], h["corner_num"], h["surface"], h["direction"]) == (52, None, 3, 0)
    assert (h["last_3F_rank"], h["Ave_3F"], h["PCI"], h["last_3F_time_diff"]) == (None, None, None, None)
    assert h["last_3F_time"] == pytest.approx(13.7)
    assert [h[f"corner{i}_rank"] for i in range(1, 5)] == [5, 5, 5, 6]
    assert (h["age_code"], h["class_code"], h["weight_code"], h["state"], h["class_ord"]) == (19, 131, 2, "良", 4)


def test_g1_and_listed():
    g1 = rows("202305010811")
    assert set(g1["class_code"]) == {195} and set(g1["class_ord"]) == {7}
    assert horse(g1, 7)["prize"] == 12000.0
    assert horse(g1, 2)["inc_dec"] is None  # 出馬表の「前計不」
    listed = rows("202306010110")
    assert set(listed["class_code"]) == {115} and set(listed["track_code"]) == {18}
    assert class_code("3歳オープン (国際)(特指)(別定)", "L", 20181230) == 131  # 2019 年より前は OP


def test_parent_names_prefer_record_master():
    df = rows("202509040311", parents={2021105236: ("ブリックスアンドモルタル", "マキシマムドパリ"),
                                        2019100792: ("Hat Trick", "Nokaze")})
    assert horse(df, 2)["father"] == "Hat Trick"       # record の名前で置き換える
    assert horse(df, 6)["father"] == "ハットトリック"  # マスタに無い馬は出馬表の名前
    assert horse(df, 3)["blinker"] == "B" and horse(df, 6)["blinker"] == "B"


def test_pci_rounds_exact_ties_to_even():
    from fractions import Fraction

    from keiba_goal.netkeiba.convert import _round_half_even, _round_half_up

    assert _round_half_even(Fraction(161, 4), 1) == 40.2 and _round_half_even(Fraction(175, 4), 1) == 43.8
    assert _round_half_up(Fraction(36525, 1000), 2) == 36.53
    h = horse(rows("202309040202"), 3)   # 厳密値 61.25 → record は 61.2
    assert h["PCI"] == pytest.approx(61.2)


def test_split_prize_is_truncated_and_parent_names_follow_jv():
    from keiba_goal.netkeiba.convert import _prize, jv_parent_name

    assert _prize(3, [1, 2, 3, 3], (550.0, 220.0, 141.0, 80.0, 55.0)) == 110.0   # (141 + 80) / 2 = 110.5
    assert _prize(5, [1, 2, 3, 4, 5, 5], (550.0, 220.0, 140.0, 83.0, 55.0)) == 27.0  # 6 着の賞金は 0
    assert _prize(0, [1], (550.0,)) == 0.0
    assert jv_parent_name("ポールポジションII") == "ポールポジション2" and jv_parent_name("タッチII") == "タッチ2"
    assert jv_parent_name("Henry VIII") == "Henry VIII" and jv_parent_name("キングカメハメハ") == "キングカメハメハ"
    assert jv_parent_name(None) is None


def test_code_tables():
    assert flat_track_code("芝", "右 外") == 18 and flat_track_code("芝", "左") == 11
    assert flat_track_code("ダ", "右") == 24 and flat_track_code("芝", "直線") == 10
    assert flat_track_code("芝", "右 内2周") == 21 and flat_track_code("芝", "左 外") == 12
    with pytest.raises(ConvertError):
        flat_track_code("芝", "")
    assert obstacle_track_code("芝 ダート", 6, 3200, COURSES) == 52
    assert age_code("障害3歳以上未勝利") == 18 and age_code("2歳新馬") == 11 and age_code("3歳以上1勝クラス") == 13
    assert weight_code("3歳以上2勝クラス (混)(ハンデ)") == 1 and weight_code("2歳未勝利 (混)[指](馬齢)") == 3
    with pytest.raises(ConvertError):
        weight_code("3歳以上")
    assert class_code("2歳未勝利 (混)", None, 20250101) == 7 and class_code("3歳以上3勝クラス", None, 20250101) == 67
    assert class_code("4歳以上1000万下", None, 2010_01_01) == 43
    assert class_code("3歳以上オープン", "GII", 20250101) == 179 and class_code("障害4歳以上オープン", "J.GI", 20250101) == 195
    assert class_code("3歳オープン", "重賞", 20210529) == 147
    assert corner_ranks("8-8-4-2", 4) == [8, 8, 4, 2] and corner_ranks("7-7", 2) == [0, 0, 7, 7]
    assert corner_ranks("", 0) == [0, 0, 0, 0] and corner_ranks("", 4) == [0, 0, 0, 0]
    assert corner_ranks("1-1-1-1-2-2", 6) == [1, 1, 2, 2]


def test_partial_passing_of_a_non_finisher_is_placed_by_corner():
    # record_data の中止馬: 4 角のレースで '4-6' → 1・2 角、2 角のレースで '5' → 3 角 (= 1 つ目のコーナー)
    assert corner_ranks("4-6", 4) == [4, 6, 0, 0] and corner_ranks("4-6-3", 4) == [4, 6, 3, 0]
    assert corner_ranks("5", 2) == [0, 0, 5, 0] and corner_ranks("3", 3) == [0, 3, 0, 0]
    assert corner_ranks("1-1-1-1", 6) == [1, 1, 0, 0] and corner_ranks("1-1-1-1-2", 6) == [1, 1, 2, 0]
    with pytest.raises(ConvertError):
        corner_ranks("1-2-3", 2)
    df = convert_race(load_result("202408010206"), None, COURSES)   # 京都 芝 2000m、6 番が中止 (通過 '4-6')
    h = horse(df, 6)
    assert (h["error_code"], h["rank"], h["time"]) == (4, 0, None)
    assert [h[f"corner{i}_rank"] for i in range(1, 5)] == [4, 6, None, None]
    assert [horse(df, 2)[f"corner{i}_rank"] for i in range(1, 5)] == [8, 8, 2, 2]


def test_unknown_course_is_an_error():
    res, card = load("202306010110")
    with pytest.raises(ConvertError, match="コース表"):
        convert_race(res, card, COURSES.filter(pl.col("track_code") != 18))


def test_card_from_another_race_is_rejected():
    res, _ = load("202306010110")
    _, other = load("202306010106")
    with pytest.raises(ConvertError, match="別レース"):
        convert_race(res, other, COURSES)


@pytest.mark.skipif(not load_config().paths.record_dir.exists(), reason="record_data が無い")
def test_fixtures_match_record_data_except_documented_quirks():
    from keiba_goal.netkeiba.course_table import load_or_build
    from keiba_goal.netkeiba.records import read_record_range

    cfg = load_config()
    courses = load_or_build(cfg.paths.interim / "course_table.parquet", cfg.paths.record_dir)
    nk = pl.concat([convert_race(*load(k), courses) for k in FIXTURE_RACES])
    rec = pl.concat([read_record_range(cfg.paths.record_dir, d, d) for d in sorted(set(nk["date"]))])
    counts, details = compare(nk, rec.filter(pl.col("race_id").is_in(nk["race_id"].implode())))
    assert counts.filter(pl.col("column") == "(行が無い)")["mismatches"].item() == 0
    bad = {(d["race_id"], d["column"]) for d in details.to_dicts()}
    known = {
        (202305210401081003, "blinker"),   # 除外馬: 出馬表は B、record は空 (record は除外馬の B を 84/1,052 件で残す)
        (202509130904031106, "father"),    # 外国産馬の親は JV が外国名 (Hat Trick / ハットトリック)
        (202301080701031107, "mother"), (202309100904020208, "mother"), (202509130904031103, "mother"),
    }
    assert bad <= known, sorted(bad - known)
    # 0 埋めの欠損は record と同じく null (父の英字名の引用符も record の読み込みが外す)
    assert any(nk[c].null_count() for c in SENTINEL_ZERO_COLUMNS)
    assert all((nk[c] == 0).sum() == 0 for c in SENTINEL_ZERO_COLUMNS)


PRE_RACE_SHARED = ("race_id", "year", "month", "day", "times", "place", "daily", "race_num", "horse", "jockey_id",
                   "horse_N", "waku_num", "horse_num", "class_code", "track_code", "corner_num", "dist", "state",
                   "weather", "age_code", "sex", "age", "basis_weight", "blinker", "weight", "inc_dec", "weight_code",
                   "father", "mother", "id", "date", "race_key", "nk_race_id", "venue_code", "surface", "direction",
                   "course_inout", "class_ord", "trainer_id", "source")


@pytest.mark.parametrize("key", [k for k in FIXTURE_RACES if k != "202509040311"])
def test_card_only_conversion_matches_result_path(key):
    res, card = load(key)
    full = convert_race(res, card, COURSES)
    pre = convert_card(card, res.race.date, COURSES, ParentNames.card_only())
    assert list(pre.schema) == list(HISTORY_DTYPES) and pre["rank"].null_count() == pre.height
    assert full.select(PRE_RACE_SHARED).equals(pre.select(PRE_RACE_SHARED)), key


def test_blinker_rule_for_scratched_horses_is_the_same_on_both_paths():
    from keiba_goal.netkeiba.parse_shutuba import CardPage

    names = ParentNames.card_only()
    res, card = load("202304010810")   # 3 番は除外 (出馬表に B): 除外は B を残し、取消だけ欠損にする
    assert card.horses.filter(pl.col("horse_num") == 3)["scratch"].item() == "除外"
    assert horse(convert_race(res, card, COURSES), 3)["blinker"] == "B"
    assert horse(convert_card(card, res.race.date, COURSES, names), 3)["blinker"] == "B"
    res, card = load("202306010106")   # 6 番は取消。全馬に B を付けても取消馬だけ欠損になる
    card_b = CardPage(card.race, card.horses.with_columns(blinker=pl.lit("B")), card.past_runs, card.source)
    assert horse(convert_race(res, card_b, COURSES), 6)["blinker"] is None
    assert horse(convert_card(card_b, res.race.date, COURSES, names), 6)["blinker"] is None
    assert horse(convert_card(card_b, res.race.date, COURSES, names), 5)["blinker"] == "B"


def _record_master(rows: list[dict]) -> pl.DataFrame:
    base = {"sex": "牡", "broodmare_sire_ref": None, "foreign_bred": None, "first_date": 20200101,
            "last_date": 20250831, "source": "record"}
    return pl.DataFrame([{**base, **r, "birth_year": r["horse_id"] // 1_000_000} for r in rows],
                        schema=horse_master.SCHEMA)


def test_card_only_parents_follow_the_same_rule_as_the_rebuilt_master():
    """ライブ (convert_card + parent_names) の父・母 = 後でそのレースを作り直したときの馬マスタの父・母。"""
    _, card = load("202509040311")
    # record にいる外国産馬 (別の馬) の出馬表の父は 'ハットトリック'、record は 'Hat Trick'
    master = _record_master([
        {"horse_id": 2019100792, "horse": "マイネルクリソーラ", "father": "スクリーンヒーロー", "mother": "マイネトゥインクル"},
        {"horse_id": 2016110099, "horse": "ガイコクサン", "father": "Hat Trick", "mother": "Some Mare"},
    ])
    before = pl.DataFrame([{"horse_id": 2016110099, "horse": "ガイコクサン", "sex": "牡", "father": "ハットトリック",
                            "mother": "サムメア", "broodmare_sire": None, "foreign_bred": True,
                            "first_date": 20250830, "last_date": 20250830},
                           {"horse_id": 2020103060, "horse": "ヴェルテンベルク", "sex": "牡", "father": "キタサンブラック",
                            "mother": "マルカアイチャン", "broodmare_sire": None, "foreign_bred": False,
                            "first_date": 20250830, "last_date": 20250830}], schema=horse_master.SEEN_SCHEMA)
    live_master = horse_master.update_from_seen(master, before)   # 前日までの馬マスタ (5 番は netkeiba 由来)
    names = horse_master.parent_names(live_master, before)
    assert names.known[2020103060] == ("キタサンブラック", "マルカアイチャン")   # 出典を問わずマスタの馬
    live = convert_card(card, 20250913, COURSES, names)
    assert horse(live, 6)["father"] == "Hat Trick"       # 新しい外国産馬: record から学んだ外国名
    assert horse(live, 3)["father"] == "Kingman" and horse(live, 2)["father"] == "スクリーンヒーロー"
    seen = horse_master.merge_seen(before, horse_master.card_horses(20250913, card))
    rebuilt = horse_master.update_from_seen(master, seen)   # crawl の作り直しが行の父・母に使うマスタ
    want = live.select("id").join(rebuilt.select(pl.col("horse_id").alias("id"), "father", "mother"), on="id",
                                  how="left")
    assert live.select("id", "father", "mother").equals(want.select("id", "father", "mother"))
    raw = convert_card(card, 20250913, COURSES, ParentNames.card_only())
    assert horse(raw, 6)["father"] == "ハットトリック"   # マスタを使わなければ出馬表の名前のまま


def test_win_payout_equals_final_odds_except_dead_heat():
    from keiba_goal.netkeiba.rebuild import payout_frame

    frames = [(convert_race(*load(k), COURSES), payout_frame(load(k)[0], datetime(2026, 1, 1)))
              for k in FIXTURE_RACES if k != "202509040311"]
    rows = pl.concat([f[0] for f in frames])
    pays = pl.concat([f[1] for f in frames])
    check = win_payout_check(rows, pays)
    assert check["checked"] == 10 and check["dead_heat_races"] == 1 and check["mismatched"] == []
