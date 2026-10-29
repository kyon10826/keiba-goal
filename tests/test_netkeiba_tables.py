import gzip

import polars as pl
import pytest

from keiba_goal.data.record import load_record_file
from keiba_goal.netkeiba import course_table, horse_master
from keiba_goal.netkeiba.http import decode
from keiba_goal.netkeiba.parse_shutuba import parse_shutuba_past
from keiba_goal.netkeiba.records import read_record_range, record_frames
from test_netkeiba_support import FX, record_line, write_record


def test_record_frames_use_annual_files_and_the_latest_snapshot(tmp_path):
    write_record(tmp_path / "record_data_2024.csv", [record_line(race_id="202401010101010101")])
    write_record(tmp_path / "record_data_2025_20250822.csv", [record_line(race_id="202507260101010101")])
    write_record(tmp_path / "record_data_2025_20250919.csv", [
        record_line(race_id="202507260101010101"), record_line(race_id="202509130101010101")])
    frames = list(record_frames(tmp_path))
    assert [f.height for f in frames] == [1, 2]
    assert [f.height for f in record_frames(tmp_path, first_year=2025)] == [2]


def test_record_rows_are_exactly_the_shared_history_representation(tmp_path):
    """P1 の読み込みと同じ表現 (引用符付きの名前・A/B/C の日次・'----' など) を、自前の解釈をはさまずに返す。"""
    p = write_record(tmp_path / "record_data_2025_20250919.csv", [
        record_line(race_id="202507260101100101", time_diff="----", father="Too Darn Hot", mother="Lady of Persia",
                    inc_dec="+6"),
        record_line(race_id="202507260101010102", time_diff=" -0.0"),
    ])
    rng = read_record_range(tmp_path, 20250726, 20250726, ["father", "daily", "time_diff", "inc_dec", "jockey_id"])
    assert rng.columns == ["race_id", "father", "daily", "time_diff", "inc_dec", "jockey_id"]
    shared = load_record_file(p)[0].select(rng.columns)
    assert rng.sort("race_id").equals(shared.sort("race_id"))
    r0 = rng.filter(pl.col("daily") == 10).row(0, named=True)
    assert r0["time_diff"] is None and r0["inc_dec"] == 6.0 and r0["jockey_id"] == "01032"
    assert rng.schema["race_id"] == pl.Int64 and rng.schema["time_diff"] == pl.Float32
    assert read_record_range(tmp_path, 20240101, 20241231).is_empty()


def test_read_record_rejects_unexpected_tokens(tmp_path):
    write_record(tmp_path / "record_data_2025_20250919.csv", [record_line(rank="X")])
    with pytest.raises(Exception):
        read_record_range(tmp_path, 20250101, 20251231, ["rank"])


def test_course_table_is_unique_and_conflicts_are_rejected(tmp_path):
    write_record(tmp_path / "record_data_2024.csv", [
        record_line(race_id="202401010101010101", track_code=17, dist=1200, corner_num=2),
        record_line(race_id="202401010101010102", track_code=17, dist=1200, corner_num=2),
        record_line(race_id="202401010101020101", track_code=54, dist=2750, corner_num=""),
    ])
    t = course_table.build_course_table(tmp_path)
    assert t.height == 2 and t.schema["corner_num"] == pl.Int8
    assert course_table.lookup(t, 1, 17, 1200)["corner_num"] == 2
    assert course_table.lookup(t, 1, 54, 2750)["corner_num"] is None
    assert course_table.lookup(t, 1, 18, 1200) is None
    assert course_table.track_codes(t, 1, 2750, set(range(51, 60))) == [54]
    write_record(tmp_path / "record_data_2023.csv", [
        record_line(race_id="202301010101010101", track_code=17, dist=1200, corner_num=3)])
    with pytest.raises(course_table.CourseConflict):
        course_table.build_course_table(tmp_path)


def _two_year_record(tmp_path):
    write_record(tmp_path / "record_data_2024.csv", [
        record_line(race_id="202401010101010101", id=2021105236, horse="アスクカムオンモア", sex="牡",
                    father="ブリックスアンドモルタル", mother="マキシマムドパリ"),
        record_line(race_id="202406010101010101", id=2019100792, horse="マイネルクリソーラ", sex="牡",
                    father="Hat Trick", mother="マイネトゥインクル"),
    ])
    write_record(tmp_path / "record_data_2025_20250919.csv", [
        record_line(race_id="202503010101010101", id=2019100792, horse="マイネルクリソーラ", sex="セ",
                    father="Hat Trick", mother="マイネトゥインクル"),
        record_line(race_id="202509130904031102", id=2023100001, horse="ゲートデビュー", sex="牝",
                    father="キタサンブラック", mother="ゲートノハハ")])


def test_horse_master_seed_and_update_from_card(tmp_path):
    _two_year_record(tmp_path)
    m = horse_master.seed_from_record(tmp_path)
    assert m.height == 3 and set(m["source"]) == {"record"}
    h = m.filter(pl.col("horse_id") == 2019100792).row(0, named=True)
    assert (h["sex"], h["first_date"], h["last_date"], h["birth_year"]) == ("セ", 20240601, 20250301, 2019)
    # 父・母は P1 の読み込みの値そのもの (引用符の扱いも history と同じ)
    shared = load_record_file(tmp_path / "record_data_2024.csv")[0]
    assert h["father"] == shared.filter(pl.col("id") == 2019100792)["father"].item()

    card = parse_shutuba_past(decode(gzip.decompress((FX / "shutuba_past_202509040311.html.gz").read_bytes())),
                              "202509040311")
    m2 = horse_master.update_from_cards(m, [(20250913, card)])
    assert m2.height == 16
    known = m2.filter(pl.col("horse_id") == 2021105236).row(0, named=True)
    assert known["source"] == "record" and known["last_date"] == 20250913 and known["broodmare_sire_ref"] == "キングカメハメハ"
    assert m2.filter(pl.col("source") == "netkeiba").height == 13
    assert set(horse_master.parents_map(m2)) == {2021105236, 2019100792, 2023100001}
    assert m2.schema == pl.Schema(horse_master.SCHEMA)


def test_as_of_seed_ignores_rows_from_the_start_date(tmp_path):
    _two_year_record(tmp_path)
    before = horse_master.seed_from_record(tmp_path, before=20250913)
    assert 2023100001 not in before["horse_id"].to_list() and before.height == 2
    assert horse_master.seed_from_record(tmp_path, before=20000101).is_empty()


def test_seen_store_keeps_the_latest_row_per_horse_and_day(tmp_path):
    a = pl.DataFrame({"horse_id": [1, 2], "horse": ["A", "B"], "sex": ["牡", "牝"], "father": ["F", None],
                      "mother": ["M", "N"], "broodmare_sire": [None, None], "foreign_bred": [False, False],
                      "first_date": [20250906, 20250906], "last_date": [20250906, 20250906]},
                     schema=horse_master.SEEN_SCHEMA)
    b = a.head(1).with_columns(pl.lit("F2").alias("father"))
    merged = horse_master.merge_seen(a, None, b)
    assert merged.height == 2 and merged.filter(pl.col("horse_id") == 1)["father"].item() == "F2"
    assert horse_master.load_seen(tmp_path / "missing.parquet").schema == pl.Schema(horse_master.SEEN_SCHEMA)
    assert horse_master.merge_seen().is_empty()


def test_foreign_bred_newcomers_get_jv_parent_names():
    master = pl.DataFrame({
        "horse_id": [2020110085], "horse": ["イングランドアイズ"], "sex": ["牝"], "birth_year": [2020],
        "father": ["Palace Malice"], "mother": ["Queen's Turf"], "broodmare_sire_ref": [None],
        "foreign_bred": [None], "first_date": [20230101], "last_date": [20250913], "source": ["record"],
    }, schema=horse_master.SCHEMA)
    seen = pl.DataFrame({
        "horse_id": [2020110085, 2023110001, 2023100002],
        "horse": ["イングランドアイズ", "ガイコクサン", "ナイコクサン"], "sex": ["牝", "牡", "牡"],
        "father": ["パレスマリス", "パレスマリス", "パレスマリス"], "mother": ["クイーンズターフ", "Foo", "バー"],
        "broodmare_sire": [None, None, None], "foreign_bred": [True, True, False],
        "first_date": [20251001] * 3, "last_date": [20251001] * 3,
    }, schema=horse_master.SEEN_SCHEMA)
    m = horse_master.update_from_seen(master, seen)
    by_id = {r["horse_id"]: r for r in m.iter_rows(named=True)}
    assert by_id[2020110085]["father"] == "Palace Malice" and by_id[2020110085]["foreign_bred"] is True
    assert by_id[2023110001]["father"] == "Palace Malice" and by_id[2023110001]["source"] == "netkeiba"
    assert by_id[2023100002]["father"] == "パレスマリス"   # 内国産馬はカナのまま (JV もカナ)
    assert horse_master.update_from_seen(master, seen.clear()).equals(master)


def test_card_marks_foreign_bred_horses():
    card = parse_shutuba_past(decode(gzip.decompress((FX / "shutuba_past_202509040311.html.gz").read_bytes())),
                              "202509040311")
    flags = dict(card.horses.select("horse_num", "foreign_bred").iter_rows())
    assert flags[3] is True and flags[6] is True and flags[1] is False
