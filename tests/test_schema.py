from keiba_goal.data.schema import (
    CLASS_ORD,
    HISTORY_DTYPES,
    POST_RACE_COLUMNS,
    PRE_RACE_COLUMNS,
    RECORD_COLUMNS,
    TRACK_CODE,
    nk_race_id_from_race_id,
    race_key_from_race_id,
)


def test_record_columns_and_dtypes_align():
    assert len(RECORD_COLUMNS) == 47
    assert set(RECORD_COLUMNS) <= set(HISTORY_DTYPES)


def test_pre_and_post_race_columns_are_disjoint_and_cover_record():
    assert not (PRE_RACE_COLUMNS & POST_RACE_COLUMNS)
    assert set(RECORD_COLUMNS) <= PRE_RACE_COLUMNS | POST_RACE_COLUMNS
    for leak in ("win_odds", "pop", "rank", "time", "last_3F_time", "error_code", "leg", "prize", "starters"):
        assert leak in POST_RACE_COLUMNS


def test_race_id_conversions():
    rid = 202407211003081206  # 2024-07-21 場10 回03 日08 R12 馬06
    assert race_key_from_race_id(rid) == 2024072110030812
    assert nk_race_id_from_race_id(rid) == "202410030812"


def test_class_and_track_maps():
    assert CLASS_ORD[195] == 7 and CLASS_ORD[23] == 1 and CLASS_ORD[115] == CLASS_ORD[131] == 4
    assert TRACK_CODE[24] == (2, 1, 0)   # ダート右
    assert TRACK_CODE[17] == (1, 1, 0)   # 芝右
    assert TRACK_CODE[55][0] == 3        # 障害
