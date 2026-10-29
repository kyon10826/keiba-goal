"""目的変数 (design §5 v2.1): 同着・取消・少頭数・支払着数の出どころ・学習対象・ノイズレース。"""

import polars as pl
import pytest

from keiba_goal.features.decision import decision_frame, with_live_columns
from keiba_goal.features.labels import LABEL_COLUMNS, SOURCE_COLUMNS, STARTERS_RULE, build_labels
from keiba_goal.features.synth import history_frame

D = 20230910


def race(race_num, ranks, errors=None, surface=1, class_ord=0):
    errors = errors or [0] * len(ranks)
    return [
        {"date": D, "venue_code": 6, "race_num": race_num, "horse_num": i + 1, "id": race_num * 100 + i,
         "rank": r, "error_code": e, "surface": surface, "class_ord": class_ord}
        for i, (r, e) in enumerate(zip(ranks, errors))
    ]


@pytest.fixture(scope="module")
def rows() -> pl.DataFrame:
    return history_frame(
        race(1, [1, 1, 3, 4, 5, 6, 7, 0], [0, 0, 0, 0, 0, 0, 0, 1])      # 8 頭登録・1 頭取消で 7 頭出走、1 着同着
        + race(2, [1, 2, 3, 4])                                            # 4 頭立て (複勝なし)
        + race(3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [0] * 9 + [4])           # 競走中止を含む 10 頭
        + race(4, [1, 2, 3, 4, 5], surface=3)                              # 障害
        + race(5, [1, 2, 3, 4, 5], class_ord=-1)                           # 対象外のクラス
        + race(6, [1, 2, 3, 4, 5, 6])                                      # 6 頭 (paid_places に無い)
    )


def labels(rows: pl.DataFrame, paid: pl.DataFrame | None) -> pl.DataFrame:
    frame = decision_frame(with_live_columns(rows.lazy()).collect(), None)
    return build_labels(rows.select(SOURCE_COLUMNS), frame, paid)


def by_race(df: pl.DataFrame, race_num: int) -> pl.DataFrame:
    return df.filter((pl.col("race_key") % 100) == race_num).sort("horse_num")


def test_starters_rule_without_paid_places(rows):
    out = labels(rows, None)
    assert out.columns == list(LABEL_COLUMNS)
    r1 = by_race(out, 1)
    assert r1["y_win"].to_list() == [1, 1, 0, 0, 0, 0, 0, None]          # 同着は両方 1、取消は null
    assert r1["n_paid_places"].unique().to_list() == [2]                   # 7 頭出走 → 2 着まで
    assert r1["y_place"].to_list() == [1, 1, 0, 0, 0, 0, 0, None]
    assert r1["started"].to_list() == [True] * 7 + [False]
    assert r1["eligible"].to_list() == [True] * 7 + [False]
    assert r1["field_dec_size"].unique().to_list() == [7] and r1["in_field_dec"].to_list() == [True] * 7 + [False]
    assert set(out["n_paid_source"].unique()) == {STARTERS_RULE}
    r2 = by_race(out, 2)
    assert r2["n_paid_places"].unique().to_list() == [0]
    assert r2["y_place"].null_count() == 4 and r2["y_win"].to_list() == [1, 0, 0, 0]
    assert not r2["eligible"].any()                                        # 出走 5 頭未満
    r3 = by_race(out, 3)
    assert r3["noise_race"].all() and r3["n_paid_places"].unique().to_list() == [3]
    assert r3["y_win"][-1] == 0 and r3["y_place"][-1] == 0 and r3["started"][-1]
    assert not by_race(out, 4)["eligible"].any() and not by_race(out, 5)["eligible"].any()
    assert not by_race(out, 1)["noise_race"].any()


def test_paid_places_table_wins_and_missing_races_fall_back(rows):
    keys = rows.select("race_key").unique().sort("race_key")["race_key"].to_list()
    paid = pl.DataFrame({
        "race_key": keys[:5],
        "n_paid_places": pl.Series([3, 0, 3, 2, None], dtype=pl.Int8),   # 1R: 発売開始時 8 頭 → 3 着まで、5R: 不明
        "field_open": pl.Series([8, 4, 10, 5, 5], dtype=pl.Int16),
        "starters": pl.Series([7, 4, 10, 5, 5], dtype=pl.Int16),
        "source": ["payout", "field_open", "payout", "payout", "payout"],
    })
    out = labels(rows, paid)
    r1 = by_race(out, 1)
    assert r1["n_paid_places"].unique().to_list() == [3]
    assert r1["y_place"].to_list() == [1, 1, 1, 0, 0, 0, 0, None]
    assert r1["n_paid_source"].unique().to_list() == ["payout"]
    r5 = by_race(out, 5)
    assert r5["n_paid_places"].null_count() == 5 and r5["y_place"].null_count() == 5
    r6 = by_race(out, 6)
    assert r6["n_paid_source"].unique().to_list() == [STARTERS_RULE] and r6["n_paid_places"].unique().to_list() == [2]


def test_duplicate_paid_places_rows_are_rejected(rows):
    key = rows["race_key"][0]
    paid = pl.DataFrame({"race_key": [key, key], "n_paid_places": [3, 3], "source": ["a", "b"]})
    with pytest.raises(ValueError, match="重複"):
        labels(rows, paid)
