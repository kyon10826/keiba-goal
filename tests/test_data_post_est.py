"""発走時刻の推定 (post_est): 合成した行の並びで規則の各場合を確かめる。時刻は分 (10:00 = 600)。"""

import numpy as np
import polars as pl
import pytest

from keiba_goal.data.post_est import estimate_post, races_after_g1


def rows(*minutes: int) -> np.ndarray:
    return np.array(minutes, dtype=np.int64)


@pytest.mark.parametrize("ref, cand, expected", [
    # 発走 10:00。区分 3 は +1 分、最後の候補行は 6 分前 (09:54) か 5 分前 (09:55): どちらも 10:00
    (601, rows(589, 594), 600),
    (601, rows(589, 595), 600),
    # 区分 3 が発走ちょうど (以前の Δ = 1 の規則が 1 分早く外れて T-6 を落とした形)
    (600, rows(589, 594), 600),
    # 区分 3 の刻印が 7 分遅れ・締切推定なし: 最後の候補行 + 6 分が上限になるので 10:05 にならない
    (607, rows(589, 594), 600),
    (607, rows(590, 595), 600),
    # 候補行が無い: ref の 5 分刻みの切り捨て
    (602, rows(), 600),
])
def test_estimate_matches_the_post_time_for_the_usual_row_patterns(ref, cand, expected):
    assert estimate_post(ref, cand) == expected


def test_estimate_errs_early_when_the_last_row_is_missing_or_the_post_is_off_grid():
    # 6・5 分前の行が無く最後の候補行が 11 分前 (09:49): 09:55 (5 分早い。T-6 は 09:49 の行か欠損になるだけ)
    assert estimate_post(601, rows(584, 589)) == 595
    # 15:01 のような 5 分刻みでない発走 (2009 年から): 区分 3 15:02、最後の候補行 14:55 → 15:00 (1 分早い)
    assert estimate_post(902, rows(890, 895)) == 900


def test_estimate_never_exceeds_ref():
    for ref in range(590, 620):
        for last in range(570, 620):
            assert estimate_post(ref, rows(last)) <= ref


def test_after_g1_moves_one_grid_step_earlier():
    assert estimate_post(601, rows(589, 594), after_g1=True) == 595
    assert estimate_post(602, rows(), after_g1=True) == 595


def test_races_after_g1():
    races = pl.DataFrame({
        "race_key": [1, 2, 3, 4, 5, 6, 7],
        "date": [20080601] * 5 + [20080602] * 2,
        "venue_code": [5, 5, 5, 8, 8, 5, 5],
        "race_num": [9, 10, 11, 10, 11, 10, 11],
        "class_ord": [3, 7, 2, 2, 7, 7, 2],     # 東京 10R G1、京都 11R G1 (最終)、翌日の東京 10R G1
    })
    races = pl.concat([races, races.head(2)])   # 馬ごとの行の重複は 1 レースとして数える
    assert races_after_g1(races) == frozenset({3, 7})
