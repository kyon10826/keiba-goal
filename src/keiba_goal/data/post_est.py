"""実際の発走時刻が無いレースの発走時刻の推定 (design §3.2。主に 2003-10〜2007-07)。

規則は 2007-07〜2024 の開催日一覧の発走時刻と JD の行の並びから作り、2008-10 年 (推定期間に最も近い年) で選んだ:
- 区分 3 は約 98% が発走の 0〜4 分後に記録される (2008-10 は 88% が +1 分)。締切推定 close_est があれば
  ref = min(区分 3, close_est + 2) で、ref ≥ 発走 がほぼ常に成り立つ。
- 締切前の最後の候補行は発走の 6 分前 (約 70%) か 5 分前 (約 27%)。
- 2008 年までの発走時刻はすべて 5 分刻み (2009 年から 15:01 のような 5 分刻みでない時刻が 7〜8% ある)。
よって 発走 ≤ min(ref, 最後の候補行 + 6 分) とし、それ以下で最も遅い 5 分刻みの時刻を推定値にする。区分 3 の刻印が
5 分以上遅れたレースでも、最後の候補行が上限を抑えるので遅い側に外れない。5 分刻みでない発走と最後の候補行が
欠けたレースでは早い側に外れる (T-x が早い行か欠損になるだけで、判断時刻より後の行は使わない)。

G1 と同じ日・同じ場で G1 より後のレースは 1 刻み (5 分) 早める。2007-16 年、一覧の時刻より JD の行の並び
(= 実際の締切) が 5 分遅いレースが G1 後のレースで 248 中 63、G2・G3 後で 1,309 中 6、それ以外で 31,022 中 27 あり
(表彰式による発走の遅れとみられる。2017 年以降の G1 後は 211 中 1)、一覧の時刻を正とすると推定が遅い側に外れるため。
2008-10 年の T-6 で推定の行が実時刻の行より遅いレースは、この補正で 24 → 8 (10,357 中) になる。
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

POST_GRID_MIN = 5
LAST_ROW_LEAD_MIN = 6
G1_CLASS_ORD = 7


def estimate_post(ref: float, cand_t: np.ndarray, after_g1: bool = False) -> float:
    """発走時刻の推定値 (1970-01-01 からの分)。cand_t はスナップショット候補行の時刻 (昇順、空でもよい)。"""
    upper = float(ref)
    if cand_t.size:
        upper = min(upper, float(cand_t[-1]) + LAST_ROW_LEAD_MIN)
    post = float(math.floor(upper / POST_GRID_MIN) * POST_GRID_MIN)
    return post - POST_GRID_MIN if after_g1 else post


def races_after_g1(races: pl.DataFrame) -> frozenset[int]:
    """G1 と同じ日・同じ場で G1 より後のレースの race_key。races: race_key, date, venue_code, race_num, class_ord。"""
    per_race = races.select("race_key", "date", "venue_code", "race_num", "class_ord").unique("race_key")
    g1 = (per_race.filter(pl.col("class_ord") == G1_CLASS_ORD)
          .group_by("date", "venue_code").agg(pl.col("race_num").min().alias("g1_race_num")))
    after = per_race.join(g1, on=["date", "venue_code"]).filter(pl.col("race_num") > pl.col("g1_race_num"))
    return frozenset(after["race_key"].to_list())
