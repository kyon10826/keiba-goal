import itertools
import math

import numpy as np
import pytest

from keiba_goal.betting.allocate import allocate, effective_odds
from keiba_goal.betting.theory import target_odds

UNIT = 100


def _assert_valid_and_minimal(alloc, odds, x, goal):
    deficit = goal - x
    total = alloc.total
    assert total == sum(alloc.stakes) <= x
    assert all(s > 0 and s % UNIT == 0 for s in alloc.stakes)
    payouts = [s * o for s, o in zip(alloc.stakes, odds)]
    assert min(payouts) >= deficit + total  # どれが当たっても A に届く
    assert alloc.min_payout == min(payouts)
    assert alloc.bankroll_if_hit(x) >= goal
    assert alloc.o_eff == pytest.approx(min(payouts) / total)
    assert alloc.o_eff <= effective_odds(odds) * (1 + 1e-12)
    for s, o in zip(alloc.stakes, odds):  # どれか 1 つ減らすと条件を破る
        assert (s - UNIT) * o < deficit + (total - UNIT)


def test_random_baskets_satisfy_goal_budget_unit_and_minimality():
    rng = np.random.default_rng(1)
    checked = infeasible = 0
    for _ in range(3_000):
        odds = rng.uniform(1.3, 120.0, int(rng.integers(1, 5)))
        x = float(rng.integers(10, 30_000) * 10)
        goal = x * rng.uniform(1.05, 6.0)
        alloc = allocate(odds, x, goal)
        if alloc is None:
            infeasible += 1
            continue
        checked += 1
        _assert_valid_and_minimal(alloc, odds, x, goal)
    assert checked > 500 and infeasible > 100


def test_matches_brute_force_least_solution():
    rng = np.random.default_rng(2)
    for _ in range(150):
        m = int(rng.integers(1, 4))
        odds = np.round(rng.uniform(1.5, 30.0, m), 2)
        x = int(rng.integers(5, 21)) * UNIT
        goal = x + int(rng.integers(1, 80)) * UNIT + int(rng.integers(0, UNIT))
        cap = x // UNIT
        grid = np.array(list(itertools.product(range(cap + 1), repeat=m)))
        total = grid.sum(axis=1)
        ok = (UNIT * total <= x) & np.all(UNIT * grid * odds >= (goal - x) + UNIT * total[:, None], axis=1)
        alloc = allocate(odds, x, goal)
        if not ok.any():
            assert alloc is None
            continue
        assert alloc is not None
        units = np.array(alloc.stakes) // UNIT
        assert alloc.total == UNIT * total[ok].min()
        assert np.all(grid[ok] >= units)  # 成分ごとに最小の解


def test_ceil_fixed_point_example_from_review():
    # 10,000 → 30,000、õ = [40, 50, 60]: 素朴な丸め [500, 400, 400] では 40 倍と 50 倍が当たっても届かない
    alloc = allocate([40.0, 50.0, 60.0], 10_000, 30_000)
    assert alloc.stakes == (600, 500, 400)
    assert [s * o for s, o in zip(alloc.stakes, [40, 50, 60])] == [24_000, 25_000, 24_000]
    assert 20_000 + alloc.total == 21_500


def test_infeasible_cases_are_not_capped():
    assert allocate([2.0, 2.0], 1_000, 2_000) is None  # Σ1/õ = 1
    assert allocate([1.5, 2.5], 1_000, 2_000) is None  # Σ1/õ > 1
    assert allocate([4.99], 4_000, 20_000) is None  # 届くには x を超える
    exact = allocate([5.0], 4_000, 20_000)  # k = 1 の下限 A/x ちょうど
    assert exact is not None and exact.stakes == (4_000,) and exact.bankroll_if_hit(4_000) == 20_000


def test_invalid_inputs():
    with pytest.raises(ValueError):
        allocate([], 1_000, 2_000)
    with pytest.raises(ValueError):
        allocate([float("nan")], 1_000, 2_000)
    with pytest.raises(ValueError):
        allocate([0.0], 1_000, 2_000)
    with pytest.raises(ValueError):
        allocate([5.0], 2_000, 2_000)


X_10R, A_10R = 1_694_430, 2_440_000
O11 = 8.4 * 0.787  # 11 番: T-5 表示 8.4 倍 × q ≈ 0.787 → õ ≈ 6.611
O2 = 23.8 * 0.787  # 2 枚目 (未公開): õ₂ ∈ [18.714, 18.747) なら PDF の配分がそのまま再現する


def test_pdf_10r_single_ticket_on_horse_11():
    alloc = allocate([O11], X_10R, A_10R)
    assert alloc.stakes == (132_900,)
    assert alloc.bankroll_if_hit(X_10R) >= A_10R
    # PDF は 11 番に 141,900。11 番だけの 132,900 との差は、2 枚目の賭け金も 11 番の払戻で賄う分
    # (2 枚目 / (õ − 1) を 100 円単位に切り上げた額)
    two = allocate([O11, O2], X_10R, A_10R)
    assert two.stakes[0] - alloc.stakes[0] == UNIT * math.ceil(two.stakes[1] / (UNIT * (O11 - 1)))


def test_pdf_10r_two_ticket_basket_reproduces_the_stakes():
    alloc = allocate([O11, O2], X_10R, A_10R)
    assert alloc.stakes == (141_900, 50_100)
    assert alloc.total == 192_000
    assert alloc.bankroll_if_hit(X_10R) >= A_10R
    o_star = target_odds(X_10R, A_10R, 5)
    assert o_star <= alloc.o_eff <= 1.10 * o_star
    assert alloc.o_eff / o_star == pytest.approx(1.031, abs=1e-3)  # PDF の実購入 (o_eff/o* = 1.031〜1.034)
    # 実際は確定 8.0 倍に下落しても 2,637,630 に到達 (PDF の最終資金)
    assert X_10R - alloc.total + alloc.stakes[0] * 8.0 == 2_637_630
    pdf = (141_900, 50_100)
    assert all(allocate([O11, o2], X_10R, A_10R).stakes == pdf for o2 in (18.7140, 18.7300, 18.7473))
    assert all(allocate([O11, o2], X_10R, A_10R).stakes != pdf for o2 in (18.7139, 18.7475))
