import math

import numpy as np
import pytest

from keiba_goal.betting.theory import (
    bold_stake,
    end_reason,
    no_edge_day,
    reach_ceiling,
    reach_prob,
    shortfall_ratio,
    target_odds,
)

A_PDF = 2_440_000  # PDF 9〜11 枚目 (2026-09-21 阪神)


@pytest.mark.parametrize(
    ("x", "k", "expected"),
    [(1_965_730, 7, 4.7930), (1_845_830, 6, 4.7671), (1_694_430, 5, 4.7370)],
)
def test_pdf_target_odds(x, k, expected):
    assert round(target_odds(x, A_PDF, k), 4) == expected


def test_pdf_reach_probability():
    assert round(reach_prob(1_965_730, A_PDF, 7, 0.20) * 100, 4) == 72.1485


def test_pdf_capital_path_recomputes_o_star_from_the_actual_capital():
    # 8R 119,900 外れ → 9R 151,400 外れ → 10R 192,000。o* は実際の資金から毎回計算し直す (4.793 → 4.767 → 4.737)
    x = 1_965_730
    for k, stake, o_star_pdf, ratio_pdf in ((7, 119_900, 4.7930, 1.0339), (6, 151_400, 4.7671, 1.0330),
                                            (5, 192_000, 4.7370, 1.0309)):
        o_star = target_odds(x, A_PDF, k)
        assert round(o_star, 4) == o_star_pdf
        # 当たればちょうど A に届く有効オッズ 1 + (A − x)/s は帯 [o*, 1.10·o*] の中 (PDF の o_eff/o* = 1.031〜1.034)
        assert round((1 + (A_PDF - x) / stake) / o_star, 4) == ratio_pdf
        assert bold_stake(x, A_PDF, o_star) > stake  # 有効オッズが o* より高いぶん bold stake より少ない
        x -= stake


def test_target_odds_is_invariant_after_a_miss_only_with_the_exact_bold_stake():
    x, k = 1_965_730, 7
    o = target_odds(x, A_PDF, k)
    exact = bold_stake(x, A_PDF, o)
    assert target_odds(x - exact, A_PDF, k - 1) == pytest.approx(o, rel=1e-12)
    rounded = math.ceil(exact / 100) * 100
    assert abs(target_odds(x - rounded, A_PDF, k - 1) - o) > 1e-5
    assert abs(target_odds(x - 119_900, A_PDF, k - 1) - o) > 0.02  # PDF の実際の賭け金


def test_exact_bold_play_ends_at_goal_or_zero():
    x, goal, k = 10_000.0, 20_000.0, 3
    o = target_odds(x, goal, k)
    for _ in range(k):
        s = bold_stake(x, goal, o)
        assert x - s + s * o == pytest.approx(goal)
        x -= s
    assert x == pytest.approx(0.0, abs=1e-6)


def test_default_day_without_edge():
    day = no_edge_day(10_000, 20_000, 3, 0.20)
    assert day.o_star == pytest.approx(4.847, abs=5e-4)
    assert round(day.reach, 3) == 0.418
    assert round(day.ceiling, 3) == 0.426
    assert day.expected_end == pytest.approx(8_358, abs=1)
    assert round(day.expected_change, 3) == -0.164
    assert round(day.share_lose_all, 2) == 0.58
    assert day.x_over_goal == 0.5
    assert day.p_star == pytest.approx(0.8 / day.o_star)


def test_reach_is_below_the_tight_ceiling_which_is_below_x_over_a():
    for xa in np.linspace(0.01, 0.99, 50):
        for k in range(1, 31):
            for c in (0.05, 0.20, 0.25, 0.50):
                p = reach_prob(xa, 1.0, k, c, unit=0)
                ceiling = reach_ceiling(xa, 1.0, c)
                assert p <= ceiling + 1e-12
                assert ceiling <= xa + 1e-12


@pytest.mark.parametrize("xa", [0.05, 1 / 3, 0.5, 0.8056])
@pytest.mark.parametrize("k", [1, 3, 7, 20])
def test_bounds_are_equal_without_takeout(xa, k):
    assert reach_prob(xa, 1.0, k, 0.0, unit=0) == pytest.approx(xa, rel=1e-12)
    assert reach_ceiling(xa, 1.0, 0.0) == pytest.approx(xa, rel=1e-12)


def test_bold_play_approaches_the_ceiling_as_k_grows():
    values = [reach_prob(1 / 3, 1.0, k, 0.2, unit=0) for k in (1, 2, 3, 6, 50, 2000)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(reach_ceiling(1 / 3, 1.0, 0.2), abs=1e-4)


def test_ceiling_holds_for_any_bet_with_return_at_most_rho():
    # g(x) = 1 − (1 − x/A)^ρ は 1 円あたりの払戻の期待値が ρ 以下のどんな賭けでも増えない (優マルチンゲール)
    rng = np.random.default_rng(0)
    rho = 0.8
    worst = -1.0
    for _ in range(5_000):
        x = rng.uniform(0.01, 0.99)
        stake = rng.uniform(0, x)
        n = int(rng.integers(1, 5))
        probs = rng.dirichlet(np.ones(n + 1))
        pay = rng.exponential(1.0, n)
        pay = pay / (probs[:n] @ pay) * rho * stake
        after = np.append(x - stake + pay, x - stake)
        g_after = np.array([reach_ceiling(v, 1.0, 1 - rho) for v in after])
        worst = max(worst, float(probs @ g_after) - reach_ceiling(x, 1.0, 1 - rho))
    assert worst <= 1e-12


def test_k1_target_is_a_over_x():
    assert target_odds(4_000, 20_000, 1) == pytest.approx(5.0)


def test_end_conditions_and_guards():
    assert end_reason(20_000, 20_000, 3) == "goal"
    assert end_reason(5_000, 20_000, 0) == "no_bets"
    assert end_reason(99, 20_000, 3) == "no_funds"
    assert end_reason(100, 20_000, 1) is None
    assert reach_prob(20_000, 20_000, 3, 0.2) == 1.0
    assert reach_prob(5_000, 20_000, 0, 0.2) == 0.0
    assert reach_prob(99, 20_000, 3, 0.2) == 0.0
    assert shortfall_ratio(15_000, 20_000) == 0.25
    for x in (20_000, 25_000):  # r ≤ 0 で r^(1/k) を計算しない
        with pytest.raises(ValueError):
            target_odds(x, 20_000, 3)
    with pytest.raises(ValueError):
        target_odds(10_000, 20_000, 0)
    with pytest.raises(ValueError):
        bold_stake(10_000, 20_000, 1.0)
    with pytest.raises(ValueError):
        bold_stake(20_000, 20_000, 3.0)
    assert reach_ceiling(20_000, 20_000, 0.2) == 1.0
    assert reach_ceiling(0, 20_000, 0.2) == 0.0
