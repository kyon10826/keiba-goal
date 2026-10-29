"""Goal-reaching の理論式と上限 (design §9.2、PDF 9 枚目)。

現在資金 x、目標 A、残り回数 k、控除率 c:
r = (A − x)/A、o*_k = 1/(1 − r^(1/k))、p* = (1 − c)/o*_k、P*_k = 1 − (1 − p*)^k、bold stake s = (A − x)/(o − 1)。
P*_k は独立性・校正・一定の控除率を仮定した理論値で、保証ではない。
"""

from __future__ import annotations

from dataclasses import dataclass


def shortfall_ratio(x: float, goal: float) -> float:
    """不足率 r = (A − x) / A。"""
    if goal <= 0:
        raise ValueError(f"目標は正の値です: {goal}")
    return (goal - x) / goal


def end_reason(x: float, goal: float, k: int, unit: int = 100) -> str | None:
    """終了条件: 資金 ≥ A は 'goal'、k = 0 は 'no_bets'、資金 < 100 円は 'no_funds'、続けるなら None。"""
    if x >= goal:
        return "goal"
    if k <= 0:
        return "no_bets"
    if x < unit:
        return "no_funds"
    return None


def target_odds(x: float, goal: float, k: int) -> float:
    """狙う有効オッズ o*_k。0 < x < A かつ k ≥ 1 でだけ定義する (r ≤ 0 で r^(1/k) を計算しない)。"""
    if k < 1:
        raise ValueError(f"残り回数 k は 1 以上です: {k}")
    if not 0 < x < goal:
        raise ValueError(f"o*_k は 0 < x < A でだけ定義されます (x={x}, A={goal})")
    r = shortfall_ratio(x, goal)
    return 1.0 / (1.0 - r ** (1.0 / k))


def fair_hit_prob(odds: float, takeout: float) -> float:
    """控除率 c の市場で有効オッズ o の賭けが当たる公正な確率 p* = (1 − c) / o。"""
    if odds <= 0:
        raise ValueError(f"オッズは正の値です: {odds}")
    return (1.0 - takeout) / odds


def reach_prob(x: float, goal: float, k: int, takeout: float, unit: int = 100) -> float:
    """エッジなしで bold play したときの到達確率 P*_k。到達済みは 1、終了 (k = 0 / x < 100) は 0。"""
    reason = end_reason(x, goal, k, unit)
    if reason == "goal":
        return 1.0
    if reason is not None:
        return 0.0
    p = fair_hit_prob(target_odds(x, goal, k), takeout)
    return 1.0 - (1.0 - p) ** k


def bold_stake(x: float, goal: float, odds: float) -> float:
    """当たればちょうど A に届く賭け金 s = (A − x)/(o − 1) (丸めない)。"""
    if odds <= 1:
        raise ValueError(f"有効オッズは 1 より大きい必要があります: {odds}")
    if x >= goal:
        raise ValueError("目標に到達済みです")
    return (goal - x) / (odds - 1.0)


def reach_ceiling(x: float, goal: float, takeout: float) -> float:
    """1 円あたりの払戻率が ρ = 1 − c 以下の賭けだけなら、どんな買い方でも P(到達) ≤ 1 − (1 − x/A)^ρ ≤ x/A。"""
    if x >= goal:
        return 1.0
    if x <= 0:
        return 0.0
    return 1.0 - (1.0 - x / goal) ** (1.0 - takeout)


@dataclass(frozen=True)
class NoEdgeDay:
    """市場と同じ精度 (エッジなし) で bold play した 1 日の理論値 (design §0 の表)。"""

    x: float
    goal: float
    k: int
    takeout: float
    o_star: float
    p_star: float
    reach: float
    ceiling: float
    x_over_goal: float
    expected_end: float
    expected_change: float
    share_lose_all: float


def no_edge_day(x: float, goal: float, k: int, takeout: float) -> NoEdgeDay:
    """bold play は A ちょうどか (k 回外れて) ほぼ 0 で終わるので、終了時資金の期待値は P*·A、全損の日の割合は 1 − P*。"""
    o_star = target_odds(x, goal, k)
    p_star = fair_hit_prob(o_star, takeout)
    reach = 1.0 - (1.0 - p_star) ** k
    expected_end = reach * goal
    return NoEdgeDay(
        x=x,
        goal=goal,
        k=k,
        takeout=takeout,
        o_star=o_star,
        p_star=p_star,
        reach=reach,
        ceiling=reach_ceiling(x, goal, takeout),
        x_over_goal=x / goal,
        expected_end=expected_end,
        expected_change=expected_end / x - 1.0,
        share_lose_all=1.0 - reach,
    )
