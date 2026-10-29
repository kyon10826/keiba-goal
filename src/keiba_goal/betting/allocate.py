"""100 円単位の配分: どの券が当たっても目標 A に届く最小の金額 (design §9.3-5)。

券 i に n_i 単位 (s_i = 100·n_i) を賭けるとき、条件は全 i で s_i·õ_i ≥ (A − x) + Σs。
n ← 0 から n_i ← ceil(((A − x) + 100·Σn)/(100·õ_i)) を変化しなくなるまで繰り返すと、
単調に増えて条件を満たす最小の n (どれか 1 つでも減らすと条件を破る) に止まる。
100·Σn > x になったら実現不能 (上限で切ると当たっても A に届かない券ができるため、切らない)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Allocation:
    stakes: tuple[int, ...]
    odds_adj: tuple[float, ...]
    total: int
    min_payout: float

    @property
    def o_eff(self) -> float:
        """丸め後の有効オッズ min_i(s_i·õ_i) / Σs。"""
        return self.min_payout / self.total

    def bankroll_if_hit(self, x: float) -> float:
        """どれか 1 枚が õ で当たったときの資金の下限。"""
        return x - self.total + self.min_payout


def effective_odds(odds_adj: Sequence[float]) -> float:
    """丸める前の有効オッズ (Σ 1/õ)^−1。丸め後の o_eff はこれを超えない。"""
    return 1.0 / sum(1.0 / o for o in odds_adj)


def _units_needed(need: float, odds: float, unit: int) -> int:
    """unit·k·odds ≥ need となる最小の k (浮動小数の誤差で 1 ずれないよう両側を確かめる)。"""
    k = max(0, math.ceil(need / (unit * odds)))
    while k > 0 and unit * (k - 1) * odds >= need:
        k -= 1
    while unit * k * odds < need:
        k += 1
    return k


def allocate(odds_adj: Sequence[float], x: float, goal: float, unit: int = 100) -> Allocation | None:
    """õ (下落補正後のオッズ) の券の組に 100 円単位で配分する。実現不能なら None。"""
    odds = tuple(float(o) for o in odds_adj)
    if not odds:
        raise ValueError("券が 1 枚もありません")
    if any(not (math.isfinite(o) and o > 0) for o in odds):
        raise ValueError(f"オッズは正の有限値です: {odds}")
    if x >= goal:
        raise ValueError("目標に到達済みです")
    if sum(1.0 / o for o in odds) >= 1.0:
        return None
    deficit = goal - x
    n = [0] * len(odds)
    # Σn は反復ごとに 1 以上増えるので、x/unit + 1 回以内に止まるか実現不能になる
    for _ in range(int(x // unit) + 2):
        need = deficit + unit * sum(n)
        new = [_units_needed(need, o, unit) for o in odds]
        if unit * sum(new) > x:
            return None
        if new == n:
            stakes = tuple(unit * k for k in n)
            payout = min(s * o for s, o in zip(stakes, odds))
            return Allocation(stakes=stakes, odds_adj=odds, total=sum(stakes), min_payout=payout)
        n = new
    raise RuntimeError("配分の反復が止まりませんでした")
