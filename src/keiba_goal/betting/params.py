"""投票の設定値 (config/default.yaml の betting 節)。既定値は design §9 と PDF の値。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

# betting 節にあるが BettingParams では使わない項目 (読むモジュール)
OTHER_MODULE_KEYS: frozenset[str] = frozenset({
    "odds_drop_quantile",  # keiba_goal.data.build: q 表の分位
})


@dataclass(frozen=True)
class BettingParams:
    takeout: float = 0.20               # 控除率 c (単勝・複勝)
    ev_gate: float = 0.8                # τ: EV_goal ≥ τ のバスケットだけ推奨
    eff_odds_band_upper: float = 1.10   # 丸め後の有効オッズを [o*_k, 1.10·o*_k] に収める
    max_market_ratio: float = 2.0       # 乖離ガード R_max (p̂ / p_mkt)
    max_odds: float = 100.0             # 判断時の単勝オッズがこれ以上の馬は単勝・複勝とも外す
    top_n: int = 12
    max_basket: int = 4
    search_prefilter: bool = True       # 探索の前に õ < o*_k の券を外す (False は PDF の文字どおりの探索)
    unit: int = 100
    min_field: int = 5
    pl_tol: float = 1e-4
    pl_max_iter: int = 2000
    pl_damping: float = 0.7
    decision_lead_min: float = 6.0      # 判断時刻 = 発走 6 分前
    deadline_min: float = 3.0           # 発走 3 分前を過ぎたら判断しない
    chain_gap_min: float = 12.0         # k_eff の連鎖: 前の発走 + 12 分以降に判断時刻が来るレース
    snapshot_max_age_sec: float = 90.0  # 表示前の安全ゲート: 判断スナップショットの古さ上限
    day_budget: int = 10_000
    day_goal: int = 20_000
    day_bets: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.search_prefilter, bool):
            raise ValueError(f"search_prefilter は true / false です: {self.search_prefilter!r}")
        if not 0 <= self.takeout < 1:
            raise ValueError(f"takeout は [0, 1) です: {self.takeout}")
        if self.ev_gate < 0 or self.max_market_ratio <= 0 or self.max_odds <= 1:
            raise ValueError("ev_gate ≥ 0、max_market_ratio > 0、max_odds > 1 である必要があります")
        if self.eff_odds_band_upper < 1:
            raise ValueError("eff_odds_band_upper は 1 以上です")
        if not 1 <= self.max_basket <= self.top_n <= 62:
            raise ValueError("1 ≤ max_basket ≤ top_n ≤ 62 である必要があります")
        if self.unit <= 0 or self.min_field < 5:
            raise ValueError("unit は正、min_field は 5 以上です (学習・評価の対象は出走 5 頭以上)")
        if not (self.pl_tol > 0 and self.pl_max_iter >= 0 and 0 < self.pl_damping <= 1):
            raise ValueError("pl_tol > 0、pl_max_iter ≥ 0、0 < pl_damping ≤ 1 である必要があります")
        if not 0 <= self.deadline_min < self.decision_lead_min:
            raise ValueError("0 ≤ deadline_min < decision_lead_min である必要があります (判断は締切より前)")
        if self.chain_gap_min < 0 or self.snapshot_max_age_sec <= 0:
            raise ValueError("chain_gap_min ≥ 0、snapshot_max_age_sec > 0 である必要があります")
        if not (0 < self.day_budget < self.day_goal and self.day_bets >= 1):
            raise ValueError("0 < day_budget < day_goal、day_bets ≥ 1 である必要があります")

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> BettingParams:
        """betting 節から読む。節に無い項目は既定値。知らない項目 (書き間違い) と型の合わない値は ValueError。"""
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(section) - set(known) - OTHER_MODULE_KEYS)
        if unknown:
            raise ValueError(f"betting 節に不明な項目があります: {unknown}")
        return cls(**{
            name: _coerce(name, section[name], type(f.default)) for name, f in known.items() if name in section
        })


def _coerce(name: str, value: Any, kind: type) -> Any:
    """YAML の値を既定値と同じ型にする。文字列や bool からの変換、12.7 → 12 の切り捨てはしない。"""
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"betting.{name} は true / false です: {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"betting.{name} は数値です: {value!r} (YAML では 1e-4 は文字列になる。1.0e-4 と書く)")
    if math.isnan(value):
        raise ValueError(f"betting.{name} が NaN です")
    if kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"betting.{name} は整数です: {value!r}")
        return int(value)
    return float(value)
