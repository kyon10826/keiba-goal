"""買い目の決定: 候補・除外・探索・ゲート (design §9.3)。

1. 候補は判断時に単勝オッズが付いている馬 (field_dec) の単勝・複勝。複勝の支払着数 m (0/2/3) は呼び出し側が
   発売開始時の頭数から決めて渡す (バックテストはデータ層の paid_places 表、ライブは P7 の規則。0 は複勝の発売なし)。
2. õ = o × q(券種, 判断時オッズの帯)。
3. 除外: 判断時の単勝 ≥ 100 倍の馬は単複とも / オッズ欠損 / p̂ / p_mkt > R_max (乖離ガード)。
   同じ馬の単勝と複勝は同じバスケットに入れない。
4. 探索: 候補を p̂·õ の降順に top_n 枚、その 4 枚以下の全組合せ。既定 (search_prefilter) は õ ≥ o*_k の候補だけから
   top_n 枚を取る (o_eff ≤ min õ なので õ < o*_k の券を含むバスケットは帯に入らない)。False は PDF の文字どおり。
5. 100 円単位の配分 → 丸め後の o_eff が [o*_k, 1.10·o*_k]、EV_goal = P_hit·o_eff ≥ τ のうち P_hit 最大
   (同点は合計金額が小さい方)。
`prepare` は資金や回数によらない部分 (除外・順序表) で、レースごとに 1 回作って使い回す。
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from keiba_goal.betting.allocate import Allocation, allocate
from keiba_goal.betting.joint import OutcomeTable, build_outcome_table, check_paid_places, subset_hit_probs
from keiba_goal.betting.params import BettingParams
from keiba_goal.betting.theory import end_reason, reach_ceiling, reach_prob, target_odds
from keiba_goal.betting.tickets import BetType, Ticket, total_stake

QTable = Mapping[float, float]  # 帯の下端 (判断時オッズ。複勝は Lo) → q
_BET_ORDER = {"win": 0, "place": 1}
_REL = 1e-12
_MARKET_SUM_TOL = 1e-4


@dataclass(frozen=True)
class RaceInput:
    """判断時点の 1 レースの馬ごとの配列 (同じ長さ・同じ並び)。

    p_win は field_dec (判断時に単勝オッズが付いている馬) の中で合計 1 に正規化済み。
    win_odds / place_lo は判断スナップショットの単勝と複勝 Lo (売られていなければ NaN)。
    p_mkt_* は判断時の市場確率 (market_probs に prepare と同じ支払着数を渡して作る)。
    """

    horse_num: np.ndarray
    p_win: np.ndarray
    p_place: np.ndarray
    win_odds: np.ndarray
    place_lo: np.ndarray
    p_mkt_win: np.ndarray
    p_mkt_place: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "horse_num", np.asarray(self.horse_num, dtype=np.int64))
        for name in ("p_win", "p_place", "win_odds", "place_lo", "p_mkt_win", "p_mkt_place"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        n = self.horse_num.size
        bad = [name for name in ("p_win", "p_place", "win_odds", "place_lo", "p_mkt_win", "p_mkt_place")
               if getattr(self, name).shape != (n,)]
        if bad:
            raise ValueError(f"配列の長さが馬番 ({n}) と違います: {bad}")
        if np.unique(self.horse_num).size != n:
            raise ValueError("馬番が重複しています")


def _normalized_inverse(odds: np.ndarray, total: float) -> np.ndarray:
    odds = np.asarray(odds, dtype=np.float64)
    ok = np.isfinite(odds) & (odds > 0)
    inv = np.full(odds.shape, np.nan)
    inv[ok] = 1.0 / odds[ok]
    if ok.any():
        inv[ok] *= total / inv[ok].sum()
    return inv


def market_probs(win_odds: np.ndarray, place_lo: np.ndarray, n_paid_places: int) -> tuple[np.ndarray, np.ndarray]:
    """判断時の市場確率: 単勝は (1/o)/Σ(1/o)、複勝は m·(1/Lo)/Σ(1/Lo) (m = 支払着数)。

    売られていない馬と、複勝の発売が無いレース (m = 0) の複勝は NaN。
    """
    m = check_paid_places(n_paid_places)
    win = _normalized_inverse(win_odds, 1.0)
    if m == 0:
        return win, np.full(np.shape(place_lo), np.nan)
    return win, _normalized_inverse(place_lo, float(m))


def drop_factor(q_table: QTable, odds: float, max_odds: float) -> float | None:
    """判断時オッズ odds が入る帯 [下端, 次の下端) の q。最後の帯の上端は max_odds。範囲外・欠損は None。

    data 層の q 表からは {b.lo: b.q for b in table.bands[券種]} で作れる (結合された帯もそのまま使える)。
    """
    if not math.isfinite(odds) or odds >= max_odds or not q_table:
        return None
    edges = sorted(q_table)
    i = bisect.bisect_right(edges, odds) - 1
    return float(q_table[edges[i]]) if i >= 0 else None


@dataclass(frozen=True)
class Candidate:
    bet_type: BetType
    horse_num: int
    odds: float      # 判断時のオッズ (複勝は Lo)
    q: float         # 下落補正
    odds_adj: float  # õ = o·q
    p_hat: float     # 単勝: 正規化済み p_win / 複勝: 射影後の複勝確率 (順序表の目標値)
    p_mkt: float

    @property
    def score(self) -> float:
        return self.p_hat * self.odds_adj

    def to_dict(self) -> dict[str, Any]:
        return {
            "bet_type": self.bet_type, "horse_num": self.horse_num, "odds": self.odds, "q": self.q,
            "odds_adj": self.odds_adj, "p_hat": self.p_hat, "p_mkt": self.p_mkt,
        }


@dataclass(frozen=True)
class Excluded:
    bet_type: BetType
    horse_num: int
    # no_odds / longshot / no_place_odds / no_place_sales / no_q / no_market_prob / disagreement / joint_not_converged
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class PreparedRace:
    """資金・回数によらない部分 (レースごとにキャッシュする)。順序表が無いときは skip_reason に理由。"""

    candidates: tuple[Candidate, ...]
    excluded: tuple[Excluded, ...]
    table: OutcomeTable | None
    masks: np.ndarray
    field_size: int
    n_paid_places: int
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.table is None) != (self.skip_reason is not None):
            raise ValueError("順序表が無いときだけ skip_reason を付けます")


@dataclass(frozen=True)
class Decision:
    """1 レースの推奨 (tickets が空なら見送りで、skip_reason に理由)。"""

    x: float
    goal: float
    k_eff: int
    p_star: float      # P*_{k_eff}(x) (エッジなしの理論値)
    ceiling: float     # 1 − (1 − x/A)^(1−c)
    o_star: float | None = None
    tickets: tuple[Ticket, ...] = ()
    lines: tuple[Candidate, ...] = ()
    o_eff: float | None = None
    p_hit: float | None = None
    ev_goal: float | None = None
    ev_q: float | None = None
    ev_raw: float | None = None
    bankroll_if_hit: float | None = None
    skip_reason: str | None = None
    search_prefilter: bool | None = None  # 探索の前絞り込みを使ったか (decide が必ず入れる)
    excluded: tuple[Excluded, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def recommended(self) -> bool:
        return bool(self.tickets)

    @property
    def total_stake(self) -> int:
        return total_stake(self.tickets)

    @property
    def model_claims_edge(self) -> bool:
        """モデルの EV > 1 (= モデルは市場に勝てると言っている。資料はどちらもそれを支持していない)。"""
        return self.ev_q is not None and self.ev_q > 1.0

    def summary(self) -> dict[str, Any]:
        """イベントログ・表示用の JSON にできる要約。"""
        return {
            "x": self.x, "goal": self.goal, "k_eff": self.k_eff, "o_star": self.o_star, "o_eff": self.o_eff,
            "p_hit": self.p_hit, "ev_goal": self.ev_goal, "ev_q": self.ev_q, "ev_raw": self.ev_raw,
            "p_star": self.p_star, "ceiling": self.ceiling, "bankroll_if_hit": self.bankroll_if_hit,
            "total_stake": self.total_stake, "skip_reason": self.skip_reason,
            "search_prefilter": self.search_prefilter, "model_claims_edge": self.model_claims_edge,
            "tickets": [t.to_dict() for t in self.tickets],
            "lines": [c.to_dict() for c in self.lines],
            "excluded": [{"bet_type": e.bet_type, "horse_num": e.horse_num, "reason": e.reason, "detail": e.detail}
                         for e in self.excluded],
            "diagnostics": dict(self.diagnostics),
        }


def _candidate(
    bet_type: BetType, horse_num: int, odds: float, q_table: QTable, p_hat: float, p_mkt: float,
    params: BettingParams,
) -> Candidate | Excluded:
    if not (math.isfinite(odds) and odds > 0):
        return Excluded(bet_type, horse_num, "no_odds" if bet_type == "win" else "no_place_odds")
    q = drop_factor(q_table, odds, params.max_odds)
    if q is None or not (math.isfinite(q) and q > 0):
        return Excluded(bet_type, horse_num, "no_q", f"odds={odds}")
    if not (math.isfinite(p_mkt) and p_mkt > 0):
        return Excluded(bet_type, horse_num, "no_market_prob")
    if p_hat / p_mkt > params.max_market_ratio:
        return Excluded(bet_type, horse_num, "disagreement", f"p_hat={p_hat:.4f} p_mkt={p_mkt:.4f}")
    return Candidate(bet_type, horse_num, float(odds), float(q), float(odds) * float(q), float(p_hat), float(p_mkt))


def _check_market_sums(race: RaceInput, m: int) -> None:
    """市場確率が同じ支払着数で作られているか (単勝の合計 1、複勝の合計 m)。違うと乖離ガードがずれる。"""
    if m == 0 and np.isfinite(race.p_mkt_place).any():
        raise ValueError("複勝の発売が無いレース (n_paid_places=0) に p_mkt_place があります (market_probs に同じ 0 を渡す)")
    checks = [("p_mkt_win", race.p_mkt_win, 1.0)] + ([("p_mkt_place", race.p_mkt_place, float(m))] if m else [])
    for name, p, total in checks:
        ok = np.isfinite(p)
        if ok.any() and abs(p[ok].sum() - total) > _MARKET_SUM_TOL * total:
            raise ValueError(f"{name} の合計 {p[ok].sum():.6f} が {total:g} ではありません "
                             f"(market_probs に prepare と同じ支払着数 {m} を渡す)")


def prepare(
    race: RaceInput, q_win: QTable, q_place: QTable, params: BettingParams, *, n_paid_places: int,
) -> PreparedRace:
    """候補と除外、順序表を作る (資金・回数によらない)。

    n_paid_places は複勝の支払着数 (発売開始時の頭数で決まる 2 / 3、複勝の発売なしは 0)。判断時の頭数からは決めない。
    """
    m = check_paid_places(n_paid_places)
    priced = np.isfinite(race.win_odds) & (race.win_odds > 0)
    excluded: list[Excluded] = [
        Excluded(b, int(h), "no_odds") for h in race.horse_num[~priced] for b in ("win", "place")
    ]
    idx = np.flatnonzero(priced)
    idx = idx[np.argsort(race.horse_num[idx], kind="stable")]
    n = int(idx.size)
    if n == 0 or n < params.min_field:
        # 判断スナップショットが無い (どの馬にも単勝オッズが無い) のは頭数不足ではなくオッズ欠損
        return PreparedRace((), tuple(excluded), None, np.zeros((0, 0), bool), n, m,
                            "no_odds" if n == 0 else "small_field")
    _check_market_sums(race, m)
    table = build_outcome_table(
        race.horse_num[idx], race.p_win[idx], race.p_place[idx] if m else None, m,
        tol=params.pl_tol, max_iter=params.pl_max_iter, damping=params.pl_damping,
    )
    candidates: list[Candidate] = []
    for j, i in enumerate(idx):
        h = int(race.horse_num[i])
        if race.win_odds[i] >= params.max_odds:
            excluded += [Excluded(b, h, "longshot", f"win_odds={race.win_odds[i]}") for b in ("win", "place")]
            continue
        tickets: list[Candidate | Excluded] = [
            _candidate("win", h, race.win_odds[i], q_win, table.p_win[j], race.p_mkt_win[i], params)
        ]
        if m == 0:
            tickets.append(Excluded("place", h, "no_place_sales"))
        elif table.converged:
            tickets.append(
                _candidate("place", h, race.place_lo[i], q_place, table.p_place[j], race.p_mkt_place[i], params)
            )
        else:
            tickets.append(Excluded("place", h, "joint_not_converged", f"max_error={table.max_error:.2e}"))
        for t in tickets:
            if isinstance(t, Candidate):
                candidates.append(t)
            else:
                excluded.append(t)
    masks = table.ticket_masks((c.bet_type, c.horse_num) for c in candidates)
    return PreparedRace(tuple(candidates), tuple(excluded), table, masks, n, m)


def _subsets(pool: list[Candidate], max_size: int) -> list[tuple[int, ...]]:
    """順位の組 (同じ馬の単勝と複勝を同時に含まない)。"""
    out = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(range(len(pool)), size):
            if len({pool[r].horse_num for r in combo}) == size:
                out.append(combo)
    return out


def decide(prepared: PreparedRace, x: float, goal: float, k_eff: int, params: BettingParams) -> Decision:
    """資金 x・目標 A・実効の残り回数 k_eff で推奨バスケットを決める。"""
    base: dict[str, Any] = {
        "x": x, "goal": goal, "k_eff": k_eff, "excluded": prepared.excluded,
        "p_star": reach_prob(x, goal, k_eff, params.takeout, params.unit),
        "ceiling": reach_ceiling(x, goal, params.takeout),
        "search_prefilter": params.search_prefilter,
    }
    diag: dict[str, Any] = {"field_size": prepared.field_size, "n_paid_places": prepared.n_paid_places,
                            "n_candidates": len(prepared.candidates)}
    if prepared.table is not None:
        diag.update(joint_converged=prepared.table.converged, joint_iterations=prepared.table.iterations,
                    joint_max_error=prepared.table.max_error)
    ended = end_reason(x, goal, k_eff, params.unit)
    if ended is not None:
        return Decision(**base, skip_reason=ended, diagnostics=diag)
    if prepared.table is None:
        return Decision(**base, skip_reason=prepared.skip_reason, diagnostics=diag)

    o_star = target_odds(x, goal, k_eff)
    lo_b, hi_b = o_star * (1 - _REL), params.eff_odds_band_upper * o_star * (1 + _REL)
    order = sorted(
        (i for i, c in enumerate(prepared.candidates) if not params.search_prefilter or c.odds_adj >= lo_b),
        key=lambda i: (-prepared.candidates[i].score, prepared.candidates[i].horse_num,
                       _BET_ORDER[prepared.candidates[i].bet_type]),
    )[: params.top_n]
    pool = [prepared.candidates[i] for i in order]
    diag["n_pool"] = len(pool)
    if not pool:
        return Decision(**base, o_star=o_star, skip_reason="no_candidates", diagnostics=diag)

    subsets = _subsets(pool, params.max_basket)
    deficit = goal - x
    kept: list[tuple[tuple[int, ...], Allocation]] = []
    for combo in subsets:
        odds = [pool[r].odds_adj for r in combo]
        inv = sum(1.0 / o for o in odds)
        # 丸め後の o_eff は (Σ1/õ)^−1 以下、合計金額は (A − x)/(o_eff − 1) 以上なので、ここで外れるものは配分しない
        if inv >= 1.0 or 1.0 / inv < lo_b or deficit / (1.0 / inv - 1.0) > x:
            continue
        alloc = allocate(odds, x, goal, params.unit)
        if alloc is not None and lo_b <= alloc.o_eff <= hi_b:
            kept.append((combo, alloc))
    diag.update(n_subsets=len(subsets), n_feasible=len(kept))
    if not kept:
        return Decision(**base, o_star=o_star, skip_reason="infeasible", diagnostics=diag)

    masks = prepared.masks[order]
    p_hit = subset_hit_probs(prepared.table.prob, masks, [c for c, _ in kept])
    ev_goal = p_hit * np.array([a.o_eff for _, a in kept])
    passing = np.flatnonzero(ev_goal >= params.ev_gate)
    diag.update(n_passed=int(passing.size), best_ev_goal=float(ev_goal.max()))
    if passing.size == 0:
        return Decision(**base, o_star=o_star, skip_reason="gate", diagnostics=diag)

    best_p = float(p_hit[passing].max())
    ties = [int(j) for j in passing if p_hit[j] >= best_p - 1e-12]
    j = min(ties, key=lambda j: (kept[j][1].total, len(kept[j][0]), kept[j][0]))
    combo, alloc = kept[j]
    lines = tuple(pool[r] for r in combo)
    _check_hit_prob(float(p_hit[j]), lines, prepared.table)
    total = alloc.total
    return Decision(
        **base,
        o_star=o_star,
        tickets=tuple(Ticket(c.bet_type, c.horse_num, s) for c, s in zip(lines, alloc.stakes)),
        lines=lines,
        o_eff=alloc.o_eff,
        p_hit=float(p_hit[j]),
        ev_goal=float(ev_goal[j]),
        ev_q=sum(c.p_hat * c.odds_adj * s for c, s in zip(lines, alloc.stakes)) / total,
        ev_raw=sum(c.p_hat * c.odds * s for c, s in zip(lines, alloc.stakes)) / total,
        bankroll_if_hit=alloc.bankroll_if_hit(x),
        diagnostics=diag,
    )


def _check_hit_prob(p_hit: float, lines: tuple[Candidate, ...], table: OutcomeTable) -> None:
    """実行時の検査: max p̂_i ≤ P_hit ≤ min(1, Σp̂_i) (複勝の p̂ は当てはめ誤差の分だけ許す)、単勝だけなら Σp̂。"""
    tol = table.max_error + 1e-9
    p = [c.p_hat for c in lines]
    if not (max(p) - tol <= p_hit <= min(1.0, sum(p)) + tol):
        raise RuntimeError(f"P_hit が範囲外です: {p_hit} (p̂={p})")
    if all(c.bet_type == "win" for c in lines) and abs(p_hit - sum(p)) > 1e-9:
        raise RuntimeError(f"単勝だけのバスケットの P_hit が Σp̂ と違います: {p_hit} (p̂={p})")


def recommend(
    race: RaceInput, x: float, goal: float, k_eff: int, q_win: QTable, q_place: QTable, params: BettingParams,
    *, n_paid_places: int,
) -> Decision:
    return decide(prepare(race, q_win, q_place, params, n_paid_places=n_paid_places), x, goal, k_eff, params)


def _malformed(decision: Decision) -> bool:
    """券と根拠の行が対応していない・同じ馬が 2 枚ある・õ が正の有限値でない。"""
    keys = [(t.bet_type, t.horse_num) for t in decision.tickets]
    return (
        len(decision.tickets) != len(decision.lines)
        or keys != [(c.bet_type, c.horse_num) for c in decision.lines]
        or len({h for _, h in keys}) != len(keys)
        or not all(math.isfinite(c.odds_adj) and c.odds_adj > 0 for c in decision.lines)
    )


def safety_check(
    decision: Decision,
    *,
    snapshot_official_at: datetime,
    now: datetime,
    post_time: datetime,
    priced_horses: Collection[int],
    bankroll: float,
    params: BettingParams,
) -> tuple[str, ...]:
    """表示前の安全ゲート (design §9.3-8)。通らなかった理由を返す (空なら表示してよい)。

    形の壊れた Decision でも例外にせず理由 (malformed_decision / stake_not_unit) を返す。
    """
    if not decision.recommended:
        return ("no_recommendation",)
    if _malformed(decision):
        return ("malformed_decision",)
    stakes = [t.stake for t in decision.tickets]
    total = sum(stakes)
    if total <= 0:
        return ("stake_not_unit",)
    reasons: list[str] = []
    age = (now - snapshot_official_at).total_seconds()
    if age > params.snapshot_max_age_sec:
        reasons.append("stale_snapshot")
    elif age < -params.snapshot_max_age_sec:
        reasons.append("clock_skew")
    if now > post_time - timedelta(minutes=params.deadline_min):
        reasons.append("too_late")
    missing = sorted({t.horse_num for t in decision.tickets} - set(priced_horses))
    if missing:
        reasons.append("not_priced:" + ",".join(map(str, missing)))
    if total > bankroll:
        reasons.append("over_budget")
    if bankroll != decision.x:
        reasons.append("bankroll_changed")
    if any(s <= 0 or s % params.unit for s in stakes):
        reasons.append("stake_not_unit")
    min_payout = min(s * c.odds_adj for s, c in zip(stakes, decision.lines))
    if bankroll - total + min_payout < decision.goal:
        reasons.append("hit_misses_goal")
    o_eff = min_payout / total
    recomputed = (o_eff, decision.x - total + min_payout, (decision.p_hit or 0.0) * o_eff)
    stored = (decision.o_eff, decision.bankroll_if_hit, decision.ev_goal)
    if any(s is None or not math.isclose(a, s, rel_tol=1e-9, abs_tol=1e-9) for a, s in zip(recomputed, stored)):
        reasons.append("recompute_mismatch")
    elif recomputed[2] < params.ev_gate:
        reasons.append("below_gate")
    return tuple(reasons)
