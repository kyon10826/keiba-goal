"""券どうしの重なりを補正した的中確率のための 2 段モデル (design §9.3-4)。

1 着は校正済みの p_win どおりに決まり、2〜m 着 (m = 複勝の支払着数) は 1 着馬を除いた残りから別の強さ v の
Plackett–Luce で決まる。v は複勝確率 (上位 m 着に入る確率) に合うよう当てはめる。
1 本の PL で単勝・複勝の両方に合わせることはできない (勝率で強さが決まってしまう) ため 2 段にしている。
上位 m 着の全順序を列挙した表 (18 頭・m=3 で 4,896 通り) をレースごとに作り、モンテカルロは使わない。

m は発売開始時の頭数で決まる (design §3.4) ので呼び出し側が渡す。判断時の頭数から決めてはいけない
(発売開始後の取消で頭数が減っても支払着数は減らない)。m = 0 (複勝の発売なし) は 1 着だけの表で単勝だけを扱う。
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from keiba_goal.betting.tickets import BET_TYPES

PAID_PLACES: tuple[int, ...] = (0, 2, 3)
_TINY = 1e-300


def check_paid_places(n_paid_places: int) -> int:
    """複勝の支払着数は 0 (発売なし)・2・3 のどれか。"""
    if isinstance(n_paid_places, bool) or n_paid_places not in PAID_PLACES:
        raise ValueError(f"複勝の支払着数は {PAID_PLACES} のどれかです: {n_paid_places!r}")
    return int(n_paid_places)


def project_place(p_win: np.ndarray, p_place: np.ndarray, m: int, eps: float = 1e-6) -> np.ndarray:
    """複勝確率を p_win + eps ≤ p ≤ 1、合計 m に射影する (一様なずらし t を二分法で決めて clip)。"""
    lo_b, hi_b = p_win + eps, np.ones_like(p_win)
    # t = lo で全馬が下限 (合計 1 + n·eps < m)、t = hi で全馬が 1 (合計 n ≥ m) になる区間から始める
    lo, hi = -float(np.max(p_place)), 1.0 - float(np.min(p_place))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.clip(p_place + mid, lo_b, hi_b).sum() < m:
            lo = mid
        else:
            hi = mid
    return np.clip(p_place + 0.5 * (lo + hi), lo_b, hi_b)


def place_marginals(p_win: np.ndarray, v: np.ndarray, m: int) -> np.ndarray:
    """2 段モデルで各馬が上位 m 着に入る確率 (n×n の行列計算。順序表と同じ値)。"""
    if m not in (2, 3):
        raise ValueError(f"2 段モデルの支払着数 m は 2 か 3 です: {m}")
    big_v = v.sum()
    m2 = p_win[:, None] * v[None, :] / np.maximum(big_v - v, _TINY)[:, None]
    np.fill_diagonal(m2, 0.0)
    out = p_win + m2.sum(axis=0)
    if m == 3:
        rest = np.maximum(big_v - v[:, None] - v[None, :], _TINY)
        r = m2 / rest
        np.fill_diagonal(r, 0.0)
        out = out + v * (r.sum() - r.sum(axis=1) - r.sum(axis=0))
    return out


@dataclass(frozen=True)
class StrengthFit:
    v: np.ndarray
    converged: bool
    iterations: int
    max_error: float


def fit_strengths(
    p_win: np.ndarray,
    p_place: np.ndarray,
    m: int,
    tol: float = 1e-4,
    max_iter: int = 2000,
    damping: float = 0.7,
) -> StrengthFit:
    """v ← p_win から始め、v ← v·((p_place − p_win)/(現在の値 − p_win))^damping を正規化しながら反復する。

    p_place は project_place 済みであること。max 誤差 < tol で収束、max_iter 回で打ち切り (converged=False)。
    """
    target = p_place - p_win
    if np.any(target <= 0):
        raise ValueError("p_place は p_win より大きい必要があります (project_place を先に通す)")
    v = np.maximum(p_win, 1e-12)
    v = v / v.sum()
    err = np.inf
    for it in range(max_iter + 1):
        cur = place_marginals(p_win, v, m)
        err = float(np.max(np.abs(cur - p_place)))
        if err < tol:
            return StrengthFit(v, True, it, err)
        if it == max_iter:
            break
        v = v * (target / np.maximum(cur - p_win, _TINY)) ** damping
        v = v / v.sum()
    return StrengthFit(v, False, max_iter, err)


@lru_cache(maxsize=32)
def ordered_tuples(n: int, m: int) -> np.ndarray:
    """0..n-1 から m 頭を選ぶ全順序 (T, m)。読み取り専用。"""
    out = np.array(list(itertools.permutations(range(n), m)), dtype=np.int64).reshape(-1, m)
    out.setflags(write=False)
    return out


def tuple_probs(p_win: np.ndarray, v: np.ndarray, order: np.ndarray) -> np.ndarray:
    """順序 (a, b[, c]) の確率 p_win[a]·v[b]/(V − v[a])·v[c]/(V − v[a] − v[b])。"""
    big_v = v.sum()
    a = order[:, 0]
    denom = np.maximum(big_v - v[a], _TINY)
    prob = p_win[a] * v[order[:, 1]] / denom
    if order.shape[1] == 3:
        b = order[:, 1]
        prob = prob * v[order[:, 2]] / np.maximum(denom - v[b], _TINY)
    return prob


@dataclass(frozen=True)
class OutcomeTable:
    """1 レースの上位 m 着 (複勝の発売が無ければ 1 着だけ) の全順序とその確率 (レースごとにキャッシュする)。"""

    horse_nums: np.ndarray
    p_win: np.ndarray
    p_place: np.ndarray | None  # 射影後の複勝確率 (当てはめの目標値)。複勝の発売なしは None
    v: np.ndarray | None
    n_paid_places: int
    order: np.ndarray
    prob: np.ndarray
    converged: bool
    iterations: int
    max_error: float

    def index_of(self, horse_num: int) -> int:
        hit = np.flatnonzero(self.horse_nums == horse_num)
        if hit.size != 1:
            raise KeyError(f"馬番 {horse_num} は表にありません")
        return int(hit[0])

    def ticket_mask(self, bet_type: str, horse_num: int) -> np.ndarray:
        """その券が当たる順序の真偽値 (T,)。"""
        if bet_type not in BET_TYPES:
            raise ValueError(f"券種は win / place です: {bet_type!r}")
        i = self.index_of(horse_num)
        if bet_type == "win":
            return self.order[:, 0] == i
        if self.n_paid_places == 0:
            raise ValueError("複勝の発売が無いレースです")
        return np.any(self.order == i, axis=1)

    def ticket_masks(self, tickets: Iterable[tuple[str, int]]) -> np.ndarray:
        rows = [self.ticket_mask(b, h) for b, h in tickets]
        return np.vstack(rows) if rows else np.zeros((0, self.prob.size), dtype=bool)

    def hit_prob(self, tickets: Iterable[tuple[str, int]]) -> float:
        """P_hit(B): バスケットの券のどれか 1 枚以上が当たる確率。"""
        masks = self.ticket_masks(tickets)
        if masks.shape[0] == 0:
            return 0.0
        return float(self.prob[np.any(masks, axis=0)].sum())

    def place_marginals(self) -> np.ndarray:
        if self.n_paid_places == 0:
            raise ValueError("複勝の発売が無いレースです")
        return table_place_marginals(self.order, self.prob, self.horse_nums.size)


def table_place_marginals(order: np.ndarray, prob: np.ndarray, n: int) -> np.ndarray:
    """順序表から各馬が上位 m 着に入る確率を集計する。"""
    return np.bincount(order.ravel(), np.repeat(prob, order.shape[1]), n)


def _win_inputs(
    horse_nums: Sequence[int] | np.ndarray, p_win: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nums = np.asarray(horse_nums, dtype=np.int64)
    w = np.asarray(p_win, dtype=np.float64)
    n = nums.size
    if n == 0 or w.shape != (n,):
        raise ValueError(f"頭数と配列の長さが合いません (n={n}, p_win {w.shape})")
    if np.unique(nums).size != n:
        raise ValueError("馬番が重複しています")
    if not np.all(np.isfinite(w)):
        raise ValueError("確率に NaN / inf があります")
    if np.any(w < 0) or abs(w.sum() - 1.0) > 1e-6 or np.any(w > 1.0 - 1e-6):
        raise ValueError(f"p_win は 0 以上・合計 1 (レース内正規化済み) である必要があります (合計 {w.sum():.6f})")
    return nums, w / w.sum()


def build_outcome_table(
    horse_nums: Sequence[int] | np.ndarray,
    p_win: Sequence[float] | np.ndarray,
    p_place: Sequence[float] | np.ndarray | None,
    n_paid_places: int,
    tol: float = 1e-4,
    max_iter: int = 2000,
    damping: float = 0.7,
) -> OutcomeTable:
    """p_win (レース内で合計 1) と複勝確率から順序表を作る。収束しなくても表は返し、converged で知らせる。

    n_paid_places = 0 (複勝の発売なし) は 1 着だけの表で、p_place は None で渡す。
    """
    m = check_paid_places(n_paid_places)
    nums, w = _win_inputs(horse_nums, p_win)
    n = nums.size
    if m == 0:
        if p_place is not None:
            raise ValueError("複勝の発売が無いレース (n_paid_places=0) に複勝確率は渡しません")
        order = ordered_tuples(n, 1)
        return OutcomeTable(horse_nums=nums, p_win=w, p_place=None, v=None, n_paid_places=0, order=order,
                            prob=w[order[:, 0]], converged=True, iterations=0, max_error=0.0)
    if p_place is None:
        raise ValueError(f"支払着数 {m} のレースには複勝確率が要ります")
    plc = np.asarray(p_place, dtype=np.float64)
    if plc.shape != (n,) or n < m:
        raise ValueError(f"頭数と配列の長さが合いません (n={n}, m={m}, p_place {plc.shape})")
    if not np.all(np.isfinite(plc)):
        raise ValueError("複勝確率に NaN / inf があります")
    target = project_place(w, plc, m)
    fit = fit_strengths(w, target, m, tol=tol, max_iter=max_iter, damping=damping)
    order = ordered_tuples(n, m)
    prob = tuple_probs(w, fit.v, order)
    err = float(np.max(np.abs(table_place_marginals(order, prob, n) - target)))
    return OutcomeTable(
        horse_nums=nums, p_win=w, p_place=target, v=fit.v, n_paid_places=m, order=order, prob=prob,
        converged=fit.converged, iterations=fit.iterations, max_error=err,
    )


def subset_hit_probs(prob: np.ndarray, masks: np.ndarray, subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """券の組合せごとの P_hit をまとめて計算する。

    各順序に「当たる券の集合」をビット列で付け、同じビット列の確率をまとめてから、
    組合せのビットと 1 つでも重なる行の確率を足す (組合せ × 異なるビット列 の 1 回の行列積)。
    """
    c = masks.shape[0]
    if c > 62:
        raise ValueError("券の数は 62 枚までです")
    if not subsets:
        return np.zeros(0)
    codes = np.zeros(prob.size, dtype=np.int64)
    for j in range(c):
        codes |= masks[j].astype(np.int64) << j
    uniq, inv = np.unique(codes, return_inverse=True)
    weight = np.bincount(inv, weights=prob, minlength=uniq.size)
    bits = np.array([sum(1 << j for j in s) for s in subsets], dtype=np.int64)
    hits = (bits[:, None] & uniq[None, :]) != 0
    return hits.astype(np.float64) @ weight
