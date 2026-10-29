import itertools
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from keiba_goal.betting.allocate import allocate
from keiba_goal.betting.basket import (
    Candidate,
    RaceInput,
    decide,
    drop_factor,
    market_probs,
    prepare,
    recommend,
    safety_check,
)
from keiba_goal.betting.joint import ordered_tuples, table_place_marginals, tuple_probs
from keiba_goal.betting.params import BettingParams
from keiba_goal.betting.theory import target_odds
from keiba_goal.betting.tickets import Ticket

PARAMS = BettingParams()
LITERAL = replace(PARAMS, search_prefilter=False)
Q_WIN = {1.0: 0.90, 3.0: 0.83, 5.0: 0.79, 10.0: 0.79, 20.0: 0.82, 50.0: 0.91}
Q_PLACE = {1.0: 0.90, 3.0: 0.86, 5.0: 0.85, 10.0: 0.84, 20.0: 0.85, 50.0: 0.90}


def paid(n: int) -> int:
    """合成レース (取消なし) の支払着数: 発売開始時の頭数 = n とした JRA の規則。本番は呼び出し側が決めて渡す。"""
    return 3 if n >= 8 else 2 if n >= 5 else 0


def _harville_place(w: np.ndarray, m: int) -> np.ndarray:
    order = ordered_tuples(w.size, m)
    return table_place_marginals(order, tuple_probs(w, w**0.8, order), w.size)


def make_race(win_odds, p_win, place_lo=None, p_place=None, m=None) -> tuple[RaceInput, int]:
    """馬番 1..n のレースと支払着数。複勝の既定値は p_win からの Harville 風の値、複勝 Lo は単勝の約 1/3。"""
    win_odds = np.asarray(win_odds, float)
    p_win = np.asarray(p_win, float)
    priced = np.isfinite(win_odds)
    m = paid(int(priced.sum())) if m is None else m
    if place_lo is None:
        place_lo = np.where(priced, np.maximum(1.0, np.round(win_odds / 3.2, 1)), np.nan)
    if p_place is None:
        p_place = np.zeros_like(p_win)
        if m:
            p_place[priced] = _harville_place(p_win[priced] / p_win[priced].sum(), m)
    pmw, pmp = market_probs(win_odds, place_lo, m)
    return RaceInput(np.arange(1, win_odds.size + 1), p_win, p_place, win_odds, place_lo, pmw, pmp), m


def random_race(seed: int, n: int | None = None, noise: float = 0.25,
                place_noise: float | None = None) -> tuple[RaceInput, int]:
    """市場 (控除 20%) と、それにノイズを足したモデル確率。"""
    rng = np.random.default_rng(seed)
    n = n or int(rng.integers(8, 19))
    w_mkt = np.clip(rng.dirichlet(np.full(n, 0.8)), 0.005, None)
    w_mkt /= w_mkt.sum()
    win_odds = np.round(0.8 / w_mkt, 1)
    plc_mkt = _harville_place(w_mkt, paid(n))
    place_lo = np.round(np.maximum(1.0, 0.7 / plc_mkt), 1)
    p_win = w_mkt * np.exp(rng.normal(0, noise, n))
    p_place = np.clip(plc_mkt * np.exp(rng.normal(0, noise * 0.6 if place_noise is None else place_noise, n)), 0, 1)
    return make_race(win_odds, p_win / p_win.sum(), place_lo, p_place)


def prep_of(case: tuple[RaceInput, int], params: BettingParams = PARAMS, q_win=Q_WIN, q_place=Q_PLACE):
    race, m = case
    return prepare(race, q_win, q_place, params, n_paid_places=m)


def brute_force(prep, x, goal, k, params=PARAMS):
    """仕様をそのまま書いた参照実装 (前段の枝刈りなし、P_hit は直接計算)。"""
    o_star = target_odds(x, goal, k)
    eligible = [c for c in prep.candidates if not params.search_prefilter or c.odds_adj >= o_star]
    pool = sorted(eligible, key=lambda c: (-c.p_hat * c.odds_adj, c.horse_num, c.bet_type != "win"))[: params.top_n]
    masks = prep.table.ticket_masks((c.bet_type, c.horse_num) for c in pool)
    valid = []
    for size in range(1, params.max_basket + 1):
        for combo in itertools.combinations(range(len(pool)), size):
            if len({pool[r].horse_num for r in combo}) < size:
                continue
            alloc = allocate([pool[r].odds_adj for r in combo], x, goal)
            if alloc is None or not (o_star * (1 - 1e-12) <= alloc.o_eff <= 1.10 * o_star * (1 + 1e-12)):
                continue
            p = float(prep.table.prob[np.any(masks[list(combo)], axis=0)].sum())
            if p * alloc.o_eff >= params.ev_gate:
                valid.append((p, alloc, combo))
    if not valid:
        return None
    best_p = max(v[0] for v in valid)
    p, alloc, combo = min((v for v in valid if v[0] >= best_p - 1e-12), key=lambda v: (v[1].total, len(v[2]), v[2]))
    return p, tuple(Ticket(pool[r].bet_type, pool[r].horse_num, s) for r, s in zip(combo, alloc.stakes))


STATES = [(10_000, 20_000, 3), (3_000, 20_000, 1), (16_000, 20_000, 6), (5_000, 30_000, 2)]


def test_search_matches_brute_force():
    seen_recommendation = seen_skip = 0
    for seed in range(12):
        prep = prep_of(random_race(seed))
        for x, goal, k in STATES:
            d = decide(prep, x, goal, k, PARAMS)
            ref = brute_force(prep, x, goal, k)
            assert d.search_prefilter is True
            if ref is None:
                assert not d.recommended
                seen_skip += 1
            else:
                assert d.tickets == ref[1]
                assert d.p_hit == pytest.approx(ref[0], abs=1e-12)
                seen_recommendation += 1
    assert seen_recommendation >= 10 and seen_skip >= 5


def test_literal_search_matches_brute_force_and_never_beats_the_prefilter():
    # 前絞り込みなしで上位に入る õ ≥ o* の券は、前絞り込みありの上位にも必ず入る → 選べるバスケットは増えるだけ
    params = replace(LITERAL, top_n=8)
    differ = 0
    for seed in range(40):
        prep = prep_of(random_race(500 + seed, place_noise=0.6), params)
        for x, goal, k in STATES:
            lit = decide(prep, x, goal, k, params)
            pre = decide(prep, x, goal, k, replace(params, search_prefilter=True))
            ref = brute_force(prep, x, goal, k, params)
            assert lit.search_prefilter is False and pre.search_prefilter is True
            assert lit.tickets == (() if ref is None else ref[1])
            if lit.recommended:
                assert pre.recommended and pre.p_hit >= lit.p_hit - 1e-12
            differ += lit.tickets != pre.tickets
    assert differ >= 3


def test_prefilter_keeps_in_band_tickets_that_low_odds_tickets_crowd_out():
    # 複勝 10 枚 (õ = 5 × 0.85 = 4.25 < o* = 4.85) が p̂·õ で上位を占め、文字どおりの上位 10 枚では帯に入るバスケットが無い
    p_win = [0.25, 0.2, 0.18, 0.1, 0.08, 0.06, 0.05, 0.04, 0.02, 0.02]
    win_odds = [3.2, 4.0, 6.5, 8.0, 10.0, 13.3, 16.0, 20.0, 40.0, 40.0]
    case = make_race(win_odds, p_win, place_lo=[5.0] * 10, p_place=[0.3] * 10, m=3)
    top10 = replace(PARAMS, top_n=10)
    pre = decide(prep_of(case, top10), 10_000, 20_000, 3, top10)
    lit = decide(prep_of(case, top10), 10_000, 20_000, 3, replace(top10, search_prefilter=False))
    single = allocate([6.5 * Q_WIN[5.0]], 10_000, 20_000)  # 3 番の単勝 1 枚 (õ = 5.135 は帯 [4.85, 5.33] の中)
    assert pre.tickets == (Ticket("win", 3, single.stakes[0]),) and pre.search_prefilter is True
    assert not lit.recommended and lit.skip_reason == "infeasible" and lit.search_prefilter is False
    assert {c.bet_type for c in prep_of(case, top10).candidates if c.score > pre.lines[0].score} == {"place"}
    assert lit.summary()["search_prefilter"] is False


def test_search_matches_brute_force_on_mixed_win_place_baskets():
    # 複勝の誤差を大きくすると複勝を含む (重なりのある) バスケットが推奨される
    with_place = mixed = 0
    for seed in range(10):
        prep = prep_of(random_race(300 + seed, place_noise=0.6))
        for x, goal, k in [(19_000, 20_000, 3), (15_000, 20_000, 2)]:
            d = decide(prep, x, goal, k, PARAMS)
            ref = brute_force(prep, x, goal, k)
            assert d.tickets == (() if ref is None else ref[1])
            with_place += any(t.bet_type == "place" for t in d.tickets)
            mixed += len({t.bet_type for t in d.tickets}) == 2
    assert with_place >= 5 and mixed >= 3


def test_recommendations_satisfy_all_invariants():
    count = 0
    for seed in range(40):
        prep = prep_of(random_race(100 + seed))
        for x, goal, k in STATES:
            d = decide(prep, x, goal, k, PARAMS)
            if not d.recommended:
                assert d.skip_reason in {"gate", "infeasible", "no_candidates"}
                continue
            count += 1
            total = d.total_stake
            payouts = [t.stake * c.odds_adj for t, c in zip(d.tickets, d.lines)]
            assert d.o_star == pytest.approx(target_odds(x, goal, k))
            assert d.o_star * (1 - 1e-12) <= d.o_eff <= 1.10 * d.o_star * (1 + 1e-12)
            assert d.o_eff == pytest.approx(min(payouts) / total)
            assert total <= x and all(t.stake % 100 == 0 and t.stake > 0 for t in d.tickets)
            assert min(payouts) >= goal - x + total
            assert d.bankroll_if_hit >= goal
            assert d.ev_goal == pytest.approx(d.p_hit * d.o_eff) and d.ev_goal >= PARAMS.ev_gate
            assert len({t.horse_num for t in d.tickets}) == len(d.tickets)  # 同じ馬の単複は同時に買わない
            assert len(d.tickets) <= PARAMS.max_basket
            p_hat = [c.p_hat for c in d.lines]
            assert max(p_hat) - 1e-4 <= d.p_hit <= min(1.0, sum(p_hat)) + 1e-4
            assert d.ev_q == pytest.approx(sum(c.p_hat * c.odds_adj * t.stake for c, t in zip(d.lines, d.tickets)) / total)
            assert d.ev_raw == pytest.approx(sum(c.p_hat * c.odds * t.stake for c, t in zip(d.lines, d.tickets)) / total)
            if all(t.bet_type == "win" for t in d.tickets):
                assert d.p_hit == pytest.approx(sum(p_hat), abs=1e-12)
    assert count >= 30


def test_market_like_model_does_not_pass_the_gate():
    # p̂ = 判断時の市場確率なら EV ≈ q / オーバーラウンド ≈ 0.65 で τ = 0.8 を超えない
    for seed in range(10):
        race, m = random_race(200 + seed, noise=0.0)
        race = replace(race, p_win=race.p_mkt_win, p_place=race.p_mkt_place)
        d = recommend(race, 10_000, 20_000, 3, Q_WIN, Q_PLACE, PARAMS, n_paid_places=m)
        assert d.skip_reason in {"gate", "infeasible"}
        if d.skip_reason == "gate":
            assert d.diagnostics["best_ev_goal"] < 0.8


def test_exclusions_with_reasons():
    nan = float("nan")
    win_odds = [3.5, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 120.0, nan, 15.0]
    place_lo = [1.3, 1.6, 2.2, 3.0, 4.5, 7.0, 11.0, 25.0, nan, nan]
    priced = np.isfinite(win_odds)
    mkt = np.where(priced, 1 / np.nan_to_num(np.array(win_odds), nan=1.0), 0.0)
    p_win = mkt / mkt.sum()
    p_win[6] *= 2.5  # 7 番: 市場の 2.5 倍 → 乖離ガード
    p_win /= p_win.sum()
    prep = prep_of(make_race(win_odds, p_win, place_lo))
    reasons = {(e.bet_type, e.horse_num): e.reason for e in prep.excluded}
    assert reasons[("win", 8)] == reasons[("place", 8)] == "longshot"  # 単勝 100 倍以上は単複とも
    assert reasons[("win", 9)] == reasons[("place", 9)] == "no_odds"
    assert reasons[("place", 10)] == "no_place_odds" and ("win", 10) not in reasons
    assert reasons[("win", 7)] == "disagreement"
    assert prep.field_size == 9 and prep.n_paid_places == prep.table.n_paid_places == 3
    kept = {(c.bet_type, c.horse_num) for c in prep.candidates}
    assert not kept & set(reasons)
    assert ("win", 1) in kept and ("place", 1) in kept
    for c in prep.candidates:
        assert c.odds_adj == pytest.approx(c.odds * c.q)
        assert c.p_hat / c.p_mkt <= PARAMS.max_market_ratio


def test_paid_places_come_from_the_caller_not_the_field_size():
    # 判断時 7 頭 (発売開始時 8 頭、開始後に 1 頭取消) は 3 着払い。逆に判断時 9 頭でも渡された m に従う
    for n, m in ((7, 3), (7, 2), (9, 2), (9, 3)):
        race, _ = random_race(40 + n, n=n)
        case = make_race(race.win_odds, race.p_win, race.place_lo, race.p_place, m=m)
        prep = prep_of(case)
        assert prep.n_paid_places == prep.table.n_paid_places == m and prep.table.order.shape[1] == m
        assert sum(prep.table.p_place) == pytest.approx(m, abs=1e-9)
        d = decide(prep, 16_000, 20_000, 3, PARAMS)
        assert d.diagnostics["n_paid_places"] == m and d.diagnostics["field_size"] == n


def test_no_place_sales_offers_only_win_tickets():
    race, _ = random_race(12, n=8)
    case = make_race(race.win_odds, race.p_win, race.place_lo, race.p_place, m=0)
    assert np.isnan(case[0].p_mkt_place).all()
    prep = prep_of(case)
    assert prep.table.n_paid_places == 0 and prep.table.order.shape == (8, 1)
    assert {c.bet_type for c in prep.candidates} == {"win"}
    assert {e.horse_num for e in prep.excluded if e.reason == "no_place_sales"} == {
        c.horse_num for c in prep.candidates}
    recommended = 0
    for x, goal, k in STATES + [(19_000, 20_000, 3)]:
        d = decide(prep, x, goal, k, replace(PARAMS, ev_gate=0.0))
        assert all(t.bet_type == "win" for t in d.tickets)
        if d.recommended:
            recommended += 1
            assert d.p_hit == pytest.approx(sum(c.p_hat for c in d.lines), abs=1e-12)
    assert recommended >= 1


def test_market_probabilities_must_use_the_same_paid_places():
    race, m = random_race(8, n=10)
    assert m == 3
    _, p2 = market_probs(race.win_odds, race.place_lo, 2)
    with pytest.raises(ValueError, match="p_mkt_place"):
        prepare(replace(race, p_mkt_place=p2), Q_WIN, Q_PLACE, PARAMS, n_paid_places=3)
    with pytest.raises(ValueError, match="p_mkt_place"):
        prepare(race, Q_WIN, Q_PLACE, PARAMS, n_paid_places=0)  # 3 着払いで作った市場確率のまま 0 を渡した
    for bad in (1, 4, None):
        with pytest.raises(ValueError, match="支払着数"):
            prepare(race, Q_WIN, Q_PLACE, PARAMS, n_paid_places=bad)
    with pytest.raises(TypeError):
        prepare(race, Q_WIN, Q_PLACE, PARAMS)  # 支払着数は省略できない


def test_missing_q_band_excludes_ticket():
    race, m = random_race(7)
    prep = prepare(race, {5.0: 0.79}, Q_PLACE, PARAMS, n_paid_places=m)  # 5 倍未満の帯が無い
    no_q = {e.horse_num for e in prep.excluded if e.reason == "no_q" and e.bet_type == "win"}
    assert no_q == {int(h) for h, o in zip(race.horse_num, race.win_odds) if o < 5.0}


def test_drop_factor_lookup():
    assert drop_factor(Q_WIN, 2.0, 100.0) == 0.90
    assert drop_factor(Q_WIN, 3.0, 100.0) == 0.83
    assert drop_factor(Q_WIN, 99.9, 100.0) == 0.91
    for odds in (100.0, 0.9, float("nan")):
        assert drop_factor(Q_WIN, odds, 100.0) is None
    assert drop_factor({1.0: 0.9, 20.0: 0.8}, 60.0, 100.0) == 0.8  # data 層で結合された帯
    assert drop_factor({}, 5.0, 100.0) is None


def test_market_probs():
    w, p = market_probs(np.array([2.0, 4.0, np.nan, 8.0]), np.array([1.2, 1.5, np.nan, np.nan]), 2)
    assert np.nansum(w) == pytest.approx(1.0) and np.isnan(w[2])
    assert np.nansum(p) == pytest.approx(2.0) and np.isnan(p[3])
    assert w[0] / w[1] == pytest.approx(2.0)
    _, p3 = market_probs(np.array([2.0, 4.0, 8.0, 9.0]), np.array([1.2, 1.5, 2.0, 2.5]), 3)
    assert p3.sum() == pytest.approx(3.0)
    w0, p0 = market_probs(np.array([2.0, 4.0]), np.array([1.2, 1.5]), 0)  # 複勝の発売なし
    assert np.isnan(p0).all() and w0.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        market_probs(np.array([2.0]), np.array([1.2]), 1)


def test_skip_reasons():
    case = random_race(3)
    prep = prep_of(case)
    assert decide(prep, 20_000, 20_000, 3, PARAMS).skip_reason == "goal"
    assert decide(prep, 10_000, 20_000, 0, PARAMS).skip_reason == "no_bets"
    assert decide(prep, 99, 20_000, 3, PARAMS).skip_reason == "no_funds"
    assert decide(prep, 100, 20_000, 1, PARAMS).skip_reason == "no_candidates"  # o*_1 = 200 倍
    small = make_race([2.5, 4.0, 6.0, 9.0, np.nan], [0.4, 0.3, 0.2, 0.1, 0.0])
    d = recommend(small[0], 10_000, 20_000, 3, Q_WIN, Q_PLACE, PARAMS, n_paid_places=small[1])
    assert d.skip_reason == "small_field" and not d.recommended
    # 判断スナップショットが無い (どの馬にも単勝オッズが無い) のは頭数不足ではなくオッズ欠損
    race = case[0]
    blank = replace(race, win_odds=np.full(race.horse_num.size, np.nan), place_lo=np.full(race.horse_num.size, np.nan),
                    p_mkt_win=np.full(race.horse_num.size, np.nan), p_mkt_place=np.full(race.horse_num.size, np.nan))
    missing = prepare(blank, Q_WIN, Q_PLACE, PARAMS, n_paid_places=case[1])
    assert missing.skip_reason == "no_odds" and missing.field_size == 0 and missing.table is None
    assert {e.reason for e in missing.excluded} == {"no_odds"}
    assert decide(missing, 10_000, 20_000, 3, PARAMS).skip_reason == "no_odds"


def test_infeasible_when_no_basket_lands_in_the_band():
    # õ がすべて 12 前後: 1 枚 12、2 枚 6、3 枚 4 は帯 [3.41, 3.76] の上、4 枚 3 は下 → 実現できるバスケットなし
    win_odds = [15.2] * 10
    race, m = make_race(win_odds, [0.1] * 10, place_lo=[3.0] * 10)
    d = recommend(race, 1_000, 2_000, 2, Q_WIN, Q_PLACE, PARAMS, n_paid_places=m)
    assert d.skip_reason == "infeasible"
    assert d.diagnostics["n_feasible"] == 0 and d.diagnostics["n_pool"] > 0


def test_ties_prefer_the_smaller_total_stake():
    # 1 番と 2 番は p̂ が同じ。1 枚ずつなら P_hit は同点で、õ が高い 2 番の方が金額が小さい
    win_odds = [6.2, 6.5, 4.0, 4.0, 4.0, 4.0]
    p_win = [0.22, 0.22, 0.14, 0.14, 0.14, 0.14]
    race, m = make_race(win_odds, p_win, place_lo=[1.5] * 6)
    d = recommend(race, 10_000, 20_000, 3, Q_WIN, Q_PLACE, replace(PARAMS, max_basket=1), n_paid_places=m)
    assert d.tickets == (Ticket("win", 2, 2_500),)
    assert d.diagnostics["n_passed"] == 2


def test_place_tickets_are_dropped_when_the_joint_fit_does_not_converge():
    params = replace(PARAMS, pl_max_iter=0)
    prep = prep_of(random_race(11, n=16), params)
    assert not prep.table.converged
    assert all(c.bet_type == "win" for c in prep.candidates)
    assert {e.horse_num for e in prep.excluded if e.reason == "joint_not_converged"} == set(range(1, 17)) - {
        e.horse_num for e in prep.excluded if e.reason == "longshot"}
    d = decide(prep, 10_000, 20_000, 3, params)
    assert all(t.bet_type == "win" for t in d.tickets)
    assert d.diagnostics["joint_converged"] is False


X_10R, A_10R = 1_694_430, 2_440_000
Q_WIN_10R = {1.0: 0.90, 3.0: 0.83, 5.0: 0.787, 10.0: 0.79, 20.0: 0.787, 50.0: 0.91}


def pdf_10r_race() -> tuple[RaceInput, int]:
    """PDF 11 枚目 10R の再構成: 11 番 8.4 倍 (q 0.787)、2 枚目に 23.8 倍の単勝、他は 100 倍以上。"""
    win_odds = np.full(12, 150.0)
    win_odds[10], win_odds[3] = 8.4, 23.8
    p_win = np.full(12, (1 - 0.179) / 10)
    p_win[10], p_win[3] = 0.135, 0.044
    return make_race(win_odds, p_win, place_lo=np.full(12, 1.5))


def pdf_10r_decision(params: BettingParams = PARAMS):
    case = pdf_10r_race()
    return case[0], decide(prep_of(case, params, q_win=Q_WIN_10R), X_10R, A_10R, 5, params)


def test_pdf_10r_reconstruction_end_to_end():
    _, d = pdf_10r_decision()
    assert d.tickets == (Ticket("win", 11, 141_900), Ticket("win", 4, 50_100))
    assert d.total_stake == 192_000
    assert d.o_star == pytest.approx(4.7370, abs=5e-5)
    assert d.o_eff / d.o_star == pytest.approx(1.031, abs=1e-3)
    assert d.p_hit == pytest.approx(0.179)
    assert d.ev_goal == pytest.approx(0.875, abs=1e-3)  # PDF の EV 0.875 = P_hit 0.179 × o_eff
    assert d.bankroll_if_hit >= A_10R
    assert not d.model_claims_edge
    loaded = json.loads(json.dumps(d.summary()))
    assert loaded["tickets"] == [t.to_dict() for t in d.tickets] and loaded["total_stake"] == 192_000
    assert loaded["search_prefilter"] is True
    # 候補は 4 枚 (≤ top_n) なので文字どおりの探索でも同じ買い目
    _, lit = pdf_10r_decision(LITERAL)
    assert lit.tickets == d.tickets and lit.search_prefilter is False


def test_safety_gate():
    race, d = pdf_10r_decision()
    post = datetime(2026, 9, 21, 14, 50)
    now = post - timedelta(minutes=5, seconds=30)
    ok = dict(snapshot_official_at=now - timedelta(seconds=30), now=now, post_time=post,
              priced_horses=set(range(1, 13)), bankroll=X_10R, params=PARAMS)
    assert safety_check(d, **ok) == ()
    assert safety_check(d, **{**ok, "snapshot_official_at": now - timedelta(seconds=91)}) == ("stale_snapshot",)
    assert safety_check(d, **{**ok, "snapshot_official_at": now + timedelta(seconds=120)}) == ("clock_skew",)
    late = post - timedelta(minutes=2, seconds=59)
    assert safety_check(d, **{**ok, "now": late, "snapshot_official_at": late}) == ("too_late",)
    assert safety_check(d, **{**ok, "priced_horses": set(range(1, 13)) - {4}}) == ("not_priced:4",)
    poor = safety_check(d, **{**ok, "bankroll": 150_000})
    assert {"over_budget", "bankroll_changed", "hit_misses_goal"} <= set(poor)
    assert safety_check(replace(d, o_eff=d.o_eff + 0.1), **ok) == ("recompute_mismatch",)
    odd = replace(d, tickets=(Ticket("win", 11, 141_950), d.tickets[1]))
    assert "stake_not_unit" in safety_check(odd, **ok)
    weak = replace(d, p_hit=0.1, ev_goal=0.1 * d.o_eff)
    assert safety_check(weak, **ok) == ("below_gate",)
    skipped = decide(prep_of((race, 3)), 20_000, 20_000, 1, PARAMS)
    assert safety_check(skipped, **ok) == ("no_recommendation",)


def test_safety_gate_reports_malformed_decisions_instead_of_raising():
    race, d = pdf_10r_decision()
    post = datetime(2026, 9, 21, 14, 50)
    now = post - timedelta(minutes=5, seconds=30)
    ok = dict(snapshot_official_at=now, now=now, post_time=post, priced_horses=set(range(1, 13)),
              bankroll=X_10R, params=PARAMS)
    zero = replace(d, tickets=tuple(Ticket(t.bet_type, t.horse_num, 0) for t in d.tickets))
    assert safety_check(zero, **ok) == ("stake_not_unit",)  # 0 で割らない
    place_11 = Candidate("place", 11, 1.5, 0.9, 1.35, 0.4, 0.3)
    malformed = [
        replace(d, lines=d.lines[:1]),                                         # 券と根拠の行の数が違う
        replace(d, tickets=d.tickets + (Ticket("win", 7, 100),)),
        replace(d, tickets=(Ticket("win", 5, 141_900), d.tickets[1])),         # 馬番が違う
        replace(d, tickets=(Ticket("place", 11, 141_900), d.tickets[1])),      # 券種が違う
        replace(d, tickets=d.tickets[:1] + (Ticket("place", 11, 50_100),), lines=d.lines[:1] + (place_11,)),  # 同じ馬
        replace(d, lines=(replace(d.lines[0], odds_adj=float("nan")), d.lines[1])),
    ]
    for bad in malformed:
        assert safety_check(bad, **ok) == ("malformed_decision",)


def test_one_decision_is_fast_for_18_runners():
    case = random_race(42, n=18)
    prep_of(case)  # 順序表の列挙をキャッシュ
    worst_decide = 0.0
    for x, goal, k in STATES:
        prep_times, decide_times = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            prep = prep_of(case)
            t1 = time.perf_counter()
            decide(prep, x, goal, k, PARAMS)
            prep_times.append(t1 - t0)
            decide_times.append(time.perf_counter() - t1)
        worst_decide = max(worst_decide, min(decide_times))
        assert min(prep_times) + min(decide_times) < 0.05
    assert worst_decide < 0.05


def test_candidate_score_and_serialization():
    c = Candidate("win", 3, 8.0, 0.79, 6.32, 0.15, 0.10)
    assert c.score == pytest.approx(0.15 * 6.32)
    assert c.to_dict()["odds_adj"] == 6.32
