import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from keiba_goal.betting.basket import Candidate, Decision
from keiba_goal.betting.params import BettingParams
from keiba_goal.betting.session import (
    BalanceSynced,
    Confirmed,
    Corrected,
    Missed,
    RaceListed,
    Recommended,
    SessionError,
    SessionStarted,
    Settled,
    append_event,
    chain_len,
    decision_context,
    end_reason,
    event_from_dict,
    event_to_dict,
    fold,
    load_state,
    read_events,
    recommended_event,
    settled_event,
    state_at,
    unconfirmed,
)
from keiba_goal.betting.settle import settle
from keiba_goal.betting.tickets import Ticket

P = BettingParams()
DAY = datetime(2026, 11, 1)
WIN1 = (Ticket("win", 1, 1_000),)


def at(h: int, m: int, s: int = 0) -> datetime:
    return DAY.replace(hour=h, minute=m, second=s)


def start(x0=10_000, goal=20_000, k0=3, venues=()):
    return SessionStarted(at=at(8, 30), x0=x0, goal=goal, k0=k0, venues=venues)


def listing(*races):
    return [RaceListed(at=at(8, 30), race_id=rid, post_time=post, venue=venue) for rid, post, venue in races]


def bought(rid, now, tickets=WIN1):
    return [Recommended(at=now, race_id=rid, tickets=tickets),
            Confirmed(at=now + timedelta(seconds=30), race_id=rid, action="bought")]


def test_two_venues_ten_minutes_apart_use_only_settled_results():
    # 東京 10:00, 10:20, … と 京都 10:10, 10:30, …。確定 (発走 + 8 分) は次のレースの判断 (発走 − 6 分) より後
    races = []
    for i in range(5):
        races += [(f"T{i}", at(10, 0) + timedelta(minutes=20 * i), "東京"),
                  (f"K{i}", at(10, 10) + timedelta(minutes=20 * i), "京都")]
    events = [start(k0=6), *listing(*races)]
    queue: list[tuple[datetime, str, int]] = []  # (精算時刻, レース, 払戻)
    ledger: list[tuple[int, int, datetime]] = []  # 参照用: (賭け金, 払戻, 精算時刻)
    outcomes = []
    for rid, post, _ in sorted(races, key=lambda r: r[1]):
        now = post - timedelta(minutes=6)
        for item in sorted(q for q in queue if q[0] <= now):
            events.append(Settled(at=item[0], race_id=item[1], payout=item[2], refund=0))
            queue.remove(item)
        ctx = decision_context(events, rid, now, P)
        done = [row for row in ledger if row[2] <= now]
        x_ref = 10_000 - sum(row[0] for row in ledger) + sum(row[1] for row in done)
        k_ref = 6 - len(done)
        if ctx.ok:
            assert (ctx.x, ctx.k) == (x_ref, k_ref)
            assert ctx.k_eff == min(k_ref, 5 - int(rid[1]))  # 連鎖できるのは残りの東京のレースだけ
            payout = 3_000 if rid == "T2" else 0
            events += bought(rid, now)
            queue.append((post + timedelta(minutes=8), rid, payout))
            ledger.append((1_000, payout, post + timedelta(minutes=8)))
            outcomes.append((rid, "bet"))
        else:
            events.append(Missed(at=now, race_id=rid, reason=ctx.skip_reason))
            outcomes.append((rid, ctx.skip_reason))
    assert outcomes == [(f"{v}{i}", "bet" if v == "T" else "pending") for i in range(5) for v in "TK"]
    final = fold(events)
    assert final.k == 6 - 4  # T4 はまだ精算されていない
    assert final.cash == 10_000 - 5_000 + 3_000


def test_settlement_stamped_after_the_decision_time_is_not_used():
    # バックテストが精算イベントを先に書いてしまっても、判断時刻より後の精算は使わない
    events = [start(), *listing(("T0", at(10, 0), "東京"), ("K0", at(10, 10), "京都"), ("T1", at(10, 20), "東京")),
              *bought("T0", at(9, 54)), Settled(at=at(10, 8), race_id="T0", payout=5_000, refund=0)]
    assert decision_context(events, "K0", at(10, 4), P).skip_reason == "pending"
    assert end_reason(events, at(10, 4), P) is None
    ctx = decision_context(events, "T1", at(10, 14), P)
    assert ctx.ok and (ctx.x, ctx.k) == (14_000, 2)


def test_events_after_now_do_not_leak_into_the_decision():
    # 後の時刻の残高同期・訂正・確認がログに先に書かれていても、判断時点の資金と状態はそれより前のイベントだけで決まる
    races = listing(("T0", at(10, 0), "東京"), ("K0", at(10, 10), "京都"), ("T1", at(10, 20), "東京"))
    events = [start(), *races, BalanceSynced(at=at(15, 0), x=3_000),
              Corrected(at=at(15, 1), race_id="T1", tickets=(Ticket("win", 2, 500),))]
    ctx = decision_context(events, "T0", at(9, 54), P)
    assert ctx.ok and (ctx.x, ctx.k, ctx.k_eff) == (10_000, 3, 2)
    assert state_at(events, at(9, 54)).cash == 10_000 and fold(events).cash == 2_500
    assert state_at(events, at(15, 0)).cash == 3_000

    rec = [start(), *races, Recommended(at=at(9, 54), race_id="T0", tickets=WIN1),
           Confirmed(at=at(10, 30), race_id="T0", action="bought")]
    early = decision_context(rec, "K0", at(10, 4), P)
    assert (early.skip_reason, early.x) == ("unconfirmed", 10_000)  # 10:30 の確認はまだ起きていない
    assert [r.race_id for r in unconfirmed(rec, at(10, 4))] == ["T0"] and unconfirmed(rec, at(10, 31)) == []
    assert decision_context(rec, "T1", at(10, 14), P).skip_reason == "unconfirmed"
    assert decision_context(rec, "T1", at(10, 31), P).skip_reason == "deadline"
    with pytest.raises(SessionError, match="始まっていません"):
        decision_context(rec, "T0", at(8, 0), P)  # セッション開始 (8:30) より前


def test_decisions_before_the_decision_time_are_too_early():
    events = [start(), *listing(("R1", at(10, 0), "東京"))]
    assert decision_context(events, "R1", at(9, 53, 59), P).skip_reason == "too_early"
    assert decision_context(events, "R1", at(8, 30), P).skip_reason == "too_early"
    assert decision_context(events, "R1", at(9, 54), P).ok
    with pytest.raises(SessionError, match="一覧にありません"):
        decision_context(events, "R9", at(9, 54), P)


def test_unconfirmed_recommendation_blocks_until_resolved():
    races = [("T0", at(10, 0), "東京"), ("N0", at(10, 5), "新潟"), ("K0", at(10, 10), "京都"), ("T1", at(10, 20), "東京")]
    events = [start(), *listing(*races), Recommended(at=at(9, 54, 30), race_id="T0", tickets=WIN1)]
    assert decision_context(events, "N0", at(9, 59), P).skip_reason == "awaiting_confirmation"
    assert decision_context(events, "K0", at(10, 4), P).skip_reason == "unconfirmed"
    assert [r.race_id for r in unconfirmed(events, at(10, 4))] == ["T0"]
    assert unconfirmed(events, at(9, 59)) == []
    assert end_reason(events, at(10, 4), P) is None
    assert decision_context(events, "T1", at(10, 14), P).skip_reason == "unconfirmed"

    not_bought = [*events, Confirmed(at=at(10, 6), race_id="T0", action="not_bought")]
    ctx = decision_context(not_bought, "T1", at(10, 14), P)
    assert ctx.ok and (ctx.x, ctx.k) == (10_000, 3)

    # day record で「実は買っていた」と記録すると精算待ちになる
    late_record = [*events, Corrected(at=at(10, 6), race_id="T0", tickets=WIN1)]
    assert decision_context(late_record, "T1", at(10, 14), P).skip_reason == "pending"
    assert fold(late_record).cash == 9_000


def test_refunds_and_remaining_bets():
    races = [("R1", at(10, 0), "東京"), ("R2", at(10, 30), "東京"), ("R3", at(11, 0), "東京")]
    codes = {h: 0 for h in range(1, 13)}
    t1 = (Ticket("win", 3, 1_000), Ticket("place", 5, 1_000))
    events = [start(), *listing(*races), *bought("R1", at(9, 54), t1)]
    s1 = settle(t1, {**codes, 3: 3}, {("win", 1): 350, ("place", 1): 130, ("place", 2): 210, ("place", 4): 160})
    events.append(settled_event("R1", at(10, 8), s1))
    state = fold(events)
    assert (state.cash, state.k) == (9_000, 2)  # 3 番は競走除外で返還、5 番の複勝は外れ → 1 回分使う

    edited = (Ticket("win", 7, 1_500),)
    events += [Recommended(at=at(10, 24), race_id="R2", tickets=(Ticket("win", 7, 2_000),)),
               Confirmed(at=at(10, 25), race_id="R2", action="edited", tickets=edited)]
    assert fold(events).cash == 7_500
    events.append(settled_event("R2", at(10, 38), settle(edited, {**codes, 7: 1}, {})))  # 全部返還: 払戻表は要らない
    state = fold(events)
    assert (state.cash, state.k) == (9_000, 2)  # 全部返還なら k は減らない
    ctx = decision_context(events, "R3", at(10, 54), P)
    assert ctx.ok and (ctx.x, ctx.k, ctx.k_eff) == (9_000, 2, 1)


def _full_day():
    races = [("R1", at(10, 0), "東京"), ("R2", at(10, 30), "東京"), ("R3", at(11, 0), "東京"), ("R4", at(11, 30), "東京")]
    return [
        start(), *listing(*races),
        RaceListed(at=at(9, 0), race_id="R4", post_time=at(11, 35), venue="東京"),  # 発走時刻の変更
        *bought("R1", at(9, 54)), Settled(at=at(10, 8), race_id="R1", payout=0, refund=0),
        Missed(at=at(10, 24), race_id="R2", reason="gate"),
        *bought("R3", at(10, 54), (Ticket("win", 2, 1_500), Ticket("win", 6, 500))),
        Settled(at=at(11, 8), race_id="R3", payout=6_300, refund=0),
        BalanceSynced(at=at(11, 10), x=13_300),
        Corrected(at=at(11, 12), race_id="R3", payout=6_400, note="払戻の訂正"),
    ]


def test_folding_is_idempotent():
    events = _full_day()
    once = fold(events)
    assert fold(events + events) == once
    assert fold([e for pair in zip(events, events) for e in pair]) == once
    assert once.races["R4"].post_time == at(11, 35)
    assert (once.cash, once.k) == (13_400, 1)


def test_goal_reached_ends_the_day():
    events = [start(), *listing(("R1", at(10, 0), "東京"), ("R2", at(10, 30), "東京")),
              *bought("R1", at(9, 54), (Ticket("win", 4, 2_500),)),
              Settled(at=at(10, 8), race_id="R1", payout=12_600, refund=0)]
    assert fold(events).cash == 20_100
    assert end_reason(events, at(10, 5), P) is None  # 精算時刻の前はまだ分からない
    assert end_reason(events, at(10, 8), P) == "goal"
    ctx = decision_context(events, "R2", at(10, 24), P)
    assert (ctx.skip_reason, ctx.end) == ("session_over", "goal")


def test_other_end_conditions():
    races = listing(("R1", at(10, 0), "東京"), ("R2", at(10, 30), "東京"))
    no_bets = [start(k0=1), *races, *bought("R1", at(9, 54)), Settled(at=at(10, 8), race_id="R1", payout=0, refund=0)]
    assert fold(no_bets).k == 0 and end_reason(no_bets, at(10, 10), P) == "no_bets"
    assert decision_context(no_bets, "R2", at(10, 24), P).end == "no_bets"

    broke = [start(x0=10_050), *races, *bought("R1", at(9, 54), (Ticket("win", 1, 10_000),)),
             Settled(at=at(10, 8), race_id="R1", payout=0, refund=0)]
    assert fold(broke).cash == 50 and end_reason(broke, at(10, 10), P) == "no_funds"

    idle = [start(), *races]
    assert end_reason(idle, at(10, 0), P) is None
    assert end_reason(idle, at(10, 27, 1), P) == "day_over"


def test_k_eff_counts_only_chainable_races_and_respects_the_venue_filter():
    races = [("T11", at(15, 0), "東京"), ("K11", at(15, 5), "京都"), ("N11", at(15, 10), "新潟"),
             ("T12", at(15, 30), "東京"), ("K12", at(15, 35), "京都"), ("N12", at(15, 40), "新潟")]
    events = [start(k0=5), *listing(*races)]
    state = fold(events)
    assert [chain_len(state, r[0], P) for r in races] == [2, 2, 2, 1, 1, 1]
    ctx = decision_context(events, "T11", at(14, 54), P)
    assert ctx.ok and (ctx.k, ctx.k_eff) == (5, 2)

    tokyo = [start(k0=5, venues=("東京",)), *listing(*races)]
    assert decision_context(tokyo, "K11", at(14, 59), P).skip_reason == "other_venue"
    assert chain_len(fold(tokyo), "T11", P) == 2
    assert end_reason(tokyo, at(15, 27), P) is None  # 東京の最終 (15:30) の締切 = 発走 3 分前
    assert end_reason(tokyo, at(15, 27, 1), P) == "day_over"


def test_deadline_and_already_decided():
    events = [start(), *listing(("R1", at(10, 0), "東京"))]
    assert decision_context(events, "R1", at(9, 57), P).ok
    assert decision_context(events, "R1", at(9, 57, 1), P).skip_reason == "deadline"
    missed = [*events, Missed(at=at(9, 54), race_id="R1", reason="gate")]
    assert decision_context(missed, "R1", at(9, 55), P).skip_reason == "already_decided"


def test_corrections_and_balance_sync():
    races = listing(("R1", at(10, 0), "東京"), ("R2", at(10, 30), "東京"), ("R3", at(11, 0), "東京"))
    base = [start(), *races, *bought("R1", at(9, 54)), Settled(at=at(10, 8), race_id="R1", payout=0, refund=0)]
    # 実際は 2,000 買っていて 5,600 の払戻だった
    fixed = [*base, Corrected(at=at(10, 15), race_id="R1", tickets=(Ticket("win", 1, 2_000),), payout=5_600)]
    state = fold(fixed)
    assert (state.cash, state.k, state.races["R1"].status) == (13_600, 2, "settled")
    # 購入内容だけを直すと精算は取り消され、精算待ちに戻る
    reopened = [*fixed, Corrected(at=at(10, 16), race_id="R1", tickets=(Ticket("win", 1, 1_500),))]
    state = fold(reopened)
    assert (state.cash, state.k, state.races["R1"].status) == (8_500, 3, "bought")
    assert decision_context(reopened, "R2", at(10, 24), P).skip_reason == "pending"
    manual = [*reopened, Corrected(at=at(10, 17), race_id="R1", payout=0)]
    assert (fold(manual).cash, fold(manual).k) == (8_500, 2)
    synced = [*manual, BalanceSynced(at=at(10, 20), x=8_400)]
    assert fold(synced).cash == 8_400
    undone = [*synced, Corrected(at=at(10, 21), race_id="R1", tickets=())]
    state = fold(undone)
    assert (state.cash, state.k, state.races["R1"].status) == (9_900, 3, "not_bought")
    # 推奨なしで自分で買ったレースも記録でき、精算まで次の判断を止める
    own = [start(), *races, Corrected(at=at(10, 1), race_id="R1", tickets=WIN1)]
    assert decision_context(own, "R2", at(10, 24), P).skip_reason == "pending"


def test_invalid_transitions_are_rejected():
    base = [start(), *listing(("R1", at(10, 0), "東京"))]
    rec = Recommended(at=at(9, 54), race_id="R1", tickets=WIN1)
    confirmed = Confirmed(at=at(9, 55), race_id="R1", action="bought")
    bad_logs = [
        base[1:],  # SessionStarted が先頭にない
        [SessionStarted(at=at(8, 30), x0=20_000, goal=20_000, k0=3)],
        [*base, confirmed],  # 推奨なしの確認
        [*base, Settled(at=at(10, 8), race_id="R1", payout=0, refund=0)],  # 買っていない
        [*base, Recommended(at=at(9, 54), race_id="R9", tickets=WIN1)],  # 一覧にない
        [*base, Recommended(at=at(9, 54), race_id="R1", tickets=())],
        [*base, rec, Missed(at=at(9, 58), race_id="R1", reason="deadline")],
        [*base, rec, confirmed, Settled(at=at(10, 8), race_id="R1", payout=0, refund=2_000)],
        [*base, Corrected(at=at(10, 0), race_id="R1")],
        [*base, Corrected(at=at(10, 0), race_id="R1", payout=100)],
    ]
    for log in bad_logs:
        with pytest.raises(SessionError):
            fold(log)
    # 同じキーの 2 回目は無視 (冪等、先に記録した方が有効)
    assert fold([*base, rec, Recommended(at=at(9, 56), race_id="R1", tickets=(Ticket("win", 2, 500),))]) == fold([*base, rec])


def test_a_second_session_start_with_different_content_is_rejected(tmp_path):
    base = [start(), *listing(("R1", at(10, 0), "東京"))]
    rerun = SessionStarted(at=at(9, 0), x0=10_000, goal=20_000, k0=3)  # 同じ内容の再実行 (時刻だけ違う) は冪等
    assert fold([*base, rerun]) == fold(base)
    for other in (SessionStarted(at=at(8, 30), x0=50_000, goal=90_000, k0=6),
                  SessionStarted(at=at(8, 30), x0=10_000, goal=20_000, k0=3, venues=("東京",))):
        with pytest.raises(SessionError, match="SessionStarted"):
            fold([*base, other])
    path = tmp_path / "events.jsonl"
    for e in base:
        append_event(path, e)
    assert append_event(path, rerun) is False
    with pytest.raises(SessionError, match="SessionStarted"):
        append_event(path, SessionStarted(at=at(8, 31), x0=50_000, goal=90_000, k0=6))
    assert read_events(path) == base


def test_confirmation_action_is_validated(tmp_path):
    for bad in (dict(action="yes"), dict(action="edited"), dict(action="bought", tickets=WIN1),
                dict(action="not_bought", tickets=WIN1)):
        with pytest.raises(SessionError, match="R1"):
            Confirmed(at=at(9, 55), race_id="R1", **bad)
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"type": "confirmed", "at": "2026-11-01T09:55:00", "race_id": "R1", "action": "yes"}\n')
    with pytest.raises(SessionError, match="1 行目"):
        read_events(path)


def test_event_json_round_trip_for_every_type():
    decision = Decision(
        x=10_000, goal=20_000, k_eff=3, p_star=0.418, ceiling=0.426, o_star=4.847,
        tickets=(Ticket("win", 3, 2_600),), lines=(Candidate("win", 3, 6.5, 0.79, 5.135, 0.22, 0.12),),
        o_eff=5.135, p_hit=0.22, ev_goal=1.13, ev_q=1.13, ev_raw=1.43, bankroll_if_hit=20_751.0,
    )
    jst_aware = datetime(2026, 11, 1, 0, 54, tzinfo=timezone.utc)  # = 09:54 JST
    events = [
        *_full_day(),
        recommended_event("R4", jst_aware, decision),
        Confirmed(at=at(11, 25), race_id="R4", action="edited", tickets=(Ticket("place", 3, 1_000),)),
    ]
    for e in events:
        back = event_from_dict(json.loads(json.dumps(event_to_dict(e), ensure_ascii=False)))
        assert back == e and back.key() == e.key()
    assert events[-2].at == at(9, 54)
    with pytest.raises(ValueError):
        recommended_event("R4", at(9, 54), Decision(x=1, goal=2, k_eff=1, p_star=0.0, ceiling=0.0, skip_reason="gate"))
    with pytest.raises(SessionError):
        event_from_dict({"type": "unknown", "at": "2026-11-01T10:00:00"})


def test_journal_round_trip_idempotency_and_validation(tmp_path):
    path = tmp_path / "sessions" / "20261101" / "events.jsonl"
    events = _full_day()
    assert all(append_event(path, e) for e in events)
    assert not append_event(path, events[3])  # 同じ冪等キー
    assert read_events(path) == events
    assert load_state(path) == fold(events)
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(events)
    with pytest.raises(SessionError):
        append_event(path, Settled(at=at(11, 40), race_id="R4", payout=0, refund=0))  # 買っていない
    assert read_events(path) == events


def test_torn_last_line_is_ignored_and_repaired(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [start(), *listing(("R1", at(10, 0), "東京"))]
    for e in events:
        append_event(path, e)
    with open(path, "ab") as f:
        f.write(b'{"type": "missed", "at": "2026-11-01T09:5')  # 書き込みの途中で止まった行
    assert read_events(path) == events
    missed = Missed(at=at(9, 54), race_id="R1", reason="gate")
    assert append_event(path, missed)
    assert read_events(path) == [*events, missed]
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def test_corrupt_complete_line_is_an_error(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, start())
    with open(path, "ab") as f:
        f.write(b"not json\n")
    with pytest.raises(SessionError, match="2 行目"):
        read_events(path)


def test_concurrent_appends_keep_every_line_intact(tmp_path):
    path = tmp_path / "events.jsonl"
    races = [(f"R{i:02d}", at(10, 0) + timedelta(minutes=5 * i), "東京") for i in range(40)]
    for e in [start(), *listing(*races)]:
        append_event(path, e)
    results: list[bool] = []

    def worker(ids):
        for rid in ids:
            results.append(append_event(path, Missed(at=at(9, 0), race_id=rid, reason="gate")))

    threads = [threading.Thread(target=worker, args=([r[0] for r in races[i::4]],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 40
    state = load_state(path)
    assert all(r.status == "missed" for r in state.races.values())
    assert len(read_events(path)) == 1 + 40 + 40
