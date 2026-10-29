import pytest

from keiba_goal.betting.settle import IncompletePayouts, settle
from keiba_goal.betting.tickets import Ticket, format_tickets, parse_tickets

# 202606040512 (2026-09-19 中山 12R) の払戻表: 単勝 5 → 420、複勝 5/7/1 → 150/160/170
PAYOUTS = {("win", 5): 420, ("place", 5): 150, ("place", 7): 160, ("place", 1): 170}
CODES = {h: 0 for h in range(1, 17)}


def test_hits_pay_per_100_yen_and_misses_pay_nothing():
    s = settle([Ticket("win", 5, 500), Ticket("place", 7, 300), Ticket("win", 7, 200)], CODES, PAYOUTS)
    assert [ln.result for ln in s.lines] == ["hit", "hit", "miss"]
    assert [ln.amount for ln in s.lines] == [2_100, 480, 0]
    assert s.payout == 2_580 and s.refund == 0 and s.cash_delta == 2_580
    assert s.counts_bet


def test_dead_heat_pays_each_row_of_the_table():
    # 1 着同着: 単勝の払戻表に 2 行 (それぞれの払戻)
    payouts = {("win", 3): 1_150, ("win", 8): 260, ("place", 3): 300, ("place", 8): 120, ("place", 2): 140}
    s = settle([Ticket("win", 3, 100), Ticket("win", 8, 1_000), Ticket("place", 2, 200)], CODES, payouts)
    assert [ln.amount for ln in s.lines] == [1_150, 2_600, 280]


def test_refunds_for_scratched_horses_and_k():
    codes = {**CODES, 4: 1, 9: 3, 11: 2}
    all_refunded = settle([Ticket("win", 4, 1_000), Ticket("place", 9, 500)], codes, PAYOUTS)
    assert [ln.result for ln in all_refunded.lines] == ["refund", "refund"]
    assert all_refunded.refund == 1_500 and all_refunded.cash_delta == 1_500
    assert not all_refunded.counts_bet  # 全部返還なら k を減らさない

    partly = settle([Ticket("win", 11, 1_000), Ticket("place", 6, 500)], codes, PAYOUTS)
    assert [ln.result for ln in partly.lines] == ["refund", "miss"]
    assert partly.cash_delta == 1_000
    assert partly.counts_bet  # 返還されなかった券が 1 枚でもあれば k を減らす


def test_cancelled_race_refunds_everything():
    s = settle([Ticket("win", 5, 500), Ticket("place", 7, 300)], CODES, PAYOUTS, race_cancelled=True)
    assert s.refund == 800 and s.payout == 0 and not s.counts_bet
    # 不成立のレースには結果も払戻表も無い
    s = settle([Ticket("win", 5, 500)], {}, {}, race_cancelled=True)
    assert [ln.result for ln in s.lines] == ["refund"] and s.cash_delta == 500


def test_incomplete_payout_table_is_not_a_miss():
    win_only = {("win", 5): 420}
    for tickets, payouts in [
        ([Ticket("win", 5, 500)], {}),                                       # 表が空
        ([Ticket("win", 7, 500)], {("place", 5): 150, ("place", 7): 160}),   # 単勝の行が無い
        ([Ticket("place", 7, 300)], win_only),                               # 複勝の券があるのに複勝の行が無い
        ([Ticket("place", 7, 300)], {**win_only, ("place", 5): 150}),        # 複勝が 1 行だけ
        ([Ticket("win", 4, 100), Ticket("place", 6, 100)], win_only),        # 一部返還でも残りの券は精算が要る
    ]:
        with pytest.raises(IncompletePayouts, match="PENDING"):
            settle(tickets, {**CODES, 4: 1}, payouts)
    # 単勝の券だけなら複勝の行は要らない。複勝 2 行 (5〜7 頭立て) で足りる
    assert settle([Ticket("win", 7, 500)], CODES, win_only).lines[0].result == "miss"
    two_places = {**win_only, ("place", 5): 130, ("place", 7): 180}
    assert settle([Ticket("place", 7, 300)], CODES, two_places).payout == 540
    # 返還される券だけなら払戻表は要らない (読めない行があっても見ない)
    assert settle([Ticket("win", 4, 100)], {**CODES, 4: 1}, {}).refund == 100
    assert settle([Ticket("win", 4, 100)], {**CODES, 4: 1}, {("win", 5): None}).refund == 100


def test_errors_and_non_refund_abnormal_codes():
    codes = {**CODES, 2: 4, 3: 7}  # 競走中止・降着は返還ではない (払戻表どおり)
    payouts = {("win", 1): 380, ("place", 1): 140, ("place", 3): 230, ("place", 6): 300}
    s = settle([Ticket("win", 2, 100), Ticket("place", 3, 100)], codes, payouts)
    assert [ln.result for ln in s.lines] == ["miss", "hit"]
    with pytest.raises(ValueError, match="結果"):
        settle([Ticket("win", 17, 100)], CODES, PAYOUTS)
    with pytest.raises(ValueError, match="100 円単位"):
        settle([Ticket("win", 5, 150)], CODES, PAYOUTS)
    for bad in ({**PAYOUTS, ("win", 5): 0}, {**PAYOUTS, ("exacta", 5): 900}, {**PAYOUTS, ("win", 5): 420.5}):
        with pytest.raises(ValueError, match="払戻"):
            settle([Ticket("win", 5, 100)], CODES, bad)
    for unreadable in (None, float("nan")):  # 読めなかった行は外れではなく表の欠け
        with pytest.raises(IncompletePayouts):
            settle([Ticket("win", 5, 100)], CODES, {**PAYOUTS, ("place", 1): unreadable})


def test_ticket_text_round_trip():
    tickets = parse_tickets("win:5:500, place:7:300")
    assert tickets == (Ticket("win", 5, 500), Ticket("place", 7, 300))
    assert format_tickets(tickets) == "win:5:500,place:7:300"
    assert parse_tickets("") == ()
    assert Ticket.from_dict(tickets[0].to_dict()) == tickets[0]
    for bad in ("win:5", "exacta:5:100", "win:5:150", "win:5:0", "win:5:100,win:5:200", "win:0:100"):
        with pytest.raises(ValueError):
            parse_tickets(bad)
