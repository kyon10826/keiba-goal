"""精算: 返還 → 払戻表 → 0 の順に決める (design §3.4、§10)。

- レース不成立、または出走取消・発走除外・競走除外 (error_code 1/2/3) の馬の券は返還 (賭け金が戻る)。
- それ以外は払戻表 (券種, 馬番) → 100 円あたりの払戻。同着は払戻表に複数行あるのでそのまま使う。
- 返還されない券があるのに払戻表が欠けていれば (単勝の行が無い、複勝の券があるのに複勝の行が 2 行未満)
  IncompletePayouts。外れ扱いにせず、呼び出し側は精算を記録しない (セッションは PENDING のまま)。
- 完全な払戻表に無い券は外れ (0)。
- 返還されなかった券が 1 枚でもあれば、その購入で残り回数 k を 1 減らす。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from keiba_goal.betting.tickets import BET_TYPES, Ticket
from keiba_goal.data.schema import NOT_STARTED_CODES

Result = Literal["refund", "hit", "miss"]


class IncompletePayouts(ValueError):
    """払戻表が欠けていて精算できない (外れとは区別する)。"""


@dataclass(frozen=True)
class SettledLine:
    ticket: Ticket
    result: Result
    amount: int  # 戻ってくる金額 (返還は賭け金、的中は払戻、外れは 0)

    def to_dict(self) -> dict[str, Any]:
        return {**self.ticket.to_dict(), "result": self.result, "amount": self.amount}


@dataclass(frozen=True)
class Settlement:
    lines: tuple[SettledLine, ...]

    @property
    def payout(self) -> int:
        return sum(ln.amount for ln in self.lines if ln.result == "hit")

    @property
    def refund(self) -> int:
        return sum(ln.amount for ln in self.lines if ln.result == "refund")

    @property
    def cash_delta(self) -> int:
        """精算で手元に戻る金額 (賭け金は購入時に引いてあるので、払戻 + 返還)。"""
        return self.payout + self.refund

    @property
    def counts_bet(self) -> bool:
        """返還されなかった券が 1 枚でもあれば True (k を 1 減らす)。"""
        return any(ln.result != "refund" for ln in self.lines)


def _check_payouts(payouts: Mapping[tuple[str, int], int]) -> None:
    for (bet_type, horse), per_100 in payouts.items():
        if bet_type not in BET_TYPES:
            raise ValueError(f"払戻表の券種が不正です: {bet_type!r}")
        if per_100 is None or per_100 != per_100:  # None / NaN は読めなかった行
            raise IncompletePayouts(f"払戻表の {bet_type} {horse} の金額がありません (精算せず PENDING のままにします)")
        if per_100 <= 0 or int(per_100) != per_100:
            raise ValueError(f"払戻は正の整数 (円) です: {bet_type} {horse} {per_100}")


def _check_complete(held: list[Ticket], payouts: Mapping[tuple[str, int], int]) -> None:
    """返還されない券を精算できるだけの行が払戻表にあるか (同着は行が増えるだけなので下限で調べる)。"""
    n_win = sum(1 for bet_type, _ in payouts if bet_type == "win")
    n_place = sum(1 for bet_type, _ in payouts if bet_type == "place")
    if held and n_win == 0:
        raise IncompletePayouts("払戻表に単勝の行がありません (精算せず PENDING のままにします)")
    if any(t.bet_type == "place" for t in held) and n_place < 2:
        raise IncompletePayouts(f"複勝の券があるのに払戻表の複勝が {n_place} 行です (精算せず PENDING のままにします)")


def settle(
    tickets: Iterable[Ticket],
    error_codes: Mapping[int, int],
    payouts: Mapping[tuple[str, int], int],
    race_cancelled: bool = False,
) -> Settlement:
    """券を精算する。error_codes は馬番 → 異常区分、payouts は (券種, 馬番) → 100 円あたりの払戻 (円)。

    レース不成立 (race_cancelled) は全額返還で、error_codes と payouts は見ない。
    """
    tickets = tuple(tickets)
    for t in tickets:
        if t.stake <= 0 or t.stake % 100:
            raise ValueError(f"金額は 100 円単位の正の値です: {t}")
    if race_cancelled:
        return Settlement(tuple(SettledLine(t, "refund", t.stake) for t in tickets))
    for t in tickets:
        if t.horse_num not in error_codes:
            raise ValueError(f"馬番 {t.horse_num} の結果がありません")
    refunded = [error_codes[t.horse_num] in NOT_STARTED_CODES for t in tickets]
    held = [t for t, r in zip(tickets, refunded) if not r]
    if held:  # 全部返還なら払戻表は見ない
        _check_payouts(payouts)
        _check_complete(held, payouts)
    lines: list[SettledLine] = []
    for t, r in zip(tickets, refunded):
        per_100 = payouts.get((t.bet_type, t.horse_num))
        if r:
            lines.append(SettledLine(t, "refund", t.stake))
        elif per_100 is None:
            lines.append(SettledLine(t, "miss", 0))
        else:
            lines.append(SettledLine(t, "hit", t.stake // 100 * int(per_100)))
    return Settlement(tuple(lines))
