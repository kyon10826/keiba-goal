"""単勝・複勝の券 (券種・馬番・金額) と `day record --tickets` の文字列形式。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

BetType = Literal["win", "place"]
BET_TYPES: tuple[BetType, ...] = ("win", "place")


@dataclass(frozen=True)
class Ticket:
    bet_type: BetType
    horse_num: int
    stake: int

    def __post_init__(self) -> None:
        if self.bet_type not in BET_TYPES:
            raise ValueError(f"券種は win / place のどちらかです: {self.bet_type!r}")
        if self.horse_num < 1:
            raise ValueError(f"馬番が不正です: {self.horse_num}")
        if self.stake < 0:
            raise ValueError(f"金額が負です: {self.stake}")

    def to_dict(self) -> dict[str, Any]:
        return {"bet_type": self.bet_type, "horse_num": self.horse_num, "stake": self.stake}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Ticket:
        return cls(str(d["bet_type"]), int(d["horse_num"]), int(d["stake"]))  # type: ignore[arg-type]


def total_stake(tickets: Iterable[Ticket]) -> int:
    return sum(t.stake for t in tickets)


def parse_tickets(text: str, unit: int = 100) -> tuple[Ticket, ...]:
    """'win:5:500,place:7:300' → 券のタプル。空文字は「買っていない」。"""
    out: list[Ticket] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError(f"券の形式は 券種:馬番:金額 です: {part!r}")
        bet_type, horse, stake = (f.strip() for f in fields)
        t = Ticket(bet_type, int(horse), int(stake))  # type: ignore[arg-type]
        if t.stake == 0 or t.stake % unit:
            raise ValueError(f"金額は {unit} 円単位の正の値です: {part!r}")
        out.append(t)
    if len({(t.bet_type, t.horse_num) for t in out}) != len(out):
        raise ValueError(f"同じ券が重複しています: {text!r}")
    return tuple(out)


def format_tickets(tickets: Iterable[Ticket]) -> str:
    return ",".join(f"{t.bet_type}:{t.horse_num}:{t.stake}" for t in tickets)
