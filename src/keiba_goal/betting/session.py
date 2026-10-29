"""1 日のセッション: イベントソース型の状態 (design §9.1、§11)。

- `sessions/{date}/events.jsonl` に追記だけを行う (flock + fsync)。状態は `fold(events)` で毎回作る (純粋関数)。
- 冪等キー (race_id, step) が同じイベントは 2 回目以降を無視するので、同じログを何度畳み込んでも同じ状態になる。
  ただし内容 (予算・目標・回数・場) の違う 2 回目の SessionStarted は SessionError (訂正の手段が無いため)。
- 時刻に依存する問い合わせ (decision_context / end_reason / unconfirmed) は、ログのうち at ≤ now のイベントだけを
  畳み込む (state_at)。バックテストが先に書いた精算や、後の時刻の残高同期・訂正・確認は判断に使わない。
- 未確定の賭けは同時に 1 つまで: それまでに買った券がすべて精算済みのときだけ次のレースを判断する。
- 資金は手元の現金 (未精算の賭け金を除く)。残り回数 k は「返還されなかった券が 1 枚でもある精算済みの購入」で減る。
- 表示した推奨が発走時刻までに確認されなければ UNCONFIRMED とし、解決 (Confirmed / Corrected) まで以後の推奨を止める。
時刻はすべて JST の naive datetime (タイムゾーン付きは JST に直してから外す)。
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal, Union
from zoneinfo import ZoneInfo

from keiba_goal.betting.basket import Decision
from keiba_goal.betting.params import BettingParams
from keiba_goal.betting.settle import Settlement
from keiba_goal.betting.tickets import Ticket, total_stake

JST = ZoneInfo("Asia/Tokyo")

RaceStatus = Literal["listed", "recommended", "bought", "not_bought", "settled", "missed"]
ConfirmAction = Literal["bought", "not_bought", "edited"]
CONFIRM_ACTIONS: tuple[ConfirmAction, ...] = ("bought", "not_bought", "edited")


class SessionError(ValueError):
    """イベントの並びが状態遷移の規則に合わない。"""


def as_jst(dt: datetime) -> datetime:
    return dt.astimezone(JST).replace(tzinfo=None) if dt.tzinfo is not None else dt


# ---------------------------------------------------------------- events


class _EventBase:
    """時刻は JST の naive に、列はタプルにそろえる (冪等キーが JSON の往復で変わらないように)。"""

    def __post_init__(self) -> None:
        for name in ("at", "post_time"):
            value = getattr(self, name, None)
            if isinstance(value, datetime):
                object.__setattr__(self, name, as_jst(value))
        for name in ("tickets", "venues", "lines"):
            value = getattr(self, name, None)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))


@dataclass(frozen=True)
class SessionStarted(_EventBase):
    TYPE: ClassVar[str] = "session_started"
    at: datetime
    x0: int
    goal: int
    k0: int
    venues: tuple[str, ...] = ()  # 空なら全場

    def key(self) -> tuple[str, str]:
        return ("", self.TYPE)

    def same_session(self, other: SessionStarted) -> bool:
        """予算・目標・回数・場が同じ (開始時刻だけが違う再実行は同じセッション)。"""
        return (self.x0, self.goal, self.k0, self.venues) == (other.x0, other.goal, other.k0, other.venues)


@dataclass(frozen=True)
class RaceListed(_EventBase):
    TYPE: ClassVar[str] = "race_listed"
    at: datetime
    race_id: str
    post_time: datetime
    venue: str

    def key(self) -> tuple[str, str]:
        # 発走時刻の変更は別のイベントとして残す (同じ時刻の再登録だけを重複とみなす)
        return (self.race_id, f"listed@{self.post_time.isoformat()}")


@dataclass(frozen=True)
class Recommended(_EventBase):
    TYPE: ClassVar[str] = "recommended"
    at: datetime
    race_id: str
    tickets: tuple[Ticket, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str]:
        return (self.race_id, "recommended")


@dataclass(frozen=True)
class Confirmed(_EventBase):
    TYPE: ClassVar[str] = "confirmed"
    at: datetime
    race_id: str
    action: ConfirmAction
    tickets: tuple[Ticket, ...] = ()  # edited のときに実際に買った券 (1 枚以上)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action not in CONFIRM_ACTIONS:
            raise SessionError(f"{self.race_id}: 確認の action は {CONFIRM_ACTIONS} のどれかです: {self.action!r}")
        if (self.action == "edited") != bool(self.tickets):
            raise SessionError(f"{self.race_id}: 券は action='edited' のときだけ 1 枚以上指定します")

    def key(self) -> tuple[str, str]:
        return (self.race_id, "confirmed")


@dataclass(frozen=True)
class Settled(_EventBase):
    TYPE: ClassVar[str] = "settled"
    at: datetime
    race_id: str
    payout: int
    refund: int
    lines: tuple[dict[str, Any], ...] = ()

    def key(self) -> tuple[str, str]:
        return (self.race_id, "settled")


@dataclass(frozen=True)
class Missed(_EventBase):
    TYPE: ClassVar[str] = "missed"
    at: datetime
    race_id: str
    reason: str  # pending / unconfirmed / deadline / gate / infeasible / no_candidates / safety:... など

    def key(self) -> tuple[str, str]:
        return (self.race_id, "missed")


@dataclass(frozen=True)
class Corrected(_EventBase):
    """`day record` の訂正。tickets: None は購入内容を変えない、() は買っていない。payout: 手入力の精算。"""

    TYPE: ClassVar[str] = "corrected"
    at: datetime
    race_id: str
    tickets: tuple[Ticket, ...] | None = None
    payout: int | None = None
    refund: int = 0
    note: str = ""

    def key(self) -> tuple[str, str]:
        return (self.race_id, f"corrected@{self.at.isoformat()}")


@dataclass(frozen=True)
class BalanceSynced(_EventBase):
    TYPE: ClassVar[str] = "balance_synced"
    at: datetime
    x: int

    def key(self) -> tuple[str, str]:
        return ("", f"balance@{self.at.isoformat()}")


Event = Union[SessionStarted, RaceListed, Recommended, Confirmed, Settled, Missed, Corrected, BalanceSynced]
EVENT_TYPES: dict[str, type] = {
    cls.TYPE: cls
    for cls in (SessionStarted, RaceListed, Recommended, Confirmed, Settled, Missed, Corrected, BalanceSynced)
}


def recommended_event(race_id: str, at: datetime, decision: Decision) -> Recommended:
    if not decision.recommended:
        raise ValueError("見送りの判断は Missed で記録します")
    return Recommended(at=at, race_id=race_id, tickets=decision.tickets, details=decision.summary())


def settled_event(race_id: str, at: datetime, settlement: Settlement) -> Settled:
    return Settled(at=at, race_id=race_id, payout=settlement.payout, refund=settlement.refund,
                   lines=tuple(ln.to_dict() for ln in settlement.lines))


def event_to_dict(event: Event) -> dict[str, Any]:
    out: dict[str, Any] = {"type": event.TYPE}
    for f in fields(event):
        value = getattr(event, f.name)
        if isinstance(value, datetime):
            value = as_jst(value).isoformat()
        elif f.name == "tickets" and value is not None:
            value = [t.to_dict() for t in value]
        elif isinstance(value, tuple):
            value = list(value)
        out[f.name] = value
    return out


def event_from_dict(d: Mapping[str, Any]) -> Event:
    cls = EVENT_TYPES.get(str(d.get("type")))
    if cls is None:
        raise SessionError(f"不明なイベント種別です: {d.get('type')!r}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        if f.name in ("at", "post_time"):
            value = datetime.fromisoformat(value)
        elif f.name == "tickets":
            value = None if value is None else tuple(Ticket.from_dict(t) for t in value)
        elif f.name in ("venues", "lines"):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------- state


@dataclass(frozen=True)
class RaceState:
    race_id: str
    venue: str
    post_time: datetime
    status: RaceStatus = "listed"
    recommended: tuple[Ticket, ...] = ()
    bought: tuple[Ticket, ...] = ()
    credit: int = 0  # 精算で戻った金額 (払戻 + 返還)
    refund: int = 0
    settled_at: datetime | None = None
    missed_reason: str | None = None

    @property
    def stake(self) -> int:
        return total_stake(self.bought)

    @property
    def counts_bet(self) -> bool:
        """精算済みで、返還されなかった券が 1 枚でもある (k を 1 減らす)。"""
        return self.status == "settled" and self.refund < self.stake


@dataclass(frozen=True)
class SessionState:
    started_at: datetime
    x0: int
    goal: int
    k0: int
    venues: tuple[str, ...]
    cash: int  # 手元の現金 (未精算の賭け金は引いてある)
    races: Mapping[str, RaceState]

    @property
    def k(self) -> int:
        return self.k0 - sum(r.counts_bet for r in self.races.values())

    def in_session(self, race: RaceState) -> bool:
        return not self.venues or race.venue in self.venues

    def with_status(self, *statuses: RaceStatus) -> list[RaceState]:
        return [r for r in self.races.values() if r.status in statuses]


def _apply(races: dict[str, RaceState], cash: int, ev: Event) -> int:
    """1 イベントを races に反映し、新しい手元の現金を返す。規則に合わなければ SessionError。

    SessionStarted は来ない (fold が先頭で受け、2 回目は冪等キーで _check_duplicate に回る)。
    """
    if isinstance(ev, BalanceSynced):
        return ev.x
    if isinstance(ev, RaceListed):
        old = races.get(ev.race_id)
        races[ev.race_id] = (
            replace(old, post_time=as_jst(ev.post_time), venue=ev.venue) if old
            else RaceState(ev.race_id, ev.venue, as_jst(ev.post_time))
        )
        return cash
    race = races.get(ev.race_id)
    if race is None:
        raise SessionError(f"一覧にないレースです: {ev.race_id}")

    def need(*statuses: RaceStatus) -> None:
        if race.status not in statuses:
            raise SessionError(f"{ev.race_id}: {race.status} のレースに {ev.TYPE} は記録できません")

    if isinstance(ev, Recommended):
        need("listed")
        if not ev.tickets:
            raise SessionError("推奨に券がありません (見送りは Missed)")
        races[ev.race_id] = replace(race, status="recommended", recommended=ev.tickets)
        return cash
    if isinstance(ev, Missed):
        need("listed")
        races[ev.race_id] = replace(race, status="missed", missed_reason=ev.reason)
        return cash
    if isinstance(ev, Confirmed):
        need("recommended")
        bought = {"bought": race.recommended, "edited": ev.tickets, "not_bought": ()}[ev.action]
        races[ev.race_id] = replace(race, status="bought" if bought else "not_bought", bought=bought)
        return cash - total_stake(bought)
    if isinstance(ev, Settled):
        need("bought")
        if not 0 <= ev.refund <= race.stake or ev.payout < 0:
            raise SessionError(f"{ev.race_id}: 精算額が不正です (払戻 {ev.payout}, 返還 {ev.refund})")
        races[ev.race_id] = replace(race, status="settled", credit=ev.payout + ev.refund, refund=ev.refund,
                                    settled_at=as_jst(ev.at))
        return cash + ev.payout + ev.refund
    if isinstance(ev, Corrected):
        if ev.tickets is None and ev.payout is None:
            raise SessionError("訂正の中身がありません")
        if ev.tickets is not None:
            # 購入内容を差し替える。精算済みなら精算も取り消す (同じイベントの payout か、改めての精算を待つ)
            cash += race.stake - total_stake(ev.tickets) - race.credit
            race = replace(race, status="bought" if ev.tickets else "not_bought", bought=ev.tickets,
                           credit=0, refund=0, settled_at=None)
        if ev.payout is not None:
            if race.status not in ("bought", "settled"):
                raise SessionError(f"{ev.race_id}: 買っていないレースに払戻は記録できません")
            if not 0 <= ev.refund <= race.stake or ev.payout < 0:
                raise SessionError(f"{ev.race_id}: 精算額が不正です (払戻 {ev.payout}, 返還 {ev.refund})")
            cash += ev.payout + ev.refund - race.credit
            race = replace(race, status="settled", credit=ev.payout + ev.refund, refund=ev.refund,
                           settled_at=as_jst(ev.at))
        races[ev.race_id] = race
        return cash
    raise SessionError(f"不明なイベントです: {ev!r}")


def _check_duplicate(first: Event, again: Event) -> None:
    """冪等キーが同じ 2 回目のイベントは無視する (先に記録した方が有効)。SessionStarted だけは内容の違いを拒否する。"""
    if isinstance(first, SessionStarted) and isinstance(again, SessionStarted) and not first.same_session(again):
        raise SessionError(
            f"SessionStarted が 2 回あり内容が違います (x0={first.x0}/{again.x0}, A={first.goal}/{again.goal}, "
            f"k0={first.k0}/{again.k0}, 場={first.venues}/{again.venues})"
        )


def fold(events: Iterable[Event]) -> SessionState:
    """イベント列 → 状態 (純粋関数)。冪等キーが重複するイベントは無視する (_check_duplicate)。"""
    seen: dict[tuple[str, str], Event] = {}
    start: SessionStarted | None = None
    races: dict[str, RaceState] = {}
    cash = 0
    for ev in events:
        key = ev.key()
        if key in seen:
            _check_duplicate(seen[key], ev)
            continue
        seen[key] = ev
        if start is None:
            if not isinstance(ev, SessionStarted):
                raise SessionError("最初のイベントは SessionStarted です")
            if not (0 < ev.x0 < ev.goal and ev.k0 >= 1):
                raise SessionError(f"予算・目標・回数が不正です (x0={ev.x0}, A={ev.goal}, k0={ev.k0})")
            start, cash = ev, ev.x0
            continue
        cash = _apply(races, cash, ev)
    if start is None:
        raise SessionError("SessionStarted がありません")
    return SessionState(as_jst(start.at), start.x0, start.goal, start.k0, tuple(start.venues), cash, dict(races))


# ---------------------------------------------------------------- queries


def decision_time(race: RaceState, params: BettingParams) -> datetime:
    return race.post_time - timedelta(minutes=params.decision_lead_min)


def _deadline(race: RaceState, params: BettingParams) -> datetime:
    return race.post_time - timedelta(minutes=params.deadline_min)


def _time_order(race: RaceState, params: BettingParams) -> tuple:
    return (decision_time(race, params), race.post_time, race.venue, race.race_id)


def chain_len(state: SessionState, race_id: str, params: BettingParams) -> int:
    """race_id から始めて、前に選んだレースの発走 + chain_gap 以降に判断時刻が来る未判断のレースを貪欲に数えた長さ。"""
    start = state.races[race_id]
    later = sorted(
        (r for r in state.races.values()
         if r.status == "listed" and state.in_session(r) and _time_order(r, params) > _time_order(start, params)),
        key=lambda r: _time_order(r, params),
    )
    count, last_post = 1, start.post_time
    gap = timedelta(minutes=params.chain_gap_min)
    for r in later:
        if decision_time(r, params) >= last_post + gap:
            count, last_post = count + 1, r.post_time
    return count


def state_at(events: Iterable[Event], now: datetime) -> SessionState:
    """now の時点の状態: at ≤ now のイベントだけを畳み込む (後の時刻の精算・残高同期・訂正・確認は使わない)。"""
    now = as_jst(now)
    past = [e for e in events if e.at <= now]
    if not any(isinstance(e, SessionStarted) for e in past):
        raise SessionError(f"{now:%Y-%m-%d %H:%M:%S} の時点でセッションが始まっていません")
    return fold(past)


def _end_reason(state: SessionState, now: datetime, params: BettingParams) -> str | None:
    if state.with_status("bought", "recommended"):
        return None
    if state.cash >= state.goal:
        return "goal"
    if state.k <= 0:
        return "no_bets"
    if state.cash < params.unit:
        return "no_funds"
    if not any(state.in_session(r) and now <= _deadline(r, params) for r in state.with_status("listed")):
        return "day_over"
    return None


def end_reason(events: Iterable[Event], now: datetime, params: BettingParams) -> str | None:
    """'goal' / 'no_bets' / 'no_funds' / 'day_over'、続くなら None。精算待ち・確認待ちの間は終わらない。

    資金・回数は now の時点で起きていたイベントだけで決める (state_at)。
    """
    now = as_jst(now)
    return _end_reason(state_at(events, now), now, params)


@dataclass(frozen=True)
class DecisionContext:
    """レース j を判断してよいか。skip_reason が None なら x・k・k_eff で decide してよい。"""

    race_id: str
    x: int
    k: int
    k_eff: int
    skip_reason: str | None = None
    end: str | None = None

    @property
    def ok(self) -> bool:
        return self.skip_reason is None


def decision_context(events: Iterable[Event], race_id: str, now: datetime, params: BettingParams) -> DecisionContext:
    """時刻 now にレースを判断してよいかと、そのときの資金・回数 (at ≤ now のイベントだけで決める)。

    判断時刻 (発走 − decision_lead_min) より前は too_early、締切 (発走 − deadline_min) を過ぎたら deadline。
    """
    now = as_jst(now)
    state = state_at(events, now)
    race = state.races.get(race_id)
    if race is None:
        raise SessionError(f"{race_id} は {now:%Y-%m-%d %H:%M:%S} の時点で一覧にありません")

    def skip(reason: str, end: str | None = None) -> DecisionContext:
        return DecisionContext(race_id, state.cash, state.k, 0, reason, end)

    if not state.in_session(race):
        return skip("other_venue")
    if race.status != "listed":
        return skip("already_decided")
    if now < decision_time(race, params):
        return skip("too_early")
    if now > _deadline(race, params):
        return skip("deadline")
    awaiting = state.with_status("recommended")
    if any(r.post_time <= now for r in awaiting):
        return skip("unconfirmed")
    if awaiting:
        return skip("awaiting_confirmation")
    if state.with_status("bought"):
        return skip("pending")
    ended = _end_reason(state, now, params)
    if ended is not None:
        return skip("session_over", ended)
    return DecisionContext(race_id, state.cash, state.k, min(state.k, chain_len(state, race_id, params)))


def unconfirmed(events: Iterable[Event], now: datetime) -> list[RaceState]:
    """発走時刻を過ぎても購入の確認がない推奨 (解決するまで以後の推奨を止める)。"""
    now = as_jst(now)
    return [r for r in state_at(events, now).with_status("recommended") if r.post_time <= now]


# ---------------------------------------------------------------- journal (I/O)


def _parse_lines(data: bytes) -> tuple[list[Event], int]:
    """完全な行 (改行で終わる行) だけを読む。戻り値の 2 つ目は完全な行の末尾までのバイト数。"""
    end = data.rfind(b"\n") + 1
    events = []
    for no, line in enumerate(data[:end].splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(event_from_dict(json.loads(line)))
        except (ValueError, KeyError, TypeError) as e:
            raise SessionError(f"events.jsonl の {no} 行目を読めません: {e}") from e
    return events, end


def read_events(path: Path) -> list[Event]:
    """イベントを読む。書き込み途中で止まった最後の行 (改行なし) は無視する。"""
    if not path.exists():
        return []
    return _parse_lines(path.read_bytes())[0]


def append_event(path: Path, event: Event) -> bool:
    """イベントを 1 行追記する (flock で排他、fsync で永続化)。

    同じ冪等キーが既にあれば何もせず False (内容の違う SessionStarted は SessionError)。
    追記後のログが fold できない (規則違反) なら SessionError で書かない。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with open(path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            data = f.read()
            events, complete = _parse_lines(data)
            existing = next((e for e in events if e.key() == event.key()), None)
            if existing is not None:
                _check_duplicate(existing, event)
                return False
            fold([*events, event])
            if complete < len(data):
                f.truncate(complete)  # 前回の書き込みが途中で止まった行を捨てる
            f.write((json.dumps(event_to_dict(event), ensure_ascii=False) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    if created:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return True


def load_state(path: Path) -> SessionState:
    return fold(read_events(path))
