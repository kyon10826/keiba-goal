"""race.netkeiba.com/api/api_get_jra_odds.html?type=1 (単勝・複勝オッズ JSON) の解析。

status: yoso = 予想オッズ (市場ではない) / middle = 発売中 / result = 確定。特徴量に使えるのは middle だけ
(design §11)。単勝は [odds, '0.0', 人気]、複勝は [lo, hi, 人気]。数値でない値は NaN / null。票数は無い。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from keiba_goal.netkeiba.common import ParseError

PARSER_VERSION = "odds-2"
STATUSES = ("yoso", "middle", "result")

WIN_SCHEMA = {"horse_num": pl.Int8, "win_odds": pl.Float32, "win_pop": pl.Int16}
PLACE_SCHEMA = {"horse_num": pl.Int8, "place_lo": pl.Float32, "place_hi": pl.Float32, "place_pop": pl.Int16}


@dataclass(frozen=True)
class OddsSnapshot:
    nk_race_id: str
    status: str
    official_datetime: datetime | None
    update_count: int | None
    win: pl.DataFrame
    place: pl.DataFrame

    @property
    def usable_for_features(self) -> bool:
        return self.status == "middle"


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) and x > 0 else math.nan


def _pop(v) -> int | None:
    try:
        x = int(str(v))
    except ValueError:
        return None
    return x if x > 0 else None


def _entries(block: object, what: str, nk_race_id: str) -> list[tuple[int, list]]:
    """{'01': [a, b, c], ...} → [(1, [a, b, c]), ...]。形が違えば ParseError。"""
    if not isinstance(block, dict):
        raise ParseError(f"{nk_race_id}: {what} が辞書でない: {type(block).__name__}")
    out = []
    for k, v in block.items():
        if not (isinstance(k, str) and re.fullmatch(r"[0-9]{1,2}", k)) or not isinstance(v, list) or len(v) < 3:
            raise ParseError(f"{nk_race_id}: {what} の要素を解釈できない: {k!r}: {v!r}")
        out.append((int(k), v))
    return out


def parse_odds(text: str, nk_race_id: str) -> OddsSnapshot:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"{nk_race_id}: オッズ API の応答が JSON でない") from e
    if not isinstance(payload, dict):
        raise ParseError(f"{nk_race_id}: オッズ API の応答がオブジェクトでない")
    status = payload.get("status")
    if status not in STATUSES:
        raise ParseError(f"{nk_race_id}: 未知の status {status!r} (reason={payload.get('reason')!r})")
    data = payload.get("data")
    odds = data.get("odds") if isinstance(data, dict) else None
    if not isinstance(odds, dict) or not isinstance(odds.get("1"), dict) or not odds["1"]:
        raise ParseError(f"{nk_race_id}: 単勝オッズ (odds['1']) が無い (status={status})")
    win = [{"horse_num": k, "win_odds": _num(v[0]), "win_pop": _pop(v[2])}
           for k, v in _entries(odds["1"], "単勝", nk_race_id)]
    place = [{"horse_num": k, "place_lo": _num(v[0]), "place_hi": _num(v[1]), "place_pop": _pop(v[2])}
             for k, v in _entries(odds.get("2") or {}, "複勝", nk_race_id)]
    raw_dt = data.get("official_datetime")
    try:
        official = datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S") if raw_dt else None
    except ValueError as e:
        raise ParseError(f"{nk_race_id}: official_datetime を解釈できない: {raw_dt!r}") from e
    uc = str(payload.get("update_count", ""))
    return OddsSnapshot(
        nk_race_id=nk_race_id,
        status=status,
        official_datetime=official,
        update_count=int(uc) if uc.isdigit() else None,
        win=pl.DataFrame(win, schema=WIN_SCHEMA).sort("horse_num"),
        place=pl.DataFrame(place, schema=PLACE_SCHEMA).sort("horse_num"),
    )
