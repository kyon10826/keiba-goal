"""netkeiba の結果ページ (+ 同じレースの shutuba_past) → record_data 形式の行 (schema.HISTORY_DTYPES)。

design §3.3 の規則。record_data (JV/TARGET) の流儀に合わせる:
- race_id 18 桁 = YYYYMMDD + 場 + 回 + 日 + R + 馬番、class_code は TARGET 式、track_code は JV。
- time_diff は最速馬との差 (秒)。最速馬は 2 番目に速い馬との差 (負、同着は -0.0)。タイムなしは欠損。
- タイムなし (取消・除外・中止) の time は 0、着順 0。上がり順位は min 順位 (無ければ 0、障害は欠損)。
- 通過順位は、そのレースで最も長い通過 (コーナー数) を 4 角までの枠にして左から当てる (5 角以上は最後の 4 つ)。
  中止馬の途中までの通過 '4-6' は 4 コーナーのレースなら [4, 6, 0, 0] (record_data と同じ)。
- 最後に schema.null_sentinels を通す (record の読み込みと同じ): 上の time 0・通過順 0・上がり順位 0 は欠損になる。
- Ave_3F / PCI / -3F差 は 0.1 秒単位の整数から厳密に計算 (Ave_3F は四捨五入、PCI は偶数丸め。障害は欠損)。
- 賞金は本賞金 (shutuba_past の「本賞金」。同着は按分して 1 万円未満切り捨て)。結果ページの賞金は付加賞を含むので予備。
- 父・母・ブリンカーは shutuba_past から。record_data に既にいる馬の父・母は馬マスタ (record 由来) を使う。
  それ以外は netkeiba の名前 (同名馬の II は JV に合わせて 2 に)。作り直し (rebuild.build) は最後に行の父・母を
  馬マスタの名前 (外国産の新馬は record から学んだ外国名) にそろえる。出馬表だけの経路 (convert_card、ライブ) は
  同じ決め方を ParentNames で行う (horse_master.parent_names で作る)。
- ブリンカーは出走取消の馬だけ欠損にする。出馬表だけの経路 (convert_card) も同じ規則
  (record_data は 2015 年以降、取消の行で 0/825 件、除外の行で 84/1,052 件 B を持つ。通常の行は 10%)。
- 増減は shutuba_past の「前計不」と、過去に出走 (取消・除外を除く) が無い馬を欠損にする (record_data と同じ)。
- leg (TARGET の脚質) は再現できないので欠損。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

import polars as pl

from keiba_goal.data.schema import (
    CLASS_ORD,
    CODE_TO_PLACE,
    HISTORY_DTYPES,
    NOT_STARTED_CODES,
    PLACE_CODES,
    TRACK_CODE,
    null_sentinels,
)
from keiba_goal.netkeiba.course_table import lookup as course_lookup
from keiba_goal.netkeiba.course_table import track_codes as course_track_codes
from keiba_goal.netkeiba.parse_result import ResultPage, ResultRace
from keiba_goal.netkeiba.parse_shutuba import CardPage, CardRace

CONVERTER_VERSION = "convert-3"

AGE_CODES = {"2歳": 11, "3歳": 12, "3歳以上": 13, "4歳以上": 14, "障害3歳以上": 18, "障害4歳以上": 19}
WEIGHT_CODES = {"ハンデ": 1, "別定": 2, "馬齢": 3, "定量": 4}
CLASS_WORDS = {
    "新馬": 15, "未出走": 11, "未勝利": 7,
    "400万下": 19, "500万下": 23, "1勝クラス": 23,
    "900万下": 39, "1000万下": 43, "2勝クラス": 43,
    "1500万下": 63, "1600万下": 67, "3勝クラス": 67,
}
GRADE_CODES = {1: 195, 2: 179, 3: 163}
LISTED_SINCE = 20190101           # リステッド競走 (115) は 2019 年から。それより前の (L) は OP (131)
GOING = {"良": "良", "稍重": "稍", "稍": "稍", "重": "重", "不良": "不", "不": "不"}
TURF_INOUT = {(): 0, ("外",): 2, ("内",): 0, ("内", "外"): 3, ("外", "内"): 4, ("内2周",): 5, ("外2周",): 6}
DIRT_INOUT = {(): 0, ("内",): 1, ("外",): 2}
CORNERS = ("corner1_rank", "corner2_rank", "corner3_rank", "corner4_rank")


class ConvertError(ValueError):
    pass


def _inout_tokens(tokens: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for t in tokens:
        out.extend(x for x in re.split(r"[-→]", t) if x)
    return tuple(out)


def flat_track_code(surface: str, detail: str) -> int:
    """芝ダ + 回り + 内外 → JV トラックコード (schema.TRACK_CODE の逆引き)。"""
    tokens = detail.split()
    if not tokens:
        raise ConvertError(f"回りが無い: {surface}{detail!r}")
    direction = {"右": 1, "左": 2, "直線": 3}.get(tokens[0])
    if direction is None:
        raise ConvertError(f"回りを解釈できない: {surface}{detail!r}")
    inout = (TURF_INOUT if surface == "芝" else DIRT_INOUT).get(_inout_tokens(tokens[1:]))
    if inout is None:
        raise ConvertError(f"内外を解釈できない: {surface}{detail!r}")
    surf = {"芝": 1, "ダ": 2}[surface]
    codes = [c for c, v in TRACK_CODE.items() if v == (surf, direction, inout) and c not in (27, 28)]
    if len(codes) != 1:
        raise ConvertError(f"トラックコードが決まらない: {surface}{detail!r} → {codes}")
    return codes[0]


def obstacle_track_code(detail: str, venue_code: int, dist: int, courses: pl.DataFrame) -> int:
    """障害: 「芝 ダート」= 52 (芝→ダート)、外/内の指定を見て、無ければ 54 (芝)。コース表で確かめる。"""
    tokens = _inout_tokens(detail.split())
    if "ダート" in tokens:
        guess = 52
    elif tokens[-2:] == ("外", "内"):
        guess = 56
    elif tokens[-2:] == ("内", "外"):
        guess = 57
    elif "外" in tokens:
        guess = 55
    else:
        guess = 54
    if course_lookup(courses, venue_code, guess, dist) is not None:
        return guess
    known = course_track_codes(courses, venue_code, dist, set(range(51, 60)))
    if len(known) == 1:
        return known[0]
    raise ConvertError(f"障害のトラックコードが決まらない: {detail!r} 場{venue_code} {dist}m 候補 {known}")


def class_code(conditions: str, grade: str | None, date: int) -> int:
    word = re.search(r"(新馬|未出走|未勝利|\d勝クラス|\d+万下|オープン)", conditions)
    if word is None:
        raise ConvertError(f"クラスを解釈できない: {conditions!r}")
    if word.group(1) != "オープン":
        if word.group(1) not in CLASS_WORDS:
            raise ConvertError(f"クラスコードの表に無いクラス: {word.group(1)!r} ({conditions!r})")
        return CLASS_WORDS[word.group(1)]
    g = (grade or "").replace("・", ".")
    m = re.fullmatch(r"(?:J\.?)?G(III|II|I|[123])", g)
    if m:
        return GRADE_CODES[{"I": 1, "II": 2, "III": 3}.get(m.group(1)) or int(m.group(1))]
    if g == "重賞":
        return 147
    if g == "L":
        return 115 if date >= LISTED_SINCE else 131
    return 131


def age_code(conditions: str) -> int:
    m = re.match(r"(障害)?(\d歳(?:以上)?)", conditions.removeprefix("サラ系"))
    key = (m.group(1) or "") + m.group(2) if m else None
    if key not in AGE_CODES:
        raise ConvertError(f"競走種別を解釈できない: {conditions!r}")
    return AGE_CODES[key]


def weight_code(conditions: str) -> int:
    found = [w for w in WEIGHT_CODES if f"({w})" in conditions]
    if len(found) != 1:
        raise ConvertError(f"重量種別を解釈できない: {conditions!r}")
    return WEIGHT_CODES[found[0]]


def _round_half_up(q: Fraction, digits: int) -> float:
    scaled = q * 10**digits
    return math.floor(scaled + Fraction(1, 2)) / 10**digits


def _round_half_even(q: Fraction, digits: int) -> float:
    # record_data の PCI はちょうど x.x5 のとき偶数側に丸めたものが 85.6% (2004-2025 の 3,620 件)。残りは再現できない
    return round(q * 10**digits) / 10**digits


def _tenths(x: float | None) -> int | None:
    return None if x is None else round(x * 10)


def passing_numbers(passing: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", passing)]


def corner_ranks(passing: str, n_corners: int) -> list[int]:
    """通過順位 → corner1-4。n_corners はそのレースの最も長い通過の長さ。

    i 番目 (1 始まり) の数字は 4 + i − n_corners 番の枠: '7-7' (2 コーナー) → [0, 0, 7, 7]、
    6 コーナーの '1-1-1-1-2-2' → [1, 1, 2, 2]、4 コーナーのレースで途中までの '4-6' → [4, 6, 0, 0]。
    """
    nums = passing_numbers(passing)
    if len(nums) > n_corners:
        raise ConvertError(f"通過 {passing!r} がレースのコーナー数 {n_corners} より長い")
    slots = [0, 0, 0, 0]
    for i, x in enumerate(nums, start=1):
        slot = 4 + i - n_corners
        if slot >= 1:
            slots[slot - 1] = x
    return slots


def _prize(rank: int, ranks: list[int], prizes: tuple[float, ...]) -> float:
    """本賞金 (万円)。同着は該当着順の賞金を合算して頭数で割り、1 万円未満を切り捨てる (record_data に端数は無い)。"""
    if rank <= 0:
        return 0.0
    k = ranks.count(rank)
    pos = [prizes[i] if i < len(prizes) else 0.0 for i in range(rank - 1, rank - 1 + k)]
    return float(math.floor(sum(pos) / k))


_ROMAN_SUFFIX = {"II": "2", "III": "3", "IV": "4", "V": "5"}


def jv_parent_name(name: str | None) -> str | None:
    """netkeiba の同名馬表記「ポールポジションII」→ JV の「ポールポジション2」(英字名には触らない)。"""
    if not name:
        return name
    m = re.fullmatch(r"(.*[^\x00-\x7f])(II|III|IV|V)", name)
    return m.group(1) + _ROMAN_SUFFIX[m.group(2)] if m else name


@dataclass(frozen=True)
class ParentNames:
    """出馬表の馬の父・母の決め方 (作り直しの馬マスタと同じ結果にする。horse_master.parent_names で作る)。

    known: 馬マスタの全馬 (出典を問わない) → (父, 母)。マスタにいない馬は出馬表の名前 (II → 2) で、外国産馬なら
    aliases ('father' / 'mother' ごとに カナ → record の外国名) で置き換える。
    """

    known: dict[int, tuple[str | None, str | None]]
    aliases: dict[str, dict[str, str]]

    @classmethod
    def card_only(cls) -> ParentNames:
        """馬マスタを使わず出馬表の名前のまま (変換器の単体テスト用)。"""
        return cls({}, {"father": {}, "mother": {}})

    def for_horse(self, horse_id: int, father: str | None, mother: str | None,
                  foreign_bred: bool | None) -> tuple[str | None, str | None]:
        if horse_id in self.known:
            return self.known[horse_id]
        f, m = jv_parent_name(father), jv_parent_name(mother)
        if foreign_bred:
            f, m = self.aliases["father"].get(f, f), self.aliases["mother"].get(m, m)
        return f, m


@dataclass(frozen=True)
class RaceCodes:
    venue_code: int
    track_code: int
    corner_num: int | None
    class_code: int
    age_code: int
    weight_code: int
    state: str
    weather: str


def _course(rid: str, surface: str, detail: str, dist: int, courses: pl.DataFrame) -> tuple[int, int | None]:
    venue_code = int(rid[4:6])
    if surface == "障":
        tc = obstacle_track_code(detail, venue_code, dist, courses)
    else:
        tc = flat_track_code(surface, detail)
    course = course_lookup(courses, venue_code, tc, dist)
    if course is None:
        raise ConvertError(f"{rid}: コース表に無い (場{venue_code}, track {tc}, {dist}m)")
    return tc, course["corner_num"]


def _state(rid: str, surface: str, turf: str | None, dirt: str | None) -> str:
    going = dirt if surface == "ダ" else turf   # 障害は芝の馬場
    if going not in GOING:
        raise ConvertError(f"{rid}: 馬場状態を解釈できない: 芝 {turf!r} ダ {dirt!r}")
    return GOING[going]


def result_codes(race: ResultRace, courses: pl.DataFrame) -> RaceCodes:
    rid = race.nk_race_id
    if PLACE_CODES.get(race.venue) != int(rid[4:6]):
        raise ConvertError(f"{rid}: 場 {race.venue} が race_id と合わない")
    if (race.times, race.daily) != (int(rid[6:8]), int(rid[8:10])):
        raise ConvertError(f"{rid}: 回・日目 {race.times}/{race.daily} が race_id と合わない")
    tc, corners = _course(rid, race.surface, race.course_detail, race.dist, courses)
    return RaceCodes(int(rid[4:6]), tc, corners, class_code(race.conditions, race.grade, race.date),
                     age_code(race.conditions), weight_code(race.conditions),
                     _state(rid, race.surface, race.goings.get("芝"), race.goings.get("ダート")), race.weather)


def card_codes(race: CardRace, date: int, courses: pl.DataFrame) -> RaceCodes:
    """出馬表だけから (レース前・ライブ用)。距離の後ろの A〜D (仮柵) は内外ではないので捨てる。"""
    rid = race.nk_race_id
    if PLACE_CODES.get(race.venue) != int(rid[4:6]) or (race.times, race.daily) != (int(rid[6:8]), int(rid[8:10])):
        raise ConvertError(f"{rid}: 場・回・日目が race_id と合わない")
    detail = " ".join(t for t in race.course_tokens if not re.fullmatch(r"[A-D]\+?", t))
    tc, corners = _course(rid, race.surface, detail, race.dist, courses)
    if race.weather is None:
        raise ConvertError(f"{rid}: 天気が未発表")
    conditions = f"{race.age_text}{race.class_text} ({race.weight_type})"
    return RaceCodes(int(rid[4:6]), tc, corners, class_code(conditions, race.grade, date), age_code(race.age_text),
                     WEIGHT_CODES[race.weight_type], _state(rid, race.surface, race.going_turf, race.going_dirt),
                     race.weather)


def _race_columns(rid: str, date: int, times: int, daily: int, c: RaceCodes, dist: int) -> dict:
    return {
        "year": date // 10000 % 100, "month": date // 100 % 100, "day": date % 100, "times": times,
        "place": CODE_TO_PLACE[c.venue_code], "daily": daily, "race_num": int(rid[10:12]),
        "class_code": c.class_code, "track_code": c.track_code, "corner_num": c.corner_num, "dist": dist,
        "state": c.state, "weather": c.weather, "age_code": c.age_code, "weight_code": c.weight_code,
        "date": date, "race_key": int(f"{date:08d}{rid[4:12]}"), "nk_race_id": rid, "venue_code": c.venue_code,
        "surface": TRACK_CODE[c.track_code][0], "direction": TRACK_CODE[c.track_code][1],
        "course_inout": TRACK_CODE[c.track_code][2], "class_ord": CLASS_ORD.get(c.class_code, -1),
        "source": "netkeiba",
    }


def convert_race(result: ResultPage, card: CardPage | None, courses: pl.DataFrame,
                 parents: dict[int, tuple[str | None, str | None]] | None = None,
                 first_seen: dict[int, int] | None = None) -> pl.DataFrame:
    """1 レース分を HISTORY_DTYPES の列・型・順で返す。

    parents: 馬 ID → (父, 母) を優先する表 (record_data 由来の馬マスタ)。
    first_seen: 馬 ID → 初めて JRA の記録に出た日 (shutuba_past が無いときの増減の欠損判定に使う)。
    """
    race, rows = result.race, result.rows
    rid = race.nk_race_id
    if card is not None:
        _check_card(result, card)
    codes = result_codes(race, courses)
    obstacle = codes.track_code >= 51
    base = _race_columns(rid, race.date, race.times, race.daily, codes, race.dist)

    recs = rows.sort("horse_num").to_dicts()
    T = {r["horse_num"]: _tenths(r["time"]) for r in recs}
    L = {r["horse_num"]: _tenths(r["last_3F_time"]) for r in recs}
    timed = sorted(t for t in T.values() if t)
    with_both = [T[h] - L[h] for h in T if T[h] and L[h] is not None]
    l3_sorted = sorted(v for v in L.values() if v is not None)
    ranks = [r["rank"] for r in recs if r["rank"] > 0]
    n_corners = max(len(passing_numbers(r["passing"])) for r in recs)
    card_h = {h["horse_num"]: h for h in card.horses.to_dicts()} if card is not None else {}
    starters = sum(r["error_code"] not in NOT_STARTED_CODES for r in recs)

    out = []
    for r in recs:
        h = r["horse_num"]
        t, l3 = T[h], L[h]
        ch = card_h.get(h)
        if ch is not None and ch["horse_id"] != r["horse_id"]:
            raise ConvertError(f"{rid}: 馬番 {h} の馬 ID が出馬表と結果で違う")
        if t:
            td = (timed[0] - timed[1] if len(timed) > 1 else None) if t == timed[0] else t - timed[0]
        else:
            td = None
        flat_both = not obstacle and bool(t) and l3 is not None
        ave = Fraction((t - l3) * 60, race.dist - 600) if flat_both else None
        father, mother = (jv_parent_name(ch["father"]), jv_parent_name(ch["mother"])) if ch is not None else (None, None)
        if parents and r["horse_id"] in parents:
            father, mother = parents[r["horse_id"]]
        out.append({
            **base,
            "race_id": int(f"{race.date:08d}{rid[4:12]}{h:02d}"),
            "horse": r["horse"],
            "jockey_id": r["jockey_id"],
            "horse_N": len(recs),
            "waku_num": r["waku_num"],
            "horse_num": h,
            "sex": r["sex"],
            "age": r["age"],
            "basis_weight": r["basis_weight"],
            "blinker": None if ch is None or r["error_code"] == 1 else ch["blinker"],
            "weight": r["weight"],
            "inc_dec": _inc_dec(r, ch, race.date, first_seen),
            "win_odds": r["win_odds"],
            "rank": r["rank"],
            "time_diff": None if td is None else td / 10,
            "time": (t or 0) / 10,
            **dict(zip(CORNERS, corner_ranks(r["passing"], n_corners))),
            "last_3F_time": None if l3 is None else l3 / 10,
            "last_3F_rank": None if obstacle else (l3_sorted.index(l3) + 1 if l3 is not None else 0),
            "Ave_3F": _round_half_up(ave, 2) if ave is not None else None,
            "PCI": _round_half_even(ave / Fraction(l3, 10) * 100 - 50, 1) if ave is not None else None,
            "last_3F_time_diff": ((t - l3) - min(with_both)) / 10 if flat_both else None,
            "leg": None,
            "pop": r["pop"],
            "prize": _prize(r["rank"], ranks, card.race.prizes) if card is not None else (r["prize"] or 0.0),
            "error_code": r["error_code"],
            "father": father,
            "mother": mother,
            "id": r["horse_id"],
            "starters": starters,
            "trainer_id": r["trainer_id"],
        })
    return null_sentinels(pl.DataFrame(out, schema=HISTORY_DTYPES).select(list(HISTORY_DTYPES)))


def convert_card(card: CardPage, date: int, courses: pl.DataFrame, names: ParentNames) -> pl.DataFrame:
    """出馬表 (shutuba_past) だけから発走前の列を作る (レース後の列は欠損)。ライブ (P7) とパリティ検査 (P3) 用。

    names は horse_master.parent_names(馬マスタ, card_horses) で作る。作り直し (rebuild.build) と同じ規則なので、
    後でそのレースを結果ページから変換した行と父・母が一致する (違いうるのは、マスタを作った後に出た出馬表で
    名前が変わった馬と、後の出馬表で別名の対応が増えた・曖昧になった外国産の新馬だけ)。
    """
    race = card.race
    rid = race.nk_race_id
    base = _race_columns(rid, date, race.times, race.daily, card_codes(race, date, courses), race.dist)
    out = []
    for h in card.horses.to_dicts():
        father, mother = names.for_horse(h["horse_id"], h["father"], h["mother"], h["foreign_bred"])
        no_prev = h["prev_weight_unknown"] or not h["n_past_starts"]
        out.append({
            **base,
            "race_id": int(f"{date:08d}{rid[4:12]}{h['horse_num']:02d}"),
            "horse": h["horse"], "jockey_id": h["jockey_id"], "horse_N": card.horses.height,
            "waku_num": h["waku_num"], "horse_num": h["horse_num"], "sex": h["sex"], "age": h["age"],
            "basis_weight": h["basis_weight"], "blinker": None if h["scratch"] == "取消" else h["blinker"],
            "weight": h["weight"], "inc_dec": None if no_prev else h["inc_dec"],
            "father": father, "mother": mother, "id": h["horse_id"], "trainer_id": h["trainer_id"],
        })
    return null_sentinels(pl.DataFrame(out, schema=HISTORY_DTYPES).select(list(HISTORY_DTYPES)))


def _inc_dec(r: dict, ch: dict | None, date: int, first_seen: dict[int, int] | None) -> float | None:
    if r["weight"] is None:
        return None
    if ch is not None:
        if ch["prev_weight_unknown"] or not ch["n_past_starts"]:
            return None
        return r["inc_dec"]
    if first_seen is not None and first_seen.get(r["horse_id"], date) >= date:
        return None
    return r["inc_dec"]


def _check_card(result: ResultPage, card: CardPage) -> None:
    rid = result.race.nk_race_id
    if card.race.nk_race_id != rid:
        raise ConvertError(f"{rid}: 出馬表が別レース ({card.race.nk_race_id})")
    if (card.race.surface, card.race.dist) != (result.race.surface, result.race.dist):
        raise ConvertError(f"{rid}: 出馬表と結果で距離・芝ダが違う")
    if card.horses.height != result.rows.height:
        raise ConvertError(f"{rid}: 出馬表 {card.horses.height} 頭と結果 {result.rows.height} 頭が合わない")
    if len(card.race.prizes) < 5:
        raise ConvertError(f"{rid}: 本賞金が 5 着分無い: {card.race.prizes}")
