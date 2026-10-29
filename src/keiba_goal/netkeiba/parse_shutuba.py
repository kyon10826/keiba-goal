"""race.netkeiba.com の出馬表の解析。shutuba_past (5 走表示) が正本、shutuba は予備。

RaceData01: 発走時刻・芝ダ障・距離・回り・内外・天気 (Icon_Weather の title)・馬場 (span.Item03 = 芝, span.Item04 = ダート)。
RaceData02: 回・場・日目・競走種別・クラス・記号・重量種別・頭数・本賞金。
馬ごと: 枠・馬番・馬 ID・馬名・性齢 (毛色)・斤量・騎手 ID・調教師 ID・馬体重/増減・ブリンカー (span.Mark 'B')・父・母・母父・直近 5 走。
取消・除外でない馬に父・母・騎手・調教師・斤量が無ければ ParseError (欠損で通すと履歴に null が黙って入る)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

from keiba_goal.data.schema import PLACE_CODES
from keiba_goal.netkeiba.common import ParseError, first, has_class, nfkc, norm, parse_doc, text_of, to_float, to_int

PARSER_VERSION = "shutuba-2"

# RaceName の Icon_GradeType 番号 → 格 (13 は WIN5 の印なので無視)
GRADE_ICONS = {
    "1": "G1", "2": "G2", "3": "G3", "4": "重賞", "5": "OP", "10": "J.G1", "11": "J.G2", "12": "J.G3",
    "15": "L", "16": "3勝", "17": "2勝", "18": "1勝",
}
WEIGHT_TYPES = ("定量", "馬齢", "別定", "ハンデ")
_POST = re.compile(r"(\d{1,2}):(\d{2})発走")
_COURSE = re.compile(r"^(芝|ダ|障)(\d{3,4})m$")
_WEIGHT = re.compile(r"^(\d{3,4})kg\((?:([+-]?\d+)|(前計不))\)$")
_HORSE_ID = re.compile(r"/horse/(\d{10})")
_PERSON_ID = re.compile(r"/(?:jockey|trainer)/(?:result/recent/)?([0-9a-zA-Z]+)/?")
_RACE_ID = re.compile(r"/race/([0-9A-Za-z]{12})")  # 海外は 2022F5a00308 のような英数字
_CLASS = re.compile(r"(新馬|未勝利|未出走|\d勝クラス|\d+万下|オープン)")
NOT_STARTED_MARKS = ("取", "除")
SCRATCH_KINDS = ("取消", "除外")
NO_WEIGHT = "999"   # 出馬表の「999kg」は計不 (除外馬などで出る)
REQUIRED_PAST = ("father", "mother", "jockey_id", "trainer_id", "basis_weight")
REQUIRED_SHUTUBA = ("jockey_id", "trainer_id", "basis_weight")
FOREIGN_MARKS = ("Icon_MaruGai", "Icon_KakuGai")   # マル外 (外国産馬) / カク外 (外国馬)


@dataclass(frozen=True)
class CardRace:
    nk_race_id: str
    race_name: str
    grade: str | None
    post_time: str               # HH:MM
    surface: str                 # 芝 / ダ / 障
    dist: int
    course_tokens: tuple[str, ...]  # 距離の後ろ: ('右', '外', 'C')、障害は ('芝', 'ダート')
    weather: str | None          # 発表前は None
    going_turf: str | None
    going_dirt: str | None
    times: int
    venue: str
    daily: int
    age_text: str                # '3歳以上'、'障害4歳以上'
    class_text: str              # '1勝クラス'、'オープン'
    symbols: str                 # '(国際)(特指)'
    weight_type: str             # 定量 / 馬齢 / 別定 / ハンデ
    n_entries: int
    prizes: tuple[float, ...]    # 本賞金 1〜5 着 (万円)


@dataclass(frozen=True)
class CardPage:
    race: CardRace
    horses: pl.DataFrame
    past_runs: pl.DataFrame
    source: str                  # 'shutuba_past' | 'shutuba'


HORSE_SCHEMA: dict[str, pl.DataType] = {
    "waku_num": pl.Int8, "horse_num": pl.Int8, "horse_id": pl.Int64, "horse": pl.Utf8, "scratched": pl.Boolean,
    "scratch": pl.Utf8,
    "sex": pl.Utf8, "age": pl.Int8, "coat": pl.Utf8, "basis_weight": pl.Float32, "jockey_id": pl.Utf8,
    "trainer_id": pl.Utf8, "trainer_area": pl.Utf8, "weight": pl.Float32, "inc_dec": pl.Float32,
    "prev_weight_unknown": pl.Boolean, "blinker": pl.Utf8, "father": pl.Utf8, "mother": pl.Utf8,
    "broodmare_sire": pl.Utf8, "n_past_runs": pl.Int8, "n_past_starts": pl.Int8, "foreign_bred": pl.Boolean,
}
PAST_SCHEMA: dict[str, pl.DataType] = {
    "horse_id": pl.Int64, "k": pl.Int8, "past_nk_race_id": pl.Utf8, "date": pl.Int32, "venue": pl.Utf8,
    "rank_text": pl.Utf8,
}


def parse_shutuba_past(text: str, nk_race_id: str) -> CardPage:
    doc = parse_doc(text)
    race = _parse_race(doc, nk_race_id)
    table = first(doc, '//table[@id="sort_table"]')
    if table is None:
        raise ParseError(f"{nk_race_id}: 5 走表示の表 (sort_table) が無い")
    horses, past = [], []
    for tr in table.xpath(f'.//tr[{has_class("HorseList")}]'):
        h, runs = _past_row(tr, nk_race_id)
        horses.append(h)
        past.extend(runs)
    return CardPage(race, _horse_frame(horses, nk_race_id), pl.DataFrame(past, schema=PAST_SCHEMA), "shutuba_past")


def parse_shutuba(text: str, nk_race_id: str) -> CardPage:
    doc = parse_doc(text)
    race = _parse_race(doc, nk_race_id)
    rows = doc.xpath(f'//table[{has_class("ShutubaTable")}]//tr[{has_class("HorseList")}]')
    horses = [_shutuba_row(tr, nk_race_id) for tr in rows]
    return CardPage(race, _horse_frame(horses, nk_race_id), pl.DataFrame([], schema=PAST_SCHEMA), "shutuba")


def _parse_race(doc, nk_race_id: str) -> CardRace:
    name_el = first(doc, f'//h1[{has_class("RaceName")}]')
    d1 = first(doc, f'//div[{has_class("RaceData01")}]')
    d2 = first(doc, f'//div[{has_class("RaceData02")}]')
    if name_el is None or d1 is None or d2 is None:
        raise ParseError(f"{nk_race_id}: RaceName / RaceData01 / RaceData02 が無い")
    grades = [GRADE_ICONS[c.removeprefix("Icon_GradeType")]
              for span in name_el.xpath(f'.//span[{has_class("Icon_GradeType")}]')
              for c in (span.get("class") or "").split()
              if c.startswith("Icon_GradeType") and c.removeprefix("Icon_GradeType") in GRADE_ICONS]
    d1_text = text_of(d1)
    post = _POST.search(d1_text)
    course_span = next((norm(s.text_content()) for s in d1.xpath("./span")
                        if _COURSE.match(norm(s.text_content()))), None)
    if post is None or course_span is None:
        raise ParseError(f"{nk_race_id}: RaceData01 の発走時刻・距離を解釈できない: {d1_text!r}")
    cm = _COURSE.match(course_span)
    after = d1_text.split(course_span, 1)[1].split("/", 1)[0]
    weather = first(d1, f'.//span[{has_class("Icon_Weather")}]')
    turf = first(d1, f'.//span[{has_class("Item03")}]')
    dirt = first(d1, f'.//span[{has_class("Item04")}]')
    spans = [nfkc(s.text_content()) for s in d2.xpath("./span")]
    joined = " ".join(spans)
    kai = re.search(r"(\d+)回", joined)
    day = re.search(r"(\d+)日目", joined)
    venue = next((s for s in spans if s in PLACE_CODES), None)
    age = next((s for s in spans if re.fullmatch(r"(サラ系|障害|アラ系)?\d歳(以上)?", s)), None)
    cls = next((s for s in spans if _CLASS.fullmatch(s)), None)
    wtype = next((s for s in spans if s in WEIGHT_TYPES), None)
    heads = re.search(r"(\d+)頭", joined)
    prize = re.search(r"本賞金:([\d,.]+)万円", joined)
    if not all((kai, day, venue, age, cls, wtype, heads, prize)):
        raise ParseError(f"{nk_race_id}: RaceData02 を解釈できない: {joined!r}")
    return CardRace(
        nk_race_id=nk_race_id,
        race_name=text_of(name_el),
        grade=grades[0] if grades else None,
        post_time=f"{int(post.group(1)):02d}:{post.group(2)}",
        surface=cm.group(1),
        dist=int(cm.group(2)),
        course_tokens=tuple(norm(after).split()),
        weather=(weather.get("title") or None) if weather is not None else None,
        going_turf=text_of(turf) or None if turf is not None else None,
        going_dirt=text_of(dirt) or None if dirt is not None else None,
        times=int(kai.group(1)),
        venue=venue,
        daily=int(day.group(1)),
        age_text=age.removeprefix("サラ系"),
        class_text=cls,
        symbols="".join(s for s in spans if s.startswith(("(", "["))),
        weight_type=wtype,
        n_entries=int(heads.group(1)),
        prizes=tuple(to_float(x, f"{nk_race_id}: 本賞金") for x in prize.group(1).split(",")),
    )


def _id(el, pattern: re.Pattern) -> str | None:
    if el is None:
        return None
    for href in el.xpath(".//a/@href"):
        m = pattern.search(href)
        if m:
            return m.group(1)
    return None


def _foreign(el) -> bool:
    return any(m in (c or "") for c in el.xpath('.//span[contains(@class, "Icon_HorseMark")]/@class')
               for m in FOREIGN_MARKS)


def _weight(text: str, nk_race_id: str) -> tuple[float | None, float | None, bool]:
    t = text.replace(" ", "")
    if not t:
        return None, None, False
    m = _WEIGHT.match(t)
    if not m:
        raise ParseError(f"{nk_race_id}: 馬体重を解釈できない: {text!r}")
    if m.group(1) == NO_WEIGHT:
        return None, None, False
    if m.group(3):
        return float(m.group(1)), None, True
    return float(m.group(1)), float(m.group(2)), False


def _sex_age(text: str, nk_race_id: str) -> tuple[str, int, str]:
    m = re.fullmatch(r"(牡|牝|セ)(\d+)(\S*)", text.replace(" ", ""))
    if not m:
        raise ParseError(f"{nk_race_id}: 性齢を解釈できない: {text!r}")
    return m.group(1), int(m.group(2)), m.group(3)


def _int_cell(tr, xpath: str, what: str, nk_race_id: str) -> int:
    return to_int(text_of(first(tr, xpath)), f"{nk_race_id}: {what}")


def _scratch_kind(tr) -> str | None:
    """取消・除外の馬の行は class に Cancel があり、Cancel_NoData のセルに「取消」「除外」と出る。"""
    t = text_of(first(tr, './td[contains(@class, "Cancel")]'))
    return t if t in SCRATCH_KINDS else None


def _require(h: dict, fields: tuple[str, ...], nk_race_id: str) -> dict:
    missing = [f for f in fields if h[f] is None]
    if missing and not h["scratched"]:
        raise ParseError(f"{nk_race_id}: 馬番 {h['horse_num']} ({h['horse']}) に {missing} が無い")
    return h


def _past_row(tr, nk_race_id: str) -> tuple[dict, list[dict]]:
    info = first(tr, f'./td[{has_class("Horse_Info")}]')
    jockey = first(tr, f'./td[{has_class("Jockey")}]')
    if info is None or jockey is None:
        raise ParseError(f"{nk_race_id}: 馬の行に Horse_Info / Jockey が無い")
    name_div = first(info, f'.//div[{has_class("Horse02")}]')
    horse_id = _id(name_div, _HORSE_ID)
    if horse_id is None:
        raise ParseError(f"{nk_race_id}: 馬 ID が無い")
    sex, age, coat = _sex_age(text_of(first(jockey, f'.//span[{has_class("Barei")}]')), nk_race_id)
    kg = [norm(s) for s in jockey.xpath("./span[not(@class)]/text()") if norm(s)]
    weight_div = first(info, f'.//div[{has_class("Weight")}]')
    weight, inc_dec, prev_unknown = _weight(text_of(weight_div), nk_race_id)
    bms = text_of(first(info, f'.//div[{has_class("Horse04")}]'))
    trainer = first(info, f'.//div[{has_class("Horse05")}]')
    runs = []
    for k, td in enumerate(tr.xpath(f'./td[{has_class("Past")}]'), start=1):
        link = _id(first(td, f'.//div[{has_class("Data02")}]'), _RACE_ID)
        head = [norm(x) for x in td.xpath(f'.//div[{has_class("Data01")}]//text()') if norm(x)]
        dm = re.match(r"(\d{4})\.(\d{2})\.(\d{2})\s*(\S*)", head[0]) if head else None
        if link is None or dm is None:
            raise ParseError(f"{nk_race_id}: 馬 {horse_id} の {k} 走前を解釈できない")
        runs.append({"horse_id": int(horse_id), "k": k, "past_nk_race_id": link,
                     "date": int(dm.group(1) + dm.group(2) + dm.group(3)), "venue": dm.group(4) or None,
                     "rank_text": text_of(first(td, f'.//span[{has_class("Num")}]')) or None})
    h = {
        "waku_num": _int_cell(tr, "./td[1]", "枠番", nk_race_id),
        "horse_num": _int_cell(tr, "./td[2]", "馬番", nk_race_id),
        "horse_id": int(horse_id),
        "horse": norm("".join(name_div.xpath("./a//text()"))),
        "scratched": "Cancel" in (tr.get("class") or "").split(),
        "scratch": _scratch_kind(tr),
        "sex": sex, "age": age, "coat": coat or None,
        "basis_weight": to_float(kg[-1], f"{nk_race_id}: 斤量") if kg else None,
        "jockey_id": _id(jockey, _PERSON_ID),
        "trainer_id": _id(trainer, _PERSON_ID),
        "trainer_area": text_of(trainer).split("・")[0] or None if trainer is not None else None,
        "weight": weight, "inc_dec": inc_dec, "prev_weight_unknown": prev_unknown,
        "blinker": "B" if "B" in [norm(x) for x in name_div.xpath(f'.//span[{has_class("Mark")}]/text()')] else None,
        "father": text_of(first(info, f'.//div[{has_class("Horse01")}]')) or None,
        "mother": text_of(first(info, f'.//div[{has_class("Horse03")}]')) or None,
        "broodmare_sire": bms.strip("()") or None,
        "n_past_runs": len(runs),
        # 取消・除外は出走していない (record_data の増減は前回「出走」時の体重との差)
        "n_past_starts": sum(r["rank_text"] not in NOT_STARTED_MARKS for r in runs),
        "foreign_bred": _foreign(name_div),
    }
    return _require(h, REQUIRED_PAST, nk_race_id), runs


def _shutuba_row(tr, nk_race_id: str) -> dict:
    info = first(tr, f'./td[{has_class("HorseInfo")}]')
    horse_id = _id(info, _HORSE_ID)
    if horse_id is None:
        raise ParseError(f"{nk_race_id}: 馬 ID が無い")
    sex, age, coat = _sex_age(text_of(first(tr, f'./td[{has_class("Barei")}]')), nk_race_id)
    kg = text_of(first(tr, f'./td[{has_class("Barei")}]/following-sibling::td[1]'))
    wtxt = text_of(first(tr, f'./td[{has_class("Weight")}]')).replace(" ", "")
    weight, inc_dec, prev_unknown = _weight(re.sub(r"^(\d{3,4})\(", r"\1kg(", wtxt), nk_race_id)
    trainer = first(tr, f'./td[{has_class("Trainer")}]')
    return _require({
        "waku_num": _int_cell(tr, f'./td[starts-with(@class, "Waku")]', "枠番", nk_race_id),
        "horse_num": _int_cell(tr, f'./td[starts-with(@class, "Umaban")]', "馬番", nk_race_id),
        "horse_id": int(horse_id),
        "horse": text_of(first(info, ".//a")),
        "scratched": "Cancel" in (tr.get("class") or "").split(),
        "scratch": _scratch_kind(tr),
        "sex": sex, "age": age, "coat": coat or None,
        "basis_weight": to_float(kg, f"{nk_race_id}: 斤量") if kg else None,
        "jockey_id": _id(first(tr, f'./td[{has_class("Jockey")}]'), _PERSON_ID),
        "trainer_id": _id(trainer, _PERSON_ID),
        "trainer_area": text_of(first(trainer, f'.//span[{has_class("Label1")}]')) or None if trainer is not None else None,
        "weight": weight, "inc_dec": inc_dec, "prev_weight_unknown": prev_unknown,
        "blinker": None, "father": None, "mother": None, "broodmare_sire": None, "n_past_runs": None,
        "n_past_starts": None, "foreign_bred": _foreign(info),
    }, REQUIRED_SHUTUBA, nk_race_id)


def _horse_frame(horses: list[dict], nk_race_id: str) -> pl.DataFrame:
    if not horses:
        raise ParseError(f"{nk_race_id}: 出走馬の行が無い")
    df = pl.DataFrame(horses, schema=HORSE_SCHEMA)
    if df["horse_num"].n_unique() != df.height:
        raise ParseError(f"{nk_race_id}: 馬番が重複")
    return df.sort("horse_num")
