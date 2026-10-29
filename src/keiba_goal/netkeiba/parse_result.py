"""db.netkeiba.com/race/{12 桁}/ (結果ページ, EUC-JP) の解析。

- 25 列中 12 列を包む `<diary_snap_cut>` タグを先に取り除き、列はヘッダの文字列で対応付ける。
- ヘッダと全行が 25 セルであること、必須列がそろうことを検査し、欠けたら ParseError (0 や '' で埋めない)。
- 払戻表 (pay_table_01) は全券種。同着は 1 券種に複数行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

from keiba_goal.netkeiba.common import ParseError, first, nfkc, norm, parse_doc, text_of, to_int

PARSER_VERSION = "result-2"
N_CELLS = 25
REQUIRED_COLUMNS = ("着順", "馬番", "タイム", "着差", "通過", "上り", "単勝", "人気", "馬体重", "調教師", "賞金(万円)")
OTHER_COLUMNS = ("枠番", "馬名", "性齢", "斤量", "騎手")

# 着順欄の文字 → JV 異常区分。数字の後ろの (降) は降着 (着順は降着後の値)。
RANK_TOKENS = {"取": 1, "除": 3, "中": 4, "失": 5}
BET_TYPES = {
    "tan": "win", "fuku": "place", "waku": "bracket_quinella", "uren": "quinella",
    "wide": "wide", "utan": "exacta", "sanfuku": "trio", "santan": "trifecta",
}

_DIARY = re.compile(r"</?diary_snap_cut[^>]*>", re.I)
_COURSE = re.compile(r"^(芝|ダ|障)(.*?)(\d{3,4})m$")
_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_KAI = re.compile(r"(\d+)回(\S+?)(\d+)日目")
_POST = re.compile(r"発走\s*:\s*(\d{1,2}):(\d{2})")
_TIME = re.compile(r"^(?:(\d+):)?(\d+)\.(\d)$")
_WEIGHT = re.compile(r"^(\d{3,4})\(([+-]?\d+)\)$")
_ID = {
    "horse": re.compile(r"/horse/(\d{10})"),
    "jockey": re.compile(r"/jockey/(?:result/recent/)?([0-9a-zA-Z]+)/?"),
    "trainer": re.compile(r"/trainer/(?:result/recent/)?([0-9a-zA-Z]+)/?"),
}

ROW_SCHEMA: dict[str, pl.DataType] = {
    "rank_text": pl.Utf8, "rank": pl.Int16, "error_code": pl.Int8, "waku_num": pl.Int8, "horse_num": pl.Int8,
    "horse": pl.Utf8, "horse_id": pl.Int64, "sex": pl.Utf8, "age": pl.Int8, "basis_weight": pl.Float32,
    "jockey_id": pl.Utf8, "time": pl.Float64, "margin": pl.Utf8, "passing": pl.Utf8, "last_3F_time": pl.Float64,
    "win_odds": pl.Float32, "pop": pl.Int16, "weight": pl.Float32, "inc_dec": pl.Float32, "weight_text": pl.Utf8,
    "trainer_id": pl.Utf8, "trainer_area": pl.Utf8, "prize": pl.Float32,
}


@dataclass(frozen=True)
class ResultRace:
    nk_race_id: str
    date: int                 # YYYYMMDD
    venue: str
    times: int                # 回
    daily: int                # 日目
    race_num: int
    race_name: str
    grade: str | None         # タイトル末尾の (GI) (L) (OP) (2勝) などの中身
    surface: str              # 芝 / ダ / 障
    course_detail: str        # 「芝右 外1600m」の「右 外」、「障芝 ダート3200m」の「芝 ダート」
    dist: int
    weather: str
    goings: dict[str, str]    # {'芝': '良', 'ダート': '稍重'}
    post_time: str            # HH:MM
    conditions: str           # 「4歳以上2勝クラス (混)[指](定量)」


@dataclass(frozen=True)
class ResultPage:
    race: ResultRace
    rows: pl.DataFrame
    payouts: pl.DataFrame

    @property
    def starters(self) -> int:
        return int(self.rows.filter(~pl.col("error_code").is_in([1, 2, 3])).height)

    def completeness(self) -> str | None:
        """完全なら None、そうでなければ理由 (manifest の error に入れる)。"""
        kinds = set(self.payouts["bet_type"].to_list())
        if "win" not in kinds:
            return "単勝の払戻が無い"
        if self.starters > 4 and "place" not in kinds:
            return "複勝の払戻が無い"
        return None


def strip_diary_tags(text: str) -> str:
    return _DIARY.sub("", text)


def parse_result(text: str, nk_race_id: str) -> ResultPage:
    doc = parse_doc(strip_diary_tags(text))
    race = _parse_race(doc, nk_race_id)
    rows = _parse_rows(doc, nk_race_id)
    payouts = _parse_payouts(doc, nk_race_id)
    return ResultPage(race, rows, payouts)


def _parse_race(doc, nk_race_id: str) -> ResultRace:
    intro = first(doc, '//div[contains(@class,"data_intro")]')
    if intro is None:
        raise ParseError(f"{nk_race_id}: data_intro が無い")
    title = text_of(first(intro, ".//h1"))
    if not title:
        raise ParseError(f"{nk_race_id}: レース名が無い")
    head = text_of(first(intro, ".//p/span"))
    parts = [norm(p) for p in head.split("/")]
    m = _COURSE.match(parts[0]) if parts else None
    if not m:
        raise ParseError(f"{nk_race_id}: コース表記を解釈できない: {head!r}")
    weather, goings, post = None, {}, None
    for p in parts[1:]:
        key, _, val = (norm(x) for x in p.partition(":"))
        if key == "天候":
            weather = val
        elif key == "発走":
            post = _POST.search(p)
        elif key in ("芝", "ダート"):
            goings[key] = val
        else:
            # 障害の「芝 : 良 ダート : 稍重」のように 1 区切りに 2 つ並ぶ場合
            for k2, v2 in re.findall(r"(芝|ダート)\s*:\s*(\S+)", p):
                goings[k2] = v2
    if not weather or not goings or post is None:
        raise ParseError(f"{nk_race_id}: 天候・馬場・発走時刻のどれかが無い: {head!r}")
    small = text_of(first(doc, '//p[contains(@class,"smalltxt")]'))
    dm, km = _DATE.search(small), _KAI.search(small)
    if not dm or not km:
        raise ParseError(f"{nk_race_id}: 日付・開催が無い: {small!r}")
    conditions = norm(small[km.end():])
    if not conditions:
        raise ParseError(f"{nk_race_id}: 競走条件が無い: {small!r}")
    gm = re.search(r"\(([^()]*)\)\s*$", title)
    return ResultRace(
        nk_race_id=nk_race_id,
        date=int(dm.group(1)) * 10000 + int(dm.group(2)) * 100 + int(dm.group(3)),
        venue=km.group(2),
        times=int(km.group(1)),
        daily=int(km.group(3)),
        race_num=int(nk_race_id[10:12]),
        race_name=title,
        grade=nfkc(gm.group(1)) if gm else None,
        surface=m.group(1),
        course_detail=norm(m.group(2)),
        dist=int(m.group(3)),
        weather=weather,
        goings=goings,
        post_time=f"{int(post.group(1)):02d}:{post.group(2)}",
        conditions=conditions,
    )


def _cells(tr) -> list:
    return tr.xpath("./td|./th")


def _header_index(header_cells: list, nk_race_id: str) -> dict[str, int]:
    names = [re.sub(r"\s+", "", c.text_content()) for c in header_cells]
    idx: dict[str, int] = {}
    for want in REQUIRED_COLUMNS + OTHER_COLUMNS + ("馬主",):
        hits = [i for i, n in enumerate(names) if n == want]
        if len(hits) != 1:
            raise ParseError(f"{nk_race_id}: ヘッダに {want} が {len(hits)} 個: {names}")
        idx[want] = hits[0]
    return idx


def _link_id(cell, kind: str) -> str | None:
    for href in cell.xpath(".//a/@href"):
        m = _ID[kind].search(href)
        if m:
            return m.group(1)
    return None


def parse_time(text: str) -> float | None:
    """'1:11.5' → 71.5、'59.8' → 59.8、'' → None。"""
    if not text:
        return None
    m = _TIME.match(text)
    if not m:
        raise ParseError(f"タイムを解釈できない: {text!r}")
    return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 10


def parse_rank(text: str) -> tuple[int, int]:
    """着順欄 → (着順, 異常区分)。取消・除外・中止・失格は着順 0、降着は降着後の着順で区分 7。"""
    t = text.replace(" ", "")
    m = re.fullmatch(r"(\d+)(\(降\))?", t)
    if m:
        return int(m.group(1)), 7 if m.group(2) else 0
    if t in RANK_TOKENS:
        return 0, RANK_TOKENS[t]
    raise ParseError(f"着順欄を解釈できない: {text!r}")


def parse_body_weight(text: str) -> tuple[float | None, float | None]:
    """'496(0)' → (496, 0)、'計不' や '' → (None, None)。"""
    t = text.replace(" ", "")
    if t in ("", "計不"):
        return None, None
    m = _WEIGHT.match(t)
    if m:
        return float(m.group(1)), float(m.group(2))
    if re.fullmatch(r"\d{3,4}", t):
        return float(t), None
    raise ParseError(f"馬体重を解釈できない: {text!r}")


def _float_or_none(text: str, what: str) -> float | None:
    t = text.replace(",", "")
    if t in ("", "---", "----", "**"):
        return None
    try:
        return float(t)
    except ValueError as e:
        raise ParseError(f"{what} を数値にできない: {text!r}") from e


def _parse_rows(doc, nk_race_id: str) -> pl.DataFrame:
    table = first(doc, '//table[contains(@class,"race_table_01")]')
    if table is None:
        raise ParseError(f"{nk_race_id}: 結果表 (race_table_01) が無い")
    trs = table.xpath("./tr|./tbody/tr")
    if len(trs) < 2:
        raise ParseError(f"{nk_race_id}: 結果表に行が無い")
    header = _cells(trs[0])
    if len(header) != N_CELLS:
        raise ParseError(f"{nk_race_id}: ヘッダが {len(header)} セル (期待 {N_CELLS})")
    ix = _header_index(header, nk_race_id)
    recs = []
    for n, tr in enumerate(trs[1:], start=1):
        cells = _cells(tr)
        if len(cells) != N_CELLS:
            raise ParseError(f"{nk_race_id}: {n} 行目が {len(cells)} セル (期待 {N_CELLS})")
        txt = [text_of(c) for c in cells]
        rank, ec = parse_rank(txt[ix["着順"]])
        horse_id = _link_id(cells[ix["馬名"]], "horse")
        jockey_id = _link_id(cells[ix["騎手"]], "jockey")
        trainer_id = _link_id(cells[ix["調教師"]], "trainer")
        if horse_id is None or jockey_id is None or trainer_id is None:
            raise ParseError(f"{nk_race_id}: {n} 行目に馬・騎手・調教師の ID が無い")
        sex_age = txt[ix["性齢"]]
        sm = re.fullmatch(r"(牡|牝|セ)(\d+)", sex_age)
        if not sm:
            raise ParseError(f"{nk_race_id}: 性齢を解釈できない: {sex_age!r}")
        weight, inc_dec = parse_body_weight(txt[ix["馬体重"]])
        pop = txt[ix["人気"]]
        area = re.match(r"\[(\S)\]", txt[ix["調教師"]])
        recs.append({
            "rank_text": txt[ix["着順"]],
            "rank": rank,
            "error_code": ec,
            "waku_num": to_int(txt[ix["枠番"]], f"{nk_race_id}: {n} 行目の枠番"),
            "horse_num": to_int(txt[ix["馬番"]], f"{nk_race_id}: {n} 行目の馬番"),
            "horse": txt[ix["馬名"]],
            "horse_id": int(horse_id),
            "sex": sm.group(1),
            "age": int(sm.group(2)),
            "basis_weight": _float_or_none(txt[ix["斤量"]], "斤量"),
            "jockey_id": jockey_id,
            "time": parse_time(txt[ix["タイム"]]),
            "margin": txt[ix["着差"]],
            "passing": txt[ix["通過"]],
            "last_3F_time": _float_or_none(txt[ix["上り"]], "上り"),
            "win_odds": _float_or_none(txt[ix["単勝"]], "単勝"),
            "pop": to_int(pop, "人気") if pop not in ("", "**", "---") else None,
            "weight": weight,
            "inc_dec": inc_dec,
            "weight_text": txt[ix["馬体重"]],
            "trainer_id": trainer_id,
            "trainer_area": area.group(1) if area else None,
            "prize": _float_or_none(txt[ix["賞金(万円)"]], "賞金"),
        })
    df = pl.DataFrame(recs, schema=ROW_SCHEMA)
    if df["horse_num"].n_unique() != df.height:
        raise ParseError(f"{nk_race_id}: 馬番が重複")
    if df["basis_weight"].null_count():
        raise ParseError(f"{nk_race_id}: 斤量が無い行がある")
    return df


PAYOUT_SCHEMA: dict[str, pl.DataType] = {
    "bet_type": pl.Utf8, "combo": pl.Utf8, "numbers": pl.List(pl.Int8), "payout_per_100": pl.Int32,
    "popularity": pl.Int16,
}


def _lines(cell) -> list[str]:
    return [norm(x) for x in cell.itertext() if norm(x)]


def _parse_payouts(doc, nk_race_id: str) -> pl.DataFrame:
    tables = doc.xpath('//table[contains(@class,"pay_table_01")]')
    if not tables:
        raise ParseError(f"{nk_race_id}: 払戻表 (pay_table_01) が無い")
    recs = []
    for table in tables:
        for tr in table.xpath(".//tr"):
            th = first(tr, "./th")
            tds = tr.xpath("./td")
            if th is None or len(tds) != 3:
                raise ParseError(f"{nk_race_id}: 払戻表の行の形が違う")
            cls = (th.get("class") or "").split()
            kind = next((BET_TYPES[c] for c in cls if c in BET_TYPES), None)
            if kind is None:
                raise ParseError(f"{nk_race_id}: 未知の券種 {cls} {text_of(th)!r}")
            combos, pays, pops = (_lines(td) for td in tds)
            if not (len(combos) == len(pays) == len(pops)) or not combos:
                raise ParseError(f"{nk_race_id}: {kind} の組と払戻の数が合わない: {combos} {pays} {pops}")
            for combo, pay, pop in zip(combos, pays, pops):
                nums = [to_int(x, f"{nk_race_id}: {kind} の組 {combo!r}") for x in combo.replace("→", "-").split("-")]
                recs.append({
                    "bet_type": kind,
                    "combo": "-".join(str(x) for x in nums),
                    "numbers": nums,
                    "payout_per_100": to_int(pay.removesuffix("円"), f"{nk_race_id}: {kind} {combo} の払戻"),
                    "popularity": to_int(pop, "人気") if re.fullmatch(r"[0-9,]+", pop) else None,
                })
    return pl.DataFrame(recs, schema=PAYOUT_SCHEMA)
