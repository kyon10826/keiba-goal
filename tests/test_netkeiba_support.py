"""netkeiba テスト共通の小道具 (テストは無い): 合成の record_data CSV と、保存ページを返す偽セッション。"""

from __future__ import annotations

import gzip
from pathlib import Path

from keiba_goal.data.schema import CODE_TO_PLACE, RECORD_COLUMNS

FX = Path(__file__).parent / "fixtures" / "netkeiba"

TEMPLATE = ("202507260101010101,25,07,26,1,札幌,1,1,セイウンダイフク,01032,12,1,1,7,17,2,1200,良,曇,11,牡,2,55.0,,"
            "428,-6,3,36.9,12, 2.5,71.4,0,0,12,12,35.1,5,36.30,53.4,2.1,後方,6,0,0,サンダースノー,ニシノマメフク,"
            "2023104705").split(",")


def _from_race_id(race_id: str) -> dict[str, str]:
    """race_id の分解と同じ値の列 (P1 の読み込みは race_id と各列の一致を検査する)。"""
    daily = int(race_id[12:14])
    return {"year": race_id[2:4], "month": race_id[4:6], "day": race_id[6:8],
            "place": CODE_TO_PLACE[int(race_id[8:10])], "times": str(int(race_id[10:12])),
            "daily": "ABC"[daily - 10] if daily >= 10 else str(daily), "race_num": str(int(race_id[14:16])),
            "horse_num": str(int(race_id[16:18]))}


def record_line(**values) -> str:
    """1 行分の CSV。race_id を渡すと年月日・場・回・日次・R・馬番もそれに合わせる (明示した値が優先)。"""
    row = list(TEMPLATE)
    if "race_id" in values:
        values = {**_from_race_id(str(values["race_id"])), **values}
    for k, v in values.items():
        row[RECORD_COLUMNS.index(k)] = str(v)
    return ",".join(f'"{x}"' if " " in x.strip() else x for x in row)


def write_record(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("cp932"))
    return path


class FakeResponse:
    def __init__(self, status: int, content: bytes):
        self.status_code, self.content = status, content


class FixtureSession:
    """URL → 保存ページ。無いものは 404。呼ばれた URL を記録する。"""

    def __init__(self):
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if "/race/" in url and "db.netkeiba.com" in url:
            name = f"result_{url.rstrip('/').rsplit('/', 1)[1]}.html.gz"
        elif "shutuba_past.html" in url:
            name = f"shutuba_past_{url.rsplit('=', 1)[1]}.html.gz"
        elif "race_list_sub.html" in url:
            name = f"race_list_{url.rsplit('=', 1)[1]}.html.gz"
        else:
            return FakeResponse(404, b"")
        path = FX / name
        if not path.exists():
            return FakeResponse(404, b"")
        return FakeResponse(200, gzip.decompress(path.read_bytes()))
