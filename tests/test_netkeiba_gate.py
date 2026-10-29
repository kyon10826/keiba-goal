"""変換器の合格ゲート: 出馬表の父・母の欠損を見逃さないこと、実キャッシュでゲート窓が record_data と一致すること。"""

import shutil
import sqlite3
from datetime import date

import polars as pl
import pytest
import requests
import yaml

from keiba_goal.config import load_config
from keiba_goal.data.schema import HISTORY_DTYPES
from keiba_goal.netkeiba import crawl, gate, horse_master, rebuild
from keiba_goal.netkeiba.http import Fetcher, Target
from keiba_goal.netkeiba.records import read_record_range

REAL = load_config()
GATE_DAY = date(2026, 10, 29)
# 窓より前の record にいない外国産馬の母: JV は 'Lady of Persia'、netkeiba は 'レディオブパーシャ'。
# 同じ母の別の外国産馬が record にいないので、カナ → 外国名の対応を学べない。
KNOWN_GATE_MISMATCHES = {(202509130904030510, "mother")}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("テストからネットワークに出ようとした")
    monkeypatch.setattr(requests.Session, "request", refuse)


def test_card_name_differences_count_missing_card_names():
    rec = pl.DataFrame({"race_id": [1, 2, 3], "nk_race_id": ["202509060101"] * 3, "horse_num": [1, 2, 3],
                        "id": [10, 20, 30], "father": ["F", "G", "Hat Trick"], "mother": ["M", "N", "O"]},
                       schema={c: HISTORY_DTYPES[c] for c in ("race_id", "nk_race_id", "horse_num", "id", "father",
                                                               "mother")})
    card = pl.DataFrame({"nk_race_id": ["202509060101"] * 3, "horse_num": [1, 2, 3], "horse_id": [10, 20, 30],
                         "father": ["F", None, "ハットトリック"], "mother": ["M", "N", "O"],
                         "broodmare_sire": [None] * 3}, schema=rebuild.CARD_PARENTS_SCHEMA)
    d = gate.card_name_differences(card, rec)
    assert d["race_id"].to_list() == [2, 3]
    assert d.filter(pl.col("father_card").is_null())["race_id"].to_list() == [2]   # 欠損も「違う」に数える
    assert gate.card_name_differences(card.clear(), rec).height == 3              # 出馬表が無い行も


def _gate_cache_ready() -> bool:
    manifest = REAL.paths.raw / "manifest.sqlite"
    if not (REAL.paths.record_dir.exists() and manifest.exists() and (REAL.paths.raw / "netkeiba").is_dir()):
        return False
    w = crawl.load_window(REAL, "gate", GATE_DAY)
    want = {Target(k, rid).url for rid in crawl.record_races(REAL, w)["nk_race_id"] for k in w.pages}
    with sqlite3.connect(f"file:{manifest}?mode=ro", uri=True) as con:
        have = {u for (u,) in con.execute("SELECT url FROM pages WHERE complete = 1")}
    return bool(want) and want <= have


def _scratch_config(tmp_path):
    """実キャッシュ (読むだけ) を指す一時データ置き場。manifest は写し、出力・レポートは一時ディレクトリへ。"""
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    (root / "interim").mkdir()
    src = sqlite3.connect(f"file:{REAL.paths.raw / 'manifest.sqlite'}?mode=ro", uri=True)
    dst = sqlite3.connect(root / "raw" / "manifest.sqlite")
    src.backup(dst)
    src.close()
    dst.close()
    (root / "raw" / "netkeiba").symlink_to(REAL.paths.raw / "netkeiba")
    course = REAL.paths.interim / "course_table.parquet"
    if course.exists():
        shutil.copy(course, root / "interim" / course.name)
    raw = dict(REAL.raw)
    raw["paths"] = {**raw["paths"], "data_root": str(root)}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_config(path)


@pytest.mark.skipif(not _gate_cache_ready(), reason="ゲート窓のキャッシュ (data/raw) か record_data が無い")
def test_gate_window_matches_record_data_from_the_real_cache(tmp_path):
    cfg = _scratch_config(tmp_path)
    w = crawl.load_window(cfg, "gate", GATE_DAY)
    seed = horse_master.seed_from_record(cfg.paths.record_dir, before=w.start)   # crawl.run と同じく答えを見ない
    b = rebuild.build(cfg, Fetcher.from_config(cfg, offline=True), w.pages, crawl.record_races(cfg, w), seed)
    rep = gate.gate_report(cfg, w.start, w.end, b)
    assert b.errors == [] and rep["dtype_problems"] == [] and rep["mismatches"]["(行が無い)"] == 0
    assert rep["rows_converted"] == rep["rows_record"] > 1500

    rec = read_record_range(cfg.paths.record_dir, w.start, w.end)
    j = rec.join(b.rows, on="race_id", suffix="_nk")
    remaining = set()
    for c in gate.COMPARE_COLUMNS[1:]:
        bad = j.filter(~pl.col(c).eq_missing(pl.col(f"{c}_nk")))
        remaining |= {(rid, c) for rid in bad["race_id"]}
    assert remaining == KNOWN_GATE_MISMATCHES
    assert {k: v for k, v in rep["mismatches"].items() if v} == {"mother": 1}

    # 父・母の 1 割以上は出馬表から来ており、出馬表の父・母は 1 行も欠けていない (欠ければ上の比較で落ちる)
    assert rep["parents"]["rows_from_card"] > 0.1 * b.rows.height
    assert rep["card_names_vs_record"]["card_null"] == 0
    assert rep["card_names_vs_record"]["differ"] < 0.05 * b.rows.height
    assert rep["win_payout_vs_odds"]["mismatched"] == [] and rep["win_payout_vs_odds"]["checked"] > 100

    from keiba_goal.data import history

    check = getattr(history, "check_nk_rows", None)   # P1 が netkeiba 行に課す検査 (あれば)
    if check is not None:
        check(history.conform(b.rows, "gate"), "gate")
