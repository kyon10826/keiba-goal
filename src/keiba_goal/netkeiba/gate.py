"""変換器の合格ゲート (design §3.3): 変換行と record_data の同じ race_id の行を leg 以外の全列で比べる。

比較は型も値も厳密 (Float32 の完全一致)。父・母が record から写されて比較が素通りにならないよう、ゲートの変換は
「窓より前の record だけで作った馬マスタ」で行い (crawl.run)、出馬表の父・母そのものも全行で record と比べる。
gate_report は結果を data/raw/reports/gate_report.json に保存する。
"""

from __future__ import annotations

from fractions import Fraction

import polars as pl

from keiba_goal.config import Config
from keiba_goal.data.schema import HISTORY_DTYPES, RECORD_COLUMNS
from keiba_goal.netkeiba.convert import CONVERTER_VERSION, jv_parent_name
from keiba_goal.netkeiba.files import save_json
from keiba_goal.netkeiba.rebuild import BuildResult
from keiba_goal.netkeiba.records import read_record_range

COMPARE_COLUMNS: tuple[str, ...] = tuple(c for c in RECORD_COLUMNS if c != "leg")
DETAIL_SCHEMA = {"race_id": pl.Int64, "column": pl.Utf8, "record": pl.Utf8, "netkeiba": pl.Utf8}


def dtype_problems(nk: pl.DataFrame) -> list[str]:
    want = list(HISTORY_DTYPES.items())
    got = list(nk.schema.items())
    return [f"{w[0]}: 期待 {w[1]} / 実際 {g}" for w, g in zip(want, got) if w != g] + (
        [f"列数 {len(got)} (期待 {len(want)})"] if len(got) != len(want) else [])


def compare(nk: pl.DataFrame, rec: pl.DataFrame, columns: tuple[str, ...] = COMPARE_COLUMNS
            ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(列ごとの不一致件数, 不一致の明細 [race_id, column, record, netkeiba])。

    片方にしか無い行は column='(行が無い)' として数える。
    """
    cols = [c for c in columns if c != "race_id"]
    r = rec.select("race_id", *cols).with_columns(_in_rec=pl.lit(True))
    n = nk.select("race_id", *cols).rename({c: f"{c}_nk" for c in cols}).with_columns(_in_nk=pl.lit(True))
    j = r.join(n, on="race_id", how="full", coalesce=True).with_columns(
        pl.col("_in_rec").fill_null(False), pl.col("_in_nk").fill_null(False))
    both = j.filter(pl.col("_in_rec") & pl.col("_in_nk"))
    counts, details = [], []
    missing = j.filter(~(pl.col("_in_rec") & pl.col("_in_nk")))
    counts.append({"column": "(行が無い)", "mismatches": missing.height, "compared": j.height})
    for row in missing.select("race_id", "_in_rec").iter_rows():
        details.append({"race_id": row[0], "column": "(行が無い)",
                        "record": "あり" if row[1] else "なし", "netkeiba": "なし" if row[1] else "あり"})
    for c in cols:
        bad = both.filter(~pl.col(c).eq_missing(pl.col(f"{c}_nk")))
        counts.append({"column": c, "mismatches": bad.height, "compared": both.height})
        for rid, a, b in bad.select("race_id", c, f"{c}_nk").iter_rows():
            details.append({"race_id": rid, "column": c, "record": repr(a), "netkeiba": repr(b)})
    return pl.DataFrame(counts), pl.DataFrame(details, schema=DETAIL_SCHEMA)


def card_name_differences(card_parents: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """出馬表の父・母 (II→2 の後) と record の父・母が違う行。出馬表側の欠損も違いに数える。"""
    card = card_parents.with_columns(
        pl.col(c).map_elements(jv_parent_name, return_dtype=pl.Utf8).alias(f"{c}_card") for c in ("father", "mother"))
    r = rec.select("race_id", "nk_race_id", "horse_num", "id", "father", "mother")
    j = r.join(card.select("nk_race_id", "horse_num", "father_card", "mother_card"), on=["nk_race_id", "horse_num"],
               how="left")
    differ = pl.col("father").ne_missing(pl.col("father_card")) | pl.col("mother").ne_missing(pl.col("mother_card"))
    return j.filter(differ).select("race_id", "id", "father", "father_card", "mother", "mother_card")


def win_payout_check(rows: pl.DataFrame, payouts: pl.DataFrame) -> dict:
    """1 着同着でないレースの単勝払戻 = 確定単勝オッズ × 100 (design §3.4)。"""
    winners = rows.filter(pl.col("rank") == 1).group_by("race_key").agg(
        n=pl.len(), horse_num=pl.col("horse_num").first(), odds=pl.col("win_odds").first())
    win = payouts.filter(pl.col("bet_type") == "win").select("race_key", "horse_num", "payout_per_100")
    j = winners.filter(pl.col("n") == 1).join(win, on=["race_key", "horse_num"], how="left")
    bad = j.filter(pl.col("payout_per_100").is_null()
                   | ((pl.col("odds") * 100).round(0).cast(pl.Int32) != pl.col("payout_per_100")))
    return {"checked": j.height, "dead_heat_races": winners.filter(pl.col("n") > 1).height,
            "mismatched": bad.to_dicts()}


def pci_ties(rows: pl.DataFrame, race_ids: list[int]) -> list[dict]:
    """PCI の不一致が「厳密値がちょうど x.x5」(record_data の丸めが再現できない場合) かどうか。"""
    out = []
    for r in rows.filter(pl.col("race_id").is_in(race_ids)).iter_rows(named=True):
        if not r["time"] or r["last_3F_time"] is None or r["dist"] <= 600:
            out.append({"race_id": r["race_id"], "pci_exact": None, "tie": False})
            continue
        t, l3 = round(r["time"] * 10), round(r["last_3F_time"] * 10)
        exact = Fraction((t - l3) * 60, r["dist"] - 600) / Fraction(l3, 10) * 100 - 50
        out.append({"race_id": r["race_id"], "pci_exact": str(exact), "tie": (exact * 10).denominator == 2})
    return out


def report(rows: pl.DataFrame, rec: pl.DataFrame, payouts: pl.DataFrame, card_parents: pl.DataFrame,
           known_horses: frozenset[int]) -> dict:
    """ゲートの結果 (JSON にできる dict)。

    known_horses は変換に使った馬マスタ (窓より前の record) にいた馬。それ以外の馬の父・母は出馬表から来ている。
    """
    counts, details = compare(rows, rec)
    from_card = rows.filter(~pl.col("id").is_in(sorted(known_horses)))
    parent_bad = details.filter(pl.col("column").is_in(["father", "mother"])
                                & pl.col("race_id").is_in(from_card["race_id"].implode()))
    names = card_name_differences(card_parents, rec)
    return {
        "races_record": int(rec["race_key"].n_unique()), "rows_record": rec.height,
        "races_converted": int(rows["race_key"].n_unique()) if rows.height else 0, "rows_converted": rows.height,
        "dtype_problems": dtype_problems(rows),
        "mismatches": {r["column"]: r["mismatches"] for r in counts.iter_rows(named=True)},
        "details": details.to_dicts(),
        "parents": {
            "rows_from_record_master": rows.height - from_card.height,
            "rows_from_card": from_card.height,
            "horses_from_card": int(from_card["id"].n_unique()),
            "mismatched_rows_from_card": int(parent_bad["race_id"].n_unique()),
        },
        "card_names_vs_record": {
            "rows": rec.height,
            "differ": names.height,
            "card_null": names.filter(pl.col("father_card").is_null() | pl.col("mother_card").is_null()).height,
            "rows_detail": names.to_dicts(),
        },
        "win_payout_vs_odds": win_payout_check(rows, payouts),
        "pci_ties": pci_ties(rows, details.filter(pl.col("column") == "PCI")["race_id"].to_list()),
    }


def gate_report(cfg: Config, start: int, end: int, b: BuildResult) -> dict:
    """作り直しの結果を record_data の窓の行と比べ、data/raw/reports/gate_report.json に保存する。"""
    rec = read_record_range(cfg.paths.record_dir, start, end)
    rep = {"window": "gate", "converter": CONVERTER_VERSION,
           **report(b.rows, rec, b.payouts, b.card_parents, b.known_horses), "build_errors": b.errors}
    save_json(rep, cfg.paths.raw / "reports" / "gate_report.json")
    return rep


def print_gate(report: dict) -> None:
    print(f"\nゲート: record {report['races_record']} レース / {report['rows_record']} 行、"
          f"変換 {report['races_converted']} レース / {report['rows_converted']} 行")
    print(f"  型の不一致: {report['dtype_problems'] or 'なし'}")
    bad = {k: v for k, v in report["mismatches"].items() if v}
    total = report["rows_record"]
    print(f"  列ごとの不一致 (leg 以外の {len(report['mismatches']) - 1} 列、厳密比較): {bad or 'すべて一致'}")
    for col in bad:
        ex = [d for d in report["details"] if d["column"] == col][:5]
        print(f"   - {col}: {bad[col]} 件 ({bad[col] / max(total, 1):.3%}) "
              f"例 {[(d['race_id'], d['record'], d['netkeiba']) for d in ex]}")
    p = report["parents"]
    print(f"  父・母の出どころ: 窓より前の record {p['rows_from_record_master']} 行 / 出馬表 {p['rows_from_card']} 行"
          f" ({p['horses_from_card']} 頭、うち record と違う行 {p['mismatched_rows_from_card']})")
    n = report["card_names_vs_record"]
    print(f"  出馬表の父・母 (II→2 の後) と record の比較: {n['rows']} 行中 {n['differ']} 行が違う"
          f" (出馬表側の欠損 {n['card_null']}。外国産馬の親は JV が外国名)")
    ties = report["pci_ties"]
    if ties:
        print(f"  PCI の不一致 {len(ties)} 件のうち厳密値がちょうど x.x5 (record の丸めが再現不能): "
              f"{sum(t['tie'] for t in ties)} 件")
    wp = report["win_payout_vs_odds"]
    print(f"  単勝払戻 = 確定オッズ×100: {wp['checked'] - len(wp['mismatched'])}/{wp['checked']} レース一致"
          f" (1 着同着 {wp['dead_heat_races']} レースは対象外)")
    if report["build_errors"]:
        print(f"  作り直しのエラー: {len(report['build_errors'])} 件 例 {report['build_errors'][:3]}")
