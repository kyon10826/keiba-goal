"""P1 データ層のビルド (design §3.1–3.4): records → history → ts → paid_places → qtable。

    .venv/bin/python -m keiba_goal.data.build --step records|history|ts|paid_places|qtable|all
        [--years 2023 2024] [--post-times PATH] [--payouts PATH] [--config PATH]

出力は cfg.paths.interim (data/interim/) の records.parquet, history.parquet, ts_snap/ ts_race/ ts_final/{year}.parquet,
paid_places.parquet, q_tables/{backtest,production,by_year}.json。
netkeiba の表は P2 の `python -m keiba_goal.netkeiba.crawl combine` がまとめたもの (nk_rows.parquet, post_times.parquet,
payouts.parquet) を読む。窓ごとのファイル (*_gate.parquet など) だけがある・まとめた方が古いときは警告する。
"""

from __future__ import annotations

import argparse
import logging
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from keiba_goal.config import Config, load_config
from keiba_goal.data.history import build_history
from keiba_goal.data.paid_places import write_paid_places
from keiba_goal.data.post_est import races_after_g1
from keiba_goal.data.qtable import BET_TYPES, build_q_by_year, build_q_tables
from keiba_goal.data.record import build_records
from keiba_goal.data.ts_odds import POST_SOURCES, SNAPSHOT_X, build_ts, jd_files_by_year, load_post_times

STEPS = ("records", "history", "ts", "paid_places", "qtable")
# (窓ごとのファイルの接頭辞, P2 の combine がまとめたファイル)
COMBINED = {"nk_rows": "nk_rows.parquet", "post_times": "post_times.parquet", "nk_payouts": "payouts.parquet"}


def warn_if_not_combined(interim: Path, prefix: str) -> None:
    """窓ごとのファイルがあるのに、まとめたファイルが無い・それより古ければ大きく警告する。"""
    parts = sorted(interim.glob(f"{prefix}_*.parquet"))
    if not parts:
        return
    combined = interim / COMBINED[prefix]
    newest = max(p.stat().st_mtime for p in parts)
    if combined.exists() and combined.stat().st_mtime >= newest:
        return
    state = "がありません" if not combined.exists() else "が窓ごとのファイルより古いです"
    bar = "!" * 78
    print(f"{bar}\n警告: {combined.name} {state} ({', '.join(p.name for p in parts)})。\n"
          f"      `python -m keiba_goal.netkeiba.crawl combine` (P2) でまとめてから作り直してください。\n{bar}")


def _years(cfg: Config, years: Sequence[int] | None) -> list[int]:
    if years is not None:
        return list(years)
    y0, y1 = cfg.section("data").get("ts_years", [2003, 2024])
    return list(range(int(y0), int(y1) + 1))


def jd_years(cfg: Config) -> list[int]:
    """data.ts_years のうち JD ファイルのある年 (ts の出力があるはずの年)。"""
    return sorted(set(jd_files_by_year(cfg.paths.ts_odds_dir)) & set(_years(cfg, None)))


def _input(path: str | None, default: Path, what: str) -> Path:
    """明示されたパスは必須 (無ければ例外)。既定のパスは無くてもよい (呼び出し側で無しとして扱う)。"""
    if path is None:
        return default
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{what} {p} がありません")
    return p


def run_records(cfg: Config) -> None:
    t0 = time.time()
    s = build_records(cfg.paths.record_dir, cfg.paths.interim / "records.parquet")
    print(f"[records] {cfg.paths.record_dir} の {len(s.files)} ファイル ({s.files[0]} … {s.files[-1]})")
    print(f"[records] {s.rows:,} 行 / {s.races:,} レース, {s.min_date}〜{s.max_date} ({time.time() - t0:.0f} 秒)")
    expected = cfg.section("data").get("expected_records")
    if expected and (s.rows, s.races) != (expected["rows"], expected["races"]):
        print(f"[records] 警告: 想定の {expected['rows']:,} 行 / {expected['races']:,} レースと違います")
    if s.unknown_class_codes:
        print(f"[records] CLASS_ORD に無い class_code (class_ord = -1): {s.unknown_class_codes}")
    if s.unknown_track_codes:
        print(f"[records] TRACK_CODE に無い track_code: {s.unknown_track_codes}")


def run_history(cfg: Config) -> None:
    interim = cfg.paths.interim
    warn_if_not_combined(interim, "nk_rows")
    nk = interim / "nk_rows.parquet"
    print(f"[history] 入力: {interim / 'records.parquet'} + {nk if nk.exists() else '(nk_rows.parquet なし)'}")
    s = build_history(interim / "records.parquet", nk, interim / "history.parquet")
    print(
        f"[history] {s.rows:,} 行 (record {s.record_rows:,} + netkeiba {s.nk_rows_added:,}、"
        f"record と重複して捨てた netkeiba 行 {s.nk_rows_replaced_by_record:,}), {s.min_date}〜{s.max_date}"
    )


def run_ts(cfg: Config, years: Sequence[int] | None, post_times_path: str | None) -> None:
    t0 = time.time()
    d = cfg.section("data")
    interim = cfg.paths.interim
    warn_if_not_combined(interim, "post_times")
    path = _input(post_times_path, interim / "post_times.parquet", "発走時刻表")
    post_times = load_post_times(path)
    by_source = {s: sum(1 for p in post_times.values() if p.source == s) for s in POST_SOURCES[:2]}
    print(f"[ts] 発走時刻表: {path} {len(post_times):,} レース {by_source}"
          + ("" if path.exists() else " (ファイルなし: 全レース推定)"))
    records = interim / "records.parquet"
    after_g1 = races_after_g1(
        pl.scan_parquet(records).filter(pl.col("date") >= 20030101)
        .select("race_key", "date", "venue_code", "race_num", "class_ord").collect()
    )
    print(f"[ts] G1 後のレース (推定を 5 分早める): {len(after_g1):,} ({records})")
    summaries = build_ts(
        cfg.paths.ts_odds_dir,
        interim,
        _years(cfg, years),
        post_times=post_times,
        after_g1=after_g1,
        processes=int(d.get("ts_processes", 8)),
        records_path=records,
    )
    print(f"[ts] JD: {cfg.paths.ts_odds_dir}")
    cols = ["year", "files", "no_k3", "two_k3", "ref_moved", "ref_moved_plateau", "dropped_rows", "dropped_empty"]
    cols += [f"avail_T{x}" for x in SNAPSHOT_X]
    cols += ["snap_unpriced", "min_vote_share", "max_vote_share", "t6_share_ge90", "final_eq_record",
             "record_races", "record_races_with_final"]
    _print_table("[ts]", cols, summaries)
    print("[ts] 発走時刻の出どころと、実時刻のあるレースの (区分 3 − 発走)・(ref − 発走)・推定値と実時刻の比較 (分)")
    _print_table(
        "[ts]",
        ["year", *[f"post_{s}" for s in POST_SOURCES], "post_rejected", "k3_minus_post_med", "ref_minus_post_p05",
         "ref_minus_post_med", "ref_minus_post_p95", "est_eq_real", "est_lt_real", "est_gt_real"],
        summaries,
    )
    print(f"[ts] {len(summaries)} 年 ({time.time() - t0:.0f} 秒)")


def run_paid_places(cfg: Config, payouts_path: str | None) -> None:
    interim = cfg.paths.interim
    warn_if_not_combined(interim, "nk_payouts")
    path = _input(payouts_path, interim / "payouts.parquet", "払戻表")
    years = jd_years(cfg)
    print(f"[paid_places] 入力: {interim / 'history.parquet'}, {interim / 'ts_race'} ({years[0] if years else '-'}"
          f"〜{years[-1] if years else '-'} の {len(years)} 年), {path if path.exists() else f'(払戻表 {path.name} なし)'}")
    s = write_paid_places(interim / "history.parquet", interim / "ts_race", years,
                          path if path.exists() else None, interim / "paid_places.parquet")
    print(f"[paid_places] {s.races:,} レース {s.by_source} → {interim / 'paid_places.parquet'}")
    total = sum(s.changed_vs_starters.values())
    print(f"[paid_places] 出走頭数の規則と違うレース {total} (年, 出どころ): {s.changed_vs_starters}")
    print(f"[paid_places] 払戻表と field_open の両方: 一致 {s.payout_vs_field_open[0]}, 不一致 {s.payout_vs_field_open[1]}; "
          f"field_open < 出走頭数 {s.field_below_starters}; history に無い払戻表のレース {s.payout_races_not_in_history}")
    for ex in s.examples_changed:
        print(f"[paid_places]   例: {ex}")


def run_qtable(cfg: Config, years: Sequence[int] | None) -> None:
    d = cfg.section("data")
    windows = d.get("q_windows", {"backtest": [2020, 2022], "production": [2022, 2024]})
    quantile = float(cfg.section("betting").get("odds_drop_quantile", 0.10))
    tables = build_q_tables(cfg.paths.interim, windows, quantile, int(d.get("q_min_band_samples", 200)))
    for name, t in tables.items():
        print(f"[qtable] {name} {t.window[0]}-{t.window[1]} (q{t.quantile:g})")
        for bt, bands in t.bands.items():
            desc = ", ".join(f"[{b.lo:g},{b.hi:g}) q={b.q:.3f} n={b.n}" for b in bands)
            print(f"[qtable]   {bt:5}: {desc}")
    by_year = build_q_by_year(cfg.paths.interim, jd_years(cfg) if years is None else years, quantile)
    print(f"[qtable] 年ごとの q (帯は結合しない、n < 200 は *) → {cfg.paths.interim / 'q_tables' / 'by_year.json'}")
    for bt in BET_TYPES:
        edges = [f"[{b['lo']:g},{b['hi']:g})" for b in next(iter(by_year.values()))[bt]] if by_year else []
        print(f"[qtable]   {bt:5} {'year':>4} " + " ".join(f"{e:>14}" for e in edges))
        for year, v in by_year.items():
            cells = [("      -" if b["q"] is None else f"{b['q']:.3f}") + ("*" if b["n"] < 200 else " ") + f"({b['n']:>5})"
                     for b in v[bt]]
            print(f"[qtable]   {bt:5} {year:>4} " + " ".join(f"{c:>14}" for c in cells))


def _print_table(tag: str, cols: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    widths = [max(10, len(c)) for c in cols]
    print(f"{tag} " + " ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    for s in rows:
        cells = []
        for c, w in zip(cols, widths):
            v = s.get(c, "")
            cells.append(f"{v:>{w}.4f}" if isinstance(v, float) else f"{v!s:>{w}}")
        print(f"{tag} " + " ".join(cells))


def _peak_mb(who: int) -> float:
    rss = resource.getrusage(who).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024  # ru_maxrss は macOS ではバイト、Linux では KB


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keiba_goal.data.build", description="P1 データ層のビルド")
    ap.add_argument("--step", required=True, choices=(*STEPS, "all"))
    ap.add_argument("--config", help="設定ファイル (既定: config/default.yaml)")
    ap.add_argument("--years", type=int, nargs="+", help="ts と年ごとの q の対象年 (既定: data.ts_years。paid_places は常に全年)")
    ap.add_argument("--post-times", help="発走時刻表 (既定: data/interim/post_times.parquet、無ければ全レース推定)")
    ap.add_argument("--payouts", help="払戻表 (既定: data/interim/payouts.parquet、無ければ JD・出走頭数から)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    steps = STEPS if args.step == "all" else (args.step,)
    for step in steps:
        if step == "records":
            run_records(cfg)
        elif step == "history":
            run_history(cfg)
        elif step == "ts":
            run_ts(cfg, args.years, args.post_times)
        elif step == "paid_places":
            run_paid_places(cfg, args.payouts)
        else:
            run_qtable(cfg, args.years)
    print(
        f"[build] 最大メモリ: このプロセス {_peak_mb(resource.RUSAGE_SELF):,.0f} MB, "
        f"子プロセス (1 つあたり最大) {_peak_mb(resource.RUSAGE_CHILDREN):,.0f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
