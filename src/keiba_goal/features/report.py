"""作成済みストアの点検 (年ごとの行数・前走のある行の割合・スピード指数の分布・目的変数の率・NaN の多い列) と、
B/year= と labels/year= が同じ入力 (今の data/interim の入力) から作られているかの検査 (問題があれば終了コード 1)。

    .venv/bin/python -m keiba_goal.features.report [--years 2003-2024] [--config path]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import polars as pl

from keiba_goal.config import load_config
from keiba_goal.features.build import parse_years
from keiba_goal.features.groups import discover
from keiba_goal.features.registry import OUTPUT_KEYS
from keiba_goal.features.store import Layout, current_inputs, nan_report, store_problems


def year_table(layout: Layout, years: Sequence[int]) -> pl.DataFrame:
    fig = pl.scan_parquet(layout.state_dir / "helpers" / "speed_fig.parquet")
    rows = []
    for y in years:
        store = pl.scan_parquet(layout.b_file(y)).select("race_id", "hs_n_starts", "sp_last", "rd_field_fallback")
        labels = pl.scan_parquet(layout.labels_file(y))
        el = labels.filter(pl.col("eligible")).join(store, on="race_id")
        figs = fig.filter((pl.col("race_id") // 10**14) == y).join(
            labels.filter(pl.col("started") & (pl.col("y_win").is_not_null())).select("race_id"), on="race_id"
        )
        stats = el.select(
            pl.len().alias("eligible"),
            (pl.col("hs_n_starts") > 0).mean().alias("has_prev"),
            pl.col("sp_last").is_not_nan().mean().alias("has_sp_last"),
            pl.col("y_win").mean().alias("win_rate"),
            (1 / pl.col("starters")).mean().alias("inv_starters"),
            pl.col("y_place").mean().alias("place_rate"),
            pl.col("noise_race").mean().alias("noise_rows"),
            pl.col("field_dec_fallback").mean().alias("fallback_rows"),
        ).collect()
        f = figs.select(pl.col("speed_fig").fill_nan(None).mean().alias("fig_mean"),
                        pl.col("speed_fig").fill_nan(None).std().alias("fig_std"),
                        pl.col("speed_fig").fill_nan(None).is_not_null().mean().alias("fig_share")).collect()
        rows.append({"year": y, "rows": pl.scan_parquet(layout.b_file(y)).select(pl.len()).collect().item(),
                     **stats.row(0, named=True), **f.row(0, named=True)})
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keiba_goal.features.report", description=__doc__)
    ap.add_argument("--years", help="2003-2024 (既定: features.years)")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    lo, hi = cfg.section("features")["years"]
    years = parse_years(args.years) or list(range(int(lo), int(hi) + 1))
    layout = Layout.from_config(cfg)
    registered = [s.name for g in discover() for s in g.specs]
    problems = {y: p for y, p in store_problems(layout, years, registered, current_inputs(layout, years)).items() if p}
    for y, ps in problems.items():
        print(f"[report] {y}: " + " / ".join(ps))
    readable = [y for y in years if layout.b_file(y).exists() and layout.labels_file(y).exists()]
    if not readable:
        print("[report] B/year= と labels/year= のそろった年がありません")
        return 1
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(year_table(layout, readable))
    last = readable[-3:]
    store = pl.concat([pl.read_parquet(layout.b_file(y)) for y in last], how="diagonal")
    features = [c for c in store.columns if c not in OUTPUT_KEYS]
    print(f"[report] {last[0]}-{last[-1]} の NaN の多い列 (全 {len(features)} 列)\n{nan_report(store, features, top=30)}")
    print(f"[report] 点検: {len(years) - len(problems)}/{len(years)} 年が整合" + (" (問題は上)" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
