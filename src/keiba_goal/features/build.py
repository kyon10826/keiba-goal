"""特徴量ストアの作成 (design §4、docs/features.md)。

    .venv/bin/python -m keiba_goal.features.build --stage A|B|labels|all [--groups g1,g2] [--years 2003-2024]

A: 段階 A のグループを全履歴で計算し、features.stage_a_min_year 以降の年を書く (--years で絞れる)。
   --groups があればそのうちの段階 A のグループだけ (1 つも無ければ例外)。
B: 年ごとに判断時刻の枠と段階 B のグループを作り、全グループの特徴量を B/year={Y}.parquet に組み立てる。
   --groups があればそのうちの段階 B のグループだけを必ず作り直し、ほかの段階 B のグループはファイルが古い・無い
   ときだけ作り直す (段階 A のグループ名だけなら組み立てだけ)。段階 A のファイルが古い・無ければ StaleError。
labels: 目的変数。all: A → B → labels。
重い処理なので同時に 1 つだけ動かす (メモリ 8GB)。
"""

from __future__ import annotations

import argparse
import resource
import sys
from collections.abc import Sequence

from keiba_goal.config import load_config
from keiba_goal.features.groups import discover
from keiba_goal.features.registry import Group, RegistryError
from keiba_goal.features.store import Layout, history_years, open_run, run_labels, run_stage_a, run_stage_b

STAGES = ("A", "B", "labels", "all")


def parse_years(text: str | None) -> list[int] | None:
    """'2003-2024' / '2022' / '2003,2010'。"""
    if not text:
        return None
    years: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        years.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(years))


def select(groups: Sequence[Group], stage: str, names: list[str] | None, strict: bool) -> list[Group]:
    """その段階のグループ (names があればそのうち names のもの)。strict なら 1 つも無いとき例外。"""
    in_stage = [g for g in groups if g.stage == stage]
    if names is None:
        return in_stage
    known = {g.name for g in groups}
    if unknown := [n for n in names if n not in known]:
        raise RegistryError(f"登録されていないグループ {unknown} (登録: {sorted(known)})")
    picked = [g for g in in_stage if g.name in names]
    if strict and not picked:
        raise RegistryError(f"{names} に段階 {stage} のグループがありません (段階 {stage}: {[g.name for g in in_stage]})")
    return picked


def _peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keiba_goal.features.build", description="特徴量ストアの作成")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--groups", help="カンマ区切りのグループ名 (既定: その段階の全グループ。B では必ず作り直すグループ)")
    ap.add_argument("--years", help="2003-2024 / 2022 / 2003,2010 (既定: A は stage_a_min_year 以降の全年、B・labels は features.years)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="B の組み立てで、段階 A のファイルが無い・古いグループ (と依存するグループ) を外して続ける")
    ap.add_argument("--config", help="設定ファイル (既定: config/default.yaml)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    params = cfg.section("features")
    groups = discover()
    names = [n.strip() for n in args.groups.split(",") if n.strip()] if args.groups else None
    years = parse_years(args.years)
    a_groups = select(groups, "A", names, strict=args.stage == "A") if args.stage in ("A", "all") else []
    b_groups = select(groups, "B", names, strict=False) if args.stage in ("B", "all") else []
    b_years = years or list(range(int(params["years"][0]), int(params["years"][1]) + 1))
    run = open_run(Layout.from_config(cfg), groups, params)
    if args.stage in ("A", "all"):
        run_stage_a(run, a_groups, years or history_years(run))
    if args.stage in ("B", "all"):
        run_stage_b(run, b_groups, b_years, allow_missing=args.allow_missing, check_labels=args.stage != "all")
    if args.stage in ("labels", "all"):
        run_labels(run, b_years)
    print(f"[build] このプロセスの最大メモリ {_peak_mb():,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
