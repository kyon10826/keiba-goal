"""固定語彙: コース (場 × 芝ダ × 距離) のコード表 (design §4.1「学習期間で固定した語彙」)。

語彙は 2003 年以前の履歴だけから作って course_vocab.json に固定する (どの評価年とも重ならない)。
作り直し: .venv/bin/python -m keiba_goal.features.vocab [--config path]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import polars as pl

from keiba_goal.config import load_config

COURSE_VOCAB_PATH = Path(__file__).with_name("course_vocab.json")
COURSE_KEYS: tuple[str, ...] = ("venue_code", "surface", "dist")
VOCAB_MAX_YEAR = 2003


def build_course_vocab(history_path: Path, max_year: int = VOCAB_MAX_YEAR) -> dict[str, object]:
    combos = (
        pl.scan_parquet(history_path)
        .filter((pl.col("date") // 10_000) <= max_year)
        .select(COURSE_KEYS)
        .drop_nulls()
        .unique()
        .sort(COURSE_KEYS)
        .collect()
    )
    return {"keys": list(COURSE_KEYS), "max_year": max_year, "courses": [list(r) for r in combos.iter_rows()]}


@cache
def course_vocab() -> pl.DataFrame:
    """(venue_code, surface, dist, course_code)。course_code は語彙の通し番号 (0 始まり)。"""
    data = json.loads(COURSE_VOCAB_PATH.read_text(encoding="utf-8"))
    rows = data["courses"]
    return pl.DataFrame(
        {
            "venue_code": pl.Series([r[0] for r in rows], dtype=pl.Int8),
            "surface": pl.Series([r[1] for r in rows], dtype=pl.Int8),
            "dist": pl.Series([r[2] for r in rows], dtype=pl.Int16),
            "course_code": pl.Series(range(len(rows)), dtype=pl.Float32),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keiba_goal.features.vocab", description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    vocab = build_course_vocab(cfg.paths.interim / "history.parquet")
    COURSE_VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[vocab] コース {len(vocab['courses'])} 件 ({vocab['max_year']} 年以前) → {COURSE_VOCAB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
