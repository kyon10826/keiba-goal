"""成績データ (record_data) を読み、正規化した records.parquet を作る (design §3.1)。

CSV は cp932・ヘッダ無し・47 列。全列を文字列で読み、前後の空白を除いてから schema.RECORD_DTYPES に変換する
(型推論させない: 日次の A/B/C や騎手コードの先頭ゼロが壊れるため)。
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from keiba_goal.data.schema import (
    CLASS_ORD,
    HISTORY_DTYPES,
    NOT_STARTED_CODES,
    PLACE_CODES,
    RECORD_COLUMNS,
    RECORD_DTYPES,
    TRACK_CODE,
    null_sentinels,
)

log = logging.getLogger(__name__)

SORT_KEYS: tuple[str, ...] = ("date", "race_key", "horse_num")
_FILE_RE = re.compile(r"^record_data_(\d{4})(?:_(\d{8}))?\.csv$")
_DAILY_HEX = {"A": "10", "B": "11", "C": "12"}


def select_record_files(record_dir: Path) -> list[Path]:
    """読むファイル: 年次ファイル全部 + 年次ファイルの無い年は日付が最大のスナップショット 1 本。

    2025 のスナップショットは古いものが新しいものの部分集合なので、全部読むと最大 5 重複する。
    """
    annual: dict[int, Path] = {}
    snapshots: dict[int, list[tuple[str, Path]]] = {}
    for p in sorted(record_dir.iterdir()):
        m = _FILE_RE.match(p.name)
        if m is None:
            continue
        year = int(m.group(1))
        if m.group(2) is None:
            annual[year] = p
        else:
            snapshots.setdefault(year, []).append((m.group(2), p))
    chosen = dict(annual)
    for year, items in snapshots.items():
        chosen.setdefault(year, max(items)[1])
    return [chosen[y] for y in sorted(chosen)]


def read_record_csv(path: Path) -> pl.DataFrame:
    """1 ファイルを全列 Utf8 で読み、strip して空文字を null にする。47 列でない行があれば例外。"""
    lines = [ln for ln in path.read_bytes().decode("cp932").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path.name}: データ行がありません")
    n_sep = len(RECORD_COLUMNS) - 1
    bad = [i + 1 for i, ln in enumerate(lines) if ln.count(",") != n_sep]
    if bad:
        raise ValueError(f"{path.name}: {len(RECORD_COLUMNS)} 列でない行が {len(bad)} 行あります (例: {bad[:5]} 行目)")
    # 引用符は父・母の英字名 ("Dream Street Rose") を囲むだけで、中に引用符やカンマを含む値は無い
    # (上の列数検査がカンマ入りの値を弾く)。囲みを外して netkeiba の表記とそろえる。
    raw = pl.read_csv(
        "\n".join(lines).encode("utf-8"),
        has_header=False,
        new_columns=list(RECORD_COLUMNS),
        infer_schema=False,
        quote_char='"',
    )
    stripped = raw.with_columns(pl.col(c).str.strip_chars() for c in RECORD_COLUMNS)
    return stripped.with_columns(pl.when(pl.col(c) != "").then(pl.col(c)).alias(c) for c in RECORD_COLUMNS)


def to_record_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """文字列の 47 列を RECORD_DTYPES に変換する。変換できない値は例外 (黙って null にしない)。

    日次 A/B/C → 10/11/12、増減 '+6' → 6、着差 '----' → null、騎手コードは 5 桁ゼロ埋め。
    """
    fixed = df.with_columns(
        pl.col("daily").replace(_DAILY_HEX),
        pl.col("inc_dec").str.strip_prefix("+"),
        pl.when(~pl.col("time_diff").str.contains(r"^-+$")).then(pl.col("time_diff")).alias("time_diff"),
        pl.col("jockey_id").str.zfill(5),
    )
    return fixed.select(pl.col(c).cast(RECORD_DTYPES[c], strict=True) for c in RECORD_COLUMNS)


def check_race_id(df: pl.DataFrame, name: str) -> None:
    """race_id が一意で、その分解 (年・月・日・場・回・日次・R・馬番) が各列と一致することを検査する。"""
    rid = pl.col("race_id")
    place_code = pl.col("place").replace_strict(PLACE_CODES, default=-1, return_dtype=pl.Int16)
    checks = {
        "year": rid // 10**14 % 100 == pl.col("year"),
        "month": rid // 10**12 % 100 == pl.col("month"),
        "day": rid // 10**10 % 100 == pl.col("day"),
        "place": rid // 10**8 % 100 == place_code,
        "times": rid // 10**6 % 100 == pl.col("times"),
        "daily": rid // 10**4 % 100 == pl.col("daily"),
        "race_num": rid // 100 % 100 == pl.col("race_num"),
        "horse_num": rid % 100 == pl.col("horse_num"),
    }
    counts = df.select((~e.fill_null(False)).sum().alias(k) for k, e in checks.items()).row(0, named=True)
    bad = {k: v for k, v in counts.items() if v}
    if bad:
        raise ValueError(f"{name}: race_id の分解と列が一致しない行があります {bad}")
    if df["race_id"].n_unique() != df.height:
        raise ValueError(f"{name}: race_id が重複しています")
    df.select((rid // 10**10).cast(pl.Utf8).str.to_date("%Y%m%d", strict=True))   # 存在しない日付なら例外


def unknown_codes(df: pl.DataFrame, column: str, known: Mapping[int, object]) -> dict[int, int]:
    """対応表に無いコードと件数。"""
    vc = df.filter(~pl.col(column).is_in(list(known)))[column].value_counts()
    return {int(code): int(n) for code, n in vc.iter_rows()}


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """DERIVED_DTYPES の列を加え、HISTORY_DTYPES の順・型にそろえる (trainer_id は record に無いので null)。"""
    rid = pl.col("race_id")

    def track_part(i: int) -> pl.Expr:
        mapping = {code: parts[i] for code, parts in TRACK_CODE.items()}
        return pl.col("track_code").replace_strict(mapping, default=pl.lit(None, dtype=pl.Int8), return_dtype=pl.Int8)

    starters = (~pl.col("error_code").is_in(sorted(NOT_STARTED_CODES))).sum().over("race_key")
    out = df.with_columns(
        (rid // 10**10).alias("date"),
        (rid // 100).alias("race_key"),
        ((rid // 10**14).cast(pl.Utf8) + (rid // 100 % 10**8).cast(pl.Utf8).str.zfill(8)).alias("nk_race_id"),
        (rid // 10**8 % 100).alias("venue_code"),
        track_part(0).alias("surface"),
        track_part(1).alias("direction"),
        track_part(2).alias("course_inout"),
        pl.col("class_code").replace_strict(CLASS_ORD, default=-1, return_dtype=pl.Int8).alias("class_ord"),
        pl.lit(None, dtype=pl.Utf8).alias("trainer_id"),
        pl.lit("record").alias("source"),
    ).with_columns(starters.alias("starters"))
    return out.select(pl.col(c).cast(t) for c, t in HISTORY_DTYPES.items())


def load_record_file(path: Path) -> tuple[pl.DataFrame, dict[int, int], dict[int, int]]:
    """1 ファイル → (派生列付き・(date, race_key, horse_num) 順のフレーム, 未知の class_code, 未知の track_code)。"""
    df = null_sentinels(to_record_dtypes(read_record_csv(path)))
    check_race_id(df, path.name)
    unknown_class = unknown_codes(df, "class_code", CLASS_ORD)
    unknown_track = unknown_codes(df, "track_code", TRACK_CODE)
    return add_derived(df).sort(SORT_KEYS), unknown_class, unknown_track


@dataclass(frozen=True)
class RecordsSummary:
    files: tuple[str, ...]
    rows: int
    races: int
    min_date: int
    max_date: int
    unknown_class_codes: dict[int, int]
    unknown_track_codes: dict[int, int]


def build_records(record_dir: Path, out_path: Path) -> RecordsSummary:
    """record_data → records.parquet ((date, race_key, horse_num) 順)。ファイルごとに読んで追記する。

    ファイル間で日付が重ならず昇順であることを検査するので、ファイル内の一意性と合わせて race_id は全体で一意になり、
    全体を並べ替えなくても出力は (date, race_key, horse_num) 順になる。
    """
    files = select_record_files(record_dir)
    if not files:
        raise FileNotFoundError(f"{record_dir} に record_data_*.csv がありません")
    races = 0
    dates: list[int] = []
    unknown_class: Counter[int] = Counter()
    unknown_track: Counter[int] = Counter()

    def parts() -> Iterator[pl.DataFrame]:
        nonlocal races
        prev_max: int | None = None
        for p in files:
            df, uc, ut = load_record_file(p)
            lo, hi = int(df["date"].min()), int(df["date"].max())
            if prev_max is not None and lo <= prev_max:
                raise ValueError(f"{p.name}: 日付 {lo} が前のファイルの最終日 {prev_max} 以前です (ファイル間で重複・逆順)")
            prev_max = hi
            dates.extend((lo, hi))
            races += df["race_key"].n_unique()
            unknown_class.update(uc)
            unknown_track.update(ut)
            yield df

    rows = write_parquet_parts(parts(), out_path)
    if unknown_class:
        log.warning("CLASS_ORD に無い class_code (class_ord = -1): %s", dict(unknown_class))
    if unknown_track:
        log.warning("TRACK_CODE に無い track_code (surface 等 = null): %s", dict(unknown_track))
    return RecordsSummary(
        files=tuple(p.name for p in files),
        rows=rows,
        races=races,
        min_date=min(dates),
        max_date=max(dates),
        unknown_class_codes=dict(unknown_class),
        unknown_track_codes=dict(unknown_track),
    )


def write_parquet_parts(parts: Iterable[pl.DataFrame], path: Path) -> int:
    """フレームを順に 1 つの parquet へ追記する (全体をメモリに載せない)。失敗したら何も残さない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for df in parts:
            if df.height == 0:
                continue
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += df.height
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    if writer is None:
        raise ValueError(f"{path.name}: 書き出す行がありません")
    writer.close()
    os.replace(tmp, path)
    return rows


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """一時ファイルに書いてから置き換える (中断で壊れたファイルを残さない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)
