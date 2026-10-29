from pathlib import Path

import polars as pl
import pytest

from keiba_goal.data.history import build_history
from keiba_goal.data.record import add_derived, build_records
from keiba_goal.data.schema import HISTORY_DTYPES, RECORD_COLUMNS

FIX = Path(__file__).parent / "fixtures" / "data" / "record"
DUP_RACE = 2025091306040311
EARLY_RACE = 2005081301010104


@pytest.fixture(scope="module")
def records_path(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("rec") / "records.parquet"
    build_records(FIX, out)
    return out


def as_nk(rows: pl.DataFrame) -> pl.DataFrame:
    """record の行 (47 列を書き換えたもの) から netkeiba 由来の行を作る。派生列は 47 列から作り直す。"""
    return add_derived(rows.select(RECORD_COLUMNS)).with_columns(
        pl.lit("netkeiba").alias("source"), pl.lit("01234").alias("trainer_id")
    )


def test_history_equals_records_without_netkeiba_rows(tmp_path: Path, records_path: Path):
    s = build_history(records_path, tmp_path / "nk_rows.parquet", tmp_path / "history.parquet")
    hist = pl.read_parquet(tmp_path / "history.parquet")
    assert dict(hist.schema) == HISTORY_DTYPES and list(hist.columns) == list(HISTORY_DTYPES)
    assert hist.equals(pl.read_parquet(records_path))
    assert (s.rows, s.nk_rows_added, s.nk_rows_replaced_by_record) == (hist.height, 0, 0)


def _nk_rows(records: pl.DataFrame) -> pl.DataFrame:
    """netkeiba 由来の行を模す: record と重なる 1 レース (値を変える) + record の翌日の 1 レース + 2005 年の別レース。"""
    dup = records.filter(pl.col("race_key") == DUP_RACE).with_columns((pl.col("weight") + 2).alias("weight"))
    new = dup.with_columns((pl.col("race_id") + 10**10).alias("race_id"), (pl.col("day") + 1).alias("day"))
    early = records.filter(pl.col("race_key") == EARLY_RACE).with_columns(   # 回を変えた別レース
        (pl.col("race_id") + 10**6).alias("race_id"), (pl.col("times") + 1).alias("times")
    )
    return pl.concat([as_nk(dup), as_nk(new), as_nk(early)]).sample(fraction=1.0, shuffle=True, seed=0)


def test_record_rows_win_and_netkeiba_rows_are_appended_in_order(tmp_path: Path, records_path: Path):
    records = pl.read_parquet(records_path)
    n_dup = records.filter(pl.col("race_key") == DUP_RACE).height
    n_early = records.filter(pl.col("race_key") == EARLY_RACE).height
    _nk_rows(records).write_parquet(tmp_path / "nk_rows.parquet")
    s = build_history(records_path, tmp_path / "nk_rows.parquet", tmp_path / "history.parquet")
    hist = pl.read_parquet(tmp_path / "history.parquet")
    assert dict(hist.schema) == HISTORY_DTYPES
    assert hist["race_id"].n_unique() == hist.height == records.height + n_dup + n_early
    assert (s.nk_rows_added, s.nk_rows_replaced_by_record, s.nk_rows_in_record_races) == (n_dup + n_early, n_dup, 0)
    assert hist.equals(hist.sort(["date", "race_key", "horse_num"]))
    kept = hist.filter(pl.col("race_key") == DUP_RACE)
    assert kept["source"].unique().to_list() == ["record"] and kept["trainer_id"].null_count() == kept.height
    added = hist.filter(pl.col("race_key") == DUP_RACE + 10**8)
    assert added["source"].unique().to_list() == ["netkeiba"] and added["nk_race_id"].unique().to_list() == ["202506040311"]
    early = hist.filter(pl.col("race_key") == EARLY_RACE + 10**4)
    assert early["nk_race_id"].unique().to_list() == ["200501020104"] and early.height == n_early
    assert hist.filter(pl.col("source") == "netkeiba")["date"].min() == 20050813
    assert s.max_date == 20250914


def _write_nk(tmp_path: Path, nk: pl.DataFrame) -> Path:
    nk.write_parquet(tmp_path / "nk_rows.parquet")
    return tmp_path / "nk_rows.parquet"


def _early_nk(records_path: Path) -> pl.DataFrame:
    records = pl.read_parquet(records_path)
    return as_nk(records.filter(pl.col("race_key") == EARLY_RACE).with_columns(
        (pl.col("race_id") + 10**6).alias("race_id"), (pl.col("times") + 1).alias("times")
    ))


def test_netkeiba_rows_with_missing_columns_are_rejected(tmp_path: Path, records_path: Path):
    nk = _write_nk(tmp_path, _early_nk(records_path).drop("starters"))
    with pytest.raises(ValueError, match="starters"):
        build_history(records_path, nk, tmp_path / "history.parquet")
    assert not (tmp_path / "history.parquet").exists()


@pytest.mark.parametrize("source", ["jrdb", "record", None])
def test_netkeiba_rows_must_be_labelled_netkeiba(tmp_path: Path, records_path: Path, source: str | None):
    nk = _write_nk(tmp_path, _early_nk(records_path).with_columns(pl.lit(source, dtype=pl.Utf8).alias("source")))
    with pytest.raises(ValueError, match="source"):
        build_history(records_path, nk, tmp_path / "history.parquet")


def test_netkeiba_rows_with_duplicate_race_id_are_rejected(tmp_path: Path, records_path: Path):
    early = _early_nk(records_path)
    nk = _write_nk(tmp_path, pl.concat([early, early.head(1)]))
    with pytest.raises(ValueError, match="重複"):
        build_history(records_path, nk, tmp_path / "history.parquet")


@pytest.mark.parametrize("column, change, match", [
    ("month", pl.col("month") + 1, "race_id の分解"),          # 47 列と race_id が食い違う
    ("date", pl.col("date") + 1, "派生列"),
    ("race_key", pl.col("race_key") + 1, "派生列"),
    ("nk_race_id", pl.lit("200501010104"), "派生列"),          # 回を変える前の netkeiba ID のまま
    ("starters", pl.col("starters") + 1, "派生列"),
])
def test_netkeiba_rows_with_inconsistent_keys_are_rejected(tmp_path: Path, records_path: Path, column, change, match):
    nk = _write_nk(tmp_path, _early_nk(records_path).with_columns(change.cast(HISTORY_DTYPES[column]).alias(column)))
    with pytest.raises(ValueError, match=match):
        build_history(records_path, nk, tmp_path / "history.parquet")


def test_incomplete_netkeiba_race_is_rejected(tmp_path: Path, records_path: Path):
    nk = _write_nk(tmp_path, _early_nk(records_path).head(2))     # starters は全頭で数えた値のまま
    with pytest.raises(ValueError, match="starters"):
        build_history(records_path, nk, tmp_path / "history.parquet")
