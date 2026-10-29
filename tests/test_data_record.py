from pathlib import Path

import polars as pl
import pytest

from keiba_goal.data.record import (
    add_derived,
    build_records,
    check_race_id,
    load_record_file,
    read_record_csv,
    select_record_files,
    to_record_dtypes,
)
from keiba_goal.data.schema import HISTORY_DTYPES, RECORD_COLUMNS, RECORD_DTYPES, SENTINEL_ZERO_COLUMNS

FIX = Path(__file__).parent / "fixtures" / "data" / "record"
# 実データの 6 行 (2021-05-02 阪神 4R の 1・2・3・5・8 番、2024-07-21 札幌 3R の 3 番) をそのままのバイトで
QUIRKS = Path(__file__).parent / "fixtures" / "data" / "record_quirks" / "quoted_names_and_zero_sentinels.csv"


def _load(name: str) -> pl.DataFrame:
    return to_record_dtypes(read_record_csv(FIX / name))


def _race(df: pl.DataFrame, race_key: int) -> pl.DataFrame:
    return df.filter((pl.col("race_id") // 100) == race_key).sort("horse_num")


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932"))
    return path


def _fixture_lines(name: str) -> list[str]:
    return (FIX / name).read_bytes().decode("cp932").splitlines()


def test_selects_annual_files_and_only_the_latest_snapshot(tmp_path: Path):
    for name in (
        "record_data_2023.csv", "record_data_2024.csv", "record_data_2025_20250822.csv",
        "record_data_2025_20250919.csv", "record_data_2025_20250905.csv", "memo.csv", "record_data_2024_x.csv",
    ):
        (tmp_path / name).write_text("")
    assert [p.name for p in select_record_files(tmp_path)] == [
        "record_data_2023.csv", "record_data_2024.csv", "record_data_2025_20250919.csv",
    ]


def test_annual_file_wins_over_snapshots_of_the_same_year(tmp_path: Path):
    for name in ("record_data_2025.csv", "record_data_2025_20250919.csv"):
        (tmp_path / name).write_text("")
    assert [p.name for p in select_record_files(tmp_path)] == ["record_data_2025.csv"]


def test_dtypes_follow_schema():
    df = _load("record_data_2015.csv")
    assert df.columns == list(RECORD_COLUMNS)
    assert dict(df.schema) == RECORD_DTYPES


def test_daily_hex_and_year_as_given():
    df = _load("record_data_2015.csv")
    daily = dict(df.group_by(pl.col("race_id") // 100).agg(pl.col("daily").first()).iter_rows())
    assert daily[2015052405021011] == 10   # 'A'
    assert daily[2015053005021102] == 11   # 'B'
    assert daily[2015053105021202] == 12   # 'C'
    assert df["year"].unique().to_list() == [15]


def test_blank_and_dash_values_become_null_and_signed_weight_change_is_numeric():
    race = _race(_load("record_data_2015.csv"), 2015053105021202)
    scratched = race.filter(pl.col("horse_num") == 13).row(0, named=True)   # 出走取消
    assert scratched["error_code"] == 1
    assert scratched["weight"] is None and scratched["inc_dec"] is None and scratched["win_odds"] is None
    assert scratched["time_diff"] is None and scratched["rank"] == 0 and scratched["pop"] is None
    pulled_up = race.filter(pl.col("horse_num") == 6).row(0, named=True)    # 競走中止
    assert pulled_up["error_code"] == 4 and pulled_up["time_diff"] is None and pulled_up["win_odds"] == pytest.approx(59.1)
    inc = dict(race.select("horse_num", "inc_dec").iter_rows())
    assert inc[1] == 4.0 and inc[3] == -2.0 and inc[8] == 0.0 and inc[6] == -16.0
    winner = race.filter(pl.col("rank") == 1).row(0, named=True)
    assert winner["time_diff"] == pytest.approx(-0.1)


def test_jump_race_blank_columns_are_null():
    race = _race(_load("record_data_2015.csv"), 2015053108031204)
    assert race["corner_num"].null_count() == race.height
    assert race["last_3F_rank"].null_count() == race.height
    assert race["Ave_3F"].null_count() == race.height


def test_jockey_id_is_zero_padded_to_five_chars(tmp_path: Path):
    lines = _fixture_lines("record_data_2005.csv")
    cells = lines[0].split(",")
    cells[9] = "135"
    df = to_record_dtypes(read_record_csv(_write_lines(tmp_path / "x.csv", [",".join(cells)] + lines[1:])))
    assert df["jockey_id"][0] == "00135"
    assert df["jockey_id"].str.len_chars().unique().to_list() == [5]


def test_dead_heat_keeps_min_rank():
    race = _race(_load("record_data_2021.csv"), 2021050209021204)
    ranks = sorted(r for r in race["rank"].to_list() if r > 0)
    assert ranks[:3] == [1, 1, 3]
    assert race.filter(pl.col("rank") == 1)["horse_num"].to_list() == [3, 7]


def test_demoted_horse_keeps_official_rank():
    race = _race(_load("record_data_2017.csv"), 2017070903020405)
    demoted = race.filter(pl.col("error_code") == 7).row(0, named=True)
    assert demoted["horse_num"] == 8 and demoted["rank"] == 4
    assert race["inc_dec"].null_count() == race.height   # 新馬戦: 前走が無い


def test_rejects_line_with_wrong_field_count(tmp_path: Path):
    lines = _fixture_lines("record_data_2005.csv")
    lines[3] = lines[3].rsplit(",", 1)[0]
    with pytest.raises(ValueError, match="47 列でない行が 1 行"):
        read_record_csv(_write_lines(tmp_path / "bad.csv", lines))


def test_rejects_unparseable_value(tmp_path: Path):
    lines = _fixture_lines("record_data_2005.csv")
    cells = lines[0].split(",")
    cells[16] = "12OO"   # dist
    with pytest.raises(pl.exceptions.InvalidOperationError):
        to_record_dtypes(read_record_csv(_write_lines(tmp_path / "bad.csv", [",".join(cells)] + lines[1:])))


@pytest.mark.parametrize("col, value", [(2, "09"), (6, "2"), (12, "13"), (5, "函館")])
def test_race_id_decomposition_mismatch_is_rejected(tmp_path: Path, col: int, value: str):
    lines = _fixture_lines("record_data_2005.csv")
    cells = lines[0].split(",")
    cells[col] = value
    df = to_record_dtypes(read_record_csv(_write_lines(tmp_path / "bad.csv", [",".join(cells)] + lines[1:])))
    with pytest.raises(ValueError, match="race_id の分解"):
        check_race_id(df, "bad.csv")


def test_duplicate_race_id_is_rejected(tmp_path: Path):
    lines = _fixture_lines("record_data_2005.csv")
    df = to_record_dtypes(read_record_csv(_write_lines(tmp_path / "dup.csv", lines + lines[:1])))
    with pytest.raises(ValueError, match="重複"):
        check_race_id(df, "dup.csv")


def test_derived_columns():
    df = add_derived(_load("record_data_2015.csv"))
    assert dict(df.schema) == HISTORY_DTYPES
    row = df.filter(pl.col("race_id") == 201505310502120203).row(0, named=True)
    assert row["date"] == 20150531 and row["race_key"] == 2015053105021202
    assert row["nk_race_id"] == "201505021202" and row["venue_code"] == 5
    assert (row["surface"], row["direction"], row["course_inout"]) == (2, 2, 0)   # track 23 = ダート左
    assert row["class_ord"] == 0 and row["trainer_id"] is None and row["source"] == "record"
    starters = dict(df.group_by("race_key").agg(pl.col("starters").first()).iter_rows())
    assert starters[2015053105021202] == 15   # 16 頭登録 − 出走取消 1 (競走中止は出走扱い)
    assert starters[2015053005021102] == 15   # 競走除外 1
    assert starters[2015053108031204] == 14   # 障害、競走中止 3
    jump = df.filter(pl.col("race_key") == 2015053108031204).row(0, named=True)
    assert jump["surface"] == 3


def test_unknown_class_code_maps_to_minus_one_and_is_reported(tmp_path: Path):
    lines = _fixture_lines("record_data_2005.csv")
    lines = [ln.split(",") for ln in lines]
    for cells in lines:
        cells[13] = "99"
    d = tmp_path / "rec"
    d.mkdir()
    _write_lines(d / "record_data_2005.csv", [",".join(c) for c in lines])
    s = build_records(d, tmp_path / "records.parquet")
    assert s.unknown_class_codes == {99: len(lines)}
    assert pl.read_parquet(tmp_path / "records.parquet")["class_ord"].unique().to_list() == [-1]


def test_build_records_from_fixture_directory(tmp_path: Path):
    out = tmp_path / "records.parquet"
    s = build_records(FIX, out)
    assert s.files[-1] == "record_data_2025_20250919.csv" and "record_data_2025_20250822.csv" not in s.files
    df = pl.read_parquet(out)
    assert dict(df.schema) == HISTORY_DTYPES
    assert df.height == s.rows == 13 + 64 + 15 + 12 + 13 + 37 + 22
    assert s.races == df["race_key"].n_unique() == 1 + 4 + 1 + 1 + 1 + 3 + 2
    assert df["race_id"].n_unique() == df.height   # 2025 の部分集合スナップショットを読まないので重複なし
    assert df.equals(df.sort(["date", "race_key", "horse_num"]))
    assert (s.min_date, s.max_date) == (20050813, 20250913)
    assert not (tmp_path / "records.parquet.tmp").exists()


def test_files_with_overlapping_dates_are_rejected(tmp_path: Path):
    d = tmp_path / "rec"
    d.mkdir()
    lines = _fixture_lines("record_data_2024.csv")
    _write_lines(d / "record_data_2024.csv", lines)
    _write_lines(d / "record_data_2025_20250919.csv", lines[:3])   # 2024 の行が 2025 のファイルに混入
    with pytest.raises(ValueError, match="前のファイルの最終日"):
        build_records(d, tmp_path / "records.parquet")
    assert not (tmp_path / "records.parquet").exists()
    assert not (tmp_path / "records.parquet.tmp").exists()


def _by_id(df: pl.DataFrame) -> dict[int, dict]:
    return {r["race_id"]: r for r in df.iter_rows(named=True)}


def test_quoted_parent_names_are_unquoted():
    raw = QUIRKS.read_bytes().decode("cp932")
    assert '"Sea The Moon"' in raw and '"Duff One"' in raw
    rows = _by_id(load_record_file(QUIRKS)[0])
    assert rows[202105020902120402]["father"] == "Sea The Moon"
    assert rows[202105020902120405]["father"] == "Medaglia d'Oro"      # 引用符の中のアポストロフィは値の一部
    assert rows[202407210101020303]["father"] == "Arrogate" and rows[202407210101020303]["mother"] == "Duff One"
    assert rows[202105020902120401]["mother"] == "ジェルヴェーズ"
    df = load_record_file(QUIRKS)[0]
    assert not df["father"].str.contains('"').any() and not df["mother"].str.contains('"').any()


def test_zero_sentinels_become_null_only_in_sentinel_columns():
    before = _by_id(to_record_dtypes(read_record_csv(QUIRKS)))
    rows = _by_id(load_record_file(QUIRKS)[0])
    excluded = 202105020902120408                              # 競走除外: タイム・通過順・上がり順位は 0 で入っている
    assert before[excluded]["time"] == 0 and before[excluded]["corner4_rank"] == 0 and before[excluded]["last_3F_rank"] == 0
    assert rows[excluded]["error_code"] == 3
    assert all(rows[excluded][c] is None for c in SENTINEL_ZERO_COLUMNS)
    assert rows[excluded]["rank"] == 0 and rows[excluded]["prize"] == 0     # rank 0 は error_code と読む約束、賞金 0 は本物
    two_corners = rows[202407210101020303]                    # 1200m: 1・2 コーナーが無い
    assert (two_corners["corner1_rank"], two_corners["corner2_rank"]) == (None, None)
    assert (two_corners["corner3_rank"], two_corners["corner4_rank"]) == (12, 15) and two_corners["time"] == pytest.approx(69.5)
    winner = rows[202105020902120403]                          # 着差 -0.0 (1 着同着)、上がり順位 1
    assert winner["time_diff"] == 0 and winner["last_3F_rank"] == 1
    leader = rows[202105020902120401]                          # -3F 差 0.0・賞金 0 は値のまま
    assert leader["last_3F_time_diff"] == 0 and leader["prize"] == 0 and leader["time_diff"] == pytest.approx(1.2)
    df = load_record_file(QUIRKS)[0]
    assert df.select((pl.col(c) == 0).sum() for c in SENTINEL_ZERO_COLUMNS).row(0) == (0,) * len(SENTINEL_ZERO_COLUMNS)


def test_built_records_have_no_quotes_and_no_zero_sentinels(tmp_path: Path):
    out = tmp_path / "records.parquet"
    build_records(FIX, out)
    df = pl.read_parquet(out)
    assert not df["father"].str.contains('"').any() and not df["mother"].str.contains('"').any()
    assert df.select((pl.col(c) == 0).sum() for c in SENTINEL_ZERO_COLUMNS).row(0) == (0,) * len(SENTINEL_ZERO_COLUMNS)
    assert df.filter(pl.col("error_code").is_in([1, 3, 4]))["time"].null_count() == df.filter(pl.col("error_code").is_in([1, 3, 4])).height
