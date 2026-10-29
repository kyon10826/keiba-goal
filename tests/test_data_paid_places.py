from pathlib import Path

import polars as pl
import pytest

from keiba_goal.data.history import build_history
from keiba_goal.data.paid_places import (
    PAID_PLACES_SCHEMA,
    build_paid_places,
    load_paid_places,
    payout_places,
    places_for_field,
    write_paid_places,
)
from keiba_goal.data.record import build_records
from keiba_goal.data.ts_odds import build_ts, extract_file

FIX = Path(__file__).parent / "fixtures" / "data"
# 2023-09-10 阪神 2R (netkeiba 202309040202): 8 頭登録、5 番が発売開始後に出走取消で 7 頭出走。
# 結果ページ (tests/fixtures/netkeiba/result_202309040202.html.gz) の複勝は 3・6・2 番 = 3 着まで払戻
RACE_8_TO_7 = 2023091009040202


def _races(**starters: int) -> pl.DataFrame:
    return pl.DataFrame({"race_key": [int(k[1:]) for k in starters], "starters": list(starters.values())},
                        schema={"race_key": pl.Int64, "starters": pl.Int16})


def _field(**field_open: int | None) -> pl.DataFrame:
    return pl.DataFrame({"race_key": [int(k[1:]) for k in field_open], "field_open": list(field_open.values())},
                        schema={"race_key": pl.Int64, "field_open": pl.Int16})


def _payouts(**paid: int | None) -> pl.DataFrame:
    return pl.DataFrame({"race_key": [int(k[1:]) for k in paid], "n_paid_places": list(paid.values())},
                        schema={"race_key": pl.Int64, "n_paid_places": pl.Int8})


def test_places_for_field_boundaries():
    got = pl.DataFrame({"n": [0, 4, 5, 7, 8, 18]}).select(places_for_field(pl.col("n")))
    assert got.to_series().to_list() == [0, 0, 2, 2, 3, 3]


def test_jd_field_open_decides_the_8_to_7_race():
    r = extract_file(str(FIX / "jd" / "JD09234202.CSV"))
    assert r.race_key == RACE_8_TO_7 and r.field_open == 8
    out = build_paid_places(_races(r2023091009040202=7), _field(r2023091009040202=r.field_open), None)
    assert out.row(0, named=True) == {"race_key": RACE_8_TO_7, "n_paid_places": 3, "field_open": 8, "starters": 7,
                                      "source": "jd_field_open"}


def test_rule_order_payout_then_field_open_then_starters():
    races = _races(r1=7, r2=7, r3=7, r4=4, r5=9)
    field = _field(r1=8, r2=8, r4=None, r5=4)
    paid = payout_places(pl.concat([_payouts(r1=2, r2=None), _payouts(r1=2)]))   # r2 は払戻表で不明 (null)
    out = build_paid_places(races, field, paid)
    assert dict(out.schema) == PAID_PLACES_SCHEMA
    got = {r["race_key"]: (r["n_paid_places"], r["source"]) for r in out.iter_rows(named=True)}
    assert got == {
        1: (2, "payout"),           # 払戻表の値が最優先 (field_open の規則と違っても)
        2: (3, "jd_field_open"),    # 払戻表が null → field_open 8
        3: (2, "starters"),         # JD なし → 出走 7 頭
        4: (0, "starters"),         # 4 頭以下は複勝なし
        5: (0, "jd_field_open"),    # JD で発売開始時 4 頭
    }


def test_conflicting_payout_values_in_one_race_raise():
    with pytest.raises(ValueError, match="食い違"):
        payout_places(_payouts(r1=2) .vstack(_payouts(r1=3)))


def test_duplicate_races_raise():
    with pytest.raises(ValueError, match="重複"):
        build_paid_places(pl.concat([_races(r1=8), _races(r1=8)]), _field(), None)


def test_write_paid_places_from_fixtures(tmp_path: Path):
    rec = tmp_path / "records.parquet"
    build_records(FIX / "record", rec)
    build_history(rec, None, tmp_path / "history.parquet")
    years = [2004, 2005, 2015, 2023, 2024]
    build_ts(FIX / "jd", tmp_path, years, processes=1)
    payouts = tmp_path / "payouts.parquet"
    # 2025-09-13 中山 11R (record にある) の払戻表は 3 着払い、record に無いレースの払戻も 1 つ混ぜる
    pl.DataFrame({"race_key": [2025091306040311, 2025091306040311, 1999010101010101], "n_paid_places": [3, 3, 2]},
                 schema={"race_key": pl.Int64, "n_paid_places": pl.Int8}).write_parquet(payouts)
    s = write_paid_places(tmp_path / "history.parquet", tmp_path / "ts_race", years, payouts, tmp_path / "paid_places.parquet")
    out = load_paid_places(tmp_path / "paid_places.parquet")
    assert out.height == s.races == pl.read_parquet(rec)["race_key"].n_unique()
    assert s.payout_races_not_in_history == 1 and s.by_source["payout"] == 1
    row = out.filter(pl.col("race_key") == 2025091306040311).row(0, named=True)
    assert (row["n_paid_places"], row["source"], row["field_open"]) == (3, "payout", None)
    jd = out.filter(pl.col("source") == "jd_field_open")
    assert jd.height == s.by_source["jd_field_open"] > 0 and (jd["field_open"] >= jd["starters"]).all()
    assert s.field_below_starters == 0


def test_missing_ts_race_or_table_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="paid_places"):
        load_paid_places(tmp_path / "paid_places.parquet")
    rec = tmp_path / "records.parquet"
    build_records(FIX / "record", rec)
    with pytest.raises(FileNotFoundError, match="--step ts"):
        write_paid_places(rec, tmp_path / "ts_race", [2023], None, tmp_path / "paid_places.parquet")
