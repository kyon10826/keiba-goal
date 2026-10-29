"""グループごとの中身 (手で書いた小さな履歴で値を確かめる) と、fixture の馬の過去走を履歴と突き合わせる。"""

import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.decision import decision_frame, pivot_snapshots, with_live_columns
from keiba_goal.features.groups import discover
from keiba_goal.features.groups.race_static import dist_band
from keiba_goal.features.groups.speed import FIG_SOURCES, MIN_SAMPLES, speed_figures
from keiba_goal.features.pipeline import build_store
from keiba_goal.features.registry import with_requirements
from keiba_goal.features.synth import history_frame

FIX = Path(__file__).parent / "fixtures" / "features"
NAN = float("nan")


def store(rows, groups, snapshots=None, year=2021):
    hist = history_frame(rows)
    return build_store(hist, snapshots, years=(year,), groups=with_requirements(discover(), groups))


def row_of(df: pl.DataFrame, date: int, horse_num: int = 1) -> dict:
    out = df.filter((pl.col("date") == date) & (pl.col("horse_num") == horse_num))
    assert out.height == 1
    return out.row(0, named=True)


def run(date, horse, **kw):
    base = {"date": date, "venue_code": 6, "race_num": 1, "horse_num": 1, "id": horse, "starters": 10}
    return {**base, **kw}


HORSE = [
    run(20210105, 1, rank=3, time_diff=0.5, last_3F_time=35.0, last_3F_rank=2, corner3_rank=4, corner4_rank=3,
        PCI=50.0, last_3F_time_diff=0.3, pop=2, win_odds=4.0, dist=1200, surface=2, class_ord=0, state="良",
        prize=100.0, weight=480.0, basis_weight=55.0, jockey_id="00001"),
    run(20210205, 1, error_code=1, rank=0, jockey_id="00009"),                                    # 取消 (出走でない)
    run(20210305, 1, rank=1, starters=8, time_diff=-0.2, corner1_rank=5, corner2_rank=4, corner3_rank=3,
        corner4_rank=2, pop=1, win_odds=2.5, dist=1600, surface=1, class_ord=1, state="稍", prize=500.0,
        weight=484.0, basis_weight=56.0, jockey_id="00002"),
    run(20210405, 1, rank=0, error_code=4, dist=1600, surface=1, class_ord=2, state="重", weight=490.0,
        basis_weight=57.0, jockey_id="00002", prize=0.0),
    run(20210505, 1, rank=2, dist=1800, surface=1, class_ord=2, weight=486.0, basis_weight=56.0, jockey_id="00002",
        inc_dec=-4.0, waku_num=1, sex="牡", age=4),
]


def test_prior_count_excludes_the_same_day_and_non_starts():
    hist = history_frame(HORSE + [run(20210505, 2, race_num=2)]).lazy()
    ev = lags.events(hist.filter(~pl.col("error_code").is_in([1, 2, 3])))
    out = lags.prior_count(hist.select("race_id", "id", "date"), ev).collect().sort("race_id")
    counts = dict(zip(out["race_id"].to_list(), out[lags.N_PRIOR].to_list()))
    assert [counts[r] for r in hist.filter(pl.col("id") == 1).collect()["race_id"]] == [0, 1, 1, 2, 3]
    third = lags.at_prior(out.lazy(), ev, 3, {"rank": "r3"}).collect()
    assert third.filter(pl.col("date") == 20210505)["r3"].to_list() == [3, None]


def test_past_runs_values():
    out = store(HORSE, ["past_runs"])
    r = row_of(out, 20210505)
    assert r["p1_dnf"] == 1.0 and math.isnan(r["p1_rank"]) and r["p1_days"] == 30.0 and r["p1_going"] == 2.0
    assert r["p2_rank"] == 1.0 and r["p2_rank_ratio"] == pytest.approx(1 / 8) and r["p2_days"] == 61.0
    assert r["p2_early_pos"] == pytest.approx(5 / 8) and r["p2_late_pos"] == pytest.approx(2 / 8)
    assert r["p2_log_odds"] == pytest.approx(math.log(2.5)) and r["p2_surface"] == 1.0 and r["p2_going"] == 1.0
    assert r["p3_rank"] == 3.0 and r["p3_early_pos"] == pytest.approx(4 / 10) and r["p3_class_ord"] == 0.0
    assert r["p3_last3f_rank_ratio"] == pytest.approx(0.2) and r["p3_dnf"] == 0.0 and r["p3_days"] == 120.0
    assert all(math.isnan(r[f"p4_{v}"]) for v in ("rank", "days", "dnf", "dist"))
    debut = row_of(out, 20210105)
    assert math.isnan(debut["p1_rank"]) and math.isnan(debut["p1_dnf"])


def test_horse_static_and_aggregates():
    out = store(HORSE, ["horse_static", "aggregates"])
    r = row_of(out, 20210505)
    assert r["hs_n_starts"] == 3.0 and r["hs_career_days"] == 120.0 and r["hs_sex"] == 1.0
    assert r["ag_all_win_rate"] == pytest.approx(1 / 3) and r["ag_all_top3_rate"] == pytest.approx(2 / 3)
    assert r["ag_all_rank_ratio_mean"] == pytest.approx((0.3 + 1 / 8) / 2)
    assert r["ag_l3_time_diff_min"] == pytest.approx(-0.2) and r["ag_all_prize_sum"] == 600.0
    assert r["ag_l3_pop_mean"] == pytest.approx(1.5)
    first = row_of(out, 20210105)
    assert first["hs_n_starts"] == 0.0 and math.isnan(first["hs_career_days"]) and math.isnan(first["ag_all_win_rate"])
    scratched_day = row_of(out, 20210205)
    assert scratched_day["hs_n_starts"] == 1.0 and scratched_day["ag_l3_win_rate"] == 0.0


def test_horse_day_compares_with_previous_starts():
    out = store(HORSE, ["horse_day"])
    r = row_of(out, 20210505)
    assert r["hd_same_jockey_1"] == 1.0 and r["hd_same_jockey_2"] == 1.0 and r["hd_same_jockey_3"] == 0.0
    assert math.isnan(r["hd_same_jockey_4"])
    assert r["hd_weight_vs_past_mean"] == pytest.approx(486 / ((480 + 484 + 490) / 3))
    assert r["hd_basis_weight_change"] == pytest.approx(-1.0)
    assert r["hd_weight_change_rate"] == pytest.approx(-4 / 486) and r["hd_blinker"] == 0.0


def test_hd_blinker_follows_the_schema_encoding():
    """'B' = 着用、null (record の空欄・出馬表に B の印が無い) = 着用なし、それ以外の値は不明 (NaN)。"""
    rows = [run(20210505, h, horse_num=h, blinker=b) for h, b in ((1, "B"), (2, None), (3, ""), (4, "?"))]
    out = store(rows, ["horse_day"]).sort("horse_num")["hd_blinker"].to_list()
    assert out[:3] == [1.0, 0.0, 0.0] and math.isnan(out[3])


def field(race_num=1, n=5, scratched=(), basis=None, date=20210505):
    basis = basis or [55.0] * n
    return [
        {"date": date, "venue_code": 6, "race_num": race_num, "horse_num": h, "id": 1000 * race_num + h,
         "error_code": 1 if h in scratched else 0, "basis_weight": basis[h - 1], "rank": 0 if h in scratched else h,
         "age": 3 if h < 3 else 4}
        for h in range(1, n + 1)
    ]


def snaps(race_key: int, odds_by_x: dict[int, list[float]]) -> pl.DataFrame:
    rows = [
        {"race_key": race_key, "horse_num": h, "x": x, "win": o, "place_lo": NAN, "place_hi": NAN}
        for x, odds in odds_by_x.items() for h, o in enumerate(odds, start=1)
    ]
    return pl.DataFrame(rows, schema={"race_key": pl.Int64, "horse_num": pl.Int8, "x": pl.Int16, "win": pl.Float32,
                                      "place_lo": pl.Float32, "place_hi": pl.Float32})


def test_decision_frame_field_dec_and_field_open():
    rows = with_live_columns(history_frame(field(1, 5, scratched=(5,)) + field(2, 6, scratched=(6,))).lazy()).collect()
    k1, k2 = sorted(rows["race_key"].unique().to_list())
    s = snaps(k1, {6: [2.0, 3.0, NAN, 5.0, NAN], 120: [2.0, 3.0, 9.0, 5.0, NAN]})   # 3 番は T-6 で売られていない
    fr = decision_frame(rows, s).sort("race_id")
    r1, r2 = fr.filter(pl.col("race_key") == k1), fr.filter(pl.col("race_key") == k2)
    assert r1["in_field_dec"].to_list() == [True, True, False, True, False] and not r1["field_dec_fallback"].any()
    assert r1["field_dec_size"].unique().to_list() == [3] and r1["field_open"].unique().to_list() == [4]
    assert r1["field_open_source"].unique().to_list() == ["snapshots"]
    assert r2["in_field_dec"].to_list() == [True] * 5 + [False] and r2["field_dec_fallback"].all()
    assert r2["field_open"].unique().to_list() == [5] and r2["field_open_source"].unique().to_list() == ["field_dec"]
    table = pl.DataFrame({"race_key": [k1], "field_open": [5]})
    assert decision_frame(rows, s, table).filter(pl.col("race_key") == k1)["field_open"].unique().to_list() == [5]
    assert set(pivot_snapshots(s).columns) >= {"win_6", "place_lo_120", "place_hi_30"}
    with pytest.raises(ValueError, match="重複"):
        pivot_snapshots(pl.concat([s, s]))


def test_race_day_horse_day_and_relative_use_field_dec_only():
    rows = field(1, 5, scratched=(5,), basis=[54.0, 55.0, 56.0, 57.0, 60.0])
    out = store(rows, ["race_day", "horse_day", "relative"]).sort("horse_num")
    assert out["rd_field_size"].unique().to_list() == [4.0] and out["rd_field_fallback"].unique().to_list() == [1.0]
    assert out["rd_n_paid_places"].unique().to_list() == [0.0]                  # field_open 4 → 複勝なし
    dev = out["hd_basis_weight_dev"].to_list()
    assert dev[:4] == pytest.approx([-1.5, -0.5, 0.5, 1.5]) and math.isnan(dev[4])
    assert out["rel_rank_hd_basis_weight"].to_list()[:4] == [1.0, 2.0, 3.0, 4.0]
    assert math.isnan(out["rel_rank_hd_basis_weight"][4]) and math.isnan(out["rel_z_hd_basis_weight"][4])
    z = out["rel_z_hd_basis_weight"].to_list()[:4]
    assert sum(z) == pytest.approx(0.0, abs=1e-6) and z[3] == pytest.approx(1.5 / math.sqrt(1.25))
    assert out["rel_z_hs_age"].to_list()[:4] == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert out["hd_horse_num_ratio"].to_list() == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.25])
    # 全頭同じ値の z は 0、値のある馬が 1 頭なら NaN
    assert out["rel_z_hs_n_starts"].to_list()[:4] == [0.0] * 4


def test_race_static_codes():
    rows = [
        {**field(1, 1)[0], "dist": 1200, "surface": 2, "class_ord": -1, "weight_code": 0, "corner_num": 2},
        {**field(2, 1)[0], "dist": 2101, "surface": 1, "class_ord": 3, "weight_code": 3},
        {**field(3, 1)[0], "dist": 1150, "surface": 2, "class_ord": 0, "weight_code": 1, "venue_code": 3},
    ]
    out = store(rows, ["race_static"]).sort("rs_race_num")
    assert out["rs_dist_band"].to_list() == [1.0, 4.0, 1.0]
    assert math.isnan(out["rs_class_ord"][0]) and math.isnan(out["rs_weight_code"][0])
    assert out["rs_class_ord"][1] == 3.0 and out["rs_month"][0] == 5.0 and out["rs_corner_num"][0] == 2.0
    assert not math.isnan(out["rs_course"][0])    # 中山ダ 1200
    assert math.isnan(out["rs_course"][2])        # 福島ダ 1150 は 2004 年からのコース (語彙外)
    dists = pl.DataFrame({"dist": [1000, 1300, 1301, 1899, 1900, 2100, 2101, 2700, 2701, None]})
    assert dists.select(dist_band().alias("band"))["band"].to_list() == [1, 1, 2, 2, 3, 3, 4, 4, 5, None]


def test_speed_figure_baseline_and_fallbacks():
    rows = []
    for year in (2018, 2019, 2020):
        for i in range(12):                                     # 中山ダ 1200 良: 3 年で 36 頭
            rows.append(run(year * 10_000 + 105, 10_000 + year * 100 + i, horse_num=i + 1, venue_code=6, dist=1200,
                            surface=2, state="良", rank=i + 1, time=70.0 + 0.1 * i, race_num=1))
        for i in range(4):                                      # 中京ダ 1200 良: 3 年で 12 頭 (< 30)
            rows.append(run(year * 10_000 + 106, 20_000 + year * 100 + i, horse_num=i + 1, venue_code=7, dist=1200,
                            surface=2, state="良", rank=i + 1, time=71.0, race_num=2))
    rows += [
        run(20210110, 1, venue_code=6, dist=1200, surface=2, state="良", rank=1, time=69.5),
        run(20210110, 2, venue_code=7, dist=1200, surface=2, state="良", rank=1, time=69.5, race_num=2),
        run(20210110, 3, venue_code=7, dist=1000, surface=2, state="良", rank=1, time=59.0, race_num=3),
        run(20180110, 4, venue_code=6, dist=1200, surface=2, state="良", rank=1, time=69.5, race_num=4),
    ]
    hist = history_frame(rows).lazy()
    figs = speed_figures(Ctx("t", history=hist, history_columns=FIG_SOURCES)).join(
        hist.select("race_id", "id").collect(), on="race_id")
    fig = dict(zip(figs["id"].to_list(), figs["speed_fig"].to_list()))
    t6 = pl.Series([70.0 + 0.1 * i for i in range(12)] * 3 + [69.5])     # 2018-01-10 の id 4 も 2018 年の完走馬
    assert fig[1] == pytest.approx((t6.mean() - 69.5) / t6.std(), rel=1e-5)
    pooled = pl.concat([t6, pl.Series([71.0] * 12)])
    assert MIN_SAMPLES == 30 and fig[2] == pytest.approx((pooled.mean() - 69.5) / pooled.std(), rel=1e-5)
    assert fig[3] is None or math.isnan(fig[3])     # 距離 1000 の標本なし
    assert fig[4] is None or math.isnan(fig[4])     # 2018 年は前 3 年の標本なし


def test_fixture_past_runs_match_history_by_hand():
    """fixture の 3 頭: 1・2 走前の着順・頭数・日数・確定オッズが履歴の前の出走と一致する。"""
    hist = pl.read_parquet(FIX / "history.parquet")
    snaps_2022 = pl.read_parquet(FIX / "ts_snap" / "2022.parquet")
    d = 20220320
    out = build_store(hist, snaps_2022, years=(2022,), dates=(d,), groups=with_requirements(discover(), ["past_runs"]))
    day = hist.filter((pl.col("date") == d) & (pl.col("error_code") == 0)).sort("race_id")
    checked = 0
    for target in day.slice(100, 3).iter_rows(named=True):
        prev = hist.filter((pl.col("id") == target["id"]) & (pl.col("date") < d) & ~pl.col("error_code").is_in([1, 2, 3]))
        prev = prev.sort("date", descending=True)
        r = out.filter(pl.col("race_id") == target["race_id"]).row(0, named=True)
        for k in (1, 2):
            if prev.height < k:
                assert math.isnan(r[f"p{k}_rank"])
                continue
            p = prev.row(k - 1, named=True)
            assert r[f"p{k}_rank"] == p["rank"] if p["rank"] > 0 else math.isnan(r[f"p{k}_rank"])
            assert r[f"p{k}_starters"] == p["starters"]
            assert r[f"p{k}_log_odds"] == pytest.approx(math.log(p["win_odds"]), abs=1e-5)
            assert r[f"p{k}_days"] == (ymd(d) - ymd(p["date"])).days
            checked += 1
    assert checked >= 4


def ymd(v: int) -> date:
    return date(v // 10_000, v // 100 % 100, v % 100)
