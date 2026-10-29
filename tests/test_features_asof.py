"""as-of 検査 (design §12-4) を登録済みの全グループに: D の特徴量が D より後の行・D の行のレース後の列に依らず、
段階 A は D の当日変わりうる列にも、段階 B は他のレースの当日の値にも依らない。

fixture (tests/fixtures/features/) の 2022 年の 3 日で、(a) そのままの batch、(b) D より後を消し D のレース後の列を
null、(c) D 以降のレース後の列を並べ替え、(d) D の偶数番のレースの当日の列・取消の印・オッズを変える、の D の行の
特徴量が一致すること ((d) は段階 A の全行と奇数番のレースの全特徴量)。新しいグループは自動で対象になる。
"""

from pathlib import Path

import polars as pl
import pytest

from keiba_goal.features import lags
from keiba_goal.features.asof import check_day, perturb_from, scramble_from, truncate_at
from keiba_goal.features.decision import with_live_columns
from keiba_goal.features.exprs import f32, started
from keiba_goal.features.groups import discover
from keiba_goal.features.pipeline import build_store
from keiba_goal.features.registry import Group, spec_maker

FIX = Path(__file__).parent / "fixtures" / "features"
# 取消・競走中止を含む日 (T-6 の無いレースは snapshots で各日に 1 つ作る)
DATES = (20220108, 20220227, 20220320)


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return pl.read_parquet(FIX / "history.parquet")


@pytest.fixture(scope="module")
def snapshots(history) -> pl.DataFrame:
    """fixture のスナップショットから、各日の最初のレースの T-6 を消す (field_dec の代替の経路を必ず通す)。"""
    snaps = pl.read_parquet(FIX / "ts_snap" / "2022.parquet")
    first = history.filter(pl.col("date").is_in(DATES)).group_by("date").agg(pl.col("race_key").min())["race_key"]
    return snaps.filter(~(pl.col("race_key").is_in(first.implode()) & (pl.col("x") == 6)))


@pytest.fixture(scope="module")
def full(history, snapshots) -> pl.DataFrame:
    return build_store(history, snapshots, years=(2022,))


@pytest.fixture(scope="module", params=DATES)
def result(request, history, snapshots, full):
    return check_day(history, snapshots, request.param, full=full, seed=request.param)


def test_fixture_days_cover_the_edge_cases(history, full):
    for d in DATES:
        day = history.filter(pl.col("date") == d)
        assert day.height > 300
        assert full.filter(pl.col("date") == d)["rd_field_fallback"].max() == 1.0
    days = history.filter(pl.col("date").is_in(DATES))
    assert days.filter(pl.col("error_code").is_in([1, 2])).height >= 3
    assert days.filter(pl.col("error_code") == 4).height >= 1


@pytest.mark.parametrize("group", discover(), ids=lambda g: g.name)
def test_group_is_as_of(group: Group, result):
    assert result.rows > 300
    assert result.failures(group.feature_names) == {}


def test_truncation_and_perturbation_touch_only_the_intended_rows(history):
    d = DATES[1]
    t = truncate_at(history, d)
    assert t["date"].max() == d
    assert t.filter(pl.col("date") == d)["rank"].null_count() == t.filter(pl.col("date") == d).height
    assert t.filter(pl.col("date") < d).equals(history.filter(pl.col("date") < d))
    p = perturb_from(history, d, seed=1)
    assert p.filter(pl.col("date") < d).equals(history.filter(pl.col("date") < d))
    after = history.filter(pl.col("date") >= d)
    changed = p.filter(pl.col("date") >= d).select((pl.col("time") != after["time"]).mean()).item()
    assert changed > 0.9


# 漏れのあるグループを harness が見つけること (検査そのものの検査)


def _own_rank(ctx):
    return ctx.history("rank").select("race_id", f32(pl.col("rank"), "leak_x")).collect()


def _next_rank(ctx):
    ev = lags.events(ctx.history("rank", "error_code").filter(started()))
    nxt = ev.with_columns(pl.col("rank").shift(-1).over("id").alias("_next"))
    return ctx.target().join(nxt.select("race_id", "_next"), on="race_id", how="left").select(
        "race_id", f32(pl.col("_next"), "leak_x")
    ).collect()


def _same_day_wins(ctx):
    rows = ctx.history("rank", "venue_code")
    wins = rows.group_by("venue_code", "date").agg((pl.col("rank") == 1).sum().alias("_w"))
    return ctx.target("venue_code").join(wins, on=["venue_code", "date"], how="left").select(
        "race_id", f32(pl.col("_w"), "leak_x")
    ).collect()


def _today_via_history(ctx):
    """段階 A なのに、履歴の D の行 (対象行そのもの) から当日の馬体重・騎手を読む。"""
    today = ctx.history("weight", "jockey_id").select(
        "race_id", (pl.col("weight") + pl.col("jockey_id").cast(pl.Float32, strict=False).fill_null(0)).alias("_v"))
    return ctx.target().join(today, on="race_id", how="left").select("race_id", f32(pl.col("_v"), "leak_x")).collect()


def _day_mean_weight(ctx):
    """段階 B で、同じ日・同じ場の全レースの馬体重の平均 (他のレースの当日の値) を使う。"""
    rows = ctx.target("weight", "venue_code")
    return rows.select("race_id", f32(pl.col("weight").mean().over("date", "venue_code"), "leak_x")).collect()


LEAKS = {
    "own": (_own_rank, "A", [], ["rank", "error_code"], "truncated"),
    "future": (_next_rank, "A", [], ["rank", "error_code"], "truncated"),
    "same_day": (_same_day_wins, "A", ["venue_code"], ["rank", "venue_code"], "truncated"),
    "today_b_columns": (_today_via_history, "A", [], ["weight", "jockey_id"], "same_day"),
    "other_races": (_day_mean_weight, "B", ["weight", "venue_code"], [], "same_day"),
}


@pytest.mark.parametrize("kind", list(LEAKS))
def test_harness_detects_leaks(kind, history, snapshots):
    build, stage, target, hist_cols, check = LEAKS[kind]
    spec = spec_maker("leak", stage)("leak_x", "num", target, hist_cols)
    res = check_day(history, snapshots, DATES[1], groups=(Group("leak", stage, (spec,), build),))
    assert check in res.failures(), f"漏れを検出できていない: {res.failures()}"


def test_same_day_scramble_changes_every_stage_b_column(history, snapshots):
    d = DATES[1]
    hist = with_live_columns(history.lazy()).collect()
    scrambled, snaps = scramble_from(hist, snapshots, d, seed=3)
    even = (pl.col("date") == d) & (pl.col("race_key") % 100 % 2 == 0)
    before, after = hist.filter(even).sort("race_id"), scrambled.filter(even).sort("race_id")
    for c in ("jockey_id", "basis_weight", "weight", "inc_dec", "blinker", "state", "weather", "horse_N", "scratched"):
        changed = after.select((~pl.col(c).eq_missing(before[c])).mean()).item()
        assert changed > 0.8, c
    assert scrambled.filter(~even).equals(hist.filter(~even))
    keys = hist.filter(even)["race_key"].unique().implode()
    assert not snaps.filter(pl.col("race_key").is_in(keys) & pl.col("win").is_finite())["win"].equals(
        snapshots.filter(pl.col("race_key").is_in(keys) & pl.col("win").is_finite())["win"])
    assert snaps.filter(~pl.col("race_key").is_in(keys)).equals(snapshots.filter(~pl.col("race_key").is_in(keys)))


def test_scratched_is_taken_from_history_before_nulling(history):
    d = DATES[1]
    t = truncate_at(with_live_columns(history.lazy()).collect(), d)
    day = t.filter(pl.col("date") == d)
    assert day["error_code"].null_count() == day.height
    assert day["scratched"].sum() == history.filter((pl.col("date") == d) & pl.col("error_code").is_in([1, 2])).height
