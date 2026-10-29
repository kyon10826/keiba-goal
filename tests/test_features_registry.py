"""特徴量レジストリの検査 (design §4.1・§12-4): 禁止列・Float32・キー列なし・目的変数の名前・接頭辞・ID・段階・依存。"""

import json
import re
from pathlib import Path

import polars as pl
import pytest

from keiba_goal.data.schema import POST_RACE_COLUMNS, PRE_RACE_COLUMNS
from keiba_goal.data.ts_odds import SNAPSHOT_X as DATA_SNAPSHOT_X
from keiba_goal.features.context import AccessError, Ctx
from keiba_goal.features.groups import discover
from keiba_goal.features.labels import LABEL_COLUMNS
from keiba_goal.features.pipeline import GroupOutputError, build_store, validate_output
from keiba_goal.features.registry import (
    DIAGNOSTIC_COLUMNS,
    FINAL_ODDS_COLUMNS,
    ID_COLUMNS,
    KEY_COLUMNS,
    NON_FEATURE_NAMES,
    SNAPSHOT_X,
    STAGE_B_COLUMNS,
    STATIC_COLUMNS,
    UNUSABLE_COLUMNS,
    FeatureSpec,
    Group,
    Helper,
    RegistryError,
    check_registry,
    registry_json,
    with_requirements,
)
from keiba_goal.features.synth import history_frame

DOCS = Path(__file__).parents[1] / "docs" / "features.md"


def _noop(ctx):
    raise AssertionError("not built")


HELPER = Helper("h", _noop, ("rank",), own_post_race=True)


def group(name="g", stage="A", specs=None, **kw) -> Group:
    specs = specs if specs is not None else (FeatureSpec(f"{name}_x", name, stage, "num", (), ()),)
    return Group(name, stage, tuple(specs), _noop, **kw)


def spec(name="g_x", group="g", stage="A", kind="num", target=(), history=(), keys=()) -> FeatureSpec:
    return FeatureSpec(name, group, stage, kind, tuple(target), tuple(history), keys=tuple(keys))


def test_registered_groups_pass_the_checks():
    groups = discover()
    check_registry(groups)
    names = [s.name for g in groups for s in g.specs]
    assert len(names) == len(set(names))
    assert not set(names) & (NON_FEATURE_NAMES | set(KEY_COLUMNS) | set(LABEL_COLUMNS))
    prefixes = [p for g in groups for p in g.prefixes]
    assert len(prefixes) == len(set(prefixes)) == len(groups)
    for g in groups:
        for s in g.specs:
            assert s.doc, f"{s.name}: doc (registry.json の説明) が空"
            assert not set(s.target_sources) & POST_RACE_COLUMNS, s.name
            assert not set(s.target_sources) & ID_COLUMNS, s.name
            assert not (set(s.target_sources) | set(s.history_sources)) & (DIAGNOSTIC_COLUMNS | FINAL_ODDS_COLUMNS), s.name
            assert "leg" not in s.history_sources
            if g.stage == "A":
                assert set(s.target_sources) <= STATIC_COLUMNS, s.name


def test_every_pre_race_column_is_classified():
    assert PRE_RACE_COLUMNS == STATIC_COLUMNS | STAGE_B_COLUMNS | (UNUSABLE_COLUMNS & PRE_RACE_COLUMNS)
    assert not STATIC_COLUMNS & STAGE_B_COLUMNS
    for col in ("jockey_id", "basis_weight", "weight", "inc_dec", "blinker", "state", "weather"):
        assert col in STAGE_B_COLUMNS


def test_snapshot_x_matches_the_data_layer():
    assert SNAPSHOT_X == tuple(DATA_SNAPSHOT_X)


@pytest.mark.parametrize(
    "bad, message",
    [
        (spec(target=["rank"]), "レース後の列"),
        (spec(stage="B", target=["win_odds"]), "レース後の列"),
        (spec(stage="B", target=["vote_share"]), "診断専用"),
        (spec(history=["staleness_min"]), "診断専用"),
        (spec(stage="B", target=["snap_time"]), "診断専用"),
        (spec(history=["win_final"]), "確定オッズ"),
        (spec(history=["leg"]), "使わない列"),
        (spec(target=["source"]), "使わない列"),
        (spec(target=["jockey_id"]), "段階 A の対象行から読めない"),
        (spec(target=["weight"]), "段階 A の対象行から読めない"),
        (spec(target=["win_6"]), "段階 A の対象行から読めない"),
        (spec(history=["in_field_dec"]), "履歴に無い列"),
        (spec(name="race_id"), "キー列"),
        (spec(name="jockey_id"), "キー列"),
        (spec(name="dist"), "キー列"),
        (spec(name="Bad-Name"), "接頭辞_名前"),
        (spec(name="gx"), "接頭辞_名前"),
        (spec(name="y_win"), "目的変数"),
        (spec(name="y_extra"), "目的変数用"),
        (spec(name="n_paid_places"), "目的変数の列"),
        (spec(name="eligible"), "目的変数の列"),
        (spec(name="field_dec_size"), "目的変数の列"),
        (spec(name="field_open_source"), "目的変数の列"),
        (spec(kind="int"), "kind"),
        (spec(stage="B", kind="cat", target=["jockey_id"]), "ID 列"),
        (spec(stage="B", kind="num", target=["jockey_id"]), "ID 列"),
        (spec(target=["father"]), "ID 列"),
        (spec(stage="B", target=["trainer_id", "dist"]), "ID 列"),
        (spec(stage="B", keys=["weight"]), "keys に書けるのは ID 列"),
        (spec(stage="B", keys=["id"]), "keys に書けるのは ID 列"),
        (spec(keys=["jockey_id"]), "段階 A の対象行から読めない"),
        (spec(group="other"), "group/stage"),
    ],
)
def test_forbidden_specs_are_rejected(bad, message):
    g = group(stage=bad.stage, specs=[bad]) if bad.group == "g" else group(specs=[bad])
    with pytest.raises(RegistryError, match=message):
        check_registry([g])


def test_stage_b_may_read_same_day_and_decision_columns():
    ok = spec(stage="B", target=["weight", "state", "win_6", "in_field_dec", "field_open"], keys=["jockey_id"])
    check_registry([group(stage="B", specs=[ok])])
    check_registry([group(specs=[spec(target=["date"], keys=["father", "mother", "trainer_id"])])])


def test_feature_names_share_one_prefix_per_group():
    check_registry([group(specs=[spec("p1_rank"), spec("p2_rank"), spec("p10_days")])])
    with pytest.raises(RegistryError, match="接頭辞がそろっていない"):
        check_registry([group(specs=[spec("g_x"), spec("h_y")])])
    with pytest.raises(RegistryError, match="接頭辞 g_ はグループ a"):
        check_registry([group("a", specs=[spec("g_x", "a")]), group("b", specs=[spec("g_y", "b")])])
    check_registry([group("a", specs=[spec("p1_x", "a")]), group("b", specs=[spec("pd_x", "b")])])


def test_duplicate_names_and_dependency_errors():
    with pytest.raises(RegistryError, match="特徴量名 g_x が重複"):
        check_registry([group("g"), group("h", specs=[spec(group="h")])])
    with pytest.raises(RegistryError, match="補助出力"):
        check_registry([group("a", specs=[spec("a_x", "a")], helpers=(HELPER,)),
                        group("b", specs=[spec("b_x", "b")], helpers=(HELPER,))])
    with pytest.raises(RegistryError, match="requires の zz が無い"):
        check_registry([group(requires=("zz",))])
    with pytest.raises(RegistryError, match="段階 B の b に依存"):
        check_registry([group("a", requires=("b",)), group("b", stage="B")])
    with pytest.raises(RegistryError, match="循環"):
        check_registry([group("a", requires=("b",)), group("b", requires=("a",))])
    with pytest.raises(RegistryError, match="inputs"):
        check_registry([group("a"), group("b", stage="B", requires=("a",), inputs=("nope",))])


def test_with_requirements_adds_dependencies_in_order():
    groups = discover()
    names = [g.name for g in with_requirements(groups, ["relative"])]
    assert names[-1] == "relative"
    assert {"speed", "past_runs", "aggregates", "horse_static", "horse_day"} <= set(names)
    assert names.index("speed") < names.index("past_runs")


def test_registry_json_lists_every_feature_as_float32():
    groups = discover()
    reg = json.loads(json.dumps(registry_json(groups), ensure_ascii=False))
    assert [f["name"] for f in reg["features"]] == [s.name for g in groups for s in g.specs]
    assert {f["dtype"] for f in reg["features"]} == {"Float32"}
    helpers = {h["name"]: h for g in reg["groups"] for h in g["helpers"]}
    assert helpers["speed_fig"]["own_post_race"] is True
    assert helpers["horse_prev"]["own_post_race"] is False


def _frame(**cols) -> pl.DataFrame:
    return pl.DataFrame({"race_id": pl.Series([1, 2, 3], dtype=pl.Int64), **cols})


def test_validate_output_enforces_columns_dtypes_and_rows():
    g = group(specs=[spec("g_x")])
    expected = pl.DataFrame({"race_id": pl.Series([3, 1, 2], dtype=pl.Int64)})
    good = _frame(g_x=pl.Series([1.0, None, 3.0], dtype=pl.Float32))
    out = validate_output(g, good, expected)
    assert out["race_id"].to_list() == [3, 1, 2]
    assert out["g_x"].is_nan().to_list() == [False, False, True]
    with pytest.raises(GroupOutputError, match="Float32"):
        validate_output(g, _frame(g_x=pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64)), expected)
    with pytest.raises(GroupOutputError, match="列が宣言と違う"):
        validate_output(g, good.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("extra")), expected)
    with pytest.raises(GroupOutputError, match="行が無い"):
        validate_output(g, good.head(2), expected)
    with pytest.raises(GroupOutputError, match="inf"):
        validate_output(g, _frame(g_x=pl.Series([1.0, float("inf"), 3.0], dtype=pl.Float32)), expected)
    with pytest.raises(GroupOutputError, match="重複"):
        validate_output(g, pl.concat([good, good.head(1)]), expected)


def test_ctx_returns_only_declared_columns():
    hist = pl.LazyFrame({c: [1] for c in KEY_COLUMNS} | {"rank": [1], "dist": [1600]})
    ctx = Ctx("g", history=hist, history_columns=["rank"], target=hist, target_columns=["dist"],
              helpers=lambda n: hist, helper_names=["h"])
    assert ctx.history("rank").collect_schema().names() == [*KEY_COLUMNS, "rank"]
    assert ctx.target().collect_schema().names() == list(KEY_COLUMNS)
    with pytest.raises(AccessError, match="dist"):
        ctx.history("dist")
    with pytest.raises(AccessError, match="rank"):
        ctx.target("rank")
    with pytest.raises(AccessError, match="other"):
        ctx.helper("other")
    with pytest.raises(AccessError, match="inputs"):
        ctx.features("sp_last")



def test_explicit_groups_are_checked_too():
    """build_store(groups=...) にもレジストリの検査をかける (discover() を通らない経路)。"""
    bad = Group("bad", "B", (spec("bad_x", "bad", "B", target=["jockey_id"]),), _noop)
    hist = history_frame([{"date": 20210105, "venue_code": 6, "race_num": 1, "horse_num": 1, "id": 1}])
    with pytest.raises(RegistryError, match="ID 列"):
        build_store(hist, years=(2021,), groups=(bad,))


def test_docs_example_group_passes_the_registry_and_builds():
    """docs/features.md §4.2 の例をそのまま実行して登録・作成できること (説明とコードがずれないように)。"""
    text = DOCS.read_text(encoding="utf-8")
    code = re.search(r"### 4\.2 例\n+```python\n(.*?)```", text, re.S).group(1)
    namespace: dict = {"__name__": "docs_example"}
    exec(compile(code, str(DOCS), "exec"), namespace)
    example = namespace["GROUP"]
    check_registry([example])
    rows = [
        {"date": d, "venue_code": 6, "race_num": 1, "horse_num": h, "id": 10 * d % 7 + h, "jockey_id": j,
         "rank": h, "error_code": 0}
        for d, pairs in ((20210105, [(1, "00001"), (2, "00002")]), (20210112, [(1, "00001"), (2, "00001")]))
        for h, j in pairs
    ]
    out = build_store(history_frame(rows), years=(2021,), groups=(example,))
    assert out.filter(pl.col("date") == 20210112).sort("horse_num")[example.specs[0].name].to_list() == [1.0, 1.0]
