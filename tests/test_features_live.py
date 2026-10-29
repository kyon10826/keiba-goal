"""CLI で作った batch のストアと、ライブの入口 (build_rows・朝/判断時刻の分割 prepare_morning/decide) の出力が
同じ日について一致すること (パリティ P3-lite) と、ストアの作成の管理 (古いファイル・入力の固定・点検)。

fixture (tests/fixtures/features/) を一時ディレクトリの data/interim に置いて `--stage all --years 2022` を通し、
その日の出馬表の行 (レース後の列 null + 取消の印) とスナップショットから作った行と比べる。
"""

import json
import os
import shutil
from pathlib import Path

import polars as pl
import pytest
import yaml

from keiba_goal.config import load_config
from keiba_goal.data.schema import POST_RACE_COLUMNS
from keiba_goal.features import build
from keiba_goal.features import io as io_mod
from keiba_goal.features import store as store_mod
from keiba_goal.features.asof import diff_counts, scramble_same_day
from keiba_goal.features.fingerprint import PACKAGE_ROOT, group_modules, module_files
from keiba_goal.features.groups import discover
from keiba_goal.features.io import file_sha256, pin_file, read_meta
from keiba_goal.features.labels import LABEL_COLUMNS
from keiba_goal.features.live import build_rows, decide, prepare_morning
from keiba_goal.features.registry import OUTPUT_KEYS
from keiba_goal.features.store import Layout, StaleError, current_inputs, store_problems

FIX = Path(__file__).parent / "fixtures" / "features"
DATES = (20220227, 20220320)


def write_config(root: Path) -> Path:
    cfg = root / "cfg.yaml"
    paths = {"record_dir": str(root), "ts_odds_dir": str(root), "spec_xlsx": str(root), "data_root": str(root / "data")}
    cfg.write_text(yaml.safe_dump({"paths": paths, "features": {"stage_a_min_year": 2003, "years": [2022, 2022]}}))
    return cfg


def make_root(root: Path) -> Path:
    interim = root / "data" / "interim"
    (interim / "ts_snap").mkdir(parents=True)
    shutil.copy(FIX / "history.parquet", interim / "history.parquet")
    shutil.copy(FIX / "ts_snap" / "2022.parquet", interim / "ts_snap" / "2022.parquet")
    return write_config(root)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("store")
    cfg = make_root(root)
    assert build.main(["--stage", "all", "--years", "2022", "--config", str(cfg)]) == 0
    return root / "data"


def copy_built(built: Path, root: Path) -> Path:
    """作成済みのストア (built) の写しを root に作り、その設定ファイルを返す (書き換えるテスト用)。"""
    shutil.copytree(built, root / "data")
    return write_config(root)


def features_now() -> list[str]:
    return [s.name for g in discover() for s in g.specs]


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return pl.read_parquet(FIX / "history.parquet")


@pytest.fixture(scope="module")
def snapshots() -> pl.DataFrame:
    return pl.read_parquet(FIX / "ts_snap" / "2022.parquet")


def test_cli_writes_the_store_layout(built: Path, history: pl.DataFrame):
    groups = discover()
    store = pl.read_parquet(built / "features" / "B" / "year=2022.parquet")
    features = [s.name for g in groups for s in g.specs]
    assert store.columns == [*OUTPUT_KEYS, *features]
    assert all(store.schema[c] == pl.Float32 for c in features)
    assert store.height == history.filter((pl.col("date") // 10_000) == 2022).height
    assert store.equals(store.sort(["date", "race_key", "horse_num"]))
    for g in groups:
        path = built / "features" / ("A" if g.stage == "A" else "B/groups") / g.name / "year=2022.parquet"
        assert read_meta(path)["features"] == list(g.feature_names)
    labels = pl.read_parquet(built / "features" / "labels" / "year=2022.parquet")
    assert labels.columns == list(LABEL_COLUMNS) and labels["race_id"].equals(store["race_id"])
    reg = json.loads((built / "features" / "registry.json").read_text(encoding="utf-8"))
    assert [f["name"] for f in reg["features"]] == features
    helper = read_meta(built / "state" / "helpers" / "speed_fig.parquet")
    assert helper["group"] == "speed" and helper["code_sha256"] and helper["inputs"]["history"]
    assert (built / "state" / "inputs" / "history.parquet").exists()
    assert (built / "state" / "inputs" / "ts_snap" / "2022.parquet").exists()
    assert not list((built / "state" / "inputs").rglob("*.tmp"))
    meta = read_meta(built / "features" / "B" / "year=2022.parquet")
    assert meta["missing_groups"] == {} and set(meta["groups"]) == {g.name for g in groups}
    assert meta["inputs"] == read_meta(built / "features" / "labels" / "year=2022.parquet")["inputs"]


def card_rows(history: pl.DataFrame, d: int) -> pl.DataFrame:
    """その日の出馬表の行: 発走前の列 + 取消の印、レース後の列は null。"""
    day = history.filter(pl.col("date") == d).with_columns(pl.col("error_code").is_in([1, 2]).alias("scratched"))
    return day.with_columns(pl.lit(None, dtype=day.schema[c]).alias(c) for c in POST_RACE_COLUMNS)


@pytest.mark.parametrize("d", DATES)
def test_live_rows_equal_the_batch_store(d, built, history, snapshots):
    card = card_rows(history, d)
    snaps = snapshots.join(card.select("race_key").unique(), on="race_key", how="semi")
    live = build_rows(history.filter(pl.col("date") < d), card, snaps)
    batch = pl.read_parquet(built / "features" / "B" / "year=2022.parquet").filter(pl.col("date") == d)
    assert live.select(OUTPUT_KEYS).equals(batch.select(OUTPUT_KEYS))
    features = [c for c in batch.columns if c not in OUTPUT_KEYS]
    assert live.columns == batch.columns
    assert {k: v for k, v in diff_counts(batch, live, features).items() if v} == {}


@pytest.fixture(scope="module")
def mornings(history) -> dict:
    """各日の朝の状態。朝の出馬表は判断時刻と当日の列 (騎手・斤量・馬場・天気・ブリンカー・取消) が違い、馬体重は未発表。"""
    out = {}
    for d in DATES:
        early = scramble_same_day(card_rows(history, d), seed=d).with_columns(
            pl.lit(None, dtype=pl.Float32).alias("weight"), pl.lit(None, dtype=pl.Float32).alias("inc_dec"))
        out[d] = prepare_morning(history.filter(pl.col("date") < d), early)
    return out


@pytest.mark.parametrize("d", DATES)
def test_morning_state_and_decision_time_equal_the_batch_store(d, built, history, snapshots, mornings):
    """朝に作った段階 A と補助出力を、判断時刻 (最新の出馬表) の段階 B に使っても batch と同じ。"""
    card = card_rows(history, d)
    snaps = snapshots.join(card.select("race_key").unique(), on="race_key", how="semi")
    batch = pl.read_parquet(built / "features" / "B" / "year=2022.parquet").filter(pl.col("date") == d)
    features = [c for c in batch.columns if c not in OUTPUT_KEYS]
    split = decide(mornings[d], card, snaps)
    assert split.columns == batch.columns and split.select(OUTPUT_KEYS).equals(batch.select(OUTPUT_KEYS))
    assert {k: v for k, v in diff_counts(batch, split, features).items() if v} == {}
    # 1 レースずつ判断しても同じ (段階 B はレースの中だけで決まる)
    for key in card["race_key"].unique().sort().gather([0, 5, 11]).to_list():
        one = decide(mornings[d], card.filter(pl.col("race_key") == key), snaps.filter(pl.col("race_key") == key))
        expected = batch.filter(pl.col("race_key") == key)
        assert one.height == expected.height and one["race_id"].equals(expected["race_id"])
        assert {k: v for k, v in diff_counts(expected, one, features).items() if v} == {}


def test_decide_rejects_rows_the_morning_did_not_have(history, mornings):
    d = DATES[0]
    card = card_rows(history, d)
    key = card["race_key"].min()
    with pytest.raises(ValueError, match="全馬"):
        decide(mornings[d], card.filter((pl.col("race_key") == key) & (pl.col("horse_num") > 1)), None)
    with pytest.raises(ValueError, match="日付"):
        decide(mornings[d], card_rows(history, DATES[1]), None)
    other = card.filter(pl.col("race_key") == key).with_columns((pl.col("race_id") + 50).alias("race_id"))
    with pytest.raises(ValueError, match="朝の出馬表に無い"):
        decide(mornings[d], pl.concat([card, other]), None)


def test_live_rejects_bad_inputs(history, snapshots):
    d = DATES[0]
    card = card_rows(history, d)
    with pytest.raises(ValueError, match="レース後の列の値"):
        build_rows(history.filter(pl.col("date") < d), card.with_columns(pl.lit(1, dtype=pl.Int16).alias("rank")), None)
    with pytest.raises(ValueError, match="scratched"):
        build_rows(history.filter(pl.col("date") < d), card.drop("scratched"), None)
    with pytest.raises(ValueError, match="以降の行"):
        build_rows(history.filter(pl.col("date") <= d), card, None)


def test_stale_stage_a_files_are_refused(built: Path, tmp_path: Path):
    cfg = copy_built(built, tmp_path)
    hist_path = tmp_path / "data" / "interim" / "history.parquet"
    pl.read_parquet(hist_path).slice(1).write_parquet(hist_path)      # データ層が history を作り直した
    with pytest.raises(StaleError, match="history が違う"):
        build.main(["--stage", "B", "--years", "2022", "--config", str(cfg)])
    args = ["--stage", "B", "--years", "2022", "--groups", "race_day", "--allow-missing", "--config", str(cfg)]
    assert build.main(args) == 0
    meta = read_meta(tmp_path / "data" / "features" / "B" / "year=2022.parquet")
    # 段階 B のグループは作り直され (horse_day は古いので自動で)、段階 A とそれに依存する relative は外れる
    assert set(meta["groups"]) == {"race_day", "horse_day"}
    assert set(meta["missing_groups"]) == {g.name for g in discover()} - {"race_day", "horse_day"}
    layout = Layout.from_config(load_config(cfg))
    assert "外したグループがある" in " ".join(store_problems(layout, [2022], features_now())[2022])


def test_stage_b_groups_are_rebuilt_when_ts_snap_changes(built: Path, tmp_path: Path):
    """データ層が ts_snap/{Y} を作り直した後の部分的な `--stage B --groups race_day` で、古い段階 B のグループの
    ファイルを混ぜない (全部が新しい判断時刻の枠から作られ、B/year= のメタデータと一致する)。"""
    cfg = copy_built(built, tmp_path)
    layout = Layout.from_config(load_config(cfg))
    before = pl.read_parquet(layout.b_file(2022))
    snap_path = layout.ts_snap_dir / "2022.parquet"
    snaps = pl.read_parquet(snap_path)
    races = snaps.filter((pl.col("x") == 6) & (pl.col("horse_num") == 1) & pl.col("win").is_finite())
    drop = pl.col("race_key").is_in(races["race_key"].unique().sort().head(40).implode()) & (pl.col("x") == 6) & (
        pl.col("horse_num") == 1)
    tmp = snap_path.with_name("2022.parquet.new")
    snaps.filter(~drop).write_parquet(tmp)
    os.replace(tmp, snap_path)                    # データ層の書き方 (一時ファイル + rename)
    assert build.main(["--stage", "B", "--years", "2022", "--groups", "race_day", "--config", str(cfg)]) == 0
    sha = file_sha256(snap_path)
    for g in discover():
        if g.stage == "B":
            assert read_meta(layout.b_group_file(g.name, 2022))["inputs"]["ts_snap"] == sha, g.name
    assert read_meta(layout.b_file(2022))["inputs"]["ts_snap"] == sha
    store = pl.read_parquet(layout.b_file(2022))
    assert (store["rd_field_size"] != before["rd_field_size"]).sum() > 400        # 40 レースの頭数が 1 減った
    sized = store.filter(pl.col("rd_field_size") > 0)
    gap = (sized["hd_horse_num_ratio"] - sized["horse_num"].cast(pl.Float32) / sized["rd_field_size"]).abs()
    assert gap.max() < 1e-6
    # labels はまだ古い入力 → 点検で見つかり、`--stage labels` で解消する
    problems = store_problems(layout, [2022], features_now(), current_inputs(layout, [2022]))[2022]
    assert any("B と labels の入力が違う" in p for p in problems)
    assert build.main(["--stage", "labels", "--years", "2022", "--config", str(cfg)]) == 0
    assert store_problems(layout, [2022], features_now(), current_inputs(layout, [2022]))[2022] == []


def test_files_built_by_other_code_are_stale(built: Path, tmp_path: Path, monkeypatch):
    cfg = copy_built(built, tmp_path)
    real = store_mod.group_code_sha
    monkeypatch.setattr(store_mod, "group_code_sha", lambda g: "changed" if g.name == "speed" else real(g))
    with pytest.raises(StaleError, match="A/speed/.*コードが変わった"):
        build.main(["--stage", "B", "--years", "2022", "--config", str(cfg)])
    assert build.main(["--stage", "A", "--years", "2022", "--groups", "speed", "--config", str(cfg)]) == 0
    with pytest.raises(StaleError, match="A/aggregates/.*依存先 speed が変わった"):
        build.main(["--stage", "B", "--years", "2022", "--config", str(cfg)])


def test_store_problems_flag_inputs_the_data_layer_replaced(built: Path, tmp_path: Path):
    cfg = copy_built(built, tmp_path)
    layout = Layout.from_config(load_config(cfg))
    assert store_problems(layout, [2022], features_now(), current_inputs(layout, [2022]))[2022] == []
    snap_path = layout.ts_snap_dir / "2022.parquet"
    pl.read_parquet(snap_path).head(1000).write_parquet(snap_path)
    problems = store_problems(layout, [2022], features_now(), current_inputs(layout, [2022]))[2022]
    assert any("今の data/interim の入力と違う" in p for p in problems)
    assert store_problems(layout, [2022], features_now()[:-1])[2022] == ["B/year= の列が今のレジストリと違う (`--stage B` で組み立て直す)"]


def test_code_hash_covers_the_imported_modules():
    by_name = {g.name: g for g in discover()}

    def files(name: str) -> set[str]:
        return {str(p.relative_to(PACKAGE_ROOT)) for m in group_modules(by_name[name]) for p in module_files(m)}

    past = files("past_runs")
    assert {"features/groups/past_runs.py", "features/groups/speed.py", "features/lags.py", "features/exprs.py",
            "data/schema.py"} <= past
    assert not {"features/registry.py", "features/context.py", "config.py", "features/store.py"} & past
    assert {"features/vocab.py", "features/course_vocab.json"} <= files("race_static")
    assert "features/decision.py" in files("horse_day") and "features/decision.py" not in files("speed")


def test_pinned_inputs_keep_their_content_when_the_source_is_replaced(tmp_path: Path):
    src, pinned = tmp_path / "src.parquet", tmp_path / "state" / "pinned.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(src)
    pin_file(src, pinned)
    pin_file(src, pinned)                     # 2 回目 (同じ inode) でも一時ファイルを残さない
    assert sorted(p.name for p in pinned.parent.iterdir()) == ["pinned.parquet"]
    tmp = tmp_path / "new.parquet"
    pl.DataFrame({"a": [2]}).write_parquet(tmp)
    os.replace(tmp, src)                      # データ層の書き方 (一時ファイル + rename)
    assert pl.read_parquet(pinned)["a"].to_list() == [1]
    assert pl.read_parquet(src)["a"].to_list() == [2]
    pin_file(src, pinned)
    assert pl.read_parquet(pinned)["a"].to_list() == [2]
    assert sorted(p.name for p in pinned.parent.iterdir()) == ["pinned.parquet"]


def test_reads_are_retried_while_a_file_is_being_replaced(tmp_path: Path, monkeypatch):
    path = tmp_path / "x.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)
    calls = {"n": 0}
    real = pl.read_parquet

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("一時的に読めない")
        return real(*args, **kwargs)

    monkeypatch.setattr(io_mod, "RETRY_SEC", 0.0)
    monkeypatch.setattr(io_mod.pl, "read_parquet", flaky)
    assert io_mod.read_parquet_retry(path)["a"].to_list() == [1] and calls["n"] == 3
