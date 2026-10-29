"""グループの実行・出力の検査・補助出力の作成と、メモリ上の特徴量作成 (as-of テスト・ライブ・小さな検査用)。

ディスクに書く batch (build.py / store.py) も、ここの run_group と HelperCache を同じように使う。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import polars as pl

from keiba_goal.features.context import Ctx
from keiba_goal.features.decision import decision_frame, with_live_columns
from keiba_goal.features.fingerprint import group_code_sha
from keiba_goal.features.groups import discover
from keiba_goal.features.io import write_parquet
from keiba_goal.features.registry import (
    KEY_COLUMNS,
    LIVE_COLUMNS,
    OUTPUT_KEYS,
    Group,
    Helper,
    RegistryError,
    check_registry,
    ordered,
)

FeatureSource = Callable[[str, Sequence[str]], pl.LazyFrame]


class GroupOutputError(ValueError):
    pass


def _lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return df.lazy() if isinstance(df, pl.DataFrame) else df


def validate_output(group: Group, out: pl.DataFrame | pl.LazyFrame, expected: pl.DataFrame) -> pl.DataFrame:
    """build の出力を検査し、expected (race_id の列) の行・順にそろえる。null は NaN にする。

    列 = race_id (Int64) + specs の列 (Float32) ちょうど、race_id は一意、expected の行がすべてある、±inf が無い。
    幅の広いグループ (past_runs で 1.1M 行 × 95 列) でも写しを増やさないよう、並びが同じなら並べ替えない。
    """
    df = out.collect() if isinstance(out, pl.LazyFrame) else out
    names = group.feature_names
    if sorted(df.columns) != sorted(["race_id", *names]):
        extra = sorted(set(df.columns) - {"race_id", *names})
        missing = sorted({"race_id", *names} - set(df.columns))
        raise GroupOutputError(f"{group.name}: 列が宣言と違う (余分 {extra}, 不足 {missing})")
    if df.schema["race_id"] != pl.Int64:
        raise GroupOutputError(f"{group.name}: race_id は Int64 ({df.schema['race_id']})")
    if bad := [c for c in names if df.schema[c] != pl.Float32]:
        raise GroupOutputError(f"{group.name}: Float32 でない列 {bad}")
    ids = df.select("race_id")
    if ids["race_id"].is_duplicated().any():
        raise GroupOutputError(f"{group.name}: race_id が重複している")
    if n_missing := expected.join(ids, on="race_id", how="anti").height:
        raise GroupOutputError(f"{group.name}: 対象行 {expected.height:,} のうち {n_missing:,} 行が無い")
    if not ids["race_id"].equals(expected["race_id"]):
        df = expected.select("race_id").join(df, on="race_id", how="left", maintain_order="left")
    if names and (inf := [c for c in names if df[c].is_infinite().any()]):
        raise GroupOutputError(f"{group.name}: ±inf を含む列 {inf} (値の無いところは NaN にする)")
    nulls = [c for c in names if df[c].null_count()]
    return df.with_columns(pl.col(c).fill_null(float("nan")) for c in nulls).select("race_id", *names)


class HelperCache:
    """補助出力を 1 回の作成につき 1 回だけ作る。directory があれば parquet に書いて遅延読み込みで返す。

    meta (入力の sha256 など) は書くファイルのメタデータに入れる (持ち主のグループのコードの sha256 も付ける)。
    """

    def __init__(self, groups: Sequence[Group], history: pl.LazyFrame, directory: Path | None = None,
                 meta: Mapping[str, object] | None = None):
        self._defs: dict[str, tuple[Group, Helper]] = {h.name: (g, h) for g in groups for h in g.helpers}
        self._history = history
        self._dir = directory
        self._meta = dict(meta or {})
        self._cache: dict[str, pl.LazyFrame] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._defs)

    def get(self, name: str) -> pl.LazyFrame:
        if name not in self._cache:
            owner, h = self._defs[name]
            ctx = Ctx(f"helper:{name}", history=self._history, history_columns=h.history_sources)
            df = h.build(ctx)
            df = df.collect() if isinstance(df, pl.LazyFrame) else df
            if self._dir is None:
                self._cache[name] = df.lazy()
            else:
                path = self._dir / f"{name}.parquet"
                meta = {"helper": name, "group": owner.name, "own_post_race": h.own_post_race,
                        "code_sha256": group_code_sha(owner), **self._meta}
                write_parquet(df, path, meta)
                self._cache[name] = pl.scan_parquet(path)
        return self._cache[name]

    def build_all(self) -> None:
        for name in self._defs:
            self.get(name)


def helper_names(group: Group, groups_by_name: Mapping[str, Group]) -> tuple[str, ...]:
    own = [h.name for h in group.helpers]
    return tuple(own + [h.name for r in group.requires for h in groups_by_name[r].helpers])


def feature_groups(group: Group, groups_by_name: Mapping[str, Group]) -> dict[str, str]:
    provided = {s.name: r for r in group.requires for s in groups_by_name[r].specs}
    return {f: provided[f] for f in group.inputs}


def run_group(
    group: Group,
    groups_by_name: Mapping[str, Group],
    *,
    history: pl.LazyFrame,
    target: pl.LazyFrame,
    expected: pl.DataFrame,
    helpers: HelperCache,
    features: FeatureSource,
    years: Sequence[int],
) -> pl.DataFrame:
    """1 グループを作って検査する。target は段階 A なら対象行の履歴、段階 B なら判断時刻の枠。"""
    ctx = Ctx(
        group.name,
        history=history,
        history_columns=group.history_columns,
        target=target,
        target_columns=group.target_columns,
        helpers=helpers.get,
        helper_names=helper_names(group, groups_by_name),
        features=features,
        feature_groups=feature_groups(group, groups_by_name),
        years=years,
    )
    return validate_output(group, group.build(ctx), expected)


def target_rows(history: pl.LazyFrame, years: Sequence[int], dates: Sequence[int] | None = None) -> pl.LazyFrame:
    lf = history.filter((pl.col("date") // 10_000).is_in(list(years)))
    return lf.filter(pl.col("date").is_in(list(dates))) if dates is not None else lf


def frame_columns(groups: Sequence[Group]) -> list[str]:
    """判断時刻の枠に載せる履歴の列: キー + 段階 B のグループが対象行から読む発走前の列 + scratched。"""
    cols = {c for g in groups if g.stage == "B" for c in g.target_columns if c not in LIVE_COLUMNS}
    return [*KEY_COLUMNS, *sorted(cols - set(KEY_COLUMNS)), "scratched"]


def build_frame(history: pl.LazyFrame, groups: Sequence[Group], year: int, dates: Sequence[int] | None,
                snapshots: pl.DataFrame | None, field_open: pl.DataFrame | None) -> pl.DataFrame:
    rows = target_rows(history, [year], dates).select(frame_columns(groups)).collect()
    return decision_frame(rows.sort(["date", "race_key", "horse_num"]), snapshots, field_open)


def memory_source(outputs: Mapping[str, pl.DataFrame]) -> FeatureSource:
    return lambda group, names: outputs[group].lazy().select("race_id", *names)


def checked_groups(groups: Sequence[Group] | None) -> tuple[Group, ...]:
    """groups (None なら登録済みの全グループ) を依存順で。明示されたグループにもレジストリの検査をかける。"""
    if groups is None:
        return discover()
    by_name = {g.name: g for g in groups}
    for g in groups:
        if missing := [r for r in g.requires if r not in by_name]:
            raise RegistryError(f"{g.name}: requires {missing} が groups に無い")
    check_registry(groups)
    return ordered(groups)


def stage_a(history: pl.LazyFrame, groups: Sequence[Group], helpers: HelperCache, years: Sequence[int],
            dates: Sequence[int] | None) -> dict[str, pl.DataFrame]:
    """段階 A のグループの出力 (グループ名 → race_id + 特徴量、対象年 (と対象日) の行)。"""
    by_name = {g.name: g for g in groups}
    target = target_rows(history, years, dates)
    expected = target.select("race_id").collect()
    outputs: dict[str, pl.DataFrame] = {}
    for g in (g for g in groups if g.stage == "A"):
        outputs[g.name] = run_group(g, by_name, history=history, target=target, expected=expected,
                                    helpers=helpers, features=memory_source(outputs), years=years)
    return outputs


def stage_b(history: pl.LazyFrame, groups: Sequence[Group], helpers: HelperCache, frame: pl.DataFrame,
            a_outputs: Mapping[str, pl.DataFrame], year: int) -> pl.DataFrame:
    """判断時刻の枠 frame の行について段階 B のグループを作り、段階 A の出力と組み立てる (OUTPUT_KEYS + 全特徴量)。"""
    by_name = {g.name: g for g in groups}
    expected = frame.select("race_id")
    outputs = {n: df.join(expected, on="race_id", how="semi") for n, df in a_outputs.items()}
    for g in (g for g in groups if g.stage == "B"):
        outputs[g.name] = run_group(g, by_name, history=history, target=frame.lazy(), expected=expected,
                                    helpers=helpers, features=memory_source(outputs), years=(year,))
    return assemble(frame.select(OUTPUT_KEYS), groups, outputs)


def build_store(
    history: pl.DataFrame | pl.LazyFrame,
    snapshots: pl.DataFrame | None = None,
    *,
    years: Sequence[int],
    dates: Sequence[int] | None = None,
    field_open: pl.DataFrame | None = None,
    groups: Sequence[Group] | None = None,
) -> pl.DataFrame:
    """メモリ上で段階 A → B を作り、OUTPUT_KEYS + 全特徴量 (Float32) を (date, race_key, horse_num) 順で返す。

    batch の store.py と同じグループ関数を通す (as-of テスト・ライブ・パリティ検査の基準)。
    """
    if not years:
        raise ValueError("years が空です")
    groups = checked_groups(groups)
    hist = with_live_columns(_lazy(history))
    helpers = HelperCache(groups, hist)
    a_outputs = stage_a(hist, groups, helpers, years, dates)
    return pl.concat([
        stage_b(hist, groups, helpers, build_frame(hist, groups, year, dates, snapshots, field_open), a_outputs, year)
        for year in years
    ])


def assemble(keys: pl.DataFrame, groups: Sequence[Group], outputs: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """キー + 各グループの特徴量 (依存順)。どのグループも keys の行をちょうど持つこと。"""
    out = keys
    for g in ordered(groups):
        part = outputs[g.name]
        if part.height != keys.height or keys.join(part, on="race_id", how="anti").height:
            raise GroupOutputError(f"{g.name}: 行がキーの {keys.height:,} 行と一致しない ({part.height:,} 行)")
        out = out.join(part, on="race_id", how="left", maintain_order="left")
    return out
