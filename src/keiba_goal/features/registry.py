"""特徴量レジストリ (design §4.1): FeatureSpec / Helper / Group の定義と、登録内容の検査。

段階 A (朝): 対象行の「前日に分かっている静的な属性」(STATIC_COLUMNS) と、D より前の履歴行だけで決まる特徴量。
段階 B (判断時刻): 当日変わりうる発走前の列 (STAGE_B_COLUMNS) と判断時刻の列 (LIVE_COLUMNS: 判断スナップショット・
field_dec など) を使う特徴量。docs/features.md に規約と追加手順。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from keiba_goal.data.schema import HISTORY_DTYPES, POST_RACE_COLUMNS, PRE_RACE_COLUMNS

if TYPE_CHECKING:
    from keiba_goal.features.context import Ctx

Stage = Literal["A", "B"]
Kind = Literal["num", "cat", "flag"]
STAGES: tuple[str, ...] = ("A", "B")
KINDS: tuple[str, ...] = ("num", "cat", "flag")

# ctx がいつでも返す行のキー (宣言不要)。B の出力ファイルのキーは OUTPUT_KEYS。
KEY_COLUMNS: tuple[str, ...] = ("race_id", "race_key", "date", "horse_num", "id")
OUTPUT_KEYS: tuple[str, ...] = ("race_id", "race_key", "date", "horse_num")

# 対象行の静的な属性 (前日の出馬表で確定し、当日は変わらない)。段階 A の target_sources はこの中だけ。
STATIC_COLUMNS: frozenset[str] = frozenset({
    "race_id", "race_key", "nk_race_id", "date", "month", "day", "times", "place", "daily", "race_num", "venue_code",
    "surface", "direction", "course_inout", "dist", "track_code", "corner_num",
    "class_code", "class_ord", "age_code", "weight_code",
    "id", "horse", "sex", "age", "waku_num", "horse_num", "father", "mother", "trainer_id",
})
# 当日 (判断時刻まで) に変わりうる発走前の列。段階 B だけが対象行から読める。
STAGE_B_COLUMNS: frozenset[str] = frozenset({
    "jockey_id", "basis_weight", "weight", "inc_dec", "blinker", "state", "weather", "horse_N",
})
# 特徴量の元にしない列 (2 桁の年・TARGET の脚質・行の出どころ)
UNUSABLE_COLUMNS: frozenset[str] = frozenset({"year", "leg", "source"})

SNAPSHOT_X: tuple[int, ...] = (120, 60, 30, 6)
DECISION_X = 6
SNAPSHOT_VALUES: tuple[str, ...] = ("win", "place_lo", "place_hi")
SNAPSHOT_COLUMNS: tuple[str, ...] = tuple(f"{v}_{x}" for x in SNAPSHOT_X for v in SNAPSHOT_VALUES)
# 判断時刻に分かる列 (段階 B の対象行の枠 decision.decision_frame が作る)。ライブでは判断時刻に作り直す。
LIVE_COLUMNS: frozenset[str] = frozenset({
    "scratched", "field_open", "in_field_dec", "field_dec_size", "field_dec_fallback", *SNAPSHOT_COLUMNS,
})
# どこにも使ってはいけない列: ts_snap の診断列 (確定票数・区分 3 の刻印に依存) と確定オッズ (ts_final)
DIAGNOSTIC_COLUMNS: frozenset[str] = frozenset({"vote_share", "snap_time", "staleness_min"})
FINAL_ODDS_COLUMNS: frozenset[str] = frozenset({"win_final", "place_lo_final", "place_hi_final"})
# 高カーディナリティの ID・名前。値として特徴量にしない (as-of 統計で置き換える)。対象行の ID を as-of 統計の
# 結合キーにするときは FeatureSpec.keys に書く (target_sources には書けない)。
ID_COLUMNS: frozenset[str] = frozenset({
    "race_id", "race_key", "nk_race_id", "id", "horse", "jockey_id", "trainer_id", "father", "mother",
})
KEY_SOURCES: frozenset[str] = ID_COLUMNS - set(KEY_COLUMNS)
NON_FEATURE_NAMES: frozenset[str] = ID_COLUMNS | {"date", "horse_num"}
# 目的変数の列 (labels.py) と、判断時刻の枠にだけある列。B/year= や labels と結合するので特徴量の名前にしない。
LABEL_NAMES: tuple[str, ...] = (
    "y_win", "y_place", "started", "eligible", "noise_race", "n_paid_places", "n_paid_source",
    "in_field_dec", "field_dec_size", "field_dec_fallback", "starters",
)
FRAME_ONLY_COLUMNS: frozenset[str] = frozenset({"field_open_source"})

TARGET_SOURCES: dict[str, frozenset[str]] = {
    "A": STATIC_COLUMNS,
    "B": (PRE_RACE_COLUMNS - UNUSABLE_COLUMNS) | LIVE_COLUMNS,
}
HISTORY_SOURCES: frozenset[str] = frozenset(HISTORY_DTYPES) - UNUSABLE_COLUMNS
RESERVED_NAMES: frozenset[str] = (
    NON_FEATURE_NAMES | set(OUTPUT_KEYS) | set(HISTORY_DTYPES) | LIVE_COLUMNS | set(LABEL_NAMES) | FRAME_ONLY_COLUMNS
)
RESERVED_PREFIXES: frozenset[str] = frozenset({"y"})     # 目的変数 (y_win, y_place, …)
# 名前 = 接頭辞 (小文字。後ろに数字を付けてよい: past_runs の p1_ … p5_) + "_" + 残り。1 グループ 1 接頭辞。
NAME_RE = re.compile(r"^([a-z]+)[0-9]*_[a-z0-9_]+$")


@dataclass(frozen=True)
class FeatureSpec:
    """1 列の特徴量。target_sources は対象行から値を読む列、history_sources はそれより前の行から読む列。

    keys: 対象行の ID 列 (KEY_SOURCES: jockey_id・father など) のうち、as-of 統計の結合や前の行との一致の判定に
    だけ使うもの (値そのものは特徴量にしない)。
    """

    name: str
    group: str
    stage: Stage
    kind: Kind = "num"
    target_sources: tuple[str, ...] = ()
    history_sources: tuple[str, ...] = ()
    doc: str = ""
    keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Helper:
    """グループが他のグループに渡す中間表 (特徴量ではない)。履歴だけから 1 回の作成につき 1 回作る。

    own_post_race = True は「行自身のレース後の列を使う値」(例: その行の走破タイムから作るスピード指数)。
    その行より後の行の特徴量にしか使ってはいけない。
    """

    name: str
    build: Callable[[Ctx], pl.DataFrame | pl.LazyFrame]
    history_sources: tuple[str, ...]
    own_post_race: bool
    doc: str = ""


@dataclass(frozen=True)
class Group:
    """特徴量グループ。build(ctx) は race_id (Int64) + specs の列 (すべて Float32) のフレームを返す。

    requires: 依存するグループ名 (その helpers を ctx.helper で、inputs に挙げた特徴量を ctx.features で読める)。
    """

    name: str
    stage: Stage
    specs: tuple[FeatureSpec, ...]
    build: Callable[[Ctx], pl.DataFrame | pl.LazyFrame]
    requires: tuple[str, ...] = ()
    helpers: tuple[Helper, ...] = ()
    inputs: tuple[str, ...] = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    @property
    def target_columns(self) -> frozenset[str]:
        return frozenset(c for s in self.specs for c in (*s.target_sources, *s.keys))

    @property
    def prefixes(self) -> frozenset[str]:
        return frozenset(m.group(1) for s in self.specs if (m := NAME_RE.match(s.name)))

    @property
    def history_columns(self) -> frozenset[str]:
        return frozenset(c for s in self.specs for c in s.history_sources)


class RegistryError(ValueError):
    pass


def spec_maker(group: str, stage: Stage) -> Callable[..., FeatureSpec]:
    """グループ内の FeatureSpec を短く書くための関数 (group と stage を埋める)。"""

    def make(name: str, kind: Kind = "num", target: Iterable[str] = (), history: Iterable[str] = (), doc: str = "",
             keys: Iterable[str] = ()) -> FeatureSpec:
        return FeatureSpec(name, group, stage, kind, tuple(target), tuple(history), doc, tuple(keys))

    return make


def _spec_errors(g: Group, s: FeatureSpec) -> list[str]:
    errors = []
    where = f"{g.name}.{s.name}"
    if (s.group, s.stage) != (g.name, g.stage):
        errors.append(f"{where}: group/stage ({s.group}, {s.stage}) がグループ ({g.name}, {g.stage}) と違う")
    if s.kind not in KINDS:
        errors.append(f"{where}: kind {s.kind!r} は {KINDS} のどれか")
    if not (m := NAME_RE.match(s.name)):
        errors.append(f"{where}: 名前は 接頭辞_名前 (小文字・数字・_、例 jk_win_rate・p1_rank)")
    elif m.group(1) in RESERVED_PREFIXES:
        errors.append(f"{where}: 接頭辞 {m.group(1)}_ は目的変数用 (使えない)")
    if s.name in RESERVED_NAMES:
        errors.append(f"{where}: キー列・元の列・目的変数の列と同じ名前は使えない")
    sources = set(s.target_sources) | set(s.history_sources) | set(s.keys)
    for bad, why in ((DIAGNOSTIC_COLUMNS, "診断専用の列"), (FINAL_ODDS_COLUMNS, "確定オッズ"), (UNUSABLE_COLUMNS, "使わない列")):
        if sources & bad:
            errors.append(f"{where}: {why} {sorted(sources & bad)} を元にしている")
    if leak := set(s.target_sources) & POST_RACE_COLUMNS:
        errors.append(f"{where}: 対象行のレース後の列 {sorted(leak)} を読んでいる")
    readable = TARGET_SOURCES.get(s.stage, frozenset())
    if extra := (set(s.target_sources) | set(s.keys)) - readable - POST_RACE_COLUMNS - UNUSABLE_COLUMNS:
        errors.append(f"{where}: 段階 {s.stage} の対象行から読めない列 {sorted(extra)}")
    if extra := set(s.history_sources) - HISTORY_SOURCES - DIAGNOSTIC_COLUMNS - FINAL_ODDS_COLUMNS - UNUSABLE_COLUMNS:
        errors.append(f"{where}: 履歴に無い列 {sorted(extra)}")
    if ids := set(s.target_sources) & ID_COLUMNS:
        errors.append(f"{where}: ID 列 {sorted(ids)} を値として読んでいる (as-of 統計の結合キーなら keys に書く)")
    if bad := set(s.keys) - KEY_SOURCES:
        errors.append(f"{where}: keys に書けるのは ID 列 {sorted(KEY_SOURCES)} だけ ({sorted(bad)})")
    return errors


def check_registry(groups: Sequence[Group]) -> None:
    """登録内容の検査。問題をまとめて RegistryError で返す。"""
    errors: list[str] = []
    by_name = {g.name: g for g in groups}
    errors += [f"グループ名 {n} が重複" for n, c in Counter(g.name for g in groups).items() if c > 1]
    errors += [f"特徴量名 {n} が重複" for n, c in Counter(s.name for g in groups for s in g.specs).items() if c > 1]
    errors += [f"補助出力名 {n} が重複" for n, c in Counter(h.name for g in groups for h in g.helpers).items() if c > 1]
    owners: dict[str, str] = {}
    for g in groups:
        if g.stage not in STAGES:
            errors.append(f"{g.name}: stage {g.stage!r} は A か B")
        if not g.specs and not g.helpers:
            errors.append(f"{g.name}: 特徴量も補助出力も無い")
        if len(g.prefixes) > 1:
            errors.append(f"{g.name}: 特徴量名の接頭辞がそろっていない {sorted(g.prefixes)} (1 グループ 1 接頭辞)")
        for p in sorted(g.prefixes):
            if owners.setdefault(p, g.name) != g.name:
                errors.append(f"{g.name}: 接頭辞 {p}_ はグループ {owners[p]} が使っている")
        for s in g.specs:
            errors += _spec_errors(g, s)
        for h in g.helpers:
            if bad := set(h.history_sources) - HISTORY_SOURCES:
                errors.append(f"{g.name}: 補助出力 {h.name} の元列 {sorted(bad)} は履歴に無いか使えない")
        for r in g.requires:
            dep = by_name.get(r)
            if dep is None or r == g.name:
                errors.append(f"{g.name}: requires の {r} が無い")
            elif dep.stage == "B" and g.stage == "A":
                errors.append(f"{g.name}: 段階 A のグループが段階 B の {r} に依存している")
        provided = {s.name for r in g.requires if r in by_name for s in by_name[r].specs}
        if missing := [f for f in g.inputs if f not in provided]:
            errors.append(f"{g.name}: inputs {missing} は requires のグループの特徴量ではない")
    if not errors:
        try:
            ordered(groups)
        except RegistryError as e:
            errors.append(str(e))
    if errors:
        raise RegistryError("特徴量レジストリの検査に失敗:\n  " + "\n  ".join(errors))


def ordered(groups: Sequence[Group]) -> tuple[Group, ...]:
    """依存先が先に来る順 (同順位は 段階, 名前 の順)。循環があれば RegistryError。"""
    pending = {g.name: g for g in groups}
    done: list[Group] = []
    names: set[str] = set()
    while pending:
        ready = sorted((g for g in pending.values() if set(g.requires) <= names), key=lambda g: (g.stage, g.name))
        if not ready:
            raise RegistryError(f"requires が循環している: {sorted(pending)}")
        g = ready[0]
        done.append(g)
        names.add(g.name)
        del pending[g.name]
    return tuple(done)


def with_requirements(groups: Sequence[Group], selected: Iterable[str]) -> tuple[Group, ...]:
    """selected とその依存先 (再帰的) を依存順で。"""
    by_name = {g.name: g for g in groups}
    need: set[str] = set()
    stack = list(selected)
    while stack:
        n = stack.pop()
        if n not in by_name:
            raise RegistryError(f"グループ {n} は登録されていない (登録: {sorted(by_name)})")
        if n not in need:
            need.add(n)
            stack.extend(by_name[n].requires)
    return tuple(g for g in ordered(groups) if g.name in need)


def registry_json(groups: Sequence[Group]) -> dict[str, Any]:
    """data/features/registry.json の中身。"""
    return {
        "version": 1,
        "stages": {
            "A_target_columns": sorted(STATIC_COLUMNS),
            "B_target_columns": sorted(TARGET_SOURCES["B"]),
            "live_columns": sorted(LIVE_COLUMNS),
            "key_sources": sorted(KEY_SOURCES),
            "forbidden": sorted(DIAGNOSTIC_COLUMNS | FINAL_ODDS_COLUMNS | UNUSABLE_COLUMNS),
            "reserved_names": sorted(RESERVED_NAMES),
        },
        "groups": [
            {
                "name": g.name, "stage": g.stage, "prefix": sorted(g.prefixes), "requires": list(g.requires),
                "inputs": list(g.inputs), "features": list(g.feature_names),
                "helpers": [
                    {"name": h.name, "history_sources": list(h.history_sources), "own_post_race": h.own_post_race, "doc": h.doc}
                    for h in g.helpers
                ],
            }
            for g in ordered(groups)
        ],
        "features": [
            {
                "name": s.name, "group": s.group, "stage": s.stage, "kind": s.kind, "dtype": "Float32",
                "target_sources": list(s.target_sources), "keys": list(s.keys), "history_sources": list(s.history_sources),
                "doc": s.doc,
            }
            for g in ordered(groups) for s in g.specs
        ],
    }
