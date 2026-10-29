"""ディスク上の特徴量ストア (段階 A・B・目的変数) の作成。CLI は build.py。

    data/features/A/{group}/year={Y}.parquet        race_id + そのグループの特徴量 (段階 A)
    data/features/B/groups/{group}/year={Y}.parquet race_id + そのグループの特徴量 (段階 B)
    data/features/B/year={Y}.parquet                race_id, race_key, date, horse_num + 全特徴量 (Float32)
    data/features/labels/year={Y}.parquet           目的変数と学習対象の印 (labels.py)
    data/features/registry.json                      レジストリ
    data/state/inputs/                               この作成で使った入力 (history, paid_places, ts_snap/{Y})
    data/state/helpers/{name}.parquet                補助出力 (ライブの「前日までの状態」の元)

入力はハードリンクで固定する (データ層が作業中に作り直しても 1 回の作成の中では同じ中身を読む)。
各グループのファイルのメタデータには指紋 (fingerprint.py: 入力の sha256・コードの sha256・依存グループの指紋) を
入れる。入力は段階 A が history、段階 B は history とその年の ts_snap と paid_places。B/year= の組み立ては、今の
入力とコードから計算した指紋と同じファイルだけを使う: 段階 A のファイルが古い・無いなら StaleError、段階 B の
グループは古い・無いならその場で作り直す (1 年 1〜5 秒)。B/year= と labels/year= のメタデータには使った入力の
sha256 を入れ、store_problems が両者 (と今の data/interim の入力) を突き合わせる。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil

from keiba_goal.config import Config
from keiba_goal.features.decision import with_live_columns
from keiba_goal.features.fingerprint import LABELS_MODULES, code_sha, fingerprints, group_code_sha
from keiba_goal.features.io import file_sha256, pin_file, read_meta, read_parquet_retry, write_parquet
from keiba_goal.features.labels import SOURCE_COLUMNS, build_labels
from keiba_goal.features.pipeline import (
    FeatureSource,
    HelperCache,
    assemble,
    build_frame,
    memory_source,
    run_group,
    target_rows,
)
from keiba_goal.features.registry import OUTPUT_KEYS, Group, registry_json

Log = Callable[[str], None]
SNAPSHOT_READ_COLUMNS: tuple[str, ...] = ("race_key", "horse_num", "x", "win", "place_lo", "place_hi")
INPUT_NAMES: dict[str, tuple[str, ...]] = {"A": ("history",), "B": ("history", "ts_snap", "paid_places")}


class StaleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    history: Path
    ts_snap_dir: Path
    paid_places: Path
    features_dir: Path
    state_dir: Path

    @classmethod
    def from_config(cls, cfg: Config) -> Layout:
        interim = cfg.paths.interim
        return cls(
            history=interim / "history.parquet",
            ts_snap_dir=interim / "ts_snap",
            paid_places=interim / "paid_places.parquet",
            features_dir=cfg.paths.features,
            state_dir=cfg.paths.data_root / "state",
        )

    def a_file(self, group: str, year: int) -> Path:
        return self.features_dir / "A" / group / f"year={year}.parquet"

    def b_group_file(self, group: str, year: int) -> Path:
        return self.features_dir / "B" / "groups" / group / f"year={year}.parquet"

    def group_file(self, group: Group, year: int) -> Path:
        return self.a_file(group.name, year) if group.stage == "A" else self.b_group_file(group.name, year)

    def b_file(self, year: int) -> Path:
        return self.features_dir / "B" / f"year={year}.parquet"

    def labels_file(self, year: int) -> Path:
        return self.features_dir / "labels" / f"year={year}.parquet"


class PeakRss:
    """with の間のこのプロセスの RSS の最大 (0.1 秒ごとに測る)。"""

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self._stop.wait(0.1)

    def __enter__(self) -> PeakRss:
        self.peak = self._proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)


@contextmanager
def step(name: str, log: Log) -> Iterator[None]:
    t0 = time.time()
    with PeakRss() as rss:
        yield
    log(f"[{name}] {time.time() - t0:.1f} 秒, ピーク RSS {rss.peak / 2**20:,.0f} MB")


def nan_report(df: pl.DataFrame, features: Sequence[str], top: int = 20) -> str:
    if not features or df.height == 0:
        return "    (特徴量なし)"
    shares = df.select(pl.col(c).is_nan().mean() for c in features).row(0)
    ranked = sorted(zip(features, shares), key=lambda t: -t[1])[:top]
    return "\n".join(f"    {name:40s} NaN {share:6.1%}" for name, share in ranked)


def split_years(df: pl.DataFrame) -> Iterator[tuple[int, pl.DataFrame]]:
    """race_id の年ごとの部分。行が年の順に並んでいれば写しを作らない切り出し。"""
    years = (df["race_id"] // 10**14).to_numpy()
    if years.size and (np.diff(years) >= 0).all():
        bounds = (np.flatnonzero(np.diff(years)) + 1).tolist()
        for a, b in zip([0, *bounds], [*bounds, years.size]):
            yield int(years[a]), df.slice(a, b - a)
    else:
        for y in np.unique(years):
            yield int(y), df.filter((pl.col("race_id") // 10**14) == y)


@dataclass(frozen=True)
class Inputs:
    """固定した入力のパスと sha256。paid_places が無ければ None、ts_snap は年 → パス。"""

    history: Path
    paid_places: Path | None
    ts_snap: Mapping[int, Path]
    sha: Mapping[str, str]


def pin_inputs(layout: Layout) -> Inputs:
    """history・paid_places・ts_snap/{Y} を data/state/inputs/ にハードリンクで固定する。"""
    root = layout.state_dir / "inputs"
    history = pin_file(layout.history, root / "history.parquet")
    paid = root / "paid_places.parquet"
    if layout.paid_places.exists():
        pin_file(layout.paid_places, paid)
    else:
        paid.unlink(missing_ok=True)
    sources = {int(p.stem): p for p in sorted(layout.ts_snap_dir.glob("*.parquet")) if p.stem.isdigit()}
    for old in (root / "ts_snap").glob("*.parquet"):
        if old.stem.isdigit() and int(old.stem) not in sources:
            old.unlink()
    snaps = {y: pin_file(p, root / "ts_snap" / p.name) for y, p in sources.items()}
    sha = {"history": file_sha256(history)}
    if paid.exists():
        sha["paid_places"] = file_sha256(paid)
    sha |= {f"ts_snap/{y}": file_sha256(p) for y, p in snaps.items()}
    return Inputs(history, paid if paid.exists() else None, snaps, sha)


def year_inputs(sha: Mapping[str, str], stage: str, year: int) -> dict[str, str | None]:
    """その段階 (A / B・labels) のその年のファイルが依る入力の sha256 (無い入力は None)。"""
    keys = {"history": "history", "ts_snap": f"ts_snap/{year}", "paid_places": "paid_places"}
    return {name: sha.get(keys[name]) for name in INPUT_NAMES["A" if stage == "A" else "B"]}


@dataclass
class Run:
    """1 回の作成 (CLI 1 回) の状態: 固定した入力・補助出力のキャッシュ・設定・グループのコードの sha256。"""

    layout: Layout
    groups: tuple[Group, ...]
    inputs: Inputs
    history: pl.LazyFrame
    helpers: HelperCache
    params: Mapping[str, Any]
    log: Log
    code: Mapping[str, str]
    _fps: dict[int, dict[str, str]] = field(default_factory=dict)

    @property
    def by_name(self) -> dict[str, Group]:
        return {g.name: g for g in self.groups}

    def inputs_for(self, stage: str, year: int) -> dict[str, str | None]:
        return year_inputs(self.inputs.sha, stage, year)

    def fingerprints(self, year: int) -> dict[str, str]:
        """今の入力とコードから計算した、その年の各グループのファイルの指紋。"""
        if year not in self._fps:
            self._fps[year] = fingerprints(self.groups, self.code, {s: self.inputs_for(s, year) for s in ("A", "B")})
        return self._fps[year]

    def group_meta(self, g: Group, year: int, rows: int) -> dict[str, Any]:
        fps = self.fingerprints(year)
        return {
            "group": g.name, "stage": g.stage, "year": year, "rows": rows, "features": list(g.feature_names),
            "fingerprint": fps[g.name], "inputs": self.inputs_for(g.stage, year), "code_sha256": self.code[g.name],
            "deps": {r: fps[r] for r in g.requires}, "built_at": datetime.now().isoformat(timespec="seconds"),
        }

    def stale_reason(self, g: Group, year: int) -> str | None:
        """そのグループのその年のファイルが今の入力・コードで作ったものでなければ理由、そうなら None。"""
        meta = read_meta(self.layout.group_file(g, year))
        if meta is None:
            return "ファイルが無い"
        fps = self.fingerprints(year)
        if meta.get("fingerprint") == fps[g.name] and meta.get("features") == list(g.feature_names):
            return None
        reasons = []
        built = meta.get("inputs") or {}
        for name, sha in self.inputs_for(g.stage, year).items():
            if built.get(name) != sha:
                reasons.append(f"{name} が違う ({str(built.get(name))[:12]} → {str(sha)[:12]})")
        if meta.get("code_sha256") != self.code[g.name]:
            reasons.append("コードが変わった")
        reasons += [f"依存先 {r} が変わった" for r in g.requires if (meta.get("deps") or {}).get(r) != fps[r]]
        if meta.get("features") != list(g.feature_names):
            reasons.append("列がレジストリと違う")
        return "、".join(reasons) or "指紋が違う"


def open_run(layout: Layout, groups: Sequence[Group], params: Mapping[str, Any], log: Log = print) -> Run:
    """入力を固定して registry.json を書く。"""
    inputs = pin_inputs(layout)
    history = with_live_columns(pl.scan_parquet(inputs.history))
    rows = history.select(pl.len()).collect().item()
    log(f"[inputs] history {rows:,} 行 (sha256 {inputs.sha['history'][:12]}), ts_snap {sorted(inputs.ts_snap)},"
        f" paid_places {'あり' if inputs.paid_places else 'なし'} → {inputs.history.parent}")
    layout.features_dir.mkdir(parents=True, exist_ok=True)
    reg = layout.features_dir / "registry.json"
    tmp = reg.with_name(reg.name + ".tmp")
    tmp.write_text(json.dumps(registry_json(groups), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, reg)
    helpers = HelperCache(groups, history, layout.state_dir / "helpers", {"inputs": {"history": inputs.sha["history"]}})
    code = {g.name: group_code_sha(g) for g in groups}
    return Run(layout, tuple(groups), inputs, history, helpers, params, log, code)


def load_group_year(run: Run, group: Group, year: int) -> pl.DataFrame:
    """作成済みのグループのファイル。今の入力・コードで作ったものでなければ StaleError。"""
    path = run.layout.group_file(group, year)
    if reason := run.stale_reason(group, year):
        raise StaleError(f"{path}: {reason} (`--stage {group.stage} --groups {group.name}` で作り直す)")
    return read_parquet_retry(path)


def disk_source(run: Run, years: Sequence[int]) -> FeatureSource:
    def source(group: str, names: Sequence[str]) -> pl.LazyFrame:
        g = run.by_name[group]
        return pl.concat([load_group_year(run, g, y).select("race_id", *names) for y in years]).lazy()

    return source


def history_years(run: Run) -> list[int]:
    lo = int(run.params["stage_a_min_year"])
    years = run.history.select((pl.col("date") // 10_000).unique().alias("y")).collect()["y"].to_list()
    return sorted(y for y in years if y >= lo)


def run_stage_a(run: Run, groups: Sequence[Group], years: Sequence[int]) -> None:
    """段階 A: グループごとに全履歴で 1 回計算し、対象年ごとに書く。"""
    if not groups:
        return
    target = target_rows(run.history, years)
    expected = target.select("race_id").collect()
    run.log(f"[A] 対象 {len(years)} 年 ({years[0]}〜{years[-1]}), {expected.height:,} 行")
    for g in groups:
        with step(f"A/{g.name}", run.log):
            out = run_group(g, run.by_name, history=run.history, target=target, expected=expected,
                            helpers=run.helpers, features=disk_source(run, years), years=years)
            for y, part in split_years(out):
                write_parquet(part, run.layout.a_file(g.name, y), run.group_meta(g, y, part.height))
            run.log(f"[A/{g.name}] {out.height:,} 行 × {len(g.feature_names)} 列\n{nan_report(out, g.feature_names)}")
            del out


def read_snapshots(run: Run, year: int) -> pl.DataFrame | None:
    path = run.inputs.ts_snap.get(year)
    if path is None:
        run.log(f"[{year}] ts_snap/{year}.parquet が無い → 全レースの field_dec を取消の印で決める (field_dec_fallback = 1)")
        return None
    return read_parquet_retry(path, SNAPSHOT_READ_COLUMNS)


def read_paid_places(run: Run) -> pl.DataFrame | None:
    if run.inputs.paid_places is None:
        run.log(f"[paid_places] {run.layout.paid_places} が無い → 目的変数の支払着数は出走頭数の規則、"
                "判断時の field_open はスナップショット (無ければ field_dec の頭数) で代替")
        return None
    return read_parquet_retry(run.inputs.paid_places)


def frame_for_year(run: Run, year: int, paid: pl.DataFrame | None) -> pl.DataFrame:
    field_open = None
    if paid is not None:
        field_open = paid.select("race_key", "field_open").filter(pl.col("field_open").is_not_null())
    return build_frame(run.history, run.groups, year, None, read_snapshots(run, year), field_open)


def stage_b_parts(run: Run, year: int, frame: pl.DataFrame, force: set[str], allow_missing: bool
                  ) -> tuple[dict[str, pl.DataFrame], dict[str, str], dict[str, str]]:
    """組み立てに使う全グループの出力。返り値は (出力, 外したグループ → 理由, 作り直した段階 B のグループ → 理由)。

    段階 A は作成済みのファイル (古い・無いなら StaleError、allow_missing なら外す)。段階 B は force のグループと、
    ファイルが古い・無いグループを作り直し、それ以外は作成済みのファイルを使う。外したグループに依存するグループも外す。
    """
    expected = frame.select("race_id")
    parts: dict[str, pl.DataFrame] = {}
    missing: dict[str, str] = {}
    rebuilt: dict[str, str] = {}
    for g in run.groups:
        if blocked := [r for r in g.requires if r in missing]:
            missing[g.name] = f"依存先 {blocked} を外した"
            continue
        reason = run.stale_reason(g, year)
        if g.stage == "A":
            if reason is None:
                parts[g.name] = read_parquet_retry(run.layout.a_file(g.name, year))
            elif allow_missing:
                missing[g.name] = reason
            else:
                raise StaleError(f"{run.layout.a_file(g.name, year)}: {reason} (`--stage A --groups {g.name}` で作り直す)")
            continue
        if g.name not in force and reason is None:
            parts[g.name] = read_parquet_retry(run.layout.b_group_file(g.name, year))
            continue
        out = run_group(g, run.by_name, history=run.history, target=frame.lazy(), expected=expected,
                        helpers=run.helpers, features=memory_source(parts), years=(year,))
        write_parquet(out, run.layout.b_group_file(g.name, year), run.group_meta(g, year, out.height))
        parts[g.name] = out
        if g.name not in force:
            rebuilt[g.name] = reason or ""
    return parts, missing, rebuilt


def run_stage_b(run: Run, groups: Sequence[Group], years: Sequence[int], allow_missing: bool = False,
                check_labels: bool = True) -> None:
    """段階 B: 年ごとに判断時刻の枠を作り、段階 B のグループ (groups のものと、ファイルが古い・無いもの) を作って、
    全グループの特徴量を B/year= に組み立てる。

    allow_missing: 段階 A のファイルが古い・無いグループ (とそれに依存するグループ) を B/year= から外して続ける
    (外したグループは表示し、メタデータの missing_groups に残す)。既定は StaleError。
    check_labels: labels/year= が別の入力のものなら警告する (同じ実行で目的変数も作るときは False)。
    """
    paid = read_paid_places(run)
    force = {g.name for g in groups if g.stage == "B"}
    for year in years:
        with step(f"B/{year}", run.log):
            frame = frame_for_year(run, year, paid)
            parts, missing, rebuilt = stage_b_parts(run, year, frame, force, allow_missing)
            for name, reason in rebuilt.items():
                run.log(f"[B/{year}] 段階 B の {name} を作り直した ({reason})")
            for name, reason in missing.items():
                run.log(f"[B/{year}] 外したグループ {name}: {reason}")
            included = [g for g in run.groups if g.name in parts]
            store = assemble(frame.select(OUTPUT_KEYS), included, parts)
            features = [c for c in store.columns if c not in OUTPUT_KEYS]
            if bad := [c for c in features if store.schema[c] != pl.Float32]:
                raise TypeError(f"B/{year}: Float32 でない列 {bad}")
            fps = run.fingerprints(year)
            write_parquet(store, run.layout.b_file(year), {
                "stage": "B", "year": year, "rows": store.height, "inputs": run.inputs_for("B", year),
                "groups": {g.name: fps[g.name] for g in included}, "missing_groups": missing,
                "built_at": datetime.now().isoformat(timespec="seconds"),
            })
            races = frame.group_by("race_key").agg(pl.col("field_dec_fallback").first(), pl.col("field_open_source").first())
            run.log(
                f"[B/{year}] {store.height:,} 行 × {len(features)} 列, レース {races.height:,} (T-6 が無く field_dec を"
                f"取消の印で決めたレース {int(races['field_dec_fallback'].sum()):,}), field_open の出どころ "
                f"{dict(races.group_by('field_open_source').len().iter_rows())}\n{nan_report(store, features)}"
            )
            if check_labels:
                warn_mismatch(run, year, "labels")


def run_labels(run: Run, years: Sequence[int]) -> None:
    paid = read_paid_places(run)
    code = code_sha(LABELS_MODULES)
    for year in years:
        with step(f"labels/{year}", run.log):
            rows = target_rows(run.history, [year]).select(SOURCE_COLUMNS).collect()
            labels = build_labels(rows, frame_for_year(run, year, paid), paid)
            write_parquet(labels, run.layout.labels_file(year), {
                "stage": "labels", "year": year, "rows": labels.height, "inputs": run.inputs_for("B", year),
                "code_sha256": code, "built_at": datetime.now().isoformat(timespec="seconds"),
            })
            el = labels.filter(pl.col("eligible"))
            sources = dict(labels.group_by("n_paid_source").agg(pl.col("race_key").n_unique()).iter_rows())
            run.log(
                f"[labels/{year}] {labels.height:,} 行 (学習対象 {el.height:,}), 単勝の正例率 {el['y_win'].mean():.4f}"
                f" (1/出走頭数の平均 {(1 / el['starters']).mean():.4f}), 複勝の正例率 {el['y_place'].mean():.4f},"
                f" ノイズレースの行 {el['noise_race'].mean():.2%}, 支払着数の出どころ (レース数) {sources}"
            )
            warn_mismatch(run, year, "B")


def warn_mismatch(run: Run, year: int, other: str) -> None:
    """今書いた年のファイルと、もう一方 (other = "B" / "labels") の入力が違えば警告する。"""
    path = run.layout.labels_file(year) if other == "labels" else run.layout.b_file(year)
    meta = read_meta(path)
    if meta is not None and meta.get("inputs") != run.inputs_for("B", year):
        run.log(f"[警告] {path} は別の入力で作られています → `--stage {other} --years {year}` で作り直す")


def current_inputs(layout: Layout, years: Sequence[int]) -> dict[str, str]:
    """data/interim の今の入力 (データ層が置いているファイル。固定したものではない) の sha256。"""
    sha = {"history": file_sha256(layout.history)}
    if layout.paid_places.exists():
        sha["paid_places"] = file_sha256(layout.paid_places)
    for y in years:
        if (p := layout.ts_snap_dir / f"{y}.parquet").exists():
            sha[f"ts_snap/{y}"] = file_sha256(p)
    return sha


def store_problems(layout: Layout, years: Sequence[int], features: Sequence[str],
                   current: Mapping[str, str] | None = None) -> dict[int, list[str]]:
    """学習に使う前の点検: 年 → 問題の一覧 (無ければ空)。

    B/year= と labels/year= がある、同じ入力 (history・ts_snap/{Y}・paid_places) から作られている、外したグループが
    無い、B/year= の列 = キー + features (今のレジストリの順)。current (current_inputs) を渡すと、今の data/interim の
    入力と違う (データ層が作り直した) ことも問題にする。
    """
    out: dict[int, list[str]] = {}
    for y in years:
        problems = []
        b, lab = read_meta(layout.b_file(y)), read_meta(layout.labels_file(y))
        if b is None:
            problems.append("B/year= が無い (`--stage B`)")
        if lab is None:
            problems.append("labels/year= が無い (`--stage labels`)")
        if b is not None and lab is not None and b.get("inputs") != lab.get("inputs"):
            problems.append(f"B と labels の入力が違う (B: {_short(b.get('inputs'))} / labels: {_short(lab.get('inputs'))})")
        if b is not None:
            if b.get("missing_groups"):
                problems.append(f"外したグループがある {sorted(b['missing_groups'])}")
            if pl.read_parquet_schema(layout.b_file(y)).names() != [*OUTPUT_KEYS, *features]:
                problems.append("B/year= の列が今のレジストリと違う (`--stage B` で組み立て直す)")
            if current is not None and b.get("inputs") != (now := year_inputs(current, "B", y)):
                problems.append(f"今の data/interim の入力と違う ({_short(b.get('inputs'))} → {_short(now)})"
                                " → `--stage B` と `--stage labels` で作り直す")
        out[y] = problems
    return out


def _short(inputs: Mapping[str, str | None] | None) -> str:
    return ", ".join(f"{k} {str(v)[:12]}" for k, v in (inputs or {}).items())
