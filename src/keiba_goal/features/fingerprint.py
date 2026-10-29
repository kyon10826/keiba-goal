"""作成済みファイルの指紋 (何から作ったか)。store.py は、今の入力とコードから計算した指紋と同じファイルだけを使う。

指紋 = sha256(グループ名・列・コードの sha256・入力の sha256・依存グループの指紋)。依存グループの指紋を含むので、
依存先 (例: speed) を作り直すと依存する側 (past_runs・aggregates・relative) の指紋も変わる。

コードの sha256 は、グループのモジュールと、そこから (推移的に) import される keiba_goal のモジュール、それらが
ファイル名で読むデータファイル (course_vocab.json など) の中身から作る。値を変えない登録・入口のモジュール
(PLUMBING) は含めない。段階 B のグループは判断時刻の枠 (decision.py) にも依る。
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path

from keiba_goal.features.io import file_sha256
from keiba_goal.features.registry import Group

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLUMBING: frozenset[str] = frozenset({
    "keiba_goal.config", "keiba_goal.features.registry", "keiba_goal.features.context", "keiba_goal.features.groups",
})
DECISION_MODULE = "keiba_goal.features.decision"
LABELS_MODULES: tuple[str, ...] = ("keiba_goal.features.labels", DECISION_MODULE)


def _imported(path: Path) -> set[str]:
    """ファイル中の import 文が名指す keiba_goal のモジュール名 (from X import y は X と X.y の両方の候補)。"""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names |= {node.module, *(f"{node.module}.{a.name}" for a in node.names)}
    return {n for n in names if n.split(".")[0] == "keiba_goal"}


def _source_file(name: str) -> Path | None:
    module = sys.modules.get(name)
    origin = getattr(module, "__file__", None) if module is not None else None
    if origin is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            return None
        origin = spec.origin if spec is not None else None
    return Path(origin).resolve() if origin and origin.endswith(".py") else None


@cache
def module_files(module: str) -> tuple[Path, ...]:
    """module と、そこから推移的に import される keiba_goal のモジュール (PLUMBING を除く) と、読むデータファイル。"""
    seen: dict[str, Path] = {}
    stack = [module]
    while stack:
        name = stack.pop()
        if name in seen or name in PLUMBING:
            continue
        path = _source_file(name)
        if path is None:
            continue
        seen[name] = path
        stack.extend(_imported(path))
    files = set(seen.values())
    for py in seen.values():
        text = py.read_text(encoding="utf-8")
        files |= {p.resolve() for p in py.parent.glob("*.json") if p.name in text}
    return tuple(sorted(files))


def _label(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT)) if path.is_relative_to(PACKAGE_ROOT) else path.name


def code_sha(modules: Iterable[str]) -> str:
    files = sorted({f for m in modules for f in module_files(m)})
    if not files:
        raise ValueError(f"コードのファイルが見つからない: {list(modules)}")
    h = hashlib.sha256()
    for f in files:
        h.update(f"{_label(f)}\0{file_sha256(f)}\n".encode())
    return h.hexdigest()


def group_modules(group: Group) -> tuple[str, ...]:
    own = group.build.__module__
    return (own, DECISION_MODULE) if group.stage == "B" else (own,)


def group_code_sha(group: Group) -> str:
    return code_sha(group_modules(group))


def fingerprint(group: Group, code: str, inputs: Mapping[str, str | None], deps: Mapping[str, str]) -> str:
    payload = {
        "group": group.name, "features": list(group.feature_names), "code": code,
        "inputs": dict(sorted(inputs.items())), "deps": dict(sorted(deps.items())),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fingerprints(groups: Sequence[Group], code: Mapping[str, str],
                 inputs: Mapping[str, Mapping[str, str | None]]) -> dict[str, str]:
    """依存順の groups の指紋。code はグループ名 → コードの sha256、inputs は段階 ("A"/"B") → 入力の sha256。"""
    out: dict[str, str] = {}
    for g in groups:
        out[g.name] = fingerprint(g, code[g.name], inputs[g.stage], {r: out[r] for r in g.requires})
    return out
