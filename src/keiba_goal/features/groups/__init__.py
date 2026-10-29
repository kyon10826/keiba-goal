"""特徴量グループ。このパッケージの各モジュール (名前が _ で始まらないもの) が GROUP = Group(...) を持つ。

モジュール名 = グループ名。新しいグループはモジュールを 1 つ足すだけで登録される (docs/features.md)。
"""

from __future__ import annotations

import importlib
import pkgutil
from functools import cache

from keiba_goal.features.registry import Group, RegistryError, check_registry, ordered


@cache
def discover() -> tuple[Group, ...]:
    """全グループを依存順で。レジストリの検査 (registry.check_registry) を通らなければ RegistryError。"""
    groups = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        group = getattr(module, "GROUP", None)
        if not isinstance(group, Group):
            raise RegistryError(f"features/groups/{info.name}.py に GROUP = Group(...) が無い")
        if group.name != info.name:
            raise RegistryError(f"features/groups/{info.name}.py: グループ名 {group.name!r} はモジュール名と同じにする")
        groups.append(group)
    check_registry(groups)
    return ordered(groups)
