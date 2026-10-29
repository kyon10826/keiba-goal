"""グループの build に渡す ctx。宣言した列・補助出力・特徴量だけを返す (宣言漏れは例外)。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import polars as pl

from keiba_goal.features.registry import KEY_COLUMNS


class AccessError(KeyError):
    pass


def _keys_plus(cols: Iterable[str]) -> list[str]:
    return [*KEY_COLUMNS, *(c for c in dict.fromkeys(cols) if c not in KEY_COLUMNS)]


class Ctx:
    """1 つのグループ (または補助出力) の作成に使う入口。

    history(*cols): 履歴の全行 (対象行を含む。ライブでは当日の出馬表の行がレース後の列 null で入る)。
        対象日 D の特徴量に使ってよいのは date < D の行だけ (as-of テストが検査する)。
    target(*cols): 特徴量を作る行。段階 A は対象年 (と対象日) の履歴行、段階 B は判断時刻の枠 (decision_frame)。
    helper(name): requires のグループ (と自分) の補助出力。
    features(*names): requires のグループの特徴量 (inputs に宣言したもの) を target の行について。
    years: 対象年。
    """

    def __init__(
        self,
        owner: str,
        *,
        history: pl.LazyFrame,
        history_columns: Iterable[str],
        target: pl.LazyFrame | None = None,
        target_columns: Iterable[str] = (),
        helpers: Callable[[str], pl.LazyFrame] | None = None,
        helper_names: Iterable[str] = (),
        features: Callable[[str, Sequence[str]], pl.LazyFrame] | None = None,
        feature_groups: Mapping[str, str] | None = None,
        years: Sequence[int] = (),
    ):
        self.owner = owner
        self.years = tuple(years)
        self._history = history
        self._history_columns = frozenset(history_columns)
        self._target = target
        self._target_columns = frozenset(target_columns)
        self._helpers = helpers
        self._helper_names = frozenset(helper_names)
        self._features = features
        self._feature_groups = dict(feature_groups or {})

    def _check(self, cols: Iterable[str], allowed: frozenset[str], what: str) -> None:
        bad = [c for c in cols if c not in allowed and c not in KEY_COLUMNS]
        if bad:
            raise AccessError(f"{self.owner}: {what} {bad} は宣言されていない (宣言: {sorted(allowed)})")

    def history(self, *cols: str) -> pl.LazyFrame:
        self._check(cols, self._history_columns, "history の列")
        return self._history.select(_keys_plus(cols))

    def target(self, *cols: str) -> pl.LazyFrame:
        if self._target is None:
            raise AccessError(f"{self.owner}: target はグループの build でだけ使える")
        self._check(cols, self._target_columns, "target の列")
        return self._target.select(_keys_plus(cols))

    def helper(self, name: str) -> pl.LazyFrame:
        if self._helpers is None or name not in self._helper_names:
            raise AccessError(f"{self.owner}: 補助出力 {name} は requires のグループに無い (使える: {sorted(self._helper_names)})")
        return self._helpers(name)

    def features(self, *names: str) -> pl.LazyFrame:
        bad = [n for n in names if n not in self._feature_groups]
        if bad or self._features is None:
            raise AccessError(f"{self.owner}: 特徴量 {bad or list(names)} は inputs に宣言されていない")
        by_group: dict[str, list[str]] = {}
        for n in names:
            by_group.setdefault(self._feature_groups[n], []).append(n)
        frames = [self._features(g, cols) for g, cols in by_group.items()]
        out = frames[0]
        for f in frames[1:]:
            out = out.join(f, on="race_id", how="left")
        return out
