"""開催日境界の拡張型時系列分割 (design.md §6)。

FLAML と Optuna が同じ分割を使えるよう、分割は事前に計算した位置インデックスで持つ。
TimeSeriesSplit を継承するのは、FLAML が TimeSeriesSplit 型の分割器に対して
学習側インデックスの並べ替え (shuffle) をせず split(X, y) で呼ぶため。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class DayBoundaryExpandingSplit(TimeSeriesSplit):
    """検証ブロック = 各検証年の全開催日、学習 = それより前の全開催日 (min_date 以降)。

    `dates` は分割対象の行と同じ順に並んだ YYYYMMDD 整数。行は日付順に並んでいる必要がある
    (同じ日が学習と検証に割れないことを保証するため、並びが崩れていれば例外)。
    `exclude_from_train` が True の行は学習側からだけ除く (ノイズレースのクレンジング用)。
    """

    def __init__(
        self,
        dates: Sequence[int] | np.ndarray,
        val_years: Sequence[int],
        min_date: int | None = None,
        exclude_from_train: Sequence[bool] | np.ndarray | None = None,
    ):
        super().__init__(n_splits=max(2, len(val_years)))
        d = np.asarray(dates, dtype=np.int64)
        if d.ndim != 1 or d.size == 0:
            raise ValueError("dates は 1 次元で 1 行以上必要です")
        if np.any(np.diff(d) < 0):
            raise ValueError("dates が日付順に並んでいません (date, race_key, horse_num で並べてから渡す)")
        excl = np.zeros(d.size, dtype=bool) if exclude_from_train is None else np.asarray(exclude_from_train, bool)
        if excl.shape != d.shape:
            raise ValueError("exclude_from_train の長さが dates と違います")
        years = d // 10_000
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for y in val_years:
            val = np.flatnonzero(years == y)
            train_mask = d < y * 10_000 + 101
            if min_date is not None:
                train_mask &= d >= min_date
            train = np.flatnonzero(train_mask & ~excl)
            if val.size == 0 or train.size == 0:
                raise ValueError(f"検証年 {y} の学習側または検証側が空です")
            folds.append((train, val))
        self.shuffle = False
        self.val_years = list(val_years)
        self._folds = folds
        self._dates = d

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803 (sklearn の引数名)
        return len(self._folds)

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # noqa: N803
        if X is not None and len(X) != self._dates.size:
            raise ValueError(f"X の行数 {len(X)} が分割作成時の行数 {self._dates.size} と違います")
        for train, val in self._folds:
            yield train.copy(), val.copy()

    def fold_table(self) -> list[dict]:
        """保存・検査用: 各分割の日付範囲と件数。"""
        out = []
        for (train, val), y in zip(self._folds, self.val_years):
            out.append({
                "val_year": int(y),
                "train_rows": int(train.size),
                "val_rows": int(val.size),
                "train_max_date": int(self._dates[train].max()),
                "val_min_date": int(self._dates[val].min()),
            })
        return out
