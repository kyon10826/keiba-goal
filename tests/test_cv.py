import numpy as np
import pytest

from keiba_goal.models.cv import DayBoundaryExpandingSplit


def _dates():
    # 2014〜2020 の毎年 3 開催日 × 各日 4 行
    days = [y * 10_000 + md for y in range(2014, 2021) for md in (105, 612, 1220)]
    return np.repeat(np.array(days, dtype=np.int64), 4)


def test_folds_are_expanding_and_day_disjoint():
    d = _dates()
    sp = DayBoundaryExpandingSplit(d, val_years=[2016, 2017, 2018, 2019, 2020], min_date=20140101)
    assert sp.get_n_splits() == 5
    prev_train = 0
    for (train, val), y in zip(sp.split(d), [2016, 2017, 2018, 2019, 2020]):
        assert d[train].max() < d[val].min()
        assert set(np.unique(d[val] // 10_000)) == {y}
        assert np.intersect1d(train, val).size == 0
        assert train.size > prev_train
        prev_train = train.size


def test_exclusion_only_affects_training_side():
    d = _dates()
    excl = np.zeros(d.size, bool)
    excl[::4] = True
    sp = DayBoundaryExpandingSplit(d, val_years=[2019, 2020], exclude_from_train=excl)
    for train, val in sp.split(d):
        assert not excl[train].any()
        assert excl[val].any()


def test_rejects_unsorted_dates_and_wrong_length():
    d = _dates()
    with pytest.raises(ValueError, match="日付順"):
        DayBoundaryExpandingSplit(d[::-1], val_years=[2020])
    sp = DayBoundaryExpandingSplit(d, val_years=[2020])
    with pytest.raises(ValueError, match="行数"):
        next(sp.split(d[:-1]))


def test_flaml_does_not_shuffle_this_splitter():
    sp = DayBoundaryExpandingSplit(_dates(), val_years=[2020])
    assert getattr(sp, "shuffle") is False
