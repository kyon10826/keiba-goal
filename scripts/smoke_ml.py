"""P0 の ML スモークテスト (design.md §2)。

全学習器の短時間学習、FLAML を開催日境界の分割で 30 秒、Optuna の sqlite スタディ 2 試行を
1 プロセスで実行し、依存関係の組み合わせが実際に動くことを確かめる。

    .venv/bin/python scripts/smoke_ml.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings

import numpy as np


def _data(n_days: int = 140, rows_per_day: int = 40, p: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = np.array(
        [y * 10_000 + 101 + (i * 37 % 1100) for y in range(2014, 2021) for i in range(n_days // 7)],
        dtype=np.int64,
    )
    days.sort()
    d = np.repeat(days, rows_per_day)
    X = rng.normal(size=(d.size, p)).astype(np.float32)
    X[rng.random(X.shape) < 0.12] = np.nan
    logit = 1.2 * np.nan_to_num(X[:, 0]) - 0.8 * np.nan_to_num(X[:, 1]) - 2.0
    y = (rng.random(d.size) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    return d, X, y


def main() -> int:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    import catboost
    import flaml
    import lightgbm as lgb
    import optuna
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import log_loss

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from keiba_goal.models.cv import DayBoundaryExpandingSplit

    d, X, y = _data()
    tr, va = d < 20190101, d >= 20190101
    print(f"data {X.shape} float32, NaN {np.isnan(X).mean():.1%}, pos {y.mean():.1%}")

    # 1. 各学習器 20 ラウンド
    m = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 4},
                  lgb.Dataset(X[tr], y[tr], params={"feature_pre_filter": False}), num_boost_round=20)
    print(f"lightgbm {lgb.__version__}: logloss {log_loss(y[va], m.predict(X[va])):.4f}")
    bx = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "nthread": 4},
                   xgb.QuantileDMatrix(X[tr], y[tr]), num_boost_round=20)
    print(f"xgboost {xgb.__version__}: logloss {log_loss(y[va], bx.predict(xgb.DMatrix(X[va]))):.4f}")
    cb = catboost.CatBoostClassifier(iterations=20, verbose=0, allow_writing_files=False, thread_count=4)
    cb.fit(X[tr], y[tr])
    print(f"catboost {catboost.__version__}: logloss {log_loss(y[va], cb.predict_proba(X[va])[:, 1]):.4f}")

    # 2. FLAML: 独自分割・NaN 維持・並べ替えなし
    split = DayBoundaryExpandingSplit(d, val_years=[2017, 2018, 2019, 2020], min_date=20140101)
    used = []
    seen_inputs = []

    class Recording(DayBoundaryExpandingSplit):
        def split(self, X=None, y=None, groups=None):  # noqa: N803
            # FLAML が変換した後の行列がここに来る: 欠損が残り、行数・並びが変わっていないことを記録
            seen_inputs.append((len(X), bool(np.isnan(np.asarray(X, dtype=np.float32)).any())))
            for tr_i, va_i in super().split(X, y, groups):
                used.append((int(d[tr_i].max()), int(d[va_i].min()), tr_i.size, va_i.size))
                yield tr_i, va_i

    rec = Recording(d, val_years=[2017, 2018, 2019, 2020], min_date=20140101)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])], copy=False)
    with tempfile.TemporaryDirectory() as wd:
        cwd = os.getcwd()
        os.chdir(wd)
        try:
            automl = flaml.AutoML()
            automl.fit(df, y, task="classification", metric="log_loss",
                       estimator_list=["lgbm", "xgboost", "catboost"], eval_method="cv",
                       split_type=rec, time_budget=30, skip_transform=True, retrain_full=False,
                       n_concurrent_trials=1, n_jobs=4, seed=17, verbose=0)
        finally:
            os.chdir(cwd)
    assert used, "FLAML が分割器を呼んでいない"
    assert all(a < b for a, b, *_ in used), "学習側の最大日付 >= 検証側の最小日付"
    print(f"flaml {flaml.__version__}: best={automl.best_estimator} loss={automl.best_loss:.4f}, "
          f"folds used={len(used)} (all day-disjoint), split table={split.fold_table()[0]}")
    assert all(n == len(d) for n, _ in seen_inputs), "FLAML が行数を変えた"
    assert all(has_nan for _, has_nan in seen_inputs), "FLAML が欠損値を補完した (skip_transform が効いていない)"
    print(f"flaml passed the untransformed frame to the splitter {len(seen_inputs)} times (NaN kept, rows {len(d)})")

    # 3. Optuna sqlite 2 試行
    with tempfile.TemporaryDirectory() as wd:
        storage = f"sqlite:///{wd}/optuna.db"
        study = optuna.create_study(direction="minimize", storage=storage, study_name="smoke",
                                    sampler=optuna.samplers.TPESampler(seed=17), load_if_exists=True)

        def objective(trial):
            params = {"objective": "binary", "verbose": -1, "num_threads": 4,
                      "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                      "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100)}
            booster = lgb.train(params, lgb.Dataset(X[tr], y[tr], params={"feature_pre_filter": False}), 20)
            return log_loss(y[va], booster.predict(X[va]))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=2)
        print(f"optuna {optuna.__version__}: 2 trials ok, best={study.best_value:.4f}")

    print(f"SMOKE OK in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
