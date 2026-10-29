import numpy as np
import pytest

from keiba_goal.betting.joint import (
    build_outcome_table,
    check_paid_places,
    ordered_tuples,
    place_marginals,
    project_place,
    subset_hit_probs,
    table_place_marginals,
    tuple_probs,
)


def _race(n: int, seed: int, noise: float = 0.15, m: int | None = None):
    """勝率と、Harville を割り引いた形にノイズを足した複勝確率 (校正済みモデルの出力の代わり)。

    支払着数 m の既定は発売開始時の頭数 = n とした JRA の規則 (取消の無い合成レース)。
    """
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(n, 0.9))
    m = m if m is not None else (3 if n >= 8 else 2)
    order = ordered_tuples(n, m)
    plc = table_place_marginals(order, tuple_probs(w, w**0.8, order), n)
    plc = np.clip(plc * np.exp(rng.normal(0, noise, n)), 0, 1)
    return np.arange(1, n + 1), w, plc, m


def test_paid_places_must_be_0_2_or_3():
    assert [check_paid_places(m) for m in (0, 2, 3)] == [0, 2, 3]
    for bad in (1, 4, -1, True, False, None, 2.5):
        with pytest.raises(ValueError, match="支払着数"):
            check_paid_places(bad)


def test_paid_places_is_an_input_not_the_field_size():
    # 8 頭登録・発売開始後に 1 頭取消 (7 頭) は 3 着払い (202309040202 型)、9 頭でも渡された m に従う
    nums, w, plc, _ = _race(7, seed=30, m=3)
    seven_paying_three = build_outcome_table(nums, w, plc, 3)
    assert seven_paying_three.n_paid_places == 3 and seven_paying_three.order.shape == (7 * 6 * 5, 3)
    assert seven_paying_three.p_place.sum() == pytest.approx(3.0, abs=1e-9)
    nums, w, plc, _ = _race(9, seed=31, m=2)
    nine_paying_two = build_outcome_table(nums, w, plc, 2)
    assert nine_paying_two.n_paid_places == 2 and nine_paying_two.order.shape == (9 * 8, 2)
    assert nine_paying_two.p_place.sum() == pytest.approx(2.0, abs=1e-9)


def test_same_race_with_two_and_three_paid_places():
    nums, w, plc3, _ = _race(10, seed=32, m=3)
    two = build_outcome_table(nums, w, plc3 * 2 / 3, 2)
    three = build_outcome_table(nums, w, plc3, 3)
    for table, m in ((two, 2), (three, 3)):
        assert table.converged and table.order.shape[1] == m
        assert table.place_marginals().sum() == pytest.approx(m, abs=1e-9)
        np.testing.assert_allclose(np.bincount(table.order[:, 0], table.prob, 10), w, atol=1e-12)
        for i, h in enumerate(nums):
            assert abs(table.hit_prob([("place", h)]) - table.p_place[i]) <= table.max_error + 1e-12
    fav = int(nums[np.argmax(w)])
    assert three.hit_prob([("place", fav)]) > two.hit_prob([("place", fav)])
    # 単勝だけのバスケットは支払着数によらない
    assert three.hit_prob([("win", 1), ("win", 2)]) == pytest.approx(two.hit_prob([("win", 1), ("win", 2)]), abs=1e-12)


def test_no_place_sales_gives_a_win_only_table():
    nums, w, plc, _ = _race(6, seed=33)
    table = build_outcome_table(nums, w, None, 0)
    assert table.n_paid_places == 0 and table.order.shape == (6, 1)
    assert table.p_place is None and table.v is None and table.converged and table.max_error == 0.0
    np.testing.assert_allclose(table.prob, w, atol=1e-15)
    assert table.hit_prob([("win", 2), ("win", 5)]) == pytest.approx(w[1] + w[4], abs=1e-12)
    with pytest.raises(ValueError, match="複勝の発売"):
        table.ticket_mask("place", 2)
    with pytest.raises(ValueError, match="複勝の発売"):
        table.place_marginals()
    with pytest.raises(ValueError, match="複勝確率"):
        build_outcome_table(nums, w, plc, 0)
    with pytest.raises(ValueError, match="複勝確率"):
        build_outcome_table(nums, w, None, 2)


def test_tuple_counts():
    assert ordered_tuples(18, 3).shape == (4_896, 3)
    assert ordered_tuples(5, 2).shape == (20, 2)


def test_tuple_probabilities_follow_the_two_stage_definition():
    w = np.array([0.5, 0.3, 0.2])
    v = np.array([0.2, 0.3, 0.5])
    order = ordered_tuples(3, 2)
    got = dict(zip(map(tuple, order), tuple_probs(w, v, order)))
    assert got[(0, 1)] == pytest.approx(0.5 * 0.3 / 0.8)
    assert got[(1, 2)] == pytest.approx(0.3 * 0.5 / 0.7)
    assert got[(2, 0)] == pytest.approx(0.2 * 0.2 / 0.5)
    assert sum(got.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [5, 7, 8, 12, 18])
def test_marginals_match_inputs(n):
    nums, w, plc, m = _race(n, seed=n)
    table = build_outcome_table(nums, w, plc, m)
    assert table.converged
    assert table.prob.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(np.bincount(table.order[:, 0], table.prob, n), w, atol=1e-12)
    assert np.abs(table.place_marginals() - table.p_place).max() < 1e-4
    assert table.max_error < 1e-4


def test_matrix_marginals_equal_the_enumerated_table():
    rng = np.random.default_rng(3)
    for n, m in ((6, 2), (11, 3), (18, 3)):
        w = rng.dirichlet(np.ones(n))
        v = rng.dirichlet(np.ones(n))
        order = ordered_tuples(n, m)
        np.testing.assert_allclose(place_marginals(w, v, m), table_place_marginals(order, tuple_probs(w, v, order), n),
                                   atol=1e-12)


def test_projection_makes_place_targets_feasible():
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
    plc = np.array([0.30, 1.20, 0.50, 0.30, 0.04, 0.10, 0.05])  # 1 番は p_win 未満、2 番は 1 超
    out = project_place(w, plc, 2)
    assert out.sum() == pytest.approx(2.0, abs=1e-9)
    assert np.all(out >= w + 1e-6 - 1e-15) and np.all(out <= 1.0)
    assert out[1] == 1.0
    far = project_place(w, np.full(7, 1.5), 2)  # 全馬 1 超でも合計 m にそろう
    assert far.sum() == pytest.approx(2.0, abs=1e-9) and np.all(far >= w)
    low = project_place(w, np.full(7, -0.5), 2)
    assert low.sum() == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize("n", [5, 18])
def test_single_ticket_reproduces_p_hat(n):
    nums, w, plc, m = _race(n, seed=10 + n)
    table = build_outcome_table(nums, w, plc, m)
    for i, h in enumerate(nums):
        assert table.hit_prob([("win", h)]) == pytest.approx(w[i], abs=1e-12)
        assert abs(table.hit_prob([("place", h)]) - table.p_place[i]) <= table.max_error + 1e-12
        # 同じ馬の単勝は複勝に含まれる
        assert table.hit_prob([("win", h), ("place", h)]) == pytest.approx(table.hit_prob([("place", h)]), abs=1e-12)


def test_win_only_basket_equals_sum_of_p_hat():
    nums, w, plc, m = _race(18, seed=4)
    table = build_outcome_table(nums, w, plc, m)
    rng = np.random.default_rng(4)
    for _ in range(50):
        pick = rng.choice(18, size=int(rng.integers(1, 6)), replace=False)
        assert table.hit_prob([("win", nums[i]) for i in pick]) == pytest.approx(w[pick].sum(), abs=1e-12)


def test_hit_probability_bounds_on_mixed_baskets():
    rng = np.random.default_rng(5)
    for n in (5, 9, 14, 18):
        nums, w, plc, m = _race(n, seed=20 + n)
        table = build_outcome_table(nums, w, plc, m)
        p_hat = {"win": dict(zip(nums, table.p_win)), "place": dict(zip(nums, table.p_place))}
        tol = table.max_error + 1e-12
        for _ in range(100):
            horses = rng.choice(nums, size=int(rng.integers(1, min(5, n) + 1)), replace=False)
            basket = [(str(rng.choice(["win", "place"])), int(h)) for h in horses]
            p = [p_hat[b][h] for b, h in basket]
            hit = table.hit_prob(basket)
            assert max(p) - tol <= hit <= min(1.0, sum(p)) + tol


def test_horse_with_almost_zero_win_probability():
    nums, w, plc, m = _race(12, seed=6)
    w[4] = 0.0
    w = w / w.sum()
    table = build_outcome_table(nums, w, plc, m)
    assert table.converged
    assert table.hit_prob([("win", 5)]) == 0.0
    assert abs(table.hit_prob([("place", 5)]) - table.p_place[4]) < 1e-4
    w[4] = 1e-12
    table = build_outcome_table(nums, w / w.sum(), plc, m)
    assert table.converged


def test_non_convergence_is_reported_and_table_stays_a_distribution():
    nums, w, plc, m = _race(18, seed=7, noise=0.4)
    table = build_outcome_table(nums, w, plc, m, max_iter=1)
    assert not table.converged
    assert table.iterations == 1
    assert table.max_error >= 1e-4
    assert table.prob.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(np.bincount(table.order[:, 0], table.prob, 18), w, atol=1e-12)


def test_target_clipped_at_one_converges_slowly_and_is_reported():
    # 複勝確率が射影で 1 に張り付くと v → ∞ が必要で、0.7 乗の更新では 2,000 回で 1e-4 に届かない
    nums, w, plc, m = _race(10, seed=8)
    plc[int(np.argmax(w))] = 1.5
    table = build_outcome_table(nums, w, plc, m)
    assert table.p_place.max() == 1.0
    assert not table.converged and table.iterations == 2_000
    assert table.max_error == pytest.approx(np.abs(table.place_marginals() - table.p_place).max())
    assert 1e-4 <= table.max_error < 2e-4
    longer = build_outcome_table(nums, w, plc, m, max_iter=20_000)
    assert longer.converged and longer.max_error < 1e-4


def test_subset_probabilities_match_direct_computation():
    nums, w, plc, m = _race(16, seed=9)
    table = build_outcome_table(nums, w, plc, m)
    tickets = [("win", 3), ("place", 3), ("win", 7), ("place", 9), ("win", 12), ("place", 1), ("win", 16)]
    masks = table.ticket_masks(tickets)
    subsets = [(0,), (0, 2), (1, 3, 4), (2, 3, 5, 6), (0, 1), (6,)]
    got = subset_hit_probs(table.prob, masks, subsets)
    want = [table.hit_prob([tickets[j] for j in s]) for s in subsets]
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_input_validation():
    nums, w, plc, m = _race(8, seed=11)
    with pytest.raises(ValueError, match="合計 1"):
        build_outcome_table(nums, w * 0.9, plc, m)
    with pytest.raises(ValueError, match="重複"):
        build_outcome_table(np.ones(8, int), w, plc, m)
    with pytest.raises(ValueError):
        build_outcome_table(nums, w, plc, 4)
    with pytest.raises(ValueError):
        build_outcome_table(nums, w, np.full(8, np.nan), m)
    table = build_outcome_table(nums, w, plc, m)
    with pytest.raises(KeyError):
        table.hit_prob([("win", 99)])
    with pytest.raises(ValueError):
        table.hit_prob([("exacta", 1)])
