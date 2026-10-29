from dataclasses import fields

import pytest

from keiba_goal.betting.params import OTHER_MODULE_KEYS, BettingParams
from keiba_goal.config import load_config


def test_yaml_defaults_equal_the_dataclass_defaults():
    section = load_config().section("betting")
    assert BettingParams.from_config(section) == BettingParams()
    # yaml がすべての項目を書いている (既定値を変えるときは両方を直す)
    assert {f.name for f in fields(BettingParams)} | OTHER_MODULE_KEYS == set(section)


def test_unknown_keys_are_rejected_but_keys_of_other_modules_are_allowed():
    with pytest.raises(ValueError, match="ev_gate_"):
        BettingParams.from_config({"ev_gate_": 0.95})
    assert BettingParams.from_config({"odds_drop_quantile": 0.10, "ev_gate": 0.9}).ev_gate == 0.9


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("top_n", 12.7),          # 切り捨てない
        ("top_n", "12"),          # 文字列から変換しない
        ("top_n", True),          # bool は整数ではない
        ("pl_tol", "1e-4"),       # YAML の 1e-4 は文字列になる
        ("ev_gate", None),
        ("ev_gate", float("nan")),
        ("search_prefilter", "false"),  # 文字列の "false" は真
        ("search_prefilter", 0),
    ],
)
def test_values_of_the_wrong_type_are_rejected(key, value):
    with pytest.raises(ValueError, match=f"betting.{key}"):
        BettingParams.from_config({key: value})


def test_integral_values_are_converted_exactly():
    p = BettingParams.from_config({"top_n": 10.0, "decision_lead_min": 6, "pl_tol": 1e-3, "search_prefilter": False})
    assert (p.top_n, p.decision_lead_min, p.pl_tol, p.search_prefilter) == (10, 6.0, 1e-3, False)
    assert type(p.top_n) is int and type(p.decision_lead_min) is float


@pytest.mark.parametrize(
    "bad",
    [
        {"takeout": 1.0}, {"max_basket": 13}, {"min_field": 4}, {"pl_damping": 0.0}, {"pl_tol": 0.0},
        {"deadline_min": 6.0}, {"day_budget": 20_000}, {"day_bets": 0}, {"eff_odds_band_upper": 0.9},
        {"max_odds": 1.0}, {"ev_gate": -0.1},
    ],
)
def test_out_of_range_values_are_rejected(bad):
    with pytest.raises(ValueError):
        BettingParams.from_config(bad)
