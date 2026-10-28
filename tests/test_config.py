from pathlib import Path

import pytest

from keiba_goal.config import REPO_ROOT, load_config


def test_default_config_resolves_paths():
    cfg = load_config()
    assert cfg.paths.data_root == (REPO_ROOT / "data").resolve()
    assert cfg.paths.features == cfg.paths.data_root / "features"
    assert cfg.section("splits")["train"] == [2004, 2020]
    assert cfg.section("betting")["ev_gate"] == 0.8


def test_missing_path_key_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths:\n  record_dir: /x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ts_odds_dir"):
        load_config(bad)
