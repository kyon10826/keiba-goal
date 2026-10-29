import json
import os
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from keiba_goal.data.build import main

FIX = Path(__file__).parent / "fixtures" / "data"


def _config(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
paths:
  record_dir: {FIX / "record"}
  ts_odds_dir: {FIX / "jd"}
  spec_xlsx: {tmp_path / "spec.xlsx"}
  data_root: {tmp_path / "data"}
betting:
  odds_drop_quantile: 0.10
data:
  ts_years: [2004, 2024]
  ts_processes: 1
  q_min_band_samples: 1
  q_windows:
    fix: [2024, 2024]
""",
        encoding="utf-8",
    )
    return cfg


def test_build_all_on_fixtures(tmp_path: Path, capsys):
    cfg = _config(tmp_path)
    assert main(["--step", "all", "--config", str(cfg)]) == 0
    interim = tmp_path / "data" / "interim"
    out = capsys.readouterr().out
    for tag in ("[records]", "[history]", "[ts]", "[paid_places]", "[qtable]", "[build]"):
        assert tag in out
    assert out.index("[history]") < out.index("[ts]") < out.index("[paid_places]") < out.index("[qtable]")
    assert "ファイルなし: 全レース推定" in out and "警告" not in out
    assert pl.read_parquet(interim / "records.parquet").height == 176
    assert sorted(p.name for p in (interim / "ts_snap").iterdir()) == [
        "2004.parquet", "2005.parquet", "2015.parquet", "2023.parquet", "2024.parquet"]
    assert json.loads((interim / "q_tables" / "fix.json").read_text(encoding="utf-8"))["window"] == [2024, 2024]
    by_year = json.loads((interim / "q_tables" / "by_year.json").read_text(encoding="utf-8"))
    assert sorted(by_year["years"]) == ["2004", "2005", "2015", "2023", "2024"]
    assert pl.read_parquet(interim / "history.parquet").equals(pl.read_parquet(interim / "records.parquet"))
    paid = pl.read_parquet(interim / "paid_places.parquet")
    assert paid.height == 13 and set(paid["source"]) == {"jd_field_open", "starters"}


def test_post_times_option_and_missing_explicit_path(tmp_path: Path, capsys):
    cfg = _config(tmp_path)
    assert main(["--step", "records", "--config", str(cfg)]) == 0
    table = tmp_path / "scratch_post_times.parquet"
    pl.DataFrame({
        "race_key": [2024031009010604], "post_time": ["11:30"], "post_datetime": [datetime(2024, 3, 10, 11, 30)],
        "source": ["race_list"],
    }).write_parquet(table)
    assert main(["--step", "ts", "--years", "2024", "--post-times", str(table), "--config", str(cfg)]) == 0
    assert str(table) in capsys.readouterr().out
    race = pl.read_parquet(tmp_path / "data" / "interim" / "ts_race" / "2024.parquet")
    assert race.filter(pl.col("post_source") == "race_list")["race_key"].to_list() == [2024031009010604]
    with pytest.raises(FileNotFoundError, match="発走時刻表"):
        main(["--step", "ts", "--years", "2024", "--post-times", str(tmp_path / "nope.parquet"), "--config", str(cfg)])


def test_warns_when_window_files_are_not_combined(tmp_path: Path, capsys):
    cfg = _config(tmp_path)
    assert main(["--step", "records", "--config", str(cfg)]) == 0
    interim = tmp_path / "data" / "interim"
    window = interim / "post_times_gate.parquet"
    pl.DataFrame({"race_key": [1], "post_time": ["10:00"], "source": ["netkeiba"]}).write_parquet(window)
    capsys.readouterr()
    assert main(["--step", "ts", "--years", "2024", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "post_times.parquet がありません" in out and "crawl combine" in out
    combined = interim / "post_times.parquet"
    pl.DataFrame({"race_key": [2024031009010604], "post_time": ["11:30"], "source": ["netkeiba"]}).write_parquet(combined)
    old = window.stat().st_mtime - 60
    os.utime(combined, (old, old))
    assert main(["--step", "ts", "--years", "2024", "--config", str(cfg)]) == 0
    assert "窓ごとのファイルより古いです" in capsys.readouterr().out
    os.utime(combined, None)
    assert main(["--step", "ts", "--years", "2024", "--config", str(cfg)]) == 0
    assert "警告" not in capsys.readouterr().out
