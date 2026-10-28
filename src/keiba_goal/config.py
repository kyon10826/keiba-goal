"""設定ファイル (config/*.yaml) の読み込みとパス解決。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    record_dir: Path
    ts_odds_dir: Path
    spec_xlsx: Path
    data_root: Path

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        return self.data_root / "interim"

    @property
    def features(self) -> Path:
        return self.data_root / "features"

    @property
    def models(self) -> Path:
        return self.data_root / "models"

    @property
    def sessions(self) -> Path:
        return self.data_root / "sessions"

    @property
    def backtest(self) -> Path:
        return self.data_root / "backtest"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    paths: Paths

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name) or {})


def _resolve(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    p = raw.get("paths") or {}
    missing = [k for k in ("record_dir", "ts_odds_dir", "spec_xlsx", "data_root") if k not in p]
    if missing:
        raise ValueError(f"{cfg_path}: paths に {missing} がありません")
    paths = Paths(**{k: _resolve(str(p[k])) for k in ("record_dir", "ts_odds_dir", "spec_xlsx", "data_root")})
    return Config(raw=raw, paths=paths)
