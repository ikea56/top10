"""設定ファイル（config.yml / themes.yml）の読み込み。"""
from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

ROOT = Path(__file__).resolve().parent.parent
JST = timezone(timedelta(hours=9), "JST")


class ConfigError(Exception):
    """設定ファイルの内容に問題があるときのエラー（原因が日本語で出る）。"""


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.yml"
    try:
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"設定ファイルが見つかりません: {p}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"config.yml の書き方に誤りがあります（インデントや「:」を確認）: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError("config.yml の中身が空、または形式が正しくありません")
    return cfg


def load_exclude_codes(cfg: Config) -> set[str]:
    p = ROOT / cfg["filters"]["exclude_codes_file"]
    codes: set[str] = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                codes.add(line)
    return codes


def load_themes() -> dict[str, list[str]]:
    p = ROOT / "themes.yml"
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k): [str(c) for c in v] for k, v in raw.items()}
