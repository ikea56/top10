"""設定ファイルの読み込み。"""
from pathlib import Path
from datetime import timezone, timedelta

import yaml

ROOT = Path(__file__).resolve().parent.parent
JST = timezone(timedelta(hours=9), "JST")


def load_config(path=None):
    p = Path(path) if path else ROOT / "config.yml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_exclude_codes(cfg):
    p = ROOT / cfg["filters"]["exclude_codes_file"]
    codes = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                codes.add(line)
    return codes


def load_themes():
    p = ROOT / "themes.yml"
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k): [str(c) for c in v] for k, v in raw.items()}
