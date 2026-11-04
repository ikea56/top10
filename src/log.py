"""ログ出力の共通設定（print の代わりにこれを使う）。"""
from __future__ import annotations

import logging
import sys

_configured = False


def get_logger(name: str) -> logging.Logger:
    """`log = get_logger(__name__)` の形で使う。時刻・レベル付きで標準出力に出る。"""
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root = logging.getLogger("top10")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _configured = True
    return logging.getLogger(f"top10.{name.split('.')[-1]}")
