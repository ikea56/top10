"""業種ランキング・テーマランキング。"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _agg(sub: pd.DataFrame) -> dict[str, Any]:
    return {
        "score": float(sub["tech_scaled"].mean()),
        "n": int(len(sub)),
        "n_high": int(sub["hi252"].sum()),
        "ret20": float(sub["ret20d"].mean()) if sub["ret20d"].notna().any() else 0.0,
        "leaders": list(sub.sort_values(["tech_base", "vol_ratio"], ascending=False)["name"].head(3)),
    }


def rank_sectors(liquid: pd.DataFrame, top: int = 5, min_members: int = 5) -> list[dict[str, Any]]:
    rows = [{"name": n, **_agg(sub)} for n, sub in liquid.groupby("sector")
            if len(sub) >= min_members and n not in ("-", "nan")]
    return sorted(rows, key=lambda r: -r["score"])[:top]


def rank_themes(liquid: pd.DataFrame, themes: dict[str, list[str]], top: int = 5,
                min_members: int = 3) -> list[dict[str, Any]]:
    rows = [{"name": n, **_agg(sub)} for n, codes in themes.items()
            if len(sub := liquid[liquid["code"].isin(codes)]) >= min_members]
    return sorted(rows, key=lambda r: -r["score"])[:top]


def themes_of(code: str, themes: dict[str, list[str]]) -> list[str]:
    return [n for n, codes in themes.items() if code in codes]
