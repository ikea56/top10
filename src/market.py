"""市場全体の状況（地合い）の判定とコメント生成。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REGIME_LABEL = {"good": "追い風", "neutral": "中立", "bad": "逆風"}


def regime_series(close: pd.Series) -> pd.Series:
    """TOPIX系列から日々の地合いを判定。good=株価>25日線>75日線 / bad=75日線割れ / それ以外 neutral。"""
    ma25, ma75 = close.rolling(25).mean(), close.rolling(75).mean()
    out = pd.Series("neutral", index=close.index, dtype=object)
    out[(close > ma25) & (ma25 > ma75)] = "good"
    out[close < ma75] = "bad"
    return out


def above_ma50_series(close: pd.Series) -> pd.Series:
    """日経平均などが50日線の上にあるか（True/False）。"""
    return (close > close.rolling(50).mean()).fillna(False)


def market_regime(close: pd.Series) -> str:
    c = close.dropna()
    return str(regime_series(c).iloc[-1]) if len(c) >= 80 else "neutral"


def build_market(bench: dict[str, pd.DataFrame], breadth_ma25: float | None,
                 n_high60: int, n_high252: int) -> dict[str, Any]:
    regime = market_regime(bench["topix"]["close"])
    parts: list[str] = []
    n225_last = n225_chg = None
    n225 = bench.get("n225")
    if n225 is not None and len(n225["close"].dropna()) >= 2:
        s = n225["close"].dropna()
        n225_last, n225_chg = float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1)
        parts.append(f"日経平均は{n225_last:,.0f}円（前営業日比{n225_chg * 100:+.1f}%）。")
    parts.append({
        "good": "TOPIXは25日線・75日線の上で、上昇トレンドが続いています。",
        "neutral": "TOPIXは短期と長期の線が交錯しており、方向感が定まりにくい状況です。",
        "bad": "TOPIXは75日線を下回り、下落基調です。",
    }[regime])
    if breadth_ma25 is not None and not np.isnan(breadth_ma25):
        parts.append(f"売買代金の条件を満たす銘柄のうち{breadth_ma25 * 100:.0f}%が25日線の上にあり、"
                     f"60日高値を更新した銘柄は{n_high60:,}社、252日高値を更新した銘柄は{n_high252:,}社です。")
    parts.append({"good": "順張りには追い風です。", "neutral": "強い銘柄だけを厳選して見ましょう。",
                  "bad": "順張りは負けやすい局面です。新規の購入は少額・慎重に。"}[regime])
    return {"regime": regime, "label": REGIME_LABEL[regime], "comment": "".join(parts),
            "n225": n225_last, "n225_chg": n225_chg}
