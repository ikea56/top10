"""市場全体の状況（地合い）の判定とコメント生成。"""
import numpy as np


def market_regime(close_series):
    """TOPIX系列から地合いを判定。good / neutral / bad"""
    c = close_series.dropna()
    if len(c) < 80:
        return "neutral"
    ma25, ma75 = c.rolling(25).mean().iloc[-1], c.rolling(75).mean().iloc[-1]
    last = c.iloc[-1]
    if last > ma25 > ma75:
        return "good"
    if last < ma75:
        return "bad"
    return "neutral"


def build_market(bench, breadth_ma25, n_high60, n_universe):
    topix = bench["topix"]["close"].dropna()
    regime = market_regime(topix)
    label = {"good": "追い風", "neutral": "中立", "bad": "逆風"}[regime]

    parts = []
    n225 = bench.get("n225")
    n225_last = n225_chg = None
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
                     f"60日高値を更新した銘柄は{n_high60:,}社です。")
    parts.append({
        "good": "順張りには追い風です。",
        "neutral": "強い銘柄だけを厳選して見ましょう。",
        "bad": "順張りは負けやすい局面です。新規の購入は少額・慎重に。",
    }[regime])

    return {"regime": regime, "label": label, "comment": "".join(parts),
            "n225": n225_last, "n225_chg": n225_chg}
