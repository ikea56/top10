"""テクニカル指標とテクニカル点（全銘柄×全日付を一括計算）。

ライブ運用（最新日だけ使う）とバックテスト（全日付を使う）で同じ関数を使うので、
「バックテストの結果」と「画面に出るスコア」が食い違いません。
"""
import numpy as np
import pandas as pd


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.where(ad > 0)
    out = 100 - 100 / (1 + rs)
    return out.where(~((ad == 0) & (au > 0)), 100.0)


def compute_features(panel):
    """panel: {open, high, low, close, volume} の幅広DataFrame（行=日付, 列=ティッカー）"""
    c, v = panel["close"], panel["volume"]
    f = {"close": c}
    f["ma25"] = c.rolling(25, min_periods=25).mean()
    f["ma75"] = c.rolling(75, min_periods=75).mean()
    # 高値更新 = 終値がN日間の終値の最高値（当日を含む）になった
    f["hi60"] = (c >= c.rolling(60, min_periods=60).max()) & c.notna()
    f["hi30"] = (c >= c.rolling(30, min_periods=30).max()) & c.notna()
    f["hi52w"] = (c >= c.rolling(250, min_periods=200).max()) & c.notna()
    # 出来高倍率 = 当日出来高 ÷ 「前日までの」20日平均（当日を含めない）
    avgv = v.shift(1).rolling(20, min_periods=15).mean()
    f["vol_ratio"] = (v / avgv.where(avgv > 0)).replace([np.inf, -np.inf], np.nan)
    f["turnover20"] = (c * v).rolling(20, min_periods=15).mean()
    f["trend"] = (c > f["ma25"]) & (f["ma25"] > f["ma75"])
    f["rsi"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_sig"], f["macd_hist"] = macd, sig, macd - sig
    f["gap25"] = c / f["ma25"] - 1
    f["ret1d"] = c.pct_change(fill_method=None)
    f["ret20d"] = c.pct_change(20, fill_method=None)
    return f


def tech_points(f, sc):
    """テクニカル点（高値更新・出来高・トレンド）。戻り値は各点数の幅広DataFrame。"""
    zeros = pd.DataFrame(0.0, index=f["close"].index, columns=f["close"].columns)

    pts_high = zeros.mask(f["hi30"], float(sc["high30"])).mask(f["hi60"], float(sc["high60"]))

    pts_vol = zeros.copy()
    for th, p in sorted(sc["vol_ratio"], key=lambda x: x[0]):   # 低い段階から順に上書き
        pts_vol = pts_vol.mask(f["vol_ratio"] >= th, float(p))

    pts_trend = zeros.mask(f["trend"], float(sc["trend"]))
    raw = pts_high + pts_vol + pts_trend
    return {"pts_high": pts_high, "pts_vol": pts_vol, "pts_trend": pts_trend, "tech_raw": raw}


def tech_max(sc):
    return sc["high60"] + max(p for _, p in sc["vol_ratio"]) + sc["trend"]
