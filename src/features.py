"""テクニカル指標（全銘柄×全日付を一括計算）と、ある日の断面（snapshot）の取り出し。

ライブ運用とバックテストは同じ `compute_features` → `snapshot` → `scoring.score_technical` を通るため、
「検証で使ったスコア」と「画面に出るスコア」が食い違いません。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Panel = dict[str, pd.DataFrame]       # open/high/low/close/volume（行=日付, 列=ティッカー）
Features = dict[str, pd.DataFrame]


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + au / ad.where(ad > 0))
    return out.where(~((ad == 0) & (au > 0)), 100.0)


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    """メモリ節約（10年分でも動くように）：小数は32bitに。"""
    return df.astype("float32") if df.dtypes.iloc[0].kind == "f" else df


def compute_features(panel: Panel, topix_close: pd.Series | None, sc: dict) -> Features:
    """全指標を計算する。高値更新は『終値がN日間の終値の最高値になった』で判定（当日を含む）。"""
    w = sc["trend"]["windows"]
    near_ratio = float(sc["trend"]["near_high_ratio"])
    c, v = panel["close"], panel["volume"]
    max_long = c.rolling(w["long"], min_periods=int(w["long"] * 0.8)).max()

    f: Features = {"close": c, "volume": v}
    f["ma25"] = c.rolling(25, min_periods=25).mean()
    f["ma75"] = c.rolling(75, min_periods=75).mean()
    f["hi20"] = (c >= c.rolling(w["short"], min_periods=w["short"]).max()) & c.notna()
    f["hi60"] = (c >= c.rolling(w["mid"], min_periods=w["mid"]).max()) & c.notna()
    f["hi252"] = (c >= max_long) & c.notna()
    f["near52w"] = (c >= max_long * near_ratio) & c.notna()
    f["dist52w"] = c / max_long - 1

    avg_prev = v.shift(1).rolling(20, min_periods=15).mean()      # 当日を含めない20日平均
    f["vol_ratio"] = (v / avg_prev.where(avg_prev > 0)).replace([np.inf, -np.inf], np.nan)
    f["avgvol20"] = v.rolling(20, min_periods=15).mean()
    f["turnover20"] = (c * v).rolling(20, min_periods=15).mean()

    f["rsi"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd"], f["macd_sig"] = macd, macd.ewm(span=9, adjust=False).mean()
    f["gap25"] = c / f["ma25"] - 1
    f["ret1d"] = c.pct_change(fill_method=None)
    f["ret20d"] = c.pct_change(20, fill_method=None)

    # --- 押し目スコア用 ---
    f["ma50"] = c.rolling(50, min_periods=50).mean()
    f["ma200"] = c.rolling(200, min_periods=200).mean()
    f["ma200_slope"] = f["ma200"] / f["ma200"].shift(20) - 1
    f["gap50"] = c / f["ma50"] - 1
    f["ret5d"] = c.pct_change(5, fill_method=None)
    f["vol_cont"] = v.rolling(10, min_periods=8).mean() / v.rolling(50, min_periods=40).mean()   # 出来高の収縮率

    if topix_close is not None and len(topix_close.dropna()) > 21:
        tx = topix_close.reindex(c.index).ffill()
        f["rs20"] = f["ret20d"].sub(tx.pct_change(20, fill_method=None), axis=0)
    else:
        f["rs20"] = f["ret20d"]
    return {k: _slim(df) for k, df in f.items()}


def snapshot(f: Features, row: int) -> pd.DataFrame:
    """ある1日の断面（行=ティッカー, 列=指標）。row=-1 で最新日。"""
    return pd.DataFrame({k: df.iloc[row] for k, df in f.items()})


def add_live_extras(f: Features) -> Features:
    """毎朝のランキング専用の指標（バックテストでは使わない＝メモリ節約）。"""
    c, r1 = f["close"], f["ret1d"]
    f["vol20"] = r1.rolling(20, min_periods=15).std()
    f["vol60"] = r1.rolling(60, min_periods=40).std()
    f["max_ret60"] = r1.rolling(60, min_periods=40).max()
    f["range10"] = c.rolling(10, min_periods=10).max() / c.rolling(10, min_periods=10).min() - 1
    return {k: (v.astype("float32") if v.dtypes.iloc[0].kind == "f" else v) for k, v in f.items()}
