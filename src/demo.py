"""デモ用の架空データ生成（ネット接続なしで動作確認するため）。
値動きはランダムで、「勝てる癖」は一切仕込んでいません。"""
from datetime import datetime

import numpy as np
import pandas as pd

from .config import JST

SECTORS = ["電気機器", "情報・通信業", "機械", "銀行業", "化学", "医薬品",
           "卸売業", "輸送用機器", "サービス業", "小売業", "建設業", "精密機器"]
MARKETS = ["プライム"] * 5 + ["スタンダード"] * 2 + ["グロース"] * 2


def make_demo(seed=7, n=180, days=820):
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(datetime.now(JST).date())
    end = today - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(end=end, periods=days)

    mkt = rng.normal(0.0004, 0.009, days)
    bench = {}
    for key, base, beta in (("topix", 2700.0, 1.0), ("n225", 39000.0, 1.1)):
        cl = base * np.exp(np.cumsum(mkt * beta))
        op = np.r_[cl[0], cl[:-1]] * (1 + rng.normal(0, 0.002, days))
        bench[key] = pd.DataFrame({"open": op, "high": np.maximum(op, cl) * 1.003,
                                   "low": np.minimum(op, cl) * 0.997, "close": cl}, index=dates)

    uni, O, H, L, C, V = [], {}, {}, {}, {}, {}
    for i in range(n):
        code = str(1301 + i * 37)
        beta, sig = rng.uniform(0.6, 1.4), rng.uniform(0.008, 0.02)
        r = beta * mkt + rng.normal(0.0001, sig, days)
        base_vol = float(rng.uniform(3e5, 6e6))
        vol = base_vol * rng.lognormal(0, 0.35, days)
        spikes = rng.random(days) < 0.03
        vol[spikes] *= rng.uniform(2.2, 4.5, spikes.sum())
        if i % 7 == 0:                              # 直近で高値更新＋出来高急増させる銘柄
            r[-25:] += 0.004
            r[-1] = rng.uniform(0.03, 0.06)
            vol[-1] *= rng.uniform(2.5, 4.2)
        close = rng.uniform(400, 9000) * np.exp(np.cumsum(r))
        op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, days))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
        tk = f"{code}.T"
        O[tk], H[tk], L[tk], C[tk], V[tk] = op, hi, lo, close, vol
        uni.append({"code": code, "name": f"デモ銘柄{i + 1:03d}", "market": MARKETS[i % len(MARKETS)],
                    "sector": SECTORS[i % len(SECTORS)], "ticker": tk})

    panel = {k: pd.DataFrame(d, index=dates) for k, d in
             (("open", O), ("high", H), ("low", L), ("close", C), ("volume", V))}

    def fund_fn(code):
        r = np.random.default_rng(int(code))
        if r.random() < 0.04:
            return {"status": "missing", "market_cap": None, "op_ttm": None,
                    "op_yoy": None, "op_basis": None, "turnaround": False}
        turnaround = bool(r.random() < 0.05)
        op_ttm = float(1e9 * np.exp(r.normal(2, 1.2)) * (1 if r.random() > 0.06 else -1))
        return {"status": "ok", "market_cap": float(10 ** r.uniform(9.5, 12.3)), "op_ttm": op_ttm,
                "op_yoy": None if turnaround else float(r.normal(0.25, 0.4)),
                "op_basis": "直近四半期・前年同期比", "turnaround": turnaround}

    return pd.DataFrame(uni), panel, bench, fund_fn
