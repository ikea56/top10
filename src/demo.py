"""デモ用の架空データ生成（ネット接続なしで動作確認するため）。
値動きはランダムで、「勝てる癖」は一切仕込んでいません。"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from .config import JST

SECTORS = ["電気機器", "情報・通信業", "機械", "銀行業", "化学", "医薬品",
           "卸売業", "輸送用機器", "サービス業", "小売業", "建設業", "精密機器"]
MARKETS = ["プライム"] * 5 + ["スタンダード"] * 2 + ["グロース"] * 2
Fund = dict


def _demo_fund(code: str) -> Fund:
    r = np.random.default_rng(int(code))
    if r.random() < 0.04:
        return {"v": 3, "status": "missing", "market_cap": None, "op_ttm": None, "op_yoy": None,
                "op_basis": None, "op_annual": []}
    base = float(1e9 * np.exp(r.normal(2, 1.2)))
    turn = r.random() < 0.12                                   # 前期赤字の企業
    annual = [-0.3 * base, base * 0.2, base * 0.4, base * 0.5] if turn else \
        [base * g for g in (1.0, 0.8, 0.7, 0.6)] if r.random() < 0.5 else [base * g for g in (0.7, 0.9, 0.8, 0.6)]
    ttm = base if not turn else base * 0.2
    eps = [x / 1e8 for x in annual]
    inc = r.random() < 0.35
    prev = float(r.uniform(10, 80))
    return {
        "v": 3, "status": "ok", "market_cap": float(10 ** r.uniform(9.5, 12.3)), "op_ttm": float(ttm),
        "op_yoy": None if turn else float(r.normal(0.25, 0.4)), "op_basis": "直近四半期・前年同期比",
        "op_annual": [float(x) for x in annual], "eps_hist": [float(x) for x in eps],
        "eps_est_0y": float(eps[0] * 1.3 if not turn else abs(eps[0]) * 0.8),
        "eps_est_1y": float(eps[0] * 1.5 if r.random() < 0.7 else -1.0),
        "eps_rev": {"cur": 110.0, "d30": float(100.0 if r.random() < 0.3 else 110.0), "d90": 100.0},
        "div_cur": prev * (float(r.choice([1.12, 1.25, 1.4])) if inc else 1.0), "div_prev": prev,
        "div_basis": "forecast", "div_streak": int(r.choice([0, 0, 1, 3, 5, 11])),
        "div_ttm": prev, "div_prev12": prev * 0.9,
        "payout_latest": float(r.uniform(0.25, 0.5)), "payout_prev": float(r.uniform(0.2, 0.45)),
        "buyback": float(1e9 * r.uniform(0.5, 5)) if r.random() < 0.3 else None,
    }


def make_demo(seed: int = 7, n: int = 180, days: int = 2900
              ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], Callable[[str], Fund]]:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(datetime.now(JST).date()) - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(end=end, periods=days)

    mkt = rng.normal(0.0004, 0.009, days)
    bench: dict[str, pd.DataFrame] = {}
    for key, base, beta in (("topix", 2700.0, 1.0), ("n225", 39000.0, 1.1)):
        cl = base * np.exp(np.cumsum(mkt * beta))
        op = np.r_[cl[0], cl[:-1]] * (1 + rng.normal(0, 0.002, days))
        bench[key] = pd.DataFrame({"open": op, "high": np.maximum(op, cl) * 1.003,
                                   "low": np.minimum(op, cl) * 0.997, "close": cl}, index=dates)

    uni: list[dict] = []
    cols: dict[str, dict[str, np.ndarray]] = {k: {} for k in ("open", "high", "low", "close", "volume")}
    for i in range(n):
        code = str(1301 + i * 37)
        beta, sig = rng.uniform(0.6, 1.4), rng.uniform(0.008, 0.02)
        r = beta * mkt + rng.normal(0.0001, sig, days)
        vol = float(rng.uniform(3e5, 6e6)) * rng.lognormal(0, 0.35, days)
        spikes = rng.random(days) < 0.03
        vol[spikes] *= rng.uniform(2.2, 4.5, spikes.sum())
        if i % 7 == 0:                              # 直近で高値更新＋出来高急増させる銘柄
            r[-25:] += 0.004
            r[-1] = rng.uniform(0.03, 0.06)
            vol[-1] *= rng.uniform(2.5, 4.2)
        close = rng.uniform(400, 9000) * np.exp(np.cumsum(r))
        op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, days))
        tk = f"{code}.T"
        cols["open"][tk], cols["close"][tk], cols["volume"][tk] = op, close, vol
        cols["high"][tk] = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, days)))
        cols["low"][tk] = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, days)))
        uni.append({"code": code, "name": f"デモ銘柄{i + 1:03d}", "market": MARKETS[i % len(MARKETS)],
                    "sector": SECTORS[i % len(SECTORS)], "ticker": tk})
    panel = {k: pd.DataFrame(d, index=dates) for k, d in cols.items()}
    return pd.DataFrame(uni), panel, bench, _demo_fund
