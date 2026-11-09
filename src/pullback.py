"""押し目スコア（Pullback Score）＝ メインランキング。

目的は「いま強い（急騰した）銘柄」ではなく、「業績・株主還元が良好で、健全な押し目を形成中の優良銘柄」。
  長期上昇トレンド20 ＋ 高値からの調整率25 ＋ 50日線への接近15 ＋ 出来高の収縮10 ＋ 業績・還元の維持20 ＋ 地合い10 = 100点
  そこから「急騰済み・高値から離れすぎ・出来高だけの上昇」を減点する。
テクニカル部分（業績維持を除く80点）はバックテストにも同じ関数を使う。株価の将来を予想する処理は含まない。
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigError
from .events import EventResult
from .scoring import tier_points


def _curve(x: pd.Series, pts: list[list[float]]) -> pd.Series:
    """折れ線 [[値, 点数], ...] で連続的に採点する（範囲外は両端の点数）。欠損値は0点。"""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    v = np.interp(x.astype(float).fillna(xs[0] if ys[0] == 0 else xs[-1]).to_numpy(), xs, ys)
    return pd.Series(v, index=x.index).where(x.notna(), 0.0)


def max_points(pb: dict) -> dict[str, float]:
    t = pb["trend"]
    top = lambda pts: max(p[1] for p in pts)  # noqa: E731
    return {"trend": t["ma50_over_200"]["max"] + t["above_ma200"] + t["ma200_rising"]["max"],
            "drawdown": top(pb["drawdown_curve"]), "ma50": top(pb["ma50_gap_curve"]),
            "volume": top(pb["volume"]["curve"]), "rs": top(pb["rs"]["curve"]),
            "bounce": pb["bounce"]["above_ma5"] + top(pb["bounce"]["ret3d_curve"]),
            "quality": pb["weights"]["quality"],
            "market": max(pb["market"]["topix"].values()) + max(pb["market"]["nikkei"].values())}


def validate_config(cfg: Config) -> None:
    pb = cfg["pullback"]
    got, want = max_points(pb), pb["weights"]
    for k in want:
        if k not in got:
            raise ConfigError(f"pullback.weights に未対応の項目があります: {k}")
        if abs(got[k] - want[k]) > 1e-9:
            raise ConfigError(f"押し目スコアの配点が合いません: {k} は weights で {want[k]}点ですが、"
                              f"各項目の最大合計は {got[k]}点です（config.yml の pullback）")
    if pb["market"]["floor"] >= 0 or pb["market"]["floor"] < -50:
        raise ConfigError("pullback.market.floor は -50〜0 の負の値にしてください（-15 を推奨）")
    if sum(want.values()) != 100:
        raise ConfigError(f"pullback.weights の合計が100点ではありません（{sum(want.values())}点）")


def tech_max(cfg: Config) -> float:
    """業績維持を除いたテクニカル部分の最大点（バックテスト・候補選定用）。"""
    w = cfg["pullback"]["weights"]
    return float(100 - w["quality"])


def market_points(regime: str, nikkei_ok: bool, topix_below200: bool, cfg: Config) -> float:
    """地合い点（-15〜+10）。良い時は加点、TOPIX・日経平均が弱い時は減点して順位全体を押し下げる。"""
    m = cfg["pullback"]["market"]
    pts = float(m["topix"][regime]) + float(m["nikkei"]["above_ma50" if nikkei_ok else "below"])
    if regime == "bad":
        pts += float(m["topix_below_ma75"])
    if topix_below200:
        pts += float(m["topix_below_ma200"])
    if not nikkei_ok:
        pts += float(m["nikkei_below_ma50"])
    return max(float(m["floor"]), min(pts, float(cfg["pullback"]["weights"]["market"])))


def earnings_penalty(next_earnings: str | None, today: date, cfg: Config) -> tuple[float, int | None]:
    """決算までの営業日数（土日を除く）に応じた減点。(減点, 営業日数)。決算日が不明・過去なら (0, None)。"""
    if not next_earnings:
        return 0.0, None
    try:
        d = date.fromisoformat(next_earnings)
    except ValueError:
        return 0.0, None
    if d < today:
        return 0.0, None
    n = int(np.busday_count(today, d))
    for tier in sorted(cfg["pullback"]["earnings_penalty"], key=lambda t: t["business_days"]):
        if n <= tier["business_days"]:
            return float(tier["pts"]), n
    return 0.0, n


def stop_line(r: Any, cfg: Config) -> dict | None:
    """ATRベースの撤退ライン＝移動平均線 − ATR×倍率。50日線→200日線の順に、現在値より下にある線を使う。"""
    a = cfg["pullback"]["atr"]
    atr, close = r.get("atr"), float(r["close"])
    if not atr or atr != atr or atr <= 0:
        return None
    for name, key in (("50日線", "ma50"), ("200日線", "ma200")):
        ma = r[key]
        if ma == ma and ma < close:
            price = float(ma) - float(atr) * float(a["multiplier"])
            return {"name": f"{name}−ATR", "ma": float(ma), "atr": float(atr), "atr_pct": float(atr) / close * 100,
                    "price": price, "pct": (price / close - 1) * 100, "risk_pct": (1 - price / close) * 100}
    price = close - float(atr) * float(a["fallback_multiplier"])
    return {"name": f"現在値−ATR×{a['fallback_multiplier']:g}", "ma": None, "atr": float(atr), "atr_pct": float(atr) / close * 100,
            "price": price, "pct": (price / close - 1) * 100, "risk_pct": (1 - price / close) * 100}


def score_technical(snap: pd.DataFrame, regime: str, nikkei_ok: bool, cfg: Config,
                    topix_below200: bool = False) -> pd.DataFrame:
    """断面（snapshot）から押し目スコアのテクニカル部分と減点を計算する。"""
    pb = cfg["pullback"]
    out = pd.DataFrame(index=snap.index)
    dd = snap["dist52w"].astype(float)

    t = pb["trend"]
    spread = (snap["ma50"] / snap["ma200"] - 1).astype(float)
    out["pb_trend"] = (_curve(spread, t["ma50_over_200"]["curve"])
                       + (snap["close"] > snap["ma200"]) * float(t["above_ma200"])
                       + _curve(snap["ma200_slope"].astype(float), t["ma200_rising"]["curve"])).astype(float)
    out["pb_dd"] = _curve(dd, pb["drawdown_curve"])
    out["pb_ma50"] = _curve(snap["gap50"].astype(float), pb["ma50_gap_curve"])

    vol = pb["volume"]
    vc = snap["vol_cont"].astype(float)
    gate = dd <= vol["pullback_gate"]                                   # 調整局面（高値から-3%以下）だけ評価
    out["pb_vol"] = _curve(vc, vol["curve"]).where(gate, 0.0)
    out["pb_rs"] = _curve(snap["rs60"].astype(float), pb["rs"]["curve"])
    bn = pb["bounce"]
    out["pb_bounce"] = ((snap["close"] > snap["ma5"]) * float(bn["above_ma5"])
                        + _curve(snap["ret3d"].astype(float), bn["ret3d_curve"])).where(gate, 0.0).astype(float)

    out["pb_market"] = market_points(regime, nikkei_ok, topix_below200, cfg)

    p = pb["penalties"]
    flags = pd.DataFrame({
        "pen_surge5": snap["ret5d"] >= p["surge_5d"]["ret"],
        "pen_surge20": snap["ret20d"] >= p["surge_20d"]["ret"],
        "pen_far": dd <= p["far_from_high"]["dd"],
        "pen_spike": (snap["vol_ratio"] >= p["volume_spike"]["vol_ratio"]) & (snap["ret1d"] >= p["volume_spike"]["ret1d"]),
        "pen_expand": (vc >= p["volume_expansion"]["ratio"]) & (dd <= vol["pullback_gate"]),
    }).fillna(False)
    pts = {"pen_surge5": p["surge_5d"]["pts"], "pen_surge20": p["surge_20d"]["pts"], "pen_far": p["far_from_high"]["pts"],
           "pen_spike": p["volume_spike"]["pts"], "pen_expand": p["volume_expansion"]["pts"]}
    out = out.join(flags)
    out["pb_penalty"] = sum(flags[k].astype(float) * v for k, v in pts.items())
    out["pb_sum"] = out[["pb_trend", "pb_dd", "pb_ma50", "pb_vol", "pb_rs", "pb_bounce", "pb_market"]].sum(axis=1)
    out["pb_tech_scaled"] = ((out["pb_sum"] - out["pb_penalty"]).clip(lower=0) / tech_max(cfg) * 100)
    return out


def quality_points(fund: dict | None, ev: EventResult | None, cfg: Config) -> tuple[float, list[str]]:
    """業績・株主還元の維持（20点）。黒字転換・最高益・増配・連続増配・自社株買いなどを評価（上限あり）。"""
    q = cfg["pullback"]["quality"]
    cap = float(cfg["pullback"]["weights"]["quality"])
    keys: list[str] = []
    if fund and (fund.get("op_ttm") or 0) > 0 and (fund.get("op_yoy") is None or fund["op_yoy"] >= 0):
        keys.append("profit_ok")
    if ev:
        keys += [k for k in ("turnaround", "record_high", "dividend", "streak", "buyback", "sustained", "revision")
                 if ev.has(k)]
    return min(sum(float(q[k]) for k in keys), cap), keys


def total(pb_sum: float, quality: float, penalty: float, extra: float = 0.0) -> float:
    """押し目スコア＋業績点−減点−追加減点（決算接近など）。地合い点が負のときは pb_sum に含まれて全体を下げる。"""
    return max(0.0, pb_sum + quality - penalty - extra)


def label(main: float, serious_negative: bool, cfg: Config) -> tuple[str, str, str]:
    """(ラベル, CSSクラス, 補足)。下方修正・減配があれば『様子見』に下げる。"""
    lb = cfg["pullback"]["labels"]
    name = "押し目良好" if main >= lb["good"] else "押し目候補" if main >= lb["fair"] else "様子見"
    note = ""
    if serious_negative and name != "様子見":
        name, note = "様子見", "下方修正・減配の開示があるため"
    return name, {"押し目良好": "bull", "押し目候補": "mild", "様子見": "neutral"}[name], note
