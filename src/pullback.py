"""押し目スコア（Pullback Score）＝ メインランキング。

目的は「いま強い（急騰した）銘柄」ではなく、「業績・株主還元が良好で、健全な押し目を形成中の優良銘柄」。
  長期上昇トレンド20 ＋ 高値からの調整率25 ＋ 50日線への接近15 ＋ 出来高の収縮10 ＋ 業績・還元の維持20 ＋ 地合い10 = 100点
  そこから「急騰済み・高値から離れすぎ・出来高だけの上昇」を減点する。
テクニカル部分（業績維持を除く80点）はバックテストにも同じ関数を使う。株価の将来を予想する処理は含まない。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigError
from .events import EventResult
from .scoring import tier_points


def _range_points(x: pd.Series, tiers: list[list[float]]) -> pd.Series:
    """[[下限, 上限, 点数], ...] のうち、当てはまる最高点を返す。"""
    out = pd.Series(0.0, index=x.index)
    for lo, hi, pts in tiers:
        out = out.mask((x >= lo) & (x <= hi) & (out < pts), float(pts))
    return out


def max_points(pb: dict) -> dict[str, float]:
    return {"trend": sum(pb["trend"].values()),
            "drawdown": max(t[2] for t in pb["drawdown_tiers"]),
            "ma50": max(t[2] for t in pb["ma50_gap_tiers"]),
            "volume": max(p for _, p in pb["volume"]["tiers"]),
            "quality": pb["weights"]["quality"],
            "market": max(pb["market"]["topix"].values()) + max(pb["market"]["nikkei"].values())}


def validate_config(cfg: Config) -> None:
    pb = cfg["pullback"]
    got, want = max_points(pb), pb["weights"]
    for k in want:
        if abs(got[k] - want[k]) > 1e-9:
            raise ConfigError(f"押し目スコアの配点が合いません: {k} は weights で {want[k]}点ですが、"
                              f"各項目の最大合計は {got[k]}点です（config.yml の pullback）")
    if sum(want.values()) != 100:
        raise ConfigError(f"pullback.weights の合計が100点ではありません（{sum(want.values())}点）")


def tech_max(cfg: Config) -> float:
    """業績維持を除いたテクニカル部分の最大点（バックテスト・候補選定用）。"""
    w = cfg["pullback"]["weights"]
    return float(100 - w["quality"])


def score_technical(snap: pd.DataFrame, regime: str, nikkei_ok: bool, cfg: Config) -> pd.DataFrame:
    """断面（snapshot）から押し目スコアのテクニカル部分と減点を計算する。"""
    pb = cfg["pullback"]
    out = pd.DataFrame(index=snap.index)
    dd = snap["dist52w"].astype(float)

    t = pb["trend"]
    out["pb_trend"] = ((snap["ma50"] > snap["ma200"]) * t["ma50_over_200"]
                       + (snap["close"] > snap["ma200"]) * t["above_ma200"]
                       + (snap["ma200_slope"] > 0) * t["ma200_rising"]).astype(float)
    out["pb_dd"] = _range_points(dd, pb["drawdown_tiers"])
    out["pb_ma50"] = _range_points(snap["gap50"].astype(float), pb["ma50_gap_tiers"])

    vol = pb["volume"]
    vc = snap["vol_cont"].astype(float)
    vpts = tier_points(-vc, [[-th, p] for th, p in vol["tiers"]])          # 比率が小さいほど高得点
    out["pb_vol"] = vpts.where(dd <= vol["pullback_gate"], 0.0)

    m = pb["market"]
    out["pb_market"] = float(m["topix"][regime]) + float(m["nikkei"]["above_ma50" if nikkei_ok else "below"])

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
    out["pb_sum"] = out[["pb_trend", "pb_dd", "pb_ma50", "pb_vol", "pb_market"]].sum(axis=1)
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


def total(pb_sum: float, quality: float, penalty: float) -> float:
    return max(0.0, pb_sum + quality - penalty)


def label(main: float, serious_negative: bool, cfg: Config) -> tuple[str, str, str]:
    """(ラベル, CSSクラス, 補足)。下方修正・減配があれば『様子見』に下げる。"""
    lb = cfg["pullback"]["labels"]
    name = "押し目良好" if main >= lb["good"] else "押し目候補" if main >= lb["fair"] else "様子見"
    note = ""
    if serious_negative and name != "様子見":
        name, note = "様子見", "下方修正・減配の開示があるため"
    return name, {"押し目良好": "bull", "押し目候補": "mild", "様子見": "neutral"}[name], note
