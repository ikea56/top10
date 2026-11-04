"""基本スコア（トレンド50＋需給25＋業績15＋地合い10＝100点）、期待値点、強気判定。

ライブ運用とバックテストの両方が `score_technical` を使う（断面DataFrameを受け取る）。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigError


def tier_points(x: Any, tiers: list[list[float]]) -> Any:
    """[[しきい値, 点数], ...] の段階点。しきい値以上のうち最も高い段階の点数を返す（重複加点なし）。
    Series / DataFrame / 数値のどれでも使える。欠損は0点。"""
    ordered = sorted(tiers, key=lambda t: t[0])
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = (x.astype(float) * 0.0).fillna(0.0)
        for th, pts in ordered:
            out = out.mask(x >= th, float(pts))
        return out
    if x is None or x != x:
        return 0.0
    pts_out = 0.0
    for th, pts in ordered:
        if x >= th:
            pts_out = float(pts)
    return pts_out


def _tier_max(tiers: list[list[float]]) -> float:
    return max(float(p) for _, p in tiers)


def max_points(sc: dict) -> dict[str, float]:
    t, d, e, m = sc["trend"], sc["demand"], sc["earnings"], sc["market"]
    return {
        "trend": max(t["high"].values()) + t["ma_alignment"] + t["macd"] + t["rsi"]["points"] + t["rel_strength"],
        "demand": _tier_max(d["vol_ratio"]) + d["up_day_volume"]["points"] + d["liquidity"]["points"],
        "earnings": _tier_max(e["op_growth"]) + _tier_max(e["mcap"]),
        "market": max(m["regime"].values()) + max(m["sector"].values()),
    }


def validate_config(cfg: Config) -> None:
    """配点の最大値が weights と一致しているか確認（設定ミスで合計が100点からずれるのを防ぐ）。"""
    got, want = max_points(cfg["scoring"]), cfg["weights"]
    for k in ("trend", "demand", "earnings", "market"):
        if abs(got[k] - want[k]) > 1e-9:
            raise ConfigError(f"配点の合計が合いません: {k} は weights で {want[k]}点ですが、"
                              f"scoring の各項目の最大合計は {got[k]}点です。config.yml を確認してください")
    if sum(want.values()) != 100:
        raise ConfigError(f"weights の合計が100点ではありません（{sum(want.values())}点）")
    ev = cfg["events"]["momentum_bonus"]
    if not all(isinstance(k, int) for k in ev):
        raise ConfigError("events.momentum_bonus のキーは整数（2, 3, 4）にしてください")


def tech_max(cfg: Config) -> float:
    """バックテスト・候補選定で使うテクニカル点（業績を除く）の最大値。"""
    w = cfg["weights"]
    return float(w["trend"] + w["demand"] + w["market"])


# ---------------------------------------------------------------- 断面スコア
def sector_points(ret20: pd.Series, sector: pd.Series, sc: dict) -> pd.Series:
    """業種ごとの平均20日騰落率で順位を付け、上位1/3・中位1/3・下位1/3に点を与える。"""
    m = sc["market"]
    pts = m["sector"]
    sec = sector.reindex(ret20.index)
    df = pd.DataFrame({"r": ret20, "s": sec}).dropna()
    g = df.groupby("s")["r"].agg(["mean", "count"])
    g = g[g["count"] >= m["sector_min_members"]]
    if g.empty:
        return pd.Series(float(pts["mid"]), index=ret20.index)
    rank = g["mean"].rank(pct=True)
    mapped = pd.Series(np.where(rank >= 2 / 3, pts["top"], np.where(rank >= 1 / 3, pts["mid"], pts["bottom"])),
                       index=g.index, dtype=float)
    return sec.map(mapped).fillna(float(pts["mid"])).astype(float)


def eligible_mask(snap: pd.DataFrame, flt: dict) -> pd.Series:
    """数値条件の除外（極端な低位株・売買代金不足・出来高不足）。ライブとバックテストで共通。"""
    return (snap["close"].notna() & (snap["volume"] > 0)
            & (snap["close"] >= flt["min_price"])
            & (snap["turnover20"] >= flt["min_turnover_yen"])
            & (snap["avgvol20"] >= flt["min_avg_volume"])
            & (snap["volume"] >= flt["min_today_volume"]))


def score_technical(snap: pd.DataFrame, regime: str, sector_pts: pd.Series, sc: dict) -> pd.DataFrame:
    """トレンド・需給・地合いの点数（業績を除く）。戻り値は snap と同じ index の DataFrame。"""
    t, d, m = sc["trend"], sc["demand"], sc["market"]
    out = pd.DataFrame(index=snap.index)

    # --- トレンド（50点）：高値更新は一番高い段階だけ ---
    h = t["high"]
    cand = pd.DataFrame({
        "n20": snap["hi20"].astype(float) * h["new20"],
        "n60": snap["hi60"].astype(float) * h["new60"],
        "near": snap["near52w"].astype(float) * h["near52w"],
        "n252": snap["hi252"].astype(float) * h["new252"]})
    out["pts_high"] = cand.max(axis=1)
    aligned = (snap["close"] > snap["ma25"]) & (snap["ma25"] > snap["ma75"])
    above25 = snap["close"] > snap["ma25"]
    out["pts_ma"] = np.where(aligned, t["ma_alignment"], np.where(above25, t["ma_partial"], 0.0))
    out["pts_macd"] = np.where((snap["macd"] > snap["macd_sig"]) & (snap["macd"] > 0), float(t["macd"]), 0.0)
    r = t["rsi"]
    out["pts_rsi"] = np.where((snap["rsi"] >= r["low"]) & (snap["rsi"] <= r["high"]), float(r["points"]), 0.0)
    out["pts_rs"] = np.where(snap["rs20"] > 0, float(t["rel_strength"]), 0.0)
    out["pts_trend"] = out[["pts_high", "pts_ma", "pts_macd", "pts_rsi", "pts_rs"]].sum(axis=1)

    # --- 需給（25点） ---
    out["pts_vol"] = tier_points(snap["vol_ratio"], d["vol_ratio"])
    up = d["up_day_volume"]
    out["pts_upvol"] = np.where((snap["ret1d"] > 0) & (snap["vol_ratio"] >= up["min_ratio"]), float(up["points"]), 0.0)
    out["pts_liq"] = np.where(snap["turnover20"] >= d["liquidity"]["min_turnover_yen"],
                              float(d["liquidity"]["points"]), 0.0)
    out["pts_demand"] = out[["pts_vol", "pts_upvol", "pts_liq"]].sum(axis=1)

    # --- 地合い（10点）：市場全体＋業種の勢い ---
    out["pts_regime"] = float(m["regime"][regime])
    out["pts_sector"] = sector_pts.reindex(snap.index).fillna(float(m["sector"]["mid"]))
    out["pts_market"] = out["pts_regime"] + out["pts_sector"]

    out["tech_base"] = out["pts_trend"] + out["pts_demand"] + out["pts_market"]
    return out


def earnings_points(fund: dict | None, sc: dict) -> tuple[float, float]:
    """業績評価（15点）＝ 営業利益の前年比 ＋ 時価総額。"""
    if not fund:
        return 0.0, 0.0
    e = sc["earnings"]
    return tier_points(fund.get("op_yoy"), e["op_growth"]), tier_points(fund.get("market_cap"), e["mcap"])


# ---------------------------------------------------------------- 期待値点・判定
def liquidity_factor(turnover: Any, es: dict) -> Any:
    return tier_points(turnover, es["liquidity_factor"])


def expected_score(total: float, win_rate: float, liq: float, es: dict) -> float:
    """期待値点 = 総合点 × (過去勝率 ÷ 基準勝率) × 流動性係数。基準勝率で割るのは総合点と同じ尺度にそろえるため。"""
    return round(total * (win_rate / es["win_rate_baseline"]) * liq, 1)


def bull_label(total: float, overheat: bool, regime: str, cfg: Config) -> tuple[str, str, str]:
    """(ラベル, CSSクラス, 補足)。過熱・逆風の日は『強気』を1段階下げる。"""
    b = cfg["bull"]
    label = "強気" if total >= b["strong"] else "やや強気" if total >= b["mild"] else "中立"
    note = ""
    if label == "強気" and overheat:
        label, note = "やや強気", "過熱気味のため格下げ"
    elif label == "強気" and regime == "bad":
        label, note = "やや強気", "市場全体が弱いため格下げ"
    return label, {"強気": "bull", "やや強気": "mild", "中立": "neutral"}[label], note
