"""「なぜ選ばれたのか」を初心者向けの日本語で説明する文章を作る。

方針：株価がこれから上がる、とは書かない。「いまどんな状態か（事実）」だけを書く。
"""
from __future__ import annotations

import math
from typing import Any

from .config import Config
from .events import EventResult


def fmt_oku(yen: float | None) -> str:
    """円 → 「1,234億円」「2.1兆円」"""
    if yen is None or (isinstance(yen, float) and math.isnan(yen)):
        return "—"
    oku = yen / 1e8
    return f"{oku / 10000:.1f}兆円" if oku >= 10000 else f"{oku:,.0f}億円"


def rsi_label(rsi: float) -> str:
    if rsi >= 80:
        return "かなり過熱"
    if rsi >= 70:
        return "やや過熱"
    return "勢いあり（適温）" if rsi >= 50 else "勢い弱め"


def macd_label(macd: float, sig: float) -> str:
    if macd > sig and macd > 0:
        return "上昇の勢いが強い"
    return "下げ止まり、上向きに転換" if macd > sig else "勢いが鈍っている"


def is_overheat(r: Any, cfg: Config) -> bool:
    oh = cfg["overheat"]
    return bool((r["rsi"] == r["rsi"] and r["rsi"] >= oh["rsi"])
                or (r["gap25"] == r["gap25"] and r["gap25"] >= oh["ma25_gap"]))


def build_indicators(r: Any) -> list[str]:
    return [f"RSI {r['rsi']:.0f}：{rsi_label(r['rsi'])}", f"MACD：{macd_label(r['macd'], r['macd_sig'])}"]


def reason_text(r: Any, fund: dict | None, ev: EventResult, regime: str, limit: int = 100) -> str:
    """初心者向けの選出理由。優先度の高い順に並べ、100文字を超える文は入れない。"""
    phrases: list[str] = []
    if ev.has("hi252"):
        phrases.append("52週高値を更新")
    elif ev.has("near52w"):
        phrases.append("52週高値付近を推移")
    elif r["hi60"]:
        phrases.append("60日高値を更新")
    elif r["hi20"]:
        phrases.append("20日高値を更新")
    vr = r["vol_ratio"]
    if vr == vr and vr >= 2:
        phrases.append(f"出来高が通常の{vr:.1f}倍に増加")
    if ev.has("turnaround"):
        phrases.append("営業利益が黒字転換")
    elif fund and fund.get("op_yoy") is not None and fund["op_yoy"] >= 0.2:
        phrases.append(f"営業利益が前年比{fund['op_yoy'] * 100:.0f}%増")
    if ev.has("record_high"):
        phrases.append("最高益を更新")
    if ev.has("dividend"):
        phrases.append("増配を実施")
    if regime == "good":
        phrases.append("市場全体の地合いも良好")
    if (r["close"] > r["ma25"] > r["ma75"]):
        phrases.append("上昇トレンドが継続")

    text = ""
    for p in phrases:
        cand = f"{text}{p}。"
        if len(cand) <= limit:
            text = cand
    return text or "株価の勢いと出来高から、条件を満たした銘柄です。"


def build_risks(r: Any, fund: dict | None, market: str, regime: str, ev: EventResult) -> list[str]:
    risks: list[str] = []
    if r["rsi"] == r["rsi"] and r["rsi"] >= 75:
        risks.append(f"RSIが{r['rsi']:.0f}と過熱圏。利益確定売りによる反落に注意")
    if r["gap25"] == r["gap25"] and r["gap25"] >= 0.20:
        risks.append(f"25日線から{r['gap25'] * 100:.0f}%上に離れており、急騰後の反落に注意")
    if r["ret1d"] == r["ret1d"] and r["ret1d"] >= 0.07:
        risks.append(f"前日に{r['ret1d'] * 100:+.1f}%と急伸。飛びつき買いは禁物")
    if r["vol_ratio"] == r["vol_ratio"] and r["vol_ratio"] >= 5:
        risks.append("出来高が急増しすぎており、一時的な材料で終わる可能性")
    mc = (fund or {}).get("market_cap")
    if mc and mc < 3e10:
        risks.append("時価総額が小さく、値動きが荒くなりやすい")
    if market == "グロース":
        risks.append("グロース市場の銘柄は値動きが大きい傾向")
    if ev.has("turnaround"):
        risks.append("黒字転換直後のため、業績が続くか次の決算で確認を")
    if ev.uses_forecast:
        risks.append("評価イベントに会社・アナリストの予想を含みます。予想は外れることがあります")
    if not fund or fund.get("status") != "ok" or fund.get("op_ttm") is None:
        risks.append("営業利益データを取得できず、業績面は未確認")
    if regime == "bad":
        risks.append("市場全体が下落基調で、順張りは勝ちにくい局面")
    return (risks or ["高値圏での購入になるため、下がったときの撤退ラインを先に決めておく"])[:3]
