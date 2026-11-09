"""「なぜ選ばれたのか」を初心者向けの日本語で説明する文章を作る。

方針：株価がこれから上がる、とは書かない。「いまどんな状態か（事実）」だけを書く。
"""
from __future__ import annotations

import math
from datetime import date
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


# ====================================================================== 押し目ランキング用の表示文
def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def pullback_status(r: Any, cfg: Config, regime: str, nikkei_ok: bool) -> list[str]:
    """【押し目状況】満たしている項目だけを✅で並べる。"""
    pb = cfg["pullback"]
    out: list[str] = []
    if r["ma50"] > r["ma200"]:
        out.append("長期上昇トレンド（50日線＞200日線）")
    dd = r["dist52w"]
    if -0.25 <= dd <= -0.03:
        out.append(f"高値から{_pct(dd)}（52週高値比）")
    elif dd > -0.03:
        out.append(f"ほぼ高値圏（高値から{_pct(dd)}・調整は浅め）")
    if -0.06 <= r["gap50"] <= 0.07:
        out.append(f"50日線付近（乖離{r['gap50'] * 100:+.1f}%）")
    vc = r["vol_cont"]
    if vc == vc and vc <= pb["volume"]["status_ratio"] and dd <= pb["volume"]["pullback_gate"]:
        out.append(f"出来高縮小（直近10日が50日平均の{vc:.2f}倍）")
    rs = r.get("rs60")
    if rs is not None and rs == rs and rs >= 0.03:
        out.append(f"相対力が強い（直近60日でTOPIXを{rs * 100:+.1f}%上回る）")
    if dd <= pb["volume"]["pullback_gate"] and r.get("ma5") == r.get("ma5") and r["close"] > r["ma5"]:
        out.append("短期の下げ止まり（株価が5日線の上）")
    if regime == "good" or nikkei_ok:
        out.append("地合いも良好（" + ("TOPIX追い風" if regime == "good" else "TOPIX中立") +
                   ("・日経平均は50日線の上）" if nikkei_ok else "）"))
    return out


def strengths(ev: EventResult, td: Any) -> list[dict]:
    """【強み】yfinance由来のイベントとTDnetの開示を、同じ種類は1つにまとめて並べる。"""
    items: dict[str, dict] = {}
    fmap = {"turnaround": "黒字転換", "record_high": "最高益更新", "dividend": "増配", "streak": None,
            "buyback": "自社株買い", "revision": "上方修正", "sustained": "継続黒字化"}
    for i in ev.items:
        if i.key in fmap and i.key != "bonus":
            key = {"dividend": "dividend_increase", "revision": "upward_revision"}.get(i.key, i.key)
            items[key] = {"label": i.label, "src": "予想" if i.basis == "予想" else "実績"}
    if td is not None and getattr(td, "status", "") == "ok":
        for h in td.hits:
            if h.points > 0:
                items[h.kind] = {"label": f"{h.label}（TDnet {h.date[5:].replace('-', '/')}）", "src": "TDnet"}
    return list(items.values())


def cautions(r: Any, fund: dict | None, td: Any, ev: EventResult, cfg: Config, regime: str, market: str,
             today: Any, earn_pen: float = 0.0, earn_days: int | None = None, down_src: list | None = None) -> list[dict]:
    """【注意事項】重要度の高い順（level=high は赤、それ以外は黄）。"""
    cp, pen = cfg["pullback"]["caution"], cfg["pullback"]["penalties"]
    out: list[dict] = []
    dpen = int(cfg["tdnet"]["downward_recent"]["penalty"])
    if td is not None and td.status == "ok":
        for h in td.hits:
            if h.kind == "downward_revision":
                recent = h.age <= int(cfg["tdnet"]["downward_recent"]["days"])
                out.append({"label": f"下方修正（{h.date.replace('-', '/')}開示）" +
                            (f"・総合点−{dpen}点、判定は様子見" if recent else "・大幅減点"), "level": "high"})
            elif h.kind == "dividend_cut":
                out.append({"label": f"減配の開示（{h.date.replace('-', '/')}）・減点", "level": "high"})
            elif h.kind == "unknown_revision":
                out.append({"label": f"修正あり（未判定）：{h.date.replace('-', '/')}開示。増額か減額か確認してください", "level": "mid"})
    elif td is not None:
        out.append({"label": "TDnetを確認できていません（TOB・下方修正などの有無が未確認）", "level": "high"})
    if down_src and "EPS予想" in down_src and "TDnet" not in down_src:
        out.append({"label": f"アナリストの予想EPSが直近30日で下方修正（総合点−{dpen}点、判定は様子見）", "level": "high"})
    ne = (fund or {}).get("next_earnings")
    if earn_pen > 0 and ne:
        out.append({"label": f"決算接近（{ne[5:].replace('-', '/')}発表予定・あと{earn_days}営業日）・窓開けリスクで総合点−{earn_pen:.0f}点", "level": "high"})
    elif ne and 0 <= (date.fromisoformat(ne) - today).days <= cp["earnings_days"]:
        out.append({"label": f"決算接近（{ne[5:].replace('-', '/')}発表予定）・発表で大きく動く可能性", "level": "mid"})
    v20, v60 = r.get("vol20"), r.get("vol60")
    if v20 == v20 and v60 == v60 and v60 and v20 / v60 >= cp["vol_ratio"]:
        out.append({"label": f"ボラティリティ上昇（直近の値動きが60日平均の{v20 / v60:.1f}倍）", "level": "mid"})
    if r["ret5d"] == r["ret5d"] and r["ret5d"] >= pen["surge_5d"]["ret"]:
        out.append({"label": f"直近5日で{_pct(r['ret5d'])}の急騰後（減点）", "level": "mid"})
    if r["dist52w"] <= pen["far_from_high"]["dd"]:
        out.append({"label": f"高値から{_pct(r['dist52w'])}と大きく下落（減点）", "level": "mid"})
    if regime == "bad":
        out.append({"label": "市場全体が下落基調（地合い減点で総合点が下がっています）", "level": "mid"})
    mc = (fund or {}).get("market_cap")
    if mc and mc < 3e10:
        out.append({"label": "時価総額が小さく値動きが荒くなりやすい", "level": "low"})
    if market == "グロース":
        out.append({"label": "グロース市場は値動きが大きい傾向", "level": "low"})
    if ev.uses_forecast:
        out.append({"label": "強みの一部は会社・アナリストの予想に基づく（外れることがあります）", "level": "low"})
    if not fund or fund.get("status") != "ok" or fund.get("op_ttm") is None:
        out.append({"label": "営業利益データを取得できず業績面は未確認", "level": "low"})
    order = {"high": 0, "mid": 1, "low": 2}
    return sorted(out, key=lambda c: order[c["level"]])[:6]


def pullback_reason(r: Any, stren: list[dict], regime: str, nikkei_ok: bool, limit: int = 100) -> str:
    """初心者向けの選出理由（100文字以内）。『いまの状態』だけを書き、将来の値動きは予想しない。"""
    phrases: list[str] = []
    dd = r["dist52w"]
    if -0.25 <= dd <= -0.03:
        phrases.append(f"高値から{_pct(dd)}まで調整")
    if -0.06 <= r["gap50"] <= 0.07:
        phrases.append("50日線付近で推移")
    if r["vol_cont"] == r["vol_cont"] and r["vol_cont"] <= 1.0 and dd <= -0.03:
        phrases.append("出来高は縮小")
    names = [s["label"].split("（")[0] for s in stren if s["label"].split("（")[0] in
             ("増配", "自社株買い", "上方修正", "黒字転換", "最高益更新", "大型受注", "株式分割")][:3]
    if names:
        phrases.append("・".join(names) + "など業績・株主還元も良好")
    if regime == "good":
        phrases.append("市場全体の地合いも良好")
    elif nikkei_ok:
        phrases.append("日経平均は50日線の上")
    if r["ma50"] > r["ma200"]:
        phrases.append("長期の上昇トレンドは継続")
    text = ""
    for p in phrases:
        cand = f"{text}{p}。"
        if len(cand) <= limit:
            text = cand
    return text or "長期トレンドを保ったまま、調整局面にある銘柄です。"
