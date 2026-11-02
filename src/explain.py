"""「なぜ選ばれたのか」を日本語で説明する文章を作る。"""
import math


def fmt_oku(yen):
    """円 → 「1,234億円」「2.1兆円」"""
    if yen is None or (isinstance(yen, float) and math.isnan(yen)):
        return "—"
    oku = yen / 1e8
    if oku >= 10000:
        return f"{oku / 10000:.1f}兆円"
    return f"{oku:,.0f}億円"


def rsi_label(rsi):
    if rsi >= 80:
        return "かなり過熱"
    if rsi >= 70:
        return "やや過熱"
    if rsi >= 50:
        return "勢いあり（適温）"
    return "勢い弱め"


def macd_label(macd, sig):
    if macd > sig and macd > 0:
        return "上昇の勢いが強い"
    if macd > sig:
        return "下げ止まり、上向きに転換"
    return "勢いが鈍っている"


def is_overheat(r, cfg):
    oh = cfg["overheat"]
    return (r["rsi"] == r["rsi"] and r["rsi"] >= oh["rsi"]) or \
           (r["gap25"] == r["gap25"] and r["gap25"] >= oh["ma25_gap"])


def build_reasons(r, fund):
    """✅付きの選出理由リスト。達成した条件だけを並べる。"""
    L = []
    if r["hi52w"]:
        L.append("🏆 52週高値を更新（この1年で最も高い株価）")
    if r["hi60"]:
        L.append("✅ 60日高値更新（約3か月ぶりの高値）")
    elif r["hi30"]:
        L.append("✅ 30日高値更新（約1.5か月ぶりの高値）")
    if r["vol_ratio"] == r["vol_ratio"] and r["vol_ratio"] >= 2:
        L.append(f"✅ 出来高{r['vol_ratio']:.1f}倍（直近20日の平均と比べて）")
    if r["trend"]:
        L.append("✅ 上昇トレンド継続（株価 ＞ 25日線 ＞ 75日線）")
    if fund:
        if fund.get("turnaround"):
            L.append("✅ 営業利益が赤字から黒字に転換")
        elif fund.get("op_yoy") is not None and fund["op_yoy"] >= 0.2:
            L.append(f"✅ 営業利益 {fund['op_yoy'] * 100:+.0f}%（{fund['op_basis']}）")
        if fund.get("market_cap") and fund["market_cap"] >= 1e10:
            L.append(f"✅ 時価総額 {fmt_oku(fund['market_cap'])}（規模は十分）")
    return L


def build_indicators(r):
    """補助指標（RSI・MACD）。"""
    return [
        f"RSI {r['rsi']:.0f}：{rsi_label(r['rsi'])}",
        f"MACD：{macd_label(r['macd'], r['macd_sig'])}",
    ]


def beginner_text(r, fund, limit=100):
    """初心者向け解説。100文字以内に収まるよう、優先度の低い文から落とす。"""
    parts = []
    if r["hi52w"]:
        parts.append("株価がこの1年で最も高い水準を更新しました。")
    elif r["hi60"]:
        parts.append("株価が約3か月ぶりの高値を更新しました。")
    elif r["hi30"]:
        parts.append("株価が直近1か月半の高値をつけました。")
    else:
        parts.append("株価が上昇の流れを保っています。")
    if r["vol_ratio"] == r["vol_ratio"] and r["vol_ratio"] >= 2:
        parts.append(f"出来高も普段の{r['vol_ratio']:.1f}倍に増え、買いが集まっています。")
    if fund and fund.get("turnaround"):
        parts.append("営業利益が黒字に転換し、業績も改善しています。")
    elif fund and fund.get("op_yoy") is not None and fund["op_yoy"] >= 0.2:
        parts.append(f"営業利益も前年より{fund['op_yoy'] * 100:.0f}%増えています。")
    if r["trend"]:
        parts.append("移動平均線も上向きで、流れは良好です。")

    text = ""
    for p in parts:
        if len(text) + len(p) <= limit:
            text += p
    return text or parts[0][:limit]


def build_risks(r, fund, market, regime):
    risks = []
    if r["rsi"] == r["rsi"] and r["rsi"] >= 75:
        risks.append(f"RSIが{r['rsi']:.0f}と過熱圏。利益確定売りによる反落に注意")
    if r["gap25"] == r["gap25"] and r["gap25"] >= 0.20:
        risks.append(f"25日線から{r['gap25'] * 100:.0f}%上に離れており、急騰後の反落に注意")
    if r["ret1d"] == r["ret1d"] and r["ret1d"] >= 0.07:
        risks.append(f"前日に{r['ret1d'] * 100:+.1f}%と急伸。寄り付きが高くなりやすく、飛びつき買いは禁物")
    if r["vol_ratio"] == r["vol_ratio"] and r["vol_ratio"] >= 5:
        risks.append("出来高が急増しすぎており、一時的な材料で終わる可能性")
    mc = (fund or {}).get("market_cap")
    if mc and mc < 3e10:
        risks.append("時価総額が小さく、値動きが荒くなりやすい")
    if market == "グロース":
        risks.append("グロース市場の銘柄は値動きが大きい傾向")
    if fund and fund.get("turnaround"):
        risks.append("黒字転換直後のため、業績が続くか次の決算で確認を")
    if not fund or fund.get("status") != "ok" or (fund.get("op_ttm") is None):
        risks.append("営業利益データを取得できず、業績面は未確認")
    if regime == "bad":
        risks.append("市場全体が下落基調で、順張りは勝ちにくい局面")
    if not risks:
        risks.append("高値圏での購入になるため、下がったときの撤退ラインを先に決めておく")
    return risks[:3]
