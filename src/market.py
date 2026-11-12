"""市場全体の状況（地合い）の判定とコメント生成。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REGIME_LABEL = {"good": "追い風", "neutral": "中立", "bad": "逆風"}


def regime_series(close: pd.Series) -> pd.Series:
    """TOPIX系列から日々の地合いを判定。good=株価>25日線>75日線 / bad=75日線割れ / それ以外 neutral。"""
    ma25, ma75 = close.rolling(25).mean(), close.rolling(75).mean()
    out = pd.Series("neutral", index=close.index, dtype=object)
    out[(close > ma25) & (ma25 > ma75)] = "good"
    out[close < ma75] = "bad"
    return out


def above_ma50_series(close: pd.Series) -> pd.Series:
    """日経平均などが50日線の上にあるか（True/False）。"""
    return (close > close.rolling(50).mean()).fillna(False)


def market_regime(close: pd.Series) -> str:
    c = close.dropna()
    return str(regime_series(c).iloc[-1]) if len(c) >= 80 else "neutral"


STANCE = {
    "buy": {"emoji": "🟢", "label": "買い優勢", "cls": "bull",
            "message": "押し目が機能しやすい環境です。",
            "guide": ["押し目候補への新規エントリーを検討できます", "それでも1銘柄に集中せず、分散して少しずつ"]},
    "neutral": {"emoji": "🟡", "label": "中立", "cls": "mild",
                "message": "良い材料と悪い材料が混在しています。",
                "guide": ["厳選してエントリー（上位でも点数・注意事項を確認）", "買う金額は控えめに"]},
    "defense": {"emoji": "🔴", "label": "防御優先", "cls": "bear",
                "message": "現在は押し目が失敗しやすい環境です。新規エントリーは慎重にしてください。",
                "guide": ["新規買いは慎重に", "現金比率を高める", "保有銘柄の監視を優先する（③保有銘柄監視を確認）"]},
}


def _vs_ma(s: pd.Series | None, n: int) -> tuple[bool | None, float | None]:
    """(移動平均線の上にあるか, 乖離率)。データ不足・無しなら (None, None)。"""
    if s is None:
        return None, None
    s = s.dropna()
    if len(s) < n:
        return None, None
    ma = float(s.rolling(n).mean().iloc[-1])
    return bool(s.iloc[-1] > ma), float(s.iloc[-1] / ma - 1)


def market_stance(bench: dict[str, pd.DataFrame], cfg: Any) -> dict[str, Any]:
    """相場環境を3段階で判定する（個別銘柄ではなく市場全体）。
    買い優勢：TOPIXが75日線・200日線の上、日経平均が75日線の上（グロースが取得でき、75日線の下なら中立に格下げ）
    防御優先：TOPIXが75日線・200日線の下、かつグロースが75日線の下（グロースが取得できない時は日経平均が75日線の下）
    それ以外：中立。売り推奨ではなく、新規買いを控える目安です。"""
    sc = cfg["stance"]
    mid, lng = int(sc["ma_mid"]), int(sc["ma_long"])
    g = lambda k: bench[k]["close"] if k in bench else None  # noqa: E731
    t75, t75d = _vs_ma(g("topix"), mid)
    t200, t200d = _vs_ma(g("topix"), lng)
    n75, n75d = _vs_ma(g("n225"), mid)
    g75, g75d = _vs_ma(g("growth"), mid)
    if t75 is None:
        key = "neutral"
    elif not t75 and t200 is False and (g75 is False if g75 is not None else n75 is False):
        key = "defense"
    elif t75 and t200 is not False and n75 is not False and g75 is not False:
        key = "buy"
    else:
        key = "neutral"
    checks = []
    for name, n, ok, d in (("TOPIX", mid, t75, t75d), ("TOPIX", lng, t200, t200d), ("日経平均", mid, n75, n75d),
                           ("東証グロース250", mid, g75, g75d)):
        checks.append({"name": f"{name}が{n}日線の{'上' if ok else '下'}" if ok is not None else f"{name}（{n}日線）は判定できず",
                       "ok": ok, "gap": None if d is None else round(d * 100, 1)})
    reasons = [(c["name"] + (f"（{c['gap']:+.1f}%）" if c["gap"] is not None else "")) for c in checks if c["ok"] is not None]
    known = [c for c in checks if c["ok"] is not None]
    weak = [r for r, c in zip(reasons, known) if not c["ok"]]
    shown = weak if key == "defense" else reasons      # 防御優先は「弱い点」、他は良い点・悪い点の両方
    s = STANCE[key]
    return {"key": key, "label": s["label"], "emoji": s["emoji"], "cls": s["cls"], "message": s["message"],
            "guide": s["guide"], "checks": checks, "reasons": shown, "weak": weak,
            "growth_weak": g75 is False, "growth_available": g75 is not None,
            "note": None if g75 is not None else "東証グロース250のデータを取得できなかったため、TOPIXと日経平均で判定しています。"}


def build_market(bench: dict[str, pd.DataFrame], breadth_ma25: float | None,
                 n_high60: int, n_high252: int, cfg: Any = None) -> dict[str, Any]:
    regime = market_regime(bench["topix"]["close"])
    parts: list[str] = []
    n225_last = n225_chg = None
    n225 = bench.get("n225")
    if n225 is not None and len(n225["close"].dropna()) >= 2:
        s = n225["close"].dropna()
        n225_last, n225_chg = float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1)
        parts.append(f"日経平均は{n225_last:,.0f}円（前営業日比{n225_chg * 100:+.1f}%）。")
    parts.append({
        "good": "TOPIXは25日線・75日線の上で、上昇トレンドが続いています。",
        "neutral": "TOPIXは短期と長期の線が交錯しており、方向感が定まりにくい状況です。",
        "bad": "TOPIXは75日線を下回り、下落基調です。",
    }[regime])
    if breadth_ma25 is not None and not np.isnan(breadth_ma25):
        parts.append(f"売買代金の条件を満たす銘柄のうち{breadth_ma25 * 100:.0f}%が25日線の上にあり、"
                     f"60日高値を更新した銘柄は{n_high60:,}社、252日高値を更新した銘柄は{n_high252:,}社です。")
    parts.append({"good": "順張りには追い風です。", "neutral": "強い銘柄だけを厳選して見ましょう。",
                  "bad": "順張りは負けやすい局面です。新規の購入は少額・慎重に。"}[regime])
    stance = market_stance(bench, cfg or {"stance": {"ma_mid": 75, "ma_long": 200}})
    return {"regime": regime, "label": REGIME_LABEL[regime], "comment": "".join(parts),
            "n225": n225_last, "n225_chg": n225_chg, "stance": stance}
