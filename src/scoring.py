"""最終スコア（テクニカル + 業績 + 時価総額）と強気判定。"""
from .features import tech_max


def earn_points(f, sc):
    if f is None:
        return 0
    if f.get("turnaround"):
        return sc["turnaround_points"]
    yoy = f.get("op_yoy")
    if yoy is None:
        return 0
    for th, p in sorted(sc["op_growth"], key=lambda x: -x[0]):
        if yoy >= th:
            return p
    return 0


def mcap_points(f, sc):
    mc = (f or {}).get("market_cap")
    if not mc:
        return 0
    for th, p in sorted(sc["mcap"], key=lambda x: -x[0]):
        if mc >= th:
            return p
    return 0


def max_earn_mcap(sc):
    e = max(max(p for _, p in sc["op_growth"]), sc["turnaround_points"])
    return e + max(p for _, p in sc["mcap"])


def max_raw(sc):
    return tech_max(sc) + max_earn_mcap(sc)


def to_score(raw, sc):
    return round(raw / max_raw(sc) * sc["normalize_to"], 1)


def bull_label(score, overheat, regime, cfg):
    """(ラベル, CSSクラス, 補足)"""
    b = cfg["bull"]
    note = ""
    if score >= b["strong"]:
        label = "強気"
    elif score >= b["mild"]:
        label = "やや強気"
    else:
        label = "中立"
    if label == "強気" and overheat:
        label, note = "やや強気", "過熱気味のため格下げ"
    elif label == "強気" and regime == "bad":
        label, note = "やや強気", "市場全体が弱いため格下げ"
    css = {"強気": "bull", "やや強気": "mild", "中立": "neutral"}[label]
    return label, css, note
