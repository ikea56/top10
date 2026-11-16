"""決算イベント評価（㉔）。決算ギャンブルを避け、決算を通過した後の押し目を優先する。

  決算接近ペナルティ … 決算までの営業日数に応じて総合点から減点（押し目スコアより優先される大きさ）
  決算通過後ボーナス … 決算通過後30営業日以内なら基本加点、さらに増配・上方修正・自社株買い・黒字転換・最高益で加点
  バッジ … ✅ 決算通過済み / ⚠ 決算接近（5営業日以内）/ ⚠ 決算跨ぎ注意（3営業日以内）

注意：決算発表は引け後が多いため、発表の翌営業日の朝はまだ株価の反応が出ていない。
      発表から2営業日以上たってから「通過済み」として扱う（それまでは「反応待ち」）。
      決算日は取得できない銘柄があり、その場合は前回の決算日＋約91日の『推定』で代用し、推定と明記する。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from . import jpholidays
from .config import Config


def _d(x: Any) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10]) if x else None
    except ValueError:
        return None


def busdays(a: date, b: date, cfg: Config) -> int:
    """a から b までの営業日数（土日・設定の祝日を除く）。a<=b なら0以上、a>bなら負。"""
    lo, hi = min(a, b), max(a, b)
    hol = [np.datetime64(h) for h in jpholidays.between(lo, hi)]
    hol += [np.datetime64(h) for h in cfg["earnings"].get("extra_holidays", [])]       # 臨時の休場日（通常は空）
    if a <= b:
        return int(np.busday_count(a, b, holidays=hol))
    return -int(np.busday_count(b, a, holidays=hol))


def _penalty(n: int, cfg: Config) -> float:
    for t in sorted(cfg["earnings"]["penalty"], key=lambda t: t["business_days"]):
        if n <= t["business_days"]:
            return float(t["pts"])
    return 0.0


def assess(fund: dict | None, today: date, cfg: Config, td_dates: list[str] | None = None) -> dict[str, Any]:
    """決算状況を判定する。pen＝減点、base＝通過後の基本加点（どちらか一方）。"""
    ec = cfg["earnings"]
    fund = fund or {}
    pasts = [d for d in (_d(x) for x in list(fund.get("earnings_past") or []) + list(td_dates or []) + [fund.get("next_earnings")]) if d and d <= today]
    last = max(pasts) if pasts else None
    nxt, est = _d(fund.get("next_earnings")), False
    if nxt and nxt < today:
        nxt = None
    if nxt is None and last:
        guess = last + timedelta(days=int(ec["estimate_days"]))
        while guess.weekday() >= 5:
            guess += timedelta(days=1)
        if guess >= today:
            nxt, est = guess, True
    days_to = busdays(today, nxt, cfg) if nxt else None
    since = busdays(last, today, cfg) if last else None
    out: dict[str, Any] = {"next": nxt.isoformat() if nxt else None, "estimate": est, "days_to": days_to,
                           "last": last.isoformat() if last else None, "days_since": since,
                           "state": "unknown", "pen": 0.0, "base": 0.0, "badges": [], "text": "決算日を取得できませんでした"}
    if days_to is not None and days_to <= ec["approach_days"]:        # 決算が近い（通過後の評価より優先）
        out.update(state="approach", pen=_penalty(days_to, cfg))
        out["badges"].append("⚠ 決算接近")
        if days_to <= ec["cross_days"]:
            out["badges"].append("⚠ 決算跨ぎ注意")
        out["text"] = f"決算まで{days_to}営業日" + ("（推定）" if est else "")
    elif since is not None and since <= 1:
        out.update(state="pending", text="決算直後（株価の反応はこれから。通過済みとしては扱いません）")
    elif since is not None and since <= ec["passed_days"]:
        out.update(state="passed", base=float(ec["passed_base"]))
        out["badges"].append("✅ 決算通過済み")
        out["text"] = f"決算通過後{since}営業日"
    elif days_to is not None:
        out.update(state="ample", pen=_penalty(days_to, cfg))
        out["text"] = f"決算まで{days_to}営業日" + ("（推定）" if est else "")
    elif since is not None:
        out["text"] = f"前回の決算から{since}営業日（次回の決算日は未取得）"
    return out


def event_bonus(earn: dict | None, td: Any, ev: Any, cfg: Config) -> tuple[float, list[str]]:
    """決算通過後の好材料ボーナス。通過済み（state=passed）のときだけ。TDnetは前回決算日以降の開示だけ数える。"""
    if not earn or earn.get("state") != "passed":
        return 0.0, []
    bc = cfg["earnings"]["bonus"]
    last = earn["last"]
    pts, labels = 0.0, []
    if td is not None and getattr(td, "status", "") == "ok":
        for kind, name in (("dividend_increase", "増配"), ("upward_revision", "上方修正"), ("buyback", "自社株買い")):
            if any(h.kind == kind and h.date >= last for h in td.hits):
                pts += float(bc[kind]); labels.append(name)
    for key, name in (("turnaround", "黒字転換"), ("record_high", "最高益更新")):
        if ev is not None and ev.has(key):
            pts += float(bc[key]); labels.append(name)
    return min(pts, float(bc["cap"])), labels


def cap_label(label: str, css: str, earn: dict | None, cfg: Config) -> tuple[str, str, str]:
    """決算直前は総合点が高くてもラベルを抑える。(ラベル, CSS, 補足)"""
    if not earn or earn.get("days_to") is None or earn["state"] != "approach":
        return label, css, ""
    ec = cfg["earnings"]
    n = earn["days_to"]
    order = ["押し目良好", "押し目候補", "様子見"]
    css_of = {"押し目良好": "bull", "押し目候補": "mild", "様子見": "neutral"}
    limit = "様子見" if n <= ec["watch_cap_days"] else "押し目候補" if n <= ec["fair_cap_days"] else None
    if limit and order.index(label) < order.index(limit):
        return limit, css_of[limit], "決算が近いため"
    return label, css, ""


def summary(earn: dict, bonus: float, labels: list[str]) -> dict[str, Any]:
    """画面・JSON用（保有銘柄の監視にも使う）。"""
    return {"text": earn["text"], "state": earn["state"], "badges": earn["badges"], "next": earn["next"], "estimate": earn["estimate"],
            "days_to": earn["days_to"], "last": earn["last"], "days_since": earn["days_since"],
            "pen": round(earn["pen"]), "base": round(earn["base"]), "bonus": round(bonus), "bonus_labels": labels}
