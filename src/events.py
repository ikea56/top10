"""評価イベント（業績イベント＋株主還元＋同時発生ボーナス）。

「機関投資家が評価しやすいイベント」を、事実（決算・配当・予想の変更）として検出して加点する。
株価の上昇を予測する処理は含まない。『予想』と付く項目は会社・アナリストの公表予想で、当サイトの予測ではない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .scoring import tier_points


@dataclass
class EventItem:
    key: str
    label: str
    points: float = 0.0
    kind: str = "earnings"       # momentum / earnings / shareholder / bonus
    basis: str = ""              # 予想 / 実績 / ""


@dataclass
class EventResult:
    items: list[EventItem] = field(default_factory=list)
    earnings: float = 0.0
    shareholder: float = 0.0
    bonus: float = 0.0
    total: float = 0.0
    uses_forecast: bool = False

    def has(self, key: str) -> bool:
        return any(i.key == key for i in self.items)


def _pos(x: Any) -> bool:
    return x is not None and x == x and x > 0


# ---------------------------------------------------------------- 業績イベント
def _turnaround(f: dict, ev: dict) -> EventItem | None:
    annual = f.get("op_annual") or []
    prior_loss = bool(annual) and annual[0] <= 0
    if prior_loss and _pos(f.get("eps_est_0y")):
        return EventItem("turnaround", "今期黒字転換（前期赤字→今期黒字予想）", ev["turnaround"], "earnings", "予想")
    if prior_loss and _pos(f.get("op_ttm")):
        return EventItem("turnaround", "黒字転換（前期赤字→直近12か月は黒字）", ev["turnaround_actual"], "earnings", "実績")
    return None


def _sustained(f: dict, ev: dict) -> EventItem | None:
    annual = f.get("op_annual") or []
    if annual and annual[0] <= 0 and _pos(f.get("eps_est_0y")) and _pos(f.get("eps_est_1y")):
        return EventItem("sustained", "継続黒字化（来期も黒字予想）", ev["sustained_profit"], "earnings", "予想")
    if len(annual) >= 2 and annual[1] <= 0 < annual[0] and _pos(f.get("op_ttm")):
        return EventItem("sustained", "黒字化が2期連続で定着", ev["sustained_profit"], "earnings", "実績")
    return None


def _record_high(f: dict, ev: dict) -> EventItem | None:
    hist = [x for x in (f.get("eps_hist") or []) if x is not None]
    est0, annual = f.get("eps_est_0y"), f.get("op_annual") or []
    if _pos(est0) and len(hist) >= 3 and est0 > max(hist):
        return EventItem("record_high", "最高益更新（今期予想EPSが過去最高）", ev["record_high"], "earnings", "予想")
    ttm = f.get("op_ttm")
    if _pos(ttm) and len(annual) >= 3 and ttm > max(annual):
        return EventItem("record_high", "最高益更新（直近12か月の営業利益が過去最高）", ev["record_high"], "earnings", "実績")
    return None


def _revision(f: dict, ev: dict) -> EventItem | None:
    r = f.get("eps_rev") or {}
    cur, d30 = r.get("cur"), r.get("d30")
    if cur is None or d30 is None or d30 == 0:
        return None
    chg = (cur - d30) / abs(d30)
    if chg >= ev["revision_min_change"]:
        return EventItem("revision", f"上方修正（予想EPSが30日前より{chg * 100:+.0f}%）", ev["upward_revision"], "earnings", "予想")
    return None


# ---------------------------------------------------------------- 株主還元
def _dividend_growth(f: dict, ev: dict) -> EventItem | None:
    cur, prev = f.get("div_cur"), f.get("div_prev")
    if not (_pos(cur) and _pos(prev)) or cur <= prev * 1.0001:
        return None
    g = round(cur / prev - 1, 6)          # 20%ちょうどが19.999…%にならないよう丸める
    pts = max(float(ev["dividend_increase"]), tier_points(g, ev["increase_tiers"]))
    big = g >= min(t[0] for t in ev["increase_tiers"])
    return EventItem("dividend", f"{'大幅' if big else ''}増配（{g * 100:+.0f}%）", pts, "shareholder",
                     "予想" if f.get("div_basis") == "forecast" else "実績")


def _streak(f: dict, ev: dict) -> EventItem | None:
    n = int(f.get("div_streak") or 0)
    pts = tier_points(n, ev["streak_tiers"])
    return EventItem("streak", f"{n}年連続増配", pts, "shareholder", "実績") if pts > 0 else None


def _payout(f: dict, ev: dict) -> EventItem | None:
    now, prev = f.get("payout_latest"), f.get("payout_prev")
    if _pos(now) and _pos(prev) and now > prev and now <= ev["payout_max"]:
        return EventItem("payout", f"配当性向が上昇し株主還元を強化（{prev * 100:.0f}%→{now * 100:.0f}%）",
                         ev["payout_improve"], "shareholder", "実績")
    return None


def _buyback(f: dict, ev: dict) -> EventItem | None:
    b, mc = f.get("buyback"), f.get("market_cap")
    if _pos(b) and _pos(mc) and b / mc >= ev["buyback_min_ratio"]:
        return EventItem("buyback", f"自社株買いを実施（時価総額の{b / mc * 100:.1f}%）", ev["buyback"], "shareholder", "実績")
    return None


# ---------------------------------------------------------------- まとめ
def evaluate_events(row: Any, fund: dict | None, cfg: Config) -> EventResult:
    """row は断面の1行（hi252 / near52w / vol_ratio を使う）。fund が None でも動く。"""
    ev = cfg["events"]
    res = EventResult()

    # 画面表示用（点数は基本点に反映済みなので0点）
    if bool(row["hi252"]):
        res.items.append(EventItem("hi252", "252日高値更新（1年ぶりの高値）", 0, "momentum"))
    elif bool(row["near52w"]):
        res.items.append(EventItem("near52w", "52週高値付近を推移", 0, "momentum"))
    vr = row["vol_ratio"]
    if vr == vr and vr >= 2:
        res.items.append(EventItem("volume", f"出来高が通常の{vr:.1f}倍に急増", 0, "momentum"))

    if fund:
        for fn, group in ((_turnaround, "earnings"), (_sustained, "earnings"), (_record_high, "earnings"),
                          (_revision, "earnings"), (_dividend_growth, "shareholder"), (_streak, "shareholder"),
                          (_payout, "shareholder"), (_buyback, "shareholder")):
            sub = ev["earnings"] if group == "earnings" else ev["shareholder"]
            item = fn(fund, sub)
            if item:
                res.items.append(item)
    res.earnings = sum(i.points for i in res.items if i.kind == "earnings")
    res.shareholder = sum(i.points for i in res.items if i.kind == "shareholder")
    res.uses_forecast = any(i.basis == "予想" for i in res.items)

    # 252日高値更新 ＋ {増配, 黒字転換, 最高益} の同時発生 → 強いモメンタムシグナル
    signals = [name for key, name in (("dividend", "増配"), ("turnaround", "黒字転換"), ("record_high", "最高益"))
               if res.has(key)]
    if res.has("hi252") and signals:
        n = 1 + len(signals)
        pts = float(ev["momentum_bonus"].get(n, 0))
        if pts > 0:
            res.bonus = pts
            res.items.append(EventItem("bonus", f"同時発生ボーナス（252日高値更新＋{'＋'.join(signals)}）", pts, "bonus"))
    res.total = min(res.earnings + res.shareholder + res.bonus, float(ev["cap"]))
    return res
