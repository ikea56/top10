"""寄り付き前の市場環境（先物・為替・米国株）。相場判定と「ギャップダウン注意」にだけ使う。個別銘柄の順位・点数は変えない。

  日経225先物・TOPIX先物・ドル円 … 東証の前営業日の引け（15:30）から、いま（朝）までの変化率
  S&P500・NASDAQ・SOX         … 米国の直近の取引日の前日比

取得できない・値が古い指標は「判定に使わない」（取得できないことを画面に出す）。
先物は重み3、その他は1。加重平均の評価点（−1〜+1）が基準以下なら、相場判定を一段階だけ弱める。
先物が大幅に下落している（gap_down）なら「⚠ 本日はギャップダウンリスクあり」を出し、判定を弱める。
判定を強める（上げる）方向の補正はしない（朝の値動きだけで買い優勢に上げるのは危険なため）。
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Callable

import pandas as pd

from .config import JST, Config
from .log import get_logger

log = get_logger(__name__)
DownloadFn = Callable[[str, str, str], pd.Series]     # (ticker, period, interval) → 終値の時系列（時刻つき）
STATUS_LABEL = {"weak": "弱い", "neutral": "中立", "strong": "強い", "stale": "古い（判定に使わず）", "missing": "取得できず"}


# ---------------------------------------------------------------- 取得
def yahoo_download(ticker: str, period: str, interval: str) -> pd.Series:
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].dropna()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s.index = s.index.tz_convert(JST)
    return s


def change_since_tokyo_close(s: pd.Series, ref: datetime, now: datetime, max_age_h: float,
                             ref_tolerance_h: float = 3.0) -> dict[str, Any]:
    """前営業日の東証の引け（ref）の時点の値 → いま の変化率。値が古い・基準となる値が無いときは理由を返す。"""
    if s is None or s.empty:
        return {"status": "missing", "why": "データなし"}
    last_ts, last = s.index[-1], float(s.iloc[-1])
    if now - last_ts.to_pydatetime() > timedelta(hours=max_age_h):
        return {"status": "stale", "why": f"最後の値が{last_ts:%m/%d %H:%M}で古い", "asof": f"{last_ts:%m/%d %H:%M}"}
    before = s[s.index <= ref]
    if before.empty or ref - before.index[-1].to_pydatetime() > timedelta(hours=ref_tolerance_h):
        return {"status": "missing", "why": "東証の引け時点の値が無い"}
    base = float(before.iloc[-1])
    return {"status": "ok", "pct": (last / base - 1) * 100, "last": last, "asof": f"{last_ts:%m/%d %H:%M}"}


def change_daily(s: pd.Series, now: datetime, max_age_h: float) -> dict[str, Any]:
    """米国の指数など：直近の取引日の前日比。取引時間中の途中の値は使わない前提（朝の実行）。"""
    if s is None or len(s) < 2:
        return {"status": "missing", "why": "データなし"}
    last_ts = s.index[-1]
    if now - last_ts.to_pydatetime() > timedelta(hours=max_age_h + 72):      # 週末・休日をまたぐ余裕
        return {"status": "stale", "why": f"最後の値が{last_ts:%m/%d}で古い", "asof": f"{last_ts:%m/%d}"}
    return {"status": "ok", "pct": (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100, "last": float(s.iloc[-1]), "asof": f"{last_ts:%m/%d}"}


def fetch(cfg: Config, data_date: pd.Timestamp, now: datetime, download: DownloadFn | None = None) -> dict[str, Any]:
    """全指標を取得して評価まで行う。1つ失敗しても他は使う。"""
    pc = cfg["premarket"]
    dl = download or yahoo_download
    ref = datetime.combine(data_date.date(), time(15, 30), tzinfo=JST)
    readings: dict[str, dict[str, Any]] = {}
    for ind in pc["indicators"]:
        try:
            if ind["mode"] == "tokyo_close":
                s = dl(ind["ticker"], "7d", "15m")
                readings[ind["key"]] = change_since_tokyo_close(s, ref, now, float(pc["max_age_hours"]))
            else:
                s = dl(ind["ticker"], "10d", "1d")
                readings[ind["key"]] = change_daily(s, now, float(pc["max_age_hours"]))
        except Exception as e:  # noqa: BLE001
            log.warning("寄り付き前の指標 %s（%s）を取得できません: %s", ind["label"], ind["ticker"], e)
            readings[ind["key"]] = {"status": "missing", "why": "取得に失敗"}
    return evaluate(readings, cfg, now)


# ---------------------------------------------------------------- 評価（純粋関数）
def evaluate(readings: dict[str, dict[str, Any]], cfg: Config, now: datetime | None = None) -> dict[str, Any]:
    pc = cfg["premarket"]
    items, num, den = [], 0.0, 0.0
    gap_hits: list[str] = []
    for ind in pc["indicators"]:
        r = dict(readings.get(ind["key"]) or {"status": "missing", "why": "データなし"})
        status = r["status"]
        if status == "ok":
            pct = r["pct"]
            status = "weak" if pct <= ind["weak"] else "strong" if pct >= ind["strong"] else "neutral"
            w = float(ind["weight"])
            num += w * {"weak": -1.0, "neutral": 0.0, "strong": 1.0}[status]
            den += w
            if ind.get("futures") and pct <= float(pc["gap_down"]):
                gap_hits.append(f"{ind['label']} {pct:+.1f}%")
        items.append({"key": ind["key"], "label": ind["label"], "pct": r.get("pct"), "asof": r.get("asof"), "futures": bool(ind.get("futures")),
                      "status": status, "status_label": STATUS_LABEL[status], "why": r.get("why"),
                      "note": ind.get("note")})
    score = num / den if den else None
    futures_ok = [i for i in items if i["futures"] and i["pct"] is not None]
    return {"items": items, "score": score, "gap_down": bool(gap_hits), "gap_detail": gap_hits,
            "available": sum(1 for i in items if i["pct"] is not None), "total": len(items),
            "futures_available": len(futures_ok), "fetched_at": f"{now:%H:%M}" if now else None,
            "weak_hit": score is not None and score <= float(pc["downgrade_score"])}


def apply_to_stance(stance: dict[str, Any], pm: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """寄り付き前の指標を相場判定に反映する。弱める方向だけ：
    ギャップダウンリスク … 買い優勢→中立、中立→防御優先 ／ 全体が弱い … 買い優勢→中立"""
    from .market import STANCE

    out = dict(stance, history_key=stance["key"], premarket=pm, pm_note=None)
    key = stance["key"]
    new = key
    if pm["gap_down"]:
        new = {"buy": "neutral", "neutral": "defense"}.get(key, key)
    elif pm["weak_hit"]:
        new = {"buy": "neutral"}.get(key, key)
    reasons = list(stance["reasons"])
    if pm["gap_down"]:
        reasons.append("寄り付き前：" + "、".join(pm["gap_detail"]) + "（先物が大幅下落）")
    elif pm["weak_hit"]:
        weak = [f"{i['label']} {i['pct']:+.1f}%" for i in pm["items"] if i["status"] == "weak"]
        reasons.append("寄り付き前：" + "、".join(weak) + "が弱い")
    if new != key:
        s = STANCE[new]
        out.update(key=new, label=s["label"], emoji=s["emoji"], cls=s["cls"], message=s["message"], guide=list(s["guide"]))
        out["pm_note"] = f"過去の株価だけなら「{STANCE[key]['label']}」ですが、寄り付き前の先物・海外市場が弱いため一段階弱めて表示しています。"
    if pm["gap_down"]:
        out["guide"] = list(out["guide"]) + ["寄り付き直後は値動きが荒れやすいため、焦って新規で買わない"]
    out["reasons"] = reasons
    if pm["futures_available"] == 0:
        out["pm_warn"] = "日経225先物・TOPIX先物を取得できなかったため、先物による判定は行っていません。"
    return out


def demo_readings(kind: str = "gapdown") -> dict[str, dict[str, Any]]:
    """デモ用の架空の値（画面の確認用）。"""
    vals = {"gapdown": (-1.9, -1.7, -0.8, -1.4, -2.2, -3.6), "calm": (0.2, 0.1, 0.1, 0.3, 0.5, 0.8)}[kind]
    keys = ("n225f", "topixf", "usdjpy", "sp500", "nasdaq", "sox")
    return {k: {"status": "ok", "pct": v, "asof": "デモ"} for k, v in zip(keys, vals)}
