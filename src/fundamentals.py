"""業績・配当・自社株買いのデータ取得（yfinance）。テクニカル上位の銘柄だけ取得して負荷を減らす。

取得できない項目は None にして、その項目の評価を単にスキップする（銘柄を落とさない）。
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd

from .config import ROOT, Config
from .log import get_logger

log = get_logger(__name__)
CACHE = ROOT / "data" / "fundamentals_cache.json"
SCHEMA_VERSION = 4          # 項目を増やしたら上げる（古いキャッシュを使わないため）
Fund = dict[str, Any]


def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _yoy(cur: float | None, prev: float | None) -> tuple[float | None, bool]:
    """(前年比, 黒字転換か)。前年が赤字・ゼロの場合は前年比を出さない。"""
    if prev is not None and prev > 0:
        return (cur - prev) / prev, False
    if cur is not None and cur > 0 and (prev is None or prev <= 0):
        return None, prev is not None
    return None, False


def _series(stmt: pd.DataFrame | None, labels: tuple[str, ...]) -> pd.Series | None:
    """財務表から指定行を取り出し、新しい期が先頭になるよう並べる。"""
    if stmt is None or getattr(stmt, "empty", True):
        return None
    for label in labels:
        if label in stmt.index:
            s = stmt.loc[label].dropna().sort_index(ascending=False)
            return s if len(s) else None
    return None


def _safe(fn: Callable[[], Any], what: str, code: str) -> Any:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001  取得できない項目があっても全体は続ける
        log.debug("%s: %s の取得に失敗（%s）", code, what, e)
        return None


def dividend_stats(div: pd.Series | None, forward: float | None, today: pd.Timestamp) -> Fund:
    """配当履歴から 増配率・連続増配年数 を計算する。
    - 予想配当が取れる場合：今期予想配当 vs 直近12か月の実績
    - 取れない場合：直近12か月 vs その前の12か月"""
    out: Fund = {"div_cur": None, "div_prev": None, "div_basis": None, "div_streak": 0,
                 "div_ttm": None, "div_prev12": None}
    if div is None or len(div) == 0:
        return out
    idx = pd.to_datetime(div.index, utc=True).tz_localize(None)
    d = pd.Series(div.values.astype(float), index=idx)
    ttm = float(d[d.index > today - pd.Timedelta(days=365)].sum())
    prev12 = float(d[(d.index > today - pd.Timedelta(days=730)) & (d.index <= today - pd.Timedelta(days=365))].sum())
    out.update(div_ttm=ttm, div_prev12=prev12)
    if forward and forward > 0 and ttm > 0:
        out.update(div_cur=float(forward), div_prev=ttm, div_basis="forecast")
    else:
        out.update(div_cur=ttm, div_prev=prev12, div_basis="actual")
    by_year = d.groupby(d.index.year).sum()
    y, streak = today.year - 1, 0                       # 当年は途中なので、前年から数える
    while y in by_year.index and (y - 1) in by_year.index and by_year[y] > by_year[y - 1] > 0:
        streak, y = streak + 1, y - 1
    out["div_streak"] = streak
    return out


def fetch_one(code: str) -> Fund:
    import yfinance as yf

    out: Fund = {"v": SCHEMA_VERSION, "status": "missing", "market_cap": None, "op_ttm": None, "op_yoy": None,
                 "op_basis": None, "op_annual": [], "eps_hist": [], "eps_est_0y": None, "eps_est_1y": None,
                 "eps_rev": None, "div_cur": None, "div_prev": None, "div_basis": None, "div_streak": 0,
                 "payout_latest": None, "payout_prev": None, "buyback": None, "next_earnings": None}
    t = yf.Ticker(f"{code}.T")
    out["market_cap"] = _f(_safe(lambda: t.fast_info["market_cap"], "時価総額", code)
                           or _safe(lambda: t.info.get("marketCap"), "時価総額(info)", code))

    q = _series(_safe(lambda: t.quarterly_income_stmt, "四半期損益", code), ("Operating Income", "OperatingIncome"))
    a_stmt = _safe(lambda: t.income_stmt, "通期損益", code)
    a = _series(a_stmt, ("Operating Income", "OperatingIncome"))
    if a is not None:
        out["op_annual"] = [float(x) for x in a.iloc[:4]]
    if q is not None and len(q) >= 4:
        out["op_ttm"] = float(q.iloc[:4].sum())
    elif a is not None:
        out["op_ttm"] = float(a.iloc[0])

    if q is not None and len(q) >= 5 and 330 <= (q.index[0] - q.index[4]).days <= 400:
        out["op_yoy"], _ = _yoy(float(q.iloc[0]), float(q.iloc[4]))
        out["op_basis"] = "直近四半期・前年同期比"
    if out["op_basis"] is None and a is not None and len(a) >= 2:
        out["op_yoy"], _ = _yoy(float(a.iloc[0]), float(a.iloc[1]))
        out["op_basis"] = "通期・前年比"

    eps = _series(a_stmt, ("Diluted EPS", "Basic EPS"))
    if eps is not None:
        out["eps_hist"] = [float(x) for x in eps.iloc[:4]]

    est = _safe(lambda: t.earnings_estimate, "EPS予想", code)
    if est is not None and not getattr(est, "empty", True) and "avg" in est.columns:
        out["eps_est_0y"] = _f(est["avg"].get("0y"))
        out["eps_est_1y"] = _f(est["avg"].get("+1y"))
    trend = _safe(lambda: t.eps_trend, "EPS予想の推移", code)
    if trend is not None and not getattr(trend, "empty", True) and "0y" in trend.index:
        row = trend.loc["0y"]
        out["eps_rev"] = {"cur": _f(row.get("current")), "d30": _f(row.get("30daysAgo")), "d90": _f(row.get("90daysAgo"))}

    fwd = _f(_safe(lambda: t.info.get("dividendRate"), "予想配当", code))
    out.update(dividend_stats(_safe(lambda: t.dividends, "配当履歴", code), fwd, pd.Timestamp(date.today())))
    if len(out["eps_hist"]) >= 2:                               # 配当性向（配当÷EPS）：取得できる場合のみ
        e0, e1 = out["eps_hist"][0], out["eps_hist"][1]
        if e0 > 0 and out.get("div_ttm"):
            out["payout_latest"] = out["div_ttm"] / e0
        if e1 > 0 and out.get("div_prev12"):
            out["payout_prev"] = out["div_prev12"] / e1

    cal = _safe(lambda: t.calendar, "決算予定日", code)
    dates = (cal.get("Earnings Date") if isinstance(cal, dict) else None) or []
    future = sorted(str(d)[:10] for d in dates if str(d)[:10] >= date.today().isoformat())
    out["next_earnings"] = future[0] if future else None

    cf = _safe(lambda: t.cashflow, "キャッシュフロー", code)
    rep = _series(cf, ("Repurchase Of Capital Stock", "Common Stock Payments", "RepurchaseOfCapitalStock"))
    if rep is not None and float(rep.iloc[0]) < 0:
        out["buyback"] = abs(float(rep.iloc[0]))

    if out["op_ttm"] is not None or out["market_cap"] is not None:
        out["status"] = "ok"
    return out


def exclusion_reason(fund: Fund, fc: dict) -> str | None:
    """赤字・継続赤字の除外判定。除外するなら理由（日本語）、残すなら None。"""
    ttm, annual = fund.get("op_ttm"), fund.get("op_annual") or []
    if ttm is None:
        return "業績データなし" if fc["exclude_if_missing"] else None
    n = int(fc["chronic_loss_years"])
    chronic = len(annual) >= n and all(x <= 0 for x in annual[:n]) and ttm <= 0
    if chronic and fc["exclude_chronic_loss"]:
        return "継続赤字"
    forecast_profit = fc["allow_forecast_turnaround"] and (fund.get("eps_est_0y") or 0) > 0
    if ttm <= 0 and fc["exclude_loss_making"] and not forecast_profit:
        return "赤字"
    return None


class FundamentalsStore:
    """取得済みデータを data/fundamentals_cache.json に保存して、同じ銘柄を何度も取りに行かない。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg["fundamentals"]
        self.cache: dict[str, dict[str, Any]] = {}
        if CACHE.exists():
            try:
                self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                log.warning("業績キャッシュを読めないため作り直します: %s", e)

    def get(self, code: str) -> Fund:
        today = date.today()
        hit = self.cache.get(code)
        if hit and hit["data"].get("v") == SCHEMA_VERSION:
            if (today - date.fromisoformat(hit["fetched"])).days <= self.cfg["cache_days"]:
                return hit["data"]
        try:
            data = fetch_one(code)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: 業績取得に失敗: %s", code, e)
            data = {"v": SCHEMA_VERSION, "status": "missing"}
        time.sleep(self.cfg["sleep"])
        if data["status"] == "ok":                      # 取得失敗はキャッシュしない
            self.cache[code] = {"fetched": today.isoformat(), "data": data}
        return data

    def save(self) -> None:
        limit = date.today() - timedelta(days=30)
        self.cache = {k: v for k, v in self.cache.items() if date.fromisoformat(v["fetched"]) >= limit}
        try:
            CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("業績キャッシュを保存できませんでした: %s", e)
