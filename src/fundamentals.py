"""業績データ（営業利益・時価総額）。テクニカル上位の銘柄だけ取得して負荷を減らす。"""
import json
import time
from datetime import date, timedelta

from .config import ROOT

CACHE = ROOT / "data" / "fundamentals_cache.json"


def _yoy(cur, prev):
    """(前年比, 黒字転換か)"""
    if prev is not None and prev > 0:
        return (cur - prev) / prev, False
    if cur is not None and cur > 0 and (prev is None or prev <= 0):
        return None, prev is not None   # 前年が赤字（またはゼロ）で今期黒字 → 黒字転換
    return None, False


def _op_series(stmt):
    if stmt is None or getattr(stmt, "empty", True):
        return None
    for label in ("Operating Income", "OperatingIncome"):
        if label in stmt.index:
            s = stmt.loc[label].dropna().sort_index(ascending=False)
            return s if len(s) else None
    return None


def fetch_one(code):
    import yfinance as yf

    out = {"status": "missing", "market_cap": None, "op_ttm": None,
           "op_yoy": None, "op_basis": None, "turnaround": False}
    t = yf.Ticker(f"{code}.T")
    try:
        out["market_cap"] = float(t.fast_info["market_cap"])
    except Exception:  # noqa: BLE001
        try:
            out["market_cap"] = float(t.info.get("marketCap"))
        except Exception:  # noqa: BLE001
            pass

    try:
        q = _op_series(t.quarterly_income_stmt)
    except Exception:  # noqa: BLE001
        q = None
    try:
        a = _op_series(t.income_stmt)
    except Exception:  # noqa: BLE001
        a = None

    if q is not None and len(q) >= 4:
        out["op_ttm"] = float(q.iloc[:4].sum())
    elif a is not None:
        out["op_ttm"] = float(a.iloc[0])

    # 前年比：直近四半期の前年同期比を優先、無ければ通期
    if q is not None and len(q) >= 5:
        gap = (q.index[0] - q.index[4]).days
        if 330 <= gap <= 400:
            out["op_yoy"], out["turnaround"] = _yoy(float(q.iloc[0]), float(q.iloc[4]))
            out["op_basis"] = "直近四半期・前年同期比"
    if out["op_basis"] is None and a is not None and len(a) >= 2:
        out["op_yoy"], out["turnaround"] = _yoy(float(a.iloc[0]), float(a.iloc[1]))
        out["op_basis"] = "通期・前年比"

    if out["op_ttm"] is not None or out["market_cap"] is not None:
        out["status"] = "ok"
    return out


class FundamentalsStore:
    def __init__(self, cfg):
        self.cfg = cfg["fundamentals"]
        self.cache = {}
        if CACHE.exists():
            try:
                self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.cache = {}

    def get(self, code):
        today = date.today()
        hit = self.cache.get(code)
        if hit:
            age = (today - date.fromisoformat(hit["fetched"])).days
            if age <= self.cfg["cache_days"]:
                return hit["data"]
        data = fetch_one(code)
        time.sleep(self.cfg["sleep"])
        if data["status"] == "ok":          # 取得失敗はキャッシュしない
            self.cache[code] = {"fetched": today.isoformat(), "data": data}
        return data

    def save(self):
        limit = date.today() - timedelta(days=30)
        self.cache = {k: v for k, v in self.cache.items()
                      if date.fromisoformat(v["fetched"]) >= limit}
        CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
