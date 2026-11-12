"""業種別リード指標（外部環境）。

銘柄 → 業種（またはコード）でグループを決める → グループの参照先（米国株・指数・為替・金利・商品）の直近の動きを見る
→ 業種リードスコア（−5〜+5）。参照先は config.yml の lead.groups で管理する（追加・変更は設定ファイルだけで済む）。

総合点への影響は補助的：本質的な評価（押し目・業績・還元）が主役で、この点は rank_effect 倍（既定0.5＝最大±2.5点）だけ足す。
取得できない参照先は使わず、使えた重みが少ないほどスコアを小さくする（情報が少ないときは影響も小さく）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .config import Config
from .log import get_logger
from .premarket import DownloadFn, change_daily, change_since_tokyo_close, yahoo_download

log = get_logger(__name__)


def match_group(code: str, sector: str, cfg: Config) -> dict | None:
    """上から順に、コード指定または業種が合致した最初のグループ。"""
    for g in cfg["lead"]["groups"]:
        if code in [str(c) for c in g.get("codes", [])] or sector in g.get("sectors", []):
            return g
    return None


def needed_refs(cands: list[dict], cfg: Config) -> dict[tuple[str, str], dict]:
    """対象銘柄が使うグループの参照先だけを重複なしで集める（(ticker, mode) → 設定）。"""
    out: dict[tuple[str, str], dict] = {}
    for c in cands:
        g = match_group(c["row"]["code"], str(c["row"]["sector"]), cfg)
        for r in (g or {}).get("refs", []):
            out.setdefault((r["ticker"], r.get("mode", "daily")), r)
    return out


def _read(s: pd.Series, ref: dict, ref_time: datetime, now: datetime, max_age_days: float) -> dict[str, Any]:
    if ref.get("mode", "daily") == "tokyo_close":
        r = change_since_tokyo_close(s, ref_time, now, max_age_days * 24)
        return r
    r = change_daily(s, now, max_age_days * 24 - 72)
    if r["status"] == "ok" and ref.get("kind") == "diff":
        r["diff"] = float(s.iloc[-1]) - float(s.iloc[-2])
    return r


def fetch_readings(refs: dict[tuple[str, str], dict], cfg: Config, data_date: pd.Timestamp, now: datetime,
                   download: DownloadFn | None = None) -> dict[tuple[str, str], dict]:
    from datetime import time

    from .config import JST

    dl = download or yahoo_download
    ref_time = datetime.combine(data_date.date(), time(15, 30), tzinfo=JST)
    age = float(cfg["lead"]["max_age_days"])
    out: dict[tuple[str, str], dict] = {}
    for key, ref in refs.items():
        try:
            intraday = ref.get("mode", "daily") == "tokyo_close"
            s = dl(ref["ticker"], "7d" if intraday else "10d", "15m" if intraday else "1d")
            out[key] = _read(s, ref, ref_time, now, age)
        except Exception as e:  # noqa: BLE001 1つ失敗しても他は使う
            log.warning("外部指標 %s（%s）を取得できません: %s", ref["label"], ref["ticker"], e)
            out[key] = {"status": "missing", "why": "取得に失敗"}
    return out


def evaluate(group: dict, readings: dict[tuple[str, str], dict], cfg: Config) -> dict[str, Any]:
    """グループの業種リードスコアと、表示用の項目（✅追い風 / ⚠逆風 / ➖横ばい）。"""
    lc = cfg["lead"]
    band = float(lc["flat_band"])
    items: list[dict] = []
    num = den = total_w = 0.0
    for ref in group["refs"]:
        w = float(ref.get("weight", 1))
        total_w += w
        r = readings.get((ref["ticker"], ref.get("mode", "daily")), {"status": "missing"})
        if r.get("status") != "ok":
            items.append({"label": ref["label"], "mark": "?", "state": "none", "text": "取得できず"})
            continue
        kind = ref.get("kind", "pct")
        chg = r["diff"] if kind == "diff" else r["pct"]
        sig = max(-1.0, min(1.0, chg / float(ref["full"])))
        if ref.get("invert"):
            sig = -sig
        num += w * sig
        den += w
        up = chg > 0
        move = ("上昇" if up else "下落") if abs(sig) >= band else "横ばい"
        shown = f"{chg * 100:+.0f}bp" if kind == "diff" else f"{chg:+.1f}%"
        state = "good" if sig >= band else "bad" if sig <= -band else "flat"
        items.append({"label": ref["label"], "mark": {"good": "✅", "bad": "⚠", "flat": "➖"}[state], "state": state,
                      "text": f"{ref['label']}{move}（{shown}）"})
    score = 0.0
    if den:
        score = float(lc["max_points"]) * (num / den) * min(1.0, den / total_w)      # 使えた重みが少ないほど小さく
    vp = float(lc["verdict_points"])
    v = "good" if score >= vp else "bad" if score <= -vp else "flat"
    n_ok = sum(1 for i in items if i["state"] != "none")
    head = {"good": f"{group['label']}に追い風", "bad": f"{group['label']}に逆風", "flat": f"{group['label']}への影響は小さめ"}[v]
    return {"key": group["key"], "group": group["label"], "score": round(score, 1), "items": items,
            "verdict": v if n_ok else "none", "headline": head if n_ok else f"{group['label']}の外部データを取得できませんでした",
            "icon": {"good": "🟢", "bad": "🔴", "flat": "🟡"}[v] if n_ok else "⚪",
            "available": sum(1 for i in items if i["state"] != "none"), "total": len(items)}


def apply(cands: list[dict], readings: dict[tuple[str, str], dict], cfg: Config) -> None:
    """各候補に c["lead"]（表示用）と c["lead_pts"]（総合点に足す点）をつける。該当するグループが無ければ None / 0。"""
    eff = float(cfg["lead"]["rank_effect"])
    for c in cands:
        g = match_group(c["row"]["code"], str(c["row"]["sector"]), cfg)
        if g is None:
            c["lead"], c["lead_pts"] = None, 0.0
            continue
        ev = evaluate(g, readings, cfg)
        c["lead"] = ev
        c["lead_pts"] = ev["score"] * eff if ev["available"] else 0.0


def check_config(cfg: Config, sectors: set[str] | None = None) -> list[str]:
    """設定の誤り（キー重複・必須項目の抜け・存在しない業種名）を返す。"""
    problems: list[str] = []
    lc = cfg["lead"]
    keys = [g["key"] for g in lc["groups"]]
    problems += [f"グループのキー {k} が重複しています" for k in set(keys) if keys.count(k) > 1]
    for g in lc["groups"]:
        if not g.get("refs"):
            problems.append(f"{g['key']}: 参照先(refs)がありません")
        if not g.get("sectors") and not g.get("codes"):
            problems.append(f"{g['key']}: 業種(sectors)もコード(codes)も指定されていません")
        for r in g.get("refs", []):
            if not {"ticker", "label", "full"} <= set(r):
                problems.append(f"{g['key']}: 参照先に ticker / label / full が必要です（{r}）")
        if sectors:
            problems += [f"{g['key']}: 業種名「{s}」が銘柄一覧にありません" for s in g.get("sectors", []) if s not in sectors]
    return problems


def demo_download(ticker: str, period: str, interval: str) -> pd.Series:
    """デモ用の架空の値（ティッカーごとに決まった変化率）。"""
    import hashlib

    from .config import JST

    pct = (int(hashlib.md5(ticker.encode()).hexdigest()[:6], 16) % 800 - 400) / 100          # −4.0〜+4.0%
    base = 4.3 if ticker == "^TNX" else 100.0
    now = datetime.now(JST)
    if interval == "1d":
        idx = pd.DatetimeIndex([now - pd.Timedelta(days=2), now - pd.Timedelta(hours=3)])
        return pd.Series([base, base * (1 + pct / 100)], index=idx)
    idx = pd.date_range(now - pd.Timedelta(days=5), now - pd.Timedelta(minutes=20), freq="15min", tz=JST)
    s = pd.Series(base, index=idx)
    s[s.index > now - pd.Timedelta(hours=12)] = base * (1 + pct / 100)
    return s


def _key(c: dict) -> tuple:
    return (c["main"], c["pb_total"], c["legacy_expected"])


def guard_order(base: list[dict], max_shift: int) -> list[dict]:
    """本質的な評価だけで並べた順（base）から、業種リードで動く順位を上下 max_shift までに抑えて並べ直す。"""
    remaining = list(enumerate(base))
    out: list[dict] = []
    for pos in range(len(base)):
        eligible = [x for x in remaining if x[0] <= pos + max_shift]                  # 上がりすぎない
        forced = [x for x in eligible if x[0] + max_shift <= pos]                       # 下がりすぎる前に置く
        pick = max(forced or eligible, key=lambda x: _key(x[1]))
        remaining.remove(pick)
        out.append(pick[1])
    return out


def finalize(kept: list[dict], cfg: Config) -> list[dict]:
    """総合点＝本質的な評価＋業種リード（少し）。順位の動きは max_rank_shift までで、総合点は順位と矛盾しないようにそろえる。"""
    base = sorted(kept, key=lambda c: (c["main_base"], c["pb_total"], c["legacy_expected"]), reverse=True)
    for c in base:
        c["main"] = c["main_base"] + c["lead_pts"]
    order = guard_order(base, int(cfg["lead"]["max_rank_shift"]))
    for i in range(1, len(order)):                      # 順位が下の銘柄が、上の銘柄より高い総合点にならないようにする（下げる方向のみ）
        if order[i]["main"] > order[i - 1]["main"]:
            order[i]["main"] = order[i - 1]["main"]
    for c in order:
        c["lead_pts"] = c["main"] - c["main_base"]
    return order
