"""追跡・検証ページ用の集計。

トップ10に載った銘柄を、載った日から毎日追いかけ、「載ったあとどうなったか」を数字とグラフで確かめる。
・基準（買値）＝載った日の翌営業日の始値（朝にランキングを見て、寄り付きで買った想定）。騰落率＝その後の終値 ÷ 買値 − 1
・TOPIX連動ETF(1306)の同じ期間の騰落率と比べ、「市場より良かったか（超過）」も出す
・売買コスト・税金・配当調整の違いは含まない（株価は配当・分割で調整済みの値）
・同じ銘柄が一度圏外に出て、しばらくしてまたトップ10に入ったときは、新しい掲載として数える（reentry_gap）
・追跡は max_days 営業日まで。終わった銘柄は主要な日（1・3・5・10・20…営業日後）の結果だけを台帳に残し、ファイルが増え続けないようにする
台帳：data/tracking_ledger.json（毎日更新。リポジトリに保存されるので、検証が途切れない）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .log import get_logger

log = get_logger(__name__)
STANCE_LABEL = {"buy": "🟢買い優勢", "neutral": "🟡中立", "defense": "🔴防御優先", None: "記録なし"}


DEFAULTS: dict[str, Any] = {"ret": None, "bench": None, "excess": None, "max_ret": None, "min_ret": None, "spark": None,
                            "last_close": None, "buy": None, "close0": None}


# ---------------------------------------------------------------- 履歴の読み込み
def scan_history(hist_dir: Path, top_n: int, reentry_gap: int) -> tuple[list[dict], dict[str, dict[str, int]], str | None]:
    """履歴（data/history）から、掲載の一覧・日ごとの順位・最新の記録日を作る。"""
    recs = []
    for fp in sorted(hist_dir.glob("*.json")):
        try:
            recs.append(json.loads(fp.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            log.warning("履歴ファイルを読めません（%s）: %s", fp.name, e)
    entries: list[dict] = []
    last_idx: dict[str, int] = {}
    ranks: dict[str, dict[str, int]] = {}
    for i, rec in enumerate(recs):
        d = rec.get("date")
        if not d:
            continue
        rk = {r["code"]: r["rank"] for r in rec.get("ranking", [])}
        rk.update({p["code"]: p["rank"] for p in rec.get("picks", [])})
        ranks[d] = rk
        for p in rec.get("picks", []):
            if p["rank"] > top_n:
                continue
            code = p["code"]
            if code not in last_idx or i - last_idx[code] > reentry_gap:
                entries.append({"key": f"{code}@{d}", "code": code, "name": p["name"], "date": d, "rank": p["rank"],
                                "main": p.get("main"), "stance": rec.get("stance"), "regime": rec.get("regime")})
            last_idx[code] = i
    return entries, ranks, (recs[-1]["date"] if recs else None)


# ---------------------------------------------------------------- 1件ずつの計算
def _spark(vals: list[float], k: int = 30) -> list[float]:
    if len(vals) <= k:
        return [round(v * 100, 1) for v in vals]
    idx = np.linspace(0, len(vals) - 1, k).round().astype(int)
    return [round(vals[i] * 100, 1) for i in idx]


def compute_entry(e: dict, c: pd.DataFrame, o: pd.DataFrame, tb: pd.DataFrame, ranks: dict[str, dict[str, int]],
                  cfg: Config) -> dict[str, Any]:
    """1つの掲載について、日ごとの推移と要約を計算する（価格が取れなければ status="価格なし"）。"""
    tc = cfg["tracking"]
    tk = e["code"] + ".T"
    out: dict[str, Any] = {**e, "series": [], "cp": {}, "status": "価格なし", "days": 0, **DEFAULTS}
    d0 = pd.Timestamp(e["date"])
    if tk not in c.columns or d0 not in c.index:
        return out
    i0 = c.index.get_loc(d0)
    close, opn = c[tk], o[tk]
    out["close0"] = _f(close.iloc[i0])
    if i0 + 1 >= len(c):
        out["status"] = "買い値待ち"            # 翌営業日の始値がまだ無い
        out["series"] = [[0, e["date"], out["close0"], None, None, ranks.get(e["date"], {}).get(e["code"])]]
        return out
    buy = _f(opn.iloc[i0 + 1])
    if buy is None or buy <= 0:
        out["status"] = "価格なし"
        return out
    out["buy"] = buy
    bo = _f(tb["open"].iloc[i0 + 1]) if "open" in tb else None
    series = [[0, e["date"], out["close0"], None, None, ranks.get(e["date"], {}).get(e["code"])]]
    rets: list[float] = []
    for k in range(1, min(len(c) - 1 - i0, int(tc["max_days"])) + 1):
        j = i0 + k
        px = _f(close.iloc[j])
        if px is None:
            continue
        r = px / buy - 1
        bc = _f(tb["close"].iloc[j])
        b = (bc / bo - 1) if (bc is not None and bo) else None
        dt = c.index[j].strftime("%Y-%m-%d")
        series.append([k, dt, px, r, b, ranks.get(dt, {}).get(e["code"])])
        rets.append(r)
    if not rets:
        out["status"] = "価格なし"
        return out
    out["series"] = series
    out["status"] = "追跡中"
    out["days"] = series[-1][0]
    last = series[-1]
    out.update(last_close=last[2], ret=last[3], bench=last[4], excess=None if last[4] is None else last[3] - last[4],
               max_ret=max(rets), min_ret=min(rets), spark=_spark(rets))
    for h in tc["horizons"]:
        row = next((s for s in series if s[0] == h), None)
        if row:
            out["cp"][str(h)] = [row[3], row[4]]
    return out


def _f(v: Any) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------- 台帳との統合
def build(panel: dict[str, pd.DataFrame], bench_topix: pd.DataFrame, hist_dir: Path, ledger_path: Path, cfg: Config) -> dict[str, Any]:
    """履歴から掲載を洗い出し、価格から推移を計算し、台帳に保存する。"""
    tc = cfg["tracking"]
    entries, ranks, last_date = scan_history(hist_dir, int(cfg["top_n"]), int(tc["reentry_gap"]))
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    except (OSError, ValueError) as e:
        log.warning("追跡台帳を読めないため作り直します: %s", e)
        ledger = {}
    c, o = panel["close"], panel["open"]
    tb = bench_topix.reindex(c.index).ffill()
    results: list[dict] = []
    for e in entries:
        old = ledger.get(e["key"])
        if old and old.get("closed"):                       # 追跡が終わったものは、台帳の値をそのまま使う
            results.append({**e, **DEFAULTS, "cp": {}, "days": 0, "status": "追跡中", **old, "series": []})
            continue
        r = compute_entry(e, c, o, tb, ranks, cfg)
        if r["status"] == "価格なし" and old and old.get("ret") is not None:
            r = {**e, **DEFAULTS, "cp": {}, "days": 0, **old, "series": [], "status": "価格を取得できません（最後の記録を表示）", "stale": True}
        elif r["days"] >= int(tc["max_days"]):
            r["closed"] = True
        results.append(r)
    for r in results:                                         # 現在の状態（最新の記録日の順位）
        cur = ranks.get(last_date or "", {}).get(r["code"])
        r["cur_rank"] = cur
        if r["status"] in ("価格なし", "買い値待ち") or r.get("stale"):
            pass
        elif r.get("closed"):
            r["status"] = "追跡終了"
        else:
            r["status"] = "掲載中" if cur is not None and cur <= int(cfg["top_n"]) else "圏外へ"
    keep = ("name", "date", "rank", "main", "stance", "regime", "code", "buy", "close0", "last_close", "ret", "bench", "excess",
            "max_ret", "min_ret", "spark", "cp", "days", "closed", "stale")
    new_ledger = {r["key"]: {k: r[k] for k in keep if k in r} for r in results}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(new_ledger, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    results.sort(key=lambda r: (r["date"], -r["rank"]), reverse=True)
    return {"entries": results, "last_date": last_date, "stats": summarize(results, cfg)}


# ---------------------------------------------------------------- 集計
def _agg(vals: list[tuple[float, float | None]]) -> dict[str, Any] | None:
    if not vals:
        return None
    r = np.array([v[0] for v in vals], dtype=float)
    ex = np.array([v[0] - v[1] for v in vals if v[1] is not None], dtype=float)
    return {"n": len(r), "mean": float(r.mean()), "median": float(np.median(r)), "win": float((r > 0).mean()),
            "excess": float(ex.mean()) if ex.size else None, "beat": float((ex > 0).mean()) if ex.size else None,
            "bench": float(np.mean([v[1] for v in vals if v[1] is not None])) if ex.size else None}


def summarize(results: list[dict], cfg: Config) -> dict[str, Any]:
    tc = cfg["tracking"]
    hs = [int(h) for h in tc["horizons"]]
    by_h = {h: _agg([(r["cp"][str(h)][0], r["cp"][str(h)][1]) for r in results if str(h) in r.get("cp", {}) and r["cp"][str(h)][0] is not None]) for h in hs}
    ch = int(tc["cut_horizon"])

    def cut(fn: Any, order: list[str]) -> list[dict]:
        rows = []
        for name in order:
            vals = [(r["cp"][str(ch)][0], r["cp"][str(ch)][1]) for r in results if str(ch) in r.get("cp", {}) and fn(r) == name
                    and r["cp"][str(ch)][0] is not None]
            rows.append({"name": name, **(_agg(vals) or {"n": 0})})
        return rows

    edges = tc["main_edges"]
    def main_band(r: dict) -> str:
        m = r.get("main")
        if m is None:
            return "記録なし"
        lab = f"{edges[0]}点未満" if m < edges[0] else f"{edges[-1]}点以上" if m >= edges[-1] else \
            next(f"{a}〜{b}点" for a, b in zip(edges[:-1], edges[1:]) if a <= m < b)
        return lab
    main_order = [f"{edges[0]}点未満"] + [f"{a}〜{b}点" for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}点以上"]
    return {"by_horizon": by_h, "cut_horizon": ch,
            "by_rank": cut(lambda r: "1〜3位" if r["rank"] <= 3 else "4位以下", ["1〜3位", "4位以下"]),
            "by_stance": cut(lambda r: STANCE_LABEL.get(r.get("stance"), "記録なし"), [STANCE_LABEL[k] for k in ("buy", "neutral", "defense")]),
            "by_main": cut(main_band, main_order),
            "n_total": len(results), "n_active": sum(1 for r in results if r["status"] in ("掲載中", "圏外へ")),
            "n_listed": sum(1 for r in results if r["status"] == "掲載中"),
            "n_pending": sum(1 for r in results if r["status"] == "買い値待ち"),
            "now": _agg([(r["ret"], r.get("bench")) for r in results if r.get("ret") is not None])}


# ---------------------------------------------------------------- 図（SVG）
def spark_svg(vals: list[float] | None, w: int = 84, h: int = 26) -> str:
    """掲載後の推移（騰落率%）の小さな折れ線。点線が0%（買値）。"""
    if not vals or len(vals) < 2:
        return ""
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    span = (hi - lo) or 1.0
    xs = np.linspace(2, w - 2, len(vals))
    ys = [2 + (hi - v) / span * (h - 4) for v in vals]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    zero = 2 + (hi - 0) / span * (h - 4)
    cls = "up" if vals[-1] >= 0 else "down"
    return (f'<svg class="spark {cls}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="掲載後の推移">'
            f'<line x1="0" x2="{w}" y1="{zero:.1f}" y2="{zero:.1f}" class="z"/><polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>')


def avg_chart_svg(stats: dict[str, Any], w: int = 340, h: int = 190) -> str:
    """掲載後の経過日ごとの平均騰落率：ランキング銘柄 と TOPIX。"""
    pts = [(h_, v) for h_, v in stats["by_horizon"].items() if v and v["n"] >= 3]
    if len(pts) < 2:
        return ""
    xs = [p[0] for p in pts]
    a = [p[1]["mean"] * 100 for p in pts]
    b = [(p[1]["bench"] or 0) * 100 for p in pts]
    lo, hi = min(min(a), min(b), 0), max(max(a), max(b), 0)
    span = (hi - lo) or 1.0
    L, R, T, B = 34, 8, 10, 22
    X = lambda x: L + (x - xs[0]) / ((xs[-1] - xs[0]) or 1) * (w - L - R)  # noqa: E731
    Y = lambda v: T + (hi - v) / span * (h - T - B)  # noqa: E731
    line = lambda ys, cls: f'<polyline class="{cls}" fill="none" stroke-width="2" points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys)) + '"/>'  # noqa: E731
    dots = lambda ys, cls: "".join(f'<circle class="{cls}" r="3" cx="{X(x):.1f}" cy="{Y(y):.1f}"/>' for x, y in zip(xs, ys))  # noqa: E731
    grid = f'<line class="zero" x1="{L}" x2="{w - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>'
    ticks = "".join(f'<text class="tick" x="{X(x):.1f}" y="{h - 6}" text-anchor="middle">{x}日</text>' for x in xs)
    yl = "".join(f'<text class="tick" x="{L - 4}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:+.0f}%</text>' for v in {round(lo), 0, round(hi)})
    return (f'<svg class="avgchart" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="掲載後の平均推移">{grid}{yl}{ticks}'
            f'{line(b, "l-bench")}{dots(b, "d-bench")}{line(a, "l-pick")}{dots(a, "d-pick")}</svg>')


# ---------------------------------------------------------------- ファイル出力
def write_files(out: Path, result: dict[str, Any], generated: str) -> None:
    """docs/tracking.json（行を開いたときの日ごとの表）と docs/tracking_daily.csv（全銘柄×全日の数値。表計算で検証用）。"""
    rows = [r for r in result["entries"] if r["series"]]
    (out / "tracking.json").write_text(json.dumps(
        {"generated_at": generated, "entries": {r["key"]: [[s[0], s[1], s[2], None if s[3] is None else round(s[3] * 100, 2),
                                                          None if s[4] is None else round(s[4] * 100, 2), s[5]] for s in r["series"]] for r in rows}},
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with open(out / "tracking_daily.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["掲載日", "コード", "銘柄名", "掲載時順位", "掲載時総合点", "掲載時の相場判定", "掲載後の営業日", "日付", "終値",
                    "買値(翌営業日始値)", "騰落率%", "TOPIX騰落率%", "超過%", "その日の順位(圏外は空欄)"])
        for r in rows:
            for k, dt, px, ret, b, rk in r["series"]:
                w.writerow([r["date"], r["code"], r["name"], r["rank"], r.get("main"), STANCE_LABEL.get(r.get("stance")), k, dt,
                            round(px, 2) if px is not None else "", round(r["buy"], 2) if r.get("buy") else "",
                            "" if ret is None else round(ret * 100, 2), "" if b is None else round(b * 100, 2),
                            "" if (ret is None or b is None) else round((ret - b) * 100, 2), rk or ""])
