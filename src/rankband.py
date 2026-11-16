"""順位別・スコア帯別の分析（㉕）。「何位まで買うべきか」「何点以上なら期待値があるか」を確かめる。

① 過去の再現（週次バックテスト・10年）：その週の点数順に並べた順位・点数で買ったら、その後どうだったか
   ※過去の「業績・TDnet」は再現できないため、点数は『押し目スコアのテクニカル部分（100点換算）』。画面の総合点とは尺度が違う。
② 実運用の記録（追跡台帳）：実際に載った順位・総合点での結果。件数が貯まるまでは参考値。
買値＝翌営業日の始値、h営業日後＝掲載日からh営業日後の終値、コスト（往復）込み。順位・点数は毎日変わるので、その日の点数順・順位順で数える。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .backtest import equity_metrics, trade_metrics
from .validation import _indep, tstat


def _f(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _profile(per_date: list[dict[int, np.ndarray]], uni: list[dict[int, float]], hs: list[int], eh: int, step: int,
             portfolio: bool, min_n: int) -> dict[str, Any]:
    """per_date[i][h]＝その日にこの条件で買った銘柄のh日後騰落率の配列。"""
    k = max(1, math.ceil(eh / step))
    out: dict[str, Any] = {"by_h": {}}
    for h in hs:
        arrs = [d.get(h, np.array([])) for d in per_date]
        arrs = [a[np.isfinite(a)] for a in arrs]
        dm = [(float(a.mean()), uni[i].get(h)) for i, a in enumerate(arrs) if a.size]
        if portfolio:
            sample = np.array([m for m, _ in dm])
        else:
            sample = np.concatenate([a for a in arrs if a.size]) if any(a.size for a in arrs) else np.array([])
        ex = [m - u for m, u in dm if u is not None and math.isfinite(u)]
        t = tstat(_indep(ex, max(1, math.ceil(h / step)))) if ex else None
        out["by_h"][h] = {"n": int(sample.size), "days": len(dm),
                          "win": float((sample > 0).mean()) if sample.size else None,
                          "avg": float(sample.mean()) if sample.size else None,
                          "excess": float(np.mean(ex)) if ex else None, "t": t}
    # 資産曲線系（eh営業日ごと・重ならない）：最大ドローダウン・シャープ・PF
    fin = [d.get(eh, np.array([]))[np.isfinite(d.get(eh, np.array([])))] for d in per_date]
    series = np.array([float(a.mean()) for a in fin if a.size][::k], dtype=float)
    em = equity_metrics(series, eh) if series.size >= 3 else None
    pooled = series if portfolio else (np.concatenate([a for a in fin if a.size]) if any(a.size for a in fin) else np.array([]))
    tm = trade_metrics(pooled) if pooled.size else None
    out.update({"pf": (tm or {}).get("pf"), "sharpe": (em or {}).get("sharpe"), "max_dd": (em or {}).get("max_dd"),
                "total_ret": (em or {}).get("total_ret"), "n_periods": int(series.size),
                "enough": bool(out["by_h"][eh]["n"] >= min_n)})
    out["equity"] = [round(float(x), 5) for x in np.cumprod(1 + series)[:: max(1, len(series) // 120)]] if series.size >= 3 else []
    return out


def status(row: dict, eh: int, rc: dict) -> tuple[str, str]:
    """(判定, 理由)。主な保有期間の平均が正で、全体平均より有意に良ければ『推奨』。"""
    c = row["by_h"][eh]
    if not row["enough"] or c["avg"] is None:
        return "判定不能", f"件数が少ない（{c['n']}件）"
    ex, t = c["excess"], c["t"]
    if c["avg"] > 0 and ex is not None and ex > 0 and t is not None and t >= rc["t_good"]:
        return "推奨", f"平均{c['avg'] * 100:+.2f}%・全体平均より{ex * 100:+.2f}%（t={t:.1f}）"
    if c["avg"] > 0 and ex is not None and ex > 0:
        return "中立", f"全体平均より良いが偶然の範囲（t={t:.1f}）" if t is not None else "全体平均より良いが偶然の範囲"
    return "見送り", f"平均{c['avg'] * 100:+.2f}%・全体平均より{(ex or 0) * 100:+.2f}%"


def analyze_replay(recs: list[dict], cfg: Any, step: int) -> dict[str, Any]:
    """recs：日ごとの {date, s（全銘柄の点数）, rk（順位順の位置）, ret{h:配列}, bench{h}}。"""
    rc = cfg["rankband"]
    hs, eh = [int(h) for h in rc["horizons"]], int(rc["equity_horizon"])
    recs = [r for r in recs if len(r["s"]) >= 50]
    if len(recs) < 20:
        return {"ok": False, "reason": "分析に使える日数が足りません"}
    uni = [{h: _f(np.nanmean(r["ret"][h])) if np.isfinite(r["ret"][h]).any() else None for h in hs} for r in recs]
    bench = [r["bench"] for r in recs]

    def pick(fn: Any) -> list[dict[int, np.ndarray]]:
        out = []
        for r in recs:
            idx = fn(r)
            out.append({h: r["ret"][h][idx] if len(idx) else np.array([]) for h in hs})
        return out

    mn = int(rc["min_trades"])
    ranks = []
    for lo, hi in rc["rank_groups"]:
        p = _profile(pick(lambda r, lo=lo, hi=hi: r["rk"][lo - 1:hi]), uni, hs, eh, step, False, mn)
        p.update(name=f"{lo}位" if lo == hi else f"{lo}〜{hi}位", lo=lo, hi=hi)
        p["status"], p["why"] = status(p, eh, rc)
        ranks.append(p)
    ports = []
    for n in rc["portfolios"]:
        p = _profile(pick(lambda r, n=n: r["rk"][:n]), uni, hs, eh, step, True, int(rc["min_days"]))
        p.update(name=f"TOP{n}", n=n)
        p["status"], p["why"] = status(p, eh, rc)
        ports.append(p)
    levels = []
    for lv in rc["score_levels"]:
        p = _profile(pick(lambda r, lv=lv: np.flatnonzero(r["s"] >= lv)), uni, hs, eh, step, False, mn)
        p.update(name=f"{lv}点以上", level=lv)
        p["days_with"] = int(sum(1 for d in pick(lambda r, lv=lv: np.flatnonzero(r["s"] >= lv)) if d[eh].size))
        p["status"], p["why"] = status(p, eh, rc)
        levels.append(p)
    allp = _profile(pick(lambda r: np.arange(len(r["s"]))), uni, hs, eh, step, False, mn)
    allp["name"] = "全銘柄の平均（基準）"
    bm = {h: _f(np.nanmean([b[h] for b in bench])) for h in hs}
    v = {"ok": True, "horizons": hs, "eh": eh, "n_dates": len(recs), "step": step,
         "period": {"start": str(recs[0]["date"].date()), "end": str(recs[-1]["date"].date())},
         "ranks": ranks, "portfolios": ports, "levels": levels, "all": allp, "bench": bm}
    v["advice"] = advice(v, rc)
    return v


def advice(v: dict, rc: dict) -> dict[str, Any]:
    """「何位まで買うべきか」「何点以上なら期待値があるか」の結論（文章）。"""
    eh = v["eh"]
    # 順位：1位から順に、見送りが出るまで
    thru, names = 0, []
    for g in v["ranks"]:
        if g["status"] in ("推奨", "中立"):
            thru = g["hi"]; names.append(g["name"])
        else:
            break
    rank_msg = (f"{thru}位まで" if thru else "どの順位帯も、平均より有意に良いとは言えません") + (
        "（1位から順に見て、平均より良い順位帯が続く範囲）" if thru else "")
    sh = [(p["sharpe"], p["name"]) for p in v["portfolios"] if p["sharpe"] is not None]
    best_port = max(sh)[1] if sh else None
    # 点数：推奨になる最も低い点数
    lv = sorted(v["levels"], key=lambda g: g["level"])
    rec = [g["level"] for g in lv if g["status"] == "推奨"]
    neu = [g["level"] for g in lv if g["status"] == "中立"]
    parts = []
    for g in sorted(v["levels"], key=lambda g: -g["level"]):
        parts.append(f"{g['level']}点以上は{g['status']}")
    if rec:
        score_msg = f"{min(rec)}点以上が推奨"
    elif neu:
        score_msg = f"{min(neu)}点以上は中立（推奨と言えるほどの差は確認できません）"
    else:
        score_msg = "70点以上でも、平均より有意に良いとは言えません"
    mono = [g["by_h"][eh]["avg"] for g in sorted(v["levels"], key=lambda g: g["level"]) if g["by_h"][eh]["avg"] is not None]
    trend = None
    if len(mono) >= 3:
        trend = "点数が高いほど平均リターンが高い傾向" if all(b >= a - 1e-9 for a, b in zip(mono, mono[1:])) else "点数が高いほど良くなる、とは単純には言えません"
    return {"rank": rank_msg, "rank_thru": thru, "best_portfolio": best_port, "score": score_msg,
            "score_detail": "／".join(parts), "trend": trend, "rec_level": min(rec) if rec else None}


# ---------------------------------------------------------------- 実運用の記録（追跡台帳）
def analyze_live(results: list[dict], cfg: Any) -> dict[str, Any]:
    rc = cfg["rankband"]
    hs = [int(h) for h in rc["horizons"]]

    def row(sel: list[dict]) -> dict[str, Any]:
        cells = {}
        for h in hs:
            vals = [r["cp"][str(h)] for r in sel if str(h) in (r.get("cp") or {}) and r["cp"][str(h)][0] is not None]
            a = np.array([v[0] for v in vals], dtype=float)
            ex = [v[0] - v[1] for v in vals if v[1] is not None]
            tm = trade_metrics(a) if a.size else None
            cells[h] = {"n": int(a.size), "win": float((a > 0).mean()) if a.size else None, "avg": float(a.mean()) if a.size else None,
                        "excess": float(np.mean(ex)) if ex else None, "pf": (tm or {}).get("pf")}
        return {"by_h": cells}

    ranks = [{"name": f"{lo}位" if lo == hi else f"{lo}〜{hi}位", **row([r for r in results if lo <= r["rank"] <= hi])} for lo, hi in rc["rank_groups"]]
    levels = [{"name": f"{lv}点以上", **row([r for r in results if r.get("main") is not None and r["main"] >= lv])} for lv in rc["score_levels"]]
    ports = []
    for n in rc["portfolios"]:                              # その日のTOPn（順位順）を均等に買った場合
        by_date: dict[str, list[dict]] = {}
        for r in results:
            if r["rank"] <= n:
                by_date.setdefault(r["date"], []).append(r)
        cells = {}
        for h in hs:
            ms = []
            for d, rs in by_date.items():
                vals = [r["cp"][str(h)][0] for r in rs if str(h) in (r.get("cp") or {}) and r["cp"][str(h)][0] is not None]
                if len(vals) >= max(1, len(rs) // 2):
                    ms.append(float(np.mean(vals)))
            a = np.array(ms)
            cells[h] = {"n": int(a.size), "win": float((a > 0).mean()) if a.size else None, "avg": float(a.mean()) if a.size else None}
        ports.append({"name": f"TOP{n}", "by_h": cells})
    done = [r for r in results if r.get("cp")]
    return {"ranks": ranks, "levels": levels, "portfolios": ports, "n_entries": len(done), "horizons": hs,
            "n_days": len({r["date"] for r in done})}


# ---------------------------------------------------------------- 図（SVG）
def bars_svg(labels: list[str], series: list[tuple[str, list[float | None]]], unit: str = "%", base: float | None = None,
             w: int = 360, h: int = 200) -> str:
    """縦棒グラフ。series＝[(凡例, 値のリスト)]。プラスは緑・マイナスは赤（色以外に値の数字も表示）。base＝基準線（勝率の50%など）。"""
    vals = [v for _, vs in series for v in vs if v is not None]
    if not vals:
        return ""
    ref = 0.0 if base is None else base                    # 棒の起点（勝率は50%を起点にして、差が見えるようにする）
    lo, hi = min(min(vals), ref), max(max(vals), ref)
    pad = ((hi - lo) or 1.0) * 0.12
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    L, R, T, B = 40, 8, 14, 34
    Y = lambda v: T + (hi - v) / span * (h - T - B)  # noqa: E731
    gw = (w - L - R) / len(labels)
    bw = min(26.0, gw / (len(series) + 0.6))
    out = [f'<svg class="barchart" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="棒グラフ">']
    for v in sorted({round(lo, 1), round(ref, 1), round(hi, 1)}):
        out.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="tick" x="{L - 4}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:g}{unit}</text>')
    if base is not None:
        out.append(f'<line class="base" x1="{L}" x2="{w - R}" y1="{Y(base):.1f}" y2="{Y(base):.1f}"/>')
    for i, lab in enumerate(labels):
        x0 = L + gw * i + (gw - bw * len(series)) / 2
        out.append(f'<text class="tick" x="{L + gw * i + gw / 2:.1f}" y="{h - 18}" text-anchor="middle">{lab}</text>')
        for j, (_, vs) in enumerate(series):
            v = vs[i]
            if v is None:
                out.append(f'<text class="tick" x="{x0 + bw * j + bw / 2:.1f}" y="{Y(ref) - 3:.1f}" text-anchor="middle">—</text>')
                continue
            y0, y1 = Y(max(v, ref)), Y(min(v, ref))
            cls = f"s{j}" + ((" pos" if v >= ref else " neg") if len(series) == 1 else "")
            out.append(f'<rect class="{cls}" x="{x0 + bw * j:.1f}" y="{y0:.1f}" width="{bw - 2:.1f}" height="{max(y1 - y0, 1):.1f}"/>'
                       f'<text class="val" x="{x0 + bw * j + (bw - 2) / 2:.1f}" y="{(y0 - 3) if v >= ref else (y1 + 10):.1f}" text-anchor="middle">{v:.1f}</text>')
    out.append("</svg>")
    leg = "".join(f'<span class="lg s{j}">{n}</span>' for j, (n, _) in enumerate(series)) if len(series) > 1 else ""
    return "".join(out) + (f'<div class="legend">{leg}</div>' if leg else "")


def lines_svg(curves: list[tuple[str, list[float]]], w: int = 360, h: int = 210) -> str:
    """資産曲線（1.0＝開始時）。複数の線を重ねる。"""
    curves = [(n, c) for n, c in curves if c and len(c) >= 3]
    if not curves:
        return ""
    allv = [v for _, c in curves for v in c] + [1.0]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    L, R, T, B = 40, 8, 10, 18
    n = max(len(c) for _, c in curves)
    X = lambda i, m: L + i / max(m - 1, 1) * (w - L - R)  # noqa: E731
    Y = lambda v: T + (hi - v) / span * (h - T - B)  # noqa: E731
    out = [f'<svg class="barchart" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="資産曲線">',
           f'<line class="base" x1="{L}" x2="{w - R}" y1="{Y(1):.1f}" y2="{Y(1):.1f}"/>']
    for v in sorted({round(lo, 2), 1.0, round(hi, 2)}):
        out.append(f'<text class="tick" x="{L - 4}" y="{Y(v) + 3:.1f}" text-anchor="end">{(v - 1) * 100:+.0f}%</text>')
    for j, (_, c) in enumerate(curves):
        out.append(f'<polyline class="ln s{j}" fill="none" stroke-width="2" points="' + " ".join(f"{X(i, len(c)):.1f},{Y(v):.1f}" for i, v in enumerate(c)) + '"/>')
    out.append("</svg>")
    return "".join(out) + '<div class="legend">' + "".join(f'<span class="lg s{j}">{n}</span>' for j, (n, _) in enumerate(curves)) + "</div>"


# ---------------------------------------------------------------- ページ用
def _norm(v: dict | None) -> dict | None:
    """JSONから読み直すと辞書のキー（日数）が文字列になるので、数値に戻す。"""
    if not v or not v.get("ok"):
        return v
    fix = lambda g: {**g, "by_h": {int(k): c for k, c in g["by_h"].items()}}  # noqa: E731
    out = dict(v)
    for k in ("ranks", "portfolios", "levels"):
        out[k] = [fix(g) for g in v[k]]
    out["all"] = fix(v["all"])
    out["bench"] = {int(k): x for k, x in v["bench"].items()}
    return out


def page_ctx(summary: dict | None, tracking_result: dict | None, cfg: Any) -> dict[str, Any]:
    v = _norm((summary or {}).get("rankband"))
    live = analyze_live(tracking_result["entries"], cfg) if tracking_result else None
    ch: dict[str, str] = {}
    if v and v.get("ok"):
        eh = v["eh"]
        pc = lambda g, h: None if g["by_h"][h]["avg"] is None else round(g["by_h"][h]["avg"] * 100, 2)   # noqa: E731
        wn = lambda g, h: None if g["by_h"][h]["win"] is None else round(g["by_h"][h]["win"] * 100, 1)   # noqa: E731
        for key, rows in (("rank", v["ranks"]), ("lv", list(reversed(v["levels"])))):
            names = [g["name"].replace("点以上", "+") for g in rows]
            hs = [eh, 20] if 20 in rows[0]["by_h"] else [eh]
            ch[f"{key}_avg"] = bars_svg(names, [(f"{h}日後", [pc(g, h) for g in rows]) for h in hs])
            ch[f"{key}_win"] = bars_svg(names, [(f"{h}日後", [wn(g, h) for g in rows]) for h in hs], base=50.0)
        ch["equity"] = lines_svg([(g["name"], g["equity"]) for g in v["portfolios"]])
    return {"rb_replay": v, "rb_live": live, "ch": ch, "rcfg": cfg["rankband"], "cost_pct": cfg["backtest"]["cost_pct"]}
