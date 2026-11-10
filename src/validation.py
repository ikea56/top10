"""有効性の検証（過去再現）。

「点数が高い銘柄ほど、その後の成績が良かったか」を、過去の株価データだけで確かめる。
週次バックテスト（python -m src.backtest）の中で自動的に実行され、結果は backtest_summary.json の "validation" に入る。

検証する内容
  1. 点数帯ごとの成績         … 点数が高い帯ほど成績が良いか（単調性）
  2. 点数と成績の相関（IC）   … 各日の「点数の順位」と「その後の騰落率の順位」の相関。偶然かどうかをt値で判定
  3. 五分位の差               … 点数上位20%と下位20%の差
  4. 上位銘柄と全体の差       … 上位N銘柄が、条件を満たす全銘柄の平均を上回ったか（信頼区間つき）
  5. 要素別の効果             … 各要素を外した場合との比較（その要素に意味があるか）
  6. 入れ替わりの多さ         … 毎週のトップNの入れ替わり率
  7. 買う日の比較             … 翌日の始値で買う場合と、翌日の終値で買う場合
  8. 撤退ラインの効果         … 「移動平均線−ATR」を割ったら売る場合との比較
  9. 地合い別・年別・直近期間 … 特定の局面だけで成績が出ていないか／検証期間（直近N年）でも成立するか
  10. 業種の偏り
  11. 総合判定                … 🟢/🟡/🔴

限界（画面にも必ず表示する）
  ・現在の上場銘柄だけで過去を再現するため、上場廃止になった銘柄が無く成績は甘めに出る（生存者バイアス）。
  ・使うのは株価・出来高だけで決まる点数（押し目スコアの技術80点分）。業績・TDnetは過去データが無く、検証できない。
  ・毎週の再現は期間が重なるため、t値や信頼区間は「重ならないように間引いた系列」で計算している。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

COMPONENTS = {"pb_trend": "長期トレンド", "pb_dd": "高値からの調整", "pb_ma50": "50日線に接近",
              "pb_vol": "出来高の収縮", "pb_rs": "相対力", "pb_bounce": "反発確認"}
REGIME_LABEL = {"good": "追い風", "neutral": "中立", "bad": "逆風"}


# ---------------------------------------------------------------- 小さな統計関数
def _num(v: Any) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _indep(values: list[float], k: int) -> np.ndarray:
    """期間が重なる系列から、k個おきに取り出して重ならないようにする（NaNは除く）。"""
    a = np.asarray(values, dtype=float)[:: max(1, k)]
    return a[np.isfinite(a)]


def tstat(a: np.ndarray) -> float | None:
    """平均÷標準誤差。データが少ない・ばらつきが無い場合は None。"""
    if a.size < 8:
        return None
    sd = float(np.std(a, ddof=1))
    return float(np.mean(a) / (sd / math.sqrt(a.size))) if sd > 0 else None


def boot_ci(a: np.ndarray, n: int, seed: int = 0) -> tuple[float, float] | None:
    """平均の95%信頼区間（リサンプリング）。乱数は固定なので毎回同じ結果になる。"""
    if a.size < 8:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(int(n), a.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float | None:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return None
    r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
    return None if r is None or not math.isfinite(r) else float(r)


def _mean(a: Any) -> float | None:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


# ---------------------------------------------------------------- 各検証
def _per_date(records: list[dict], rcol: str, n_top: int) -> list[dict]:
    """日ごとに、上位N銘柄・全体の平均騰落率などを集計する（先の期間が無い日は除く）。"""
    out = []
    for rec in records:
        x = rec["x"]
        r = x[rcol].to_numpy(dtype=float)
        ok = np.isfinite(r)
        if ok.sum() < 50:
            continue
        pk = [t for t in rec["pb_picks"] if t in x.index]
        rp = x.loc[pk, rcol].to_numpy(dtype=float) if pk else np.array([])
        rp = rp[np.isfinite(rp)]
        if rp.size < max(3, n_top // 2):
            continue
        lp = [t for t in rec.get("lg_picks", []) if t in x.index]
        rl = x.loc[lp, rcol].to_numpy(dtype=float) if lp else np.array([])
        rl = rl[np.isfinite(rl)]
        out.append({"rec": rec, "date": rec["date"], "regime": rec["regime"], "univ": float(r[ok].mean()),
                    "top": float(rp.mean()), "top_win": float((rp > 0).mean()),
                    "legacy": float(rl.mean()) if rl.size else float("nan"),
                    "bench": rec["bench"].get(rcol[2:], float("nan"))})
    return out


def score_bands(records: list[dict], rcol: str, edges: list[float], min_obs: int) -> dict[str, Any]:
    s = np.concatenate([rec["x"]["s"].to_numpy(dtype=float) for rec in records])
    r = np.concatenate([rec["x"][rcol].to_numpy(dtype=float) for rec in records])
    ex = np.concatenate([(rec["x"][rcol] - rec["x"][rcol].mean()).to_numpy(dtype=float) for rec in records])
    ok = np.isfinite(s) & np.isfinite(r)
    s, r, ex = s[ok], r[ok], ex[ok]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        n = int(m.sum())
        rows.append({"lo": lo, "hi": hi, "n": n,
                     "avg": float(r[m].mean()) if n >= min_obs else None,
                     "excess": float(ex[m].mean()) if n >= min_obs else None,
                     "win": float((r[m] > 0).mean()) if n >= min_obs else None})
    good = [(i, x["excess"]) for i, x in enumerate(rows) if x["excess"] is not None]
    rho = spearman(np.array([i for i, _ in good], dtype=float), np.array([e for _, e in good], dtype=float), min_n=4) if len(good) >= 4 else None
    return {"rows": rows, "rho": rho}


def information_coefficients(records: list[dict], rcol: str, k: int, cols: list[str]) -> dict[str, dict]:
    """各日の Spearman 相関（点数の順位 × その後の騰落率の順位）。平均が正で、t値が大きいほど『点数に意味がある』。"""
    out: dict[str, dict] = {}
    for col in cols:
        ics = []
        for rec in records:
            x = rec["x"]
            ics.append(spearman(x[col].to_numpy(dtype=float), x[rcol].to_numpy(dtype=float)))
        arr = np.array([i if i is not None else np.nan for i in ics], dtype=float)
        ind = _indep(list(arr), k)
        out[col] = {"mean": _mean(arr), "t": tstat(ind), "pos": float((arr[np.isfinite(arr)] > 0).mean()) if np.isfinite(arr).any() else None,
                    "n": int(ind.size), "series": ics}
    return out


def quantile_table(records: list[dict], rcol: str, q: int, k: int) -> dict[str, Any]:
    """点数順に q 分位へ分け、各分位の平均超過リターン（その日の全体平均との差）と、上位−下位の差を出す。"""
    sums = np.zeros(q)
    cnt = np.zeros(q)
    spreads = []
    for rec in records:
        x = rec["x"]
        m = x["s"].notna() & x[rcol].notna()
        if m.sum() < q * 10:
            spreads.append(np.nan)
            continue
        d = x[m]
        grp = (d["s"].rank(method="first", pct=True) * q).clip(upper=q - 1e-9).astype(int).clip(upper=q - 1)
        ex = d[rcol] - d[rcol].mean()
        means = ex.groupby(grp).mean().reindex(range(q))
        for g in range(q):
            if np.isfinite(means.iloc[g]):
                sums[g] += means.iloc[g]
                cnt[g] += 1
        spreads.append(float(means.iloc[q - 1] - means.iloc[0]))
    ind = _indep(spreads, k)
    return {"means": [float(sums[g] / cnt[g]) if cnt[g] else None for g in range(q)],
            "spread": _mean(spreads), "spread_t": tstat(ind), "n": int(ind.size)}


def top_vs_universe(days: list[dict], k: int, boot: int) -> dict[str, Any]:
    ex = [d["top"] - d["univ"] for d in days]
    exb = [d["top"] - d["bench"] if np.isfinite(d["bench"]) else np.nan for d in days]
    exl = [d["legacy"] - d["univ"] for d in days]
    ind = _indep(ex, k)
    ci = boot_ci(ind, boot)
    return {"n_dates": len(days), "n_indep": int(ind.size),
            "top": _mean([d["top"] for d in days]), "univ": _mean([d["univ"] for d in days]),
            "excess": _mean(ex), "t": tstat(ind), "ci": ci, "hit": float((np.asarray(ex) > 0).mean()) if ex else None,
            "excess_bench": _mean(exb), "win": _mean([d["top_win"] for d in days]),
            "legacy_excess": _mean(exl)}


def ablation(records: list[dict], rcol: str, n_top: int, k: int) -> dict[str, dict]:
    """各要素を外した場合のトップNの超過リターンとの差。差が正なら『その要素は成績に貢献』、負なら『むしろ足を引っ張っている』。"""
    cols = list(COMPONENTS) + ["pen"]
    full_s, wo = [], {c: [] for c in cols}
    for rec in records:
        x = rec["x"]
        r = x[rcol].to_numpy(dtype=float)
        ok = np.isfinite(r)
        if ok.sum() < 50:
            continue
        comp = x[list(COMPONENTS)].to_numpy(dtype=float)
        raw = np.nansum(comp, axis=1) - x["pen"].to_numpy(dtype=float)
        base = r[ok].mean()

        def top_ex(score: np.ndarray) -> float:
            sc = np.where(ok, score, -np.inf)
            idx = np.argsort(-sc, kind="stable")[:n_top]
            return float(np.nanmean(r[idx]) - base)

        full = top_ex(raw)
        full_s.append(full)
        for j, c in enumerate(COMPONENTS):
            wo[c].append(top_ex(raw - np.nan_to_num(comp[:, j])))
        wo["pen"].append(top_ex(raw + x["pen"].to_numpy(dtype=float)))
    out: dict[str, dict] = {}
    for c in cols:
        diff = np.array(full_s) - np.array(wo[c]) if full_s else np.array([])
        ind = _indep(list(diff), k)
        out[c] = {"delta": _mean(diff), "t": tstat(ind)}
    out["_full"] = {"excess": _mean(full_s), "n": len(full_s)}
    return out


def turnover(records: list[dict], key: str, n_top: int) -> dict[str, Any]:
    ret = []
    prev: set[str] | None = None
    for rec in records:
        cur = set(rec.get(key, [])[:n_top])
        if prev is not None and cur:
            ret.append(len(prev & cur) / max(1, len(cur)))
        prev = cur
    return {"retention": _mean(ret), "new_per_step": None if not ret else (1 - float(np.mean(ret))) * n_top, "n": len(ret)}


def entry_timing(records: list[dict], rcol: str, ccol: str, k: int) -> dict[str, Any]:
    a, b = [], []
    for rec in records:
        x = rec["x"]
        pk = [t for t in rec["pb_picks"] if t in x.index]
        if not pk or ccol not in x:
            continue
        ra, rb = x.loc[pk, rcol].to_numpy(dtype=float), x.loc[pk, ccol].to_numpy(dtype=float)
        m = np.isfinite(ra) & np.isfinite(rb)
        if m.sum() >= 3:
            a.append(float(ra[m].mean()))
            b.append(float(rb[m].mean()))
    diff = np.array(a) - np.array(b)
    return {"open": _mean(a), "close": _mean(b), "diff": _mean(diff), "t": tstat(_indep(list(diff), k)), "n": len(a)}


def stop_effect(records: list[dict], k: int) -> dict[str, Any] | None:
    ns, st, hit, diffs = [], [], [], []
    for rec in records:
        s = rec.get("stop")
        if not s:
            continue
        a, b, h = np.asarray(s["ns"], dtype=float), np.asarray(s["st"], dtype=float), np.asarray(s["hit"], dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            continue
        ns.append(a[m])
        st.append(b[m])
        hit.append(h[m])
        diffs.append(float(b[m].mean() - a[m].mean()))
    if not ns:
        return None
    A, B, H = np.concatenate(ns), np.concatenate(st), np.concatenate(hit)
    pct = lambda v: float(np.percentile(v, 5))  # noqa: E731
    return {"no_stop": float(A.mean()), "with_stop": float(B.mean()), "diff": float(np.mean(diffs)),
            "t": tstat(_indep(diffs, k)), "win_no": float((A > 0).mean()), "win_st": float((B > 0).mean()),
            "worst_no": float(A.min()), "worst_st": float(B.min()), "p5_no": pct(A), "p5_st": pct(B),
            "hit": float(H.mean()), "n": int(A.size)}


def by_regime(days: list[dict], k: int) -> list[dict]:
    rows = []
    for rg in ("good", "neutral", "bad"):
        d = [x for x in days if x["regime"] == rg]
        ex = [x["top"] - x["univ"] for x in d]
        rows.append({"regime": rg, "n": len(d), "excess": _mean(ex), "top": _mean([x["top"] for x in d]),
                     "win": _mean([x["top_win"] for x in d]), "hit": float((np.asarray(ex) > 0).mean()) if ex else None})
    return rows


def by_year(days: list[dict], ics: list[float | None], dates: list) -> list[dict]:
    ic_by = {}
    for dt, ic in zip(dates, ics):
        if ic is not None:
            ic_by.setdefault(pd.Timestamp(dt).year, []).append(ic)
    rows = []
    for y in sorted({pd.Timestamp(d["date"]).year for d in days}):
        d = [x for x in days if pd.Timestamp(x["date"]).year == y]
        ex = [x["top"] - x["univ"] for x in d]
        rows.append({"year": y, "n": len(d), "excess": _mean(ex), "ic": _mean(ic_by.get(y, []))})
    return rows


# ---------------------------------------------------------------- 判定
def _status(good: bool, weak: bool) -> str:
    return "good" if good else "weak" if weak else "bad"


def verdict(v: dict, cfg: Config) -> dict[str, Any]:
    c = cfg["validation"]["verdict"]
    tg, tw = c["t_good"], c["t_weak"]
    ic, top, yrs, oos = v["ic"]["s"], v["top"], v["years"], v["oos"]
    ic_m, ic_t = ic.get("mean"), ic.get("t")
    s1 = _status(bool(ic_m and ic_m > 0 and ic_t is not None and ic_t >= tg), bool(ic_m and ic_m > 0 and ic_t is not None and ic_t >= tw))
    ex, ex_t, ci = top.get("excess"), top.get("t"), top.get("ci")
    s2 = _status(bool(ex and ex > 0 and ex_t is not None and ex_t >= tg and ci and ci[0] > 0),
                 bool(ex and ex > 0 and ex_t is not None and ex_t >= tw))
    pos = [r for r in yrs if r["excess"] is not None]
    share = sum(1 for r in pos if r["excess"] > 0) / len(pos) if pos else None
    oos_ex = (oos.get("top") or {}).get("excess")
    s3 = _status(bool(share is not None and share >= c["positive_year_good"] and oos_ex is not None and oos_ex > 0),
                 bool(share is not None and share >= c["positive_year_weak"]))
    if s1 == "bad" or s2 == "bad":
        level, head = "bad", "過去の再現では、有効性を確認できませんでした"
        use = "このランキングだけで売買を判断する根拠は、いまのところありません。候補を探す参考にとどめ、前向きの記録（実際の成績）が貯まるまで売買の金額は小さく保ってください。"
    elif s1 == s2 == s3 == "good":
        level, head = "good", "過去の再現では、有効性が確認できました"
        use = "候補を絞るフィルターとして使う根拠はあります。ただし、買う金額・銘柄数・撤退ラインは別に決めてください。業績・開示の部分は検証できていません。"
    else:
        level, head = "weak", "一部で有効性が確認できましたが、確実とは言えません"
        use = "少額から使い、前向きの記録（実際の成績）で確かめながら判断してください。"
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    pc = lambda x: "—" if x is None else f"{x * 100:+.2f}%"  # noqa: E731
    lines = [
        {"status": s1, "text": f"点数と成績の関係：平均相関 {f(ic_m, 3)}、t値 {f(ic_t)}（2以上で偶然とは言いにくい目安）。"},
        {"status": s2, "text": f"上位{v['n_top']}銘柄と全体平均の差：平均 {pc(ex)}、t値 {f(ex_t)}"
                               + (f"、95%信頼区間 {pc(ci[0])}〜{pc(ci[1])}" if ci else "") + "。"},
        {"status": s3, "text": "安定性：" + ("年ごとのプラス率 " + f"{share * 100:.0f}%" if share is not None else "年別データ不足")
                               + f"、直近{cfg['validation']['oos_years']}年の差 {pc(oos_ex)}。"},
    ]
    return {"level": level, "headline": head, "use": use, "lines": lines, "share_years": share}


# ---------------------------------------------------------------- 本体
def analyze(records: list[dict], cfg: Config, end: pd.Timestamp, step_days: int, horizons: dict[str, int]) -> dict[str, Any]:
    vc, n_top = cfg["validation"], int(cfg["top_n"])
    hz = vc["horizon"]
    rcol, ccol = f"r_{hz}", f"rc_{hz}"
    k = max(1, math.ceil(horizons[hz] / step_days))
    records = [r for r in records if r.get("x") is not None and len(r["x"])]
    days = _per_date(records, rcol, n_top)
    if len(days) < 20:
        return {"ok": False, "reason": "検証に使える日数が足りません（20日未満）"}
    recs = [d["rec"] for d in days]
    cols = ["s"] + list(COMPONENTS)
    ics = information_coefficients(recs, rcol, k, cols)
    oos_from = end - pd.DateOffset(years=int(vc["oos_years"]))
    oos_days = [d for d in days if d["date"] > oos_from]
    is_days = [d for d in days if d["date"] <= oos_from]
    oos_recs = [d["rec"] for d in oos_days]
    ic_oos = information_coefficients(oos_recs, rcol, k, ["s"])["s"] if len(oos_recs) >= 20 else {}
    ic_is = information_coefficients([d["rec"] for d in is_days], rcol, k, ["s"])["s"] if len(is_days) >= 20 else {}
    v: dict[str, Any] = {
        "ok": True, "horizon": hz, "n_top": n_top, "n_dates": len(days), "step_days": step_days, "k": k,
        "period": {"start": str(days[0]["date"].date()), "end": str(days[-1]["date"].date())},
        "bands": score_bands(recs, rcol, vc["band_edges"], vc["min_obs"]),
        "ic": {c: {kk: vv for kk, vv in d.items() if kk != "series"} for c, d in ics.items()},
        "quantiles": quantile_table(recs, rcol, int(vc["quantiles"]), k),
        "top": top_vs_universe(days, k, vc["bootstrap"]),
        "ablation": ablation(recs, rcol, n_top, k),
        "turnover": {"pb": turnover(records, "pb_picks", n_top), "legacy": turnover(records, "lg_picks", n_top)},
        "entry": entry_timing(recs, rcol, ccol, k),
        "stop": stop_effect(recs, k),
        "regime": by_regime(days, k),
        "years": by_year(days, ics["s"]["series"], [r["date"] for r in recs]),
        "oos": {"years": int(vc["oos_years"]), "top": top_vs_universe(oos_days, k, vc["bootstrap"]) if len(oos_days) >= 20 else {},
                "ic": {kk: vv for kk, vv in ic_oos.items() if kk != "series"},
                "is_top": top_vs_universe(is_days, k, vc["bootstrap"]) if len(is_days) >= 20 else {},
                "is_ic": {kk: vv for kk, vv in ic_is.items() if kk != "series"}},
        "sector": {"distinct": _mean([r["sector"][0] for r in recs if r.get("sector")]),
                   "top_share": _mean([r["sector"][1] for r in recs if r.get("sector")])},
    }
    v["verdict"] = verdict(v, cfg)
    v["tables"] = build_tables(v, cfg)
    return v


# ---------------------------------------------------------------- 画面用の表
def _p(x: float | None, d: int = 2) -> str:
    return "—" if x is None else f"{x * 100:+.{d}f}%"


def _n(x: float | None, d: int = 2) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _w(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def build_tables(v: dict, cfg: Config) -> list[dict]:
    hz = v["horizon"]
    label = {"1w": "1週間後", "1m": "1か月後", "3m": "3か月後"}.get(hz, hz)
    N = v["n_top"]
    T: list[dict] = []
    rows = [[f"{int(b['lo'])}〜{int(min(b['hi'], 100))}点", f"{b['n']:,}", _p(b["avg"]), _p(b["excess"]), _w(b["win"])] for b in v["bands"]["rows"]]
    T.append({"id": "bands", "title": "① 点数帯ごとの成績", "cols": ["点数帯", "データ数", f"平均騰落率（{label}）", "全体平均との差", "勝率"], "rows": rows,
              "note": f"点数が高い帯ほど「全体平均との差」が大きくなっていれば、点数に意味があります（順位相関 {_n(v['bands']['rho'])}、1に近いほど単調）。データ数は期間が重なるため独立ではありません。"})
    q = v["quantiles"]
    T.append({"id": "quantiles", "title": "② 点数順の五分位", "cols": ["分位（低→高）"] + [f"Q{i + 1}" for i in range(len(q["means"]))] + ["上位−下位"],
              "rows": [[f"全体平均との差（{label}）"] + [_p(m) for m in q["means"]] + [_p(q["spread"]) + f"（t={_n(q['spread_t'])}）"]],
              "note": "Q5（点数が最も高い20%）がQ1（最も低い20%）を上回っていれば、点数の高さが成績に結びついています。"})
    t = v["top"]
    ci = t.get("ci")
    rows = [[f"押し目トップ{N}", _p(t["top"]), _p(t["excess"]), _n(t["t"]), (f"{_p(ci[0])}〜{_p(ci[1])}" if ci else "—"), _w(t["hit"]), _w(t["win"])],
            ["（比較）従来の「今強い」", "—", _p(t.get("legacy_excess")), "—", "—", "—", "—"]]
    T.append({"id": "top", "title": f"③ 上位{N}銘柄は、全体の平均を上回ったか", "cols": ["", f"平均騰落率（{label}）", "全体平均との差", "t値", "95%信頼区間", "上回った割合", "個別の勝率"], "rows": rows,
              "note": f"全体平均＝条件を満たす全銘柄の平均。信頼区間が0をまたがなければ、偶然とは言いにくい差です（重ならないよう{v['k']}回おきに間引いて計算、{t['n_indep']}期間）。"})
    ab = v["ablation"]
    rows = []
    for c, name in COMPONENTS.items():
        ic = v["ic"].get(c, {})
        a = ab.get(c, {})
        d = a.get("delta")
        tt = a.get("t")
        if d is None or tt is None or abs(tt) < 1:
            verdict_txt = "差が小さく、効果は不明"
        else:
            verdict_txt = "貢献" if d > 0 else "足を引っ張っている可能性"
        rows.append([name, _n(ic.get("mean"), 3), _n(ic.get("t")), _p(d), _n(a.get("t")), verdict_txt])
    pen = ab.get("pen", {})
    rows.append(["急騰などの減点", "—", "—", _p(pen.get("delta")), _n(pen.get("t")), "—"])
    T.append({"id": "components", "title": "④ 要素ごとの効果", "cols": ["要素", "平均相関", "t値", "外した場合との差", "t値", "見立て"], "rows": rows,
              "note": f"「外した場合との差」＝その要素を除いて上位{N}銘柄を選び直したときと比べた、{label}の超過リターンの差。負の要素は配点を下げる検討対象です（偶然の可能性もあるため、t値が1未満なら断定しません）。"})
    tp, tl = v["turnover"]["pb"], v["turnover"]["legacy"]
    T.append({"id": "turnover", "title": "⑤ 入れ替わりの多さ", "cols": ["", "前回から残る割合", f"毎回の入れ替わり（{N}銘柄中）"],
              "rows": [["押し目（メイン）", _w(tp["retention"]), _n(tp["new_per_step"], 1) + "銘柄"], ["従来の「今強い」", _w(tl["retention"]), _n(tl["new_per_step"], 1) + "銘柄"]],
              "note": f"{v['step_days']}営業日おきの比較。入れ替わりが多いほど、売買コストと約定のずれが成績を削ります。"})
    e = v["entry"]
    T.append({"id": "entry", "title": "⑥ 買う日の比較", "cols": ["", f"平均騰落率（{label}・同じ日に売却）"],
              "rows": [["翌営業日の始値で買う", _p(e["open"])], ["翌営業日の終値で買う", _p(e["close"])], ["差（始値−終値）", _p(e["diff"]) + f"（t={_n(e['t'])}）"]],
              "note": "終値で買う場合は、その日の値動きを見てから入れる代わりに、保有日数が1日短くなります。"})
    s = v["stop"]
    if s:
        T.append({"id": "stop", "title": "⑦ 撤退ライン（移動平均線−ATR）の効果", "cols": ["", "なし", "あり"],
                  "rows": [["平均騰落率", _p(s["no_stop"]), _p(s["with_stop"])], ["勝率", _w(s["win_no"]), _w(s["win_st"])],
                           ["最悪の1件", _p(s["worst_no"]), _p(s["worst_st"])], ["下位5%の損失", _p(s["p5_no"]), _p(s["p5_st"])],
                           ["撤退ラインに達した割合", "—", _w(s["hit"])]],
                  "note": f"終値が撤退ラインを割った日の終値で売る想定。平均の差 {_p(s['diff'])}（t={_n(s['t'])}）。損失の大きさ（最悪・下位5%）がどれだけ抑えられるかが主な効果です。"})
    T.append({"id": "regime", "title": "⑧ 地合い別の成績", "cols": ["地合い", "期間数", f"平均騰落率（{label}）", "全体平均との差", "上回った割合"],
              "rows": [[REGIME_LABEL[r["regime"]], str(r["n"]), _p(r["top"]), _p(r["excess"]), _w(r["hit"])] for r in v["regime"]],
              "note": "特定の地合いだけで成績が出ていないかを確認します。逆風時に成績が悪いなら、逆風の日は新規購入を見送る根拠になります。"})
    T.append({"id": "years", "title": "⑨ 年別の成績", "cols": ["年", "期間数", "全体平均との差", "平均相関"],
              "rows": [[str(r["year"]), str(r["n"]), _p(r["excess"]), _n(r["ic"], 3)] for r in v["years"]],
              "note": "毎年プラスで安定しているかを見ます。1〜2年だけ大きく勝っていて他の年が負けているなら、偶然の可能性があります。"})
    o = v["oos"]
    rows = []
    if o.get("is_top"):
        rows.append([f"それ以前（設計時に近い期間）", _p(o["is_top"]["excess"]), _n(o["is_top"]["t"]), _n((o.get("is_ic") or {}).get("mean"), 3)])
    if o.get("top"):
        rows.append([f"直近{o['years']}年（検証期間）", _p(o["top"]["excess"]), _n(o["top"]["t"]), _n((o.get("ic") or {}).get("mean"), 3)])
    if rows:
        T.append({"id": "oos", "title": "⑩ 直近の期間でも成立するか", "cols": ["期間", "全体平均との差", "t値", "平均相関"], "rows": rows,
                  "note": "配点や形は過去のデータを見て決めたものではありませんが、直近の期間でも成績が保たれているかを別に確認します。"})
    sc = v["sector"]
    T.append({"id": "sector", "title": "⑪ 業種の偏り", "cols": ["", "値"],
              "rows": [[f"上位{N}銘柄の業種数（平均）", _n(sc["distinct"], 1)], ["最も多い業種の割合（平均）", _w(sc["top_share"])]],
              "note": "割合が高いほど、同じ業種に偏ったランキングになりやすいことを意味します。"})
    return T


def empty_verdict() -> dict:
    return {"ok": False}
