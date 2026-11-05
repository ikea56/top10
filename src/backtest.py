"""バックテスト。

  1. 過去再現（replay）  … 毎週「その日のトップ10」を過去に作り直し、買っていた場合の成績を計算（週1回・自動）
  2. 実運用記録（forward）… このサイトが実際に出したトップ10を data/history に保存し、後日の騰落率を計算（毎朝・自動）

買い方の想定：ランキングが出た翌営業日の始値で買い、N営業日後の終値で売る。往復コストを差し引く。
過去再現の対象は「トレンド・需給・地合い」の点数（業績・イベントの過去データは無料で入手できないため含まない）。
実行：  python -m src.backtest   （--demo で架空データ）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import pullback, scoring
from .config import JST, ROOT, Config, load_config
from .features import Panel, compute_features, snapshot
from .log import get_logger
from .market import above_ma50_series, regime_series

log = get_logger(__name__)

HORIZON_LABEL = {"1w": "1週間後", "1m": "1か月後", "3m": "3か月後"}
MIN_HISTORY = 210          # 252日高値・75日線の計算に必要な助走日数
GROUPS = ("pb_top10", "pb_ok", "top10", "top10_ok", "tech80", "hi252")
CAVEATS = [
    "過去再現は「現在の上場銘柄」で計算しているため、上場廃止になった銘柄が含まれず成績がやや良く出ます（生存者バイアス）。",
    "過去再現に使うのは株価・出来高だけで決まる点数です（押し目スコアは業績維持20点を除く80点分、従来スコアはトレンド・需給・地合い）。業績・増配・TDnetの開示は過去データが無いため含みません。",
    "ランキングの翌営業日の始値で買い、往復の売買コストを差し引いた計算です。ストップ高で買えない・約定が滑るといった実際のズレは含みません。",
    "期間が重なる計算のため、1か月後などの結果は独立した試行ではありません。数字は目安として見てください。",
    "過去の成績は将来の利益を保証しません。",
]


def paths(demo: bool = False) -> tuple[Path, Path]:
    base = ROOT / "data" / "demo" if demo else ROOT / "data"
    return base / "history", base / "backtest_summary.json"


# ---------------------------------------------------------------- 指標の計算（純粋関数）
def trade_metrics(r: np.ndarray) -> dict[str, float | int | None] | None:
    """1回ごとの売買結果（騰落率の配列）から 勝率・平均利益率・平均損失率・PF を出す。"""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = abs(float(losses.sum()))
    return {"n_trades": int(r.size), "win": float((r > 0).mean()),
            "avg_win": float(wins.mean()) if wins.size else None,
            "avg_loss": float(losses.mean()) if losses.size else None,
            "pf": float(wins.sum()) / gross_loss if gross_loss > 0 else None}


def equity_metrics(period_returns: np.ndarray, hold_days: int) -> dict[str, float | None] | None:
    """重ならない保有期間ごとの平均リターン列から 年率リターン・シャープレシオ・最大ドローダウン を出す。"""
    r = period_returns[np.isfinite(period_returns)]
    if r.size < 3:
        return None
    eq = np.r_[1.0, np.cumprod(1.0 + r)]
    years = r.size * hold_days / 252
    std = float(np.std(r, ddof=1))
    return {"total_ret": float(eq[-1] - 1),
            "ann_ret": float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else None,
            "sharpe": float(np.mean(r) / std * np.sqrt(252 / hold_days)) if std > 0 else None,
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min())}


def _mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def period_stats(records: list[dict], years: int, end: pd.Timestamp, bt: dict,
                 group: str = "top10") -> dict[str, Any] | None:
    """直近 years 年ぶんの成績（トップ10を毎週買った場合）。期間に満たなければ None。"""
    if not records:
        return None
    start = end - pd.DateOffset(years=years)
    if records[0]["date"] > start + pd.Timedelta(days=60):
        return None
    win = [r for r in records if r["date"] > start]
    hold = int(bt["horizons"][bt["equity_horizon"]])
    k = max(1, hold // int(bt["step_days"]))
    h = bt["equity_horizon"]
    trades = np.concatenate([r["groups"][group][h] for r in win]) if win else np.array([])
    sub = win[::k]                                              # 保有期間が重ならないように間引く
    strat = np.array([_mean(r["groups"][group][h]) for r in sub])
    bench = np.array([r["bench"][h] for r in sub], dtype=float)
    out: dict[str, Any] = {"n_dates": len(win), "n_periods": len(sub)}
    out.update(trade_metrics(trades) or {})
    out.update(equity_metrics(strat, hold) or {})
    if np.isfinite(bench).any() and np.nanmin(bench) < -0.4:
        log.warning("TOPIX比較に1期間で-40%超の値があります。指数データ（分割の調整漏れなど）を確認してください")
    be = equity_metrics(bench, hold) or {}
    out.update({"bench_ann": be.get("ann_ret"), "bench_sharpe": be.get("sharpe"), "bench_max_dd": be.get("max_dd")})
    return out


def group_stats(records: list[dict], group: str, horizons: dict[str, int]) -> dict[str, Any]:
    """画面の「1週間／1か月／3か月後」表用。"""
    out: dict[str, Any] = {}
    for h in horizons:
        means, bench, allr = [], [], []
        for r in records:
            if group.endswith("_ok") and r["regime"] == "bad":
                continue
            a = r["groups"][{"top10_ok": "top10", "pb_ok": "pb_top10"}.get(group, group)][h]
            a = a[np.isfinite(a)]
            if a.size and np.isfinite(r["bench"][h]):
                means.append(float(a.mean()))
                bench.append(r["bench"][h])
                allr.extend(a.tolist())
        if not means:
            out[h] = None
            continue
        arr = np.array(allr)
        avg, b = float(np.mean(means)), float(np.mean(bench))
        out[h] = {"avg": avg, "median": float(np.median(arr)), "win": float((arr > 0).mean()), "bench": b,
                  "excess": avg - b, "n": len(means), "n_picks": int(arr.size)}
    return out


def band_stats(records: list[dict], es: dict, end: pd.Timestamp) -> dict[str, Any]:
    """テクニカル点（100点換算）のスコア帯ごとの勝率（期待値点の「過去勝率」に使う）。"""
    start = end - pd.DateOffset(years=int(es["winrate_years"]))
    sel = [r for r in records if r["date"] > start]
    x = np.concatenate([r["band"][0] for r in sel]) if sel else np.array([])
    y = np.concatenate([r["band"][1] for r in sel]) if sel else np.array([])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = es["bands"]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "win": float((y[m] > 0).mean()) if m.any() else None,
                     "avg": float(y[m].mean()) if m.any() else None})
    return {"horizon": es["winrate_horizon"], "years": int(es["winrate_years"]), "rows": rows}


def win_rate_lookup(summary: dict | None, scaled: float, es: dict) -> tuple[float, int, bool]:
    """(勝率, サンプル数, 実データか)。バックテスト未実行・サンプル不足なら仮置き勝率を返す。"""
    default = float(es["default_win_rate"])
    if not summary or not summary.get("bands"):
        return default, 0, False
    for row in summary["bands"]["rows"]:
        if row["lo"] <= scaled < row["hi"]:
            if row["n"] >= es["min_samples"] and row["win"] is not None:
                return float(row["win"]), int(row["n"]), True
            break
    return default, 0, False


# ---------------------------------------------------------------- 1. 過去再現
def run_replay(panel: Panel, bench: dict[str, pd.DataFrame], sectors: pd.Series, cfg: Config) -> dict[str, Any]:
    bt, sc, flt = cfg["backtest"], cfg["scoring"], cfg["filters"]
    horizons: dict[str, int] = bt["horizons"]
    cost, top_n, tmax = bt["cost_pct"] / 100, cfg["top_n"], scoring.tech_max(cfg)
    c, o = panel["close"], panel["open"]
    topix = bench["topix"].reindex(c.index).ffill()

    log.info("指標を計算中（%d銘柄 × %d日）", c.shape[1], c.shape[0])
    f = compute_features(panel, topix["close"], sc)
    regime = regime_series(topix["close"]).fillna("neutral")
    below200 = (topix["close"] < topix["close"].rolling(200).mean()).fillna(False)
    n225 = bench.get("n225")
    nikkei = (above_ma50_series(n225["close"].reindex(c.index).ffill()) if n225 is not None
              else pd.Series(False, index=c.index))

    first = max(MIN_HISTORY, int(c.index.searchsorted(c.index[-1] - pd.DateOffset(years=bt["replay_years"]))))
    idx = list(range(first, len(c) - 1, int(bt["step_days"])))
    entry = o.shift(-1).where(lambda x: x > 0)
    fwd = {k: (c.shift(-h) / entry - 1 - cost).iloc[idx] for k, h in horizons.items()}
    del entry
    bentry = topix["open"].shift(-1)
    bfwd = {k: (topix["close"].shift(-h) / bentry - 1 - cost).iloc[idx] for k, h in horizons.items()}

    records: list[dict] = []
    for pos, i in enumerate(idx):
        snap = snapshot(f, i)
        s = snap[scoring.eligible_mask(snap, flt)]
        if s.empty:
            continue
        tech = scoring.score_technical(s, regime.iloc[i], scoring.sector_points(s["ret20d"], sectors, sc), sc)
        scaled = tech["tech_base"] / tmax * 100
        pb = pullback.score_technical(s, regime.iloc[i], bool(nikkei.iloc[i]), cfg, bool(below200.iloc[i]))["pb_tech_scaled"]
        pb_order = pd.DataFrame({"s": pb, "vr": s["vol_cont"]}).query("s > 0").sort_values(["s", "vr"], ascending=[False, True])
        order = pd.DataFrame({"s": scaled, "vr": s["vol_ratio"]}).sort_values(["s", "vr"], ascending=False)
        pick = {"pb_top10": pb_order.index[:top_n], "top10": order.index[:top_n],
                "tech80": scaled.index[scaled >= 80],
                "hi252": s.index[s["hi252"].astype(bool)]}
        groups = {g: {k: fwd[k].iloc[pos].reindex(ix).to_numpy(dtype=float) for k in horizons} for g, ix in pick.items()}
        records.append({"date": c.index[i], "regime": regime.iloc[i],
                        "bench": {k: float(bfwd[k].iloc[pos]) for k in horizons},
                        "groups": groups,
                        "band": (scaled.to_numpy(dtype=float),
                                 fwd[cfg["expected_score"]["winrate_horizon"]].iloc[pos].reindex(scaled.index).to_numpy(dtype=float))})
    if not records:
        raise RuntimeError("過去再現で条件を満たす日が1日もありませんでした（データ期間や除外条件を確認）")

    end = c.index[-1]
    table_recs = [r for r in records if r["date"] > end - pd.DateOffset(years=bt["table_years"])]
    summary: dict[str, Any] = {
        "generated": datetime.now(JST).isoformat(timespec="minutes"),
        "years": bt["table_years"], "replay_years": bt["replay_years"],
        "period": {"start": str(records[0]["date"].date()), "end": str(records[-1]["date"].date())},
        "n_universe": int(c.shape[1]), "n_dates": len(table_recs), "cost_pct": bt["cost_pct"],
        "hold": bt["equity_horizon"],
        "periods": {f"{y}y": period_stats(records, y, end, bt, "pb_top10") for y in bt["periods"]},
        "periods_legacy": {f"{y}y": period_stats(records, y, end, bt, "top10") for y in bt["periods"]},
        "bands": band_stats(records, cfg["expected_score"], end),
    }
    for g in GROUPS:
        summary[g] = group_stats(table_recs, g, horizons)
    return summary


# ---------------------------------------------------------------- 2. 実運用記録
def forward_stats(panel: Panel, bench_topix: pd.DataFrame, cfg: Config, hist_dir: Path) -> dict[str, Any]:
    bt, top_n = cfg["backtest"], cfg["top_n"]
    horizons, cost = bt["horizons"], bt["cost_pct"] / 100
    c = panel["close"]
    tb = bench_topix.reindex(c.index).ffill()
    entry = panel["open"].shift(-1).where(lambda x: x > 0)
    acc: dict[str, dict[str, list]] = {k: {"means": [], "bench": [], "all": []} for k in horizons}
    n_records = 0
    for fp in sorted(hist_dir.glob("*.json")):
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
            d = pd.Timestamp(rec["date"])
        except (OSError, ValueError, KeyError) as e:
            log.warning("履歴ファイルを読めません（%s）: %s", fp.name, e)
            continue
        if d not in c.index:
            continue
        n_records += 1
        i = c.index.get_loc(d)
        tickers = [p["code"] + ".T" for p in rec["picks"] if p["rank"] <= top_n and p["code"] + ".T" in c.columns]
        for k, h in horizons.items():
            if i + h >= len(c) or not tickers:
                continue
            r = (c.iloc[i + h][tickers] / entry.iloc[i][tickers] - 1 - cost).dropna()
            b = tb["close"].iloc[i + h] / tb["open"].iloc[i + 1] - 1 - cost
            if len(r) and np.isfinite(b):
                acc[k]["means"].append(float(r.mean()))
                acc[k]["bench"].append(float(b))
                acc[k]["all"].extend(r.tolist())
    stats: dict[str, Any] = {}
    for k, a in acc.items():
        if not a["means"]:
            stats[k] = None
            continue
        arr, avg, b = np.array(a["all"]), float(np.mean(a["means"])), float(np.mean(a["bench"]))
        stats[k] = {"avg": avg, "median": float(np.median(arr)), "win": float((arr > 0).mean()),
                    "bench": b, "excess": avg - b, "n": len(a["means"]), "n_picks": int(arr.size)}
    return {"n_records": n_records, "stats": stats}


# ---------------------------------------------------------------- 画面用の表
def _cell(s: dict | None, show_n: bool = False) -> dict[str, str]:
    if not s:
        return {"txt": "—", "sub": "", "cls": "na"}
    sub = f"勝率{s['win'] * 100:.0f}%" + (f"・{s['n']}回" if show_n else "")
    return {"txt": f"{s['avg'] * 100:+.1f}%", "sub": sub, "cls": "pos" if s["avg"] > 0 else "neg"}


def build_table(summary: dict | None, forward: dict | None, cfg: Config) -> dict[str, Any] | None:
    """画面の「1週間／1か月／3か月後」表。"""
    hs = list(cfg["backtest"]["horizons"])
    if not summary and not (forward and forward["n_records"]):
        return None
    t: dict[str, Any] = {"horizons": [HORIZON_LABEL.get(h, h) for h in hs], "rows": [], "headline": None,
                         "caveats": CAVEATS, "period": None, "forward_note": None}
    if summary:
        t["period"] = f"{summary['period']['start']} 〜 {summary['period']['end']}（直近{summary['years']}年・{summary['n_dates']}週分）"
        for key, label, note in (("pb_top10", "押し目トップ10（メイン）", "押し目スコアの技術80点分で再現"),
                                 ("pb_ok", "　うち地合いが「逆風」でない週", "TOPIXが75日線の上"),
                                 ("top10", "従来の「今強い」トップ10（補助）", "過去再現"),
                                 ("tech80", "テクニカル80点以上の全銘柄", "過去再現"),
                                 ("hi252", "252日高値を更新した全銘柄", "過去再現（モメンタム）")):
            if summary.get(key):
                t["rows"].append({"label": label, "note": note, "cells": [_cell(summary[key].get(h)) for h in hs]})
        bench_cells = []
        for h in hs:
            s = (summary.get("top10") or {}).get(h)
            bench_cells.append({"txt": f"{s['bench'] * 100:+.1f}%", "sub": "", "cls": "pos" if s["bench"] > 0 else "neg"}
                               if s else _cell(None))
        t["rows"].append({"label": "TOPIX（比較用）", "note": "同じ期間・同じ売買ルール", "cells": bench_cells})
        s1, s0 = (summary.get("pb_top10") or {}).get("1m"), (summary.get("top10") or {}).get("1m")
        if s1:
            t["headline"] = (f"過去{summary['years']}年の再現検証では、毎週の押し目トップ10を買った場合の1か月後の平均騰落率は"
                             f"{s1['avg'] * 100:+.1f}%（同期間のTOPIXは{s1['bench'] * 100:+.1f}%"
                             + (f"、従来の「今強い」トップ10は{s0['avg'] * 100:+.1f}%" if s0 else "") + "）でした。")
    if forward is not None:
        n = forward["n_records"]
        if any(forward["stats"].values()):
            ready = n >= cfg["backtest"]["min_forward_samples"]
            t["rows"].append({"label": "このサイトの実績（毎朝のトップ10）",
                              "note": f"{n}日分の記録" + ("" if ready else "・参考値"),
                              "cells": [_cell(forward["stats"].get(h), show_n=True) for h in hs]})
        else:
            t["forward_note"] = f"実績の記録を蓄積中です（現在{n}日分）。1週間後の結果は記録の翌週から表示されます。"
    return t


def _fmt(v: float | None, kind: str) -> str:
    if v is None or v != v:
        return "—"
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "spct":
        return f"{v * 100:+.1f}%"
    return f"{v:.2f}"


def build_period_table(summary: dict | None, key: str = "periods") -> dict[str, Any] | None:
    """画面の「期間別の成績（1年・3年・5年・10年）」表。"""
    if not summary or not summary.get(key):
        return None
    periods = summary[key]
    keys = list(periods)
    spec = [("勝率", "win", "pct"), ("平均利益率", "avg_win", "spct"), ("平均損失率", "avg_loss", "spct"),
            ("PF（総利益÷総損失）", "pf", "num"), ("最大ドローダウン", "max_dd", "spct"),
            ("シャープレシオ", "sharpe", "num"), ("年率リターン", "ann_ret", "spct"),
            ("（比較）TOPIX年率", "bench_ann", "spct")]
    rows = [{"label": label, "cells": [_fmt((periods[k] or {}).get(kcol), kind) for k in keys]}
            for label, kcol, kind in spec]
    return {"cols": [k.replace("y", "年") for k in keys], "rows": rows,
            "note": f"毎週のトップ10を翌営業日の始値で買い、約1か月（{summary.get('hold', '1m')}）保有→入れ替え、を繰り返した場合。"
                    "「—」は取得した株価の期間が足りない項目です。"}


# ---------------------------------------------------------------- 実行
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認")
    args = ap.parse_args()
    cfg = load_config()
    scoring.validate_config(cfg)
    pullback.validate_config(cfg)
    _, summary_path = paths(args.demo)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from .demo import make_demo
        uni, panel, bench, _ = make_demo()
    else:
        from .prices import download_benchmarks, download_prices
        from .universe import load_universe
        uni = load_universe(cfg)
        d = cfg["data"]
        start = (datetime.now(JST) - timedelta(days=int(cfg["backtest"]["replay_years"] * 365 + 400))).strftime("%Y-%m-%d")
        log.info("株価を%s以降で取得します（時間がかかります）", start)
        panel = download_prices(list(uni["ticker"]), chunk_size=d["chunk_size"], sleep=d["sleep_between_chunks"],
                                min_success_ratio=d["min_success_ratio"], start=start)
        bench = download_benchmarks(start=start)

    summary = run_replay(panel, bench, uni.set_index("ticker")["sector"], cfg)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("保存しました: %s", summary_path)
    for k, p in summary["periods"].items():           # 押し目トップ10（メイン）
        if p:
            log.info("  %s: 勝率%.0f%% 平均利益%s 平均損失%s PF%s DD%s シャープ%s 年率%s", k, p["win"] * 100,
                     _fmt(p.get("avg_win"), "spct"), _fmt(p.get("avg_loss"), "spct"), _fmt(p.get("pf"), "num"),
                     _fmt(p.get("max_dd"), "spct"), _fmt(p.get("sharpe"), "num"), _fmt(p.get("ann_ret"), "spct"))
        else:
            log.info("  %s: データ不足", k)


if __name__ == "__main__":
    main()
