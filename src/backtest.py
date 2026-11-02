"""バックテスト。

2種類あります。
  1. 過去再現（replay）  … 毎週「その日のトップ10」を過去N年ぶん（config.ymlのreplay_years）作り直し、買っていたらどうなったかを計算（週1回・自動）
  2. 実運用記録（forward）… このサイトが実際に毎朝出したトップ10を data/history に保存し、後日の騰落率を計算（毎朝・自動）

買い方の想定：ランキングが出た翌営業日の始値で買い、N営業日後の終値で売る。往復コストを差し引く。
実行：  python -m src.backtest          （過去再現を実行して data/backtest_summary.json を更新）
"""
import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import ROOT, JST, load_config
from .features import compute_features, tech_points, tech_max

HORIZON_LABEL = {"1w": "1週間後", "1m": "1か月後", "3m": "3か月後"}

CAVEATS = [
    "過去再現は「現在の上場銘柄」で計算しているため、上場廃止になった銘柄が含まれず、成績がやや良く出ます（生存者バイアス）。",
    "過去再現には業績・時価総額の点数が含まれません（過去時点の業績データを無料で入手できないため）。テクニカル点だけで計算しています。",
    "ランキングの翌営業日の始値で買い、往復の売買コストを差し引いた計算です。ストップ高で買えない・約定が滑るといった実際のズレは含みません。",
    "過去の成績は将来の利益を保証しません。",
]


def paths(demo=False):
    base = ROOT / "data" / "demo" if demo else ROOT / "data"
    return base / "history", base / "backtest_summary.json"


# ---------------------------------------------------------------- 共通部品
def _fwd(close, open_, horizons, cost):
    entry = open_.shift(-1).where(lambda x: x > 0)
    return {k: close.shift(-h) / entry - 1 - cost for k, h in horizons.items()}


def _new_acc(horizons):
    return {k: {"dates": [], "bench": [], "all": []} for k in horizons}


def _add(acc_h, rets, bench_val):
    rets = rets.dropna()
    if rets.empty or bench_val is None or not np.isfinite(bench_val):
        return
    acc_h["dates"].append(float(rets.mean()))
    acc_h["bench"].append(float(bench_val))
    acc_h["all"].extend(rets.astype(float).tolist())


def _stats(acc):
    out = {}
    for k, a in acc.items():
        if not a["dates"]:
            out[k] = None
            continue
        avg, bench = float(np.mean(a["dates"])), float(np.mean(a["bench"]))
        allr = np.array(a["all"])
        out[k] = {"avg": avg, "median": float(np.median(allr)), "win": float((allr > 0).mean()),
                  "bench": bench, "excess": avg - bench, "n": len(a["dates"]), "n_picks": int(len(allr))}
    return out


def _bench_fwd(bench, index, horizons, cost):
    b = bench.reindex(index)
    return _fwd(b["close"], b["open"], horizons, cost), b["close"].ffill()


# ---------------------------------------------------------------- 1. 過去再現
def run_replay(panel, bench_topix, cfg):
    bt, sc, flt = cfg["backtest"], cfg["scoring"], cfg["filters"]
    horizons, cost, top_n = bt["horizons"], bt["cost_pct"] / 100, cfg["top_n"]

    f = compute_features(panel)
    pts = tech_points(f, sc)
    scaled = pts["tech_raw"] / tech_max(sc) * 100
    c = panel["close"]
    eligible = (c >= flt["min_price"]) & (f["turnover20"] >= flt["min_turnover_yen"]) \
        & c.notna() & (panel["volume"] > 0)

    fwd = _fwd(c, panel["open"], horizons, cost)
    bfwd, bclose = _bench_fwd(bench_topix, c.index, horizons, cost)
    regime_on = bclose > bclose.rolling(75).mean()

    dates = c.index
    start = dates[-1] - pd.DateOffset(years=bt["replay_years"])
    idxs = [i for i in range(80, len(dates) - 1) if dates[i] >= start]
    sample = idxs[::bt["step_days"]]

    acc = {name: _new_acc(horizons) for name in ("top10", "top10_regime", "tech80", "tech60")}
    for i in sample:
        row = scaled.iloc[i].where(eligible.iloc[i]).dropna()
        row = row[row > 0]
        if row.empty:
            continue
        vr = f["vol_ratio"].iloc[i].reindex(row.index)
        order = pd.DataFrame({"s": row, "vr": vr}).sort_values(["s", "vr"], ascending=False)
        top = order.head(top_n).index
        for k in horizons:
            b = bfwd[k].iloc[i]
            r = fwd[k].iloc[i]
            _add(acc["top10"][k], r.reindex(top), b)
            if bool(regime_on.iloc[i]):
                _add(acc["top10_regime"][k], r.reindex(top), b)
            for name, th in (("tech80", 80), ("tech60", 60)):
                sel = row[row >= th].index
                _add(acc[name][k], r.reindex(sel), b)

    return {
        "generated": datetime.now(JST).isoformat(timespec="minutes"),
        "years": bt["replay_years"],
        "period": {"start": str(dates[sample[0]].date()) if sample else None,
                   "end": str(dates[sample[-1]].date()) if sample else None},
        "n_universe": int(c.shape[1]),
        "n_dates": len(sample),
        "cost_pct": bt["cost_pct"],
        **{name: _stats(a) for name, a in acc.items()},
    }


# ---------------------------------------------------------------- 2. 実運用記録
def forward_stats(panel, bench_topix, cfg, hist_dir):
    bt, top_n = cfg["backtest"], cfg["top_n"]
    horizons, cost = bt["horizons"], bt["cost_pct"] / 100
    c = panel["close"]
    fwd = _fwd(c, panel["open"], horizons, cost)
    bfwd, _ = _bench_fwd(bench_topix, c.index, horizons, cost)
    acc = _new_acc(horizons)
    n_records = 0
    for fp in sorted(hist_dir.glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        d = pd.Timestamp(rec["date"])
        if d not in c.index:
            continue
        n_records += 1
        i = c.index.get_loc(d)
        tickers = [p["code"] + ".T" for p in rec["picks"] if p["rank"] <= top_n]
        for k, h in horizons.items():
            if i + h >= len(c):
                continue
            _add(acc[k], fwd[k].iloc[i].reindex(tickers), bfwd[k].iloc[i])
    return {"n_records": n_records, "stats": _stats(acc)}


# ---------------------------------------------------------------- 画面用の表を作る
def _cell(s, show_n=False):
    if not s:
        return {"txt": "—", "sub": "", "cls": "na"}
    sub = f"勝率{s['win'] * 100:.0f}%"
    if show_n:
        sub += f"・{s['n']}回"
    return {"txt": f"{s['avg'] * 100:+.1f}%", "sub": sub, "cls": "pos" if s["avg"] > 0 else "neg"}


def build_table(summary, forward, cfg):
    """画面に出す検証結果。データが無ければ None。"""
    hs = list(cfg["backtest"]["horizons"].keys())
    if not summary and not (forward and forward["n_records"]):
        return None
    t = {"horizons": [HORIZON_LABEL.get(h, h) for h in hs], "rows": [], "headline": None,
         "caveats": CAVEATS, "period": None, "forward_note": None}

    if summary:
        t["period"] = f"{summary['period']['start']} 〜 {summary['period']['end']}（{summary['n_dates']}週分）"
        for key, label, note in (
            ("top10", "毎週のトップ10を買った場合", "過去再現"),
            ("top10_regime", "　うち地合いが良い週だけ", "TOPIXが75日線の上"),
            ("tech80", "テクニカル80点以上の全銘柄", "過去再現"),
        ):
            t["rows"].append({"label": label, "note": note,
                              "cells": [_cell(summary.get(key, {}).get(h)) for h in hs]})
        bench = summary["top10"].get("1m") if summary.get("top10") else None
        bench_cells = []
        for h in hs:
            s = (summary.get("top10") or {}).get(h)
            bench_cells.append({"txt": f"{s['bench'] * 100:+.1f}%", "sub": "",
                                "cls": "pos" if s["bench"] > 0 else "neg"} if s else _cell(None))
        t["rows"].append({"label": "TOPIX（比較用）", "note": "同じ期間・同じ売買ルール",
                          "cells": bench_cells})
        s80 = (summary.get("tech80") or {}).get("1m")
        if s80:
            t["headline"] = (f"過去{summary['years']}年の再現検証では、テクニカル80点以上の銘柄を買った場合の"
                             f"1か月後の平均騰落率は{s80['avg'] * 100:+.1f}%"
                             f"（同期間のTOPIXは{s80['bench'] * 100:+.1f}%）でした。")
        elif bench:
            t["headline"] = (f"過去{summary['years']}年の再現検証では、毎週のトップ10を買った場合の"
                             f"1か月後の平均騰落率は{bench['avg'] * 100:+.1f}%でした。")

    if forward is not None:
        n = forward["n_records"]
        ready = n >= cfg["backtest"]["min_forward_samples"]
        if forward["stats"] and any(forward["stats"].values()):
            t["rows"].append({"label": "このサイトの実績（毎朝のトップ10）",
                              "note": f"{n}日分の記録" + ("" if ready else "・参考値"),
                              "cells": [_cell(forward["stats"].get(h), show_n=True) for h in hs]})
        else:
            t["forward_note"] = f"実績の記録を蓄積中です（現在{n}日分）。1週間後の結果は記録の翌週から表示されます。"
    return t


# ---------------------------------------------------------------- 実行
def main():
    from .universe import load_universe
    from .prices import download_prices, download_benchmarks

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認")
    args = ap.parse_args()
    cfg = load_config()
    _, summary_path = paths(args.demo)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from .demo import make_demo
        _, panel, bench, _ = make_demo()
    else:
        uni = load_universe(cfg)
        d = cfg["data"]
        # 検証期間 + 約半年ぶん（75日線などの助走期間）をさかのぼって取得
        start = (datetime.now(JST) - timedelta(days=int(cfg["backtest"]["replay_years"] * 365 + 200))).strftime("%Y-%m-%d")
        print(f"株価を{start}以降で取得します（時間がかかります）")
        panel = download_prices(list(uni["ticker"]), chunk_size=d["chunk_size"], sleep=d["sleep_between_chunks"],
                                min_success_ratio=d["min_success_ratio"], start=start)
        bench = download_benchmarks(start=start)

    summary = run_replay(panel, bench["topix"], cfg)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存しました: {summary_path}")
    for key in ("top10", "top10_regime", "tech80"):
        s = summary[key].get("1m")
        if s:
            print(f"  {key:13s} 1か月後 平均{s['avg'] * 100:+.2f}%  勝率{s['win'] * 100:.0f}%  "
                  f"TOPIX{s['bench'] * 100:+.2f}%  回数{s['n']}")


if __name__ == "__main__":
    main()
