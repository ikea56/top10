"""毎朝の処理本体。  python -m src.main        （デモ:  python -m src.main --demo）"""
import argparse
import json
from datetime import datetime

import pandas as pd

from . import backtest, charts, explain, features, rankings, scoring
from .config import JST, ROOT, load_config, load_exclude_codes, load_themes
from .market import build_market
from .render import render_site

WEEKDAY = "月火水木金土日"


def trim_to_valid_last(panel):
    """最終行が読み込み途中（銘柄の半分以上が欠損）なら切り捨てる。"""
    counts = panel["close"].notna().sum(axis=1)
    valid = counts[counts >= 0.5 * counts.max()]
    last = valid.index[-1]
    return {k: v.loc[:last] for k, v in panel.items()}


def build_snapshot(f, pts, panel, universe, cfg, exclude_codes):
    """最新日の全銘柄一覧（除外条件を適用済み）。"""
    last = lambda df: df.iloc[-1]  # noqa: E731
    snap = pd.DataFrame({
        "close": last(f["close"]), "volume": last(panel["volume"]),
        "ret1d": last(f["ret1d"]), "ret20d": last(f["ret20d"]),
        "vol_ratio": last(f["vol_ratio"]), "turnover20": last(f["turnover20"]),
        "ma25": last(f["ma25"]), "ma75": last(f["ma75"]), "gap25": last(f["gap25"]),
        "hi60": last(f["hi60"]), "hi30": last(f["hi30"]), "hi52w": last(f["hi52w"]),
        "trend": last(f["trend"]), "rsi": last(f["rsi"]),
        "macd": last(f["macd"]), "macd_sig": last(f["macd_sig"]),
        "pts_high": last(pts["pts_high"]), "pts_vol": last(pts["pts_vol"]),
        "pts_trend": last(pts["pts_trend"]), "tech_raw": last(pts["tech_raw"]),
    })
    snap = snap.join(universe.set_index("ticker")[["code", "name", "market", "sector"]], how="inner")
    snap["tech_scaled"] = snap["tech_raw"] / features.tech_max(cfg["scoring"]) * 100

    flt = cfg["filters"]
    ok = (snap["close"].notna() & (snap["volume"] > 0) & (snap["close"] >= flt["min_price"])
          & (snap["turnover20"] >= flt["min_turnover_yen"]) & ~snap["code"].isin(exclude_codes))
    for kw in flt["exclude_name_keywords"]:
        ok &= ~snap["name"].str.contains(kw, na=False)
    return snap[ok]


def select_top(snap, fund_fn, cfg):
    """テクニカル上位から順に業績を取得し、「これ以上は上位10位に入れない」と確定した時点で打ち切る。"""
    sc, fc, top_n = cfg["scoring"], cfg["fundamentals"], cfg["top_n"]
    max_extra = scoring.max_earn_mcap(sc)
    cands = snap.sort_values(["tech_raw", "vol_ratio"], ascending=False)
    finals, fetched, excluded = [], 0, 0
    for tk, r in cands.iterrows():
        if fetched >= fc["max_fetch"]:
            break
        if fetched >= fc["min_fetch"] and len(finals) >= top_n:
            tenth = sorted((x["raw_total"] for x in finals), reverse=True)[top_n - 1]
            if r["tech_raw"] + max_extra < tenth:
                break
        fund = fund_fn(r["code"])
        fetched += 1
        loss = fund.get("op_ttm") is not None and fund["op_ttm"] <= 0
        missing = fund.get("op_ttm") is None
        if (fc["exclude_loss_making"] and loss) or (fc["exclude_if_missing"] and missing):
            excluded += 1
            continue
        pe, pm = scoring.earn_points(fund, sc), scoring.mcap_points(fund, sc)
        raw_total = float(r["tech_raw"] + pe + pm)
        finals.append({"ticker": tk, "row": r, "fund": fund, "pe": pe, "pm": pm,
                       "raw_total": raw_total, "score": scoring.to_score(raw_total, sc)})
    finals.sort(key=lambda x: (x["score"], x["row"]["vol_ratio"] if x["row"]["vol_ratio"] == x["row"]["vol_ratio"] else 0),
                reverse=True)
    return finals[:top_n], fetched, excluded


def make_cards(top, f, panel, themes, regime, cfg):
    sc = cfg["scoring"]
    max_vol = max(p for _, p in sc["vol_ratio"])
    max_earn = max(max(p for _, p in sc["op_growth"]), sc["turnaround_points"])
    max_mcap = max(p for _, p in sc["mcap"])
    cards = []
    for rank, x in enumerate(top, 1):
        r, fund, tk = x["row"], x["fund"], x["ticker"]
        code = r["code"]
        overheat = explain.is_overheat(r, cfg)
        label, css, note = scoring.bull_label(x["score"], overheat, regime, cfg)

        n = 60
        sl = slice(-n, None)
        svg = charts.candle_svg(
            panel["open"][tk].iloc[sl].to_numpy(), panel["high"][tk].iloc[sl].to_numpy(),
            panel["low"][tk].iloc[sl].to_numpy(), panel["close"][tk].iloc[sl].to_numpy(),
            panel["volume"][tk].iloc[sl].to_numpy(),
            f["ma25"][tk].iloc[sl].to_numpy(), f["ma75"][tk].iloc[sl].to_numpy())

        stop = None
        if r["ma25"] == r["ma25"] and r["ma25"] < r["close"]:
            stop = {"price": float(r["ma25"]), "pct": float(r["ma25"] / r["close"] - 1) * 100}

        cards.append({
            "rank": rank, "code": code, "name": r["name"], "market": r["market"], "sector": r["sector"],
            "themes": rankings.themes_of(code, themes),
            "close": float(r["close"]), "ret1d": None if r["ret1d"] != r["ret1d"] else float(r["ret1d"]) * 100,
            "score": x["score"], "label": label, "css": css, "label_note": note,
            "breakdown": [
                {"name": "高値更新", "pts": int(r["pts_high"]), "max": sc["high60"]},
                {"name": "出来高", "pts": int(r["pts_vol"]), "max": max_vol},
                {"name": "トレンド", "pts": int(r["pts_trend"]), "max": sc["trend"]},
                {"name": "業績", "pts": int(x["pe"]), "max": max_earn},
                {"name": "規模", "pts": int(x["pm"]), "max": max_mcap},
            ],
            "reasons": explain.build_reasons(r, fund),
            "indicators": explain.build_indicators(r),
            "explain": explain.beginner_text(r, fund),
            "risks": explain.build_risks(r, fund, r["market"], regime),
            "stop": stop, "svg": svg,
            "vol_ratio": None if r["vol_ratio"] != r["vol_ratio"] else float(r["vol_ratio"]),
        })
    return cards


def save_history(hist_dir, data_date, cards, market, demo):
    hist_dir.mkdir(parents=True, exist_ok=True)
    rec = {"date": data_date.strftime("%Y-%m-%d"), "demo": demo, "regime": market["regime"],
           "picks": [{"rank": c["rank"], "code": c["code"], "name": c["name"],
                      "score": c["score"], "close": c["close"]} for c in cards]}
    (hist_dir / f"{rec['date']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認（ネット不要）")
    args = ap.parse_args()
    cfg = load_config()
    themes = load_themes()
    hist_dir, summary_path = backtest.paths(args.demo)

    # ---- データ取得 ----
    store = None
    if args.demo:
        from .demo import make_demo
        universe, panel, bench, fund_fn = make_demo()
        print("[demo] 架空データで実行します")
    else:
        from .universe import load_universe
        from .prices import download_prices, download_benchmarks
        from .fundamentals import FundamentalsStore
        universe = load_universe(cfg)
        print(f"対象銘柄: {len(universe)}")
        d = cfg["data"]
        panel = download_prices(list(universe["ticker"]), period=d["period"], chunk_size=d["chunk_size"],
                                sleep=d["sleep_between_chunks"], min_success_ratio=d["min_success_ratio"])
        bench = download_benchmarks("3y")
        store = FundamentalsStore(cfg)
        fund_fn = store.get

    panel = trim_to_valid_last(panel)
    data_date = panel["close"].index[-1]
    print(f"データ基準日: {data_date.date()}")

    # ---- 指標とスコア ----
    f = features.compute_features(panel)
    pts = features.tech_points(f, cfg["scoring"])
    snap = build_snapshot(f, pts, panel, universe, cfg, load_exclude_codes(cfg))
    print(f"条件を満たす銘柄: {len(snap)} / {len(universe)}")
    if snap.empty:
        raise RuntimeError("条件を満たす銘柄が0件です。データ取得に問題がある可能性があります")

    bench_trim = {k: v.loc[:data_date] for k, v in bench.items()}
    market = build_market(bench_trim, float((snap["close"] > snap["ma25"]).mean()),
                          int(snap["hi60"].sum()), len(snap))

    top, fetched, excluded = select_top(snap, fund_fn, cfg)
    print(f"業績を確認した銘柄: {fetched}（赤字などで除外 {excluded}）")
    if store:
        store.save()
    cards = make_cards(top, f, panel, themes, market["regime"], cfg)

    # ---- 履歴・検証 ----
    save_history(hist_dir, data_date, cards, market, args.demo)
    forward = backtest.forward_stats(panel, bench["topix"], cfg, hist_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    bt_table = backtest.build_table(summary, forward, cfg)

    # ---- 画面生成 ----
    now = datetime.now(JST)
    sc = cfg["scoring"]
    ctx = {
        "title": cfg["site_title"], "demo": args.demo,
        "generated_at": f"{now:%Y年%m月%d日} {now:%H:%M}",
        "generated_iso": now.isoformat(timespec="minutes"),
        "data_date_label": f"{data_date.month}/{data_date.day}（{WEEKDAY[data_date.weekday()]}）",
        "data_date_iso": data_date.strftime("%Y-%m-%d"),
        "market": market, "cards": cards, "bt": bt_table,
        "sectors": rankings.rank_sectors(snap), "themes": rankings.rank_themes(snap, themes),
        "n_universe": len(universe), "n_scanned": len(snap), "n_fetched": fetched,
        "bull": cfg["bull"], "max_raw": scoring.max_raw(sc), "sc": sc, "top_n": cfg["top_n"],
        "min_turnover_oku": cfg["filters"]["min_turnover_yen"] / 1e8,
        "min_price": cfg["filters"]["min_price"],
    }
    out = render_site(ctx)
    print(f"生成しました: {out}")
    for c in cards[:3]:
        print(f"  {c['rank']}位 {c['name']}({c['code']}) {c['score']}点 {c['label']}")


if __name__ == "__main__":
    main()
