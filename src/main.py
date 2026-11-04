"""毎朝の処理本体。  python -m src.main        （デモ:  python -m src.main --demo）

流れ：データ取得 → 指標 → 除外 → 基本点(100) → 候補の業績取得 → イベント点 → 期待値点 → ランキング → HTML
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from . import backtest, charts, events, explain, features, fundamentals, rankings, scoring
from .config import JST, ROOT, Config, ConfigError, load_config, load_exclude_codes, load_themes
from .log import get_logger
from .market import build_market
from .render import render_site

log = get_logger(__name__)
WEEKDAY = "月火水木金土日"
FundFn = Callable[[str], dict]


# ---------------------------------------------------------------- 1. データ取得
def trim_to_valid_last(panel: features.Panel) -> features.Panel:
    """最終行が読み込み途中（銘柄の半分以上が欠損）なら切り捨てる。"""
    counts = panel["close"].notna().sum(axis=1)
    last = counts[counts >= 0.5 * counts.max()].index[-1]
    return {k: v.loc[:last] for k, v in panel.items()}


def load_data(cfg: Config, demo: bool) -> tuple[pd.DataFrame, features.Panel, dict, FundFn, Any]:
    if demo:
        from .demo import make_demo
        log.info("[demo] 架空データで実行します")
        universe, panel, bench, fund_fn = make_demo()
        return universe, panel, bench, fund_fn, None
    from .prices import download_benchmarks, download_prices
    from .universe import load_universe
    universe = load_universe(cfg)
    log.info("対象銘柄: %d", len(universe))
    d = cfg["data"]
    panel = download_prices(list(universe["ticker"]), period=d["period"], chunk_size=d["chunk_size"],
                            sleep=d["sleep_between_chunks"], min_success_ratio=d["min_success_ratio"])
    store = fundamentals.FundamentalsStore(cfg)
    return universe, panel, download_benchmarks("3y"), store.get, store


# ---------------------------------------------------------------- 2. 除外・基本点
def apply_exclusions(snap: pd.DataFrame, cfg: Config, demo: bool) -> pd.DataFrame:
    """除外条件を順に適用し、理由ごとの件数をログに出す。"""
    flt = cfg["filters"]
    codes = load_exclude_codes(cfg)
    if flt["exclude_supervision"] and not demo:
        from .supervision import fetch_supervision_codes
        codes |= fetch_supervision_codes(flt["supervision_urls"])
    kw = "|".join(flt["exclude_name_keywords"])
    steps: list[tuple[str, pd.Series]] = [
        ("極端な低位株", ~(snap["close"] >= flt["min_price"])),
        ("売買代金不足", ~(snap["turnover20"] >= flt["min_turnover_yen"])),
        ("出来高不足", ~((snap["avgvol20"] >= flt["min_avg_volume"]) & (snap["volume"] >= flt["min_today_volume"]))),
        ("監理・整理銘柄", snap["code"].isin(codes) | (snap["name"].str.contains(kw, na=False) if kw else False)),
    ]
    drop = pd.Series(False, index=snap.index)
    for label, mask in steps:
        newly = mask & ~drop
        log.info("除外 %-8s %5d件", label, int(newly.sum()))
        drop |= mask
    return snap[~drop & snap["close"].notna() & (snap["volume"] > 0)]


def attach_scores(snap: pd.DataFrame, regime: str, cfg: Config, summary: dict | None) -> pd.DataFrame:
    """基本点（トレンド・需給・地合い）、勝率、流動性係数を付ける。"""
    sc, es = cfg["scoring"], cfg["expected_score"]
    tech = scoring.score_technical(snap, regime, scoring.sector_points(snap["ret20d"], snap["sector"], sc), sc)
    out = snap.join(tech)
    out["tech_scaled"] = out["tech_base"] / scoring.tech_max(cfg) * 100
    wr = out["tech_scaled"].map(lambda s: backtest.win_rate_lookup(summary, s, es))
    out["win_rate"], out["win_n"], out["win_known"] = (wr.map(lambda t: t[i]) for i in range(3))
    out["liq"] = scoring.liquidity_factor(out["turnover20"], es)
    out["prelim"] = out["tech_base"] * (out["win_rate"] / es["win_rate_baseline"]) * out["liq"]
    return out


# ---------------------------------------------------------------- 3. 候補の評価
def evaluate_candidates(snap: pd.DataFrame, fund_fn: FundFn, cfg: Config) -> tuple[list[dict], int, Counter]:
    """基本点の上位から順に業績・配当を取得し、除外・イベント点・期待値点を計算する。"""
    sc, fc, es, top_n = cfg["scoring"], cfg["fundamentals"], cfg["expected_score"], cfg["top_n"]
    results: list[dict] = []
    excluded: Counter = Counter()
    fetched = 0
    for tk, r in snap.sort_values("prelim", ascending=False).iterrows():
        if fetched >= fc["max_fetch"] or (fetched >= fc["min_fetch"] and len(results) >= top_n * fc["survivor_multiple"]):
            break
        fund = fund_fn(r["code"])
        fetched += 1
        reason = fundamentals.exclusion_reason(fund, fc) if fund.get("status") == "ok" else \
            ("業績データなし" if fc["exclude_if_missing"] else None)
        if reason:
            excluded[reason] += 1
            continue
        op_pts, mc_pts = scoring.earnings_points(fund, sc)
        ev = events.evaluate_events(r, fund, cfg)
        base = float(r["tech_base"]) + op_pts + mc_pts
        total = base + ev.total
        results.append({"ticker": tk, "row": r, "fund": fund, "op_pts": op_pts, "mc_pts": mc_pts,
                        "ev": ev, "base": base, "total": total,
                        "expected": scoring.expected_score(total, float(r["win_rate"]), float(r["liq"]), es)})
    results.sort(key=lambda x: (x["expected"], x["total"]), reverse=True)
    return results[:top_n], fetched, excluded


# ---------------------------------------------------------------- 4. 画面用データ
def make_cards(top: list[dict], f: features.Features, panel: features.Panel, themes: dict,
               regime: str, cfg: Config) -> list[dict]:
    sc, mx = cfg["scoring"], cfg["weights"]
    cards: list[dict] = []
    for rank, x in enumerate(top, 1):
        r, fund, ev, tk = x["row"], x["fund"], x["ev"], x["ticker"]
        label, css, note = scoring.bull_label(x["total"], explain.is_overheat(r, cfg), regime, cfg)
        sl = slice(-60, None)
        svg = charts.candle_svg(*(panel[k][tk].iloc[sl].to_numpy() for k in ("open", "high", "low", "close", "volume")),
                                f["ma25"][tk].iloc[sl].to_numpy(), f["ma75"][tk].iloc[sl].to_numpy())
        stop = ({"price": float(r["ma25"]), "pct": float(r["ma25"] / r["close"] - 1) * 100}
                if r["ma25"] == r["ma25"] and r["ma25"] < r["close"] else None)
        scores = {"trend": (float(r["pts_trend"]), mx["trend"]), "demand": (float(r["pts_demand"]), mx["demand"]),
                  "earnings": (x["op_pts"] + x["mc_pts"], mx["earnings"]), "market": (float(r["pts_market"]), mx["market"])}
        cards.append({
            "rank": rank, "code": r["code"], "name": r["name"], "market": r["market"], "sector": r["sector"],
            "themes": rankings.themes_of(r["code"], themes),
            "close": float(r["close"]), "ret1d": None if r["ret1d"] != r["ret1d"] else float(r["ret1d"]) * 100,
            "total": round(x["total"]), "expected": x["expected"], "base": round(x["base"]),
            "events_pts": round(ev.total), "scores": {k: [round(a), b] for k, (a, b) in scores.items()},
            "win_rate": float(r["win_rate"]), "win_n": int(r["win_n"]), "win_known": bool(r["win_known"]),
            "liq": float(r["liq"]), "label": label, "css": css, "label_note": note,
            "events": [{"label": i.label, "points": round(i.points), "basis": i.basis, "kind": i.kind} for i in ev.items],
            "explain": explain.reason_text(r, fund, ev, regime),
            "indicators": explain.build_indicators(r),
            "risks": explain.build_risks(r, fund, r["market"], regime, ev),
            "stop": stop, "svg": svg,
        })
    return cards


def save_history(hist_dir, data_date: pd.Timestamp, cards: list[dict], market: dict, demo: bool) -> None:
    hist_dir.mkdir(parents=True, exist_ok=True)
    rec = {"date": data_date.strftime("%Y-%m-%d"), "demo": demo, "regime": market["regime"],
           "picks": [{k: c[k] for k in ("rank", "code", "name", "total", "expected", "close")} for c in cards]}
    (hist_dir / f"{rec['date']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 実行
def run(demo: bool) -> None:
    cfg = load_config()
    scoring.validate_config(cfg)
    themes = load_themes()
    hist_dir, summary_path = backtest.paths(demo)

    universe, panel, bench, fund_fn, store = load_data(cfg, demo)
    panel = trim_to_valid_last(panel)
    data_date = panel["close"].index[-1]
    log.info("データ基準日: %s", data_date.date())

    sc = cfg["scoring"]
    bench = {k: v.loc[:data_date] for k, v in bench.items()}
    f = features.compute_features(panel, bench["topix"]["close"], sc)
    snap = features.snapshot(f, -1).join(universe.set_index("ticker")[["code", "name", "market", "sector"]], how="inner")
    snap = apply_exclusions(snap, cfg, demo)
    log.info("条件を満たす銘柄: %d / %d", len(snap), len(universe))
    if snap.empty:
        raise RuntimeError("条件を満たす銘柄が0件です。データ取得に問題がある可能性があります")

    market = build_market(bench, float((snap["close"] > snap["ma25"]).mean()), int(snap["hi60"].sum()), int(snap["hi252"].sum()))
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("バックテスト結果を読めません（勝率は仮置きになります）: %s", e)
    else:
        log.warning("バックテスト結果が未作成です。期待値点の勝率は仮置き(%.0f%%)になります（weekly-backtest を実行してください）",
                    cfg["expected_score"]["default_win_rate"] * 100)

    snap = attach_scores(snap, market["regime"], cfg, summary)
    top, fetched, excluded = evaluate_candidates(snap, fund_fn, cfg)
    log.info("業績を確認した銘柄: %d（除外: %s）", fetched, dict(excluded) or "なし")
    if store:
        store.save()
    if not top:
        raise RuntimeError("ランキングが0件です（業績取得が全滅した可能性）。時間をおいて再実行してください")
    cards = make_cards(top, f, panel, themes, market["regime"], cfg)

    save_history(hist_dir, data_date, cards, market, demo)
    forward = backtest.forward_stats(panel, bench["topix"], cfg, hist_dir)
    now = datetime.now(JST)
    ctx = {
        "title": cfg["site_title"], "demo": demo,
        "generated_at": f"{now:%Y年%m月%d日} {now:%H:%M}", "generated_iso": now.isoformat(timespec="minutes"),
        "data_date_label": f"{data_date.month}/{data_date.day}（{WEEKDAY[data_date.weekday()]}）",
        "data_date_iso": data_date.strftime("%Y-%m-%d"),
        "market": market, "cards": cards, "bt": backtest.build_table(summary, forward, cfg),
        "pt": backtest.build_period_table(summary),
        "sectors": rankings.rank_sectors(snap), "themes": rankings.rank_themes(snap, themes),
        "n_universe": len(universe), "n_scanned": len(snap), "n_fetched": fetched,
        "bull": cfg["bull"], "w": cfg["weights"], "sc": sc, "ev": cfg["events"], "es": cfg["expected_score"],
        "top_n": cfg["top_n"], "min_turnover_oku": cfg["filters"]["min_turnover_yen"] / 1e8,
        "min_price": cfg["filters"]["min_price"], "win_known_any": any(c["win_known"] for c in cards),
    }
    log.info("生成しました: %s", render_site(ctx))
    for c in cards[:3]:
        log.info("  %d位 %s(%s) 期待値%.1f 総合%d %s", c["rank"], c["name"], c["code"], c["expected"], c["total"], c["label"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認（ネット不要）")
    try:
        run(ap.parse_args().demo)
    except ConfigError as e:
        log.error("設定エラー: %s", e)
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
