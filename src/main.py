"""毎朝の処理本体。  python -m src.main        （デモ:  python -m src.main --demo）

流れ（押し目ランキングがメイン。従来の「今強い銘柄」スコアは補助指標）
  データ取得 → 指標 → 除外(数値・監理整理・TOB疑い) → 押し目スコア(技術) → 上位の業績・配当取得(赤字除外)
  → 押し目スコア確定 → 上位50をTDnet解析（企業再編は除外・イベント点を加減点）→ 最終順位 → HTML・CSV
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from typing import Any, Callable

import pandas as pd

from . import backtest, charts, events, explain, features, fundamentals, pullback, rankings, scoring, tdnet, tdnet_events
from .config import JST, Config, ConfigError, load_config, load_exclude_codes, load_themes
from .log import get_logger
from .market import above_ma50_series, build_market
from .render import render_site, write_excluded_csv

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


# ---------------------------------------------------------------- 2. 除外
def apply_exclusions(snap: pd.DataFrame, cfg: Config, demo: bool, today: date) -> tuple[pd.DataFrame, list[dict]]:
    """数値条件・監理整理・TOB疑い（価格固着）の除外。除外理由ごとの件数をログに出し、CSV用の行も返す。"""
    flt = cfg["filters"]
    codes = load_exclude_codes(cfg)
    jpx: set[str] = set()
    if flt["exclude_supervision"] and not demo:
        from .supervision import fetch_supervision_codes
        jpx = fetch_supervision_codes(flt["supervision_urls"])
    kw = "|".join(flt["exclude_name_keywords"])
    name_hit = snap["name"].str.contains(kw, na=False) if kw else pd.Series(False, index=snap.index)
    lock = flt["tob_price_lock"]
    price_lock = ((snap["max_ret60"] >= lock["jump"]) & (snap["range10"] <= lock["range10"])
                  if lock["enabled"] else pd.Series(False, index=snap.index))

    steps: list[tuple[str, pd.Series]] = [
        ("極端な低位株", ~(snap["close"] >= flt["min_price"])),
        ("売買代金不足", ~(snap["turnover20"] >= flt["min_turnover_yen"])),
        ("出来高不足", ~((snap["avgvol20"] >= flt["min_avg_volume"]) & (snap["volume"] >= flt["min_today_volume"]))),
        ("監理・整理銘柄", snap["code"].isin(jpx | codes) | name_hit),
        ("TOB疑い（価格固着）", price_lock.fillna(False)),
    ]
    drop = pd.Series(False, index=snap.index)
    rows: list[dict] = []
    for label, mask in steps:
        newly = mask & ~drop
        log.info("除外 %-12s %5d件", label, int(newly.sum()))
        if label in ("監理・整理銘柄", "TOB疑い（価格固着）"):
            for _, r in snap[newly].iterrows():
                ev = ("JPX監理・整理銘柄一覧" if r["code"] in jpx else "除外リスト／銘柄名に「監理」「整理」") \
                    if label == "監理・整理銘柄" else f"直近60日に1日で{r['max_ret60'] * 100:+.0f}%のギャップ後、10日間の値幅{r['range10'] * 100:.1f}%に固着"
                rows.append({"code": r["code"], "name": r["name"], "reason": label, "events": ev, "date": today.isoformat()})
        drop |= mask
    return snap[~drop & snap["close"].notna() & (snap["volume"] > 0)], rows


# ---------------------------------------------------------------- 3. スコア（従来＝補助 ＋ 押し目＝メイン）
def attach_scores(snap: pd.DataFrame, regime: str, nikkei_ok: bool, cfg: Config, summary: dict | None) -> pd.DataFrame:
    sc, es = cfg["scoring"], cfg["expected_score"]
    tech = scoring.score_technical(snap, regime, scoring.sector_points(snap["ret20d"], snap["sector"], sc), sc)
    out = snap.join(tech)
    out["tech_scaled"] = out["tech_base"] / scoring.tech_max(cfg) * 100
    wr = out["tech_scaled"].map(lambda s: backtest.win_rate_lookup(summary, s, es))
    out["win_rate"], out["win_n"], out["win_known"] = (wr.map(lambda t: t[i]) for i in range(3))
    out["liq"] = scoring.liquidity_factor(out["turnover20"], es)
    return out.join(pullback.score_technical(snap, regime, nikkei_ok, cfg))


# ---------------------------------------------------------------- 4. 候補の評価（業績・配当 → 押し目スコア確定）
def evaluate_candidates(snap: pd.DataFrame, fund_fn: FundFn, cfg: Config) -> tuple[list[dict], int, Counter]:
    sc, fc, es = cfg["scoring"], cfg["fundamentals"], cfg["expected_score"]
    cands: list[dict] = []
    excluded: Counter = Counter()
    fetched = 0
    order = snap[snap["pb_tech_scaled"] > 0].sort_values(["pb_tech_scaled", "tech_base"], ascending=False)
    for tk, r in order.iterrows():
        if fetched >= fc["max_fetch"] or (fetched >= fc["min_fetch"] and len(cands) >= fc["survivor_target"]):
            break
        fund = fund_fn(r["code"])
        fetched += 1
        reason = fundamentals.exclusion_reason(fund, fc) if fund.get("status") == "ok" else \
            ("業績データなし" if fc["exclude_if_missing"] else None)
        if reason:
            excluded[reason] += 1
            continue
        ev = events.evaluate_events(r, fund, cfg)
        q, qkeys = pullback.quality_points(fund, ev, cfg)
        op_pts, mc_pts = scoring.earnings_points(fund, sc)
        legacy_base = float(r["tech_base"]) + op_pts + mc_pts
        legacy_total = legacy_base + ev.total
        cands.append({"ticker": tk, "row": r, "fund": fund, "ev": ev, "quality": q, "qkeys": qkeys,
                      "pb_total": pullback.total(float(r["pb_sum"]), q, float(r["pb_penalty"])),
                      "op_pts": op_pts, "mc_pts": mc_pts, "legacy_base": legacy_base, "legacy_total": legacy_total,
                      "legacy_expected": scoring.expected_score(legacy_total, float(r["win_rate"]), float(r["liq"]), es)})
    cands.sort(key=lambda c: (c["pb_total"], c["legacy_expected"]), reverse=True)
    return cands, fetched, excluded


# ---------------------------------------------------------------- 5. TDnet解析（上位のみ）・企業再編の除外
def run_tdnet(cands: list[dict], client: Any, cfg: Config, today: date) -> tuple[list[dict], list[dict], dict]:
    """押し目スコア上位から順にTDnetを解析。企業再編イベントがあれば除外して次の銘柄を繰り上げる（解析は最大 max_check 銘柄）。"""
    t = cfg["tdnet"]
    stats = {"enabled": client is not None, "checked": 0, "unavailable": 0, "excluded": 0}
    if client is None:
        return [dict(c, td=None) for c in cands[:t["top_k"]]], [], stats
    pdf_fn = None if isinstance(client, tdnet.DemoTdnet) else lambda u: tdnet.pdf_text(u, t["timeout"])
    kept: list[dict] = []
    rows: list[dict] = []
    for c in cands:
        if len(kept) >= t["top_k"] or stats["checked"] >= t["max_check"]:
            break
        stats["checked"] += 1
        r = c["row"]
        try:
            res = tdnet_events.analyze(client.fetch(r["code"]), c["fund"], cfg, today, pdf_fn)
        except tdnet.TdnetError as e:
            stats["unavailable"] += 1
            log.warning("%s", e)
            if t["on_error"] == "exclude":
                rows.append({"code": r["code"], "name": r["name"], "reason": "TDnet取得失敗（安全のため除外）",
                             "events": str(e)[:80], "date": today.isoformat()})
                continue
            res = tdnet_events.TdnetResult(status="unavailable", error=str(e))
        if res.excluded:
            stats["excluded"] += 1
            rows.append({"code": r["code"], "name": r["name"], "reason": " / ".join(dict.fromkeys(x["reason"] for x in res.excluded)),
                         "events": " / ".join(x["event"] for x in res.excluded)[:300],
                         "date": max(x["date"] for x in res.excluded)})
            continue
        kept.append(dict(c, td=res))
    return kept, rows, stats


# ---------------------------------------------------------------- 6. 画面用データ
def _stop(r: Any) -> dict | None:
    for name, key in (("50日線", "ma50"), ("200日線", "ma200")):
        if r[key] == r[key] and r[key] < r["close"]:
            return {"name": name, "price": float(r[key]), "pct": float(r[key] / r["close"] - 1) * 100}
    return None


def make_cards(top: list[dict], f: features.Features, panel: features.Panel, themes: dict, regime: str,
               nikkei_ok: bool, cfg: Config, today: date) -> list[dict]:
    pbw, w, sc = cfg["pullback"]["weights"], cfg["weights"], cfg["scoring"]
    legacy_rank = {c["ticker"]: i for i, c in enumerate(sorted(top, key=lambda c: c["legacy_expected"], reverse=True), 1)}
    cards: list[dict] = []
    for rank, x in enumerate(top, 1):
        r, fund, ev, td, tk = x["row"], x["fund"], x["ev"], x["td"], x["ticker"]
        td_score = td.score if td is not None and td.status == "ok" else 0.0
        main = x["pb_total"] + td_score
        label, css, note = pullback.label(main, bool(td and td.serious_negative), cfg)
        sl = slice(-60, None)
        svg = charts.candle_svg(*(panel[k][tk].iloc[sl].to_numpy() for k in ("open", "high", "low", "close", "volume")),
                                f["ma25"][tk].iloc[sl].to_numpy(), f["ma50"][tk].iloc[sl].to_numpy())
        stren = explain.strengths(ev, td)
        parts = {"trend": (r["pb_trend"], pbw["trend"]), "drawdown": (r["pb_dd"], pbw["drawdown"]),
                 "ma50": (r["pb_ma50"], pbw["ma50"]), "volume": (r["pb_vol"], pbw["volume"]),
                 "quality": (x["quality"], pbw["quality"]), "market": (r["pb_market"], pbw["market"])}
        cards.append({
            "rank": rank, "code": r["code"], "name": r["name"], "market": r["market"], "sector": r["sector"],
            "themes": rankings.themes_of(r["code"], themes),
            "close": float(r["close"]), "ret1d": None if r["ret1d"] != r["ret1d"] else float(r["ret1d"]) * 100,
            "main": round(main), "pb_total": round(x["pb_total"]), "td_score": round(td_score),
            "penalty": round(float(r["pb_penalty"])),
            "parts": {k: [round(float(a)), b] for k, (a, b) in parts.items()},
            "label": label, "css": css, "label_note": note,
            "strengths": stren,
            "status": explain.pullback_status(r, cfg, regime, nikkei_ok),
            "cautions": explain.cautions(r, fund, td, ev, cfg, regime, r["market"], today),
            "explain": explain.pullback_reason(r, stren, regime, nikkei_ok),
            "indicators": explain.build_indicators(r), "stop": _stop(r), "svg": svg,
            "tdnet": "ok" if td is not None and td.status == "ok" else ("off" if td is None else "ng"),
            "legacy": {"rank": legacy_rank[tk], "total": round(x["legacy_total"]), "base": round(x["legacy_base"]),
                       "events_pts": round(ev.total), "expected": x["legacy_expected"],
                       "win_rate": float(r["win_rate"]), "win_known": bool(r["win_known"]), "liq": float(r["liq"]),
                       "trend": [round(float(r["pts_trend"])), w["trend"]], "demand": [round(float(r["pts_demand"])), w["demand"]],
                       "earnings": [round(x["op_pts"] + x["mc_pts"]), w["earnings"]],
                       "market": [round(float(r["pts_market"])), w["market"]]},
        })
    return cards


def save_history(hist_dir, data_date: pd.Timestamp, cards: list[dict], market: dict, demo: bool) -> None:
    hist_dir.mkdir(parents=True, exist_ok=True)
    rec = {"date": data_date.strftime("%Y-%m-%d"), "demo": demo, "regime": market["regime"],
           "picks": [{"rank": c["rank"], "code": c["code"], "name": c["name"], "main": c["main"],
                      "pb_total": c["pb_total"], "close": c["close"]} for c in cards]}
    (hist_dir / f"{rec['date']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 実行
def run(demo: bool) -> None:
    cfg = load_config()
    scoring.validate_config(cfg)
    pullback.validate_config(cfg)
    themes, today = load_themes(), datetime.now(JST).date()
    hist_dir, summary_path = backtest.paths(demo)

    universe, panel, bench, fund_fn, store = load_data(cfg, demo)
    panel = trim_to_valid_last(panel)
    data_date = panel["close"].index[-1]
    log.info("データ基準日: %s", data_date.date())

    bench = {k: v.loc[:data_date] for k, v in bench.items()}
    f = features.add_live_extras(features.compute_features(panel, bench["topix"]["close"], cfg["scoring"]))
    snap = features.snapshot(f, -1).join(universe.set_index("ticker")[["code", "name", "market", "sector"]], how="inner")
    snap, excluded_rows = apply_exclusions(snap, cfg, demo, today)
    log.info("条件を満たす銘柄: %d / %d", len(snap), len(universe))
    if snap.empty:
        raise RuntimeError("条件を満たす銘柄が0件です。データ取得に問題がある可能性があります")

    market = build_market(bench, float((snap["close"] > snap["ma25"]).mean()), int(snap["hi60"].sum()), int(snap["hi252"].sum()))
    nikkei_ok = bool(above_ma50_series(bench["n225"]["close"]).iloc[-1]) if "n225" in bench else False
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("バックテスト結果を読めません（従来スコアの勝率は仮置きになります）: %s", e)

    snap = attach_scores(snap, market["regime"], nikkei_ok, cfg, summary)
    cands, fetched, excl = evaluate_candidates(snap, fund_fn, cfg)
    log.info("業績を確認した銘柄: %d（除外: %s） 押し目候補: %d", fetched, dict(excl) or "なし", len(cands))
    if store:
        store.save()

    client = tdnet.make_client(cfg, demo)
    if isinstance(client, tdnet.TdnetClient):
        tdnet.selfcheck(client)
    kept, td_rows, td_stats = run_tdnet(cands, client, cfg, today)
    if client:
        client.save()
    log.info("TDnet解析: %d銘柄（企業再編などで除外 %d／取得失敗 %d）", td_stats["checked"], td_stats["excluded"], td_stats["unavailable"])
    excluded_rows += td_rows

    for c in kept:
        c["main"] = c["pb_total"] + (c["td"].score if c["td"] is not None and c["td"].status == "ok" else 0.0)
    kept.sort(key=lambda c: (c["main"], c["legacy_expected"]), reverse=True)
    top = kept[:cfg["top_n"]]
    if not top:
        raise RuntimeError("ランキングが0件です（業績取得が全滅した可能性）。時間をおいて再実行してください")
    cards = make_cards(top, f, panel, themes, market["regime"], nikkei_ok, cfg, today)

    save_history(hist_dir, data_date, cards, market, demo)
    forward = backtest.forward_stats(panel, bench["topix"], cfg, hist_dir)
    now = datetime.now(JST)
    pb = cfg["pullback"]
    ctx = {
        "title": cfg["site_title"], "demo": demo,
        "generated_at": f"{now:%Y年%m月%d日} {now:%H:%M}", "generated_iso": now.isoformat(timespec="minutes"),
        "data_date_label": f"{data_date.month}/{data_date.day}（{WEEKDAY[data_date.weekday()]}）",
        "data_date_iso": data_date.strftime("%Y-%m-%d"),
        "market": market, "nikkei_ok": nikkei_ok, "cards": cards,
        "bt": backtest.build_table(summary, forward, cfg),
        "pt": backtest.build_period_table(summary, "periods"), "pt_legacy": backtest.build_period_table(summary, "periods_legacy"),
        "sectors": rankings.rank_sectors(snap), "themes": rankings.rank_themes(snap, themes),
        "n_universe": len(universe), "n_scanned": len(snap), "n_fetched": fetched,
        "td": td_stats, "tdnet_warn": bool(client) and td_stats["checked"] > 0 and td_stats["unavailable"] > td_stats["checked"] / 2,
        "n_excluded_events": len(excluded_rows),
        "pb": pb, "w": cfg["weights"], "tdc": cfg["tdnet"], "ev": cfg["events"], "es": cfg["expected_score"],
        "top_n": cfg["top_n"], "min_turnover_oku": cfg["filters"]["min_turnover_yen"] / 1e8, "min_price": cfg["filters"]["min_price"],
    }
    out = render_site(ctx)
    write_excluded_csv(excluded_rows, out.parent / "excluded_events.csv")
    log.info("生成しました: %s（除外イベント %d件 → excluded_events.csv）", out, len(excluded_rows))
    for c in cards[:3]:
        log.info("  %d位 %s(%s) 総合%d（押し目%d＋イベント%+d） %s", c["rank"], c["name"], c["code"], c["main"],
                 c["pb_total"], c["td_score"], c["label"])


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
