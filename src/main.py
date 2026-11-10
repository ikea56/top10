"""毎朝の処理本体。  python -m src.main        （デモ:  python -m src.main --demo）

流れ（押し目ランキングがメイン。従来の「今強い銘柄」スコアは補助指標）
  データ取得 → 指標 → 除外(数値・監理整理・TOB疑い) → 押し目スコア(技術) → 上位の業績・配当取得(赤字除外)
  → 押し目スコア確定 → 上位50をTDnet解析（企業再編は除外・イベント点を加減点）→ 最終順位 → HTML・CSV
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import date, datetime
from typing import Any, Callable

import pandas as pd

from . import (backtest, charts, events, explain, features, fundamentals, holdings, movers, pullback, rankings,
               scoring, tdnet, tdnet_events)
from .config import JST, Config, ConfigError, load_config, load_exclude_codes, load_themes
from .log import get_logger
from .market import above_ma50_series, build_market
from .render import render_site, write_excluded_csv, write_holdings_files

log = get_logger(__name__)
WEEKDAY = "月火水木金土日"
FundFn = Callable[[str], dict]


# ---------------------------------------------------------------- 1. データ取得
def trim_to_valid_last(panel: features.Panel) -> features.Panel:
    """最終行が読み込み途中（銘柄の半分以上が欠損）なら切り捨てる。"""
    counts = panel["close"].notna().sum(axis=1)
    last = counts[counts >= 0.5 * counts.max()].index[-1]
    return {k: v.loc[:last] for k, v in panel.items()}


def drop_unfinished_bar(panel: features.Panel, now: datetime, close_hour: float = 16.0) -> features.Panel:
    """取引時間中に実行された場合、今日の足は未確定（途中の値）なので切り捨てる。
    東証の終了は15:30。16:00より前に今日の日付の足があれば、それは「終値」ではない。"""
    last = panel["close"].index[-1]
    if last.date() >= now.date() and now.hour + now.minute / 60 < close_hour:
        log.info("今日(%s)の足は取引時間中の未確定値のため使いません", last.date())
        return {k: v.iloc[:-1] for k, v in panel.items()}
    return panel


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
def attach_scores(snap: pd.DataFrame, regime: str, nikkei_ok: bool, cfg: Config, summary: dict | None,
                  topix_below200: bool = False) -> pd.DataFrame:
    sc, es = cfg["scoring"], cfg["expected_score"]
    tech = scoring.score_technical(snap, regime, scoring.sector_points(snap["ret20d"], snap["sector"], sc), sc)
    out = snap.join(tech)
    out["tech_scaled"] = out["tech_base"] / scoring.tech_max(cfg) * 100
    wr = out["tech_scaled"].map(lambda s: backtest.win_rate_lookup(summary, s, es))
    out["win_rate"], out["win_n"], out["win_known"] = (wr.map(lambda t: t[i]) for i in range(3))
    out["liq"] = scoring.liquidity_factor(out["turnover20"], es)
    return out.join(pullback.score_technical(snap, regime, nikkei_ok, cfg, topix_below200))


# ---------------------------------------------------------------- 4. 候補の評価（業績・配当 → 押し目スコア確定）
def evaluate_candidates(snap: pd.DataFrame, fund_fn: FundFn, cfg: Config, today: date) -> tuple[list[dict], int, Counter]:
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
        earn_pen, earn_days = pullback.earnings_penalty(fund.get("next_earnings"), today, cfg)   # 決算接近の減点
        cands.append({"ticker": tk, "row": r, "fund": fund, "ev": ev, "quality": q, "qkeys": qkeys,
                      "earn_pen": earn_pen, "earn_days": earn_days,
                      "pb_total": pullback.total(float(r["pb_sum"]), q, float(r["pb_penalty"]), earn_pen),
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
def final_score(c: dict, cfg: Config) -> tuple[float, float, float, list[str]]:
    """(総合点, TDnetイベント点, 下方修正の減点, 下方修正の根拠)。総合点＝押し目スコア＋イベント点−下方修正の減点。"""
    td = c.get("td")
    td_score = td.score if td is not None and td.status == "ok" else 0.0
    src = tdnet_events.recent_downward(td, c["fund"], cfg)
    pen = float(cfg["tdnet"]["downward_recent"]["penalty"]) if src else 0.0
    return c["pb_total"] + td_score - pen, td_score, pen, src


def make_cards(top: list[dict], f: features.Features, panel: features.Panel, themes: dict, regime: str,
               nikkei_ok: bool, cfg: Config, today: date) -> list[dict]:
    pbw, w, sc = cfg["pullback"]["weights"], cfg["weights"], cfg["scoring"]
    legacy_rank = {c["ticker"]: i for i, c in enumerate(sorted(top, key=lambda c: c["legacy_expected"], reverse=True), 1)}
    cards: list[dict] = []
    for rank, x in enumerate(top, 1):
        r, fund, ev, td, tk = x["row"], x["fund"], x["ev"], x["td"], x["ticker"]
        main, td_score, down_pen, down_src = final_score(x, cfg)
        label, css, note = pullback.label(main, tdnet_events.serious_negative(td, fund, cfg), cfg)
        if note:
            note = "直近30日以内に下方修正があるため" if down_src else "減配の開示があるため"
        sl = slice(-60, None)
        svg = charts.candle_svg(*(panel[k][tk].iloc[sl].to_numpy() for k in ("open", "high", "low", "close", "volume")),
                                f["ma25"][tk].iloc[sl].to_numpy(), f["ma50"][tk].iloc[sl].to_numpy())
        stren = explain.strengths(ev, td)
        parts = {"trend": (r["pb_trend"], pbw["trend"]), "drawdown": (r["pb_dd"], pbw["drawdown"]),
                 "ma50": (r["pb_ma50"], pbw["ma50"]), "volume": (r["pb_vol"], pbw["volume"]),
                 "strength": (r["pb_rs"] + r["pb_bounce"], pbw["rs"] + pbw["bounce"]),
                 "quality": (x["quality"], pbw["quality"]), "market": (r["pb_market"], pbw["market"])}
        cards.append({
            "rank": rank, "code": r["code"], "name": r["name"], "market": r["market"], "sector": r["sector"],
            "themes": rankings.themes_of(r["code"], themes),
            "close": float(r["close"]), "ret1d": None if r["ret1d"] != r["ret1d"] else float(r["ret1d"]) * 100,
            "main": round(main, 1), "pb_total": round(x["pb_total"], 1), "td_score": round(td_score),
            "penalty": round(float(r["pb_penalty"]) + x["earn_pen"]), "earn_pen": round(x["earn_pen"]),
            "down_pen": round(down_pen), "market_pts": round(float(r["pb_market"])),
            "parts": {k: [round(float(a), 1), b] for k, (a, b) in parts.items()},
            "label": label, "css": css, "label_note": note,
            "strengths": stren,
            "status": explain.pullback_status(r, cfg, regime, nikkei_ok),
            "cautions": explain.cautions(r, fund, td, ev, cfg, regime, r["market"], today, x["earn_pen"], x["earn_days"], down_src),
            "explain": explain.pullback_reason(r, stren, regime, nikkei_ok),
            "indicators": explain.build_indicators(r), "stop": pullback.stop_line(r, cfg), "svg": svg,
            "tdnet": "ok" if td is not None and td.status == "ok" else ("off" if td is None else "ng"),
            "legacy": {"rank": legacy_rank[tk], "total": round(x["legacy_total"]), "base": round(x["legacy_base"]),
                       "events_pts": round(ev.total), "expected": x["legacy_expected"],
                       "win_rate": float(r["win_rate"]), "win_known": bool(r["win_known"]), "liq": float(r["liq"]),
                       "trend": [round(float(r["pts_trend"])), w["trend"]], "demand": [round(float(r["pts_demand"])), w["demand"]],
                       "earnings": [round(x["op_pts"] + x["mc_pts"]), w["earnings"]],
                       "market": [round(float(r["pts_market"])), w["market"]]},
        })
    return cards


def save_history(hist_dir, data_date: pd.Timestamp, cards: list[dict], market: dict, demo: bool,
                 ranking: list[dict] | None = None) -> None:
    hist_dir.mkdir(parents=True, exist_ok=True)
    rec = {"date": data_date.strftime("%Y-%m-%d"), "demo": demo, "regime": market["regime"],
           "picks": [{"rank": c["rank"], "code": c["code"], "name": c["name"], "main": c["main"],
                      "pb_total": c["pb_total"], "close": c["close"]} for c in cards],
           "ranking": ranking or []}            # 上位30（前回比較用）。保有銘柄は記録しない（公開リポジトリのため）
    (hist_dir / f"{rec['date']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 7. 保有銘柄の監視（ランキング順位に関係なく常に評価）
def monitor_holdings(codes: list[str], snap_full: pd.DataFrame, snap: pd.DataFrame, rank_of: dict[str, int],
                     regime: str, nikkei_ok: bool, topix_below200: bool, fund_fn: FundFn, client: Any,
                     cfg: Config, today: date, demo: bool) -> list[dict]:
    """保有銘柄ごとに、押し目スコア・イベント点・順位・開示・注意事項・推奨アクション（🟢🟡🔴）を作る。
    売買代金などの除外条件は掛けない（流動性が低くても、TOBでも、監視を続ける）。"""
    hc, flt = cfg["holdings"], cfg["filters"]
    jpx: set[str] = set()
    if flt["exclude_supervision"] and not demo:
        try:
            from .supervision import fetch_supervision_codes
            jpx = fetch_supervision_codes(flt["supervision_urls"])
        except Exception as e:                               # noqa: BLE001 取得できなくても監視は続ける
            log.warning("監理・整理銘柄一覧を取得できませんでした（保有銘柄の判定は他の情報で行います）: %s", e)
    manual = load_exclude_codes(cfg)
    kw = "|".join(flt["exclude_name_keywords"])
    lock = flt["tob_price_lock"]
    pdf_fn = None if client is None or isinstance(client, tdnet.DemoTdnet) else \
        lambda u: tdnet.pdf_text(u, cfg["tdnet"]["timeout"])
    by_code = {r["code"]: (tk, r) for tk, r in snap_full.iterrows()}
    out: list[dict] = []
    for code in codes:
        if code not in by_code:
            out.append({"code": code, "name": "（銘柄が見つかりません）", "found": False,
                        "action": {"key": "watch", **holdings.ACTIONS["watch"], "reasons": ["東証の上場銘柄に見つかりません。証券コードを確認してください"]}})
            continue
        tk, r0 = by_code[code]
        r = r0.copy()
        for k, v in pullback.score_technical(snap_full.loc[[tk]], regime, nikkei_ok, cfg, topix_below200).iloc[0].items():
            r[k] = v
        fund = fund_fn(code)
        ev = events.evaluate_events(r, fund, cfg)
        q, _ = pullback.quality_points(fund, ev, cfg)
        earn_pen, earn_days = pullback.earnings_penalty(fund.get("next_earnings"), today, cfg)
        td = None
        td_status = "off"
        if client is not None:
            try:
                td = tdnet_events.analyze(client.fetch(code), fund, cfg, today, pdf_fn)
                td_status = "ok"
            except tdnet.TdnetError as e:
                log.warning("%s（保有銘柄）", e)
                td, td_status = tdnet_events.TdnetResult(status="unavailable", error=str(e)), "unavailable"
        cand = {"pb_total": pullback.total(float(r["pb_sum"]), q, float(r["pb_penalty"]), earn_pen), "td": td, "fund": fund}
        main, td_score, down_pen, down_src = final_score(cand, cfg)
        hits = list(td.hits) if td is not None and td.status == "ok" else []
        down_hits = [h for h in hits if h.kind == "downward_revision"]
        cut = [h for h in hits if h.kind == "dividend_cut"]
        close, ma50, ma200, atr = float(r["close"]), r["ma50"], r["ma200"], r.get("atr")
        broken: list[str] = []
        data_ok = ma200 == ma200 and ma50 == ma50
        if data_ok:
            if close < ma200:
                broken.append("株価が200日線を下回りました（長期トレンド割れ）")
            elif ma50 < ma200:
                broken.append("50日線が200日線を下回っています（トレンドが弱まっています）")
            if atr == atr and atr and close < ma50 - atr:
                broken.append("株価が「50日線−ATR」を下回りました（押し目が崩れた可能性）")
            if r["dist52w"] <= hc["collapse_dd"]:
                broken.append(f"52週高値から{r['dist52w'] * 100:.0f}%下落しています")
        ne = fund.get("next_earnings")
        e_days = (date.fromisoformat(ne) - today).days if ne and date.fromisoformat(ne) >= today else None
        locked = bool(lock["enabled"] and r.get("max_ret60", 0) >= lock["jump"] and r.get("range10", 1) <= lock["range10"])
        act = holdings.decide_action({
            "corporate": [x["event"] for x in (td.excluded if td is not None else [])],
            "supervised": code in jpx or code in manual or bool(kw and __import__("re").search(kw, str(r["name"]))),
            "price_lock": locked, "down_recent": down_src,
            "down_days": min((h.age for h in down_hits), default=None),
            "dividend_cut_days": min((h.age for h in cut), default=None),
            "broken": broken, "earnings_days": e_days, "earn_pen": earn_pen,
            "unknown_rev": any(h.kind == "unknown_revision" for h in hits), "td_status": td_status, "data_ok": data_ok,
        }, cfg)
        has = lambda *ks: [h for h in hits if h.kind in ks]  # noqa: E731
        flag = lambda label, hs, bad=False: {"label": label, "state": ("bad" if bad else "good") if hs else "none",  # noqa: E731
                                              "detail": hs[0].date[5:].replace("-", "/") if hs and not isinstance(hs[0], str) else (hs[0] if hs else "")}
        tech_rank = int((snap["pb_tech_scaled"] > r["pb_tech_scaled"]).sum()) + 1
        label, css, _ = pullback.label(main, tdnet_events.serious_negative(td, fund, cfg), cfg)
        out.append({
            "code": code, "name": r["name"], "found": True, "market": r["market"], "sector": r["sector"],
            "close": close, "ret1d": None if r["ret1d"] != r["ret1d"] else float(r["ret1d"]) * 100,
            "pb_total": round(cand["pb_total"], 1), "td_score": round(td_score), "main": round(main, 1),
            "down_pen": round(down_pen), "label": label, "css": css,
            "rank_main": rank_of.get(code), "rank_tech": tech_rank, "n_tech": int(len(snap)),
            "events": [flag("増配", has("dividend_increase")), flag("上方修正", has("upward_revision")),
                       flag("下方修正", down_hits or ([] if not down_src else ["予想EPS"]), bad=True),
                       flag("自社株買い", has("buyback"))],
            "earnings": {"date": ne, "days": e_days, "penalty": round(earn_pen)},
            "td_events": [{"label": h.label, "date": h.date, "points": round(h.points, 1)} for h in hits],
            "td_status": td_status,
            "cautions": explain.cautions(r, fund, td, ev, cfg, regime, r["market"], today, earn_pen, earn_days, down_src),
            "stop": pullback.stop_line(r, cfg), "action": act,
        })
    order = {"sell": 0, "watch": 1, "hold": 2}
    return sorted(out, key=lambda h: (order[h["action"]["key"]], h["code"]))          # 悪化している銘柄を先頭に


# ---------------------------------------------------------------- 実行
def run(demo: bool) -> None:
    cfg = load_config()
    scoring.validate_config(cfg)
    pullback.validate_config(cfg)
    themes, today = load_themes(), datetime.now(JST).date()
    hist_dir, summary_path = backtest.paths(demo)

    universe, panel, bench, fund_fn, store = load_data(cfg, demo)
    panel = trim_to_valid_last(panel)
    if not demo:
        panel = drop_unfinished_bar(panel, datetime.now(JST))
    data_date = panel["close"].index[-1]
    log.info("データ基準日: %s", data_date.date())

    bench = {k: v.loc[:data_date] for k, v in bench.items()}
    f = features.add_live_extras(features.compute_features(panel, bench["topix"]["close"], cfg["scoring"]), panel,
                                 int(cfg["pullback"]["atr"]["period"]))
    snap = features.snapshot(f, -1).join(universe.set_index("ticker")[["code", "name", "market", "sector"]], how="inner")
    snap_full = snap                                     # 保有銘柄の監視用（除外条件を掛ける前）
    snap, excluded_rows = apply_exclusions(snap, cfg, demo, today)
    log.info("条件を満たす銘柄: %d / %d", len(snap), len(universe))
    if snap.empty:
        raise RuntimeError("条件を満たす銘柄が0件です。データ取得に問題がある可能性があります")

    market = build_market(bench, float((snap["close"] > snap["ma25"]).mean()), int(snap["hi60"].sum()), int(snap["hi252"].sum()))
    nikkei_ok = bool(above_ma50_series(bench["n225"]["close"]).iloc[-1]) if "n225" in bench else False
    tc = bench["topix"]["close"].dropna()
    topix_below200 = bool(len(tc) >= 200 and tc.iloc[-1] < tc.rolling(200).mean().iloc[-1])
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("バックテスト結果を読めません（従来スコアの勝率は仮置きになります）: %s", e)

    snap = attach_scores(snap, market["regime"], nikkei_ok, cfg, summary, topix_below200)
    cands, fetched, excl = evaluate_candidates(snap, fund_fn, cfg, today)
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
        c["main"] = final_score(c, cfg)[0]
    kept.sort(key=lambda c: (c["main"], c["pb_total"], c["legacy_expected"]), reverse=True)   # 同点は押し目スコア→従来の期待値の順
    top = kept[:cfg["top_n"]]
    if not top:
        raise RuntimeError("ランキングが0件です（業績取得が全滅した可能性）。時間をおいて再実行してください")
    cards = make_cards(top, f, panel, themes, market["regime"], nikkei_ok, cfg, today)

    # ② 前回ランキングとの比較（上位30を記録）
    mv = cfg["movers"]
    ranking = [{"rank": i, "code": c["row"]["code"], "name": c["row"]["name"], "main": round(c["main"], 1)}
               for i, c in enumerate(kept[:mv["save_top"]], 1)]
    prev_date, prev_rank = movers.load_previous(hist_dir, data_date.strftime("%Y-%m-%d"), demo)
    mover = movers.compare(prev_rank, ranking, mv) if prev_rank else None
    changes = movers.annotate(prev_rank, ranking, mv["show_top"])
    for c in cards:
        c["rank_change"] = changes.get(c["code"])
    summary_changes = {k: sum(1 for c in cards if (c["rank_change"] or {}).get("kind") == k) for k in ("new", "up", "down", "same")} if changes else None

    # ③ 保有銘柄の監視（保存先から一覧を取得 → 順位に関係なく評価 → 暗号化して出力）
    hold_cards, hold_key = None, None
    hc = cfg["holdings"]
    endpoint = os.environ.get("HOLDINGS_ENDPOINT") or hc["endpoint"]
    if demo:
        ucodes = list(universe["code"])
        pick = lambda m: next((c for c in ucodes if c.isdigit() and int(c) % m == 0), ucodes[3])  # noqa: E731 デモ開示の再現用
        sample = [c["code"] for c in cards[:2]] + [pick(11), pick(17), ucodes[8], "9999"]
        hold_codes, hold_key, endpoint = list(dict.fromkeys(sample)), "demo-passphrase", endpoint or "https://example.invalid/exec"
    elif endpoint and os.environ.get("HOLDINGS_KEY"):
        hold_key, hold_codes = os.environ["HOLDINGS_KEY"], []
        try:
            hold_codes = holdings.fetch_codes(endpoint, hold_key)
        except holdings.HoldingsError as e:
            log.warning("%s → 保有銘柄の監視は今回スキップします（前回の監視データはそのまま残ります）", e)
            hold_key = None
    elif endpoint:
        log.warning("保有銘柄の合言葉（GitHub Secrets の HOLDINGS_KEY）が未設定のため、保有銘柄の監視をスキップします")
    if hold_key:
        rank_of = {c["row"]["code"]: i for i, c in enumerate(kept, 1)}
        hold_cards = monitor_holdings(hold_codes, snap_full, snap, rank_of, market["regime"], nikkei_ok,
                                      topix_below200, fund_fn, client, cfg, today, demo)
        cnt = Counter(h["action"]["text"] for h in hold_cards)
        log.info("保有銘柄の監視: %d銘柄（%s）", len(hold_cards), " / ".join(f"{k}{v}" for k, v in cnt.items()) or "なし")

    save_history(hist_dir, data_date, cards, market, demo, ranking)
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
        "val": (summary or {}).get("validation"),
        "pt": backtest.build_period_table(summary, "periods"), "pt_legacy": backtest.build_period_table(summary, "periods_legacy"),
        "sectors": rankings.rank_sectors(snap), "themes": rankings.rank_themes(snap, themes),
        "n_universe": len(universe), "n_scanned": len(snap), "n_fetched": fetched,
        "td": td_stats, "tdnet_warn": bool(client) and td_stats["checked"] > 0 and td_stats["unavailable"] > td_stats["checked"] / 2,
        "n_excluded_events": len(excluded_rows),
        "movers": mover, "prev_date": prev_date, "mv": mv, "change_summary": summary_changes,
        "hold": {"endpoint": endpoint, "enabled": bool(endpoint), "demo": demo, "n": None if hold_cards is None else len(hold_cards)},
        "pb": pb, "w": cfg["weights"], "tdc": cfg["tdnet"], "ev": cfg["events"], "es": cfg["expected_score"],
        "top_n": cfg["top_n"], "min_turnover_oku": cfg["filters"]["min_turnover_yen"] / 1e8, "min_price": cfg["filters"]["min_price"],
    }
    out = render_site(ctx)
    write_excluded_csv(excluded_rows, out.parent / "excluded_events.csv")
    write_holdings_files(out.parent, universe, hold_cards, hold_key, data_date, today, hc["pbkdf2_iterations"])
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
