"""10時エントリー判定。朝のランキング（順位・点数）は一切変えず、寄り付き後の値動きだけで 🟢エントリー可 / 🟡様子見 / 🔴見送り を付ける。

    朝 7:10  ランキング生成（docs/latest.json） → 9:00 寄り付き → 10:00 までの5分足で判定 → docs/entry.json に出力
    画面（static/entry.js）が entry.json を読んで各銘柄カードに「10時判定」を後から追記する。順位は動かさない。

判定に使う4項目
  ① ギャップ率      = 当日の寄り付き ÷ 前日終値 − 1
  ② 寄付き後騰落率  = 10:00の値 ÷ 当日の寄り付き − 1
  ③ 出来高進捗      = 10:00までの出来高 ÷ 普段の同じ時間帯（寄り〜10:00）の出来高（過去の中央値）
  ④ 市場環境        = TOPIX・日経平均・グロース250・日経225先物・TOPIX先物（弱いときは 🟢 を 🟡 に下げるだけ）
目的は「有望銘柄の発見」ではなく「高値掴みを避けて、期待値の高いタイミングで入ること」。買い推奨ではなく、最終判断はご自身で。
Yahooの日本株は約20分遅れのことがあるため、10:15ごろに実行して10:27まで足の到着を待ち、実際に取れた最後の足の時刻を「基準時刻」として表示する。
"""
from __future__ import annotations

import json
import sys
import time as _time
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import JST, ROOT, Config, load_config
from .log import get_logger

log = get_logger(__name__)
Download5m = Callable[[list[str], str], dict[str, pd.DataFrame]]      # (tickers, period) → {ticker: 5分足(JST・Open/High/Low/Close/Volume)}

JUDGE = {
    "go":   {"emoji": "🟢", "label": "エントリー可"},
    "wait": {"emoji": "🟡", "label": "様子見"},
    "skip": {"emoji": "🔴", "label": "見送り"},
    "na":   {"emoji": "⚪", "label": "判定不能"},
}


# ---------------------------------------------------------------- 取得
def download_5m(tickers: list[str], period: str = "10d", batch: int = 40) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch):
        part = tickers[i:i + batch]
        try:
            raw = yf.download(part, period=period, interval="5m", auto_adjust=False, progress=False,
                              group_by="ticker", threads=True)
        except Exception as e:  # noqa: BLE001
            log.warning("5分足の取得に失敗（%s…）: %s", part[0], e)
            continue
        for tk in part:
            try:
                df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(subset=["Close"])
                if df.empty:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(JST)
                out[tk] = df[["Open", "High", "Low", "Close", "Volume"]]
            except Exception:  # noqa: BLE001
                continue
    return out


# ---------------------------------------------------------------- 1銘柄の指標（純粋関数）
def session_stats(df: pd.DataFrame | None, day, cutoff: time = time(10, 0), min_last_start: time = time(9, 50),
                  am_close: time = time(11, 30)) -> dict[str, Any] | None:
    """day（date）の寄り付き・cutoff時点の値・累積出来高・前場引けの値。足の時刻は「開始時刻」（9:55の足＝9:55〜10:00）。"""
    if df is None or df.empty:
        return None
    d = df[df.index.date == day]
    d = d[d.index.time >= time(9, 0)]
    if d.empty:
        return None
    o = float(d["Open"].iloc[0])
    pre = d[d.index.time < cutoff]
    if pre.empty:
        return None
    last_start = pre.index[-1]
    res: dict[str, Any] = {"open": o, "p_cut": float(pre["Close"].iloc[-1]), "vol_cut": float(pre["Volume"].sum()),
                           "asof": (last_start + timedelta(minutes=5)).strftime("%H:%M"),
                           "complete": last_start.time() >= min_last_start}
    am = d[d.index.time < am_close]
    res["p_am"] = float(am["Close"].iloc[-1]) if len(am) and am.index[-1].time() >= time(11, 20) else None
    return res


def normal_volume(df: pd.DataFrame | None, day, cutoff: time = time(10, 0), n_days: int = 10, min_days: int = 3) -> float | None:
    """普段の「寄り〜cutoff」の出来高 = 直前 n_days 営業日の中央値。日数が足りなければ None。"""
    if df is None or df.empty:
        return None
    vals = []
    for d in sorted({x for x in df.index.date if x < day}, reverse=True):
        s = df[(df.index.date == d) & (df.index.time >= time(9, 0)) & (df.index.time < cutoff)]
        if len(s) >= 10 and s["Volume"].sum() > 0:           # 9:00〜10:00は12本。大きく欠けた日は使わない
            vals.append(float(s["Volume"].sum()))
        if len(vals) >= n_days:
            break
    return float(np.median(vals)) if len(vals) >= min_days else None


# ---------------------------------------------------------------- 市場環境
def market_state(readings: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    """readings: [{key,label,pct,futures}]。先物の重みを大きくした加重平均（−1〜+1）で weak / neutral / strong。"""
    mc = cfg["entry10"]["market"]
    num = den = 0.0
    items = []
    for r in readings:
        pct = r.get("pct")
        if pct is None:
            items.append({**r, "status": "missing", "status_label": "取得できず"})
            continue
        weak, strong = (mc["fut_weak"], mc["fut_strong"]) if r.get("futures") else (mc["idx_weak"], mc["idx_strong"])
        st = "weak" if pct <= weak else "strong" if pct >= strong else "neutral"
        w = float(mc["fut_weight"] if r.get("futures") else 1)
        num += w * {"weak": -1, "neutral": 0, "strong": 1}[st]
        den += w
        items.append({**r, "status": st, "status_label": {"weak": "弱い", "neutral": "横ばい", "strong": "強い"}[st]})
    score = num / den if den else None
    if score is None:
        key = "unknown"
    elif score <= float(mc["weak_score"]):
        key = "weak"
    elif score >= float(mc["strong_score"]):
        key = "strong"
    else:
        key = "neutral"
    label = {"weak": "弱い（新規は慎重に）", "neutral": "中立", "strong": "強い", "unknown": "判定できず"}[key]
    return {"key": key, "label": label, "score": score, "items": items}


# ---------------------------------------------------------------- 判定（純粋関数）
def vol_label(vr: float | None, ec: dict[str, Any]) -> str:
    if vr is None:
        return "不明"
    if vr >= ec["vol_hot"]:
        return "過熱"
    if vr >= ec["vol_ok"]:
        return "良好"
    if vr >= ec["vol_thin"]:
        return "やや少ない"
    return "少ない"


def judge(gap: float | None, ret: float | None, vr: float | None, mkt_key: str, cfg: Config) -> dict[str, Any]:
    """gap・ret は%、vr は倍率。理由つきで 🟢/🟡/🔴 を返す。"""
    ec = cfg["entry10"]
    if gap is None or ret is None:
        return {"key": "na", "reasons": ["寄り付き後の値が取れませんでした"], "vol_label": vol_label(vr, ec)}
    total = gap + ret                                          # 前日終値からの上昇（高値掴みの目安）
    skip, wait, good = [], [], []
    if gap >= ec["gap_skip"]:
        skip.append(f"大幅ギャップアップ（{gap:+.1f}%）")
    if ret >= ec["ret_skip"]:
        skip.append(f"寄り付き直後の急騰（{ret:+.1f}%）")
    if total >= ec["total_skip"] and not skip:
        skip.append(f"前日終値から{total:+.1f}%上昇済み（高値掴みの恐れ）")
    if vr is not None and vr >= ec["vol_hot"] and (gap >= ec["gap_ok"] or ret >= ec["hot_ret"]):
        skip.append(f"出来高が普段の{vr:.1f}倍で過熱感あり")
    if ret <= ec["ret_fall_skip"]:
        skip.append(f"寄り付き後に急落中（{ret:+.1f}%）")
    if skip:
        return {"key": "skip", "reasons": skip, "vol_label": vol_label(vr, ec)}
    ok = True
    if gap > ec["gap_ok"]:
        ok = False
        wait.append(f"ギャップアップがやや大きい（{gap:+.1f}%）")
    elif gap < ec["gap_down_wait"]:
        ok = False
        wait.append(f"ギャップダウン（{gap:+.1f}%）。下げ止まりを確認")
    if ret > ec["ret_ok_max"]:
        ok = False
        wait.append(f"寄り付き後の上昇が大きい（{ret:+.1f}%）。押しを待つ")
    elif ret < ec["ret_ok_min"]:
        ok = False
        wait.append(f"寄り付き後に下げている（{ret:+.1f}%）。方向が定まらない")
    if vr is None:
        ok = False
        wait.append("普段の出来高が不明で出来高を評価できません")
    elif vr < ec["vol_ok"]:
        ok = False
        wait.append(f"出来高が少ない（普段の{vr:.1f}倍）")
    elif vr >= ec["vol_hot"]:
        ok = False
        wait.append(f"出来高が多すぎる（普段の{vr:.1f}倍）")
    if ok:
        good.append(f"ギャップ{gap:+.1f}%・寄付き後{ret:+.1f}%で落ち着き、出来高も良好")
        if mkt_key == "weak":
            return {"key": "wait", "reasons": ["個別は良好だが、市場環境が弱いため様子見に下げました"], "vol_label": vol_label(vr, ec), "mkt_downgrade": True}
        return {"key": "go", "reasons": good, "vol_label": vol_label(vr, ec)}
    return {"key": "wait", "reasons": wait, "vol_label": vol_label(vr, ec)}


def evaluate_stock(code: str, df: pd.DataFrame | None, day, prev_close: float | None, mkt_key: str, cfg: Config) -> dict[str, Any]:
    ec = cfg["entry10"]
    cut = time.fromisoformat(ec["cutoff"])
    st = session_stats(df, day, cut, time.fromisoformat(ec["min_last_bar"]))
    if st is None:
        return {"code": code, "key": "na", **JUDGE["na"], "reasons": ["本日の5分足がありません"], "gap": None, "ret": None, "vr": None, "vol_label": "不明"}
    pc = prev_close if prev_close and prev_close > 0 else None
    gap = (st["open"] / pc - 1) * 100 if pc else None
    ret = (st["p_cut"] / st["open"] - 1) * 100 if st["open"] > 0 else None
    nv = normal_volume(df, day, cut, int(ec["vol_days"]), int(ec["min_vol_days"]))
    vr = st["vol_cut"] / nv if nv else None
    j = judge(gap, ret, vr, mkt_key, cfg)
    return {"code": code, "key": j["key"], **JUDGE[j["key"]], "gap": _r(gap), "ret": _r(ret), "vr": _r(vr, 2),
            "vol_label": j["vol_label"], "reasons": j["reasons"], "asof": st["asof"], "complete": st["complete"],
            "price": round(st["p_cut"], 1), "open": round(st["open"], 1)}


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ---------------------------------------------------------------- 市場環境の取得
def fetch_market(cfg: Config, day, prev_data_date, now: datetime, download: Download5m, fut_download=None) -> dict[str, Any]:
    from . import premarket

    ec = cfg["entry10"]
    cut = time.fromisoformat(ec["cutoff"])
    readings = []
    idx = ec["market"]["indices"]
    data = download([i["ticker"] for i in idx], "7d")
    for i in idx:
        df = data.get(i["ticker"])
        pct, asof = None, None
        if df is not None and not df.empty:
            today = df[(df.index.date == day) & (df.index.time < cut)]
            prev = df[df.index.date < day]
            if not today.empty and not prev.empty:
                pct = (float(today["Close"].iloc[-1]) / float(prev["Close"].iloc[-1]) - 1) * 100
                asof = (today.index[-1] + timedelta(minutes=5)).strftime("%H:%M")
        readings.append({"key": i["key"], "label": i["label"], "pct": _r(pct), "asof": asof, "futures": False})
    ref = datetime.combine(prev_data_date, time(15, 30), tzinfo=JST)
    fdl = fut_download or premarket.yahoo_download
    asof_dt = datetime.combine(day, cut, tzinfo=JST)
    for f in ec["market"]["futures"]:
        pct, asof = None, None
        try:
            s = fdl(f["ticker"], "7d", "15m")
            s = s[s.index <= asof_dt]
            r = premarket.change_since_tokyo_close(s, ref, asof_dt, 6.0)
            if r["status"] == "ok":
                pct, asof = r["pct"], r["asof"]
        except Exception as e:  # noqa: BLE001
            log.warning("%s を取得できません: %s", f["label"], e)
        readings.append({"key": f["key"], "label": f["label"], "pct": _r(pct), "asof": asof, "futures": True})
    return market_state(readings, cfg)


# ---------------------------------------------------------------- 全体
def build(cfg: Config, latest: dict[str, Any], now: datetime, download: Download5m, fut_download=None) -> dict[str, Any]:
    """latest.json の内容と5分足から entry.json の中身を作る。"""
    ec = cfg["entry10"]
    picks = latest.get("picks", [])
    ranking_date = pd.Timestamp(latest["data_date"]).date()
    day = now.date()
    out: dict[str, Any] = {"generated_at": now.isoformat(timespec="minutes"), "trade_date": str(day),
                           "ranking_data_date": str(ranking_date), "cutoff": ec["cutoff"], "status": "ok", "items": {}}
    tickers = [p["code"] + ".T" for p in picks]
    data = download(tickers, f"{int(ec['vol_days']) + 6}d") if tickers else {}
    mkt = fetch_market(cfg, day, ranking_date, now, download, fut_download)
    out["market"] = mkt
    n_today = 0
    for p in picks:
        df = data.get(p["code"] + ".T")
        item = evaluate_stock(p["code"], df, day, p.get("close"), mkt["key"], cfg)
        item["rank"] = p["rank"]
        out["items"][p["code"]] = item
        n_today += item["key"] != "na"
    if not picks:
        out["status"] = "no_picks"
    elif n_today == 0:
        out["status"] = "no_data"                      # 休場日・まだ寄り付き前・取得失敗
    counts = {k: sum(1 for i in out["items"].values() if i["key"] == k) for k in JUDGE}
    out["counts"] = counts
    out["asof"] = max((i["asof"] for i in out["items"].values() if i.get("asof")), default=None)
    return out


def wait_for_bars(cfg: Config, download: Download5m, now_fn=lambda: datetime.now(JST), sleep=_time.sleep) -> bool:
    """10:00までの足が出そろうまで待つ（Yahooの遅延・GitHubの起動遅れ対策）。待つのは deadline（10:27）まで。
    すでに deadline を過ぎて起動したときは待たずに、取れた分で判定する。"""
    ec = cfg["entry10"]
    probe = ec["market"]["indices"][0]["ticker"]
    need, cut = time.fromisoformat(ec["min_last_bar"]), time.fromisoformat(ec["cutoff"])
    deadline = time.fromisoformat(ec["deadline"])
    last = None
    while True:
        now = now_fn()
        df = download([probe], "2d").get(probe)
        if df is not None and not df.empty:
            d = df[(df.index.date == now.date()) & (df.index.time < cut)]
            last = d.index[-1].strftime("%H:%M") if not d.empty else None
            if not d.empty and d.index[-1].time() >= need:
                return True
        if now.time() >= deadline:
            return False
        log.info("10:00までの5分足を待っています（取得できた最後の足: %s）", last or "なし")
        sleep(60)


def already_done(out: Path, today: str) -> bool:
    """今日の判定が、10:00までの足がそろった状態で出来ているか（予備の実行を省くため）。"""
    try:
        d = json.loads((out / "entry.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    its = [i for i in d.get("items", {}).values() if i.get("key") != "na"]
    return d.get("trade_date") == today and d.get("status") == "ok" and bool(its) and all(i.get("complete") for i in its)


def run(demo: bool = False, out_dir: Path | None = None, wait: bool = True, skip_if_done: bool = False) -> dict[str, Any]:
    cfg = load_config()
    out = out_dir or ROOT / "docs"
    if skip_if_done and not demo and already_done(out, datetime.now(JST).date().isoformat()):
        log.info("今日の10時判定は完了済みのため、この予備の実行は何もしません")
        return json.loads((out / "entry.json").read_text(encoding="utf-8"))
    t0 = _time.perf_counter()
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    now = datetime.now(JST)
    if demo:
        res = demo_entry(latest, now)
    else:
        if wait:
            if not wait_for_bars(cfg, download_5m):
                log.warning("待機時間内に10:00までの足がそろいませんでした。取得できた分で判定します")
        res = build(cfg, latest, datetime.now(JST), download_5m)
    res["seconds"] = round(_time.perf_counter() - t0, 1)
    (out / "entry.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log_dir = ROOT / "data" / "entry_log"
    if not demo and res["status"] == "ok":
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{res['trade_date']}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    log.info("10時判定: %s（%.1f秒）", res["counts"], res["seconds"])
    return res


def demo_entry(latest: dict[str, Any], now: datetime) -> dict[str, Any]:
    """画面確認用の架空の判定（実在の値ではありません）。"""
    cycle = [("go", 0.4, 0.3, 1.2), ("wait", 0.2, -0.9, 0.5), ("skip", 4.2, 1.8, 3.1), ("go", -0.3, 0.6, 0.9), ("wait", 2.1, 0.2, 1.4)]
    items = {}
    for n, p in enumerate(latest.get("picks", [])):
        k, g, r, v = cycle[n % len(cycle)]
        reasons = {"go": ["落ち着いていて出来高も良好（デモ）"], "wait": ["方向が定まらない（デモ）"], "skip": ["過熱感あり（デモ）"]}[k]
        items[p["code"]] = {"code": p["code"], "rank": p["rank"], "key": k, **JUDGE[k], "gap": g, "ret": r, "vr": v,
                            "vol_label": "過熱" if k == "skip" else "良好" if v >= 0.7 else "やや少ない", "reasons": reasons, "asof": "10:00", "complete": True}
    counts = {k: sum(1 for i in items.values() if i["key"] == k) for k in JUDGE}
    mk = [{"key": "topix", "label": "TOPIX", "pct": 0.2, "status_label": "横ばい"}, {"key": "n225", "label": "日経平均", "pct": 0.3, "status_label": "横ばい"},
          {"key": "growth", "label": "グロース250", "pct": -0.1, "status_label": "横ばい"}, {"key": "n225f", "label": "日経225先物", "pct": 0.4, "status_label": "横ばい"},
          {"key": "topixf", "label": "TOPIX先物", "pct": 0.3, "status_label": "横ばい"}]
    return {"generated_at": now.isoformat(timespec="minutes"), "trade_date": str(now.date()), "ranking_data_date": latest["data_date"],
            "cutoff": "10:00", "status": "ok", "demo": True, "items": items, "counts": counts, "asof": "10:00",
            "market": {"key": "neutral", "label": "中立", "score": 0.0, "items": mk}}


if __name__ == "__main__":
    run(demo="--demo" in sys.argv, wait="--nowait" not in sys.argv, skip_if_done="--skip-if-done" in sys.argv)
