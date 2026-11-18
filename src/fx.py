"""ドル円（USDJPY）の市場参加者別（東京・欧州・NY）の買い圧力・売り圧力の分析。

1時間足（Yahoo Finance JPY=X）から、毎日を3つの時間帯（セッション）に分けて損益（pips）を作る：
  東京 09:00〜15:30 JST ／ 欧州 15:30〜22:00 JST ／ NY 22:00〜翌06:00 JST（開始した日の記録として扱う）
各セッションの「始値→終値」の動き＝その時間帯の参加者が買い（上昇）・売り（下落）のどちらに傾けたか。

単純な時間別の統計ではなく、直近の動き（既定は20営業日）を、勝率・平均pips・期待値・累積損益・連続記録から総合して
『買いスコア／売りスコア』にする。全期間の統計は、直近の動きが普段と比べてどうかを見る基準に使う。

注意：1時間足なので、15:30は15:00〜16:00の足の中間を線形に補間した価格で代用している（約定価格ではない）。
      スプレッド・スリッページは含まない。過去の傾向は将来を保証しない。
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import JST, ROOT, Config, load_config
from .log import get_logger

log = get_logger(__name__)
BAR = pd.Timedelta(hours=1)
WEEKDAYS = ["月", "火", "水", "木", "金"]
SIDE_WORD = {1: "買い", -1: "売り", 0: "中立"}


# ---------------------------------------------------------------- 取得・蓄積
def yahoo_hourly(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval="1h", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close"]].dropna()


def normalize(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars[["Open", "High", "Low", "Close"]].astype(float).dropna().copy()
    b.index = pd.DatetimeIndex(b.index)
    b.index = (b.index.tz_localize("UTC") if b.index.tz is None else b.index).tz_convert(JST)
    b = b[~b.index.duplicated(keep="last")].sort_index()
    return b[(b["High"] >= b["Low"]) & (b["Close"] > 0)]


def load_store(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return normalize(df)


def save_store(path: Path, bars: pd.DataFrame, keep_days: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cut = bars.index.max() - pd.Timedelta(days=keep_days)
    out = bars[bars.index >= cut].copy()
    out.index = out.index.tz_convert("UTC")
    out.round(4).to_csv(path)


def update_store(cfg: Config, download: Callable[[str, str], pd.DataFrame] | None = None) -> tuple[pd.DataFrame, str | None]:
    """蓄積済みの1時間足に、新しく取得した分を足して保存する。取得に失敗しても、蓄積済みのデータで続ける。(足, 警告)"""
    fc = cfg["fx"]
    path = ROOT / fc["store"]
    old = load_store(path)
    period = fc["first_period"] if old.empty else fc["update_period"]
    warn = None
    try:
        new = normalize((download or yahoo_hourly)(fc["ticker"], period))
        if new.empty:
            raise ValueError("データが空でした")
        bars = normalize(pd.concat([old, new])) if not old.empty else new
        save_store(path, bars, int(fc["keep_days"]))
    except Exception as e:  # noqa: BLE001
        log.warning("ドル円の1時間足を取得できません: %s", e)
        bars, warn = old, f"最新のデータを取得できなかったため、前回までのデータで表示しています（{type(e).__name__}）"
    return bars, warn


# ---------------------------------------------------------------- セッション別の損益
def price_at(bars: pd.DataFrame, t: pd.Timestamp) -> float | None:
    """時刻tの価格。足の途中なら、その足の始値→終値の線形補間。足が無い（途切れている）ときは None。"""
    i = bars.index.searchsorted(t, side="right") - 1
    if i < 0:
        return None
    start = bars.index[i]
    dt = t - start
    if dt > BAR:                                                      # ちょうど足の終わり（dt==BAR）は、その足の終値
        return None
    o, c = float(bars["Open"].iloc[i]), float(bars["Close"].iloc[i])
    return o + (c - o) * (dt / BAR)


def _hm(s: str) -> pd.Timedelta:
    h, m = (int(x) for x in s.split(":"))
    return pd.Timedelta(hours=h, minutes=m)


def build_sessions(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """日×セッションの表。complete=False は、まだ終わっていない（進行中の）セッション。"""
    fc = cfg["fx"]
    pip = float(fc["pip"])
    if bars.empty:
        return pd.DataFrame()
    last_end = bars.index[-1] + BAR
    rows: list[dict[str, Any]] = []
    for d in pd.date_range(bars.index[0].normalize(), bars.index[-1].normalize(), freq="D"):
        if d.weekday() > 4:                                           # 土日に「始まる」セッションは無い
            continue
        for key, s in fc["sessions"].items():
            start = d + _hm(s["start"])
            end = d.normalize() + _hm(s["end"])
            if end <= start:
                end += pd.Timedelta(days=1)
            if start >= last_end:
                continue
            win = bars[(bars.index >= start.floor("h")) & (bars.index < end)]
            expected = math.ceil((end - start) / BAR)
            inside = int(((win.index >= start) & (win.index < end)).sum())
            complete = end <= last_end
            if inside < (max(2, math.ceil(expected * float(fc["min_coverage"]))) if complete else 1):     # 進行中は1本でもあれば「途中経過」として扱う
                continue
            o = price_at(bars, start)
            c = price_at(bars, end) if complete else float(win["Close"].iloc[-1])
            if o is None or c is None:
                continue
            hi, lo = max(float(win["High"].max()), o, c), min(float(win["Low"].min()), o, c)
            rng = (hi - lo) / pip
            rows.append({"date": d.date(), "session": key, "weekday": d.weekday(), "open": o, "close": c, "high": hi, "low": lo,
                         "pips": (c - o) / pip, "range": rng, "clv": (c - lo) / (hi - lo) if hi > lo else 0.5, "complete": complete})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 統計
def streak(p: pd.Series) -> int:
    """直近の連続日数。上昇が続いていればプラス、下落ならマイナス（横ばいで途切れる）。"""
    n, sign = 0, 0
    for v in reversed(p.to_list()):
        s = 1 if v > 0 else -1 if v < 0 else 0
        if s == 0 or (sign and s != sign):
            break
        sign, n = s, n + 1
    return sign * n


def max_streaks(p: pd.Series) -> tuple[int, int]:
    best = {1: 0, -1: 0}
    run, sign = 0, 0
    for v in p.to_list():
        s = 1 if v > 0 else -1 if v < 0 else 0
        run = run + 1 if s and s == sign else (1 if s else 0)
        sign = s
        if s:
            best[s] = max(best[s], run)
    return best[1], best[-1]


def stats(p: pd.Series, rng: pd.Series | None = None, clv: pd.Series | None = None) -> dict[str, Any]:
    """pips（買い持ちの損益）の系列から集計する。"""
    n = len(p)
    if n == 0:
        return {"n": 0}
    up, down = int((p > 0).sum()), int((p < 0).sum())
    w, l = p[p > 0], p[p < 0]
    mu = float(p.mean())
    sd = float(p.std(ddof=1)) if n > 1 else 0.0
    mx_up, mx_dn = max_streaks(p)
    return {"n": n, "up": up, "down": down, "flat": n - up - down, "win": up / n,
            "avg_move": float(p.abs().mean()), "avg_up": float(w.mean()) if len(w) else 0.0, "avg_down": float(l.mean()) if len(l) else 0.0,
            "avg_range": float(rng.mean()) if rng is not None and len(rng) else None,
            "cum": float(p.sum()), "exp": mu, "t": (mu / (sd / math.sqrt(n))) if sd > 0 else 0.0,
            "pf": (float(w.sum()) / abs(float(l.sum()))) if len(l) and l.sum() != 0 else None,
            "streak": streak(p), "max_up": mx_up, "max_down": mx_dn,
            "clv": float(clv.mean()) if clv is not None and len(clv) else None}


# ---------------------------------------------------------------- スコア（買い/売りの強さ）
def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def score_session(recent: pd.Series, base: pd.Series, cfg: Config) -> dict[str, Any]:
    """直近の動き（勝率・平均pips・期待値・累積損益・連続記録）を総合して -1（強い売り）〜+1（強い買い）の指数にする。
    単純な勝率順にならないよう、動きの大きさ（平均pips・期待値・累積）と連続記録も同じ重みで見る。"""
    sc = cfg["fx"]["score"]
    n = len(recent)
    if n < int(sc["min_days"]):
        return {"ok": False, "why": f"日数が少ない（{n}日）"}
    typical = max(float(base.abs().mean()) if len(base) else 0.0, 1e-9)       # このセッションの普段の動きの大きさ
    up, down = int((recent > 0).sum()), int((recent < 0).sum())
    wins, losses = float(recent[recent > 0].sum()), abs(float(recent[recent < 0].sum()))
    comp = {
        "win": _clip(((up / (up + down) if up + down else 0.5) - 0.5) / 0.30),                       # 勝率：上昇日の割合（80%で満点）
        "avg": math.tanh(float(recent.mean()) / typical * 3.0),                                       # 平均pips：普段の動きに対する大きさ
        "exp": _clip((wins - losses) / (wins + losses)) if wins + losses else 0.0,                    # 期待値：利益と損失の偏り（ペイオフ込み）
        "cum": math.tanh(float(recent.sum()) / (typical * math.sqrt(n)) / 2.0),                       # 累積損益：偶然では説明しにくい偏りか
        "streak": _clip(streak(recent) / float(sc["streak_full"])),                                   # 連続記録：同じ向きが続いているか
    }
    w = sc["weights"]
    s = sum(float(w[k]) * v for k, v in comp.items()) / sum(float(x) for x in w.values())
    return {"ok": True, "s": s, "score": int(round(abs(s) * 100)), "comp": comp, "typical": typical}


def judge(s: float, cfg: Config) -> tuple[int, str, str]:
    """(向き, 強さ, 表示の文言)。向き：+1買い／-1売り／0中立"""
    t = cfg["fx"]["score"]["levels"]
    a = abs(s)
    if a < float(t["mild"]):
        return 0, "none", "中立"
    side = 1 if s > 0 else -1
    word = SIDE_WORD[side]
    if a >= float(t["strong"]):
        return side, "strong", f"強い{word}優勢"
    if a >= float(t["normal"]):
        return side, "normal", f"{word}優勢"
    return side, "mild", f"やや{word}優勢"


def trend_text(recent: pd.Series, name: str, k: int) -> dict[str, Any]:
    """直近k日：勝敗・平均pips・連続日数と、そこからの判断。"""
    r = recent.iloc[-k:]
    up, down = int((r > 0).sum()), int((r < 0).sum())
    avg = float(r.mean()) if len(r) else 0.0
    st = streak(r)
    side = 1 if up >= 0.625 * len(r) and avg > 0 else -1 if down >= 0.625 * len(r) and avg < 0 else 0
    win_for = up if side >= 0 else down
    lose_for = down if side >= 0 else up
    verdict = f"{name}勢{SIDE_WORD[side]}優勢" if side else f"{name}勢は拮抗（方向感なし）"
    streak_txt = f"{abs(st)}日連続{'上昇' if st > 0 else '下落'}" if abs(st) >= 2 else "連続なし"
    return {"k": len(r), "up": up, "down": down, "avg": avg, "streak": st, "side": side, "verdict": verdict,
            "line1": f"{'買い' if side >= 0 else '売り'}側から見て {win_for}勝{lose_for}敗" if side else f"上昇{up}日・下落{down}日",
            "streak_txt": streak_txt}


# ---------------------------------------------------------------- 全体の分析
def analyze(sessions: pd.DataFrame, cfg: Config, bars_last: pd.Timestamp | None = None) -> dict[str, Any]:
    fc = cfg["fx"]
    names = {k: v["name"] for k, v in fc["sessions"].items()}
    if sessions.empty or not (sessions["complete"]).any():
        return {"ok": False, "reason": "分析できるデータがありません"}
    done = sessions[sessions["complete"]]
    n_recent, k = int(fc["recent_days"]), int(fc["trend_days"])
    out: dict[str, Any] = {"ok": True, "sessions": {}, "names": names, "recent_days": n_recent}
    for key in fc["sessions"]:
        d = done[done["session"] == key].sort_values("date")
        if d.empty:
            continue
        p = d["pips"].reset_index(drop=True)
        info: dict[str, Any] = {"key": key, "name": names[key], "windows": {}}
        for label, n in (("recent", n_recent), ("mid", int(fc["mid_days"])), ("all", None)):
            dd = d.iloc[-n:] if n else d
            info["windows"][label] = {"label": {"recent": f"直近{n_recent}営業日", "mid": f"直近{fc['mid_days']}営業日", "all": "全期間"}[label],
                                      **stats(dd["pips"], dd["range"], dd["clv"])}
        recent = d.iloc[-n_recent:]["pips"]
        sc = score_session(recent, p, cfg)
        info["score"] = sc
        if sc["ok"]:
            info["side"], info["strength"], info["label"] = judge(sc["s"], cfg)
        else:
            info["side"], info["strength"], info["label"] = 0, "none", "判定できません"
        info["trend"] = trend_text(p, names[key], k)
        info["last_days"] = [{"date": str(r.date), "pips": round(float(r.pips), 1)} for r in d.iloc[-n_recent:].itertuples()]
        info["cum_curve"] = [round(float(x), 1) for x in d.iloc[-int(fc["mid_days"]):]["pips"].cumsum()]
        info["last_day"] = {"date": str(d.iloc[-1]["date"]), "pips": round(float(d.iloc[-1]["pips"]), 1)}
        live = sessions[(sessions["session"] == key) & (~sessions["complete"])]
        info["live"] = None if live.empty else {"date": str(live.iloc[-1]["date"]), "pips": round(float(live.iloc[-1]["pips"]), 1)}
        # 曜日別（全期間）
        wd = []
        for w_ in range(5):
            x = d[d["weekday"] == w_]["pips"]
            st = stats(x) if len(x) else {"n": 0}
            wd.append({"weekday": WEEKDAYS[w_], **st})
        info["weekdays"] = wd
        out["sessions"][key] = info
    rank = sorted((v for v in out["sessions"].values() if v["score"]["ok"]), key=lambda v: -v["score"]["score"])
    out["ranking"] = [{"rank": i, "name": v["name"], "key": v["key"], "side": v["side"], "word": SIDE_WORD[v["side"]],
                       "score": v["score"]["score"], "label": v["label"]} for i, v in enumerate(rank, 1)]
    # 総合：セッションごとの普段の動きの大きさで重みづけして、向きを合算
    tot = sum(v["score"]["typical"] for v in rank) or 1.0
    net = sum(v["score"]["s"] * v["score"]["typical"] for v in rank) / tot if rank else 0.0
    nside, nstrength, nlabel = judge(net, cfg)
    out["net"] = {"s": net, "side": nside, "strength": nstrength,
                  "headline": {1: "いまは買い相場", -1: "いまは売り相場", 0: "いまは方向感なし"}[nside], "label": nlabel,
                  "score": int(round(abs(net) * 100))}
    leader = rank[0] if rank and rank[0]["side"] != 0 else None
    out["leader"] = None if leader is None else {"name": leader["name"], "side": leader["side"], "word": SIDE_WORD[leader["side"]], "score": leader["score"]["score"]}
    sides = {v["side"] for v in rank if v["side"] != 0}
    out["split"] = len(sides) > 1
    parts = [f"{v['name']}勢は{v['label']}" for v in rank]
    out["summary"] = "。".join(parts) + "。" if parts else ""
    if out["split"]:
        out["note"] = "市場ごとに向きが割れています。時間帯によって優勢な側が入れ替わる相場です。"
    elif leader:
        out["note"] = f"{leader['name']}勢が{SIDE_WORD[leader['side']]}で相場を主導しています。"
    else:
        out["note"] = "どの市場参加者も強い方向感を出していません。"
    out["last_date"] = str(done["date"].max())
    out["n_days"] = int(done["date"].nunique())
    out["period"] = {"start": str(done["date"].min()), "end": str(done["date"].max())}
    return out


# ---------------------------------------------------------------- 図（SVG）
def pips_bars_svg(days: list[dict], w: int = 300, h: int = 64) -> str:
    """直近の日ごとのpips（買い＝緑・売り＝赤）。最新が右端。"""
    if not days:
        return ""
    vals = [d["pips"] for d in days]
    m = max(max(abs(v) for v in vals), 1e-9)
    bw = w / len(vals)
    mid = h / 2
    out = [f'<svg class="fxbars" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="直近の日ごとの値動き（pips）"><line class="zero" x1="0" x2="{w}" y1="{mid}" y2="{mid}"/>']
    for i, v in enumerate(vals):
        hh = abs(v) / m * (mid - 2)
        y = mid - hh if v >= 0 else mid
        out.append(f'<rect class="{"up" if v >= 0 else "dn"}" x="{i * bw + 1:.1f}" y="{y:.1f}" width="{max(bw - 2, 1):.1f}" height="{max(hh, 0.8):.1f}"><title>{days[i]["date"]} {v:+.1f}pips</title></rect>')
    out.append("</svg>")
    return "".join(out)


def cum_svg(curves: list[tuple[str, list[float]]], w: int = 360, h: int = 210) -> str:
    """セッション別の累積pips（買い持ちの累積損益）。上がっている＝そのセッションは買いが優勢、下がっている＝売りが優勢。"""
    curves = [(n, c) for n, c in curves if len(c) >= 2]
    if not curves:
        return ""
    allv = [0.0] + [v for _, c in curves for v in c]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    L, R, T, B = 44, 8, 10, 16
    Y = lambda v: T + (hi - v) / span * (h - T - B)  # noqa: E731
    X = lambda i, m: L + i / max(m - 1, 1) * (w - L - R)  # noqa: E731
    out = [f'<svg class="fxcum" viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="セッション別の累積pips">',
           f'<line class="zero" x1="{L}" x2="{w - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>']
    for v in sorted({round(lo), 0, round(hi)}):
        out.append(f'<text class="tick" x="{L - 4}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:+d}</text>')
    for j, (_, c) in enumerate(curves):
        pts = " ".join(f"{X(i, len(c)):.1f},{Y(v):.1f}" for i, v in enumerate(c))
        out.append(f'<polyline class="ln s{j}" fill="none" stroke-width="2.2" points="{pts}"/>')
    out.append("</svg>")
    leg = "".join(f'<span class="lg s{j}">{n}</span>' for j, (n, _) in enumerate(curves))
    return "".join(out) + f'<div class="legend">{leg}</div>'


# ---------------------------------------------------------------- ページ生成
def page_context(res: dict[str, Any], warn: str | None, bars_last: pd.Timestamp | None, generated: datetime, demo: bool) -> dict[str, Any]:
    ctx: dict[str, Any] = {"res": res, "warn": warn, "demo": demo, "generated": f"{generated:%Y-%m-%d %H:%M}",
                           "bars_last": None if bars_last is None else f"{bars_last.tz_convert(JST) + BAR:%m/%d %H:%M}",
                           "today_wd": generated.astimezone(JST).weekday() if generated.astimezone(JST).weekday() < 5 else None}
    if res.get("ok"):
        ctx["bars_charts"] = {k: pips_bars_svg(v["last_days"]) for k, v in res["sessions"].items()}
        ctx["cum_chart"] = cum_svg([(v["name"], v["cum_curve"]) for v in res["sessions"].values()])
        ctx["order"] = [k for k in ("tokyo", "europe", "ny") if k in res["sessions"]]
        if bars_last is not None:
            age_h = (generated.astimezone(JST) - (bars_last.tz_convert(JST) + BAR).to_pydatetime()).total_seconds() / 3600
            ctx["stale"] = age_h > 54 or (age_h > 30 and generated.weekday() < 5 and generated.weekday() != 0)
    return ctx


def write_page(ctx: dict[str, Any], sessions: pd.DataFrame, out_dir: Path) -> Path:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=select_autoescape(["html"]))
    env.filters["pips"] = lambda v, d=1: "—" if v is None or v != v else f"{v:+.{d}f}"
    env.filters["pct0"] = lambda v: "—" if v is None or v != v else f"{v * 100:.0f}%"
    import shutil
    shutil.copy(ROOT / "static" / "style.css", out_dir.parent / "style.css")                 # 株のページと同じ見た目（../style.css）
    (out_dir / "index.html").write_text(env.get_template("fx.html").render(**ctx), encoding="utf-8")
    if not sessions.empty:
        sessions.drop(columns=["open", "close", "high", "low"]).round(2).to_csv(out_dir / "usdjpy_sessions.csv", index=False, encoding="utf-8-sig")
    return out_dir / "index.html"


# ---------------------------------------------------------------- 人工データ（デモ・テスト用）
def demo_bars(days: int = 520, seed: int = 7) -> pd.DataFrame:
    """架空の1時間足。東京は上がりやすく・欧州は下がりやすく・NYは上がりやすい癖をつけ、直近は欧州が売り優勢。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp("2026-11-18 06:00", tz=JST), periods=days * 24, freq="h")
    idx = idx[[not ((t.weekday() == 5 and t.hour >= 6) or t.weekday() == 6 or (t.weekday() == 0 and t.hour < 6)) for t in idx]]
    px, rows = 150.0, []
    for t in idx:
        hm = t.hour + t.minute / 60
        drift = 0.012 if 9 <= hm < 15.5 else (-0.015 if 15.5 <= hm < 22 else 0.010)
        if (idx[-1] - t) < pd.Timedelta(days=12) and 15.5 <= hm < 22:
            drift -= 0.05
        o = px
        c = o + drift + rng.normal(0, 0.06)
        rows.append((o, max(o, c) + abs(rng.normal(0, 0.02)), min(o, c) - abs(rng.normal(0, 0.02)), c))
        px = c
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


# ---------------------------------------------------------------- 実行
def run(demo: bool = False, out_dir: Path | None = None) -> Path:
    cfg = load_config()
    now = datetime.now(JST)
    if demo:
        bars, warn = demo_bars(), None
    else:
        bars, warn = update_store(cfg)
    sessions = build_sessions(bars, cfg) if not bars.empty else pd.DataFrame()
    res = analyze(sessions, cfg) if not sessions.empty else {"ok": False, "reason": "1時間足のデータがありません（初回の取得に失敗した可能性があります）"}
    ctx = page_context(res, warn, bars.index[-1] if not bars.empty else None, now, demo)
    path = write_page(ctx, sessions, out_dir or ROOT / "docs" / "fx")
    log.info("ドル円ページを出力しました: %s", path)
    return path


if __name__ == "__main__":
    run(demo="--demo" in sys.argv)
