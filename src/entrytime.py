"""エントリー時刻別検証。同じ銘柄・同じ売却日で、買う時刻だけを変えると成績がどう変わるかを比べる。

  ① 前日終値（ランキング日の引けで買った場合）  ② 当日寄付き  ③ 10:00  ④ 前場引け（11:30）
  売却は4つとも同じ日の終値（5営業日後／20営業日後）。差は「買う時刻」だけなので、どの時刻が有利かを比べられる。

対象は、過去の各営業日の押し目トップ（技術点のみの再現）。5分足は Yahoo が約60日分しか出さないため、
計算できた分を data/intraday/entry_prices.csv に毎週ためていく（件数は週ごとに増える）。件数が少ないうちは「参考値」。
"""
from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ROOT, Config
from .entry10 import download_5m, session_stats
from .log import get_logger

log = get_logger(__name__)
TYPES = [("prev", "① 前日終値"), ("open", "② 当日寄付き"), ("p1000", "③ 10:00"), ("pam", "④ 前場引け")]
COLS = ["pick_date", "code", "prev_close", "open", "p1000", "pam"]


def build_records(picks: list[tuple[pd.Timestamp, str]], panel_close: pd.DataFrame, panel_open: pd.DataFrame,
                  data5: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """picks（ランキング日, 銘柄コード）ごとに、翌営業日の 前日終値・寄付き・10:00・前場引け の値を作る。"""
    idx = panel_close.index
    rows = []
    for d, code in picks:
        tk = code if code.endswith(".T") else code + ".T"
        if d not in idx or tk not in panel_close.columns:
            continue
        i = idx.get_loc(d)
        if i + 1 >= len(idx):
            continue
        st = session_stats(data5.get(tk), idx[i + 1].date(), time(10, 0), time(9, 50))
        if not st or not st["complete"]:
            continue
        o = float(panel_open[tk].iloc[i + 1])
        pc = float(panel_close[tk].iloc[i])
        if not (o > 0 and pc > 0):
            continue
        rows.append({"pick_date": d.strftime("%Y-%m-%d"), "code": tk, "prev_close": pc, "open": o,
                     "p1000": st["p_cut"], "pam": st["p_am"] if st["p_am"] else np.nan})
    return pd.DataFrame(rows, columns=COLS)


def merge_store(old: pd.DataFrame, new: pd.DataFrame, keep_days: int = 800) -> pd.DataFrame:
    df = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    df = df.drop_duplicates(["pick_date", "code"], keep="last").sort_values(["pick_date", "code"])
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
    return df[df["pick_date"] >= cutoff].reset_index(drop=True)


def compute(rec: pd.DataFrame, panel_close: pd.DataFrame, horizons: list[int], cost: float, min_n: int, t_good: float) -> dict[str, Any]:
    """4つの買い時刻 × 各保有日数の平均騰落率・勝率。同じ取引どうしで比べる（対応のある差のt値つき）。"""
    idx = panel_close.index
    out: dict[str, Any] = {"ok": False, "horizons": horizons, "rows": {}, "n_records": int(len(rec))}
    if rec.empty:
        out["reason"] = "5分足から計算できた件数がまだありません（週次バックテストの実行で増えていきます）"
        return out
    pos = {d: i for i, d in enumerate(idx.strftime("%Y-%m-%d"))}
    for h in horizons:
        r = {k: [] for k, _ in TYPES}
        for row in rec.itertuples(index=False):
            i = pos.get(row.pick_date)
            if i is None or i + h >= len(idx) or row.code not in panel_close.columns:
                continue
            ex = float(panel_close[row.code].iloc[i + h])
            if not np.isfinite(ex) or ex <= 0:
                continue
            vals = {"prev": row.prev_close, "open": row.open, "p1000": row.p1000, "pam": row.pam}
            if not all(np.isfinite(v) and v > 0 for v in vals.values()):
                continue
            for k, v in vals.items():
                r[k].append(ex / v - 1 - cost)
        n = len(r["open"])
        if n == 0:
            out["rows"][str(h)] = None
            continue
        arr = {k: np.array(v) for k, v in r.items()}
        row_out = {}
        for k, label in TYPES:
            a = arr[k]
            d = a - arr["open"]
            t = None
            if k != "open" and n >= 5 and d.std(ddof=1) > 0:
                t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
            row_out[k] = {"label": label, "n": n, "mean": float(a.mean()), "win": float((a > 0).mean()),
                          "diff_open": float(d.mean()), "t": t}
        best = max(row_out, key=lambda k: row_out[k]["mean"])
        out["rows"][str(h)] = {"n": n, "types": row_out, "best": best, "reference": n < min_n}
    avail = [h for h in horizons if out["rows"].get(str(h))]
    if avail:
        out["ok"] = True
        h0 = avail[-1]
        r0 = out["rows"][str(h0)]
        b = r0["types"][r0["best"]]
        sig = b["t"] is not None and abs(b["t"]) >= t_good and r0["best"] != "open"
        out["conclusion"] = (f"{h0}営業日後まで持った場合、平均騰落率が最も高いのは「{b['label']}」で買った場合（{b['mean'] * 100:+.2f}%）でした。"
                             f"当日寄付きとの差は{b['diff_open'] * 100:+.2f}%ポイント"
                             + ("（統計的にも差が出ています）" if sig else "（偶然の範囲かもしれません）") + f"。件数 {r0['n']}件"
                             + ("で少ないため、参考値です。" if r0["reference"] else "。"))
    else:
        out["reason"] = "売却日（5・20営業日後）がまだ来ていない取引ばかりです。日が経つと計算できます"
    return out


def analyze(picks: list[tuple[pd.Timestamp, str]], panel: dict[str, pd.DataFrame], cfg: Config,
            download: Callable[[list[str], str], dict[str, pd.DataFrame]] | None = None,
            prev_store: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """prev_store＝前回までにためた記録（data/backtest_summary.json の entrytime_store）。結果の "store" を、次回に引き継ぐ。
    ※別ファイルに書くとGitHub Actionsの保存で取りこぼすため、summaryのJSONの中に入れてためる。"""
    ec = cfg["entrytime"]
    if not ec.get("enabled", True):
        return {"ok": False, "reason": "設定で無効になっています"}
    old = pd.DataFrame(prev_store or [], columns=COLS)
    if old.empty:                                         # 旧版（CSV保存）からの引き継ぎ（読むだけ）
        legacy = ROOT / ec["store"]
        if legacy.exists():
            try:
                old = pd.read_csv(legacy, dtype={"code": str})[COLS]
            except Exception:  # noqa: BLE001
                old = pd.DataFrame(columns=COLS)
    tickers = sorted({(c if c.endswith(".T") else c + ".T") for _, c in picks})
    new = pd.DataFrame(columns=COLS)
    if tickers:
        try:
            data5 = (download or download_5m)(tickers, "60d")
            new = build_records(picks, panel["close"], panel["open"], data5, cfg)
            log.info("エントリー時刻別検証: 5分足から %d件を新たに計算", len(new))
        except Exception as e:  # noqa: BLE001
            log.warning("5分足の取得・計算に失敗（保存済みの分だけで検証します）: %s", e)
    rec = merge_store(old, new)
    res = compute(rec, panel["close"], [int(h) for h in ec["horizons"]], cfg["backtest"]["cost_pct"] / 100, int(ec["min_n"]), float(ec["t_good"]))
    res["period"] = (rec["pick_date"].min(), rec["pick_date"].max()) if len(rec) else None
    res["store"] = [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()} for r in rec.round(4).to_dict("records")]
    return res
