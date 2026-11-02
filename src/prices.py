"""株価データ（yfinance）の取得。"""
import time

import pandas as pd

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _naive_index(df):
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df


def _extract(raw):
    """yf.download の結果を {Open: 幅広DataFrame, ...} に変換する。"""
    if raw is None or raw.empty:
        return {}
    cols = raw.columns
    if not isinstance(cols, pd.MultiIndex):
        return {}
    lvl = 0 if "Close" in cols.get_level_values(0) else 1
    return {f: raw.xs(f, axis=1, level=lvl) for f in FIELDS if f in cols.get_level_values(lvl)}


def download_prices(tickers, period="2y", chunk_size=150, sleep=1.5,
                    min_success_ratio=0.5, retries=3):
    import yfinance as yf

    parts = {f: [] for f in FIELDS}
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for n, chunk in enumerate(chunks, 1):
        frames = {}
        for attempt in range(retries):
            try:
                raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                frames = _extract(raw)
                if frames:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[warn] chunk {n}: {e}")
            time.sleep(5 * (attempt + 1))
        for f, df in frames.items():
            parts[f].append(df)
        print(f"  価格取得 {n}/{len(chunks)}")
        time.sleep(sleep)

    if not parts["Close"]:
        raise RuntimeError("株価を1銘柄も取得できませんでした（Yahoo側の制限の可能性）")

    panel = {}
    for f in FIELDS:
        df = pd.concat(parts[f], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        panel[f.lower()] = _naive_index(df).sort_index()
    close = panel["close"]
    ok = close.columns[close.notna().sum() > 20]
    ratio = len(ok) / max(len(tickers), 1)
    print(f"  取得できた銘柄: {len(ok)}/{len(tickers)} ({ratio:.0%})")
    if ratio < min_success_ratio:
        raise RuntimeError(f"取得成功率が低すぎます({ratio:.0%})。更新を中止します")
    return {k: v[ok] for k, v in panel.items()}


def download_benchmarks(period="3y"):
    """TOPIX連動ETF(1306)と日経平均。"""
    import yfinance as yf

    out = {}
    for key, tk in (("topix", "1306.T"), ("n225", "^N225")):
        try:
            raw = yf.download(tk, period=period, interval="1d", auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = _naive_index(raw).dropna(subset=["Close"])
            out[key] = raw[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {tk} の取得に失敗: {e}")
    if "topix" not in out and "n225" in out:
        out["topix"] = out["n225"]
    if not out:
        raise RuntimeError("指数データを取得できませんでした")
    return out
