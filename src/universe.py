"""対象銘柄リスト（JPX公式の上場銘柄一覧）を取得する。"""
from __future__ import annotations
import io
import time

import pandas as pd
import requests

from .config import ROOT
from .log import get_logger

log = get_logger(__name__)

# JPXは公開ファイルの形式を変えることがあるため、新しい順に試す
_BASE = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/"
JPX_URLS = [_BASE + "data_j.xlsx", _BASE + "data_j.xls"]
CACHE = ROOT / "data" / "universe.csv"
CACHE_MAX_AGE_DAYS = 7


def _download() -> pd.DataFrame:
    last_err = None
    for url in JPX_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return pd.read_excel(io.BytesIO(r.content))   # .xlsx は openpyxl、.xls は xlrd が必要
        except Exception as e:  # noqa: BLE001
            log.warning("%s -> %s", url, e)
            last_err = e
    raise RuntimeError(f"JPXの銘柄一覧を取得できませんでした: {last_err}")


def _clean(raw: pd.DataFrame, markets: list[str]) -> pd.DataFrame:
    need = ["コード", "銘柄名", "市場・商品区分", "33業種区分"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"JPXの表の列名が変わりました（見つからない列: {missing} / 実際の列: {list(raw.columns)}）")
    df = raw.copy()
    df = df[df["市場・商品区分"].astype(str).str.contains("内国株式", na=False)]
    df["market"] = df["市場・商品区分"].astype(str).str.extract(r"(プライム|スタンダード|グロース)")[0]
    df = df[df["market"].isin(markets)]
    code = df["コード"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out = pd.DataFrame({
        "code": code.values,
        "name": df["銘柄名"].astype(str).values,
        "market": df["market"].values,
        "sector": df["33業種区分"].astype(str).values,
    })
    out["ticker"] = out["code"] + ".T"
    return out.drop_duplicates("code").reset_index(drop=True)


def load_universe(cfg: dict) -> pd.DataFrame:
    markets = cfg["markets"]
    if CACHE.exists() and (time.time() - CACHE.stat().st_mtime) < CACHE_MAX_AGE_DAYS * 86400:
        return pd.read_csv(CACHE, dtype=str, encoding="utf-8-sig")
    try:
        uni = _clean(_download(), markets)
        if len(uni) < 1500:
            raise RuntimeError(f"銘柄数が少なすぎます({len(uni)})。JPXの形式が変わった可能性があります")
        uni.to_csv(CACHE, index=False, encoding="utf-8-sig")
        return uni
    except Exception as e:  # noqa: BLE001
        log.warning("銘柄一覧の取得に失敗: %s", e)
        if CACHE.exists():
            log.warning("保存済みの銘柄一覧を使います")
            return pd.read_csv(CACHE, dtype=str, encoding="utf-8-sig")
        raise
