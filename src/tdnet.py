"""TDnet（適時開示）の取得層。

・JPX公式のTDnet無料閲覧は「直近31日分」だけ（確認済み）。当システムは直近180日を見たいので、
  銘柄コード別に過去分を返す外部の無料API（やのしんWebAPI）を既定の取得元にしている。
  ※ 外部サービスのためレスポンス形式は変わる可能性がある。形式が違っても落ちないよう寛容に読み、
    取得できない銘柄は「TDnet未確認」として画面に警告を出す（TOB等の確認ができていない状態を隠さない）。
・取得元を変えたいときは Disclosure を返す fetch(code) を持つクラスに差し替えればよい（tdnet_events.py は取得元に依存しない）。
・確認用：  python -m src.tdnet 7203      （取得できた開示の件数とタイトルを表示）
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import requests

from .config import ROOT, Config
from .log import get_logger

log = get_logger(__name__)
CACHE = ROOT / "data" / "tdnet_cache.json"
UA = "jp-stock-top10 (personal research; low-rate)"


class TdnetError(Exception):
    """TDnetを取得できなかったときのエラー。"""


@dataclass(frozen=True)
class Disclosure:
    date: str      # YYYY-MM-DD
    title: str
    url: str = ""


def _to_date(text: Any) -> str | None:
    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", str(text or ""))
    return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}" if m else None


def parse_payload(payload: Any, code: str) -> list[Disclosure]:
    """やのしんWebAPIのJSON（items[].Tdnet.{pubdate,company_code,title,document_url}）を読む。
    形式が少し違っても読めるよう、Tdnetで包まれていない場合も許容する。"""
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise TdnetError("レスポンスの形式が想定と違います（items が見つかりません）")
    out: list[Disclosure] = []
    for it in items:
        d = it.get("Tdnet", it) if isinstance(it, dict) else None
        if not isinstance(d, dict):
            continue
        cc = str(d.get("company_code") or "")
        if cc and not cc.upper().startswith(code.upper()):      # 5桁コード（例 72030）の先頭4桁で照合
            continue
        day, title = _to_date(d.get("pubdate")), str(d.get("title") or "").strip()
        if day and title:
            out.append(Disclosure(day, title, str(d.get("document_url") or "")))
    return out


class TdnetClient:
    """銘柄別の適時開示を取得する。結果は data/tdnet_cache.json に保存し、同じ銘柄を短時間に何度も取りに行かない。"""

    def __init__(self, cfg: Config) -> None:
        self.c = cfg["tdnet"]
        self.cache: dict[str, dict[str, Any]] = {}
        self.errors = 0
        self.calls = 0
        if CACHE.exists():
            try:
                self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                log.warning("TDnetキャッシュを読めないため作り直します: %s", e)

    def fetch(self, code: str) -> list[Disclosure]:
        hit = self.cache.get(code)
        if hit and time.time() - hit["t"] < self.c["cache_hours"] * 3600:
            return [Disclosure(*x) for x in hit["items"]]
        self.calls += 1
        url = self.c["base_url"].format(code=code)
        last: Exception | None = None
        for attempt in range(int(self.c["retries"]) + 1):
            try:
                r = requests.get(url, params={"limit": 300}, timeout=self.c["timeout"], headers={"User-Agent": UA})
                r.raise_for_status()
                items = parse_payload(r.json(), code)
                time.sleep(self.c["sleep"])
                self.cache[code] = {"t": time.time(), "items": [[d.date, d.title, d.url] for d in items]}
                return items
            except (requests.RequestException, ValueError, TdnetError) as e:
                last = e
                time.sleep(1.0 + attempt)
        self.errors += 1
        raise TdnetError(f"{code}: TDnetを取得できませんでした（{last}）")

    def save(self) -> None:
        limit = time.time() - 14 * 86400
        self.cache = {k: v for k, v in self.cache.items() if v["t"] >= limit}
        try:
            CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("TDnetキャッシュを保存できませんでした: %s", e)


class DemoTdnet:
    """デモ用の架空の開示（TOB・下方修正などを一部の銘柄に仕込んで、除外や減点の表示を確認できるようにする）。"""

    errors = calls = 0

    def fetch(self, code: str) -> list[Disclosure]:
        n, today = int(code), date.today()
        ago = lambda d: (today - timedelta(days=d)).isoformat()    # noqa: E731
        out = [Disclosure(ago(40), "第2四半期決算短信〔日本基準〕（連結）")]
        if n % 5 == 0:
            out.append(Disclosure(ago(20), "配当予想の修正（増配）に関するお知らせ"))
        if n % 7 == 0:
            out.append(Disclosure(ago(15), "自己株式の取得に係る事項の決定に関するお知らせ"))
        if n % 11 == 0:
            out.append(Disclosure(ago(30), "通期業績予想の下方修正に関するお知らせ"))
        if n % 13 == 0:
            out.append(Disclosure(ago(10), "大型受注に関するお知らせ"))
        if n % 17 == 0:
            out.append(Disclosure(ago(25), "株式会社○○による当社株式に対する公開買付けに関する意見表明のお知らせ"))
        if n % 19 == 0:
            out.append(Disclosure(ago(12), "自己株式の公開買付けの結果に関するお知らせ".replace("の結果", "")))
        return out

    def save(self) -> None:
        return None


def pdf_text(url: str, timeout: float = 20.0, pages: int = 2) -> str | None:
    """開示PDFの先頭ページの文字を取り出す（pypdf が無い・取得できない場合は None）。"""
    try:
        from io import BytesIO

        from pypdf import PdfReader
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        reader = PdfReader(BytesIO(r.content))
        return "".join((reader.pages[i].extract_text() or "") for i in range(min(pages, len(reader.pages))))
    except Exception as e:  # noqa: BLE001
        log.debug("PDFを読めませんでした（%s）: %s", url, e)
        return None


def make_client(cfg: Config, demo: bool) -> TdnetClient | DemoTdnet | None:
    if not cfg["tdnet"]["enabled"]:
        return None
    return DemoTdnet() if demo else TdnetClient(cfg)


def selfcheck(client: TdnetClient, code: str = "7203") -> bool:
    """起動時の動作確認。取得できなければ警告（TOB等の確認ができない状態になる）。"""
    try:
        items = client.fetch(code)
    except TdnetError as e:
        log.warning("TDnet取得の動作確認に失敗: %s", e)
        return False
    log.info("TDnet動作確認 %s: %d件（最新: %s）", code, len(items), items[0].title[:30] if items else "なし")
    return bool(items)


if __name__ == "__main__":
    import sys

    from .config import load_config
    code = sys.argv[1] if len(sys.argv) > 1 else "7203"
    cl = TdnetClient(load_config())
    for d in cl.fetch(code)[:20]:
        print(d.date, d.title)
