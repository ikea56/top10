"""JPX公式「監理銘柄・整理銘柄一覧」から対象コードを取得する（取得できなくても処理は止めない）。"""
from __future__ import annotations

import re
from datetime import date, datetime
from html.parser import HTMLParser

import requests

from .log import get_logger

log = get_logger(__name__)
CODE_RE = re.compile(r"^\d{3}[0-9A-Z]$")


class _TableParser(HTMLParser):
    """HTMLの表を [表][行][セル文字列] に分解するだけの小さなパーサ（外部ライブラリ不要）。"""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self.tables[-1].append([])
        elif tag in ("td", "th") and self.tables and self.tables[-1]:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self.tables and self.tables[-1]:
            self.tables[-1][-1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _released(text: str, today: date) -> bool:
    m = re.search(r"(\d{4})[/年.-](\d{1,2})[/月.-](\d{1,2})", text)
    if not m:
        return False
    try:
        return datetime(int(m[1]), int(m[2]), int(m[3])).date() <= today
    except ValueError:
        return False


def parse_codes(html: str, today: date | None = None) -> set[str]:
    """『コード』列を持つ表から銘柄コードを集める。『解除年月日』が過去の行は除く。"""
    today = today or date.today()
    p = _TableParser()
    p.feed(html)
    codes: set[str] = set()
    for rows in p.tables:
        header = next((r for r in rows if any("コード" in c for c in r)), None)
        if header is None:
            continue
        ci = next(i for i, c in enumerate(header) if "コード" in c)
        ri = next((i for i, c in enumerate(header) if "解除" in c), None)
        for r in rows:
            if r is header or len(r) <= ci:
                continue
            code = re.sub(r"\s", "", r[ci]).upper()
            if not CODE_RE.match(code):
                continue
            if ri is not None and len(r) > ri and _released(r[ri], today):
                continue
            codes.add(code)
    return codes


def fetch_supervision_codes(urls: list[str]) -> set[str]:
    codes: set[str] = set()
    for url in urls:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            found = parse_codes(r.text)
            log.info("監理・整理銘柄（%s）: %d件", url, len(found))
            codes |= found
        except Exception as e:  # noqa: BLE001  取得失敗でも更新全体は止めない
            log.warning("監理・整理銘柄一覧を取得できませんでした（銘柄名キーワードと手動リストで代替）: %s", e)
    return codes
