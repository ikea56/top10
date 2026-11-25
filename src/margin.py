"""信用売りできる状態かの判定（情報提供のみ。売買の推奨ではありません）。

  貸借銘柄        … 制度信用で「売り」ができる銘柄（JPX「制度信用・貸借銘柄一覧」。毎月第1営業日に更新されるExcel）
  制度信用のみ    … 制度信用では売りができない（買いのみ）。※一般信用で売れるかは証券会社しだい
  規制中          … JPX「信用取引に関する規制等」の規制中銘柄（増担保など。売りは可能だが保証金が上がる）
  貸借取引停止等  … 日本証券金融の銘柄別制限措置（停止・注意喚起）。取得できた範囲で判定
※ 公式ページのレイアウトが変わると読み取れなくなります。読み取れないときは「不明」と表示し、除外はしません。
"""
from __future__ import annotations

import io
import re
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import urljoin

import requests

from .log import get_logger

log = get_logger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (jp-stock-top10; personal research; low-rate)"}
CODE = re.compile(r"^[0-9]{3}[0-9A-Z]$")


class _Tables(HTMLParser):
    """HTMLの表を [[セルの文字,…],…] のリストにする（標準ライブラリだけ）。"""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._t: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._t = []
        elif tag == "tr" and self._t is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._t is not None:
            if self._row:
                self._t.append(self._row)
            self._row = None
        elif tag == "table":
            self._depth -= 1
            if self._depth == 0 and self._t is not None:
                self.tables.append(self._t)
                self._t = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_tables(html: str) -> list[list[list[str]]]:
    p = _Tables()
    p.feed(html)
    return p.tables


# ---------------------------------------------------------------- JPX 制度信用・貸借銘柄一覧（Excel）
def find_list_url(html: str, base: str) -> str | None:
    """ページ内の「制度信用・貸借銘柄一覧」のExcelへのリンク（ファイル名が毎月変わるため、ページから探す）。"""
    hrefs = re.findall(r'href="([^"]+\.xlsx?)"', html)
    cand = [h for h in hrefs if re.search(r"list", h, re.I)] or hrefs
    return urljoin(base, cand[0]) if cand else None


def parse_margin_xlsx(content: bytes) -> tuple[set[str], set[str]] | None:
    """戻り値：(制度信用の銘柄, 貸借銘柄)。貸借を見分けられなければ None（＝判定しない）。
    見分け方：①シート名に「貸借」 ②ある列の値に「貸借」（区分列） ③見出し行に「貸借」「制度信用」のある列ブロック。"""
    import pandas as pd

    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None, dtype=str)
    except Exception as e:  # noqa: BLE001
        log.warning("制度信用・貸借銘柄一覧のExcelを読めません: %s", e)
        return None
    seido: set[str] = set()
    lend: set[str] = set()
    for name, df in sheets.items():
        df = df.fillna("")
        head = " ".join(" ".join(df.iloc[i].astype(str)) for i in range(min(6, len(df))))
        for _, row in df.iterrows():
            cells = [str(c).strip() for c in row]
            codes = [c for c in cells if CODE.match(c)]
            if not codes:
                continue
            code = codes[0]
            text = " ".join(cells)
            is_lend = "貸借" in text.replace("貸借銘柄選定", "") or "貸借" in str(name)
            if not is_lend and "貸借" in head and "制度信用" not in head:
                is_lend = True
            seido.add(code)
            if is_lend:
                lend.add(code)
    if not seido or not lend:                # 貸借を見分けられない＝判定しない
        return None
    return seido, lend


def fetch_margin_lists(page: str, get: Callable[..., Any] | None = None) -> tuple[set[str], set[str]] | None:
    get = get or requests.get
    try:
        r = get(page, timeout=30, headers=UA)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        url = find_list_url(r.text, page)
        if not url:
            log.warning("制度信用・貸借銘柄一覧のExcelへのリンクが見つかりません")
            return None
        x = get(url, timeout=60, headers=UA)
        x.raise_for_status()
        res = parse_margin_xlsx(x.content)
        if res:
            log.info("制度信用 %d銘柄・貸借 %d銘柄（%s）", len(res[0]), len(res[1]), url)
        return res
    except Exception as e:  # noqa: BLE001
        log.warning("制度信用・貸借銘柄一覧を取得できません: %s", e)
        return None


# ---------------------------------------------------------------- JPX 信用取引規制（HTMLの表）
def parse_regulated(html: str) -> dict[str, str]:
    """「規制を行っている銘柄」の表（列に『実施日』がある表）→ {コード: 規制の内容}。解除済み（『解除日』の表）は含めない。"""
    out: dict[str, str] = {}
    for t in parse_tables(html):
        if not t:
            continue
        hdr = " ".join(t[0])
        if "実施日" not in hdr or "解除日" in hdr:
            continue
        for row in t[1:]:
            code = next((c for c in row if CODE.match(c)), None)
            if code:
                out[code] = next((c for c in row if "規制" in c or "保証金" in c or "売り" in c), "規制中")
    return out


def fetch_regulated(page: str, get: Callable[..., Any] | None = None) -> dict[str, str] | None:
    get = get or requests.get
    try:
        r = get(page, timeout=30, headers=UA)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return parse_regulated(r.text)
    except Exception as e:  # noqa: BLE001
        log.warning("JPXの信用取引規制を取得できません: %s", e)
        return None


# ---------------------------------------------------------------- 日本証券金融（貸借取引の停止・注意喚起）
STOP = re.compile(r"貸借取引.{0,6}停止|申込.{0,4}停止|新規売り.{0,6}(停止|制限|不可)|貸株.{0,4}停止|品貸.{0,4}停止|売り.{0,6}(できません|不可)")
CAUTION = re.compile(r"注意喚起|貸株超過|貸株不足|品貸料|逆日歩|貸株.{0,4}制限")


def parse_jsf(html: str) -> dict[str, str]:
    """表の行から、コードと『停止』『注意喚起』の語を拾う（ページの形式が未確認のため、緩く判定）。{コード: 'stop'|'caution'}"""
    out: dict[str, str] = {}
    for t in parse_tables(html):
        for row in t:
            code = next((c for c in row if CODE.match(c)), None)
            if not code:
                continue
            text = " ".join(row)
            if STOP.search(text):
                out[code] = "stop"
            elif CAUTION.search(text) and out.get(code) != "stop":
                out[code] = "caution"
    return out


def fetch_jsf(page: str, get: Callable[..., Any] | None = None) -> dict[str, str] | None:
    get = get or requests.get
    try:
        r = get(page, timeout=30, headers=UA)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return parse_jsf(r.text)
    except Exception as e:  # noqa: BLE001
        log.warning("日本証券金融の銘柄別制限措置を取得できません: %s", e)
        return None


# ---------------------------------------------------------------- 銘柄ごとの判定
def short_status(code: str, lists: tuple[set[str], set[str]] | None, regulated: dict[str, str] | None, jsf: dict[str, str] | None) -> dict[str, Any]:
    """戻り値 key：ok（貸借・規制なし）/ ok_warn（貸借だが規制・注意あり）/ no_lending（制度信用のみ）/ no_margin（信用取引の対象外）/ stop（貸借取引停止等）/ unknown。
    ok・ok_warn・unknown は売り可能（または不明）、それ以外は信用売りできない状態。"""
    notes: list[str] = []
    if jsf and jsf.get(code) == "stop":
        return {"key": "stop", "label": "貸借取引停止等", "can_short": False, "notes": ["日本証券金融の銘柄別制限措置で、貸借取引の停止・制限の対象"]}
    if lists is not None:
        seido, lend = lists
        if code not in seido:
            return {"key": "no_margin", "label": "信用取引の対象外", "can_short": False, "notes": ["制度信用・貸借銘柄に選定されていません"]}
        if code not in lend:
            return {"key": "no_lending", "label": "制度信用のみ（売り不可）", "can_short": False, "notes": ["貸借銘柄ではないため、制度信用では売りができません（一般信用の売りは証券会社しだい）"]}
    else:
        notes.append("貸借銘柄かどうかを確認できませんでした")
    if regulated and code in regulated:
        notes.append(f"信用取引規制中：{regulated[code]}")
    if jsf and jsf.get(code) == "caution":
        notes.append("日本証券金融の注意喚起の対象（今後、売りが制限される可能性）")
    if lists is None:
        return {"key": "unknown", "label": "貸借かどうか不明", "can_short": None, "notes": notes}
    if notes:
        return {"key": "ok_warn", "label": "売り可（貸借）・注意あり", "can_short": True, "notes": notes}
    return {"key": "ok", "label": "売り可（貸借銘柄）", "can_short": True, "notes": []}
