"""TDnetの開示タイトルから、企業イベントの判定・Event Score・企業再編の除外判定を行う。

判定は開示タイトルのキーワードで行う（取得元に依存しない）。
「業績予想の修正」「配当予想の修正」は、タイトルだけでは増額・減額が分からないことが多いため、
  ① タイトルの語（上方/下方/増額/減額） → ② 開示PDFの本文（直近30日分のみ取得可） → ③ アナリスト予想の変化・配当実績
の順に方向を判定し、それでも不明なら「方向不明の修正」として小さく減点して注意表示する。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from .config import Config
from .tdnet import Disclosure

# ---------------------------------------------------------------- 除外（企業再編・上場廃止）
EXCLUDE_RULES: list[tuple[str, re.Pattern]] = [
    ("TOB・公開買付", re.compile(r"公開買付|ＴＯＢ|TOB")),
    ("MBO", re.compile(r"MBO|ＭＢＯ|マネジメント・?バイアウト")),
    ("完全子会社化", re.compile(r"完全子会社化|完全子会社となる|完全子会社とする")),
    ("株式交換", re.compile(r"株式交換")),
    ("スクイーズアウト", re.compile(r"スクイーズ|株式売渡請求|全部取得条項付|特別支配株主")),
    ("上場廃止予定", re.compile(r"上場廃止")),
    ("監理銘柄", re.compile(r"監理銘柄|監理ポスト")),
    ("整理銘柄", re.compile(r"整理銘柄|整理ポスト")),
]
SELF_TENDER = re.compile(r"自己株式.{0,6}公開買付")      # 会社が自社株を買う（正の材料）。TOB（買収）とは別

# ---------------------------------------------------------------- 加点・減点イベント
BUYBACK = re.compile(r"自己株式(の)?取得|自己株式.{0,6}公開買付")
BUYBACK_SKIP = re.compile(r"状況|結果|終了|消却|処分|中止|変更")
EVENT_RULES: list[tuple[str, str, re.Pattern, re.Pattern | None]] = [
    # (種別, 表示名, 一致, 除外)
    ("stock_split", "株式分割", re.compile(r"株式分割|株式の分割"), re.compile(r"併合|無償割当")),
    ("large_order", "大型受注", re.compile(r"受注"), None),                # 下で「大型」語を別途確認
    ("alliance", "業務提携", re.compile(r"業務提携|資本提携|資本業務提携|業務・資本提携|戦略的提携|アライアンス"), None),
    ("factory", "工場増設", re.compile(r"工場.{0,8}(増設|新設|建設)|新工場|増設"), None),
    ("capex", "設備投資拡大", re.compile(r"設備投資|生産能力.{0,6}(増強|拡大)|能力増強|増産投資"), None),
    ("new_product", "新製品発表", re.compile(r"新製品|新商品|新サービス|新型|新モデル|新発売|発売開始|提供開始"), None),
]
BIG_ORDER = re.compile(r"大型|大口|大規模|多額|大量|過去最大|億円")
REV_ANY = re.compile(r"業績予想.{0,6}修正|予想.{0,4}修正|通期.{0,6}修正|業績修正")
REV_UP, REV_DOWN = re.compile(r"上方修正|上方に修正|増額"), re.compile(r"下方修正|下方に修正|減額")
DIV_ANY = re.compile(r"配当予想.{0,4}修正|配当.{0,4}修正")
DIV_UP, DIV_DOWN = re.compile(r"増配|記念配当|特別配当|復配|増額"), re.compile(r"減配|無配|減額")

LABELS = {"upward_revision": "上方修正", "downward_revision": "下方修正", "dividend_increase": "増配",
          "dividend_cut": "減配", "buyback": "自社株買い", "large_order": "大型受注", "alliance": "業務提携",
          "new_product": "新製品発表", "capex": "設備投資拡大", "factory": "工場増設", "stock_split": "株式分割",
          "unknown_revision": "業績予想の修正（増額・減額不明）"}


def direction_from_text(text: str) -> str | None:
    """開示本文の語の多数決で増額(up)・減額(down)を判定。強い語だけを数え、同数なら不明。"""
    up = len(re.findall(r"増額|上方修正|上方に修正", text))
    down = len(re.findall(r"減額|下方修正|下方に修正", text))
    return "up" if up > down else "down" if down > up else None


@dataclass
class EventHit:
    kind: str
    date: str
    title: str
    points: float = 0.0

    @property
    def label(self) -> str:
        return LABELS[self.kind]


@dataclass
class TdnetResult:
    status: str = "ok"                       # ok / unavailable / disabled
    excluded: list[dict] = field(default_factory=list)   # {reason, event, date}
    hits: list[EventHit] = field(default_factory=list)
    score: float = 0.0
    error: str = ""

    @property
    def has_down_revision(self) -> bool:
        return any(h.kind == "downward_revision" for h in self.hits)

    @property
    def serious_negative(self) -> bool:
        return any(h.kind in ("downward_revision", "dividend_cut") for h in self.hits)


def detect_exclusions(d: Disclosure) -> list[tuple[str, str]]:
    """(除外理由, 検出イベント=タイトル) を返す。自社株の公開買付けはTOB扱いにしない。"""
    hits = []
    for reason, rx in EXCLUDE_RULES:
        if not rx.search(d.title):
            continue
        if reason == "TOB・公開買付" and SELF_TENDER.search(d.title):
            continue
        hits.append((reason, d.title))
    return hits


def _revision_direction(d: Disclosure, fund: dict | None, cfg: Config, today: date,
                        pdf_fn: Callable[[str], str | None] | None) -> str | None:
    t = cfg["tdnet"]
    if REV_UP.search(d.title):
        return "up"
    if REV_DOWN.search(d.title):
        return "down"
    age = (today - date.fromisoformat(d.date)).days
    if t["parse_pdf"] and pdf_fn and d.url and age <= t["pdf_max_age_days"]:
        text = pdf_fn(d.url)
        if text and (dr := direction_from_text(text)):
            return dr
    r = (fund or {}).get("eps_rev") or {}                     # アナリスト予想EPSの90日間の変化で補う
    cur, d90 = r.get("cur"), r.get("d90")
    if cur is not None and d90 and d90 > 0:
        chg = cur / d90 - 1
        if chg <= -0.03:
            return "down"
        if chg >= 0.03:
            return "up"
    return None


def _dividend_direction(d: Disclosure, fund: dict | None) -> str | None:
    if DIV_UP.search(d.title) and not DIV_DOWN.search(d.title):
        return "up"
    if DIV_DOWN.search(d.title):
        return "down"
    cur, prev = (fund or {}).get("div_cur"), (fund or {}).get("div_prev")
    if cur and prev and prev > 0:
        return "up" if cur > prev * 1.0001 else "down" if cur < prev * 0.9999 else None
    return None


def analyze(discs: list[Disclosure], fund: dict | None, cfg: Config, today: date | None = None,
            pdf_fn: Callable[[str], str | None] | None = None) -> TdnetResult:
    """直近 lookback_days の開示を分類し、除外判定とEvent Scoreを返す。同じ種別は新しい1件だけ数える。"""
    today = today or date.today()
    t = cfg["tdnet"]
    since = (today - timedelta(days=int(t["lookback_days"]))).isoformat()
    res = TdnetResult()
    found: dict[str, EventHit] = {}
    seen_ex: set[tuple[str, str]] = set()

    def put(kind: str, d: Disclosure) -> None:
        if kind not in found or d.date > found[kind].date:
            found[kind] = EventHit(kind, d.date, d.title, float(t["points"][kind]))

    for d in sorted((x for x in discs if x.date >= since), key=lambda x: x.date, reverse=True):
        for reason, ev in detect_exclusions(d):
            if (reason, d.date) not in seen_ex:
                seen_ex.add((reason, d.date))
                res.excluded.append({"reason": reason, "event": ev, "date": d.date})
        title = d.title
        if BUYBACK.search(title) and not BUYBACK_SKIP.search(title):
            put("buyback", d)
        for kind, _, rx, skip in EVENT_RULES:
            if rx.search(title) and not (skip and skip.search(title)):
                if kind == "large_order" and not BIG_ORDER.search(title):
                    continue
                put(kind, d)
        if REV_ANY.search(title) or REV_UP.search(title) or REV_DOWN.search(title):
            if not (DIV_ANY.search(title) and not re.search(r"業績", title)):          # 配当だけの修正は下で扱う
                dr = _revision_direction(d, fund, cfg, today, pdf_fn)
                put({"up": "upward_revision", "down": "downward_revision"}.get(dr, "unknown_revision"), d)
        if DIV_ANY.search(title) or DIV_UP.search(title) or DIV_DOWN.search(title):
            if "業績" not in title or DIV_ANY.search(title):
                dr = _dividend_direction(d, fund)
                if dr:
                    put("dividend_increase" if dr == "up" else "dividend_cut", d)

    res.hits = sorted(found.values(), key=lambda h: -abs(h.points))
    plus = sum(h.points for h in res.hits if h.points > 0)
    minus = sum(h.points for h in res.hits if h.points < 0)
    score = min(plus, t["cap"]["plus"]) + max(minus, -t["cap"]["minus"])
    if res.has_down_revision:
        score -= float(t["downward_extra_penalty"])         # 下方修正は最重要減点
    res.score = score
    return res
