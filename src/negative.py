"""㉗ 信用リスク監視ページ。企業の信用低下や、継続的な株価下落につながりうるネガティブイベントを監視する。

※ 空売り推奨ではありません。「信用リスクの監視」が目的で、点数は売買の指示ではなく『確認の優先度』です。

情報源（取得できた範囲）
  TDnet（適時開示）        … やのしんWebAPIの日付範囲の全社開示。タイトルを分類し、方向が不明な「業績予想の修正」はPDFも読む
  JPX公式 監理・整理銘柄一覧 … 公式一覧に載っている銘柄
  Googleニュース（検索RSS）  … ネガティブ語の横断検索＋ランキング上位・検出済み銘柄の個別検索
  Yahoo!ニュース（公式RSS） … 経済トピックの見出しのうち、ネガティブ語と会社名を含むもの
  財務データ（Yahoo Finance）… 営業利益の推移から「赤字転落／継続赤字」（取得済みの銘柄のみ）
  X（旧Twitter）・各社IRページ … 無料で安定して使える取得手段が無いため未対応（TDnetが各社IRの法定開示をカバー）

要約は「AI」ではなく、決まった規則による自動要約（見出しの整理）です。
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time as _time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from .config import JST, ROOT, Config, load_config
from .log import get_logger

log = get_logger(__name__)
UA = "jp-stock-top10 (personal research; low-rate)"


# ---------------------------------------------------------------- データ
@dataclass
class Item:
    source: str                 # TDnet / JPX公式 / Googleニュース / Yahoo!ニュース / 財務データ
    date: str                   # 公表日 YYYY-MM-DD
    code: str | None
    name: str
    title: str
    url: str = ""
    kind: str = "news"          # official（TDnet・JPX）/ news / financial


@dataclass
class Event:
    key: str                    # points のキー
    label: str                  # 画面に出す分類名（依頼の17分類のどれか）
    item: Item
    base: float
    severe: int = 0             # 「重大・隠蔽…」などの語の数
    direction: str | None = None


# ---------------------------------------------------------------- 分類ルール
# (key, 分類名, 当てはまる正規表現, 除外する正規表現)。タイトル（見出し）だけで判定する。1つの見出しが複数に当てはまってもよい。
LABELS = {
    "downward_revision": "下方修正", "dividend_cut": "減配", "loss_turn": "赤字転落", "continuing_loss": "継続赤字",
    "going_concern": "継続赤字", "kanri": "監理銘柄", "seiri": "整理銘柄", "admin_action": "行政処分",
    "quality_fraud": "品質不正", "inspection_fraud": "検査不正", "accounting": "会計問題", "window_dressing": "粉飾疑惑",
    "data_leak": "情報漏洩", "outage": "大規模障害", "recall": "リコール", "litigation": "訴訟",
    "deterioration": "業績悪化", "delisting": "上場廃止関連",
}
CATEGORY_ORDER = ["上場廃止関連", "整理銘柄", "監理銘柄", "粉飾疑惑", "会計問題", "品質不正", "検査不正", "行政処分", "赤字転落", "継続赤字",
                  "下方修正", "減配", "情報漏洩", "リコール", "大規模障害", "訴訟", "業績悪化"]
RX = re.compile
RULES: list[tuple[str, Any, Any]] = [
    ("delisting", RX(r"上場廃止(?!.*(公開買付|ＭＢＯ|MBO|非公開化|完全子会社|株式交換|株式併合.{0,10}スクイーズ))|猶予期間|特別注意銘柄|特設注意市場|上場維持|"
                     r"上場契約違約|破産|民事再生|会社更生|事業再生ＡＤＲ|事業再生ADR|特別清算|不渡り|債務不履行|デフォルト|倒産"), None),
    ("seiri", RX(r"整理銘柄"), RX(r"指定解除|解除")),
    ("kanri", RX(r"監理銘柄|監理ポスト"), RX(r"指定解除|解除")),
    ("window_dressing", RX(r"粉飾|架空(売上|取引|計上|在庫)|循環取引|水増し|不正会計|不正経理|虚偽記載|横領|不正流用"), None),
    ("accounting", RX(r"不適切(な)?(会計|経理|処理|取引)|会計不正|過年度.{0,6}(訂正|修正)|(決算|財務諸表|有価証券報告書).{0,6}訂正|訂正(有価証券|報告書|決算短信)|"
                     r"決算発表.{0,6}(延期|遅延)|(有価証券報告書|四半期報告書|半期報告書).{0,10}(提出遅延|提出期限|延長)|意見不表明|不適正意見|限定付適正|"
                     r"開示すべき重要な不備|特別調査委員会|第三者委員会|調査委員会"), None),
    ("quality_fraud", RX(r"品質(不正|偽装|不適切|問題|不祥事)|(データ|数値)(の)?(改ざん|改竄|偽装|不正)|認証(不正|取得.{0,4}不正)|性能.{0,4}偽装|品質保証.{0,6}(問題|不備)"), None),
    ("inspection_fraud", RX(r"(検査|試験|測定|点検).{0,6}(不正|不適切|偽装|改ざん|改竄|不備)|無資格.{0,4}(検査|作業)|完成検査"), None),
    ("admin_action", RX(r"行政処分|業務停止|業務改善命令|課徴金|排除措置命令|措置命令|立入検査|是正勧告|是正命令|改善命令|登録取消|許可取消|認可取消|"
                       r"業務改善勧告|公正取引委員会.{0,6}(調査|審査|処分)|景品表示法違反|金融庁.{0,6}処分"), RX(r"解除|終了")),
    ("going_concern", RX(r"継続企業の前提|重要事象等|重要な疑義|ゴーイングコンサーン"), None),
    ("continuing_loss", RX(r"継続赤字|連続(の)?赤字|[2-9２-９]期連続.{0,6}(赤字|損失|営業損失|最終損失)|営業損失.{0,6}(継続|連続)"), None),
    ("loss_turn", RX(r"赤字転落|赤字に転落|赤字(となる|になる|見込|予想)|営業赤字|経常赤字|最終赤字|損失に転落|損失(となる|を計上する)見込|"
                     r"(営業|経常|当期純|親会社株主に帰属する当期純)損失.{0,4}(計上|見込|の見通し)"), None),
    ("downward_revision", RX(r"下方修正|下方に修正|業績予想.{0,8}(減額|引き下げ|引下げ)|予想.{0,6}(減額|引き下げ|引下げ)|通期.{0,6}(減額|引下げ)"), None),
    ("dividend_cut", RX(r"減配|無配|配当.{0,8}(減額|引き下げ|引下げ|取りやめ|見送り|廃止|ゼロ)|(期末|中間|年間)配当.{0,6}(未定|なし)"), RX(r"増配|増額")),
    ("data_leak", RX(r"情報漏(えい|洩)|(個人|顧客|取引先).{0,4}(情報|データ).{0,6}(流出|漏えい|漏洩)|不正アクセス|ランサムウェア|ランサムウエア|サイバー攻撃|"
                    r"ウイルス感染|不正ログイン"), None),
    ("outage", RX(r"システム障害|大規模(な)?障害|通信障害|サービス(の)?(停止|障害)|(運航|運行|操業|生産|稼働)(の)?停止|工場.{0,4}(火災|爆発)|爆発事故|火災.{0,6}(発生|被害)|"
                  r"出荷停止|ネットワーク障害"), RX(r"再開")),
    ("recall", RX(r"リコール|自主回収|無償(回収|交換|修理)|回収・?交換|販売停止|改善対策"), None),
    ("litigation", RX(r"訴訟|提訴|訴えを提起|訴え提起|損害賠償(請求|を求め)|集団訴訟|差止(請求|め)|控訴|仲裁(申立|請求)|調停申立"),
     RX(r"和解|(訴訟|裁判).{0,8}(解決|終結|取下|勝訴|棄却)|勝訴|判決.{0,6}(当社の主張|請求棄却)")),
    ("deterioration", RX(r"業績悪化|減収減益|大幅(な)?(減益|減少|悪化)|最終減益|(営業|経常|純)利益.{0,6}(減少|悪化|急減)|特別損失|減損損失|のれん.{0,4}減損|"
                         r"在庫評価損|事業(撤退|縮小|譲渡の中止)|リストラ|希望退職|早期退職|工場閉鎖|資金繰り|債務超過|財務制限条項|格下げ|格付.{0,6}(引き下げ|引下げ|下方)|"
                         r"減益"), None),
]
# 「業績予想の修正」「配当予想の修正」の方向がタイトルから分からないもの（PDFを読む）
REV_AMBIG = RX(r"業績予想.{0,4}の?修正|通期.{0,6}業績予想.{0,4}の?修正|業績予想及び|業績予想並びに|業績予想.{0,6}に関する(お知らせ|開示)")
SEVERE = RX(r"重大|多額|過去最大|大規模|複数|組織的|常態化|隠蔽|隠ぺい|逮捕|家宅捜索|捜索|特捜|告発|刑事|検察|不正.{0,3}発覚|巨額|大幅|急落|ストップ安|経営破綻")
GLOBAL_EXCLUDE = RX(r"(疑い|事実).{0,6}(なし|認められ|ありません)|報道.{0,10}(否定|事実ではありません|事実はありません)|(再開|復旧|解消).{0,4}(の)?お知らせ$")
TOB_LIKE = RX(r"公開買付|ＭＢＯ|MBO|非公開化|完全子会社|株式交換|スクイーズアウト")


# ---------------------------------------------------------------- 文字の正規化・社名の結びつけ
def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").replace("　", " ").strip()


SUFFIX = RX(r"(ホールディングス|グループホールディングス|グループ|ＨＤ|HD|コーポレーション|カンパニー)$")
PREFIX = RX(r"^(株式会社|㈱|\(株\))|(株式会社|㈱|\(株\))$")
STOP_NAMES = {"日本", "東京", "大阪", "三菱", "三井", "住友", "ジャパン", "アジア", "グローバル", "ネクスト", "ホーム", "システム", "サービス", "テクノ", "ニッポン"}


def stems(name: str) -> list[str]:
    """見出しの中で探す社名（正式名称と、末尾の「ホールディングス」「グループ」等を除いた短い名前）。3文字未満・一般語は使わない。"""
    n = PREFIX.sub("", norm(name)).strip()
    out = {n}
    s = SUFFIX.sub("", n).strip()
    if s:
        out.add(s)
    return [x for x in out if len(x) >= 3 and x not in STOP_NAMES]


def build_name_index(uni: Any) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    info, pairs = {}, []
    for r in uni.itertuples(index=False):
        code = str(r.code)
        info[code] = {"name": str(r.name), "market": str(getattr(r, "market", "")), "sector": str(getattr(r, "sector", ""))}
        for s in stems(str(r.name)):
            pairs.append((s, code))
    pairs.sort(key=lambda p: -len(p[0]))                     # 長い名前を優先（「ソフトバンクグループ」と「ソフトバンク」）
    return info, pairs


CODE_IN_TITLE = RX(r"[（(【\[]\s*(\d{4})\s*[）)】\]]|証券コード[:：]?\s*(\d{4})|東証[^\d]{0,6}(\d{4})")


def match_code(title: str, pairs: list[tuple[str, str]], info: dict[str, dict]) -> str | None:
    t = norm(title)
    m = CODE_IN_TITLE.search(t)
    if m:
        c = next(g for g in m.groups() if g)
        if c in info:
            return c
    for s, code in pairs:
        if s in t:
            return code
    return None


# ---------------------------------------------------------------- 分類
def classify(item: Item, direction: str | None = None) -> list[Event]:
    title = norm(item.title)
    if GLOBAL_EXCLUDE.search(title):
        return []
    evs: list[Event] = []
    severe = min(3, len(SEVERE.findall(title)))
    for key, rx, ex in RULES:
        if not rx.search(title):
            continue
        if ex is not None and ex.search(title):
            continue
        if key == "delisting" and TOB_LIKE.search(title):          # TOB・MBOによる上場廃止は信用リスクではない
            continue
        if key == "downward_revision" and re.search(r"上方修正|増額|引き上げ", title) and not re.search(r"下方修正|減額|引き下げ", title):
            continue
        if key == "deterioration" and item.kind == "official" and not re.search(r"減損損失|特別損失|債務超過|資金繰り|希望退職|早期退職|財務制限条項|工場閉鎖|格下げ|事業撤退", title):
            continue                                                  # TDnetの「減益」は決算短信などで頻出するため、深刻な語だけ拾う
        evs.append(Event(key, LABELS[key], item, 0.0, severe))
    if direction == "down" and not any(e.key == "downward_revision" for e in evs) and REV_AMBIG.search(title):
        evs.append(Event("downward_revision", LABELS["downward_revision"], item, 0.0, severe, "down"))
    # 同じ見出しに「赤字転落」と「下方修正」が両方あれば両方つける（別の信用リスクとして数える）。ただし継続企業の前提は継続赤字に統合しない
    return evs


def is_ambiguous_revision(item: Item) -> bool:
    t = norm(item.title)
    return item.kind == "official" and bool(REV_AMBIG.search(t)) and not re.search(r"上方|下方|増額|減額|引き上げ|引き下げ|引上げ|引下げ", t)


# ---------------------------------------------------------------- スコア
def recency_factor(age: int, cfg: Config) -> float:
    r = cfg["negative"]["recency"]
    full, mn, span = int(r["full_days"]), float(r["min_factor"]), int(cfg["negative"]["days"])
    if age <= full:
        return 1.0
    if age >= span:
        return mn
    return 1.0 - (1.0 - mn) * (age - full) / max(1, span - full)


def event_points(ev: Event, today: date, cfg: Config) -> float:
    n = cfg["negative"]
    age = max(0, (today - date.fromisoformat(ev.item.date)).days)
    rel = float(n["reliability"][ev.item.kind])
    bonus = min(float(n["severe_words"]["cap"]), ev.severe * float(n["severe_words"]["per_word"]))
    return min(100.0, (float(n["points"][ev.key]) + bonus) * recency_factor(age, cfg) * rel)


def company_score(events: list[Event], today: date, cfg: Config) -> dict[str, Any]:
    """会社ごとの点数＝分類ごとに『一番高い1件』を取り（同じ種類の重複は数えない）、分類をまたいで確率の和の形で合成する：
         score = 100 × (1 − Π(1 − p_i / 100))
       同じ分類を複数の情報源が報じていれば p_i を +10%ずつ（最大1.25倍）。点数は『確認の優先度』で、株価の予想ではない。"""
    n = cfg["negative"]
    best: dict[str, tuple[float, Event]] = {}
    srcs: dict[str, set[str]] = {}
    for ev in events:
        p = event_points(ev, today, cfg)
        srcs.setdefault(ev.key, set()).add(ev.item.source)
        if ev.key not in best or p > best[ev.key][0]:
            best[ev.key] = (p, ev)
    keep, prod = [], 1.0
    for key, (p, ev) in best.items():
        mult = min(float(n["corroboration"]["cap"]), 1 + float(n["corroboration"]["per_source"]) * (len(srcs[key]) - 1))
        p2 = min(100.0, p * mult)
        prod *= 1 - p2 / 100
        keep.append({"key": key, "points": round(p2, 1), "event": ev, "sources": sorted(srcs[key])})
    score = round(100 * (1 - prod))
    keep.sort(key=lambda x: -x["points"])
    return {"score": score, "parts": keep}


def band(score: int, cfg: Config) -> tuple[str, str]:
    b = cfg["negative"]["bands"]
    if score >= b["critical"]:
        return "critical", "重大"
    if score >= b["warn"]:
        return "warn", "警戒"
    if score >= b["caution"]:
        return "caution", "注意"
    return "minor", "軽微"


# ---------------------------------------------------------------- 要約（規則ベース）
TAIL = RX(r"(に関する|についての|に係る|に伴う)?(お知らせ|ご報告|開示|報告)$")


def core_title(item: Item, name: str) -> str:
    t = norm(item.title)
    t = re.sub(r"\s*[-－–|｜]\s*[^-－–|｜]{2,25}$", "", t) if item.kind == "news" else t       # ニュースの「 - 媒体名」を除く
    for s in [name] + stems(name):
        t = t.replace(s, "")
    t = PREFIX.sub("", t).strip(" 　:：、,・-")
    t = TAIL.sub("", t).strip(" 　:：、,・-") or norm(item.title)
    return t[:70]


def summarize(name: str, parts: list[dict[str, Any]], limit: int = 3) -> str:
    """検出イベントを、点数の高い順に最大limit件、「分類：内容（情報源 月/日）」で並べる。"""
    out = []
    for p in parts[:limit]:
        ev = p["event"]
        d = date.fromisoformat(ev.item.date)
        tag = "" if ev.item.kind == "official" else "【報道】" if ev.item.kind == "news" else "【財務データ】"
        out.append(f"{ev.label}：{tag}{core_title(ev.item, name)}（{ev.item.source} {d.month}/{d.day}）")
    return " ／ ".join(out)


# ---------------------------------------------------------------- 取得：TDnet
def parse_tdnet_all(payload: Any) -> list[Item]:
    """やのしんWebAPIの日付範囲JSON（items[].Tdnet.{pubdate,company_code,company_name,title,document_url}）。"""
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("レスポンスの形式が想定と違います（items が見つかりません）")
    out = []
    for it in items:
        d = it.get("Tdnet", it) if isinstance(it, dict) else None
        if not isinstance(d, dict):
            continue
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(d.get("pubdate") or ""))
        title = str(d.get("title") or "").strip()
        cc = str(d.get("company_code") or "").strip()
        if not (m and title):
            continue
        code = cc[:4] if len(cc) >= 4 else None
        out.append(Item("TDnet", f"{m[1]}-{m[2]}-{m[3]}", code, str(d.get("company_name") or "").strip(), title, str(d.get("document_url") or ""), "official"))
    return out


def fetch_tdnet(cfg: Config, today: date, get: Callable[..., Any] | None = None) -> tuple[list[Item], dict[str, Any]]:
    n = cfg["negative"]
    t = n["tdnet"]
    get = get or requests.get
    items, ok, fail = [], 0, 0
    for k in range(int(n["days"]) + 1):
        d = today - timedelta(days=k)
        if d.weekday() >= 5:
            continue
        s = d.strftime("%Y%m%d")
        url = t["day_url"].format(start=s, end=s)
        try:
            r = get(url, params={"limit": int(t["limit"])}, timeout=t["timeout"], headers={"User-Agent": UA})
            r.raise_for_status()
            items += parse_tdnet_all(r.json())
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            log.warning("TDnet（%s）を取得できません: %s", s, e)
        _time.sleep(float(t["sleep"]))
    return items, {"name": "TDnet（適時開示）", "ok": ok > 0, "n": len(items), "note": f"{ok}日分を取得" + (f"、{fail}日分は失敗" if fail else "")}


# ---------------------------------------------------------------- 取得：ニュース
def parse_rss(xml_text: str, source: str, today: date, days: int) -> list[Item]:
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter("item"):
        title = html.unescape((it.findtext("title") or "").strip())
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        try:
            dt = parsedate_to_datetime(pub).astimezone(JST).date() if pub else None
        except (TypeError, ValueError):
            dt = None
        if not title or dt is None or (today - dt).days > days or dt > today:
            continue
        out.append(Item(source, dt.isoformat(), None, "", title, link, "news"))
    return out


def google_url(q: str, days: int) -> str:
    return f"https://news.google.com/rss/search?q={quote(q + f' when:{days}d')}&hl=ja&gl=JP&ceid=JP:ja"


def fetch_news(cfg: Config, today: date, watch: list[tuple[str, str]], get: Callable[..., Any] | None = None) -> tuple[list[Item], list[dict[str, Any]]]:
    """watch: [(code, name)]＝個別に検索する銘柄。戻り値：記事（code は未設定のものがある）と、情報源ごとの取得状況。"""
    n = cfg["negative"]
    nc = n["news"]
    days, get = int(n["days"]), get or requests.get
    items: list[Item] = []
    status = []
    if not nc["enabled"]:
        return items, [{"name": "Googleニュース／Yahoo!ニュース", "ok": False, "n": 0, "note": "設定で無効"}]

    def grab(url: str, source: str) -> list[Item]:
        r = get(url, timeout=nc["timeout"], headers={"User-Agent": UA})
        r.raise_for_status()
        _time.sleep(float(nc["sleep"]))
        return parse_rss(r.text, source, today, days)
    g_ok = g_fail = 0
    g_items: list[Item] = []
    for q in nc["google_queries"]:
        try:
            g_items += grab(google_url(q, days), "Googleニュース")
            g_ok += 1
        except Exception as e:  # noqa: BLE001
            g_fail += 1
            log.warning("Googleニュース（%s…）を取得できません: %s", q[:12], e)
    kw = "下方修正 OR 不正 OR 訴訟 OR リコール OR 障害 OR 処分 OR 赤字 OR 減配 OR 漏洩 OR 破産 OR 上場廃止"
    w_ok = w_fail = 0
    for code, name in watch[: int(nc["per_company_max"])]:
        q = f'"{PREFIX.sub("", norm(name))}" ({kw})'
        try:
            for it in grab(google_url(q, days), "Googleニュース"):
                it.code = code
                g_items.append(it)
            w_ok += 1
        except Exception as e:  # noqa: BLE001
            w_fail += 1
            log.warning("Googleニュース（%s）を取得できません: %s", name, e)
    items += g_items
    status.append({"name": "Googleニュース（検索RSS）", "ok": g_ok > 0 or w_ok > 0, "n": len(g_items),
                   "note": f"横断検索 {g_ok}/{g_ok + g_fail}件・個別検索 {w_ok}/{w_ok + w_fail}件が成功"})
    y_items, y_ok, y_fail = [], 0, 0
    for url in nc["yahoo_feeds"]:
        try:
            y_items += grab(url, "Yahoo!ニュース")
            y_ok += 1
        except Exception as e:  # noqa: BLE001
            y_fail += 1
            log.warning("Yahoo!ニュースRSSを取得できません: %s", e)
    items += y_items
    status.append({"name": "Yahoo!ニュース（公式RSS・経済）", "ok": y_ok > 0, "n": len(y_items), "note": f"{y_ok}/{y_ok + y_fail}フィード"})
    return items, status


# ---------------------------------------------------------------- 取得：JPX・財務データ
def supervision_items(cfg: Config, today: date, fetch: Callable[[list[str]], set[str]] | None = None) -> tuple[list[Item], dict[str, Any]]:
    codes: set[str] = set()
    try:
        if fetch is None:
            from .supervision import fetch_supervision_codes as fetch
        codes = set(fetch(cfg["filters"]["supervision_urls"]))
    except Exception as e:  # noqa: BLE001
        log.warning("JPXの監理・整理銘柄一覧を取得できません: %s", e)
    items = [Item("JPX公式", today.isoformat(), c, "", "監理銘柄・整理銘柄一覧（JPX公式）に掲載", cfg["filters"]["supervision_urls"][0], "official") for c in sorted(codes)]
    return items, {"name": "JPX公式 監理・整理銘柄一覧", "ok": bool(codes), "n": len(codes), "note": "公式一覧に載っている銘柄" if codes else "取得できませんでした（0件の可能性もあります）"}


def financial_items(cache: dict[str, Any], today: date, cfg: Config) -> list[Item]:
    """取得済みの財務データ（data/fundamentals_cache.json）から、赤字転落・継続赤字。直近days日以内に取得したものだけ。"""
    out = []
    span = int(cfg["negative"]["days"])
    for code, hit in cache.items():
        try:
            fetched = date.fromisoformat(hit["fetched"])
            d = hit["data"]
        except (KeyError, ValueError, TypeError):
            continue
        if d.get("status") != "ok" or (today - fetched).days > span:
            continue
        a, ttm = d.get("op_annual") or [], d.get("op_ttm")
        if ttm is None:
            continue
        if len(a) >= 2 and ttm <= 0 and a[0] <= 0 and a[1] <= 0:
            out.append(Item("財務データ", fetched.isoformat(), code, "", "営業利益が直近12か月・直近2期ともマイナス（継続赤字）", "", "financial"))
        elif len(a) >= 2 and ttm <= 0 and a[1] > 0:
            out.append(Item("財務データ", fetched.isoformat(), code, "", "営業利益が黒字からマイナスに転落（赤字転落）", "", "financial"))
    return out


def fin_events(item: Item) -> list[Event]:
    key = "continuing_loss" if "継続赤字" in item.title else "loss_turn"
    return [Event(key, LABELS[key], item, 0.0)]


# ---------------------------------------------------------------- 検出日の記録
def update_seen(path: Path, entries: list[tuple[str, str]], today: date, keep_days: int = 60) -> dict[str, str]:
    """entries: [(キー, 公表日)]。初めて見たキーの検出日＝今日。60日より古い記録は消す。"""
    seen: dict[str, str] = {}
    if path.exists():
        try:
            seen = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            seen = {}
    for k, _ in entries:
        seen.setdefault(k, today.isoformat())
    lim = (today - timedelta(days=keep_days)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= lim}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, ensure_ascii=False), encoding="utf-8")
    return seen


def event_key(code: str, ev: Event) -> str:
    raw = f"{code}|{ev.key}|{ev.item.source}|{ev.item.url or norm(ev.item.title)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 全体
def analyze(items: list[Item], info: dict[str, dict], pairs: list[tuple[str, str]], today: date, cfg: Config,
            pdf_direction: Callable[[Item], str | None] | None = None) -> dict[str, dict[str, Any]]:
    """記事・開示を会社ごとにまとめて分類→採点。universe（info）に無い銘柄（ETF・REIT等）は対象外。"""
    n = cfg["negative"]
    span = int(n["days"])
    by_code: dict[str, list[Event]] = {}
    pdf_used = 0
    seen_pairs: set[tuple] = set()
    for it in items:
        if not it.date or (today - date.fromisoformat(it.date)).days > span or date.fromisoformat(it.date) > today:
            continue
        code = it.code or match_code(it.title, pairs, info)
        if not code or code not in info:
            continue
        sig = (code, it.source, norm(it.title))
        if sig in seen_pairs:
            continue
        seen_pairs.add(sig)
        if it.kind == "financial":
            evs = fin_events(it)
        else:
            direction = None
            if pdf_direction and is_ambiguous_revision(it) and pdf_used < int(n["pdf_max"]):
                pdf_used += 1
                direction = pdf_direction(it)
            evs = classify(it, direction)
        if it.source == "JPX公式":
            evs = [Event("kanri", "監理銘柄", it, 0.0)]       # 公式一覧は監理・整理の両方を含むため、点数は低い方（監理）で扱う。タイトルに「整理」があれば整理
        by_code.setdefault(code, []).extend(evs)
    res = {}
    for code, evs in by_code.items():
        if not evs:
            continue
        sc = company_score(evs, today, cfg)
        res[code] = {**sc, "events": evs}
    return res


def build_rows(res: dict[str, dict[str, Any]], info: dict[str, dict], seen: dict[str, str], today: date, cfg: Config) -> list[dict[str, Any]]:
    n = cfg["negative"]
    rows = []
    for code, r in res.items():
        if r["score"] < int(n["min_score"]):
            continue
        cls, blabel = band(r["score"], cfg)
        evs = sorted(r["events"], key=lambda e: (-event_points(e, today, cfg), e.item.date))
        dedup, seen_k = [], set()
        for e in evs:
            k = (e.key, e.item.source, norm(e.item.title))
            if k in seen_k:
                continue
            seen_k.add(k)
            dedup.append({"label": e.label, "key": e.key, "source": e.item.source, "date": e.item.date, "title": core_title(e.item, info[code]["name"]) or e.item.title,
                          "full_title": e.item.title, "url": e.item.url, "kind": e.item.kind,
                          "detected": seen.get(event_key(code, e), today.isoformat())})
        labels = sorted({p["event"].label for p in r["parts"]}, key=CATEGORY_ORDER.index)
        rows.append({"code": code, "name": info[code]["name"], "market": info[code]["market"], "sector": info[code]["sector"],
                     "score": r["score"], "band": cls, "band_label": blabel, "labels": labels,
                     "summary": summarize(info[code]["name"], r["parts"]),
                     "events": dedup[:8], "n_events": len(dedup),
                     "detected": min((d["detected"] for d in dedup), default=today.isoformat()),
                     "latest": max((d["date"] for d in dedup), default=None),
                     "official": any(d["kind"] == "official" for d in dedup),
                     "parts": [{"label": p["event"].label, "points": p["points"], "sources": p["sources"]} for p in r["parts"]]})
    rows.sort(key=lambda x: (-x["score"], -len(x["labels"]), x["code"]))
    return rows[: int(n["max_rows"])]


def page_context(rows: list[dict[str, Any]], status: list[dict[str, Any]], now: datetime, cfg: Config, demo: bool = False) -> dict[str, Any]:
    n = cfg["negative"]
    counts = {lab: sum(1 for r in rows if lab in r["labels"]) for lab in CATEGORY_ORDER}
    bands = {b: sum(1 for r in rows if r["band"] == b) for b in ("critical", "warn", "caution", "minor")}
    unsupported = [{"name": "X（旧Twitter）", "note": "無料で安定して使える取得手段が無いため未対応です"},
                   {"name": "各社のIR情報ページ", "note": "会社ごとに形式が違うため個別取得はしていません（法定開示はTDnetで確認しています）"}]
    return {"rows": rows, "status": status, "unsupported": unsupported, "counts": counts, "bands": bands, "categories": CATEGORY_ORDER,
            "generated": f"{now:%Y-%m-%d %H:%M}", "generated_iso": now.isoformat(), "days": n["days"], "min_score": n["min_score"], "demo": demo,
            "points": {LABELS[k]: v for k, v in n["points"].items() if k not in ("going_concern",)} | {"継続赤字（継続企業の前提）": n["points"]["going_concern"]},
            "bands_cfg": n["bands"], "rel": n["reliability"], "rec": n["recency"], "corr": n["corroboration"], "sev": n["severe_words"],
            "all_failed": not any(s["ok"] for s in status)}


def write_page(ctx: dict[str, Any], out_dir: Path) -> Path:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=select_autoescape(["html"]))
    shutil.copy(ROOT / "static" / "style.css", out_dir.parent / "style.css")
    (out_dir / "index.html").write_text(env.get_template("negative.html").render(**ctx), encoding="utf-8")
    slim = {"generated_at": ctx["generated_iso"], "days": ctx["days"], "rows": [{k: r[k] for k in ("code", "name", "score", "band_label", "labels", "summary", "detected", "latest")} for r in ctx["rows"]]}
    (out_dir.parent / "negative.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir / "index.html"


def run(demo: bool = False, out_dir: Path | None = None) -> Path:
    cfg = load_config()
    now = datetime.now(JST)
    today = now.date()
    if demo:
        info, pairs, items, status = demo_data(today)
    else:
        from . import universe
        uni = universe.load_universe(cfg)
        info, pairs = build_name_index(uni)
        items, status = [], []
        td, st = fetch_tdnet(cfg, today)
        items += td
        status.append(st)
        sup, st = supervision_items(cfg, today)
        items += sup
        status.append(st)
        cache_p = ROOT / "data" / "fundamentals_cache.json"
        try:
            fin = financial_items(json.loads(cache_p.read_text(encoding="utf-8")), today, cfg) if cache_p.exists() else []
        except (OSError, ValueError):
            fin = []
        items += fin
        status.append({"name": "財務データ（Yahoo Finance）", "ok": True, "n": len(fin), "note": "取得済みの銘柄の営業利益から判定（未取得の銘柄は対象外）"})
        pre = analyze(td + sup, info, pairs, today, cfg)
        watch = [(c, info[c]["name"]) for c, _ in sorted(pre.items(), key=lambda kv: -kv[1]["score"])]
        try:
            latest = json.loads(((out_dir or ROOT / "docs") / "latest.json").read_text(encoding="utf-8"))
            watch = [(p["code"], p["name"]) for p in latest.get("picks", []) if p["code"] in info] + watch
        except (OSError, ValueError, KeyError):
            pass
        seen_c, w2 = set(), []
        for c, nm in watch:
            if c not in seen_c:
                seen_c.add(c)
                w2.append((c, nm))
        news, nst = fetch_news(cfg, today, w2)
        items += news
        status += nst
        status += [{"name": "X（旧Twitter）", "ok": False, "n": 0, "note": "未対応"}]
    from .tdnet import pdf_text
    from .tdnet_events import direction_from_table, direction_from_text

    def pdf_dir(it: Item) -> str | None:
        t = pdf_text(it.url) if it.url and not demo else None
        return (direction_from_table(t) or direction_from_text(t)) if t else None
    res = analyze(items, info, pairs, today, cfg, pdf_dir)
    entries = [(event_key(code, e), e.item.date) for code, r in res.items() for e in r["events"]]
    seen = update_seen((ROOT / cfg["negative"]["seen_file"]) if not demo else (ROOT / "data" / "demo" / "negative_seen.json"), entries, today)
    rows = build_rows(res, info, seen, today, cfg)
    ctx = page_context(rows, status, now, cfg, demo)
    path = write_page(ctx, (out_dir or ROOT / "docs") / "negative")
    log.info("信用リスク監視ページを出力しました: %d銘柄（%s）", len(rows), path)
    return path


# ---------------------------------------------------------------- デモ（架空）
def demo_data(today: date):
    import pandas as pd
    uni = pd.DataFrame({"code": ["9001", "9002", "9003", "9004", "9005", "9006"],
                        "name": ["架空製作所", "サンプル電機", "デモ運輸", "テスト化学", "ダミー食品", "例示システムズ"],
                        "market": ["プライム"] * 6, "sector": ["機械", "電気機器", "陸運業", "化学", "食料品", "情報・通信業"]})
    info, pairs = build_name_index(uni)
    ago = lambda d: (today - timedelta(days=d)).isoformat()    # noqa: E731
    it = [Item("TDnet", ago(2), "9001", "架空製作所", "通期連結業績予想の下方修正に関するお知らせ", "", "official"),
          Item("TDnet", ago(2), "9001", "架空製作所", "配当予想の修正（減配）に関するお知らせ", "", "official"),
          Item("Googleニュース", ago(1), None, "", "架空製作所、最終赤字に転落へ 通期予想を下方修正 - 経済新聞（デモ）", "", "news"),
          Item("TDnet", ago(4), "9002", "サンプル電機", "不適切な会計処理に関する第三者委員会の設置のお知らせ", "", "official"),
          Item("TDnet", ago(9), "9002", "サンプル電機", "決算発表の延期に関するお知らせ", "", "official"),
          Item("Googleニュース", ago(3), None, "", "サンプル電機に粉飾の疑い、検察が家宅捜索 - 日刊デモ", "", "news"),
          Item("TDnet", ago(6), "9003", "デモ運輸", "システム障害の発生に関するお知らせ", "", "official"),
          Item("Yahoo!ニュース", ago(5), None, "", "デモ運輸で大規模なシステム障害、運行停止 - デモニュース", "", "news"),
          Item("TDnet", ago(12), "9004", "テスト化学", "当社製品の自主回収（リコール）に関するお知らせ", "", "official"),
          Item("TDnet", ago(14), "9005", "ダミー食品", "当社に対する訴訟の提起に関するお知らせ", "", "official"),
          Item("JPX公式", ago(0), "9006", "", "監理銘柄・整理銘柄一覧（JPX公式）に掲載", "", "official"),
          Item("TDnet", ago(1), "9006", "例示システムズ", "上方修正に関するお知らせ", "", "official")]
    status = [{"name": "TDnet（適時開示）", "ok": True, "n": 12, "note": "デモ"}, {"name": "Googleニュース（検索RSS）", "ok": True, "n": 3, "note": "デモ"},
              {"name": "Yahoo!ニュース（公式RSS・経済）", "ok": True, "n": 1, "note": "デモ"}, {"name": "X（旧Twitter）", "ok": False, "n": 0, "note": "未対応"}]
    return info, pairs, it, status


if __name__ == "__main__":
    run(demo="--demo" in sys.argv)
