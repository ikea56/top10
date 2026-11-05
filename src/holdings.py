"""保有銘柄の管理・監視。

保存先：Google スプレッドシート＋Apps Script（apps_script/Code.gs）。Webページから追加・削除でき、端末に依存しない。
監視結果：保有銘柄は公開サイトに載せたくないので、合言葉で暗号化（AES-256-GCM）した docs/holdings_status.enc.json に出力する。
          ページ側（static/holdings.js）が、合言葉を入力した人のブラウザの中だけで復号して表示する。
"""
from __future__ import annotations

import base64
import json
import os
import re
from datetime import date
from typing import Any

from .config import Config
from .log import get_logger

log = get_logger(__name__)
CODE_RE = re.compile(r"^[0-9]{3}[0-9A-Z]$")
ACTIONS = {
    "hold": {"emoji": "🟢", "text": "保有継続", "css": "bull"},
    "watch": {"emoji": "🟡", "text": "要監視", "css": "mild"},
    "sell": {"emoji": "🔴", "text": "売却検討", "css": "bad"},
}


class HoldingsError(RuntimeError):
    pass


def normalize_code(s: Any) -> str | None:
    """全角→半角・大文字化。4桁の証券コードとして不正なら None。"""
    t = str(s or "").strip().translate(str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
                                                      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")).upper()
    return t if CODE_RE.match(t) else None


# ------------------------------------------------------------------ 保存先との通信
def fetch_codes(endpoint: str, key: str, timeout: int = 30, retries: int = 2) -> list[str]:
    """Apps Script から保有銘柄コードの一覧を取得する（合言葉はHTTPSのPOST本文で送る。URLには載せない）。"""
    import time

    import requests
    last: Exception | None = None
    for i in range(retries + 1):
        try:
            r = requests.post(endpoint, data=json.dumps({"key": key, "action": "list"}),
                              headers={"Content-Type": "text/plain;charset=utf-8"}, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            if not j.get("ok"):
                raise HoldingsError(f"保有銘柄の取得に失敗: {j.get('error')}（合言葉の設定を確認してください）")
            codes = [c for c in (normalize_code(x) for x in j.get("holdings", [])) if c]
            return list(dict.fromkeys(codes))
        except HoldingsError:
            raise
        except Exception as e:                                   # noqa: BLE001 通信・形式の失敗は再試行
            last = e
            time.sleep(2 * (i + 1))
    raise HoldingsError(f"保有銘柄の取得に失敗しました（Apps Scriptに接続できません）: {last}")


# ------------------------------------------------------------------ 暗号化（ブラウザの WebCrypto で復号できる形式）
def encrypt_payload(obj: Any, passphrase: str, iterations: int = 200_000) -> dict:
    """PBKDF2-SHA256（200,000回）で鍵を作り、AES-256-GCM で暗号化する。"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations).derive(passphrase.encode("utf-8"))
    ct = AESGCM(key).encrypt(iv, json.dumps(obj, ensure_ascii=False).encode("utf-8"), None)
    b = lambda x: base64.b64encode(x).decode("ascii")  # noqa: E731
    return {"v": 1, "iter": iterations, "salt": b(salt), "iv": b(iv), "ct": b(ct)}


def decrypt_payload(blob: dict, passphrase: str) -> Any:
    """テスト用（ページ側は JavaScript で同じ処理をする）。"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    d = lambda k: base64.b64decode(blob[k])  # noqa: E731
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=d("salt"), iterations=int(blob["iter"])).derive(passphrase.encode("utf-8"))
    return json.loads(AESGCM(key).decrypt(d("iv"), d("ct"), None).decode("utf-8"))


# ------------------------------------------------------------------ 推奨アクションの判定
def decide_action(i: dict, cfg: Config) -> dict:
    """🔴売却検討／🟡要監視／🟢保有継続 を、理由つきで返す。

    i のキー：corporate(list[str] TOB等の検出)・supervised・price_lock・down_recent(list)・down_days(下方修正からの日数)・
              dividend_cut_days・trend_ok・broken(list[str] 押し目崩れの理由)・earnings_days(カレンダー日数)・
              earn_pen・unknown_rev・td_status("ok"/"unavailable"/"off")・data_ok
    """
    h = cfg["holdings"]
    sell: list[str] = []
    watch: list[str] = []
    for ev in i.get("corporate", []):
        sell.append(f"{ev}（企業再編・上場廃止関連）")
    if i.get("supervised"):
        sell.append("監理銘柄・整理銘柄に指定されています")
    if i.get("price_lock"):
        sell.append("TOBの疑い（急騰後に価格が固着）")
    if i.get("down_recent"):
        sell.append("直近30日以内に下方修正（" + "・".join(i["down_recent"]) + "）")
    elif i.get("down_days") is not None and i["down_days"] <= h["down_watch_days"]:
        watch.append(f"下方修正から{i['down_days']}日（30日を過ぎましたが警戒期間）")
    if i.get("dividend_cut_days") is not None and i["dividend_cut_days"] <= h["dividend_cut_days"]:
        sell.append(f"減配の開示（{i['dividend_cut_days']}日前）")
    if not i.get("data_ok", True):
        watch.append("株価データが不足しています（状態を判定できません）")
    for b in i.get("broken", []):
        watch.append(b)
    if i.get("earn_pen") or (i.get("earnings_days") is not None and 0 <= i["earnings_days"] <= h["earnings_watch_days"]):
        watch.append(f"決算が近い（あと{i['earnings_days']}日）・発表後に大きく動く可能性")
    if i.get("unknown_rev"):
        watch.append("業績予想の修正あり（増額か減額か未判定）")
    if i.get("td_status") != "ok":
        watch.append("TDnet（適時開示）を確認できていません")
    key = "sell" if sell else "watch" if watch else "hold"
    reasons = sell + watch if key != "hold" else ["下方修正・減配なし、トレンド維持"]
    return {"key": key, **ACTIONS[key], "reasons": reasons}
