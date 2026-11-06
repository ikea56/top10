"""前回ランキングとの比較：新規ランクイン・急上昇・急下降・ランクアウト。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_previous(hist_dir: Path, current_date: str, demo: bool) -> tuple[str | None, list[dict]]:
    """今回より前で、いちばん新しい記録のランキング（上位30。古い形式なら上位10）を返す。"""
    best: tuple[str, dict] | None = None
    for p in hist_dir.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        d = rec.get("date") or p.stem
        if d < current_date and bool(rec.get("demo")) == demo and (best is None or d > best[0]):
            best = (d, rec)
    if best is None:
        return None, []
    rec = best[1]
    return best[0], list(rec.get("ranking") or rec.get("picks") or [])


def compare(prev: list[dict], cur: list[dict], cfg: dict[str, Any]) -> dict[str, list[dict]]:
    """prev / cur は [{code, name, rank, main}]。show_top 位以内の出入りと、jump 位以上の変動を抜き出す。"""
    top, jump, limit = int(cfg["show_top"]), int(cfg["jump"]), int(cfg["limit"])
    pr = {x["code"]: x["rank"] for x in prev}
    cr = {x["code"]: x["rank"] for x in cur}
    names = {x["code"]: x["name"] for x in prev + cur}
    item = lambda c, **kw: {"code": c, "name": names[c], **kw}  # noqa: E731
    new = [item(c, rank=r, prev=pr.get(c)) for c, r in cr.items() if r <= top and pr.get(c, 10**9) > top]
    out = [item(c, rank=cr.get(c), prev=r) for c, r in pr.items() if r <= top and cr.get(c, 10**9) > top]
    skip = {x["code"] for x in new} | {x["code"] for x in out}
    both = [c for c in cr if c in pr and c not in skip]
    up = sorted((item(c, rank=cr[c], prev=pr[c], delta=pr[c] - cr[c]) for c in both if pr[c] - cr[c] >= jump), key=lambda x: -x["delta"])
    down = sorted((item(c, rank=cr[c], prev=pr[c], delta=cr[c] - pr[c]) for c in both if cr[c] - pr[c] >= jump), key=lambda x: -x["delta"])
    return {"new": sorted(new, key=lambda x: x["rank"])[:limit], "up": up[:limit], "down": down[:limit],
            "out": sorted(out, key=lambda x: x["prev"])[:limit]}


def annotate(prev: list[dict], cur: list[dict], top: int) -> dict[str, dict]:
    """各銘柄の「前回からの変化」。kind は new（新規ランクイン）/ up / down / same。
    new は『前回のトップN（既定10）に入っていなかった』銘柄で、前回が圏外（記録の上位30にもない）なら prev=None。
    prev が空（比較できる記録がない）なら空の辞書を返し、画面には何も出さない。"""
    if not prev:
        return {}
    pr = {x["code"]: x for x in prev}
    out: dict[str, dict] = {}
    for x in cur:
        p = pr.get(x["code"])
        if p is None or p["rank"] > top:
            kind, delta = "new", None
        else:
            delta = p["rank"] - x["rank"]
            kind = "up" if delta > 0 else "down" if delta < 0 else "same"
        out[x["code"]] = {"kind": kind, "prev": None if p is None else p["rank"], "delta": delta,
                          "main_delta": None if p is None or "main" not in p or p["main"] is None else x["main"] - p["main"]}
    return out
