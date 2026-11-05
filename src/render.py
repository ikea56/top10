"""HTMLの生成（docs/index.html）。"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import ROOT
from .explain import fmt_oku


def _yen(v: float | None) -> str:
    if v is None or v != v:
        return "—"
    return f"¥{v:,.0f}" if v >= 1000 else f"¥{v:,.1f}".replace(".0", "")


def _pct(v: float | None, digits: int = 1) -> str:
    return "—" if v is None or v != v else f"{v:+.{digits}f}%"


def render_site(ctx: dict[str, Any], out_dir: Path | None = None) -> Path:
    out = out_dir or ROOT / "docs"
    out.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=select_autoescape(["html"]))
    env.filters.update(yen=_yen, pct=_pct, oku=fmt_oku)
    (out / "index.html").write_text(env.get_template("index.html").render(**ctx), encoding="utf-8")
    shutil.copy(ROOT / "static" / "style.css", out / "style.css")
    shutil.copy(ROOT / "static" / "holdings.js", out / "holdings.js")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    keys = ("rank", "code", "name", "main", "pb_total", "td_score", "penalty", "parts", "label", "close",
            "strengths", "status", "cautions", "explain", "legacy")
    (out / "latest.json").write_text(json.dumps(
        {"generated_at": ctx["generated_iso"], "data_date": ctx["data_date_iso"], "demo": ctx["demo"],
         "picks": [{k: c[k] for k in keys} for c in ctx["cards"]]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out / "index.html"


def write_excluded_csv(rows: list[dict], path: Path) -> None:
    """除外された銘柄の一覧。Excelでそのまま開けるよう BOM付きUTF-8 で出力する。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["銘柄コード", "銘柄名", "除外理由", "検出イベント", "検出日"])
        for r in sorted(rows, key=lambda r: (r["reason"], r["code"])):
            w.writerow([r["code"], r["name"], r["reason"], r["events"], r["date"]])


def write_holdings_files(out: Path, universe, cards: list[dict] | None, passphrase: str | None, data_date, today, iterations: int) -> None:
    """names.json（コード→銘柄名。入力チェック用・公開情報）と、暗号化した保有銘柄の監視データを出力する。
    cards が None（保存先に接続できなかった等）のときは、前回の暗号化データをそのまま残す。"""
    from .holdings import encrypt_payload
    names = {str(c): str(n) for c, n in zip(universe["code"], universe["name"])}
    (out / "names.json").write_text(json.dumps(names, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if cards is None or not passphrase:
        return
    payload = {"generated_at": str(today), "data_date": data_date.strftime("%Y-%m-%d"),
               "holdings": {h["code"]: h for h in cards}}
    (out / "holdings_status.enc.json").write_text(
        json.dumps(encrypt_payload(payload, passphrase, iterations), separators=(",", ":")), encoding="utf-8")
