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
