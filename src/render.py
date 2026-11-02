"""HTMLの生成（docs/index.html）。"""
import json
import shutil

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import ROOT
from .explain import fmt_oku


def _yen(v):
    if v is None or v != v:
        return "—"
    return f"¥{v:,.0f}" if v >= 1000 else f"¥{v:,.1f}".replace(".0", "")


def _pct(v, digits=1):
    if v is None or v != v:
        return "—"
    return f"{v:+.{digits}f}%"


def render_site(ctx, out_dir=None):
    out = out_dir or ROOT / "docs"
    out.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")),
                      autoescape=select_autoescape(["html"]))
    env.filters.update(yen=_yen, pct=_pct, oku=fmt_oku)
    html = env.get_template("index.html").render(**ctx)
    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copy(ROOT / "static" / "style.css", out / "style.css")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    slim = [{k: c[k] for k in ("rank", "code", "name", "score", "label", "close", "reasons", "explain", "risks")}
            for c in ctx["cards"]]
    (out / "latest.json").write_text(
        json.dumps({"generated_at": ctx["generated_iso"], "data_date": ctx["data_date_iso"],
                    "demo": ctx["demo"], "picks": slim}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out / "index.html"
