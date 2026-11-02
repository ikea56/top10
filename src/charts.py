"""ローソク足（直近60日）をSVGで描く。外部サービスに頼らないので常に表示できる。"""
import numpy as np


def candle_svg(o, h, l, c, v, ma25, ma75, w=320, ht=150):
    n = len(c)
    pad = 6
    vol_h = 26
    price_h = ht - pad * 2 - vol_h - 4

    series = [h, l]
    for m in (ma25, ma75):
        series.append(m)
    hi = float(np.nanmax(h))
    lo = float(np.nanmin(l))
    for m in (ma25, ma75):
        if np.isfinite(m).any():
            hi = max(hi, float(np.nanmax(m)))
            lo = min(lo, float(np.nanmin(m)))
    span = (hi - lo) or 1.0

    def y(val):
        return pad + (hi - val) / span * price_h

    step = (w - pad * 2) / n
    bw = max(step * 0.62, 1.2)
    vmax = float(np.nanmax(v)) or 1.0
    parts = []
    base = ht - pad
    for i in range(n):
        if not np.isfinite(c[i]) or not np.isfinite(o[i]):
            continue
        cx = pad + step * (i + 0.5)
        cls = "up" if c[i] >= o[i] else "dn"
        top, bot = y(max(o[i], c[i])), y(min(o[i], c[i]))
        parts.append(f'<line class="{cls}" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(h[i]):.1f}" y2="{y(l[i]):.1f}"/>')
        parts.append(f'<rect class="{cls}" x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{max(bot - top, 0.8):.1f}"/>')
        vh = (v[i] / vmax) * vol_h if np.isfinite(v[i]) else 0
        parts.append(f'<rect class="vol {cls}" x="{cx - bw / 2:.1f}" y="{base - vh:.1f}" '
                     f'width="{bw:.1f}" height="{max(vh, 0.5):.1f}"/>')

    def path(arr, cls):
        pts = [f"{pad + step * (i + 0.5):.1f},{y(val):.1f}" for i, val in enumerate(arr) if np.isfinite(val)]
        if len(pts) > 1:
            parts.append(f'<polyline class="{cls}" points="{" ".join(pts)}"/>')

    path(ma25, "ma25")
    path(ma75, "ma75")
    return (f'<svg viewBox="0 0 {w} {ht}" class="candle" role="img" '
            f'aria-label="直近{n}日のローソク足と25日・75日移動平均線">' + "".join(parts) + "</svg>")
