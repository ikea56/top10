"""東証の休場日（土日以外）を自動計算する。設定の更新は不要。
日本の祝日（2020年以降の制度）＋振替休日＋国民の休日＋東証の年末年始休場（12/31・1/1〜1/3）。
春分・秋分の日は天文計算の近似式（1980〜2099年に有効）。臨時の祝日（即位礼など）は config の earnings.extra_holidays に書く。"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _nth_monday(y: int, m: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(7 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _equinox(y: int, spring: bool) -> int:
    c = 20.8431 if spring else 23.2488
    return int(c + 0.242194 * (y - 1980) - (y - 1980) // 4)


@lru_cache(maxsize=None)
def holidays(y: int) -> frozenset[date]:
    h = {date(y, 1, 1), date(y, 2, 11), date(y, 2, 23), date(y, 4, 29), date(y, 5, 3), date(y, 5, 4), date(y, 5, 5),
         date(y, 8, 11), date(y, 11, 3), date(y, 11, 23),
         _nth_monday(y, 1, 2), _nth_monday(y, 7, 3), _nth_monday(y, 9, 3), _nth_monday(y, 10, 2),
         date(y, 3, _equinox(y, True)), date(y, 9, _equinox(y, False))}
    for d in sorted(h):                                   # 振替休日（日曜の祝日は次の平日以降）
        if d.weekday() == 6:
            n = d + timedelta(days=1)
            while n in h:
                n += timedelta(days=1)
            h.add(n)
    for d in sorted(h):                                   # 国民の休日（祝日に挟まれた平日）
        m = d + timedelta(days=2)
        if m in h and (d + timedelta(days=1)) not in h and (d + timedelta(days=1)).weekday() < 6:
            h.add(d + timedelta(days=1))
    h |= {date(y, 12, 31), date(y, 1, 2), date(y, 1, 3)}  # 東証の年末年始休場
    return frozenset(h)


def between(a: date, b: date) -> list[date]:
    return sorted(d for y in range(a.year, b.year + 1) for d in holidays(y) if a <= d <= b)
