"""動作確認テスト。  python tests/smoke_test.py   （ネット接続不要・約30秒）
壊れていなければ最後に「ALL OK」と出ます。"""
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import backtest, events, explain, features, fundamentals, scoring   # noqa: E402
from src.config import ConfigError, load_config                              # noqa: E402
from src.supervision import parse_codes                                      # noqa: E402

cfg = load_config()
sc = cfg["scoring"]


def make_panel(last_vol_mult=3.5, n=300):
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(np.linspace(100, 200, n))
    close.iloc[-1] = close.iloc[-2] * 1.05
    vol = pd.Series(5_000_000.0, index=range(n))
    vol.iloc[-1] *= last_vol_mult
    mk = lambda s: pd.DataFrame({"T.T": s.values}, index=idx)  # noqa: E731
    return {"open": mk(close), "high": mk(close), "low": mk(close), "close": mk(close), "volume": mk(vol)}


def score_last(panel):
    f = features.compute_features(panel, None, sc)
    snap = features.snapshot(f, -1)
    t = scoring.score_technical(snap, "good", pd.Series({"T.T": 4.0}), sc)
    return snap, t.iloc[0]


def test_config_and_weights():
    scoring.validate_config(cfg)
    assert scoring.max_points(sc) == {"trend": 50, "demand": 25, "earnings": 15, "market": 10}
    bad = json.loads(json.dumps(cfg))
    bad["scoring"]["trend"]["macd"] = 9
    try:
        scoring.validate_config(bad)
        raise AssertionError("配点ずれを検出できていない")
    except ConfigError:
        pass
    print("ok  配点50/25/15/10と設定チェック")


def test_momentum_points():
    snap, t = score_last(make_panel(3.5))
    r = snap.iloc[0]
    assert r["hi252"] and r["hi60"] and r["hi20"] and r["vol_ratio"] == 3.5
    assert t["pts_high"] == 22, "252日高値更新が最大点"
    assert t["pts_ma"] == 15 and t["pts_vol"] == 18 and t["pts_upvol"] == 3 and t["pts_liq"] == 4
    assert t["pts_regime"] == 6 and t["pts_market"] == 10
    _, t2 = score_last(make_panel(1.2))
    assert t2["pts_vol"] == 0
    # 252日高値を割っている（ただし97%以内）→「52週高値接近」
    p = make_panel()
    for k in p:
        p[k] = p[k].copy()
    p["close"].iloc[-1] = p["close"].iloc[-2] * 0.99
    p["close"].iloc[-3] = p["close"].iloc[-2] * 1.0001
    snap3, t3 = score_last(p)
    assert not snap3.iloc[0]["hi252"] and snap3.iloc[0]["near52w"] and t3["pts_high"] == 12
    print("ok  モメンタム（20/60/252日高値・52週高値接近・出来高・トレンド）")


def test_no_lookahead():
    p = make_panel(3.5, 320)
    f_full = features.compute_features(p, None, sc)
    cut = {k: v.iloc[:290] for k, v in p.items()}
    f_cut = features.compute_features(cut, None, sc)
    for k in ("ma25", "hi252", "vol_ratio", "rsi", "rs20"):
        a, b = f_full[k].iloc[:290].astype(float).fillna(-1), f_cut[k].astype(float).fillna(-1)
        assert np.allclose(a, b), f"{k} に未来データが混入"
    print("ok  未来データ混入なし")


def fund(**kw):
    base = {"status": "ok", "market_cap": 6e10, "op_ttm": 5e8, "op_yoy": 0.3, "op_annual": [4e8, 3e8, 2e8, 1e8],
            "eps_hist": [40, 30, 20, 10], "eps_est_0y": None, "eps_est_1y": None, "eps_rev": None,
            "div_cur": 10, "div_prev": 10, "div_basis": "forecast", "div_streak": 0,
            "payout_latest": None, "payout_prev": None, "buyback": None}
    base.update(kw)
    return base


ROW = {"hi252": True, "near52w": True, "vol_ratio": 4.2}


def test_events():
    # 黒字転換（予想）＋継続黒字化＋増配20%＋最高益 ＋ 252日高値更新 → ボーナス
    f = fund(op_annual=[-1e8, -2e8, 1e8], op_ttm=-5e7, eps_est_0y=50, eps_est_1y=60, eps_hist=[-5, -3, 2],
             div_cur=12, div_prev=10, div_streak=5)
    r = events.evaluate_events(ROW, f, cfg)
    keys = {i.key for i in r.items}
    assert {"turnaround", "sustained", "record_high", "dividend", "streak", "bonus", "hi252", "volume"} <= keys
    pts = {i.key: i.points for i in r.items}
    assert pts["turnaround"] == 10 and pts["sustained"] == 4 and pts["record_high"] == 5
    assert pts["dividend"] == 6 and pts["streak"] == 4 and pts["bonus"] == 15   # 4つ同時
    assert r.total == cfg["events"]["cap"] and r.uses_forecast
    # 例：252日高値更新＋増配＋黒字転換 → +10
    f2 = fund(op_annual=[-1e8, 1e8, 1e8], op_ttm=-1e7, eps_est_0y=5, eps_hist=[-3, 5, 6], div_cur=11, div_prev=10)
    r2 = events.evaluate_events(ROW, f2, cfg)
    assert r2.bonus == 10 and not r2.has("record_high")
    # 252日高値でなければボーナスなし
    r3 = events.evaluate_events({**ROW, "hi252": False, "near52w": False}, f2, cfg)
    assert r3.bonus == 0 and r3.has("turnaround")
    # 増配率の段階（9%→2点, 10%→4点, 30%→8点）
    for g, want in ((1.09, 2), (1.10, 4), (1.30, 8)):
        it = events.evaluate_events(ROW, fund(div_cur=10 * g), cfg)
        assert {i.key: i.points for i in it.items}["dividend"] == want, g
    # 上方修正・自社株買い・配当性向
    r4 = events.evaluate_events(ROW, fund(eps_rev={"cur": 105, "d30": 100, "d90": 90}, buyback=3e8,
                                          payout_latest=0.45, payout_prev=0.35), cfg)
    assert {"revision", "buyback", "payout"} <= {i.key for i in r4.items}
    # 取得できない項目は評価しない
    r5 = events.evaluate_events(ROW, None, cfg)
    assert r5.total == 0
    print("ok  評価イベント（黒字転換・継続黒字化・最高益・上方修正・増配・連続増配・自社株買い・ボーナス）")


def test_dividend_and_exclusion():
    idx = pd.to_datetime([f"{y}-{m:02d}-01" for y in range(2019, 2027) for m in (6, 12)][:-1]).tz_localize("Asia/Tokyo")
    vals = [10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17]
    d = fundamentals.dividend_stats(pd.Series(vals, index=idx), 36.0, pd.Timestamp("2026-11-04"))
    assert d["div_streak"] == 6 and d["div_cur"] == 36.0 and d["div_prev"] == 33.0
    fc = cfg["fundamentals"]
    ok = {"op_ttm": 1e8, "op_annual": [1e8, 1e8]}
    assert fundamentals.exclusion_reason(ok, fc) is None
    assert fundamentals.exclusion_reason({"op_ttm": -1e8, "op_annual": [-1e8, 1e8]}, fc) == "赤字"
    assert fundamentals.exclusion_reason({"op_ttm": -1e8, "op_annual": [-1e8, -2e8]}, fc) == "継続赤字"
    # 前期だけ赤字で今期黒字予想 → 黒字転換候補として残す。継続赤字は予想があっても除外
    assert fundamentals.exclusion_reason({"op_ttm": -1e8, "op_annual": [-1e8, 1e8], "eps_est_0y": 5}, fc) is None
    assert fundamentals.exclusion_reason({"op_ttm": -1e8, "op_annual": [-1e8, -2e8], "eps_est_0y": 5}, fc) == "継続赤字"
    assert fundamentals._yoy(150, 100) == (0.5, False) and fundamentals._yoy(10, -5) == (None, True)
    html = ("<table><tr><th>銘柄名</th><th>コード</th><th>解除年月日</th></tr><tr><td>A</td><td>1234</td><td></td></tr>"
            "<tr><td>B</td><td>130A</td><td>2020/01/01</td></tr></table>")
    assert parse_codes(html, date(2026, 11, 4)) == {"1234"}
    flt = cfg["filters"]
    snap = pd.DataFrame({"close": [50, 500, 500, 500, 500], "volume": [1e6, 1e6, 100, 1e6, 1e6],
                         "turnover20": [1e9, 5e7, 1e9, 1e9, 1e9], "avgvol20": [1e6, 1e6, 1e6, 1e3, 1e6]})
    assert list(scoring.eligible_mask(snap, flt)) == [False, False, False, False, True]
    print("ok  配当計算・赤字/継続赤字の除外・監理銘柄パーサ・数値除外条件")


def test_expected_and_metrics():
    es = cfg["expected_score"]
    assert scoring.expected_score(80, 0.6, 1.0, es) == 96.0 and scoring.expected_score(80, 0.5, 0.85, es) == 68.0
    assert scoring.liquidity_factor(2e8, es) == 0.7 and scoring.liquidity_factor(6e9, es) == 1.0
    summ = {"bands": {"rows": [{"lo": 0, "hi": 50, "n": 500, "win": 0.4, "avg": 0},
                              {"lo": 50, "hi": 101, "n": 10, "win": 0.9, "avg": 0}]}}
    assert backtest.win_rate_lookup(summ, 30, es) == (0.4, 500, True)
    assert backtest.win_rate_lookup(summ, 80, es) == (0.5, 0, False)       # サンプル不足 → 仮置き
    assert backtest.win_rate_lookup(None, 80, es) == (0.5, 0, False)
    tm = backtest.trade_metrics(np.array([0.10, -0.05, 0.20, -0.05]))
    assert tm["win"] == 0.5 and abs(tm["avg_win"] - 0.15) < 1e-9 and abs(tm["avg_loss"] + 0.05) < 1e-9
    assert abs(tm["pf"] - 3.0) < 1e-9
    em = backtest.equity_metrics(np.array([0.1, -0.1, 0.1, 0.1]), 63)    # 63営業日×4期＝1年
    eq = 1.1 * 0.9 * 1.1 * 1.1
    assert abs(em["ann_ret"] - (eq - 1)) < 1e-9 and abs(em["max_dd"] + 0.1) < 1e-9
    r = np.array([0.1, -0.1, 0.1, 0.1])
    assert abs(em["sharpe"] - r.mean() / r.std(ddof=1) * 2) < 1e-9
    print("ok  期待値点・勝率の引き当て・勝率/PF/DD/シャープ/年率")


FORBIDDEN = ("上がります", "上昇します", "値上がり確実", "必ず上", "儲かり", "予想される株価", "上昇が見込", "買うべき")


def test_text():
    _, t = score_last(make_panel(3.5))
    snap, _ = score_last(make_panel(3.5))
    r = snap.iloc[0]
    r = r.copy()
    ev = events.evaluate_events(r, fund(op_annual=[-1, 1, 1], op_ttm=-1, eps_est_0y=3, eps_hist=[-1, 2, 2],
                                        div_cur=12, div_prev=10, op_yoy=0.35), cfg)
    txt = explain.reason_text(r, fund(op_yoy=0.35), ev, "good")
    assert 0 < len(txt) <= 100 and "出来高が通常の3.5倍に増加" in txt and "市場全体の地合いも良好" in txt
    assert not any(w in txt for w in FORBIDDEN)
    for lbl in scoring.bull_label(80, True, "good", cfg)[:1] + scoring.bull_label(80, False, "bad", cfg)[:1]:
        assert lbl == "やや強気"
    print("ok  理由文（100文字以内・予測表現なし）")


def test_pullback():
    from src import pullback
    pullback.validate_config(cfg)
    pb = cfg["pullback"]
    assert sum(pb["weights"].values()) == 100
    idx = pd.Index(["A.T", "B.T", "C.T"])
    snap = pd.DataFrame({
        "close": [100.0] * 3, "ma50": [100.0, 100.0, 100.0], "ma200": [80.0] * 3, "ma200_slope": [0.01] * 3,
        "gap50": [0.0, 0.0, 0.0], "dist52w": [-0.10, -0.10, -0.10], "vol_cont": [0.6, 0.6, 0.6],
        "ret5d": [0.0, 0.20, 0.0], "ret20d": [0.0, 0.0, 0.0], "ret1d": [0.0, 0.0, 0.0],
        "vol_ratio": [1.0, 1.0, 1.0], "rs60": [0.05] * 3, "ma5": [99.0] * 3, "ret3d": [0.01] * 3}, index=idx)
    snap.loc["C.T", "dist52w"] = -0.40
    try:
        t = pullback.score_technical(snap, "good", True, cfg)
    except KeyError as e:   # 列名が違う場合は実データの列名に合わせる
        raise AssertionError(f"snapshotの列名不一致: {e}")
    a, b, c = (t.loc[k] for k in idx)
    assert a["pb_sum"] - a["pb_penalty"] > b["pb_sum"] - b["pb_penalty"], "急騰は減点される"
    assert b["pb_penalty"] >= pb["penalties"]["surge_5d"]["pts"]
    assert a["pb_sum"] - a["pb_penalty"] > c["pb_sum"] - c["pb_penalty"], "高値から遠すぎる銘柄は低評価"
    assert pullback.total(80, 10, 5) == 85
    # 連続採点：近い値の銘柄でも点数が分かれる（同点が並びにくい）／各項目は満点を超えない
    xs = pd.DataFrame({"close": [100.0] * 5, "ma50": [100.0 / (1 + g) for g in (0.0, 0.01, 0.02, 0.03, 0.04)], "ma200": [80.0] * 5,
                       "ma200_slope": [0.01] * 5, "dist52w": [-0.06, -0.08, -0.10, -0.12, -0.14], "vol_cont": [0.6, 0.7, 0.75, 0.8, 0.85],
                       "ret5d": [0.0] * 5, "ret20d": [0.0] * 5, "ret1d": [0.0] * 5, "vol_ratio": [1.0] * 5,
                       "rs60": [0.02, 0.04, 0.06, 0.08, 0.10], "ma5": [99.0] * 5, "ret3d": [0.01] * 5,
                       "gap50": [0.0, 0.01, 0.02, 0.03, 0.04]}, index=[f"T{i}" for i in range(5)])
    tx = pullback.score_technical(xs, "good", True, cfg)
    assert tx["pb_sum"].nunique() == 5, tx["pb_sum"].tolist()
    mx = pullback.max_points(cfg["pullback"])
    for col, key in (("pb_trend", "trend"), ("pb_dd", "drawdown"), ("pb_ma50", "ma50"), ("pb_vol", "volume"),
                     ("pb_rs", "rs"), ("pb_bounce", "bounce")):
        assert (tx[col] <= mx[key] + 1e-9).all() and (tx[col] >= 0).all(), col
    # 相対力：TOPIXを上回るほど高得点／反発確認：調整局面で5日線の上に戻っていれば加点、高値圏では評価しない
    lo, hi = xs.iloc[[0]].copy(), xs.iloc[[0]].copy()
    lo["rs60"], hi["rs60"] = -0.05, 0.10
    assert pullback.score_technical(hi, "good", True, cfg)["pb_rs"].iloc[0] > pullback.score_technical(lo, "good", True, cfg)["pb_rs"].iloc[0]
    up, dn, hv = xs.iloc[[0]].copy(), xs.iloc[[0]].copy(), xs.iloc[[0]].copy()
    dn["ma5"], dn["ret3d"] = 101.0, -0.03
    hv["dist52w"] = -0.01
    assert pullback.score_technical(up, "good", True, cfg)["pb_bounce"].iloc[0] > pullback.score_technical(dn, "good", True, cfg)["pb_bounce"].iloc[0] == 0
    assert pullback.score_technical(hv, "good", True, cfg)["pb_bounce"].iloc[0] == 0 == pullback.score_technical(hv, "good", True, cfg)["pb_vol"].iloc[0]
    lab = pullback.label(90, True, cfg)[0]
    assert lab == "様子見", "下方修正・減配があれば様子見"
    print("ok  押し目スコア（急騰・高値乖離の減点／悪材料でラベル降格）")


def test_tdnet():
    from src import tdnet, tdnet_events as te
    D = tdnet.Disclosure
    today = date(2026, 11, 2)
    d = lambda t, dt="2026-10-20": D(dt, t, "")  # noqa: E731
    # 除外判定
    assert te.detect_exclusions(d("株式会社○○による当社株式に対する公開買付けに関する意見表明のお知らせ"))
    assert te.detect_exclusions(d("MBOの実施及び応募の推奨に関するお知らせ"))
    assert te.detect_exclusions(d("株式交換契約締結のお知らせ"))
    assert te.detect_exclusions(d("上場廃止に関するお知らせ"))
    assert te.detect_exclusions(d("監理銘柄（確認中）への指定のお知らせ"))
    assert not te.detect_exclusions(d("自己株式の取得及び自己株式の公開買付けに関するお知らせ")), "自己株式TOBは除外しない"
    # イベント点
    r = te.analyze([d("自己株式取得に係る事項の決定に関するお知らせ"), d("剰余金の配当（増配）に関するお知らせ"),
                    d("業績予想の上方修正に関するお知らせ")], None, cfg, today)
    assert r.score == 12 and not r.excluded, r.score          # 自社株買いは規模不明で+1、増配+5、上方修正+6
    r = te.analyze([d("業績予想の下方修正に関するお知らせ")], None, cfg, today)
    assert r.has_down_revision and te.serious_negative(r, None, cfg) and r.score == -10, r.score
    r = te.analyze([d("配当予想の修正（減配）に関するお知らせ")], None, cfg, today)
    assert r.score == -10, r.score
    # 同種イベントは1回だけ数える
    r = te.analyze([d("上方修正のお知らせ", "2026-10-01"), d("業績予想の上方修正のお知らせ", "2026-10-20")], None, cfg, today)
    assert r.score == 6, r.score
    # パーサ（形式の揺れに耐える）
    payload = {"items": [{"Tdnet": {"pubdate": "2026-10-20 15:00:00", "company_code": "72030",
                                    "title": "増配のお知らせ", "document_url": "https://x/y.pdf"}}]}
    ds = tdnet.parse_payload(payload, "7203")
    assert len(ds) == 1 and ds[0].title == "増配のお知らせ" and ds[0].date == "2026-10-20"
    assert tdnet.parse_payload({"items": []}, "7203") == []
    for bad in ({}, None, "x"):          # 形式が違えばエラー → 「TDnet未確認」警告になる
        try:
            tdnet.parse_payload(bad, "7203")
            raise AssertionError("想定外の形式でエラーにならない")
        except tdnet.TdnetError:
            pass
    print("ok  TDnet解析（除外・自己株TOB非除外・加点減点・重複・パーサ）")


def test_accuracy_21():
    from src import pullback, tdnet, tdnet_events as te
    D = tdnet.Disclosure
    today = date(2026, 11, 2)                                     # 月曜
    d = lambda t, dt, url="": D(dt, t, url)  # noqa: E731
    # ① 地合い -15〜+10
    mp = lambda rg, nk, b200: pullback.market_points(rg, nk, b200, cfg)  # noqa: E731
    assert mp("good", True, False) == 10 and mp("neutral", True, False) == 7
    assert mp("bad", True, False) == -6 and mp("bad", False, False) == -13
    assert mp("bad", False, True) == -15, "下限は-15"
    assert mp("good", False, True) == 6 - 5 - 3
    # ⑤ 決算接近（営業日）
    from src import earnings as E
    ea = lambda nx, past=None: E.assess({"next_earnings": nx, "earnings_past": past or []}, today, cfg)  # noqa: E731
    # 今日=11/2(月)、11/3は文化の日（休場）。営業日：11/4=1, 11/5=2, 11/6=3, 11/9=4, 11/10=5, 11/11=6
    assert [ea(d)["days_to"] for d in ("2026-11-02", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-10", "2026-11-11")] == [0, 1, 2, 3, 5, 6]
    assert [ea(d)["pen"] for d in ("2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-17", "2026-11-18")] == [25, 15, 15, 10, 10, 5, 5, 0]
    assert ea("2026-11-02")["pen"] == 25 and ea(None)["pen"] == 0 and ea("2026-10-30")["pen"] == 0
    assert ea("2026-11-06")["badges"] == ["⚠ 決算接近", "⚠ 決算跨ぎ注意"] and ea("2026-11-10")["badges"] == ["⚠ 決算接近"] and ea("2026-11-11")["badges"] == []
    assert ea("2026-11-30")["pen"] == 0 and ea("2026-11-30")["state"] == "ample"
    p1 = ea("2027-02-05", ["2026-10-20"])                    # 9営業日前に発表 → 通過済み
    assert p1["state"] == "passed" and p1["badges"] == ["✅ 決算通過済み"] and p1["base"] == 3 and p1["text"].startswith("決算通過後")
    assert ea("2027-02-05", ["2026-11-02"])["state"] == "pending", "発表当日〜翌営業日は反応待ち"
    assert ea("2027-02-05", ["2026-09-01"])["state"] == "ample", "30営業日を過ぎたら通過ボーナスなし"
    assert ea("2026-11-06", ["2026-10-20"])["state"] == "approach", "接近が優先"
    es_ = ea(None, ["2026-08-20"])                           # 次回不明 → 前回+91日の推定
    assert es_["estimate"] and es_["next"] == "2026-11-19" and es_["days_to"] == 12 and "推定" in es_["text"]
    assert ea(None)["state"] == "unknown" and "取得できません" in ea(None)["text"]
    assert E.busdays(date(2026, 11, 20), date(2026, 11, 24), cfg) == 1, "11/23（祝日）と土日を除く（金のみ）"
    from src import jpholidays as J, main as MAIN
    hd = lambda y: {d.strftime("%m-%d") for d in J.holidays(y) if d.weekday() < 5}  # noqa: E731
    assert {"09-21", "09-22", "09-23", "05-06", "03-20", "12-31"} <= hd(2026), "2026：敬老・国民の休日・秋分・振替・春分"
    assert {"03-22", "09-20", "09-23", "05-03", "01-11", "07-19"} <= hd(2027) and "11-23" in hd(2027)
    from types import SimpleNamespace as NS
    td = NS(status="ok", hits=[NS(kind="dividend_increase", date="2026-10-25"), NS(kind="upward_revision", date="2026-10-01"),
                                NS(kind="buyback", date="2026-10-28")])
    evx = NS(has=lambda k: k in ("record_high", "turnaround"))
    lb = E.good_news(p1, td, evx)
    assert set(lb) == {"増配", "自社株買い", "黒字転換", "最高益更新"}, "上方修正は決算前の開示なので対象外（表示のみ・点数なし）"
    assert E.good_news(ea("2026-11-06"), td, evx) == [], "通過済みでなければ表示なし"
    assert "bonus" not in cfg["earnings"], "好材料の再加点（二重加点）は設定にも存在しない"
    # 二重加点がないこと：決算通過後の加点は基本点だけで、好材料の有無で総合点が変わらない
    import copy
    ca = {"pb_total": 70.0, "fund": {}, "earn": p1, "ev": evx, "td": None}
    cb = dict(ca, ev=NS(has=lambda k: False))
    assert MAIN.final_score(ca, cfg)[0] == MAIN.final_score(cb, cfg)[0] == 70.0
    assert E.cap_label("押し目良好", "bull", ea("2026-11-04"), cfg)[0] == "様子見"
    assert E.cap_label("押し目良好", "bull", ea("2026-11-06"), cfg)[0] == "押し目候補"
    assert E.cap_label("押し目良好", "bull", ea("2026-11-11"), cfg)[0] == "押し目良好"
    assert E.cap_label("押し目良好", "bull", p1, cfg)[0] == "押し目良好"
    # 決算直前の良い押し目は、通過後の押し目より下に来る
    hi = pullback.total(70, 15, 0, ea("2026-11-04")["pen"])
    lo = pullback.total(60, 15, 0, p1["base"] * -1)
    assert hi < lo, "決算前（-25）は押し目スコアが10点高くても通過後に負ける"
    # ③ 鮮度
    assert [te.freshness(x, cfg) for x in (0, 30, 31, 90, 91, 180, 181)] == [1, 1, 0.7, 0.7, 0.4, 0.4, 0]
    r = te.analyze([d("業績予想の上方修正のお知らせ", "2026-09-03")], None, cfg, today)     # 60日前
    assert abs(r.score - 6 * 0.7) < 1e-9 and r.hits[0].weight == 0.7
    r = te.analyze([d("業績予想の上方修正のお知らせ", "2026-10-20")], None, cfg, today)
    assert r.score == 6
    assert te.analyze([d("業績予想の上方修正のお知らせ", "2026-03-01")], None, cfg, today).score == 0   # 180日超
    # ④ 自社株買い：時価総額比
    assert [te.buyback_points(x, cfg) for x in (0.005, 0.01, 0.02, 0.03, 0.08, None)] == [1, 3, 3, 5, 5, 1]
    assert abs(te.buyback_ratio_from_text("発行済株式総数（自己株式を除く）に対する割合 2.50％", None) - 0.025) < 1e-9
    assert abs(te.buyback_ratio_from_text("取得価額の総額 上限 30億円", 1e11) - 0.03) < 1e-9
    assert te.buyback_ratio_from_text("関係なし", 1e11) is None
    pdf = lambda u: "取得する株式の総数 1,000,000株（発行済株式総数（自己株式を除く）に対する割合 3.2％）"  # noqa: E731
    r = te.analyze([d("自己株式取得に係る事項の決定に関するお知らせ", "2026-10-25", "x.pdf")], {"market_cap": 1e11}, cfg, today, pdf)
    assert r.score == 5 and "3.2%" in r.hits[0].label, (r.score, r.hits[0].label)
    # ② 下方修正：30日以内は-15・様子見。それより前は対象外
    r = te.analyze([d("業績予想の下方修正に関するお知らせ", "2026-10-20")], None, cfg, today)
    assert te.recent_downward(r, None, cfg) == ["TDnet"] and te.serious_negative(r, None, cfg)
    r = te.analyze([d("業績予想の下方修正に関するお知らせ", "2026-09-01")], None, cfg, today)     # 62日前
    assert te.recent_downward(r, None, cfg) == [] and abs(r.score - (-10 * 0.7)) < 1e-9
    okr = te.TdnetResult()
    assert te.recent_downward(okr, {"eps_rev": {"cur": 90, "d30": 100, "d90": 100}}, cfg) == ["EPS予想"]
    assert te.recent_downward(okr, {"eps_rev": {"cur": 99, "d30": 100, "d90": 100}}, cfg) == []
    assert pullback.label(95, True, cfg)[0] == "様子見" and pullback.label(95, False, cfg)[0] == "押し目良好"
    # ⑦ 修正の方向（表・語・△）
    up = "修正前 修正後\n売上高 1,000 1,100\n営業利益 100 150\n経常利益 100 140\n増減額 50"
    dn = "売上高 1,000 900\n営業利益 100 △50\n経常利益 100 80\n増減額 △50"
    assert te.direction_from_text(up) == "up" and te.direction_from_text(dn) == "down"
    assert te.direction_from_text("特に記載なし") is None
    r = te.analyze([d("業績予想の修正に関するお知らせ", "2026-10-28", "u.pdf")], None, cfg, today, lambda u: dn)
    assert r.hits[0].kind == "downward_revision"
    r = te.analyze([d("通期業績予想の引き上げに関するお知らせ", "2026-10-28")], None, cfg, today)
    assert r.hits[0].kind == "upward_revision"
    r = te.analyze([d("業績予想の修正に関するお知らせ", "2026-10-28")], None, cfg, today)
    assert r.hits[0].kind == "unknown_revision" and r.hits[0].label == "修正あり（未判定）"
    # ⑥ ATR撤退ライン
    row = {"close": 1000.0, "ma50": 960.0, "ma200": 800.0, "atr": 20.0}
    sl = pullback.stop_line(row, cfg)
    assert abs(sl["price"] - 940) < 1e-9 and abs(sl["risk_pct"] - 6.0) < 1e-9 and abs(sl["atr_pct"] - 2.0) < 1e-9
    row2 = {"close": 1000.0, "ma50": 1100.0, "ma200": 1200.0, "atr": 20.0}
    assert abs(pullback.stop_line(row2, cfg)["price"] - 960) < 1e-9
    assert pullback.stop_line({**row, "atr": float("nan")}, cfg) is None
    from src import features
    n = 40; ix = pd.bdate_range("2026-01-01", periods=n)
    base = pd.DataFrame({"A": 100.0 + np.arange(n)}, index=ix)
    panel = {"close": base, "high": base + 2, "low": base - 2, "open": base, "volume": base * 0 + 1}
    fx = features.add_live_extras({"close": base, "ret1d": base.pct_change()}, panel, 14)
    assert abs(float(fx["atr"].iloc[-1, 0]) - 4.0) < 0.3, fx["atr"].iloc[-1, 0]
    print("ok  ㉑ 地合い・決算接近・鮮度・自社株買い・下方修正・修正判定・ATR")


def test_holdings():
    from src import holdings as H, movers
    # コードの正規化
    assert H.normalize_code("７２０３") == "7203" and H.normalize_code("285a") == "285A"
    assert H.normalize_code("12") is None and H.normalize_code("72031") is None and H.normalize_code("") is None
    # 暗号化（往復・誤った合言葉・毎回違う暗号文）
    blob = H.encrypt_payload({"a": "日本語"}, "pass-pass-1", 1000)
    assert H.decrypt_payload(blob, "pass-pass-1") == {"a": "日本語"}
    assert H.encrypt_payload({"a": 1}, "pass-pass-1", 1000)["ct"] != H.encrypt_payload({"a": 1}, "pass-pass-1", 1000)["ct"]
    try:
        H.decrypt_payload(blob, "wrong-pass-1")
        raise AssertionError("誤った合言葉で復号できた")
    except AssertionError:
        raise
    except Exception:
        pass
    assert "日本語" not in json.dumps(blob)
    # 推奨アクション
    base = {"corporate": [], "supervised": False, "price_lock": False, "down_recent": [], "down_days": None,
            "dividend_cut_days": None, "broken": [], "earnings_days": None, "earn_pen": 0, "unknown_rev": False,
            "td_status": "ok", "data_ok": True}
    act = lambda **kw: H.decide_action({**base, **kw}, cfg)  # noqa: E731
    assert act()["key"] == "hold" and act()["text"] == "保有継続"
    assert act(down_recent=["TDnet"])["key"] == "sell"
    assert act(dividend_cut_days=40)["key"] == "sell" and act(dividend_cut_days=120)["key"] == "hold"
    assert act(corporate=["公開買付け"])["key"] == "sell" and act(supervised=True)["key"] == "sell"
    assert act(price_lock=True)["key"] == "sell"
    assert act(down_days=60)["key"] == "watch", "30日を過ぎた下方修正は要監視"
    assert act(broken=["200日線割れ"])["key"] == "watch" and act(earnings_days=4)["key"] == "watch"
    assert act(earnings_days=5)["key"] == "watch" and "要監視" in act(earnings_days=5)["reasons"][0]
    assert "決算跨ぎ注意" in act(earnings_days=3)["reasons"][0] and act(earnings_days=8)["key"] == "hold" and act(earnings_days=30)["key"] == "hold"
    assert act(unknown_rev=True)["key"] == "watch" and act(td_status="unavailable")["key"] == "watch"
    assert act(data_ok=False)["key"] == "watch"
    assert act(down_recent=["TDnet"], broken=["x"])["key"] == "sell", "売却検討が最優先"
    # 前回比較（判定のテストはトップ10基準で行う）
    mc = {**cfg["movers"], "show_top": 10}
    mk = lambda ranks: [{"code": c, "name": f"N{c}", "rank": r, "main": 80} for c, r in ranks.items()]  # noqa: E731
    prev = mk({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 20, "M": 25})
    cur = mk({"A": 1, "K": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "N": 10, "B": 18, "M": 14, "L": 29, "J": 30})
    r = movers.compare(prev, cur, mc)
    assert [x["code"] for x in r["new"]] == ["K", "N"] and r["new"][1]["prev"] is None
    assert [x["code"] for x in r["out"]] == ["B", "J"] and r["out"][0]["rank"] == 18
    assert [x["code"] for x in r["up"]] == ["M"] and r["up"][0]["delta"] == 11
    assert [x["code"] for x in r["down"]] == ["L"] and r["down"][0]["delta"] == 9
    ann = movers.annotate(prev, cur, 10)
    assert ann["K"]["kind"] == "new" and ann["K"]["prev"] == 11, "前回11位→今回2位はトップ10の新規ランクイン"
    assert ann["N"]["kind"] == "new" and ann["N"]["prev"] is None, "前回圏外は prev=None"
    assert ann["A"]["kind"] == "same" and ann["A"]["delta"] == 0 and ann["A"]["main_delta"] == 0
    assert movers.annotate(mk({"A": 1}), mk({"A": 1}), 10)["A"]["kind"] == "same"
    p2 = mk({"X": 5}); c2 = mk({"X": 2}); a2 = movers.annotate(p2, c2, 10)["X"]
    assert a2["kind"] == "up" and a2["delta"] == 3 and movers.annotate(c2, p2, 10)["X"]["kind"] == "down"
    assert movers.annotate([], cur, 10) == {}, "前回の記録がなければ何も出さない"
    cur2 = mk({"A": 1, "L": 9}); prev2 = mk({"A": 1, "L": 4})
    assert movers.compare(prev2, cur2, mc)["down"][0]["code"] == "L"
    print("ok  保有銘柄（コード正規化・暗号化・推奨アクション・前回比較）")


def test_excluded_csv(tmp=ROOT / "data" / "demo" / "_ex_test.csv"):
    from src.render import write_excluded_csv
    tmp.parent.mkdir(parents=True, exist_ok=True)
    write_excluded_csv([{"code": "1234", "name": "テスト", "reason": "TOB・公開買付", "events": "公開買付け", "date": "2026-10-11"}], tmp)
    raw = tmp.read_text(encoding="utf-8-sig").splitlines()
    assert raw[0] == "銘柄コード,銘柄名,除外理由,検出イベント,検出日" and raw[1].startswith("1234,テスト")
    tmp.unlink()
    print("ok  excluded_events.csv の列と内容")


def test_data_fixes():
    from datetime import datetime
    from src import main as m, prices
    # 分割の調整漏れ（10分割で価格が1/10に）を補正
    idx = pd.bdate_range("2024-01-01", periods=10)
    c = pd.Series([100, 101, 102, 103, 104, 10.5, 10.6, 10.7, 10.8, 10.9], index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c})
    fx = prices.fix_splits(df)
    assert (fx["close"].pct_change().abs().dropna() < 0.05).all(), fx["close"].tolist()
    assert abs(fx["close"].iloc[-1] - 10.9) < 1e-9
    # 通常の値動き（-10%）は触らない
    c2 = pd.Series([100, 90, 95, 99], index=idx[:4]); d2 = pd.DataFrame({"open": c2, "high": c2, "low": c2, "close": c2})
    assert prices.fix_splits(d2)["close"].tolist() == c2.tolist()
    # 取引時間中の今日の足は捨てる／引け後は使う
    jst = __import__("datetime").timezone(__import__("datetime").timedelta(hours=9))
    pidx = pd.to_datetime(["2026-11-04", "2026-11-05"])
    panel = {"close": pd.DataFrame({"A": [1.0, 2.0]}, index=pidx)}
    assert len(m.drop_unfinished_bar(panel, datetime(2026, 11, 5, 10, 36, tzinfo=jst))["close"]) == 1
    assert len(m.drop_unfinished_bar(panel, datetime(2026, 11, 5, 7, 0, tzinfo=jst))["close"]) == 1
    assert len(m.drop_unfinished_bar(panel, datetime(2026, 11, 5, 17, 0, tzinfo=jst))["close"]) == 2
    assert len(m.drop_unfinished_bar(panel, datetime(2026, 11, 6, 7, 0, tzinfo=jst))["close"]) == 2
    print("ok  分割補正・取引時間中の未確定足の除外")


def _fake_records(signal, n_dates=120, n=300, seed=1):
    import numpy as np
    import pandas as pd
    from src import validation as V
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        idx = [f"{1000 + j}.T" for j in range(n)]
        s = rng.uniform(0, 100, n)
        r = signal * (s - 50) / 100 + rng.normal(0, 0.08, n)
        x = pd.DataFrame({"s": s, "pen": 0.0, "r_1m": r, "rc_1m": r - 0.001}, index=idx)
        for c in V.COMPONENTS:
            x[c] = rng.uniform(0, 10, n)
        top = list(x.sort_values("s", ascending=False).index[:10])
        recs.append({"date": pd.Timestamp("2020-01-06") + pd.Timedelta(days=7 * d), "regime": ["good", "neutral", "bad"][d % 3],
                     "bench": {"1m": 0.0}, "x": x, "pb_picks": top, "lg_picks": top, "stop": None, "sector": (5.0, 0.3)})
    return recs


def test_validation():
    import numpy as np
    from src import validation as V
    hz = {"1w": 5, "1m": 20, "3m": 60}
    # 点数と成績が結びつくデータ → 有効と判定、乱数データ → 有効と判定されない
    good = V.analyze(_fake_records(0.3), cfg, pd.Timestamp("2022-12-31"), 5, hz)
    assert good["ok"] and good["verdict"]["level"] in ("good", "weak"), good["verdict"]
    assert good["ic"]["s"]["mean"] > 0.2 and good["top"]["excess"] > 0 and good["top"]["ci"][0] > 0
    rows = [b["excess"] for b in good["bands"]["rows"] if b["excess"] is not None]
    assert rows == sorted(rows), "点数帯が高いほど成績が良いはず"
    assert good["quantiles"]["spread"] > 0
    rnd = V.analyze(_fake_records(0.0, seed=7), cfg, pd.Timestamp("2022-12-31"), 5, hz)
    assert rnd["verdict"]["level"] != "good", rnd["verdict"]
    assert abs(rnd["ic"]["s"]["mean"]) < 0.05
    # 同じ入力なら同じ結果（信頼区間の乱数固定）
    again = V.analyze(_fake_records(0.0, seed=7), cfg, pd.Timestamp("2022-12-31"), 5, hz)
    assert again["top"]["ci"] == rnd["top"]["ci"]
    # 補助関数
    a = np.arange(10.0)
    assert len(V._indep(list(a), 4)) == 3 and V.tstat(np.array([1.0] * 10)) is None
    assert V.spearman(np.arange(50.0), np.arange(50.0)) > 0.99 and V.spearman(np.arange(5.0), np.arange(5.0)) is None
    r = [{"pb_picks": ["a", "b"]}, {"pb_picks": ["a", "c"]}]
    assert abs(V.turnover(r, "pb_picks", 2)["retention"] - 0.5) < 1e-9
    # 撤退ラインのシミュレーション：下落した銘柄だけ早く売られる
    c = pd.DataFrame({"A": [100, 99, 90, 80, 70, 60], "B": [100, 101, 102, 103, 104, 105]}, index=range(6)).to_numpy(float)
    ma = pd.Series({"A": 95.0, "B": 95.0}); atr = pd.Series({"A": 2.0, "B": 2.0})
    st = backtest._simulate_stop(c, pd.Index(["A", "B"]), 0, 4, ["A", "B"], ma, ma, atr, cfg["pullback"]["atr"], 0.0,
                                 pd.Series({"A": 70 / 99 - 1, "B": 104 / 101 - 1}))
    assert st["hit"] == [1.0, 0.0] and st["st"][0] > st["ns"][0] and st["st"][1] == st["ns"][1], st
    # 画面用の表に禁止語が無い
    txt = json.dumps(good["tables"], ensure_ascii=False) + json.dumps(good["verdict"], ensure_ascii=False)
    assert not any(w in txt for w in FORBIDDEN)
    print("ok  有効性の検証（有意な点数と乱数の区別・信頼区間の再現性・撤退シミュレーション・回転率）")


def test_stance():
    from src import market as M, pullback as PB
    n = 260
    idx = pd.bdate_range(end="2026-11-10", periods=n)
    up = pd.Series(np.linspace(100, 150, n), index=idx)
    down = pd.Series(np.linspace(150, 100, n), index=idx)
    mk = lambda **k: {key: pd.DataFrame({"close": v}) for key, v in k.items()}  # noqa: E731
    assert M.market_stance(mk(topix=up, n225=up, growth=up), cfg)["key"] == "buy"
    assert M.market_stance(mk(topix=up, n225=up), cfg)["key"] == "buy"                 # グロース無しでも判定できる
    assert M.market_stance(mk(topix=up, n225=up, growth=down), cfg)["key"] == "neutral"
    assert M.market_stance(mk(topix=up, n225=down, growth=up), cfg)["key"] == "neutral"
    d = M.market_stance(mk(topix=down, n225=down, growth=down), cfg)
    assert d["key"] == "defense" and d["label"] == "防御優先" and len(d["reasons"]) == 4 and d["growth_weak"]
    assert "新規買いは慎重に" in d["guide"] and "現金比率を高める" in d["guide"]
    assert M.market_stance(mk(topix=down, n225=up, growth=up), cfg)["key"] == "neutral"
    short = M.market_stance(mk(topix=up.iloc[:30]), cfg)
    assert short["key"] == "neutral"
    # 地合い点：グロースが弱いと減点、下限は-15
    base = PB.market_points("good", True, False, cfg)
    assert PB.market_points("good", True, False, cfg, True) == base + cfg["pullback"]["market"]["growth_below_ma75"]
    assert PB.market_points("bad", False, True, cfg, True) == cfg["pullback"]["market"]["floor"]
    txt = json.dumps(M.STANCE, ensure_ascii=False)
    assert "売り優勢" not in txt and not any(w in txt for w in FORBIDDEN)
    print("ok  相場環境の判定（買い優勢/中立/防御優先・グロース欠損時・地合い点の減点）")


def test_premarket():
    from datetime import datetime, timedelta
    from src import market as M, premarket as P
    from src.config import JST
    now = datetime(2026, 11, 13, 7, 10, tzinfo=JST)
    ref = datetime(2026, 11, 12, 15, 30, tzinfo=JST)
    idx = pd.date_range("2026-11-12 14:00", "2026-11-13 07:00", freq="15min", tz=JST)
    s = pd.Series(100.0, index=idx); s[s.index > ref] = 97.0
    r = P.change_since_tokyo_close(s, ref, now, 14)
    assert r["status"] == "ok" and abs(r["pct"] + 3.0) < 1e-6, r
    assert P.change_since_tokyo_close(s.iloc[:8], ref, now, 14)["status"] == "stale"        # 値が古い
    assert P.change_since_tokyo_close(pd.Series(dtype=float), ref, now, 14)["status"] == "missing"
    late = pd.Series(1.0, index=pd.date_range("2026-11-13 05:00", "2026-11-13 07:00", freq="15min", tz=JST))
    assert P.change_since_tokyo_close(late, ref, now, 14)["status"] == "missing"             # 引け時点の値が無い
    d = pd.Series([100.0, 98.0], index=pd.to_datetime(["2026-11-11", "2026-11-12"]).tz_localize(JST))
    assert abs(P.change_daily(d, now, 14)["pct"] + 2.0) < 1e-6
    ok = lambda p: {"status": "ok", "pct": p, "asof": "t"}  # noqa: E731
    keys = ("n225f", "topixf", "usdjpy", "sp500", "nasdaq", "sox")
    calm = P.evaluate({k: ok(0.1) for k in keys}, cfg, now)
    assert not calm["gap_down"] and not calm["weak_hit"] and calm["available"] == 6
    gap = P.evaluate({**{k: ok(0.0) for k in keys}, "n225f": ok(-1.6)}, cfg, now)
    assert gap["gap_down"] and "日経225先物" in gap["gap_detail"][0]
    # 先物は重み3：先物だけが弱い場合と、米国3指数だけが弱い場合で評価点が違う
    only_f = P.evaluate({**{k: ok(0.0) for k in keys}, "n225f": ok(-1.2), "topixf": ok(-1.2)}, cfg, now)
    only_us = P.evaluate({**{k: ok(0.0) for k in keys}, "sp500": ok(-2.0), "nasdaq": ok(-3.0), "sox": ok(-4.0)}, cfg, now)
    assert only_f["score"] < only_us["score"] and only_f["weak_hit"] and not only_us["gap_down"]
    miss = P.evaluate({}, cfg, now)
    assert miss["score"] is None and miss["available"] == 0 and not miss["gap_down"]
    # 判定への反映：弱める方向だけ
    mk = lambda key: {"key": key, **M.STANCE[key], "reasons": [], "guide": list(M.STANCE[key]["guide"]), "label": M.STANCE[key]["label"]}  # noqa: E731
    assert P.apply_to_stance(mk("buy"), gap, cfg)["key"] == "neutral"
    assert P.apply_to_stance(mk("neutral"), gap, cfg)["key"] == "defense"
    assert P.apply_to_stance(mk("defense"), gap, cfg)["key"] == "defense"
    assert P.apply_to_stance(mk("buy"), only_f, cfg)["key"] == "neutral" and P.apply_to_stance(mk("neutral"), only_f, cfg)["key"] == "neutral"
    strong = P.evaluate({k: ok(2.0) for k in keys}, cfg, now)
    assert P.apply_to_stance(mk("defense"), strong, cfg)["key"] == "defense" and P.apply_to_stance(mk("neutral"), strong, cfg)["key"] == "neutral"
    assert "先物による判定は行っていません" in P.apply_to_stance(mk("buy"), miss, cfg)["pm_warn"]
    # 取得関数：一部が失敗しても他は使う
    def dl(tk, period, interval):
        if tk == "TPY=F":
            raise RuntimeError("boom")
        if interval == "15m":
            return s
        return d
    got = P.fetch(cfg, pd.Timestamp("2026-11-12"), now, dl)
    by = {i["key"]: i for i in got["items"]}
    assert by["topixf"]["status"] == "missing" and by["n225f"]["status"] == "weak" and got["gap_down"]
    print("ok  寄り付き前の指標（変化率・古い/欠損の扱い・先物重視・弱める方向のみ・一部失敗）")


def test_lead():
    from datetime import datetime
    from src import lead as L
    from src.config import JST
    now = datetime(2026, 11, 13, 7, 10, tzinfo=JST)
    mk = lambda code, sector: {"row": {"code": code, "sector": sector}}  # noqa: E731
    # グループの選び方：コード指定が業種より優先（半導体は電気機器の中でも別扱い）
    assert L.match_group("8035", "電気機器", cfg)["key"] == "semiconductor"
    assert L.match_group("6501", "電気機器", cfg)["key"] == "electronics"
    assert L.match_group("7203", "輸送用機器", cfg)["key"] == "auto" and L.match_group("8306", "銀行業", cfg)["key"] == "bank"
    assert L.match_group("9999", "水産・農林業", cfg) is None
    # 設定の点検：実在しない業種名・重複・必須項目の抜けを見つける
    sectors33 = set("水産・農林業 鉱業 建設業 食料品 繊維製品 パルプ・紙 化学 医薬品 石油・石炭製品 ゴム製品 ガラス・土石製品 鉄鋼 非鉄金属 金属製品 機械 電気機器 輸送用機器 精密機器 その他製品 電気・ガス業 陸運業 海運業 空運業 倉庫・運輸関連業 情報・通信業 卸売業 小売業 銀行業 証券、商品先物取引業 保険業 その他金融業 不動産業 サービス業".split())
    assert L.check_config(cfg, sectors33) == [], L.check_config(cfg, sectors33)
    bad = {**cfg, "lead": {**cfg["lead"], "groups": cfg["lead"]["groups"] + [{"key": "auto", "label": "x", "sectors": ["存在しない業"], "refs": [{"ticker": "A"}]}]}}
    assert len(L.check_config(bad, sectors33)) >= 3
    # スコア：上昇は追い風、下落は逆風、invert は反対、金利は差（bp）、欠損は無視して小さく
    ok = lambda **k: {"status": "ok", **k}  # noqa: E731
    semi = L.match_group("8035", "電気機器", cfg)
    up = {(r["ticker"], r.get("mode", "daily")): ok(pct=10.0) for r in semi["refs"]}
    ev = L.evaluate(semi, up, cfg)
    assert ev["score"] == cfg["lead"]["max_points"] and all(i["state"] == "good" for i in ev["items"]) and "上昇" in ev["items"][0]["text"]
    down = {k: ok(pct=-10.0) for k in up}
    assert L.evaluate(semi, down, cfg)["score"] == -cfg["lead"]["max_points"]
    dv = L.evaluate(semi, down, cfg)["items"][0]
    assert dv["mark"] == "⚠" and "下落" in dv["text"]
    part = {k: v for k, v in list(up.items())[:1]}                               # 重み2/5だけ取得できた場合
    ev2 = L.evaluate(semi, part, cfg)
    assert 0 < ev2["score"] < cfg["lead"]["max_points"] and ev2["available"] == 1 and ev2["items"][1]["text"] == "取得できず"
    assert L.evaluate(semi, {}, cfg)["score"] == 0.0 and L.evaluate(semi, {}, cfg)["verdict"] == "none"
    assert ev["verdict"] == "good" and "追い風" in ev["headline"] and L.evaluate(semi, down, cfg)["verdict"] == "bad" and "逆風" in L.evaluate(semi, down, cfg)["headline"]
    util = L.match_group("9501", "電気・ガス業", cfg)                              # 原油上昇は電力の逆風
    assert L.evaluate(util, {("CL=F", "daily"): ok(pct=6.0)}, cfg)["score"] < 0
    bank = L.match_group("8306", "銀行業", cfg)
    e = L.evaluate(bank, {("^TNX", "daily"): ok(pct=1.0, diff=0.10), ("XLF", "daily"): ok(pct=0.0)}, cfg)
    assert e["score"] > 0 and "bp" in e["items"][0]["text"]
    # 取得：一部が失敗しても他は使う
    refs = L.needed_refs([mk("8035", "電気機器"), mk("8306", "銀行業"), mk("9999", "水産・農林業")], cfg)
    assert ("^SOX", "daily") in refs and ("^TNX", "daily") in refs and len(refs) == len(semi["refs"]) + len(bank["refs"])
    def dl(tk, period, interval):
        if tk == "NVDA":
            raise RuntimeError("boom")
        return L.demo_download(tk, period, interval)
    rd = L.fetch_readings(refs, cfg, pd.Timestamp(now.date()) - pd.Timedelta(days=1), datetime.now(JST), dl)
    assert rd[("NVDA", "daily")]["status"] == "missing" and rd[("^SOX", "daily")]["status"] == "ok"
    # 順位への影響は補助的：本質的な評価の順から、動くのは max_rank_shift まで。総合点は順位と矛盾しない
    import random
    rnd = random.Random(3)
    items = [{"row": {"code": str(i)}, "main_base": 80 - i * 0.3, "pb_total": 80 - i * 0.3, "legacy_expected": 0.0,
              "lead_pts": rnd.uniform(-2.5, 2.5)} for i in range(40)]
    base_pos = {c["row"]["code"]: i for i, c in enumerate(items)}
    s = cfg["lead"]["max_rank_shift"]
    order = L.finalize(items, cfg)
    assert sorted(c["row"]["code"] for c in order) == sorted(base_pos)
    assert all(abs(i - base_pos[c["row"]["code"]]) <= s for i, c in enumerate(order)), "順位が動きすぎ"
    mains = [c["main"] for c in order]
    assert mains == sorted(mains, reverse=True), "総合点が順位と矛盾"
    assert all(abs(c["lead_pts"]) <= 2.5 + 1e-9 for c in order)
    zero = [dict(c, lead_pts=0.0) for c in items]
    assert [c["row"]["code"] for c in L.finalize(zero, cfg)] == [c["row"]["code"] for c in items]      # 影響0なら並びは変わらない
    txt = json.dumps(cfg["lead"], ensure_ascii=False)
    assert not any(w in txt for w in FORBIDDEN)
    print("ok  業種リード指標（グループ選定・スコア・欠損・順位の上限・設定の点検）")


def _tracking_fixture(tmp, n_days=40):
    """架空の履歴・株価：A銘柄は毎日+1%（始値は前日終値）、B銘柄は毎日−1%。"""
    import numpy as np
    idx = pd.bdate_range("2026-09-01", periods=n_days)
    tks = ["1111.T", "2222.T", "3333.T"]
    close = pd.DataFrame({"1111.T": 100 * 1.01 ** np.arange(n_days), "2222.T": 100 * 0.99 ** np.arange(n_days), "3333.T": 100.0}, index=idx)
    opn = close.shift(1).fillna(close.iloc[0])
    panel = {"close": close, "open": opn, "high": close, "low": close, "volume": close * 0 + 1e6}
    topix = pd.DataFrame({"close": 100 * 1.002 ** np.arange(n_days), "open": 100 * 1.002 ** np.arange(n_days) / 1.002}, index=idx)
    hist = tmp / "history"; hist.mkdir(parents=True)
    # 1111：2日目から連続で載る。2222：5日目に1回だけ載り、30日後にもう一度（再掲載）。3333：載らない
    for i, d in enumerate(idx[:35]):
        picks = []
        if i >= 1:
            picks.append({"rank": 1, "code": "1111", "name": "上がる株", "main": 82.0, "pb_total": 80, "close": 1})
        if i == 4 or i == 33:
            picks.append({"rank": 5, "code": "2222", "name": "下がる株", "main": 71.0, "pb_total": 70, "close": 1})
        (hist / f"{d:%Y-%m-%d}.json").write_text(json.dumps({"date": f"{d:%Y-%m-%d}", "demo": False, "regime": "good", "stance": ["buy", "neutral"][i % 2],
                                                            "picks": picks, "ranking": [{"rank": p["rank"], "code": p["code"], "name": p["name"], "main": p["main"]} for p in picks]}), encoding="utf-8")
    return panel, topix, hist, idx


def test_rankband():
    """㉕ 順位別・スコア帯別：人工データで、点数が高いほど上がる世界／無関係な世界を見分けられるか。"""
    from src import rankband as RB
    cfg = load_config()
    rng = np.random.default_rng(3)
    hs = [1, 3, 5, 10, 20]

    def world(edge: float, n_days: int = 300) -> list[dict]:
        recs = []
        for d in range(n_days):
            n = 200
            sc = rng.uniform(40, 100, n)
            order = np.argsort(-sc)
            base = rng.normal(0.001, 0.03, n)
            ret = {h: (base + edge * (sc - 70) / 30 * (h / 5)).astype(float) for h in hs}
            recs.append({"date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=7 * d), "s": sc, "rk": order[:10],
                         "ret": ret, "bench": {h: 0.001 for h in hs}})
        return recs

    good = RB.analyze_replay(world(0.01), cfg, 5)
    assert good["ok"]
    st = {g["name"]: g["status"] for g in good["levels"]}
    assert st["90点以上"] == "推奨" and st["85点以上"] == "推奨", st
    assert good["advice"]["rec_level"] is not None and "推奨" in good["advice"]["score"]
    assert all(g["by_h"][5]["avg"] > good["all"]["by_h"][5]["avg"] for g in good["ranks"]), "上位は全体平均より高い"
    assert good["levels"][0]["by_h"][5]["avg"] > good["levels"][4]["by_h"][5]["avg"], "点数が高いほど平均が高い"
    assert good["advice"]["rank_thru"] >= 3 and good["advice"]["best_portfolio"] in {"TOP1", "TOP3", "TOP5", "TOP10"}
    for g in good["ranks"] + good["portfolios"] + good["levels"]:
        for k in ("pf", "sharpe", "max_dd"):
            assert k in g
        assert set(g["by_h"]) == set(hs) and 0 <= g["by_h"][5]["win"] <= 1
    assert good["portfolios"][0]["max_dd"] <= 0
    flat = RB.analyze_replay(world(0.0), cfg, 5)
    assert flat["advice"]["rec_level"] is None, "点数に意味が無い世界で『推奨』を出さない"
    assert all(g["status"] != "推奨" for g in flat["ranks"] + flat["levels"])
    tiny = RB.analyze_replay(world(0.01, 10), cfg, 5)
    assert not tiny["ok"], "日数が足りないときは分析しない"
    # 件数が少ない帯は判定不能（90点以上かつ少数）
    few = world(0.01, 60)
    assert RB.analyze_replay(few, cfg, 5)["levels"][0]["status"] in ("判定不能", "推奨", "中立", "見送り")
    # JSONを経由してもページ用に読み直せる（キーが文字列になる）
    v = RB._norm(json.loads(json.dumps({"rankband": good}))["rankband"])
    assert isinstance(next(iter(v["levels"][0]["by_h"])), int)
    svg = RB.bars_svg(["a", "b"], [("5日後", [1.0, -2.0]), ("20日後", [2.0, None])])
    assert "<svg" in svg and "—" in svg and RB.bars_svg(["a"], [("x", [None])]) == ""
    # 実運用の記録（追跡台帳）：順位・総合点で切り分け
    res = [{"rank": r, "main": m, "date": f"2026-10-{d:02d}", "cp": {"5": (x, 0.0), "20": (x * 2, 0.0)}}
           for d, (r, m, x) in enumerate([(1, 88, 0.05), (2, 82, -0.02), (3, 76, 0.01), (5, 72, -0.03), (8, 71, 0.02)], 1)]
    lv = RB.analyze_live(res, cfg)
    assert lv["ranks"][0]["by_h"][5]["n"] == 1 and lv["ranks"][0]["by_h"][5]["win"] == 1.0
    assert lv["levels"][3]["name"] == "75点以上" and lv["levels"][3]["by_h"][5]["n"] == 3     # 88・82・76
    assert lv["levels"][0]["by_h"][5]["n"] == 0 and lv["portfolios"][0]["name"] == "TOP1"
    print("ok  順位別・スコア帯別の分析（点数に意味がある/無い世界の見分け・判定不能・実運用の記録）")


def test_quality_first():
    """良い企業の押し目を優先する仕組み：減益は除外・増益は加点・高値突破の実績がない銘柄は大きく減点・値動きが穏やかなら加点。"""
    from src import fundamentals as FU, pullback
    cfg = load_config()
    fc = cfg["fundamentals"]
    ok = {"op_ttm": 100.0, "op_yoy": 0.12, "op_annual": [100, 80]}
    assert FU.exclusion_reason(ok, fc) is None
    assert "減益" in FU.exclusion_reason(dict(ok, op_yoy=-0.08), fc), "減益の企業は除外"
    assert FU.exclusion_reason(dict(ok, op_yoy=None), fc) is None, "前年比が出せない（黒字転換など）は残す"
    assert "引き下げ" in FU.exclusion_reason(dict(ok, eps_rev={"cur": 80.0, "d90": 100.0}), fc)
    assert FU.exclusion_reason(dict(ok, eps_rev={"cur": 95.0, "d90": 100.0}), fc) is None
    # 営業増益が大きいほど高得点
    qp = lambda yoy: pullback.quality_points(dict(ok, op_yoy=yoy), None, cfg)[0]  # noqa: E731
    assert qp(0.0) < qp(0.10) < qp(0.25) < qp(0.50) and qp(0.50) - qp(0.0) >= 7.9
    assert pullback.quality_points(ok, None, cfg)[0] <= cfg["pullback"]["weights"]["quality"]
    # 高値突破・保有のしやすさ
    idx = ["A", "B", "C"]
    snap = pd.DataFrame({"close": 100.0, "ma5": 99.0, "ma50": 100.0, "ma200": 90.0, "ma200_slope": 0.03, "gap50": 0.0,
                         "dist52w": -0.10, "vol_cont": 0.7, "rs60": 0.1, "ret3d": 0.02, "ret5d": 0.0, "ret20d": 0.0, "ret1d": 0.0,
                         "vol_ratio": 1.0, "hi252_recent": [1.0, 0.0, 1.0], "vol60": [0.012, 0.012, 0.04]}, index=idx)
    o = pullback.score_technical(snap, "good", True, cfg)
    assert o.loc["A", "pb_penalty"] == 0 and o.loc["B", "pb_penalty"] == cfg["pullback"]["penalties"]["no_breakout"]["pts"]
    assert o.loc["A", "pb_hold"] == cfg["pullback"]["weights"]["hold"] and o.loc["C", "pb_hold"] == 0
    assert o.loc["A", "pb_tech_scaled"] > o.loc["C", "pb_tech_scaled"] > o.loc["B", "pb_tech_scaled"]
    assert sum(pullback.max_points(cfg["pullback"]).values()) == 100
    print("ok  良い企業優先（減益除外・増益加点・高値突破・保有のしやすさ・配点合計100）")


def test_entry10():
    """㉖ 10時判定・エントリー時刻別検証・業績キャッシュの規則。"""
    from datetime import date, datetime, time as dtime
    from src import entry10 as E, entrytime as ET, fundamentals as FU
    cfg = load_config()
    J = lambda g, r, v, m="neutral": E.judge(g, r, v, m, cfg)["key"]      # noqa: E731
    assert J(0.4, 0.3, 1.2) == "go"
    assert J(3.5, 0.0, 1.0) == "skip" and J(0.5, 3.2, 1.0) == "skip" and J(1.0, 1.5, 3.0) == "skip"
    assert J(2.0, 2.8, 1.0) == "skip", "前日終値から+4.8%上昇済み"
    assert J(0.2, -0.9, 0.5) == "wait" and J(0.2, 0.2, 0.3) == "wait" and J(-2.5, 0.0, 1.0) == "wait"
    assert J(0.3, 0.2, 1.0, "weak") == "wait", "市場が弱いと🟢は🟡に下がる"
    assert J(0.3, 0.2, 1.0, "weak") != "skip" and J(0.3, -3.0, 1.0) == "skip"
    assert J(None, None, 1.0) == "na" and J(0.3, 0.2, None) == "wait"
    # 5分足 → 指標
    JST = E.JST

    def bars(day, o, closes, vol):
        idx = pd.date_range(f"{day} 09:00", periods=len(closes), freq="5min", tz=JST)
        return pd.DataFrame({"Open": [o] + closes[:-1], "High": closes, "Low": closes, "Close": closes, "Volume": [vol] * len(closes)}, index=idx)
    hist = pd.concat([bars(f"2026-11-{d:02d}", 100.0, [100.0] * 24, 1000) for d in (9, 10, 11, 12, 13, 16)])
    today = bars("2026-11-20", 101.0, [101.0 + 0.05 * k for k in range(24)], 1500)
    df = pd.concat([hist, today])
    it = E.evaluate_stock("1234", df, date(2026, 11, 20), 100.0, "neutral", cfg)
    assert abs(it["gap"] - 1.0) < 1e-6 and it["key"] in ("go", "wait"), it
    assert abs(it["vr"] - 1.5) < 1e-6, it           # 12本×1500 ÷ 12本×1000
    assert it["asof"] == "10:00" and it["complete"]
    assert E.session_stats(df, date(2026, 11, 20))["p_cut"] == today["Close"].iloc[11], "10:00の値は9:55の足の終値"
    assert E.evaluate_stock("1234", None, date(2026, 11, 20), 100.0, "neutral", cfg)["key"] == "na"
    # 市場環境
    weak = E.market_state([{"key": "a", "label": "x", "pct": -1.5, "futures": True}, {"key": "b", "label": "y", "pct": -1.0, "futures": False}], cfg)
    assert weak["key"] == "weak" and E.market_state([], cfg)["key"] == "unknown"
    # build / entry.json
    latest = {"data_date": "2026-11-19", "picks": [{"rank": 1, "code": "1234", "close": 100.0}, {"rank": 2, "code": "5678", "close": 50.0}]}

    def dl(tks, period):
        out = {}
        for t in tks:
            if t == "1234.T":
                out[t] = df
            elif t in ("1306.T", "^N225", "2516.T"):
                out[t] = pd.concat([bars("2026-11-19", 100.0, [100.0] * 24, 1), bars("2026-11-20", 100.0, [100.2] * 24, 1)])
        return out
    res = E.build(cfg, latest, datetime(2026, 11, 20, 10, 25, tzinfo=JST), dl, lambda tk, p, i: pd.Series(dtype=float))
    assert res["status"] == "ok" and res["items"]["5678"]["key"] == "na" and res["items"]["1234"]["rank"] == 1
    assert res["trade_date"] == "2026-11-20" and res["ranking_data_date"] == "2026-11-19" and res["market"]["key"] in ("neutral", "strong")
    assert E.build(cfg, latest, datetime(2026, 11, 21, 10, 25, tzinfo=JST), lambda t, p: {}, lambda *a: pd.Series(dtype=float))["status"] == "no_data"
    # ランキングの並びは判定で変わらない（items は順位をそのまま持つだけ）
    assert [res["items"][c]["rank"] for c in ("1234", "5678")] == [1, 2]
    # エントリー時刻別
    idx = pd.bdate_range("2026-09-01", periods=60)
    rng = np.random.default_rng(5)
    close = pd.DataFrame({"1234.T": 100 * np.cumprod(1 + rng.normal(0.002, 0.01, 60))}, index=idx)
    opn = close * 0.999
    rec = pd.DataFrame([{"pick_date": idx[k].strftime("%Y-%m-%d"), "code": "1234.T", "prev_close": float(close.iloc[k, 0]),
                         "open": float(opn.iloc[k + 1, 0]), "p1000": float(opn.iloc[k + 1, 0]) * 1.002,
                         "pam": float(opn.iloc[k + 1, 0]) * 1.004} for k in range(5, 35)])
    r = ET.compute(rec, close, [5, 20], 0.002, 30, 2.0)
    assert r["ok"] and r["rows"]["5"]["n"] == 30 and not r["rows"]["5"]["reference"]
    assert r["rows"]["5"]["types"]["pam"]["mean"] < r["rows"]["5"]["types"]["open"]["mean"], "後で買うほど（高く買うので）平均は下がる"
    assert r["rows"]["5"]["best"] in ("prev", "open") and "最も高い" in r["conclusion"]
    assert not ET.compute(pd.DataFrame(columns=ET.COLS), close, [5], 0.002, 30, 2.0)["ok"]
    # 記録は別ファイルではなく summary(JSON) の中に引き継ぐ（Actionsの保存で取りこぼさない）
    import json as _j, tempfile as _t
    with _t.TemporaryDirectory() as _d:
        ET.ROOT = Path(_d)                                   # 旧CSVの読み込み先を空にして、ファイルを書かないことも確認
        r2 = ET.analyze([], {"close": close, "open": opn}, cfg, lambda t, p: {}, rec.to_dict("records"))
        assert len(r2["store"]) == 30 and r2["ok"] and not list(Path(_d).rglob("*")), "新規ファイルを作らない"
        _j.dumps(r2)                                         # JSONにできる（NaNなし）
    merged = ET.merge_store(rec.iloc[:10], rec.iloc[5:15])
    assert len(merged) == 15
    # 業績キャッシュ：30日まで使い回し、決算発表を過ぎたら取り直す
    hit = lambda f, ne: {"fetched": f, "data": {"next_earnings": ne}}      # noqa: E731
    t = date(2026, 11, 20)
    assert FU.cache_valid(hit("2026-11-01", None), t, 30) and not FU.cache_valid(hit("2026-10-10", None), t, 30)
    assert not FU.cache_valid(hit("2026-11-01", "2026-11-12"), t, 30), "決算発表前に取ったデータは決算後に取り直す"
    assert FU.cache_valid(hit("2026-11-15", "2026-11-12"), t, 30), "決算後に取ったものは使い回す"
    assert FU.cache_valid(hit("2026-11-01", "2026-12-01"), t, 30)
    # 待機：期限（10:27）まで待ち、足がそろえば即終了、期限後に起動したら待たない
    clock = [datetime(2026, 11, 20, 10, 5, tzinfo=JST)]
    ready_at = datetime(2026, 11, 20, 10, 20, tzinfo=JST)

    def dl_wait(tks, period):
        return {tks[0]: df} if clock[0] >= ready_at else {}
    def sl(sec):
        clock[0] += pd.Timedelta(seconds=sec)
    assert E.wait_for_bars(cfg, dl_wait, lambda: clock[0], sl) and clock[0] < datetime(2026, 11, 20, 10, 27, tzinfo=JST)
    clock[0] = datetime(2026, 11, 20, 10, 5, tzinfo=JST)
    assert not E.wait_for_bars(cfg, lambda t, p: {}, lambda: clock[0], sl) and clock[0].time() >= dtime(10, 27)
    clock[0] = datetime(2026, 11, 20, 10, 40, tzinfo=JST)
    assert not E.wait_for_bars(cfg, lambda t, p: {}, lambda: clock[0], sl) and clock[0] == datetime(2026, 11, 20, 10, 40, tzinfo=JST)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        od = Path(td)
        assert not E.already_done(od, "2026-11-20")
        (od / "entry.json").write_text(json.dumps({"trade_date": "2026-11-20", "status": "ok", "items": {"1": {"key": "go", "complete": True}}}), encoding="utf-8")
        assert E.already_done(od, "2026-11-20") and not E.already_done(od, "2026-11-21")
        (od / "entry.json").write_text(json.dumps({"trade_date": "2026-11-20", "status": "ok", "items": {"1": {"key": "go", "complete": False}}}), encoding="utf-8")
        assert not E.already_done(od, "2026-11-20"), "足が不完全なら再実行する"
    print("ok  ㉖ 10時判定（判定規則・5分足の指標・市場環境・entry.json・時刻別検証・業績キャッシュ）")


def test_negative():
    """㉗ 信用リスク監視：分類規則・誤検出の除外・点数・社名の結びつけ・取得物の解析・検出日・ページ。"""
    import tempfile
    from datetime import date, datetime, timedelta
    from src.config import JST
    from src import negative as N
    cfg = load_config()
    today = date(2026, 11, 24)
    ago = lambda d: (today - timedelta(days=d)).isoformat()      # noqa: E731
    T = lambda title, kind="official", d=1: N.Item("TDnet" if kind == "official" else "Googleニュース", ago(d), "9001", "", title, "", kind)   # noqa: E731
    keys = lambda title, kind="official": {e.key for e in N.classify(T(title, kind))}                                  # noqa: E731
    assert "downward_revision" in keys("通期業績予想の下方修正に関するお知らせ")
    assert "downward_revision" not in keys("通期業績予想の上方修正に関するお知らせ")
    assert "dividend_cut" in keys("期末配当予想の修正（無配）に関するお知らせ") and "dividend_cut" not in keys("配当予想の修正（増配）に関するお知らせ")
    assert "loss_turn" in keys("最終赤字に転落へ", "news") and "going_concern" in keys("継続企業の前提に関する注記の記載について")
    assert "continuing_loss" in keys("3期連続の営業赤字")
    assert "kanri" in keys("監理銘柄（確認中）への指定に関するお知らせ") and "kanri" not in keys("監理銘柄の指定解除について")
    assert "seiri" in keys("整理銘柄への指定について")
    assert "delisting" in keys("上場廃止基準に係る猶予期間入りに関するお知らせ") and "delisting" in keys("民事再生手続開始の申立てに関するお知らせ")
    assert "delisting" not in keys("MBOの実施及び応募の推奨に関するお知らせ（上場廃止となる見込み）"), "TOB・MBOによる上場廃止は対象外"
    assert "accounting" in keys("第三者委員会の調査報告書受領に関するお知らせ") and "accounting" in keys("決算発表の延期に関するお知らせ")
    assert "window_dressing" in keys("架空売上の計上が判明", "news") and "quality_fraud" in keys("品質データの改ざんが発覚", "news")
    assert "inspection_fraud" in keys("完成検査における不適切事案について") and "admin_action" in keys("金融庁から業務改善命令を受けました")
    assert "data_leak" in keys("不正アクセスによる情報漏えいのお知らせ") and "recall" in keys("製品の自主回収（リコール）について")
    assert "outage" in keys("システム障害の発生について") and "litigation" in keys("当社に対する訴訟の提起に関するお知らせ")
    assert "litigation" not in keys("訴訟の和解に関するお知らせ") and not keys("第2四半期決算短信〔日本基準〕（連結）") and not keys("自己株式の取得に係る事項の決定に関するお知らせ")
    assert "deterioration" in keys("希望退職者の募集に関するお知らせ") and "deterioration" not in keys("減益決算の通期見通し", "official"), "TDnetの『減益』は深刻語がなければ対象外"
    assert "deterioration" in keys("減益決算", "news")
    assert not keys("一部報道について（事実ではありません）"), "会社が否定した報道は対象外"
    assert N.is_ambiguous_revision(T("業績予想の修正に関するお知らせ")) and not N.is_ambiguous_revision(T("業績予想の下方修正に関するお知らせ"))
    assert any(e.key == "downward_revision" for e in N.classify(T("業績予想の修正に関するお知らせ"), "down"))
    # 点数：新しいほど高い・報道は公式より低い・重なると上がる（100超えない）
    pts = lambda title, kind, d: N.event_points(N.classify(T(title, kind, d))[0], today, cfg)                              # noqa: E731
    assert pts("下方修正", "official", 1) > pts("下方修正", "official", 14) > 0 and pts("下方修正", "official", 1) > pts("下方修正", "news", 1)
    one = N.company_score(N.classify(T("下方修正", "official")), today, cfg)["score"]
    many = N.company_score(N.classify(T("下方修正", "official")) + N.classify(T("粉飾の疑い", "news")) + N.classify(T("最終赤字に転落", "news")), today, cfg)["score"]
    assert 0 < one < many <= 100
    two_src = N.company_score(N.classify(T("下方修正", "official")) + N.classify(T("下方修正を発表", "news")), today, cfg)["score"]
    assert two_src > one, "同じ種類を別の情報源も報じていれば上がる"
    same = N.company_score(N.classify(T("下方修正", "official")) * 3, today, cfg)["score"]
    assert same == one, "同じ情報源の同じ種類の重複は数えない"
    sev = N.company_score(N.classify(T("組織的な不正会計が発覚", "official")), today, cfg)["score"]
    assert sev >= N.company_score(N.classify(T("不正会計が発覚", "official")), today, cfg)["score"]
    assert N.band(85, cfg)[0] == "critical" and N.band(65, cfg)[0] == "warn" and N.band(45, cfg)[0] == "caution" and N.band(10, cfg)[0] == "minor"
    # 社名の結びつけ
    import pandas as pd
    uni = pd.DataFrame({"code": ["9001", "9002", "9003"], "name": ["架空製作所", "サンプルホールディングス", "日本"], "market": ["プライム"] * 3, "sector": ["機械"] * 3})
    info, pairs = N.build_name_index(uni)
    assert N.match_code("架空製作所が下方修正", pairs, info) == "9001" and N.match_code("サンプルが不正会計", pairs, info) == "9002"
    assert N.match_code("日本で訴訟が相次ぐ", pairs, info) is None, "一般語・短い名前は使わない"
    assert N.match_code("某社（9001）が下方修正", pairs, info) == "9001" and N.match_code("ABC(1234)が訴訟", pairs, info) is None
    # 取得物の解析
    td = N.parse_tdnet_all({"items": [{"Tdnet": {"pubdate": "2026-11-20 15:00:00", "company_code": "90010", "company_name": "架空製作所", "title": "下方修正のお知らせ", "document_url": "http://x/a.pdf"}},
                                       {"Tdnet": {"pubdate": "bad", "title": "x"}}]})
    assert len(td) == 1 and td[0].code == "9001" and td[0].kind == "official" and td[0].date == "2026-11-20"
    rss = """<?xml version="1.0"?><rss><channel><item><title>架空製作所が赤字転落 - 経済紙</title><link>http://n/1</link><pubDate>Fri, 20 Nov 2026 03:00:00 GMT</pubDate></item>
      <item><title>古い記事</title><link>http://n/2</link><pubDate>Mon, 01 Jun 2026 03:00:00 GMT</pubDate></item></channel></rss>"""
    nws = N.parse_rss(rss, "Googleニュース", today, 15)
    assert len(nws) == 1 and nws[0].date == "2026-11-20" and nws[0].url == "http://n/1"
    fin = N.financial_items({"9001": {"fetched": ago(3), "data": {"status": "ok", "op_ttm": -5.0, "op_annual": [-3.0, -2.0, 4.0]}},
                             "9002": {"fetched": ago(3), "data": {"status": "ok", "op_ttm": -1.0, "op_annual": [2.0, 3.0]}},
                             "9003": {"fetched": ago(40), "data": {"status": "ok", "op_ttm": -1.0, "op_annual": [-2.0, -3.0]}}}, today, cfg)
    assert {(f.code, "継続" in f.title) for f in fin} == {("9001", True), ("9002", False)}, "古い取得分は対象外"
    # 全体：会社ごとにまとめる・対象外の銘柄（ETF等）を除く・失敗した情報源があっても動く
    items = [N.Item("TDnet", ago(2), "9001", "架空製作所", "通期業績予想の下方修正に関するお知らせ", "", "official"),
             N.Item("Googleニュース", ago(1), None, "", "架空製作所、不正会計の疑い 検察が家宅捜索 - 経済紙", "", "news"),
             N.Item("TDnet", ago(1), "1306", "TOPIX ETF", "訴訟の提起", "", "official"),
             N.Item("TDnet", ago(30), "9002", "サンプルHD", "不正会計の疑い", "", "official")]
    res = N.analyze(items, info, pairs, today, cfg)
    assert set(res) == {"9001"} and res["9001"]["score"] >= 60
    with tempfile.TemporaryDirectory() as td_:
        sp = Path(td_) / "seen.json"
        ev = res["9001"]["events"][0]
        k = N.event_key("9001", ev)
        s1 = N.update_seen(sp, [(k, ev.item.date)], today)
        s2 = N.update_seen(sp, [(k, ev.item.date)], today + timedelta(days=3))
        assert s1[k] == s2[k] == today.isoformat(), "検出日は最初に見つけた日のまま"
        rows = N.build_rows(res, info, s2, today + timedelta(days=3), cfg)
        assert rows[0]["detected"] == today.isoformat() and rows[0]["labels"] and "下方修正" in rows[0]["summary"]
        ctx = N.page_context(rows, [{"name": "TDnet", "ok": True, "n": 1, "note": ""}], datetime(2026, 11, 24, 12, 0, tzinfo=JST), cfg)
        page = N.write_page(ctx, Path(td_) / "docs" / "negative").read_text(encoding="utf-8")
        assert "空売りの推奨ではありません" in page and "架空製作所" in page and "X（旧Twitter）" in page
    assert "売り" not in "".join(N.summarize("架空製作所", res["9001"]["parts"])), "要約に売買の推奨表現を入れない"
    # 取引できない銘柄の除外（TOB決定・上場廃止決定・法的整理・取引停止）
    U = lambda *ts: [r["reason"] for r in N.untradable_reasons("9001", [(t, ago(5), "TDnet") for t in ts])]      # noqa: E731
    assert U("株式会社○○による当社株式に対する公開買付けに関する意見表明のお知らせ") == ["TOB・MBO・買収が決定"]
    assert U("完全子会社化に関する株式交換契約締結のお知らせ") and U("MBOの実施及び応募の推奨に関するお知らせ")
    assert not U("自己株式の公開買付けに関するお知らせ"), "自社株買いのTOBは対象外にしない"
    assert U("上場廃止の決定に関するお知らせ") == ["上場廃止が決定"] and U("整理銘柄への指定について") == ["上場廃止が決定"]
    assert U("破産手続開始の決定に関するお知らせ") and not U("上場廃止基準に該当するおそれのある事実の発生について") and not U("通期業績予想の下方修正に関するお知らせ")
    assert not N.untradable_reasons("9001", [("上場廃止が決定", ago(1), "Googleニュース")]), "報道だけでは上場廃止を断定しない"
    assert N.untradable_reasons("9001", [("TOBを発表", ago(1), "Googleニュース")]), "TOBは報道でも対象外"
    idx_ = pd.bdate_range(end=pd.Timestamp(today), periods=8)
    ok_df = pd.DataFrame({"Close": 100.0, "Volume": 5000000.0}, index=idx_)
    assert N.price_problem(ok_df, today) is None and N.price_problem(None, today) is None
    assert "止まって" in N.price_problem(ok_df.iloc[:2], today) and "出来高" in N.price_problem(ok_df.assign(Volume=0.0), today)
    rows_ = [{"code": c, "name": c, "score": 70} for c in ("9001", "9002", "9003", "9004")]
    tdf = lambda code: [type("D", (), {"title": "公開買付けに関する意見表明", "date": ago(30)})()] if code == "9002" else []      # noqa: E731
    pf = lambda codes: {c: (ok_df if c != "9003" else ok_df.iloc[:2]) for c in codes}                                      # noqa: E731
    keep_, out_, st_ = N.filter_tradable(rows_, {}, today, cfg, tdf, pf, {"9004"})
    assert [r["code"] for r in keep_] == ["9001"] and {x["code"] for x in out_} == {"9002", "9003", "9004"}
    k2, o2, s2 = N.filter_tradable(rows_, {}, today, cfg, tdf, lambda codes: {}, set())
    assert s2["price_skipped"] and [r["code"] for r in k2] == ["9001", "9003", "9004"], "株価が取れないときは価格で除外しない"
    def tdfail(code):
        raise RuntimeError("x")
    k3, o3, s3 = N.filter_tradable(rows_, {}, today, cfg, tdfail, None, set())
    assert len(k3) == 4 and s3["td_fail"] == 4, "TDnet履歴を取れなくても動く（確認不完全として表示）"
    # ㉗追補：信用売りできる状態・動いていない銘柄・規模・新着順
    from src import margin as MG
    import io as _io
    lists = ({"1111", "2222", "3333"}, {"1111", "2222"})
    assert MG.short_status("1111", lists, {}, {})["key"] == "ok" and MG.short_status("3333", lists, {}, {})["key"] == "no_lending"
    assert MG.short_status("9999", lists, {}, {})["key"] == "no_margin" and MG.short_status("1111", lists, {}, {"1111": "stop"})["key"] == "stop"
    w = MG.short_status("1111", lists, {"1111": "新規売りの委託保証金率50%以上"}, {"1111": "caution"})
    assert w["key"] == "ok_warn" and w["can_short"] is True and len(w["notes"]) == 2
    assert MG.short_status("1111", None, {}, {})["key"] == "unknown" and MG.short_status("1111", None, {}, {})["can_short"] is None
    rg = MG.parse_regulated("<table><tr><th>銘柄名</th><th>コード</th><th>実施日</th><th>規制の内容</th></tr><tr><td>A</td><td>1111</td><td>2026/11/20</td><td>新規売りの委託保証金率引上げ</td></tr></table>"
                            "<table><tr><th>銘柄名</th><th>コード</th><th>解除日</th><th>規制の内容</th></tr><tr><td>B</td><td>2222</td><td>2026/11/19</td><td>x</td></tr></table>")
    assert set(rg) == {"1111"}, "解除済みは含めない"
    assert MG.parse_jsf("<table><tr><td>1111</td><td>貸借取引停止</td></tr><tr><td>2222</td><td>注意喚起</td></tr></table>") == {"1111": "stop", "2222": "caution"}
    df_x = pd.DataFrame([["区分", "銘柄名", "コード"], ["貸借", "A", "1111"], ["制度信用", "B", "3333"]])
    bio = _io.BytesIO()
    df_x.to_excel(bio, header=False, index=False)
    assert MG.parse_margin_xlsx(bio.getvalue()) == ({"1111", "3333"}, {"1111"}) and MG.parse_margin_xlsx(b"broken") is None
    assert MG.find_list_url('<a href="/a/20261102_list.xlsx">', "https://www.jpx.co.jp/x/index.html") == "https://www.jpx.co.jp/a/20261102_list.xlsx"
    rows_ = [{"code": c, "name": c, "score": 70} for c in ("1111", "3333", "9999")]
    kp, ex = N.apply_short(rows_, lists, {}, {})
    assert [r["code"] for r in kp] == ["1111"] and len(ex) == 2
    kp2, ex2 = N.apply_short(rows_, None, {}, {})
    assert len(kp2) == 3 and not ex2, "貸借一覧を取れないときは除外しない"
    # 動いていない銘柄
    idx_ = pd.bdate_range(end=pd.Timestamp(today), periods=25)
    mk = lambda c, hi, lo, vol: pd.DataFrame({"Close": c, "High": hi, "Low": lo, "Volume": vol}, index=idx_)       # noqa: E731
    active = mk(np.linspace(1000, 1100, 25), np.linspace(1010, 1110, 25), np.linspace(990, 1090, 25), 500000.0)
    pinned = mk(1000.0, 1002.0, 999.0, 500000.0)
    thin = mk(np.linspace(100, 110, 25), np.linspace(101, 111, 25), np.linspace(99, 109, 25), 1000.0)
    assert N.inactive_reason(N.price_stats(active, cfg), cfg) is None
    assert "動いていません" in N.inactive_reason(N.price_stats(pinned, cfg), cfg) and "売買代金" in N.inactive_reason(N.price_stats(thin, cfg), cfg)
    kp3, ex3, _ = N.filter_tradable([{"code": c, "name": c, "score": 70} for c in ("1111", "2222", "3333")], {}, today, cfg, None,
                                    lambda codes: {"1111": active, "2222": pinned, "3333": thin}, set())
    assert [r["code"] for r in kp3] == ["1111"] and kp3[0]["stats"]["ret5"] is not None
    # 規模
    z = cfg["negative"]["size"]
    assert N.size_reason({"revenue": 5e9, "market_cap": 1e11}, cfg) and N.size_reason({"revenue": 5e10, "market_cap": 1e9}, cfg)
    assert N.size_reason({"revenue": 5e10, "market_cap": 1e11}, cfg) is None and N.size_reason({"revenue": None, "market_cap": None}, cfg) is None and N.size_reason(None, cfg) is None
    with tempfile.TemporaryDirectory() as td2:
        cp = Path(td2) / "size.json"
        calls = []
        def getter(c):
            calls.append(c)
            return {"revenue": 5e10, "market_cap": 1e11}
        a1 = N.fetch_size(["1111"], cp, today, cfg, getter)
        a2 = N.fetch_size(["1111"], cp, today + timedelta(days=5), cfg, getter)
        assert a1 == a2 and calls == ["1111"], "30日はキャッシュを使う"
    # 新着順：出たばかりのイベントが上に来る
    def mkres(day_ago):
        it_ = N.Item("TDnet", ago(day_ago), "9001", "架空製作所", "不適切な会計処理に関するお知らせ", "", "official")
        return {"9001": {"score": 70, "parts": N.company_score(N.classify(it_), today, cfg)["parts"], "events": N.classify(it_)}}
    old_r = N.build_rows(mkres(9), info, {}, today, cfg)[0]
    new_r = N.build_rows(mkres(0), info, {}, today, cfg)[0]
    assert new_r["is_new"] and not old_r["is_new"] and new_r["rank_score"] > old_r["rank_score"] + 10
    both = {"9001": mkres(0)["9001"], "9002": {**mkres(9)["9001"]}}
    two = N.build_rows(both, info, {}, today, cfg)
    assert two[0]["code"] == "9001", "新しい不祥事が上位"
    print("ok  ㉗ 売買できない銘柄の除外・信用売り可否・動いていない銘柄・規模・新着順")
    print("ok  ㉗ 信用リスク監視（17分類・誤検出の除外・点数・社名結びつけ・RSS/TDnet解析・検出日・ページ）")


def test_fx():
    """ドル円の市場参加者分析：セッション別の損益・統計・スコア・判定・ページ。"""
    import copy
    import tempfile
    from datetime import datetime
    from src import fx
    from src.config import JST
    cfg = load_config()
    # ① 値動きが分かっている人工データ：東京+0.10円/h・欧州-0.20円/h・NY+0.05円/h（線形）
    def linear(days: int = 12) -> pd.DataFrame:
        idx = pd.date_range("2026-10-05 00:00", periods=days * 24, freq="h", tz=JST)
        idx = idx[[not (t.weekday() >= 5) for t in idx]]
        rows, px = [], 150.0
        for t in idx:
            hm = t.hour + t.minute / 60
            d = 0.10 if 9 <= hm < 15.5 else (-0.20 if 15.5 <= hm < 22 else 0.05) if True else 0
            if 15 <= hm < 16:                                   # 15:30をまたぐ足は半分ずつ
                d = 0.5 * 0.10 + 0.5 * -0.20
            rows.append((px, px + max(d, 0) + 0.01, px + min(d, 0) - 0.01, px + d)); px += d
        return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])
    b = linear()
    ss = fx.build_sessions(b, cfg)
    one = ss[(ss["date"] == pd.Timestamp("2026-10-07").date())].set_index("session")
    # 15:30をまたぐ足（15:00〜16:00）は、始値→終値の中間で補間する（この足は全体で-0.05円 → 15:30時点は-0.025円）
    assert abs(one.loc["tokyo", "pips"] - 57.5) < 0.5 and abs(one.loc["europe", "pips"] + 122.5) < 0.5, one["pips"].to_dict()
    assert abs(one.loc["ny", "pips"] - 40.0) < 0.5, "NYは22:00〜翌06:00（8時間）"
    assert one.loc["tokyo", "clv"] > 0.9 and one.loc["europe", "clv"] < 0.1, "上がり続ければ高値引け・下がり続ければ安値引け"
    assert ss["weekday"].max() <= 4 and set(ss[~ss["complete"]]["date"]) <= {ss["date"].max()}, "終わっていないのは最終日だけ"
    # 進行中のセッション（足が途中まで）は集計に入れない
    part = fx.build_sessions(b[b.index <= pd.Timestamp("2026-10-08 23:00", tz=JST)], cfg)
    assert (~part["complete"]).sum() >= 1 and part[~part["complete"]]["session"].iloc[-1] == "ny"
    assert fx.price_at(b, pd.Timestamp("2026-10-07 15:30", tz=JST)) is not None and fx.price_at(b, pd.Timestamp("2026-10-10 12:00", tz=JST)) is None
    # ② 連続・統計
    assert fx.streak(pd.Series([-1, 2, 3, 4])) == 3 and fx.streak(pd.Series([1, -2, -3])) == -2 and fx.streak(pd.Series([1, 0])) == 0
    assert fx.max_streaks(pd.Series([1, 1, -1, -1, -1, 1])) == (2, 3)
    st = fx.stats(pd.Series([10.0, -20.0, 30.0, -10.0]))
    assert st["up"] == 2 and st["down"] == 2 and st["win"] == 0.5 and st["cum"] == 10 and abs(st["exp"] - 2.5) < 1e-9 and st["avg_move"] == 17.5
    assert abs(st["pf"] - 40 / 30) < 1e-9
    # ③ スコア：偏りがあれば強く、偶然なら弱く。向きは符号で決まる
    rng = np.random.default_rng(5)
    base_s = pd.Series(rng.normal(0, 15, 400))
    sell = pd.Series(-abs(rng.normal(12, 5, 20)))
    buy = pd.Series(abs(rng.normal(12, 5, 20)))
    ss_, sb_ = fx.score_session(sell, base_s, cfg), fx.score_session(buy, base_s, cfg)
    assert ss_["s"] < -0.8 and sb_["s"] > 0.8 and fx.judge(ss_["s"], cfg)[0] == -1 and fx.judge(sb_["s"], cfg)[2] == "強い買い優勢"
    assert fx.judge(0.1, cfg)[0] == 0 and fx.judge(0.4, cfg)[2] == "やや買い優勢" and fx.judge(-0.6, cfg)[2] == "売り優勢"
    assert not fx.score_session(sell.iloc[:5], base_s, cfg)["ok"], "日数が少ないときは判定しない"
    noise = [abs(fx.score_session(pd.Series(rng.normal(0, 15, 20)), base_s, cfg)["s"]) for _ in range(600)]
    assert np.mean([x >= 0.7 for x in noise]) < 0.10 and np.mean([x < 0.3 for x in noise]) > 0.35, "偶然の値動きで『強い』を出しすぎない"
    # 勝率だけでなく大きさも見る：同じ勝率でも、負け日が大きければ弱い
    a = pd.Series([5.0] * 14 + [-5.0] * 6)
    c_ = pd.Series([5.0] * 14 + [-60.0] * 6)
    assert fx.score_session(a, base_s, cfg)["s"] > fx.score_session(c_, base_s, cfg)["s"] + 0.3
    # ④ 全体の分析（デモの人工データ：東京買い・欧州売り・NY買い、直近は欧州が強い売り）
    res = fx.analyze(fx.build_sessions(fx.demo_bars(), cfg), cfg)
    assert res["ok"] and res["ranking"][0]["key"] == "europe" and res["ranking"][0]["side"] == -1 and res["ranking"][0]["score"] >= 70
    assert [r["rank"] for r in res["ranking"]] == [1, 2, 3] and res["split"] and res["leader"]["name"] == "欧州"
    eu = res["sessions"]["europe"]
    assert eu["trend"]["verdict"] == "欧州勢売り優勢" and eu["trend"]["streak"] <= -5 and len(eu["weekdays"]) == 5 and all(w["n"] > 20 for w in eu["weekdays"])
    assert eu["windows"]["all"]["n"] > eu["windows"]["mid"]["n"] > eu["windows"]["recent"]["n"] and res["net"]["headline"].startswith("いまは")
    # ⑤ 蓄積と、取得失敗時の継続
    with tempfile.TemporaryDirectory() as td:
        c2 = copy.deepcopy(cfg)
        c2["fx"]["store"] = str(Path(td) / "h.csv")
        full = fx.demo_bars(60)
        calls = []
        def dl(tk, period):
            calls.append(period)
            return full.iloc[:600] if len(calls) == 1 else full.iloc[400:]
        bars1, w1 = fx.update_store(c2, dl)
        bars2, w2 = fx.update_store(c2, dl)
        assert calls == [cfg["fx"]["first_period"], cfg["fx"]["update_period"]] and w1 is None and w2 is None
        assert len(bars2) == len(full) and not bars2.index.has_duplicates, "足して重複を除く"
        def boom(tk, period):
            raise RuntimeError("接続できません")
        bars3, w3 = fx.update_store(c2, boom)
        assert len(bars3) == len(full) and "前回までのデータ" in w3
        e_bars, e_w = fx.update_store(dict(c2, fx=dict(c2["fx"], store=str(Path(td) / "none.csv"))), boom)
        assert e_bars.empty and e_w
    # ⑥ ページ
    with tempfile.TemporaryDirectory() as td:
        ctx = fx.page_context(res, None, fx.demo_bars().index[-1], datetime(2026, 11, 18, 7, 15, tzinfo=JST), True)
        html = fx.write_page(ctx, fx.build_sessions(fx.demo_bars(), cfg), Path(td) / "fx").read_text(encoding="utf-8")
        for w in ("市場勢力図", "売りスコア", "買いスコア", "ランキング", "曜日別", "累積pips", "月曜欧州", "欧州勢売り優勢", "見方・注意", "デモ表示"):
            assert w in html, w
        bad = fx.page_context({"ok": False, "reason": "テスト"}, "取得失敗", None, datetime(2026, 11, 18, 7, 15, tzinfo=JST), False)
        assert "まだ分析できません" in fx.write_page(bad, pd.DataFrame(), Path(td) / "fx2").read_text(encoding="utf-8")
    print("ok  ドル円分析（セッション損益・統計・スコア・偶然の値動きでの誤判定・蓄積と失敗時の継続・ページ）")


def test_tracking():
    import tempfile
    from pathlib import Path
    from src import tracking as TR
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        panel, topix, hist, idx = _tracking_fixture(tmp)
        c = {**cfg, "tracking": {**cfg["tracking"], "max_days": 20}}
        led = tmp / "ledger.json"
        res = TR.build(panel, topix, hist, led, c)
        ents = {(e["code"], e["date"]): e for e in res["entries"]}
        assert len(res["entries"]) == 3, [(e["code"], e["date"]) for e in res["entries"]]       # 1111が1件、2222が2件（再掲載）
        a = ents[("1111", f"{idx[1]:%Y-%m-%d}")]
        assert abs(a["buy"] - panel["open"]["1111.T"].iloc[2]) < 1e-9                              # 買値＝翌営業日の始値
        s1 = a["series"][1]
        assert s1[0] == 1 and abs(s1[3] - (panel["close"]["1111.T"].iloc[2] / panel["open"]["1111.T"].iloc[2] - 1)) < 1e-9
        assert a["days"] == 20 and a["closed"] and a["status"] == "追跡終了" and set(a["cp"]) >= {"1", "5", "10", "20"}
        assert a["ret"] > 0 and a["excess"] is not None and a["max_ret"] >= a["ret"] >= a["min_ret"]
        b = ents[("2222", f"{idx[4]:%Y-%m-%d}")]
        assert b["ret"] < 0 and b["excess"] < 0
        again = ents[("2222", f"{idx[33]:%Y-%m-%d}")]                                                # 載ってから数日しか経っていない
        assert again["status"] in ("掲載中", "圏外へ") and again["days"] == 6 and again["rank"] == 5 and again["status"] == "圏外へ"
        assert ents[("2222", f"{idx[4]:%Y-%m-%d}")]["cur_rank"] is None and a["cur_rank"] == 1
        # 集計：件数・平均・勝率・TOPIX比、区分別
        h5 = res["stats"]["by_horizon"][5]
        assert h5["n"] == 3 and abs(h5["win"] - 1 / 3) < 1e-9 and h5["excess"] is not None
        assert sum(r["n"] for r in res["stats"]["by_stance"]) <= res["stats"]["by_horizon"][10]["n"]
        assert [r["n"] for r in res["stats"]["by_rank"]] == [1, 1]
        # 暴騰率：h営業日以内の最高の騰落率で数える（期間が終わった銘柄だけ）
        sg = {r["name"]: r for r in res["stats"]["surge"]["rows"]}
        c5 = sg["全体"]["cells"][5]
        assert c5["n"] == 3 and c5["lv"][10.0]["k"] == 0 and sg["1位"]["cells"][5]["n"] == 1 and sg["1〜3位"]["cells"][5]["n"] == 1
        assert a["pk"]["5"] >= 0.04 and b["pk"]["5"] < 0.01                              # 毎日+1%の株は5日で約+5%、毎日-1%の株は上がらない
        hi = {**cfg["tracking"], "surge_levels": [3, 10]}
        assert TR.surge_table(res["entries"], hi)["rows"][-1]["cells"][5]["lv"][3.0]["k"] == 1       # +3%以上に達したのは上がる株だけ
        # 台帳：追跡が終わった銘柄は、株価データが無くなっても結果が残る／取得できない銘柄は最後の記録を表示
        gone = {k: v.drop(columns=["1111.T", "2222.T"]) for k, v in panel.items()}
        res2 = TR.build(gone, topix, hist, led, c)
        e2 = {(e["code"], e["date"]): e for e in res2["entries"]}
        assert e2[("1111", f"{idx[1]:%Y-%m-%d}")]["ret"] == a["ret"] and e2[("1111", f"{idx[1]:%Y-%m-%d}")]["status"] == "追跡終了"
        st2 = e2[("2222", f"{idx[33]:%Y-%m-%d}")]                                                       # 追跡中に株価が取れなくなった → 最後の記録を表示
        assert st2["stale"] and st2["status"].startswith("価格を取得できません") and st2["ret"] == again["ret"]
        assert e2[("2222", f"{idx[4]:%Y-%m-%d}")]["ret"] == b["ret"] and e2[("2222", f"{idx[4]:%Y-%m-%d}")].get("closed") or e2[("2222", f"{idx[4]:%Y-%m-%d}")]["stale"]
        # 翌営業日の始値が無い（最新日に載った）→ 買値待ち
        pend = {k: v.iloc[:2] for k, v in panel.items()}
        res3 = TR.build(pend, topix.iloc[:2], hist, tmp / "l3.json", c)
        assert any(e["status"] == "買い値待ち" for e in res3["entries"])
        # 出力：SVG・CSV・JSON
        assert "<svg" in TR.spark_svg(a["spark"]) and TR.spark_svg([1.0]) == "" and "<svg" in TR.avg_chart_svg(res["stats"])
        out = tmp / "docs"; out.mkdir()
        TR.write_files(out, res, "2026-11-13T07:00")
        lines = (out / "tracking_daily.csv").read_text(encoding="utf-8-sig").splitlines()
        assert len(lines) == 1 + sum(len(e["series"]) for e in res["entries"]) and lines[0].startswith("掲載日,コード")
        js = json.loads((out / "tracking.json").read_text(encoding="utf-8"))
        assert any(len(v) > 5 for v in js["entries"].values())
        txt = json.dumps(res["stats"], ensure_ascii=False) + "".join(TR.STANCE_LABEL.values())
        assert not any(w in txt for w in FORBIDDEN)
    print("ok  追跡・検証（掲載の検出・再掲載・買値=翌始値・TOPIX比・台帳の保持・買値待ち・CSV）")


def test_end_to_end():
    for cmd in (["-m", "src.backtest", "--demo"], ["-m", "src.main", "--demo"]):
        res = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
    d = json.loads((ROOT / "docs" / "latest.json").read_text(encoding="utf-8"))
    assert len(d["picks"]) == cfg["top_n"]
    for p in d["picks"]:
        assert 0 < len(p["explain"]) <= 100, p["name"]
        assert not any(w in p["explain"] for w in FORBIDDEN)
        assert set(p["parts"]) == {"trend", "drawdown", "ma50", "volume", "strength", "hold", "quality", "market"}
        assert abs(p["main"] - p["lead_pts"] - p["main_base"]) <= 0.11 and abs(p["lead_pts"]) <= cfg["lead"]["max_points"] * cfg["lead"]["rank_effect"] + 1e-6
        assert isinstance(p["cautions"], list) and "legacy" in p
    mains = [p["main"] for p in d["picks"]]
    assert mains == sorted(mains, reverse=True), "総合点の順になっていない"
    csv_text = (ROOT / "docs" / "excluded_events.csv").read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0] == "銘柄コード,銘柄名,除外理由,検出イベント,検出日"
    excluded_codes = {ln.split(",")[0] for ln in csv_text.splitlines()[1:]}
    assert not excluded_codes & {p["code"] for p in d["picks"]}, "除外銘柄がランキングに残っている"
    summ = json.loads((ROOT / "data" / "demo" / "backtest_summary.json").read_text(encoding="utf-8"))
    assert set(summ["periods"]) == {"1y", "3y", "5y", "10y"} and all(summ["periods"].values())
    assert "periods_legacy" in summ
    assert summ["validation"]["ok"] and len(summ["validation"]["tables"]) >= 10
    for k in ("win", "avg_win", "avg_loss", "pf", "max_dd", "sharpe", "ann_ret"):
        assert k in summ["periods"]["5y"], k
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    for word in ("【強み】", "【押し目状況】", "【注意事項】", "押し目スコア", "イベント点", "なぜ選ばれた",
                 "補助指標", "期間別の成績", "excluded_events.csv", "推奨撤退ライン", "リスク幅", "ATR", "有効性の検証", "点数帯ごとの成績", "現在の相場判定", "運用ガイド", "本日はギャップダウンリスクあり", "寄り付き前の指標", "【外部環境】", "順位への影響はごく小さく", "最終更新", "追跡・検証ページ", "順位別・スコア帯別の分析", "【決算状況】", "決算まで"):
        assert word in html, word
    # ㉒ 保有銘柄：暗号化ファイル・名前一覧・前回比較・3ブロック
    from src import holdings as H
    enc = json.loads((ROOT / "docs" / "holdings_status.enc.json").read_text(encoding="utf-8"))
    hd = H.decrypt_payload(enc, "demo-passphrase")["holdings"]
    assert {h["action"]["key"] for h in hd.values()} >= {"hold", "sell"}, "デモに保有継続と売却検討が両方出るはず"
    assert any(not h["found"] for h in hd.values()), "存在しないコードは『見つかりません』で残す"
    top_codes = {p["code"] for p in d["picks"]}
    assert any(c not in top_codes for c in hd), "ランキング外の保有銘柄も監視されている"
    outside = [c for c, h in hd.items() if c not in top_codes and h["found"]]
    public = (ROOT / "docs" / "latest.json").read_text(encoding="utf-8") + (ROOT / "docs" / "holdings_status.enc.json").read_text(encoding="utf-8")
    assert not any(f'"{c}"' in public for c in outside), "保有銘柄が公開ファイルに平文で出ている"
    names = json.loads((ROOT / "docs" / "names.json").read_text(encoding="utf-8"))
    assert len(names) >= 100 and all(len(k) == 4 for k in names)
    hist = sorted((ROOT / "data" / "demo" / "history").glob("*.json")) or sorted((ROOT / "data" / "history").glob("*.json"))
    rec = json.loads(hist[-1].read_text(encoding="utf-8"))
    assert len(rec["ranking"]) == cfg["movers"]["save_top"]
    assert not any(f'"{c}"' in json.dumps(rec) for c in outside if c not in {x["code"] for x in rec["ranking"]}), "保有銘柄を履歴に書いてはいけない"
    assert all("rank_change" in p for p in d["picks"])
    if any(p["rank_change"] for p in d["picks"]):
        assert "前回" in html and "rank-move" in html
    for word in ("① 本日の押し目候補", "急上昇・急下降", "保有銘柄管理・監視", 'id="holdings"', "holdings.js"):
        assert word in html, word
    if shutil.which("node"):
        res = subprocess.run(["node", "tests/holdings_js_test.js", "docs/holdings_status.enc.json", "demo-passphrase"],
                             cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0 and "JS OK" in res.stdout, res.stdout + res.stderr
    print("ok  デモ一式（10銘柄・総合点順・除外CSV・押し目/強み/注意・期間別成績・保有銘柄の暗号化）")


if __name__ == "__main__":
    for fn in (test_config_and_weights, test_momentum_points, test_no_lookahead, test_events,
               test_dividend_and_exclusion, test_expected_and_metrics, test_text, test_pullback, test_tdnet, test_accuracy_21, test_holdings,
               test_excluded_csv, test_data_fixes, test_validation, test_stance, test_premarket, test_lead, test_tracking, test_rankband, test_quality_first, test_fx, test_negative, test_entry10, test_end_to_end):
        fn()
    print("\nALL OK")
