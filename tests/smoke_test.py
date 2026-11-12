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
    ep = lambda dt: pullback.earnings_penalty(dt, today, cfg)  # noqa: E731
    assert ep("2026-11-05")[0] == 10 and ep("2026-11-06")[0] == 5 and ep("2026-11-09")[0] == 5
    assert ep("2026-11-10")[0] == 0 and ep("2026-10-30")[0] == 0 and ep(None)[0] == 0
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
    assert act(broken=["200日線割れ"])["key"] == "watch" and act(earn_pen=5, earnings_days=4)["key"] == "watch"
    assert act(earnings_days=8)["key"] == "watch" and act(earnings_days=30)["key"] == "hold"
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


def test_end_to_end():
    for cmd in (["-m", "src.backtest", "--demo"], ["-m", "src.main", "--demo"]):
        res = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
    d = json.loads((ROOT / "docs" / "latest.json").read_text(encoding="utf-8"))
    assert len(d["picks"]) == cfg["top_n"]
    for p in d["picks"]:
        assert 0 < len(p["explain"]) <= 100, p["name"]
        assert not any(w in p["explain"] for w in FORBIDDEN)
        assert set(p["parts"]) == {"trend", "drawdown", "ma50", "volume", "strength", "quality", "market"}
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
                 "補助指標", "期間別の成績", "excluded_events.csv", "推奨撤退ライン", "リスク幅", "ATR", "有効性の検証", "点数帯ごとの成績", "現在の相場判定", "運用ガイド", "本日はギャップダウンリスクあり", "寄り付き前の指標", "【外部環境】", "順位への影響はごく小さく"):
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
               test_excluded_csv, test_data_fixes, test_validation, test_stance, test_premarket, test_lead, test_end_to_end):
        fn()
    print("\nALL OK")
