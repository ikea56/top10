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
        "vol_ratio": [1.0, 1.0, 1.0]}, index=idx)
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
    # 前回比較
    mk = lambda ranks: [{"code": c, "name": f"N{c}", "rank": r, "main": 80} for c, r in ranks.items()]  # noqa: E731
    prev = mk({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 20, "M": 25})
    cur = mk({"A": 1, "K": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "N": 10, "B": 18, "M": 14, "L": 29, "J": 30})
    r = movers.compare(prev, cur, cfg["movers"])
    assert [x["code"] for x in r["new"]] == ["K", "N"] and r["new"][1]["prev"] is None
    assert [x["code"] for x in r["out"]] == ["B", "J"] and r["out"][0]["rank"] == 18
    assert [x["code"] for x in r["up"]] == ["M"] and r["up"][0]["delta"] == 11
    assert [x["code"] for x in r["down"]] == ["L"] and r["down"][0]["delta"] == 9
    cur2 = mk({"A": 1, "L": 9}); prev2 = mk({"A": 1, "L": 4})
    assert movers.compare(prev2, cur2, cfg["movers"])["down"][0]["code"] == "L"
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


def test_end_to_end():
    for cmd in (["-m", "src.backtest", "--demo"], ["-m", "src.main", "--demo"]):
        res = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
    d = json.loads((ROOT / "docs" / "latest.json").read_text(encoding="utf-8"))
    assert len(d["picks"]) == cfg["top_n"]
    for p in d["picks"]:
        assert 0 < len(p["explain"]) <= 100, p["name"]
        assert not any(w in p["explain"] for w in FORBIDDEN)
        assert set(p["parts"]) == {"trend", "drawdown", "ma50", "volume", "quality", "market"}
        assert p["main"] == p["pb_total"] + p["td_score"] or abs(p["main"] - p["pb_total"] - p["td_score"]) <= 1
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
    for k in ("win", "avg_win", "avg_loss", "pf", "max_dd", "sharpe", "ann_ret"):
        assert k in summ["periods"]["5y"], k
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    for word in ("【強み】", "【押し目状況】", "【注意事項】", "押し目スコア", "イベント点", "なぜ選ばれた",
                 "補助指標", "期間別の成績", "excluded_events.csv", "推奨撤退ライン", "リスク幅", "ATR"):
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
               test_excluded_csv, test_data_fixes, test_end_to_end):
        fn()
    print("\nALL OK")
