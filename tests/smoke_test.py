"""動作確認テスト。  python tests/smoke_test.py   （ネット接続不要・約10秒）
壊れていなければ最後に「ALL OK」と出ます。"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import features, scoring, explain          # noqa: E402
from src.config import load_config                  # noqa: E402
from src.fundamentals import _yoy                   # noqa: E402

cfg = load_config()
sc = cfg["scoring"]


def make_panel(last_vol_mult=3.5, n=120):
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(np.linspace(100, 160, n))
    close.iloc[-1] = close.iloc[-2] * 1.05
    vol = pd.Series(1000.0, index=range(n))
    vol.iloc[-1] = 1000 * last_vol_mult
    mk = lambda s: pd.DataFrame({"T.T": s.values}, index=idx)  # noqa: E731
    return {"open": mk(close), "high": mk(close), "low": mk(close), "close": mk(close), "volume": mk(vol)}


def test_points():
    f = features.compute_features(make_panel(3.5))
    p = features.tech_points(f, sc)
    assert f["vol_ratio"].iloc[-1, 0] == 3.5, "出来高倍率は『前日までの20日平均』との比"
    assert p["pts_high"].iloc[-1, 0] == 30 and p["pts_vol"].iloc[-1, 0] == 30 and p["pts_trend"].iloc[-1, 0] == 20
    f2 = features.compute_features(make_panel(2.2))
    assert features.tech_points(f2, sc)["pts_vol"].iloc[-1, 0] == 20
    f3 = features.compute_features(make_panel(1.5))
    assert features.tech_points(f3, sc)["pts_vol"].iloc[-1, 0] == 0
    print("ok  テクニカル点（高値・出来高・トレンド）")


def test_no_lookahead():
    """未来のデータを足しても、過去日のスコアが変わらないこと"""
    p = make_panel(3.5, 150)
    full = features.tech_points(features.compute_features(p), sc)["tech_raw"]
    cut = {k: v.iloc[:100] for k, v in p.items()}
    part = features.tech_points(features.compute_features(cut), sc)["tech_raw"]
    assert np.allclose(full.iloc[:100].fillna(-1), part.fillna(-1)), "未来の情報が混ざっています"
    print("ok  未来データ混入なし")


def test_fundamental_points():
    assert scoring.earn_points({"op_yoy": 0.68}, sc) == 25
    assert scoring.earn_points({"op_yoy": 0.20}, sc) == 15
    assert scoring.earn_points({"op_yoy": 0.19}, sc) == 0
    assert scoring.earn_points({"op_yoy": None, "turnaround": True}, sc) == 15
    assert scoring.mcap_points({"market_cap": 6e10}, sc) == 10
    assert scoring.mcap_points({"market_cap": 2e10}, sc) == 5
    assert scoring.mcap_points({"market_cap": 5e9}, sc) == 0
    assert scoring.max_raw(sc) == 115 and scoring.to_score(115, sc) == 100
    assert _yoy(150, 100) == (0.5, False) and _yoy(10, -5) == (None, True) and _yoy(-5, 10) == (-1.5, False)
    print("ok  業績・時価総額の点数／前年比")


def test_labels_and_text():
    assert scoring.bull_label(80, False, "good", cfg)[0] == "強気"
    assert scoring.bull_label(80, True, "good", cfg)[0] == "やや強気"      # 過熱で格下げ
    assert scoring.bull_label(80, False, "bad", cfg)[0] == "やや強気"      # 逆風で格下げ
    assert scoring.bull_label(40, False, "good", cfg)[0] == "中立"
    r = {"hi52w": True, "hi60": True, "hi30": True, "vol_ratio": 12.3, "trend": True}
    assert len(explain.beginner_text(r, {"op_yoy": 12.34, "turnaround": False})) <= 100
    print("ok  強気判定・100文字制限")


def test_end_to_end():
    for cmd in (["-m", "src.backtest", "--demo"], ["-m", "src.main", "--demo"]):
        res = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
    import json
    d = json.loads((ROOT / "docs" / "latest.json").read_text(encoding="utf-8"))
    assert len(d["picks"]) == cfg["top_n"]
    for p in d["picks"]:
        assert p["reasons"], "選出理由が空"
        assert 0 < len(p["explain"]) <= 100, f"解説が100文字超: {p['explain']}"
        assert p["risks"], "リスクが空"
    scores = [p["score"] for p in d["picks"]]
    assert scores == sorted(scores, reverse=True), "スコア順になっていない"
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    for word in ("なぜ選ばれた", "初心者向け解説", "リスク", "TradingView", "viewport", "prefers-color-scheme"):
        assert word in html or word in (ROOT / "docs" / "style.css").read_text(encoding="utf-8"), word
    print("ok  デモ一式の生成（10銘柄・理由・100文字以内・リスク）")


if __name__ == "__main__":
    test_points()
    test_no_lookahead()
    test_fundamental_points()
    test_labels_and_text()
    test_end_to_end()
    print("\nALL OK")
