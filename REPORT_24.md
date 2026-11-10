# ㉔ 有効性の検証（バックテスト）

## 追加したもの
- `src/validation.py` 新規：点数帯別・IC・五分位・上位Nと全体平均の差(t値/ブートストラップ95%CI)・要素除外(ablation)・入れ替わり率・始値/終値買い・ATR撤退シミュレーション・地合い別/年別/直近OOS・業種偏り・🟢🟡🔴判定
- `src/backtest.py`：run_replay が検証用データを集め summary["validation"] に保存（失敗しても既存の成績表は残る）
- `src/features.py`：atr_frame 追加 / `config.yml`：validation 設定 / テンプレート・CSS：「🔬 有効性の検証」欄
- `tests/smoke_test.py`：test_validation（有意なデータ→有効、乱数→有効と判定しない、CI再現性、撤退シミュ）

## 使い方
Actions → weekly-backtest を実行 → daily-update を実行。サイトに結果が出ます。

## 限界（正直に）
- この環境からは Yahoo に接続できず、実データの結果は未確認です。デモ（乱数）では「有効性なし」と正しく判定されることだけ確認済み。
- 生存者バイアス（成績は甘め）。業績・TDnet・PER は過去データが無く検証不可。
- 重なる期間はt値を間引いて計算していますが、多数の指標を見るため偶然の当たりが混じり得ます。
- 判定が🟢でも将来を保証せず、売買判断・金額はご自身で決めてください。
