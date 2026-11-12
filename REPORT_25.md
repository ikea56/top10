# ㉓ 相場環境判定
- `src/market.py`：market_stance（3段階・根拠・運用ガイド）／`config.yml`：stance、market.growth_below_ma75(-3)
- `src/pullback.py`：market_points に growth_weak（グロースが75日線の下で−3、下限−15は維持）
- `src/prices.py`：2516.T を取得（失敗しても動作）／`src/main.py`：stance を履歴・各計算へ反映
- 画面：ランキング最上部に「🧭 現在の相場判定」（理由・運用ガイド）。🔴でもランキングは表示
- テスト：test_stance（買い優勢/中立/防御優先、グロース欠損、減点）
## 注意
- 過去再現（バックテスト）にはグロースの減点を含みません（ETFの履歴が短いため）。
- グロースは指数そのものではなく連動ETF(2516)で代用しています。
- 判定は新規買いの積極度の目安で、売りの推奨ではありません。
