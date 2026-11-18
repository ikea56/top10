# ドル円 市場参加者分析ページ
- 新規 src/fx.py・templates/fx.html・.github/workflows/fx.yml・config.yml fx:。株のランキングとは独立（python -m src.fx）。
- 1時間足を data/fx/usdjpy_1h.csv に蓄積（取得失敗時は蓄積分で継続）。セッション別pips→統計→買い/売りスコア→判定・ランキング・曜日別。
- 未検証：実際のYahooデータ（この環境から接続不可）。人工データで計算・ページ・失敗時の継続をテスト。
