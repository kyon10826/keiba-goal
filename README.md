# keiba-goal

JRA の普通の開催日に、**単勝・複勝それぞれの的中確率**を校正済みの確率として出し、
「1 日の予算 x を目標額 A に届かせる確率」を最大にする買い目と金額を推奨するツールです。
馬券の自動購入はしません (推奨の表示と、買ったかどうかの記録まで)。

- 設計書: [docs/design.md](docs/design.md) (仕様の正本)
- 特徴量ストアの作り方: [docs/features.md](docs/features.md)

## 何をするか

| 部分 | 内容 |
|---|---|
| 予測モデル | 券種別 (単勝 = 1 着、複勝 = 支払着数以内) の直接分類。数百の特徴量 (過去 5 走・通算/条件別成績・スピード指数・騎手/血統の前日時点統計・発走 6 分前までの時系列オッズ)、ノイズレースの除外、FLAML (LightGBM / XGBoost / CatBoost) + Optuna、開催日境界の時系列 CV、5 seed 平均 + Isotonic 校正 |
| 投票 | Goal-reaching: 残り回数 k から狙う有効オッズ o\*_k を計算し、bold stake・オッズ下落補正 (q0.1)・単勝/複勝の重なりを考慮した的中確率 (2 段 Plackett–Luce の厳密列挙)・100 円単位の配分で、目標に届く確率が最大のバスケットを選ぶ |
| 運用 | 1 日 1 セッション (朝に予算・目標・最大回数を決める)。イベントソース型の記録で、払戻の確定まで次の判断を止める |

参考にした資料:
- Shohei Mitani「競馬で学ぶ機械学習の基本と実践」(YAPC::Fukuoka 2025, [Speaker Deck](https://speakerdeck.com/shoheimitani/machine-learning-with-horse-racing)) の改良モデル (83 枚目以降)
- Hiroshi Kishi「競馬 AI コンペ解法 — 券種別 LightGBM と『目標到達確率』を最大化する投票戦略」(AI 競馬予想マスターズ 2026 の解法共有) の投票方式

## 期待値について (大事)

Goal-reaching は**期待値がマイナスのまま「目標に届く確率」を最大にする方策**で、長期で儲かる手法ではありません。
市場と同じ精度 (エッジなし) なら、既定の 1 日 (10,000 円 → 20,000 円、最大 3 回) は次のとおりです。

| 項目 | 値 |
|---|---|
| 目標到達確率 (理論値) | 41.8% |
| どんな買い方でも超えられない上限 | 42.6% |
| 終了時資金の期待値 | 約 8,360 円 (−16%) |
| 資金をほぼ全部失う日の割合 | 約 58% |

## データ (リポジトリには含みません)

- 学習データ (成績データ 47 列・時系列オッズ) は、配布元の利用条件により再配布できないため含めていません。
- netkeiba から補う部分 (2025 年 9 月以降の結果、払戻、発走時刻) は、公開ページへのログインなしの GET のみです。
  全プロセス共通で 2.5 秒以上の間隔を空け、取得済みのページはキャッシュして再取得しません。
  取得したページそのものは含めていません。利用する場合は netkeiba の利用規約に従ってください。
- 生成物はすべて `data/` (git 管理外) に出力されます。

## セットアップ

Python 3.12 と [uv](https://docs.astral.sh/uv/) を使います。

```bash
uv sync --extra dev
cp config/default.example.yaml config/default.yaml   # paths を手元のデータの場所に書き換える
```

主なコマンド:

```bash
.venv/bin/python -m keiba_goal.data.build --step all          # 成績データ・時系列オッズ → data/interim/
.venv/bin/python -m keiba_goal.features.build --stage all     # 特徴量ストア → data/features/
```

## テスト

```bash
.venv/bin/python -m pytest
```

- ネットワークには触れません (保存ページの fixture で検査します)。
- 実データ由来の fixture (netkeiba の保存ページ、成績データ・時系列オッズの抜粋、それから作った parquet) は
  `tests/fixtures/` に置く前提で、同梱していません。無い環境では、それを読むテストは理由付きで skip され、
  合成データだけで動くテスト (投票の数理・CV・登録簿・パーサの異常系など) が実行されます。
- 特徴量の fixture は、データがあれば `python -m keiba_goal.features.fixture` で作り直せます。
