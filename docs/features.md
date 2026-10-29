# 特徴量ストア (P3) — 規約・API・グループ一覧

design.md §4 (特徴量)・§5 (目的変数)・§12-4 (as-of テスト) の実装。後からグループを足す人はまず「4. グループの足し方」を読む。

- コード: `src/keiba_goal/features/` (グループは `features/groups/` に 1 モジュール 1 グループ)
- 作成: `.venv/bin/python -m keiba_goal.features.build --stage A|B|labels|all [--groups g1,g2] [--years 2003-2024]`
- テスト: `tests/test_features_*.py` (レジストリ検査・as-of 検査・ライブとの一致は登録済みの全グループに自動で効く)
- ライブ: `features/live.py` (朝に段階 A と補助出力 → 判断時刻に段階 B だけ、§9)

---

## 1. 全体の流れ

```
history.parquet ──(固定: data/state/inputs/)──┬─ 段階 A (全履歴を 1 回、グループごと) → features/A/{group}/year={Y}.parquet
ts_snap/{Y}.parquet ─────────────────────────┤
paid_places.parquet (あれば) ────────────────┴─ 段階 B (年ごと: 判断時刻の枠 + 段階 B のグループ + 組み立て)
                                                   → features/B/groups/{group}/year={Y}.parquet
                                                   → features/B/year={Y}.parquet   (race_id, race_key, date, horse_num + 全特徴量)
                                                 目的変数 → features/labels/year={Y}.parquet
```

- 段階 A は 1986 年からの全履歴で計算し、`features.stage_a_min_year` (2003) 以降の年を書く。**年ごとに切って計算しない** (前走が欠ける)。
- 段階 B と目的変数の既定の対象年は `features.years` (2003-2024 = 時系列オッズのある年)。
- B/year= の行 = その年の history の全行 (取消・障害・対象外クラスも含む)。学習で使う行は labels の `eligible` で選ぶ。
- 全特徴量は Float32、値の無いところは NaN (0 埋めしない)。±inf は出力の検査で弾く。
- 作成の最初に入力 (history・paid_places・ts_snap/{Y}) を `data/state/inputs/` にハードリンクで固定する。データ層が作業中にファイルを置き換えても 1 回の作成の中では同じ中身を読む。
- 各ファイルのメタデータに**指紋** (何から作ったか: 入力の sha256・コードの sha256・依存グループの指紋) を入れ、B/year= の組み立ては今の入力とコードで作ったファイルだけを使う (§10.1)。データ層が ts_snap や paid_places を作り直した後に一部のグループだけ作り直しても、古い判断時刻の枠から作ったファイルは混ざらない。

## 2. 段階の定義 (design §4.1 の精密化)

対象行 = 特徴量を作る行 (レース D のある馬)。「履歴」= history の全行 (対象行も含む。ライブでは当日の出馬表の行がレース後の列 null で入る)。

| 段階 | 対象行から読める列 | 履歴から読める行 | 作る単位 |
|---|---|---|---|
| A (朝) | 静的な属性 `registry.STATIC_COLUMNS` | date < D の行 (レース後の列も可)。**D の行 (対象行そのものも) の当日変わりうる列は読まない** | 全履歴を 1 回 |
| B (判断時刻) | 発走前の全列 (`PRE_RACE_COLUMNS` − 使わない列) + 判断時刻の列 `LIVE_COLUMNS` | date < D の行 | 年ごと (ライブはレースごとに判断時刻に作り直す)。**同じレースの行だけで決まる** (他のレースの当日の値・オッズを使わない) |

- 段階 A は朝に作ってキャッシュし、判断時刻の最新の出馬表 (騎手変更・取消・馬場変更・馬体重) では作り直さない。だから `ctx.history()` から D の行の `STAGE_B_COLUMNS` を読んではいけない (宣言では防げないので as-of 検査の (d) が見る、§8)。
- 段階 B はレースごとに判断時刻に作る。後のレースの馬体重・オッズは判断時刻にまだ無いので、`.over("race_key")` より広い範囲の当日の値を使わない (as-of 検査の (d) と、1 レースずつの判断のパリティ検査が見る)。

- **静的な属性** (前日の出馬表で決まり当日変わらない): `date venue_code race_num surface direction course_inout dist track_code corner_num class_code class_ord age_code weight_code id sex age waku_num horse_num` + race_id を分解した列 (`race_id race_key nk_race_id month day times place daily`) + 馬の属性 `horse father mother trainer_id` (血統・調教師は登録時に決まる。trainer_id は netkeiba 由来の行だけ値がある)。
- **当日変わりうる発走前の列** (`STAGE_B_COLUMNS`): `jockey_id basis_weight weight inc_dec blinker state weather horse_N`。騎手の統計・騎手 Elo・乗り替わり・斤量・馬体重・馬場・天気・頭数の特徴量は必ず段階 B。
- **判断時刻の列** (`LIVE_COLUMNS`、`decision.decision_frame` が作る。§3): `scratched field_open in_field_dec field_dec_size field_dec_fallback` と判断スナップショット `win_{x} place_lo_{x} place_hi_{x}` (x = 120, 60, 30, 6)。
- **どこにも使えない列**: 対象行のレース後の列 (`schema.POST_RACE_COLUMNS`: 着順・タイム・確定オッズ・確定人気・error_code・starters など)、ts_snap の診断列 (`vote_share snap_time staleness_min`)、確定オッズ (ts_final の `win_final place_lo_final place_hi_final`)、使わない列 (`year` 2 桁年・`leg` TARGET の脚質・`source`)。
- **「前日終了時点」(D−1)**: 騎手・種牡馬・コース基準時計・Elo などの状態は date < D の行だけから作る。**同じ日の先のレースの結果は使わない** (ライブで再現できない)。馬の過去走も date < D の出走 (同じ馬が同じ日に 2 回走る行は history に無い。それでも shift ではなく date < D で引く)。
- 過去の行のレース後の列 (前走の着順・確定オッズ・確定人気など) は使ってよい。

## 3. 判断時刻の枠 (decision frame)

段階 B のグループの `ctx.target(...)` が返す行。batch とライブで同じ関数 `decision.decision_frame(rows, snapshots, field_open)` を通す。

| 列 | 意味 | batch での作り方 | ライブ |
|---|---|---|---|
| `scratched` | 判断時刻までに取消・除外と分かっている馬 | error_code ∈ {1, 2} (出走取消・発走除外) | 出馬表の取消の印 |
| `win_{x}` `place_lo_{x}` `place_hi_{x}` | T-x のオッズ (欠損 NaN) | ts_snap/{Y} を 1 馬 1 行に | API の middle オッズを同じ形に |
| `in_field_dec` | 判断時点の出走馬集合 field_dec に入るか | T-6 で単勝オッズが付いている馬。そのレースに T-6 が 1 頭も無ければ scratched でない馬 | 同じ |
| `field_dec_fallback` | 上の代替を使ったレース | | |
| `field_dec_size` | field_dec の頭数 | | |
| `field_open` | 発売開始時に売られていた頭数 (判断時の複勝の支払着数を決める) | paid_places.field_open → 無ければ T-120/60/30/6 のどれかで単勝オッズが付いた馬の数 → それも無ければ field_dec_size | design §4.1 の規則 (登録頭数 − 前日発売の開始前に取消だった馬) で呼び出し側が渡す |

- 出走頭数・馬番/頭数・斤量偏差・レース内の順位/z 値・市場確率の正規化はすべて **field_dec** で作る (error_code は使わない)。
- field_dec 外の馬 (判断前の取消) の行も B/year= に残る。field_dec 上の相対値は NaN。
- `field_open_source` (paid_places / snapshots / field_dec) は作成ログ用で、特徴量の元には使えない。

## 4. グループの足し方

### 4.1 手順

1. `src/keiba_goal/features/groups/<name>.py` を作り、`GROUP = Group(...)` を置く (**モジュール名 = グループ名**)。登録はそれだけ (`groups.discover()` がパッケージ内のモジュールを全部読む。共有の一覧ファイルは無い)。
2. 特徴量ごとに `FeatureSpec` を宣言する (下の例の `S(...)`)。名前は `接頭辞_名前` (小文字・数字・_)。接頭辞はグループで 1 つ (後ろに数字を付けてよい: `p1_ … p5_` は接頭辞 `p`)、他のグループと重ならないこと (例: `jk_`, `elo_`, `od_`)。`y_` (目的変数用) と、キー列・history の列・判断時刻の列・目的変数の列 (labels の `eligible` `n_paid_places` など) と同じ名前は使えない。
3. `build(ctx) -> pl.DataFrame` を書く: 列は `race_id` (Int64) + 宣言した特徴量 (すべて Float32) ちょうど、行は `ctx.target()` の全行。
4. `.venv/bin/python -m pytest tests/test_features_registry.py tests/test_features_asof.py tests/test_features_live.py -q -p no:cacheprovider` (自分のグループが自動で入る) と、自分のグループの単体テスト `tests/test_features_<name>.py` (手で書いた小さな履歴 `features.synth.history_frame` で値を確かめる)。
5. 実データ: 段階 A のグループは `--stage A --groups <name>` の後に `--stage B --groups <name>` (段階 A の名前だけなら B/year= の組み立てだけ。古い・無い段階 B のグループはそのとき自動で作り直される)。段階 B のグループは `--stage B --groups <name>`。最後に `.venv/bin/python -m keiba_goal.features.report` が終了コード 0 であること (§10)。他の人の段階 A のグループがまだ作られていない・古いときだけ `--allow-missing` (外したグループを表示し、メタデータに残す。report は問題として出す)。

### 4.2 例

```python
"""騎手の前日までの成績 (段階 B)。"""
import polars as pl
from keiba_goal.features import lags
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import f32, started
from keiba_goal.features.registry import Group, spec_maker

NAME = "jockey"
S = spec_maker(NAME, "B")                       # group と stage を埋める
SPECS = (
    # 今日の騎手 (jockey_id) は値ではなく結合キー → keys。履歴の jockey_id・error_code は history_sources
    S("jk_rides", "num", [], ["jockey_id", "error_code"], doc="D より前の騎乗数", keys=["jockey_id"]),
)


def build(ctx: Ctx) -> pl.DataFrame:
    rides = lags.events(ctx.history("jockey_id", "error_code").filter(started()), by=("jockey_id",))
    rows = lags.prior_count(ctx.target("jockey_id"), rides, by=("jockey_id",))   # date < D の騎乗数 = _n
    return rows.select("race_id", f32(pl.col(lags.N_PRIOR), "jk_rides")).collect()


GROUP = Group(NAME, "B", SPECS, build)
```

### 4.3 API

`FeatureSpec(name, group, stage, kind="num", target_sources=(), history_sources=(), doc="", keys=())`
- `kind`: `num` / `cat` (固定語彙のコード。LightGBM の名義変数候補) / `flag` (0/1)。
- `target_sources`: 対象行から**値を**読む列 (段階 A は STATIC_COLUMNS の中だけ、段階 B は発走前の列 + LIVE_COLUMNS)。ID 列 (`registry.ID_COLUMNS`: `jockey_id trainer_id father mother horse nk_race_id` など) は書けない (ID は特徴量にしない、design §4.1)。
- `keys`: 対象行の ID 列 (`registry.KEY_SOURCES`) のうち、as-of 統計の結合や前の行との一致の判定にだけ使うもの (例: 騎手の統計は `keys=["jockey_id"]`、`hd_same_jockey_k` も同じ)。段階の制限は target_sources と同じ (段階 A は jockey_id を使えない)。値そのもの (ID を数にしたもの) を出力しないこと。`id race_id race_key` はいつも ctx が返すので書かない。
- `history_sources`: それより前の行から読む列 (レース後の列・ID 列も可)。
- `spec_maker(group, stage)` は `S(name, kind, target, history, doc=, keys=)` の短縮形を返す。

`Group(name, stage, specs, build, requires=(), helpers=(), inputs=())`
- `requires`: 依存するグループ名。そのグループの `helpers` を `ctx.helper(name)` で、`inputs` に挙げた特徴量を `ctx.features(*names)` で読める。段階 A は段階 B に依存できない。循環は禁止。
- `helpers`: 他のグループに渡す中間表 `Helper(name, build, history_sources, own_post_race, doc)`。履歴だけから 1 回の作成につき 1 回作り、CLI では `data/state/helpers/{name}.parquet` に書く (ライブの「前日までの状態」の元)。**`own_post_race=True` は行自身のレース後の列を使う値** (例: speed_fig はその行のタイムから作る) で、その行より後の行の特徴量にしか使ってはいけない。
- `inputs`: requires のグループの特徴量のうち読むもの (例: relative は speed.sp_last など)。

`ctx` (`features.context.Ctx`。宣言していない列・補助出力・特徴量を読むと `AccessError`):
- `ctx.history(*cols) -> LazyFrame`: 履歴の全行 (キー列 `race_id race_key date horse_num id` は常に付く)。cols は `history_sources` の和集合。
- `ctx.target(*cols) -> LazyFrame`: 特徴量を作る行。段階 A は対象年 (ライブは当日) の履歴行、段階 B は判断時刻の枠。cols は `target_sources` と `keys` の和集合。
- `ctx.helper(name) -> LazyFrame`、`ctx.features(*names) -> LazyFrame` (race_id + names、target の行)、`ctx.years`。

出力の検査 (`pipeline.validate_output`): 列 = race_id + 宣言した特徴量ちょうど、Float32、race_id 一意、target の全行がある、±inf なし。null は NaN にそろえる。
レジストリの検査 (`registry.check_registry`) は `discover()` のほか、`build_store` / `live` / `asof.check_day` にグループを直接渡したときにもかかる。

### 4.4 道具

- `features.lags`: 「D より前の出来事」を引く。`events(lf, by)` は (by, date, race_key) 順に通し番号 `_seq` を付け、`prior_count(targets, events, by)` は対象行ごとに date < D の出来事の数 `_n` を付ける (同じ日は数えない。`join_asof(..., allow_exact_matches=False)`)。k 個前は `at_prior(targets, events, k, {列: 名前})` (= `_seq == _n − k`)、直前までの累積・窓の集計は出来事の表で `rolling_*().over(by)` / `cum_*().over(by)` を作って `at_prior(..., 1, ...)` で引く (aggregates.py)。馬以外 (騎手・種牡馬) も `by=("jockey_id",)` などで同じ。
- `features.exprs`: `started()` (error_code ∉ {1,2,3})、`going()` (良 0 / 稍 1 / 重 2 / 不 3)、`coded(col, 語彙)`、`positive(col)` (0 以下 → null)、`ratio(a, b)` (b ≤ 0 → null)、`days_between`、`n_paid_rule`、`f32(expr, name)` (Float32 + NaN)。語彙 `GOING WEATHER SEX` は固定。
- `features.groups.speed.started_with_figures(ctx, *cols)`: 出走行 + speed_fig の出来事の表 (requires=("speed",) のグループ用)。
- `features.vocab.course_vocab()`: コース (場 × 芝ダ × 距離) の固定語彙 (2003 年以前、`course_vocab.json`)。
- `features.synth.history_frame(rows)`: テスト用の小さな履歴 (HISTORY_DTYPES にそろえ、race_id などを作る)。
- `features.asof.check_day(...)`: as-of 検査 (§8)。`features.live.prepare_morning / decide / build_rows`: ライブの入口 (§9)。

### 4.5 守ること (レビューで見る点)

- 対象行のレース後の列を読まない (宣言で弾かれ、読めても as-of テストが落ちる)。過去の行を引くときは必ず date < D。
- 平滑化の事前分布・Elo の K と初期値など、データから決める定数は **2004 年より前のデータ**で決めてコードに固定する (design §4.1)。語彙も固定 (語彙外 = NaN)。
- 段階 B で重い全履歴の計算 (騎手の累積成績・騎手 Elo) は、年ごとにやり直さず Helper にする (例: `(jockey_id, date)` → その日の朝の値の表を Helper で作り、段階 B で `join_asof(..., by="jockey_id", allow_exact_matches=False)`)。Elo のような日ごとの逐次計算は「その日の全レースを朝のレーティングで評価してから更新」(同じ日の先のレースの結果を後のレースに入れない)。
- Helper はライブでは朝に 1 回作り、判断時刻に作り直さない (§9)。だから Helper の値も D の行の当日変わりうる列 (`STAGE_B_COLUMNS`) に依らないこと (date < D の行だけから作る。朝と判断時刻で出馬表が違う場合のパリティ検査が見る)。
- 段階 A のグループは `ctx.history()` の D の行 (対象行と同じ日の行) から当日変わりうる列を読まない。段階 B のグループは同じレースの行の値だけを使う (§2、as-of 検査 (d))。
- メモリ: 1 プロセスのピーク 2GB 以下 (M2 / 8GB、スワップ中)。全履歴の出来事の表は必要な列だけ Float32 で 1 回 collect、幅の広い出力は部分ごとに作って `hstack` (past_runs.py)、行ごとの Python ループは避ける。`--stage A --groups <name>` の表示 (秒・ピーク RSS) で確かめる。
- 段階 A の出力は対象年の全行 (約 111 万行) を持つ。対象行だけを `ctx.target()` から始めて結合すると速い。
- 同じ列を 2 回作らない (例: 出走数は `hs_n_starts`、乗り替わりは `hd_same_jockey_1`、直近 3/5 走のスピード指数の平均・最大は `sp_*`)。
- 出力の並びは自由 (検査で target の並びにそろえる。並びが同じなら写しを作らない)。

## 5. 登録済みのグループ (8 グループ、203 列)

| グループ | 段階 | 列数 | 依存 | 内容 |
|---|---|---|---|---|
| race_static | A | 13 | | `rs_venue rs_surface rs_direction rs_inout` (cat), `rs_dist`, `rs_dist_band` (SMILE: ≤1300 / ≤1899 / ≤2100 / ≤2700 / それ以上 → 1-5), `rs_class_ord` (-1 → NaN), `rs_age_code` (cat), `rs_weight_code` (0 不明 → NaN), `rs_race_num`, `rs_month`, `rs_course` (場×芝ダ×距離、2003 年以前の固定語彙、語彙外 NaN), `rs_corner_num` |
| horse_static | A | 6 | | `hs_waku hs_horse_num`, `hs_sex` (牡 1 / 牝 2 / セ 3), `hs_age`, `hs_career_days` (初出走からの日数、前の出走が無ければ NaN), `hs_n_starts` (D より前の出走数) |
| speed | A | 6 | | `sp_last`, `sp_mean3 sp_max3 sp_mean5 sp_max5`, `sp_same_surface_last` (今回と同じ芝ダの直近)。Helper `speed_fig` (own_post_race) |
| past_runs | A | 95 | speed | 1〜5 走前 (`p{k}_`): `rank rank_ratio time_diff last3f last3f_rank_ratio early_pos late_pos pci last3f_diff pop log_odds dist surface class_ord going starters days speed dnf` |
| aggregates | A | 36 | speed | 直近 3/5/10 走と通算 (`ag_l3_ ag_l5_ ag_l10_ ag_all_`): `rank_ratio_mean win_rate top3_rate time_diff_mean time_diff_min last3f_rank_ratio_mean pop_mean prize_sum`、10 走と通算の `speed_mean speed_max` |
| race_day | B | 5 | | `rd_going`, `rd_weather` (cat), `rd_field_size` (field_dec), `rd_n_paid_places` (field_open から ≥8 → 3、5-7 → 2、≤4 → 0), `rd_field_fallback` |
| horse_day | B | 14 | | `hd_basis_weight`, `hd_basis_weight_dev` (field_dec の平均との差), `hd_horse_num_ratio` (馬番/field_dec の頭数), `hd_blinker` ('B' → 1、null・空 → 0、それ以外 → NaN。下の注), `hd_weight hd_inc_dec`, `hd_weight_change_rate` (増減/体重), `hd_weight_vs_past_mean` (体重/前の出走の平均体重), `hd_basis_weight_change` (前走との斤量差), `hd_same_jockey_{1..5}` (k 走前と同じ騎手、`keys=["jockey_id"]`)。Helper `horse_prev` |
| relative | B | 28 | 元の特徴量のグループ | `config features.relative.base` の各特徴量 f について `rel_rank_{f}` (field_dec の値のある馬の中の昇順の順位、同値は平均順位) と `rel_z_{f}` (母標準偏差。値のある馬が 2 頭未満は NaN、全頭同じ値は 0)。field_dec 外は NaN |

細部:
- **過去走** = 同じ馬 (`id`) の date < D の**出走** (error_code ∉ {1,2,3}。取消・除外の行は数えない、競走中止は数えて `dnf` = 1)。着順 0 (中止・失格) は NaN。先行度 `early_pos` = 最初にある通過順位 (corner1→4) / 出走頭数、`late_pos` = 4 角 / 出走頭数 (2 角のレースも最後のコーナーは corner4 に入っている)。
- **スピード指数** `speed_fig` = (基準平均 − 走破タイム) / 基準標準偏差 (大きいほど速い)。基準は (場, 芝ダ, 距離, 馬場) ごとに、その行の年より前の 3 暦年の完走馬 (着順 ≥ 1、タイムあり)。標本 30 未満なら場をまたいだ (芝ダ, 距離, 馬場)、それも 30 未満なら NaN。クラス・斤量の補正はしない。
- **集計の窓** は「D より前の直近 w 回の出走」。平均・最小・最大は値のある走だけ、勝率・3 着内率は出走数で割る (中止は負け)。前の出走が無ければ NaN (賞金の合計も)。
- `rel_*` の元の特徴量の並びは config で変えられる (import 時に `config/default.yaml` から読む。`--config` では変わらない)。
- **ブリンカー** (`hd_blinker`、後で作る「前走のブリンカー」も同じ): history の約束は 'B' = 着用、null = 着用なし。record は空欄が着用なし、netkeiba の行は出馬表 (shutuba_past) に B の印が無い馬が null。**出馬表の無い netkeiba の行 (着用が不明) は history に入らない**: 窓 1 は出馬表の無いレースを捨て (`netkeiba/rebuild.py` の build)、出馬表を取らない窓 2/3 の行は同じ race_id の record の行が優先される (`data/history.py`)。この約束が変わって不明の値 ('B'・null・'' 以外) が来たら NaN になる (0 にしない)。netkeiba 層が出馬表なしの行を history に入れるようにするなら、不明を null 以外 (例 '?') で書くこと。

## 6. 後から作るグループ (段階の割り当て)

| グループ (design §4.2) | 段階 | メモ |
|---|---|---|
| 条件別成績 (同芝ダ・同距離帯・同場・同コース・重不良・同回り・±200m、ベイズ平滑) | A | `lags` を by=("id", 条件) で。事前分布は 2004 年より前で固定。**「同騎手」での成績だけは B** (騎手が当日変わる) |
| 変化 (距離・芝ダ・クラス・場・間隔・初出走・キャリア) | A | past_runs の `p1_*` を `inputs` で読めば結合し直さなくてよい。キャリア日数は `hs_career_days` が既にある |
| 変化のうち 騎手・斤量・体重変化率・体重の通算平均比・ブリンカー着用変化 | B | 乗り替わり・斤量差・体重変化率・通算平均比は horse_day に**既にある** (作り直さない)。ブリンカー着用変化は前走のブリンカーを Helper で (null の意味は §5 の注) |
| 調子の推移 (傾き・最高との差・連続改善数) | A | aggregates と同じ出来事の表で |
| Elo (馬 Elo・芝ダ別馬 Elo) | A | 日ごとの逐次計算を Helper に (同じ日の結果は翌日から)。K・初期値は 2004 年より前で固定 |
| Elo (騎手 Elo) | B | 騎手 × 日の表を Helper にして段階 B で as-of 結合 |
| 騎手 (D−1: 勝率・複勝率の平滑平均と 95% 下限、365 日、芝ダ・コース・距離帯・場別、騎乗数、騎手×馬) | B | 同上 (年ごとに全履歴をやり直さない) |
| 種牡馬・母父・母 (D−1) | A | 父・母は STATIC_COLUMNS。母の 10 桁 ID の解決は design §4.2 の規則 (子の生年より後の行を使わない) |
| 時系列オッズ (T-120/60/30/6) | B | `ctx.target("win_6", ..., "in_field_dec")` だけで作れる。市場確率の正規化は field_dec の上で。欠損フラグ付き。確定オッズ・診断列は読めない |
| レース内相対 (主要 30 特徴量 × 順位/z) | B | relative グループ。`config features.relative.base` に "グループ.特徴量" を足す |
| (v2) 調教師・厩舎 | A (trainer_id は静的) | 窓 3 のクロール後 |

## 7. 目的変数 (`features/labels/year={Y}.parquet`, labels.py)

| 列 | 型 | 定義 |
|---|---|---|
| race_id, race_key, date, horse_num | キー | |
| `y_win` | Int8 | 着順 1 (同着は両方 1)。出走していない馬は null |
| `y_place` | Int8 | 1 ≤ 着順 ≤ 支払着数。支払着数 0 (複勝なし) か不明のレース、出走していない馬は null |
| `started` | Bool | error_code ∉ {1,2,3} |
| `eligible` | Bool | 平地 (surface != 3)・出走・出走頭数 ≥ 5・class_ord ≥ 0 |
| `noise_race` | Bool | 競走中止 (4) か落馬再騎乗 (6) を含むレース (学習時にレースごと除く、design §5) |
| `n_paid_places` | Int8 | paid_places.parquet の n_paid_places (データ層 §3.4)。表が無い・表に無いレースは出走頭数の規則 (≥8 → 3、5-7 → 2、≤4 → 0) |
| `n_paid_source` | Utf8 | paid_places の `source`、規則なら `starters_rule` |
| `in_field_dec`, `field_dec_size`, `field_dec_fallback` | Bool/Int16/Bool | §3 (判断時の出走馬集合。候補・正規化用) |
| `starters` | Int16 | 実際の出走頭数 |

paid_places.parquet が無い間は全レースが `starters_rule` (作成ログに表示)。データ層が作ったら `--stage B` と `--stage labels` を作り直す (field_open も表の値に変わる)。

## 8. 検査

- **レジストリ** (`test_features_registry.py`): 名前の一意・形 (`接頭辞_名前`)・1 グループ 1 接頭辞で重ならない・キー列や元の列や目的変数の列と同じ名前の禁止 (`y_` も)・対象行のレース後の列・診断列・確定オッズ・使わない列・段階ごとに対象行で読める列・ID 列を値として読まない (`keys` だけ)・依存 (存在・段階・循環・inputs)・直接渡したグループにも検査がかかる・この文書の §4.2 の例がそのまま動く。出力の Float32・行・inf は `validate_output`。
- **as-of** (`test_features_asof.py`, `features.asof.check_day`): fixture の 2022 年の 3 日 (取消・競走中止・T-6 の無いレースを含む) で、(a) そのまま、(b) date > D を消し D の行のレース後の列を null、(c) date ≥ D のレース後の列を乱数で並べ替え、(d) D の偶数番のレースの当日変わりうる列 (`STAGE_B_COLUMNS` 全部)・scratched・スナップショットのオッズを別の値にする、の D の行の特徴量が一致 (NaN 同士は一致、1e-6。(d) は段階 A の特徴量を D の全行で、全特徴量を奇数番のレースの行で)。**登録済みの全グループに自動で効く**。scratched は加工の前に作る (ライブでは出馬表の値)。検査自体の検査として、行自身の着順・次走の着順・同じ日の勝ち数 ((b)(c) が検出)、段階 A が `ctx.history()` から当日の馬体重・騎手を読む・段階 B が同じ日の他のレースの馬体重を使う ((d) だけが検出) グループを見つけることも確かめている。
- **ライブとの一致 (P3-lite)** (`test_features_live.py`): fixture を CLI (`--stage all --years 2022`) で作ったストアと、ライブの入口の出力が D の全行・全列で一致: `build_rows(D より前の履歴, D の出馬表の行 (レース後の列 null + scratched), スナップショット)`、朝の出馬表 (騎手・斤量・馬場・天気・ブリンカー・取消を変え、馬体重は未発表) で作った `prepare_morning` の状態に判断時刻の出馬表で `decide`、1 レースずつの `decide`。
- **ストアの管理** (`test_features_live.py`): history が変わった後は段階 A のファイルを使わない (StaleError)、ts_snap/{Y} を作り直した後の `--stage B --groups race_day` で全ての段階 B のグループが新しい枠から作り直される (同じ B/year= の中で `hd_horse_num_ratio` = 馬番 / `rd_field_size`)、labels が古いことを `store_problems` が見つけ `--stage labels` で解消する、コード (依存先のコードも) が変わったファイルを使わない、入力の固定が一時ファイルを残さない、コードの sha256 が import されるモジュール (speed.py・lags.py・schema.py・course_vocab.json・段階 B の decision.py) を含む。
- 実データの P1 (オッズ経路)・P2 (出馬表経路)・P3 (履歴経路) のパリティは netkeiba 層が揃ってから (design §4.3)。

## 9. ライブの入口 (`features/live.py`)

```python
morning = prepare_morning(history_before_d, card_rows_morning)        # 朝 1 回: 段階 A + 全グループの補助出力
rows = decide(morning, card_rows_latest, snapshots, field_open)       # 各レースの判断時刻: 段階 B だけ
rows = build_rows(history_before_d, card_rows, snapshots, field_open) # = decide(prepare_morning(...), ...)
```
- `card_rows`: `netkeiba.convert.convert_card` の行 (HISTORY_DTYPES、レース後の列は null。値があれば例外) + `scratched` (全行必須)。朝は全レース、`decide` には朝の出馬表のレースのうち 1 つ以上を**全馬の行で** (レース内の相対値・頭数がレースの全馬で決まるため。一部の馬だけ・朝に無い行・別の日は例外)。
- `snapshots`: ts_snap 形式 (race_key, horse_num, x, win, place_lo, place_hi)。`field_open`: (race_key, field_open)。None ならスナップショットの代替。
- 履歴に D 以降の行があれば例外 (`predict --race_id` が当日より前の履歴しか使わないことの保証)。
- batch と同じグループ関数を通す (`pipeline.stage_a` / `stage_b`)。朝の状態を判断時刻に使ってよいのは、段階 A と補助出力が D の行の当日変わりうる列に依らないから (as-of (d) と、朝と判断時刻で出馬表が違う場合のパリティ検査が確かめる)。
- 実データ (2024-05-26 と 2013-10-13、全履歴、M2 / 8GB): `prepare_morning` 5.7〜7.4 秒・ピーク RSS 0.75GB、`decide` は 1 レース 約 0.05 秒 (最大 0.07 秒)・1 日分 0.05 秒 (design §11 の「段階 B + 予測 + バスケット計算が 10 秒以内」の段階 B の分)。どちらも batch のストアと全行・全列一致 (朝の出馬表を変えても)。`Morning` は補助出力 (horse_prev・speed_fig、全履歴の行) をメモリに持つ (56〜79MB)。

## 10. CLI と出力の管理

```
.venv/bin/python -m keiba_goal.features.build --stage A                    # 段階 A の全グループ (2003〜2025 年を書く)
.venv/bin/python -m keiba_goal.features.build --stage B                    # 2003-2024 の B/year= (全グループの組み立て)
.venv/bin/python -m keiba_goal.features.build --stage labels
.venv/bin/python -m keiba_goal.features.build --stage all                  # A → B → labels (1 プロセス)
.venv/bin/python -m keiba_goal.features.build --stage A --groups speed,past_runs --years 2022
.venv/bin/python -m keiba_goal.features.build --stage B --groups race_day --allow-missing
```

- `--groups`: 段階 A はそのうちの段階 A のグループだけを作る (1 つも無ければ例外)。段階 B はそのうちの段階 B のグループを必ず作り直し、ほかの段階 B のグループはファイルが古い・無いときだけ作り直す (段階 A の名前だけなら組み立てだけ)。`--groups` 無しはその段階の全グループ。
- 段階ごとに 行数・列数・NaN の多い特徴量 (上位 20)・秒・ピーク RSS を表示する。重い処理なので同時に 1 つだけ動かす。
- 実データ (M2 / 8GB、2026-10-29 08:51、スワップ中): `--stage all` 全体で 40 秒・ピーク RSS 0.99GB。段階 A 15 秒 (horse_static 1.1 / race_static 0.6 / speed 3.1 / aggregates 4.5 / past_runs 6.0 秒、111 万行)、段階 B 0.7〜1.0 秒/年 (初年は補助出力 horse_prev の作成で 2.9 秒)、目的変数 0.1 秒/年。report 2 秒。
- 点検: `.venv/bin/python -m keiba_goal.features.report [--years 2003-2024]` (§10.1 の整合の検査 (問題があれば表示して終了コード 1)、年ごとの行数・学習対象・前走のある行の割合・単勝/複勝の正例率と 1/出走頭数・ノイズレース・field_dec の代替の割合・スピード指数の平均/標準偏差、直近 3 年の NaN の多い列)。
- `data/features/registry.json`: グループ (接頭辞)・特徴量 (元列・keys・段階・種類)・補助出力・段階ごとの使える列・使えない名前。
- `data/state/inputs/` (固定した入力)、`data/state/helpers/{name}.parquet` (補助出力。メタデータに持ち主のグループ・コードの sha256・history の sha256)。

### 10.1 指紋と古いファイル

- 各グループの年ごとのファイルのメタデータ (`features.io.read_meta`): `fingerprint`、`inputs` (段階 A: history の sha256、段階 B: history・その年の ts_snap・paid_places)、`code_sha256`、`deps` (依存グループの指紋)、`features`、行数、作成時刻。
- 指紋 (`features/fingerprint.py`) = sha256(グループ名・列・コードの sha256・inputs・deps)。コードの sha256 はグループのモジュールと、そこから推移的に import される keiba_goal のモジュール (登録・入口のモジュール registry・context・config・groups/__init__ を除く) と、それらがファイル名で読むデータファイル (course_vocab.json) の中身。段階 B は判断時刻の枠 (decision.py) も含む。依存先の指紋を含むので、speed を作り直すと past_runs・aggregates・relative も古くなる。
- B/year= の組み立て: 今の入力・コードから計算した指紋とファイルの指紋を比べる。段階 A が違えば `StaleError` (理由: history が違う / コードが変わった / 依存先 X が変わった / 列が違う、と作り直すコマンド)、段階 B は違えばその場で作り直す (ログに理由)。B/year= のメタデータ: `inputs`、`groups` (グループ → 指紋)、`missing_groups` (`--allow-missing` で外したグループ → 理由)。
- labels/year= のメタデータにも `inputs` (B と同じ 3 つ) と `code_sha256` (labels.py + decision.py)。B か labels の一方だけを作り直すと、もう一方が別の入力のものなら警告する。
- **学習側 (P5) は読む前に `store.store_problems(layout, years, registry の特徴量名, store.current_inputs(layout, years))` が全年で空であることを確かめる** (report と同じ検査): B と labels が同じ入力から作られている、外したグループが無い、B/year= の列 = キー + 今のレジストリの特徴量、今の data/interim の入力 (データ層が作り直していない) と同じ。

## 11. fixture (`tests/fixtures/features/`)

- `history.parquet`: 2022-01-01〜04-30 の中山・中京・阪神の全レースの全行 + それらに出た馬の 04-30 までの全行 (他場・2013 年からの過去年。前走が欠けないように)。61,298 行、HISTORY_DTYPES。
- `ts_snap/2022.parquet`: その 2022 年の行のレースの ts_snap 行 (データ層の列のまま)。63,952 行。
- 作り直し: `.venv/bin/python -m keiba_goal.features.fixture` (data/interim から。作り直すと as-of テストの日付の前提 (取消・T-6 の無いレースを含む) を `test_fixture_days_cover_the_edge_cases` が確かめる)。合計 約 2.2MB。
- 語彙の作り直し: `.venv/bin/python -m keiba_goal.features.vocab` (2003 年以前の history → `features/course_vocab.json`)。

## 12. 既知の制約

- `rs_course` の語彙は 2003 年以前で固定のため、2004 年以降に新設・改修されたコース (2012 年改修後の中京の多く、福島ダ 1150 など) は NaN (2003-2025 の行の 6.7%)。場・芝ダ・距離は別の列にあるので情報は失われない。
- paid_places.parquet (データ層) が無い間: 目的変数の支払着数は出走頭数の規則 (発売開始後の取消があるレース 約 0.07% で実際と違う)、判断時の `field_open` はスナップショットで代替 (発売開始後・T-120 前の取消を数えない)。
- `keys` は「ID 列を値として出力しない」ことを宣言で示すだけで、keys に書いた ID を数に変えて出力することまでは検査できない (レビューで見る)。
- コードの sha256 は import 文から辿るので、import せずに読むファイル (例: 文字列で組み立てたパスのファイル) は入らない。data/schema.py (共有) を変えると全グループのファイルが古くなる (作り直しが必要。値に効かない変更でも)。
- 判断スナップショットの T-6 が無いレースが 2004-2022 年で 24-32%、2023-24 年で 0.2-0.3%、2003 年で 84% (10 月からしかオッズが無い) ある (2026-10-29 08:27 の ts_snap。データ層が発走時刻を取り込み中)。そのレースは field_dec を取消の印で決める (`rd_field_fallback = 1`)。ts_snap や paid_places が作り直されたら `--stage B` と `--stage labels` を作り直す (段階 A は history だけに依る)。
- スピード指数にクラス・斤量・馬場差の補正は無い (design の定義どおり)。1986-1988 年は基準の年が足りず NaN が多い (2003 年以降の特徴量には影響しない)。
- 「各馬の直近 10 走」の状態表は持たず、ライブは朝の `prepare_morning` で全履歴から作る (§9)。
