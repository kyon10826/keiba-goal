"""成績データ (record_data 47 列) と、正規化後の履歴表 `history` の列定義。

record_data の 47 列は公式仕様 (データ仕様書 xlsx「成績データ」シート) の順。
netkeiba から補完した行も、変換器 (netkeiba/convert.py) で同じ 47 列 + 派生列にそろえる。
"""

from __future__ import annotations

import polars as pl

# record_data の列 (ファイル上の順番どおり)。日本語は仕様書の項目名。
RECORD_COLUMNS: tuple[str, ...] = (
    "race_id",            # レースID (18 桁)
    "year",               # 年 (2 桁。使わない: date は race_id から作る)
    "month",              # 月
    "day",                # 日
    "times",              # 回次
    "place",              # 場所 (漢字)
    "daily",              # 日次 (1-9, A/B/C = 10/11/12)
    "race_num",           # レース番号
    "horse",              # 馬名
    "jockey_id",          # 騎手コード (5 桁ゼロ埋め文字列として扱う)
    "horse_N",            # 頭数 (取消を含む登録頭数)
    "waku_num",           # 枠番
    "horse_num",          # 馬番
    "class_code",         # クラスコード (TARGET 式)
    "track_code",         # トラックコード (JV)
    "corner_num",         # コーナー回数
    "dist",               # 距離
    "state",              # 馬場状態 (良/稍/重/不)
    "weather",            # 天候
    "age_code",           # 競走種別コード
    "sex",                # 性別
    "age",                # 年齢
    "basis_weight",       # 斤量
    "blinker",            # ブリンカー ('B' or 空)
    "weight",             # 馬体重
    "inc_dec",            # 増減
    "weight_code",        # 重量種別
    "win_odds",           # 確定単勝オッズ (レース後の値)
    "rank",               # 確定着順 (0 = 着外/取消等)
    "time_diff",          # 着差タイム (秒)
    "time",               # 走破タイム (秒)
    "corner1_rank",
    "corner2_rank",
    "corner3_rank",
    "corner4_rank",
    "last_3F_time",       # 上がり 3F
    "last_3F_rank",
    "Ave_3F",
    "PCI",
    "last_3F_time_diff",  # -3F 差
    "leg",                # 脚質 (TARGET 式。再現できないので特徴量に使わない)
    "pop",                # 確定人気 (レース後の値)
    "prize",              # 本賞金 (万円)
    "error_code",         # 異常区分 (JV)
    "father",
    "mother",
    "id",                 # 血統登録番号 (10 桁)
)
assert len(RECORD_COLUMNS) == 47

# 正規化後の型。CSV はすべて文字列で読み、strip してからこの型に変換する。
RECORD_DTYPES: dict[str, pl.DataType] = {
    "race_id": pl.Int64,
    "year": pl.Int16,
    "month": pl.Int8,
    "day": pl.Int8,
    "times": pl.Int8,
    "place": pl.Utf8,
    "daily": pl.Int8,            # A/B/C は 10/11/12 に変換してから
    "race_num": pl.Int8,
    "horse": pl.Utf8,
    "jockey_id": pl.Utf8,        # 先頭ゼロを保つ
    "horse_N": pl.Int16,
    "waku_num": pl.Int8,
    "horse_num": pl.Int8,
    "class_code": pl.Int16,
    "track_code": pl.Int16,
    "corner_num": pl.Int8,
    "dist": pl.Int16,
    "state": pl.Utf8,
    "weather": pl.Utf8,
    "age_code": pl.Int16,
    "sex": pl.Utf8,
    "age": pl.Int8,
    "basis_weight": pl.Float32,
    "blinker": pl.Utf8,
    "weight": pl.Float32,        # 欠損あり
    "inc_dec": pl.Float32,       # '+6' / '-2' / '0'、欠損あり
    "weight_code": pl.Int8,
    "win_odds": pl.Float32,
    "rank": pl.Int16,
    "time_diff": pl.Float32,     # '----' は欠損
    "time": pl.Float32,
    "corner1_rank": pl.Int16,
    "corner2_rank": pl.Int16,
    "corner3_rank": pl.Int16,
    "corner4_rank": pl.Int16,
    "last_3F_time": pl.Float32,
    "last_3F_rank": pl.Int16,
    "Ave_3F": pl.Float32,
    "PCI": pl.Float32,
    "last_3F_time_diff": pl.Float32,
    "leg": pl.Utf8,
    "pop": pl.Int16,
    "prize": pl.Float32,
    "error_code": pl.Int8,
    "father": pl.Utf8,
    "mother": pl.Utf8,
    "id": pl.Int64,
}

# history 表で record の 47 列に加える派生列 (P1 の loader と P2 の変換器の両方が作る)
DERIVED_DTYPES: dict[str, pl.DataType] = {
    "date": pl.Int32,            # YYYYMMDD (race_id 先頭 8 桁)
    "race_key": pl.Int64,        # race_id // 100 (16 桁, 時系列オッズのレース ID)
    "nk_race_id": pl.Utf8,       # netkeiba 12 桁 = YYYY + 場 + 回 + 日 + R
    "venue_code": pl.Int8,       # 01-10
    "surface": pl.Int8,          # 1 芝, 2 ダート, 3 障害
    "direction": pl.Int8,        # 1 右, 2 左, 3 直線
    "course_inout": pl.Int8,     # 0 指定なし, 1 内, 2 外, 3 内→外, 4 外→内, 5 内2周, 6 外2周
    "class_ord": pl.Int8,        # クラス序数 (下の CLASS_ORD)。-1 = 学習対象外のクラス
    "starters": pl.Int16,        # 実際の出走頭数 (error_code ∉ {1,2,3}) — ラベルと払戻規則専用
    "trainer_id": pl.Utf8,       # 調教師 ID (netkeiba 由来の行のみ。record は null)
    "source": pl.Utf8,           # 'record' | 'netkeiba'
}

HISTORY_DTYPES: dict[str, pl.DataType] = {**RECORD_DTYPES, **DERIVED_DTYPES}

# 発走前に分かる列 (特徴量の元にしてよい)。それ以外はレース後の情報で、過去走としてだけ使う。
PRE_RACE_COLUMNS: frozenset[str] = frozenset({
    "race_id", "year", "month", "day", "times", "place", "daily", "race_num", "horse", "jockey_id",
    "horse_N", "waku_num", "horse_num", "class_code", "track_code", "corner_num", "dist", "state",
    "weather", "age_code", "sex", "age", "basis_weight", "blinker", "weight", "inc_dec", "weight_code",
    "father", "mother", "id",
    "date", "race_key", "nk_race_id", "venue_code", "surface", "direction", "course_inout", "class_ord",
    "trainer_id", "source",
})
POST_RACE_COLUMNS: frozenset[str] = frozenset(set(RECORD_COLUMNS) - (PRE_RACE_COLUMNS & set(RECORD_COLUMNS))) | {"starters"}

PLACE_CODES: dict[str, int] = {
    "札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5,
    "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10,
}
CODE_TO_PLACE: dict[int, str] = {v: k for k, v in PLACE_CODES.items()}

# JV 異常区分
ERROR_CODES: dict[int, str] = {
    0: "正常", 1: "出走取消", 2: "発走除外", 3: "競走除外", 4: "競走中止", 5: "失格", 6: "落馬再騎乗", 7: "降着",
}
NOT_STARTED_CODES: frozenset[int] = frozenset({1, 2, 3})   # 出走していない (返還対象)
NOISE_RACE_CODES: frozenset[int] = frozenset({4, 6})       # 学習時にレースごと除く (design §5)

# TARGET 式 class_code → 序数 (design §3.1)。-1 は学習対象外 (アラブ系・名称不明クラス)。
CLASS_ORD: dict[int, int] = {
    15: 0, 11: 0, 7: 0,          # 新馬 / 未出走 / 未勝利
    19: 1, 23: 1,                # 400万下 / 500万下・1勝
    39: 2, 43: 2,                # 900万下 / 1000万下・2勝
    63: 3, 67: 3,                # 1500万下 / 1600万下・3勝
    115: 4, 131: 4,              # L / OP
    147: 5, 163: 5,              # 重賞 (格付けなし) / G3
    179: 6,                      # G2
    195: 7,                      # G1
    27: -1, 31: -1, 35: -1, 59: -1,
}

# JV トラックコード → (surface, direction, course_inout)
TRACK_CODE: dict[int, tuple[int, int, int]] = {
    10: (1, 3, 0),   # 芝・直線
    11: (1, 2, 0),   # 芝・左
    12: (1, 2, 2),   # 芝・左・外
    13: (1, 2, 3),   # 芝・左・内→外
    14: (1, 2, 4),   # 芝・左・外→内
    15: (1, 2, 5),   # 芝・左・内2周
    16: (1, 2, 6),   # 芝・左・外2周
    17: (1, 1, 0),   # 芝・右
    18: (1, 1, 2),   # 芝・右・外
    19: (1, 1, 3),   # 芝・右・内→外
    20: (1, 1, 4),   # 芝・右・外→内
    21: (1, 1, 5),   # 芝・右・内2周
    22: (1, 1, 6),   # 芝・右・外2周
    23: (2, 2, 0),   # ダート・左
    24: (2, 1, 0),   # ダート・右
    25: (2, 2, 1),   # ダート・左・内
    26: (2, 1, 2),   # ダート・右・外
    27: (2, 2, 0),   # サンド・左
    28: (2, 1, 0),   # サンド・右
    29: (2, 3, 0),   # ダート・直線
    **{c: (3, 0, 0) for c in range(51, 60)},  # 障害
}

AGE_CODES: dict[int, str] = {
    11: "2歳", 12: "3歳", 13: "3歳以上", 14: "4歳以上", 18: "障害3歳以上", 19: "障害4歳以上",
    21: "アラブ2歳", 22: "アラブ3歳", 23: "アラブ3歳以上",
}
WEIGHT_CODES: dict[int, str] = {0: "不明", 1: "ハンデ", 2: "別定", 3: "馬齢", 4: "定量"}


def nk_race_id_from_race_id(race_id: int) -> str:
    """18 桁 race_id → netkeiba 12 桁 (YYYY + 場 + 回 + 日 + R)。"""
    s = f"{race_id:018d}"
    return s[0:4] + s[8:16]


def race_key_from_race_id(race_id: int) -> int:
    return race_id // 100
