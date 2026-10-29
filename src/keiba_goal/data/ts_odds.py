"""時系列オッズ JD (単勝・複勝 Lo/Hi) から T-x スナップショットと確定オッズを作る (design §3.2)。

1 ファイル = 1 レース。行 = レースID, 区分, 月日時分, 頭数, 単勝票数, 複勝票数, 馬ごとに (単勝, 複勝Lo, 複勝Hi)。
区分: 1 中間 / 2 前日売最終 / 3 最終 (締切) / 4 確定 (5 確定(月曜), 0 削除, 9 中止)。
スナップショットには最初の区分 3 より前の区分 1・2 の行だけを使う。区分 3・4 は確定値なので特徴量に入れてはいけない。

基準時刻 ref は最初の区分 3 行の時刻 (確定オッズは確定 = 区分 4 行から)。2022 年以降は締切後の区分 1 行が混ざり、
区分 3 の刻印も遅れることがある。そこで締切を 2 通りに推定し、早い方を close_est とする:
- 最初の区分 3 より前で確定票数の 99% に達した最初の行 (締切時に紛れ込んだ区分 4 行を含む)
- 票数の頭打ち (vote_plateau): 確定票数の 80% 以上を持ち、その後の票数の増え方が急に落ちた最初の区分 1・2 行
close_est があれば ref = min(ref, close_est + 2 分)。スナップショットの候補からは、確定票数の 97% 以上を持つ行、
close_est 以降の行、単勝票数 0 の行 (発売直後でオッズが全部 NaN) を捨てる。よって「その馬のオッズが NaN」が欠損の印になる。

発走時刻 (T-x の基準) は発走時刻表 (結果ページ・開催日一覧) の実時刻を正とし、JD の時刻と矛盾しない
([ref − 20 分, ref + 3 分]) ときだけ使う。無ければ post_est.estimate_post の推定値。ts_race に出どころ
post_source ∈ {result, race_list, estimated}・実時刻 post_real・推定値 post_estimate (常に計算) を残す。

DIAGNOSTIC_COLUMNS (vote_share, snap_time, staleness_min) は診断専用で、特徴量の元列にしてはいけない。
vote_share は確定票数で割った値、snap_time と staleness_min は発走時刻 (学習では区分 3 の刻印からの推定を含む)
に対する行の時刻で、ライブ (API の official_datetime の古さ) とは意味が違う。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from keiba_goal.data.post_est import estimate_post
from keiba_goal.data.record import write_parquet_atomic
from keiba_goal.data.schema import NOT_STARTED_CODES

log = logging.getLogger(__name__)

SNAPSHOT_X: tuple[int, ...] = (120, 60, 30, 6)
STALENESS_CAP_MIN: dict[int, float] = {120: 10.0, 60: 10.0, 30: 5.0, 6: 3.0}
VOTE_GUARD = 0.97
CLOSE_SHARE = 0.99
CLOSE_MARGIN_MIN = 2.0
# 締切直前は票数が加速し続けるので、確定票数の 80% 以上を持つ行のあと 3 分以上先の行までの増え方が、直前 3 分以上の
# 増え方の 1/4 未満なら締切後の行 (集計の残りだけが入る) とみなす。実測: 区分 3 の 2 分前と 7 分前に行がある
# 2022-24 の 6,259 レースで、締切後の 2 分前の行の比は 99.8% が < 0.2 (最大 0.33)、締切前の 7 分前の行は 99.9% が
# ≥ 1.1。前売りの厚い G1 の締切前の行は、区分 3 の刻印が遅れて比が薄まっても 2003-24 で ≥ 0.28。
PLATEAU_SHARE = 0.8
PLATEAU_RATE_RATIO = 0.25
PLATEAU_WINDOW_MIN = 3
# 発走時刻表の値は ref − 20 分〜ref + 3 分のときだけ使い、外れたら表の誤りとみなして推定値にする。
# 締切 (発走 − 2 分) は区分 3 と締切推定の時刻以前なので発走 ≤ ref + 2 分。区分 3 の刻印は締切推定から最大 18 分
# 遅れる (2003-24)。同じ場の前後のレースは 25 分以上離れているので、取り違えた発走時刻は弾ける
POST_TIME_WINDOW_MIN = (-20.0, 3.0)
SNAP_KUBUN = (1, 2)
FINAL_KUBUN = 3
CONFIRMED_KUBUN = (4, 5)
DELETED_KUBUN = 0
DIAGNOSTIC_COLUMNS: frozenset[str] = frozenset({"vote_share", "snap_time", "staleness_min"})
SNAP_POST_RACE_COLUMNS = DIAGNOSTIC_COLUMNS   # 旧名
POST_SOURCES: tuple[str, ...] = ("result", "race_list", "estimated")
# 発走時刻表の source 列 → post_source ('netkeiba' は P2 の結果ページ由来)
TABLE_SOURCES: dict[str, str] = {"netkeiba": "result", "result": "result", "race_list": "race_list"}

_EPOCH = date(1970, 1, 1)
_US_PER_MIN = 60_000_000
_NAT = np.iinfo(np.int64).min


class PostTime(NamedTuple):
    """発走時刻表の 1 レース分。minutes は 1970-01-01 00:00 からの分 (実際に走った日の壁時計)。"""

    race_key: int
    minutes: float
    source: str   # 'result' | 'race_list'


@dataclass(frozen=True)
class JdRace:
    """JD ファイル 1 本の中身 (時刻順)。時刻は 1970-01-01 00:00 (JST の壁時計) からの分。"""

    race_key: int
    n_heads: int
    n_rows: int               # ファイルのデータ行数 (削除行を含む)
    kubun: np.ndarray         # (n,) int8
    t_min: np.ndarray         # (n,) int64
    win_votes: np.ndarray     # (n,) int64
    place_votes: np.ndarray   # (n,) int64
    odds: np.ndarray          # (n, n_heads, 3) float32 = 単勝, 複勝Lo, 複勝Hi。0.0 は NaN


def jd_minutes(race_key: int, codes: np.ndarray) -> np.ndarray:
    """月日時分 (MMDDHHMM, 年なし) → 1970-01-01 からの分。

    年は race_key から取る。前日までの発売分は前の日付で入っているので、レース月より大きい月は前年
    (1 月のレースに付いた 12 月の行)。
    """
    race_year, race_month = race_key // 10**12, race_key // 10**10 % 100
    codes = np.asarray(codes, dtype=np.int64)
    month, day = codes // 10**6, codes // 10**4 % 100
    hour, minute = codes // 100 % 100, codes % 100
    if ((hour > 23) | (minute > 59)).any():
        raise ValueError("月日時分の時・分が範囲外です")
    year = np.where(month > race_month, race_year - 1, race_year)
    ymd = year * 10_000 + month * 100 + day
    uniq, inv = np.unique(ymd, return_inverse=True)
    base = np.array(
        [(date(int(u) // 10_000, int(u) // 100 % 100, int(u) % 100) - _EPOCH).days * 1440 for u in uniq],
        dtype=np.int64,
    )
    return base[inv] + hour * 60 + minute


def parse_jd(data: bytes) -> JdRace:
    """JD ファイルの bytes を解析する。ヘッダ (cp932) は読み飛ばし、区分 0 (削除) の行は除く。行は時刻順であること。"""
    rows = [ln.split(b",") for ln in data.splitlines()[1:] if ln.strip()]
    if not rows:
        raise ValueError("データ行がありません")
    n_heads = int(rows[0][3])
    width = 6 + 3 * n_heads
    if any(len(r) != width for r in rows):
        raise ValueError(f"列数が 6 + 3×頭数 = {width} でない行があります")
    cells = np.array(rows)
    meta = cells[:, :6].astype(np.int64)
    race_key = int(meta[0, 0])
    if (meta[:, 0] != race_key).any():
        raise ValueError("レースID が行によって違います")
    if (meta[:, 3] != n_heads).any():
        raise ValueError("頭数が行によって違います")
    odds = cells[:, 6:].astype(np.float32).reshape(len(rows), n_heads, 3)
    odds[~(odds > 0)] = np.nan
    keep = meta[:, 1] != DELETED_KUBUN
    t_min = jd_minutes(race_key, meta[keep, 2])
    if (np.diff(t_min) < 0).any():
        raise ValueError("月日時分が時刻順に並んでいません")
    return JdRace(
        race_key=race_key,
        n_heads=n_heads,
        n_rows=len(rows),
        kubun=meta[keep, 1].astype(np.int8),
        t_min=t_min,
        win_votes=meta[keep, 4],
        place_votes=meta[keep, 5],
        odds=odds[keep],
    )


def jd_file_name(race_key: int) -> str:
    """race_key → JD ファイル名 (場 + 年 2 桁 + 回 + 日 (16 進 1 桁) + R)。"""
    s = f"{race_key:016d}"
    return f"JD{s[8:10]}{s[2:4]}{int(s[10:12]):X}{int(s[12:14]):X}{s[14:16]}.CSV"


@dataclass(frozen=True)
class Candidates:
    """締切の推定とスナップショット候補行 (cand は race の行番号、時刻順)。"""

    i3: int                          # 最初の区分 3 行
    final_votes: int
    close_votes: int | None          # 確定票数の 99% に達した最初の行
    close_plateau: int | None        # 票数の頭打ち (vote_plateau)
    close_est: int | None            # close_votes と close_plateau の早い方
    ref: float
    cand: np.ndarray
    n_dropped_postclose: int         # 97% 以上・close_est 以降で捨てた行
    n_dropped_empty: int             # 単勝票数 0 またはオッズが全部 NaN で捨てた行


@dataclass(frozen=True)
class RaceSnapshots:
    """1 レースの抽出結果。時刻は 1970-01-01 からの分 (ref / post_* は小数可)。"""

    race_key: int
    n_heads: int
    n_rows: int
    n_k3: int
    first_k3: int | None
    k4: int | None
    close_est: int | None
    close_votes: int | None
    close_plateau: int | None
    ref: float | None
    post_est: float | None           # T-x の基準にした発走時刻 (post_real が無ければ post_estimate)
    post_time: float | None          # 発走時刻表の値 (採否によらず。無ければ None)
    post_real: float | None          # 採用した実時刻 (無い・JD と矛盾して不採用なら None)
    post_estimate: float | None      # estimate_post の値 (実時刻があっても計算する)
    post_source: str | None          # POST_SOURCES (区分 3 の無いレースは None)
    final_win_votes: int | None
    final_place_votes: int | None
    field_open: int | None           # 候補行のどれかで単勝オッズが付いた馬の数 (発売開始時の頭数)
    n_dropped_postclose: int
    n_dropped_empty: int
    snap_odds: np.ndarray | None     # (len(SNAPSHOT_X), n_heads, 3)、欠損は NaN
    snap_time: np.ndarray | None     # (len(SNAPSHOT_X),) int64 分、欠損は −1
    staleness: np.ndarray | None     # (len(SNAPSHOT_X),) float32 分、欠損は NaN
    vote_share: np.ndarray | None    # (len(SNAPSHOT_X),) float32
    final_odds: np.ndarray | None    # (n_heads, 3)

    @property
    def has_k3(self) -> bool:
        return self.first_k3 is not None

    @property
    def post_actual(self) -> bool:
        return self.post_real is not None


def _last_row_per_minute(idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """同じ時刻の行が複数あれば後の行だけ残す (時刻順)。"""
    if idx.size == 0:
        return idx
    idx = idx[np.argsort(t[idx], kind="stable")]
    ts = t[idx]
    return idx[np.append(ts[1:] != ts[:-1], True)]


def vote_plateau(race: JdRace, cand: np.ndarray, final_votes: int) -> int | None:
    """票数の頭打ちで見た締切の時刻。cand は時刻順の区分 1・2 の行。頭打ちの行が無ければ None。

    頭打ちの行 = 単勝票数が確定の PLATEAU_SHARE 以上で、次の 2 つの増え方の比が PLATEAU_RATE_RATIO 未満の最初の行:
    その行 → PLATEAU_WINDOW_MIN 分以上先の最初の行 (区分を問わない) / PLATEAU_WINDOW_MIN 分以上前の最後の cand → その行。
    """
    t, v = race.t_min, race.win_votes
    tc = t[cand]
    for i in cand[v[cand] >= PLATEAU_SHARE * final_votes]:
        before = int(np.searchsorted(tc, t[i] - PLATEAU_WINDOW_MIN, side="right")) - 1
        after = int(np.searchsorted(t, t[i] + PLATEAU_WINDOW_MIN, side="left"))
        if before < 0 or after >= t.size:
            continue
        p = int(cand[before])
        rate_before = (v[i] - v[p]) / (t[i] - t[p])
        rate_after = (v[after] - v[i]) / (t[after] - t[i])
        if rate_before > 0 and rate_after < PLATEAU_RATE_RATIO * rate_before:
            return int(t[i])
    return None


def find_candidates(race: JdRace) -> Candidates | None:
    """締切を推定し、スナップショット候補行を選ぶ。区分 3 の行が無ければ None。

    候補行 = 最初の区分 3 より前の区分 1・2 の行 (同じ時刻は後の行) のうち、単勝票数 > 0 で単勝オッズが 1 頭以上あり、
    単勝票数 < 確定票数の 97%、close_est より前の行。
    """
    k, t, v = race.kubun, race.t_min, race.win_votes
    k3 = np.flatnonzero(k == FINAL_KUBUN)
    if k3.size == 0:
        return None
    i3 = int(k3[0])
    final_votes = int(v[i3])
    rows = _last_row_per_minute(np.flatnonzero(np.isin(k[:i3], SNAP_KUBUN)), t)
    close_votes = close_plateau = None
    if final_votes > 0:
        reached = np.flatnonzero(v[:i3] >= CLOSE_SHARE * final_votes)
        close_votes = int(t[reached[0]]) if reached.size else None
        close_plateau = vote_plateau(race, rows, final_votes)
    found = [c for c in (close_votes, close_plateau) if c is not None]
    close_est = min(found) if found else None
    first_k3 = float(t[i3])
    ref = first_k3 if close_est is None else min(first_k3, close_est + CLOSE_MARGIN_MIN)

    has_votes = (v[rows] > 0) & np.isfinite(race.odds[rows, :, 0]).any(axis=1)
    before_close = v[rows] < VOTE_GUARD * final_votes
    if close_est is not None:
        before_close &= t[rows] < close_est
    return Candidates(
        i3=i3,
        final_votes=final_votes,
        close_votes=close_votes,
        close_plateau=close_plateau,
        close_est=close_est,
        ref=ref,
        cand=rows[has_votes & before_close],
        n_dropped_postclose=int((has_votes & ~before_close).sum()),
        n_dropped_empty=int((~has_votes).sum()),
    )


def pick_snapshots(cand_t: np.ndarray, post: float) -> tuple[np.ndarray, np.ndarray]:
    """x ごとに、時刻 ≤ post − x の最後の候補行の位置 (cand_t の添字、無い・鮮度上限超えは −1) と鮮度 (分)。"""
    pos = np.full(len(SNAPSHOT_X), -1, dtype=np.int64)
    lag = np.full(len(SNAPSHOT_X), np.nan, dtype=np.float32)
    for j, x in enumerate(SNAPSHOT_X):
        target = post - x
        p = int(np.searchsorted(cand_t, target, side="right")) - 1
        if p >= 0 and target - float(cand_t[p]) <= STALENESS_CAP_MIN[x]:
            pos[j] = p
            lag[j] = target - float(cand_t[p])
    return pos, lag


def choose_post(ref: float, post: PostTime | None) -> tuple[float | None, str]:
    """(採用する実時刻, post_source)。実時刻は [ref − 20 分, ref + 3 分] に入るときだけ使う。"""
    lo, hi = POST_TIME_WINDOW_MIN
    if post is not None and ref + lo <= post.minutes <= ref + hi:
        return post.minutes, post.source
    return None, "estimated"


def extract_snapshots(race: JdRace, post: PostTime | None = None, after_g1: bool = False) -> RaceSnapshots:
    """T-x スナップショット・確定オッズ・基準時刻を取り出す。

    基準時刻と確定票数は最初の区分 3 行、確定オッズは最初の区分 3 より後の確定 (区分 4) 行 (無ければ最後の区分 3 行)。
    T-x = 候補行 (find_candidates) のうち時刻 ≤ 発走 − x の最後の行。鮮度 (発走 − x) − 時刻 が STALENESS_CAP_MIN を
    超えたら欠損。発走は採用できる実時刻 (choose_post)、無ければ estimate_post の推定値 (after_g1 はその入力)。
    """
    if post is not None and post.race_key != race.race_key:
        raise ValueError(f"発走時刻表のレース {post.race_key} と JD のレース {race.race_key} が違います")
    c = find_candidates(race)
    post_table = None if post is None else post.minutes
    if c is None:
        return RaceSnapshots(
            race_key=race.race_key, n_heads=race.n_heads, n_rows=race.n_rows, n_k3=0, first_k3=None, k4=None,
            close_est=None, close_votes=None, close_plateau=None, ref=None, post_est=None, post_time=post_table,
            post_real=None, post_estimate=None, post_source=None, final_win_votes=None, final_place_votes=None,
            field_open=None, n_dropped_postclose=0, n_dropped_empty=0,
            snap_odds=None, snap_time=None, staleness=None, vote_share=None, final_odds=None,
        )
    k, t, v = race.kubun, race.t_min, race.win_votes
    k3 = np.flatnonzero(k == FINAL_KUBUN)
    confirmed = np.flatnonzero(np.isin(k, CONFIRMED_KUBUN) & (np.arange(k.size) > c.i3))
    # 確定オッズは確定 (区分4) 行から。区分3 が 2 行あるファイルは締切後の競走除外で 2 行目以降が再計算されており、
    # 払戻 (= record の win_odds) は再計算後の値
    i_final = int(confirmed[0]) if confirmed.size else int(k3[-1])
    tc = t[c.cand]
    estimate = estimate_post(c.ref, tc, after_g1)
    post_real, source = choose_post(c.ref, post)
    post_used = estimate if post_real is None else post_real
    pos, lag = pick_snapshots(tc, post_used)

    n_x = len(SNAPSHOT_X)
    snap_odds = np.full((n_x, race.n_heads, 3), np.nan, dtype=np.float32)
    snap_time = np.full(n_x, -1, dtype=np.int64)
    share = np.full(n_x, np.nan, dtype=np.float32)
    for j in np.flatnonzero(pos >= 0):
        i = int(c.cand[pos[j]])
        snap_odds[j] = race.odds[i]
        snap_time[j] = t[i]
        share[j] = v[i] / c.final_votes
    priced = np.isfinite(race.odds[c.cand, :, 0]).any(axis=0)
    return RaceSnapshots(
        race_key=race.race_key,
        n_heads=race.n_heads,
        n_rows=race.n_rows,
        n_k3=int(k3.size),
        first_k3=int(t[c.i3]),
        k4=int(t[confirmed[0]]) if confirmed.size else None,
        close_est=c.close_est,
        close_votes=c.close_votes,
        close_plateau=c.close_plateau,
        ref=c.ref,
        post_est=post_used,
        post_time=post_table,
        post_real=post_real,
        post_estimate=estimate,
        post_source=source,
        final_win_votes=c.final_votes,
        final_place_votes=int(race.place_votes[c.i3]),
        field_open=int(priced.sum()) if c.cand.size else None,
        n_dropped_postclose=c.n_dropped_postclose,
        n_dropped_empty=c.n_dropped_empty,
        snap_odds=snap_odds,
        snap_time=snap_time,
        staleness=lag,
        vote_share=share,
        final_odds=race.odds[i_final].copy(),
    )


def _minutes_to_datetime(values: Iterable[float | None]) -> pl.Series:
    us = [None if v is None else round(v * _US_PER_MIN) for v in values]
    return pl.Series(us, dtype=pl.Int64).cast(pl.Datetime("us"))


def snapshot_frames(results: list[RaceSnapshots]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(ts_snap, ts_race, ts_final)。ts_snap は区分 3 のあるレースの全馬 × 全 x (欠損は NaN、snap_time は null)。"""
    with_k3 = [r for r in results if r.has_k3]
    n_x = len(SNAPSHOT_X)
    xs = np.array(SNAPSHOT_X, dtype=np.int16)

    def cat(parts: list[np.ndarray], dtype: object, width: int | None = None) -> np.ndarray:
        if parts:
            return np.concatenate(parts)
        return np.empty((0, width) if width else 0, dtype=dtype)

    # 1 レースの行の並び: x ごとに馬番 1..頭数 (snap_odds の (x, 馬, 3) をそのまま平らにした順)
    snap_t = cat([np.repeat(r.snap_time, r.n_heads) for r in with_k3], np.int64)
    odds = cat([r.snap_odds.reshape(-1, 3) for r in with_k3], np.float32, 3)
    final = cat([r.final_odds for r in with_k3], np.float32, 3)
    snap = pl.DataFrame({
        "race_key": pl.Series(cat([np.full(n_x * r.n_heads, r.race_key) for r in with_k3], np.int64), dtype=pl.Int64),
        "horse_num": pl.Series(cat([np.tile(np.arange(1, r.n_heads + 1), n_x) for r in with_k3], np.int64), dtype=pl.Int8),
        "x": pl.Series(cat([np.repeat(xs, r.n_heads) for r in with_k3], np.int16), dtype=pl.Int16),
        "win": pl.Series(odds[:, 0], dtype=pl.Float32),
        "place_lo": pl.Series(odds[:, 1], dtype=pl.Float32),
        "place_hi": pl.Series(odds[:, 2], dtype=pl.Float32),
        "staleness_min": pl.Series(cat([np.repeat(r.staleness, r.n_heads) for r in with_k3], np.float32), dtype=pl.Float32),
        "vote_share": pl.Series(cat([np.repeat(r.vote_share, r.n_heads) for r in with_k3], np.float32), dtype=pl.Float32),
        "snap_time": pl.Series(
            np.where(snap_t >= 0, snap_t, _NAT).view("datetime64[m]").astype("datetime64[us]"), dtype=pl.Datetime("us")
        ),
    })
    race = pl.DataFrame({
        "race_key": pl.Series([r.race_key for r in results], dtype=pl.Int64),
        "n_heads": pl.Series([r.n_heads for r in results], dtype=pl.Int8),
        "first_k3_time": _minutes_to_datetime(r.first_k3 for r in results),
        "k4_time": _minutes_to_datetime(r.k4 for r in results),
        "close_est": _minutes_to_datetime(r.close_est for r in results),
        "close_votes": _minutes_to_datetime(r.close_votes for r in results),
        "close_plateau": _minutes_to_datetime(r.close_plateau for r in results),
        "ref_time": _minutes_to_datetime(r.ref for r in results),
        "post_est": _minutes_to_datetime(r.post_est for r in results),
        "post_source": pl.Series([r.post_source for r in results], dtype=pl.Utf8),
        "post_real": _minutes_to_datetime(r.post_real for r in results),
        "post_estimate": _minutes_to_datetime(r.post_estimate for r in results),
        "post_time": _minutes_to_datetime(r.post_time for r in results),
        "post_actual": pl.Series([r.post_actual for r in results], dtype=pl.Boolean),
        "field_open": pl.Series([r.field_open for r in results], dtype=pl.Int16),
        "final_win_votes": pl.Series([r.final_win_votes for r in results], dtype=pl.Int64),
        "final_place_votes": pl.Series([r.final_place_votes for r in results], dtype=pl.Int64),
        "has_k3": pl.Series([r.has_k3 for r in results], dtype=pl.Boolean),
        "n_rows": pl.Series([r.n_rows for r in results], dtype=pl.Int32),
        "n_dropped_postclose": pl.Series([r.n_dropped_postclose for r in results], dtype=pl.Int32),
        "n_dropped_empty": pl.Series([r.n_dropped_empty for r in results], dtype=pl.Int32),
        "n_k3": pl.Series([r.n_k3 for r in results], dtype=pl.Int8),
    })
    final_df = pl.DataFrame({
        "race_key": pl.Series(cat([np.full(r.n_heads, r.race_key) for r in with_k3], np.int64), dtype=pl.Int64),
        "horse_num": pl.Series(cat([np.arange(1, r.n_heads + 1) for r in with_k3], np.int64), dtype=pl.Int8),
        "win_final": pl.Series(final[:, 0], dtype=pl.Float32),
        "place_lo_final": pl.Series(final[:, 1], dtype=pl.Float32),
        "place_hi_final": pl.Series(final[:, 2], dtype=pl.Float32),
    })
    return (
        snap.sort(["race_key", "x", "horse_num"]),
        race.sort("race_key"),
        final_df.sort(["race_key", "horse_num"]),
    )


def load_post_times(path: Path) -> dict[int, PostTime]:
    """発走時刻表 → {race_key: PostTime}。無ければ空。

    必要な列: race_key, source ('netkeiba'/'result' = 結果ページ, 'race_list' = 開催日一覧) と、post_datetime
    (実際に走った日の日時。順延したレースは race_key と日付が違う) か post_time ('HH:MM' または Time、日付は race_key)。
    post_datetime があればそれを使う。発走時刻が null の行は使わない。
    """
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    missing = [c for c in ("race_key", "source") if c not in df.columns]
    if missing or not ({"post_datetime", "post_time"} & set(df.columns)):
        raise ValueError(f"{path.name}: race_key, source と post_datetime か post_time の列が要ります (列: {df.columns})")
    if df["race_key"].n_unique() != df.height:
        raise ValueError(f"{path.name}: race_key が重複しています")
    unknown = sorted(set(df["source"].drop_nulls().unique()) - set(TABLE_SOURCES))
    if unknown or df["source"].null_count():
        raise ValueError(f"{path.name}: source が {sorted(TABLE_SOURCES)} でない行があります {unknown}")
    if "post_datetime" in df.columns:
        if not isinstance(df.schema["post_datetime"], pl.Datetime) or df.schema["post_datetime"].time_zone is not None:
            raise ValueError(f"{path.name}: post_datetime はタイムゾーン無しの Datetime にしてください ({df.schema['post_datetime']})")
        minutes = pl.col("post_datetime").cast(pl.Datetime("us")).cast(pl.Int64) / _US_PER_MIN
    else:
        pt = pl.col("post_time")
        if df.schema["post_time"] == pl.Utf8:
            pt = pt.str.strip_chars().str.to_time("%H:%M", strict=True)
        elif df.schema["post_time"] != pl.Time:
            raise ValueError(f"{path.name}: post_time は 'HH:MM' 文字列か Time 型にしてください ({df.schema['post_time']})")
        day = (pl.col("race_key") // 10**8).cast(pl.Utf8).str.to_date("%Y%m%d", strict=True).cast(pl.Int64)
        minutes = day * 1440 + pt.dt.hour().cast(pl.Int64) * 60 + pt.dt.minute().cast(pl.Int64)
    out = df.select(pl.col("race_key"), minutes.cast(pl.Float64).alias("m"), pl.col("source").replace_strict(TABLE_SOURCES))
    n_null = out["m"].null_count()
    if n_null:
        log.warning("%s: 発走時刻が null の %d 行は使いません", path.name, n_null)
    return {int(k): PostTime(int(k), float(m), s) for k, m, s in out.drop_nulls("m").iter_rows()}


def jd_files_by_year(ts_dir: Path) -> dict[int, list[Path]]:
    """JD ファイルを年 (ファイル名の 2 桁年 → 西暦) ごとに。"""
    out: dict[int, list[Path]] = {}
    for name in sorted(os.listdir(ts_dir)):
        if name.startswith("JD") and name.upper().endswith(".CSV") and len(name) == 14:
            out.setdefault(2000 + int(name[4:6]), []).append(ts_dir / name)
    return out


def extract_file(path: str, post: PostTime | None = None, after_g1: bool = False) -> RaceSnapshots:
    """ファイル 1 本 → RaceSnapshots。ファイル名と中のレースID が一致しなければ例外。"""
    name = Path(path).name
    try:
        race = parse_jd(Path(path).read_bytes())
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from e
    if jd_file_name(race.race_key) != name.upper():
        raise ValueError(f"{name}: 中のレースID {race.race_key} とファイル名が一致しません")
    return extract_snapshots(race, post, after_g1)


FileArgs = tuple[str, PostTime | None, bool]


@contextmanager
def file_mapper(processes: int) -> Iterator[Callable[[list[FileArgs]], list[RaceSnapshots]]]:
    """(ファイル, 発走時刻, G1 後のレースか) の並列抽出器。processes <= 1 ならこのプロセスで順に処理する。"""
    if processes <= 1:
        yield lambda args: [extract_file(*a) for a in args]
        return
    with Pool(processes) as pool:
        yield lambda args: pool.starmap(extract_file, args, chunksize=64)


def _minutes(a: pl.Series, b: pl.Series) -> np.ndarray:
    return ((a - b).dt.total_seconds() / 60).to_numpy()


def summarize_year(
    year: int, snap: pl.DataFrame, race: pl.DataFrame, final: pl.DataFrame, records: pl.DataFrame | None
) -> dict[str, object]:
    """検証用の数値: x ごとの取得率、選んだスナップショットの票数比、確定単勝 = record win_odds の一致率、
    発走時刻の出どころ、実時刻のあるレースの (区分 3 − 発走)・(ref − 発走)・(推定 − 実時刻) の分布など。"""
    k3 = race.filter(pl.col("has_k3"))
    # 頭打ちの規則が無かった場合の ref (区分 3 と 99% の行だけ)
    ref_votes = pl.min_horizontal("first_k3_time", pl.col("close_votes") + pl.duration(minutes=CLOSE_MARGIN_MIN))
    out: dict[str, object] = {
        "year": year,
        "files": race.height,
        "races_k3": k3.height,
        "no_k3": race.height - k3.height,
        "two_k3": k3.filter(pl.col("n_k3") > 1).height,
        "ref_moved": k3.filter(pl.col("ref_time") < pl.col("first_k3_time")).height,
        "ref_moved_plateau": k3.filter(pl.col("ref_time") < ref_votes).height,
        "dropped_rows": int(race["n_dropped_postclose"].sum()),
        "dropped_empty": int(race["n_dropped_empty"].sum()),
        "races_with_dropped": race.filter(pl.col("n_dropped_postclose") > 0).height,
        "post_table": k3.filter(pl.col("post_time").is_not_null()).height,
        "post_actual": k3.filter(pl.col("post_actual")).height,
    }
    out["post_rejected"] = out["post_table"] - out["post_actual"]
    for s in POST_SOURCES:
        out[f"post_{s}"] = k3.filter(pl.col("post_source") == s).height
    per_race = snap.group_by(["x", "race_key"]).agg(
        pl.col("snap_time").first().is_not_null().alias("ok"),
        pl.col("vote_share").first().alias("share"),
        pl.col("win").is_finite().any().alias("priced"),
    )
    got = dict(per_race.group_by("x").agg(pl.col("ok").mean()).iter_rows())
    for x in SNAPSHOT_X:
        out[f"avail_T{x}"] = float(got.get(x, 0.0))
    out["snap_unpriced"] = per_race.filter(pl.col("ok") != pl.col("priced")).height
    shares = per_race["share"].drop_nans().drop_nulls()
    out["max_vote_share"] = float(shares.max()) if shares.len() else float("nan")
    out["min_vote_share"] = float(shares.min()) if shares.len() else float("nan")
    t6 = per_race.filter((pl.col("x") == 6) & pl.col("ok"))["share"]
    out["t6_share_ge90"] = int((t6 >= 0.9).sum())
    timed = k3.filter(pl.col("post_actual"))
    if timed.height:
        k3_lag = _minutes(timed["first_k3_time"], timed["post_real"])
        ref_lag = _minutes(timed["ref_time"], timed["post_real"])
        est_err = _minutes(timed["post_estimate"], timed["post_real"])
        out["k3_minus_post_med"] = float(np.median(k3_lag))
        out["ref_minus_post_p05"], out["ref_minus_post_med"], out["ref_minus_post_p95"] = (
            float(q) for q in np.quantile(ref_lag, [0.05, 0.5, 0.95])
        )
        out["est_eq_real"] = float(np.mean(est_err == 0))
        out["est_lt_real"] = float(np.mean(est_err < 0))
        out["est_gt_real"] = int(np.sum(est_err > 0))
    if records is not None:
        starters = records.filter(~pl.col("error_code").is_in(sorted(NOT_STARTED_CODES)))
        joined = starters.join(final, on=["race_key", "horse_num"], how="inner")
        out["record_races"] = records["race_key"].n_unique()
        out["record_races_with_final"] = joined["race_key"].n_unique()
        out["starters_with_final"] = joined.height
        same = (joined["win_final"] - joined["win_odds"]).abs() < 1e-3
        out["final_eq_record"] = float(same.fill_null(False).mean()) if joined.height else float("nan")
    return out


def build_ts(
    ts_dir: Path,
    out_dir: Path,
    years: Iterable[int],
    *,
    post_times: Mapping[int, PostTime] | None = None,
    after_g1: frozenset[int] = frozenset(),
    processes: int = 8,
    records_path: Path | None = None,
) -> list[dict[str, object]]:
    """年ごとに JD を全部読み、out_dir/{ts_snap,ts_race,ts_final}/{year}.parquet を書く。年ごとの検証値を返す。

    post_times: 発走時刻表 (load_post_times)。after_g1: G1 後のレース (post_est.races_after_g1、推定の入力)。
    """
    by_year = jd_files_by_year(ts_dir)
    by_file = {jd_file_name(k): p for k, p in (post_times or {}).items()}
    g1_files = {jd_file_name(k) for k in after_g1}
    summaries: list[dict[str, object]] = []
    with file_mapper(processes) as extract_all:
        for year in years:
            paths = by_year.get(year, [])
            if not paths:
                log.warning("%d 年の JD ファイルがありません", year)
                continue
            args = [(str(p), by_file.get(p.name.upper()), p.name.upper() in g1_files) for p in paths]
            snap, race, final = snapshot_frames(extract_all(args))
            write_parquet_atomic(snap, out_dir / "ts_snap" / f"{year}.parquet")
            write_parquet_atomic(race, out_dir / "ts_race" / f"{year}.parquet")
            write_parquet_atomic(final, out_dir / "ts_final" / f"{year}.parquet")
            records = None
            if records_path is not None and records_path.exists():
                records = (
                    pl.scan_parquet(records_path)
                    .filter((pl.col("date") // 10_000) == year)
                    .select("race_key", "horse_num", "error_code", "win_odds")
                    .collect()
                )
            summary = summarize_year(year, snap, race, final, records)
            if summary["post_rejected"]:
                rejected = race.filter(pl.col("has_k3") & pl.col("post_time").is_not_null() & ~pl.col("post_actual"))
                log.warning(
                    "%d 年: 発走時刻表の %d レースが JD の時刻 (ref %+g〜%+g 分) と合わないので推定値を使いました (例: %s)",
                    year, summary["post_rejected"], *POST_TIME_WINDOW_MIN, rejected["race_key"].head(5).to_list(),
                )
            summaries.append(summary)
            del snap, race, final, records
    return summaries
