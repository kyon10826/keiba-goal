import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import keiba_goal.data.ts_odds as ts_odds
from keiba_goal.data.record import build_records
from keiba_goal.data.ts_odds import (
    DIAGNOSTIC_COLUMNS,
    SNAP_POST_RACE_COLUMNS,
    SNAPSHOT_X,
    STALENESS_CAP_MIN,
    PostTime,
    build_ts,
    extract_file,
    extract_snapshots,
    jd_file_name,
    jd_minutes,
    load_post_times,
    parse_jd,
    snapshot_frames,
    summarize_year,
)

FIX = Path(__file__).parent / "fixtures" / "data"
JD = FIX / "jd"
T6 = SNAPSHOT_X.index(6)
T30 = SNAPSHOT_X.index(30)
T60 = SNAPSHOT_X.index(60)


def minute(s: str) -> int:
    """'2024-03-10 11:34' → 1970-01-01 からの分。"""
    return int((datetime.strptime(s, "%Y-%m-%d %H:%M") - datetime(1970, 1, 1)).total_seconds()) // 60


def clock(m: float | None) -> str | None:
    """1970-01-01 からの分 → 'YYYY-MM-DD HH:MM' (欠損 −1 / None は None)。"""
    if m is None or m < 0:
        return None
    return (datetime(1970, 1, 1) + timedelta(minutes=float(m))).strftime("%Y-%m-%d %H:%M")


def jd_bytes(race_key: int, rows: list[tuple[int, str, int, list[tuple[float, float, float]]]]) -> bytes:
    """合成 JD: rows = (区分, MMDDHHMM, 単勝票数, 馬ごとの (単勝, 複勝Lo, 複勝Hi))。複勝票数は単勝票数と同じにする。"""
    heads = len(rows[0][3])
    header = "レースID,区分,月日時分,頭数,単勝票数,複勝票数," + ",".join(f"{i}単,{i}複Lo,{i}複Hi" for i in range(1, heads + 1))
    lines = [header]
    for kubun, code, votes, odds in rows:
        cells = [str(race_key), str(kubun), code, str(heads), f"{votes:>11}", f"{votes:>11}"]
        cells += [f"{v:>6.1f}" for o in odds for v in o]
        lines.append(",".join(cells))
    return ("\r\n".join(lines) + "\r\n").encode("cp932")


ODDS2 = [(2.0, 1.1, 1.3), (4.0, 1.5, 2.0)]
RK = 2024061501010101   # 2024-06-15 札幌 1 回 1 日 1R (合成)


def real(s: str, source: str = "race_list") -> PostTime:
    return PostTime(RK, float(minute(s)), source)


def synthetic(rows, post: PostTime | None = None, after_g1: bool = False):
    return extract_snapshots(parse_jd(jd_bytes(RK, rows)), post, after_g1)


def plateau_rows(at_1140: int, at_1148: int) -> list:
    """11:30 100 → 11:35 300 → 11:40 → 11:48 → 区分3 11:50 1000 (確定 1000 票)。"""
    return [
        (1, "06151130", 100, ODDS2),
        (1, "06151135", 300, ODDS2),
        (1, "06151140", at_1140, ODDS2),
        (1, "06151148", at_1148, ODDS2),
        (3, "06151150", 1000, ODDS2),
    ]


def test_file_name_from_race_key_uses_hex_kai_and_nichi():
    assert jd_file_name(2024070702011001) == "JD02241A01.CSV"
    assert jd_file_name(2015053005021102) == "JD05152B02.CSV"
    assert jd_file_name(2003100401020701) == "JD01032701.CSV"


def test_parse_real_file_with_hex_nichi():
    race = parse_jd((JD / "JD02241A01.CSV").read_bytes())
    assert race.race_key == 2024070702011001 and race.n_heads == 11
    assert race.odds.shape == (race.kubun.size, 11, 3) and race.odds.dtype == np.float32
    assert extract_file(str(JD / "JD02241A01.CSV")).race_key == race.race_key


def test_previous_day_rows_are_dated_before_race_day():
    race = parse_jd((JD / "JD05152B02.CSV").read_bytes())
    assert clock(race.t_min[0]) == "2015-05-29 22:06"
    assert clock(race.t_min[-1]) == "2015-05-30 10:48"
    assert (np.diff(race.t_min) >= 0).all()


def test_december_rows_of_a_january_race_belong_to_the_previous_year():
    m = jd_minutes(2024010606010101, np.array([12312355, 1051900, 1060930]))
    assert [clock(v) for v in m] == ["2023-12-31 23:55", "2024-01-05 19:00", "2024-01-06 09:30"]


def test_invalid_clock_value_is_rejected():
    with pytest.raises(ValueError):
        jd_minutes(2024010606010101, np.array([1062460]))


def test_rows_out_of_time_order_are_rejected():
    with pytest.raises(ValueError, match="時刻順"):
        parse_jd(jd_bytes(RK, [(1, "06150905", 100, ODDS2), (1, "06150900", 150, ODDS2)]))


def test_zero_odds_become_nan_and_deleted_rows_are_dropped():
    rows = [
        (1, "06150900", 100, [(0.0, 0.0, 0.0), (4.0, 1.5, 2.0)]),
        (0, "06150905", 150, ODDS2),
        (3, "06151000", 1000, ODDS2),
    ]
    race = parse_jd(jd_bytes(RK, rows))
    assert race.kubun.tolist() == [1, 3] and race.n_rows == 3
    assert np.isnan(race.odds[0, 0]).all() and race.odds[0, 1, 0] == pytest.approx(4.0)


def test_row_width_must_match_heads():
    data = jd_bytes(RK, [(1, "06150900", 100, ODDS2)]).replace(b"2.0\r\n", b"2.0,  1.0\r\n")
    with pytest.raises(ValueError, match="列数"):
        parse_jd(data)


def test_file_name_must_match_race_key(tmp_path: Path):
    wrong = tmp_path / "JD02241A02.CSV"
    shutil.copyfile(JD / "JD02241A01.CSV", wrong)
    with pytest.raises(ValueError, match="ファイル名"):
        extract_file(str(wrong))


def test_reference_is_the_first_of_two_k3_rows_and_final_odds_are_the_confirmed_ones():
    # 11:15 と 11:16 の区分3。2 行目は締切後の競走除外 (2 番) で再計算され、区分4 (確定・払戻) と同じ
    r = extract_file(str(JD / "JD03243104.CSV"))
    assert r.n_k3 == 2
    assert clock(r.first_k3) == "2024-11-02 11:15" and clock(r.ref) == "2024-11-02 11:15"
    assert clock(r.k4) == "2024-11-02 11:23"
    assert r.final_win_votes == 385653
    assert r.final_odds[0, 0] == pytest.approx(7.5) and r.final_odds[0, 1] == pytest.approx(2.3)   # 最初の区分3 は Lo 2.4
    assert np.isnan(r.final_odds[1]).all() and r.final_odds[2, 0] == pytest.approx(5.1)
    # 11:14 の区分1 (96.9%) は票数の頭打ち = 締切後なので、97% 未満でも候補にしない
    assert clock(r.close_plateau) == "2024-11-02 11:14" and r.n_dropped_postclose == 1
    # 発走の推定: 最後の候補行 11:09 + 6 分と ref 11:15 の小さい方を 5 分刻みに → 11:15、T-6 は 11:09
    assert r.post_source == "estimated" and clock(r.post_est) == clock(r.post_estimate) == "2024-11-02 11:15"
    assert clock(r.snap_time[T6]) == "2024-11-02 11:09" and r.staleness[T6] == 0


def test_post_close_k1_row_moves_reference_earlier_and_is_never_a_snapshot():
    r = extract_file(str(JD / "JD09241604.CSV"))
    assert clock(r.first_k3) == "2024-03-10 11:42"
    assert clock(r.close_est) == "2024-03-10 11:34"            # 区分1 で確定票数の 99.998%
    assert clock(r.ref) == "2024-03-10 11:36" and clock(r.post_est) == "2024-03-10 11:35"
    assert r.n_dropped_postclose == 1
    assert clock(r.snap_time[T6]) == "2024-03-10 11:29" and r.staleness[T6] == 0
    assert r.vote_share[T6] == pytest.approx(387606 / 604996)


def test_premature_k4_row_marks_the_close():
    # 区分 1;4;1;3;4: 11:34 の区分4 (確定票数の 99.999%)、11:47 の締切後の区分1、11:52 の区分3
    r = extract_file(str(JD / "JD08231304.CSV"))
    assert clock(r.close_est) == "2023-04-29 11:34" and clock(r.ref) == "2023-04-29 11:36"
    assert clock(r.k4) == "2023-04-29 11:58"   # 確定は最初の区分3 より後の区分4
    assert r.n_dropped_postclose == 1
    assert clock(r.post_est) == "2023-04-29 11:35" and clock(r.snap_time[T6]) == "2023-04-29 11:29"


def test_row_holding_97_percent_of_final_votes_is_dropped():
    r = extract_file(str(JD / "JD02241A01.CSV"))     # 09:49 の行が 417293 / 430110 = 97.02%
    assert clock(r.close_plateau) == "2024-07-07 09:49" and clock(r.ref) == "2024-07-07 09:51"
    assert r.n_dropped_postclose == 1
    assert clock(r.post_est) == "2024-07-07 09:50" and clock(r.snap_time[T6]) == "2024-07-07 09:44"


def test_row_holding_97_percent_is_dropped_without_a_plateau():
    # 前売りの厚いレースの形: 90% → 95% → 97.5% と増え続けるので頭打ちではないが、97% 以上の行は候補にしない
    rows = [(1, "06151130", 900, ODDS2), (1, "06151135", 950, ODDS2), (1, "06151140", 975, ODDS2),
            (3, "06151147", 1000, ODDS2)]
    r = synthetic(rows)
    assert r.close_est is None and r.n_dropped_postclose == 1
    # 推定は 11:35 + 6 分 → 11:40。T-6 の目標 11:34 に 11:30 は 4 分古いので欠損
    assert clock(r.post_est) == "2024-06-15 11:40" and r.snap_time[T6] == -1


@pytest.mark.parametrize("name, plateau, close_est, ref, t6, t6_share", [
    # 小倉 3R: 10:44 33.7% → 10:49 96.0% → 10:57 100% → 区分3 10:59 (9 分遅れの刻印)。10:49 は締切後
    ("JD10231803.CSV", "2023-02-05 10:49", "2023-02-05 10:49", "2023-02-05 10:51", "2023-02-05 10:44", 124334 / 368827),
    # 東京 2R: 10:34 96.4% (締切後) → 10:40 99.94% (99% の行) → 区分3 10:42
    ("JD05231802.CSV", "2023-02-19 10:34", "2023-02-19 10:34", "2023-02-19 10:36", "2023-02-19 10:29", 201409 / 491377),
    # 中山 5R: 12:29 86.3% の後は 7 分後の区分3 だけ。増え方の比 0.206 (締切後の行の分布の裾)
    ("JD06233505.CSV", "2023-04-08 12:29", "2023-04-08 12:29", "2023-04-08 12:31", "2023-04-08 12:24", 153387 / 397906),
])
def test_vote_plateau_catches_post_close_rows_below_the_vote_guard(name, plateau, close_est, ref, t6, t6_share):
    r = extract_file(str(JD / name))
    assert clock(r.close_plateau) == plateau and clock(r.close_est) == close_est and clock(r.ref) == ref
    assert clock(r.snap_time[T6]) == t6 and r.staleness[T6] == 0
    assert r.vote_share[T6] == pytest.approx(t6_share)
    assert (r.snap_time[r.snap_time >= 0] < r.close_plateau).all()


@pytest.mark.parametrize("name, t6, t6_share", [
    # 2015 日本ダービー: 15:28/15:29 の 1 分差の行 (次の行だけ見る判定なら誤って頭打ちになる)、15:34 93.8% は締切前
    ("JD05152C10.CSV", "2015-05-31 15:34", 11616017 / 12389579),
    # 2004 フェブラリーS: 15:34 91.6% の次は 10 分後の区分3 (比 0.28 まで薄まる) が、締切前
    ("JD05041811.CSV", "2004-02-22 15:34", 3435041 / 3748434),
])
def test_high_share_pre_close_rows_of_g1_races_are_kept(name, t6, t6_share):
    r = extract_file(str(JD / name))
    assert r.close_plateau is None and r.close_est is None
    assert clock(r.snap_time[T6]) == t6 and r.vote_share[T6] == pytest.approx(t6_share)


def test_vote_plateau_rule():
    r = synthetic(plateau_rows(960, 1000))     # 11:35→11:40 は 132 票/分、11:40→11:48 は 5 票/分
    assert clock(r.close_plateau) == "2024-06-15 11:40" and clock(r.close_est) == "2024-06-15 11:40"
    assert clock(r.ref) == "2024-06-15 11:42" and r.n_dropped_postclose == 2
    # 候補は 11:30 と 11:35 だけ。推定 11:40 の T-6 (11:34) に 11:35 は使えず、11:30 は 4 分古いので欠損
    assert clock(r.post_est) == "2024-06-15 11:40" and r.snap_time[T6] == -1
    assert synthetic(plateau_rows(800, 810)).close_plateau == minute("2024-06-15 11:40")   # 票数比 80% ちょうど
    assert synthetic(plateau_rows(790, 800)).close_plateau is None                         # 80% 未満は見ない
    # 11:35→11:40 は 100 票/分。11:40→11:48 が 199 票 (24.875 票/分) なら頭打ち、200 票 (比 1/4 ちょうど) なら違う
    assert synthetic(plateau_rows(800, 999)).close_plateau == minute("2024-06-15 11:40")
    assert synthetic(plateau_rows(800, 1000)).close_plateau is None
    stalled = [(1, "06151135", 900, ODDS2), (1, "06151140", 900, ODDS2), (1, "06151148", 900, ODDS2),
               (3, "06151150", 1000, ODDS2)]
    assert synthetic(stalled).close_plateau is None                                         # 増えていない行は判定しない


def test_vote_plateau_compares_windows_of_at_least_three_minutes():
    rows = [
        (1, "06151135", 500, ODDS2),
        (1, "06151139", 850, ODDS2),     # 1 分後の 11:40 だけを見ると 1 票/分で頭打ちに見える
        (1, "06151140", 851, ODDS2),
        (1, "06151145", 1000, ODDS2),
        (3, "06151150", 1000, ODDS2),
    ]
    r = synthetic(rows)
    assert clock(r.close_plateau) == "2024-06-15 11:45"


def test_rows_at_or_after_the_close_estimate_are_never_snapshots(monkeypatch):
    # 発走時刻表の許容幅を広げて目標時刻を締切後に置いても、close_est 以降の行 (11:40, 96%) は使わない
    monkeypatch.setattr(ts_odds, "POST_TIME_WINDOW_MIN", (-20.0, 10.0))
    r = synthetic(plateau_rows(960, 1000), post=real("2024-06-15 11:47"))
    assert r.post_actual and r.post_source == "race_list" and clock(r.close_est) == "2024-06-15 11:40"
    assert r.snap_time[T6] == -1 and np.isnan(r.snap_odds[T6]).all()
    assert r.n_dropped_postclose == 2


@pytest.mark.parametrize("name", sorted(p.name for p in JD.iterdir()))
def test_every_snapshot_obeys_the_guards(name: str):
    race = parse_jd((JD / name).read_bytes())
    r = extract_file(str(JD / name))
    if not r.has_k3:
        return
    assert r.post_est <= r.ref
    for j, x in enumerate(SNAPSHOT_X):
        if r.snap_time[j] < 0:
            assert np.isnan(r.snap_odds[j]).all() and np.isnan(r.staleness[j])
            continue
        assert 0 < r.vote_share[j] < 0.97
        assert np.isfinite(r.snap_odds[j, :, 0]).any()          # 存在するスナップショットには単勝オッズがある
        assert r.snap_time[j] <= r.post_est - x
        assert r.close_est is None or r.snap_time[j] < r.close_est
        assert r.snap_time[j] <= r.first_k3
        assert 0 <= r.staleness[j] <= STALENESS_CAP_MIN[x]
        at = race.kubun[race.t_min == r.snap_time[j]]
        assert set(at.tolist()) <= {1, 2}


def test_zero_vote_row_at_sales_open_is_not_a_snapshot():
    # 2004-01-17 中山 1R: 09:00 の発売開始の行は単勝票数 0 で単勝オッズが全部 NaN。推定発走 10:00 の T-60 の目標は
    # 09:00 ちょうどだが、その行は候補にしないので T-60 は欠損 (次の行 09:06 は目標より後)
    r = extract_file(str(JD / "JD06041501.CSV"))
    assert r.n_dropped_empty == 1 and clock(r.post_est) == "2004-01-17 10:00"
    assert r.snap_time[T60] == -1 and np.isnan(r.staleness[T60])
    assert clock(r.snap_time[T30]) == "2004-01-17 09:30" and clock(r.snap_time[T6]) == "2004-01-17 09:54"


def test_rows_with_votes_but_no_win_odds_are_not_candidates():
    no_win = [(0.0, 1.1, 1.3), (0.0, 1.5, 2.0)]
    rows = [(1, "06151148", 300, ODDS2), (1, "06151153", 400, no_win), (3, "06151200", 1000, ODDS2)]
    r = synthetic(rows, post=real("2024-06-15 11:59"))
    assert r.n_dropped_empty == 1
    assert r.snap_time[T6] == -1          # 11:53 は使えず、11:48 は T-6 の目標 11:53 に 5 分古い


def test_k3_and_k4_rows_are_never_used_for_snapshots():
    rows = [
        (1, "06150922", 100, ODDS2),
        (4, "06150924", 500, [(9.9, 9.9, 9.9)] * 2),   # 締切前に紛れ込んだ区分4 (T-6 の目標ちょうど)
        (3, "06150931", 1000, ODDS2),
        (4, "06150937", 1000, ODDS2),
    ]
    r = synthetic(rows, post=real("2024-06-15 09:30"))
    assert clock(r.snap_time[T6]) == "2024-06-15 09:22" and r.staleness[T6] == 2
    assert r.snap_odds[T6, 0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("last_row, t6_ok", [("06151150", True), ("06151149", False)])
def test_staleness_caps(last_row: str, t6_ok: bool):
    rows = [
        (1, "06150949", 100, ODDS2),    # T-120 目標 09:59 → 10 分前: 上限 10 分ちょうどで採用
        (1, "06151048", 200, ODDS2),    # T-60 目標 10:59 → 11 分前: 欠損
        (1, "06151124", 300, ODDS2),    # T-30 目標 11:29 → 5 分前: 採用
        (1, last_row, 400, ODDS2),      # T-6 目標 11:53 → 3 分前は採用、4 分前は欠損
        (3, "06151200", 1000, ODDS2),
        (4, "06151206", 1000, ODDS2),
    ]
    r = synthetic(rows, post=real("2024-06-15 11:59"))
    got = dict(zip(SNAPSHOT_X, r.staleness.tolist()))
    assert got[120] == 10 and np.isnan(got[60]) and got[30] == 5
    assert (got[6] == 3) if t6_ok else np.isnan(got[6])
    assert (r.snap_time[T60] == -1) and np.isnan(r.snap_odds[T60]).all()


def test_duplicate_timestamps_keep_the_last_row():
    rows = [
        (1, "06151149", 400, [(2.0, 1.1, 1.3), (4.0, 1.5, 2.0)]),
        (1, "06151149", 410, [(2.2, 1.2, 1.4), (3.8, 1.5, 2.0)]),
        (3, "06151156", 1000, ODDS2),
    ]
    r = synthetic(rows)
    assert clock(r.post_est) == "2024-06-15 11:55"
    assert r.snap_odds[T6, 0, 0] == pytest.approx(2.2) and r.vote_share[T6] == pytest.approx(0.41)


def test_zero_final_votes_give_no_close_estimate_and_no_snapshots():
    r = synthetic([(1, "06151150", 0, ODDS2), (3, "06151200", 0, ODDS2)])
    assert r.close_est is None and clock(r.ref) == "2024-06-15 12:00"
    assert (r.snap_time == -1).all() and r.n_dropped_empty == 1 and r.n_dropped_postclose == 0
    assert r.field_open is None


def test_field_open_counts_horses_priced_before_the_close():
    # 2023-09-10 阪神 2R: 8 頭登録、5 番は発売開始後 (09-10 06:26 まで値が付いている) に出走取消で 7 頭出走。
    # 発売開始時は 8 頭 → 複勝は 3 着まで払戻 (結果ページ 202309040202 の払戻表)
    r = extract_file(str(JD / "JD09234202.CSV"))
    assert r.n_heads == 8 and r.field_open == 8
    rows = [
        (1, "06151140", 300, [(2.0, 1.1, 1.3), (0.0, 0.0, 0.0), (4.0, 1.5, 2.0)]),   # 2 番は発売前に取消 (値が無い)
        (1, "06151150", 960, [(2.0, 1.1, 1.3), (0.0, 0.0, 0.0), (4.0, 1.5, 2.0)]),
        (1, "06151153", 990, [(2.0, 1.1, 1.3), (9.0, 2.0, 3.0), (4.0, 1.5, 2.0)]),   # 締切後の行だけに値
        (3, "06151200", 1000, [(2.0, 1.1, 1.3), (9.0, 2.0, 3.0), (4.0, 1.5, 2.0)]),
    ]
    assert synthetic(rows).field_open == 2


def test_real_post_time_overrides_the_estimate():
    race = parse_jd((JD / "JD09241604.CSV").read_bytes())
    r = extract_snapshots(race, PostTime(race.race_key, float(minute("2024-03-10 11:30")), "result"))
    assert r.post_actual and r.post_source == "result" and clock(r.post_est) == clock(r.post_real) == "2024-03-10 11:30"
    assert clock(r.post_estimate) == "2024-03-10 11:35"          # 推定値も残す (比較用)
    assert clock(r.ref) == "2024-03-10 11:36"
    assert clock(r.snap_time[T6]) == "2024-03-10 11:24" and r.staleness[T6] == 0


@pytest.mark.parametrize("post, used", [
    ("11:39", True), ("11:40", False),   # ref 11:36 + 3 分まで (締切 = 発走 − 2 分は区分 3・締切推定より後にならない)
    ("11:16", True), ("11:15", False),   # ref − 20 分まで (区分 3 の刻印の遅れ)
])
def test_post_times_inconsistent_with_the_jd_clock_fall_back_to_the_estimate(post: str, used: bool):
    race = parse_jd((JD / "JD09241604.CSV").read_bytes())   # ref 11:36、推定 11:35
    r = extract_snapshots(race, PostTime(race.race_key, float(minute(f"2024-03-10 {post}")), "race_list"))
    assert r.post_actual is used and r.post_time == minute(f"2024-03-10 {post}")
    assert r.post_source == ("race_list" if used else "estimated")
    assert clock(r.post_real) == (f"2024-03-10 {post}" if used else None)
    assert clock(r.post_est) == (f"2024-03-10 {post}" if used else "2024-03-10 11:35")


def test_post_time_of_another_race_is_rejected():
    race = parse_jd((JD / "JD09241604.CSV").read_bytes())
    with pytest.raises(ValueError, match="違います"):
        extract_snapshots(race, PostTime(race.race_key + 1, float(minute("2024-03-10 11:35")), "result"))


def test_g1_flag_moves_the_estimate_one_grid_step_earlier():
    race = parse_jd((JD / "JD09241604.CSV").read_bytes())
    r = extract_snapshots(race, None, after_g1=True)
    assert clock(r.post_estimate) == "2024-03-10 11:30" and r.post_source == "estimated"
    assert clock(r.snap_time[T6]) == "2024-03-10 11:24"


def _post_table(tmp_path: Path, **cols) -> Path:
    p = tmp_path / "post_times.parquet"
    pl.DataFrame(cols).write_parquet(p)
    return p


def test_load_post_times_prefers_post_datetime_and_maps_sources(tmp_path: Path):
    # 2008-02-09 の 1R は順延で 2/11 に走った (race_key の日付と post_datetime の日付が違う) とする
    p = _post_table(
        tmp_path,
        race_key=[2008020908020101, 2024031009010604, 2024070702011001],
        post_time=["10:00", "11:30", "09:45"],
        post_datetime=[datetime(2008, 2, 11, 10, 0), datetime(2024, 3, 10, 11, 30), datetime(2024, 7, 7, 9, 45)],
        source=["race_list", "netkeiba", "result"],
    )
    got = load_post_times(p)
    assert got == {
        2008020908020101: PostTime(2008020908020101, float(minute("2008-02-11 10:00")), "race_list"),
        2024031009010604: PostTime(2024031009010604, float(minute("2024-03-10 11:30")), "result"),
        2024070702011001: PostTime(2024070702011001, float(minute("2024-07-07 09:45")), "result"),
    }


def test_load_post_times_without_post_datetime_uses_the_race_key_date(tmp_path: Path):
    p = _post_table(tmp_path, race_key=[2024031009010604, 2024070702011001], post_time=["11:30", " 9:45"],
                    source=["race_list", "race_list"])
    assert {k: v.minutes for k, v in load_post_times(p).items()} == {
        2024031009010604: minute("2024-03-10 11:30"), 2024070702011001: minute("2024-07-07 09:45"),
    }
    p = _post_table(tmp_path, race_key=[2024031009010604], post_time=[datetime(2024, 3, 10, 11, 30).time()],
                    source=["result"])
    assert load_post_times(p)[2024031009010604].minutes == minute("2024-03-10 11:30")


def test_load_post_times_skips_null_times_and_missing_file(tmp_path: Path):
    p = _post_table(tmp_path, race_key=[2024031009010604, 2024070702011001],
                    post_datetime=[datetime(2024, 3, 10, 11, 30), None], source=["race_list", "race_list"])
    assert list(load_post_times(p)) == [2024031009010604]
    assert load_post_times(tmp_path / "none.parquet") == {}


@pytest.mark.parametrize("cols, match", [
    ({"race_key": [1, 1], "post_time": ["11:30", "11:35"], "source": ["result", "result"]}, "重複"),
    ({"race_key": [1], "post_time": ["11:30"]}, "source"),
    ({"race_key": [1], "post_time": ["11:30"], "source": ["guess"]}, "source"),
    ({"race_key": [1], "source": ["result"]}, "post_datetime か post_time"),
    ({"race_key": [1], "post_time": [1130], "source": ["result"]}, "HH:MM"),
    ({"race_key": [1], "post_datetime": [datetime(2024, 3, 10, 2, 30, tzinfo=timezone.utc)], "source": ["result"]},
     "タイムゾーン"),
])
def test_load_post_times_rejects_bad_tables(tmp_path: Path, cols: dict, match: str):
    with pytest.raises(ValueError, match=match):
        load_post_times(_post_table(tmp_path, **cols))


@pytest.mark.real_fixtures
def test_frames_schema():
    snap, race, final = snapshot_frames([extract_file(str(p)) for p in sorted(JD.iterdir())])
    assert dict(snap.schema) == {
        "race_key": pl.Int64, "horse_num": pl.Int8, "x": pl.Int16, "win": pl.Float32, "place_lo": pl.Float32,
        "place_hi": pl.Float32, "staleness_min": pl.Float32, "vote_share": pl.Float32, "snap_time": pl.Datetime("us"),
    }
    assert DIAGNOSTIC_COLUMNS == {"vote_share", "snap_time", "staleness_min"}
    assert DIAGNOSTIC_COLUMNS <= set(snap.columns) and SNAP_POST_RACE_COLUMNS is DIAGNOSTIC_COLUMNS
    assert dict(final.schema) == {
        "race_key": pl.Int64, "horse_num": pl.Int8, "win_final": pl.Float32, "place_lo_final": pl.Float32,
        "place_hi_final": pl.Float32,
    }
    for c in ("first_k3_time", "k4_time", "close_est", "close_votes", "close_plateau", "ref_time", "post_est",
              "post_real", "post_estimate", "post_time"):
        assert race.schema[c] == pl.Datetime("us")
    assert race.schema["post_actual"] == pl.Boolean and race.schema["post_source"] == pl.Utf8
    assert race.schema["field_open"] == pl.Int16 and race.schema["n_dropped_empty"] == pl.Int32
    assert race.height == 13 and race["has_k3"].sum() == 12
    k3 = race.filter(pl.col("has_k3"))
    assert set(k3["post_source"]) == {"estimated"} and race.filter(~pl.col("has_k3"))["post_source"].null_count() == 1
    assert k3["post_real"].null_count() == k3.height and k3["post_estimate"].null_count() == 0
    assert snap.height == 4 * final.height
    assert snap.group_by("race_key", "x").agg(pl.col("snap_time").n_unique()).filter(pl.col("snap_time") > 1).height == 0
    present = snap.filter(pl.col("snap_time").is_not_null()).group_by("race_key", "x").agg(pl.col("win").is_finite().any())
    assert present["win"].all()


@pytest.fixture(scope="module")
def records_path(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("rec") / "records.parquet"
    build_records(FIX / "record", out)
    return out


def test_build_ts_with_worker_pool(tmp_path: Path, records_path: Path):
    summaries = build_ts(JD, tmp_path, [2024, 2026], processes=2, records_path=records_path)
    assert [s["year"] for s in summaries] == [2024]   # 2026 は JD なし
    s = summaries[0]
    assert s["files"] == 3 and s["two_k3"] == 1 and s["ref_moved"] == 1 and s["ref_moved_plateau"] == 0
    assert s["dropped_rows"] == 3 and s["post_table"] == 0 and "ref_minus_post_med" not in s
    assert (s["post_result"], s["post_race_list"], s["post_estimated"]) == (0, 0, 3)
    assert s["final_eq_record"] == 1.0 and s["record_races_with_final"] == 3
    assert s["max_vote_share"] < 0.97 and s["min_vote_share"] > 0 and s["t6_share_ge90"] == 0
    assert s["snap_unpriced"] == 0
    for name in ("ts_snap", "ts_race", "ts_final"):
        assert (tmp_path / name / "2024.parquet").exists()


def test_build_ts_passes_post_times_and_the_g1_flag_to_workers(tmp_path: Path):
    rk = 2024031009010604
    posts = {rk: PostTime(rk, float(minute("2024-03-10 11:30")), "race_list")}
    build_ts(JD, tmp_path, [2024], post_times=posts, after_g1=frozenset({2024070702011001}), processes=2)
    race = pl.read_parquet(tmp_path / "ts_race" / "2024.parquet")
    row = race.filter(pl.col("race_key") == rk).row(0, named=True)
    assert row["post_source"] == "race_list" and row["post_real"] == datetime(2024, 3, 10, 11, 30)
    assert row["post_estimate"] == datetime(2024, 3, 10, 11, 35)
    g1 = race.filter(pl.col("race_key") == 2024070702011001).row(0, named=True)
    assert g1["post_source"] == "estimated" and g1["post_estimate"] == datetime(2024, 7, 7, 9, 45)   # 09:50 の 1 刻み前


@pytest.mark.real_fixtures
def test_summary_counts_plateau_moves_and_measures_post_time_lags(tmp_path: Path):
    post_times = {
        2023020510010803: PostTime(2023020510010803, float(minute("2023-02-05 10:50")), "race_list"),  # ref 10:51、区分3 10:59
        2023021905010802: PostTime(2023021905010802, float(minute("2023-02-19 10:35")), "result"),     # ref 10:36、区分3 10:42
        2023040806030505: PostTime(2023040806030505, float(minute("2023-04-08 13:00")), "race_list"),  # ref 12:31 より 29 分後
    }
    summaries = build_ts(JD, tmp_path, [2023], post_times=post_times, processes=1)
    s = summaries[0]
    assert s["files"] == 5 and s["ref_moved"] == 4 and s["ref_moved_plateau"] == 3
    assert (s["post_table"], s["post_actual"], s["post_rejected"]) == (3, 2, 1)
    assert (s["post_result"], s["post_race_list"], s["post_estimated"]) == (1, 1, 3)
    assert s["k3_minus_post_med"] == pytest.approx(8.0)          # 9 分と 7 分
    assert s["ref_minus_post_med"] == pytest.approx(1.0)
    assert (s["est_eq_real"], s["est_lt_real"], s["est_gt_real"]) == (1.0, 0.0, 0)
    race = pl.read_parquet(tmp_path / "ts_race" / "2023.parquet")
    assert race.filter(pl.col("post_actual")).height == 2 and race["post_time"].null_count() == 2
    direct = summarize_year(
        2023,
        pl.read_parquet(tmp_path / "ts_snap" / "2023.parquet"),
        race,
        pl.read_parquet(tmp_path / "ts_final" / "2023.parquet"),
        None,
    )
    assert direct["post_rejected"] == 1


def test_final_win_odds_equal_record_win_odds_for_starters(tmp_path: Path, records_path: Path):
    summaries = build_ts(JD, tmp_path, [2005, 2015, 2023], processes=1, records_path=records_path)
    by_year = {s["year"]: s for s in summaries}
    assert by_year[2005]["no_k3"] == 1 and by_year[2005]["record_races_with_final"] == 0
    assert by_year[2015]["final_eq_record"] == 1.0 and by_year[2015]["starters_with_final"] == 15
    assert by_year[2023]["final_eq_record"] == 1.0
    final = pl.read_parquet(tmp_path / "ts_final" / "2015.parquet")
    excluded = final.filter((pl.col("race_key") == 2015053005021102) & (pl.col("horse_num") == 13)).row(0, named=True)
    assert np.isnan(excluded["win_final"])   # 競走除外: 確定オッズなし
