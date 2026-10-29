import fcntl
import gzip
import json
import sqlite3

import polars as pl
import pytest
import requests
import yaml

from keiba_goal.config import load_config
from keiba_goal.netkeiba import crawl, files, horse_master, rebuild, validate
from keiba_goal.netkeiba.http import Fetcher, FetchSettings, Target, decode
from keiba_goal.netkeiba.parse_result import parse_result
from test_netkeiba_support import FX, FakeResponse, FixtureSession, record_line, write_record

RACE = "202306010106"   # 2023-01-05 中山 6R、10 頭 (取消 1)
MAINTENANCE = "<html><body>ただいまメンテナンス中です</body></html>".encode()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """本物の HTTP に出たら失敗させる (偽セッションだけを使う)。"""
    def refuse(*args, **kwargs):
        raise AssertionError("テストからネットワークに出ようとした")
    monkeypatch.setattr(requests.Session, "request", refuse)


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def env(tmp_path, monkeypatch):
    """一時ディレクトリの設定 + 保存ページを返すセッション (ネットワークに出ない)。"""
    rec_dir = tmp_path / "record"
    res = parse_result(decode(gzip.decompress((FX / f"result_{RACE}.html.gz").read_bytes()),
                              "https://db.netkeiba.com/"), RACE)
    lines = [record_line(race_id=f"20230105{RACE[4:]}{r['horse_num']:02d}", year=23, month="01", day="05",
                         times=1, place="中山", daily=1, race_num=6, horse_N=10, horse_num=r["horse_num"],
                         waku_num=r["waku_num"], track_code=17, corner_num=4, dist=2000, id=r["horse_id"])
             for r in res.rows.iter_rows(named=True)]
    write_record(rec_dir / "record_data_2023.csv", lines)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "paths": {"record_dir": str(rec_dir), "ts_odds_dir": str(tmp_path), "spec_xlsx": str(tmp_path / "x.xlsx"),
                  "data_root": str(tmp_path / "data")},
        "netkeiba": {"min_interval_sec": 2.5, "backoff_base_sec": 1, "incomplete_ttl_hours": 6, "windows": {
            "gate": {"start": "2023-01-05", "end": "2023-01-05", "pages": ["result", "shutuba_past"], "ids": "record"},
            "1": {"start": "2026-09-19", "end": "2026-09-20", "pages": ["result", "shutuba_past"], "ids": "race_list"},
            "2": {"start": "2023-01-05", "end": "2023-01-05", "pages": ["result"], "ids": "record"},
            "3": {"start": "2004-01-01", "end": "2004-12-31", "pages": ["result"], "ids": "record"}}},
    }, allow_unicode=True), encoding="utf-8")
    session = FixtureSession()
    clock = Clock()

    class TestFetcher(Fetcher):
        @classmethod
        def from_config(cls, cfg, **kw):
            return cls(cfg.paths.raw, FetchSettings.from_section(cfg.section("netkeiba")), session=session,
                       clock=clock, sleep=clock.sleep, **kw)

    monkeypatch.setattr(crawl, "Fetcher", TestFetcher)
    return load_config(cfg_path), session, clock


def offline(cfg) -> Fetcher:
    return Fetcher(cfg.paths.raw, FetchSettings(), session=FixtureSession(), offline=True)


def rewrite_record(cfg, **changes):
    """env の record_data の全行で列を書き換える (馬マスタの作り直しの確認用)。"""
    res = parse_result(decode(gzip.decompress((FX / f"result_{RACE}.html.gz").read_bytes()),
                              "https://db.netkeiba.com/"), RACE)
    lines = [record_line(race_id=f"20230105{RACE[4:]}{r['horse_num']:02d}", horse_N=10, waku_num=r["waku_num"],
                         track_code=17, corner_num=4, dist=2000, id=r["horse_id"], **changes)
             for r in res.rows.iter_rows(named=True)]
    write_record(cfg.paths.record_dir / "record_data_2023.csv", lines)


def outputs(cfg, window):
    d = cfg.paths.interim
    return [d / f"nk_rows_{window}.parquet", d / f"nk_payouts_{window}.parquet", d / f"post_times_{window}.parquet"]


def fixture_bytes(name: str) -> bytes:
    return gzip.decompress((FX / name).read_bytes())


# --- 取得と作り直しの基本 ---------------------------------------------------------------------

def test_dry_run_touches_no_network_and_writes_nothing(env, capsys):
    cfg, session, _ = env
    assert crawl.run(cfg, "gate", None, dry=True, parse_only=False) == 0
    assert session.calls == []
    assert not any(p.exists() for p in outputs(cfg, "gate"))
    assert "約 2 GET" in capsys.readouterr().out


def test_gate_fetches_validates_builds_and_then_reuses_cache(env, capsys):
    cfg, session, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    assert len(session.calls) == 2
    f = offline(cfg)
    for kind in ("result", "shutuba_past"):
        rec = f.manifest.get(Target(kind, RACE).url)
        assert rec.complete is True and rec.error is None and rec.parser_version == validate.VERSIONS[kind]
    rows, pays, posts = (pl.read_parquet(p) for p in outputs(cfg, "gate"))
    assert rows.height == 10 and set(rows["source"]) == {"netkeiba"}
    assert set(pays["bet_type"]) >= {"win", "place"} and set(pays["n_paid_places"]) == {3}
    assert pays.filter(pl.col("bet_type") == "win")["horse_num"].to_list() == [5]
    post = posts.row(0, named=True)
    assert (post["post_time"], post["source"], str(post["post_datetime"])) == ("12:45", "result", "2023-01-05 12:45:00")
    report = json.loads((cfg.paths.raw / "reports" / "gate_report.json").read_text())
    assert report["rows_converted"] == 10 and report["dtype_problems"] == []
    assert (cfg.paths.interim / "horse_master.parquet").exists()
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    assert len(session.calls) == 2  # 完全なページは取り直さない
    before = manifest_dump(cfg)
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=True) == 0
    assert len(session.calls) == 2 and pl.read_parquet(outputs(cfg, "gate")[0]).height == 10
    assert manifest_dump(cfg) == before   # --parse-only は manifest に書かない
    assert not [p for p in cfg.paths.interim.iterdir() if p.name.endswith(".tmp")]


def manifest_dump(cfg) -> tuple[list, list]:
    with sqlite3.connect(cfg.paths.raw / "manifest.sqlite") as con:
        return (con.execute("SELECT * FROM pages ORDER BY url").fetchall(),
                con.execute("SELECT * FROM requests ORDER BY id").fetchall())


def test_dry_run_counts_only_pages_that_need_a_get(env, capsys):
    cfg, session, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False, no_rebuild=True) == 0
    offline(cfg).mark(Target("result", RACE), complete=True, parser_version="result-0")   # 古いパーサで完全
    capsys.readouterr()
    assert crawl.run(cfg, "gate", None, dry=True, parse_only=False) == 0
    assert "{'result': 0, 'shutuba_past': 0}  → 約 0 GET" in capsys.readouterr().out   # 写しの検査し直しで足りる
    offline(cfg).cache_path(Target("result", RACE)).unlink()
    assert crawl.run(cfg, "gate", None, dry=True, parse_only=False) == 0
    assert "{'result': 1, 'shutuba_past': 0}  → 約 1 GET" in capsys.readouterr().out


def test_limit_zero_never_touches_network(env):
    cfg, session, _ = env
    assert crawl.run(cfg, "1", 0, dry=False, parse_only=False) == 0
    assert crawl.run(cfg, "2", 0, dry=False, parse_only=False) == 0
    assert session.calls == []
    assert pl.read_parquet(outputs(cfg, "2")[0]).is_empty()


def test_no_rebuild_fetches_but_writes_no_outputs(env, capsys):
    cfg, session, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False, no_rebuild=True) == 0
    assert len(session.calls) == 2 and offline(cfg).manifest.get(Target("result", RACE).url).complete is True
    assert not any(p.exists() for p in outputs(cfg, "gate"))
    assert not (cfg.paths.interim / "horse_master.parquet").exists()
    assert "--parse-only" in capsys.readouterr().out
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=True) == 0   # そろってから 1 回だけ作り直す
    assert len(session.calls) == 2 and pl.read_parquet(outputs(cfg, "gate")[0]).height == 10


def test_window1_enumerates_by_race_list_and_respects_limit(env):
    cfg, session, _ = env
    assert crawl.run(cfg, "1", 1, dry=False, parse_only=False) == 0
    assert session.calls[0].endswith("kaisai_date=20260919")
    assert len(session.calls) == 3  # 一覧 1 + 1 レース分 (出馬表・結果) だけ
    f = offline(cfg)
    assert f.manifest.get(Target("race_list", "20260919").url).complete is True
    assert f.manifest.get(Target("result", "202606040501").url).error == "http 404"


def test_incomplete_result_is_marked_and_rows_count_checked(env):
    cfg, session, _ = env
    f = Fetcher(cfg.paths.raw, FetchSettings(), session=session)
    page = f.get(Target("result", RACE))
    assert validate.mark(f, page, expected_entries=11) == (False, "行数 10 ≠ 登録頭数 11")
    assert f.manifest.get(Target("result", RACE).url).complete is False


@pytest.mark.skipif(not load_config().paths.record_dir.exists(), reason="record_data が無い")
def test_check_history_reports_race_day_content(env, tmp_path, capsys):
    cfg, _, _ = env
    real = load_config()
    cfg_path = tmp_path / "cfg2.yaml"
    raw = dict(cfg.raw)
    raw["paths"] = {**raw["paths"], "record_dir": str(real.paths.record_dir)}
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert crawl.check_history(load_config(cfg_path)) == 0
    out = capsys.readouterr().out
    assert "ブリンカー 15/15" in out and "レース当日の内容を返している" in out


def test_a_page_with_an_odd_value_is_logged_and_the_crawl_continues(env, monkeypatch):
    cfg, session, _ = env
    real_get = session.get

    def odd_payout(url, **kw):
        r = real_get(url, **kw)
        if "db.netkeiba.com" in url:
            r.content = r.content.replace("14,150".encode("euc_jp"), "特払".encode("euc_jp"))
        return r

    monkeypatch.setattr(session, "get", odd_payout)
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    rec = offline(cfg).manifest.get(Target("result", RACE).url)
    assert rec.complete is False and "払戻" in rec.error
    errors = json.loads((cfg.paths.raw / "reports" / "build_errors_gate.json").read_text())
    assert [e["stage"] for e in errors] == ["result"] and "特払" in errors[0]["error"]
    assert pl.read_parquet(outputs(cfg, "gate")[0]).is_empty()


# --- 一覧: 空の容器だけがレースの無い日 --------------------------------------------------------

def test_window1_a_200_page_without_the_list_container_is_incomplete_and_refetched_after_ttl(env, monkeypatch):
    cfg, session, clock = env
    real_get = session.get
    state = {"maintenance": True}

    def get(url, **kw):
        if "race_list_sub" in url and state["maintenance"]:
            session.calls.append(url)
            return FakeResponse(200, MAINTENANCE)
        return real_get(url, **kw)

    monkeypatch.setattr(session, "get", get)
    w = crawl.load_window(cfg, "1", crawl.jst_today())
    f = crawl.Fetcher.from_config(cfg)
    lists = crawl.listed_races(f, w, crawl.Crawl(f))
    rec = f.manifest.get(Target("race_list", "20260919").url)
    assert lists.races.is_empty() and rec.complete is False and "項目が無い" in rec.error
    assert [d.status for d in lists.days] == ["unreadable", "unreadable"]
    n = len(session.calls)
    crawl.listed_races(f, w, crawl.Crawl(f))
    assert len(session.calls) == n      # TTL 内は取り直さない
    state["maintenance"] = False
    clock.t += 7 * 3600
    lists = crawl.listed_races(f, w, crawl.Crawl(f))
    assert lists.races.height == 24 and f.manifest.get(Target("race_list", "20260919").url).complete is True


def test_window1_only_the_empty_container_counts_as_a_day_without_races(env, monkeypatch):
    cfg, session, _ = env
    real_get = session.get

    def get(url, **kw):
        if url.endswith("kaisai_date=20260920"):
            session.calls.append(url)
            return FakeResponse(200, fixture_bytes("race_list_20040105_empty.html.gz"))
        return real_get(url, **kw)

    monkeypatch.setattr(session, "get", get)
    assert crawl.run(cfg, "1", None, dry=False, parse_only=False, no_rebuild=True) == 0
    rec = offline(cfg).manifest.get(Target("race_list", "20260920").url)
    assert rec.complete is True and rec.error is None
    n = len(session.calls)
    assert crawl.run(cfg, "1", None, dry=False, parse_only=False, no_rebuild=True) == 0
    assert not any("kaisai_date" in u for u in session.calls[n:])   # 完全な一覧は取り直さない
    rep = json.loads((cfg.paths.raw / "reports" / "suspicious_days_1.json").read_text())
    assert rep["by_status"] == {"races": 1, "empty": 1, "unreadable": 0}
    assert rep["empty_weekend_or_holiday"] == [{"date": "20260920", "weekday": "日", "holiday": False}]


# --- 失敗の扱い: 1 つの URL では止まらない、続いたら止まる --------------------------------------

class SyntheticSession:
    """結果ページはどの race_id にも 202306010106 の保存ページを返す。fail の URL は 500、invalid は 200 のエラーページ。"""

    def __init__(self, fail=(), invalid=()):
        self.calls: list[str] = []
        self.fail, self.invalid = set(fail), set(invalid)
        self.body = fixture_bytes(f"result_{RACE}.html.gz")

    def get(self, url, **kw):
        self.calls.append(url)
        rid = url.rstrip("/").rsplit("/", 1)[1]
        if rid in self.fail or "*" in self.fail:
            return FakeResponse(500, b"")
        if rid in self.invalid or "*" in self.invalid:
            return FakeResponse(200, b"<html><body>error</body></html>")
        return FakeResponse(200, self.body)


def synthetic(tmp_path, session, n=6):
    clock = Clock()
    f = Fetcher(tmp_path / "raw", FetchSettings(backoff_base_sec=1), session=session, clock=clock, sleep=clock.sleep)
    races = pl.DataFrame({"nk_race_id": [f"2023010101{r:02d}" for r in range(1, n + 1)], "date": [20230101] * n,
                          "horse_N": [10] * n}, schema=crawl.RACES_SCHEMA)
    return f, races, clock


def test_one_permanently_failing_url_does_not_stop_the_crawl(tmp_path):
    session = SyntheticSession(fail={"202301010103"})
    f, races, clock = synthetic(tmp_path, session)
    c = crawl.Crawl(f)
    assert crawl.fetch_races(c, races, ("result",), None) == 6
    assert c.failed == [Target("result", "202301010103").url]
    for rid in races["nk_race_id"]:
        rec = f.manifest.get(Target("result", rid).url)
        if rid == "202301010103":
            assert (rec.complete, rec.http_status, rec.error) == (False, 500, "http 500 ×3")
        else:
            assert rec.complete is True
    assert len(session.calls) == 5 + 3
    crawl.fetch_races(crawl.Crawl(f), races, ("result",), None)
    assert len(session.calls) == 8      # TTL 内は失敗した URL も取り直さない
    session.fail.clear()
    clock.t += 7 * 3600
    assert crawl.fetch_races(crawl.Crawl(f), races, ("result",), None) == 1
    assert len(session.calls) == 9 and f.manifest.get(Target("result", "202301010103").url).complete is True


def test_consecutive_failed_requests_across_urls_stop_the_crawl(tmp_path):
    session = SyntheticSession(fail={"*"})
    f, races, _ = synthetic(tmp_path, session)
    with pytest.raises(crawl.FetchBlocked):
        crawl.fetch_races(crawl.Crawl(f), races, ("result",), None)
    assert len(session.calls) == 5      # 1 つ目の URL で 3 回、2 つ目で 2 回 → 連続 5 回で停止
    assert f.manifest.get(Target("result", "202301010101").url).error == "http 500 ×3"
    assert f.manifest.get(Target("result", "202301010102").url) is None


def test_failed_requests_stop_a_window_run_with_a_nonzero_exit(env, monkeypatch):
    cfg, session, _ = env
    monkeypatch.setattr(session, "get", lambda url, **kw: FakeResponse(503, b""))
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 2
    assert not any(p.exists() for p in outputs(cfg, "gate"))   # 止めたら作り直さない


def test_ten_consecutive_invalid_200_pages_stop_the_crawl(tmp_path):
    session = SyntheticSession(invalid={"*"})
    f, races, _ = synthetic(tmp_path, session, n=12)
    with pytest.raises(crawl.CrawlStopped, match="10 件続いた"):
        crawl.fetch_races(crawl.Crawl(f), races, ("result",), None)
    assert len(session.calls) == 10
    session2 = SyntheticSession(invalid={f"2023010101{r:02d}" for r in range(1, 10)})
    f2, races2, _ = synthetic(tmp_path / "b", session2, n=12)
    assert crawl.fetch_races(crawl.Crawl(f2), races2, ("result",), None) == 12   # 9 件で途切れれば止めない


def test_invalid_pages_stop_a_window_run_with_a_nonzero_exit(env, monkeypatch, capsys):
    cfg, session, _ = env
    cfg.raw["netkeiba"]["max_consecutive_invalid_pages"] = 2
    monkeypatch.setattr(session, "get", lambda url, **kw: FakeResponse(200, b"<html><body>error</body></html>"))
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 2
    assert "2 件続いた" in capsys.readouterr().err


# --- キャッシュの検査し直し・消えたキャッシュ ------------------------------------------------------

def test_cached_copy_is_revalidated_before_any_refetch(env):
    cfg, session, clock = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False, no_rebuild=True) == 0
    f = crawl.Fetcher.from_config(cfg)
    result, card = Target("result", RACE), Target("shutuba_past", RACE)
    f.mark(result, complete=False, parser_version="result-0", error="古いパーサで失敗")
    f.mark(card, complete=True, parser_version="shutuba-0")
    clock.t += 30 * 86400   # TTL を過ぎても、今のパーサで通るならネットワークに触れない
    n = len(session.calls)
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False, no_rebuild=True) == 0
    assert len(session.calls) == n
    for t in (result, card):
        rec = f.manifest.get(t.url)
        assert rec.complete is True and rec.parser_version == validate.VERSIONS[t.kind] and rec.error is None


def test_a_complete_page_whose_cache_file_is_lost_is_refetched(env):
    cfg, session, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    f = offline(cfg)
    f.cache_path(Target("result", RACE)).unlink()
    assert f.cached(Target("result", RACE)) is None
    n = len(session.calls)
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    assert session.calls[n:] == [Target("result", RACE).url]
    assert json.loads((cfg.paths.raw / "reports" / "build_errors_gate.json").read_text()) == []


def test_build_uses_only_pages_the_manifest_calls_complete(env):
    cfg, _, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    offline(cfg).mark(Target("result", RACE), complete=False, parser_version="result-2", error="単勝の払戻が無い")
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=True) == 0
    errors = json.loads((cfg.paths.raw / "reports" / "build_errors_gate.json").read_text())
    assert errors == [{"nk_race_id": RACE, "date": 20230105, "stage": "result",
                       "error": "manifest で不完全 (単勝の払戻が無い)"}]
    assert all(pl.read_parquet(p).is_empty() for p in outputs(cfg, "gate"))


# --- 馬マスタ・ロック ---------------------------------------------------------------------------

def test_master_is_reseeded_from_record_and_keeps_card_horses_of_other_windows(env):
    cfg, _, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    seen = pl.read_parquet(cfg.paths.raw / rebuild.CARD_HORSES)
    assert seen.height == 10 and set(seen["first_date"]) == {20230105}
    m = pl.read_parquet(cfg.paths.interim / "horse_master.parquet")
    assert m["broodmare_sire_ref"].null_count() == 0 and set(m["source"]) == {"record"}
    rewrite_record(cfg, father="Too Darn Hot")   # record の読み込み結果が変わったら、マスタも作り直される
    assert crawl.run(cfg, "3", 0, dry=False, parse_only=False) == 0
    m2 = pl.read_parquet(cfg.paths.interim / "horse_master.parquet")
    from keiba_goal.netkeiba.records import read_record_range

    want = read_record_range(cfg.paths.record_dir, 20230105, 20230105, ["father"])["father"].unique().to_list()
    assert m2["father"].unique().to_list() == want == ["Too Darn Hot"]   # P1 の読み込みは引用符を外す
    assert m2["broodmare_sire_ref"].null_count() == 0              # ゲートの出馬表の馬は残る
    assert pl.read_parquet(cfg.paths.raw / rebuild.CARD_HORSES).equals(seen)


def test_update_master_rereads_card_horses_written_by_another_window(env):
    cfg, _, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    w = crawl.load_window(cfg, "2", crawl.jst_today())
    seed = horse_master.seed_from_record(cfg.paths.record_dir)
    b = rebuild.build(cfg, offline(cfg), w.pages, crawl.record_races(cfg, w), seed)
    # build の後・書く前に、別の窓 (窓 1) の作り直しが card_horses に馬を足した
    path = cfg.paths.raw / rebuild.CARD_HORSES
    other = pl.read_parquet(path).head(1).with_columns(horse_id=pl.lit(2025100001, pl.Int64),
                                                        first_date=pl.lit(20251001, pl.Int32),
                                                        last_date=pl.lit(20251001, pl.Int32))
    files.save_parquet(pl.concat([pl.read_parquet(path), other]), path)
    with files.build_lock(cfg.paths.raw):
        master = rebuild.update_master(cfg, w.start, w.end, seed, b)
    assert 2025100001 in pl.read_parquet(path)["horse_id"].to_list()
    assert 2025100001 in master["horse_id"].to_list()


def test_build_lock_is_an_exclusive_file_lock_and_nests_in_one_process(tmp_path):
    def try_lock() -> bool:
        with open(tmp_path / files.BUILD_LOCK, "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(fh, fcntl.LOCK_UN)
            return True

    with files.build_lock(tmp_path):
        assert not try_lock()            # ほかのプロセス (別の open) は待たされる
        with files.build_lock(tmp_path):   # 同じプロセスの入れ子 (作り直し → combine) は素通り
            assert not try_lock()
        assert not try_lock()
    assert try_lock()


def test_gate_uses_only_record_rows_before_the_window_for_parents(env, capsys):
    cfg, _, _ = env
    rewrite_record(cfg, father="レコードノチチ")   # 窓の record 行にだけある名前は変換に使わない
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    rows = pl.read_parquet(outputs(cfg, "gate")[0])
    assert "レコードノチチ" not in rows["father"].to_list()   # 出馬表の父 (窓より前の record に馬がいない)
    report = json.loads((cfg.paths.raw / "reports" / "gate_report.json").read_text())
    assert report["mismatches"]["father"] == 10 and report["parents"]["rows_from_card"] == 10
    assert report["card_names_vs_record"]["differ"] == 10 and report["card_names_vs_record"]["card_null"] == 0
    assert "father: 10 件" in capsys.readouterr().out


# --- まとめ (combine) --------------------------------------------------------------------------

def _racelist_post_times(cfg, rows: list[tuple[int, str, str]]) -> None:
    df = pl.DataFrame([{"race_key": k, "nk_race_id": rid, "date": k // 10**8, "post_time": t,
                        "post_datetime": f"{k // 10**8} {t}", "source": "race_list"} for k, rid, t in rows])
    df = df.with_columns(pl.col("post_datetime").str.strptime(pl.Datetime("ms"), "%Y%m%d %H:%M"))
    files.save_parquet(df.cast(rebuild.POST_SCHEMA), cfg.paths.interim / "post_times_racelist.parquet")


def test_combine_merges_windows_and_prefers_result_post_times_over_the_race_list(env):
    from keiba_goal.data.ts_odds import load_post_times

    cfg, _, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    d = cfg.paths.interim
    gate_key = 2023010506010106
    # 一覧は同じレースに別の時刻 (結果ページが優先)、ほかのレースは一覧だけ
    _racelist_post_times(cfg, [(gate_key, RACE, "12:40"), (2008020305010401, "200805010401", "10:05")])
    written = rebuild.combine(cfg)
    n_pay = pl.read_parquet(d / "nk_payouts_gate.parquet").height
    assert written == {"nk_rows.parquet": 10, "payouts.parquet": n_pay, "post_times.parquet": 2}
    assert pl.read_parquet(d / "nk_rows.parquet").equals(pl.read_parquet(d / "nk_rows_gate.parquet"))
    assert pl.read_parquet(d / "payouts.parquet").equals(pl.read_parquet(d / "nk_payouts_gate.parquet"))
    posts = pl.read_parquet(d / "post_times.parquet")
    assert posts.schema == pl.Schema(rebuild.POST_SCHEMA)
    assert posts.select("race_key", "post_time", "source").rows() == [
        (2008020305010401, "10:05", "race_list"), (gate_key, "12:45", "result")]
    assert rebuild.post_time_sources(cfg) == {"race_list": 1, "result": 1}
    pt = load_post_times(d / "post_times.parquet")       # P1 の読み込み (post_datetime を使う)
    assert set(pt) == {gate_key, 2008020305010401} and pt[gate_key].source == "result"


def test_combine_refuses_the_same_race_in_two_window_files(env):
    cfg, _, _ = env
    assert crawl.run(cfg, "gate", None, dry=False, parse_only=False) == 0
    d = cfg.paths.interim
    files.save_parquet(pl.read_parquet(d / "nk_rows_gate.parquet"), d / "nk_rows_2.parquet")
    with pytest.raises(rebuild.CombineError, match="nk_rows.parquet"):
        rebuild.combine(cfg)
    (d / "nk_rows_2.parquet").unlink()
    files.save_parquet(pl.read_parquet(d / "post_times_gate.parquet"), d / "post_times_2.parquet")
    with pytest.raises(rebuild.CombineError, match="post_times.parquet"):
        rebuild.combine(cfg)   # 同じ出どころ (結果ページ) の重複は選ばずに止める
