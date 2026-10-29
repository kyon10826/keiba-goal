import fcntl
import gzip
import threading
from pathlib import Path

import pytest
import requests

from keiba_goal.netkeiba.http import (
    FetchBlocked,
    FetchError,
    Fetcher,
    FetchSettings,
    Manifest,
    RateLimiter,
    Target,
    decode,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """本物の HTTP に出たら失敗させる (偽セッションだけを使う)。"""
    def refuse(*args, **kwargs):
        raise AssertionError("テストからネットワークに出ようとした")
    monkeypatch.setattr(requests.Session, "request", refuse)


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class FakeResponse:
    def __init__(self, status: int, content: bytes):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_fetcher(tmp_path: Path, responses, **settings) -> tuple[Fetcher, FakeSession, FakeClock]:
    clock = FakeClock()
    session = FakeSession(responses)
    fetcher = Fetcher(tmp_path, FetchSettings(**settings), session=session, clock=clock, sleep=clock.sleep)
    return fetcher, session, clock


RESULT = Target("result", "202509040311")
PAST = Target("shutuba_past", "202509040311")


def test_target_validates_key_and_builds_url():
    assert RESULT.url == "https://db.netkeiba.com/race/202509040311/"
    assert Target("race_list", "20250913").url.endswith("kaisai_date=20250913")
    with pytest.raises(ValueError):
        Target("result", "2025090403")
    with pytest.raises(ValueError):
        Target("horse", "202509040311")


def test_rate_limiter_rejects_interval_below_floor(tmp_path):
    with pytest.raises(ValueError):
        RateLimiter(tmp_path / "l", tmp_path / "s", 1.0)


def test_rate_limiter_waits_between_slots_across_instances(tmp_path):
    clock = FakeClock()
    a = RateLimiter(tmp_path / "lock", tmp_path / "last", 2.5, clock, clock.sleep)
    b = RateLimiter(tmp_path / "lock", tmp_path / "last", 2.5, clock, clock.sleep)  # 別プロセス相当
    with a.slot():
        pass
    clock.t += 1.0
    with b.slot():
        pass
    assert clock.sleeps == [pytest.approx(1.5)]
    clock.t += 10
    with a.slot():
        pass
    assert len(clock.sleeps) == 1


def test_rate_limiter_caps_wait_when_clock_goes_back(tmp_path):
    clock = FakeClock()
    lim = RateLimiter(tmp_path / "lock", tmp_path / "last", 2.5, clock, clock.sleep)
    with lim.slot():
        pass
    clock.t -= 3600
    with lim.slot():
        pass
    assert clock.sleeps == [2.5]


def test_rate_limiter_holds_an_exclusive_file_lock(tmp_path):
    lim = RateLimiter(tmp_path / "lock", tmp_path / "last", 2.5)
    seen: list[str] = []

    def try_lock():
        with open(tmp_path / "lock", "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append("acquired")
            except BlockingIOError:
                seen.append("blocked")

    with lim.slot():
        t = threading.Thread(target=try_lock)
        t.start()
        t.join()
    assert seen == ["blocked"]


def test_manifest_is_wal_and_roundtrips(tmp_path):
    m = Manifest(tmp_path / "manifest.sqlite")
    assert m.journal_mode == "wal"
    from datetime import datetime

    from keiba_goal.netkeiba.http import JST
    m.record_fetch(RESULT, datetime(2026, 10, 29, 12, 0, tzinfo=JST), 200, b"abc", "netkeiba/result/x.gz")
    rec = m.get(RESULT.url)
    assert rec.kind == "result" and rec.bytes == 3 and rec.complete is None and rec.error is None
    m.mark(RESULT.url, True, "p1", None)
    rec = m.get(RESULT.url)
    assert rec.complete is True and rec.parser_version == "p1"
    with pytest.raises(KeyError):
        m.mark("https://db.netkeiba.com/race/000000000000/", True, "p1", None)


def test_fetch_caches_gzip_and_reuses_within_ttl(tmp_path):
    f, s, clock = make_fetcher(tmp_path, [FakeResponse(200, b"<html>1</html>")])
    p1 = f.get(RESULT)
    assert not p1.from_cache and p1.ok
    path = f.cache_path(RESULT)
    assert gzip.decompress(path.read_bytes()) == b"<html>1</html>"
    assert not list(path.parent.glob("*.tmp"))
    p2 = f.get(RESULT)  # 未判定だが TTL 内 → キャッシュ
    assert p2.from_cache and p2.content == p1.content
    assert len(s.calls) == 1
    assert f.manifest.request_count() == 1
    assert s.calls[0][1]["allow_redirects"] is False


def test_incomplete_page_is_refetched_after_ttl_but_complete_is_not(tmp_path):
    f, s, clock = make_fetcher(tmp_path, [FakeResponse(200, b"v1"), FakeResponse(200, b"v2")],
                               incomplete_ttl_hours=1)
    f.get(RESULT)
    f.mark(RESULT, complete=False, parser_version="t", error="payouts missing")
    clock.t += 1800
    assert f.get(RESULT).content == b"v1"
    clock.t += 3600
    p = f.get(RESULT)
    assert p.content == b"v2" and not p.from_cache
    assert f.manifest.get(RESULT.url).complete is None  # 取り直したら判定をやり直す
    f.mark(RESULT, complete=True, parser_version="t")
    clock.t += 10 * 86400
    p = f.get(RESULT)
    assert p.from_cache and p.complete is True
    assert len(s.calls) == 2


def test_live_pages_are_not_cached_but_can_be_snapshotted(tmp_path):
    odds = Target("odds", "202509040311")
    f, s, _ = make_fetcher(tmp_path, [FakeResponse(200, b'{"status":"middle"}'), FakeResponse(200, b"<html/>"),
                                      FakeResponse(200, b"<html/>")])
    f.get(odds, snapshot=True)
    f.get(PAST, live=True)
    f.get(PAST, live=True)
    assert len(s.calls) == 3
    assert f.manifest.get(odds.url) is None and f.manifest.get(PAST.url) is None
    assert not f.cache_path(PAST).exists()
    snaps = list((tmp_path / "snapshots" / "odds" / "202509040311").glob("*.gz"))
    assert len(snaps) == 1 and gzip.decompress(snaps[0].read_bytes()) == b'{"status":"middle"}'


def test_backoff_then_success(tmp_path):
    f, s, clock = make_fetcher(tmp_path, [FakeResponse(503, b""), FakeResponse(429, b""), FakeResponse(200, b"ok")],
                               backoff_base_sec=10)
    p = f.get(RESULT)
    assert p.content == b"ok"
    backoffs = [x for x in clock.sleeps if x > 2.5]
    assert backoffs == [10, 20]
    assert f.manifest.request_count() == 3


def test_stops_after_five_consecutive_failures(tmp_path):
    responses = [FakeResponse(403, b"")] * 3 + [requests.ConnectionError("x")] * 2
    f, s, _ = make_fetcher(tmp_path, responses, max_retries=3)
    with pytest.raises(FetchError) as e:
        f.get(RESULT)
    assert not isinstance(e.value, FetchBlocked) and (e.value.status, e.value.reason) == (403, "http 403 ×3")
    with pytest.raises(FetchBlocked):
        f.get(PAST)   # URL をまたいで 5 回目の失敗
    assert len(s.calls) == 5
    assert f.manifest.get(PAST.url) is None   # 止めた URL は記録しない (次の実行ですぐ取り直す)


def test_a_recorded_failure_is_served_from_the_manifest_until_the_ttl_passes(tmp_path):
    responses = [FakeResponse(500, b"")] * 3 + [FakeResponse(200, b"<html>ok</html>")]
    f, s, clock = make_fetcher(tmp_path, responses, max_retries=3, incomplete_ttl_hours=1)
    with pytest.raises(FetchError) as e:
        f.get(RESULT)
    assert f.manifest.get(RESULT.url) is None   # 記録するかは呼ぶ側 (クロール) が決める
    f.record_failure(RESULT, e.value)
    rec = f.manifest.get(RESULT.url)
    assert (rec.http_status, rec.complete, rec.error, rec.path) == (500, False, "http 500 ×3", None)
    cached = f.cached(RESULT)
    assert cached.status == 500 and not cached.ok and cached.error == "http 500 ×3"
    assert f.get(RESULT).from_cache and len(s.calls) == 3   # TTL 内は取り直さない
    clock.t += 3601
    p = f.get(RESULT)
    assert p.ok and not p.from_cache and len(s.calls) == 4
    rec = f.manifest.get(RESULT.url)
    assert rec.http_status == 200 and rec.complete is None and rec.error is None


def test_connection_errors_are_recorded_without_a_status(tmp_path):
    f, s, _ = make_fetcher(tmp_path, [requests.ConnectionError("down")] * 3, max_retries=3)
    with pytest.raises(FetchError) as e:
        f.get(RESULT)
    assert e.value.status is None and e.value.reason.startswith("通信エラー ×3")
    f.record_failure(RESULT, e.value)
    rec = f.manifest.get(RESULT.url)
    assert rec.http_status is None and rec.complete is False and rec.error.startswith("通信エラー ×3")


def test_not_found_is_final_and_recorded(tmp_path):
    f, s, _ = make_fetcher(tmp_path, [FakeResponse(404, b"")])
    p = f.get(RESULT)
    assert p.status == 404 and not p.ok
    assert f.manifest.get(RESULT.url).error == "http 404"
    assert f.get(RESULT).from_cache  # TTL 内は取り直さない
    assert len(s.calls) == 1


def test_corrupt_cache_is_ignored(tmp_path):
    f, s, _ = make_fetcher(tmp_path, [FakeResponse(200, b"good"), FakeResponse(200, b"again")])
    f.get(RESULT)
    f.mark(RESULT, complete=True, parser_version="t")
    f.cache_path(RESULT).write_bytes(gzip.compress(b"tampered"))
    assert f.get(RESULT).content == b"again"


def test_offline_mode_never_touches_network(tmp_path):
    f, s, clock = make_fetcher(tmp_path, [FakeResponse(200, b"v1")], incomplete_ttl_hours=1)
    f.get(RESULT)
    clock.t += 7200
    off = Fetcher(tmp_path, FetchSettings(), session=FakeSession([]), clock=clock, sleep=clock.sleep, offline=True)
    assert off.get(RESULT).content == b"v1"  # TTL 切れでもオフラインはキャッシュを返す
    with pytest.raises(FetchError):
        off.get(PAST)
    with pytest.raises(FetchError):
        off.get(Target("odds", "202509040311"))


def test_decode_uses_meta_charset_and_replaces_bad_bytes():
    euc = '<meta charset="EUC-JP">馬名'.encode("euc_jp") + b"\xff\xfe"
    text = decode(euc, "https://db.netkeiba.com/race/x/")
    assert "馬名" in text and "�" in text
    assert decode("出馬表".encode("utf-8"), "https://race.netkeiba.com/race/x") == "出馬表"
