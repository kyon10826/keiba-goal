"""netkeiba への GET (design §3.3 クローラ)。

- 全プロセス共通のレート制限: fcntl のファイルロック + 最終リクエスト時刻ファイル (間隔 ≥ 2.5 秒)。
- SQLite (WAL) の manifest: (url, kind, fetched_at, http_status, bytes, sha256, parser_version, complete, error)。
- gzip キャッシュは一時ファイルに書いてから rename。complete と判定されたページだけ再利用し、
  不完全なページは TTL を過ぎたら取り直す。ライブのページ (オッズ API・当日の出馬表) はキャッシュしない。
- 403/429/5xx と通信エラーは指数バックオフ。1 つの URL が max_retries 回失敗したら FetchError (status・reason 付き)。
  クロールは record_failure で manifest に不完全 (最後の HTTP 状態と理由) として記録して次へ進み、TTL 後に取り直す。
  リクエストの失敗が URL をまたいで max_consecutive_failures 回続いたら FetchBlocked で止める。
- 取得先は race.netkeiba.com と db.netkeiba.com だけ。GET のみ、リダイレクトは追わない、Cookie は保存しない。
"""

from __future__ import annotations

import codecs
import fcntl
import gzip
import hashlib
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from http.cookiejar import DefaultCookiePolicy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

from keiba_goal.netkeiba.files import save_bytes

JST = ZoneInfo("Asia/Tokyo")
ALLOWED_HOSTS = frozenset({"race.netkeiba.com", "db.netkeiba.com"})
MIN_INTERVAL_FLOOR = 2.5

URL_TEMPLATES: dict[str, str] = {
    "result": "https://db.netkeiba.com/race/{key}/",
    "shutuba_past": "https://race.netkeiba.com/race/shutuba_past.html?race_id={key}",
    "shutuba": "https://race.netkeiba.com/race/shutuba.html?race_id={key}",
    "race_list": "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={key}",
    "odds": "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={key}&type=1&action=update",
}
LIVE_ONLY_KINDS = frozenset({"odds"})


class FetchError(RuntimeError):
    """取得の失敗。status は最後の HTTP 状態 (通信エラーは None)、reason は manifest の error に入れる短い理由。"""

    def __init__(self, message: str, *, status: int | None = None, reason: str | None = None):
        super().__init__(message)
        self.status, self.reason = status, reason


class FetchBlocked(FetchError):
    """連続失敗の上限に達した (ブロックされた可能性)。クロールを止める。"""


@dataclass(frozen=True)
class Target:
    kind: str
    key: str  # 12 桁 race_id、race_list は YYYYMMDD

    def __post_init__(self) -> None:
        if self.kind not in URL_TEMPLATES:
            raise ValueError(f"未知の kind: {self.kind}")
        digits = 8 if self.kind == "race_list" else 12
        if not re.fullmatch(rf"\d{{{digits}}}", self.key):
            raise ValueError(f"{self.kind} のキーは {digits} 桁の数字: {self.key!r}")

    @property
    def url(self) -> str:
        return URL_TEMPLATES[self.kind].format(key=self.key)


@dataclass(frozen=True)
class Page:
    target: Target
    status: int
    content: bytes
    fetched_at: datetime
    from_cache: bool
    complete: bool | None = None
    error: str | None = None   # manifest の error (不完全な理由)

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.content)

    @property
    def text(self) -> str:
        return decode(self.content, self.target.url)


_META_CHARSET = re.compile(rb"charset=[\"']?([A-Za-z0-9_\-]+)", re.I)


def decode(content: bytes, url: str = "") -> str:
    """meta の charset (無ければホスト) で復号する。EUC-JP の範囲外の文字は置換文字にする。"""
    m = _META_CHARSET.search(content[:4096])
    enc = m.group(1).decode("ascii").lower() if m else ("euc_jp" if "db.netkeiba.com" in url else "utf-8")
    try:
        codecs.lookup(enc)
    except LookupError:
        enc = "utf-8"
    return content.decode(enc, errors="replace")


@dataclass(frozen=True)
class FetchSettings:
    min_interval_sec: float = MIN_INTERVAL_FLOOR
    timeout_sec: float = 30.0
    max_retries: int = 3
    backoff_base_sec: float = 10.0
    max_consecutive_failures: int = 5
    max_consecutive_invalid_pages: int = 10   # 200 だが検査に通らないページがこれだけ続いたらクロールを止める
    incomplete_ttl_hours: float = 6.0
    user_agent: str = "keiba-goal/0.1 (personal research)"

    @classmethod
    def from_section(cls, section: dict[str, Any]) -> FetchSettings:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in section.items() if k in known})


class RateLimiter:
    """ロックを持ったまま「前回のリクエスト終了 + 間隔」まで待ち、リクエストを 1 本だけ通す。"""

    def __init__(self, lock_path: Path, stamp_path: Path, min_interval: float,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        if min_interval < MIN_INTERVAL_FLOOR:
            raise ValueError(f"min_interval_sec は {MIN_INTERVAL_FLOOR} 秒以上: {min_interval}")
        self.lock_path, self.stamp_path = lock_path, stamp_path
        self.min_interval, self.clock, self.sleep = min_interval, clock, sleep

    def last_request_end(self) -> float:
        try:
            return float(self.stamp_path.read_text().strip() or 0.0)
        except (FileNotFoundError, ValueError):
            return 0.0

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                wait = self.last_request_end() + self.min_interval - self.clock()
                if wait > 0:
                    # 時計が戻った場合でも待ちは間隔 1 回分まで
                    self.sleep(min(wait, self.min_interval))
                yield
            finally:
                save_bytes(self.stamp_path, f"{self.clock():.6f}".encode())
                fcntl.flock(lock, fcntl.LOCK_UN)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    http_status INTEGER,
    bytes INTEGER,
    sha256 TEXT,
    path TEXT,
    parser_version TEXT,
    complete INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS pages_kind_key ON pages (kind, key);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    http_status INTEGER,
    elapsed_ms INTEGER,
    error TEXT
);
"""


@dataclass(frozen=True)
class PageRecord:
    url: str
    kind: str
    key: str
    fetched_at: datetime
    http_status: int | None
    bytes: int | None
    sha256: str | None
    path: str | None
    parser_version: str | None
    complete: bool | None
    error: str | None


class Manifest:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=60, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.executescript(_SCHEMA)

    @property
    def journal_mode(self) -> str:
        return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    def get(self, url: str) -> PageRecord | None:
        row = self._conn.execute(
            "SELECT url, kind, key, fetched_at, http_status, bytes, sha256, path, parser_version, complete, error "
            "FROM pages WHERE url = ?", (url,)).fetchone()
        return _record(row) if row else None

    def records(self, kind: str) -> list[PageRecord]:
        rows = self._conn.execute(
            "SELECT url, kind, key, fetched_at, http_status, bytes, sha256, path, parser_version, complete, error "
            "FROM pages WHERE kind = ? ORDER BY key", (kind,)).fetchall()
        return [_record(r) for r in rows]

    def record_fetch(self, target: Target, fetched_at: datetime, status: int, content: bytes,
                     path: str | None) -> None:
        error = None if status == 200 else f"http {status}"
        self._conn.execute(
            "INSERT INTO pages (url, kind, key, fetched_at, http_status, bytes, sha256, path, parser_version, complete, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?) "
            "ON CONFLICT(url) DO UPDATE SET fetched_at=excluded.fetched_at, http_status=excluded.http_status, "
            "bytes=excluded.bytes, sha256=excluded.sha256, path=excluded.path, parser_version=NULL, "
            "complete=NULL, error=excluded.error",
            (target.url, target.kind, target.key, fetched_at.isoformat(timespec="seconds"), status, len(content),
             hashlib.sha256(content).hexdigest(), path, error))

    def record_failure(self, target: Target, at: datetime, status: int | None, error: str) -> None:
        """取得に失敗した (再試行の上限)。前の内容は捨て、TTL 後に取り直す不完全なページとして残す。"""
        self._conn.execute(
            "INSERT INTO pages (url, kind, key, fetched_at, http_status, bytes, sha256, path, parser_version, complete, error) "
            "VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, 0, ?) "
            "ON CONFLICT(url) DO UPDATE SET fetched_at=excluded.fetched_at, http_status=excluded.http_status, "
            "bytes=0, sha256=NULL, path=NULL, parser_version=NULL, complete=0, error=excluded.error",
            (target.url, target.kind, target.key, at.isoformat(timespec="seconds"), status, error))

    def mark(self, url: str, complete: bool, parser_version: str, error: str | None) -> None:
        cur = self._conn.execute("UPDATE pages SET complete = ?, parser_version = ?, error = ? WHERE url = ?",
                                 (int(complete), parser_version, error, url))
        if cur.rowcount != 1:
            raise KeyError(f"manifest に無い URL: {url}")

    def log_request(self, url: str, requested_at: datetime, status: int | None, elapsed_ms: int,
                    error: str | None) -> None:
        self._conn.execute(
            "INSERT INTO requests (url, requested_at, http_status, elapsed_ms, error) VALUES (?, ?, ?, ?, ?)",
            (url, requested_at.isoformat(timespec="seconds"), status, elapsed_ms, error))

    def request_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _record(row: tuple) -> PageRecord:
    vals = list(row)
    vals[3] = datetime.fromisoformat(vals[3])
    vals[9] = None if vals[9] is None else bool(vals[9])
    return PageRecord(*vals)


def _new_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Language": "ja,en;q=0.5"})
    s.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
    return s


def _is_retryable(status: int | None) -> bool:
    return status is None or status in (403, 429) or status >= 500


class Fetcher:
    def __init__(self, raw_dir: Path, settings: FetchSettings, *, session: Any = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 offline: bool = False):
        self.raw_dir = raw_dir
        self.settings = settings
        self.cache_dir = raw_dir / "netkeiba"
        self.snapshot_dir = raw_dir / "snapshots"
        self.manifest = Manifest(raw_dir / "manifest.sqlite")
        self.limiter = RateLimiter(raw_dir / "netkeiba.lock", raw_dir / "netkeiba.last",
                                   settings.min_interval_sec, clock, sleep)
        self.session = session if session is not None else _new_session(settings.user_agent)
        self.clock, self.sleep, self.offline = clock, sleep, offline
        self.network_requests = 0
        self._consecutive_failures = 0

    @classmethod
    def from_config(cls, cfg: Any, **kw: Any) -> Fetcher:
        return cls(cfg.paths.raw, FetchSettings.from_section(cfg.section("netkeiba")), **kw)

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), JST)

    def cache_path(self, target: Target) -> Path:
        return self.cache_dir / target.kind / target.key[:4] / f"{target.key}.gz"

    def get(self, target: Target, *, live: bool = False, snapshot: bool = False) -> Page:
        """キャッシュ規則に従って 1 ページ返す。live=True はキャッシュを読まず・書かない。"""
        if live or target.kind in LIVE_ONLY_KINDS:
            if self.offline:
                raise FetchError(f"オフラインではライブのページを取得できない: {target.url}")
            page = self._download(target)
            if snapshot and page.ok:
                self._save_snapshot(page)
            return page
        cached = self.cached(target)
        if cached is not None and (self.offline or self._reusable(cached)):
            return cached
        if self.offline:
            raise FetchError(f"オフライン: キャッシュが無い {target.url}")
        page = self._download(target)
        self._store(page)
        return page

    def cached(self, target: Target) -> Page | None:
        """manifest とキャッシュにあるもの (不完全・HTTP エラーを含む)。壊れたファイルは無いものとして扱う。"""
        rec = self.manifest.get(target.url)
        if rec is None:
            return None
        if rec.http_status != 200:
            return Page(target, rec.http_status or 0, b"", rec.fetched_at, True, rec.complete, rec.error)
        path = self.raw_dir / rec.path if rec.path else self.cache_path(target)
        try:
            content = gzip.decompress(path.read_bytes())
        except (FileNotFoundError, OSError, EOFError):
            return None
        if hashlib.sha256(content).hexdigest() != rec.sha256:
            return None
        return Page(target, 200, content, rec.fetched_at, True, rec.complete, rec.error)

    def mark(self, target: Target, *, complete: bool, parser_version: str, error: str | None = None) -> None:
        self.manifest.mark(target.url, complete, parser_version, error)

    def record_failure(self, target: Target, error: FetchError) -> None:
        """再試行の上限まで失敗したページを不完全として残す (get は TTL を過ぎるまでこの失敗を返す)。"""
        self.manifest.record_failure(target, self.now(), error.status, error.reason or str(error))

    def _reusable(self, page: Page) -> bool:
        if page.complete:
            return True
        ttl = timedelta(hours=self.settings.incomplete_ttl_hours)
        return self.now() - page.fetched_at < ttl

    def _download(self, target: Target) -> Page:
        url = target.url
        if urlparse(url).hostname not in ALLOWED_HOSTS:
            raise ValueError(f"許可されていないホスト: {url}")
        status: int | None = None
        error: str | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            with self.limiter.slot():
                requested_at = self.now()
                t0 = time.monotonic()
                try:
                    resp = self.session.get(url, timeout=self.settings.timeout_sec, allow_redirects=False)
                    status, content, error = resp.status_code, resp.content, None
                except requests.RequestException as e:
                    status, content, error = None, b"", repr(e)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
            self.network_requests += 1
            self.manifest.log_request(url, requested_at, status, elapsed_ms, error)
            if not _is_retryable(status):
                self._consecutive_failures = 0
                return Page(target, status, content, requested_at, False)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.settings.max_consecutive_failures:
                raise FetchBlocked(f"{self._consecutive_failures} 回連続で失敗したので停止 (最後: {url} "
                                   f"status={status} error={error})")
            if attempt < self.settings.max_retries:
                self.sleep(self.settings.backoff_base_sec * 2 ** (attempt - 1))
        n = self.settings.max_retries
        reason = f"http {status} ×{n}" if status is not None else f"通信エラー ×{n} ({error})"
        raise FetchError(f"{url}: {reason}", status=status, reason=reason)

    def _store(self, page: Page) -> None:
        rel = None
        if page.status == 200:
            path = self.cache_path(page.target)
            save_bytes(path, gzip.compress(page.content, mtime=0))
            rel = str(path.relative_to(self.raw_dir))
        self.manifest.record_fetch(page.target, page.fetched_at, page.status, page.content, rel)

    def _save_snapshot(self, page: Page) -> Path:
        stamp = page.fetched_at.strftime("%Y%m%dT%H%M%S")
        path = self.snapshot_dir / page.target.kind / page.target.key / f"{stamp}.gz"
        save_bytes(path, gzip.compress(page.content, mtime=0))
        return path
