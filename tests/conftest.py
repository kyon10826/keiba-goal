"""実データ由来の fixture (配布しない) が手元に無いとき、それを読むテストを理由付きで skip にする。

tests/fixtures/ の netkeiba の保存ページ・record_data・時系列オッズの抜粋とそれから作った parquet は、
再配布できないのでリポジトリに含めない (.gitignore)。手元のデータから作り直す方法は README を参照。
合成 fixture だけで動くテストは、どの環境でも実行される。
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REAL_FIXTURE_MARKERS = (
    FIXTURES / "data" / "jd",
    FIXTURES / "data" / "record",
    FIXTURES / "netkeiba",
    FIXTURES / "features",
)


def real_fixtures_present() -> bool:
    return all(p.is_dir() and any(f.name != ".gitkeep" and "synthetic" not in f.name for f in p.rglob("*") if f.is_file())
               for p in REAL_FIXTURE_MARKERS)


REAL_PRESENT = real_fixtures_present()
SKIP_REASON = "実データ由来の fixture が無い (再配布できないため非同梱。README の「テスト」を参照)"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if REAL_PRESENT or not rep.failed or call.excinfo is None:
        return
    if _caused_by_missing_fixture(call.excinfo.value):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {SKIP_REASON}")


def _caused_by_missing_fixture(exc: BaseException | None) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, FileNotFoundError | NotADirectoryError):
            return True
        if "fixtures" in str(exc) and ("No such file" in str(exc) or "not found" in str(exc).lower()):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def pytest_report_header(config):
    return f"real-data fixtures: {'present' if REAL_PRESENT else 'absent (tests that need them are skipped)'}"


def pytest_configure(config):
    config.addinivalue_line("markers", "real_fixtures: 実データ由来の fixture が無ければ skip する")


def pytest_collection_modifyitems(config, items):
    if REAL_PRESENT:
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if "real_fixtures" in item.keywords:
            item.add_marker(skip)
