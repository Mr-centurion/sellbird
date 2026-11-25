"""Shared test setup.

# Unit tests run without real models or network calls.
"""

import os

# An exported SELLBIRD_HOME (manual checks) must not move the home that
# sellbird.paths resolves at import, so drop it before any sellbird import.
# Tests that need it pass env= to resolve_home or use monkeypatch.
os.environ.pop("SELLBIRD_HOME", None)

import pytest  # noqa: E402


def pytest_configure(config):
    pass


@pytest.fixture(autouse=True)
def _private_dashboard_token(tmp_path_factory):
    """Never create or read the checkout's real data/dashboard-token."""
    import sellbird.dashboard as dash

    dash.token_store.configure(tmp_path_factory.mktemp("token") / "dashboard-token")
    yield
    dash.token_store.configure(dash.TOKEN_FILE)


import sellbird.humanize  # noqa: E402


@pytest.fixture(autouse=True)
def _no_installed_humanizer(monkeypatch, tmp_path):
    """Tests never pick up the operator's real Humanizer skill."""
    monkeypatch.setattr(sellbird.humanize, "SKILL_PATH", tmp_path / "no-humanizer.md")


# Every CLI command seeds the home folder first (sellbird.home). Tests that
# call cli.main() would create the checkout's data/, so seeding is a no-op
# here; tests of seeding itself call the real function.
@pytest.fixture(autouse=True)
def _no_home_seeding(monkeypatch):
    import sellbird.home

    monkeypatch.setattr(sellbird.home, "seed_home", lambda home, source: [])


from pathlib import Path
from sellbird.paths import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(autouse=True)
def _isolate_test_database(tmp_path_factory, monkeypatch):
    """Ensure every test runs against a temporary database, never data/sellbird.db."""
    temp_dir = tmp_path_factory.mktemp("test_home")
    temp_data = temp_dir / "data"
    temp_data.mkdir(parents=True, exist_ok=True)
    temp_db = temp_data / "test_sellbird.db"

    import sellbird.state as state_mod
    import sellbird.dashboard as dash_mod
    import sellbird.ollama_runtime as runtime_mod

    monkeypatch.setattr(state_mod, "DB_PATH", temp_db)
    monkeypatch.setattr(dash_mod, "DB_PATH", temp_db)
    monkeypatch.setattr(dash_mod, "AGENT_TOKEN_FILE", temp_data / "agent-token")
    monkeypatch.setattr(runtime_mod, "DB_PATH", temp_db)

    yield temp_db


@pytest.fixture(scope="session", autouse=True)
def _assert_no_data_writes_during_session():
    """Fail if the test suite writes or modifies any database or runtime file in data/."""
    initial_mtimes = {}
    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*"):
            if p.is_file():
                try:
                    initial_mtimes[str(p.resolve())] = p.stat().st_mtime
                except OSError:
                    pass

    yield

    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*"):
            if p.is_file():
                path_str = str(p.resolve())
                if p.name in ("sellbird.db", "sellbird.db-wal", "sellbird.db-shm", "offline_test.db"):
                    if path_str in initial_mtimes:
                        mtime_now = p.stat().st_mtime
                        if mtime_now != initial_mtimes[path_str]:
                            raise AssertionError(
                                f"TEST SUITE VIOLATION: {p} in data/ was modified during tests! "
                                f"Tests must only use temporary databases."
                            )
                    else:
                        raise AssertionError(
                            f"TEST SUITE VIOLATION: New database {p} was created in data/ during tests! "
                            f"Tests must only use temporary databases."
                        )


from tests.fake_ollama import fake_ollama, FakeOllamaServer  # noqa: E402, F401
