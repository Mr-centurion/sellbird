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
    """Tests never pick up the operator's real ~/.claude Humanizer skill."""
    monkeypatch.setattr(sellbird.humanize, "SKILL_PATH", tmp_path / "no-humanizer.md")


# Every CLI command seeds the home folder first (sellbird.home). Tests that
# call cli.main() would create the checkout's data/, so seeding is a no-op
# here; tests of seeding itself call the real function.
@pytest.fixture(autouse=True)
def _no_home_seeding(monkeypatch):
    import sellbird.home

    monkeypatch.setattr(sellbird.home, "seed_home", lambda home, source: [])


from tests.fake_ollama import fake_ollama, FakeOllamaServer  # noqa: E402, F401
