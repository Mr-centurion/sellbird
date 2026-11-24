"""Security defaults and fallback crawler tests."""

from types import SimpleNamespace
import pytest
import pytest_asyncio
from sellbird.paths import PROJECT_ROOT
from sellbird.state import StateManager


@pytest_asyncio.fixture
async def state(tmp_path):
    state = StateManager(str(tmp_path / "test.db"))
    await state.init_db()
    return state


@pytest.mark.asyncio
async def test_training_on_a_local_url_says_why_it_was_refused():
    from sellbird.trainer import FallbackCrawler
    crawler = FallbackCrawler()
    assert await crawler.crawl("http://localhost:3000", max_pages=3) == {}
    assert crawler.blocked is True


def test_status_still_works_when_the_config_is_invalid(tmp_path, monkeypatch, capsys):
    # `sellbird status` is what people run when something's wrong, so an
    # invalid sellbird.yaml must not crash the sending-hold check.
    import sellbird.config
    import sellbird.state
    from sellbird.cli import cmd_status
    from sellbird.config import SellbirdConfig
    from pydantic import ValidationError

    def invalid():
        SellbirdConfig(channels={"email": {"max_daily_sends": "lots"}})

    try:
        invalid()
    except ValidationError:
        pass
    real_state = sellbird.state.StateManager
    monkeypatch.setattr(sellbird.state, "StateManager", lambda: real_state(str(tmp_path / "s.db")))
    monkeypatch.setattr(sellbird.config, "load_config", invalid)
    cmd_status(SimpleNamespace(check=False))
    assert "Prospects:" in capsys.readouterr().out
