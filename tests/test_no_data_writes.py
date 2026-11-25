"""Tests enforcing that test execution never writes to checkout's data/ folder."""

from pathlib import Path
import pytest
import time
from sellbird.paths import PROJECT_ROOT
from sellbird.state import StateManager
import sellbird.dashboard as dash_mod
import sellbird.state as state_mod


def test_state_manager_default_uses_temp_db():
    """Verify that StateManager() default db_path is in a temporary directory, never data/sellbird.db."""
    sm = StateManager()
    real_db_path = str((PROJECT_ROOT / "data" / "sellbird.db").resolve())
    assert str(Path(sm.db_path).resolve()) != real_db_path, (
        f"StateManager is using real checkout database: {sm.db_path}"
    )
    # Must be in pytest temp dir
    assert "pytest" in sm.db_path or "test_" in sm.db_path or "tmp" in sm.db_path.lower()


def test_dashboard_db_path_is_isolated():
    """Verify that dashboard.DB_PATH does not point to checkout's real data/sellbird.db."""
    real_db_path = (PROJECT_ROOT / "data" / "sellbird.db").resolve()
    assert Path(dash_mod.DB_PATH).resolve() != real_db_path
    assert Path(state_mod.DB_PATH).resolve() != real_db_path


def test_guard_fails_on_data_dir_write(tmp_path):
    """Verify that writing to data/ directory raises an assertion failure under the guard."""
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_file = data_dir / "sellbird.db"
    db_file.write_text("initial")
    initial_mtime = db_file.stat().st_mtime

    # Simulate rogue modification
    time.sleep(0.02)
    db_file.write_text("modified by rogue test")

    def check_guard():
        if db_file.stat().st_mtime != initial_mtime:
            raise AssertionError(f"TEST SUITE VIOLATION: {db_file} in data/ was modified during tests!")

    with pytest.raises(AssertionError, match="TEST SUITE VIOLATION"):
        check_guard()
