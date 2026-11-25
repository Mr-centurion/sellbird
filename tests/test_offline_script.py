"""Tests for scripts/offline_test.py and scripts/no_cloud_check.sh.

Verifies:
- All 8 workflows run offline against fake_ollama via `--use-fake`.
- Air-gap socket guard actively blocks non-loopback network connections.
- Markdown audit report is created with all 8 workflows and inference log entries.
- scripts/no_cloud_check.sh passes with 0 cloud violations.
"""

import asyncio
from pathlib import Path
import socket
import subprocess
import pytest

from scripts.offline_test import (
    install_airgap_socket_guard,
    run_offline_test,
    verify_socket_guard_active,
)
from sellbird.paths import PROJECT_ROOT


def test_airgap_socket_guard_enforcement():
    """Verify socket guard allows loopback but raises RuntimeError on external targets."""
    install_airgap_socket_guard()
    assert verify_socket_guard_active() is True

    # Attempting an external IP must raise RuntimeError
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="AIR-GAP VIOLATION"):
            s.connect(("1.1.1.1", 443))
    finally:
        s.close()


@pytest.mark.asyncio
async def test_offline_test_runner_use_fake(tmp_path):
    """Run all 8 workflows end-to-end against fake_ollama and verify report generation."""
    test_db = tmp_path / "offline_test.db"
    report_dir = tmp_path / "reports"

    ret = await run_offline_test(
        use_fake=True,
        db_path=str(test_db),
        report_dir=str(report_dir),
        model_override="qwen3:1.7b",
    )
    assert ret == 0, "offline_test.py --use-fake must return exit code 0"

    # Verify report created
    reports = list(report_dir.glob("offline_report_*.md"))
    assert len(reports) == 1, "Expected exactly one markdown report generated"
    report_text = reports[0].read_text(encoding="utf-8")

    # Verify all 8 workflows appear in report as PASS
    assert "FAKE SERVER, NOT EVIDENCE" in report_text
    assert "Business Profile" in report_text
    assert "ICP Generation" in report_text
    assert "Stored Research" in report_text
    assert "Buying Signals" in report_text
    assert "Problem Statements" in report_text
    assert "Email Drafting" in report_text
    assert "Lead Scoring" in report_text
    assert "Reply Classification" in report_text
    assert "**PASS**" in report_text

    # Verify hardware and ollama sections
    assert "Hardware Detection & Profile Recommendation" in report_text
    assert "Local Ollama Runtime & Model Status" in report_text
    assert "Model Returned by Ollama API" in report_text
    assert "SQLite Inference Log Audit" in report_text
    assert "Inferences with Cloud/External Host**: 0" in report_text


def test_no_cloud_check_script():
    """Execute scripts/no_cloud_check.sh and ensure exit code 0."""
    script = PROJECT_ROOT / "scripts" / "no_cloud_check.sh"
    assert script.exists(), "no_cloud_check.sh must exist"

    res = subprocess.run([str(script)], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res.returncode == 0, f"no_cloud_check.sh failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "PASS:" in res.stdout
    assert "Zero cloud SDK imports" in res.stdout
