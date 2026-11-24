"""Tests for the pattern-first email pipeline."""

import os
import tempfile

import pytest
import pytest_asyncio

from sellbird.state import StateManager
from sellbird.integrations import email_finder as ef
from sellbird.integrations.email_finder import (
    EmailResult,
    build_email,
    classify_mx,
    derive_pattern,
    find_email,
    generate_patterns,
    infer_pattern_from_email,
)


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


class Env:
    def __init__(self, **kw):
        pass


# ── pattern building / inference ──


def test_build_email_patterns():
    assert build_email("{f}.{l}", "Jane", "Doe", "acme.com") == "jane.doe@acme.com"
    assert build_email("{fi}{l}", "Jane", "Doe", "acme.com") == "jdoe@acme.com"
    assert build_email("{f}", "Jane", "Doe", "acme.com") == "jane@acme.com"
    assert build_email("{f}.{l}", "", "Doe", "acme.com") == ""
    assert build_email("nonexistent", "Jane", "Doe", "acme.com") == ""


def test_generate_patterns_dedup_and_order():
    patterns = generate_patterns("Jane", "Doe", "acme.com")
    assert patterns[0] == "jane.doe@acme.com"   # most common first
    assert "jdoe@acme.com" in patterns
    assert len(patterns) == len(set(patterns))


def test_infer_pattern_from_email():
    assert infer_pattern_from_email("jane.doe@acme.com", "Jane", "Doe") == "{f}.{l}"
    assert infer_pattern_from_email("jdoe@acme.com", "Jane", "Doe") == "{fi}{l}"
    assert infer_pattern_from_email("janedoe@acme.com", "Jane", "Doe") == "{f}{l}"
    assert infer_pattern_from_email("random@acme.com", "Jane", "Doe") is None


# ── MX classification ──


def test_classify_mx():
    assert classify_mx("aspmx.l.google.com") == "google"
    assert classify_mx("acme-com.mail.protection.outlook.com") == "microsoft"
    assert classify_mx("mx.pphosted.com") == "gateway"
    assert classify_mx("mail.smallco.com") == "other"
    assert classify_mx("") == "none"


# ── pattern derivation with cache + known emails ──


@pytest.mark.asyncio
async def test_derive_pattern_from_known_email(state):
    pattern, source, conf = await derive_pattern(
        "acme.com",
        known_emails=[("john.smith@acme.com", "John", "Smith")],
        state=state,
    )
    assert pattern == "{f}.{l}"
    assert source == "known_email"
    assert conf >= 0.9


@pytest.mark.asyncio
async def test_derive_pattern_uses_cache(state):
    await state.save_email_pattern(
        "acme.com", pattern="{fi}{l}", source="scraped_mailto", confidence=0.85
    )
    pattern, source, conf = await derive_pattern("acme.com", state=state)
    assert pattern == "{fi}{l}"
    assert source == "scraped_mailto"


@pytest.mark.asyncio
async def test_derive_pattern_default_no_signal(state):
    pattern, source, conf = await derive_pattern("unknown-co.com", state=state)
    assert pattern == "{f}.{l}"
    assert source == "default"
    assert conf < 0.5


# ── find_email end-to-end (mocked network) ──


@pytest.mark.asyncio
async def test_find_email_no_mx_is_invalid(state, monkeypatch):
    async def no_mx(domain):
        return None
    monkeypatch.setattr(ef, "get_mx_host", no_mx)
    result = await find_email("Jane", "Doe", "acme.com", state=state)
    assert result.status == "invalid"
    assert result.mx_type == "none"


@pytest.mark.asyncio
async def test_find_email_verify_false_is_guess(state, monkeypatch):
    async def google_mx(domain):
        return "aspmx.l.google.com"
    monkeypatch.setattr(ef, "get_mx_host", google_mx)

    result = await find_email("Jane", "Doe", "acme.com", verify=False, state=state)
    assert result.status == "guess"
    assert result.email == "jane.doe@acme.com"
    assert result.mx_type == "google"
    # Pattern still cached even without verification
    cached = await state.get_email_pattern("acme.com")
    assert cached["mx_type"] == "google"


@pytest.mark.asyncio
async def test_find_email_inconclusive_is_guess_not_verified(state, monkeypatch):
    """No verifier confirmation → 'guess', never 'verified'."""
    async def google_mx(domain):
        return "aspmx.l.google.com"
    monkeypatch.setattr(ef, "get_mx_host", google_mx)
    result = await find_email("Jane", "Doe", "acme.com", env=Env(), state=state)
    assert result.status == "guess"
    assert not result.verified
