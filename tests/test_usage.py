"""Tests for usage tracking: schema, aggregations, inference logging."""

import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from sellbird.state import StateManager


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        sm = StateManager(db_path)
        await sm.init_db()
        yield sm


# ── usage_events table + aggregations ──


@pytest.mark.asyncio
async def test_record_usage_event_and_totals(state):
    ok = await state.record_usage_event(
        agent="writer", task="write_sequence", session_id="sess-1",
        model="qwen3:1.7b", input_tokens=100, output_tokens=500,
        cache_read_tokens=2000, cache_creation_tokens=300,
        cost_usd=0.0, duration_ms=1200, num_turns=1,
    )
    assert ok

    totals = await state.usage_totals()
    assert totals["today"]["calls"] == 1
    assert totals["today"]["output_tokens"] == 500
    assert totals["week"]["input_tokens"] == 100


@pytest.mark.asyncio
async def test_usage_by_agent_groups_and_counts_sessions(state):
    # Two model-rows from the SAME call (same session) → one "call"
    await state.record_usage_event(
        agent="scout", session_id="sess-a", model="qwen3:1.7b",
        output_tokens=10, cost_usd=0.0,
    )
    await state.record_usage_event(
        agent="scout", session_id="sess-a", model="qwen3:1.7b",
        output_tokens=5, cost_usd=0.0,
    )
    await state.record_usage_event(
        agent="handler", session_id="sess-b", model="qwen3:1.7b",
        output_tokens=20, cost_usd=0.0,
    )

    rows = await state.usage_by_agent(days=7)
    by_name = {r["agent"]: r for r in rows}
    assert by_name["scout"]["calls"] == 1
    assert by_name["scout"]["output_tokens"] == 15
    assert by_name["handler"]["calls"] == 1


@pytest.mark.asyncio
async def test_request_key_dedup(state):
    first = await state.record_usage_event(
        request_key="msg_1:req_1", model="qwen3:1.7b",
        output_tokens=50, source="transcript",
    )
    second = await state.record_usage_event(
        request_key="msg_1:req_1", model="qwen3:1.7b",
        output_tokens=50, source="transcript",
    )
    assert first is True
    assert second is False

    totals = await state.usage_totals()
    assert totals["today"]["output_tokens"] == 50


@pytest.mark.asyncio
async def test_inference_log_table(state):
    import aiosqlite
    async with aiosqlite.connect(state.db_path) as db:
        await db.execute(
            """INSERT INTO inference_log (ts, agent, task, model, host, prompt_tokens, output_tokens, latency_ms, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("2026-11-23T12:00:00Z", "writer", "draft", "qwen3:1.7b", "localhost", 10, 20, 150, "ok")
        )
        await db.commit()
        async with db.execute("SELECT agent, model, status FROM inference_log") as cur:
            row = await cur.fetchone()
            assert row == ("writer", "qwen3:1.7b", "ok")
