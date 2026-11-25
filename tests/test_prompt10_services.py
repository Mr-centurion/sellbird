"""Tests for Prompt 10 Services: Product Scoring, Validated Email Drafter, and 8-Label Reply Classification.

RULES: No external LLM or cloud calls. All tests run locally against fake_ollama only.
"""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import sellbird.dashboard as dash
from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.email_drafter import (
    count_calls_to_action,
    draft_validated_email,
    extract_numbers,
    mentions_top_signal,
    validate_email_draft,
)
from sellbird.models.prospect import Prospect
from sellbird.ollama_runtime import OllamaUnavailable
from sellbird.paths import PROJECT_ROOT
from sellbird.reply_classifier import (
    classify_reply,
    classify_reply_heuristic,
    VALID_LABELS,
)
from sellbird.scoring import (
    calculate_score_breakdown,
    score_prospect_for_product,
)
from sellbird.state import StateManager
from tests.fake_ollama import FakeOllamaServer


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Authenticated TestClient on an isolated database."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "data" / "sellbird.db"
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "DB_PATH", db_path)
    monkeypatch.setattr(dash, "PID_FILE", tmp_path / "data" / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", tmp_path / "data" / "sellbird.log")
    monkeypatch.setattr(dash, "WEB_DIR", PROJECT_ROOT / "sellbird" / "web")

    dash.token_store.configure(tmp_path / "data" / "dashboard-token")
    token = dash.token_store.get()

    asyncio.run(StateManager(str(db_path)).init_db())

    with TestClient(
        dash.app,
        base_url="http://127.0.0.1:5555",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        client.db_path = db_path
        yield client


# ── 1. Scoring & Breakdown Maths Tests ────────────────────────────────


@pytest.mark.asyncio
async def test_score_breakdown_maths(monkeypatch):
    """Verify component points add up to maximum 100 points and validate rule logic."""
    monkeypatch.setattr("sellbird.scoring.compute_embedding", AsyncMock(return_value=[0.1] * 384))

    icp = {
        "titles": ["VP Supply Chain Operations", "Head of Logistics"],
        "industries": ["Logistics", "Freight"],
        "company_size": "50-500",
        "buyer_persona": "Operations executives burdened by manual fulfillment workflows.",
        "pain_points": ["Manual shipment tracking", "Inventory discrepancies across ERP"],
    }
    product = {
        "id": "acme_flow",
        "name": "Acme Flow",
        "description": "Workflow automation engine for supply chain and warehouse fulfillment.",
        "value_props": "Reduces manual order processing by 75%.",
    }
    signals = [
        {"label": "expansion", "evidence_quote": "opening of two new fulfillment hubs in Chicago and Dallas"},
        {"label": "leadership_change", "evidence_quote": "Marcus Vance joins as VP"},
        {"label": "tech_adoption", "evidence_quote": "migrating off legacy ERP"},
    ]

    # Perfect match prospect
    perfect_prospect = Prospect(
        id="p-01",
        company="Apex Logistics",
        title="VP Supply Chain Operations",
        industry="Logistics",
        size="250 employees",
        company_size="250 employees",
        stored_text="Apex Logistics is expanding nationwide fulfillment and warehousing hubs.",
    )

    bd = await calculate_score_breakdown(perfect_prospect, product, icp, signals)

    # 1. Title match: 20 points
    assert bd["title_match"] == 20
    # 2. Industry match: 20 points
    assert bd["industry_match"] == 20
    # 3. Company size match: 15 points
    assert bd["company_size_match"] == 15
    # 4. Buying signals: 20 points (min(20, 3 * 7) = 20)
    assert bd["buying_signals"] == 20
    # 5. Embedding similarity: 0-25 points
    assert 0 <= bd["embedding_similarity"] <= 25

    # Check maximum possible score is 100
    max_score = 20 + 20 + 15 + 20 + 25
    assert max_score == 100
    assert 0 <= bd["total"] <= 100

    # Non-matching prospect
    mismatch_prospect = Prospect(
        id="p-02",
        company="Tiny Bakery",
        title="Pastry Chef",
        industry="Food & Beverage",
        size="5",
        company_size="5",
        stored_text="Artisanal pastries and wedding cakes.",
    )
    bd_low = await calculate_score_breakdown(mismatch_prospect, product, icp, [])
    assert bd_low["title_match"] == 0
    assert bd_low["industry_match"] == 0
    assert bd_low["company_size_match"] == 0
    assert bd_low["buying_signals"] == 0
    assert bd_low["total"] < 30


@pytest.mark.asyncio
async def test_score_prospect_for_product_with_fake_ollama(tmp_path, monkeypatch):
    """Test score_prospect_for_product with fake Ollama LLM rationale."""
    monkeypatch.setattr("sellbird.scoring.compute_embedding", AsyncMock(return_value=[0.1] * 384))

    db_path = str(tmp_path / "test_scoring.db")
    state = StateManager(db_path)
    await state.init_db()

    # Seed product & prospect
    await state.add_product(product_id="prod_flow", name="Acme Flow", description="Logistics automation")
    await state.save_icp(
        product_id="prod_flow",
        icp_data={"titles": ["VP Operations"], "industries": ["Logistics"], "company_size": "50-500"},
    )
    p = Prospect(
        id="prospect_1",
        first_name="Marcus",
        last_name="Vance",
        company="Apex Logistics",
        title="VP Operations",
        industry="Logistics",
        size="200",
        stored_text="Apex Logistics modernizing distribution centers.",
    )
    await state.save_prospect(p)

    def router(prompt, body):
        return {
            "rationale": "Direct executive title match and logistics expansion create high operational fit.",
            "top_reasons": ["Executive role match", "Logistics industry alignment", "Active modernization need"],
            "main_risk": "Existing procurement backlog",
        }

    with FakeOllamaServer() as server:
        server.set_reply_fn(router)
        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))
        res = await score_prospect_for_product("prospect_1", "prod_flow", state=state, brain=brain)

        assert res["score"] > 0
        assert res["mode"] == "llm"
        assert len(res["rationale"].split()) <= 40
        assert len(res["top_reasons"]) == 3
        assert res["main_risk"] == "Existing procurement backlog"

        # Verify persisted in database
        scores = await state.get_scores_by_product("prod_flow")
        assert len(scores) == 1
        assert scores[0]["prospect_id"] == "prospect_1"
        assert scores[0]["score"] == res["score"]


@pytest.mark.asyncio
async def test_score_prospect_for_product_non_llm_mode(tmp_path, monkeypatch):
    """Verify fallback to non-LLM mode rationale when Ollama is unavailable."""
    monkeypatch.setattr("sellbird.scoring.compute_embedding", AsyncMock(return_value=[0.1] * 384))

    db_path = str(tmp_path / "test_scoring_down.db")
    state = StateManager(db_path)
    await state.init_db()

    await state.add_product(product_id="prod_flow", name="Acme Flow")
    p = Prospect(id="p1", first_name="A", last_name="B", company="Alpha Corp", title="Director")
    await state.save_prospect(p)

    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b"))
    brain.unavailable = "Ollama connection refused"

    res = await score_prospect_for_product("p1", "prod_flow", state=state, brain=brain)
    assert res["mode"] == "non-LLM mode"
    assert "non-LLM mode" in res["rationale"]
    assert len(res["rationale"].split()) <= 40


# ── 2. Validated Email Drafter Tests ──────────────────────────────────


def test_email_validation_all_rules():
    """Verify all 4 email validation rules: word count, signal mention, single CTA, grounded numbers."""
    top_signal = {
        "label": "expansion",
        "evidence_quote": "Apex Logistics announces the opening of two new fulfillment hubs in Chicago and Dallas.",
    }
    inputs_text = "Apex Logistics Marcus Vance VP Supply Chain 35% growth opening of two new fulfillment hubs"

    # Valid draft (no ungrounded numbers)
    valid_body = (
        "Hi Marcus,\n\n"
        "Noticed Apex Logistics is expanding with new fulfillment hubs to support volume growth. "
        "Rapid facility expansion often strains dispatch operations.\n\n"
        "Acme Flow automates order routing and eliminates manual bottlenecks.\n\n"
        "Would you be open to a brief walkthrough next week?\n\n"
        "Best,\nAlex"
    )
    ok, err = validate_email_draft(valid_body, top_signal, inputs_text)
    assert ok is True, f"Valid draft failed: {err}"
    assert err == ""

    # Rule 1: Body >= 110 words fails
    long_body = "word " * 115 + "expanding? Next week?"
    ok, err = validate_email_draft(long_body, top_signal, inputs_text)
    assert ok is False
    assert "under 110 words" in err

    # Rule 2: Top signal missing fails
    no_signal_body = (
        "Hi Marcus,\n\n"
        "We build logistics automation tools for operations leaders.\n\n"
        "Would you be open to a brief chat next Tuesday?\n\n"
        "Best,\nAlex"
    )
    ok, err = validate_email_draft(no_signal_body, top_signal, inputs_text)
    assert ok is False
    assert "does not mention the top signal" in err

    # Rule 3: No CTA fails; Multiple CTAs fail
    no_cta_body = (
        "Hi Marcus,\n\n"
        "Noticed Apex Logistics is expanding with new hubs.\n\n"
        "We help streamline warehouse operations.\n\n"
        "Best,\nAlex"
    )
    ok, err = validate_email_draft(no_cta_body, top_signal, inputs_text)
    assert ok is False
    assert "no call to action" in err

    multi_cta_body = (
        "Hi Marcus,\n\n"
        "Noticed Apex Logistics is expanding with new hubs.\n\n"
        "Are you available Tuesday? Or would Thursday work better?\n\n"
        "Best,\nAlex"
    )
    ok, err = validate_email_draft(multi_cta_body, top_signal, inputs_text)
    assert ok is False
    assert "calls to action" in err

    # Rule 4: Ungrounded number fails
    hallucinated_number_body = (
        "Hi Marcus,\n\n"
        "Noticed Apex Logistics is expanding with new hubs. "
        "We helped similar logistics companies cut costs by 99%.\n\n"
        "Would you be open to a brief conversation next week?\n\n"
        "Best,\nAlex"
    )
    # 99 is not in inputs_text!
    ok, err = validate_email_draft(hallucinated_number_body, top_signal, inputs_text)
    assert ok is False
    assert "contains numbers not found in input data" in err


@pytest.mark.asyncio
async def test_draft_validated_email_retry(tmp_path):
    """Test retry mechanism when attempt 1 violates validation rules and attempt 2 corrects."""
    db_path = str(tmp_path / "test_email_retry.db")
    state = StateManager(db_path)
    await state.init_db()

    await state.add_product(product_id="prod_1", name="Acme Flow", description="Automation")
    await state.save_icp("prod_1", {"buyer_persona": "Ops"})
    p = Prospect(
        id="prospect_apex",
        first_name="Marcus",
        last_name="Vance",
        company="Apex Logistics",
        title="VP Operations",
        stored_text="Apex Logistics expanding new hubs in Chicago.",
    )
    await state.save_prospect(p)
    await state.save_prospect_signal(
        prospect_id="prospect_apex",
        product_id="prod_1",
        label="expansion",
        confidence=0.9,
        evidence_quote="Apex Logistics expanding new hubs in Chicago.",
    )

    attempt_counter = 0

    def router(prompt, body):
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            # First attempt violates CTA rule (2 questions)
            return {
                "subject": "Question",
                "body": "Hi Marcus, expanding hubs? Are you free tomorrow? Would next week work?",
            }
        else:
            # Second attempt corrects strictly
            return {
                "subject": "Warehouse dispatching at Apex Logistics",
                "body": (
                    "Hi Marcus,\n\n"
                    "Noticed Apex Logistics is expanding with new hubs in Chicago. "
                    "We help automate dispatch workflows.\n\n"
                    "Would you be open to a brief chat next Tuesday?\n\n"
                    "Best,\nAlex"
                ),
            }

    with FakeOllamaServer() as server:
        server.set_reply_fn(router)
        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))
        draft = await draft_validated_email("prospect_apex", "prod_1", state=state, brain=brain)

        assert draft is not None
        assert draft["subject"] == "Warehouse dispatching at Apex Logistics"
        assert attempt_counter == 2, "Expected exactly 2 attempts due to retry"

        # Verify saved in outbox / emails table
        saved_drafts = await state.get_email_drafts(product_id="prod_1")
        assert len(saved_drafts) == 1
        assert saved_drafts[0]["product_id"] == "prod_1"
        assert saved_drafts[0]["status"] == "draft"


@pytest.mark.asyncio
async def test_draft_validated_email_ollama_down_queues_task(tmp_path):
    """When Ollama is down, draft_validated_email queues task in tasks table and returns None."""
    db_path = str(tmp_path / "test_email_queue.db")
    state = StateManager(db_path)
    await state.init_db()

    await state.add_product(product_id="prod_1", name="Acme Flow")
    p = Prospect(id="p1", first_name="Marcus", last_name="V", company="Apex")
    await state.save_prospect(p)

    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b"))
    brain.unavailable = "Ollama daemon unreachable on port 11434"

    res = await draft_validated_email("p1", "prod_1", state=state, brain=brain)
    assert res is None, "Should return None when Ollama is unavailable"

    # Assert task is queued in tasks table
    tasks = await state.get_tasks(status="queued")
    assert len(tasks) == 1
    assert tasks[0]["type"] == "email_draft"
    assert "p1" in tasks[0]["payload"]
    assert tasks[0]["status"] == "queued"


# ── 3. Reply Classification Tests ─────────────────────────────────────


def test_reply_classification_all_8_labels_heuristic():
    """Verify keyword heuristic classifies all 8 labels on sample_threads.json."""
    threads_path = PROJECT_ROOT / "data" / "replies" / "sample_threads.json"
    assert threads_path.exists(), "data/replies/sample_threads.json must exist"

    with open(threads_path, "r", encoding="utf-8") as f:
        threads = json.load(f)

    assert len(threads) == 32, "Expected 32 sample threads"

    correct = 0
    seen_labels = set()
    for t in threads:
        expected = t["expected_label"]
        seen_labels.add(expected)
        res = classify_reply_heuristic(t["reply_text"])
        if res["label"] == expected:
            correct += 1

    # Verify all 8 labels exist in test set
    assert seen_labels == set(VALID_LABELS)
    accuracy = correct / len(threads)
    assert accuracy >= 0.90, f"Expected heuristic accuracy >= 90%, got {accuracy * 100}%"


@pytest.mark.asyncio
async def test_reply_classification_llm_with_fake_ollama(tmp_path):
    """Test LLM reply classifier with fake Ollama server."""
    state = StateManager(str(tmp_path / "test_reply.db"))
    await state.init_db()

    def router(prompt, body):
        return {
            "label": "interested",
            "confidence": 0.95,
            "reasoning": "Prospect requested demo and pricing details.",
            "suggested_next_action": "Send calendar link and product overview",
        }

    with FakeOllamaServer() as server:
        server.set_reply_fn(router)
        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))
        res = await classify_reply("Can you share pricing details or a demo link for next week?", brain=brain)

        assert res["label"] == "interested"
        assert res["confidence"] == 0.95
        assert len(res["reasoning"].split()) <= 25
        assert res["suggested_next_action"] == "Send calendar link and product overview"
        assert res["mode"] == "llm"


@pytest.mark.asyncio
async def test_eval_replies_logs_to_inference_log_with_reply_classifier(tmp_path):
    """Verify that classify_reply logs to inference_log with agent='reply_classifier'."""
    import aiosqlite
    db_file = tmp_path / "eval_log.db"
    state = StateManager(str(db_file))
    await state.init_db()

    with FakeOllamaServer() as server:
        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))
        res = await classify_reply(
            "Can we schedule a call next Tuesday?",
            brain=brain,
            agent="reply_classifier",
            allow_fallback=False,
        )
        assert res["label"] == "interested"
        assert res["model"] == "qwen3:1.7b"
        assert "latency_ms" in res

        # Verify inference_log
        async with aiosqlite.connect(str(db_file)) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT agent, task, model, host, mode FROM inference_log") as cur:
                rows = [dict(r) for r in await cur.fetchall()]
        assert len(rows) == 1
        assert rows[0]["agent"] == "reply_classifier"
        assert rows[0]["task"] == "classify_reply"
        assert rows[0]["model"] == "qwen3:1.7b"
        assert rows[0]["mode"] == "fake"


@pytest.mark.asyncio
async def test_classify_reply_no_fallback_raises_when_ollama_unreachable(tmp_path):
    """When allow_fallback=False, Ollama outages raise OllamaUnavailable instead of falling back silently."""
    db_file = tmp_path / "down.db"
    state = StateManager(str(db_file))
    await state.init_db()
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url="http://127.0.0.1:59999"))
    with pytest.raises(OllamaUnavailable):
        await classify_reply(
            "Sounds good, send a demo link",
            brain=brain,
            agent="reply_classifier",
            allow_fallback=False,
        )


def test_eval_replies_cli(tmp_path):
    """Test CLI command `sellbird eval-replies` runs successfully and outputs evaluation."""
    import os
    env = os.environ.copy()
    env["SELLBIRD_HOME"] = str(tmp_path)
    res = subprocess.run(
        [sys.executable, "-m", "sellbird.cli", "eval-replies", "--threads", "data/replies/sample_threads.json"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    assert res.returncode == 0, f"eval-replies failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "SELLBIRD REPLY CLASSIFICATION EVALUATION" in res.stdout
    assert "Heuristic Path" in res.stdout
    assert "LLM Model Path" in res.stdout
    assert "CONFUSION & PER-LABEL ACCURACY SUMMARY" in res.stdout


@pytest.mark.asyncio
async def test_dashboard_api_product_scores_and_emails(test_client):
    """Test dashboard endpoints for product scores and email drafts."""
    # 1. Add product
    resp = test_client.post("/api/products", json={"name": "Acme Cloud", "description": "Cloud telemetry"})
    assert resp.status_code == 200
    prod_id = resp.json()["id"]

    # 2. Get scores (empty initially)
    resp = test_client.get(f"/api/products/{prod_id}/scores")
    assert resp.status_code == 200
    assert resp.json() == []

    # 3. Get email drafts
    resp = test_client.get(f"/api/emails?product_id={prod_id}")
    assert resp.status_code == 200
    assert resp.json() == []
