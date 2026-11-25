"""Tests for stored research, buying signals, problem statements, and air-gapped offline resilience.

Rules:
- No external LLM or cloud calls.
- Do not hard-code generated outputs.
- Test with fake_ollama only.
- Strict verbatim quote enforcement.
- Word limits enforcement.
- Sockets blocked to external network.
"""

import asyncio
import json
import re
import socket
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

import sellbird.dashboard as dash
from sellbird.brain import Brain
from sellbird.buying_signals import (
    SIGNAL_LABELS,
    classify_buying_signals,
    detect_signals_regex,
)
from sellbird.config import BrainConfig
from sellbird.paths import PROJECT_ROOT
from sellbird.problem_statement import generate_problem_statement
from sellbird.state import StateManager
from sellbird.stored_research import (
    build_research_summary,
    chunk_stored_text,
    compute_embedding,
    retrieve_top_k_snippets,
)
from tests.fake_ollama import FakeOllamaServer


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    """Authenticated TestClient on an isolated database."""
    (tmp_path / "data").mkdir()
    db_path = tmp_path / "data" / "sellbird.db"
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "DB_PATH", db_path)
    monkeypatch.setattr(dash, "PID_FILE", tmp_path / "data" / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", tmp_path / "data" / "sellbird.log")
    monkeypatch.setattr(dash, "WEB_DIR", PROJECT_ROOT / "sellbird" / "web")

    dash.token_store.configure(tmp_path / "data" / "dashboard-token")
    token = dash.token_store.get()

    asyncio.run(StateManager(str(db_path)).init_db())

    with TestClient(dash.app, base_url="http://127.0.0.1:5555",
                    headers={"Authorization": f"Bearer {token}"}) as client:
        client.db_path = db_path
        yield client


# ── Step 1: Seed data and import tests ───────────────────────────────


def test_sample_prospects_json_structure():
    """Verify data/prospects/sample_prospects.json adheres to all requirements."""
    path = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    assert path.exists(), f"{path} does not exist"

    with open(path, "r", encoding="utf-8") as f:
        prospects = json.load(f)

    assert len(prospects) == 15, f"Expected 15 prospects, got {len(prospects)}"

    for p in prospects:
        assert p.get("sample_data") == "SAMPLE DATA"
        assert p.get("company"), "Company name is required"
        assert p.get("industry"), "Industry is required"
        assert p.get("size"), "Size is required"
        assert p.get("contact_name"), "Contact name is required"
        assert p.get("contact_title"), "Contact title is required"
        assert p.get("contact_email"), "Contact email is required"

        desc = p.get("website_description", "")
        words = desc.split()
        assert 40 <= len(words) <= 80, f"{p['company']} description has {len(words)} words, expected 40-80"
        assert "SAMPLE DATA" in desc

        news = p.get("recent_news", [])
        assert len(news) in (2, 3), f"{p['company']} has {len(news)} news lines, expected 2-3"
        for line in news:
            assert re.match(r"^\d{4}-\d{2}-\d{2}:", line), f"News line missing date: {line}"


@pytest.mark.asyncio
async def test_prospect_import_into_database(tmp_path):
    """Test importing prospects into an isolated SQLite database."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    count = await state.import_prospects_from_file(sample_file)
    assert count == 15

    # Verify rows in prospects table
    prospects = await state.get_prospects()
    assert len(prospects) == 15

    # Verify stored_text exists and contains sample marker and news
    first_p = prospects[0]
    stored_text = await state.get_prospect_stored_text(first_p.id)
    assert "SAMPLE DATA" in stored_text
    assert "Website Description:" in stored_text
    assert "Recent News:" in stored_text


# ── Step 2: Stored research tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_stored_research_with_fake_ollama(tmp_path):
    """Verify research from stored data using sentence-transformers and SQLite embedding cache."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    # Import prospects
    sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    await state.import_prospects_from_file(sample_file)
    prospects = await state.get_prospects()
    prospect = prospects[0]

    with FakeOllamaServer() as server:
        canned_research = {
            "summary": "Apex Logistics is a tech-enabled 3PL provider streamlining freight fulfillment across North America.",
            "key_facts": [
                "Operates enterprise third-party logistics and automated fulfillment",
                "Centralized platform unifies carrier dispatching and inventory routing",
            ],
            "recent_events": [
                "2026-08-14: Opened two new fulfillment hubs in Chicago and Dallas",
                "2026-10-02: Marcus Vance joined as VP of Supply Chain Operations",
            ],
            "gaps": ["Current transportation management software integrations", "Contract SLA terms"],
        }
        server.set_canned_reply(canned_research)

        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))
        res = await build_research_summary(prospect.id, state=state, brain=brain)

        assert res["prospect_id"] == prospect.id
        assert "Apex Logistics" in res["summary"]
        assert len(res["key_facts"]) == 2
        assert len(res["recent_events"]) == 2
        assert len(res["gaps"]) == 2
        assert res["mode"] == "llm"

        # Verify embeddings were cached in SQLite embedding_cache table
        import aiosqlite
        async with aiosqlite.connect(str(db_file)) as db:
            async with db.execute("SELECT count(*) FROM embedding_cache") as cur:
                (cache_count,) = await cur.fetchone()
                assert cache_count > 0, "Expected embedding cache rows"

        # Verify research stored in DB
        saved = await state.get_prospect_research(prospect.id)
        assert saved is not None
        assert saved["summary"] == canned_research["summary"]


@pytest.mark.asyncio
async def test_stored_research_non_llm_fallback(tmp_path):
    """Verify deterministic fallback when Ollama is unreachable."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    await state.import_prospects_from_file(sample_file)
    prospects = await state.get_prospects()
    prospect = prospects[0]

    # Dead brain host
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url="http://127.0.0.1:59999"))
    res = await build_research_summary(prospect.id, state=state, brain=brain)

    assert res["mode"] == "non-LLM mode"
    assert "non-LLM mode" in res["summary"]
    assert len(res["key_facts"]) > 0


# ── Step 3: Buying signals tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_buying_signals_verbatim_drop_rule(tmp_path):
    """Verify that any signal whose evidence_quote is not in stored_text is dropped."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    await state.import_prospects_from_file(sample_file)
    prospects = await state.get_prospects()
    prospect = next(p for p in prospects if p.company == "Apex Logistics")

    stored_text = await state.get_prospect_stored_text(prospect.id)

    # Valid verbatim quote from stored text
    valid_quote = "opening of two new fulfillment hubs in Chicago and Dallas"
    assert valid_quote in stored_text

    # Fabricated / paraphrased quote NOT in stored text
    invalid_quote = "Apex plans to build massive new distribution centers soon"
    assert invalid_quote not in stored_text

    with FakeOllamaServer() as server:
        server.set_canned_reply({
            "signals": [
                {"label": "expansion", "evidence_quote": valid_quote},
                {"label": "funding", "evidence_quote": invalid_quote},
                {"label": "none", "evidence_quote": "nothing to see here"},
            ]
        })

        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))
        signals = await classify_buying_signals(prospect.id, state=state, brain=brain)

        # Only the verbatim signal must be accepted
        labels = [s["label"] for s in signals]
        assert "expansion" in labels
        assert "funding" not in labels, "Non-verbatim signal must be DROPPED"
        assert "none" not in labels, "'none' must be dropped"

        # Verify saved in signals table
        db_signals = await state.get_prospect_signals(prospect.id)
        assert len(db_signals) == 1
        assert db_signals[0]["evidence_quote"] == valid_quote


@pytest.mark.asyncio
async def test_buying_signals_regex_fallback(tmp_path):
    """Verify non-LLM regex signal detector runs when Ollama is unavailable."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    await state.import_prospects_from_file(sample_file)
    prospects = await state.get_prospects()
    prospect = prospects[0]

    # Offline brain
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url="http://127.0.0.1:59999"))
    signals = await classify_buying_signals(prospect.id, state=state, brain=brain)

    assert len(signals) > 0
    for s in signals:
        assert s["mode"] == "non-LLM mode"
        stored_text = await state.get_prospect_stored_text(prospect.id)
        # Verify verbatim match in stored text
        assert s["evidence_quote"] in stored_text


# ── Step 4: Problem statements tests ────────────────────────────────


@pytest.mark.asyncio
async def test_problem_statement_constraints_and_multi_product(tmp_path):
    """Verify problem statement generation enforces word limits and stores per product."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    # Add two products
    prod_a = await state.add_product("FlowDispatch", "Automated carrier dispatching", ["dispatch speed", "SLA compliance"])
    prod_b = await state.add_product("ColdTrack", "Cold chain temperature monitoring", ["temperature alerts", "waste reduction"])

    # Import prospects
    sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
    await state.import_prospects_from_file(sample_file)
    prospects = await state.get_prospects()
    prospect = prospects[0]

    with FakeOllamaServer() as server:
        # Long generated texts to test truncation constraint
        long_ps = " ".join(["problem"] * 55)
        long_wn = " ".join(["urgent"] * 35)
        server.set_canned_reply({
            "problem_statement": long_ps,
            "why_now": long_wn,
            "evidence": ["Reports supply chain bottlenecks due to legacy ERP"],
        })

        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))

        # Product A
        res_a = await generate_problem_statement(prospect.id, product_id=prod_a["id"], state=state, brain=brain)
        assert res_a["product_id"] == prod_a["id"]
        assert len(res_a["problem_statement"].split()) <= 40
        assert len(res_a["why_now"].split()) <= 25

        # Product B
        res_b = await generate_problem_statement(prospect.id, product_id=prod_b["id"], state=state, brain=brain)
        assert res_b["product_id"] == prod_b["id"]
        assert len(res_b["problem_statement"].split()) <= 40
        assert len(res_b["why_now"].split()) <= 25

        # Verify DB queries per product
        pss_a = await state.get_problem_statements(prospect_id=prospect.id, product_id=prod_a["id"])
        pss_b = await state.get_problem_statements(prospect_id=prospect.id, product_id=prod_b["id"])
        assert len(pss_a) >= 1
        assert len(pss_b) >= 1
        assert pss_a[0]["product_id"] == prod_a["id"]
        assert pss_b[0]["product_id"] == prod_b["id"]


# ── Step 5: Network-disabled air-gap test ────────────────────────────


@pytest.mark.asyncio
async def test_offline_air_gap_with_blocked_sockets(tmp_path):
    """Verify that all workflows pass when external socket connections are blocked."""
    db_file = tmp_path / "test.db"
    state = StateManager(str(db_file))
    await state.init_db()

    prod = await state.add_product("LogiFlow", "Automated logistics orchestration", ["lower freight spend"])

    with FakeOllamaServer() as server:
        def reply_router(prompt, body):
            p_lower = prompt.lower()
            if "research" in p_lower:
                return {
                    "summary": "Apex Logistics operates 3PL services across North America.",
                    "key_facts": ["Enterprise logistics fulfillment", "Guaranteed SLA schedules"],
                    "recent_events": ["2026-08-14: Opened two new hubs in Chicago and Dallas"],
                    "gaps": ["Current software stack"],
                }
            if "buying signals" in p_lower:
                return {
                    "signals": [
                        {
                            "label": "expansion",
                            "evidence_quote": "Apex Logistics announces the opening of two new fulfillment hubs in Chicago and Dallas",
                        }
                    ]
                }
            if "problem statement" in p_lower:
                return {
                    "problem_statement": "Legacy ERP dispatching creates severe supply chain bottlenecks.",
                    "why_now": "Rapid 35% freight volume growth demands automated dispatch today.",
                    "evidence": ["opening of two new fulfillment hubs"],
                }
            return {"status": "ok"}

        server.set_reply_fn(reply_router)

        # Patch socket connect to block all external network connections
        orig_connect = socket.socket.connect

        def guarded_connect(self, address):
            host = address[0]
            if host in ("127.0.0.1", "localhost", "::1"):
                return orig_connect(self, address)
            raise OSError(f"AIR-GAP VIOLATION: Attempted socket connection to external host {host}")

        with patch.object(socket.socket, "connect", guarded_connect):
            # 1. Import sample prospects
            sample_file = PROJECT_ROOT / "data" / "prospects" / "sample_prospects.json"
            imported = await state.import_prospects_from_file(sample_file)
            assert imported == 15

            prospects = await state.get_prospects()
            prospect = prospects[0]

            brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b", ollama_url=server.url))

            # 2. Research from stored data
            research = await build_research_summary(prospect.id, state=state, brain=brain)
            assert "Apex Logistics" in research["summary"]

            # 3. Buying signals with verbatim quote
            signals = await classify_buying_signals(prospect.id, state=state, brain=brain)
            assert len(signals) == 1
            assert signals[0]["label"] == "expansion"

            # 4. Problem statement per prospect & product
            ps = await generate_problem_statement(prospect.id, product_id=prod["id"], state=state, brain=brain)
            assert "bottlenecks" in ps["problem_statement"]
            assert len(ps["problem_statement"].split()) <= 40
            assert len(ps["why_now"].split()) <= 25


# ── Dashboard API integration tests ─────────────────────────────────


def test_dashboard_prospects_api_endpoints(isolated_client):
    """Test dashboard endpoints for sample import, research, signals, and problem statements."""
    with FakeOllamaServer() as fake:
        def reply_router(prompt, body):
            p_lower = prompt.lower()
            if "sales strategist" in p_lower or "problem_statement" in p_lower:
                return {
                    "problem_statement": "Legacy dispatch systems create delivery bottlenecks.",
                    "why_now": "Opening new hubs makes modernizing dispatch urgent today.",
                    "evidence": ["Reports dispatching bottlenecks due to legacy ERP"],
                }
            if "buying signals" in p_lower or "buying signal" in p_lower:
                return {
                    "signals": [
                        {
                            "label": "expansion",
                            "evidence_quote": "Apex Logistics announces the opening of two new fulfillment hubs in Chicago and Dallas",
                        }
                    ]
                }
            if "sales intelligence researcher" in p_lower or "research" in p_lower:
                return {
                    "summary": "Apex Logistics provides enterprise 3PL services.",
                    "key_facts": ["Enterprise logistics fulfillment", "Guaranteed SLA schedules"],
                    "recent_events": ["2026-08-14: Opened two new hubs in Chicago and Dallas"],
                    "gaps": ["Current software stack"],
                }
            return {"status": "ok"}

        fake.set_reply_fn(reply_router)

        with patch("sellbird.config.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.brain = BrainConfig(model="qwen3:1.7b", ollama_url=fake.url)
            mock_cfg.return_value = cfg

            # 1. Import sample prospects
            res_import = isolated_client.post("/api/prospects/import-sample")
            assert res_import.status_code == 200
            data_import = res_import.json()
            assert data_import["status"] == "ok"
            assert data_import["imported"] == 15

            # 2. List prospects
            res_list = isolated_client.get("/api/prospects")
            assert res_list.status_code == 200
            prospects = res_list.json()
            assert len(prospects) == 15
            apex = next(p for p in prospects if p["company"] == "Apex Logistics")
            pid = apex["id"]

            # 3. Research endpoints
            res_res = isolated_client.get(f"/api/prospects/{pid}/research")
            assert res_res.status_code == 200

            # Generate research with fake_ollama
            res_gen_res = isolated_client.post(f"/api/prospects/{pid}/research")
            assert res_gen_res.status_code == 200
            res_data = res_gen_res.json()
            assert res_data["mode"] == "llm"
            assert "Apex Logistics" in res_data["summary"]

            # 4. Signals endpoints
            res_sig_post = isolated_client.post(f"/api/prospects/{pid}/signals", json={})
            assert res_sig_post.status_code == 200
            signals = res_sig_post.json()
            assert len(signals) == 1
            assert signals[0]["label"] == "expansion"

            res_sig_get = isolated_client.get(f"/api/prospects/{pid}/signals")
            assert res_sig_get.status_code == 200
            assert len(res_sig_get.json()) == 1

            # 5. Problem statement endpoints
            res_ps_post = isolated_client.post(f"/api/prospects/{pid}/problem-statements", json={"product_id": "default"})
            assert res_ps_post.status_code == 200
            ps_data = res_ps_post.json()
            assert ps_data["mode"] == "llm"
            assert len(ps_data["problem_statement"].split()) <= 40

            res_ps_get = isolated_client.get(f"/api/prospects/{pid}/problem-statements")
            assert res_ps_get.status_code == 200
            assert len(res_ps_get.json()) >= 1
