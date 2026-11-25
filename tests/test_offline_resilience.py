"""Tests for offline resilience, hardware detection, task queueing, and non-LLM fallbacks.

Uses fake_ollama and unit mocks only; NEVER makes real network or external LLM calls.
"""

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import sellbird.dashboard as dash
from sellbird.brain import Brain
from sellbird.config import BrainConfig, SellbirdConfig
from sellbird.hardware import detect_gpu, get_hardware_info, recommend_profile
from sellbird.models.prospect import Prospect
from sellbird.ollama_runtime import OllamaUnavailable
from sellbird.state import StateManager
from sellbird.task_queue import drain_task_queue
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
    monkeypatch.setattr(dash, "WEB_DIR", Path(__file__).resolve().parent.parent / "sellbird" / "web")

    dash.token_store.configure(tmp_path / "data" / "dashboard-token")
    token = dash.token_store.get()

    asyncio.run(StateManager(str(db_path)).init_db())

    with TestClient(dash.app, base_url="http://127.0.0.1:5555",
                    headers={"Authorization": f"Bearer {token}"}) as client:
        client.db_path = db_path
        yield client


# ── Item 0: Cloud AI used computed metric ──


def test_cloud_ai_used_computed(isolated_client):
    """Cloud AI count computes rows where host is not localhost/127.0.0.1/::1."""
    with sqlite3.connect(isolated_client.db_path) as conn:
        conn.execute(
            "INSERT INTO inference_log (ts, agent, task, model, host, prompt_tokens, output_tokens, latency_ms, status) "
            "VALUES (?, ?, ?, ?, ?, 10, 10, 50, 'ok')",
            ("2026-11-24T12:00:00", "writer", "draft", "qwen3:1.7b", "localhost"),
        )
        conn.execute(
            "INSERT INTO inference_log (ts, agent, task, model, host, prompt_tokens, output_tokens, latency_ms, status) "
            "VALUES (?, ?, ?, ?, ?, 10, 10, 50, 'ok')",
            ("2026-11-24T12:01:00", "scout", "score", "qwen3:1.7b", "127.0.0.1"),
        )
        conn.execute(
            "INSERT INTO inference_log (ts, agent, task, model, host, prompt_tokens, output_tokens, latency_ms, status) "
            "VALUES (?, ?, ?, ?, ?, 10, 10, 50, 'ok')",
            ("2026-11-24T12:02:00", "trainer", "icp", "qwen3:1.7b", "api.external-cloud.com"),
        )

    res = isolated_client.get("/api/inference")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 3
    assert data["cloud_count"] == 1


# ── Item 1: Hardware detection and profiles ──


def test_hardware_recommendation_tiers():
    """Verify RAM-based recommendation tiers."""
    # <= 8 GB -> standard laptop, qwen3:1.7b, 2048
    r8 = recommend_profile(8.0)
    assert r8["profile"] == "standard laptop"
    assert r8["model"] == "qwen3:1.7b"
    assert r8["num_ctx"] == 2048

    # 8-16 GB -> standard laptop, qwen3:4b, 4096
    r16 = recommend_profile(16.0)
    assert r16["profile"] == "standard laptop"
    assert r16["model"] == "qwen3:4b"
    assert r16["num_ctx"] == 4096

    # 16-32 GB -> high performance, qwen3:8b, 8192
    r32 = recommend_profile(32.0)
    assert r32["profile"] == "high performance"
    assert r32["model"] == "qwen3:8b"
    assert r32["num_ctx"] == 8192

    # > 32 GB -> high performance, qwen3:14b, 8192
    r64 = recommend_profile(64.0)
    assert r64["profile"] == "high performance"
    assert r64["model"] == "qwen3:14b"
    assert r64["num_ctx"] == 8192


def test_hardware_config_override():
    """Verify user configuration overrides recommendations."""
    cfg = MagicMock()
    cfg.brain = BrainConfig(model="qwen3:8b", hardware_profile="high performance", num_ctx=8192)

    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.total = 8 * (1024 ** 3)  # 8 GB system
        info = get_hardware_info(cfg)
        assert info["ram_gb"] == 8.0
        assert info["recommended_model"] == "qwen3:1.7b"
        assert info["effective_model"] == "qwen3:8b"
        assert info["effective_profile"] == "high performance"
        assert info["is_override"] is True


def test_api_hardware_endpoint(isolated_client):
    """GET /api/hardware returns structured hardware and profile metrics."""
    res = isolated_client.get("/api/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "ram_gb" in data
    assert "cpu_cores" in data
    assert "recommended_profile" in data
    assert "recommended_model" in data
    assert data["recommended_profile"] in ("standard laptop", "high performance")


# ── Item 2: GET /api/status endpoint ──


def test_api_status_with_fake_ollama(isolated_client):
    """GET /api/status reports Ollama reachability, installed model, and queue length."""
    with FakeOllamaServer() as fake:
        with patch("sellbird.config.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.brain = BrainConfig(model="qwen3:1.7b", ollama_url=fake.url)
            mock_cfg.return_value = cfg

            res = isolated_client.get("/api/status")
            assert res.status_code == 200
            data = res.json()
            assert data["ollama_reachable"] is True
            assert data["configured_model_installed"] is True
            assert data["configured_model"] == "qwen3:1.7b"
            assert data["queue_length"] == 0


def test_api_status_when_ollama_offline(isolated_client):
    """GET /api/status reports unreachable when server is down."""
    with patch("sellbird.config.load_config") as mock_cfg:
        cfg = MagicMock()
        cfg.brain = BrainConfig(model="qwen3:1.7b", ollama_url="http://127.0.0.1:59999")
        mock_cfg.return_value = cfg

        res = isolated_client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["ollama_reachable"] is False
        assert data["configured_model_installed"] is False


# ── Item 3: Tasks queue and retry worker ──


@pytest.mark.asyncio
async def test_tasks_queue_and_drain(tmp_path):
    """Tasks are saved on Ollama failure and drained when Ollama is restored."""
    db_path = tmp_path / "sellbird.db"
    state = StateManager(str(db_path))
    await state.init_db()

    # 1. Queue a task directly or via Brain
    task_id = await state.queue_task(
        task_type="write_sequence",
        payload={"prompt": "Write email", "agent": "writer", "task": "write_sequence"},
        error="Connection refused",
    )
    assert task_id > 0
    assert await state.count_queued_tasks() == 1

    # 2. Attempt drain when Ollama is down -> drains 0
    with patch("sellbird.task_queue.OllamaClient.preflight", return_value="Ollama unreachable"):
        res = await drain_task_queue(state)
        assert res["success"] is False
        assert res["drained"] == 0
        assert res["remaining"] == 1

    # 3. Drain when Ollama is available -> drains 1
    with FakeOllamaServer() as fake:
        cfg = BrainConfig(model="qwen3:1.7b", ollama_url=fake.url)
        brain = Brain(state, cfg)
        with patch("sellbird.config.load_config") as mock_cfg:
            c = MagicMock()
            c.brain = cfg
            mock_cfg.return_value = c
            with patch.object(brain, "think", return_value="Drafted response"):
                res = await drain_task_queue(state, brain=brain)
                assert res["success"] is True
                assert res["drained"] == 1
                assert res["remaining"] == 0

    assert await state.count_queued_tasks() == 0


def test_api_tasks_endpoints(isolated_client):
    """GET /api/tasks and POST /api/tasks/retry operate properly."""
    res = isolated_client.get("/api/tasks")
    assert res.status_code == 200
    assert res.json()["queue_length"] == 0

    with patch("sellbird.task_queue.OllamaClient.preflight", return_value="Ollama offline"):
        res_retry = isolated_client.post("/api/tasks/retry")
        assert res_retry.status_code == 200
        assert res_retry.json()["success"] is False


# ── Item 4: Non-LLM Fallbacks for Workflows ──


@pytest.mark.asyncio
async def test_offer_suggestions_non_llm_fallback():
    """Offer suggestion falls back to non-LLM template when brain returns empty."""
    from sellbird.offer import suggest_offers
    brain = MagicMock()
    brain.think_json = AsyncMock(return_value=None)
    cfg = MagicMock()
    cfg.persona.company = "Acme Tools"
    cfg.persona.name = "Alex"
    cfg.product.name = "Acme Tool"
    cfg.product.description = "Dev tools"
    cfg.product.key_benefits = ["Speed", "Reliability"]
    cfg.product.pricing = "Free"
    cfg.target.icp.industries = ["Software"]
    cfg.target.icp.company_size = "10-50"

    offers = await suggest_offers(brain, config=cfg)
    assert len(offers) > 0
    assert any("non-LLM mode" in o["offer"] for o in offers)


@pytest.mark.asyncio
async def test_trainer_non_llm_fallback():
    """Trainer extracts product info and ICP in non-LLM mode when Brain fails."""
    from sellbird.trainer import Trainer

    brain = MagicMock()
    brain.think_json = AsyncMock(return_value=None)
    trainer = Trainer()
    trainer.brain = brain
    trainer.scraped_pages = {"https://example.com": "Example homepage text"}

    product_info = await trainer._extract_product_info()
    assert product_info.get("mode") == "non-LLM mode"
    assert "non-LLM mode" in product_info.get("product_description", "")

    icp = await trainer._extract_icp()
    assert icp.get("mode") == "non-LLM mode"
    assert "Technology" in icp.get("industries", [])


@pytest.mark.asyncio
async def test_trainer_generic_fallback_made_up_company():
    """Fallback works on arbitrary non-Acme company with zero canned Acme text."""
    from sellbird.trainer import Trainer

    brain = MagicMock()
    brain.think_json = AsyncMock(return_value=None)
    trainer = Trainer()
    trainer.brain = brain
    trainer.scraped_pages = {
        "about.txt": "About Stellar Dynamics\nStellar Dynamics engineers satellite constellation communication systems.",
        "products.txt": "Product 1: StellarLink Hub\nOverview: Low-latency ground station router for orbital data relay.\n\nProduct 2: OrbitalMesh Core\nOverview: Inter-satellite laser mesh networking software.",
    }

    product_info = await trainer._extract_product_info()
    assert product_info.get("mode") == "non-LLM mode"
    assert product_info.get("company_name") == "Stellar Dynamics"
    assert product_info.get("product_name") == "StellarLink Hub"
    assert len(product_info.get("products", [])) == 2
    assert product_info["products"][0]["name"] == "StellarLink Hub"
    assert product_info["products"][1]["name"] == "OrbitalMesh Core"

    # Confirm no Acme strings in product info
    info_str = str(product_info).lower()
    assert "acme" not in info_str
    assert "pulse" not in info_str
    assert "flow" not in info_str

    icp = await trainer._extract_icp(product=product_info["products"][0])
    assert icp.get("mode") == "non-LLM mode"
    assert any(ind in icp.get("industries", []) for ind in ("Software", "Technology", "Telecommunications"))

    # Confirm no Acme strings in ICP
    icp_str = str(icp).lower()
    assert "acme" not in icp_str
    assert "pulse" not in icp_str
    assert "flow" not in icp_str


@pytest.mark.asyncio
async def test_trainer_fallback_empty_content_raises_error():
    """Empty content raises clear ValueError instead of inventing content."""
    from sellbird.trainer import Trainer

    brain = MagicMock()
    brain.think_json = AsyncMock(return_value=None)
    trainer = Trainer()
    trainer.brain = brain
    trainer.scraped_pages = {"empty.txt": "   \n\t  "}

    with pytest.raises(ValueError, match="stored text is empty"):
        await trainer._extract_product_info()

    with pytest.raises(ValueError, match="stored text is empty"):
        await trainer._extract_icp()


@pytest.mark.asyncio
async def test_scout_scoring_non_llm_fallback(tmp_path):
    """Scout lead scoring uses heuristic score when Brain fails, tagging non-LLM mode."""
    from sellbird.agents.scout import Scout
    state = StateManager(str(tmp_path / "scout.db"))
    await state.init_db()

    brain = MagicMock()
    brain.think_json.side_effect = OllamaUnavailable("Offline")

    cfg = MagicMock()
    cfg.target = None
    cfg.brain = None
    scout = Scout(brain=brain, state=state, config=cfg, env=MagicMock())

    contacts = [
        Prospect(id="p1", email="lead@corp.com", first_name="Chris", company="Corp", title="VP Engineering", seniority="vp")
    ]
    scored = await scout._score_contacts(contacts)
    assert len(scored) == 1
    assert scored[0].score > 30
    assert "non-LLM mode" in scored[0].personalization_notes


@pytest.mark.asyncio
async def test_handler_intent_non_llm_fallback():
    """Handler intent classification uses deterministic keyword patterns when Brain is down."""
    from sellbird.agents.handler import Handler
    brain = MagicMock()
    brain.think = AsyncMock(return_value="")
    brain.paused_reason = "Ollama down"

    handler = Handler(brain=brain, state=MagicMock(), config=MagicMock(), env=MagicMock())
    prospect = Prospect(id="p1", email="p@co.test", first_name="Alex", company="Co")

    # Unsubscribe keyword
    assert await handler._classify_intent("Please unsubscribe me immediately.", prospect) == "unsubscribe"
    # Not interested keyword
    assert await handler._classify_intent("We are not interested right now.", prospect) == "not_interested"
    # Interested keyword
    assert await handler._classify_intent("This sounds good, let's talk next week.", prospect) == "interested"
    # Wrong person keyword
    assert await handler._classify_intent("Wrong person, reach out to Sarah.", prospect) == "wrong_person"
    # OOO keyword
    assert await handler._classify_intent("I am out of office until Monday.", prospect) == "ooo"
    # Ambiguous -> escalate
    assert await handler._classify_intent("Interesting point.", prospect) == "escalate"
