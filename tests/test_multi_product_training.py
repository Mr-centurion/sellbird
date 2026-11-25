"""Tests for folder loader, multi-product database migrations, and per-product ICP generation.

RULES: No external LLM or cloud calls. Tests run against fake_ollama only.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import sellbird.dashboard as dash
from sellbird.brain import Brain
from sellbird.config import BrainConfig, load_brain_config
from sellbird.paths import PROJECT_ROOT
from sellbird.state import StateManager
from sellbird.trainer import Trainer
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

    with TestClient(dash.app, base_url="http://127.0.0.1:5555",
                    headers={"Authorization": f"Bearer {token}"}) as client:
        client.db_path = db_path
        yield client


# ── 1. Folder Loader Tests ──


def test_sample_acme_files_exist_and_start_with_sample_data():
    """Verify data/companies/sample_acme contains 3 text files starting with 'SAMPLE DATA'."""
    sample_dir = PROJECT_ROOT / "data" / "companies" / "sample_acme"
    assert sample_dir.exists() and sample_dir.is_dir()

    files = [sample_dir / "about.txt", sample_dir / "products.txt", sample_dir / "pricing.txt"]
    for f in files:
        assert f.exists(), f"Expected {f.name} to exist"
        content = f.read_text(encoding="utf-8").strip()
        assert content.startswith("SAMPLE DATA"), f"{f.name} must start with 'SAMPLE DATA'"

    products_content = (sample_dir / "products.txt").read_text(encoding="utf-8")
    assert "Acme Flow" in products_content
    assert "Acme Pulse" in products_content


@pytest.mark.asyncio
async def test_folder_loader_reads_local_files_without_crawling(tmp_path):
    """Trainer.train_from_folder populates scraped_pages from local .md/.txt files."""
    folder = tmp_path / "sample_company"
    folder.mkdir()
    (folder / "about.txt").write_text("About us content", encoding="utf-8")
    (folder / "features.md").write_text("# Features content", encoding="utf-8")
    (folder / "image.png").write_bytes(b"\x89PNG")  # non-text file should be ignored

    trainer = Trainer(brain_config=BrainConfig())
    # Mock _run_analysis to inspect scraped_pages before full execution
    with patch.object(trainer, "_run_analysis", return_value={"mock": "config"}) as mock_run:
        res = await trainer.train_from_folder(str(folder))
        assert res == {"mock": "config"}
        assert len(trainer.scraped_pages) == 2
        assert "about.txt" in trainer.scraped_pages
        assert "features.md" in trainer.scraped_pages
        assert "image.png" not in trainer.scraped_pages
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_folder_loader_nonexistent_or_empty_folder():
    """Trainer.train_from_folder returns None for invalid or empty directories."""
    trainer = Trainer(brain_config=BrainConfig())
    assert await trainer.train_from_folder("/nonexistent/directory/path") is None


# ── 2. Database Migrations Tests ──


@pytest.mark.asyncio
async def test_multi_product_migrations_fresh_db(tmp_path):
    """Fresh database creates products, icps, signals, problem_statements, emails, scores."""
    db_file = tmp_path / "fresh.db"
    state = StateManager(str(db_file))
    await state.init_db()

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        for expected in ("products", "icps", "signals", "problem_statements", "emails", "scores"):
            assert expected in tables, f"Expected table {expected} in database"

        for tbl in ("icps", "signals", "problem_statements", "emails", "scores"):
            cursor.execute(f"PRAGMA table_info({tbl})")
            cols = {row[1] for row in cursor.fetchall()}
            assert "product_id" in cols, f"Expected column product_id in table {tbl}"


@pytest.mark.asyncio
async def test_multi_product_migrations_backward_compatibility(tmp_path):
    """Legacy tables without product_id column are migrated cleanly by init_db."""
    db_file = tmp_path / "legacy.db"
    # Pre-populate tables without product_id to simulate an older database
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE icps (id TEXT PRIMARY KEY, titles TEXT);")
        conn.execute("CREATE TABLE signals (id TEXT PRIMARY KEY, code TEXT);")
        conn.execute("CREATE TABLE problem_statements (id TEXT PRIMARY KEY, statement TEXT);")
        conn.execute("CREATE TABLE emails (id TEXT PRIMARY KEY, subject TEXT);")
        conn.execute("CREATE TABLE scores (id TEXT PRIMARY KEY, score INTEGER);")
        conn.execute("PRAGMA user_version = 18;")
        conn.commit()

    state = StateManager(str(db_file))
    await state.init_db()

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        for tbl in ("icps", "signals", "problem_statements", "emails", "scores"):
            cursor.execute(f"PRAGMA table_info({tbl})")
            cols = {row[1] for row in cursor.fetchall()}
            assert "product_id" in cols, f"product_id was not migrated into {tbl}"


@pytest.mark.asyncio
async def test_state_product_and_icp_crud(tmp_path):
    """StateManager supports CRUD on products and saving per-product ICPs."""
    db_file = tmp_path / "test_crud.db"
    state = StateManager(str(db_file))
    await state.init_db()

    # Add product
    prod = await state.add_product(
        name="Acme Flow",
        description="Workflow engine for supply chain",
        value_props="Reduces order delay by 75%",
        status="active",
        product_id="acme_flow",
    )
    assert prod["id"] == "acme_flow"
    assert prod["name"] == "Acme Flow"

    # List products
    prods = await state.list_products()
    assert len(prods) == 1
    assert prods[0]["id"] == "acme_flow"

    # Save ICP for product
    icp_input = {
        "industries": ["Logistics", "Supply Chain"],
        "company_size": "50-500 employees",
        "titles": ["VP Operations", "Director of Supply Chain"],
        "pain_points": ["Manual entry errors", "ERP sync delays"],
        "disqualifiers": ["No ERP in place"],
    }
    saved = await state.save_icp(product_id="acme_flow", icp_data=icp_input, model="qwen3:1.7b")
    assert saved["product_id"] == "acme_flow"
    assert saved["model"] == "qwen3:1.7b"
    assert "Manual entry errors" in saved["pain_points_list"]
    assert "VP Operations" in saved["titles_list"]
    assert "No ERP in place" in saved["disqualifiers_list"]

    # Retrieve ICP
    fetched = await state.get_icp("acme_flow")
    assert fetched is not None
    assert fetched["product_id"] == "acme_flow"
    assert "Logistics" in fetched["industries_list"]


@pytest.mark.asyncio
async def test_single_product_backward_compatibility(tmp_path):
    """When products table is empty, list_products and get_product fall back to sellbird.yaml."""
    db_file = tmp_path / "compat.db"
    state = StateManager(str(db_file))
    await state.init_db()

    prods = await state.list_products()
    assert len(prods) >= 1
    assert prods[0]["id"] == "default"

    default_prod = await state.get_product("default")
    assert default_prod is not None
    assert default_prod["id"] == "default"

    default_icp = await state.get_icp("default")
    assert default_icp is not None
    assert default_icp["product_id"] == "default"
    assert default_icp["model"] == "config"


# ── 3. ICP Generation for Two Products Tests ──


def test_api_products_and_icp_endpoints_with_fake_ollama(test_client):
    """API supports GET/POST /api/products, POST /api/products/{id}/icp, GET /api/products/{id}/icp."""
    with FakeOllamaServer() as fake:
        def fake_reply(prompt, body):
            p = prompt.lower()
            if "pulse" in p:
                return {
                    "industries": ["Software", "Internet"],
                    "company_size": "20-250 engineers",
                    "titles": ["VP Engineering", "CTO", "Head of DevOps"],
                    "geography": ["United States"],
                    "buyer_persona": "Engineering leaders tracking velocity",
                    "pain_points": ["PR review bottlenecks", "Flaky test suites"],
                    "disqualifiers": ["Non-technical companies"],
                }
            elif "flow" in p:
                return {
                    "industries": ["Logistics", "Manufacturing"],
                    "company_size": "50-500 employees",
                    "titles": ["VP Operations", "COO", "Supply Chain Director"],
                    "geography": ["United States"],
                    "buyer_persona": "Operations leaders automating logistics",
                    "pain_points": ["Manual shipment tracking", "Inventory discrepancies"],
                    "disqualifiers": ["No warehouse operations"],
                }
            return {
                "industries": ["General"],
                "titles": ["Manager"],
                "company_size": "10-100",
                "pain_points": ["General inefficiency"],
                "disqualifiers": ["Low budget"],
            }

        fake.set_reply_fn(fake_reply)

        # 1. Create two products via POST /api/products
        res1 = test_client.post("/api/products", json={
            "id": "acme_flow",
            "name": "Acme Flow",
            "description": "Logistics workflow automation",
            "value_props": "75% faster order turnaround",
        })
        assert res1.status_code == 200
        assert res1.json()["id"] == "acme_flow"

        res2 = test_client.post("/api/products", json={
            "id": "acme_pulse",
            "name": "Acme Pulse",
            "description": "Engineering velocity telemetry",
            "value_props": "Real-time DORA metrics dashboards",
        })
        assert res2.status_code == 200
        assert res2.json()["id"] == "acme_pulse"

        # 2. Verify GET /api/products returns both products
        list_res = test_client.get("/api/products")
        assert list_res.status_code == 200
        ids = {p["id"] for p in list_res.json()}
        assert "acme_flow" in ids
        assert "acme_pulse" in ids

        # 3. Generate ICP for Acme Flow via POST /api/products/{id}/icp
        with patch("sellbird.config.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.brain = BrainConfig(model="qwen3:1.7b", ollama_url=fake.url)
            mock_cfg.return_value = cfg

            flow_gen = test_client.post("/api/products/acme_flow/icp")
            assert flow_gen.status_code == 200
            flow_data = flow_gen.json()
            assert flow_data["product_id"] == "acme_flow"
            assert flow_data["model"] == "qwen3:1.7b"
            assert "Logistics" in flow_data["icp"]["industries"]
            assert "VP Operations" in flow_data["icp"]["titles"]

            # 4. Generate ICP for Acme Pulse via POST /api/products/{id}/icp
            pulse_gen = test_client.post("/api/products/acme_pulse/icp")
            assert pulse_gen.status_code == 200
            pulse_data = pulse_gen.json()
            assert pulse_data["product_id"] == "acme_pulse"
            assert pulse_data["model"] == "qwen3:1.7b"
            assert "Software" in pulse_data["icp"]["industries"]
            assert "VP Engineering" in pulse_data["icp"]["titles"]

        # 5. Fetch ICP for both via GET /api/products/{id}/icp
        flow_get = test_client.get("/api/products/acme_flow/icp")
        assert flow_get.status_code == 200
        assert "acme_flow" == flow_get.json()["product_id"]
        assert any("Manual shipment" in pt for pt in flow_get.json()["pain_points_list"])

        pulse_get = test_client.get("/api/products/acme_pulse/icp")
        assert pulse_get.status_code == 200
        assert "acme_pulse" == pulse_get.json()["product_id"]
        assert any("PR review" in pt for pt in pulse_get.json()["pain_points_list"])


@pytest.mark.asyncio
async def test_train_from_folder_sample_acme_end_to_end(tmp_path):
    """sellbird train --from-folder runs end-to-end on sample_acme using fake_ollama."""
    with FakeOllamaServer() as fake:
        def reply_fn(prompt, body):
            p = prompt.lower()
            if "extract comprehensive product" in p:
                return {
                    "company_name": "Acme Technologies",
                    "product_name": "Acme Flow",
                    "product_description": "Workflow automation and engineering analytics software for mid-market teams.",
                    "pricing": "Starting at $499/mo",
                    "key_benefits": ["Process automation", "Cross-system sync", "Engineering velocity tracking"],
                    "features": ["Workflow engine", "Engineering telemetry", "ERP integrations"],
                    "target_audience": "Operations leaders and Engineering executives",
                    "value_proposition": "Eliminate operational bottlenecks and accelerate delivery.",
                    "products": [
                        {
                            "name": "Acme Flow",
                            "description": "Intelligent workflow automation and ERP orchestration for operations & logistics.",
                            "value_props": "Reduces manual order processing time by 75% with automated error handling.",
                            "pricing": "Starter $499/mo, Growth $1,499/mo, Enterprise $3,500/mo",
                        },
                        {
                            "name": "Acme Pulse",
                            "description": "Continuous engineering velocity and reliability telemetry platform.",
                            "value_props": "Real-time DORA metrics dashboards and sprint health analytics with zero code storage.",
                            "pricing": "Team $15/dev/mo, Org $35/dev/mo, Enterprise custom",
                        },
                    ],
                }
            if "objection" in p:
                return {
                    "Too expensive": "We offer flexible tiers to match your volume.",
                    "Already have a solution": "We integrate directly with your existing software stack.",
                }
            if "specifically for this product" in p and "pulse" in p:
                return {
                    "industries": ["Software", "Internet"],
                    "company_size": "20-250 engineers",
                    "titles": ["VP Engineering", "CTO", "Head of DevOps"],
                    "geography": ["United States"],
                    "buyer_persona": "Engineering leaders tracking velocity",
                    "pain_points": ["PR review bottlenecks", "Flaky test suites"],
                    "disqualifiers": ["Non-technical companies"],
                }
            if "specifically for this product" in p and "flow" in p:
                return {
                    "industries": ["Logistics", "Manufacturing"],
                    "company_size": "50-500 employees",
                    "titles": ["VP Operations", "COO", "Supply Chain Director"],
                    "geography": ["United States"],
                    "buyer_persona": "Operations leaders automating logistics",
                    "pain_points": ["Manual shipment tracking", "Inventory discrepancies"],
                    "disqualifiers": ["No warehouse operations"],
                }
            return {
                "industries": ["Technology", "Services"],
                "company_size": "10-200 employees",
                "titles": ["VP", "Director", "Head of", "Owner", "Founder"],
                "geography": ["United States"],
                "pain_points": ["Manual overhead", "Slow operational velocity"],
                "disqualifiers": ["No budget", "Outside operating region"],
            }

        fake.set_reply_fn(reply_fn)

        fake_cfg = BrainConfig(model="qwen3:1.7b", ollama_url=fake.url)
        trainer = Trainer(brain_config=fake_cfg)
        trainer.state = StateManager(str(tmp_path / "trained.db"))

        sample_folder = PROJECT_ROOT / "data" / "companies" / "sample_acme"
        out_yaml = tmp_path / "trained.yaml"

        config = await trainer.train_from_folder(str(sample_folder), output_path=str(out_yaml))
        assert config is not None
        assert "product" in config
        assert out_yaml.exists()

        # Check that both products were saved in DB
        products = await trainer.state.list_products()
        prod_ids = {p["id"] for p in products}
        assert "acme_flow" in prod_ids
        assert "acme_pulse" in prod_ids

        # Check that ICP was stored for both products
        icp_flow = await trainer.state.get_icp("acme_flow")
        assert icp_flow is not None
        assert icp_flow["product_id"] == "acme_flow"

        icp_pulse = await trainer.state.get_icp("acme_pulse")
        assert icp_pulse is not None
        assert icp_pulse["product_id"] == "acme_pulse"
