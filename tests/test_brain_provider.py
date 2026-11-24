"""Brain provider tests: Ollama is the default and only provider.

Verifies config loading, validation, preflight, think, think_json, and budget checks.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
import yaml
from pydantic import ValidationError

import sellbird.brain as brain_mod
from sellbird.brain import Brain, BRAIN_STATUS_KEY
from sellbird.config import BrainConfig, ConfigError, SellbirdConfig, load_brain_config
from sellbird.ollama_runtime import OllamaClient, OllamaUnavailable
from sellbird.state import StateManager

BASE = {
    "persona": {"name": "H", "company": "A", "role": "B", "email": "e", "linkedin": "", "tone": "t"},
    "product": {"name": "P", "description": "D", "pricing": "$1", "key_benefits": [],
                "objection_responses": {}},
    "icp": {"industries": [], "company_size": "", "titles": [], "geography": []},
}


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


def test_ollama_is_the_default():
    assert SellbirdConfig(**BASE).brain.provider == "ollama"
    assert BrainConfig().provider == "ollama"
    assert BrainConfig().model == "qwen3:1.7b"
    assert BrainConfig().ollama_url == "http://localhost:11434"
    assert BrainConfig().num_ctx == 4096


def test_brain_config_validation():
    cfg = BrainConfig(provider="ollama", model="qwen3:4b", num_ctx=8192)
    assert (cfg.provider, cfg.model, cfg.num_ctx) == ("ollama", "qwen3:4b", 8192)

    for bad in ({"provider": "claude"}, {"provider": "codex"}, {"provider": "openai"},
                {"model": "--dangerously-skip-permissions"}, {"model": "model; rm -rf /"}):
        with pytest.raises(ValidationError):
            BrainConfig(**bad)


def test_load_brain_config(tmp_path):
    path = tmp_path / "sellbird.yaml"
    assert load_brain_config(str(path)) == BrainConfig()  # missing file
    path.write_text(yaml.dump({"persona": "placeholder"}))
    assert load_brain_config(str(path)).provider == "ollama"
    path.write_text(yaml.dump({"brain": {"provider": "ollama", "model": "qwen3:4b"}}))
    assert load_brain_config(str(path)) == BrainConfig(provider="ollama", model="qwen3:4b")
    path.write_text(yaml.dump({"brain": {"provider": "gemini"}}))
    with pytest.raises(ConfigError):
        load_brain_config(str(path))


@pytest.mark.asyncio
async def test_brain_preflight_success(state):
    brain = Brain(state)
    with patch.object(brain.ollama, "preflight", new=AsyncMock(return_value="")):
        assert await brain.preflight() == ""
        assert brain.paused_reason == ""


@pytest.mark.asyncio
async def test_brain_preflight_failure_marks_unavailable(state):
    brain = Brain(state)
    with patch.object(brain.ollama, "preflight", new=AsyncMock(return_value="Ollama unreachable")):
        assert await brain.preflight() == "Ollama unreachable"
        assert brain.paused_reason == "Ollama unreachable"


@pytest.mark.asyncio
async def test_brain_think_calls_ollama(state):
    brain = Brain(state)
    with patch.object(brain.ollama, "chat", new=AsyncMock(return_value="Hello prospect!")) as mock_chat:
        resp = await brain.think("Hi there", agent="writer", task="intro")
        assert resp == "Hello prospect!"
        mock_chat.assert_awaited_once()
        assert await state.get_usage_today("ollama") == 1


@pytest.mark.asyncio
async def test_brain_think_json(state):
    brain = Brain(state)
    with patch.object(brain.ollama, "chat", new=AsyncMock(return_value='```json\n{"score": 95, "intent": "interested"}\n```')):
        parsed = await brain.think_json("Analyze lead", agent="scout", task="score")
        assert parsed == {"score": 95, "intent": "interested"}


@pytest.mark.asyncio
async def test_brain_think_unavailable_stops(state):
    brain = Brain(state)
    with patch.object(brain.ollama, "chat", new=AsyncMock(side_effect=OllamaUnavailable("Model missing"))):
        resp = await brain.think("Hi")
        assert resp == ""
        assert "Model missing" in brain.paused_reason


@pytest.mark.asyncio
async def test_is_within_budget(state):
    brain = Brain(state)
    assert await brain.is_within_budget(100) is True
    for _ in range(5):
        await state.increment_usage(provider="ollama")
    assert await brain.is_within_budget(4) is False
    assert await brain.is_within_budget(10) is True
