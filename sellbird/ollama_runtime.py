"""Ollama runtime — Sellbird's local thinking engine.

Runs inference against a local Ollama instance (default: http://localhost:11434)
using open-weight models such as Qwen 3. Never falls back to cloud providers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from sellbird.paths import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "sellbird.db"

logger = logging.getLogger("sellbird.ollama")

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_NUM_CTX = 4096
DEFAULT_TIMEOUT_SECONDS = 120.0

INIT_INFERENCE_LOG_SQL = """
CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    agent TEXT,
    task TEXT,
    model TEXT NOT NULL,
    host TEXT NOT NULL,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms INTEGER,
    status TEXT NOT NULL
);
"""


class OllamaUnavailable(Exception):
    """Raised when Ollama is unreachable, returns an error status, or the model is missing."""


def strip_think_blocks(text: str) -> str:
    """Remove any <think>...</think> reasoning traces from reasoning models."""
    if not text:
        return ""
    # Strip paired tags including multiline content
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip unclosed opening or stray closing tag if any
    cleaned = re.sub(r"</?think>", "", cleaned)
    return cleaned.strip()


def validate_local_host(url: str) -> str:
    """Ensure Ollama URL points strictly to loopback/localhost."""
    raw = (url or DEFAULT_OLLAMA_URL).strip()
    if not re.match(r"^https?://", raw):
        raw = f"http://{raw}"
    parsed = urllib.parse.urlparse(raw)
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_HOSTS:
        raise ValueError(
            f"OllamaClient refuses non-local host '{hostname}'. "
            f"Sellbird runs strictly local models; host must be localhost, 127.0.0.1, or ::1."
        )
    port = parsed.port or 11434
    scheme = parsed.scheme or "http"
    return f"{scheme}://{hostname}:{port}"


class OllamaClient:
    """Async client communicating with local Ollama daemon."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        num_ctx: int = DEFAULT_NUM_CTX,
        db_path: Path | str | None = None,
    ):
        self.base_url = validate_local_host(base_url)
        self.host = urllib.parse.urlparse(self.base_url).hostname or "localhost"
        self.model = model.strip() or DEFAULT_MODEL
        self.last_model = self.model
        self.num_ctx = int(num_ctx) if num_ctx else DEFAULT_NUM_CTX
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self._db_initialized = False

    async def _ensure_log_table(self):
        """Ensure inference_log table exists in the database."""
        if self._db_initialized:
            return
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(INIT_INFERENCE_LOG_SQL)
                await db.commit()
            self._db_initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize inference_log table: {e}")

    async def log_inference(
        self,
        agent: str,
        task: str,
        prompt_tokens: int,
        output_tokens: int,
        latency_ms: int,
        status: str,
        model: str | None = None,
    ):
        """Record an inference call in the SQLite inference_log table."""
        await self._ensure_log_table()
        import aiosqlite

        agent = (agent or "").strip() or "brain"
        task = (task or "").strip() or "chat"
        log_model = (model or self.last_model or self.model).strip()
        ts = datetime.now(timezone.utc).isoformat()
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO inference_log (
                        ts, agent, task, model, host,
                        prompt_tokens, output_tokens, latency_ms, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        agent,
                        task,
                        log_model,
                        self.host,
                        prompt_tokens,
                        output_tokens,
                        latency_ms,
                        status,
                    ),
                )
                await db.commit()
        except Exception as e:
            logger.warning(f"Failed to record inference log row: {e}")

    async def preflight(self) -> str:
        """Check Ollama availability and model existence.

        Returns "" when ready, or a descriptive reason string when unavailable.
        """
        tags_url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(tags_url)
        except httpx.ConnectError:
            return (
                f"Ollama is unreachable at {self.base_url}. "
                "Ensure the Ollama service is running ('ollama serve' or 'brew services start ollama')."
            )
        except httpx.TimeoutException:
            return f"Ollama connection timed out at {self.base_url}."
        except Exception as e:
            return f"Ollama preflight check failed ({type(e).__name__}): {e}"

        if resp.status_code != 200:
            return f"Ollama returned HTTP {resp.status_code} at {tags_url}: {resp.text[:200]}"

        try:
            data = resp.json()
        except Exception:
            return f"Ollama returned non-JSON output from {tags_url}."

        models = data.get("models", [])
        model_names = set()
        for m in models:
            name = m.get("name", "")
            model_names.add(name)
            if ":" in name:
                model_names.add(name.split(":")[0])

        target = self.model
        target_base = target.split(":")[0] if ":" in target else target

        # Check exact or base match
        matched = any(
            target == m or f"{target}:latest" == m or m == target_base
            for m in model_names
        )
        if not matched:
            return (
                f"Model '{self.model}' not found in Ollama at {self.base_url}. "
                f"Download it with: 'ollama pull {self.model}'."
            )

        return ""

    async def chat(
        self,
        prompt: str,
        session_id: str | None = None,
        agent: str = "",
        task: str = "",
        expect_json: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        num_predict: int | None = None,
    ) -> str:
        """Send chat prompt to Ollama /api/chat and return cleaned response string."""
        agent = (agent or "").strip()
        if not agent and session_id:
            for known in ("trainer", "scout", "writer", "handler", "analyst"):
                if known in session_id.lower():
                    agent = known
                    break
        if not agent:
            agent = "brain"
        task = (task or "").strip() or (session_id or "") or "chat"
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self.num_ctx,
            },
        }
        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict
        if expect_json:
            payload["format"] = "json"

        start_time = time.monotonic()
        prompt_tokens = 0
        output_tokens = 0
        status = "error"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.ConnectError as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            await self.log_inference(agent, task, 0, 0, latency_ms, "unreachable")
            raise OllamaUnavailable(
                f"Ollama is unreachable at {self.base_url}. "
                "Ensure Ollama is running ('ollama serve')."
            ) from e
        except httpx.TimeoutException as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            await self.log_inference(agent, task, 0, 0, latency_ms, "timeout")
            raise OllamaUnavailable(
                f"Ollama call timed out after {timeout:.0f}s at {self.base_url}."
            ) from e
        except Exception as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            await self.log_inference(agent, task, 0, 0, latency_ms, f"error:{type(e).__name__}")
            raise OllamaUnavailable(f"Ollama request error: {e}") from e

        latency_ms = int((time.monotonic() - start_time) * 1000)

        if resp.status_code == 404:
            await self.log_inference(agent, task, 0, 0, latency_ms, "model_missing")
            raise OllamaUnavailable(
                f"Model '{self.model}' not found in Ollama. Pull it with: 'ollama pull {self.model}'"
            )

        if resp.status_code != 200:
            await self.log_inference(agent, task, 0, 0, latency_ms, f"http_{resp.status_code}")
            raise OllamaUnavailable(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            await self.log_inference(agent, task, 0, 0, latency_ms, "invalid_json_response")
            raise OllamaUnavailable(f"Invalid JSON returned by Ollama: {resp.text[:200]}") from e

        # Extract token counts from response if present
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)

        # Message content
        msg = data.get("message") or {}
        raw_text = msg.get("content") or data.get("response") or ""

        # Strip any <think> blocks
        cleaned_text = strip_think_blocks(raw_text)

        status = "ok"
        resp_model = data.get("model") or self.model
        self.last_model = resp_model
        await self.log_inference(
            agent=agent,
            task=task or (session_id or ""),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
            model=resp_model,
        )

        return cleaned_text
