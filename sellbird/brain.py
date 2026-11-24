"""Brain — Sellbird's local thinking engine.

Runs on local Ollama using open-weight models (default: qwen3:1.7b).
The provider is fixed to Ollama; it never falls back to any cloud provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from sellbird.ollama_runtime import OllamaClient, OllamaUnavailable
from sellbird.paths import PROJECT_ROOT
from sellbird.state import StateManager
from sellbird.textio import read_user_text
from sellbird.workspace_home import current_home

logger = logging.getLogger("sellbird.brain")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"

# Subprocess / HTTP safety limits
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
RETRY_BASE_DELAY = 5.0

PROVIDER_LABELS = {"ollama": "Ollama"}

# Settings key holding the brain's last health transition, for `sellbird
# status` and the dashboard: {"provider", "status": "ok"|"paused", "reason", "since"}.
BRAIN_STATUS_KEY = "brain_status"


def parse_brain_status(raw: str) -> dict:
    """The stored BRAIN_STATUS_KEY value as a dict ({} if absent or bad)."""
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def active_brain_status(raw: str, provider: str = "ollama") -> dict:
    """The stored status if it belongs to ``provider``, else {}."""
    status = parse_brain_status(raw)
    return status if status.get("provider") == provider else {}


class BrainUnavailable(Exception):
    """The provider can't be used until the operator acts."""


def _read_knowledge(folder: str, name: str) -> str:
    """``<home>/<folder>/<name>.md``, or the shipped copy when the home has
    none. Raises FileNotFoundError when neither exists."""
    own = getattr(current_home(), f"{folder}_dir") / f"{name}.md"
    try:
        return read_user_text(own)
    except FileNotFoundError:
        from sellbird.home import bundled_defaults

        shipped = bundled_defaults()
        if shipped is None or (shipped / folder / f"{name}.md") == own:
            raise
        return read_user_text(shipped / folder / f"{name}.md")


class Brain:
    def __init__(self, state: StateManager, brain_config=None, transport=None):
        """``transport``, when given, replaces the local Ollama runtime: an
        ``async transport(prompt, *, agent, task, timeout, expect_json) -> str``
        that returns the model's text ("" on failure). It may carry a
        ``paused_reason`` attribute (str) to pause calls."""
        from sellbird.config import BrainConfig

        self.state = state
        self.transport = transport
        self.config = brain_config or BrainConfig()
        self.provider = "ollama"
        self.model = getattr(self.config, "model", None) or "qwen3:1.7b"
        self.ollama_url = getattr(self.config, "ollama_url", "http://localhost:11434")
        self.num_ctx = getattr(self.config, "num_ctx", 4096)
        # Whether the last think_json got no reply at all (outage or timeout).
        self.last_reply_empty = False
        self.unavailable = ""
        self._unavailable_at = 0.0
        self._status_synced = False
        self.ollama = OllamaClient(
            base_url=self.ollama_url,
            model=self.model,
            num_ctx=self.num_ctx,
            db_path=getattr(self.state, "db_path", None) if self.state is not None else None,
        )

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, "Ollama")

    @property
    def paused_reason(self) -> str:
        """Why calls are currently being skipped ("" when they aren't)."""
        transport_paused = (
            str(getattr(self.transport, "paused_reason", "") or "")
            if self.transport is not None
            else ""
        )
        if transport_paused:
            return transport_paused
        if self.unavailable:
            return self.unavailable
        return ""

    async def preflight(self) -> str:
        """Can this Brain be used right now? Returns "" or the reason why not."""
        reason = await self.ollama.preflight()
        if reason:
            await self._mark_unavailable(reason)
            return reason
        if self.unavailable:
            await self._mark_available()
        elif not self._status_synced:
            self._status_synced = True
            if self.state is not None:
                try:
                    stored = parse_brain_status(await self.state.get_setting(BRAIN_STATUS_KEY))
                except Exception:
                    stored = {}
                if stored.get("status") == "paused":
                    await self._mark_available()
        return ""

    async def _mark_unavailable(self, error: Any):
        err_msg = str(error)
        first = not self.unavailable or self.unavailable != err_msg
        self.unavailable = err_msg
        self._unavailable_at = time.monotonic()
        if first:
            await self._record_status("paused", self.unavailable, kind="setup")

    async def _mark_available(self):
        logger.info(f"{self.provider_label} is available again; resuming.")
        self.unavailable = ""
        await self._record_status("ok", "")

    async def _record_status(self, status: str, reason: str, **extra):
        if self.state is None:
            return
        try:
            await self.state.set_setting(
                BRAIN_STATUS_KEY,
                json.dumps(
                    {
                        "provider": self.provider,
                        "status": status,
                        "reason": reason[:500],
                        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        **extra,
                    }
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to record brain status: {e}")

    async def think(
        self,
        prompt: str,
        session_id: str | None = None,
        expect_json: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        agent: str = "",
        task: str = "",
    ) -> str:
        """Send a prompt to Ollama and return the response text.

        Retries transient failures with exponential backoff and enforces a
        hard timeout. Returns "" on unrecoverable failure.
        """
        agent = (agent or "").strip()
        if not agent:
            if session_id:
                for known in ("trainer", "scout", "writer", "handler", "analyst"):
                    if known in session_id.lower():
                        agent = known
                        break
            if not agent:
                agent = "brain"

        if self.transport is not None:
            return await self._think_transport(
                prompt,
                expect_json=expect_json,
                timeout=timeout,
                agent=agent,
                task=task,
            )

        paused = self.paused_reason
        if paused:
            logger.debug(f"Ollama call skipped: {paused}")
            return ""

        logger.debug(f"Brain call via Ollama (session={session_id}): {prompt[:100]}...")

        last_error = ""
        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info(
                    f"Retrying Ollama call in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries + 1})..."
                )
                await asyncio.sleep(delay)

            try:
                response = await self.ollama.chat(
                    prompt=prompt,
                    session_id=session_id,
                    agent=agent,
                    task=task or (session_id or ""),
                    expect_json=expect_json,
                    timeout=timeout,
                )
                if self.state is not None:
                    try:
                        await self.state.increment_usage(provider="ollama")
                    except Exception as e:
                        logger.warning(f"Failed to record usage counter: {e}")

                logger.debug(f"Brain response: {response[:200]}...")
                return response
            except OllamaUnavailable as e:
                logger.error(f"Ollama call failed: {e}")
                await self._mark_unavailable(e)
                return ""
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Brain error: {e}")
                last_error = str(e)
                continue

        logger.error(
            f"Brain call failed after {max_retries + 1} attempts. "
            f"Last error: {str(last_error)[:200]}"
        )
        return ""

    async def think_json(
        self,
        prompt: str,
        session_id: str | None = None,
        agent: str = "",
        task: str = "",
    ) -> dict | list | None:
        """Send a prompt and parse the response as JSON."""
        full_prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. No markdown, no explanation."
        )
        response = await self.think(
            full_prompt, session_id=session_id, expect_json=True, agent=agent, task=task
        )
        self.last_reply_empty = not response
        if not response:
            return None
        parsed = self._extract_json(response)
        if parsed is None:
            logger.error(f"Failed to parse JSON from brain: {response[:200]}")
        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict | list | None:
        """Best-effort JSON extraction: strips code fences, then falls back to
        locating the outermost JSON object/array in surrounding prose."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fall back: model wrapped the JSON in explanation text
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    async def check_usage(self) -> float:
        """Check today's usage on Ollama (call count)."""
        if self.state is None:
            return 0.0
        try:
            return await self.state.get_usage_today(provider="ollama")
        except Exception as e:
            logger.warning(f"Could not read usage from state: {e}")
            return 0.0

    async def is_within_budget(
        self, max_daily_calls: int = 200, max_percent: float | None = None
    ) -> bool:
        """Check whether Sellbird is within call budget."""
        calls = await self.check_usage()
        return calls < max_daily_calls

    async def _think_transport(
        self,
        prompt: str,
        *,
        expect_json: bool,
        timeout: float,
        agent: str,
        task: str,
    ) -> str:
        """One call through ``self.transport``. Never raises."""
        if self.paused_reason:
            return ""
        try:
            text = await asyncio.wait_for(
                self.transport(
                    prompt,
                    agent=agent,
                    task=task,
                    timeout=timeout,
                    expect_json=expect_json,
                ),
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(
                f"Brain transport call failed ({agent}/{task}): {type(e).__name__}: {e}"
            )
            return ""
        return str(text or "")

    def load_prompt(self, prompt_name: str, **kwargs) -> str:
        """Load a prompt template from the prompts/ directory and fill in variables."""
        try:
            template = _read_knowledge("prompts", prompt_name)
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_name}.md")
            return ""
        except OSError as e:
            logger.error(f"Could not read prompt file {prompt_name}.md: {e}")
            return ""
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        leftover = re.findall(r"\{\{(\w+)\}\}", template)
        if leftover:
            logger.warning(
                f"Prompt '{prompt_name}' has unfilled variables: {sorted(set(leftover))}"
            )
        return template

    def load_skill(self, skill_name: str) -> str:
        """Load a skill knowledge file from the skills/ directory."""
        try:
            return _read_knowledge("skills", skill_name)
        except FileNotFoundError:
            logger.warning(f"Skill file not found: {skill_name}.md")
            return ""
        except OSError as e:
            logger.error(f"Could not read skill file {skill_name}.md: {e}")
            return ""

    def load_skills_for_agent(self, agent_name: str) -> str:
        """Load all relevant skills for a specific sub-agent."""
        skill_map = {
            "scout": [
                "prospecting_tactics",
                "lead_qualification",
                "account_navigation",
                "signal_playbook",
                "product_knowledge",
            ],
            "writer": ["product_knowledge", "competitive_intel"],
            "handler": [
                "objection_handling",
                "sales_methodology",
                "offer_strategy",
                "product_knowledge",
                "competitive_intel",
            ],
            "sender": ["email_frameworks", "product_knowledge"],
        }

        skill_names = skill_map.get(agent_name, [])
        if not skill_names:
            return ""

        sections = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                sections.append(content)

        if not sections:
            return ""

        return (
            "\n\n---\n## FOUNDATIONAL KNOWLEDGE\n"
            "Use the following frameworks and best practices to guide your work:\n\n"
            + "\n\n---\n\n".join(sections)
        )
