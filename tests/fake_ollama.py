"""A localhost stand-in for the Ollama API, returning canned JSON and text responses
for unit and integration tests with no real model and no network access.

Supports both programmatic usage (via pytest fixture `fake_ollama` or `FakeOllamaServer`
context manager) and standalone CLI process execution:
    python fake_ollama.py [log.jsonl] [port-file]
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
import pytest

DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_TEXT_REPLY = "Hello from fake_ollama!"
DEFAULT_JSON_REPLY = {
    "status": "ok",
    "summary": "fake ollama canned response",
    "score": 85,
    "intent": "interested",
    "category": "interested",
    "steps": [
        {
            "step": 1,
            "subject": "quick question",
            "body": "Hi there,\nSaw your website and had a question about your booking setup.\nBest,\nSam",
        }
    ],
}


class FakeOllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "FakeOllamaServer"

    def log_message(self, *args):  # Keep stderr clean during test runs
        pass

    def _send(self, code: int, payload: Any, content_type: str = "application/json"):
        if isinstance(payload, bytes):
            data = payload
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload).encode("utf-8")

        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @property
    def ollama(self) -> "FakeOllamaServer":
        return getattr(self.server, "ollama_server", None)

    def _record_request(self, body: Any = None):
        entry = {
            "method": self.command,
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": body,
            "timestamp": time.time(),
        }
        if self.ollama and hasattr(self.ollama, "record"):
            self.ollama.record(entry)

    def do_GET(self):
        self._record_request()
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "":
            return self._send(200, "Ollama is running", content_type="text/plain")
        if path == "/api/version":
            return self._send(200, {"version": "0.40.17"})
        if path == "/api/tags":
            return self._send(
                200,
                {
                    "models": [
                        {
                            "name": "qwen3:1.7b",
                            "model": "qwen3:1.7b",
                            "modified_at": "2026-11-23T11:00:00Z",
                            "size": 1400000000,
                            "details": {"format": "gguf", "family": "qwen"},
                        },
                        {
                            "name": "qwen3:4b",
                            "model": "qwen3:4b",
                            "modified_at": "2026-11-23T11:00:00Z",
                            "size": 2500000000,
                            "details": {"format": "gguf", "family": "qwen"},
                        },
                    ]
                },
            )
        return self._send(404, {"error": f"Endpoint {path} not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = raw.decode("utf-8", "replace")

        self._record_request(body)
        path = self.path.split("?", 1)[0]

        if path == "/api/show":
            return self._send(
                200,
                {
                    "license": "Apache 2.0",
                    "modelfile": "FROM qwen3",
                    "parameters": "stop <|im_end|>",
                    "template": "{{ .Prompt }}",
                },
            )

        if path in ("/api/generate", "/api/chat"):
            model = body.get("model", DEFAULT_MODEL) if isinstance(body, dict) else DEFAULT_MODEL
            stream = bool(body.get("stream", False)) if isinstance(body, dict) else False
            prompt = body.get("prompt", "") if isinstance(body, dict) else ""
            if not prompt and isinstance(body, dict) and "messages" in body:
                msgs = body.get("messages", [])
                if msgs and isinstance(msgs, list):
                    prompt = msgs[-1].get("content", "")

            # Determine response content
            canned = self.ollama.get_canned_reply(prompt, body) if self.ollama else DEFAULT_TEXT_REPLY
            if isinstance(canned, (dict, list)):
                reply_text = json.dumps(canned)
            else:
                reply_text = str(canned)

            if stream:
                # Stream NDJSON
                chunks = [
                    {"model": model, "response": reply_text, "done": False},
                    {"model": model, "response": "", "done": True, "total_duration": 1000},
                ]
                stream_data = "".join(json.dumps(c) + "\n" for c in chunks).encode("utf-8")
                return self._send(200, stream_data, content_type="application/x-ndjson")

            if path == "/api/chat":
                return self._send(
                    200,
                    {
                        "model": model,
                        "created_at": "2026-11-23T11:00:00Z",
                        "message": {"role": "assistant", "content": reply_text},
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": 1000000,
                        "load_duration": 100000,
                        "prompt_eval_count": 10,
                        "eval_count": 20,
                    },
                )

            # Default /api/generate
            return self._send(
                200,
                {
                    "model": model,
                    "created_at": "2026-11-23T11:00:00Z",
                    "response": reply_text,
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 1000000,
                    "load_duration": 100000,
                    "prompt_eval_count": 10,
                    "eval_count": 20,
                },
            )

        if path in ("/api/embeddings", "/api/embed"):
            return self._send(200, {"embedding": [0.01] * 128})

        return self._send(404, {"error": f"Unknown endpoint {path}"})


class FakeOllamaServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, log_path: Optional[str] = None):
        self.host = host
        self.requested_port = port
        self.log_path = log_path
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._custom_replies: List[Any] = []
        self._reply_fn: Optional[Callable[[str, Any], Any]] = None

    @property
    def port(self) -> int:
        if self.server:
            return self.server.server_address[1]
        return self.requested_port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def record(self, entry: Dict[str, Any]):
        with self._lock:
            self.requests.append(entry)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

    def set_canned_reply(self, reply: Any):
        with self._lock:
            self._custom_replies = [reply]

    def queue_canned_reply(self, reply: Any):
        with self._lock:
            self._custom_replies.append(reply)

    def set_reply_fn(self, fn: Callable[[str, Any], Any]):
        with self._lock:
            self._reply_fn = fn

    def get_canned_reply(self, prompt: str, body: Any) -> Any:
        with self._lock:
            if self._reply_fn is not None:
                return self._reply_fn(prompt, body)
            if self._custom_replies:
                if len(self._custom_replies) > 1:
                    return self._custom_replies.pop(0)
                return self._custom_replies[0]

        # Default intelligent response based on prompt
        p_lower = prompt.lower()
        is_json = (
            (isinstance(body, dict) and body.get("format") == "json")
            or "json" in p_lower
            or "respond only with valid json" in p_lower
        )

        if "classify" in p_lower:
            if "pricing" in p_lower or "interested" in p_lower or "cost" in p_lower:
                return "interested" if not is_json else {"intent": "interested"}
            if "unsubscribe" in p_lower or "stop" in p_lower:
                return "opt_out" if not is_json else {"intent": "opt_out"}
            return "question" if not is_json else {"intent": "question"}

        if is_json:
            return DEFAULT_JSON_REPLY

        return DEFAULT_TEXT_REPLY

    def start(self):
        self.server = ThreadingHTTPServer((self.host, self.requested_port), FakeOllamaHandler)
        self.server.ollama_server = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@pytest.fixture
def fake_ollama(monkeypatch):
    """Pytest fixture providing a local stub Ollama server."""
    with FakeOllamaServer() as s:
        monkeypatch.setenv("OLLAMA_HOST", s.url)
        monkeypatch.setenv("SELLBIRD_OLLAMA_URL", s.url)
        yield s


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    port_file = sys.argv[2] if len(sys.argv) > 2 else None
    server = FakeOllamaServer(log_path=log_file)
    server.start()
    if port_file:
        with open(port_file, "w") as f:
            f.write(str(server.port))
    else:
        print(f"Fake Ollama server running on {server.url}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()


# ── Shared Heartbeat Test Harness ──
import asyncio
from sellbird.config import EnvConfig, SellbirdConfig
from sellbird.state import StateManager
from tests.test_outbox_native import FakeProvider, _now_iso, seed_prospect
import sellbird.brain as brain_mod
import sellbird.main as main_mod

BASE_CONFIG = {
    "persona": {"name": "Sellbird", "company": "Acme Tools", "role": "BD",
                "email": "sellbird@acme.example", "linkedin": "", "tone": "direct"},
    "product": {"name": "Widget", "description": "Widgets for teams", "pricing": "$10",
                "key_benefits": ["fast"], "objection_responses": {}},
    "icp": {"industries": ["software"], "company_size": "10-50", "titles": ["VP"],
            "geography": ["US"]},
    "channels": {"email": {"provider": "smtp", "require_approval": True}},
}

FAKE_SUBPROCESS_REPLY = (
    "import json, sys; sys.stdin.read() if not sys.stdin.isatty() else None; "
    "print(json.dumps({'type': 'result', 'result': 'question', 'session_id': 's',"
    " 'usage': {'input_tokens': 3, 'output_tokens': 1}}))"
)


class Harness:
    """One disposable Sellbird install wired to fixtures for e2e tests."""

    def __init__(self, monkeypatch, tmp_path, provider="ollama", model="", db=None):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.state = StateManager(str(db or tmp_path / "sellbird.db"))
        self.mail = FakeProvider()
        self.cycles = 0
        self.stop_after = 1
        self.between = {}
        self.scout_runs = 0
        self.set_config(provider, model)

        async def fake_preflight(_self):
            return ""

        monkeypatch.setattr(brain_mod.Brain, "preflight", fake_preflight)
        monkeypatch.setattr(brain_mod, "RETRY_BASE_DELAY", 0)
        real_exec = asyncio.create_subprocess_exec

        async def exec_routed(program, *args, **kwargs):
            return await real_exec(sys.executable, "-c", FAKE_SUBPROCESS_REPLY, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_routed)

        import sellbird.agents.handler as handler_mod
        import sellbird.agents.scout as scout_mod
        import sellbird.agents.sender as sender_mod

        monkeypatch.setattr(handler_mod, "get_mail_provider", lambda c, e: self.mail)
        monkeypatch.setattr(sender_mod, "get_mail_provider", lambda c, e: self.mail)

        async def scout_run(_self):
            self.scout_runs += 1

        async def no_profile(state, limit=25, **_):
            return None

        monkeypatch.setattr(scout_mod.Scout, "run", scout_run)
        monkeypatch.setattr(main_mod, "run_profile_stage", no_profile)
        monkeypatch.setattr(main_mod, "load_config", lambda: self.config)
        monkeypatch.setattr(main_mod, "load_env", lambda: EnvConfig())
        monkeypatch.setattr(main_mod, "StateManager", lambda: self.state)
        monkeypatch.setattr(main_mod, "in_quiet_hours", lambda config: False)

        async def one_cycle_sleep(seconds, stop_event):
            self.cycles += 1
            step = self.between.get(self.cycles)
            if step is not None:
                result = step()
                if asyncio.iscoroutine(result):
                    await result
            return self.cycles >= self.stop_after

        monkeypatch.setattr(main_mod, "_interruptible_sleep", one_cycle_sleep)
        real_init = sender_mod.Sender.__init__

        def init_no_pacing(s, *a, **k):
            real_init(s, *a, **k)
            s.send_pacing = False

        monkeypatch.setattr(sender_mod.Sender, "__init__", init_no_pacing)

    def set_config(self, provider="ollama", model=""):
        if provider == "claude" or not provider:
            provider = "ollama"
        brain = {"provider": provider}
        if model:
            brain["model"] = model
        self.config = SellbirdConfig(**BASE_CONFIG, brain=brain)

    async def run_cycles(self, n=1, between=None):
        self.cycles = 0
        self.stop_after = n
        self.between = between or {}
        await asyncio.wait_for(main_mod.heartbeat(asyncio.Event()), 120)


async def _approved_item(state, email="bob@beta.com"):
    pid = await seed_prospect(state, email=email, status="contacted")
    return await state.add_outbox_item(
        prospect_id=pid, to_email=email, subject="following up",
        body="Hi Bob, any thoughts on this?", send_at=_now_iso(), status="approved",
        campaign_id="c-1", step=2,
    )

