"""Test that every dashboard route and API endpoint loads and produces no 500 responses.

Also verifies that client-side JavaScript assets (app.js, auth.js) are free of syntax errors.
"""

import asyncio
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount, Route

import sellbird.dashboard as dash
from sellbird.state import StateManager

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_JS = ROOT_DIR / "sellbird" / "web" / "app.js"
AUTH_JS = ROOT_DIR / "sellbird" / "web" / "auth.js"


def _discover_routes(app_or_router, prefix=""):
    """Collect (method, path) tuples for every route and mounted subrouter."""
    routes = []
    for r in getattr(app_or_router, "routes", []):
        if isinstance(r, Route):
            for m in sorted(r.methods - {"HEAD"}):
                routes.append((m, prefix + r.path))
        elif isinstance(r, Mount):
            routes.extend(_discover_routes(r.app, prefix=prefix + r.path))
        elif hasattr(r, "routes"):
            routes.extend(_discover_routes(r, prefix=prefix))
        elif hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
            routes.extend(_discover_routes(r.original_router, prefix=prefix))
    return sorted(routes)


ALL_ROUTES = _discover_routes(dash.app)


def test_app_js_syntax():
    """Ensure app.js has no syntax errors (such as dangling statements or illegal returns)."""
    assert APP_JS.exists(), f"{APP_JS} missing"
    assert AUTH_JS.exists(), f"{AUTH_JS} missing"
    node = shutil.which("node")
    if node:
        res1 = subprocess.run([node, "-c", str(APP_JS)], capture_output=True, text=True)
        assert res1.returncode == 0, f"app.js syntax error:\n{res1.stderr}"
        res2 = subprocess.run([node, "-c", str(AUTH_JS)], capture_output=True, text=True)
        assert res2.returncode == 0, f"auth.js syntax error:\n{res2.stderr}"


@pytest.fixture
def dash_client(tmp_path, monkeypatch):
    """Authenticated TestClient on an isolated database."""
    (tmp_path / "data").mkdir()
    db_path = tmp_path / "data" / "sellbird.db"
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "DB_PATH", db_path)
    monkeypatch.setattr(dash, "PID_FILE", tmp_path / "data" / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", tmp_path / "data" / "sellbird.log")
    monkeypatch.setattr(dash, "WEB_DIR", ROOT_DIR / "sellbird" / "web")

    dash.token_store.configure(tmp_path / "data" / "dashboard-token")
    token = dash.token_store.get()

    asyncio.run(StateManager(str(db_path)).init_db())

    with patch("sellbird.offer.suggest_offers", return_value=["Mock Offer"]), \
         patch("sellbird.offer.preview_email", return_value={"subject": "Mock", "body": "Mock"}):
        with TestClient(dash.app, base_url="http://127.0.0.1:5555",
                        headers={"Authorization": f"Bearer {token}"}) as client:
            yield client


def test_route_count():
    """Verify route discovery discovered all endpoints."""
    assert len(ALL_ROUTES) >= 65, f"Expected at least 65 routes, found {len(ALL_ROUTES)}"


@pytest.mark.parametrize("method, raw_path", ALL_ROUTES)
def test_dashboard_route_no_500(dash_client, method, raw_path):
    """Load every route/endpoint and assert it never returns a 500 Internal Server Error."""
    path = raw_path
    if "{path:path}" in path:
        path = path.replace("{path:path}", "app.js")
    path = re.sub(r"\{[^}]+\}", "1", path)

    kwargs = {}
    if method in ("POST", "PUT", "PATCH"):
        kwargs["json"] = {}

    resp = dash_client.request(method, path, **kwargs)

    assert resp.status_code != 500, (
        f"{method} {path} returned 500 Internal Server Error:\n{resp.text}"
    )
    assert resp.status_code < 500, (
        f"{method} {path} returned server error status {resp.status_code}:\n{resp.text}"
    )


def test_public_routes_anonymous_access(tmp_path, monkeypatch):
    """Verify public routes are reachable without token and protected routes return 401."""
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "DB_PATH", tmp_path / "data" / "sellbird.db")
    monkeypatch.setattr(dash, "PID_FILE", tmp_path / "data" / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", tmp_path / "data" / "sellbird.log")
    monkeypatch.setattr(dash, "WEB_DIR", ROOT_DIR / "sellbird" / "web")
    dash.token_store.configure(tmp_path / "data" / "dashboard-token")

    with TestClient(dash.app, base_url="http://127.0.0.1:5555") as anon:
        # Static and shell routes
        assert anon.get("/").status_code == 200
        assert anon.get("/static/app.js").status_code == 200
        assert anon.get("/api/instance").status_code == 200

        # Protected routes return 401
        assert anon.get("/api/companies").status_code == 401
