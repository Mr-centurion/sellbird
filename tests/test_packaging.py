"""Opt-in: build the wheel from a copy of the sources, install it into an
empty target, and use it with no working-tree imports.

    SELLBIRD_PACKAGE_TESTS=1 pytest tests/test_packaging.py

Needs setuptools in the running environment (no network: the build runs
without isolation). Covers the CLI entry point and dashboard assets.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE = Path(__file__).with_name("fake_ollama.py")

pytestmark = pytest.mark.skipif(
    os.environ.get("SELLBIRD_PACKAGE_TESTS") != "1" or sys.platform == "win32",
    reason="set SELLBIRD_PACKAGE_TESTS=1 (needs setuptools)",
)

CHECK = textwrap.dedent('''
    import asyncio, base64, contextlib, io, json, os, sys, tempfile, time
    from datetime import datetime, timezone
    SITE, FAKE, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, SITE)
    # An editable install of the checkout must not shadow the wheel.
    sys.meta_path[:] = [f for f in sys.meta_path
                        if "editable" not in type(f).__name__.lower()
                        and "editable" not in getattr(f, "__module__", "")]
    import sellbird
    from sellbird.brain import Brain
    from sellbird.config import BrainConfig
    from sellbird.state import StateManager
    from sellbird.ollama_runtime import OllamaClient
    assert os.path.realpath(sellbird.__file__).startswith(os.path.realpath(SITE)), sellbird.__file__

    def cli(*argv):
        from sellbird.cli import main
        out = io.StringIO()
        sys.argv = ["sellbird", *argv]
        with contextlib.redirect_stdout(out):
            try:
                main()
            except SystemExit:
                pass
        return out.getvalue()

    assert "train" in cli("--help")
    train = cli("train", "--help")
    assert "--provider" in train and "--model" in train

    async def main():
        state = StateManager(os.path.join(WORK, "sellbird.db"))
        await state.init_db()
        brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b"))
        assert brain.provider == "ollama"

    asyncio.run(main())
    import sellbird.dashboard as dash
    from sellbird import ollama_runtime  # noqa: F401
    web = os.path.join(os.path.dirname(sellbird.__file__), "web")
    assert os.path.exists(os.path.join(web, "app.js"))
    assert os.path.exists(os.path.join(web, "auth.js"))
    # The installed dashboard is locked: the token lives under WORK here.
    from fastapi.testclient import TestClient
    dash.token_store.configure(os.path.join(WORK, "data", "dashboard-token"))
    dash.DB_PATH = __import__("pathlib").Path(WORK, "dash.db")
    token = dash.token_store.get()
    with TestClient(dash.app, base_url="http://127.0.0.1:5555") as c:
        assert c.get("/").status_code == 200 and token not in c.get("/").text
        assert c.get("/static/auth.js").status_code == 200
        assert c.get("/api/companies").status_code == 401
        assert c.get("/openapi.json").status_code == 401
        ok = {"Authorization": "Bearer " + token}
        assert c.get("/api/companies", headers=ok).status_code == 200
        assert c.get("/openapi.json", headers=ok).status_code == 404
    print("WHEEL-OK")
''')


def test_installed_wheel_runs_without_the_checkout(tmp_path):
    pytest.importorskip("setuptools")
    src = tmp_path / "src"
    src.mkdir()
    # Everything pyproject names: the package and the defaults it ships.
    for name in ("pyproject.toml", "README.md", "LICENSE", "sellbird.yaml", ".env.example"):
        shutil.copy(ROOT / name, src / name)
    for folder in ("sellbird", "prompts", "skills"):
        shutil.copytree(ROOT / folder, src / folder,
                        ignore=shutil.ignore_patterns("__pycache__"))
    dist, site, work = tmp_path / "dist", tmp_path / "site", tmp_path / "work"
    work.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps",
                    "--no-build-isolation", "-w", str(dist), str(src)],
                   check=True, cwd=tmp_path, timeout=300)
    wheel = next(dist.glob("sellbird-*.whl"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--target", str(site), str(wheel)], check=True, timeout=300)
    script = tmp_path / "check.py"
    script.write_text(CHECK)
    # An installed package uses the per-user home folder; keep it in tmp_path.
    env = {**os.environ, "SELLBIRD_HOME": str(work / "home")}
    out = subprocess.run([sys.executable, "-I", str(script), str(site), str(FAKE), str(work)],
                         cwd=work, capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "WHEEL-OK" in out.stdout
