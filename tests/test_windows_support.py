"""Windows: text files are UTF-8 whatever the locale.

Based on community PR #18 by Tariz Shahid.
"""

import ast
import io
from pathlib import Path

import pytest

SELLBIRD = Path(__file__).resolve().parent.parent / "sellbird"


def _text_io_without_encoding(tree):
    """open()/read_text()/write_text() calls in text mode with no encoding=."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("open", "read_text", "write_text"):
            continue
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and \
                func.value.id in ("os", "webbrowser", "aiosqlite", "sqlite3"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        mode = next((a.value for a in node.args[1:2] if isinstance(a, ast.Constant)), None)
        mode = next((k.value.value for k in node.keywords
                     if k.arg == "mode" and isinstance(k.value, ast.Constant)), mode)
        if name == "open" and isinstance(mode, str) and "b" in mode:
            continue
        yield node.lineno


def test_every_text_file_is_read_and_written_as_utf8():
    # Windows defaults to the locale code page (cp1252): sellbird train
    # crashed on "★" and prompts came out garbled.
    offenders = [
        f"{path.relative_to(SELLBIRD.parent)}:{line}"
        for path in sorted(SELLBIRD.rglob("*.py"))
        for line in _text_io_without_encoding(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []


def test_config_and_prompts_with_non_ascii_text_load(tmp_path, monkeypatch):
    from sellbird.brain import Brain
    import sellbird.brain as brain_mod
    from sellbird.config import load_config

    import yaml
    data = yaml.safe_load((SELLBIRD.parent / "sellbird.yaml").read_text(encoding="utf-8"))
    data["persona"]["company"] = "Acme — Studio ★"
    config = tmp_path / "sellbird.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    loaded = load_config(str(config))
    assert loaded.persona.company == "Acme — Studio ★"

    (tmp_path / "x.md").write_text("Rules ★ — {{name}}", encoding="utf-8")
    monkeypatch.setattr(brain_mod, "PROMPTS_DIR", tmp_path)
    assert Brain.load_prompt(object.__new__(Brain), "x", name="Zoë") == "Rules ★ — Zoë"
