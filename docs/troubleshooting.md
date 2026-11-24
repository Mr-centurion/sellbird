# Troubleshooting

## Troubleshooting

**`command not found: sellbird`** — `source .venv/bin/activate` first.

**`ModuleNotFoundError: No module named 'sellbird'` after install (macOS)** — Python 3.13 silently ignores `.pth` files carrying the macOS hidden flag, and some Macs propagate that flag into `.venv`. Run `sellbird install` again (it auto-fixes), or: `ln -s "$(pwd)/sellbird" .venv/lib/python3.13/site-packages/sellbird`

**`externally-managed-environment`** — Use a venv, not system Python.

**`SyntaxError` or odd `ImportError`s right after install (macOS)** — Your venv is Python 3.9. `python3 -m venv` builds from `/usr/bin/python3`, which macOS keeps at 3.9. Check with `.venv/bin/python -V`, then rebuild: `rm -rf .venv && python3.13 -m venv .venv && source .venv/bin/activate && pip install -e .`

**Ollama calls fail** — Ensure the Ollama service is running (`ollama serve`) and the configured model is installed (`ollama pull qwen3:1.7b`). Test with `sellbird status --check`.

**Dashboard says "locked"** — Open the link `sellbird dashboard` printed (or `sellbird dashboard --url`). If it says the token file can't be used, follow the message: usually delete `data/dashboard-token` and restart the dashboard. Visit it as `127.0.0.1` or `localhost`; other host names are refused.

**Discovery finds nothing** — Confirm signals first (`sellbird signals --confirm free`); nothing is collected until you do. OpenStreetMap (OSM) covers mapped local businesses and trades.

**Overpass is throttling** — It's free volunteer infrastructure. Sellbird walks three mirrors before giving up. Wait a few minutes before retrying.

**Sellbird does nothing during the day** — Check `quiet_hours`. `sellbird status` shows current state; the `actions` table logs every decision.

**How do I stop it right now?** — `Ctrl+C`, or `sellbird sending pause` to stop outbound while leaving the loop running. State is in SQLite, so it resumes cleanly.

**Where does my data live?** — All local: `data/sellbird.db`, `.env`, `sellbird.yaml`. Nothing goes anywhere except the APIs you configured.
