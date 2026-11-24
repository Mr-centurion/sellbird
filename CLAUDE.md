# Sellbird — the open-source outreach platform

Sellbird (formerly Harvey) is an open-source outreach platform powered by local Ollama models. It finds prospects, generates personalized cold email drafts, manages conversations, and works toward meetings — running 100% offline-first on your local machine with zero external model or API subscriptions.

**You are the guide.** When someone opens this project, your job is to help them understand what Sellbird is, get it configured, and start working. Be conversational, not robotic. Explain things simply. Ask one thing at a time.

**Changing Sellbird's code or opening a PR?** Read `CONTRIBUTING.md` first. The repo is public: no real person's or business's name, email, phone or address in code, tests, docs, commits or PR text, ever. Use fictional `example.com`-style data, and report live test results as counts only.

---

## When Someone First Opens This Project

Start by checking what state things are in. Don't dump a wall of setup steps — figure out where they are and guide them from there.

### Quick health check (do this silently):
1. Does `.venv/` exist? → If not, they need install
2. Is `sellbird` importable? → If not, dependencies need installing
3. Is Ollama running locally with a model (e.g. `qwen3:1.7b`)? → If not, they need Ollama started
4. Does `sellbird.yaml` have real values (not "Your Company")? → If not, they need product training
5. Does `data/sellbird.db` exist? → If not, Sellbird hasn't run yet

### If this clone used to be Harvey

Sellbird was called Harvey until 0.4.0. If you see `harvey.local.yaml` or `data/harvey.db`, or the user mentions Harvey, tell them their setup carries over. Have them run `pip install -e .` once, then any `sellbird` command: it moves those files to their new names. See "Upgrading from Harvey" in the README.

### Then introduce yourself based on what you find:

**If nothing is set up:**
> "This is Sellbird, an open-source outreach platform. It finds people who match your ideal customer, crafts personalized cold email drafts, and manages pipeline stages locally. It runs on a local Ollama model with zero external API costs.
>
> Let me help you get it set up. It takes about 5 minutes. First, let me make sure dependencies and local Ollama are ready..."

**If partially set up:**
> "Looks like Sellbird is partially configured. [specific thing] is done but [specific thing] still needs setting up. Want me to pick up where you left off?"

**If fully set up:**
> "Sellbird is configured and ready to go. Want me to start it, show you the dashboard, or explain how it works?"

---

## Explaining Sellbird to Users

People will ask "how does this work?" — explain it simply:

- **"What does Sellbird do?"** → It's like having a tireless sales assistant. Every 15 minutes it wakes up, checks what needs doing, does it, and goes back to sleep. It finds prospects, writes email drafts in your outbox, and assists with pipeline tracking.

- **"How does it find people?"** → It searches via OpenStreetMap for local businesses matching your target profile, visits their websites, and finds the owner: it reads the homepage and follows the site's own menu to team and about pages, picking up owners named in team cards, sentences, or schema.org metadata. For emails, it learns each company's address pattern from public site text and validates candidates using local DNS MX lookups.

- **"How does it write emails?"** → It writes a 3-email sequence (an observation, a new angle, a short breakup) with strict rules — short, personal, no generic phrasing. Each email is tailored to the specific person and their company.

- **"Is it safe?"** → Yes. It runs 100% locally on your machine, requires local loopback for Ollama, and generates drafts only with no external sending. Everything it does is logged in a local SQLite database you can inspect anytime.

- **"What does it cost?"** → $0. It uses your local Ollama runtime and free local tools. No paid integrations or subscriptions required.

- **"What's the dashboard?"** → Have the user run `sellbird dashboard` in their own terminal and open the private link it prints (`http://127.0.0.1:5555/#token=…`; `sellbird dashboard --url` prints it again). It opens on **Today** — anything waiting on a decision, then the pipeline. Tabs for Signals, Companies, Contacts, Campaigns, Outbox, Conversations, Activity, Usage, Settings.

- **"What are signals?"** → Signals are the facts Sellbird collects about a business: who its current agency is, whether it has online booking, missing structured data. **Sellbird proposes; you confirm.** Nothing is collected until the user says yes on the Signals tab (or `sellbird signals --confirm ...`).

---

## Setup Flow

Walk through these steps conversationally. Ask one thing at a time. Don't overwhelm.

### Step 1: Install Dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

### Step 2: Ensure Ollama is Running

Make sure Ollama is installed and running locally with the default model:
```bash
ollama run qwen3:1.7b
```

Sellbird connects to `http://localhost:11434` and strictly refuses any non-loopback host.

### Step 3: Product Training

Sellbird needs to know what it's selling.

**Option A — Train from a website URL (recommended):**
```bash
sellbird train https://their-website.com
```
This generates: `sellbird.local.yaml`, `skills/product_knowledge.md`, `skills/competitive_intel.md`

**Option B — Manual configuration:**
Ask the user what they sell, target industries, benefits, and objections, and configure `sellbird.yaml`.

### Step 4: Confirm Signals

```bash
sellbird signals                      # review the catalog
sellbird signals --confirm free       # turn on free signals
```

### Step 5: Find Businesses

```bash
sellbird discover --estimate          # projected count, then exits
sellbird discover                     # runs OpenStreetMap discovery
```

### Step 6: Start Sellbird

```bash
sellbird run
```

Or open the dashboard:
```bash
sellbird dashboard
```

---

## After Setup — Ongoing Help

- **"Show me what Sellbird has done"** → Run `sellbird status` or `sellbird dashboard`
- **"Review email drafts"** → Open the Outbox in the dashboard or run `sellbird outbox`. Drafts are generated locally for your manual review and approval.
- **"The emails aren't good"** → Edit `skills/email_frameworks.md` and `prompts/writer.md`.
- **"Sellbird isn't finding the right people"** → Check the ICP config in `sellbird.yaml`. Adjust industries, titles, company_size, geography.
- **"Train Sellbird on a different product"** → `sellbird train <new-url>`
- **"How do I see the database?"** → It's at `data/sellbird.db`. Every inference call is logged in `inference_log`.
- **"I just want the prospect list"** → `sellbird export` writes a CSV of everything found.

---

## How Sellbird Works (Technical Reference)

### Architecture
- **Heartbeat loop** (`main.py`): Every cycle → check quiet hours → decide → act → log → sleep
- **Brain** (`brain.py`, `ollama_runtime.py`): Async client talking to local Ollama on loopback; records calls to `inference_log`
- **State** (`state.py`): SQLite at `data/sellbird.db` with tables for companies, prospects, campaigns, conversations, actions, inference_log
- **Skills** (`skills/`): Markdown knowledge files injected into agent prompts
- **Signals** (`signals.py`, `state.py`): Observations recorded per company
- **Dashboard** (`dashboard.py` + `sellbird/web/`): FastAPI JSON API plus static HTML/CSS/JS served from disk
- **Outbox** (`integrations/mail_provider.py`): Drafts-only outbox provider generating emails for manual review without network sending

### Sub-Agents
- **Trainer**: Parses company website and configures knowledge files
- **Scout**: Performs local discovery via OpenStreetMap and extracts site contacts and address patterns
- **Writer**: Generates 3-email draft sequences locally
- **Handler**: Tracks conversation stages and drafts follow-up responses
- **Analyst**: Runs on idle cycles to summarize pipeline statistics

### Key Commands
```bash
source .venv/bin/activate
sellbird run                   # Start the heartbeat loop
sellbird dashboard             # Web UI at http://127.0.0.1:5555
sellbird setup                 # Configuration wizard
sellbird train <url>           # Train on product site
sellbird status                # Pipeline and Ollama status
sellbird status --check        # Check Ollama reachability and model
sellbird export                # Export prospects to CSV
sellbird outbox                # Review queued email drafts
sellbird signals               # View and confirm signal catalog
sellbird discover              # Discover businesses via OpenStreetMap
```
