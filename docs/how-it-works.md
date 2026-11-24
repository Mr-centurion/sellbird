# How Sellbird works

## Why local inference matters

Every other autonomous sales agent bills you per token or uses third-party cloud services. An agent that thinks in a loop, all day, is an agent that burns cloud API credits in a loop, all day.

Sellbird runs 100% offline-first using local Ollama models (`qwen3:1.7b` by default, or `qwen3:4b`). No tokens are sent to any external provider. Sellbird handles prospecting, verification, drafting, and outbox operations locally on your machine.

**What this costs you:**

| | |
|---|---|
| Model inference | **$0** (Local Ollama) |
| Finding businesses | **$0** on the default OSM source |
| Reading their websites | **$0** — three HTTP requests each, no model call |
| Writing and outbox | **$0** — drafts generated locally to Outbox |
| Prospecting tools | **none.** No external verifier or search subscriptions. |

**Sellbird also budgets itself.** It tracks inference calls in `inference_log` and limits daily activity according to `usage.max_daily_percent`.

---

## What Sellbird actually does

Sellbird runs a heartbeat: wake up, check the budget, decide what most needs doing, do it, log it, sleep. Every 15 minutes by default.

```
                 ┌──────────────────────────────────────┐
                 ↓                                      │
   Wake  →  Quiet hours?  →  Budget OK?  →  Decide  →  Act  →  Log  →  Sleep
                 │               │
              (sleep)         (sleep)
```

It decides deterministically, not by asking a model what to do — that would burn a call on something derivable from four counts:

```
handle replies  >  send campaigns  >  write campaigns  >  prospect  >  idle (analyze)
```

Profiling and outbox draining ride along on **every** cycle regardless, because they cost nothing and have to happen on schedule.

## Project structure

```
sellbird/
├── main.py              # heartbeat loop
├── brain.py             # Ollama provider routing + prompt handling
├── ollama_runtime.py    # Local Ollama client with inference logging
├── state.py             # SQLite schema, migrations, observations, cohorts
├── signals.py           # the 23-signal catalog Sellbird proposes
├── pipeline.py          # DISCOVER → PROFILE chaining
├── gate.py              # deterministic pre-send checks
├── dashboard.py         # FastAPI JSON API
├── dashboard_auth.py    # dashboard token, Host/Origin checks
├── collectors/
│   ├── discover.py      # provider adapters, junk filtering, entity resolution
│   └── profile.py       # the free HTTP profiler
├── agents/              # scout, writer, sender, handler, analyst
├── integrations/        # mail providers, email finder, quota
└── web/                 # dashboard HTML/CSS/JS — no build step

skills/                  # ← the sales knowledge. Editable Markdown.
prompts/                 # ← agent system prompts. Also editable Markdown.
tests/                   # 242 tests
```

---

## Philosophy

**Autonomous doesn't mean unsupervised.** Quiet hours, spend caps, send limits, an approval queue, a deterministic pre-send gate, and a kill switch. Sellbird asks before it spends and before it sends, until you tell it not to.

**You decide what a good prospect is.** Sellbird proposes signals; you confirm them. A prospect list you can't explain is a prospect list you shouldn't send to.

**If it can be gotten deterministically, don't use a model.** Rank position, tech detection, junk filtering, the priority decision, the pre-send gate — all plain code. Ollama is reserved for the parts that need judgment.

**A failure is an observation, not silence.** "No team page found" is a finding. "Overpass is throttling" is a finding. Silent failures look exactly like clean results, which is how a broken run gets mistaken for an empty market.

**The database is the asset, not the agent.** The initial list is worth little. The same signals observed over quarters — who changed agencies, who started spending on ads, who let their content go stale — is worth a great deal.

**Everything is editable.** Prompts, skills, config, the dashboard. All plain text. You don't need to be a developer to change how Sellbird sells.
