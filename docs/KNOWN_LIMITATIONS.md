# Known Limitations & Project Scope

This document provides a transparent accounting of known limitations, synthetic test data files, hardware constraints, legacy database artifacts, and testing methodology boundaries.

---

## 1. Network Air-Gap Testing Boundary (Honesty Rule)

> [!IMPORTANT]
> **Air-Gap Verification Method**:  
> The automated offline tests (`scripts/offline_test.py`) were executed with all non-loopback network communication blocked **programmatically in code via a socket guard interceptor** (`socket.socket.connect` patched to reject any IP address outside `127.0.0.1`, `localhost`, `::1`, and `0.0.0.0`), verified by an active canary test against `8.8.8.8:53`.
> 
> **Limitation**: The system was **not tested with the network interface physically off** (i.e., Wi-Fi / Ethernet physically disconnected at the hardware level). Loopback sockets between Python and the local Ollama daemon at `127.0.0.1:11434` remain open and active.

---

## 2. Sample Data Files (Synthetic / Fictional)

The repository contains strictly fictional sample datasets used for demonstration, offline testing, and reproducibility. All sample files are explicitly watermarked with `"SAMPLE DATA"`:

1. **`data/prospects/sample_prospects.json`**:
   - Contains 15 fictional prospect companies across logistics, healthcare, cybersecurity, fintech, and enterprise software.
   - Each entry contains company name, industry, size, contact name, contact title, 40–80 word website description, and 2–3 dated recent news items.
   - Explicitly tagged with `"sample_data": "SAMPLE DATA"`.
2. **`data/replies/sample_threads.json`**:
   - Contains 20 fictional cold outreach reply email threads.
   - Balances across all 8 supported classification categories (`interested`, `objection`, `not_now`, `not_interested`, `unsubscribe`, `out_of_office`, `referral`, `unclear`).
   - Explicitly tagged with `"sample_flag": "SAMPLE DATA"`.
3. **`data/companies/sample_acme/` (`about.txt`, `products.txt`, `pricing.txt`)**:
   - Fictional B2B software company ("Acme Technologies Inc.") used for product onboarding and ICP generation tests.
   - Header marked with `"SAMPLE DATA: About Acme Technologies Inc."`.

---

## 3. Small-Model Quality Limits & Inference Latency

- **Generation Latency on Entry Hardware**:
  - Running `qwen3:1.7b` on an entry-level workstation (8 GB RAM / Apple Silicon) produces latencies of ~30 to 41 seconds for complex generative workflows (such as Problem Statement synthesis or Validated Email Drafting across a 3 prospect × 2 product matrix).
  - While fast classification (signals, reply handling) takes 1.7s–2.6s per item, bulk generative tasks should be dispatched asynchronously via the task queue rather than waited on synchronously in the UI.
- **Context Size Constraints**:
  - `qwen3:1.7b` operates with a default context window of 2,048 tokens (`num_ctx: 2048`). Very large stored prospect website dossiers must be chunked or retrieved via top-k embedding similarity snippets before being passed to the model.
- **Validation Retries**:
  - Compact 1.7B parameter models occasionally miss structured output constraints (e.g., omitting a direct call-to-action or exceeding word limits). Sellbird implements automatic in-code validation and single-retry loops to enforce quality.

---

## 4. Sending Removed (Drafts Only)

- In the current build, live email sending capabilities (via SMTP, Google Workspace, or Microsoft 365) have been disabled.
- All drafted emails are stored with `status = 'draft'` in the SQLite `outbox` table.
- Users can review, edit, approve, or reject drafts in the Outbox dashboard, but no emails are automatically transmitted to external mail servers.

---

## 5. Legacy SQLite Schema Columns

To maintain backward compatibility with previous versions of Harvey and avoid breaking database migrations, several legacy column names are retained in `sellbird/state.py`:
- `claude_calls` (integer column in `usage_log`)
- `codex_calls` (integer column in `usage_log`)
- `instantly_campaign_id` (text column in `prospects`)
- `provider TEXT DEFAULT 'claude'` (in `usage_events`)

These columns are legacy schema artifacts from older prototypes. In version 0.4.0+, they are neither queried by nor populated by any cloud service. Sellbird connects strictly to local loopback Ollama instances.

---

## 6. Development Tooling Disclosure

- An AI coding assistant was utilized strictly as an interactive software development and coding tool during codebase development, refactoring, and test writing.
- The compiled Sellbird software, runtime daemons, background workers, and CLI binaries make **zero cloud AI calls** and contain no cloud SDK dependencies.
