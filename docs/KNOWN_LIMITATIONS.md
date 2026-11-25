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
   - Contains 32 fictional cold outreach reply email threads (20 clear standard threads + 12 harder indirect-wording threads added for robust boundary evaluation).
   - Balances across all 8 supported classification categories (`interested`, `objection`, `not_now`, `not_interested`, `unsubscribe`, `out_of_office`, `referral`, `unclear`).
   - Explicitly tagged with `"sample_flag": "SAMPLE DATA"`.
3. **`data/companies/sample_acme/` (`about.txt`, `products.txt`, `pricing.txt`)**:
   - Fictional B2B software company ("Acme Technologies Inc.") used for product onboarding and ICP generation tests.
   - Header marked with `"SAMPLE DATA: About Acme Technologies Inc."`.

---

## 3. Small-Model Quality Limits & Inference Latency

- **Reply Classification Weakness on 1.7B Model & Unmeasured Larger Models**:
  - The compact 1.7B parameter model (`qwen3:1.7b`) is weak on nuanced, indirect reply phrasing, scoring 16/32 (50.0%) accuracy on the 32 sample threads in `sellbird eval-replies`.
  - The model frequently misclassifies polite deferrals (e.g., "reach out next quarter during budget planning") and referrals (e.g., "looping in our VP of Infrastructure") as `not_interested` or `objection`.
  - A larger local model (`qwen3:4b` or `qwen3:8b`) is expected to achieve higher classification accuracy on nuanced language, but was not measured in this evidence benchmark.
  - While the deterministic heuristic fallback achieved 32/32 (100.0%) accuracy on the sample set, note that the heuristic rules were written alongside the sample set, so this is **NOT** a held-out measure of its quality on unseen real-world replies.
- **Generation Latency on Entry Hardware**:
  - Running `qwen3:1.7b` on an entry-level workstation (8 GB RAM / Apple Silicon) produces latencies of ~12 to 17 seconds for complex multi-item generative workflows (such as Problem Statement synthesis or Validated Email Drafting across a 3 prospect × 2 product matrix).
  - While fast classification (signals, reply handling) takes 0.9s–1.4s per item, bulk generative tasks should be dispatched asynchronously via the task queue rather than waited on synchronously in the UI.
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

---

## 7. Pytest Test Suite vs Real Inference Evidence

- The automated `pytest` test suite runs strictly against a lightweight stub model server (`fake_ollama`) with loopback network isolation. This ensures that unit and integration tests run deterministically, fast (<2 minutes for ~2,880 tests), and without requiring a GPU or active Ollama model daemon.
- Real model results, empirical token latencies, and output quality assessments come **only** from the empirical evidence report ([offline_report_20261125_131553.md](file:///Users/anandvardhanchauhan/projects/harvey/docs/evidence/offline_report_20261125_131553.md)) and the live `sellbird eval-replies` CLI run.
