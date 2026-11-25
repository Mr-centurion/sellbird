# Offline Capability & Failure Handling

Sellbird is engineered from the ground up to operate offline-first. This document outlines exactly what capabilities run without internet access, what external services require network connectivity, how the system behaves if the local Ollama daemon becomes unreachable, and which workflows provide deterministic fallbacks.

---

## 1. Feature Offline Matrix

| Feature / Subsystem | Offline Capable? | Network Requirements | Details |
|---|---|---|---|
| **All 8 AI Workflows** | **Yes (100% Offline)** | None | Runs entirely against local Ollama (`localhost:11434`) and local embeddings. |
| **Local Web Dashboard** | **Yes (100% Offline)** | None | Served locally via FastAPI/uvicorn at `http://127.0.0.1:5555`. Assets are bundled vanilla JS/CSS. |
| **SQLite Storage & Querying** | **Yes (100% Offline)** | None | Stored in local file `data/sellbird.db` via `aiosqlite`. |
| **Vector Embeddings & Cache** | **Yes (100% Offline)** | None | `all-MiniLM-L6-v2` runs locally via PyTorch/Sentence-Transformers; embeddings cached in SQLite table `embedding_cache`. |
| **Sample Data & Prospect Import** | **Yes (100% Offline)** | None | `sellbird import-prospects <file>` loads local JSON files directly into SQLite without network calls. |
| **Reply Evaluation CLI** | **Yes (100% Offline)** | None | `sellbird eval-replies` reads `data/replies/sample_threads.json` and evaluates accuracy locally. |
| **Initial Model Weights Download** | No | One-time internet | Downloading Ollama models (`ollama pull qwen3:1.7b`) and sentence-transformer weights requires an initial connection. Once downloaded, all models run strictly offline. |
| **OpenStreetMap Company Discovery** | No | Internet required | `sellbird discover` queries the public Overpass API for local business discovery. |
| **Live Website Crawling & Verification** | No | Internet required | Real-time web crawling (`Scout`), MX record lookups, and SMTP syntax validation require network access to prospect sites. *(Note: Not required when using stored research or sample data).* |
| **Email Sending** | No (currently disabled) | Internet required | SMTP or Google/Microsoft 365 dispatching requires connectivity. In the current build, all emails are saved as `draft` only and are not sent. |

---

## 2. Ollama Outage & Downtime Behavior

If the local Ollama server stops running, crashes, or the configured model is not installed, Sellbird gracefully degrades rather than crashing the application:

### A. Dashboard Alert Banner
The web interface dynamically polls Ollama health (`GET /api/ollama/status`) and renders a prominent red alert banner at the top of the interface:
- **Ollama Offline**: `"Ollama is offline or unreachable. Cannot reach Ollama at http://localhost:11434. Local inference is paused; new tasks will queue."`
- **Model Missing**: `"Configured model not installed. Ollama is running, but model 'qwen3:1.7b' is not pulled. Run: ollama pull qwen3:1.7b"`

### B. Persistent SQLite Task Queue
When Ollama is unreachable, generative requests (e.g., email drafting) do not fail silently. Instead:
1. The request payload is serialized and inserted into the `tasks` SQLite table with `status = 'pending'`.
2. The background worker (`sellbird/tasks.py`) periodically checks Ollama availability.
3. Once Ollama returns to service, queued tasks are dequeued, processed, and stored in order.

### C. Non-LLM Mode Activation
For analytical workflows that support deterministic logic, the system bypasses the LLM and executes rule-based algorithms, tagging the output with a visible **"non-LLM mode"** badge in the database and dashboard UI.

---

## 3. Workflow Deterministic Fallback Matrix

| # | Workflow | Has Deterministic Fallback? | Fallback Mechanism | Output Label |
|---|---|---|---|---|
| **1** | Business Profile | No | Pauses / queues task. Raw stored text is preserved in DB until Ollama is online. | N/A |
| **2** | ICP Generation | No | Pauses / queues task. Requires LLM to synthesize target customer attributes. | N/A |
| **3** | Stored Research | Partial | Top-k snippet retrieval works offline via cached embeddings; LLM summary generation queues until Ollama is up. | N/A |
| **4** | Buying Signals | **Yes** | Fallback to regex detector (`detect_signals_regex`) scanning for keyword patterns across all 8 signal categories. | `"non-LLM mode"` |
| **5** | Problem Statements | **Yes** | Rule-based template generator (`_generate_deterministic_problem_statement`) synthesizing ICP pain points and buying signals. | `"non-LLM mode"` |
| **6** | Validated Email Drafting | No | Generates no low-quality boilerplate; enqueues drafting task in SQLite `tasks` table for automatic retry. | N/A |
| **7** | Lead Scoring | **Yes** | Numeric score calculation (`calculate_score_breakdown`) is 100% deterministic code (rule weights + cached cosine similarity); rationale falls back to rule-based template. | `"non-LLM mode"` |
| **8** | Reply Classification | **Yes** | Fallback to rule-based keyword heuristic (`classify_reply_heuristic`) covering all 8 reply labels. | `"non-LLM mode"` |
