# Testing Evidence & Verification Results

This document summarizes the testing evidence for Sellbird, encompassing automated unit/integration test counts, empirical air-gapped run metrics from the real evidence report, and reply classification accuracy benchmarks.

---

## 1. Pytest Test Suite Results

The automated test suite runs with network access disabled, validating core logic against mock fixtures and `fake_ollama`:

- **Total Tests Collected**: **2,881**
- **Tests Passed**: **2,878**
- **Tests Skipped**: **3** (conditional platform checks)
- **Tests Failed**: **0**

### Test Coverage Highlights
- **Mathematical Integrity**: Unit tests verify the exact formula of the 0–100 lead score breakdown (rule weights + cached cosine similarity).
- **Email Validation**: Tests verify that drafted emails strictly conform to word count limits (<110 words), include the prospect's top buying signal, contain exactly one call to action, and introduce no hallucinated numbers.
- **Air-Gap Enforcement**: Unit tests patch `socket.socket.connect` to assert that workflows run cleanly against loopback endpoints while external connection attempts fail loudly.
- **Isolated Databases**: Tests use temporary databases and assert that no modifications or writes occur in `data/` during test suite execution.
- **Database Migrations**: State tests ensure backward compatibility across all SQLite table migrations.

---

## 2. Real Air-Gapped Evidence Report Summary

The real evidence report (`docs/evidence/offline_report_20261125_120715.md`) contains empirical results from running all 8 end-to-end sales workflows on real hardware against a live local Ollama server without cloud connectivity.

### Environment & Hardware Profile
- **Report Date**: 2026-11-25T06:37:15.332948+00:00
- **Total Duration**: 227.04 seconds
- **Hardware Profile**: `standard laptop` (8.0 GB RAM, Apple Silicon arm64, 8 CPU cores / 8 threads)
- **Local Model**: `qwen3:1.7b` (Alibaba Cloud, Apache-2.0, num_ctx: 2048)
- **Local Ollama Daemon**: v0.40.17 at `http://127.0.0.1:11434`
- **Network Enforcement**: Patched `socket.socket.connect` blocking non-loopback IPs; 1 canary external socket attempt to `8.8.8.8:53` blocked and verified.

### End-to-End Workflow Results (All 8 Workflows)

| # | Workflow | Status | Model Returned by Ollama | Latency (ms) | Output Summary |
|---|---|---|---|---|---|
| **1** | Business Profile | **PASS** | `qwen3:1.7b` | 34,087 ms | Extracted company profile & 2 products for Acme Technologies |
| **2** | ICP Generation | **PASS** | `qwen3:1.7b` | 35,485 ms | Synthesized ICP definitions for products `acme_flow` and `acme_pulse` |
| **3** | Stored Research | **PASS** | `qwen3:1.7b` | 30,121 ms | Researched 3 stored prospects (Apex, CloudMetrics, MedPulse) via stored text |
| **4** | Buying Signals | **PASS** | `qwen3:1.7b` | 5,942 ms | Classified signals across 3 prospects with verbatim quote enforcement |
| **5** | Problem Statements | **PASS** | `qwen3:1.7b` | 41,276 ms | Generated 6 problem statements (3 prospects × 2 products) |
| **6** | Validated Email Drafting | **PASS** | `qwen3:1.7b` | 38,839 ms | Generated 6 validated email drafts (3 prospects × 2 products, 1 retry handled) |
| **7** | Lead Scoring | **PASS** | `qwen3:1.7b` | 27,551 ms | Scored 6 prospect-product combinations (average score: 57) + rationales |
| **8** | Reply Classification | **PASS** | `qwen3:1.7b` | 13,263 ms | Classified 5 real reply threads into 8-label schema |

### Inference Log Audit Verification
- **Total Inferences in Log**: 60
- **Localhost / 127.0.0.1 Inferences**: 60 (100%)
- **External / Cloud Inferences**: 0 (0%)
- **Verification Query**:
  ```sql
  SELECT COUNT(*) FROM inference_log WHERE mode = 'real' AND task != 'usage' AND host NOT IN ('localhost', '127.0.0.1', '::1');
  -- Returned: 0
  ```
  *(Audit and Controls panels filter on `mode = 'real'` and `task != 'usage'` to count only genuine model inferences).*

---

## 3. Reply Classification Accuracy Benchmark (`sellbird eval-replies`)

The reply classification evaluation was conducted over `data/replies/sample_threads.json` containing 32 multi-turn fictional email threads (including 12 harder, indirect-wording threads added for robust boundary evaluation) covering all 8 supported labels:

| Classification Mode | Samples Evaluated | Correct Predictions | Accuracy |
|---|---|---|---|
| **Rule-Based Heuristic Fallback** (`classify_reply_heuristic`) | 32 | 32 | **100.0%** |
| **Local LLM Model** (`classify_reply` via `qwen3:1.7b`) | 32 | 16 | **50.0%** |

### Per-Label Confusion & Accuracy Summary (`qwen3:1.7b`)

| Label | Total | Heuristic Accuracy | LLM Accuracy | LLM Misclassifications |
|---|---|---|---|---|
| `interested` | 5 | 5/5 (100%) | 2/5 (40%) | `->objection`, `->not_interested`, `->not_interested` |
| `not_interested` | 4 | 4/4 (100%) | 3/4 (75%) | `->unsubscribe` |
| `not_now` | 4 | 4/4 (100%) | 1/4 (25%) | `->not_interested`, `->out_of_office`, `->not_interested` |
| `objection` | 5 | 5/5 (100%) | 3/5 (60%) | `->not_interested`, `->not_interested` |
| `out_of_office` | 3 | 3/3 (100%) | 2/3 (67%) | `->not_now` |
| `referral` | 4 | 4/4 (100%) | 0/4 (0%) | `->not_interested` (3x), `->out_of_office` |
| `unclear` | 3 | 3/3 (100%) | 1/3 (33%) | `->not_interested`, `->objection` |
| `unsubscribe` | 4 | 4/4 (100%) | 4/4 (100%) | *none* |

*Note*: As detailed in [KNOWN_LIMITATIONS.md](file:///Users/anandvardhanchauhan/projects/harvey/docs/KNOWN_LIMITATIONS.md), the 1.7B parameter model (`qwen3:1.7b`) struggles with nuanced indirect phrasing, often defaulting to `not_interested` on polite deferrals and referrals. The deterministic heuristic fallback achieves 100% accuracy on these threads and acts as an immediate reliable fallback when the model encounters ambiguity or when Ollama is unavailable.
