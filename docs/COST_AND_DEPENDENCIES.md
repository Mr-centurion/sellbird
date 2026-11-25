# Zero-Cost Statement & Dependency Audit

This document verifies the zero-cost architecture of Sellbird, linking directly to the local inference audit trail, compliance verification scripts, and empirical test reports.

---

## 1. Zero-Cost Statement

Sellbird incurs **$0.00 in cloud model API fees**, subscription charges, or per-token inference costs:

- **Local Inference**: All language generation, extraction, and classification run on local hardware through Ollama (`qwen3:1.7b` and `qwen3:4b`).
- **Local Embeddings**: Semantic retrieval and cosine similarity scoring use the open-source `all-MiniLM-L6-v2` model running locally in PyTorch/Sentence-Transformers.
- **Zero Third-Party Prospecting Services**: Sellbird requires no subscriptions to commercial prospecting databases (e.g., Apollo, ZoomInfo, Clearbit, or Clay). Business discovery is supported via open data (OpenStreetMap) or stored JSON imports.
- **Zero Cloud SDKs**: The repository contains zero dependencies on commercial cloud LLM SDKs (no `anthropic`, `openai`, `boto3`, `cohere`, `azure-ai`, or `google-generativeai`).

---

## 2. SQLite Inference Log Audit

Every LLM inference invocation is audited by `sellbird/brain.py` and recorded directly in the SQLite `inference_log` table.

During the verified end-to-end air-gapped test run on November 25, 2026:
- **Total Inferences Executed**: 60
- **Inferences Executed on Localhost (`127.0.0.1` / `localhost`)**: 60 (100.0%)
- **Inferences Executed on Cloud / External Hosts**: 0 (0.0%)
- **Blocked External Socket Connection Attempts**: 1 (automated canary test to `8.8.8.8:53` verified the socket guard was active)

```sql
-- Audit verification query executed against data/offline_test.db:
SELECT COUNT(*) FROM inference_log WHERE mode = 'real' AND task != 'usage' AND host NOT IN ('localhost', '127.0.0.1', '::1');
-- Result: 0
```

> **Note on Inference Counting**: All reports, audits, and dashboard controls count only genuine inference rows (`mode = 'real'` and `task != 'usage'`), strictly excluding bookkeeping events and test mock executions.

The detailed 60-row audit log, including timestamps, agent names, tasks, latencies, and host addresses, is documented in the full report:  
👉 [docs/evidence/offline_report_20261125_120715.md](file:///Users/anandvardhanchauhan/projects/harvey/docs/evidence/offline_report_20261125_120715.md)

---

## 3. Automated Compliance Script (`scripts/no_cloud_check.sh`)

To prevent accidental introduction of cloud APIs, keys, or SDKs, the project maintains an automated compliance scanner: `scripts/no_cloud_check.sh`.

### Verification Scope
The script executes `git grep` across the codebase asserting:
1. **Zero Cloud SDK Imports**: Checks for `import anthropic`, `import openai`, `import boto3`, `import cohere`, `import azure.ai`, `import google.generativeai`.
2. **Zero Cloud API Keys**: Scans for patterns like `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `sk-[a-zA-Z0-9]{20,}`.
3. **Zero Banned Cloud Provider References**: Scans for mentions of proprietary cloud platforms (`anthropic`, `openai`, `claude`, `codex`, `bedrock`, `chatgpt`), excluding historical database migration SQL columns preserved for backward compatibility.

### Verification Output
```bash
$ ./scripts/no_cloud_check.sh
=== Running No-Cloud & Air-Gap Compliance Check ===
PASS: Zero cloud SDK imports, API keys, or banned provider references found.
```

---

## 4. Evidence Report Link

The complete hardware configuration, model versions, socket canary test results, and per-workflow latencies are preserved in the non-fake evidence report:
- **Report Path**: [docs/evidence/offline_report_20261125_120715.md](file:///Users/anandvardhanchauhan/projects/harvey/docs/evidence/offline_report_20261125_120715.md)
- **Timestamp**: 2026-11-25T06:37:15.332948+00:00
- **Duration**: 227.04 seconds
- **Status**: 8/8 Workflows PASS
