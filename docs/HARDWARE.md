# Hardware Detection, Profiles & Benchmark Latencies

This document details Sellbird's automated hardware detection subsystem (`sellbird/hardware.py`), the sizing tiers, and empirical latency benchmarks measured during the real air-gapped test on an 8 GB Apple Silicon Mac.

---

## 1. Hardware Detection Subsystem

Sellbird dynamically inspects the host system on startup using `psutil` and platform utilities without requiring cloud checks:

- **System Memory**: Reads total physical RAM in GB (`psutil.virtual_memory().total`).
- **CPU Capabilities**: Determines physical CPU cores (`psutil.cpu_count(logical=False)`) and logical thread count (`psutil.cpu_count(logical=True)`).
- **GPU Accelerator Detection**:
  - **Apple Silicon (macOS Darwin)**: Detected via `platform.system() == "Darwin"` and machine architecture `arm64`/`aarch64`. Apple Silicon's Unified Memory Architecture (UMA) is detected as GPU VRAM, allowing the LLM weights and context cache to reside directly in shared memory.
  - **NVIDIA Hardware**: Detected via non-blocking execution of `nvidia-smi` (`--query-gpu=name,memory.total`).
  - **CPU-only fallback**: Used when no supported accelerator is found.

---

## 2. Hardware Profiles & Recommendation Sizing Tiers

Sellbird organizes compute recommendations into two primary named profiles across 4 RAM tiers:

| RAM Tier | Hardware Profile | Recommended Model | Context Window (`num_ctx`) | Target Workstation |
|---|---|---|---|---|
| **≤ 8.0 GB** | `standard laptop` | `qwen3:1.7b` | 2,048 tokens | Base MacBook Air, entry laptops |
| **8.0 – 16.0 GB** | `standard laptop` | `qwen3:4b` | 4,096 tokens | Mid-tier laptops, Mac Mini (16GB) |
| **16.0 – 32.0 GB** | `high performance` | `qwen3:8b` | 8,192 tokens | High-end laptops, developer desktops |
| **> 32.0 GB** | `high performance` | `qwen3:14b` | 8,192 tokens | Workstations, Mac Studio, GPU rigs |

### Configuration Overrides
The auto-detected profile and model can be explicitly overridden in `sellbird.local.yaml`:
```yaml
brain:
  hardware_profile: "standard laptop"
  model: "qwen3:1.7b"
  num_ctx: 2048
```

---

## 3. Measured Latencies from Real Air-Gapped Evidence Run

The benchmark below reflects real, verified measurements from `docs/evidence/offline_report_20261125_131553.md`.

### Host Environment Under Test
- **Operating System**: macOS Darwin (Apple Silicon arm64)
- **RAM**: 8.0 GB Unified Memory
- **CPU**: 8 cores / 8 threads
- **GPU Accelerator**: Apple Silicon (Unified Memory: 8.0 GB)
- **Model Evaluated**: `qwen3:1.7b` (`num_ctx`: 2048)
- **Ollama Version**: 0.40.17
- **Total Test Suite Duration**: 108.12 seconds

### Per-Workflow Measured Latency Table

| # | Workflow Name | Status | Model | Measured Latency | Execution Details & Workload |
|---|---|---|---|---|---|
| **1** | Business Profile Extraction | **PASS** | `qwen3:1.7b` | **17,957 ms** | Synthesized company profile & extracted 2 products for Acme Technologies |
| **2** | ICP Generation | **PASS** | `qwen3:1.7b` | **17,306 ms** | Generated ICP definitions for 2 distinct products (`acme_flow`, `acme_pulse`) |
| **3** | Stored Research | **PASS** | `qwen3:1.7b` | **21,981 ms** | Researched 3 stored prospects (Apex Logistics, CloudMetrics AI, MedPulse Health) |
| **4** | Buying Signals | **PASS** | `qwen3:1.7b` | **3,422 ms** | Classified 3 signals across 3 prospects with verbatim quote enforcement |
| **5** | Problem Statements | **PASS** | `qwen3:1.7b` | **16,966 ms** | Generated 6 statements (3 prospects × 2 products) |
| **6** | Validated Email Drafting | **PASS** | `qwen3:1.7b` | **12,610 ms** | Drafted 6 validated emails (3 prospects × 2 products) |
| **7** | Lead Scoring & Breakdown | **PASS** | `qwen3:1.7b` | **10,503 ms** | Scored 6 combinations via cached cosine similarity + LLM rationale (avg score 47) |
| **8** | Reply Classification | **PASS** | `qwen3:1.7b` | **6,875 ms** | Classified 5 real reply threads into 8-label schema |

### Per-Task Average Latencies from Evidence Report

| Task | Total Calls | Average Latency (ms) | Description | Source Database |
|---|---|---|---|---|
| `buying_signals` | 6 | **918 ms** | Signal classification with verbatim evidence quote check | `data/offline_test.db` |
| `classify_reply` | 5 | **1,359 ms** | Multi-turn email reply classification (offline test run, 5 threads) | `data/offline_test.db` |
| `score_rationale` | 6 | **1,567 ms** | Numeric lead score rationales, top 3 reasons, and main risk | `data/offline_test.db` |
| `email_draft` | 6 | **2,078 ms** | Cold email drafting with strict in-code rule validation | `data/offline_test.db` |
| `problem_statement` | 6 | **2,440 ms** | Prospect-product problem statement synthesis | `data/offline_test.db` |
| `stored_research` | 3 | **4,331 ms** | Prospect stored website dossier synthesis | `data/offline_test.db` |

> **Run Distinction**: The 1,359 ms average for `classify_reply` above is from Workflow 8 of the air-gapped run recorded in `data/offline_test.db` (5 threads). In the separate 32-thread `sellbird eval-replies` benchmark recorded in `data/sellbird.db`, `classify_reply` averaged **1,354 ms** over 32 inferences.

### Key Observations
1. **Interactive vs Batch Execution**: Fast classification tasks (Buying Signals, Reply Classification) average 0.9s to 1.4s per item on 8 GB hardware.
2. **Generative Multi-item Pipelines**: Drafting and problem synthesis across multi-product matrices take ~12 to 17 seconds for 6 full generations, demonstrating the utility of background task queues during bulk runs.
