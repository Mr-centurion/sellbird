<p align="center">
  <a href="https://sellbird.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/brand/sellbird-wordmark-white.svg">
      <img src="docs/brand/sellbird-wordmark.svg" alt="Sellbird" width="300">
    </picture>
  </a>
</p>

<p align="center"><strong>The open-source outreach platform that runs on local Ollama models.</strong></p>

<p align="center">
Sellbird finds companies that need what you sell, learns why they would buy, drafts personalized cold emails, scores leads, and classifies prospect replies.
It runs 100% on your own workstation using local open-weight models, with zero cloud API calls, zero token bills, and complete data privacy.
</p>

<p align="center">
  <a href="https://x.com/sellbirdagent"><img alt="Follow @sellbirdagent on X" src="https://img.shields.io/badge/follow-%40sellbirdagent-000000?logo=x&logoColor=white"></a>
  <a href="https://github.com/ethanplusai/sellbird/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/ethanplusai/sellbird?style=flat&logo=github&label=stars&color=244b62&cacheSeconds=3600"></a>
  <a href="https://github.com/ethanplusai/sellbird/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/ethanplusai/sellbird/actions/workflows/tests.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-244b62"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-244b62?logo=python&logoColor=white">
  <a href="https://sellbird.ai"><img alt="Hosted at sellbird.ai" src="https://img.shields.io/badge/hosted-sellbird.ai-244b62"></a>
  <a href="https://sellbird.ai/harvey/"><img alt="Formerly Harvey" src="https://img.shields.io/badge/formerly-Harvey-8aa0ad"></a>
</p>

> **[Harvey is now Sellbird.](https://sellbird.ai/harvey/)** Same project, same history, new name. Originally created as Harvey by Ethan Rogers and contributors under the MIT license, Sellbird 0.4.0+ is licensed under the AGPL-3.0-only.

---

## What It Is

Sellbird is a self-hosted, local-first sales automation pipeline. It performs business research, extracts buying signals from stored company data, synthesizes ideal customer profiles (ICPs), generates tailored problem statements, drafts validated outbound emails, and classifies prospect replies into actionable categories.

- **Zero Cloud API Fees**: Thinks strictly through local Ollama models (default `qwen3:1.7b` or `qwen3:4b`). No Anthropic, OpenAI, or third-party cloud LLM billing.
- **Local Semantic Retrieval**: Uses local sentence transformers (`all-MiniLM-L6-v2`) cached in SQLite for instant cosine-similarity prospect scoring and stored research retrieval.
- **Air-Gapped Operation**: All 8 pipeline workflows can run completely isolated from external networks against a local loopback Ollama daemon.
- **Safety First (Drafts Only)**: Outbound emails are validated in code and saved as drafts in your Outbox for human review. Nothing is transmitted without your explicit approval.

---

## Setup Steps from Fresh Clone (macOS)

Follow these exact steps to set up and run Sellbird on macOS (Apple Silicon or Intel):

### 1. Prerequisites
Ensure you have Python 3.11 or newer (Python 3.13 recommended) and Homebrew installed:
```bash
brew install python@3.13 ollama
```

### 2. Clone and Create Virtual Environment
```bash
git clone https://github.com/ethanplusai/sellbird.git
cd sellbird

# Create and activate virtual environment using Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install Editable Package & Dependencies
```bash
pip install --upgrade pip
pip install -e .
```

### 4. Start Ollama and Pull Local Models
Start the local Ollama background server (in another terminal or as a service):
```bash
ollama serve
```

In your main terminal, pull the required local open-weight models:
```bash
# Pull standard laptop model (1.7B)
ollama pull qwen3:1.7b

# Pull mid-tier model (4B)
ollama pull qwen3:4b
```

### 5. Download Local Embedding Model (One-Time)
Run a one-time command to download and cache the Apache-2.0 `all-MiniLM-L6-v2` embedding weights:
```bash
python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2').encode(['test']); print('cached OK')"
```

### 6. Launch the Local Dashboard
```bash
sellbird dashboard
```
The command starts the local FastAPI server on `http://127.0.0.1:5555` and prints your private one-click token link:
```
Sellbird dashboard running at http://127.0.0.1:5555/?token=sb_...
```
Open the printed link in your browser to inspect prospects, ICPs, lead scores, drafts, and conversations.

---

## Verification & Compliance Testing

Sellbird includes built-in scripts to verify air-gap compliance and end-to-end local inference:

### 1. Run the Air-Gapped Offline Test
Executes all 8 sales workflows end-to-end under an in-process socket guard blocking all external network traffic:
```bash
# Against your local Ollama server (writes report to docs/evidence/)
python scripts/offline_test.py

# Or against the fast in-memory mock server (used in pytest)
python scripts/offline_test.py --use-fake
```

### 2. Run the No-Cloud Compliance Scanner
Greps the repository to ensure zero cloud SDK imports (`anthropic`, `openai`, `boto3`, etc.), zero API key patterns, and zero banned cloud provider names:
```bash
./scripts/no_cloud_check.sh
```

### 3. Evaluate Reply Classification Accuracy
Tests the 8-label reply classifier against the 20 sample threads:
```bash
sellbird eval-replies
```

### 4. Run the Full Test Suite
```bash
pytest -q
```

---

## Documentation

Comprehensive project documentation is organized below:

| Document | Description |
|---|---|
| [System Architecture](docs/ARCHITECTURE.md) | Mermaid data-flow diagram (`UI -> FastAPI -> workflows -> Brain -> Ollama`), SQLite tables, task queue, and web research separation |
| [Models and Licenses](docs/MODELS_AND_LICENSES.md) | Verified license and version matrix for models (`qwen3:1.7b`, `qwen3:4b`, `all-MiniLM-L6-v2`) and dependencies |
| [Hardware Detection & Benchmarks](docs/HARDWARE.md) | Auto-detection logic, RAM sizing tiers, hardware profiles, and empirical latencies from real 8 GB Apple Silicon benchmarks |
| [Offline & Failure Handling](docs/OFFLINE_AND_FAILURE.md) | Offline vs online capabilities, Ollama downtime handling, dashboard alert banner, task queue, and deterministic fallbacks |
| [Zero-Cost Statement & Dependencies](docs/COST_AND_DEPENDENCIES.md) | $0.00 model cost verification, SQLite inference log audit (60/60 local calls), and compliance script details |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | Air-gap testing boundary (socket guard vs physical off), synthetic sample data files, small-model constraints, and legacy schema columns |
| [Testing Evidence](docs/TESTING_EVIDENCE.md) | Complete unit test results (2,873 passed), real air-gapped test summary, and reply classification accuracy |
| [How Sellbird Works](docs/how-it-works.md) | Detailed operational heartbeat, pipeline lifecycle, and background agents |
| [Signals & Pipeline](docs/signals-and-pipeline.md) | The 29 buying signals, discovery, enrichment, and verification stages |
| [Dashboard Guide](docs/dashboard.md) | Overview of each UI screen, prospect scoring breakdown, and Outbox management |
| [Configuration](docs/configuration.md) | Settings in `sellbird.yaml`, `sellbird.local.yaml`, and environment variables |

---

## Attribution & License

Sellbird is open-source software licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)**.

### Historical Attribution
Sellbird was originally created and released as **Harvey** by Ethan Rogers and contributors under the MIT License. Releases up to and including version `0.3.x` remain available under the MIT License, and the original copyright notice is preserved in the **[NOTICE](NOTICE)** file.

Copyright (C) 2026 Ethan Rogers and the Sellbird contributors.
