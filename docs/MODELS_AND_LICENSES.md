# Models and Licenses

This document catalogs every AI model, runtime component, and primary software dependency utilized by Sellbird, including version numbers, verified open-source licenses, and upstream sources.

All licenses have been verified directly against installed package metadata (`importlib.metadata`), distribution classifiers, and upstream model repository cards.

---

## 1. Machine Learning & Embedding Models

Sellbird operates entirely with local models. No proprietary cloud weights or gated APIs are contacted.

| Model | Purpose | Version / Tag | Parameters | License | Source / Verification |
|---|---|---|---|---|---|
| **Qwen 2.5 / Qwen 3 (1.7B)** | Default local LLM for standard laptops (inference, extraction, drafting, classification) | `qwen3:1.7b` | 1.7 Billion | Apache-2.0 | Verified via `ollama show qwen3:1.7b --license` & Alibaba Cloud Qwen model card |
| **Qwen 2.5 / Qwen 3 (4B)** | High-tier local LLM for mid-spec hardware (8–16 GB RAM) | `qwen3:4b` | 4.0 Billion | Apache-2.0 | Verified via Qwen open-source repository & Ollama library |
| **all-MiniLM-L6-v2** | Local sentence embeddings for stored research retrieval and prospect ICP cosine scoring | `v2` (Sentence-Transformers) | ~22.7 Million | Apache-2.0 | Verified via Hugging Face model repository & Sentence-Transformers project |

---

## 2. Core Application & Runtime Dependencies

Installed environment metadata (`python 3.13` on macOS / Darwin):

| Package | Installed Version | Verified License | Classification / Upstream Source |
|---|---|---|---|
| **sellbird** | `0.4.0` | **AGPL-3.0-only** | Verified via `pyproject.toml` and root `LICENSE` file. Former releases (≤0.3.x) MIT under `NOTICE`. |
| **fastapi** | `0.165.0` | **MIT** | Verified via package metadata: `https://github.com/fastapi/fastapi` |
| **uvicorn** | `0.54.0` | **BSD-3-Clause** | Verified via package metadata: `https://github.com/encode/uvicorn` |
| **pydantic** | `2.14.4` | **MIT** | Verified via package metadata: `https://github.com/pydantic/pydantic` |
| **httpx** | `0.28.1` | **BSD-3-Clause** | Verified via package metadata: `https://github.com/encode/httpx` |
| **aiosqlite** | `0.22.1` | **MIT** | Verified via classifier: `License :: OSI Approved :: MIT License` |
| **psutil** | `7.2.2` | **BSD-3-Clause** | Verified via package metadata: `https://github.com/giampaolo/psutil` |
| **sentence-transformers** | `6.3.0` | **Apache-2.0** | Verified via package metadata: `https://www.SBERT.net` |
| **torch** | `2.15.0` | **Apache-2.0 / BSD-3 / MIT** | Multi-licensed core engine; verified via package metadata (`https://pytorch.org`) |
| **transformers** | `5.24.0` | **Apache-2.0** | Verified via package metadata: `https://github.com/huggingface/transformers` |
| **numpy** | `2.5.5` | **BSD-3-Clause** | Verified via package metadata: `https://numpy.org` |
| **jinja2** | `3.1.6` | **BSD-3-Clause** | Verified via classifier: `License :: OSI Approved :: BSD License` |
| **pyyaml** | `6.0.4` | **MIT** | Verified via package metadata: `https://pyyaml.org/` |
| **rich** | `15.0.0` | **MIT** | Verified via package metadata: `https://rich.readthedocs.io/` |
| **pytest** | `8.4.2` | **MIT** | Verified via package metadata: `https://docs.pytest.org/` |

---

## 3. Local Runtime Server

| Component | Minimum Version | Tested Version | License | Host Binding |
|---|---|---|---|---|
| **Ollama Daemon** | `0.1.30+` | `0.40.17` | MIT | `http://127.0.0.1:11434` (Strict loopback) |

---

## 4. License Compliance Summary

1. **Permissive Inferences & Embeddings**: Both model weights (`qwen3:1.7b`, `qwen3:4b`, `all-MiniLM-L6-v2`) are licensed under Apache-2.0, permitting private, internal, commercial, and offline execution without proprietary cloud entanglements.
2. **Copyleft Protection**: Sellbird itself is licensed under the GNU Affero General Public License v3 (AGPL-3.0-only).
3. **No Unverified Packages**: Every installed package and model has been verified via `importlib.metadata` and local package inspection. None are marked unverified.
