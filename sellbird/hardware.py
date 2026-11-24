"""Hardware detection and model profile recommendation using psutil.

Profiles:
- 'standard laptop': qwen3:1.7b or 4b, num_ctx 2048-4096
- 'high performance': qwen3:8b or larger, num_ctx 8192

Recommendation tiers:
- <= 8 GB RAM:  qwen3:1.7b, num_ctx 2048, standard laptop
- 8-16 GB RAM:  qwen3:4b,   num_ctx 4096, standard laptop
- 16-32 GB RAM: qwen3:8b,   num_ctx 8192, high performance
- > 32 GB RAM:  qwen3:14b,  num_ctx 8192, high performance
"""

import os
import platform
import shutil
import subprocess
from typing import Any, Optional

import psutil


def detect_gpu() -> dict[str, Any]:
    """Detect available GPU accelerator.
    
    Apple Silicon counts as a GPU with unified memory.
    NVIDIA GPUs are detected via nvidia-smi if present.
    """
    sys_name = platform.system()
    machine = platform.machine().lower()

    # Apple Silicon
    if sys_name == "Darwin" and machine in ("arm64", "aarch64"):
        total_mem = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        return {
            "has_gpu": True,
            "gpu_type": "apple_silicon",
            "gpu_name": f"Apple Silicon (Unified Memory: {total_mem} GB)",
            "vram_gb": total_mem,
        }

    # NVIDIA GPU via nvidia-smi
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                first = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                name = parts[0]
                vram = round(float(parts[1]) / 1024.0, 1) if len(parts) > 1 else None
                return {
                    "has_gpu": True,
                    "gpu_type": "nvidia",
                    "gpu_name": name,
                    "vram_gb": vram,
                }
        except Exception:
            pass

    return {
        "has_gpu": False,
        "gpu_type": "none",
        "gpu_name": "None (CPU only)",
        "vram_gb": 0.0,
    }


def recommend_profile(ram_gb: float) -> dict[str, Any]:
    """Determine named profile, recommended model, and context window from total RAM."""
    if ram_gb <= 8.0:
        return {
            "profile": "standard laptop",
            "model": "qwen3:1.7b",
            "num_ctx": 2048,
        }
    elif ram_gb <= 16.0:
        return {
            "profile": "standard laptop",
            "model": "qwen3:4b",
            "num_ctx": 4096,
        }
    elif ram_gb <= 32.0:
        return {
            "profile": "high performance",
            "model": "qwen3:8b",
            "num_ctx": 8192,
        }
    else:
        return {
            "profile": "high performance",
            "model": "qwen3:14b",
            "num_ctx": 8192,
        }


def get_hardware_info(config: Optional[Any] = None) -> dict[str, Any]:
    """Detect complete system hardware metrics and recommended Ollama settings.
    
    Supports config overrides in config.brain (e.g. model, hardware_profile, num_ctx).
    """
    mem = psutil.virtual_memory()
    total_ram_gb = round(mem.total / (1024 ** 3), 1)
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or cpu_cores

    gpu = detect_gpu()
    rec = recommend_profile(total_ram_gb)

    # Config overrides
    brain_cfg = getattr(config, "brain", None) if config else None
    override_model = getattr(brain_cfg, "model", None)
    override_profile = getattr(brain_cfg, "hardware_profile", None)
    override_num_ctx = getattr(brain_cfg, "num_ctx", None)

    effective_profile = override_profile or rec["profile"]
    effective_model = override_model or rec["model"]
    effective_num_ctx = override_num_ctx or rec["num_ctx"]

    has_override = bool(
        override_profile or (override_model and override_model != rec["model"])
    )

    return {
        "ram_gb": total_ram_gb,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "has_gpu": gpu["has_gpu"],
        "gpu_type": gpu["gpu_type"],
        "gpu_name": gpu["gpu_name"],
        "vram_gb": gpu["vram_gb"],
        "recommended_profile": rec["profile"],
        "recommended_model": rec["model"],
        "recommended_num_ctx": rec["num_ctx"],
        "effective_profile": effective_profile,
        "effective_model": effective_model,
        "effective_num_ctx": effective_num_ctx,
        "is_override": has_override,
    }
