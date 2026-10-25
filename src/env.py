"""Environment detection, hardware inspection, and research metadata capture.

Ensures strict reproducibility by recording git commit, config hashes,
library versions, seed, device capabilities, and wall-clock time for all runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_git_commit(cwd: Optional[str | Path] = None) -> str:
    """Retrieve the current git commit hash (8 chars) or 'uncommitted' if dirty.

    Args:
        cwd: Directory within the git repository.

    Returns:
        String containing 8-char commit hash, or 'uncommitted' if changes exist.
    """
    try:
        git_dir = str(cwd) if cwd else os.path.dirname(os.path.abspath(__file__))
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=git_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=git_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if status:
            return "uncommitted"
        return commit[:8]
    except Exception:
        return "unversioned"


def compute_config_hash(config_path_or_dict: str | Path | Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a configuration file or dictionary.

    Args:
        config_path_or_dict: Path to yaml/json config or loaded dict.

    Returns:
        Hexadecimal SHA-256 string (first 12 characters).
    """
    if isinstance(config_path_or_dict, (str, Path)):
        p = Path(config_path_or_dict)
        if p.is_file():
            content = p.read_bytes()
        else:
            content = str(config_path_or_dict).encode("utf-8")
    else:
        import json
        content = json.dumps(config_path_or_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:12]


def detect_device() -> Dict[str, Any]:
    """Detect computation hardware (CUDA vs CPU) and extract device properties.

    Returns:
        Dictionary with device type, GPU model name, and VRAM memory.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            "device": "cuda",
            "device_name": gpu_name,
            "device_memory_gb": round(vram_gb, 2),
            "cuda_version": torch.version.cuda,
        }
    return {
        "device": "cpu",
        "device_name": platform.processor() or "CPU",
        "device_memory_gb": None,
        "cuda_version": None,
    }


def get_package_versions() -> Dict[str, str]:
    """Capture library versions for reproducibility tracking."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "not_installed"

    try:
        import peft
        versions["peft"] = peft.__version__
    except ImportError:
        versions["peft"] = "not_installed"

    try:
        import accelerate
        versions["accelerate"] = accelerate.__version__
    except ImportError:
        versions["accelerate"] = "not_installed"

    return versions


def build_run_metadata(
    config_name: str,
    config_hash: str,
    seed: int,
    model_name: str,
    num_train_samples: int = 0,
    num_eval_samples: int = 0,
    wall_clock_seconds: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a full research metadata record for auditing and results logging.

    Args:
        config_name: Identifier or file name of the configuration.
        config_hash: SHA-256 hash of the configuration.
        seed: Random seed used for reproducibility.
        model_name: Pretrained model name or path.
        num_train_samples: Number of training samples processed.
        num_eval_samples: Number of evaluation samples processed.
        wall_clock_seconds: Total execution duration in seconds.
        extra: Additional key-values to include.

    Returns:
        Comprehensive dictionary matching the research integrity standard.
    """
    dev_info = detect_device()
    pkg_info = get_package_versions()
    now_iso = datetime.now(timezone.utc).isoformat()
    git_hash = get_git_commit()

    record: Dict[str, Any] = {
        "timestamp": now_iso,
        "git_commit": git_hash,
        "config_name": config_name,
        "config_hash": config_hash,
        "seed": seed,
        "device": dev_info["device"],
        "device_name": dev_info["device_name"],
        "device_memory_gb": dev_info["device_memory_gb"],
        "torch_version": pkg_info["torch"],
        "transformers_version": pkg_info["transformers"],
        "peft_version": pkg_info["peft"],
        "model_name": model_name,
        "num_train_samples": num_train_samples,
        "num_eval_samples": num_eval_samples,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
    }

    if extra:
        record.update(extra)

    return record


class Timer:
    """Simple high-precision wall-clock timer."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time
