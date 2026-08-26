"""Capture the hardware and software environment a result set was produced on.

Everything here is read-only observation.  The report embeds it verbatim so a
reader can judge whether two result sets are comparable, and so a third party
can reconstruct the environment.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(command: list[str]) -> str | None:
    """Run a short informational command, returning None if unavailable."""
    if shutil.which(command[0]) is None:
        return None
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - absence is the answer
        return None


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "torch_num_threads": None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    # /proc/cpuinfo carries the marketing name on Linux, which platform.processor
    # does not expose there.
    model = _run(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])
    if model:
        info["model_name"] = model
    try:
        import torch

        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return info


def memory_info() -> dict[str, Any]:
    try:
        import psutil

        virtual = psutil.virtual_memory()
        return {
            "total_bytes": int(virtual.total),
            "available_bytes": int(virtual.available),
            "used_percent": float(virtual.percent),
        }
    except ImportError:
        return {"total_bytes": None, "available_bytes": None, "used_percent": None}


def gpu_info() -> dict[str, Any]:
    """Enumerate accelerators through both PyTorch and NVML.

    PyTorch reports what the process can use; NVML reports the physical device,
    including driver version, power cap and temperature.  Both are recorded
    because a mismatch between them is itself diagnostic.
    """
    result: dict[str, Any] = {"available": False, "count": 0, "devices": [], "nvml": None}
    try:
        import torch
    except ImportError:
        return result
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
    result["hip_version"] = getattr(torch.version, "hip", None)
    if not torch.cuda.is_available():
        return result
    result["available"] = True
    result["count"] = torch.cuda.device_count()
    for index in range(result["count"]):
        properties = torch.cuda.get_device_properties(index)
        result["devices"].append({
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": properties.multi_processor_count,
            "compute_capability": f"{properties.major}.{properties.minor}",
        })
    result["visible_devices_env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    smi = _run([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.limit,clocks.max.sm",
        "--format=csv,noheader",
    ])
    if smi:
        result["nvml"] = [line.strip() for line in smi.splitlines()]
    result["sdpa_backends"] = {
        "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    result["cudnn_version"] = torch.backends.cudnn.version()
    result["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    return result


def software_info() -> dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": platform.system(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "numpy": _package_version("numpy"),
        "safetensors": _package_version("safetensors"),
        "huggingface_hub": _package_version("huggingface_hub"),
        "accelerate": _package_version("accelerate"),
        "aether_runtime": _package_version("aether-runtime"),
    }
    try:
        from aether.core.constants import AETHER_VERSION

        info["aether_version_constant"] = AETHER_VERSION
    except Exception:  # noqa: BLE001
        info["aether_version_constant"] = None
    info["aether_git_commit"] = _run(["git", "rev-parse", "HEAD"])
    info["aether_git_dirty"] = bool(_run(["git", "status", "--porcelain"]))
    return info


def resolve_revision(model_id: str) -> str | None:
    """Resolve a model repository to an immutable commit sha, if reachable."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:  # noqa: BLE001 - offline or rate-limited is not fatal
        return None


def collect(models: list[str] | None = None) -> dict[str, Any]:
    """Return the full environment record embedded in every result file."""
    record: dict[str, Any] = {
        "cpu": cpu_info(),
        "ram": memory_info(),
        "gpu": gpu_info(),
        "software": software_info(),
        "env": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES", "AETHER_TORCH_DTYPE",
                "AETHER_EXECUTION_DEVICES", "AETHER_FORCE_TENSOR_PARALLEL",
                "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "KAGGLE_KERNEL_RUN_TYPE",
            )
        },
    }
    if models:
        record["model_revisions"] = {model: resolve_revision(model) for model in models}
    return record
