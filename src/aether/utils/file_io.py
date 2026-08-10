"""
File I/O utilities for Aether.

Provides helpers for resolving cache directories, reading/writing AEG artifacts,
and safely handling file paths across platforms.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from aether.core.constants import DEFAULT_CACHE_DIR


def aether_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve and ensure the Aether cache directory exists.

    Args:
        cache_dir: Custom cache directory, or None for default.

    Returns:
        Resolved cache directory path.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("AETHER_CACHE_DIR")
        if not cache_dir:
            # ``~/.aether`` is appropriate on POSIX, but Windows installations
            # can have a redirected/protected profile root.  Use the standard
            # per-user local application directory there so first-run model
            # downloads do not fail with an avoidable ACL error.
            if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
                cache_dir = str(Path(os.environ["LOCALAPPDATA"]) / "Aether")
            else:
                cache_dir = DEFAULT_CACHE_DIR
    elif os.name == "nt" and str(cache_dir).replace("\\", "/") in {"~/.aether", DEFAULT_CACHE_DIR} and os.environ.get("AETHER_CACHE_DIR") is None and os.environ.get("LOCALAPPDATA"):
        # CompilerConfig carries the POSIX default as a concrete dataclass
        # value, so normalize that default here as well.
        cache_dir = str(Path(os.environ["LOCALAPPDATA"]) / "Aether")
    root = Path(cache_dir).expanduser().resolve()
    subdirs = [
        "models",
        "kernels",
        "config",
        "logs",
        "hub",
    ]
    try:
        root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Sandboxed/locked-down hosts may deny both the profile and
        # LOCALAPPDATA locations.  Keep the cache usable in a per-user temp
        # directory rather than failing before the model path is even known.
        root = Path(tempfile.gettempdir()) / "aether-runtime"
        root.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def resolve_model_path(model_id: str, cache_dir: str | Path | None = None) -> Path | None:
    """Resolve a model to a local path in the Aether cache.

    Args:
        model_id: Model identifier (HuggingFace ID or local path).
        cache_dir: Custom cache directory.

    Returns:
        Path to the local model directory, or None if not found.
    """
    cache = aether_cache_dir(cache_dir)
    model_cache = cache / "models"
    safe_id = safe_model_id_path(model_id)
    if (model_cache / safe_id).exists():
        return model_cache / safe_id
    # Check if model is a local path
    local_path = Path(model_id)
    if local_path.exists():
        return local_path
    return None


def save_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Save a JSON file, ensuring the parent directory exists.

    Args:
        path: Path to write.
        data: JSON-serializable data.
        indent: JSON indentation level.

    Returns:
        Path to the written file.
    """
    file_path = Path(path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=indent, sort_keys=True, default=str), encoding="utf-8")
    return file_path


def load_json(path: str | Path) -> Any:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)
    return json.loads(file_path.read_text(encoding="utf-8"))


def safe_write(path: str | Path, content: str) -> Path:
    """Write text content to a file safely using atomic write.

    Args:
        path: Output path.
        content: Text content.

    Returns:
        Path to the written file.
    """
    file_path = Path(path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.rename(file_path)
    return file_path


def safe_read(path: str | Path) -> str:
    """Read text content from a file.

    Args:
        path: Path to the file.

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)
    return file_path.read_text(encoding="utf-8")


def safe_model_id_path(model_id: str) -> str:
    """Convert a model identifier into a safe filesystem directory name.

    Replaces path separators and other unsafe characters with underscores so the
    model ID can be used as a directory name under the Aether cache.
    """
    safe = model_id.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    safe = safe.strip(".")
    if not safe:
        safe = "unnamed_model"
    return safe


def glob_models(cache_dir: str | Path | None = None) -> list[Path]:
    """Find all AEG packages in the Aether cache.

    Args:
        cache_dir: Custom cache directory.

    Returns:
        List of paths to AEG package (manifest) directories.
    """
    cache = aether_cache_dir(cache_dir)
    model_cache = cache / "models"
    if not model_cache.exists():
        return []
    return sorted(model_cache.iterdir())


def delete_model(model_id: str, cache_dir: str | Path | None = None) -> None:
    """Delete a cached model from the Aether cache.

    Args:
        model_id: Model identifier.
        cache_dir: Custom cache directory.
    """
    cache = aether_cache_dir(cache_dir)
    model_path = (cache / "models" / safe_model_id_path(model_id)).resolve()
    models_root = (cache / "models").resolve()
    if not model_path.is_relative_to(models_root):
        raise ValueError("model identifier resolves outside the Aether cache")
    if model_path.exists():
        import shutil
        shutil.rmtree(model_path)


def format_size_gb(size_bytes: int) -> str:
    """Format a byte size as a human-readable GB string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "38.5 GB").
    """
    gb = size_bytes / (1024**3)
    return f"{gb:.1f} GB"


def format_tokens_per_second(tps: float) -> str:
    """Format throughput as a human-readable string.

    Args:
        tps: Tokens per second.

    Returns:
        Formatted string (e.g., "152.3 t/s").
    """
    return f"{tps:.1f} t/s"


def format_ms(ms: float) -> str:
    """Format milliseconds as a human-readable string.

    Args:
        ms: Milliseconds.

    Returns:
        Formatted string (e.g., "234.5 ms").
    """
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.1f} ms"
