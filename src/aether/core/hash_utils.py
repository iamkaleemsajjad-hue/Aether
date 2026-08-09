"""
Content-addressed hashing utilities for Aether.

Aether uses SHA-256 hashes to uniquely identify AEG graphs, compiled kernels,
weight blobs, and manifest entries. This module provides stable, reusable
hashing helpers that are deterministic across platforms and Python versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import dataclasses
import enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether.core.aeg_ir import AEGIRModule
    from aether.core.graph import AEGGraph


def _canonicalize_for_hash(value: Any) -> Any:
    """Convert graph metadata into deterministic JSON-safe values.

    In-memory graphs legitimately carry weight arrays on nodes during
    compilation. Serializing those arrays directly makes packaging fail and
    would encourage callers to drop them before hashing. Preserve their
    identity instead using dtype, shape, and a content digest; the actual
    payload is hashed separately in the AEG weight manifest.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }
        }
    if np is not None and isinstance(value, np.generic):
        return _canonicalize_for_hash(value.item())
    if isinstance(value, enum.Enum):
        return _canonicalize_for_hash(value.value)
    if isinstance(value, dict):
        return {str(key): _canonicalize_for_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize_for_hash(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize_for_hash(dataclasses.asdict(value))
    # Compiler metadata may contain rich value objects (for example the
    # pruning mask produced by Pass 9).  Prefer their explicit wire format so
    # hashes are based on semantic content rather than object identity.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonicalize_for_hash(to_dict())
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest(), "length": len(value)}
    raise TypeError(
        f"Cannot canonicalize object of type {type(value).__name__}; "
        "provide a JSON-compatible value or a deterministic to_dict() method"
    )
    return value


def _ensure_bytes(data: object) -> bytes:
    """Convert a string or bytes object into bytes for hashing.

    Args:
        data: The input to convert. Strings are encoded as UTF-8.

    Returns:
        The bytes representation of the input.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    msg = f"Cannot hash object of type {type(data).__name__}; expected str or bytes"
    raise TypeError(msg)


def compute_content_hash(data: object) -> str:
    """Compute the SHA-256 hash of arbitrary content.

    The content is converted to bytes (UTF-8 for strings, directly for bytes,
    or canonical JSON for other objects). The result is a lowercase hex string
    prefixed with the algorithm name.

    Args:
        data: The content to hash. Strings, bytes, or JSON-serializable objects.

    Returns:
        A content-addressed hash string of the form "sha256:<hex>".
    """
    hasher = hashlib.sha256()
    if isinstance(data, (str, bytes)):
        hasher.update(_ensure_bytes(data))
    else:
        try:
            canonical = json.dumps(
                _canonicalize_for_hash(data),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            hasher.update(canonical.encode("utf-8"))
        except TypeError as exc:
            msg = "Content is not JSON-serializable; convert to string or bytes first"
            raise TypeError(msg) from exc
    return f"sha256:{hasher.hexdigest()}"


def compute_file_hash(path: str | Path) -> str:
    """Compute the SHA-256 hash of a file on disk.

    The file is read in chunks so large files can be hashed without loading
    them entirely into memory.

    Args:
        path: The path to the file to hash.

    Returns:
        A content-addressed hash string of the form "sha256:<hex>".

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def compute_graph_hash(graph: AEGGraph | AEGIRModule) -> str:
    """Compute a content-addressed hash of an AEG graph or AEG-IR module.

    The hash is deterministic and reflects the structure of the graph, not
    in-memory object identity. It is used as a cache key for compiled kernels
    and Hub lookups.

    Args:
        graph: An AEG computation graph or AEG-IR module.

    Returns:
        A content-addressed hash string of the form "sha256:<hex>".
    """
    if hasattr(graph, "to_dict"):
        data = graph.to_dict()
    elif hasattr(graph, "to_json"):
        data = json.loads(graph.to_json())
    else:
        data = graph.__dict__
    return compute_content_hash(data)


def compute_kernel_cache_key(
    graph_hash: str,
    target_id: str,
    aether_version: str,
    backend_name: str = "pytorch",
    precision_profile: str = "default",
) -> str:
    """Compute a content-addressed cache key for a compiled kernel set.

    The key combines the graph hash, hardware target, Aether version, backend,
    and precision profile. It is used both for local disk cache and Aether Hub
    lookups.

    Args:
        graph_hash: Content-addressed hash of the AEG graph or IR.
        target_id: Target hardware identifier (e.g., "cuda_sm90").
        aether_version: Aether Runtime version string.
        backend_name: Name of the backend plugin used to execute the kernel.
        precision_profile: Precision profile identifier.

    Returns:
        A stable, content-addressed cache key.
    """
    payload = {
        "graph_hash": graph_hash,
        "target_id": target_id,
        "aether_version": aether_version,
        "backend_name": backend_name,
        "precision_profile": precision_profile,
    }
    return compute_content_hash(payload)


def compute_manifest_hash(manifest: dict[str, Any]) -> str:
    """Compute a hash of an AEG manifest dictionary.

    The manifest hash is stored inside the manifest itself to detect tampering
    and to provide a top-level content identifier for the entire AEG package.

    Args:
        manifest: The AEG manifest dictionary.

    Returns:
        A content-addressed hash string of the form "sha256:<hex>".
    """
    # Exclude the existing hash fields so we don't hash the hash.
    stripped = {k: v for k, v in manifest.items() if k not in ("manifest_hash", "graph_hash")}
    return compute_content_hash(stripped)


def verify_content_hash(data: object, expected_hash: str) -> bool:
    """Verify that content matches an expected SHA-256 hash.

    Args:
        data: The content to hash and compare.
        expected_hash: Expected hash string, with or without "sha256:" prefix.

    Returns:
        True if the computed hash matches the expected hash, False otherwise.
    """
    actual = compute_content_hash(data)
    return _normalize_hash(actual) == _normalize_hash(expected_hash)


def verify_file_hash(path: str | Path, expected_hash: str) -> bool:
    """Verify that a file matches an expected SHA-256 hash.

    Args:
        path: The file path to verify.
        expected_hash: Expected hash string, with or without "sha256:" prefix.

    Returns:
        True if the file hash matches the expected hash, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    actual = compute_file_hash(path)
    return _normalize_hash(actual) == _normalize_hash(expected_hash)


def _normalize_hash(hash_value: str) -> str:
    """Normalize a hash string to the lowercase "sha256:<hex>" form.

    Args:
        hash_value: A hash string that may or may not include the prefix.

    Returns:
        Normalized hash string.
    """
    lower = hash_value.lower().strip()
    if lower.startswith("sha256:"):
        return lower
    return f"sha256:{lower}"


def compute_directory_hash(directory: str | Path) -> str:
    """Compute a hash of every file in a directory, sorted lexicographically.

    The hash is path-agnostic beyond the directory root: it hashes each file's
    relative path and content, then hashes the concatenated results. This makes
    the hash stable across moves of the directory itself.

    Args:
        directory: The directory to hash.

    Returns:
        A content-addressed hash string of the form "sha256:<hex>".
    """
    root = Path(directory)
    if not root.is_dir():
        msg = f"Not a directory: {root}"
        raise NotADirectoryError(msg)

    files = sorted(root.rglob("*"))
    hasher = hashlib.sha256()
    for file_path in files:
        if file_path.is_file():
            relative_path = file_path.relative_to(root).as_posix()
            file_hash = compute_file_hash(file_path)
            entry = f"{relative_path}:{file_hash}\n"
            hasher.update(entry.encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def compute_hash_prefix(hash_value: str, length: int = 12) -> str:
    """Return a short, collision-resistant prefix of a hash string.

    Used for human-readable cache directory names and filenames.

    Args:
        hash_value: A full hash string.
        length: Desired prefix length.

    Returns:
        The hash prefix, stripped of any "sha256:" prefix.
    """
    normalized = _normalize_hash(hash_value)
    hex_part = normalized.split(":", 1)[1]
    return hex_part[:length]


def compute_aeg_cache_key(
    model_id: str,
    aether_version: str,
    optimization_level: int = 2,
    quality_budget: float = 0.02,
    calibration_dataset: str = "wikitext-2",
) -> str:
    """Compute a cache key for a compiled AEG artifact.

    AEG artifacts are cached by the inputs that affect their contents, not by
    the model ID alone. This ensures that changing the compiler settings
    produces a new cache key.

    Args:
        model_id: The model identifier (e.g., HuggingFace repo ID).
        aether_version: Aether Runtime version string.
        optimization_level: Compiler optimization level.
        quality_budget: Maximum perplexity increase budget.
        calibration_dataset: Calibration dataset name.

    Returns:
        A content-addressed cache key for the AEG artifact.
    """
    payload = {
        "model_id": model_id,
        "aether_version": aether_version,
        "optimization_level": optimization_level,
        "quality_budget": quality_budget,
        "calibration_dataset": calibration_dataset,
    }
    return compute_content_hash(payload)


def combine_hashes(*hashes: str) -> str:
    """Combine multiple hash strings into a single deterministic hash.

    Args:
        hashes: Hash strings to combine.

    Returns:
        A new content-addressed hash string.
    """
    normalized = sorted(_normalize_hash(h) for h in hashes)
    return compute_content_hash("|".join(normalized))


def hash_file_stream(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash a file and return the digest.

    This is a low-level streaming variant useful for progress-bar integrations.

    Args:
        file_path: Path to the file.
        chunk_size: Number of bytes to read per iteration.

    Returns:
        A content-addressed hash string.
    """
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def hash_environment_state() -> str:
    """Compute a hash capturing relevant environment state for reproducibility.

    This includes Python version, Aether version, and platform information. It
    is used for debugging cache misses and reproducibility reports.

    Returns:
        A content-addressed hash string.
    """
    from aether.core.constants import AETHER_VERSION

    payload = {
        "python_version": os.sys.version,
        "aether_version": AETHER_VERSION,
        "platform": os.name,
    }
    return compute_content_hash(payload)


def is_hash_equal(hash_a: str, hash_b: str) -> bool:
    """Compare two hash strings, tolerating missing prefixes.

    Args:
        hash_a: First hash string.
        hash_b: Second hash string.

    Returns:
        True if the hashes are equal, False otherwise.
    """
    return _normalize_hash(hash_a) == _normalize_hash(hash_b)
