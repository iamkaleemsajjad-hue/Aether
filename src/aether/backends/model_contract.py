"""Classify execution inputs without relying on model names or families."""

from __future__ import annotations

from pathlib import Path


def model_format(model_id: str, aeg_path: str | None = None) -> str:
    """Return the storage/API contract that a loader must actually consume."""
    if aeg_path is not None:
        return "aeg"
    path = Path(model_id)
    suffix = path.suffix.lower()
    if suffix in {".aeg", ".aegpkg"} or (path.is_dir() and (path / "manifest.json").is_file()):
        return "aeg"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".gguf":
        return "gguf"
    return "source"
