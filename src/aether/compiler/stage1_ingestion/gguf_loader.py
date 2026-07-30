"""
GGUF model loader.

Loads GGUF files produced by llama.cpp. Aether can ingest GGUF models to
compile them into AEG artifacts, preserving quantization metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import gguf
except ImportError:
    gguf = None  # type: ignore

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class GGUFLoader:
    """Loads GGUF model files and metadata."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if gguf is None:
            msg = "gguf package is not installed"
            raise UnsupportedFormatError(msg)

    def load(self) -> dict[str, Any]:
        """Load GGUF tensors and metadata.

        Returns a dictionary with keys 'tensors', 'metadata', 'arch'.
        """
        if not self.model_path.exists():
            msg = f"GGUF file not found: {self.model_path}"
            raise IngestionError(msg)
        reader = gguf.GGUFReader(self.model_path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        metadata = {k: reader.get_field(k) for k in reader.fields}
        arch = reader.architecture
        logger.info("Loaded GGUF", path=str(self.model_path), tensors=len(tensors), arch=arch)
        return {"tensors": tensors, "metadata": metadata, "arch": arch}

    def __repr__(self) -> str:
        return f"GGUFLoader({self.model_path})"
