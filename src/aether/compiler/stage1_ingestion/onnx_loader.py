"""
ONNX model loader.

Loads ONNX models for ingestion into AEG-IR. The loader extracts the model
graph, inputs, outputs, and initializers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ONNXLoader:
    """Loads ONNX models and extracts their graph."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if onnx is None:
            msg = "onnx package is not installed"
            raise UnsupportedFormatError(msg)

    def load(self) -> dict[str, Any]:
        """Load the ONNX model and return graph information."""
        if not self.model_path.exists():
            msg = f"ONNX file not found: {self.model_path}"
            raise IngestionError(msg)
        try:
            model = onnx.load(self.model_path)
        except Exception as exc:
            msg = f"Failed to load ONNX model: {exc}"
            raise UnsupportedFormatError(msg) from exc
        graph = model.graph
        nodes = [{
            "op_type": n.op_type,
            "inputs": list(n.input),
            "outputs": list(n.output),
            "attrs": {a.name: a for a in n.attribute},
        } for n in graph.node]
        inputs = [{
            "name": i.name,
            "shape": [d.dim_value if d.HasField("dim_value") else None for d in i.type.tensor_type.shape.dim],
            "dtype": i.type.tensor_type.elem_type,
        } for i in graph.input]
        outputs = [{
            "name": o.name,
            "shape": [d.dim_value if d.HasField("dim_value") else None for d in o.type.tensor_type.shape.dim],
            "dtype": o.type.tensor_type.elem_type,
        } for o in graph.output]
        logger.info("Loaded ONNX", path=str(self.model_path), nodes=len(nodes))
        return {
            "model": model,
            "nodes": nodes,
            "inputs": inputs,
            "outputs": outputs,
            "initializers": {init.name: init for init in graph.initializer},
        }

    def __repr__(self) -> str:
        return f"ONNXLoader({self.model_path})"
