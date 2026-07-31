"""
ONNX model loader with full ONNX→AEGGraph lowering.

Loads ONNX protobuf models and lowers the operator graph into AEG-IR nodes.
Supports extracting weight initializers as float32 numpy arrays for downstream
quantization and sensitivity passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

# ONNX element type → numpy dtype
_ONNX_ELEM_TYPE: dict[int, str] = {
    1:  "float32",   # FLOAT
    2:  "uint8",     # UINT8
    3:  "int8",      # INT8
    4:  "uint16",    # UINT16
    5:  "int16",     # INT16
    6:  "int32",     # INT32
    7:  "int64",     # INT64
    8:  "object",    # STRING
    9:  "bool",      # BOOL
    10: "float16",   # FLOAT16
    11: "float64",   # DOUBLE
    12: "uint32",    # UINT32
    13: "uint64",    # UINT64
    16: "bfloat16",  # BFLOAT16
}

# ONNX op_type → AEG op_type mapping (best-effort)
_ONNX_OP_MAP: dict[str, str] = {
    "MatMul":           "linear",
    "Gemm":             "linear",
    "Conv":             "conv",
    "Relu":             "relu",
    "Gelu":             "gelu",
    "Sigmoid":          "sigmoid",
    "Tanh":             "tanh",
    "Softmax":          "softmax",
    "LayerNormalization": "rmsnorm",
    "RmsNormalization": "rmsnorm",
    "Reshape":          "reshape",
    "Transpose":        "transpose",
    "Concat":           "concat",
    "Add":              "add",
    "Mul":              "mul",
    "Gather":           "embedding",
    "EmbedLayerNormalization": "embedding",
    "Attention":        "qkv_proj",
    "MultiHeadAttention": "qkv_proj",
    "RotaryEmbedding":  "rope",
    "Unsqueeze":        "unsqueeze",
    "Squeeze":          "squeeze",
    "Slice":            "slice",
    "Split":            "split",
    "ReduceMean":       "reduce_mean",
    "Pow":              "pow",
    "Sqrt":             "sqrt",
    "Div":              "div",
    "Sub":              "sub",
    "Cast":             "cast",
    "Expand":           "expand",
    "Where":            "where",
    "Less":             "less",
    "Range":            "range",
}


class ONNXLoader:
    """
    Loads ONNX models and extracts their graph, lowering ONNX ops to AEG-IR.

    Weight initializers are extracted as numpy float32 arrays, matching the
    interface expected by ``IngestionPipeline._bind_weights``.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """
        Load the ONNX model and return AEG-compatible graph information.

        Returns:
            dict with keys:
              - ``nodes``: list of AEG node descriptors
              - ``inputs``: list of input descriptors
              - ``outputs``: list of output descriptors
              - ``initializers``: name → float32 numpy array
              - ``weights``: same as initializers (alias)
              - ``opset``: ONNX opset version
              - ``ir_version``: ONNX IR version
        """
        if not self.model_path.exists():
            msg = f"ONNX file not found: {self.model_path}"
            raise IngestionError(msg)

        try:
            import onnx
            model = onnx.load(str(self.model_path))
            onnx.checker.check_model(model)
        except ImportError:
            # Fall back to raw protobuf parsing without onnx package
            return self._load_raw_protobuf()
        except Exception as exc:
            msg = f"Failed to load ONNX model {self.model_path}: {exc}"
            raise IngestionError(msg) from exc

        graph = model.graph

        # Extract nodes with AEG op mapping
        nodes: list[dict[str, Any]] = []
        for onnx_node in graph.node:
            aeg_op = _ONNX_OP_MAP.get(onnx_node.op_type, onnx_node.op_type.lower())
            attrs: dict[str, Any] = {}
            for attr in onnx_node.attribute:
                if attr.type == 1:    # FLOAT
                    attrs[attr.name] = attr.f
                elif attr.type == 2:  # INT
                    attrs[attr.name] = attr.i
                elif attr.type == 3:  # STRING
                    attrs[attr.name] = attr.s.decode("utf-8", errors="replace")
                elif attr.type == 4:  # TENSOR
                    arr = self._tensor_to_numpy(attr.t)
                    attrs[attr.name] = arr
                elif attr.type == 6:  # FLOATS
                    attrs[attr.name] = list(attr.floats)
                elif attr.type == 7:  # INTS
                    attrs[attr.name] = list(attr.ints)
            nodes.append({
                "name": onnx_node.name or f"{onnx_node.op_type}_{len(nodes)}",
                "op_type": aeg_op,
                "onnx_op_type": onnx_node.op_type,
                "inputs": list(onnx_node.input),
                "outputs": list(onnx_node.output),
                "attrs": attrs,
            })

        # Extract graph inputs/outputs with shapes
        inputs = [self._value_info_to_dict(vi) for vi in graph.input]
        outputs = [self._value_info_to_dict(vi) for vi in graph.output]

        # Extract weight initializers as numpy arrays
        initializers: dict[str, np.ndarray] = {}
        for init in graph.initializer:
            arr = self._tensor_to_numpy(init)
            if arr is not None:
                initializers[init.name] = arr

        opset = model.opset_import[0].version if model.opset_import else 17

        logger.info(
            "Loaded ONNX",
            path=str(self.model_path),
            nodes=len(nodes),
            weights=len(initializers),
            opset=opset,
        )
        return {
            "nodes": nodes,
            "inputs": inputs,
            "outputs": outputs,
            "initializers": initializers,
            "weights": initializers,
            "opset": opset,
            "ir_version": model.ir_version,
        }

    def _tensor_to_numpy(self, tensor: Any) -> np.ndarray | None:
        """Convert an ONNX TensorProto to a numpy array."""
        try:
            import onnx.numpy_helper
            arr = onnx.numpy_helper.to_array(tensor)
            return arr.astype(np.float32) if arr.dtype.kind == "f" else arr
        except Exception:
            return None

    def _value_info_to_dict(self, vi: Any) -> dict[str, Any]:
        """Convert ONNX ValueInfoProto to a shape descriptor."""
        try:
            tt = vi.type.tensor_type
            shape = []
            if tt.HasField("shape"):
                for dim in tt.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        shape.append(dim.dim_param)  # dynamic dim
                    else:
                        shape.append(None)
            dtype = _ONNX_ELEM_TYPE.get(tt.elem_type, "unknown")
            return {"name": vi.name, "shape": shape, "dtype": dtype}
        except Exception:
            return {"name": getattr(vi, "name", "?"), "shape": [], "dtype": "unknown"}

    def _load_raw_protobuf(self) -> dict[str, Any]:
        """
        Fallback path that reads ONNX files without the onnx package.

        Returns a minimal descriptor — nodes and weights will be empty but
        the structure is valid for architecture graph construction.
        """
        logger.warning(
            "onnx package not installed; ONNX file loaded without protobuf parsing",
            path=str(self.model_path),
        )
        data = self.model_path.read_bytes()
        return {
            "nodes": [],
            "inputs": [],
            "outputs": [],
            "initializers": {},
            "weights": {},
            "opset": 17,
            "ir_version": 8,
            "raw_bytes": len(data),
        }

    def __repr__(self) -> str:
        return f"ONNXLoader({self.model_path})"
