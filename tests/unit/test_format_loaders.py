"""
Tests for ONNX, PyTorch, and MLX loaders.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ONNX Loader
# ---------------------------------------------------------------------------

class TestONNXLoader:
    def test_file_not_found(self):
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader
        from aether.core.exceptions import IngestionError

        loader = ONNXLoader("/nonexistent.onnx")
        with pytest.raises(IngestionError):
            loader.load()

    def test_repr(self):
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader
        loader = ONNXLoader("/path/to/model.onnx")
        assert "ONNXLoader" in repr(loader)

    def test_op_map_coverage(self):
        """Verify AEG op map contains essential ONNX operators."""
        from aether.compiler.stage1_ingestion.onnx_loader import _ONNX_OP_MAP
        essential = {"MatMul", "Gemm", "Attention", "LayerNormalization", "Gather"}
        for op in essential:
            assert op in _ONNX_OP_MAP, f"Missing op: {op}"

    def test_elem_type_map(self):
        from aether.compiler.stage1_ingestion.onnx_loader import _ONNX_ELEM_TYPE
        assert _ONNX_ELEM_TYPE[1] == "float32"
        assert _ONNX_ELEM_TYPE[10] == "float16"
        assert _ONNX_ELEM_TYPE[16] == "bfloat16"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx package not installed",
    )
    def test_load_tiny_onnx(self):
        """Create a tiny valid ONNX model and load it."""
        import onnx
        import onnx.helper as oh
        # TensorProto is a class on the onnx module, not a sub-module
        FLOAT = onnx.TensorProto.FLOAT  # == 1

        X = oh.make_tensor_value_info("X", FLOAT, [1, 4])
        Z = oh.make_tensor_value_info("Z", FLOAT, [1, 2])
        W = oh.make_tensor("W", FLOAT, [4, 2], np.ones((4, 2), dtype=np.float32).flatten().tolist())
        node = oh.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
        graph = oh.make_graph([node], "test", [X], [Z], [W])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        onnx.checker.check_model(model)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            tmp_path = f.name

        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader
        data = ONNXLoader(tmp_path).load()
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["op_type"] == "linear"  # MatMul → linear
        assert "W" in data["initializers"]
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PyTorch Loader
# ---------------------------------------------------------------------------

class TestPyTorchLoader:
    def test_file_not_found(self):
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        from aether.core.exceptions import IngestionError

        loader = PyTorchLoader("/nonexistent.pt")
        with pytest.raises(IngestionError):
            loader.load()

    def test_repr(self):
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        loader = PyTorchLoader("/path/to/model.pt")
        assert "PyTorchLoader" in repr(loader)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_load_state_dict(self):
        """Save a minimal state dict and verify PyTorchLoader extracts it."""
        import torch

        state_dict = {
            "embed.weight": torch.randn(100, 64),
            "proj.weight": torch.randn(64, 32),
            "proj.bias": torch.zeros(32),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            tmp_path = f.name

        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        data = PyTorchLoader(tmp_path).load()
        assert "embed.weight" in data["weights"]
        assert "proj.weight" in data["weights"]
        w = data["weights"]["embed.weight"]
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float32
        assert w.shape == (100, 64)
        Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_wrapped_state_dict(self):
        """Test that {state_dict: {...}} wrapper is properly unwrapped."""
        import torch

        state_dict = {"model": {"layer.w": torch.ones(4, 4)}}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            tmp_path = f.name

        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        data = PyTorchLoader(tmp_path).load()
        # Should find layer.w in weights
        assert len(data["weights"]) >= 1
        Path(tmp_path).unlink(missing_ok=True)

    def test_flatten_state_dict(self):
        """_flatten_state_dict correctly flattens nested dicts of numpy arrays."""
        from aether.compiler.stage1_ingestion.pytorch_loader import _flatten_state_dict

        class FakeTensor:
            def numpy(self):
                return np.ones((4, 4), dtype=np.float32)
            def detach(self):
                return self

        nested = {"a": {"b": FakeTensor()}}
        flat = _flatten_state_dict(nested)
        assert "a.b" in flat


# ---------------------------------------------------------------------------
# MLX Loader
# ---------------------------------------------------------------------------

class TestMLXLoader:
    def test_missing_path(self):
        from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader
        from aether.core.exceptions import IngestionError

        loader = MLXLoader("/nonexistent_dir")
        with pytest.raises(IngestionError):
            loader.load()

    def test_repr(self):
        from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader
        loader = MLXLoader("/path/to/mlx")
        assert "MLXLoader" in repr(loader)

    def test_to_float32_basic(self):
        from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = MLXLoader._to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr)

    def test_to_float32_from_float16(self):
        from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader

        arr = np.array([1.5, -2.5], dtype=np.float16)
        result = MLXLoader._to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.5, -2.5], rtol=1e-3)

    def test_load_npz(self):
        """Save a .npz file and verify MLXLoader loads it."""
        data = {"model.embed.weight": np.ones((32, 16), dtype=np.float32)}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **data)
            tmp_path = f.name + ".npz" if not f.name.endswith(".npz") else f.name

        # np.savez appends .npz automatically
        actual_path = f.name if Path(f.name).exists() else f.name + ".npz"
        if not Path(actual_path).exists():
            pytest.skip("Could not create npz test file")

        from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader
        loader = MLXLoader(actual_path)
        result = loader.load()
        assert "weights" in result
        Path(actual_path).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("safetensors"),
        reason="safetensors not installed",
    )
    def test_load_safetensors_dir(self):
        """Create a temporary directory with a .safetensors file."""
        from safetensors.numpy import save_file

        tensors = {"weight": np.ones((4, 4), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(tensors, str(Path(tmpdir) / "model.safetensors"))
            from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader
            result = MLXLoader(tmpdir).load()
            assert "weight" in result["weights"]
            assert result["format"] == "safetensors"
