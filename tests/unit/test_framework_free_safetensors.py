"""Regression tests for framework-free SafeTensors ingestion."""

from __future__ import annotations

import json
import os
import subprocess
import struct
import sys

import numpy as np

from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader


def test_bfloat16_safetensors_loads_without_torch(tmp_path) -> None:
    values = np.asarray([1.0, -0.5, 3.25], dtype=np.float32)
    bfloat_bits = (values.view(np.uint32) >> 16).astype("<u2")
    header = json.dumps(
        {"weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bfloat_bits.tobytes())

    loaded = SafeTensorsLoader(path).load()

    np.testing.assert_array_equal(loaded["weight"], values)


def test_native_backend_is_not_derived_from_optional_framework_backend() -> None:
    """The executable CPU backend must have no PyTorch import boundary."""
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    code = (
        "import sys; "
        "from aether.backends.base import Backend; "
        "from aether.backends.native_cpu_backend import NativeCPUBackend; "
        "assert NativeCPUBackend.__bases__ == (Backend,); "
        "assert 'torch' not in sys.modules; "
        "assert 'aether.backends.torch_backend' not in sys.modules"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = source_root
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
