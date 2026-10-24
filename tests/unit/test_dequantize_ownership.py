"""Decoder ownership is explicit; real AEG loads reuse only private payloads."""

from __future__ import annotations

import gc
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aether import Compiler, CompilerConfig
from aether.core.weight_store import WeightStore
from aether.quantization.formats import QuantizedTensor, dequantize_tensor, quantize_tensor
from aether.runtime.aeg_loader import load_engine_from_package

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@pytest.mark.parametrize("precision", ["FP32", "FP16", "BF16", "Q4_K_M", "NF4", "FP8"])
def test_default_decoder_result_remains_independent(precision: str) -> None:
    """Existing callers may mutate decoded weights without changing the input."""
    tensor = quantize_tensor(np.linspace(-2, 2, 35, dtype=np.float32), precision)
    original = tensor.data.copy()
    decoded = dequantize_tensor(tensor)
    assert not np.shares_memory(decoded, tensor.data)
    decoded[:] = 19
    np.testing.assert_array_equal(tensor.data, original)


def test_explicit_reuse_preserves_fp32_bits_and_lifetime() -> None:
    """Views retain their payload, including signed zero and a NaN payload."""
    bits = np.array([0, 0x80000000, 0x3F800000, 0x7FC12345], np.uint32)
    values = bits.view(np.float32)
    tensor = QuantizedTensor("FP32", (2, 2), values)
    owner = weakref.ref(bits)
    decoded = dequantize_tensor(tensor, copy=False)
    assert np.shares_memory(decoded, values)
    assert decoded.flags.writeable
    np.testing.assert_array_equal(decoded.view(np.uint32).reshape(-1), bits)
    del tensor, values, bits
    gc.collect()
    assert owner() is not None
    decoded[0, 0] = 2
    assert decoded[0, 0] == 2


def test_readonly_reuse_does_not_fabricate_writable_storage() -> None:
    """Opting into aliasing retains input protection; the default stays writable."""
    tensor = quantize_tensor(np.arange(16, dtype=np.float32), "FP32")
    tensor.data.flags.writeable = False
    reused = dequantize_tensor(tensor, copy=False)
    copied = dequantize_tensor(tensor)
    assert np.shares_memory(reused, tensor.data)
    assert not reused.flags.writeable
    assert copied.flags.writeable
    assert not np.shares_memory(copied, tensor.data)


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(">f4")])
def test_reuse_still_converts_incompatible_dtype(dtype: np.dtype[np.generic]) -> None:
    """The no-copy preference cannot skip real conversion or byte-order changes."""
    values = np.array([1, -2, 0.125, 3], dtype=dtype)
    tensor = QuantizedTensor("FP16" if dtype.itemsize == 2 else "FP32", (2, 2), values)
    decoded = dequantize_tensor(tensor, copy=False)
    assert decoded.dtype == np.float32
    assert not np.shares_memory(decoded, values)
    np.testing.assert_array_equal(decoded, values.astype(np.float32).reshape(2, 2))


@pytest.mark.parametrize("precision", ["FP16", "BF16", "Q4_K_M", "NF4", "FP8"])
def test_reuse_does_not_change_other_codec_values(precision: str) -> None:
    """Stored BF16 bits and block padding still traverse their actual codecs."""
    tensor = quantize_tensor(np.linspace(-3, 3, 35, dtype=np.float32), precision)
    original = tensor.data.copy()
    expected = dequantize_tensor(tensor)
    actual = dequantize_tensor(tensor, copy=False)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert not np.shares_memory(actual, tensor.data)
    np.testing.assert_array_equal(tensor.data, original)


def test_legacy_float32_passthrough_and_invalid_shape() -> None:
    """Legacy dense BF16 payloads retain values; malformed shapes still raise."""
    values = np.arange(12, dtype=np.float32)
    tensor = QuantizedTensor("BF16", (3, 4), values)
    actual = dequantize_tensor(tensor, copy=False)
    assert np.shares_memory(actual, values)
    np.testing.assert_array_equal(actual, dequantize_tensor(tensor))
    tensor.shape = (3, 5)
    with pytest.raises(ValueError, match="reshape"):
        dequantize_tensor(tensor, copy=False)


def test_actual_loader_keeps_private_buffers_and_independent_model_loads(
    tiny_local_safetensors_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse actual reads without sharing weights across models or changing AEG bytes."""
    package = Compiler(
        CompilerConfig(
            quality_mode="diagnostic",
            precision="FP32",
            precision_assignment_mode="uniform",
            targets=["cpu_generic"],
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "model.aeg")
    read_tensor = WeightStore.load_tensor
    payloads: dict[str, list[NDArray[np.generic]]] = {}

    def record_read(store: WeightStore, name: str) -> QuantizedTensor:
        tensor = read_tensor(store, name)
        payloads.setdefault(name, []).append(tensor.data)
        return tensor

    monkeypatch.setattr(WeightStore, "load_tensor", record_read)
    engine = load_engine_from_package(package, prepare_native_kernels=False, allow_uncertified=True)
    for name, actual in (
        ("embedding", engine.weights.embedding),
        ("layer_0_q_proj", engine.weights.layers[0].q_proj),
    ):
        assert actual.flags.writeable
        assert any(np.shares_memory(actual, payload) for payload in payloads[name])
    second = load_engine_from_package(package, prepare_native_kernels=False, allow_uncertified=True)
    assert not np.shares_memory(engine.weights.embedding, second.weights.embedding)
    original = second.weights.embedding.copy()
    engine.weights.embedding[1, 0] += 2
    np.testing.assert_array_equal(second.weights.embedding, original)
    package.verify_integrity()
    store = package.weight_store()
    store.close()
    store.blob_path.unlink()
    logits, _ = second.forward(np.asarray([1, 2], np.int64))
    assert np.isfinite(logits).all()
