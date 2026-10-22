"""Preserve source normalization constants in actual deserialized execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from normalization_epsilon_fixtures import FAMILIES, write_package

from aether.runtime.aeg_loader import load_engine_from_package

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("epsilon", [0.0, 0.002])
def test_loaded_normalization_matches_independent_math(
    tmp_path: Path, family: str, epsilon: float,
) -> None:
    """A valid zero is a model parameter, not an instruction to substitute 1e-5.

    RMS contract: HF Transformers v4.57.3 LlamaRMSNorm.forward,
    https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/llama/modeling_llama.py
    LayerNorm uses population variance and subtracts the feature mean.
    """
    package = write_package(tmp_path / "model.aeg", family, epsilon)
    engine: Any = load_engine_from_package(
        package, prepare_native_kernels=False, allow_uncertified=True,
    )
    value = np.asarray([[1, -2, 3, 4, -5, 6, 7, -8]], dtype=np.float32) * 1e-8
    scale = np.linspace(0.7, 1.3, 8, dtype=np.float32)
    reference = value.astype(np.float64)
    if family in {"encoder", "rwkv"}:
        reference -= reference.mean(axis=-1, keepdims=True)
    expected = reference / np.sqrt(np.mean(reference**2, axis=-1, keepdims=True) + epsilon) * scale
    if family == "encoder":
        actual = engine._layer_norm(value, scale)
    elif family == "rwkv":
        actual = engine._norm(value, scale, engine.weights.norm_eps)
    elif family == "hybrid":
        actual = engine._transformer._norm(value, scale)
        np.testing.assert_allclose(
            engine._mamba._norm(value, scale), expected, rtol=2e-6, atol=1e-12,
        )
    else:
        actual = engine._norm(value, scale)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize("family", [family for family in FAMILIES if family != "seq2seq"])
def test_null_epsilon_keeps_existing_family_fallback(tmp_path: Path, family: str) -> None:
    """Null metadata remains compatible with established decoder/encoder defaults."""
    package = write_package(tmp_path / "null.aeg", family, None)
    engine: Any = load_engine_from_package(
        package, prepare_native_kernels=False, allow_uncertified=True,
    )
    expected = 1e-12 if family == "encoder" else 1e-5
    assert engine.weights.norm_eps == expected


@pytest.mark.parametrize("family,backend", [
    ("llama", "cpu"), ("llama", "torch_cpu"), ("llama", "onnx_cpu"),
    ("mamba2", "cpu"), ("mamba2", "torch_cpu"),
])
def test_zero_epsilon_compiled_logits_and_greedy_ids_match_original_hf(
    tmp_path: Path, tiny_local_safetensors_model: Path, family: str, backend: str,
) -> None:
    """Execute complete source checkpoints; compare fresh/cached rows and generation."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mamba2_normalization_fixtures import write_checkpoint

    from aether import Compiler, CompilerConfig

    source = tiny_local_safetensors_model
    if family == "mamba2":
        source = tmp_path / "mamba2"
        write_checkpoint(source, epsilon=0.0)
    else:
        from safetensors.numpy import load_file, save_file

        config_path = source / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["rms_norm_eps"] = 0.0
        config_path.write_text(json.dumps(config), encoding="utf-8")
        # Small embedding variance makes a substituted epsilon observable.
        # Use ordinary projection scales so FP32 backend order differences
        # cannot dominate this configuration regression.
        weights_path = source / "model.safetensors"
        tensors = load_file(str(weights_path))
        for name, value in tensors.items():
            if value.ndim == 2:
                tensors[name] = value * (1e-4 if "embed_tokens" in name else 0.02)
        save_file(tensors, str(weights_path))
    artifact = tmp_path / "compiled.aeg"
    Compiler(CompilerConfig(
        quality_mode="diagnostic", targets=["cpu_generic"], precision="FP32",
        precision_assignment_mode="uniform", skip_download=True,
        cache_dir=str(tmp_path / "cache"),
    )).compile(str(source), output_path=artifact)
    from aether.runtime.aeg_loader import load_engine_from_path

    engine: Any = load_engine_from_path(
        artifact, prepare_native_kernels=False, allow_uncertified=True,
    )
    if backend == "torch_cpu":
        if family == "mamba2":
            from aether.runtime.torch_state_engine import TorchMamba2AEGEngine

            engine = TorchMamba2AEGEngine(engine, "cpu")
        else:
            from aether.runtime.torch_engine import TorchAEGEngine

            engine = TorchAEGEngine(engine, "cpu")
    elif backend == "onnx_cpu":
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        from aether.runtime.onnx_export import ONNXAEGEngine, export_engine_to_onnx

        bundle = export_engine_to_onnx(engine, tmp_path / "onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        engine = ONNXAEGEngine(bundle, *[
            ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            for path in (bundle.prefill_path, bundle.decode_path)
        ])
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32,
    ).eval()
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    with torch.inference_mode():
        expected = reference(torch.tensor(ids[None, :])).logits[0].numpy()
        generated = ids[:3].tolist()
        for _ in range(8):
            logits = reference(torch.tensor([generated])).logits[0, -1]
            assert bool(torch.isfinite(logits).all())
            generated.append(int(logits.argmax()))
    actual, _ = engine.forward(ids)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    _, cache = engine.forward(ids[:3])
    tail, _ = engine.forward(ids[3:], cache)
    np.testing.assert_allclose(tail, expected[3:], rtol=3e-5, atol=3e-6)
    assert engine.generate(ids[:3], max_tokens=8, temperature=0.0) == generated[3:]
