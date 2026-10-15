"""Compiled storage codecs are decoded once before framework-free CPU execution."""

import builtins
import json
from dataclasses import fields, replace

import numpy as np
import pytest

from aether import Compiler, CompilerConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.core.aeg_format import AEGPackage
from aether.kernels.native_cpu import NativeCPUKernels
from aether.quantization.formats import dequantize_tensor, quantize_tensor
from aether.runtime.aeg_loader import load_engine_from_package
from aether.runtime.cpu_engine import CPUExecutionEngine

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _write_learned_checkpoint(path):
    from safetensors.numpy import save_file

    _write_probe_checkpoint(path)
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_layer": 1,
        "n_embd": 16,
        "n_head": 2,
        "n_inner": 32,
        "n_positions": 32,
        "n_ctx": 32,
        "vocab_size": 32,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu_new",
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rng = np.random.default_rng(193)
    sizes = {
        "transformer.wte.weight": (32, 16),
        "transformer.wpe.weight": (32, 16),
        "lm_head.weight": (32, 16),
        "transformer.h.0.attn.c_attn.weight": (16, 48),
        "transformer.h.0.attn.c_proj.weight": (16, 16),
        "transformer.h.0.mlp.c_fc.weight": (16, 32),
        "transformer.h.0.mlp.c_proj.weight": (32, 16),
    }
    tensors = {name: rng.normal(0, 0.1, shape).astype(np.float32) for name, shape in sizes.items()}
    for prefix in ("transformer.ln_f", "transformer.h.0.ln_1", "transformer.h.0.ln_2"):
        tensors[prefix + ".weight"] = np.ones(16, np.float32)
        tensors[prefix + ".bias"] = np.zeros(16, np.float32)
    for prefix, size in (
        ("attn.c_attn", 48),
        ("attn.c_proj", 16),
        ("mlp.c_fc", 32),
        ("mlp.c_proj", 16),
    ):
        tensors["transformer.h.0." + prefix + ".bias"] = np.zeros(size, np.float32)
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture(scope="module", params=["rope", "learned"])
def checkpoint(request, tmp_path_factory):
    """Write actual independent RoPE/RMSNorm and learned-position/LayerNorm checkpoints."""
    root = tmp_path_factory.mktemp("quantized-" + request.param)
    source = root / "checkpoint"
    (_write_probe_checkpoint if request.param == "rope" else _write_learned_checkpoint)(source)
    return source, request.param


def _compile(source, destination, precision):
    config = CompilerConfig(
        targets=["cpu_generic"],
        precision=precision,
        optimization_level=0,
        cache_dir=str(destination.parent / "compiler-cache"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    Compiler(config).compile(str(source), output_path=destination)
    return AEGPackage(destination).load()


def _reference_from_stored_codecs(package, engine):
    """Build an independent numeric reference directly from every stored codec."""
    store = package.weight_store()
    layers = []
    for index, layer in enumerate(engine.weights.layers):
        projections = {
            name: dequantize_tensor(store.load_tensor(f"layer_{index}_{name}"))
            for name in PROJECTIONS
            if getattr(layer, name) is not None
        }
        layers.append(replace(layer, **projections, quant_meta=None, storage_quant_meta=None))
    return CPUExecutionEngine(
        replace(engine.weights, layers=layers),
        engine.num_heads,
        engine.num_kv_heads,
        kernels=NativeCPUKernels.reference(),
    )


@pytest.mark.parametrize(
    "precision,block_size",
    [
        ("FP32", None),
        ("BF16", None),
        ("INT8", None),
        ("INT4", None),
        ("FP8", None),
        ("FP8_E5M2", None),
        ("Q4_K_M", None),
        ("Q8_0", None),
        ("NF4", 31),
        ("MXFP4", 31),
        ("TERNARY", 31),
        ("Q3_K", 31),
    ],
)
def test_compiled_codec_prefill_decode_and_request_clones(
    checkpoint, tmp_path, monkeypatch, precision, block_size
):
    """Stored formats preserve exact decoded prefill/decode across static request clones."""
    source, architecture = checkpoint
    package = _compile(
        source, tmp_path / "quantized.aeg", precision if block_size is None else "FP32"
    )
    if block_size is not None:
        # Exercise real public codecs (including padded block tails) on a real
        # compiled graph. NF4 is a storage codec, not an advertised CLI enum.
        store = package.weight_store()
        package.weights = {name: store.load_tensor(name) for name in store.entries}
        for name in tuple(package.weights):
            if name.startswith("layer_") and name.endswith(PROJECTIONS):
                package.weights[name] = quantize_tensor(
                    dequantize_tensor(package.weights[name]), precision, block_size=block_size
                )
        package.save()
        package = AEGPackage(package.root).load()
    engine = load_engine_from_package(package, prepare_native_kernels=False)
    if architecture == "learned":
        assert engine.weights.norm_type.lower() == "layernorm"
        assert engine.weights.position_embedding is not None
    layer = engine.weights.layers[0]
    assert layer.quant_meta is None
    store = package.weight_store()
    quantized = precision not in {"FP32", "BF16"}
    assert bool(layer.storage_quant_meta) == quantized
    for name, (fmt, scales, zeroes) in (layer.storage_quant_meta or {}).items():
        stored = store.load_tensor("layer_0_" + name)
        assert fmt == stored.precision
        np.testing.assert_array_equal(scales, stored.scales)
        np.testing.assert_array_equal(zeroes, stored.zero_points)
        np.testing.assert_array_equal(getattr(layer, name), dequantize_tensor(stored))
    reference = _reference_from_stored_codecs(package, engine)
    task = engine.with_task_deltas({"layer_0_q_proj": np.zeros_like(layer.q_proj)})
    rank = 2
    lora = engine.with_lora_adapter(
        {
            "zero": {
                (0, "q_proj"): (
                    np.zeros((rank, layer.q_proj.shape[1]), np.float32),
                    np.zeros((layer.q_proj.shape[0], rank), np.float32),
                    1.0,
                )
            }
        },
        "zero",
    )
    assert task.weights.layers[0].storage_quant_meta is layer.storage_quant_meta
    assert lora.weights.layers[0].storage_quant_meta is layer.storage_quant_meta
    from aether.runtime.execution_precision import NumpyPrecisionAdapter

    assert all(
        "storage_quant_meta" not in binding.name
        for binding in NumpyPrecisionAdapter(engine).bindings()
    )

    import aether.kernels.quantized_linear as quantized_linear

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("decoded CPU AEG must not invoke encoded/GPU quantized dispatch")

    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("decoded CPU AEG must not import optional Torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(quantized_linear, "QuantizedLinear", unexpected_dispatch)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    prefix, suffix = np.array([1, 2, 3]), np.array([4])
    expected, expected_cache = reference.forward(prefix)
    expected_decode, _ = reference.forward(suffix, expected_cache)
    for actual_engine in (engine, task, lora):
        actual, cache = actual_engine.forward(prefix)
        np.testing.assert_array_equal(actual, expected)
        actual_decode, _ = actual_engine.forward(suffix, cache)
        np.testing.assert_array_equal(actual_decode, expected_decode)
    delta = np.ones_like(layer.q_proj) * np.float32(0.03)
    selected = engine.with_task_deltas({"layer_0_q_proj": delta})
    explicit_layer = replace(
        reference.weights.layers[0], q_proj=reference.weights.layers[0].q_proj + delta
    )
    explicit = CPUExecutionEngine(
        replace(reference.weights, layers=[explicit_layer]),
        engine.num_heads,
        engine.num_kv_heads,
        kernels=engine.kernels,
    )
    for ids in (prefix, suffix):
        actual, _ = selected.forward(ids)
        expected_delta, _ = explicit.forward(ids)
        np.testing.assert_array_equal(actual, expected_delta)


def test_active_encoded_projection_delta_is_rejected_without_mutation(tmp_path):
    """A manually encoded projection cannot receive arithmetic in its code space."""
    source = tmp_path / "source"
    _write_probe_checkpoint(source)
    package = _compile(source, tmp_path / "encoded.aeg", "INT8")
    engine = load_engine_from_package(package, prepare_native_kernels=False)
    encoded = np.ones_like(engine.weights.layers[0].q_proj, dtype=np.int8)
    layer = replace(
        engine.weights.layers[0],
        q_proj=encoded,
        quant_meta={"q_proj": ("int8", np.array([0.1]), None)},
    )
    engine = CPUExecutionEngine(
        replace(engine.weights, layers=[layer]),
        engine.num_heads,
        engine.num_kv_heads,
        kernels=engine.kernels,
    )
    with pytest.raises(ValueError, match="encoded quantized weights"):
        engine.with_task_deltas({"layer_0_q_proj": np.zeros_like(encoded, dtype=np.float32)})
    np.testing.assert_array_equal(engine.weights.layers[0].q_proj, encoded)
    assert engine.weights.layers[0].quant_meta is layer.quant_meta
    # Preserve the manual encoded API: signed per-tensor INT8 values plus
    # their actual scale still execute correctly on NumPy CPU activations.
    activation = np.ones((1, encoded.shape[1]), np.float32)
    actual = engine._linear(activation, encoded, (0, "q_proj"))
    np.testing.assert_allclose(actual, activation @ (encoded.astype(np.float32) * 0.1).T)
