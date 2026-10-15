"""Real ONNX operator parity; no provider registration is execution evidence."""

from dataclasses import replace

import numpy as np
import pytest

from aether.kernels.native_cpu import NativeCPUKernels
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights


def make_engine(variant="rope"):
    """Build generic dense decoder operators covering independent semantic variants."""
    rng = np.random.default_rng(19)
    hidden, intermediate, heads, kv_heads, dim = 16, 24, 4, 2, 4

    def matrix(out, features):
        return (rng.normal(size=(out, features)) * 0.13).astype(np.float32)

    layers = [
        LayerWeights(
            attention_norm=np.ones(hidden, np.float32),
            q_proj=matrix(heads * dim, hidden),
            k_proj=matrix(kv_heads * dim, hidden),
            v_proj=matrix(kv_heads * dim, hidden),
            o_proj=matrix(hidden, heads * dim),
            ffn_norm=np.ones(hidden, np.float32),
            gate_proj=matrix(intermediate, hidden),
            up_proj=matrix(intermediate, hidden),
            down_proj=matrix(hidden, intermediate),
        )
        for _ in range(2)
    ]
    weights = ModelWeights(
        embedding=matrix(29, hidden),
        layers=layers,
        final_norm=np.ones(hidden, np.float32),
        lm_head=matrix(29, hidden),
        context_length=32,
    )
    if variant == "learned":
        weights = replace(
            weights,
            position_type="learned",
            position_embedding=matrix(32, hidden),
            norm_type="LayerNorm",
            ffn_type="GELU",
        )
        for layer in layers:
            layer.up_proj = None
            layer.attention_norm_bias = rng.normal(size=hidden).astype(np.float32) * 0.01
            layer.ffn_norm_bias = rng.normal(size=hidden).astype(np.float32) * 0.01
            layer.q_proj_bias = rng.normal(size=heads * dim).astype(np.float32) * 0.01
    elif variant == "alibi":
        weights = replace(
            weights, position_type="ALiBi", norm_type="LayerNorm", parallel_residual=True
        )
    elif variant == "scaled_sandwich":
        weights = replace(
            weights,
            norm_offset_one=True,
            norm_placement="sandwich",
            embedding_scale=2.0,
            residual_scale=0.7,
            logit_scale=0.5,
            attn_logit_softcap=2.0,
            final_logit_softcap=3.0,
        )
        for layer in layers:
            layer.post_attention_norm = np.ones(hidden, np.float32) * 0.3
            layer.post_ffn_norm = np.ones(hidden, np.float32) * 0.2
    elif variant == "partial_interleaved":
        weights = replace(
            weights,
            rope_partial_dim=2,
            rope_interleaved=True,
            attention_layers=["local", "global"],
            attention_window=2,
            rope_local_theta=5000.0,
        )
    elif variant == "qk_post":
        weights = replace(weights, norm_placement="post", qk_norm_scope="full")
        for layer in layers:
            layer.q_norm = np.ones(heads * dim, np.float32)
            layer.k_norm = np.ones(kv_heads * dim, np.float32)
    elif variant == "exact_geglu":
        weights = replace(weights, ffn_type="GeGLU", gelu_approximate=False)
    return CPUExecutionEngine(weights, heads, kv_heads, kernels=NativeCPUKernels.reference())


def sessions(bundle, provider="CPUExecutionProvider"):
    """Create real strict provider sessions for exported prefill and cached decode."""
    ort = pytest.importorskip("onnxruntime")

    def session(path, phase):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        if provider == "DmlExecutionProvider":
            if provider not in ort.get_available_providers():
                pytest.skip("real DirectML provider is not installed")
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.enable_profiling = True
            options.profile_file_prefix = str(bundle.root / f"execution-{phase}")
        return ort.InferenceSession(str(path), options, providers=[provider])

    from aether.runtime.onnx_export import ONNXAEGEngine

    return ONNXAEGEngine(
        bundle,
        session(bundle.prefill_path, "prefill"),
        session(bundle.decode_path, "decode"),
    )


@pytest.mark.parametrize(
    "variant",
    [
        "rope",
        "learned",
        "alibi",
        "scaled_sandwich",
        "partial_interleaved",
        "qk_post",
        "exact_geglu",
    ],
)
def test_real_exported_operators_prefill_cached_decode_and_cpu_cache_handoff(tmp_path, variant):
    """Match actual CPU ORT prefix, decode and transferred-cache operator results."""
    pytest.importorskip("onnx")
    from aether.runtime.onnx_export import export_engine_to_onnx

    reference = make_engine(variant)
    bundle = export_engine_to_onnx(reference, tmp_path / "onnx")
    engine = sessions(bundle)
    expected, cpu_cache = reference.forward(np.array([1, 4, 2]))
    actual, cache = engine.forward(np.array([1, 4, 2]))
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    expected_decode, cpu_cache = reference.forward(np.array([7, 3]), cpu_cache)
    actual_decode, cache = engine.forward(np.array([7, 3]), cache)
    np.testing.assert_allclose(actual_decode, expected_decode, rtol=3e-5, atol=2e-6)
    assert cache.length == 5
    for index in range(2):
        np.testing.assert_allclose(
            cache.layer_view(index)[0], cpu_cache.layer_view(index)[0], rtol=3e-5, atol=2e-6
        )
    # A real CPU-produced cache can continue in the provider executor.
    expected_next, _ = reference.forward(np.array([5]), cpu_cache.clone())
    actual_next, _ = engine.forward(np.array([5]), cpu_cache.clone())
    np.testing.assert_allclose(actual_next, expected_next, rtol=3e-5, atol=2e-6)
    assert bundle.metadata["arithmetic_precision"] == "fp32"
    assert (bundle.root / "weights.data").is_file()


def test_export_is_explicit_bounded_and_rejects_unsupported_semantics(tmp_path):
    """Preserve existing destinations and reject unsupported execution contracts."""
    pytest.importorskip("onnx")
    from aether.runtime.onnx_export import export_engine_to_onnx

    engine = make_engine()
    engine.sparse_attention_plan = {"format": "active"}
    with pytest.raises(Exception, match="cannot discard"):
        export_engine_to_onnx(engine, tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()
    engine.sparse_attention_plan = None
    bundle = export_engine_to_onnx(engine, tmp_path / "valid", max_context=3)
    with pytest.raises(FileExistsError):
        export_engine_to_onnx(engine, bundle.root)
    executor = sessions(bundle)
    _, cache = executor.forward(np.array([1, 2, 3]))
    with pytest.raises(Exception, match="exceeds exported context"):
        executor.forward(np.array([4]), cache)


def test_real_directml_exported_aeg_operators(tmp_path):
    """Require physical DirectML node placement and actual CPU-reference parity."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("physical DirectML execution requires its optional installed runtime")
    from aether.runtime.onnx_export import export_engine_to_onnx

    reference = make_engine()
    engine = sessions(
        export_engine_to_onnx(reference, tmp_path / "directml"), "DmlExecutionProvider"
    )
    expected, reference_cache = reference.forward(np.array([1, 2, 3]))
    actual, cache = engine.forward(np.array([1, 2, 3]))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-5)
    expected, _ = reference.forward(np.array([4]), reference_cache)
    actual, _ = engine.forward(np.array([4]), cache)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-5)
    import json

    for session in (engine.prefill_session, engine.decode_session):
        with open(session.end_profiling(), encoding="utf-8") as profile_file:
            profile = json.load(profile_file)
        attributed = [
            event["args"]["provider"]
            for event in profile
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        ]
        assert attributed and set(attributed) == {"DmlExecutionProvider"}


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "DmlExecutionProvider"])
def test_actual_compiled_artifact_backend_generation_stream_and_unload(
    tmp_path, provider, monkeypatch
):
    """Execute compiler artifacts, public generation and owned provider-worker pools."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if provider not in ort.get_available_providers():
        pytest.skip(f"real {provider} is not installed")
    from dataclasses import fields

    from aether import Compiler, CompilerConfig
    from aether.backends.base import GenerationRequest
    from aether.backends.hardware_validation import _write_probe_checkpoint
    from aether.backends.onnx_backend import ONNXBackend
    from aether.core.aeg_format import load_aeg_package
    from aether.runtime.aeg_loader import load_engine_from_path

    _write_probe_checkpoint(tmp_path / "source")
    config = CompilerConfig(
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "cache"),
    )
    for field in fields(config):
        if field.name.startswith("enable_"):
            setattr(config, field.name, field.name == "enable_precision_assignment")
    artifact = tmp_path / "model.aeg"
    Compiler(config).compile(str(tmp_path / "source"), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.supports_portable_backend("onnx")
    backend = ONNXBackend()
    assert backend.available_for_model("logical", str(artifact))
    handle = backend.load_model(
        "logical",
        str(artifact),
        providers=[provider],
        allow_provider_fallback=False,
        enable_profiling=True,
        profile_file_prefix=str(tmp_path / "profile"),
    )
    engine = handle.engine
    reference = load_engine_from_path(artifact, prepare_native_kernels=False)
    prompt = np.array([1, 2])
    expected = reference.generate(prompt, max_tokens=3, temperature=0.0)
    request = GenerationRequest(
        model_id="logical", prompt="hello world", max_tokens=3, temperature=0.0
    )
    completion = backend.generate(request)
    assert completion.text == backend._tokenizers["logical"].decode(expected)
    assert completion.completion_tokens == 3
    assert completion.metrics["cached_decode"] is True
    assert "".join(backend.generate_stream(request)) == completion.text
    attribution = backend.provider_execution_evidence("logical", provider)
    assert attribution["executed_node_count"] > 0
    assert attribution["execution_scope"] == "aeg_onnx_prefill_cached_decode"
    bundle_path = engine.bundle.root
    assert bundle_path.is_dir()
    with pytest.raises(Exception, match="different providers"):
        backend.load_model("logical", str(artifact), providers=["not-the-same-provider"])
    backend.unload_model("logical")
    assert not backend._models and not bundle_path.exists()
    if provider in {"DmlExecutionProvider", "CPUExecutionProvider"}:
        # Actual public Runtime routes must reach the requested provider. The
        # DirectML variant uses the real local GPU inventory and node profiles.
        from aether import Runtime, RuntimeConfig
        from aether.runtime.execution_features import RuntimeExecutionFeatures

        device = "directml:0" if provider == "DmlExecutionProvider" else "cpu"
        # Only memory-policy input is controlled: model execution and node
        # attribution still use actual ORT providers and the real local device.
        monkeypatch.setattr(
            RuntimeExecutionFeatures,
            "memory_pressure",
            staticmethod(lambda handle: (0.99, "system_ram")),
        )

        runtime = Runtime(
            RuntimeConfig(
                backend_name="onnx",
                execution_devices=[device],
                prefill_devices=[device],
                decode_devices=[device],
                precision_calibration_tokens=[[1, 2]],
                speculative_decoding=False,
                disaggregate_prefill_decode=True,
                prefill_chunk_size=128,
                enable_semantic_cache=False,
                enable_prefix_caching=False,
                model_cache_dir=str(tmp_path / "runtime-cache"),
            )
        )
        adapter = runtime.backend_registry.get_backend("onnx")
        load = adapter.load_model

        def profiled_load(model_id, aeg_path=None, **kwargs):
            kwargs.setdefault("enable_profiling", True)
            kwargs.setdefault("profile_file_prefix", str(tmp_path / "runtime-profile"))
            return load(model_id, aeg_path, **kwargs)

        monkeypatch.setattr(adapter, "load_model", profiled_load)
        result = runtime.generate(str(artifact), "hello world", max_tokens=3, temperature=0.0)
        assert result.text == completion.text
        assert result.metrics.active_precision == "float32"
        if provider == "DmlExecutionProvider":
            assert result.metrics.kernel_target == "directml"
        else:
            assert result.metrics.kernel_target.startswith("cpu_")
        assert runtime._loaded_models[str(artifact)].engine.device == device
        precision = result.metrics.extra["precision_adjustment"]
        assert precision["action"] == "unsupported"
        assert precision["execution_changed"] is False
        assert "mutable structured weights" in precision["reason"]
        with pytest.raises(ValueError, match="mutable structured weights"):
            runtime.adjust_precision(str(artifact), [[1, 2]])
        assert result.metrics.extra["transport"] == "in_process_typed_snapshot"
        assert result.metrics.extra["state_bytes_handed_off"] > 0
        assert (
            "".join(
                runtime.generate_stream(str(artifact), "hello world", max_tokens=3, temperature=0.0)
            )
            == completion.text
        )
        proof = runtime._loaded_backends[str(artifact)].provider_execution_evidence(
            str(artifact), provider
        )
        assert proof["executed_node_count"] > 0
        runtime.close()
