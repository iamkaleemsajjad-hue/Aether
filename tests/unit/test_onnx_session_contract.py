"""Actual compiled ONNX sessions retain provenance and reject unsupported hooks."""

from dataclasses import fields

import numpy as np
import pytest

from aether import Compiler, CompilerConfig
from aether.backends.base import GenerationRequest
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.backends.onnx_backend import ONNXBackend
from aether.runtime.session_contract import session_execution_contract


@pytest.fixture
def artifact(tmp_path):
    """Compile a complete local decoder; ONNX remains an optional test dependency."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    source = tmp_path / "checkpoint"
    _write_probe_checkpoint(source)
    config = CompilerConfig(
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "compiler-cache"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    result = tmp_path / "model.aeg"
    Compiler(config).compile(str(source), output_path=result)
    return str(result)


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "DmlExecutionProvider"])
def test_actual_onnx_session_reuses_only_bound_state(artifact, provider, tmp_path, monkeypatch):
    """Unknown provenance recomputes the prefix through the real selected provider."""
    import onnxruntime as ort

    if provider not in ort.get_available_providers():
        pytest.skip(f"real {provider} is not installed")
    backend = ONNXBackend()
    handle = backend.load_model(
        "model",
        artifact,
        providers=[provider],
        allow_provider_fallback=False,
        enable_profiling=True,
        profile_file_prefix=str(tmp_path / "actual-profile"),
    )
    calls = []
    original = handle.engine.forward

    def observed(token_ids, *args, **kwargs):
        calls.append(np.asarray(token_ids).size)
        return original(token_ids, *args, **kwargs)

    monkeypatch.setattr(handle.engine, "forward", observed)
    request = GenerationRequest(
        model_id="model",
        prompt="hello world",
        max_tokens=1,
        temperature=0,
        extra={"aether_kv_session_id": "session"},
    )
    try:
        backend.generate(request)
        assert handle.session_contracts["session"] == session_execution_contract(handle)
        assert handle.session_target_ids["session"] == id(handle.engine)
        prefix, cached = handle.session_caches["session"]
        original_cache = cached.clone()
        continuation = handle.tokenizer.decode(prefix.tolist(), skip_special_tokens=False)
        np.testing.assert_array_equal(
            handle.tokenizer(continuation, return_tensors="np")["input_ids"][0], prefix
        )
        request.prompt = continuation
        expected = backend.generate(
            GenerationRequest(
                model_id="model",
                prompt=continuation,
                max_tokens=1,
                temperature=0,
            )
        )
        calls.clear()
        assert backend.generate(request).text == expected.text
        assert all(count == 1 for count in calls)  # Decode only; no repeated prefix forward.
        handle.session_caches["session"] = (prefix.copy(), original_cache)
        handle.session_contracts.pop("session")  # Real state, deliberately unknown provenance.
        calls.clear()
        assert backend.generate(request).text == expected.text
        assert calls[0] == len(prefix)
        assert handle.session_contracts["session"] == session_execution_contract(handle)
        evidence = backend.provider_execution_evidence("model", provider)
        assert evidence["executed_node_count"] > 0
    finally:
        backend.unload_model("model")


@pytest.mark.parametrize("extra", [{"adapter_name": "active"}, {"task_weights": {"task": 1}}])
def test_raw_onnx_cannot_silently_ignore_weight_selection(extra):
    """Unsupported direct backend requests fail before their opaque session runs."""
    from aether.core.exceptions import BackendError

    class NoExecution:
        def run(self, *_args):
            pytest.fail("unsupported weight selection reached base execution")

    backend = ONNXBackend()
    backend._models["model"] = NoExecution()
    backend._tokenizers["model"] = object()
    with pytest.raises(BackendError, match=r"requested controls|unknown vector"):
        backend.generate(GenerationRequest(model_id="model", prompt="hello", extra=extra))
