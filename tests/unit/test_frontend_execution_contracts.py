"""Vendor API contract doubles; these never certify physical hardware."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.backends.base import GenerationRequest
from aether.backends.llamacpp_backend import LlamaCppBackend
from aether.backends.mlx_backend import MLXBackend
from aether.backends.vllm_backend import vLLMBackend
from aether.core.exceptions import BackendError


@pytest.mark.parametrize("backend", [LlamaCppBackend, MLXBackend, vLLMBackend])
@pytest.mark.parametrize("model", ["decoder.aeg", "encoder.onnx"])
def test_source_frontends_reject_incompatible_artifacts(backend, model):
    with pytest.raises(BackendError):
        backend().load_model(model)


def test_cpu_llama_build_does_not_satisfy_gpu_selection(monkeypatch):
    backend = LlamaCppBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_gpu_devices", lambda: [])
    assert backend.available_for_target("cpu_generic")
    assert not backend.available_for_target("cuda_sm89")
    assert not backend.available_for_target("metal_m1")


def test_llama_automatic_selection_carries_a_real_device_ordinal(monkeypatch):
    backend = LlamaCppBackend()
    monkeypatch.setattr(
        backend, "_gpu_devices", lambda: [{"name": "CUDA1", "backend": "CUDA", "main_gpu": 1}]
    )
    assert backend.automatic_execution_devices("cuda_sm89") == ["cuda:1"]
    assert backend.automatic_execution_devices("cpu_generic") == ["cpu"]
    assert backend.automatic_execution_devices("metal") == ["mps"]


def test_llama_gpu_offload_flag_without_device_api_is_insufficient(monkeypatch):
    module = SimpleNamespace(
        llama_supports_gpu_offload=lambda: True, llama_cpp=SimpleNamespace(_lib=SimpleNamespace())
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    assert not LlamaCppBackend().available_for_target("cuda_sm89")


@pytest.mark.parametrize(
    "device,name,backend_name",
    [("cuda:0", "CUDA0", "CUDA"), ("cuda:1", "HIP1", "ROCm"), ("mps", "Metal", "Metal")],
)
def test_llama_selected_gpu_gets_layers_and_correct_single_device_options(
    monkeypatch, device, name, backend_name
):
    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))
    backend = LlamaCppBackend()
    monkeypatch.setattr(
        backend, "_gpu_devices", lambda: [{"name": name, "backend": backend_name, "main_gpu": 2}]
    )
    assert backend.load_model("any.gguf", execution_devices=[device]) is model
    assert constructor.call_args.kwargs["n_gpu_layers"] == -1
    assert constructor.call_args.kwargs["main_gpu"] == 2
    assert constructor.call_args.kwargs["split_mode"] == 0


@pytest.mark.parametrize(
    "mesh,layers", [(["cpu"], 4), (["cuda:0"], 0), (["cpu", "cuda:0"], -1), (["cuda:7"], -1)]
)
def test_llama_conflicting_or_unavailable_mesh_fails(monkeypatch, mesh, layers):
    backend = LlamaCppBackend()
    monkeypatch.setattr(
        backend, "_gpu_devices", lambda: [{"name": "CUDA0", "backend": "CUDA", "main_gpu": 0}]
    )
    with pytest.raises(BackendError):
        backend.load_model("any.gguf", execution_devices=mesh, n_gpu_layers=layers)


def test_llama_only_server_binary_can_satisfy_http_backend(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "llama-cli" if name == "llama-cli" else None)
    assert LlamaCppBackend()._find_llama_binary() is None


def test_llama_chat_uses_checkpoint_chat_handler_and_actual_usage():
    model = Mock()
    model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1, "prompt_tokens": 7},
    }
    backend = LlamaCppBackend()
    backend._models["any.gguf"] = model
    request = GenerationRequest(
        model_id="any.gguf", messages=[{"role": "user", "content": "hi"}], top_k=0
    )
    result = backend.generate(request)
    assert result.text == "ok" and result.completion_tokens == 1 and result.prompt_tokens == 7
    assert model.create_chat_completion.call_args.kwargs["top_k"] == 0
    assert "ttft_ms" not in result.metrics
    model.assert_not_called()


def test_llama_stream_uses_real_events_and_unloads_model():
    model = Mock(return_value=iter([{"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}]))
    backend = LlamaCppBackend()
    backend._models["any.gguf"] = model
    assert list(backend.generate_stream(GenerationRequest(model_id="any.gguf", prompt="x"))) == [
        "a",
        "b",
    ]
    assert model.call_args.kwargs["stream"] is True
    backend.unload_model("any.gguf")
    model.close.assert_called_once()
    assert not backend._models


def mlx_modules(monkeypatch, texts=("a", "b")):
    events = [
        SimpleNamespace(
            text=text,
            prompt_tokens=3,
            generation_tokens=index + 1,
            finish_reason="stop" if index == len(texts) - 1 else None,
        )
        for index, text in enumerate(texts)
    ]
    stream = Mock(side_effect=lambda *args, **kwargs: iter(events))
    sampler = Mock(return_value="sampler")
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "checkpoint prompt"
    module = SimpleNamespace(load=Mock(return_value=(object(), tokenizer)), stream_generate=stream)
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=sampler))
    core = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    return module, stream, sampler, tokenizer


def test_mlx_readiness_requires_generation_frontend(monkeypatch):
    mlx_modules(monkeypatch)
    assert MLXBackend().is_available()
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    assert not MLXBackend().is_available()


def test_mlx_actual_event_counts_sampler_and_checkpoint_chat(monkeypatch):
    _, stream, sampler, tokenizer = mlx_modules(monkeypatch)
    backend = MLXBackend()
    result = backend.generate(
        GenerationRequest(
            model_id="org/any-family",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=99,
            temperature=0.2,
            top_k=8,
            top_p=0.7,
        )
    )
    assert result.text == "ab" and result.completion_tokens == 2 and result.prompt_tokens == 3
    assert result.finish_reason == "stop"
    sampler.assert_called_once_with(temp=0.2, top_p=0.7, top_k=8)
    assert stream.call_args.kwargs["sampler"] == "sampler"
    assert stream.call_args.kwargs["prompt"] == "checkpoint prompt"
    tokenizer.apply_chat_template.assert_called_once()


def test_mlx_stream_stop_spanning_events_is_not_leaked(monkeypatch):
    mlx_modules(monkeypatch, ("hello<", "STOP", ">ignored"))
    request = GenerationRequest(model_id="org/source", prompt="hi", stop=["<STOP>"])
    backend = MLXBackend()
    assert "".join(backend.generate_stream(request)) == "hello"
    result = backend.generate(request)
    assert result.text == "hello" and result.finish_reason == "stop"


def test_mlx_incomplete_stop_prefix_is_flushed(monkeypatch):
    mlx_modules(monkeypatch, ("hi<ST",))
    assert (
        "".join(
            MLXBackend().generate_stream(
                GenerationRequest(model_id="org/source", prompt="x", stop=["<STOP>"])
            )
        )
        == "hi<ST"
    )


def test_mlx_cpu_mesh_is_not_ignored(monkeypatch):
    module, *_ = mlx_modules(monkeypatch)
    with pytest.raises(BackendError):
        MLXBackend().load_model("org/source", execution_devices=["cpu"])
    module.load.assert_not_called()


def vllm_module(monkeypatch):
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "checkpoint prompt"
    engine = Mock()
    engine.get_tokenizer.return_value = tokenizer
    engine.generate.return_value = [
        SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[SimpleNamespace(text="answer", token_ids=[4, 5], finish_reason="stop")],
        )
    ]
    module = SimpleNamespace(
        LLM=Mock(return_value=engine), SamplingParams=Mock(return_value="sampling")
    )
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    )
    return module, engine


def test_vllm_runtime_options_do_not_leak_into_vendor_engine(monkeypatch):
    module, engine = vllm_module(monkeypatch)
    backend = vLLMBackend()
    assert (
        backend.load_model(
            "org/any-family",
            execution_devices=["cuda:0", "cuda:1"],
            offline=False,
            download_timeout_s=7,
        )
        is engine
    )
    arguments = module.LLM.call_args.kwargs
    assert arguments["dtype"] == "auto" and arguments["tensor_parallel_size"] == 2
    assert not set(arguments).intersection({"offline", "execution_devices", "download_timeout_s"})
    with pytest.raises(BackendError, match="another execution mesh"):
        backend.load_model("org/any-family", execution_devices=["cpu"])


@pytest.mark.parametrize(
    "devices", [["cpu"], ["cuda:1"], ["cuda:0", "cuda:2"], ["cuda:0", "cuda:1", "cuda:2"]]
)
def test_vllm_unsupported_mesh_is_not_ignored(monkeypatch, devices):
    module, _ = vllm_module(monkeypatch)
    with pytest.raises(BackendError):
        vLLMBackend().load_model("org/source", execution_devices=devices)
    module.LLM.assert_not_called()


def test_vllm_sampling_usage_chat_and_buffered_stream(monkeypatch):
    module, engine = vllm_module(monkeypatch)
    backend = vLLMBackend()
    request = GenerationRequest(
        model_id="org/source", messages=[{"role": "user", "content": "x"}], top_k=0
    )
    result = backend.generate(request)
    assert result.text == "answer" and result.completion_tokens == 2 and result.prompt_tokens == 3
    assert module.SamplingParams.call_args.kwargs["top_k"] == -1
    assert result.metrics["throughput_tps"] > 0
    assert result.metrics["measurement_source"] == "token_ids_and_wall_clock"
    assert engine.generate.call_args.args[0] == ["checkpoint prompt"]
    assert list(backend.generate_stream(request)) == ["answer"]
