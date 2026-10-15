"""Actual compiled task weights must govern responses, streams and request caches."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import shutil
import threading

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.core.aeg_format import load_aeg_package
from aether.core.exceptions import RuntimeError as AetherRuntimeError


def config(root, **kwargs):
    return RuntimeConfig(
        backend_name="aether_cpu", execution_devices=["cpu"], hf_offline=True,
        model_cache_dir=str(root / "models"), dynamic_precision=False,
        speculative_decoding=False, **kwargs,
    )


@pytest.fixture(scope="module")
def task_artifact(tmp_path_factory):
    from safetensors.numpy import save_file

    root = tmp_path_factory.mktemp("request-weight-contract")
    source = root / "checkpoint"
    _write_probe_checkpoint(source)
    compiler_config = CompilerConfig(
        targets=["cpu_generic"], precision="FP32", optimization_level=0,
        cache_dir=str(root / "compile-cache"),
    )
    for option in fields(compiler_config):
        if option.name.startswith("enable_"):
            setattr(compiler_config, option.name, option.name == "enable_precision_assignment")
    package = Compiler(compiler_config).compile(str(source), output_path=root / "base.aeg")
    with Runtime(config(root, enable_semantic_cache=False)) as runtime:
        runtime._load_model(str(package.root))
        handle = runtime._loaded_models[str(package.root)]
        prompt = "hello world"
        ids = handle.tokenizer(prompt, return_tensors="np")["input_ids"][0]
        logits, cache = handle.engine.forward(np.asarray(ids, dtype=np.int64))
        baseline_token = int(np.argmax(logits[-1]))
        target_token = 3 if baseline_token != 3 else 4
        delta = np.zeros_like(handle.engine.weights.lm_head)
        delta[target_token] = cache.last_hidden * np.float32(1000)
        head = package.weight_store().dequantize_all()["lm_head"]
        boost = root / "boost.safetensors"
        neutral = root / "neutral.safetensors"
        save_file({"lm_head": head + delta}, str(boost))
        save_file({"lm_head": head}, str(neutral))
        # A zero boost coefficient and a unit zero-delta coefficient retain
        # base weights while persisting both sources for request selection.
        report = runtime.merge(str(package.root), [
            {"name": "boost", "path": str(boost), "coefficient": 0.0},
            {"name": "neutral", "path": str(neutral), "coefficient": 1.0},
        ])
        expected = handle.tokenizer.decode([target_token], skip_special_tokens=True)
    merged = load_aeg_package(report["output_model"])
    merged.verify_integrity()
    return str(merged.root), prompt, expected


def test_real_task_defaults_bypass_response_cache_without_poisoning_base_entries(task_artifact, tmp_path):
    model, prompt, expected = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=True)) as runtime:
        ordinary = runtime.generate(model, prompt, max_tokens=1, temperature=0)
        cached = runtime.generate(model, prompt, max_tokens=1, temperature=0)
        assert cached.text == ordinary.text and cached.metrics.backend_name == "semantic_cache"
        runtime.set_task_weights({"boost": 1})
        modified = runtime.generate(model, prompt, max_tokens=1, temperature=0)
        assert modified.text == expected and modified.text != ordinary.text
        assert modified.metrics.backend_name != "semantic_cache"
        assert modified.metrics.extra["task_reweighting"]["vectors"] == ["boost"]
        base_override = runtime.generate(model, prompt, max_tokens=1, temperature=0, task_weights=None)
        assert base_override.text == ordinary.text
        assert base_override.metrics.backend_name != "semantic_cache"
        assert runtime._semantic_cache.stats()["exact_cache_size"] == 1


def test_configured_task_weights_apply_to_actual_streaming(task_artifact, tmp_path):
    model, prompt, expected = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=False)) as runtime:
        runtime.set_task_weights({"neutral": 1})
        runtime.set_task_weights(model, boost=1)
        actual = "".join(runtime.generate_stream(model, prompt, max_tokens=1, temperature=0))
        assert actual == expected
        base = "".join(runtime.generate_stream(
            model, prompt, max_tokens=1, temperature=0, task_weights=None,
        ))
        assert base != expected


def test_real_task_mapping_keeps_reserved_labels_as_data(task_artifact, tmp_path):
    model, prompt, expected = task_artifact
    path = tmp_path / "reserved.aeg"
    shutil.copytree(model, path)
    package = load_aeg_package(path)
    for vector in package.metadata["task_vectors"]["vectors"]:
        vector["name"] = "weights" if vector["name"] == "boost" else "self"
    package.save()
    package.verify_integrity()
    with Runtime(config(tmp_path, enable_semantic_cache=False)) as runtime:
        assert runtime.set_model_task_weights(str(path), {"weights": 1., "self": 0.}) == {"weights": 1.}
        response = runtime.generate(str(path), prompt, max_tokens=1, temperature=0)
        assert response.text == expected
        assert response.metrics.extra["task_reweighting"]["vectors"] == ["weights"]


def test_concurrent_setter_does_not_change_admitted_real_request(task_artifact, tmp_path, monkeypatch):
    model, prompt, expected = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=False)) as runtime:
        runtime.set_task_weights({"boost": 1})
        execute = runtime._generate_with_features
        entered, release = threading.Event(), threading.Event()

        def paused(backend, request):
            entered.set()
            assert release.wait(5)
            return execute(backend, request)

        monkeypatch.setattr(runtime, "_generate_with_features", paused)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(runtime.generate, model, prompt, max_tokens=1, temperature=0)
            assert entered.wait(5)
            try:
                runtime.set_task_weights({"neutral": 1})
            finally:
                release.set()
            assert pending.result(5).text == expected
        monkeypatch.setattr(runtime, "_generate_with_features", execute)
        assert runtime.generate(model, prompt, max_tokens=1, temperature=0).text != expected


@pytest.mark.parametrize("extras", [
    {"stop": ["unmatched-stop"]}, {"apply_chat_template": False}, {"seed": 123},
])
def test_output_or_backend_controls_are_not_served_from_ordinary_response_cache(task_artifact, tmp_path, extras):
    model, prompt, _ = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=True)) as runtime:
        runtime.generate(model, prompt, max_tokens=1, temperature=0)
        assert runtime.generate(model, prompt, max_tokens=1, temperature=0).metrics.backend_name == "semantic_cache"
        requested = runtime.generate(model, prompt, max_tokens=1, temperature=0, **extras)
        assert requested.metrics.backend_name != "semantic_cache"
        assert runtime._semantic_cache.stats()["exact_cache_size"] == 1


@pytest.mark.parametrize("extras", [
    {"adapter_id": "not-batchable"}, {"task_weights": {"boost": 1}},
    {"ttt_engine": object()}, {"grammar_session": object()},
    {"multi_agent_kv_coordinator": object()}, {"peagle_engine": object()},
    {"spec_draft_engine": object()},
])
def test_true_batch_rejects_hooks_before_executing_base_weights(task_artifact, tmp_path, monkeypatch, extras):
    model, prompt, _ = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=False)) as runtime:
        backend = runtime._load_model(model)
        monkeypatch.setattr(backend, "generate_batch", lambda requests: pytest.fail("unsupported batch executed"))
        with pytest.raises(AetherRuntimeError, match="active request hooks"):
            runtime.generate_batch(model, [prompt, prompt], max_tokens=1, temperature=0, **extras)


def test_configured_task_defaults_are_not_silently_dropped_for_batches(task_artifact, tmp_path):
    model, prompt, _ = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=False)) as runtime:
        runtime.set_task_weights({"boost": 1})
        with pytest.raises(AetherRuntimeError, match="task_weights"):
            runtime.generate_batch(model, [prompt, prompt], max_tokens=1, temperature=0)


def test_ordinary_true_batch_remains_executable_and_labels_its_scope(task_artifact, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    # This regression measures request routing and actual batched arithmetic,
    # not the optional projection microprobe policy covered by its own suite.
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    model, prompt, _ = task_artifact
    with Runtime(config(tmp_path, enable_semantic_cache=False)) as runtime:
        ordinary = runtime.generate(model, prompt, max_tokens=1, temperature=0)
        actual = runtime.generate_batch(model, [prompt, prompt], max_tokens=1, temperature=0)
        assert [item.text for item in actual] == [ordinary.text, ordinary.text]
        assert all(item.metrics.extra["execution_mode"] == "batched_forward" for item in actual)
        assert all(item.metrics.extra["batch_size"] == 2 for item in actual)
