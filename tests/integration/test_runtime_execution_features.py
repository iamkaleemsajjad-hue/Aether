"""Public Runtime actually executes phase pools and persists resumable state."""

import threading
from dataclasses import fields

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.runtime.execution_features import ModelExecutionGate


@pytest.fixture
def executable_artifact(tmp_path):
    source = tmp_path / "source"
    _write_probe_checkpoint(source)
    config = CompilerConfig(
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "compiler-cache"),
    )
    for field in fields(config):
        if field.name.startswith("enable_"):
            setattr(config, field.name, field.name == "enable_precision_assignment")
    artifact = tmp_path / "model.aeg"
    Compiler(config).compile(str(source), output_path=artifact)
    return str(artifact)


def _config(tmp_path, **kwargs):
    return RuntimeConfig(
        backend_name="aether_cpu",
        execution_devices=["cpu"],
        enable_semantic_cache=False,
        speculative_decoding=False,
        model_cache_dir=str(tmp_path / "runtime-cache"),
        **kwargs,
    )


def test_public_runtime_separate_pools_match_completion_stream_and_batch(
    executable_artifact, tmp_path
):
    with Runtime(_config(tmp_path)) as ordinary:
        baseline = ordinary.generate(
            executable_artifact, "hello world", max_tokens=4, temperature=0.0
        )
    config = _config(
        tmp_path, disaggregate_prefill_decode=True, prefill_devices=["cpu"], decode_devices=["cpu"]
    )
    with Runtime(config) as runtime:
        actual = runtime.generate(executable_artifact, "hello world", max_tokens=4, temperature=0.0)
        assert actual.text == baseline.text
        stats = actual.metrics.extra["disaggregated"]
        assert stats["state_bytes_handed_off"] > 0
        assert stats["ttft_ms"] > 0
        assert stats["prefill_chunks"] >= 1
        assert (
            "".join(
                runtime.generate_stream(
                    executable_artifact, "hello world", max_tokens=4, temperature=0.0
                )
            )
            == baseline.text
        )
        results = runtime.generate_batch(
            executable_artifact, ["hello world", "hello world"], max_tokens=4, temperature=0.0
        )
        assert [result.text for result in results] == [baseline.text, baseline.text]
        assert all(
            result.metrics.extra["disaggregated"]["state_bytes_handed_off"] > 0
            for result in results
        )
        workers = list(runtime._execution_features._pools[executable_artifact][0]._threads)
    assert all(not worker.is_alive() for worker in workers)


def test_public_session_persist_restart_and_actual_decode_parity(executable_artifact, tmp_path):
    config = _config(tmp_path, kv_cache_path=str(tmp_path / "persistent-state"))
    runtime = Runtime(config)
    runtime.generate(
        executable_artifact,
        "hello world",
        max_tokens=2,
        temperature=0.0,
        aether_kv_session_id="session",
    )
    handle = runtime._loaded_models[executable_artifact]
    prefix_ids, state = handle.session_caches["session"]
    logits, _ = handle.engine.forward(np.asarray([1], dtype=np.int64), state.clone())
    saved = runtime.persist_session(executable_artifact, "session")
    assert saved["logical_length"] == len(prefix_ids)
    assert "session" not in handle.session_caches
    runtime.close()
    with Runtime(config) as restarted:
        restored = restarted.restore_session(executable_artifact, "session", saved["key"])
        assert restored["logical_length"] == saved["logical_length"]
        restored_handle = restarted._loaded_models[executable_artifact]
        resumed_ids, resumed_state = restored_handle.session_caches["session"]
        np.testing.assert_array_equal(resumed_ids, prefix_ids)
        actual, _ = restored_handle.engine.forward(np.asarray([1], dtype=np.int64), resumed_state)
        np.testing.assert_array_equal(actual, logits)


def test_failed_persistence_preserves_live_session(executable_artifact, tmp_path):
    with Runtime(_config(tmp_path)) as runtime:
        runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=1,
            temperature=0.0,
            aether_kv_session_id="live",
        )
        state = runtime._loaded_models[executable_artifact].session_caches["live"]
        with pytest.raises(ValueError, match="not configured"):
            runtime.persist_session(executable_artifact, "live")
        assert runtime._loaded_models[executable_artifact].session_caches["live"] is state


def test_runtime_precision_api_changes_execution_and_invalidates_sessions(
    executable_artifact, tmp_path
):
    with Runtime(_config(tmp_path, precision_max_slowdown=1000)) as runtime:
        runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=1,
            temperature=0.0,
            aether_kv_session_id="old",
        )
        handle = runtime._loaded_models[executable_artifact]
        # The tensor values are exactly representable, as real FP16-origin
        # checkpoints expanded into the FP32 reference executor are.
        from aether.runtime.execution_precision import NumpyPrecisionAdapter

        for binding in NumpyPrecisionAdapter(handle.engine).bindings():
            binding.set(binding.get().astype(np.float16).astype(np.float32))
        report = runtime.adjust_precision(executable_artifact, [[1, 2]])
        assert report["execution_changed"] and report["saved_bytes"] > 0
        assert not handle.session_caches
        assert handle.engine.weights.layers[0].q_proj.dtype == np.float16
        compact_response = runtime.generate(
            executable_artifact, "hello", max_tokens=1, temperature=0
        )
        assert compact_response.metrics.active_precision == "float32"
        assert compact_response.metrics.extra["precision_adjustment"]["storage_dtype"] == "float16"
        assert runtime.adjust_precision(executable_artifact, restore=True)["execution_changed"]
        assert handle.engine.weights.layers[0].q_proj.dtype == np.float32


def test_runtime_unload_discards_compact_backups_without_restoration(executable_artifact, tmp_path):
    from aether.runtime.execution_precision import NumpyPrecisionAdapter

    runtime = Runtime(_config(tmp_path, precision_max_slowdown=1000))
    runtime._load_model(executable_artifact)
    handle = runtime._loaded_models[executable_artifact]
    for binding in NumpyPrecisionAdapter(handle.engine).bindings():
        binding.set(binding.get().astype(np.float16).astype(np.float32))
    assert runtime.adjust_precision(executable_artifact, [[1, 2]])["execution_changed"]
    controller = runtime._execution_features._controllers[executable_artifact]
    for backup in controller.directory.glob("*.npy"):
        backup.write_bytes(b"corrupt")
    runtime.close()
    assert handle.engine.weights.layers[0].q_proj.dtype == np.float16
    assert not list(controller.directory.glob("*.npy"))
    assert not runtime._loaded_models and not runtime._loaded_backends


def test_failed_handoff_adapter_construction_joins_started_workers(
    executable_artifact, tmp_path, monkeypatch
):
    from aether.runtime.disaggregated_execution import make_cpu_adapter_factory

    with Runtime(_config(tmp_path, disaggregate_prefill_decode=True)) as runtime:
        runtime._load_model(executable_artifact)
        engine = runtime._loaded_models[executable_artifact].engine
        created_on = []
        workers = []
        creation_lock = threading.Lock()
        start_thread = threading.Thread.start

        def observed_start(thread):
            start_thread(thread)
            if thread.name.startswith("aether-"):
                workers.append(thread)

        monkeypatch.setattr(threading.Thread, "start", observed_start)

        def phase_factory(*, model_fingerprint):
            original = make_cpu_adapter_factory(
                engine, model_fingerprint=model_fingerprint, cache_kind="attention_kv_v1"
            )

            def construct():
                with creation_lock:
                    created_on.append(threading.current_thread())
                    failed = len(created_on) == 3
                if failed:
                    raise MemoryError("handoff adapter allocation failed")
                return original()

            return construct

        engine.create_phase_adapter_factory = phase_factory
        with pytest.raises(MemoryError, match="handoff adapter allocation"):
            runtime.generate(executable_artifact, "hello", max_tokens=1, temperature=0)
        assert len(workers) == 2 and all(not worker.is_alive() for worker in workers)
        assert not runtime._execution_features._pools


def test_precision_writer_waits_for_active_requests_but_readers_can_overlap():
    gate = ModelExecutionGate()
    readers_started = threading.Barrier(3)
    release_readers = threading.Event()
    writer_finished = threading.Event()

    def reader():
        with gate.read():
            readers_started.wait(timeout=2)
            assert release_readers.wait(2)

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for thread in threads:
        thread.start()
    readers_started.wait(timeout=2)

    def writer():
        with gate.write():
            writer_finished.set()

    pending = threading.Thread(target=writer)
    pending.start()
    assert not writer_finished.wait(0.05)
    release_readers.set()
    for thread in threads:
        thread.join(2)
    pending.join(2)
    assert writer_finished.is_set()


@pytest.mark.parametrize("budget", [0, 1, 5])
def test_public_explicit_draft_preserves_bounded_completion_stream_and_session(
    executable_artifact, tmp_path, budget
):
    with Runtime(_config(tmp_path)) as ordinary:
        expected = ordinary.generate(
            executable_artifact, "hello world", max_tokens=budget, temperature=0.0
        )
    config = _config(
        tmp_path,
        draft_model_id=executable_artifact,
        draft_backend_name="aether_cpu",
        draft_execution_devices=["cpu"],
    )
    with Runtime(config) as runtime:
        actual = runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=budget,
            temperature=0.0,
            aether_kv_session_id="spec",
        )
        assert actual.text == expected.text
        assert actual.usage["completion_tokens"] == budget
        assert 0 <= actual.metrics.spec_accept_rate <= 1
        assert (
            "".join(
                runtime.generate_stream(
                    executable_artifact, "hello world", max_tokens=budget, temperature=0.0
                )
            )
            == expected.text
        )
        handle = runtime._loaded_models[executable_artifact]
        prefix, cache = handle.session_caches["spec"]
        assert len(prefix) == cache.length
        continued_prompt = (
            handle.tokenizer.decode(prefix.tolist(), skip_special_tokens=False) + " hello"
        )
        continued = runtime.generate(
            executable_artifact,
            continued_prompt,
            max_tokens=budget,
            temperature=0.0,
            aether_kv_session_id="spec",
        )
        with Runtime(_config(tmp_path)) as ordinary:
            reference = ordinary.generate(
                executable_artifact, continued_prompt, max_tokens=budget, temperature=0.0
            )
        assert continued.text == reference.text
        # Changing draft policy retains the real target cache, rather than
        # passing a read-only composite wrapper to an ordinary engine.
        runtime.config.draft_model_id = None
        ordinary_result = runtime.generate(
            executable_artifact,
            continued_prompt,
            max_tokens=budget,
            temperature=0.0,
            aether_kv_session_id="spec",
        )
        assert ordinary_result.text == reference.text


def test_cli_runtime_json_executes_real_phase_workers(executable_artifact, tmp_path):
    import json

    from click.testing import CliRunner

    from aether.cli import cli

    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(_config(tmp_path, disaggregate_prefill_decode=True).to_dict()), encoding="utf-8"
    )
    result = CliRunner().invoke(
        cli,
        [
            "--runtime-config",
            str(path),
            "run",
            executable_artifact,
            "--prompt",
            "hello world",
            "--raw",
            "--max-tokens",
            "2",
            "--temperature",
            "0",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "disaggregated" in result.output and "state_bytes_handed_off" in result.output


def test_cli_runtime_json_rejects_unknown_controls(tmp_path):
    import json

    from click.testing import CliRunner

    from aether.cli import cli

    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"dynamic_precison": True}), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "--runtime-config",
            str(path),
            "run",
            "missing-model",
            "--prompt",
            "hello",
            "--non-interactive",
        ],
    )
    assert result.exit_code != 0
    assert "unknown runtime configuration fields" in result.output


@pytest.mark.parametrize(
    "settings",
    [
        {"disaggregate_prefill_decode": "false"},
        {"precision_allow_lossy": "false"},
        {"dynamic_precision": 1},
        {"disaggregate_prefill_workers": "2"},
        {"precision_cooldown_seconds": float("inf")},
        {"execution_devices": [{}]},
    ],
)
def test_cli_runtime_json_rejects_malformed_types_without_enabling_features(tmp_path, settings):
    import json

    from click.testing import CliRunner

    from aether.cli import cli

    path = tmp_path / "invalid-types.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "--runtime-config",
            str(path),
            "run",
            "missing-model",
            "--prompt",
            "hello",
            "--non-interactive",
        ],
    )
    assert result.exit_code != 0
    assert "must" in result.output
    assert "Traceback" not in result.output


def test_causal_speculation_rejects_encoder_semantics_before_forward(executable_artifact, tmp_path):
    from aether.runtime.execution_features import validate_draft_tokenizers

    with Runtime(_config(tmp_path)) as runtime:
        runtime._load_model(executable_artifact)
        handle = runtime._loaded_models[executable_artifact]
        handle.manifest["architecture"]["is_encoder_decoder"] = True
        with pytest.raises(ValueError, match="causal decoder"):
            validate_draft_tokenizers(handle, handle)
