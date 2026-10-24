"""Actual compiled CPU requests, admission races and owned stream shutdown."""

from concurrent.futures import ThreadPoolExecutor
import shutil
import threading

import pytest

from aether import Runtime
from tests.integration.test_runtime_execution_features import executable_artifact, _config


def test_close_rejects_requests_at_the_actual_backend_unload_boundary(
    executable_artifact, tmp_path
):
    runtime = Runtime(_config(tmp_path, disaggregate_prefill_decode=True))
    runtime.generate(executable_artifact, "hello world", max_tokens=1, temperature=0)
    pools = list(runtime._execution_features._pools.values())
    backend = runtime._loaded_backends[executable_artifact]
    unload = backend.unload_model
    entered, release = threading.Event(), threading.Event()
    errors = []

    def blocked(model):
        entered.set()
        assert release.wait(5)
        unload(model)

    backend.unload_model = blocked

    def close():
        try:
            runtime.close(timeout=5)
        except BaseException as error:
            errors.append(error)

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="closing or closed"):
            runtime.generate(executable_artifact, "hello world", max_tokens=1)
        assert not runtime._execution_features._pools
    finally:
        release.set()
        closer.join(5)
        backend.unload_model = unload
    assert not closer.is_alive() and not errors
    assert not runtime._loaded_models
    assert all(not worker.is_alive() for entry in pools for worker in entry[0]._threads)
    runtime.close()  # Idempotent only after actual cleanup succeeded.


def test_paused_actual_stream_is_closed_without_waiting_for_another_next(
    executable_artifact, tmp_path
):
    runtime = Runtime(_config(tmp_path))
    stream = runtime.generate_stream(
        executable_artifact, "hello world", max_tokens=10, temperature=0
    )
    assert next(stream)
    assert runtime._execution_features.gate(executable_artifact)._readers == 1
    runtime.close(timeout=1)
    assert runtime._execution_features.gate(executable_artifact)._readers == 0
    assert list(stream) == []


def test_unstarted_stream_is_owned_and_closed_without_loading_a_model(tmp_path):
    runtime = Runtime(_config(tmp_path))
    stream = runtime.generate_stream("not-loaded", "hello world")
    runtime.close(timeout=1)
    assert list(stream) == []
    assert not runtime._loaded_models


def test_running_actual_request_times_out_honestly_then_shutdown_can_be_retried(
    executable_artifact, tmp_path
):
    runtime = Runtime(_config(tmp_path))
    backend = runtime._load_model(executable_artifact)
    generate = backend.generate
    entered, release = threading.Event(), threading.Event()

    def blocked(request):
        entered.set()
        assert release.wait(5)
        return generate(request)

    backend.generate = blocked
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            runtime.generate, executable_artifact, "hello world", max_tokens=1, temperature=0
        )
        try:
            assert entered.wait(5)
            with pytest.raises(TimeoutError, match="shutdown has not completed"):
                runtime.close(timeout=0.01)
            assert runtime._loaded_models
            with pytest.raises(RuntimeError, match="closing or closed"):
                runtime._load_model(executable_artifact)
        finally:
            release.set()
        assert pending.result(timeout=5).usage["completion_tokens"] == 1
    backend.generate = generate
    runtime.close(timeout=5)
    assert not runtime._loaded_models


def test_admission_preserves_concurrent_actual_readers(executable_artifact, tmp_path):
    with Runtime(_config(tmp_path)) as runtime:
        backend = runtime._load_model(executable_artifact)
        generate = backend.generate
        barrier = threading.Barrier(2)

        def concurrent(request):
            barrier.wait(timeout=5)
            return generate(request)

        backend.generate = concurrent
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                requests = [
                    executor.submit(
                        runtime.generate,
                        executable_artifact,
                        "hello world",
                        max_tokens=2,
                        temperature=0,
                        aether_kv_session_id=session,
                    )
                    for session in ("left", "right")
                ]
                responses = [request.result(timeout=10) for request in requests]
            assert responses[0].text == responses[1].text
            assert all(response.usage["completion_tokens"] == 2 for response in responses)
        finally:
            backend.generate = generate


@pytest.mark.parametrize(
    "call",
    [
        lambda runtime, model: runtime.generate(model, "hello"),
        lambda runtime, model: runtime.generate_batch(model, ["hello", "world"]),
        lambda runtime, model: runtime.generate_stream(model, "hello"),
        lambda runtime, model: runtime._load_model(model),
        lambda runtime, model: runtime.adjust_precision(model, [[1, 2]]),
        lambda runtime, model: runtime.persist_session(model, "session"),
        lambda runtime, model: runtime.restore_session(model, "session", "key"),
        lambda runtime, model: runtime.remove(model),
        lambda runtime, model: runtime.benchmark(model),
        lambda runtime, model: runtime.embed(model, ["hello"]),
    ],
)
def test_closed_runtime_rejects_creation_and_model_operations(tmp_path, call):
    runtime = Runtime(_config(tmp_path))
    runtime.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        call(runtime, "unused-model")


def test_cached_model_remove_closes_its_paused_stream_before_deleting(
    executable_artifact, tmp_path
):
    cache_root = tmp_path / "cached"
    from aether.utils.file_io import aeg_cache_candidates

    config = _config(cache_root)
    destination = aeg_cache_candidates("audit-model", config.model_cache_dir)[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(executable_artifact, destination)
    with Runtime(config) as runtime:
        stream = runtime.generate_stream("audit-model", "hello world", max_tokens=10, temperature=0)
        assert next(stream)
        runtime.remove("audit-model")
        assert list(stream) == []
        assert not destination.exists()
        assert "audit-model" not in runtime._loaded_models


def test_close_from_inside_an_operation_fails_without_self_deadlock(executable_artifact, tmp_path):
    with Runtime(_config(tmp_path)) as runtime:
        backend = runtime._load_model(executable_artifact)
        generate = backend.generate

        def callback(request):
            with pytest.raises(RuntimeError, match="inside an active operation"):
                runtime.close(timeout=0.01)
            return generate(request)

        backend.generate = callback
        try:
            assert (
                runtime.generate(executable_artifact, "hello", max_tokens=1).usage[
                    "completion_tokens"
                ]
                == 1
            )
        finally:
            backend.generate = generate


def test_same_session_actual_requests_serialize_without_mutable_cache_corruption(
    executable_artifact, tmp_path
):
    with Runtime(_config(tmp_path)) as runtime:
        runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=1,
            temperature=0,
            aether_kv_session_id="same",
        )
        handle = runtime._loaded_models[executable_artifact]
        prefix, _ = handle.session_caches["same"]
        prompt = handle.tokenizer.decode(prefix.tolist(), skip_special_tokens=False)
        backend = runtime._loaded_backends[executable_artifact]
        generate = backend.generate
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def controlled(request):
            calls.append(request)
            if len(calls) == 1:
                entered.set()
                assert release.wait(5)
            else:
                second.set()
            return generate(request)

        backend.generate = controlled
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(
                    runtime.generate,
                    executable_artifact,
                    prompt,
                    max_tokens=1,
                    temperature=0,
                    aether_kv_session_id="same",
                )
                assert entered.wait(5)
                later = executor.submit(
                    runtime.generate,
                    executable_artifact,
                    prompt,
                    max_tokens=1,
                    temperature=0,
                    aether_kv_session_id="same",
                )
                try:
                    assert not second.wait(0.05)
                finally:
                    release.set()
                responses = [call.result(timeout=5) for call in (first, later)]
            assert responses[0].text == responses[1].text
            saved_ids, saved = handle.session_caches["same"]
            assert len(saved_ids) == saved.length
            assert not runtime._lifecycle._sessions
        finally:
            release.set()
            backend.generate = generate


def test_session_stream_can_be_closed_on_another_thread_and_releases_lock(
    executable_artifact, tmp_path
):
    runtime = Runtime(_config(tmp_path))
    stream = runtime.generate_stream(
        executable_artifact,
        "hello world",
        max_tokens=5,
        temperature=0,
        aether_kv_session_id="stream",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(next, stream).result(timeout=5)
    assert runtime._lifecycle._sessions
    runtime.close(timeout=2)
    assert not runtime._lifecycle._sessions
    assert list(stream) == []


def test_paused_session_reentry_is_explicit_instead_of_self_deadlocking(
    executable_artifact, tmp_path
):
    with Runtime(_config(tmp_path)) as runtime:
        stream = runtime.generate_stream(
            executable_artifact,
            "hello world",
            max_tokens=5,
            temperature=0,
            aether_kv_session_id="stream",
        )
        assert next(stream)
        try:
            with pytest.raises(RuntimeError, match="active operation or stream"):
                runtime.generate(
                    executable_artifact, "hello world", max_tokens=1, aether_kv_session_id="stream"
                )
        finally:
            stream.close()
        assert (
            runtime.generate(
                executable_artifact, "hello world", max_tokens=1, aether_kv_session_id="stream"
            ).usage["completion_tokens"]
            == 1
        )


def test_aborted_actual_stream_does_not_leave_a_partially_mutated_session(
    executable_artifact, tmp_path
):
    with Runtime(_config(tmp_path)) as runtime:
        runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=1,
            temperature=0,
            aether_kv_session_id="stream",
        )
        handle = runtime._loaded_models[executable_artifact]
        prefix, _ = handle.session_caches["stream"]
        prompt = handle.tokenizer.decode(prefix.tolist(), skip_special_tokens=False)
        stream = runtime.generate_stream(
            executable_artifact, prompt, max_tokens=5, temperature=0, aether_kv_session_id="stream"
        )
        assert next(stream)
        assert next(stream)
        stream.close()
        cached = handle.session_caches.get("stream")
        assert cached is None or len(cached[0]) == cached[1].length


def test_stateful_session_generation_cannot_be_satisfied_by_response_cache(
    executable_artifact, tmp_path
):
    config = _config(tmp_path)
    config.enable_semantic_cache = True
    with Runtime(config) as runtime:
        runtime.generate(executable_artifact, "hello world", max_tokens=1, temperature=0)
        runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=1,
            temperature=0,
            aether_kv_session_id="live",
        )
        handle = runtime._loaded_models[executable_artifact]
        assert "live" in handle.session_caches


def test_same_session_batch_keeps_real_cache_and_token_prefix_consistent(
    executable_artifact, tmp_path
):
    with Runtime(_config(tmp_path)) as runtime:
        results = runtime.generate_batch(
            executable_artifact,
            ["hello world", "hello world"],
            max_tokens=1,
            temperature=0,
            aether_kv_session_id="batch",
        )
        assert results[0].text == results[1].text
        prefix, cache = runtime._loaded_models[executable_artifact].session_caches["batch"]
        assert len(prefix) == cache.length
        assert not runtime._lifecycle._sessions


def test_session_persistence_waits_for_execution_and_automatic_restore_is_reentrant(
    executable_artifact, tmp_path
):
    config = _config(tmp_path, kv_cache_path=str(tmp_path / "disk"))
    with Runtime(config) as runtime:
        runtime.generate(
            executable_artifact,
            "hello world",
            max_tokens=1,
            temperature=0,
            aether_kv_session_id="same",
        )
        backend = runtime._loaded_backends[executable_artifact]
        generate = backend.generate
        entered, release = threading.Event(), threading.Event()

        def controlled(request):
            entered.set()
            assert release.wait(5)
            return generate(request)

        backend.generate = controlled
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                request = executor.submit(
                    runtime.generate,
                    executable_artifact,
                    "hello world",
                    max_tokens=1,
                    temperature=0,
                    aether_kv_session_id="same",
                )
                assert entered.wait(5)
                persisted = executor.submit(runtime.persist_session, executable_artifact, "same")
                try:
                    assert not persisted.done()
                finally:
                    release.set()
                assert request.result(timeout=5).usage["completion_tokens"] == 1
                snapshot = persisted.result(timeout=5)
            assert snapshot["logical_length"] > 0
        finally:
            release.set()
            backend.generate = generate
        # Request preparation restores the stored state while already owning
        # the same session's lease; it must not acquire its own Lock twice.
        assert (
            runtime.generate(
                executable_artifact,
                "hello world",
                max_tokens=1,
                temperature=0,
                aether_kv_session_id="same",
            ).usage["completion_tokens"]
            == 1
        )
        assert not runtime._lifecycle._sessions


def test_process_exit_closes_a_paused_actual_stream(executable_artifact, tmp_path):
    import subprocess
    import sys

    code = (
        "from aether import Runtime,RuntimeConfig; "
        "rt=Runtime(RuntimeConfig(allow_uncertified=True,backend_name='aether_cpu',"
        f"execution_devices=['cpu'],model_cache_dir={str(tmp_path / 'child-cache')!r},"
        "enable_semantic_cache=False,speculative_decoding=False)); "
        f"stream=rt.generate_stream({executable_artifact!r},'hello world',max_tokens=5,temperature=0); "
        "assert next(stream); print('actual stream paused')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stderr
    assert "actual stream paused" in completed.stdout


def test_remove_releases_paused_session_before_waiting_for_its_admitted_waiter(
    executable_artifact, tmp_path
):
    import time
    from aether.utils.file_io import aeg_cache_candidates

    config = _config(tmp_path / "cache")
    for alias in ("remove-me", "keep-me"):
        destination = aeg_cache_candidates(alias, config.model_cache_dir)[0]
        shutil.copytree(executable_artifact, destination)
    with Runtime(config) as runtime:
        target = runtime.generate_stream(
            "remove-me", "hello world", max_tokens=5, temperature=0, aether_kv_session_id="same"
        )
        other = runtime.generate_stream("keep-me", "hello world", max_tokens=5, temperature=0)
        assert next(target) and next(other)
        with ThreadPoolExecutor(max_workers=2) as executor:
            waiting = executor.submit(
                runtime.generate,
                "remove-me",
                "hello world",
                max_tokens=1,
                temperature=0,
                aether_kv_session_id="same",
            )
            deadline = time.monotonic() + 5
            while runtime._lifecycle._active == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert runtime._lifecycle._active == 1
            removed = executor.submit(runtime.remove, "remove-me")
            try:
                removed.result(timeout=5)
            finally:
                target.close()
            assert waiting.result(timeout=5).usage["completion_tokens"] == 1
        assert list(target) == []
        assert "remove-me" not in runtime._loaded_models
        assert next(other)  # Unrelated model streams remain open.
        other.close()


def test_runtime_closes_actual_owned_mmap_and_default_daemon(executable_artifact, tmp_path):
    config = _config(tmp_path, cxl_pool_size_gb=0.002)
    runtime = Runtime(config)
    runtime._load_model(executable_artifact)
    pool = runtime._cxl_pool
    assert pool is not None and pool._backend.mode == "emulated"
    mapping, file, worker = pool._backend._mmap, pool._backend._file, pool._defrag_thread
    assert mapping is not None and not mapping.closed and worker.is_alive()
    runtime.close()
    assert mapping.closed and file.closed and not worker.is_alive()
    assert runtime._cxl_pool is None
    assert not runtime._aeg_packages and not runtime._peagle_engines


def test_runtime_disconnects_actual_owned_stdio_process(tmp_path):
    import sys
    from aether.runtime.r6_mcp_integration import MCPIntegrationLayer
    from tests.unit.test_mcp_transport_contracts import SERVER

    script = tmp_path / "local server.py"
    script.write_text(SERVER, encoding="utf-8")
    runtime = Runtime(_config(tmp_path))
    layer = runtime.mcp_layer = MCPIntegrationLayer(timeout_s=3)
    try:
        assert layer.add_server(
            "local", transport="stdio", command=[sys.executable, "-u", str(script)]
        )
        process = layer._clients["local"]._process
        assert process.poll() is None
        runtime.close()
        assert process.poll() is not None
        assert not layer.connected_servers and runtime.mcp_layer is None
    finally:
        layer.disconnect_all()
        runtime.close()


def test_failed_owned_resource_cleanup_keeps_owner_for_actual_retry(tmp_path):
    runtime = Runtime(_config(tmp_path))

    class Owner:
        attempts = 0

        def disconnect_all(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("disconnect failed")

    owner = runtime.mcp_layer = Owner()
    with pytest.raises(RuntimeError, match="resource cleanup failed.*disconnect failed"):
        runtime.close()
    assert runtime.mcp_layer is owner
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime._load_model("unused")
    runtime.close()
    assert owner.attempts == 2 and runtime.mcp_layer is None


def test_close_joins_an_owned_compile_job_or_reports_its_running_state(tmp_path):
    runtime = Runtime(_config(tmp_path))
    entered, release = threading.Event(), threading.Event()

    def work():
        entered.set()
        assert release.wait(5)

    worker = threading.Thread(target=work)
    runtime._compile_jobs["fixture"] = {"_thread": worker}
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="compiler job is still executing"):
            runtime.close(timeout=0.01)
    finally:
        release.set()
        worker.join(5)
    runtime.close()
    assert not worker.is_alive()
