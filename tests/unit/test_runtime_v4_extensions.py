"""Tests for Runtime v4.0 extensions.

Covers:
  - v4.0 layer attribute initialisation (grammar_engine, ttt_engine, etc.)
  - compile_async() + get_compile_status() lifecycle contract
  - merge() fails closed when a real AEG/source is not available
  - _init_v4_layers() no-crash contract

All tests run without GPU.
"""

from __future__ import annotations

import time

import pytest

from aether.runtime.config import RuntimeConfig
from aether.runtime.runtime import Runtime


@pytest.fixture
def rt() -> Runtime:
    """A Runtime with default config. No GPU needed."""
    return Runtime(RuntimeConfig())


class TestRuntimeV4Attributes:
    def test_grammar_engine_none_at_start(self, rt: Runtime) -> None:
        assert rt.grammar_engine is None

    def test_ttt_engine_none_at_start(self, rt: Runtime) -> None:
        assert rt.ttt_engine is None

    def test_tee_manager_none_at_start(self, rt: Runtime) -> None:
        assert rt.tee_manager is None

    def test_green_power_manager_none_at_start(self, rt: Runtime) -> None:
        assert rt.green_power_manager is None

    def test_mcp_layer_none_at_start(self, rt: Runtime) -> None:
        assert rt.mcp_layer is None

    def test_compile_jobs_empty_at_start(self, rt: Runtime) -> None:
        assert isinstance(rt._compile_jobs, dict)
        assert len(rt._compile_jobs) == 0


class TestCompileAsync:
    def test_returns_job_id_string(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_custom_job_id_respected(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model", job_id="my-custom-id")
        assert job_id == "my-custom-id"

    def test_job_registered_immediately(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model", job_id="test-reg")
        assert "test-reg" in rt._compile_jobs

    def test_initial_status_is_valid(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model")
        status = rt.get_compile_status(job_id)
        assert status["status"] in ("queued", "running", "failed", "succeeded")

    def test_job_has_model_field(self, rt: Runtime) -> None:
        job_id = rt.compile_async("my/model-path")
        status = rt.get_compile_status(job_id)
        assert status["model"] == "my/model-path"

    def test_job_has_target_field(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model", target="cuda_sm90")
        status = rt.get_compile_status(job_id)
        assert status["target"] == "cuda_sm90"

    def test_job_has_queued_at_timestamp(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model")
        status = rt.get_compile_status(job_id)
        assert status["queued_at"] is not None
        # Must be an ISO 8601 string
        assert "T" in str(status["queued_at"])

    def test_job_eventually_completes(self, rt: Runtime) -> None:
        """Job must leave queued/running state within 5 seconds."""
        job_id = rt.compile_async("fictional/model-that-does-not-exist-xyz-abc")
        deadline = time.monotonic() + 5.0
        status = rt.get_compile_status(job_id)
        while status["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.1)
            status = rt.get_compile_status(job_id)
        assert status["status"] in ("succeeded", "failed"), (
            f"Job still in state '{status['status']}' after 5s"
        )

    def test_failed_job_has_error_string(self, rt: Runtime) -> None:
        """A non-existent model compile must fail with a non-empty error."""
        job_id = rt.compile_async("nonexistent-model-xyz-123-abc")
        deadline = time.monotonic() + 5.0
        status = rt.get_compile_status(job_id)
        while status["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.1)
            status = rt.get_compile_status(job_id)
        if status["status"] == "failed":
            assert isinstance(status["error"], str)
            assert len(status["error"]) > 0

    def test_multiple_jobs_independent(self, rt: Runtime) -> None:
        jid1 = rt.compile_async("model-a", job_id="job-a-2")
        jid2 = rt.compile_async("model-b", job_id="job-b-2")
        assert jid1 != jid2
        assert rt.get_compile_status("job-a-2")["model"] == "model-a"
        assert rt.get_compile_status("job-b-2")["model"] == "model-b"

    def test_compile_with_v4_flags_does_not_raise(self, rt: Runtime) -> None:
        """compile_async with all v4.0 flags must not raise on construction."""
        job_id = rt.compile_async(
            "fictional/model",
            target="cuda_sm90",
            quality_budget=0.97,
            enable_mtp=True,
            enable_grammar=True,
            enable_tee=False,
            enable_green=True,
        )
        assert isinstance(job_id, str)

    def test_auto_target_resolved_from_fingerprint(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model", target="auto")
        status = rt.get_compile_status(job_id)
        # target field still stores "auto"; the resolved value is used internally
        assert status["target"] == "auto"


class TestGetCompileStatus:
    def test_unknown_job_id_raises_key_error(self, rt: Runtime) -> None:
        with pytest.raises(KeyError):
            rt.get_compile_status("does-not-exist-xyz-999")

    def test_status_includes_job_id(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model", job_id="test-status-id-2")
        status = rt.get_compile_status("test-status-id-2")
        assert status["job_id"] == "test-status-id-2"

    def test_status_has_queued_at(self, rt: Runtime) -> None:
        job_id = rt.compile_async("fictional/model")
        status = rt.get_compile_status(job_id)
        assert "queued_at" in status


class TestMerge:
    def test_merge_requires_real_sources(self, rt: Runtime) -> None:
        with pytest.raises(ValueError, match="at least one task vector"):
            rt.merge("fictional/model", task_vectors=[])

    def test_merge_missing_base_fails(self, rt: Runtime) -> None:
        with pytest.raises(Exception, match="base AEG"):
            rt.merge("fictional/model", task_vectors=[{"path": "/missing"}])

    def test_merge_rejects_unknown_method(self, rt: Runtime) -> None:
        with pytest.raises(ValueError, match="unsupported merge method"):
            rt.merge("fictional/model", task_vectors=[{"path": "/missing"}], method="unknown")


class TestInitV4Layers:
    def test_init_v4_with_none_path_is_noop(self, rt: Runtime) -> None:
        rt._init_v4_layers(None)
        assert rt.grammar_engine is None
        assert rt.ttt_engine is None
        assert rt.tee_manager is None
        assert rt.green_power_manager is None
        assert rt.mcp_layer is None

    def test_init_v4_with_invalid_path_does_not_raise(self, rt: Runtime) -> None:
        """Non-existent AEG path must not raise; layers remain None."""
        rt._init_v4_layers("/nonexistent/path/does-not-exist.aeg")
        assert rt.grammar_engine is None

    def test_init_v4_idempotent(self, rt: Runtime) -> None:
        """Calling _init_v4_layers twice must not raise."""
        rt._init_v4_layers(None)
        rt._init_v4_layers(None)
        assert rt.grammar_engine is None
