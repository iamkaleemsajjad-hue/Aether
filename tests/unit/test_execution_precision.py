"""Real tensor mutations must pass measured quality/latency gates and restore."""

import dataclasses
import weakref
from pathlib import Path

import numpy as np
import pytest
from test_batched_inference import _reference_engine

from aether.runtime.config import RuntimeConfig
from aether.runtime.execution_precision import (
    ExecutionPrecisionController,
    NumpyPrecisionAdapter,
    PrecisionPolicy,
    TorchPrecisionAdapter,
)

IDS = np.asarray([1, 2, 3, 4], dtype=np.int64)


def _lossless_engine():
    engine = _reference_engine()
    adapter = NumpyPrecisionAdapter(engine)
    for binding in adapter.bindings():
        binding.set(binding.get().astype(np.float16).astype(np.float32))
    return engine


def test_lossless_storage_changes_real_weights_and_restores_logits(tmp_path):
    """Lossless projection storage changes preserve actual logits and restoration."""
    engine = _lossless_engine()
    original_logits, _ = engine.forward(IDS)
    original = engine.weights.layers[0].q_proj.copy()
    protected_embedding = engine.weights.embedding
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )
    report = controller.downgrade([IDS.tolist()])
    assert report["action"] == "downgrade"
    assert report["saved_bytes"] > 0
    assert report["baseline_seconds"] > 0 and report["candidate_seconds"] > 0
    assert engine.weights.layers[0].q_proj.dtype == np.float16
    assert engine.weights.embedding is protected_embedding
    repeat = controller.downgrade([IDS.tolist()])
    assert repeat["action"] == "already_compact" and not repeat["execution_changed"]
    np.testing.assert_array_equal(engine.forward(IDS)[0], original_logits)
    assert controller.restore()["execution_changed"]
    np.testing.assert_array_equal(engine.weights.layers[0].q_proj, original)
    assert engine.weights.layers[0].q_proj.dtype == np.float32
    np.testing.assert_array_equal(engine.forward(IDS)[0], original_logits)
    assert not list(tmp_path.glob("*.npy"))


def test_random_fp32_weights_are_not_silently_rounded(tmp_path):
    """Default permission leaves weights that require lossy rounding unchanged."""
    engine = _reference_engine()
    before = engine.forward(IDS)[0]
    controller = ExecutionPrecisionController(NumpyPrecisionAdapter(engine), tmp_path)
    report = controller.downgrade([IDS.tolist()])
    assert not report["execution_changed"]
    assert engine.weights.layers[0].q_proj.dtype == np.float32
    np.testing.assert_array_equal(engine.forward(IDS)[0], before)


def test_zero_quality_budget_rejects_lossy_change_transactionally(tmp_path):
    """A measured quality rejection restores exact original CPU tensor identities."""
    engine = _reference_engine()
    before = engine.forward(IDS)[0]
    original = engine.weights.layers[0].q_proj
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine),
        tmp_path,
        PrecisionPolicy(allow_lossy=True, measurements=1, max_slowdown=1000),
    )
    report = controller.downgrade([IDS.tolist()])
    assert report["action"] == "rejected"
    assert "quality gate" in report["reason"]
    assert engine.weights.layers[0].q_proj is original
    np.testing.assert_array_equal(engine.forward(IDS)[0], before)
    assert not list(tmp_path.glob("*.npy"))


def test_tied_projection_aliases_survive_roundtrip(tmp_path):
    """Compact conversion and restoration preserve actual tied projection coordinates."""
    engine = _lossless_engine()
    layer = engine.weights.layers[0]
    layer.v_proj = layer.k_proj
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    assert layer.v_proj is layer.k_proj
    controller.restore()
    assert layer.v_proj is layer.k_proj


def test_corrupt_restore_data_does_not_partially_mutate_model(tmp_path):
    """Checksum rejection happens before any execution tensor is restored."""
    engine = _lossless_engine()
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    backups = list(tmp_path.glob("*.npy"))
    backups[-1].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        controller.restore()
    assert engine.weights.layers[0].q_proj.dtype == np.float16


def test_owned_engine_discard_never_restores_or_reads_corrupt_backups(tmp_path):
    """Unload does not allocate original tensors or trust obsolete backup data."""
    engine = _lossless_engine()
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    for backup in tmp_path.glob("*.npy"):
        backup.write_bytes(b"corrupt")
    compact = engine.weights.layers[0].q_proj
    controller.discard()
    assert engine.weights.layers[0].q_proj is compact
    assert not controller.is_compact
    assert not list(tmp_path.glob("*.npy"))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_portable_torch_precision_executes_and_restores(tmp_path, dtype):
    """Actual CPU Torch operators run after both FP16 and BF16 transformations."""
    torch = pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(_reference_engine(), "cpu")
    original_dtype = engine.compute_dtype
    original_logits = engine.forward(IDS)[0]
    controller = ExecutionPrecisionController(
        TorchPrecisionAdapter(engine),
        tmp_path,
        PrecisionPolicy(
            storage_dtype=dtype,
            allow_lossy=True,
            max_logit_error=0.1,
            require_same_argmax=False,
            measurements=1,
            max_slowdown=1000,
        ),
    )
    result = controller.downgrade([IDS.tolist()])
    assert result["action"] == "downgrade", result
    assert engine.compute_dtype == getattr(torch, dtype)
    assert engine.embedding.dtype == getattr(torch, dtype)
    assert np.all(np.isfinite(engine.forward(IDS)[0]))
    controller.restore()
    assert engine.compute_dtype == original_dtype
    np.testing.assert_array_equal(engine.forward(IDS)[0], original_logits)


def test_runtime_config_roundtrips_real_feature_controls():
    """User execution-feature controls survive public configuration serialization."""
    config = RuntimeConfig(
        precision_calibration_tokens=[[1, 2]],
        draft_model_id="draft.aeg",
        prefill_devices=["cpu"],
        decode_devices=["cpu"],
        kv_cache_path="cache-data",
        disaggregate_prefill_decode=True,
        precision_allow_lossy=True,
        precision_max_logit_error=0.01,
    )
    assert dataclasses.asdict(RuntimeConfig.from_dict(config.to_dict())) == dataclasses.asdict(
        config
    )


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_nonfinite_or_negative_quality_bound_is_invalid(value):
    """Reject acceptance gates whose comparison semantics cannot be trusted."""
    with pytest.raises(ValueError):
        PrecisionPolicy(max_logit_error=value)


def _torch_controller(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("AETHER_TORCH_DTYPE", "float32")
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(_reference_engine(), "cpu")
    adapter = TorchPrecisionAdapter(engine)
    controller = ExecutionPrecisionController(
        adapter,
        tmp_path,
        PrecisionPolicy(
            allow_lossy=True,
            max_logit_error=1.0,
            require_same_argmax=False,
            max_slowdown=1000,
            measurements=1,
        ),
    )
    return torch, engine, adapter, controller


def test_torch_top_bindings_release_original_before_next_conversion(tmp_path, monkeypatch):
    """Actual CPU tensor lifetimes verify prompt engine binding replacement."""
    torch, engine, adapter, controller = _torch_controller(tmp_path, monkeypatch)
    original_embedding = weakref.ref(engine.embedding)
    convert = adapter.convert
    observations = []

    def observe(value, policy):
        observations.append((engine.embedding.dtype, original_embedding() is not None))
        return convert(value, policy)

    monkeypatch.setattr(adapter, "convert", observe)
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    assert observations[0] == (torch.float32, True)
    assert observations[1] == (torch.float16, False)
    controller.restore()


def test_torch_restore_streams_groups_without_retaining_all_compact_tensors(tmp_path, monkeypatch):
    """Real reconstruction commits each alias group before loading the next."""
    from aether.runtime.execution_precision import _bindings

    torch, engine, adapter, controller = _torch_controller(tmp_path, monkeypatch)
    expected = engine.forward(IDS)[0]
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    decode = adapter.decode
    observations = []

    def observe(value, metadata, like):
        observations.append(
            (
                engine.embedding.dtype,
                sum(
                    binding.get().dtype == torch.float16
                    for binding in _bindings(adapter._root, torch_module=torch)
                ),
            )
        )
        return decode(value, metadata, like)

    monkeypatch.setattr(adapter, "decode", observe)
    assert controller.restore()["execution_changed"]
    assert observations[0][0] == torch.float16
    assert observations[1][0] == torch.float32
    assert observations[-1][1] < observations[0][1]
    np.testing.assert_array_equal(engine.forward(IDS)[0], expected)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_torch_streaming_restore_failure_rolls_back_exact_compact_aliases(
    tmp_path, monkeypatch, dtype
):
    """An actual reconstruction failure preserves compact bytes and tied tensors."""
    from aether.runtime.execution_precision import _bindings

    torch, engine, _, _ = _torch_controller(tmp_path, monkeypatch)
    engine.lm_head = engine.embedding
    engine.layers[0]["v_proj"] = engine.layers[0]["k_proj"]
    adapter = TorchPrecisionAdapter(engine)
    controller = ExecutionPrecisionController(
        adapter,
        tmp_path,
        PrecisionPolicy(
            storage_dtype=dtype,
            allow_lossy=True,
            max_logit_error=1.0,
            require_same_argmax=False,
            max_slowdown=1000,
            measurements=1,
        ),
    )
    expected = engine.forward(IDS)[0]
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    compact = {
        binding.name: binding.get().contiguous().view(torch.uint8).numpy().tobytes()
        for binding in _bindings(adapter._root, torch_module=torch)
    }
    decode = adapter.decode
    calls = 0

    def fail_once(value, metadata, like):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("real restoration allocation failed")
        return decode(value, metadata, like)

    monkeypatch.setattr(adapter, "decode", fail_once)
    with pytest.raises(RuntimeError, match="allocation failed"):
        controller.restore()
    assert engine.compute_dtype == getattr(torch, dtype)
    assert engine.embedding is engine.lm_head
    assert engine.layers[0]["v_proj"] is engine.layers[0]["k_proj"]
    assert compact == {
        binding.name: binding.get().contiguous().view(torch.uint8).numpy().tobytes()
        for binding in _bindings(adapter._root, torch_module=torch)
    }
    monkeypatch.setattr(adapter, "decode", decode)
    assert controller.restore()["execution_changed"]
    assert engine.embedding is engine.lm_head
    assert engine.layers[0]["v_proj"] is engine.layers[0]["k_proj"]
    np.testing.assert_array_equal(engine.forward(IDS)[0], expected)


def test_restore_validates_every_encoded_schema_before_mutating_weights(tmp_path):
    """A checksummed but structurally invalid later backup cannot partially restore."""
    from aether.runtime.execution_precision import _file_digest

    engine = _lossless_engine()
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    bindings, path, metadata, _ = controller._backups[-1]
    np.save(path, np.zeros(metadata["_encoded_shape"], dtype=np.float64), allow_pickle=False)
    controller._backups[-1] = bindings, path, metadata, _file_digest(path)
    with pytest.raises(ValueError, match="encoded schema"):
        controller.restore()
    assert engine.weights.layers[0].q_proj.dtype == np.float16


def test_precision_checksums_never_read_entire_backup_files(tmp_path, monkeypatch):
    """Backup hashing and validation use bounded reads through real restoration."""
    engine = _lossless_engine()
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )

    def forbidden(path):
        raise AssertionError(f"unbounded file read: {path}")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert controller.downgrade([IDS.tolist()])["execution_changed"]
    assert controller.restore()["execution_changed"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_precision_interruption_restores_real_execution_and_reraises(
    tmp_path, monkeypatch, interruption, framework
):
    """Cancellation after real compact operators restores weights and their aliases."""
    if framework == "torch":
        _torch, engine, _, _ = _torch_controller(tmp_path, monkeypatch)
        engine.lm_head = engine.embedding
        engine.layers[0]["v_proj"] = engine.layers[0]["k_proj"]
        adapter = TorchPrecisionAdapter(engine)
        original_dtype = engine.compute_dtype
        original_identity = None
    else:
        engine = _lossless_engine()
        engine.weights.layers[0].v_proj = engine.weights.layers[0].k_proj
        adapter = NumpyPrecisionAdapter(engine)
        original_identity = engine.weights.layers[0].q_proj
        original_dtype = original_identity.dtype
    expected = engine.forward(IDS)[0]
    controller = ExecutionPrecisionController(
        adapter,
        tmp_path,
        PrecisionPolicy(
            allow_lossy=framework == "torch",
            max_logit_error=1.0,
            require_same_argmax=False,
            max_slowdown=1000,
            measurements=1,
        ),
    )
    evaluate = adapter.evaluate
    interrupted = False

    def cancel_after_real_evaluation(ids):
        nonlocal interrupted
        logits = evaluate(ids)
        compact = (
            engine.compute_dtype != original_dtype
            if framework == "torch"
            else engine.weights.layers[0].q_proj.dtype != original_dtype
        )
        if compact and not interrupted:
            interrupted = True
            raise interruption("actual candidate calibration interrupted")
        return logits

    monkeypatch.setattr(adapter, "evaluate", cancel_after_real_evaluation)
    with pytest.raises(interruption, match="candidate calibration interrupted"):
        controller.downgrade([IDS.tolist()])
    assert interrupted and not controller.is_compact
    assert not list(tmp_path.glob("*.npy"))
    if framework == "torch":
        assert engine.compute_dtype == original_dtype
        assert engine.embedding is engine.lm_head
        assert engine.layers[0]["v_proj"] is engine.layers[0]["k_proj"]
    else:
        assert engine.weights.layers[0].q_proj is original_identity
        assert engine.weights.layers[0].v_proj is engine.weights.layers[0].k_proj
    np.testing.assert_array_equal(engine.forward(IDS)[0], expected)


@pytest.mark.parametrize("failure_stage", ["write", "digest"])
def test_failed_backup_creation_leaves_no_orphan_or_execution_change(
    tmp_path, monkeypatch, failure_stage
):
    """Actual temporary backup cleanup covers write and pre-staging digest errors."""
    import aether.runtime.execution_precision as precision

    engine = _lossless_engine()
    original = engine.weights.layers[0].q_proj
    expected = engine.forward(IDS)[0]
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )

    def fail(*args, **kwargs):
        if failure_stage == "write":
            args[0].write(b"partial real backup")
        raise OSError(f"{failure_stage} failed before staging")

    monkeypatch.setattr(
        precision.np if failure_stage == "write" else precision,
        "save" if failure_stage == "write" else "_file_digest",
        fail,
    )
    report = controller.downgrade([IDS.tolist()])
    assert report["action"] == "rejected" and not report["execution_changed"]
    assert "before staging" in report["reason"]
    assert not list(tmp_path.glob("*.npy"))
    assert engine.weights.layers[0].q_proj is original
    np.testing.assert_array_equal(engine.forward(IDS)[0], expected)


def test_interrupted_precision_commit_clears_restoration_state(tmp_path, monkeypatch):
    """Cancellation after calibrated commit metadata cannot retain deleted backups."""
    engine = _lossless_engine()
    original = engine.weights.layers[0].q_proj
    expected = engine.forward(IDS)[0]
    controller = ExecutionPrecisionController(
        NumpyPrecisionAdapter(engine), tmp_path, PrecisionPolicy(measurements=1, max_slowdown=1000)
    )
    interrupted = False

    def interrupt_commit(instance, name, value):
        nonlocal interrupted
        object.__setattr__(instance, name, value)
        if instance is controller and name == "_backups" and value and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("actual calibrated precision commit interrupted")

    monkeypatch.setattr(ExecutionPrecisionController, "__setattr__", interrupt_commit)
    with pytest.raises(KeyboardInterrupt, match="precision commit interrupted"):
        controller.downgrade([IDS.tolist()])
    assert interrupted and not controller.is_compact
    assert not list(tmp_path.glob("*.npy"))
    assert engine.weights.layers[0].q_proj is original
    np.testing.assert_array_equal(engine.forward(IDS)[0], expected)
