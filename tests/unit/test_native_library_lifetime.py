"""Native library ownership across removable artifacts and later compilation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from aether.kernels import native_cpu as native


@pytest.mark.parametrize("system,suffix", [("nt", ".dll"), ("posix", ".so"), ("posix", ".dylib")])
def test_packaged_library_cache_owns_its_bytes_on_every_platform(
    tmp_path, monkeypatch, system, suffix
):
    """Storage policy only: dummy bytes never count as native execution evidence."""
    kernels = native.NativeCPUKernels.reference()
    identity = kernels.build_identity()
    artifact = tmp_path / ("removable" + suffix)
    payload = b"verified library storage policy"
    artifact.write_bytes(payload)
    monkeypatch.setattr(
        native,
        "os",
        SimpleNamespace(
            name=system,
            environ={**os.environ, "AETHER_KERNEL_CACHE": str(tmp_path / "cache")},
            replace=os.replace,
            fsync=os.fsync,
        ),
    )
    loaded_paths = []
    monkeypatch.setattr(
        native, "_open_shared_library", lambda path, _: loaded_paths.append(path) or object()
    )
    monkeypatch.setattr(kernels, "_bind_signatures", lambda: None)
    assert kernels.load_library(artifact, hashlib.sha256(payload).hexdigest(), identity)
    artifact.unlink()
    assert loaded_paths[0] != artifact.resolve()
    assert kernels.library_path.read_bytes() == payload


def test_real_artifact_removal_does_not_poison_later_compilation(tmp_path, monkeypatch):
    """Actual compile→load→remove→emit→compile retains independent native ownership."""
    from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
    from aether.backends.hardware_validation import _write_probe_checkpoint
    from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
    from aether.utils.file_io import aeg_cache_candidates

    monkeypatch.setenv("AETHER_KERNEL_CACHE", str(tmp_path / "persistent-kernels"))
    monkeypatch.setattr(native, "_INSTANCE", None)
    shared = native.get_native_kernels()
    if not shared.ensure_compiled():
        pytest.skip(f"Actual native compilation unavailable: {shared.build_error}")
    host_library = shared.library_path
    host_identity = shared.build_identity()
    checkpoint = tmp_path / "checkpoint"
    _write_probe_checkpoint(checkpoint)
    config = CompilerConfig(
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "compiler-cache"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    runtime_config = RuntimeConfig(
        backend_name="aether_cpu",
        execution_devices=["cpu"],
        speculative_decoding=False,
        dynamic_precision=False,
        enable_semantic_cache=False,
        model_cache_dir=str(tmp_path / "runtime-cache"),
        kv_cache_path=str(tmp_path / "sessions"),
    )
    removable = aeg_cache_candidates("removable-model", runtime_config.model_cache_dir)[0]
    artifact = Compiler(config).compile(str(checkpoint), output_path=removable)
    assert artifact.metadata["kernel_artifacts"]
    with Runtime(runtime_config) as runtime:
        stream = runtime.generate_stream(
            "removable-model", "hello world", max_tokens=4, temperature=0
        )
        assert next(stream)
        engine_kernels = runtime._loaded_models["removable-model"].engine.kernels
        runtime.remove("removable-model")
        assert not removable.exists()
        assert list(stream) == []
    assert engine_kernels.argmax(np.asarray([1.0, 5.0, 2.0], dtype=np.float32)) == 1
    emitted = KernelEmitter("cpu_generic").emit_executable(
        "argmax", tmp_path / ("later-kernel" + host_library.suffix)
    )
    emitted_provider = native.NativeCPUKernels.reference()
    emitted_provider.toolchain = shared.toolchain
    assert emitted_provider.load_library(emitted.artifact_path, emitted.sha256, host_identity)
    assert emitted_provider.argmax(np.asarray([2.0, 1.0, 7.0], dtype=np.float32)) == 2
    later = Compiler(config).compile(str(checkpoint), output_path=tmp_path / "later.aeg")
    later.verify_integrity()
    assert later.metadata["kernel_artifacts"]
    assert all((later.root / item["path"]).is_file() for item in later.metadata["kernel_artifacts"])
    assert native.get_native_kernels() is shared
    assert shared.library_path == host_library
    assert shared.build_identity() == host_identity
    assert host_library.is_file()
    assert engine_kernels is not shared
    assert not engine_kernels.library_path.is_relative_to(removable)
