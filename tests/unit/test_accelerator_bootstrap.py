"""Accelerator deserialization verifies real weights without compiling unused CPU code."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from aether import Compiler, CompilerConfig
from aether.runtime import aeg_loader


@pytest.fixture
def compiled_cpu_model(tiny_local_safetensors_model, tmp_path):
    config = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "cache"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    return Compiler(config).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "fixture.aeg")


def test_reference_bootstrap_skips_native_preparation_and_compiler_discovery(compiled_cpu_model, monkeypatch):
    expected = aeg_loader.load_engine_from_path(compiled_cpu_model.root, allow_uncertified=True)
    expected_logits, _ = expected.forward(np.array([1, 2, 3]))
    with monkeypatch.context() as patch:
        def unused(*args, **kwargs):
            pytest.fail("accelerator bootstrap prepared an unused native CPU provider")
        patch.setattr(aeg_loader, "_prepare_native_kernels", unused)
        patch.setattr(aeg_loader, "get_native_kernels", unused)
        patch.setattr("aether.runtime.cpu_engine.get_native_kernels", unused)
        patch.setattr("aether.kernels.native_cpu.detect_toolchain", unused)
        reference = aeg_loader.load_engine_from_path(
            compiled_cpu_model.root, prepare_native_kernels=False, allow_uncertified=True
        )
        package_reference = aeg_loader.load_engine_from_package(
            compiled_cpu_model, prepare_native_kernels=False, allow_uncertified=True
        )
        actual, cache = reference.forward(np.array([1, 2, 3]))
        package_actual, _ = package_reference.forward(np.array([1, 2, 3]))
        assert reference.kernels.is_native is False
        np.testing.assert_allclose(actual, expected_logits, rtol=2e-4, atol=2e-5)
        np.testing.assert_array_equal(actual, package_actual)
        assert cache.length == 3
    prepare = []
    original = aeg_loader._prepare_native_kernels
    monkeypatch.setattr(aeg_loader, "_prepare_native_kernels", lambda package: prepare.append(package.root) or original(package))
    cpu = aeg_loader.load_engine_from_path(compiled_cpu_model.root, allow_uncertified=True)
    assert prepare == [compiled_cpu_model.root]
    assert cpu.kernels.is_native == expected.kernels.is_native


def test_reference_bootstrap_still_rejects_corrupt_authenticated_weights(compiled_cpu_model, monkeypatch):
    blob = compiled_cpu_model.root / "weights/quantized/model.aeg-quant"
    payload = bytearray(blob.read_bytes())
    payload[-1] ^= 0xFF
    blob.write_bytes(payload)
    monkeypatch.setattr(aeg_loader, "_prepare_native_kernels", lambda _: pytest.fail("integrity check reached native preparation"))
    with pytest.raises(Exception, match="[Hh]ash|integrity|checksum"):
        aeg_loader.load_engine_from_path(compiled_cpu_model.root, prepare_native_kernels=False)
