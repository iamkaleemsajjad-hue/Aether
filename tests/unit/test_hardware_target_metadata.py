"""Portable target metadata and fake discovery contracts; no physical GPU proof."""

from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from aether.backends.capabilities import HardwareCapabilities
from aether.cli import cli
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.constants import BACKEND_BY_TARGET, SUPPORTED_TARGET_IDS, SUPPORTED_TARGETS
from aether.core.types import HardwareTarget


@pytest.mark.parametrize("target", list(HardwareTarget))
def test_all_target_metadata_and_profile_aliases_use_canonical_definitions(target):
    """Every enum target, including generic aliases, has complete consistent metadata."""
    assert target.value in SUPPORTED_TARGET_IDS
    assert target.vendor
    assert target.display_name == SUPPORTED_TARGETS[target.value]
    assert target.backend_candidates == BACKEND_BY_TARGET[target.value]
    for alias in (target, target.value, " " + target.value.upper() + " "):
        profile = HardwareProfile.from_target_id(alias)
        assert profile.target_id == target.value
        assert profile.backend_candidates == target.backend_candidates
        assert profile.recommended_backend == target.backend_candidates[0]
    candidates = target.backend_candidates
    candidates.clear()
    assert target.backend_candidates == BACKEND_BY_TARGET[target.value]


def _cpu():
    return HardwareCapabilities(
        vendor="CPU",
        device="Discovery CPU fixture",
        architecture="fixture_cpu",
        target_id="cpu_generic",
        implemented=True,
        available=True,
        memory_bytes=32 * 1024**3,
    )


def _metal(target="metal", device="AMD Radeon Pro [contract fixture]", vendor="AMD", memory_gb=4):
    return HardwareCapabilities(
        vendor=vendor,
        device=device,
        architecture="queried Metal architecture fixture",
        target_id=target,
        memory_bytes=memory_gb * 1024**3,
        supports_fp16=True,
        supports_fp8=False,
        supports_bf16=False,
        implemented=True,
        available=True,
        hardware_present=True,
        runtime_available=True,
        driver_version="fixture driver",
        extra={"system_default": True, "device_index": 0},
    )


@pytest.mark.parametrize(
    "metal",
    [
        _metal(),
        _metal(device="Intel Iris Pro [contract fixture]", vendor="Intel", memory_gb=2),
        _metal("metal_m3", "Apple M4 Max [contract fixture]", "Apple", memory_gb=48),
    ],
)
def test_auto_metal_preserves_actual_discovery_contract_and_one_probe(monkeypatch, metal):
    """Auto selection preserves supplied evidence without substituting an M1/catalog GPU."""
    probe = Mock(return_value=[_cpu(), metal])
    monkeypatch.setattr("aether.backends.hardware_detector.detect_all_capabilities", probe)
    assert HardwareTarget.auto().value == metal.target_id
    probe.assert_called_once()
    probe.reset_mock()
    profile = HardwareProfile.auto()
    probe.assert_called_once()
    assert profile.target_id == metal.target_id
    assert profile.name == metal.device
    assert profile.memory_gb == metal.memory_bytes / (1024**3)
    assert profile.supports_fp8 is False
    assert profile.supports_bf16 is False
    assert profile.tensor_core_flops == profile.memory_bandwidth_gb_s == 0
    assert profile.warp_size == 0
    assert profile.supports_flash_attention is False
    evidence = profile.attributes["capabilities"]
    assert evidence["vendor"] == metal.vendor
    assert evidence["device"] == metal.device
    assert evidence["driver_version"] == metal.driver_version
    assert evidence["execution_tested"] is False
    assert evidence["production_validated"] is False
    restored = HardwareProfile.from_dict(profile.to_dict())
    assert restored.to_dict() == profile.to_dict()


def test_auto_ignores_unavailable_unimplemented_and_unknown_discovery(monkeypatch):
    """Only supported, available implementation records participate in target preference."""
    hidden, absent, unknown = _metal(), _metal(), _metal()
    hidden.implemented = False
    absent.available = False
    unknown.target_id = "unregistered_metal_device"
    monkeypatch.setattr(
        "aether.backends.hardware_detector.detect_all_capabilities",
        lambda: [hidden, absent, unknown, _cpu()],
    )
    assert HardwareTarget.auto() is HardwareTarget.CPU_GENERIC
    assert HardwareProfile.auto().name == _cpu().device


def test_auto_empty_discovery_has_conservative_portable_fallback(monkeypatch):
    """A missing device must not acquire assumed CPU ISA, RAM or accelerator features."""
    monkeypatch.setattr("aether.backends.hardware_detector.detect_all_capabilities", lambda: [])
    assert HardwareTarget.auto() is HardwareTarget.CPU_GENERIC
    profile = HardwareProfile.auto()
    assert profile.target_id == "cpu_generic"
    assert profile.memory_gb == 0
    assert profile.warp_size == 0
    assert profile.backend_candidates == BACKEND_BY_TARGET["cpu_generic"]
    assert profile.recommended_backend == "aether_cpu"


def test_missing_catalog_and_backend_entries_do_not_invent_cpu_adapter(monkeypatch):
    """A valid generic alias retains unknown specifications without becoming a CPU target."""
    import aether.compiler.stage3_targeting.hardware_profile as profiles

    monkeypatch.delitem(profiles._TARGET_PROFILES, "metal")
    monkeypatch.delitem(BACKEND_BY_TARGET, "metal")
    profile = HardwareProfile.from_target_id(HardwareTarget.METAL)
    assert profile.name == SUPPORTED_TARGETS["metal"]
    assert profile.target_id == "metal"
    assert profile.backend_candidates == []
    assert profile.recommended_backend is None
    assert profile.memory_gb == profile.tensor_core_flops == 0
    assert profile.warp_size == 0


def test_generic_metal_kernels_cli_uses_actual_host_name_and_catalog_scope(monkeypatch):
    """Kernel inventory works with generic Metal and distinguishes templates from readiness."""
    metal = _metal()
    probe = Mock(return_value=[metal, _cpu()])
    monkeypatch.setattr("aether.backends.hardware_detector.detect_all_capabilities", probe)
    result = CliRunner().invoke(cli, ["kernels"], terminal_width=160)
    assert result.exit_code == 0, result.output
    probe.assert_called_once()
    assert "Kernel Target Profiles" in result.output
    assert "Current target: metal (" + metal.device + ")" in result.output
    assert (
        "Profiles do not establish installed backend readiness or successful execution"
        in result.output
    )


def test_generic_metal_fingerprint_keeps_physical_identity(monkeypatch):
    """Runtime fingerprints keep supplied device evidence separate from profile labels."""
    from aether.runtime.hardware import HardwareDetector

    metal = _metal()
    monkeypatch.setattr(
        "aether.backends.hardware_detector.detect_all_capabilities", lambda: [metal, _cpu()]
    )
    result = HardwareDetector().detect()
    assert result.target_id == "metal"
    assert result.target_name == result.gpu_name == metal.device
    assert result.gpu_memory_gb == 4
    assert result.capabilities[0]["vendor"] == "AMD"
    assert result.capabilities[0]["execution_tested"] is False
