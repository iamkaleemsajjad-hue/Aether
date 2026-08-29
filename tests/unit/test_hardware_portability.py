"""Nothing in the runtime may assume which accelerator it is running on.

Aether's premise is one artifact that runs on whatever hardware is present, so a
vendor name in a code path is a defect rather than a detail.  These tests pin the
places where the runtime *asks* the backend about itself instead of assuming: the
calibration key, the device barrier, the memory probes, and the residency precision.

They run on CPU.  Foreign backends are represented by stubs, because the property
under test is that an unknown backend is handled correctly — and an unknown backend is
by definition one this machine does not have.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from aether.placement import bootstrap as boot  # noqa: E402
from aether.placement.census import _device_barrier  # noqa: E402
from aether.runtime.decode_profile import _synchronizer  # noqa: E402
from aether.runtime.kernel_strategy import StrategyCalibrator  # noqa: E402
from aether.runtime.torch_engine import TorchAEGEngine  # noqa: E402


def fake_torch(*, hip: str | None = None, arch: str = "", capability=(8, 0), **namespaces):
    """A stand-in exposing only what the code under test is allowed to consult."""
    calls: list[str] = []

    def synchronize(*_a, **_k) -> None:
        calls.append("cuda")

    cuda = SimpleNamespace(
        synchronize=synchronize,
        get_device_capability=lambda *_a: capability,
        get_device_properties=lambda *_a: SimpleNamespace(gcnArchName=arch),
    )
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=None if hip else "12.1"),
        cuda=cuda,
        calls=calls,
        **namespaces,
    )


def device(kind: str, index: int | None = 0):
    return SimpleNamespace(type=kind, index=index)


# ── the calibration key names the vendor, not the device type ──────────────────

def test_a_rocm_device_is_not_keyed_as_cuda() -> None:
    """PyTorch exposes AMD GPUs through ``torch.cuda``; the key must not repeat that."""
    calibrator = StrategyCalibrator(
        fake_torch(hip="6.0", arch="gfx942:sramecc+:xnack-"), device("cuda")
    )
    assert calibrator.device_kind == "rocm-gfx942"


def test_a_cuda_device_is_keyed_by_compute_capability() -> None:
    calibrator = StrategyCalibrator(fake_torch(capability=(7, 5)), device("cuda"))
    assert calibrator.device_kind == "cuda-sm75"


def test_two_vendors_cannot_collide_on_one_calibration_key() -> None:
    """The bug this prevents: an MI300X reusing an NVIDIA card's measured winner."""
    amd = StrategyCalibrator(fake_torch(hip="6.0", arch="gfx90a"), device("cuda"))
    nvidia = StrategyCalibrator(fake_torch(capability=(9, 0)), device("cuda"))
    assert amd.device_kind != nvidia.device_kind


@pytest.mark.parametrize("kind", ["xpu", "mps", "cpu", "npu", "some-future-backend"])
def test_every_other_backend_contributes_its_own_type(kind: str) -> None:
    """An unrecognised backend gets its own key rather than an exception or a default."""
    calibrator = StrategyCalibrator(fake_torch(), device(kind))
    assert calibrator.device_kind == kind


def test_a_backend_that_refuses_to_identify_itself_is_not_fatal() -> None:
    def explode(*_a, **_k):
        raise RuntimeError("driver not initialised")

    broken = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(get_device_capability=explode),
    )
    assert StrategyCalibrator(broken, device("cuda")).device_kind == "cuda"


# ── barriers are looked up, not enumerated ────────────────────────────────────

def test_a_cpu_needs_no_barrier() -> None:
    assert _device_barrier(fake_torch(), device("cpu")) is None
    assert _synchronizer(SimpleNamespace(torch=fake_torch(), device=device("cpu")))() is None


def test_an_unknown_accelerator_barrier_is_found_by_namespace() -> None:
    """A backend Aether has never seen still gets a correct barrier."""
    seen: list[str] = []
    module = fake_torch(
        npu=SimpleNamespace(synchronize=lambda: seen.append("npu"))
    )
    barrier = _device_barrier(module, device("npu"))
    assert barrier is not None
    barrier()
    assert seen == ["npu"]


def test_an_accelerator_without_a_barrier_is_reported_as_unmeasurable() -> None:
    """Timing queued work without a barrier would report launch time as execution time."""
    module = fake_torch(fpga=SimpleNamespace())
    assert _device_barrier(module, device("fpga")) is None


def test_the_profiler_barrier_is_also_generic() -> None:
    seen: list[str] = []
    engine = SimpleNamespace(
        torch=fake_torch(xpu=SimpleNamespace(synchronize=lambda: seen.append("xpu"))),
        device=device("xpu"),
    )
    _synchronizer(engine)()
    assert seen == ["xpu"]


def test_the_calibrator_barrier_never_raises_on_a_hostile_backend() -> None:
    def explode() -> None:
        raise RuntimeError("no queue to drain")

    calibrator = StrategyCalibrator(
        fake_torch(tpu=SimpleNamespace(synchronize=explode)), device("tpu")
    )
    calibrator._synchronize()  # must not propagate


# ── memory probes report "unknown" rather than inventing a number ─────────────

def test_an_unknown_accelerator_defers_the_bootstrap() -> None:
    """A fabricated zero would be folded into sigma as though it were measured."""
    assert boot.read_memory("fpga:0") is None


def test_resetting_peak_stats_on_an_unknown_backend_is_a_no_op() -> None:
    boot.reset_peak_stats(["fpga:0", "cpu"])  # must not raise


def test_the_xpu_reading_marks_the_unmodellable_term_as_unmeasured() -> None:
    """Intel publishes no driver-wide total, so the non-framework term must read zero."""
    reading = boot.MemoryReading(
        device_id="xpu:0", peak_allocated_bytes=1 << 20,
        reserved_bytes=1 << 21, driver_used_bytes=1 << 21,
    )
    assert reading.non_framework_bytes == 0
    assert reading.fragmentation == pytest.approx(2.0)


# ── residency precision is probed, not assumed ────────────────────────────────

def test_a_cpu_stays_at_full_precision() -> None:
    """Half precision on a CPU is emulated and slower, so it is never selected."""
    TorchAEGEngine._HALF_PRECISION_SUPPORT.clear()
    assert TorchAEGEngine._probe_compute_dtype(torch, torch.device("cpu")) is torch.float32


def test_the_precision_probe_is_cached_per_device_type() -> None:
    TorchAEGEngine._HALF_PRECISION_SUPPORT.clear()
    TorchAEGEngine._probe_compute_dtype(torch, torch.device("cpu"))
    assert "cpu" in TorchAEGEngine._HALF_PRECISION_SUPPORT
    calls = {"n": 0}

    class Counting:
        float32 = torch.float32
        float16 = torch.float16

        @staticmethod
        def ones(*_a, **_k):
            calls["n"] += 1
            raise AssertionError("the probe must not run again for a cached type")

    TorchAEGEngine._probe_compute_dtype(Counting, torch.device("cpu"))
    assert calls["n"] == 0


def test_a_backend_that_cannot_do_half_precision_falls_back_to_full() -> None:
    TorchAEGEngine._HALF_PRECISION_SUPPORT.clear()

    class Refusing:
        float32 = torch.float32
        float16 = torch.float16

        @staticmethod
        def ones(*_a, **_k):
            raise RuntimeError("half precision is not implemented for this backend")

    verdict = TorchAEGEngine._probe_compute_dtype(Refusing, SimpleNamespace(type="fpga"))
    assert verdict is torch.float32
    TorchAEGEngine._HALF_PRECISION_SUPPORT.clear()


def test_a_backend_whose_half_precision_is_wrong_falls_back_to_full() -> None:
    """Executing FP16 is not the same as computing FP16 correctly."""
    TorchAEGEngine._HALF_PRECISION_SUPPORT.clear()

    class Garbage:
        float32 = torch.float32
        float16 = torch.float16
        allclose = staticmethod(lambda *_a, **_k: False)

        @staticmethod
        def ones(shape, **_kwargs):
            return torch.ones(shape, dtype=torch.float32)

        @staticmethod
        def full(shape, value, **_kwargs):
            return torch.full(shape, value, dtype=torch.float32)

    verdict = TorchAEGEngine._probe_compute_dtype(Garbage, SimpleNamespace(type="weird"))
    assert verdict is torch.float32
    TorchAEGEngine._HALF_PRECISION_SUPPORT.clear()


def test_cuda_precision_is_not_probed_so_existing_behaviour_is_unchanged() -> None:
    """A CUDA host must keep FP16 residency exactly as before, probe or no probe."""
    import inspect

    source = inspect.getsource(TorchAEGEngine.__init__)
    assert 'elif self.device.type == "cuda":' in source
    assert "self.compute_dtype = torch.float16" in source
