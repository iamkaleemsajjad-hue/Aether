"""What the host actually is, and what that makes possible.

:mod:`benchmark.system_info` already records the environment for the report. This
module answers the narrower question the suite needs *before* it runs anything:
which engines are even applicable here, and which precision every engine should
be held to.

Nothing here guesses. ``bf16_native`` is torch's own answer for the device, not
an inference from the device name; ``nvidia`` is the presence of a CUDA device,
not the presence of the CUDA toolkit.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any

#: Thread-count variables that change CPU inference performance. Pinned to one
#: value for every engine so a CPU comparison is not silently a comparison of
#: thread budgets, and recorded in the result either way.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class Hardware:
    """The facts an applicability decision is allowed to depend on."""

    platform: str
    os_name: str
    logical_cores: int
    physical_cores: int | None
    ram_bytes: int | None
    nvidia: bool = False
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_vram_bytes: list[int] = field(default_factory=list)
    compute_capabilities: list[str] = field(default_factory=list)
    bf16_native: bool = False
    #: What torch itself answered for ``is_bf16_supported()``. Recorded separately
    #: from ``bf16_native`` because recent builds answer True on pre-Ampere cards
    #: by emulating the format, and the difference decides which engines can run.
    torch_reports_bf16: bool = False
    fp16_native: bool = False
    rocm: bool = False
    mps: bool = False
    threads: int = 1

    @property
    def accelerator(self) -> str:
        if self.nvidia:
            return "cuda"
        if self.rocm:
            return "rocm"
        if self.mps:
            return "mps"
        return "cpu"

    @property
    def smallest_vram_bytes(self) -> int | None:
        return min(self.gpu_vram_bytes) if self.gpu_vram_bytes else None

    def to_dict(self) -> dict[str, Any]:
        record = {
            key: getattr(self, key) for key in (
                "platform", "os_name", "logical_cores", "physical_cores", "ram_bytes",
                "nvidia", "gpu_count", "gpu_names", "gpu_vram_bytes",
                "compute_capabilities", "bf16_native", "torch_reports_bf16",
                "fp16_native", "rocm", "mps", "threads",
            )
        }
        record["accelerator"] = self.accelerator
        return record


def detect() -> Hardware:
    """Read the host capabilities. Never raises; absence is reported as absence."""
    physical = None
    ram = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        ram = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass

    hardware = Hardware(
        platform=platform.machine(),
        os_name=platform.system(),
        logical_cores=os.cpu_count() or 1,
        physical_cores=physical,
        ram_bytes=ram,
    )
    try:
        import torch
    except ImportError:
        return hardware

    hardware.rocm = bool(getattr(torch.version, "hip", None))
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        try:
            hardware.mps = bool(mps_backend.is_available())
        except Exception:  # noqa: BLE001
            hardware.mps = False
    if torch.cuda.is_available():
        # ROCm builds also report through torch.cuda; the hip version above is
        # what distinguishes them, so nvidia stays False for a ROCm host.
        hardware.nvidia = not hardware.rocm
        hardware.gpu_count = torch.cuda.device_count()
        for index in range(hardware.gpu_count):
            properties = torch.cuda.get_device_properties(index)
            hardware.gpu_names.append(properties.name)
            hardware.gpu_vram_bytes.append(int(properties.total_memory))
            hardware.compute_capabilities.append(
                f"{properties.major}.{properties.minor}"
            )
        hardware.fp16_native = True
        # bf16 tensor cores start at compute capability 8.0 (Ampere). Newer torch
        # builds answer ``is_bf16_supported()`` True on older cards because they can
        # emulate the format in software, which is a different claim: several engines
        # in this field (vLLM among them) refuse bf16 below sm_80 outright, so a
        # benchmark that believed torch here would silently exclude them. Both
        # answers are recorded, because a disagreement between them is diagnostic.
        try:
            hardware.torch_reports_bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001
            hardware.torch_reports_bf16 = False
        hardware.bf16_native = all(
            (properties.major, properties.minor) >= (8, 0)
            for properties in (
                torch.cuda.get_device_properties(index)
                for index in range(hardware.gpu_count)
            )
        ) if hardware.gpu_count else False
    else:
        # CPU: torch exposes bf16 arithmetic everywhere, but only AVX512-BF16 or
        # AMX hosts make it fast. Treat it as available, not as native.
        hardware.bf16_native = False
        hardware.fp16_native = False
    hardware.threads = torch.get_num_threads()
    return hardware


def resolve_precision(requested: str, hardware: Hardware) -> tuple[str, str]:
    """Choose the precision every engine will be held to, and say why.

    Resolved from the device, and specifically from the widest 16-bit format the
    *whole field* can execute there. That second clause is the one that matters: a
    precision only one engine supports is not a fair benchmark precision, it is a
    way of excluding engines. On pre-Ampere CUDA that rules out bf16, which several
    engines refuse below compute capability 8.0.

    The resolution is returned with its reason so the report states it instead of
    leaving the reader to infer it.
    """
    if requested != "auto":
        return requested, f"requested explicitly on the command line ({requested})"
    if hardware.nvidia:
        if hardware.bf16_native:
            return "bf16", (
                "every CUDA device here is compute capability 8.0 or newer, so bf16 "
                "runs on tensor cores, and every benchmark checkpoint is published in "
                "bf16 - so no engine has to round another engine's weights"
            )
        capabilities = ", ".join(hardware.compute_capabilities) or "unknown"
        emulated = (
            " torch reports bf16 as supported here, but that is software emulation "
            "rather than a tensor-core path."
            if hardware.torch_reports_bf16 else ""
        )
        return "fp16", (
            f"CUDA device (compute capability {capabilities}) has no bf16 tensor-core "
            f"path; bf16 needs 8.0 or newer.{emulated} Engines in this field refuse "
            "bf16 below 8.0 outright, so choosing it would exclude them rather than "
            "measure them. fp16 is the widest format every engine here executes "
            "natively. The checkpoints are published in bf16, so each engine holds "
            "its own 16-bit rendering of the same source values; that storage "
            "difference is recorded next to every comparison. Pass --precision bf16 "
            "for the weight-exact configuration, at the cost of the engines that "
            "cannot run it"
        )
    return "fp32", (
        "no accelerator detected; fp32 is the only format every CPU engine "
        "executes without an emulation layer"
    )


def pin_threads(threads: int | None) -> dict[str, Any]:
    """Fix the thread budget for this process, and report what was fixed.

    Called in every worker before torch initializes, so each engine gets the same
    budget. When ``threads`` is None the environment is left alone and reported as
    inherited: an uncontrolled but *disclosed* configuration, which is the only
    acceptable alternative to a controlled one.
    """
    if threads is None:
        return {
            "controlled": False,
            "requested": None,
            "env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        }
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(threads)
    applied: int | None = None
    try:
        import torch

        torch.set_num_threads(threads)
        applied = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    return {
        "controlled": True,
        "requested": threads,
        "torch_num_threads": applied,
        "env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }


def default_threads(hardware: Hardware) -> int:
    """The thread budget to pin when the caller does not name one.

    Physical cores, not logical: hyperthread siblings share execution units, so
    counting them inflates the budget without adding arithmetic throughput, and
    an oversubscribed GEMM is slower than a right-sized one.
    """
    return int(hardware.physical_cores or hardware.logical_cores or 1)


def can_hold_weights(
    hardware: Hardware, parameter_count: int | None, precision: str
) -> bool:
    """Whether the model weights plausibly fit the smallest visible accelerator.

    A guard, not a prediction: it exists so a 3.8B checkpoint on a 4 GiB card is
    reported as inapplicable up front instead of failing halfway through a load
    and leaving a partial result.
    """
    if parameter_count is None:
        return True
    smallest = hardware.smallest_vram_bytes
    if smallest is None:
        return True
    per_parameter = 4 if precision == "fp32" else 2
    # Weights plus a working allowance for activations, the KV cache and the
    # allocator's fragmentation; deliberately generous, since a false "fits" is
    # recorded honestly as an OOM while a false "does not fit" silently removes
    # an engine from the field.
    return parameter_count * per_parameter * 1.25 + 512 * 1024 ** 2 < smallest


def min_compute_capability(hardware: Hardware) -> tuple[int, int] | None:
    """The lowest compute capability among visible devices, as a comparable pair.

    Engines state their hardware floor as a capability ("sm_80 or newer"), and a
    probe has to compare against the *weakest* visible device rather than the best,
    because a run that lands on the weaker one would fail.
    """
    parsed: list[tuple[int, int]] = []
    for value in hardware.compute_capabilities:
        try:
            major, _, minor = str(value).partition(".")
            parsed.append((int(major), int(minor or 0)))
        except ValueError:
            continue
    return min(parsed) if parsed else None


def meets_capability(hardware: Hardware, required: tuple[int, int]) -> bool:
    """Whether every visible device is at least ``required``."""
    lowest = min_compute_capability(hardware)
    return lowest is not None and lowest >= required


def visible_devices(count: int | None) -> dict[str, Any]:
    """Restrict this process to the first ``count`` accelerators, and report it.

    Called in the worker before torch initializes a CUDA context, because
    ``CUDA_VISIBLE_DEVICES`` is only read at that point. The reason it exists is
    parity: a runtime that shards across every visible device would be measured on
    more hardware than a runtime that does not, and comparing those two numbers
    would be comparing machines rather than engines.

    This restricts *visibility*, so no engine's placement logic is modified or
    bypassed in its own code - each simply finds one device and does what it does
    with one device.
    """
    if not count or count < 1:
        return {
            "restricted": False,
            "requested": count,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    value = ",".join(str(index) for index in range(count))
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    # Some stacks read their own variable; set the common ones so a single engine
    # cannot quietly opt itself back into the full device set.
    os.environ.setdefault("HIP_VISIBLE_DEVICES", value)
    os.environ["AETHER_EXECUTION_DEVICES"] = ",".join(
        f"cuda:{index}" for index in range(count)
    )
    forced = None
    if count == 1:
        # A stray AETHER_FORCE_TENSOR_PARALLEL in the environment would ask Aether to
        # shard across devices it can no longer see. Cleared, and reported, so the run
        # cannot half-apply a setting the operator forgot they exported.
        forced = os.environ.pop("AETHER_FORCE_TENSOR_PARALLEL", None)
    return {
        "restricted": True,
        "requested": count,
        "CUDA_VISIBLE_DEVICES": value,
        "AETHER_EXECUTION_DEVICES": os.environ["AETHER_EXECUTION_DEVICES"],
        "cleared_force_tensor_parallel": forced,
    }
