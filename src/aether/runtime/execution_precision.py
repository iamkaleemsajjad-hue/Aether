"""Transactional precision changes backed by actual inference measurements.

The older precision managers are planners. This module changes execution
tensors, keeps exact restoration data on disk, and measures a candidate before
committing it. Calibration bounds are observations on the supplied inputs,
never a claim about every possible prompt. Lossy changes are opt-in.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def _file_digest(path: Path) -> str:
    """Hash restoration bytes with bounded host memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_array(path: Path, metadata: dict[str, Any], digest: str) -> np.ndarray:
    """Verify checksum and encoded schema without loading full host tensors."""
    if _file_digest(path) != digest:
        raise ValueError("precision restoration data failed its checksum")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.hasobject
        or value.dtype.str != metadata.get("_encoded_dtype")
        or list(value.shape) != metadata.get("_encoded_shape")
    ):
        raise ValueError("precision restoration data failed its encoded schema")
    return value


@dataclass(frozen=True)
class PrecisionPolicy:
    """Explicit precision, calibration quality and measured latency acceptance bounds."""

    storage_dtype: str = "float16"
    allow_lossy: bool = False
    max_logit_error: float = 0.0
    require_same_argmax: bool = True
    max_slowdown: float = 0.05
    measurements: int = 3

    def __post_init__(self) -> None:
        if self.storage_dtype not in {"float16", "bfloat16"}:
            raise ValueError("precision dtype must be float16 or bfloat16")
        if any(not math.isfinite(v) or v < 0 for v in (self.max_logit_error, self.max_slowdown)):
            raise ValueError("precision quality and latency limits must be finite and non-negative")
        if self.measurements < 1:
            raise ValueError("precision measurements must be positive")


@dataclass
class TensorBinding:
    """A named execution tensor coordinate that can be replaced transactionally."""

    name: str
    get: Callable[[], Any]
    set: Callable[[Any], None]


def _bindings(
    root: Any, *, torch_module: Any = None, prefix: str = "weights"
) -> list[TensorBinding]:
    """Visit structured weights without architecture or checkpoint-name guesses."""
    result: list[TensorBinding] = []
    visited: set[int] = set()

    def visit(
        value: Any, name: str, getter: Callable[[], Any], setter: Callable[[Any], None]
    ) -> None:
        tensor = (
            isinstance(value, np.ndarray)
            if torch_module is None
            else isinstance(value, torch_module.Tensor)
        )
        if tensor:
            result.append(TensorBinding(name, getter, setter))
            return
        if id(value) in visited:
            return
        visited.add(id(value))
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                key = field.name
                visit(
                    getattr(value, key),
                    f"{name}.{key}",
                    lambda o=value, k=key: getattr(o, k),
                    lambda v, o=value, k=key: setattr(o, k, v),
                )
        elif isinstance(value, dict):
            for key in value:
                visit(
                    value[key],
                    f"{name}.{key}",
                    lambda o=value, k=key: o[k],
                    lambda v, o=value, k=key: o.__setitem__(k, v),
                )
        elif isinstance(value, list):
            for key, item in enumerate(value):
                visit(
                    item,
                    f"{name}.{key}",
                    lambda o=value, k=key: o[k],
                    lambda v, o=value, k=key: o.__setitem__(k, v),
                )

    visit(root, prefix, lambda: root, lambda _: None)
    return result


class NumpyPrecisionAdapter:
    """Compact floating projection storage; execution retains FP32 arithmetic."""

    def __init__(self, engine: Any) -> None:
        if not hasattr(engine, "weights") or not callable(getattr(engine, "forward", None)):
            raise ValueError("engine has no mutable structured weights and forward contract")
        self.engine = engine

    def bindings(self) -> list[TensorBinding]:
        """Select FP32 projections while protecting embeddings and their aliases."""
        candidates = _bindings(self.engine.weights)
        # Embedding, output heads and normalization remain at compiled precision.
        # Any aliases of a protected tensor are protected as well.
        protected = {
            id(b.get())
            for b in candidates
            if any(
                part in {"embedding", "lm_head", "position_embedding", "router", "quant_meta"}
                for part in b.name.split(".")
            )
        }
        return [
            b
            for b in candidates
            if id(b.get()) not in protected and b.get().dtype == np.float32 and b.get().ndim >= 2
        ]

    def convert(self, value: np.ndarray, policy: PrecisionPolicy) -> np.ndarray:
        """Create finite compact storage, enforcing lossless conversion by default."""
        if policy.storage_dtype != "float16":
            raise ValueError(
                "NumPy projection storage supports float16; bfloat16 requires a tensor adapter"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = value.astype(np.float16)
        if not np.all(np.isfinite(converted)):
            raise ValueError("precision conversion produced non-finite weights")
        if not policy.allow_lossy and not np.array_equal(value, converted.astype(value.dtype)):
            raise ValueError("weights are not losslessly representable at requested precision")
        return converted

    def encode(self, value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Expose exact typed restoration bytes without pickle."""
        return value, {"dtype": value.dtype.str}

    def decode(self, value: np.ndarray, metadata: dict[str, Any], like: Any) -> np.ndarray:
        """Reconstruct owned NumPy weights at their original dtype."""
        # Restored weights own their bytes; a mapped backup must be closeable
        # and removable on Windows after the transactional replacement.
        return np.array(value, dtype=np.dtype(metadata["dtype"]), copy=True)

    def changed(self, policy: PrecisionPolicy | None) -> None:
        """NumPy linear kernels convert compact weights to FP32 on demand."""

    def evaluate(self, ids: np.ndarray) -> np.ndarray:
        """Execute real prefill and a cached decode step for calibration."""
        logits, cache = self.engine.forward(ids, None)
        decoded, _ = self.engine.forward(ids[-1:], cache)
        return np.concatenate((np.atleast_2d(logits), np.atleast_2d(decoded)), axis=0).astype(
            np.float32
        )

    def synchronize(self) -> None:
        """NumPy execution is synchronous and requires no device fence."""
        pass

    @staticmethod
    def nbytes(value: np.ndarray) -> int:
        """Measure the tensor's storage bytes without inferring allocator usage."""
        return int(value.nbytes)


class TorchPrecisionAdapter:
    """Portable tensor execution changes, including CUDA/HIP/MPS/CPU adapters.

    This requires explicit lossy calibration: changing arithmetic precision may
    alter outputs even when every weight is exactly representable.
    """

    disk_rollback = True

    def __init__(self, engine: Any) -> None:
        if not hasattr(engine, "torch") or not hasattr(engine, "compute_dtype"):
            raise ValueError("engine does not expose a portable tensor precision contract")
        if not hasattr(engine, "layers") or hasattr(engine, "mesh"):
            raise ValueError("sharded precision changes require a collective adapter")
        self.engine = engine
        self.original_dtype = engine.compute_dtype
        self._root = {
            name: getattr(engine, name)
            for name in (
                "embedding",
                "lm_head",
                "position_embedding",
                "layers",
                "final_norm",
                "final_norm_bias",
                "embedding_norm",
                "embedding_norm_bias",
            )
            if hasattr(engine, name)
        }

    def bindings(self) -> list[TensorBinding]:
        """Bind eligible tensors directly to live engine coordinates."""
        result = _bindings(self._root, torch_module=self.engine.torch)

        # Top-level tensors must leave the actual engine immediately. Keeping
        # its old embedding/head until changed() retains large device originals
        # throughout conversion despite the disk rollback contract.
        def top_binding(name: str, field: str) -> TensorBinding:
            def replace(value: Any) -> None:
                self._root[field] = value
                setattr(self.engine, field, value)

            return TensorBinding(name, lambda: getattr(self.engine, field), replace)

        result = [
            top_binding(binding.name, binding.name.split(".")[1])
            if len(binding.name.split(".")) == 2
            else binding
            for binding in result
        ]
        return [
            b
            for b in result
            if b.get().is_floating_point() and b.get().dtype == self.original_dtype
        ]

    def convert(self, value: Any, policy: PrecisionPolicy) -> Any:
        """Change tensor dtype only with explicit arithmetic calibration permission."""
        if not policy.allow_lossy:
            raise ValueError(
                "tensor compute precision changes require explicit lossy calibration permission"
            )
        dtype = getattr(self.engine.torch, policy.storage_dtype)
        converted = value.to(dtype=dtype)
        if not bool(self.engine.torch.isfinite(converted).all().item()):
            raise ValueError("precision conversion produced non-finite weights")
        return converted

    def encode(self, value: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Preserve exact original bits, including BF16, in a typed disk representation."""
        torch = self.engine.torch
        cpu = value.detach().contiguous().cpu()
        if cpu.dtype == torch.bfloat16:
            return cpu.view(torch.uint16).numpy(), {"dtype": "bfloat16"}
        return cpu.numpy(), {"dtype": str(cpu.dtype).split(".")[-1]}

    def decode(self, value: np.ndarray, metadata: dict[str, Any], like: Any) -> Any:
        """Restore owned tensors to the existing device and original dtype."""
        torch = self.engine.torch
        restored = torch.from_numpy(value.copy())
        dtype = getattr(torch, metadata["dtype"])
        if dtype == torch.bfloat16:
            restored = restored.view(dtype)
        return restored.to(device=like.device, dtype=dtype)

    def changed(self, policy: PrecisionPolicy | None) -> None:
        """Refresh execution dtype and invalidate previous rotary/kernel selections."""
        engine = self.engine
        for name, value in self._root.items():
            setattr(engine, name, value)
        engine.compute_dtype = (
            self.original_dtype if policy is None else getattr(engine.torch, policy.storage_dtype)
        )
        for name in ("_cos", "_sin", "_local_cos", "_local_sin"):
            if hasattr(engine, name):
                setattr(engine, name, None)
        # A measured kernel/graph selection at one dtype is invalid at another.
        for name in ("_projection_choices", "_projection_table"):
            if hasattr(engine, name):
                setattr(engine, name, {})
        if hasattr(engine, "_reference_projection"):
            engine._projection = engine._reference_projection
        self.synchronize()

    def evaluate(self, ids: np.ndarray) -> np.ndarray:
        """Run the actual framework prefill and cached decode calibration operators."""
        logits, cache = self.engine.forward(ids, None)
        decoded, _ = self.engine.forward(ids[-1:], cache)
        return np.concatenate((np.atleast_2d(logits), np.atleast_2d(decoded)), axis=0).astype(
            np.float32
        )

    def synchronize(self) -> None:
        """Fence framework accelerator work before timing or restoring tensors."""
        torch, device = self.engine.torch, self.engine.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "xpu":
            torch.xpu.synchronize(device)

    @staticmethod
    def nbytes(value: Any) -> int:
        """Measure logical tensor storage without claiming reserved device-memory savings."""
        return int(value.numel() * value.element_size())


class ExecutionPrecisionController:
    """One-model, reversible transformation; caller holds the model execution lock."""

    def __init__(
        self, adapter: Any, directory: str | Path, policy: PrecisionPolicy | None = None
    ) -> None:
        self.adapter = adapter
        self.policy = policy or PrecisionPolicy()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._backups: list[tuple[list[TensorBinding], Path, dict[str, Any], str]] = []
        self.last_report: dict[str, Any] = {"action": "none", "execution_changed": False}

    def _measure(self, samples: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        # Warm the actual execution path so compilation is excluded from the
        # comparison; no claim of whole-workload speed follows from this probe.
        for ids in samples:
            self.adapter.evaluate(ids)
        elapsed: list[float] = []
        outputs: list[np.ndarray] = []
        for _ in range(self.policy.measurements):
            self.adapter.synchronize()
            start = time.perf_counter()
            outputs = [self.adapter.evaluate(ids) for ids in samples]
            self.adapter.synchronize()
            elapsed.append(time.perf_counter() - start)
        if any(not np.all(np.isfinite(output)) for output in outputs):
            raise ValueError("calibration produced non-finite logits")
        return outputs, statistics.median(elapsed)

    def downgrade(self, calibration_tokens: list[list[int]]) -> dict[str, Any]:
        """Measure and commit compact execution, or restore it when acceptance fails."""
        with self._lock:
            if self._backups:
                return dict(self.last_report, action="already_compact", execution_changed=False)
            samples = [np.asarray(ids, dtype=np.int64) for ids in calibration_tokens]
            if not samples or any(ids.ndim != 1 or ids.size == 0 for ids in samples):
                raise ValueError("real nonempty calibration token sequences are required")
            baseline, baseline_s = self._measure(samples)
            groups: dict[int, list[TensorBinding]] = {}
            for binding in self.adapter.bindings():
                groups.setdefault(id(binding.get()), []).append(binding)
            staged: list[tuple[list[TensorBinding], Any, Path, dict[str, Any], str]] = []
            pending_path: Path | None = None
            before_bytes = after_bytes = 0
            try:
                for bindings in groups.values():
                    original = bindings[0].get()
                    try:
                        converted = self.adapter.convert(original, self.policy)
                    except ValueError:
                        if not self.policy.allow_lossy:
                            continue
                        raise
                    original_bytes = self.adapter.nbytes(original)
                    compact_bytes = self.adapter.nbytes(converted)
                    if compact_bytes >= original_bytes:
                        continue
                    data, metadata = self.adapter.encode(original)
                    metadata = dict(
                        metadata, _encoded_dtype=data.dtype.str, _encoded_shape=list(data.shape)
                    )
                    with tempfile.NamedTemporaryFile(
                        dir=self.directory, suffix=".npy", delete=False
                    ) as file:
                        pending_path = Path(file.name)
                        np.save(file, data, allow_pickle=False)
                        path = pending_path
                    digest = _file_digest(path)
                    # Device originals must not stay resident for the entire
                    # candidate: that would require the full model plus its
                    # compact copy at precisely the point memory is scarce.
                    rollback_value = (
                        None if getattr(self.adapter, "disk_rollback", False) else original
                    )
                    staged.append((bindings, rollback_value, path, metadata, digest))
                    pending_path = None
                    for binding in bindings:
                        binding.set(converted)
                    before_bytes += original_bytes
                    after_bytes += compact_bytes
                    del data, original, converted
                if not staged:
                    self.last_report = {
                        "action": "none",
                        "execution_changed": False,
                        "reason": "no smaller lossless execution representation",
                    }
                    return dict(self.last_report)
                self.adapter.changed(self.policy)
                candidate, candidate_s = self._measure(samples)
                errors = [
                    float(np.max(np.abs(a - b))) for a, b in zip(baseline, candidate, strict=True)
                ]
                same_argmax = all(
                    np.array_equal(np.argmax(a, axis=-1), np.argmax(b, axis=-1))
                    for a, b in zip(baseline, candidate, strict=True)
                )
                max_error = max(errors)
                slowdown = candidate_s / max(baseline_s, 1e-12) - 1.0
                if max_error > self.policy.max_logit_error or (
                    self.policy.require_same_argmax and not same_argmax
                ):
                    raise ValueError(
                        f"measured quality gate rejected precision change (logit error={max_error})"
                    )
                if slowdown > self.policy.max_slowdown:
                    raise ValueError(
                        f"measured latency gate rejected precision change (slowdown={slowdown})"
                    )
                self._backups = [
                    (bindings, path, metadata, digest)
                    for bindings, _, path, metadata, digest in staged
                ]
                self.last_report = {
                    "action": "downgrade",
                    "execution_changed": True,
                    "storage_dtype": self.policy.storage_dtype,
                    "saved_bytes": before_bytes - after_bytes,
                    "memory_scope": (
                        "mutable execution tensors; excludes other aliases "
                        "and allocator reservations"
                    ),
                    "baseline_seconds": baseline_s,
                    "candidate_seconds": candidate_s,
                    "max_logit_error": max_error,
                    "same_argmax": same_argmax,
                    "calibration_sequences": len(samples),
                    "calibration_scope": "supplied token sequences",
                    "lossy_allowed": self.policy.allow_lossy,
                }
                return dict(self.last_report)
            except BaseException as exc:
                for bindings, original, path, metadata, digest in staged:
                    if original is None:
                        original = self.adapter.decode(
                            _backup_array(path, metadata, digest),
                            metadata,
                            bindings[0].get(),
                        )
                    for binding in bindings:
                        binding.set(original)
                    path.unlink(missing_ok=True)
                if pending_path is not None:
                    pending_path.unlink(missing_ok=True)
                self._backups.clear()
                self.adapter.changed(None)
                self.last_report = {
                    "action": "rejected",
                    "execution_changed": False,
                    "reason": str(exc) or type(exc).__name__,
                }
                if not isinstance(exc, Exception):
                    raise
                return dict(self.last_report)

    def restore(self) -> dict[str, Any]:
        """Verify all backups, then restore alias groups with bounded tensor residency."""
        with self._lock:
            # Check every file before the first mutation. A memory-mapped header
            # validates schema without retaining a full original model in RAM.
            for _, path, metadata, digest in self._backups:
                value = _backup_array(path, metadata, digest)
                del value
            committed: list[tuple[list[TensorBinding], Path, dict[str, Any], str]] = []
            try:
                for bindings, path, metadata, digest in self._backups:
                    value = _backup_array(path, metadata, digest)
                    original = self.adapter.decode(value, metadata, bindings[0].get())
                    del value
                    committed.append((bindings, path, metadata, digest))
                    for binding in bindings:
                        binding.set(original)
                    del original
                if committed:
                    self.adapter.changed(None)
            except BaseException:
                # Reconstruct exactly the accepted compact representation from
                # each original, one alias group at a time. Keeping all compact
                # tensors for rollback would recreate the full+compact peak.
                for bindings, path, metadata, digest in reversed(committed):
                    value = _backup_array(path, metadata, digest)
                    original = self.adapter.decode(value, metadata, bindings[0].get())
                    compact = self.adapter.convert(original, self.policy)
                    for binding in bindings:
                        binding.set(compact)
                    del value, original, compact
                if committed:
                    self.adapter.changed(self.policy)
                raise
            for _, path, _, _ in self._backups:
                path.unlink(missing_ok=True)
            self._backups.clear()
            self.last_report = {
                "action": "restore" if committed else "none",
                "execution_changed": bool(committed),
            }
            return dict(self.last_report)

    def close(self) -> None:
        """Restore committed tensors before disposing of restoration files."""
        self.restore()

    def discard(self) -> None:
        """Delete restoration files when the owner is disposing of the engine.

        Unlike close(), this leaves compact execution tensors unchanged. The
        runtime uses it only while unloading the owned model, avoiding a full
        precision allocation solely to destroy that allocation immediately.
        """
        with self._lock:
            for _, path, _, _ in self._backups:
                path.unlink(missing_ok=True)
            self._backups.clear()
            self.last_report = {"action": "discard", "execution_changed": False}

    @property
    def is_compact(self) -> bool:
        """Whether execution currently uses a committed compact representation."""
        with self._lock:
            return bool(self._backups)

    def report(self) -> dict[str, Any]:
        """Return a detached copy of the actual calibration/change evidence."""
        with self._lock:
            return json.loads(json.dumps(self.last_report))
