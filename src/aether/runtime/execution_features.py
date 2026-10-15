"""Runtime lifecycle and execution bridges for optional measured features."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from aether.runtime.execution_precision import (
    ExecutionPrecisionController,
    NumpyPrecisionAdapter,
    PrecisionPolicy,
    TorchPrecisionAdapter,
)
from aether.runtime.kv_payload import CachePayload

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class ModelExecutionGate:
    """Allow concurrent requests; precision writers wait for existing readers."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers = 0
        self._writer = False
        self._waiting = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        """Hold a concurrent request boundary until execution or streaming ends."""
        with self._condition:
            while self._writer or self._waiting:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        """Wait for readers and hold exclusive mutable execution ownership."""
        with self._condition:
            self._waiting += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


def model_fingerprint(handle: Any) -> str:
    """Bind state to the integrity-checked artifact manifest, rather than an alias."""
    manifest = getattr(handle, "manifest", None)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("execution features require an integrity-checked compiled artifact handle")
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_draft_tokenizers(target: Any, draft: Any) -> None:
    """Full token-to-ID and special-token identity, rather than family/size guesses."""
    for handle in (target, draft):
        architecture = handle.manifest.get("architecture", {})
        if architecture.get("is_encoder") or architecture.get("is_encoder_decoder"):
            raise ValueError(
                "speculation requires causal decoder logits and an incremental cache contract"
            )

    def vocabulary(tokenizer: Any) -> dict[str, int]:
        native = getattr(tokenizer, "_tokenizer", tokenizer)
        getter = getattr(native, "get_vocab", None)
        if not callable(getter):
            raise ValueError("tokenizer does not expose its complete token-to-ID mapping")
        return dict(getter())

    if vocabulary(target.tokenizer) != vocabulary(draft.tokenizer):
        raise ValueError("draft and target token-to-ID vocabularies differ")
    for name in ("bos_token_id", "eos_token_id", "eos_token_ids", "pad_token_id", "unk_token_id"):
        if getattr(target.tokenizer, name, None) != getattr(draft.tokenizer, name, None):
            raise ValueError(f"draft and target tokenizer special-token identity differs: {name}")
    target_vocab = target.manifest.get("architecture", {}).get("vocab_size")
    draft_vocab = draft.manifest.get("architecture", {}).get("vocab_size")
    if target_vocab != draft_vocab:
        raise ValueError("draft and target output vocabulary dimensions differ")


class _PooledEngine:
    def __init__(self, engine: Any, pooled: Any, adapter: Any) -> None:
        self.engine, self.pooled, self.adapter = engine, pooled, adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def _payload(self, cache: Any) -> CachePayload | None:
        return (
            cache
            if cache is None or isinstance(cache, CachePayload)
            else self.adapter.export_cache(cache)
        )

    def generate_with_cache(
        self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any
    ) -> tuple[list[int], Any]:
        kwargs["cache"] = self._payload(kwargs.get("cache"))
        tokens, payload = self.pooled.generate_with_cache(prompt_ids, max_tokens, **kwargs)
        return tokens, self.adapter.import_cache(payload)

    def generate_iter(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> Iterator[int]:
        kwargs["cache"] = self._payload(kwargs.get("cache"))
        callback = kwargs.pop("cache_callback", None)
        if callback is not None:
            kwargs["cache_callback"] = lambda payload: callback(self.adapter.import_cache(payload))
        yield from self.pooled.generate_iter(prompt_ids, max_tokens, **kwargs)

    def generate(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens, **kwargs))

    def generation_stats(self) -> dict[str, Any]:
        return self.pooled.generation_stats()


class RuntimeExecutionFeatures:
    """Own measured precision controllers, phase pools and model execution gates."""

    def __init__(self, config: Any, cache_root: Path) -> None:
        self.config, self.cache_root = config, cache_root
        self._lock = threading.RLock()
        self._gates: dict[str, ModelExecutionGate] = {}
        self._controllers: dict[str, ExecutionPrecisionController] = {}
        self._pools: dict[str, tuple[Any, Any, Any, tuple[Any, ...]]] = {}
        self._last_adjustment: dict[str, float] = {}

    def gate(self, model_id: str) -> ModelExecutionGate:
        """Return the shared execution boundary for one loaded model."""
        with self._lock:
            return self._gates.setdefault(model_id, ModelExecutionGate())

    @staticmethod
    def memory_pressure(handle: Any) -> tuple[float | None, str]:
        """Read actual device memory or system RAM, retaining the measured scope."""
        engine = getattr(handle, "engine", None)
        device = getattr(engine, "device", None)
        try:
            if device is not None and getattr(device, "type", None) == "cuda":
                free, total = engine.torch.cuda.mem_get_info(device)
                return (1.0 - free / total, "device_memory") if total else (None, "unknown")
            import psutil

            info = psutil.virtual_memory()
            return (
                (1.0 - info.available / info.total, "system_ram")
                if info.total
                else (None, "unknown")
            )
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
            return None, "unknown"

    def adjust_precision(
        self,
        model_id: str,
        handle: Any,
        *,
        calibration_tokens: list[list[int]] | None = None,
        restore: bool = False,
    ) -> dict[str, Any]:
        """Measure, commit or restore tensors under exclusive model ownership."""
        if not self.config.dynamic_precision:
            raise ValueError("dynamic precision is disabled")
        with self.gate(model_id).write():
            engine = getattr(handle, "engine", None)
            if engine is None:
                raise ValueError("this backend has no mutable compiled execution engine")
            architecture = getattr(handle, "manifest", {}).get("architecture", {})
            if architecture.get("is_encoder") or architecture.get("is_encoder_decoder"):
                raise ValueError(
                    "precision calibration requires the causal incremental forward contract"
                )
            controller = self._controllers.get(model_id)
            if controller is None:
                adapter = (
                    TorchPrecisionAdapter(engine)
                    if hasattr(engine, "torch")
                    else NumpyPrecisionAdapter(engine)
                )
                policy = PrecisionPolicy(
                    storage_dtype=self.config.precision_storage_dtype,
                    allow_lossy=self.config.precision_allow_lossy,
                    max_logit_error=self.config.precision_max_logit_error,
                    max_slowdown=self.config.precision_max_slowdown,
                )
                controller = ExecutionPrecisionController(
                    adapter, self.cache_root / "precision" / model_fingerprint(handle), policy
                )
                self._controllers[model_id] = controller
            if controller.is_compact and not restore:
                return controller.downgrade([])
            if not controller.is_compact and restore:
                return controller.restore()
            self._close_pool(model_id)
            report = (
                controller.restore()
                if restore
                else controller.downgrade(
                    calibration_tokens
                    if calibration_tokens is not None
                    else self.config.precision_calibration_tokens
                )
            )
            self._last_adjustment[model_id] = time.monotonic()
            if report["execution_changed"]:
                clear_sessions = getattr(handle, "clear_session_caches", None)
                if callable(clear_sessions):
                    clear_sessions()
                else:
                    handle.session_caches.clear()
                handle.batched_engine = None
                handle.metadata["_precision_epoch"] = (
                    int(handle.metadata.get("_precision_epoch", 0)) + 1
                )
            return report

    def automatic_precision(self, model_id: str, handle: Any) -> None:
        """Apply calibrated policy only when actual memory pressure warrants it."""
        if not self.config.dynamic_precision or not self.config.precision_calibration_tokens:
            return
        if self._precision_unsupported_reason(handle) is not None:
            return
        now = time.monotonic()
        if (
            now - self._last_adjustment.get(model_id, float("-inf"))
            < self.config.precision_cooldown_seconds
        ):
            return
        pressure, _ = self.memory_pressure(handle)
        if pressure is None or 0.60 < pressure < 0.90:
            return
        if pressure <= 0.60 and model_id not in self._controllers:
            return
        self.adjust_precision(model_id, handle, restore=pressure <= 0.60)

    def wrap_engine(self, model_id: str, handle: Any, engine: Any, request: Any) -> Any:
        """Construct a compatible phase-pool or explicit-draft execution bridge."""
        draft = request.extra.get("_explicit_draft_engine")
        if draft is not None and not self.config.disaggregate_prefill_decode:
            from aether.runtime.speculative_engine import SpeculativeGenerationAdapter

            return SpeculativeGenerationAdapter(engine, draft, gamma=self.config.speculative_gamma)
        if not self.config.disaggregate_prefill_decode:
            return engine
        if self.config.draft_model_id:
            raise ValueError(
                "combining separate phase pools with a speculative draft needs "
                "a decode-worker draft adapter"
            )
        if engine is not handle.engine or any(
            request.extra.get(key) is not None
            for key in ("ttt_engine", "peagle_engine", "task_weights", "adapter_id", "adapter_name")
        ):
            raise ValueError(
                "separate phase pools require workers constructed with the request's "
                "active weight/adapter contract"
            )
        from aether.runtime.disaggregated_execution import (
            DisaggregatedExecutionPool,
            PooledGenerationAdapter,
        )

        with self._lock:
            resident = self._pools.get(model_id)
            signature = (
                id(engine),
                tuple(self.config.prefill_devices or []),
                tuple(self.config.decode_devices or []),
                self.config.disaggregate_prefill_workers,
                self.config.disaggregate_decode_workers,
                self.config.disaggregate_queue_capacity,
                self.config.prefill_chunk_size,
            )
            if resident is None or resident[3] != signature:
                self._close_pool(model_id)
                fingerprint = model_fingerprint(handle)
                custom_factory = getattr(engine, "create_phase_adapter_factory", None)
                if callable(custom_factory):
                    factory = custom_factory(model_fingerprint=fingerprint)
                elif hasattr(engine, "torch"):
                    from aether.runtime.torch_cache_handoff import make_torch_adapter_factory

                    factory = make_torch_adapter_factory(engine, model_fingerprint=fingerprint)
                else:
                    import numpy as np

                    from aether.runtime.cache_handoff import export_cpu_cache
                    from aether.runtime.disaggregated_execution import make_cpu_adapter_factory

                    _, probe_cache = engine.forward(np.asarray([0], dtype=np.int64), None)
                    kind = export_cpu_cache(probe_cache, model_fingerprint=fingerprint).cache_kind
                    factory = make_cpu_adapter_factory(
                        engine, model_fingerprint=fingerprint, cache_kind=kind
                    )
                prefill = self.config.prefill_devices
                decode = self.config.decode_devices
                if hasattr(engine, "torch"):
                    from aether.runtime.torch_cache_handoff import make_torch_adapter_factory

                    # Each device hosts owned executor state, preserving the
                    # loaded tensor dtypes. These are phase replicas, not a
                    # guessed tensor-parallel mesh.
                    prefill_factories = [
                        make_torch_adapter_factory(
                            engine, model_fingerprint=fingerprint, device_id=device
                        )
                        for device in (prefill or [str(engine.device)])
                        for _ in range(self.config.disaggregate_prefill_workers)
                    ]
                    decode_factories = [
                        make_torch_adapter_factory(
                            engine, model_fingerprint=fingerprint, device_id=device
                        )
                        for device in (decode or [str(engine.device)])
                        for _ in range(self.config.disaggregate_decode_workers)
                    ]
                else:
                    if any(
                        mesh is not None and mesh != [str(getattr(engine, "device", "cpu"))]
                        for mesh in (prefill, decode)
                    ):
                        raise ValueError(
                            "phase placement differs from this provider's device; "
                            "an explicit device factory is required"
                        )
                    prefill_factories = [factory] * self.config.disaggregate_prefill_workers
                    decode_factories = [factory] * self.config.disaggregate_decode_workers
                pool = DisaggregatedExecutionPool(
                    prefill_factories,
                    decode_factories,
                    max_pending=self.config.disaggregate_queue_capacity,
                    prefill_chunk_size=self.config.prefill_chunk_size,
                )
                pooled = PooledGenerationAdapter(pool)
                try:
                    adapter = factory()
                except BaseException:
                    pool.close()
                    raise
                resident = (pool, pooled, adapter, signature)
                self._pools[model_id] = resident
            return _PooledEngine(engine, resident[1], resident[2])

    def _close_pool(self, model_id: str) -> None:
        with self._lock:
            resident = self._pools.get(model_id)
        if resident is not None:
            try:
                resident[0].close()
            finally:
                resident[2].close()
            with self._lock:
                if self._pools.get(model_id) is resident:
                    self._pools.pop(model_id)

    def report(self, model_id: str, handle: Any) -> dict[str, Any]:
        """Report actual arithmetic precision and scoped memory/change evidence."""
        pressure, scope = self.memory_pressure(handle)
        report: dict[str, Any] = {"memory_pressure": pressure, "memory_pressure_scope": scope}
        engine = getattr(handle, "engine", None)
        dtype = getattr(engine, "compute_dtype", None)
        if dtype is not None:
            report["active_precision"] = str(dtype).removeprefix("torch.")
        elif engine is not None:
            from aether.runtime.cpu_engine import CPUExecutionEngine
            from aether.runtime.hybrid_engine import HybridExecutionEngine

            if isinstance(engine, (CPUExecutionEngine, HybridExecutionEngine)):
                report["active_precision"] = "float32"
            else:
                precision = getattr(getattr(engine, "bundle", None), "metadata", {}).get(
                    "arithmetic_precision"
                )
                if precision == "fp32":
                    report["active_precision"] = "float32"
        if model_id in self._controllers:
            report["precision_adjustment"] = self._controllers[model_id].report()
        elif self.config.dynamic_precision and self.config.precision_calibration_tokens:
            reason = self._precision_unsupported_reason(handle)
            if reason is not None:
                report["precision_adjustment"] = {
                    "action": "unsupported",
                    "execution_changed": False,
                    "reason": reason,
                }
        return report

    @staticmethod
    def _precision_unsupported_reason(handle: Any) -> str | None:
        engine = getattr(handle, "engine", None)
        architecture = getattr(handle, "manifest", {}).get("architecture", {})
        if architecture.get("is_encoder") or architecture.get("is_encoder_decoder"):
            return "precision calibration requires the causal incremental forward contract"
        try:
            if hasattr(engine, "torch"):
                TorchPrecisionAdapter(engine)
            else:
                NumpyPrecisionAdapter(engine)
        except ValueError as exc:
            return str(exc)
        return None

    def unload(self, model_id: str) -> None:
        """Join workers and discard backups for an engine the owner is unloading."""
        with self.gate(model_id).write():
            self._close_pool(model_id)
            controller = self._controllers.get(model_id)
            if controller is not None:
                controller.discard()
                self._controllers.pop(model_id, None)
            self._last_adjustment.pop(model_id, None)

    def close(self) -> None:
        """Release every owned execution feature before backend destruction."""
        for model_id in set(self._gates) | set(self._pools) | set(self._controllers):
            self.unload(model_id)
