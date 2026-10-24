"""
Aether Runtime — the main execution engine.

The Runtime is the primary Python API. It loads compiled AEG artifacts, detects
hardware, selects the best backend, manages the KV cache, runs speculative
decoding, and serves generation requests.
"""

from __future__ import annotations

import atexit
import datetime
import json
import copy
import os
import sys
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

from aether.backends.base import Backend, GenerationRequest, GenerationResult
from aether.backends.artifact_validation import ArtifactValidationError, validate_aeg_evaluation
from aether.backends.registry import BackendRegistry
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.core.constants import DEFAULT_SERVER_PORT
from aether.core.exceptions import BackendError, BackendNotAvailableError, ModelNotFoundError, RuntimeError as AetherRuntimeError
from aether.core.types import HardwareTarget
from aether.runtime.config import RuntimeConfig
from aether.runtime.hardware import HardwareDetector
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.lifecycle import (
    RuntimeLifecycle, runtime_lifecycle, runtime_operation, runtime_exclusive, runtime_stream,
    runtime_session_operation,
)
from aether.runtime.scheduler import DisaggregatedScheduler, ScheduledRequest
from aether.runtime.speculative import TreeSpeculativeEngine
from aether.utils.file_io import aether_cache_dir, resolve_model_path, safe_model_id_path
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics returned with an inference response."""

    throughput_tps: float = 0.0
    """Tokens per second."""

    ttft_ms: float = 0.0
    """Time to first token in milliseconds."""

    p95_latency_ms: float | None = None
    """P95 decode latency in milliseconds."""

    kernel_target: str | None = None
    """Active hardware target (e.g., 'cuda_sm90')."""

    active_precision: str | None = None
    """Observed arithmetic dtype; None when the backend has no measured dtype contract."""

    spec_accept_rate: float | None = None
    """Speculative decoding acceptance rate."""

    kv_cache_hit_rate: float | None = None
    """KV cache prefix hit rate."""

    memory_pressure: float | None = None
    """Measured memory pressure; extra.memory_pressure_scope identifies device memory or system RAM."""

    backend_name: str | None = None
    """Backend plugin that executed the request."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional measured/request metadata such as cascade routing."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "throughput_tps": self.throughput_tps,
            "ttft_ms": self.ttft_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "kernel_target": self.kernel_target,
            "active_precision": self.active_precision,
            "spec_accept_rate": self.spec_accept_rate,
            "kv_cache_hit_rate": self.kv_cache_hit_rate,
            "memory_pressure": self.memory_pressure,
            "backend_name": self.backend_name,
            **self.extra,
        }


@dataclass
class GenerationResponse:
    """Normalized response from `Runtime.generate()`."""

    text: str
    """Generated text."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage (prompt_tokens, completion_tokens, total_tokens)."""

    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    """Inference metrics."""

    finish_reason: str = "stop"
    """Generation finish reason."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage,
            "metrics": self.metrics.to_dict(),
            "finish_reason": self.finish_reason,
        }


class AttestationReport(dict[str, Any]):
    """Mapping-compatible attestation report with PRD attribute access.

    The REST API needs a JSON mapping, while the documented Python SDK uses
    ``report.model_hash`` and ``report.enclave_measurement``.  Keeping one
    object that supports both prevents callers from receiving different
    semantics depending on the transport.
    """

    def __getattr__(self, name: str) -> Any:
        if name == "enclave_measurement":
            return self.get("tdx_report_hash") or self.get("snp_report_hash")
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class EvalGateReport(dict[str, Any]):
    """Mapping-compatible evaluation result with SDK attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _MultiAgentAgent:
    """Async facade for one real Runtime-backed agent in an R2 session."""

    def __init__(
        self,
        runtime: "Runtime",
        session: "_MultiAgentSessionHandle",
        model_id: str,
        session_id: str,
        context: str = "",
        prefix_hash: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._session = session
        self.model_id = model_id
        self.session_id = session_id
        self.context = context
        self.prefix_hash = prefix_hash

    async def generate(self, prompt: str, **kwargs: Any) -> GenerationResponse:
        """Run inference through the parent Runtime for this agent."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("agent prompt must be a non-empty string")
        full_prompt = f"{self.context}\n\n{prompt}" if self.context else prompt
        if self.context and self.prefix_hash:
            kwargs.setdefault("multi_agent_kv_coordinator", self._session._coordinator)
            kwargs.setdefault("multi_agent_prefix", f"{self.context}\n\n")
            kwargs.setdefault("multi_agent_prefix_hash", self.prefix_hash)
        return self._runtime.generate(self.model_id, full_prompt, **kwargs)


class _MultiAgentSessionHandle(dict[str, Any]):
    """Dictionary-compatible async context manager for R2 multi-agent KV."""

    _VALID_COORDINATION = frozenset({"relay", "kvcomm", "droidspeak", "swarm"})

    def __init__(
        self,
        runtime: "Runtime",
        models: list[str],
        coordination: str,
        agent_count: int,
        shared_prefix: str,
    ) -> None:
        if coordination not in self._VALID_COORDINATION:
            raise ValueError(
                f"unsupported coordination mode {coordination!r}; "
                f"choose one of {sorted(self._VALID_COORDINATION)}"
            )
        super().__init__()
        from aether.runtime.r2_multi_agent_kv import MultiAgentKVCoordinator

        self._runtime = runtime
        self._coordinator = getattr(runtime, "_multi_agent_coordinator", None)
        if self._coordinator is None:
            self._coordinator = MultiAgentKVCoordinator()
            runtime._multi_agent_coordinator = self._coordinator
        self._models = list(dict.fromkeys(models))
        self._coordination = coordination
        self._shared_prefix = shared_prefix
        self._shared_prefix_hash = (
            self._coordinator.hash_prefix(shared_prefix) if shared_prefix else None
        )
        self._session_ids: set[str] = set()
        self._agents: dict[str, _MultiAgentAgent] = {}
        self._closed = False
        session_id = str(uuid.uuid4())
        self.update(
            {
                "session_id": session_id,
                "agent_sessions": [],
                "agent_count": 0,
                "models": list(self._models),
                "coordination": coordination,
                "shared_prefix_len": len(shared_prefix),
                "kv_sharing_enabled": True,
                "research_basis": "SGLang RadixAttention 2024 + MemServe 2025",
            }
        )
        if agent_count:
            for index in range(agent_count):
                selected = self._models[index % len(self._models)] if self._models else ""
                self._create_agent(selected, context=shared_prefix)

    def _create_agent(
        self,
        model_id: str,
        *,
        context: str = "",
        prefix_hash: str | None = None,
    ) -> _MultiAgentAgent:
        if self._closed:
            raise AetherRuntimeError("multi-agent session is closed")
        if self._models and model_id not in self._models:
            raise ValueError(f"model {model_id!r} is not registered in this session")
        agent_session_id = f"{self['session_id']}/agent_{len(self._session_ids)}"
        effective_hash = prefix_hash or self._shared_prefix_hash
        self._coordinator.create_agent_session(
            session_id=agent_session_id,
            prefix_hash=effective_hash,
        )
        agent = _MultiAgentAgent(
            self._runtime,
            self,
            model_id,
            agent_session_id,
            context=context,
            prefix_hash=effective_hash,
        )
        self._session_ids.add(agent_session_id)
        self._agents[agent_session_id] = agent
        self["agent_sessions"].append(agent_session_id)
        self["agent_count"] = len(self._session_ids)
        return agent

    async def spawn_agent(
        self,
        model: str,
        *,
        context: Any = "",
        inherit_kv_from: _MultiAgentAgent | None = None,
    ) -> _MultiAgentAgent:
        """Create an agent backed by the registered model and R2 coordinator."""
        if not isinstance(model, str) or not model:
            raise ValueError("spawn_agent requires a non-empty model identifier")
        context_text = context if isinstance(context, str) else json.dumps(context, sort_keys=True)
        inherited_hash = (
            inherit_kv_from.prefix_hash
            if isinstance(inherit_kv_from, _MultiAgentAgent)
            else None
        )
        return self._create_agent(
            model,
            context=context_text,
            prefix_hash=inherited_hash,
        )

    def close(self) -> None:
        """Release all coordinator sessions owned by this handle."""
        if self._closed:
            return
        for session_id in tuple(self._session_ids):
            self._coordinator.release_session(session_id)
        self._session_ids.clear()
        self._closed = True
        self["status"] = "closed"

    async def __aenter__(self) -> "_MultiAgentSessionHandle":
        if self._closed:
            raise AetherRuntimeError("multi-agent session is closed")
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class _AgenticRuntimeSession:
    """Async conversation wrapper with explicit session lifecycle.

    The wrapper preserves conversation context by sending the accumulated
    messages through the real Runtime.  It records a session in the agentic
    KV manager, but does not claim zero-prefill reuse until the selected
    backend exposes actual K/V tensors.
    """

    def __init__(self, runtime: "Runtime", model_id: str, system: str) -> None:
        self._runtime = runtime
        self.model_id = model_id
        self.system = system
        self.session_id = str(uuid.uuid4())
        self._manager: Any | None = None
        self._history: list[tuple[str, str]] = []
        self._closed = False

    async def __aenter__(self) -> "_AgenticRuntimeSession":
        from aether.runtime.agentic_session import AgenticKVSessionManager

        if self._closed:
            raise AetherRuntimeError("agentic session is closed")
        self._manager = getattr(self._runtime, "_agentic_session_manager", None)
        if self._manager is None:
            self._manager = AgenticKVSessionManager()
            self._runtime._agentic_session_manager = self._manager
        # Token IDs are intentionally not fabricated here.  The manager still
        # tracks lifecycle; actual prefix KV registration waits for backend
        # tensor hooks.
        self._manager.create_session(self.session_id, metadata={"model_id": self.model_id})
        return self

    async def generate(self, prompt: str, **kwargs: Any) -> GenerationResponse:
        if self._closed:
            raise AetherRuntimeError("agentic session is closed")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("agentic prompt must be a non-empty string")
        turns = []
        if self.system:
            turns.append(f"System: {self.system}")
        for role, text_value in self._history:
            turns.append(f"{role.title()}: {text_value}")
        turns.append(f"User: {prompt}")
        # Keep the assistant role marker in the cached prompt.  Subsequent
        # turns then contain the exact token prefix (including the marker)
        # before the prior generated tokens, which makes CPU KV reuse safe.
        turns.append("Assistant:")
        kwargs["aether_kv_session_id"] = self.session_id
        response = self._runtime.generate(self.model_id, "\n\n".join(turns), **kwargs)
        self._history.extend([("user", prompt), ("assistant", response.text)])
        response.metrics.extra.update(
            {
                "agentic_session_id": self.session_id,
                "agentic_turn": len(self._history) // 2,
            }
        )
        # Backends that do not expose a session cache remain explicit rather
        # than inheriting a misleading positive claim.
        response.metrics.extra.setdefault("kv_reuse", False)
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._runtime._release_agentic_kv(self.model_id, self.session_id)
        if self._manager is not None:
            self._manager.close_session(self.session_id)
        self._closed = True

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class Runtime:
    """Aether runtime API.

    Usage:
        rt = Runtime()
        response = rt.generate("Qwen/Qwen3-0.6B", "Explain AEG.")
        print(response.text)
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        """Initialize the Aether runtime.

        Args:
            config: Runtime configuration, or None for defaults.
        """
        self.config = config or RuntimeConfig()
        self.hardware_detector = HardwareDetector()
        self.fingerprint = self.hardware_detector.detect()
        self.backend_registry = BackendRegistry()
        self.target_registry = TargetRegistry()
        payload_store = None
        if self.config.kv_cache_path:
            from aether.runtime.kv_storage import KVPayloadStore

            payload_store = KVPayloadStore(self.config.kv_cache_path,
                max_bytes=int(self.config.kv_cache_nvme_gb * 1024**3))
        self.kv_cache = KVCacheManager(
            dtype=self.config.kv_cache_dtype,
            cpu_budget_gb=self.config.kv_cache_cpu_gb,
            nvme_budget_gb=self.config.kv_cache_nvme_gb,
            payload_store=payload_store,
        )
        self.scheduler = DisaggregatedScheduler(
            max_batch_size=self.config.max_batch_size,
            prefill_chunk_size=self.config.prefill_chunk_size,
        )
        self.slo_scheduler: Any | None = None
        if self.config.scheduler == "slo_aware":
            from aether.runtime.r4_slo_scheduler import SLOScheduler

            self.slo_scheduler = SLOScheduler(
                max_batch_tokens=self.config.max_batch_size * self.config.prefill_chunk_size,
                max_prefill_chunk_tokens=self.config.prefill_chunk_size,
            )
        self._loaded_models: dict[str, Any] = {}
        self._loaded_backends: dict[str, Backend] = {}
        self._model_targets: dict[str, str] = {}
        # Artifact paths for models this process has loaded, as
        # ``{model_id: (aeg_path, resolved_path)}``.  Written only by
        # :meth:`_load_model` and cleared only by :meth:`remove`, so an entry
        # exists exactly while the model it describes is loaded.
        self._aeg_paths: dict[str, tuple[str | None, str | None]] = {}
        self._aeg_packages: dict[str, AEGPackage] = {}
        self._lock = threading.RLock()
        self._lifecycle = RuntimeLifecycle()
        from aether.runtime.execution_features import RuntimeExecutionFeatures

        feature_root = Path(self.config.model_cache_dir) if self.config.model_cache_dir else aether_cache_dir()
        self._execution_features = RuntimeExecutionFeatures(self.config, feature_root)
        self._draft_runtime: Runtime | None = None
        self._draft_pairs: set[tuple[int, int, int, int]] = set()
        self._stored_sessions: dict[tuple[str, str], str] = {}
        from aether.observability.otel import AetherTracer, MetricsCollector

        self.tracer = AetherTracer()
        self.metrics_collector = MetricsCollector()
        self.safety_engine: Any | None = None
        if self.config.enable_safety_layer:
            from aether.safety.policy import ContentPolicyEngine

            safety_root = Path(self.config.model_cache_dir) if self.config.model_cache_dir else aether_cache_dir()
            self.safety_engine = ContentPolicyEngine(
                audit_path=safety_root / "safety" / "audit.jsonl",
            )

        # v4.0 runtime layer handles — initialized lazily on first model load
        # or when explicitly enabled via RuntimeConfig.
        self.grammar_engine: Any | None = None
        """R3 Grammar FSM Engine (Pass 11). Set by _init_v4_layers()."""

        self.ttt_engine: Any | None = None
        """R5 TTT Fast-Weight Engine (Pass 13). Set by _init_v4_layers()."""

        self.tee_manager: Any | None = None
        """R8 TEE Runtime Manager (Pass 17). Set by _init_v4_layers()."""

        self.green_power_manager: Any | None = None
        """R7 Green Power Manager (Pass 16). Set by _init_v4_layers()."""

        self.mcp_layer: Any | None = None
        """R6 MCP Integration Layer. Set by _init_v4_layers()."""

        self.peagle_engine: Any | None = None
        """R1 P-EAGLE engine loaded from persisted MTP speculation blobs."""
        self._peagle_engines: dict[str, Any] = {}

        # Async compilation job registry: {job_id: {status, model, ...}}
        self._compile_jobs: dict[str, dict[str, Any]] = {}

        # Register an atexit handler that joins all pending compile threads
        # before the interpreter tears down C extension state.
        #
        # Root cause of exit-139 (SIGSEGV): compile_async() spawns daemon
        # threads that call into C extensions (the Aether compiler). Daemon
        # threads are *not* joined by the interpreter on exit — they are
        # abandoned at whatever point they have reached. If a thread is in
        # the middle of a C extension call when the interpreter begins
        # finalizing (freeing module globals, calling Py_Finalize), the
        # thread can dereference already-freed memory → SIGSEGV (exit 139).
        #
        # atexit handlers run *before* C extension teardown, so joining here
        # gives in-flight threads a chance to finish cleanly. We use a weakref
        # so the handler itself does not prevent the Runtime from being
        # garbage-collected in long-running processes that create many runtimes.
        _self_ref: weakref.ref[Runtime] = weakref.ref(self)

        def _join_compile_threads() -> None:
            rt = _self_ref()
            if rt is None:
                return
            try:
                rt.close(timeout=30)
            except Exception as exc:
                logger.warning("Runtime exit cleanup did not complete", error=str(exc))
            for _job in list(rt._compile_jobs.values()):
                _thread = _job.get("_thread")
                if isinstance(_thread, threading.Thread) and _thread.is_alive():
                    # 30 s is generous: a real compile can be slow on first
                    # run, but for CI smoke-tests the job always fails fast
                    # (model does not exist) so the join returns almost
                    # immediately.
                    _thread.join(timeout=30)

        atexit.register(_join_compile_threads)

        logger.info(
            "Aether runtime initialized",
            target=self.fingerprint.target_id,
            backend=self.config.backend_name,
            optimize_for=self.config.optimize_for,
        )

    def hardware(self) -> dict[str, Any]:  # type: ignore[valid-type]
        """Return the current hardware fingerprint."""
        return self.fingerprint.to_dict()

    def _resolve_backend(
        self,
        target_id: str | None = None,
        model_id: str | None = None,
        aeg_path: str | None = None,
    ) -> Backend:
        """Resolve a compatible backend; automatic selection may use CPU."""
        if target_id is None:
            return self._resolve_model_backend(model_id)[0]
        with self._lock:
            target = target_id
            profile = HardwareProfile.from_target_id(target)
            if profile is None:
                raise BackendNotAvailableError(f"Unknown execution target {target!r}", target_id=target)
            explicit_backend = self.config.backend_name is not None
            is_onnx_model = bool(
                model_id and Path(model_id).suffix.lower() == ".onnx"
            )
            if self.config.backend_name:
                backend = self.backend_registry.get_backend(self.config.backend_name)
                if self._backend_accepts_model(backend, profile.target_id, model_id, aeg_path):
                    return backend
                msg = (
                    f"Configured backend '{self.config.backend_name}' is not available "
                    f"for target {profile.target_id}"
                )
                raise BackendNotAvailableError(msg, backend_name=self.config.backend_name)
            # Prefer cached backend
            if profile.target_id in self._loaded_backends:
                cached = self._loaded_backends[profile.target_id]
                if (
                    (cached.name != "onnx" or is_onnx_model or explicit_backend)
                    and self._backend_accepts_model(cached, profile.target_id, model_id, aeg_path)
                ):
                    return cached

            candidates = profile.backend_candidates
            for backend_name in candidates:
                if backend_name in {"onnx", "onnxruntime"} and not is_onnx_model:
                    continue
                backend = self.backend_registry.get_backend(backend_name)
                if self._backend_accepts_model(backend, profile.target_id, model_id, aeg_path):
                    self._loaded_backends[profile.target_id] = backend
                    return backend
            # A backend for another device is not a valid fallback.
            for backend in self.backend_registry.get_available_backends():
                if backend.name == "onnx" and not is_onnx_model:
                    continue
                if self._backend_accepts_model(backend, profile.target_id, model_id, aeg_path):
                    self._loaded_backends[profile.target_id] = backend
                    return backend

            msg = f"No backend available for target {target}"
            raise BackendNotAvailableError(msg, target_id=target)

    @staticmethod
    def _backend_accepts_model(
        backend: Backend | None, target: str, model_id: str | None, aeg_path: str | None = None,
    ) -> bool:
        if backend is None or not backend.available_for_target(target):
            return False
        accepts = getattr(backend, "available_for_model", None)
        if model_id is None or not callable(accepts):
            return True
        return bool(accepts(model_id, aeg_path=aeg_path)) if aeg_path else bool(accepts(model_id))

    def _execution_candidates(
        self, *, allow_portable_accelerators: bool = False,
    ) -> tuple[list[str], list[str] | None]:
        """Intersect detected targets with an explicit execution mesh, once at load.

        Hardware discovery is cached in the fingerprint. It is neither a backend
        readiness test nor evidence of execution. Accelerator preference is applied
        only after an artifact or model adapter authorizes that target.
        """
        capabilities = list(self.fingerprint.capabilities)
        if allow_portable_accelerators and any(
            item.get("target_id") == "rocm_unknown"
            and item.get("runtime_available") and item.get("hardware_present")
            for item in capabilities
        ):
            # Stable HIP C APIs may expose a real device without its GFX
            # identifier. A ready, authorized Torch runtime can supply that
            # missing architecture from actual device properties. Never infer
            # CDNA/RDNA from a marketing name or alter detector validation flags.
            torch_backend = self.backend_registry.get_backend("pytorch")
            if torch_backend is not None and torch_backend.is_available():
                import torch
                from aether.backends.hardware_detector import inspect_torch_hip_devices

                capabilities.extend(
                    item.to_dict() for item in inspect_torch_hip_devices(torch)
                    if item.implemented and item.available
                )
        available = list(dict.fromkeys([
            self.fingerprint.target_id,
            *self.fingerprint.attributes.get("available_targets", []),
            *[
                item["target_id"] for item in capabilities
                if item.get("available") and item.get("target_id")
            ],
        ]))
        if not any(target.startswith("cpu_") for target in available):
            from aether.backends.hardware_detector import detect_cpu

            available.append(detect_cpu().target_id)
        devices = self.config.execution_devices
        if devices is None:
            devices = [
                item.strip() for item in os.environ.get("AETHER_EXECUTION_DEVICES", "").split(",")
                if item.strip()
            ] or None
        if devices is not None:
            devices = ["cpu" if device == "cpu:0" else device for device in devices]
            if any(device.startswith("directml:") for device in devices):
                from aether.backends.hardware_inventory import probe_gpu_inventory

                onnx_backend = self.backend_registry.get_backend("onnx")
                adapters = probe_gpu_inventory().get("devices", [])
                ordinals = {record["directml_device_index"] for record in adapters if "directml_device_index" in record}
                requested = [device for device in devices if device.startswith("directml:")]
                if (onnx_backend is None or "DmlExecutionProvider" not in onnx_backend.available_providers()
                    or any(not device[9:].isdigit() or int(device[9:]) not in ordinals for device in requested)):
                    raise BackendNotAvailableError(f"Requested DirectML adapters {requested} are not enumerated with a registered execution provider")
                if "directml" not in available:
                    available.append("directml")
            prefixes = {
                "cpu": ("cpu_",), "cuda": ("cuda_", "rocm_", "amd_mi"), "mps": ("metal",),
                "directml": ("directml",),
            }
            kinds = {device.split(":", 1)[0] for device in devices}
            malformed = any(
                device != "cpu" and device != "mps" and not (
                    device.startswith("cuda:") and device[5:].isdigit()
                    or device.startswith("directml:") and device[9:].isdigit()
                )
                for device in devices
            )
            if not devices or len(set(devices)) != len(devices) or malformed or not kinds.issubset(prefixes):
                raise BackendNotAvailableError(f"Invalid execution devices {devices!r}")
            if any(
                not any(target.startswith(prefixes[kind]) for target in available)
                for kind in kinds
            ):
                raise BackendNotAvailableError(
                    f"Requested execution devices {devices} are unavailable; host targets: {available}"
                )
            # An ordinal belongs to the executing runtime, not NVML's physical
            # order. Where detectors provide that mapping, use its architecture.
            indexed = [
                item for item in capabilities
                if item.get("available")
                and str(item.get("extra", {}).get("execution_device") or "").split(":", 1)[0] in kinds - {"cpu"}
            ]
            accelerators = [device for device in devices if device != "cpu"]
            if accelerators and indexed:
                known_devices = {item["extra"]["execution_device"] for item in indexed}
                missing = [device for device in accelerators if device not in known_devices]
                if missing:
                    raise BackendNotAvailableError(f"Requested execution devices {missing} are unavailable")
                selected_targets = {
                    item["target_id"] for item in indexed
                    if item["extra"]["execution_device"] in accelerators
                }
                available = [target for target in available if target in selected_targets]
            else:
                # CPU accompanying an accelerator is an offload participant;
                # it must not override the requested accelerator execution.
                execution_kinds = kinds - {"cpu"} or {"cpu"}
                available = [
                    target for target in available
                    if any(target.startswith(prefixes[kind]) for kind in execution_kinds)
                ]
        return sorted(available, key=lambda target: target.startswith("cpu_")), devices

    def _resolve_model_backend(self, model_id: str | None) -> tuple[Backend, str, list[str] | None]:
        from aether.backends.model_contract import model_format

        targets, devices = self._execution_candidates(
            allow_portable_accelerators=model_id is not None and model_format(model_id) == "source",
        )
        failures = []
        for target in targets:
            try:
                backend = self._resolve_backend(target_id=target, model_id=model_id)
            except BackendNotAvailableError as exc:
                failures.append(str(exc))
                continue
            automatic = getattr(type(backend), "automatic_execution_devices", None)
            selected_devices = devices
            if selected_devices is None:
                selected_devices = automatic(backend, target) if callable(automatic) else (["cpu"] if target.startswith("cpu_") else None)
            return backend, target, selected_devices
        raise BackendNotAvailableError(
            "No compatible model execution backend is available; " + "; ".join(failures),
            backend_name=self.config.backend_name,
            target_id=self.fingerprint.target_id,
        )

    def _resolve_aeg_backend(
        self, package: AEGPackage, model_id: str, excluded_targets: set[str] | None = None,
    ) -> tuple[Backend, str, list[str] | None]:
        """Intersect artifact contracts, host devices, and installed backends.

        Prefer a ready compatible accelerator, including an explicitly authorized
        portable graph, and otherwise use the artifact's CPU execution contract.
        Explicit backend and device requests are constraints, not preferences.
        """
        available, devices = self._execution_candidates(
            allow_portable_accelerators=package.supports_portable_backend("pytorch"),
        )
        if package.supports_portable_backend("onnx") and devices is None:
            onnx_backend = self.backend_registry.get_backend("onnx")
            if onnx_backend is not None and "DmlExecutionProvider" in onnx_backend.available_providers():
                cpu_position = next((i for i, target in enumerate(available) if target.startswith("cpu_")), len(available))
                if "directml" not in available:
                    available.insert(cpu_position, "directml")
        candidates = [target for target in available if package.supports_runtime_target(target)
                      and target not in (excluded_targets or set())]
        failures = []
        for target in candidates:
            if (
                not self.config.backend_name
                and not target.startswith("cpu_")
                and package.manifest is not None
                and package.manifest.kernels.variant_status.get(target) == "executable"
            ):
                plan_backend = package.get_backend_plan(target)
                native = self.backend_registry.get_backend(plan_backend) if plan_backend else None
                if self._backend_accepts_model(native, target, model_id, str(package.root)):
                    return native, target, devices
            if self.config.backend_name:
                name = self.config.backend_name
            elif target.startswith("cpu_"):
                name = "aether_cpu"
            elif target in {"directml", "openvino_gpu", "openvino_npu"} and package.supports_portable_backend("onnx"):
                name = "onnx"
            elif package.supports_portable_backend("pytorch"):
                name = "pytorch"
            else:
                try:
                    backend = self._resolve_backend(
                        target_id=target, model_id=model_id, aeg_path=str(package.root),
                    )
                except BackendNotAvailableError as exc:
                    failures.append(str(exc))
                    continue
                return backend, target, devices
            backend = self.backend_registry.get_backend(name)
            if self._backend_accepts_model(backend, target, model_id, str(package.root)):
                return backend, target, devices or (["cpu"] if target.startswith("cpu_") else None)
            failures.append(f"backend {name!r} is not installed or available for {target!r}")
        compiled = package.manifest.kernels.targets if package.manifest else []
        raise BackendNotAvailableError(
            f"AEG artifact has no executable variant with an available backend for this request; "
            f"compiled variants: {compiled}; host targets: {available}; "
            + ("; ".join(failures) if failures else "compile a compatible executable target"),
            backend_name=self.config.backend_name,
            target_id=self.fingerprint.target_id,
        )

    def _load_onnx_model(
        self, model_id: str, load_options: dict[str, Any],
    ) -> tuple[Backend, Any, str]:
        """Initialize the requested ONNX provider without hidden CPU fallback.

        Registration only creates a candidate. A failed automatic accelerator
        initialization can try CPU; an explicit accelerator request must fail.
        Session initialization is not recorded as hardware execution validation.
        """
        configured_backend = self.config.backend_name
        if configured_backend and BackendRegistry._normalize_name(configured_backend) != "onnx":
            raise BackendNotAvailableError(
                f"Configured backend {configured_backend!r} cannot execute a raw .onnx model",
                backend_name=configured_backend,
            )
        backend = self.backend_registry.get_backend("onnx")
        if backend is None or not backend.is_available():
            raise BackendNotAvailableError(
                "ONNX Runtime is required to execute .onnx models", backend_name="onnx",
            )
        targets, devices = self._execution_candidates()
        if devices and len([device for device in devices if device != "cpu"]) > 1:
            raise BackendNotAvailableError("ONNX provider execution supports one accelerator device per session")
        registered = set(backend.available_providers())
        # DirectML has its own adapter ordinal and a raw ONNX contract. It can
        # be considered even before detection has proven a functioning session.
        if devices is None and "DmlExecutionProvider" in registered and "directml" not in targets:
            cpu_position = next((index for index, target in enumerate(targets) if target.startswith("cpu_")), len(targets))
            targets.insert(cpu_position, "directml")
        attempts = []
        for target in targets:
            if target.startswith("cpu_"):
                providers = ["CPUExecutionProvider"]
            elif target.startswith("cuda_"):
                index = next((int(device.split(":", 1)[1]) for device in devices or [] if device.startswith("cuda:")), 0)
                providers = [("CUDAExecutionProvider", {"device_id": index})]
            elif target.startswith(("rocm_", "amd_mi")):
                index = next((int(device.split(":", 1)[1]) for device in devices or [] if device.startswith("cuda:")), 0)
                provider = "MIGraphXExecutionProvider" if "MIGraphXExecutionProvider" in registered else "ROCMExecutionProvider"
                providers = [(provider, {"device_id": index})]
            elif target in {"openvino_gpu", "openvino_npu"}:
                providers = [("OpenVINOExecutionProvider", {"device_type": "GPU" if target.endswith("gpu") else "NPU"})]
            elif target == "directml":
                index = next((int(device[9:]) for device in devices or [] if device.startswith("directml:")), 0)
                providers = [("DmlExecutionProvider", {"device_id": index})]
            else:
                continue
            names = [entry[0] if isinstance(entry, tuple) else entry for entry in providers]
            if not set(names).issubset(registered):
                attempts.append(f"{target}: providers {names} are not registered")
                continue
            try:
                model = backend.load_model(
                    model_id, providers=providers, allow_provider_fallback=False, **load_options,
                )
            except BackendError as exc:
                if devices is not None:
                    raise
                attempts.append(f"{target}: {exc}")
                logger.warning("ONNX provider initialization failed for %s: %s", target, exc)
                continue
            return backend, model, target
        raise BackendNotAvailableError(
            "No compatible ONNX provider could initialize this model; " + "; ".join(attempts),
            backend_name="onnx",
        )

    @runtime_operation
    def _load_model(self, model_id: str) -> Backend:
        """Ensure a model is loaded and return the backend that loaded it."""
        with self._lock:
            if model_id in self._loaded_models:
                backend = self._loaded_backends.get(model_id)
                if backend:
                    return backend
            # Find or compile AEG
            aeg_path = self._resolve_aeg_path(model_id)
            requested_path = Path(model_id)
            local_onnx = requested_path.is_file() and requested_path.suffix.lower() == ".onnx"
            if aeg_path is None and not local_onnx and (
                requested_path.suffix.lower() in {".aeg", ".aegpkg"}
                or (
                    not requested_path.exists()
                    and (requested_path.is_absolute() or model_id.startswith(("./", "../", ".\\", "..\\")))
                )
            ):
                raise ModelNotFoundError(
                    f"AEG artifact does not exist or has no manifest: {model_id}",
                    model_id=model_id,
                )
            package = None
            target = self.fingerprint.target_id
            execution_devices = self.config.execution_devices
            load_options = {
                **({"allow_uncertified": True} if self.config.allow_uncertified else {}),
                "offline": self.config.hf_offline,
                "download_timeout_s": self.config.model_download_timeout_s,
                "trust_remote_code": self.config.allow_remote_code,
                "retain_host_weights": bool(self.config.disaggregate_prefill_decode
                    and (self.config.prefill_devices is not None or self.config.decode_devices is not None)),
            }
            if aeg_path is not None:
                try:
                    package = AEGPackage(Path(aeg_path)).load()
                except Exception as exc:
                    raise AetherRuntimeError(
                        f"Unable to validate AEG target compatibility for {aeg_path}: {exc}"
                    ) from exc
                validate_aeg_evaluation(package, allow_uncertified=self.config.allow_uncertified)
            # A local ONNX file has an explicit execution contract. Do not
            # route it to the generic PyTorch backend merely because that
            # backend is available on the host.
            if requested_path.suffix.lower() == ".onnx":
                backend, loaded_model, target = self._load_onnx_model(model_id, load_options)
            elif package is not None:
                backend, target, execution_devices = self._resolve_aeg_backend(package, model_id)
            else:
                backend, target, execution_devices = self._resolve_model_backend(model_id)
            if requested_path.suffix.lower() != ".onnx":
                attempted: set[str] = set()
                while True:
                    try:
                        loaded_model = backend.load_model(
                            model_id, aeg_path, execution_devices=execution_devices,
                            **({"target_id": target} if backend.name == "onnx" else {}), **load_options,
                        )
                        break
                    except BackendError as exc:
                        # A detected device is not proof that this particular
                        # graph/provider can execute. Automatic selection can
                        # try another authorized target; explicit constraints
                        # always preserve the initialization error.
                        if isinstance(exc, ArtifactValidationError):
                            raise
                        if package is None or self.config.backend_name or self.config.execution_devices is not None:
                            raise
                        if os.environ.get("AETHER_EXECUTION_DEVICES", "").strip():
                            raise
                        attempted.add(target)
                        logger.warning("AEG backend initialization failed for %s: %s", target, exc)
                        backend, target, execution_devices = self._resolve_aeg_backend(
                            package, model_id, excluded_targets=attempted,
                        )
            self._loaded_models[model_id] = loaded_model
            self._loaded_backends[model_id] = backend
            self._model_targets[model_id] = target
            self._aeg_paths[model_id] = (
                aeg_path,
                str(Path(aeg_path).resolve()) if aeg_path is not None else None,
            )
            # Optional v4/v5 layers are initialized at the same reachability
            # boundary as model loading.  They remain feature-gated by the
            # artifact manifest, but no longer exist only in isolated stats
            # helpers while inference bypasses them.
            if aeg_path is not None:
                self._init_v4_layers(aeg_path)
                self._init_v5_layers(aeg_path)
            return backend

    def _resolve_aeg_path(self, model_id: str) -> str | None:
        """Find the cached AEG package for a model reference.

        Resolution order, most explicit first:

        1. an explicit path the caller gave (a directory holding a ``manifest.json``);
        2. the canonical cache path from :func:`aeg_cache_candidates`, which is the
           *same* function the compiler writes through — so a model compiled by ID is
           discoverable by that ID, which is the invariant this method exists to
           uphold;
        3. earlier cache layouts, so upgrading Aether never orphans an artifact a
           user already has on disk.

        A candidate must contain a ``manifest.json`` to count. Accepting a bare
        directory, as this once did, meant a partially written or unrelated directory
        resolved here and then failed further downstream with an error that pointed
        at execution rather than at the artifact.
        """
        from aether.utils.file_io import aeg_cache_candidates

        path_candidate = Path(model_id)
        if path_candidate.is_dir() and (path_candidate / "manifest.json").is_file():
            return str(path_candidate.resolve())

        for candidate in aeg_cache_candidates(model_id, self.config.model_cache_dir):
            if (candidate / "manifest.json").is_file():
                return str(candidate)

        local = resolve_model_path(model_id, self.config.model_cache_dir)
        if local and (local / "manifest.json").is_file():
            return str(local)
        return None

    def _prepare_execution_features(self, request: GenerationRequest) -> Any:
        """Validate request-local weights before delegating model execution.

        Frontend plugins may opt in through explicit ``task_reweighting`` or
        ``lora_adapter_selection`` capabilities; each opt-in promises to apply
        the corresponding validated request inputs. Dependency presence alone
        does not establish this execution contract.
        """
        from aether.backends.compiled_handle import CompiledAEGHandle
        from aether.runtime.session_contract import validate_task_weights

        handle = self._loaded_models.get(request.model_id)
        backend = self._loaded_backends.get(request.model_id)
        backend_name = getattr(backend, "name", self.config.backend_name)
        try:
            coefficients = validate_task_weights(request.extra.get("task_weights"))
        except ValueError as exc:
            raise BackendError(str(exc), backend_name=backend_name) from exc
        adapter_id = request.extra.get("adapter_id", request.extra.get("adapter_name"))
        if request.extra.get("adapter_id") is not None and request.extra.get("adapter_name") is not None and request.extra["adapter_id"] != request.extra["adapter_name"]:
            raise BackendError("adapter_id and adapter_name select different adapters", backend_name=backend_name)
        if adapter_id is not None and (not isinstance(adapter_id, str) or not adapter_id):
            raise BackendError("adapter identity must be a nonempty string", backend_name=backend_name)
        if coefficients or adapter_id is not None:
            if isinstance(handle, CompiledAEGHandle):
                if coefficients:
                    vectors = handle.metadata.get("task_vectors", {})
                    catalog = vectors.get("vectors", []) if isinstance(vectors, dict) else []
                    available = {item.get("name") for item in catalog
                                 if isinstance(item, dict) and isinstance(item.get("name"), str)} if isinstance(catalog, list) else set()
                    try:
                        validate_task_weights(coefficients, available=available)
                    except ValueError as exc:
                        raise BackendError(str(exc), backend_name=backend_name) from exc
                    if not callable(getattr(handle.engine, "with_task_deltas", None)):
                        raise BackendError("the loaded engine has no task-vector application contract", backend_name=backend_name)
                if adapter_id is not None:
                    if adapter_id not in handle.lora_adapters:
                        raise BackendError(f"compiled LoRA adapter {adapter_id!r} is absent", backend_name=backend_name)
                    if not callable(getattr(handle.engine, "with_lora_adapter", None)):
                        raise BackendError("the loaded engine has no compiled LoRA application contract", backend_name=backend_name)
            else:
                get_capabilities = getattr(backend, "get_capabilities", None)
                try:
                    capabilities = get_capabilities() if callable(get_capabilities) else []
                except NotImplementedError:
                    capabilities = []
                if not isinstance(capabilities, (list, tuple, set, frozenset)) or not all(isinstance(item, str) for item in capabilities):
                    raise BackendError("backend returned an invalid request-state capability contract", backend_name=backend_name)
                required = (["task_reweighting"] if coefficients else []) + (["lora_adapter_selection"] if adapter_id is not None else [])
                if any(capability not in capabilities for capability in required):
                    raise BackendError(
                        f"backend {backend_name!r} has no request-state execution contract for {required}",
                        backend_name=backend_name,
                    )
        if getattr(handle, "engine", None) is None:
            if self.config.disaggregate_prefill_decode or self.config.draft_model_id:
                raise BackendError("this backend has no compiled phase/cache execution contract", backend_name=self.config.backend_name)
            return handle
        request.extra["_execution_features"] = self._execution_features
        precision_epoch = handle.metadata.get("_precision_epoch", 0)
        self._execution_features.automatic_precision(request.model_id, handle)
        if handle.metadata.get("_precision_epoch", 0) != precision_epoch:
            self._stored_sessions = {pair: key for pair, key in self._stored_sessions.items()
                                     if pair[0] != request.model_id}
        session_id = request.extra.get("aether_kv_session_id")
        stored = self._stored_sessions.get((request.model_id, session_id)) if isinstance(session_id, str) else None
        if stored is not None:
            self.restore_session(request.model_id, session_id, stored)
        if self.config.draft_model_id:
            with self._lock:
                if self._draft_runtime is None:
                    draft_config = self.config.to_dict()
                    draft_config.update(backend_name=self.config.draft_backend_name,
                        execution_devices=self.config.draft_execution_devices,
                        draft_model_id=None, dynamic_precision=False,
                        disaggregate_prefill_decode=False, enable_semantic_cache=False,
                        kv_cache_path=None, ttt_enabled=False)
                    self._draft_runtime = Runtime(RuntimeConfig.from_dict(draft_config))
            draft_runtime = self._draft_runtime
            draft_runtime._load_model(self.config.draft_model_id)
            draft = draft_runtime._loaded_models[self.config.draft_model_id]
            if getattr(draft, "engine", None) is None or getattr(draft, "tokenizer", None) is None:
                raise BackendError("speculative draft requires a loaded incremental engine and tokenizer")
            from aether.runtime.execution_features import validate_draft_tokenizers

            pair = (id(handle), id(draft), id(handle.tokenizer), id(draft.tokenizer))
            with self._lock:
                if pair not in self._draft_pairs:
                    validate_draft_tokenizers(handle, draft)
                    self._draft_pairs.add(pair)
            request.extra["_explicit_draft_engine"] = draft.engine
            request.extra["spec_gamma"] = self.config.speculative_gamma
        return handle

    def _generate_with_features(self, backend: Backend, request: GenerationRequest) -> GenerationResult:
        lifecycle = runtime_lifecycle(self)
        with lifecycle.session(request.model_id, request.extra.get("aether_kv_session_id")) as key:
            try:
                with lifecycle.preparing_session(key):
                    handle = self._prepare_execution_features(request)
                with self._execution_features.gate(request.model_id).read():
                    result = backend.generate(request)
                    result.metrics.update(self._execution_features.report(request.model_id, handle))
                return result
            except BaseException:
                if key is not None:
                    handle = self._loaded_models.get(request.model_id)
                    clear = getattr(handle, "clear_session_cache", None)
                    if callable(clear):
                        clear(key[1])
                raise

    def _stream_with_features(self, backend: Backend, request: GenerationRequest) -> Iterator[str]:
        lifecycle = runtime_lifecycle(self)
        with lifecycle.session(request.model_id, request.extra.get("aether_kv_session_id")) as key:
            completed = False
            try:
                with lifecycle.preparing_session(key):
                    self._prepare_execution_features(request)
                with self._execution_features.gate(request.model_id).read():
                    iterator = backend.generate_stream(request)
                    try:
                        yield from iterator
                        completed = True
                    finally:
                        close = getattr(iterator, "close", None)
                        if callable(close):
                            close()
            finally:
                # Ordinary engines can mutate a reused cache before their
                # final callback. Canceled/error state cannot be committed
                # safely without the complete consumed token prefix.
                if not completed and key is not None:
                    handle = self._loaded_models.get(request.model_id)
                    clear = getattr(handle, "clear_session_cache", None)
                    if callable(clear):
                        clear(key[1])

    @runtime_operation
    def adjust_precision(self, model_id: str, calibration_tokens: list[list[int]] | None = None,
                         *, restore: bool = False) -> dict[str, Any]:
        """Measure and apply a reversible change; calibration is not a universal quality certificate."""
        self._load_model(model_id)
        report = self._execution_features.adjust_precision(model_id, self._loaded_models[model_id],
            calibration_tokens=calibration_tokens, restore=restore)
        if report["execution_changed"]:
            self._stored_sessions = {pair: key for pair, key in self._stored_sessions.items() if pair[0] != model_id}
        return report

    @runtime_session_operation
    def persist_session(self, model_id: str, session_id: str, *, tier: str = "disk") -> dict[str, Any]:
        """Persist exact target state; rebuild speculative drafts from the saved prefix."""
        from aether.core.types import MemoryTier
        from aether.runtime.execution_features import model_fingerprint
        from aether.runtime.kv_payload import CachePayload
        from aether.runtime.session_contract import validate_persisted_contract

        if tier not in {"disk", "hub"}:
            raise ValueError("session storage tier must be disk or hub")
        self._load_model(model_id)
        handle = self._loaded_models[model_id]
        with self._execution_features.gate(model_id).write():
            cached = getattr(handle, "session_caches", {}).get(session_id)
            if cached is None:
                raise ValueError("session has no live execution cache")
            ids, cache = cached
            contract = validate_persisted_contract(handle, handle.session_contracts.get(session_id))
            from aether.runtime.spec_generation import SpeculativeSessionCache

            speculative_resume = {}
            if isinstance(cache, SpeculativeSessionCache):
                if cache.source_identity[0] != handle.session_target_ids.get(session_id):
                    raise ValueError("speculative target cache belongs to a different execution engine")
                if tuple(ids.tolist()) != cache.token_ids or cache.length != len(ids):
                    raise ValueError("speculative target cache differs from its recorded token prefix")
                # The target owns the authoritative decoding state. Draft state
                # is acceleration-only and can be reconstructed by a freshly
                # tokenizer-validated draft from this exact consumed prefix.
                cache = cache.target_cache
                speculative_resume = {"speculative_resume": {
                    "target_state": "lossless", "draft_state": "rebuild_from_token_prefix"
                }}
            fingerprint = model_fingerprint(handle)
            if hasattr(handle.engine, "torch"):
                from aether.runtime.torch_cache_handoff import export_torch_cache

                payload = export_torch_cache(cache, engine=handle.engine, model_fingerprint=fingerprint)
            else:
                from aether.runtime.cache_handoff import export_cpu_cache

                payload = export_cpu_cache(cache, model_fingerprint=fingerprint)
            payload = CachePayload(payload.model_fingerprint, payload.cache_kind, payload.logical_length,
                payload.tensors, {**payload.metadata, "session_token_ids": ids.tolist(),
                                 "precision_epoch": handle.metadata.get("_precision_epoch", 0),
                                 "execution_contract": contract,
                                 **speculative_resume})
            block = self.kv_cache.import_payload(payload)
            try:
                self.kv_cache.swap(block.block_id, MemoryTier.L3_NVME if tier == "disk" else MemoryTier.L4_HUB)
            except Exception:
                self.kv_cache.remove(block.block_id)
                raise
            key = self.kv_cache.payload_key(block.block_id)
            handle.clear_session_cache(session_id)
            self._stored_sessions[model_id, session_id] = key
            return {"key": key, "tier": tier, "tensor_bytes": payload.byte_size,
                    "logical_length": payload.logical_length, "model_fingerprint": fingerprint,
                    "execution_contract": contract,
                    **speculative_resume}

    @runtime_operation
    def configure_cache_hub(self, client: Any, namespace: str) -> None:
        """Bind an explicit authenticated Hub client; this does not initiate I/O."""
        if not isinstance(namespace, str) or not namespace or any(
            not callable(getattr(client, method, None)) for method in ("put_kv", "get_kv", "delete_kv")):
            raise ValueError("a cache namespace and a Hub byte-storage client are required")
        self.kv_cache.hub_client = client
        self.kv_cache.hub_namespace = namespace

    @runtime_session_operation
    def restore_session(self, model_id: str, session_id: str, key: str) -> dict[str, Any]:
        """Restore a verified snapshot, including after process restart."""
        import numpy as np
        from aether.runtime.execution_features import model_fingerprint
        from aether.runtime.session_contract import validate_persisted_contract

        self._load_model(model_id)
        handle = self._loaded_models[model_id]
        # Callers restoring automatically already own a request preparation
        # boundary; this writer runs before the request acquires its read gate.
        with self._execution_features.gate(model_id).write():
            fingerprint = model_fingerprint(handle)
            block = self.kv_cache.block_for_payload_key(key)
            if block is None:
                if self.kv_cache.hub_client is None or not self.kv_cache.hub_namespace:
                    raise ValueError("cache snapshot key is not present in configured storage")
                payload = self.kv_cache.hub_client.get_kv(self.kv_cache.hub_namespace, key,
                                                        expected_model_fingerprint=fingerprint)
            else:
                payload = self.kv_cache.export_payload(block.block_id, expected_model_fingerprint=fingerprint)
            ids = payload.metadata.get("session_token_ids")
            if not isinstance(ids, list) or len(ids) != payload.logical_length or any(
                type(token) is not int or token < 0 for token in ids):
                raise ValueError("persisted session token prefix does not match its logical cache length")
            if payload.metadata.get("precision_epoch", 0) != handle.metadata.get("_precision_epoch", 0):
                raise ValueError("persisted session uses a different execution precision state")
            contract = validate_persisted_contract(handle, payload.metadata.get("execution_contract"))
            if hasattr(handle.engine, "torch"):
                from aether.runtime.torch_cache_handoff import restore_torch_cache

                cache = restore_torch_cache(payload, expected_model_fingerprint=fingerprint, engine=handle.engine)
            else:
                from aether.runtime.cache_handoff import restore_cpu_cache

                cache = restore_cpu_cache(payload, expected_model_fingerprint=fingerprint)
            handle.save_session_cache(session_id, np.asarray(ids, dtype=np.int64), cache, contract=contract)
            if block is not None:
                self.kv_cache.release_payload_residency(block.block_id)
            self._stored_sessions.pop((model_id, session_id), None)
            return {"logical_length": payload.logical_length, "tensor_bytes": payload.byte_size,
                    "model_fingerprint": fingerprint, "execution_contract": contract}

    def close(self, *, timeout: float = 10.0) -> None:
        """Join owned phase workers, discard restoration files, and unload models."""
        lifecycle = runtime_lifecycle(self)
        claimed, errors = lifecycle.begin_close(timeout)
        if not claimed:
            return
        try:
            self._execution_features.close()
            deadline = time.monotonic() + timeout
            for job in list(self._compile_jobs.values()):
                worker = job.get("_thread")
                if isinstance(worker, threading.Thread) and worker.is_alive():
                    if worker is threading.current_thread():
                        raise RuntimeError("An executing compiler job cannot close its owning Runtime")
                    worker.join(max(0, deadline - time.monotonic()))
                    if worker.is_alive():
                        raise TimeoutError("Runtime compiler job is still executing; shutdown has not completed")
            if self._draft_runtime is not None:
                self._draft_runtime.close(timeout=timeout)
                self._draft_runtime = None
            for attribute, method in (("mcp_layer", "disconnect_all"), ("_cxl_pool", "close")):
                owner = getattr(self, attribute, None)
                if owner is None:
                    continue
                try:
                    close_owner = getattr(owner, method, None)
                    if callable(close_owner):
                        close_owner()
                    setattr(self, attribute, None)
                except BaseException as error:
                    errors.append(error)
            for model_id in list(self._loaded_models):
                backend = self._loaded_backends.get(model_id)
                unload = getattr(backend, "unload_model", None)
                try:
                    if callable(unload):
                        unload(model_id)
                    self._loaded_models.pop(model_id, None)
                    self._loaded_backends.pop(model_id, None)
                    self._model_targets.pop(model_id, None)
                    self._aeg_paths.pop(model_id, None)
                except BaseException as error:
                    errors.append(error)
            if not self._loaded_models:
                self._loaded_backends.clear()
                self._aeg_packages.clear()
                self._peagle_engines.clear()
                self._draft_pairs.clear()
                self._stored_sessions.clear()
                for attribute in ("peagle_engine", "ttt_engine", "grammar_engine", "tee_manager",
                                  "_diffusion_engine", "_semantic_cache", "_multi_agent_coordinator"):
                    setattr(self, attribute, None)
                for attribute in ("_task_weights_by_model", "_task_weights"):
                    weights = getattr(self, attribute, None)
                    if isinstance(weights, dict):
                        weights.clear()
                # A caller may retain this closed Runtime/manager object.
                # Release owned host snapshots while keeping committed disk
                # and Hub keys available to a new Runtime after restart.
                self.kv_cache.discard_resident_payloads()
            if errors:
                raise RuntimeError(f"Runtime resource cleanup failed: {errors[0]}") from errors[0]
        except BaseException:
            lifecycle.finish_close(False)
            raise
        lifecycle.finish_close(True)

    def __enter__(self) -> Runtime:
        with runtime_lifecycle(self).operation():
            return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _loaded_aeg_path(self, model_id: str) -> tuple[str | None, str | None]:
        """The artifact path for a model this process has already loaded.

        Returns ``(aeg_path, resolved_path)``.  :meth:`_load_model` resolved both
        and recorded the answer, so a second request for the same model does not
        walk the cache layouts again: that walk is several ``is_file`` probes plus a
        ``Path.resolve``, it sits on the critical path *ahead* of the prompt, and its
        cost belongs to the host filesystem rather than to Aether — an overlay or
        network mount charges far more for a stat than local disk does.

        On a miss the resolution happens exactly as before.  The memo is filled by
        loading a model, never consulted in place of resolving one, which keeps
        discovery a property of the filesystem rather than of process state.
        """
        with self._lock:
            memo = self._aeg_paths.get(model_id)
        if memo is not None:
            return memo
        path = self._resolve_aeg_path(model_id)
        return path, (str(Path(path).resolve()) if path is not None else None)

    def pull(self, model_id: str) -> None:
        """Download and compile a model to a local AEG package.

        If the AEG already exists in cache, this is a no-op. Otherwise, it
        invokes the compiler to produce a local AEG artifact.
        """
        aeg_path = self._resolve_aeg_path(model_id)
        if aeg_path is not None:
            logger.info(f"Model {model_id} already available at {aeg_path}")
            return
        logger.info(f"Pulling and compiling model {model_id}")
        from aether import Compiler, CompilerConfig
        compiler = Compiler(CompilerConfig(
            cache_dir=str(aether_cache_dir(self.config.model_cache_dir)),
            quality_mode="diagnostic" if self.config.allow_uncertified else "require_hf",
        ))
        # No explicit output path: the compiler derives the canonical cache path
        # from the same naming contract this class resolves through, so passing one
        # here could only reintroduce a divergence between writer and reader.
        compiler.compile(model_id)

    def list(self) -> list[str]:
        """Return a list of model IDs with cached AEG artifacts."""
        from aether.utils.file_io import aether_cache_dir

        cache_root = aether_cache_dir(self.config.model_cache_dir)
        model_dir = cache_root / "models"
        if not model_dir.is_dir():
            return []
        # Directory names are returned as-is rather than de-mangled back to a
        # Hugging Face ID: ``--`` stands in for ``/``, but a repo name may itself
        # contain ``--``, so reversing it is ambiguous. Every name returned here is
        # accepted by :meth:`_resolve_aeg_path`, which is the property that matters —
        # output of ``aether list`` can be fed straight back to info/graph/run.
        return sorted(
            entry.name
            for entry in model_dir.iterdir()
            if (entry / "manifest.json").is_file()
        )

    def info(self, model_id: str) -> dict[str, Any]:
        """Return metadata and precision info for a compiled model."""
        aeg_path = self._resolve_aeg_path(model_id)
        if aeg_path is None:
            msg = f"Model {model_id} not found"
            raise ModelNotFoundError(msg, model_id=model_id)
        aeg = load_aeg_package(aeg_path)
        if not aeg.manifest:
            msg = f"AEG for {model_id} has no manifest"
            raise AetherRuntimeError(msg)
        return {
            "model_id": aeg.manifest.model_id,
            "format_version": aeg.manifest.format_version,
            "aether_version": aeg.manifest.aether_version,
            "architecture": aeg.manifest.architecture.to_dict(),
            "targets": aeg.manifest.kernels.targets,
            "variant_status": dict(aeg.manifest.kernels.variant_status),
            "portable_backends": list(aeg.manifest.kernels.portable_backends),
            "precision_map": aeg.get_precision_map(),
            "memory": aeg.manifest.memory_requirements.to_dict(),
            "sharding_plans": {k: v.to_dict() for k, v in aeg.sharding_plans.items()},
        }

    @runtime_exclusive
    def remove(self, model_id: str) -> None:
        """Remove a model from the local AEG cache."""
        from aether.utils.file_io import cached_model_paths, delete_model

        paths = cached_model_paths(model_id, self.config.model_cache_dir)
        if not paths:
            raise ModelNotFoundError(f"Cached model {model_id} not found", model_id=model_id)
        with self._lock:
            aliases = {model_id}
            for alias, (_, resolved) in self._aeg_paths.items():
                if resolved and Path(resolved) in paths:
                    aliases.add(alias)
            errors = runtime_lifecycle(self).cancel_streams(aliases)
            if errors:
                raise RuntimeError(f"Runtime stream cleanup failed: {errors[0]}") from errors[0]
            for alias in aliases:
                self._execution_features.unload(alias)
                backend = self._loaded_backends.get(alias)
                unload = getattr(backend, "unload_model", None)
                if callable(unload):
                    unload(alias)
            delete_model(model_id, self.config.model_cache_dir)
            for alias in aliases:
                self._loaded_models.pop(alias, None)
                self._loaded_backends.pop(alias, None)
                self._model_targets.pop(alias, None)
                self._aeg_paths.pop(alias, None)
        logger.info(f"Removed model {model_id} from cache")

    @runtime_operation
    def generate(
        self,
        model_id: str,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int = 0,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate text from a model.

        Args:
            model_id: Model identifier or local AEG path.
            prompt: Text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            stream: Whether to stream output.
            stop: Stop sequences.
            kwargs: Additional backend parameters.

        Returns:
            A GenerationResponse with text, usage, and metrics.
        """
        if prompt is None and messages is None:
            raise ValueError("either prompt or messages must be provided")
        safety_text = prompt or "\n".join(
            str(message.get("content", ""))
            for message in (messages or [])
            if isinstance(message, dict)
        )
        if safety_text:
            self._check_safety_prompt(safety_text)
        slo_deadline_ms = kwargs.pop("slo_deadline_ms", None)
        slo_request: Any | None = None
        if self.slo_scheduler is not None:
            from aether.runtime.r4_slo_scheduler import SLOTier

            requested_tier = kwargs.pop("slo_tier", SLOTier.BALANCED)
            try:
                slo_request = self.slo_scheduler.submit(
                    request_id=uuid.uuid4().hex,
                    prompt_tokens=len(safety_text.split()),
                    max_new_tokens=max_tokens or self.config.default_max_tokens,
                    slo_tier=requested_tier,
                    metadata={"model_id": model_id},
                    ttft_deadline_ms=slo_deadline_ms,
                )
                selected = self.slo_scheduler.next_batch()
                if not any(item.request_id == slo_request.request_id for item in selected):
                    raise AetherRuntimeError("SLO scheduler did not admit the request")
            except (ValueError, KeyError) as exc:
                raise AetherRuntimeError(f"invalid SLO tier {requested_tier!r}") from exc
        task_weights = self._effective_task_weights(model_id, kwargs)
        if task_weights is not None:
            kwargs["task_weights"] = task_weights
        # The response cache's current key covers ordinary prompt sampling,
        # not arbitrary backend hooks, prompt formatting or mutable weights.
        # Such requests must execute and must not seed ordinary cache entries.
        cache_bypass = bool(kwargs.pop("cache_bypass", False)) or bool(kwargs) or bool(stop) or messages is not None
        cache_bypass = cache_bypass or self.ttt_engine is not None or self.config.ttt_enabled
        cache_bypass = cache_bypass or self.config.disaggregate_prefill_decode or bool(self.config.draft_model_id)
        cache_bypass = cache_bypass or self.slo_scheduler is not None
        with self._lock:
            cache_handle = self._loaded_models.get(model_id)
            precision_epoch = getattr(cache_handle, "metadata", {}).get("_precision_epoch", 0)
        cache_config = {
            "max_new_tokens": max_tokens if max_tokens is not None else self.config.default_max_tokens,
            "temperature": temperature if temperature is not None else self.config.default_temperature,
            "top_p": top_p if top_p is not None else self.config.default_top_p,
            "top_k": top_k,
            "precision_epoch": precision_epoch,
        }
        if self.config.enable_semantic_cache:
            self._init_v5_layers()
            cache = getattr(self, "_semantic_cache", None)
            if cache is not None and prompt is not None:
                cached = cache.lookup(prompt, model_id, cache_config, bypass=cache_bypass)
                if cached is not None:
                    cached_text = self._check_safety_output(cached.response)
                    completion_tokens = len(cached.response.split())
                    return GenerationResponse(
                        text=cached_text,
                        usage={
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": completion_tokens,
                            "total_tokens": len(prompt.split()) + completion_tokens,
                        },
                        metrics=InferenceMetrics(
                            kernel_target=self._model_targets.get(model_id, self.fingerprint.target_id),
                            backend_name="semantic_cache",
                            extra={"cache_hit": True, "similarity": cached.similarity_score},
                        ),
                    )
        backend = self._load_model(model_id)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens if max_tokens is not None else self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            top_p=top_p if top_p is not None else self.config.default_top_p,
            top_k=top_k,
            stream=stream,
            stop=stop,
            extra=kwargs,
        )
        if self.ttt_engine is not None:
            request.extra.setdefault("ttt_engine", self.ttt_engine)
            request.extra.setdefault("ttt_request_id", uuid.uuid4().hex)
        aeg_key_for_request = self._loaded_aeg_path(model_id)[1]
        peagle_engine = (
            self._peagle_engines.get(aeg_key_for_request)
            if aeg_key_for_request is not None
            else self.peagle_engine
        )
        if peagle_engine is not None:
            request.extra.setdefault("peagle_engine", peagle_engine)
        start = datetime.datetime.now(datetime.timezone.utc)
        start_ns = time.time_ns()
        try:
            result = self._generate_with_features(backend, request)
        except Exception as exc:
            error_span = self.tracer.start_span(
                "aether.inference", attributes={"model_id": model_id, "error": str(exc)}
            )
            error_span.set_error(str(exc))
            self.tracer.finish_span(error_span)
            self.metrics_collector.record(0.0, 0.0, (time.time_ns() - start_ns) / 1_000_000, is_error=True)
            raise
        result.text = self._check_safety_output(result.text)
        end = datetime.datetime.now(datetime.timezone.utc)
        duration_s = (end - start).total_seconds()
        end_ns = time.time_ns()
        self.tracer.trace_request(
            request_id=uuid.uuid4().hex,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.completion_tokens,
            ttft_ms=result.metrics.get("ttft_ms", duration_s * 1000),
            total_ms=duration_s * 1000,
            model_id=model_id,
            actual_start_time_ns=start_ns,
            actual_end_time_ns=end_ns,
        )
        self.metrics_collector.record(
            result.metrics.get("ttft_ms", duration_s * 1000),
            result.completion_tokens / max(duration_s, 1e-9),
            duration_s * 1000,
            eagle_accept_rate=float(result.metrics.get("speculative", {}).get("acceptance_rate", 0.0)),
            kv_hit_rate=self.kv_cache.hit_rate(),
        )
        backend_metrics = {
            key: value
            for key, value in result.metrics.items()
            if key not in {"ttft_ms", "throughput_tps"}
        }
        if self.green_power_manager is not None:
            energy_mj, energy_source = self.green_power_manager.measure_request_energy(
                duration_s,
                n_prompt_tokens=result.prompt_tokens,
                n_gen_tokens=result.completion_tokens,
            )
            carbon_gco2 = self.green_power_manager.estimate_carbon(energy_mj)
            self.green_power_manager.record_request(
                energy_mj,
                carbon_gco2,
                source=energy_source,
            )
            backend_metrics.update(
                {
                    "energy_mj": energy_mj,
                    "carbon_gco2": carbon_gco2,
                    "energy_source": energy_source,
                }
            )
        if slo_request is not None:
            backend_metrics["slo_tier"] = slo_request.slo_tier.value
            backend_metrics["slo_priority"] = slo_request.priority
            backend_metrics["slo_deadline_s"] = slo_request.ttft_deadline_s - slo_request.arrival_time
            self.slo_scheduler.record_batch_latency(duration_s * 1000.0)
        metrics = InferenceMetrics(
            throughput_tps=result.completion_tokens / max(duration_s, 1e-6),
            ttft_ms=result.metrics.get("ttft_ms", duration_s * 1000),
            kernel_target=self._model_targets.get(model_id, self.fingerprint.target_id),
            active_precision=result.metrics.get("active_precision"),
            spec_accept_rate=float(result.metrics.get("speculative", {}).get("acceptance_rate", 0.0)),
            kv_cache_hit_rate=self.kv_cache.hit_rate(),
            memory_pressure=result.metrics.get("memory_pressure"),
            backend_name=result.backend_name or backend.name,
            extra=backend_metrics,
        )
        response = GenerationResponse(
            text=result.text,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            metrics=metrics,
            finish_reason=result.finish_reason,
        )
        if self.config.enable_semantic_cache and prompt is not None and not cache_bypass:
            cache = getattr(self, "_semantic_cache", None)
            if cache is not None:
                cache.store(prompt, result.text, model_id, cache_config, tokens_saved=0)
        return response

    @runtime_operation
    def generate_batch(
        self,
        model_id: str,
        prompts: list[str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int = 0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> list[GenerationResponse]:
        """Generate for several prompts in one batched forward pass.

        The prompts are decoded together in a single KV tensor, each row
        independent: one row's tokens, positions and stop condition cannot affect
        another's. Responses come back in prompt order.

        This is a throughput interface, not a latency one. Batching raises tokens
        per second in aggregate while each individual response takes at least as
        long as it would alone, so a single-prompt caller should keep using
        :meth:`generate`.

        A single-prompt list is forwarded to :meth:`generate`, so callers can size
        batches dynamically without special-casing one.

        The semantic response cache is deliberately not consulted here: a partial
        hit would split the batch and make the measured work depend on cache
        contents. Route cacheable single prompts through :meth:`generate`.
        """
        if not prompts:
            return []
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must be a list of strings")
        resolved_max = max_tokens if max_tokens is not None else self.config.default_max_tokens
        resolved_temperature = (
            temperature if temperature is not None else self.config.default_temperature
        )
        resolved_top_p = top_p if top_p is not None else self.config.default_top_p
        if len(prompts) == 1:
            return [
                self.generate(
                    model_id,
                    prompts[0],
                    max_tokens=resolved_max,
                    temperature=resolved_temperature,
                    top_p=resolved_top_p,
                    top_k=top_k,
                    stop=stop,
                    **kwargs,
                )
            ]

        for prompt in prompts:
            self._check_safety_prompt(prompt)

        backend = self._load_model(model_id)
        task_weights = self._effective_task_weights(model_id, kwargs)
        if task_weights is not None:
            kwargs["task_weights"] = task_weights
        unsupported = {
            key for key in (
                "adapter_id", "adapter_name", "task_weights", "ttt_engine", "ttt_slots",
                "grammar_session", "multi_agent_kv_coordinator", "multi_agent_prefix",
                "multi_agent_prefix_hash", "peagle_engine", "spec_draft_engine",
            ) if kwargs.get(key) is not None and (key != "task_weights" or bool(kwargs[key]))
        }
        if self.ttt_engine is not None:
            unsupported.add("ttt_engine")
        batch_aeg_key = self._loaded_aeg_path(model_id)[1]
        batch_peagle = self._peagle_engines.get(batch_aeg_key) if batch_aeg_key is not None else self.peagle_engine
        if batch_peagle is not None:
            unsupported.add("peagle_engine")
        if unsupported:
            raise AetherRuntimeError(
                "batched generation has no execution adapter for active request hooks "
                f"{sorted(unsupported)}; submit individually rather than ignoring them"
            )
        run_batch = getattr(backend, "generate_batch", None)
        if run_batch is None:
            raise AetherRuntimeError(
                f"backend {backend.name!r} does not implement batched generation; "
                "submit the prompts individually rather than expecting this call to "
                "loop over them"
            )
        requests = [
            GenerationRequest(
                model_id=model_id,
                prompt=prompt,
                max_tokens=resolved_max,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                top_k=top_k,
                stop=stop,
                extra=dict(kwargs),
            )
            for prompt in prompts
        ]

        start = datetime.datetime.now(datetime.timezone.utc)
        start_ns = time.time_ns()
        try:
            concurrent_requests = self.config.disaggregate_prefill_decode or self.config.draft_model_id or isinstance(
                kwargs.get("aether_kv_session_id"), str
            )
            if concurrent_requests:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(len(requests), self.config.disaggregate_queue_capacity)) as executor:
                    results = list(executor.map(lambda request: self._generate_with_features(backend, request), requests))
            else:
                handle = self._loaded_models.get(model_id)
                self._execution_features.automatic_precision(model_id, handle)
                with self._execution_features.gate(model_id).read():
                    results = run_batch(requests)
                for result in results:
                    result.metrics.update(self._execution_features.report(model_id, handle))
        except Exception as exc:
            error_span = self.tracer.start_span(
                "aether.inference.batch",
                attributes={"model_id": model_id, "batch_size": len(prompts), "error": str(exc)},
            )
            error_span.set_error(str(exc))
            self.tracer.finish_span(error_span)
            self.metrics_collector.record(
                0.0, 0.0, (time.time_ns() - start_ns) / 1_000_000, is_error=True
            )
            raise
        duration_s = (
            datetime.datetime.now(datetime.timezone.utc) - start
        ).total_seconds()

        produced = sum(result.completion_tokens for result in results)
        responses: list[GenerationResponse] = []
        for result in results:
            result.text = self._check_safety_output(result.text)
            backend_metrics = {
                key: value
                for key, value in result.metrics.items()
                if key not in {"ttft_ms", "throughput_tps"}
            }
            backend_metrics.setdefault("batch_size", len(prompts))
            backend_metrics["execution_mode"] = "concurrent_requests" if concurrent_requests else "batched_forward"
            # The wall time belongs to the batch, so the aggregate rate is the
            # figure batching actually improves.  Both are labelled rather than
            # letting a per-row number stand in for batch throughput.
            backend_metrics["batch_throughput_tps"] = produced / max(duration_s, 1e-9)
            responses.append(
                GenerationResponse(
                    text=result.text,
                    usage={
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.prompt_tokens + result.completion_tokens,
                    },
                    metrics=InferenceMetrics(
                        throughput_tps=result.completion_tokens / max(duration_s, 1e-6),
                        ttft_ms=result.metrics.get("ttft_ms", duration_s * 1000),
                        kernel_target=self._model_targets.get(model_id, self.fingerprint.target_id),
                        active_precision=result.metrics.get("active_precision"),
                        backend_name=result.backend_name or backend.name,
                        extra=backend_metrics,
                    ),
                    finish_reason=result.finish_reason,
                )
            )
        self.metrics_collector.record(
            duration_s * 1000,
            produced / max(duration_s, 1e-9),
            duration_s * 1000,
        )
        return responses

    def _check_safety_prompt(self, prompt: str) -> str:
        """Apply the configured prompt safety policy before inference."""
        if self.safety_engine is None:
            return prompt
        decision = self.safety_engine.check_prompt(prompt)
        if not decision.allowed:
            raise AetherRuntimeError(
                f"prompt rejected by safety policy: {', '.join(decision.reasons) or 'policy violation'}"
            )
        return prompt

    @runtime_stream
    def generate_stream(
        self,
        model_id: str,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int = 0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Yield text deltas from the backend's incremental decoder.

        Streaming intentionally bypasses the semantic response cache because a
        cache hit has no incremental decoder state. When safety is enabled,
        output is buffered until the backend finishes so the complete response
        can be checked before any text is released.
        """
        if prompt is None and messages is None:
            raise ValueError("either prompt or messages must be provided")
        safety_text = prompt or "\n".join(
            str(message.get("content", ""))
            for message in (messages or [])
            if isinstance(message, dict)
        )
        safety_text = self._check_safety_prompt(safety_text)
        slo_deadline_ms = kwargs.pop("slo_deadline_ms", None)
        slo_request: Any | None = None
        if self.slo_scheduler is not None:
            from aether.runtime.r4_slo_scheduler import SLOTier

            requested_tier = kwargs.pop("slo_tier", SLOTier.BALANCED)
            try:
                slo_request = self.slo_scheduler.submit(
                    request_id=uuid.uuid4().hex,
                    prompt_tokens=len(safety_text.split()),
                    max_new_tokens=max_tokens or self.config.default_max_tokens,
                    slo_tier=requested_tier,
                    metadata={"model_id": model_id, "stream": True},
                    ttft_deadline_ms=slo_deadline_ms,
                )
                selected = self.slo_scheduler.next_batch()
                if not any(item.request_id == slo_request.request_id for item in selected):
                    raise AetherRuntimeError("SLO scheduler did not admit the streaming request")
            except (ValueError, KeyError) as exc:
                raise AetherRuntimeError(f"invalid SLO tier {requested_tier!r}") from exc
        backend = self._load_model(model_id)
        request_extra = {"cache_bypass": True, **kwargs}
        if self.ttt_engine is not None:
            request_extra.setdefault("ttt_engine", self.ttt_engine)
            request_extra.setdefault("ttt_request_id", uuid.uuid4().hex)
        task_weights = self._effective_task_weights(model_id, request_extra)
        if task_weights is not None:
            request_extra["task_weights"] = task_weights
        stream_aeg_key = self._loaded_aeg_path(model_id)[1]
        stream_peagle = (
            self._peagle_engines.get(stream_aeg_key)
            if stream_aeg_key is not None
            else self.peagle_engine
        )
        if stream_peagle is not None:
            request_extra.setdefault("peagle_engine", stream_peagle)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens if max_tokens is not None else self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            top_p=top_p if top_p is not None else self.config.default_top_p,
            top_k=top_k,
            stream=True,
            stop=stop,
            extra=request_extra,
        )
        stream = self._stream_with_features(backend, request)
        stream_started = time.perf_counter()
        pending = ""
        # Output policy checks must cover the complete generated response.  A
        # chunk-by-chunk check can leak a secret split across chunks, and once
        # emitted text cannot be retracted.  In safety-enabled mode we
        # therefore buffer the response, apply the same policy as generate(),
        # and only then release it.  Safety-disabled mode retains true
        # incremental delivery.
        safety_buffer: str | None = "" if self.safety_engine is not None else None
        retain = max((len(sequence) - 1 for sequence in (stop or []) if sequence), default=0)
        try:
            for chunk in stream:
                if safety_buffer is not None:
                    safety_buffer += str(chunk)
                    if stop:
                        cutoff = min(
                            (
                                safety_buffer.find(sequence)
                                for sequence in stop
                                if sequence and sequence in safety_buffer
                            ),
                            default=-1,
                        )
                        if cutoff >= 0:
                            safe_text = self._check_safety_output(safety_buffer[:cutoff])
                            if safe_text:
                                yield safe_text
                            return
                    continue
                pending += str(chunk)
                if stop:
                    cutoff = min(
                        (
                            pending.find(sequence)
                            for sequence in stop
                            if sequence and sequence in pending
                        ),
                        default=-1,
                    )
                    if cutoff >= 0:
                        if cutoff:
                            yield pending[:cutoff]
                        return
                if retain and len(pending) > retain:
                    yield pending[:-retain]
                    pending = pending[-retain:]
                elif not retain:
                    yield pending
                    pending = ""
            if safety_buffer is not None:
                safe_text = self._check_safety_output(safety_buffer)
                if safe_text:
                    yield safe_text
            elif pending:
                yield pending
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
            if slo_request is not None:
                self.slo_scheduler.record_batch_latency((time.perf_counter() - stream_started) * 1000.0)

    def _check_safety_output(self, output: str) -> str:
        """Filter generated output and reject content the policy blocks."""
        if self.safety_engine is None:
            return output
        decision = self.safety_engine.check_output(output)
        if not decision.allowed:
            raise AetherRuntimeError(
                f"output rejected by safety policy: {', '.join(decision.reasons) or 'policy violation'}"
            )
        return decision.redacted_text or output

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> GenerationResponse:
        """Chat completion with a list of messages."""
        return self.generate(model_id, messages=messages, **kwargs)

    @runtime_operation
    def embed(self, model_id: str, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        backend = self._load_model(model_id)
        if hasattr(backend, "embed"):
            return backend.embed(model_id, input)
        msg = f"Backend {backend.name} does not support embeddings"
        raise AetherRuntimeError(msg)

    @runtime_operation
    def rerank(self, model_id: str, query: str, documents: list[str]) -> list[dict[str, Any]]:
        """Rerank documents for a query."""
        backend = self._load_model(model_id)
        if hasattr(backend, "rerank"):
            return backend.rerank(model_id, query, documents)
        msg = f"Backend {backend.name} does not support reranking"
        raise AetherRuntimeError(msg)

    @runtime_operation
    def transcribe(self, model_id: str, audio: str, language: str | None = None) -> str:
        """Transcribe an audio file."""
        backend = self._load_model(model_id)
        if hasattr(backend, "transcribe"):
            return backend.transcribe(model_id, audio, language=language)
        msg = f"Backend {backend.name} does not support transcription"
        raise AetherRuntimeError(msg)

    @runtime_operation
    def benchmark(self, model_id: str, prompt: str = "Hello, my name is", max_tokens: int = 128) -> dict[str, Any]:
        """Run a simple benchmark on a model."""
        backend = self._load_model(model_id)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        import time
        start = time.perf_counter()
        result = backend.generate(request)
        end = time.perf_counter()
        tps = result.completion_tokens / max(end - start, 1e-6)
        return {
            "model_id": model_id,
            "backend": backend.name,
            "kernel_target": self._model_targets.get(model_id, self.fingerprint.target_id),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "duration_s": end - start,
            "throughput_tps": tps,
            "ttft_ms": result.metrics.get("ttft_ms", 0.0),
        }

    def get_attestation_report(self, model_id: str | None = None) -> AttestationReport:
        """Return the active TEE report, or an explicit unavailable result."""
        model_hash: str | None = None
        if model_id is not None:
            aeg_path = self._resolve_aeg_path(model_id)
            if aeg_path is not None:
                self._init_v4_layers(aeg_path)
                try:
                    package = load_aeg_package(aeg_path)
                    model_hash = package.manifest.manifest_hash if package.manifest else None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Unable to read AEG hash for attestation", error=str(exc))
        if self.tee_manager is None:
            return AttestationReport(
                enabled=False,
                hardware_backed=False,
                model_hash=model_hash,
                enclave_measurement=None,
                reason="TEE is not enabled by the loaded AEG",
            )
        report = AttestationReport(self.tee_manager.get_attestation_report())
        # An initialized software shim is not a confidential enclave and must
        # not be exposed as an enabled attestation service.  Keep the raw
        # ``enclave_initialized`` field for diagnostics, but gate the public
        # capability on hardware-backed evidence.
        report["enabled"] = bool(
            report.get("enclave_initialized", False)
            and report.get("hardware_backed", False)
        )
        report["model_hash"] = model_hash
        report["enclave_measurement"] = (
            report.get("tdx_report_hash") or report.get("snp_report_hash")
        )
        return report

    def quantization_report(self, model_id: str) -> dict[str, Any]:
        """Return quantization metadata from a loaded AEG artifact."""
        aeg_path = self._resolve_aeg_path(model_id)
        if aeg_path is None:
            raise ModelNotFoundError(f"Model {model_id} not found", model_id=model_id)
        aeg = load_aeg_package(aeg_path)
        info = self.info(model_id)
        precision_map = info["precision_map"]
        if not precision_map:
            raise AetherRuntimeError(f"AEG {model_id!r} contains no precision map")
        counts: dict[str, int] = {}
        for precision in precision_map.values():
            counts[precision] = counts.get(precision, 0) + 1
        entries = aeg.weight_store().load_index()
        weight_bytes = aeg.weight_store().total_bytes if entries else None
        total_elements = sum(entry.num_elements for entry in entries.values())
        weighted_bits = sum(entry.num_elements * entry.bits for entry in entries.values())
        effective_bits = weighted_bits / total_elements if total_elements else None
        bf16_bytes = total_elements * 2 if total_elements else None
        reduction = (bf16_bytes / weight_bytes) if bf16_bytes and weight_bytes else None
        unique_precisions = sorted(set(precision_map.values()))
        precision = unique_precisions[0] if len(unique_precisions) == 1 else "mixed"
        return {
            "model_id": model_id,
            "status": "measured" if entries else "metadata_only",
            "precision": precision,
            "precision_map": precision_map,
            "precision_counts": counts,
            "tensor_count": len(precision_map),
            "weight_tensor_count": len(entries),
            "memory_mb": (weight_bytes / (1024 * 1024)) if weight_bytes is not None else None,
            "weight_bytes": weight_bytes,
            "effective_bits_per_weight": effective_bits,
            "vs_bf16_reduction": f"{reduction:.2f}x" if reduction is not None else None,
            "energy_savings_est_pct": None,
        }

    def _effective_task_weights(
        self, model_id: str, request_extra: dict[str, Any] | None = None,
    ) -> dict[str, float] | None:
        """Snapshot validated request, model-specific or global coefficients."""
        from aether.runtime.session_contract import validate_task_weights

        with self._lock:
            if request_extra is not None and "task_weights" in request_extra:
                selected = request_extra["task_weights"]
            else:
                selected = getattr(self, "_task_weights_by_model", {}).get(
                    model_id, getattr(self, "_task_weights", None),
                )
            return None if selected is None else validate_task_weights(selected)

    @runtime_operation
    def set_task_weights(
        self,
        weights: dict[str, float] | str,
        **task_weights: float,
    ) -> dict[str, float]:
        """Set normalized global defaults or coefficients for one model.

        Per-model defaults override a mapping-only global default. Each request
        snapshots its coefficients, so a concurrent update affects later
        requests without mutating an admitted request's selected engine.
        Use ``set_model_task_weights(model_id, mapping)`` for arbitrary labels
        that cannot be represented safely as Python keyword arguments.
        """
        task_model_id = weights if isinstance(weights, str) else None
        if task_model_id is not None:
            weights = task_weights
        elif task_weights:
            raise ValueError("keyword task weights cannot be combined with a mapping")
        return self._set_task_weight_defaults(task_model_id, weights)

    @runtime_operation
    def set_model_task_weights(self, model_id: str, weights: dict[str, float]) -> dict[str, float]:
        """Set one model's coefficients from a mapping with unrestricted labels.

        Labels are data rather than Python parameter names. Validation,
        normalization and request snapshot semantics match ``set_task_weights``.
        """
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a nonempty string")
        return self._set_task_weight_defaults(model_id, weights)

    def _set_task_weight_defaults(self, task_model_id: str | None, weights: Any) -> dict[str, float]:
        """Validate and atomically publish coefficients for both public forms."""
        from aether.runtime.session_contract import validate_task_weights

        normalized = validate_task_weights(weights, normalize=True)
        if not normalized:
            raise ValueError("at least one task weight must be greater than zero")
        if task_model_id is not None:
            aeg_path = self._resolve_aeg_path(task_model_id)
            if aeg_path is None:
                raise ModelNotFoundError(f"model {task_model_id!r} was not found", model_id=task_model_id)
            package = load_aeg_package(aeg_path)
            task_vectors = package.metadata.get("task_vectors", {})
            vectors = task_vectors.get("vectors", []) if isinstance(task_vectors, dict) else []
            available = {str(vector.get("name")) for vector in vectors if isinstance(vector, dict)}
            if not available:
                raise AetherRuntimeError(
                    f"AEG {task_model_id!r} contains no persisted executable task-vector payloads"
                )
            validate_task_weights(weights, available=available)
        with self._lock:
            if task_model_id is not None:
                if not hasattr(self, "_task_weights_by_model"):
                    self._task_weights_by_model: dict[str, dict[str, float]] = {}
                self._task_weights_by_model[task_model_id] = dict(normalized)
            else:
                self._task_weights = dict(normalized)
        return dict(normalized)

    def generate_cascade(
        self,
        query: str,
        *,
        model_routing: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Route a request to a configured model tier and execute it.

        Routing is deterministic and complexity-based.  A missing tier is an
        explicit configuration error; the runtime never returns a successful
        response from a nonexistent or simulated model.
        """
        routes = dict(model_routing or self.config.model_routing)
        if not routes:
            raise AetherRuntimeError("cascade routing requires RuntimeConfig.model_routing")
        from aether.runtime.cascade_router import CascadeRouter, ModelTier

        router = CascadeRouter()
        tier_order = (("simple", 0.25), ("complex", 0.7), ("reasoning", 1.0))
        for index, (name, limit) in enumerate(tier_order):
            model_id = routes.get(name)
            if model_id:
                router.register_tier(
                    ModelTier(
                        tier_id=index, model_id=model_id, max_complexity=limit,
                        supports_reasoning=name == "reasoning",
                    )
                )
        if not router.tiers:
            raise AetherRuntimeError("model_routing must define at least one of simple, complex, reasoning")
        decision = router.route(query)
        response = self.generate(decision.tier.model_id, query, **kwargs)
        response.metrics.extra["cascade"] = decision.to_dict()
        return response

    def generate_with_tools(
        self,
        model_id: str,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Execute explicitly requested MCP tools, then generate from results.

        Tool execution is fail-closed: an MCP layer and registered server must
        exist, and tool errors are surfaced rather than converted into model
        text that could be mistaken for a successful call.
        """
        mcp_tools = kwargs.pop("mcp_tools", None)
        max_rounds = int(kwargs.pop("max_tool_rounds", 4))
        if max_rounds < 0 or max_rounds > 16:
            raise ValueError("max_tool_rounds must be between 0 and 16")
        if mcp_tools is not None:
            if tools is not None:
                raise ValueError("provide only one of tools or mcp_tools")
            tools = [
                {"name": item, "arguments": {}} if isinstance(item, str) else item
                for item in mcp_tools
            ]
        self._init_v4_layers(self._resolve_aeg_path(model_id))
        allowed_tool_names: set[str] | None = None
        if tools:
            if self.mcp_layer is None:
                raise AetherRuntimeError("MCP is not enabled by the loaded AEG")
            allowed_tool_names = set()
            results = []
            for tool in tools:
                name = tool.get("name")
                arguments = tool.get("arguments", {})
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("each tool request requires string name and object arguments")
                allowed_tool_names.add(name)
                result = self.mcp_layer.call_tool(name, arguments)
                if result.get("isError"):
                    raise AetherRuntimeError(f"MCP tool {name!r} failed: {result}")
                results.append({"tool": name, "result": result})
            prompt = f"{prompt}\n\nTool results:\n{results}"
        response = self.generate(model_id, prompt, **kwargs)
        # A model may emit a valid MCP call even when the caller supplied no
        # explicit arguments. Dispatch that structured call, inject the real
        # result into a fresh context, and continue generation. This is capped
        # to prevent an accidentally tool-calling model from looping forever.
        if self.mcp_layer is not None:
            tool_calls: list[dict[str, Any]] = []
            detect_tool_call = getattr(self.mcp_layer, "detect_tool_call", None)
            if not callable(detect_tool_call):
                return response
            for _ in range(max_rounds):
                detected = detect_tool_call(response.text)
                if detected is None:
                    break
                name = detected.get("tool")
                arguments = detected.get("arguments", {})
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise AetherRuntimeError("model emitted an invalid MCP tool call")
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    raise AetherRuntimeError(f"model requested unapproved MCP tool {name!r}")
                result = self.mcp_layer.call_tool(name, arguments)
                if result.get("isError"):
                    raise AetherRuntimeError(f"MCP tool {name!r} failed: {result}")
                tool_calls.append({"name": name, "arguments": arguments})
                continuation = (
                    f"{prompt}\n\nModel tool call:\n"
                    f"{json.dumps({'name': name, 'arguments': arguments}, sort_keys=True)}\n\n"
                    f"Tool result:\n{json.dumps(result, sort_keys=True)}\n\n"
                    "Continue the answer using the tool result. Do not emit another "
                    "tool call unless another external action is required."
                )
                response = self.generate(model_id, continuation, **kwargs)
            if tool_calls:
                response.metrics.extra["mcp_tool_calls"] = tool_calls
                response.metrics.extra["mcp_tool_rounds"] = len(tool_calls)
        return response

    # ── v4.0 Runtime Extensions ────────────────────────────────────────────────

    def _init_v4_layers(self, aeg_path: str | None = None) -> None:
        """Initialize v4.0 runtime layers from AEG package config.

        Called lazily on first model load. Each layer is imported and
        constructed only if its config flag is enabled in the AEG package
        and/or RuntimeConfig.

        Layers that fail to initialize log a warning and remain None —
        the runtime continues to function without them.
        """
        if aeg_path is None:
            return

        # Native AEG/1.1 and AEG/2.x artifacts do not necessarily expose the
        # AEGPackageV2 convenience manifest API.  Initialize optional runtime
        # layers from the artifact's actual files first.  This keeps the
        # runtime coupled to persisted artifacts rather than to compiler-side
        # Python objects, which is essential after a process restart.
        root = Path(aeg_path)
        if (root / "manifest.json").is_file():
            def _read_json(path: Path) -> dict[str, Any] | None:
                try:
                    if not path.is_file():
                        return None
                    value = json.loads(path.read_text(encoding="utf-8"))
                    return value if isinstance(value, dict) else None
                except (OSError, ValueError) as exc:
                    logger.warning("optional AEG configuration unreadable", path=str(path), error=str(exc))
                    return None

            # R1 P-EAGLE / native MTP speculation.  The loader only enables
            # this when the AEG contains real, validated MTP blobs; the CPU
            # backend then performs exact-greedy target verification.
            mtp_path = root / "speculation" / "mtp_config.json"
            mtp_key = str(root.resolve())
            speculation_enabled = self.config.speculative_decoding
            if (
                mtp_path.is_file()
                and mtp_key not in self._peagle_engines
                and speculation_enabled is not False
                and str(speculation_enabled).lower() not in {"off", "none", "disabled"}
            ):
                try:
                    from aether.runtime.r1_peagle_engine import PEAGLEEngine

                    try:
                        import torch

                        mtp_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        mtp_device = "cpu"

                    loaded_peagle = PEAGLEEngine(
                        draft_K=min(self.config.speculative_tree_depth, 8),
                        mode="mtp",
                        mtp_config_path=str(mtp_path),
                        device=mtp_device,
                    )
                    if not getattr(loaded_peagle, "_mtp_weights", None) or not all(
                        weight is not None for weight in loaded_peagle._mtp_weights
                    ):
                        raise RuntimeError("AEG MTP speculation has no executable weight tensors")
                    self._peagle_engines[mtp_key] = loaded_peagle
                    self.peagle_engine = loaded_peagle
                    logger.info("R1 P-EAGLE MTP engine initialized", heads=len(loaded_peagle._mtp_heads))
                except Exception as exc:
                    logger.warning("R1 P-EAGLE MTP init failed; using target decoding", error=str(exc))

            # R3 Grammar FSM Engine
            grammar_path = root / "grammar" / "fsm_config.json"
            if grammar_path.is_file() and self.grammar_engine is None:
                try:
                    from aether.runtime.r3_grammar_fsm import GrammarFSMEngine
                    engine = GrammarFSMEngine()
                    if engine.load_from_config(str(root)):
                        self.grammar_engine = engine
                        logger.info("R3 Grammar FSM Engine initialized")
                except Exception as exc:
                    logger.warning("R3 Grammar FSM Engine init failed", error=str(exc))

            # R5 TTT Fast-Weight Engine
            ttt_path = root / "ttt" / "fast_weight_config.json"
            if ttt_path.is_file() and self.ttt_engine is None:
                try:
                    from aether.runtime.r5_ttt_engine import TTTFastWeightEngine
                    self.ttt_engine = TTTFastWeightEngine(ttt_config_path=str(ttt_path))
                    logger.info("R5 TTT Fast-Weight Engine initialized")
                except Exception as exc:
                    logger.warning("R5 TTT Engine init failed", error=str(exc))

            # R7 Green Power Manager.  Ignore disabled/default profiles so a
            # normal AEG does not unexpectedly activate energy scheduling.
            green_path = root / "green" / "energy_profile.json"
            if not green_path.is_file():
                green_path = root / "metadata" / "green_profile.json"
            green_data = _read_json(green_path)
            if green_data and self.green_power_manager is None:
                enabled = bool(green_data.get("enabled", True))
                if enabled and (self.config.green_power_management or green_data.get("estimated_joules_per_token", 0) or green_data.get("dvfs_hints")):
                    try:
                        from aether.runtime.r7_green_power_manager import GreenPowerManager
                        self.green_power_manager = GreenPowerManager(green_profile_path=str(green_path))
                        logger.info("R7 Green Power Manager initialized")
                    except Exception as exc:
                        logger.warning("R7 Green Power Manager init failed", error=str(exc))

            # R8 TEE Runtime Manager.  A CPU process must not pretend that a
            # TEE exists; only initialize a configured backend and let the
            # manager report unsupported hardware at use time.
            tee_path = root / "tee" / "enclave_config.json"
            if not tee_path.is_file():
                tee_path = root / "security" / "tee_config.json"
            tee_data = _read_json(tee_path)
            tee_backend = (tee_data or {}).get("backend") or (tee_data or {}).get("tee_backend")
            if tee_backend is None and self.config.tee_mode not in ("", "auto", "none"):
                tee_backend = self.config.tee_mode
            if tee_backend and tee_backend != "none" and self.tee_manager is None:
                try:
                    from aether.runtime.r8_tee_manager import TEERuntimeManager
                    self.tee_manager = TEERuntimeManager(tee_config_path=str(tee_path), backend=str(tee_backend))
                    logger.info("R8 TEE Runtime Manager initialized", backend=str(tee_backend))
                except Exception as exc:
                    logger.warning("R8 TEE Manager init failed", error=str(exc))

            # R6 MCP Integration Layer.  Registration is persisted in the AEG
            # when present, while RuntimeConfig can supply additional servers.
            mcp_path = root / "mcp" / "mcp_config.json"
            mcp_data = _read_json(mcp_path)
            servers = list((mcp_data or {}).get("server_registry", []))
            if self.config.mcp_servers:
                servers.extend({"id": key, **value} for key, value in self.config.mcp_servers.items())
            if servers and self.mcp_layer is None:
                try:
                    from aether.runtime.r6_mcp_integration import MCPIntegrationLayer
                    self.mcp_layer = MCPIntegrationLayer(timeout_s=self.config.mcp_timeout_ms / 1000.0)
                    for server in servers:
                        if isinstance(server, dict) and server.get("id"):
                            self.mcp_layer.add_server(
                                str(server["id"]),
                                str(server.get("transport", "stdio")),
                                server.get("endpoint"),
                                server.get("command"),
                            )
                    logger.info("R6 MCP Integration Layer initialized", servers=len(servers))
                except Exception as exc:
                    logger.warning("R6 MCP Layer init failed", error=str(exc))
            return

        try:
            from aether.compiler.aeg_format_v2 import AEGPackageV2
            pkg = AEGPackageV2(aeg_path)
            manifest = pkg.read_manifest()

            # R3 Grammar FSM Engine
            if manifest.has_grammar_fsm and self.grammar_engine is None:
                try:
                    from aether.runtime.r3_grammar_fsm import GrammarFSMEngine
                    self.grammar_engine = GrammarFSMEngine()
                    self.grammar_engine.load_from_config(aeg_path)
                    logger.info("R3 Grammar FSM Engine initialized")
                except Exception as exc:
                    logger.warning("R3 Grammar FSM Engine init failed", error=str(exc))

            # R5 TTT Fast-Weight Engine
            if manifest.has_ttt_fast_weights and self.ttt_engine is None:
                try:
                    from aether.runtime.r5_ttt_engine import TTTFastWeightEngine
                    self.ttt_engine = TTTFastWeightEngine(
                        ttt_config_path=str(Path(aeg_path) / "ttt" / "fast_weight_config.json")
                    )
                    logger.info("R5 TTT Fast-Weight Engine initialized")
                except Exception as exc:
                    logger.warning("R5 TTT Engine init failed", error=str(exc))

            # R7 Green Power Manager
            if manifest.has_green_profile and self.green_power_manager is None:
                try:
                    from aether.runtime.r7_green_power_manager import GreenPowerManager
                    self.green_power_manager = GreenPowerManager(
                        green_profile_path=str(Path(aeg_path) / "green" / "energy_profile.json")
                    )
                    logger.info("R7 Green Power Manager initialized")
                except Exception as exc:
                    logger.warning("R7 Green Power Manager init failed", error=str(exc))

            # R8 TEE Runtime Manager
            if manifest.has_tee_enclave and self.tee_manager is None:
                try:
                    from aether.runtime.r8_tee_manager import TEERuntimeManager
                    tee_cfg = pkg.read_tee_config()
                    self.tee_manager = TEERuntimeManager(
                        tee_config_path=str(Path(aeg_path) / "tee" / "enclave_config.json"),
                        backend=tee_cfg.tee_backend if tee_cfg else "nvidia_cc",
                    )
                    logger.info("R8 TEE Runtime Manager initialized")
                except Exception as exc:
                    logger.warning("R8 TEE Manager init failed", error=str(exc))

            # R6 MCP Integration Layer
            if manifest.has_mcp_config and self.mcp_layer is None:
                try:
                    from aether.runtime.r6_mcp_integration import MCPIntegrationLayer
                    mcp_cfg = pkg.read_mcp_config()
                    self.mcp_layer = MCPIntegrationLayer(
                        timeout_s=self.config.mcp_timeout_ms / 1000.0,
                    )
                    for server in (mcp_cfg.server_registry if mcp_cfg else []):
                        if server.get("id"):
                            self.mcp_layer.add_server(
                                server["id"],
                                server.get("transport", "stdio"),
                                server.get("endpoint"),
                                server.get("command"),
                            )
                    logger.info("R6 MCP Integration Layer initialized")
                except Exception as exc:
                    logger.warning("R6 MCP Layer init failed", error=str(exc))

        except Exception as exc:
            logger.warning("v4.0 layer init skipped", error=str(exc))

    @runtime_operation
    def compile_async(
        self,
        model_id: str,
        job_id: str | None = None,
        target: str = "auto",
        quantization: str | None = None,
        quality_budget: float = 0.98,
        enable_mtp: bool = False,
        enable_grammar: bool = False,
        enable_tee: bool = False,
        enable_green: bool = False,
    ) -> str:
        """Start an async compilation job.

        Enqueues a background thread that compiles the model with the given
        configuration. Returns the job_id immediately.

        Args:
            model_id: Source model identifier or path.
            job_id: Optional job ID (auto-generated if None).
            target: Hardware target ID ('auto' = detect from fingerprint).
            quantization: Quantization scheme or None for auto.
            quality_budget: Quality preservation budget (0.0–1.0).
            enable_mtp: Enable Pass 10 MTP head compilation.
            enable_grammar: Enable Pass 11 grammar FSM pre-compilation.
            enable_tee: Enable Pass 17 TEE kernel wrapping.
            enable_green: Enable Pass 16 green energy profiling.

        Returns:
            Job ID string.
        """
        if job_id is None:
            job_id = str(uuid.uuid4())

        self._compile_jobs[job_id] = {
            "status": "queued",
            "model": model_id,
            "target": target,
            "queued_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        }

        def _run_compile() -> None:
            try:
                self._compile_jobs[job_id]["status"] = "running"  # type: ignore[index]
                self._compile_jobs[job_id]["started_at"] = (  # type: ignore[index]
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                from aether.compiler import Compiler, CompilerConfig
                resolved_target = target if target != "auto" else self.fingerprint.target_id
                # Build kwargs with only known CompilerConfig fields
                cfg_kwargs: dict[str, Any] = {
                    "quality_mode": "diagnostic" if self.config.allow_uncertified else "require_hf",
                    "targets": [resolved_target],
                    "quality_budget": quality_budget,
                    "enable_mtp_head": enable_mtp,
                    "enable_grammar_constraint": enable_grammar,
                    "enable_tee": enable_tee,
                    "enable_green_energy": enable_green,
                }
                # Map quantization shorthand to kv_cache_dtype if provided
                if quantization is not None:
                    cfg_kwargs["kv_cache_dtype"] = quantization
                cfg = CompilerConfig(**cfg_kwargs)
                compiler = Compiler(cfg)
                result = compiler.compile(model_id)
                self._compile_jobs[job_id]["status"] = "succeeded"  # type: ignore[index]
                self._compile_jobs[job_id]["output_path"] = str(result.output_path)  # type: ignore[index]
            except Exception as exc:
                self._compile_jobs[job_id]["status"] = "failed"  # type: ignore[index]
                self._compile_jobs[job_id]["error"] = str(exc)  # type: ignore[index]
                # Do NOT call logger.error() here.  The structlog/logging stack
                # writes to a BufferedWriter (stdout).  When this daemon thread
                # is still alive as the interpreter begins shutdown, Python's
                # atexit machinery acquires the BufferedWriter lock to flush it.
                # If logger.error() then tries to acquire the same lock we get:
                #   Fatal Python error: _enter_buffered_busy  →  SIGABRT (134)
                # sys.is_finalizing() is NOT a reliable guard because the lock
                # is acquired between is_finalizing() returning False and the
                # actual write (TOCTOU race).
                #
                # os.write(fd, bytes) bypasses all Python buffer locking: it is
                # a raw syscall, safe to call from any thread at any time.
                try:
                    msg = (
                        f"[aether compile-async] FAILED job_id={job_id} "
                        f"error={exc!r}\n"
                    ).encode("utf-8", errors="replace")
                    os.write(2, msg)  # fd 2 = stderr, unbuffered, lock-free
                except OSError:
                    pass  # fd closed during very late shutdown — nothing to do
            finally:
                # Also guard this block: datetime.now() involves no I/O and is
                # safe, but the dict assignment could race if the interpreter
                # has already cleared globals.  Wrap defensively.
                try:
                    self._compile_jobs[job_id]["completed_at"] = (  # type: ignore[index]
                        datetime.datetime.now(datetime.timezone.utc).isoformat()
                    )
                except Exception:  # noqa: BLE001
                    pass

        thread = threading.Thread(target=_run_compile, daemon=True, name=f"compile-{job_id[:8]}")
        # Store the thread so callers can join() before process exit if needed,
        # preventing the daemon-thread vs interpreter-shutdown race.
        self._compile_jobs[job_id]["_thread"] = thread
        thread.start()
        logger.info("Compilation job queued", job_id=job_id, model=model_id, target=target)
        return job_id

    def get_compile_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of an async compilation job.

        Args:
            job_id: Job ID returned by compile_async().

        Returns:
            Status dict with keys: job_id, status, model, target, queued_at,
            started_at (if started), completed_at (if done), error (if failed),
            output_path (if succeeded).

        Raises:
            KeyError: If job_id is not found.
        """
        if job_id not in self._compile_jobs:
            msg = f"Compilation job '{job_id}' not found."
            raise KeyError(msg)
        return {"job_id": job_id, **self._compile_jobs[job_id]}

    def merge(
        self,
        model_id: str,
        task_vectors: list[dict[str, Any]],
        method: str = "task_arithmetic",
        density: float = 1.0,
    ) -> dict[str, Any]:
        """Publish an integrity-checked AEG with real, persisted task-vector deltas.

        Each source may update a different subset of base parameters, but must
        contribute at least one matching tensor. ``density`` controls DARE's
        retained probability or TIES's per-tensor magnitude trimming; dare_ties
        applies DARE followed by TIES at that retention fraction. Other methods
        require density=1. FREE/evolutionary retain their existing L2 proxy
        search, which is not a model-quality evaluation. All outputs require
        fresh quality evaluation, signatures and weight-dependent attestations.
        """
        import math
        import re
        import shutil
        import tempfile
        from collections.abc import Mapping

        if not task_vectors:
            raise ValueError("at least one task vector source is required")
        if (
            isinstance(density, bool)
            or not isinstance(density, (int, float))
            or not math.isfinite(density)
            or not 0.0 < density <= 1.0
        ):
            raise ValueError("density must be finite, greater than 0 and no more than 1")
        method_aliases = {"task-arithmetic": "task_arithmetic", "free_merging": "free"}
        method = method_aliases.get(method, method)
        supported_methods = {
            "task_arithmetic",
            "dare",
            "ties",
            "dare_ties",
            "free",
            "evolutionary",
            "soup",
        }
        if method not in supported_methods:
            raise ValueError(
                f"unsupported merge method {method!r}; choose one of {sorted(supported_methods)}"
            )
        if density != 1.0 and method not in {"dare", "ties", "dare_ties"}:
            raise ValueError(f"density is not supported for merge method {method!r}; use density=1")
        coefficients: list[float] = []
        names: list[str] = []
        for index, item in enumerate(task_vectors):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"].strip()
            ):
                raise ValueError("each task vector requires a readable 'path'")
            coefficient = item.get("coefficient", 1.0 / len(task_vectors))
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))
                or not math.isfinite(coefficient)
                or coefficient < 0
            ):
                raise ValueError("task vector coefficients must be finite non-negative numbers")
            coefficients.append(float(coefficient))
            name = str(item.get("name") or Path(item["path"]).stem or f"task_{index}")
            if name in names:
                raise ValueError(f"duplicate task vector name {name!r}")
            names.append(name)
        total = sum(coefficients)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("task vector coefficients must have a finite positive sum")
        # Task arithmetic coefficients are absolute delta scales. Convex and
        # proxy-search algorithms use normalized initial coefficients instead.
        if method in {"soup", "free", "evolutionary"}:
            coefficients = [value / total for value in coefficients]
        if method == "soup" and any(
            not math.isclose(value, 1.0 / len(coefficients), abs_tol=1e-12)
            for value in coefficients
        ):
            raise ValueError(
                "soup requires uniform coefficients; use task_arithmetic for weighted task vectors"
            )

        # Runtime merging operates on native AEG artifacts.  It deliberately
        # does not return a recorded/config-only success: a caller receives a
        # new artifact only after real source weights were read and persisted.
        base_path = self._resolve_aeg_path(model_id)
        if base_path is None:
            raise ModelNotFoundError(
                f"base AEG model {model_id!r} was not found", model_id=model_id
            )
        base = load_aeg_package(base_path)
        from aether.compiler.aeg_merging import (
            _validate_input_tree,
            invalidate_weight_dependent_evidence,
        )

        _validate_input_tree(base.root)
        base.verify_integrity()
        if base.manifest is None or base.ir is None or not base.has_weights:
            raise AetherRuntimeError(f"base AEG {model_id!r} has no persisted weights")
        if "tee_kernel_wrapping" in base.metadata.get("optimizer_passes", []):
            raise AetherRuntimeError(
                "TEE-bound AEG weights require a fresh attested compilation after merging"
            )
        output_root = base.root.with_name(base.root.name + ".merged")
        if output_root.exists():
            raise AetherRuntimeError(
                f"merge output already exists: {output_root}; remove it explicitly first"
            )
        if not (base.root / "tokenizer").is_dir():
            raise AetherRuntimeError(
                "base AEG has no packaged tokenizer; merged text generation cannot be portable"
            )

        import numpy as np
        from aether.compiler.stage2_optimizer.pass12_model_merging import (
            _apply_delta,
            _compute_task_vectors,
            _get_merger,
            _load_source_weights,
        )
        from aether.core.weight_store import WeightStore
        from aether.quantization.formats import quantize_tensor

        store = base.weight_store()
        entries = store.load_index()
        base_weights = {
            name: tensor.reshape(-1).astype("float32").tolist()
            for name, tensor in store.dequantize_all().items()
        }
        if any(not np.isfinite(values).all() for values in base_weights.values()):
            raise ValueError("base merge tensors must contain finite coordinates")
        sources: list[dict[str, list[float]]] = []
        for item in task_vectors:
            source = _load_source_weights(item["path"], base)
            if not source:
                raise AetherRuntimeError(
                    f"task vector source has no readable weights: {item['path']}"
                )
            sources.append(source)
        task_delta = _compute_task_vectors(base_weights, sources)
        for name, vector in zip(names, task_delta):
            if not vector:
                raise AetherRuntimeError(
                    f"task vector {name!r} has no overlapping matching base tensor"
                )
        if method in {"ties", "dare_ties"}:
            # Vectorized trim/elect/disjoint-mean supports partial adapters:
            # missing source coordinates never count toward its denominator.
            rng = np.random.default_rng(42)
            merged_delta = {}
            for key in sorted(set().union(*(vector.keys() for vector in task_delta))):
                vectors = []
                for coefficient, vector in zip(coefficients, task_delta):
                    if key not in vector:
                        continue
                    values = np.asarray(vector[key], dtype=np.float64)
                    if method == "dare_ties":
                        values = np.where(rng.random(values.size) < density, values / density, 0.0)
                    keep = max(1, math.ceil(values.size * density))
                    magnitudes = np.abs(values)
                    threshold = np.partition(magnitudes, values.size - keep)[values.size - keep]
                    retained = magnitudes > threshold
                    # Include only the required number of tied coordinates,
                    # in stable index order, so density keeps its exact meaning.
                    tied = np.flatnonzero(magnitudes == threshold)
                    retained[tied[: keep - int(retained.sum())]] = True
                    vectors.append((coefficient, np.where(retained, values, 0.0)))
                elected = np.sign(sum(coefficient * vector for coefficient, vector in vectors))
                numerator = np.zeros(elected.shape, dtype=np.float64)
                denominator = np.zeros(elected.shape, dtype=np.float64)
                for coefficient, vector in vectors:
                    aligned = (np.sign(vector) == elected) & (vector != 0)
                    numerator += coefficient * np.where(aligned, vector, 0.0)
                    denominator += coefficient * aligned
                merged_delta[key] = np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
                ).tolist()
        elif method == "dare":
            from aether.compiler.stage2_optimizer.pass12_model_merging import _DAREMerger

            merged_delta = _DAREMerger(drop_prob=1.0 - density).merge(
                task_delta, coefficients, self.config
            )
        else:
            merged_delta = _get_merger(method).merge(task_delta, coefficients, self.config)
        if not merged_delta:
            raise AetherRuntimeError("task vector sources have no overlapping tensor names")
        merged_weights = _apply_delta(base_weights, merged_delta)
        if any(
            not np.isfinite(np.asarray(values, dtype=np.float32)).all()
            for values in merged_weights.values()
        ):
            raise ValueError("Merge overflowed the FP32 output format")
        record = {
            "format": "aether_merge_manifest_v1",
            "method": method,
            "density": density,
            "sources": [item["path"] for item in task_vectors],
            "coefficients": coefficients,
            "tensor_count": len(merged_delta),
            "n_sources": len(task_vectors),
            "n_tensors_updated": len(merged_delta),
            "precision_policy": "preserve_base_storage_formats_and_block_sizes",
            "auxiliary_policy": "preserve_base_requires_reevaluation",
            "quality_validated": False,
            "seed": 42 if method in {"dare", "dare_ties", "free", "evolutionary"} else None,
            "density_policy": "drop_then_per_tensor_trim"
            if method == "dare_ties"
            else (
                "retained_probability"
                if method == "dare"
                else "per_tensor_top_magnitude"
                if method == "ties"
                else "not_applicable"
            ),
            "coefficient_policy": "initial_for_l2_proxy_search"
            if method in {"free", "evolutionary"}
            else "applied",
            "task_vector_policy": "raw_source_minus_original_base; runtime reweighting adds deltas to the merged model",
        }

        class MergedTensors(Mapping):
            def __len__(self):
                return len(entries)

            def __iter__(self):
                return iter(entries)

            def __getitem__(self, name):
                entry = entries[name]
                if name not in merged_delta:
                    return store.load_tensor(name)
                return quantize_tensor(
                    np.asarray(merged_weights[name], dtype=np.float32).reshape(entry.shape),
                    entry.precision,
                    block_size=entry.block_size,
                )

        # Only a verified staged artifact is published. All mutation, optional
        # payload preservation and cleanup occur under the base's parent, so
        # exceptions cannot leave a partial output at the public model path.
        with tempfile.TemporaryDirectory(
            prefix=".aether-runtime-merge-", dir=output_root.parent
        ) as temporary:
            staging = Path(temporary) / "merged.aeg"
            shutil.copytree(base.root, staging)
            merged = load_aeg_package(staging)
            merged.verify_integrity()
            invalidate_weight_dependent_evidence(merged)
            merged.manifest.model_id = f"{base.manifest.model_id or model_id}+merged"
            merged.manifest.compiled_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if merged.manifest.format_version not in {"AEG/2.0", "AEG/3.0"}:
                merged.manifest.format_version = "AEG/2.0"
            merged.metadata["model_merge"] = record
            merged.metadata["model_merging"] = record
            passes = list(merged.metadata.get("optimizer_passes", []))
            if "model_merging" not in passes:
                passes.append("model_merging")
            merged.metadata["optimizer_passes"] = passes
            merged.metadata.setdefault("pass_reports", {})["model_merging"] = {
                "status": "applied",
                "details": record,
            }
            task_vector_root = staging / "merging" / "task_vectors"
            task_vector_root.mkdir(parents=True, exist_ok=True)
            descriptors = []
            for index, (name, delta) in enumerate(zip(names, task_delta)):
                safe_name = (
                    re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:80] or f"task_{index}"
                )
                file_name = f"{index:03d}_{safe_name}.npz"
                arrays = {}
                tensor_descriptors = []
                for tensor_index, (tensor_name, values) in enumerate(sorted(delta.items())):
                    key = f"tensor_{tensor_index:06d}"
                    array = np.asarray(values, dtype=np.float32)
                    if not np.isfinite(array).all():
                        raise ValueError(
                            f"task vector {name!r} overflowed its persisted FP32 format"
                        )
                    arrays[key] = array
                    tensor_descriptors.append(
                        {"name": tensor_name, "key": key, "shape": list(entries[tensor_name].shape)}
                    )
                np.savez_compressed(task_vector_root / file_name, **arrays)
                descriptors.append(
                    {
                        "name": name,
                        "path": f"merging/task_vectors/{file_name}",
                        "tensors": tensor_descriptors,
                    }
                )
            merged.metadata["task_vectors"] = {
                "format": "aether_task_vectors_v1",
                "vectors": descriptors,
            }
            (staging / "merging" / "merge_manifest.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            merged.metadata["weight_bytes"] = WeightStore(staging / "weights" / "quantized").save(
                MergedTensors()
            )
            merged.weights = {}
            merged._weight_store = None  # noqa: SLF001 - staged payload was replaced
            merged.save()
            load_aeg_package(staging).verify_integrity()
            if output_root.exists():
                raise AetherRuntimeError(
                    f"merge output already exists: {output_root}; remove it explicitly first"
                )
            from aether.utils.artifact_publication import publish_staged_directory

            publish_staged_directory(staging, output_root)
        return {
            "model": model_id,
            "output_model": str(output_root),
            "method": method,
            "task_count": len(task_vectors),
            "density": density,
            "tensor_count": len(merged_delta),
            "quality_validated": False,
            "status": "merged",
        }

    # ── v5.0 Runtime Extensions (PRD v5.0) ────────────────────────────────────

    def _init_v5_layers(self, aeg_path: str | None = None) -> None:
        """Initialize v5.0 runtime layers: R9 diffusion spec, R11 semantic cache, R12 CXL pool."""
        # R9 Diffusion Speculative Engine
        if not self.config.enable_diffusion_spec:
            self._diffusion_engine = None
        elif not hasattr(self, "_diffusion_engine"):
            try:
                from aether.runtime.r9_diffusion_spec_engine import DiffusionSpecEngine
                vocab_size = getattr(self.config, "vocab_size", 128000)
                self._diffusion_engine = DiffusionSpecEngine(
                    vocab_size=vocab_size,
                    use_adaptive_scheduling=True,
                )
                # A drafter is executable only when its persisted weights
                # load successfully.  Keep ordinary AR serving on the normal
                # path instead of retaining an engine that can only report a
                # synthetic/fallback draft.
                draft_root = Path(aeg_path) / "graph" if aeg_path else None
                has_draft_payload = bool(
                    draft_root
                    and (
                        (draft_root / "mdlm_draft_head_config.json").is_file()
                        or (draft_root / "mdlm_draft_head.npz").is_file()
                        or (draft_root / "mdlm_draft_head.safetensors").is_file()
                    )
                )
                if not has_draft_payload or not self._diffusion_engine.load_from_aeg(aeg_path):
                    self._diffusion_engine = None
                    logger.info("R9: MDLM drafter unavailable; using autoregressive decoding")
                else:
                    logger.info("R9: DiffusionSpecEngine initialized with executable draft head")
            except Exception as exc:
                self._diffusion_engine = None
                logger.warning(f"R9 init failed: {exc}")

        # R11 Semantic Request Cache
        if self.config.enable_semantic_cache and not hasattr(self, "_semantic_cache"):
            try:
                from aether.runtime.r11_semantic_kv_cache import SemanticRequestCache
                threshold = getattr(self.config, "semantic_cache_threshold", 0.92)
                persist_path = None
                if hasattr(self.config, "model_cache_dir") and self.config.model_cache_dir:
                    from pathlib import Path as _Path
                    persist_path = str(_Path(self.config.model_cache_dir) / "semantic_cache.json")
                self._semantic_cache = SemanticRequestCache(
                    similarity_threshold=threshold,
                    max_entries=int(getattr(self.config, "semantic_cache_size", 100_000)),
                    persist_path=persist_path,
                )
                logger.info("R11: SemanticRequestCache initialized")
            except Exception as exc:
                self._semantic_cache = None
                logger.warning(f"R11 init failed: {exc}")

        # R12 CXL Rack-Scale KV Pool
        if not hasattr(self, "_cxl_pool"):
            try:
                from aether.runtime.r12_cxl_kv_pool import CXLRackScaleKVPool
                pool_size_gb = getattr(self.config, "cxl_pool_size_gb", 0.0)
                if pool_size_gb > 0:
                    emulated_path = None
                    if hasattr(self.config, "model_cache_dir") and self.config.model_cache_dir:
                        from pathlib import Path as _Path
                        emulated_path = str(_Path(self.config.model_cache_dir) / "cxl_pool.bin")
                    self._cxl_pool = CXLRackScaleKVPool(
                        pool_size_gb=pool_size_gb,
                        emulated_path=emulated_path,
                    )
                    logger.info(f"R12: CXLRackScaleKVPool initialized ({pool_size_gb} GB)")
                else:
                    self._cxl_pool = None
            except Exception as exc:
                self._cxl_pool = None
                logger.warning(f"R12 init failed: {exc}")

    def _create_grammar_session(
        self,
        model_id: str,
        grammar: str | None = None,
        schema: dict | None = None,
        regex: str | None = None,
    ) -> Any:
        """Validate a persisted tokenizer-aware constraint and create a session."""
        if sum(value is not None for value in (grammar, schema, regex)) != 1:
            raise AetherRuntimeError("Provide exactly one of grammar, schema, or regex")
        backend = self._load_model(model_id)
        if self.grammar_engine is None:
            raise AetherRuntimeError(
                "No trusted tokenizer-aware grammar FSM is loaded; compile the requested constraint with a tokenizer-aware grammar backend first"
            )
        source = grammar
        if schema is not None:
            source = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        assert source is not None
        loaded = getattr(backend, "_models", {}).get(model_id)
        tokenizer = getattr(loaded, "tokenizer", None)
        if not self.grammar_engine.matches_compiled_constraint(source, tokenizer=tokenizer):
            raise AetherRuntimeError(
                "The requested constraint does not match a trusted tokenizer-aware grammar compiled into this AEG"
            )
        if not backend.supports("grammar_constraints"):
            raise AetherRuntimeError(
                f"Backend {backend.name!r} does not implement decode-time grammar constraints"
            )
        return self.grammar_engine.create_session()

    def generate_constrained(
        self,
        model_id: str,
        prompt: str | None = None,
        grammar: str | None = None,
        schema: dict | None = None,
        regex: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate text constrained to a grammar, JSON schema, or regex pattern.

        Uses R3 GrammarFSMEngine to mask invalid tokens at every decode step,
        guaranteeing 100% syntactically valid output (<50µs overhead per step).

        Args:
            model_id: Model identifier.
            prompt: Text prompt.
            grammar: EBNF/ABNF/GBNF grammar string (XGrammar format).
            schema: JSON schema dict for structured JSON output.
            regex: Regex pattern for constrained generation.
            kwargs: Additional generation parameters.

        Returns:
            GenerationResponse with guaranteed valid output.
        """
        kwargs["grammar_session"] = self._create_grammar_session(
            model_id, grammar=grammar, schema=schema, regex=regex
        )
        return self.generate(model_id, prompt, messages=messages, **kwargs)

    def generate_constrained_stream(
        self,
        model_id: str,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        grammar: str | None = None,
        schema: dict | None = None,
        regex: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream tokenizer-constrained output through the real decode path."""
        kwargs["grammar_session"] = self._create_grammar_session(
            model_id, grammar=grammar, schema=schema, regex=regex
        )
        return self.generate_stream(
            model_id,
            prompt,
            messages=messages,
            **kwargs,
        )

    def grpo_train_step(
        self,
        model_id: str | None = None,
        prompts: list[str] | None = None,
        group_size: int = 8,
        domain: str = "math",
        learning_rate: float = 1e-6,
        clip_ratio: float = 0.2,
        max_tokens: int = 2048,
        *,
        model: str | None = None,
        verifier_domain: str | None = None,
        ground_truths: list[str] | None = None,
        test_suites: list[str] | None = None,
        model_forward_fn: Callable[..., Any] | None = None,
        optimizer_step_fn: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one GRPO (Group Relative Policy Optimization) training step.

        Implements DeepSeek-R1 GRPO post-training on top of a compiled AEG model.
        For each prompt:
          1. Sample G=group_size responses from the current policy
          2. Score responses with a rule-based verifier (RLVR)
          3. Compute group-relative advantages: A_i = (r_i - mean(r)) / std(r)
          4. Apply clipped policy gradient update (PPO-style)

        This is fully on-device: no external reward model, no human labels.

        Research basis: GRPO (DeepSeek arXiv 2025), RLVR (DeepSeek-R1 2025),
        K2V (arXiv 2026), Flow-GRPO (arXiv 2026).

        Args:
            model_id: Base policy model AEG path.
            prompts: Batch of training prompts.
            group_size: G in GRPO — number of responses per prompt.
            domain: Verifier domain ('math', 'code', 'logic', 'general').
            learning_rate: Policy gradient learning rate.
            clip_ratio: PPO clip ratio ε.
            max_tokens: Max tokens per response sample.

        Returns:
            Training step report with loss, mean reward, and policy update stats.
        """
        model_id = model_id or model
        prompts = prompts or []
        if verifier_domain is not None:
            domain = verifier_domain
        if not model_id:
            raise ValueError("grpo_train_step requires model or model_id")
        if not prompts:
            raise ValueError("grpo_train_step requires at least one prompt")
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")

        if model_forward_fn is None or optimizer_step_fn is None:
            return {
                "status": "failed",
                "error": (
                    "GRPO training requires explicit model_forward_fn and optimizer_step_fn "
                    "callbacks backed by a gradient-capable policy; inference-only Runtime "
                    "refuses to claim a weight update"
                ),
                "prompts": len(prompts),
                "group_size": group_size,
                "domain": domain,
            }

        aeg_path = self._resolve_aeg_path(model_id)
        if aeg_path is None:
            raise ModelNotFoundError(f"GRPO model AEG not found: {model_id}")
        rlvr_config = Path(aeg_path) / "training" / "rlvr_config.json"
        if not rlvr_config.is_file():
            raise AetherRuntimeError(
                "GRPO requires an AEG compiled with the RLVR verifier pass; "
                "training/rlvr_config.json is missing"
            )
        from aether.runtime.r12_rlvr_harness import RLVRTrainingHarness

        harness = RLVRTrainingHarness(
            rlvr_config_path=str(rlvr_config),
            model_forward_fn=model_forward_fn,
            optimizer_step_fn=optimizer_step_fn,
        )
        harness._K = group_size  # caller-selected group size is validated above
        results: list[dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            result = harness.train_step(
                prompt=prompt,
                ground_truth=(ground_truths[index] if ground_truths and index < len(ground_truths) else None),
                test_suite=(test_suites[index] if test_suites and index < len(test_suites) else None),
                max_new_tokens=max_tokens,
            )
            results.append(
                {
                    "rewards": result.rewards,
                    "advantages": result.advantages,
                    "loss": result.loss,
                    "grad_norm": result.grad_norm,
                    "pass_at_k": result.pass_at_k,
                    "elapsed_ms": result.elapsed_ms,
                }
            )
        return {
            "status": "ok",
            "prompts": len(results),
            "group_size": group_size,
            "domain": domain,
            "mean_loss": sum(item["loss"] for item in results) / max(1, len(results)),
            "mean_pass_at_k": sum(item["pass_at_k"] for item in results) / max(1, len(results)),
            "optimizer_steps": len(results),
            "training_backend": "caller_supplied_policy_and_optimizer",
            "steps": results,
            "harness": harness.summary(),
        }

    @runtime_operation
    def generate_video(
        self,
        model_id: str | None = None,
        video_path: str | None = None,
        prompt: str | None = None,
        compression: str = "stc",
        max_visual_tokens: int = 4096,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate a response about a video using Pass 20 STC/STORM compression.

        Compresses the video's visual tokens (>75% reduction) before feeding to
        the VLM model, enabling long-video understanding without OOM.

        Args:
            model_id: Video VLM model AEG path.
            video_path: Path to video file (.mp4, .avi, .mov, etc.).
            prompt: Text question/instruction about the video.
            compression: Compression strategy ('stc', 'storm', 'streaming_tom').
            max_visual_tokens: Maximum visual tokens to keep after compression.
            kwargs: Additional generation parameters.

        Returns:
            GenerationResponse with video understanding output.

        Research basis: STC CVPR 2026, STORM arXiv 2026, Mage-VL 2026.
        """
        model_id = model_id or model
        if not model_id or not video_path or not prompt:
            raise ValueError("generate_video requires model, video_path, and prompt")
        video = Path(video_path)
        if not video.is_file():
            raise FileNotFoundError(f"video input does not exist: {video_path}")
        # Dispatch only to a real vision backend.  The CPU AEG engine does not
        # contain a pixel decoder/vision tower, so it must remain fail-closed;
        # an installed VLM backend may implement this method and advertise the
        # capability explicitly.
        try:
            backend = self._load_model(model_id)
        except ModelNotFoundError as exc:
            raise ModelNotFoundError(
                f"video generation cannot start because the requested video AEG is unavailable: {exc}",
                model_id=model_id,
            ) from exc
        generator = getattr(backend, "generate_video", None)
        supports_vision = getattr(backend, "supports", lambda _name: False)("vision")
        if callable(generator) and supports_vision:
            result = generator(
                model_id,
                str(video),
                prompt,
                compression=compression,
                max_visual_tokens=max_visual_tokens,
                **kwargs,
            )
            if isinstance(result, GenerationResponse):
                return result
            raise AetherRuntimeError(
                f"Vision backend {backend.name!r} returned an unsupported video result type"
            )
        raise AetherRuntimeError(
            f"video generation requires a runtime VLM/video encoder backend; "
            f"backend {backend.name!r} does not advertise executable vision support"
        )

    def semantic_cache_stats(self) -> dict[str, Any]:
        """Return semantic request cache statistics (R11).

        Returns:
            Stats dict with hit_rate, total_requests, tokens_saved, etc.
        """
        self._init_v5_layers()
        cache = getattr(self, "_semantic_cache", None)
        if cache is None:
            return {"enabled": False, "reason": "R11 SemanticRequestCache not initialized"}
        return {"enabled": True, **cache.stats()}

    def kv_transfer_stats(self) -> dict[str, Any]:
        """Return KV network transfer statistics (R10 NIKA+NIXL).

        Returns:
            Stats dict with NIKA policy results and transfer metrics.
        """
        # The CPU manager can measure local tier movements.  Network/RDMA
        # engines are not installed on this host and must remain explicit
        # unavailable rather than being represented by synthetic numbers.
        stats = self.kv_cache.get_transfer_stats()
        return {
            "enabled": False,
            "network_available": False,
            "network_backend": None,
            "network_scope": "rdma_nixl_uccl",
            "hub_configured": self.kv_cache.hub_client is not None and bool(self.kv_cache.hub_namespace),
            "hub_namespace": self.kv_cache.hub_namespace,
            "fallback_backend": "local_tier_cache",
            "fallback_active": True,
            "research_basis": "NIKA SCITEPRESS 2026 + NIXL NVIDIA 2026",
            **stats,
        }

    def cxl_pool_status(self) -> dict[str, Any]:
        """Return CXL rack-scale KV pool status (R12).

        Returns:
            Pool stats dict with utilization, hit_rate, block_count, etc.
        """
        self._init_v5_layers()
        pool = getattr(self, "_cxl_pool", None)
        if pool is None:
            return {"enabled": False, "reason": "R12 CXL pool not configured (cxl_pool_size_gb=0)"}
        return {"enabled": True, **pool.get_stats()}

    def eval_gate(
        self,
        model_id: str | None = None,
        domain: str = "general",
        num_examples: int = 100,
        quality_threshold: float = 0.98,
        *,
        model: str | None = None,
        benchmarks: list[str] | None = None,
        baseline_model: str | None = None,
        max_regression: float | None = None,
        evaluator: Any | None = None,
        baselines: dict[str, float] | None = None,
    ) -> EvalGateReport:
        """Run a quality evaluation gate before deployment.

        Evaluates the compiled model against a domain benchmark. If quality drops
        below quality_threshold vs BF16 baseline, returns fail=True.

        Args:
            model_id: Compiled AEG model to evaluate.
            domain: Benchmark domain ('math', 'code', 'general', 'reasoning').
            num_examples: Number of eval examples to run.
            quality_threshold: Min relative quality (0.98 = 2% max regression).

        Returns:
            Eval report with pass/fail, quality_score, and detailed metrics.
        """
        model_id = model_id or model
        if not model_id:
            raise ValueError("eval_gate requires model or model_id")
        if benchmarks:
            domain = ",".join(benchmarks)
        if max_regression is not None:
            if not 0 <= max_regression <= 1:
                raise ValueError("max_regression must be between 0 and 1")
            quality_threshold = 1.0 - max_regression

        if evaluator is not None:
            if not callable(evaluator):
                raise TypeError("evaluator must be callable")
            from aether.observability.ci_pipeline import CIEvalPipeline

            requested = tuple(benchmarks or ([domain] if domain in {"hellaswag", "mmlu", "gsm8k", "math-500", "humaneval", "aime"} else []))
            if not requested:
                raise ValueError(
                    "A configured evaluator requires benchmark names such as "
                    "hellaswag, mmlu, gsm8k, math-500, humaneval, or aime"
                )
            pipeline = CIEvalPipeline(
                aeg_path=model_id,
                max_regression=1.0 - quality_threshold,
                required_benchmarks=requested,
                evaluator=evaluator,
            )
            quality = pipeline.run(list(requested), baselines=baselines)
            payload = quality.to_dict()
            payload.update(
                {
                    "model_id": model_id,
                    "domain": domain,
                    "status": "passed" if quality.gate_decision.passed else "failed",
                    "passed": quality.gate_decision.passed,
                    "quality_threshold": quality_threshold,
                    "num_examples_requested": num_examples,
                    "baseline_model": baseline_model,
                }
            )
            return EvalGateReport(payload)

        # A generation being non-empty is not a benchmark score.  The previous
        # implementation accepted artifacts on that proxy, which could allow a
        # severely regressed model through the PRD quality gate.  Real dataset
        # evaluators are supplied through CIEvalPipeline/BenchmarkRunner; this
        # public convenience method must fail closed until one is configured.
        return EvalGateReport(
            model_id=model_id,
            domain=domain,
            num_examples_requested=num_examples,
            quality_threshold=quality_threshold,
            benchmarks=benchmarks or [domain],
            baseline_model=baseline_model,
            passed=False,
            status="unavailable",
            reason=(
                "No real benchmark evaluator is configured. Configure a dataset evaluator "
                "and measured baseline/candidate scores before accepting this artifact."
            ),
        )

    def _get_eval_prompts(self, domain: str, n: int) -> list[str]:
        """Return domain-specific evaluation prompts."""
        prompts_by_domain: dict[str, list[str]] = {
            "math": [
                "What is 15% of 240?",
                "Solve: 3x + 7 = 22. What is x?",
                "What is the area of a circle with radius 5?",
                "If a train travels 60 mph for 2.5 hours, how far does it go?",
                "What is the derivative of x^3 + 2x?",
            ],
            "code": [
                "Write a Python function to reverse a string.",
                "What is a binary search tree?",
                "Write a SQL query to find duplicate rows.",
                "Explain what a decorator is in Python.",
                "Write a function to check if a number is prime.",
            ],
            "general": [
                "What is the capital of France?",
                "Explain what photosynthesis is in one sentence.",
                "What year did World War II end?",
                "What is the speed of light?",
                "Name three programming languages.",
            ],
            "reasoning": [
                "If all cats are animals and some animals are domestic, are some cats domestic?",
                "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
                "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
                "What comes next: 2, 4, 8, 16, ?",
                "If John is taller than Mary and Mary is taller than Sue, who is tallest?",
            ],
        }
        domain_prompts = prompts_by_domain.get(domain, prompts_by_domain["general"])
        # Repeat to fill n if needed
        result = []
        while len(result) < n:
            result.extend(domain_prompts)
        return result[:n]

    def ab_rollout(
        self,
        model_a: str,
        model_b: str,
        prompt: str = "",
        traffic_split_pct: int = 50,
        *,
        traffic_split: float | None = None,
        auto_rollout: bool = False,
        rollback_on_regression: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run an A/B traffic split between two compiled models.

        Routes the request to model_a or model_b based on traffic_split_pct.
        Returns both responses for comparison in development mode.

        Args:
            model_a: First model AEG path (control).
            model_b: Second model AEG path (experiment).
            prompt: Prompt to evaluate.
            traffic_split_pct: Percentage of traffic to route to model_a (0-100).
            kwargs: Additional generation parameters.

        Returns:
            Dict with model_a_response, model_b_response, and routing decision.
        """
        if traffic_split is not None:
            if not 0.0 <= traffic_split <= 1.0:
                raise ValueError("traffic_split must be between 0 and 1")
            traffic_split_pct = int(round(traffic_split * 100))
        if not 0 <= traffic_split_pct <= 100:
            raise ValueError("traffic_split_pct must be between 0 and 100")
        import random
        route_to_a = random.randint(1, 100) <= traffic_split_pct

        resp_a = self.generate(model_a, prompt, **kwargs)
        resp_b = self.generate(model_b, prompt, **kwargs)

        return {
            "traffic_split_pct": traffic_split_pct,
            "traffic_split": traffic_split_pct / 100.0,
            "auto_rollout": auto_rollout,
            "rollback_on_regression": rollback_on_regression,
            "route_to_a": route_to_a,
            "served_model": model_a if route_to_a else model_b,
            "model_a": {
                "model_id": model_a,
                "text": resp_a.text,
                "metrics": resp_a.metrics.to_dict(),
                "usage": resp_a.usage,
            },
            "model_b": {
                "model_id": model_b,
                "text": resp_b.text,
                "metrics": resp_b.metrics.to_dict(),
                "usage": resp_b.usage,
            },
        }

    def _release_agentic_kv(self, model_id: str, session_id: str) -> None:
        """Release backend-owned KV state for a closed agentic session."""
        backend = self._loaded_backends.get(model_id)
        releaser = getattr(backend, "release_session_cache", None)
        if callable(releaser):
            try:
                releaser(model_id, session_id)
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask close
                logger.warning("agentic KV cleanup failed", session_id=session_id, error=str(exc))

    def agentic_session(
        self,
        model_id: str,
        system: str = "",
    ) -> _AgenticRuntimeSession:
        """Return an async multi-turn session backed by this Runtime.

        The method matches the v3.1 SDK contract.  Conversation history is
        preserved and each turn uses the real model backend; backends that do
        not expose reusable KV tensors are explicitly reported as
        ``kv_reuse=False`` in response metrics.
        """
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("agentic_session requires a model identifier")
        if not isinstance(system, str):
            raise ValueError("agentic_session system prompt must be a string")
        return _AgenticRuntimeSession(self, model_id, system)

    def multi_agent_session(
        self,
        models: list[str] | str | int | None = None,
        coordination: str = "relay",
        agent_count: int | str | None = None,
        shared_prefix: str = "",
        model_id: str = "",
    ) -> _MultiAgentSessionHandle:
        """Create a real R2 session usable as an async context manager.

        The PRD form is ``multi_agent_session(models=[...],
        coordination="relay")``.  The historical positional form used by the
        REST endpoint (``agent_count, shared_prefix, model``) is recognized as
        well and returns the same dictionary-compatible handle.
        """
        # Backward-compatible interpretation of the old three-positional API.
        if isinstance(models, int):
            old_count = models
            old_shared_prefix = coordination
            old_model = agent_count if isinstance(agent_count, str) else model_id
            models = [old_model] if old_model else []
            coordination = "relay"
            agent_count = old_count
            shared_prefix = old_shared_prefix
        elif isinstance(models, str):
            models = [models]
        elif models is None:
            models = [model_id] if model_id else []

        if agent_count is None:
            initial_count = 0
        elif isinstance(agent_count, int) and agent_count >= 0:
            initial_count = agent_count
        else:
            raise ValueError("agent_count must be a non-negative integer")
        return _MultiAgentSessionHandle(
            self,
            [model for model in models if isinstance(model, str) and model],
            coordination,
            initial_count,
            shared_prefix,
        )

    def __repr__(self) -> str:
        return (
            f"Runtime(target={self.fingerprint.target_id}, "
            f"backends={self.backend_registry.get_available_backend_names()}, "
            f"loaded_models={list(self._loaded_models.keys())}, "
            f"v5_layers={[l for l in ['_diffusion_engine','_semantic_cache','_cxl_pool'] if getattr(self, l, None)]!r})"
        )
