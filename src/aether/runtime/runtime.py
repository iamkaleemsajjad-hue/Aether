"""
Aether Runtime — the main execution engine.

The Runtime is the primary Python API. It loads compiled AEG artifacts, detects
hardware, selects the best backend, manages the KV cache, runs speculative
decoding, and serves generation requests.
"""

from __future__ import annotations

import datetime
import json
import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

from aether.backends.base import Backend, GenerationRequest, GenerationResult
from aether.backends.registry import BackendRegistry
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.core.constants import DEFAULT_SERVER_PORT
from aether.core.exceptions import BackendNotAvailableError, ModelNotFoundError, RuntimeError as AetherRuntimeError
from aether.core.types import HardwareTarget
from aether.runtime.config import RuntimeConfig
from aether.runtime.hardware import HardwareDetector
from aether.runtime.kv_cache import KVCacheManager
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
    """Active precision string (e.g., 'mixed_fp8_q4')."""

    spec_accept_rate: float | None = None
    """Speculative decoding acceptance rate."""

    kv_cache_hit_rate: float | None = None
    """KV cache prefix hit rate."""

    memory_pressure: float = 0.0
    """Current VRAM utilization (0.0 - 1.0)."""

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
        self.kv_cache = KVCacheManager(
            dtype=self.config.kv_cache_dtype,
            cpu_budget_gb=self.config.kv_cache_cpu_gb,
            nvme_budget_gb=self.config.kv_cache_nvme_gb,
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
        self._aeg_packages: dict[str, AEGPackage] = {}
        self._lock = threading.RLock()
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
    ) -> Backend:
        """Resolve and return the best available backend for the target."""
        with self._lock:
            target = target_id or self.fingerprint.target_id
            profile = HardwareProfile.from_target_id(target) or HardwareProfile.auto()
            explicit_backend = self.config.backend_name is not None
            is_onnx_model = bool(
                model_id and Path(model_id).suffix.lower() == ".onnx"
            )
            if self.config.backend_name:
                backend = self.backend_registry.get_backend(self.config.backend_name)
                if backend is not None and backend.available_for_target(profile.target_id):
                    return backend
                msg = (
                    f"Configured backend '{self.config.backend_name}' is not available "
                    f"for target {profile.target_id}"
                )
                raise BackendNotAvailableError(msg, backend_name=self.config.backend_name)
            # Prefer cached backend
            if profile.target_id in self._loaded_backends:
                cached = self._loaded_backends[profile.target_id]
                if cached.name != "onnx" or is_onnx_model or explicit_backend:
                    return cached

            candidates = profile.backend_candidates
            for backend_name in candidates:
                if backend_name in {"onnx", "onnxruntime"} and not is_onnx_model:
                    continue
                backend = self.backend_registry.get_backend(backend_name)
                if backend is not None and backend.available_for_target(profile.target_id):
                    self._loaded_backends[profile.target_id] = backend
                    return backend
            # A backend for another device is not a valid fallback.
            for backend in self.backend_registry.get_available_backends():
                if backend.name == "onnx" and not is_onnx_model:
                    continue
                if backend.available_for_target(profile.target_id):
                    self._loaded_backends[profile.target_id] = backend
                    return backend

            msg = f"No backend available for target {target}"
            raise BackendNotAvailableError(msg, target_id=target)

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
                or requested_path.parent != Path(".")
            ):
                raise ModelNotFoundError(
                    f"AEG artifact does not exist or has no manifest: {model_id}",
                    model_id=model_id,
                )
            if aeg_path is not None:
                # An AEG is a portable graph/weight container, not a license
                # to execute an incompatible target variant.  Validate the
                # current target before backend selection so a GPU/accelerator
                # request cannot silently fall back to the CPU reference path.
                portable_pytorch = False
                try:
                    from aether.core.aeg_format import AEGPackage

                    package = AEGPackage(Path(aeg_path)).load()
                    portable_pytorch = package.supports_portable_backend("pytorch")
                    if not package.supports_runtime_target(self.fingerprint.target_id):
                        compiled_targets = list(package.manifest.kernels.targets) if package.manifest else []
                        raise BackendNotAvailableError(
                            "AEG artifact has no executable variant for the detected target "
                            f"{self.fingerprint.target_id!r}; compiled variants: {compiled_targets}. "
                            "Compile the same source model for this target or include both targets "
                            "in one AEG package.",
                            target_id=self.fingerprint.target_id,
                        )
                except BackendNotAvailableError:
                    raise
                except Exception as exc:
                    raise AetherRuntimeError(
                        f"Unable to validate AEG target compatibility for {aeg_path}: {exc}"
                    ) from exc
            # A local ONNX file has an explicit execution contract. Do not
            # route it to the generic PyTorch backend merely because that
            # backend is available on the host.
            if requested_path.suffix.lower() == ".onnx":
                backend = self.backend_registry.get_backend("onnx")
                if backend is None or not backend.is_available():
                    raise BackendNotAvailableError(
                        "ONNX Runtime is required to execute .onnx models",
                        backend_name="onnx",
                    )
            elif aeg_path is not None and self.config.backend_name is None:
                # A packaged AEG is already the compiler's executable
                # contract. On CPU, prefer Aether's framework-free engine so a
                # clean base installation does not route a native artifact
                # through an optional PyTorch wrapper.
                target = self.fingerprint.target_id
                if target.startswith("cpu_"):
                    native = self.backend_registry.get_backend("aether_cpu")
                    if native is None or not native.is_available():
                        raise BackendNotAvailableError(
                            "The base installation could not load its native CPU backend",
                            backend_name="aether_cpu",
                            target_id=target,
                        )
                    backend = native
                else:
                    # A portable PyTorch AEG must stay on the backend that
                    # implements its graph/weight contract.  Selecting vLLM
                    # or another frontend here would either ignore the AEG or
                    # silently execute a different model path.
                    if portable_pytorch and self.fingerprint.target_id.startswith(("cuda_", "rocm_", "metal_")):
                        backend = self.backend_registry.get_backend("pytorch")
                        if backend is None or not backend.available_for_target(self.fingerprint.target_id):
                            raise BackendNotAvailableError(
                                "portable AEG execution requires an installed PyTorch backend "
                                f"for target {self.fingerprint.target_id}",
                                backend_name="pytorch",
                                target_id=self.fingerprint.target_id,
                            )
                    else:
                        backend = self._resolve_backend(model_id=model_id)
            else:
                backend = self._resolve_backend(model_id=model_id)
            self._loaded_models[model_id] = backend.load_model(
                model_id,
                aeg_path,
                offline=self.config.hf_offline,
                download_timeout_s=self.config.model_download_timeout_s,
                trust_remote_code=self.config.allow_remote_code,
                execution_devices=self.config.execution_devices,
            )
            self._loaded_backends[model_id] = backend
            # Optional v4/v5 layers are initialized at the same reachability
            # boundary as model loading.  They remain feature-gated by the
            # artifact manifest, but no longer exist only in isolated stats
            # helpers while inference bypasses them.
            if aeg_path is not None:
                self._init_v4_layers(aeg_path)
                self._init_v5_layers(aeg_path)
            return backend

    def _resolve_aeg_path(self, model_id: str) -> str | None:
        """Find the AEG package path for a model, downloading/compile if needed."""
        from aether.utils.file_io import aether_cache_dir

        path_candidate = Path(model_id)
        if path_candidate.exists() and (path_candidate / "manifest.json").exists():
            return str(path_candidate.resolve())

        cache_root = aether_cache_dir(self.config.model_cache_dir)
        aeg_path = cache_root / "models" / safe_model_id_path(model_id)
        if aeg_path.exists():
            return str(aeg_path)
        # Check local compiled path
        local = resolve_model_path(model_id, self.config.model_cache_dir)
        if local and (local / "manifest.json").exists():
            return str(local)
        return None

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
        from aether import Compiler
        compiler = Compiler()
        aeg = compiler.compile(
            model_id,
            output_path=self._resolve_aeg_path(model_id)
            or aether_cache_dir(self.config.model_cache_dir) / "models" / safe_model_id_path(model_id),
        )
        aeg.save()

    def list(self) -> list[str]:
        """Return a list of model IDs with cached AEG artifacts."""
        from aether.utils.file_io import aether_cache_dir

        cache_root = aether_cache_dir(self.config.model_cache_dir)
        model_dir = cache_root / "models"
        if not model_dir.exists():
            return []
        return sorted(d.name for d in model_dir.iterdir() if (d / "manifest.json").exists())

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

    def remove(self, model_id: str) -> None:
        """Remove a model from the local AEG cache."""
        from aether.utils.file_io import delete_model

        delete_model(model_id, self.config.model_cache_dir)
        with self._lock:
            self._loaded_models.pop(model_id, None)
            self._loaded_backends.pop(model_id, None)
        logger.info(f"Removed model {model_id} from cache")

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
        cache_bypass = bool(kwargs.pop("cache_bypass", False))
        cache_config = {
            "max_new_tokens": max_tokens or self.config.default_max_tokens,
            "temperature": temperature if temperature is not None else self.config.default_temperature,
            "top_p": top_p if top_p is not None else self.config.default_top_p,
            "top_k": top_k,
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
                            kernel_target=self.fingerprint.target_id,
                            backend_name="semantic_cache",
                            extra={"cache_hit": True, "similarity": cached.similarity_score},
                        ),
                    )
        backend = self._load_model(model_id)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens or self.config.default_max_tokens,
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
        aeg_path_for_request = self._resolve_aeg_path(model_id)
        peagle_engine = (
            self._peagle_engines.get(str(Path(aeg_path_for_request).resolve()))
            if aeg_path_for_request is not None
            else self.peagle_engine
        )
        if peagle_engine is not None:
            request.extra.setdefault("peagle_engine", peagle_engine)
        task_weights = getattr(self, "_task_weights_by_model", {}).get(model_id)
        if task_weights is not None:
            request.extra.setdefault("task_weights", dict(task_weights))
        start = datetime.datetime.now(datetime.timezone.utc)
        start_ns = time.time_ns()
        try:
            result = backend.generate(request)
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
            eagle_accept_rate=self.kv_cache.hit_rate(),
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
            kernel_target=self.fingerprint.target_id,
            active_precision="mixed",  # Could be refined from AEG
            spec_accept_rate=self.kv_cache.hit_rate(),
            kv_cache_hit_rate=self.kv_cache.hit_rate(),
            memory_pressure=0.0,
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
        if self.config.enable_semantic_cache and prompt is not None:
            cache = getattr(self, "_semantic_cache", None)
            if cache is not None:
                cache.store(prompt, result.text, model_id, cache_config, tokens_saved=0)
        return response

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
        task_weights = getattr(self, "_task_weights_by_model", {}).get(model_id)
        if task_weights is not None:
            request_extra.setdefault("task_weights", dict(task_weights))
        stream_aeg_path = self._resolve_aeg_path(model_id)
        stream_peagle = (
            self._peagle_engines.get(str(Path(stream_aeg_path).resolve()))
            if stream_aeg_path is not None
            else self.peagle_engine
        )
        if stream_peagle is not None:
            request_extra.setdefault("peagle_engine", stream_peagle)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            top_p=top_p if top_p is not None else self.config.default_top_p,
            top_k=top_k,
            stream=True,
            stop=stop,
            extra=request_extra,
        )
        stream = backend.generate_stream(request)
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

    def embed(self, model_id: str, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        backend = self._load_model(model_id)
        if hasattr(backend, "embed"):
            return backend.embed(model_id, input)
        msg = f"Backend {backend.name} does not support embeddings"
        raise AetherRuntimeError(msg)

    def rerank(self, model_id: str, query: str, documents: list[str]) -> list[dict[str, Any]]:
        """Rerank documents for a query."""
        backend = self._load_model(model_id)
        if hasattr(backend, "rerank"):
            return backend.rerank(model_id, query, documents)
        msg = f"Backend {backend.name} does not support reranking"
        raise AetherRuntimeError(msg)

    def transcribe(self, model_id: str, audio: str, language: str | None = None) -> str:
        """Transcribe an audio file."""
        backend = self._load_model(model_id)
        if hasattr(backend, "transcribe"):
            return backend.transcribe(model_id, audio, language=language)
        msg = f"Backend {backend.name} does not support transcription"
        raise AetherRuntimeError(msg)

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

    def set_task_weights(
        self,
        weights: dict[str, float] | str,
        **task_weights: float,
    ) -> dict[str, float]:
        """Set normalized task-routing weights used by this runtime session."""
        # The v4 public API accepts ``set_task_weights(model_id, legal=..., ...)``
        # while the internal API accepts a mapping.  The model identifier is
        # retained for future per-artifact routing, but weights are normalized
        # and applied to the active runtime session immediately.
        task_model_id = weights if isinstance(weights, str) else None
        if task_model_id is not None:
            weights = task_weights
        elif task_weights:
            raise ValueError("keyword task weights cannot be combined with a mapping")
        if not weights or any(not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
            raise ValueError("task weights must be a non-empty mapping of non-negative numbers")
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("at least one task weight must be greater than zero")
        normalized = {name: float(value) / total for name, value in weights.items()}
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
            unknown = sorted(set(normalized) - available)
            if unknown:
                raise ValueError(
                    f"task weights reference unknown vectors {unknown}; available vectors are {sorted(available)}"
                )
            if not hasattr(self, "_task_weights_by_model"):
                self._task_weights_by_model: dict[str, dict[str, float]] = {}
                self._task_weights_by_model: dict[str, dict[str, float]] = {}
            self._task_weights_by_model[task_model_id] = normalized
        self._task_weights = normalized
        return dict(self._task_weights)

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
                logger.error("Async compilation failed", job_id=job_id, error=str(exc))
            finally:
                self._compile_jobs[job_id]["completed_at"] = (  # type: ignore[index]
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                )

        thread = threading.Thread(target=_run_compile, daemon=True, name=f"compile-{job_id[:8]}")
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
        """Apply model merging task vectors using Pass 12 logic.

        Merges a base model with one or more task-specific delta-weight
        vectors (task_arithmetic / dare_ties / free_merging / evolutionary).

        Args:
            model_id: Base model AEG path or identifier.
            task_vectors: List of task vector configs. Each entry:
                {name: str, coefficient: float, path: str}
            method: Merge strategy: task_arithmetic | dare_ties |
                    free_merging | heterogeneous | evolutionary.
            density: Pruning density for DARE/TIES (0.0-1.0, default 1.0).

        Returns:
            Dict with merge result metadata.

        Research basis:
            Task Arithmetic (ICLR 2023), TIES-Merging (NeurIPS 2023),
            DARE-TIES (arXiv 2024), FREE-Merging (arXiv 2026).
        """
        if not task_vectors:
            raise ValueError("at least one task vector source is required")
        if not 0.0 < density <= 1.0:
            raise ValueError("density must be greater than 0 and no more than 1")
        method_aliases = {"dare_ties": "dare", "task-arithmetic": "task_arithmetic"}
        method = method_aliases.get(method, method)
        supported_methods = {"task_arithmetic", "dare", "ties", "free", "evolutionary", "soup"}
        if method not in supported_methods:
            raise ValueError(f"unsupported merge method {method!r}; choose one of {sorted(supported_methods)}")

        # Runtime merging operates on native AEG artifacts.  It deliberately
        # does not return a recorded/config-only success: a caller receives a
        # new artifact only after real source weights were read and persisted.
        base_path = self._resolve_aeg_path(model_id)
        if base_path is None:
            raise ModelNotFoundError(f"base AEG model {model_id!r} was not found", model_id=model_id)
        base = load_aeg_package(base_path)
        if not base.has_weights:
            raise AetherRuntimeError(f"base AEG {model_id!r} has no persisted weights")

        import numpy as np
        from aether.compiler.stage2_optimizer.pass12_model_merging import (
            _apply_delta,
            _compute_task_vectors,
            _get_merger,
            _load_source_weights,
        )
        from aether.quantization.formats import quantize_tensor

        store = base.weight_store()
        entries = store.load_index()
        base_weights = {
            name: tensor.reshape(-1).astype("float32").tolist()
            for name, tensor in store.dequantize_all().items()
        }
        sources: list[dict[str, list[float]]] = []
        coefficients: list[float] = []
        for item in task_vectors:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("each task vector requires a readable 'path'")
            source = _load_source_weights(item["path"], base)
            if not source:
                raise AetherRuntimeError(f"task vector source has no readable weights: {item['path']}")
            sources.append(source)
            coefficient = item.get("coefficient", 1.0 / len(task_vectors))
            if not isinstance(coefficient, (int, float)) or coefficient < 0:
                raise ValueError("task vector coefficients must be non-negative numbers")
            coefficients.append(float(coefficient))
        if sum(coefficients) <= 0:
            raise ValueError("at least one task vector coefficient must be greater than zero")

        # Keep coefficients as supplied for task arithmetic; normalize only
        # for algorithms whose semantics require a convex combination.
        if method in {"soup", "free", "evolutionary"}:
            total = sum(coefficients)
            coefficients = [value / total for value in coefficients]
        merger = _get_merger(method)
        task_delta = _compute_task_vectors(base_weights, sources)
        merged_delta = merger.merge(task_delta, coefficients, self.config)
        if not merged_delta:
            raise AetherRuntimeError("task vector sources have no overlapping tensor names")
        merged_weights = _apply_delta(base_weights, merged_delta)

        output_root = Path(base_path).with_name(Path(base_path).name + ".merged")
        if output_root.exists():
            raise AetherRuntimeError(f"merge output already exists: {output_root}; remove it explicitly first")
        tokenizer_root = Path(base_path) / "tokenizer"
        if not tokenizer_root.is_dir():
            raise AetherRuntimeError("base AEG has no packaged tokenizer; merged text generation cannot be portable")
        import shutil

        # AEG metadata may reference immutable payloads outside the graph and
        # weights trees, most importantly packaged native kernels.  Copy the
        # complete source artifact first so the merged package remains
        # self-contained; ``save()`` below then replaces the mutable manifest,
        # graph, metadata, and quantized weight payloads with their merged
        # versions.  Copying only the tokenizer would leave a valid-looking
        # manifest pointing at missing executable artifacts after reload.
        shutil.copytree(base.root, output_root)

        merged = copy.deepcopy(base)
        merged.root = output_root.resolve()
        merged._weight_store = None  # noqa: SLF001 - invalidate copied disk reader
        merged.weights = {}
        merged.manifest.model_id = f"{base.manifest.model_id or model_id}+merged"
        merged.metadata["model_merge"] = {
            "method": method,
            "density": density,
            "sources": [item["path"] for item in task_vectors],
            "coefficients": coefficients,
            "tensor_count": len(merged_delta),
        }
        # Persist the actual per-source task deltas so runtime reweighting does
        # not depend on source files remaining available after compilation.
        # AEGPackage.save() includes these files in the manifest artifact hash.
        task_delta_vectors = _compute_task_vectors(base_weights, sources)
        task_vector_descriptors: list[dict[str, Any]] = []
        import re

        task_vector_root = output_root / "merging" / "task_vectors"
        task_vector_root.mkdir(parents=True, exist_ok=True)
        task_vector_names: set[str] = set()
        for index, (item, delta) in enumerate(zip(task_vectors, task_delta_vectors)):
            requested_name = str(item.get("name") or Path(str(item["path"])).stem or f"task_{index}")
            if requested_name in task_vector_names:
                raise ValueError(f"duplicate task vector name {requested_name!r}")
            task_vector_names.add(requested_name)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", requested_name).strip("._") or f"task_{index}"
            file_name = f"{index:03d}_{safe_name}.npz"
            file_path = task_vector_root / file_name
            arrays: dict[str, Any] = {}
            tensor_descriptors: list[dict[str, Any]] = []
            for tensor_index, (tensor_name, values) in enumerate(sorted(delta.items())):
                key = f"tensor_{tensor_index:06d}"
                arrays[key] = np.asarray(values, dtype=np.float32)
                base_entry = entries.get(tensor_name)
                shape = list(base_entry.shape) if base_entry is not None else [len(values)]
                tensor_descriptors.append({"name": tensor_name, "key": key, "shape": shape})
            if not arrays:
                raise AetherRuntimeError(f"task vector {requested_name!r} has no persisted tensor deltas")
            np.savez_compressed(file_path, **arrays)
            task_vector_descriptors.append(
                {
                    "name": requested_name,
                    "path": f"merging/task_vectors/{file_name}",
                    "tensors": tensor_descriptors,
                }
            )
        merged.metadata["task_vectors"] = {
            "format": "aether_task_vectors_v1",
            "vectors": task_vector_descriptors,
        }
        for name, values in merged_weights.items():
            entry = entries.get(name)
            precision = (entry.precision if entry is not None else base.precision_map.get(name, "BF16"))
            shape = entry.shape if entry is not None else (len(values),)
            merged.weights[name] = quantize_tensor(
                np.asarray(values, dtype=np.float32).reshape(shape), precision
            )
        # ``copytree(base.root, output_root)`` above already preserved the
        # tokenizer together with every other immutable package payload.
        merged.save()
        # Reload verifies manifest, payload hashes, graph and weight index.
        verified = load_aeg_package(output_root)
        verified.verify_integrity()
        return {
            "model": model_id,
            "output_model": str(output_root),
            "method": method,
            "task_count": len(task_vectors),
            "density": density,
            "tensor_count": len(merged_delta),
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
