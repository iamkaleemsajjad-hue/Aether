"""
ONNX Runtime backend for cross-platform inference.

This backend wraps ONNX Runtime and is particularly useful for Intel OpenVINO
NPUs, Azure execution providers, and cross-platform CPU deployment.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import AetherError, BackendError


class ONNXAEGHandle(CompiledAEGHandle):
    """Common compiled artifact handle with provider inspection delegated."""

    def get_providers(self) -> list[str]:
        return self.engine.get_providers()


class ONNXBackend(Backend):
    """ONNX Runtime backend."""

    model_formats = frozenset({"onnx", "aeg"})

    def __init__(self) -> None:
        info = BackendInfo(
            name="onnx",
            version=AETHER_VERSION,
            supported_targets=[
                "cpu_generic",
                "cpu_avx2",
                "cpu_avx512",
                "cpu_neon",
                "openvino_npu",
                "openvino_gpu",
                "directml",
            ],
            capabilities=["generate", "chat", "stream", "embed", "cross_platform"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._model_options: dict[str, dict[str, Any]] = {}
        self._requested_providers: dict[str, list[Any]] = {}
        self._aeg_resources: dict[str, Any] = {}
        self._aeg_sessions: dict[str, tuple[str, str]] = {}
        self._session_locks: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401

            return True
        except (ImportError, OSError):
            return False

    def available_providers(self) -> list[str]:
        """Provider registration evidence; this does not prove device execution."""
        try:
            import onnxruntime as ort

            return list(ort.get_available_providers())
        except (ImportError, OSError):
            return []

    def available_for_model(self, model_id: str, aeg_path: str | None = None) -> bool:
        from aether.backends.model_contract import model_format

        if model_format(model_id, aeg_path) != "aeg":
            return model_format(model_id, aeg_path) == "onnx"
        try:
            import onnx  # noqa: F401

            from aether.core.aeg_format import load_aeg_package
            from aether.runtime.onnx_export import onnx_export_eligibility

            package = load_aeg_package(aeg_path or model_id)
            return bool(
                package.supports_portable_backend("onnx")
                and onnx_export_eligibility(package)["supported"]
            )
        except (ImportError, OSError, ValueError, RuntimeError, AetherError):
            return False

    def available_for_target(self, target_id: str) -> bool:
        if target_id not in self.info.supported_targets or not self.is_available():
            return False
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if target_id.startswith("cpu_"):
            return "CPUExecutionProvider" in providers
        if target_id == "directml":
            # Registration + a physical graphics adapter make this a candidate.
            # Strict session creation still gates actual load; only a profiled
            # successful workload establishes execution evidence.
            from aether.backends.hardware_inventory import probe_gpu_inventory

            return bool(
                "DmlExecutionProvider" in providers
                and (
                    probe_gpu_inventory().get("devices")
                    or any(
                        self._model_options[model_id].get("strict_directml")
                        and "DmlExecutionProvider" in session.get_providers()
                        for model_id, session in self._models.items()
                    )
                )
            )
        if "OpenVINOExecutionProvider" not in providers:
            return False
        from aether.backends.hardware_detector import detect_openvino, detect_openvino_gpu

        return bool(
            (detect_openvino() if target_id == "openvino_npu" else detect_openvino_gpu()).available
        )

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load raw ONNX or an explicitly authorized semantic AEG export.

        AEGs require their own portable ``onnx`` backend contract and supported
        semantic lowering; a target plan alone never authorizes conversion.
        Provider registration is checked here; execution attribution requires
        a real run plus explicit profiling through ``provider_execution_evidence``.
        """
        from aether.backends.model_contract import model_format

        if (
            isinstance(self._models.get(model_id), ONNXAEGHandle)
            or aeg_path is not None
            or model_format(model_id) == "aeg"
        ):
            return self._load_aeg_model(model_id, aeg_path or model_id, **kwargs)
        if "providers" in kwargs and (
            not isinstance(kwargs["providers"], (list, tuple)) or not kwargs["providers"]
        ):
            raise BackendError("providers must be a nonempty list", backend_name=self.name)
        if "allow_provider_fallback" in kwargs and not isinstance(
            kwargs["allow_provider_fallback"], bool
        ):
            raise BackendError("allow_provider_fallback must be a bool", backend_name=self.name)
        if model_id in self._models:
            if (
                "providers" in kwargs
                and list(kwargs["providers"]) != self._requested_providers[model_id]
            ):
                raise BackendError(
                    "ONNX model is already loaded with different providers; unload it before changing providers",
                    backend_name=self.name,
                )
            if "allow_provider_fallback" in kwargs and (
                kwargs["allow_provider_fallback"]
                != self._model_options[model_id]["allow_provider_fallback"]
            ):
                raise BackendError(
                    "ONNX model is already loaded with a different fallback policy; unload it first",
                    backend_name=self.name,
                )
            return self._models[model_id]
        try:
            import onnxruntime as ort
        except ImportError:
            msg = "onnxruntime is not installed. Install with: pip install aether-runtime[onnxruntime]"
            raise ImportError(msg)

        available_providers = list(ort.get_available_providers())
        providers = kwargs.get("providers", available_providers)
        if not isinstance(providers, (list, tuple)) or not providers:
            raise BackendError("providers must be a nonempty list", backend_name=self.name)
        providers = list(providers)
        if any(
            isinstance(entry, tuple) and (len(entry) != 2 or not isinstance(entry[1], dict))
            for entry in providers
        ):
            raise BackendError(
                "provider option entries must be (name, options) tuples", backend_name=self.name
            )
        names = [entry[0] if isinstance(entry, tuple) else entry for entry in providers]
        if any(not isinstance(name, str) for name in names):
            raise BackendError("provider names must be strings", backend_name=self.name)
        missing = [name for name in names if name not in available_providers]
        if missing:
            raise BackendError(
                f"requested ONNX Runtime providers {missing} are not registered; available={available_providers}",
                backend_name=self.name,
            )
        tokenizer = kwargs.get("tokenizer")
        tokenizer_path = kwargs.get("tokenizer_path")
        if tokenizer is not None and tokenizer_path is not None:
            raise ValueError("provide tokenizer or tokenizer_path, not both")
        if tokenizer_path is not None:
            tokenizer = self._load_tokenizer(tokenizer_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        directml = "DmlExecutionProvider" in names
        allow_fallback = kwargs.get("allow_provider_fallback", "providers" not in kwargs)
        strict_provider = not allow_fallback
        strict_directml = directml and strict_provider
        required_providers = [name for name in names if name != "CPUExecutionProvider"]
        if directml:
            # Required by ORT's DirectML contract, irrespective of whether a
            # caller intentionally permits fallback for unsupported operators.
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if strict_provider and required_providers:
            sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        profiling = bool(kwargs.get("enable_profiling", False))
        sess_options.enable_profiling = profiling
        if profiling:
            # ORT appends a timestamp with only second precision. Independent
            # sessions can otherwise write into the same JSON file, corrupting
            # placement evidence even when the requested prefix includes a phase.
            prefix = kwargs.get("profile_file_prefix") or "onnxruntime_profile"
            sess_options.profile_file_prefix = f"{prefix}-{uuid.uuid4().hex}"
        try:
            session = ort.InferenceSession(model_id, sess_options, providers=providers)
            configured = session.get_providers()
            missing_runtime = [name for name in names if name not in configured]
            if missing_runtime and (strict_provider or directml):
                raise RuntimeError(
                    f"requested providers {missing_runtime} fell back to another provider during session initialization"
                )
            if strict_provider:
                session.disable_fallback()
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"failed to create ONNX Runtime session for {model_id!r}: {exc}",
                backend_name=self.name,
            ) from exc
        self._models[model_id] = session
        self._session_locks[model_id] = threading.Lock()
        self._requested_providers[model_id] = providers
        self._model_options[model_id] = {
            key: kwargs[key]
            for key in (
                "input_ids_name",
                "attention_mask_name",
                "position_ids_name",
                "logits_output_name",
                "eos_token_id",
            )
            if key in kwargs
        }
        self._model_options[model_id]["profiling_enabled"] = profiling
        self._model_options[model_id]["strict_directml"] = strict_directml
        self._model_options[model_id]["allow_provider_fallback"] = allow_fallback
        if tokenizer is not None:
            self._tokenizers[model_id] = tokenizer
        return session

    def _load_aeg_model(self, model_id: str, path: str, **kwargs: Any) -> Any:
        """Materialize authenticated portable operators, keeping native AEG intact."""
        if model_id in self._models:
            if model_id not in self._aeg_sessions:
                raise BackendError(
                    "model ID already refers to a raw ONNX session; unload it first",
                    backend_name=self.name,
                )
            from aether.backends.artifact_validation import validate_cached_evaluation

            validate_cached_evaluation(
                self._models[model_id], allow_uncertified=kwargs.get("allow_uncertified", False),
                backend_name=self.name,
            )
            engine = self._models[model_id].engine
            requested_mesh = kwargs.get("execution_devices")
            if requested_mesh and list(requested_mesh) not in (
                [engine.device],
                ["cpu:0"] if engine.device == "cpu" else [],
            ):
                raise BackendError(
                    "ONNX AEG is already loaded on another execution device; unload it first",
                    backend_name=self.name,
                )
            target = kwargs.get("target_id")
            names = [
                item[0] if isinstance(item, tuple) else item
                for item in self._requested_providers[model_id]
            ]
            if target and (
                (target == "directml" and names != ["DmlExecutionProvider"])
                or (target.startswith("cpu_") and names != ["CPUExecutionProvider"])
            ):
                raise BackendError(
                    "ONNX AEG is already loaded for another target; unload it first",
                    backend_name=self.name,
                )
            for session_id in self._aeg_sessions[model_id]:
                self.load_model(
                    session_id,
                    **{
                        key: kwargs[key]
                        for key in ("providers", "allow_provider_fallback")
                        if key in kwargs
                    },
                )
            return self._models[model_id]
        resource = None
        session_ids: list[str] = []
        try:
            from aether.backends.native_cpu_backend import PackagedTokenizer
            from aether.core.aeg_format import load_aeg_package
            from aether.runtime.onnx_export import (
                ONNXAEGEngine,
                export_aeg_to_onnx,
                onnx_export_eligibility,
            )

            package = load_aeg_package(path)
            from aether.backends.artifact_validation import validate_aeg_evaluation, verify_aeg_integrity

            numerical_quality_admitted = validate_aeg_evaluation(
                package, backend_name=self.name,
                allow_uncertified=kwargs.get("allow_uncertified", False),
            )
            if not package.supports_portable_backend("onnx"):
                raise BackendError(
                    "AEG export requires a declared portable 'onnx' contract",
                    backend_name=self.name,
                )
            eligibility = onnx_export_eligibility(package)
            if not eligibility["supported"]:
                raise BackendError(
                    f"AEG export is unsupported: {eligibility['reason']}", backend_name=self.name
                )
            verify_aeg_integrity(package, backend_name=self.name)
            tokenizer = kwargs.get("tokenizer") or PackagedTokenizer(
                package.root / "tokenizer" / "tokenizer.json"
            )
            resource = tempfile.TemporaryDirectory(prefix="aether-onnx-")
            bundle = export_aeg_to_onnx(
                package, Path(resource.name) / "export", max_context=kwargs.get("max_context"),
                allow_uncertified=kwargs.get("allow_uncertified", False),
            )
            options = {
                key: kwargs[key]
                for key in (
                    "providers",
                    "allow_provider_fallback",
                    "enable_profiling",
                    "profile_file_prefix",
                )
                if key in kwargs
            }
            target = kwargs.get("target_id")
            mesh = kwargs.get("execution_devices")
            device = str(mesh[0]) if mesh else None
            if mesh and len(mesh) != 1:
                raise BackendError(
                    "ONNX AEG currently requires a single explicit execution device",
                    backend_name=self.name,
                )
            if device and device.startswith("directml:"):
                if not device[9:].isdigit():
                    raise BackendError("invalid DirectML device ordinal", backend_name=self.name)
                device_index = int(device[9:])
            else:
                device_index = int(kwargs.get("device_index", 0))
            if target == "directml" or (device and device.startswith("directml:")):
                expected_provider = "DmlExecutionProvider"
                options.setdefault("providers", [(expected_provider, {"device_id": device_index})])
            elif target in {"openvino_gpu", "openvino_npu"}:
                expected_provider = "OpenVINOExecutionProvider"
                options.setdefault(
                    "providers",
                    [
                        (
                            expected_provider,
                            {"device_type": "GPU" if target == "openvino_gpu" else "NPU"},
                        )
                    ],
                )
            elif device in {"cpu", "cpu:0"} or (target and target.startswith("cpu_")):
                expected_provider = "CPUExecutionProvider"
                options.setdefault("providers", [expected_provider])
            else:
                expected_provider = None
                options.setdefault("providers", ["CPUExecutionProvider"])
            if device and device not in {"cpu", "cpu:0"} and not device.startswith("directml:"):
                raise BackendError(
                    f"ONNX AEG has no explicit execution-device mapping for {device!r}",
                    backend_name=self.name,
                )
            configured_names = [
                item[0] if isinstance(item, tuple) else item for item in options["providers"]
            ]
            if expected_provider and configured_names != [expected_provider]:
                raise BackendError(
                    f"ONNX provider request must match selected target/device {expected_provider}",
                    backend_name=self.name,
                )
            options.setdefault("allow_provider_fallback", False)
            sessions = []
            for phase, model_path in (
                ("prefill", bundle.prefill_path),
                ("decode", bundle.decode_path),
            ):
                session_id = str(model_path)
                session_ids.append(session_id)
                phase_options = dict(options)
                if phase_options.get("profile_file_prefix"):
                    phase_options["profile_file_prefix"] = (
                        f"{phase_options['profile_file_prefix']}-{phase}"
                    )
                sessions.append(self.load_model(session_id, **phase_options))
            engine = ONNXAEGEngine(bundle, *sessions)
            if expected_provider == "DmlExecutionProvider":
                engine.device = f"directml:{device_index}"
            elif expected_provider == "CPUExecutionProvider":
                engine.device = "cpu"
            handle = ONNXAEGHandle(
                model_id=model_id,
                aeg_path=package.root,
                manifest=package.manifest.to_dict(),
                metadata=package.metadata,
                precision_map=package.get_precision_map(),
                engine=engine,
                tokenizer=tokenizer,
                numerical_quality_admitted=numerical_quality_admitted,
            )
            self._models[model_id] = handle
            self._tokenizers[model_id] = tokenizer
            self._aeg_resources[model_id] = resource
            self._aeg_sessions[model_id] = tuple(session_ids)
            self._requested_providers[model_id] = self._requested_providers[session_ids[0]]
            self._model_options[model_id] = dict(self._model_options[session_ids[0]])
            return handle
        except Exception as exc:
            for session_id in session_ids:
                self.unload_model(session_id)
            if resource is not None:
                resource.cleanup()
            if isinstance(exc, BackendError):
                raise
            raise BackendError(
                f"AEG export/session initialization failed: {exc}", backend_name=self.name
            ) from exc

    def provider_execution_evidence(self, model_id: str, required_provider: str) -> dict[str, Any]:
        """Read node attribution from a profiled real run, then stop profiling.

        Session.get_providers() only lists configured providers. This method
        requires node execution events and rejects CPU/other-provider work.
        It does not mutate hardware validation flags or imply AEG support.
        """
        if model_id in self._aeg_sessions:
            with self._models[model_id].engine._lock:
                phases = [
                    self.provider_execution_evidence(session_id, required_provider)
                    for session_id in self._aeg_sessions[model_id]
                ]
            return {
                "execution_provider": required_provider,
                "execution_scope": "aeg_onnx_prefill_cached_decode",
                "executed_node_count": sum(phase["executed_node_count"] for phase in phases),
                "phases": phases,
            }
        session = self._models.get(model_id)
        options = self._model_options.get(model_id, {})
        if session is None or not options.get("profiling_enabled"):
            raise BackendError(
                "provider execution evidence requires an explicitly profiled session",
                backend_name=self.name,
            )
        options["profiling_enabled"] = False
        with self._session_locks[model_id]:
            profile_path = Path(session.end_profiling())
        try:
            events = json.loads(profile_path.read_text(encoding="utf-8"))
            if not isinstance(events, list):
                raise ValueError("profile is not an event list")
            nodes = [
                {"name": str(event.get("name", "")), "provider": event["args"]["provider"]}
                for event in events
                if isinstance(event, dict)
                and event.get("cat") == "Node"
                and isinstance(event.get("args"), dict)
                and event["args"].get("provider")
            ]
        except (OSError, ValueError, TypeError) as exc:
            raise BackendError(
                f"cannot read ONNX execution profile: {exc}", backend_name=self.name
            ) from exc
        if not nodes:
            raise BackendError(
                "ONNX profile contains no attributed node execution; provider execution is unconfirmed",
                backend_name=self.name,
            )
        other = [node for node in nodes if node["provider"] != required_provider]
        if other:
            raise BackendError(
                f"ONNX workload executed nodes outside {required_provider}: {other}",
                backend_name=self.name,
            )
        return {
            "execution_provider": required_provider,
            "executed_node_count": len(nodes),
            "executed_nodes": nodes,
        }

    def unload_model(self, model_id: str) -> None:
        for session_id in self._aeg_sessions.pop(model_id, ()):
            self.unload_model(session_id)
        self._models.pop(model_id, None)
        self._tokenizers.pop(model_id, None)
        self._model_options.pop(model_id, None)
        self._requested_providers.pop(model_id, None)
        self._session_locks.pop(model_id, None)
        resource = self._aeg_resources.pop(model_id, None)
        if resource is not None:
            resource.cleanup()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text with an explicit tokenizer-backed ONNX decode loop.

        The ONNX model must expose ``input_ids`` plus optional attention and
        position inputs, and return logits shaped ``[batch, sequence, vocab]``
        or ``[batch, vocab]``. KV-cache inputs and arbitrary encoder/decoder
        contracts are rejected until a dedicated adapter is supplied.
        """
        completed: GenerationResult | None = None
        for event in self._decode_events(request):
            if event["type"] == "done":
                completed = event["result"]
        if completed is None:
            raise BackendError(
                "ONNX decode loop ended without a completion result",
                backend_name=self.name,
            )
        return completed

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Yield decoded token deltas from the real ONNX decode loop."""
        for event in self._decode_events(request):
            if event["type"] == "token" and event["text"]:
                yield event["text"]

    def _decode_events(self, request: GenerationRequest) -> Any:
        """Run one validated decode loop and emit token/done events."""
        session = self._models.get(request.model_id)
        if session is None:
            session = self.load_model(request.model_id)
        tokenizer = self._tokenizers.get(request.model_id)
        request_tokenizer = request.extra.get("tokenizer")
        if request_tokenizer is not None:
            tokenizer = request_tokenizer
        if tokenizer is None:
            raise BackendError(
                "ONNX Runtime model loaded, but no tokenizer/decode adapter is "
                "attached; refusing fabricated output.",
                backend_name=self.name,
            )

        iterator = None
        try:
            compiled = isinstance(session, ONNXAEGHandle)
            execution_engine = session.engine if compiled else None
            features = request.extra.get("_execution_features")
            from aether.runtime.session_contract import validate_task_weights

            unsupported = {
                key
                for key in (
                    "adapter_id", "adapter_name", "lora_adapter", "task_deltas",
                    "ttt_slots", "ttt_engine", "peagle_engine", "spec_draft_engine",
                )
                if request.extra.get(key) is not None
            }
            if validate_task_weights(request.extra.get("task_weights"), available=set()):
                unsupported.add("task_weights")
            if unsupported:
                raise BackendError(
                    f"ONNX has no execution lowering for requested controls: {sorted(unsupported)}",
                    backend_name=self.name,
                )
            if compiled:
                from aether.backends import prompt_format
                from aether.runtime.session_contract import session_execution_contract
                cache_contract = session_execution_contract(session, engine=session.engine)
                if features is not None:
                    execution_engine = features.wrap_engine(
                        request.model_id, session, execution_engine, request
                    )
                prompt = prompt_format.render_prompt(request, tokenizer)
                encoded = prompt_format.encode_prompt(
                    prompt, request, tokenizer, return_tensors="np"
                )
                prompt_ids = np.asarray(encoded["input_ids"]).reshape(-1).tolist()
            else:
                prompt = self._build_prompt(request, tokenizer)
                prompt_ids = self._encode(tokenizer, prompt)
            if not prompt_ids:
                raise ValueError("tokenizer produced no prompt tokens")
            max_tokens = int(request.max_tokens)
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            options = dict(self._model_options.get(request.model_id, {}))
            options.update(
                {
                    key: request.extra[key]
                    for key in (
                        "input_ids_name",
                        "attention_mask_name",
                        "position_ids_name",
                        "logits_output_name",
                        "eos_token_id",
                    )
                    if key in request.extra
                }
            )
            if not compiled:
                input_specs = list(session.get_inputs())
                output_specs = list(session.get_outputs())
                input_ids_name = self._resolve_input_name(
                    input_specs, options.get("input_ids_name"), ("input_ids", "input_ids:0", "ids")
                )
                self._reject_unsupported_inputs(input_specs, input_ids_name, options)
                output_name = self._resolve_output_name(
                    output_specs, options.get("logits_output_name")
                )
            eos_token_id = options.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
            from aether.runtime.stopping import stop_token_set

            eos_ids = stop_token_set(eos_token_id)
            if compiled and "eos_token_id" not in options:
                eos_ids |= stop_token_set(getattr(tokenizer, "eos_token_ids", None))
            seed = request.extra.get("seed")
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
            generated: list[int] = []
            emitted_text = ""
            finish_reason = "length"
            start = time.perf_counter()
            first_token_time = None
            cache = None
            cache_holder = [None]
            session_id = request.extra.get("aether_kv_session_id")
            step_prompt = prompt_ids
            if compiled and isinstance(session_id, str) and session_id:
                existing = session.get_session_cache(session_id, contract=cache_contract)
                if existing is not None:
                    previous_ids, previous_cache = existing
                    prefix = np.asarray(previous_ids, dtype=np.int64).reshape(-1)
                    if len(prefix) <= len(prompt_ids) and np.array_equal(
                        prefix, prompt_ids[: len(prefix)]
                    ):
                        cache = session.cache_for_engine(previous_cache, execution_engine)
                        step_prompt = prompt_ids[len(prefix) :]
                    else:
                        session.clear_session_cache(session_id)
            if compiled:
                iterator = execution_engine.generate_iter(
                    np.asarray(step_prompt, dtype=np.int64),
                    max_tokens=max_tokens,
                    cache=cache,
                    cache_callback=lambda value: cache_holder.__setitem__(0, value),
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    eos_token_id=eos_ids,
                    seed=seed,
                    grammar_session=request.extra.get("grammar_session"),
                    **({"cached_prefix_ids": prompt_ids[:cache.length]} if cache is not None
                       and request.extra.get("_explicit_draft_engine") is not None else {}),
                )

            for _ in range(max_tokens):
                try:
                    if compiled:
                        try:
                            next_id = next(iterator)
                        except StopIteration:
                            finish_reason = "stop"
                            break
                    else:
                        token_array = np.asarray(
                            [prompt_ids + generated],
                            dtype=self._integer_dtype(input_specs, input_ids_name),
                        )
                        feeds = {input_ids_name: token_array}
                        self._add_optional_inputs(feeds, input_specs, token_array.shape[1], options)
                        with self._session_locks[request.model_id]:
                            raw_outputs = session.run([output_name], feeds)
                        logits = self._extract_logits(raw_outputs[0], output_name)
                except Exception as exc:  # noqa: BLE001
                    raise BackendError(
                        f"ONNX autoregressive execution failed: {exc}",
                        backend_name=self.name,
                    ) from exc
                if not compiled:
                    next_id = self._sample(logits, request, rng=rng)
                generated.append(next_id)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                decoded = self._decode(tokenizer, generated)
                cutoff = min(
                    (
                        decoded.find(stop)
                        for stop in (request.stop or [])
                        if stop and stop in decoded
                    ),
                    default=-1,
                )
                if cutoff >= 0:
                    visible = decoded[:cutoff]
                    delta = (
                        visible[len(emitted_text) :]
                        if visible.startswith(emitted_text)
                        else visible
                    )
                    emitted_text = visible
                    if delta:
                        yield {"type": "token", "text": delta, "token_id": next_id}
                    finish_reason = "stop"
                    break
                delta = (
                    decoded[len(emitted_text) :] if decoded.startswith(emitted_text) else decoded
                )
                emitted_text = decoded
                if delta:
                    yield {"type": "token", "text": delta, "token_id": next_id}
                if next_id in eos_ids:
                    finish_reason = "stop"
                    break

            if iterator is not None and finish_reason == "length":
                try:
                    next(iterator)
                except StopIteration:
                    pass
                else:
                    raise BackendError(
                        "compiled ONNX generation exceeded max_tokens", backend_name=self.name
                    )
            text = self._decode(tokenizer, generated)
            if request.stop:
                for stop in request.stop:
                    if stop in text:
                        text = text.split(stop, 1)[0]
            elapsed = time.perf_counter() - start
            if iterator is not None:
                iterator.close()
            if (
                compiled
                and isinstance(session_id, str)
                and session_id
                and cache_holder[0] is not None
            ):
                session.save_session_cache(
                    session_id, np.asarray(prompt_ids + generated, dtype=np.int64), cache_holder[0],
                    contract=cache_contract, target_engine=session.engine,
                )
            yield {
                "type": "done",
                "result": GenerationResult(
                    text=text,
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(generated),
                    finish_reason=finish_reason,
                    backend_name=self.name,
                    metrics={
                        "ttft_ms": ((first_token_time or time.perf_counter()) - start) * 1000.0,
                        "throughput_tps": len(generated) / max(elapsed, 1e-9),
                        "device": "onnxruntime",
                        "providers": list(session.get_providers()),
                        "providers_are_execution_evidence": False,
                        "execution_contract": "aether_onnx_decoder_v1" if compiled else "raw_onnx",
                        "cached_decode": compiled,
                        **(
                            execution_engine.generation_stats()
                            if compiled
                            and callable(getattr(execution_engine, "generation_stats", None))
                            else {}
                        ),
                    },
                ),
            }
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"ONNX autoregressive adapter validation failed: {exc}",
                backend_name=self.name,
            ) from exc
        finally:
            if iterator is not None:
                iterator.close()

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        """Load a local tokenizer only; remote model code is never executed."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as transformers_exc:
            try:
                from tokenizers import Tokenizer

                return Tokenizer.from_file(tokenizer_path)
            except Exception as tokenizers_exc:
                raise ValueError(
                    f"unable to load local tokenizer {tokenizer_path!r}: "
                    f"transformers={transformers_exc}; tokenizers={tokenizers_exc}"
                ) from tokenizers_exc

    @staticmethod
    def _build_prompt(request: GenerationRequest, tokenizer: Any) -> str:
        if request.messages:
            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                return str(
                    apply_template(request.messages, tokenize=False, add_generation_prompt=True)
                )
            return (
                "\n".join(
                    f"<{message.get('role', 'user')}>\n{message.get('content', '')}"
                    for message in request.messages
                )
                + "\n<assistant>\n"
            )
        return request.prompt or ""

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        ids = getattr(encoded, "ids", encoded)
        return [int(value) for value in ids]

    @staticmethod
    def _decode(tokenizer: Any, token_ids: list[int]) -> str:
        try:
            return str(tokenizer.decode(token_ids, skip_special_tokens=True))
        except TypeError:
            return str(tokenizer.decode(token_ids))

    @staticmethod
    def _resolve_input_name(specs: list[Any], requested: Any, candidates: tuple[str, ...]) -> str:
        names = [str(spec.name) for spec in specs]
        if requested is not None:
            if str(requested) not in names:
                raise ValueError(
                    f"configured ONNX input {requested!r} is not present; inputs={names}"
                )
            return str(requested)
        for candidate in candidates:
            if candidate in names:
                return candidate
        raise ValueError(f"ONNX session has no input_ids input; inputs={names}")

    @staticmethod
    def _resolve_output_name(specs: list[Any], requested: Any) -> str:
        names = [str(spec.name) for spec in specs]
        if requested is not None:
            if str(requested) not in names:
                raise ValueError(
                    f"configured ONNX output {requested!r} is not present; outputs={names}"
                )
            return str(requested)
        for name in names:
            if "logit" in name.lower():
                return name
        if len(names) == 1:
            return names[0]
        raise ValueError(f"cannot identify logits output; outputs={names}")

    @staticmethod
    def _reject_unsupported_inputs(
        specs: list[Any], input_ids_name: str, options: dict[str, Any]
    ) -> None:
        recognized = {input_ids_name}
        for key in ("attention_mask_name", "position_ids_name"):
            if options.get(key):
                recognized.add(str(options[key]))
        for spec in specs:
            name = str(spec.name)
            lowered = name.lower()
            if name in recognized or lowered in {
                "attention_mask",
                "attention_mask:0",
                "position_ids",
                "position_ids:0",
            }:
                continue
            raise ValueError(
                f"ONNX session input {name!r} requires a dedicated adapter; "
                "KV-cache/encoder-decoder contracts are not implicitly guessed"
            )

    @staticmethod
    def _integer_dtype(specs: list[Any], input_ids_name: str) -> Any:
        spec = next(item for item in specs if str(item.name) == input_ids_name)
        type_name = str(getattr(spec, "type", ""))
        if "int32" in type_name:
            return np.int32
        if "int64" in type_name:
            return np.int64
        raise ValueError(f"input_ids must use int32 or int64, got {type_name!r}")

    @staticmethod
    def _add_optional_inputs(
        feeds: dict[str, np.ndarray],
        specs: list[Any],
        sequence_length: int,
        options: dict[str, Any],
    ) -> None:
        names = {str(spec.name): spec for spec in specs}
        attention_name = options.get("attention_mask_name")
        if attention_name is None:
            attention_name = next(
                (name for name in names if name.lower() in {"attention_mask", "attention_mask:0"}),
                None,
            )
        if attention_name is not None:
            spec = names.get(str(attention_name))
            if spec is None:
                raise ValueError(
                    f"configured attention mask input {attention_name!r} is not present"
                )
            dtype = np.int32 if "int32" in str(getattr(spec, "type", "")) else np.int64
            feeds[str(attention_name)] = np.ones((1, sequence_length), dtype=dtype)

        position_name = options.get("position_ids_name")
        if position_name is None:
            position_name = next(
                (name for name in names if name.lower() in {"position_ids", "position_ids:0"}),
                None,
            )
        if position_name is not None:
            if str(position_name) not in names:
                raise ValueError(f"configured position ids input {position_name!r} is not present")
            spec = names[str(position_name)]
            dtype = np.int32 if "int32" in str(getattr(spec, "type", "")) else np.int64
            feeds[str(position_name)] = np.arange(sequence_length, dtype=dtype)[None, :]

    @staticmethod
    def _extract_logits(output: Any, output_name: str) -> np.ndarray:
        logits = np.asarray(output)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0, -1]
        elif logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits[0]
        else:
            raise ValueError(
                f"ONNX logits output {output_name!r} must be [1, sequence, vocab] "
                f"or [1, vocab], got {tuple(logits.shape)}"
            )
        if logits.ndim != 1 or logits.size == 0 or not np.isfinite(logits).all():
            raise ValueError(f"ONNX logits output {output_name!r} is empty or non-finite")
        return logits.astype(np.float32, copy=False)

    @staticmethod
    def _sample(logits: np.ndarray, request: GenerationRequest, *, rng: Any | None = None) -> int:
        if request.temperature <= 0.0:
            return int(np.argmax(logits))
        scaled = logits / max(float(request.temperature), 1e-6)
        if request.top_k > 0 and request.top_k < scaled.size:
            keep = np.argpartition(scaled, -request.top_k)[-request.top_k :]
            filtered = np.full_like(scaled, -np.inf)
            filtered[keep] = scaled[keep]
            scaled = filtered
        shifted = scaled - np.nanmax(scaled)
        probs = np.exp(shifted)
        probs[~np.isfinite(probs)] = 0.0
        if 0.0 < request.top_p < 1.0:
            order = np.argsort(-probs)
            cumulative = np.cumsum(probs[order]) / max(float(probs.sum()), 1e-12)
            remove = cumulative > request.top_p
            if remove.any():
                remove[0] = False
                probs[order[remove]] = 0.0
        total = float(probs.sum())
        if total <= 0.0:
            return int(np.argmax(logits))
        probs /= total
        seed = request.extra.get("seed")
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
        return int(rng.choice(logits.size, p=probs))

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"ONNXBackend(models={len(self._models)})"


def validate_directml_execution(device_index: int = 0) -> dict[str, Any]:
    """Execute a small offline ONNX MatMul strictly on the selected DirectML adapter.

    This verifies ONNX/DirectML inference only. It proves neither a general AEG
    export pipeline nor any untested precision or production workload. Missing
    dependencies, wrong output, missing node attribution, and CPU fallback are
    failures; provider registration alone never satisfies this probe.
    """
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("DirectML device_index must be a nonnegative integer")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise BackendError(
            "DirectML validation requires onnxruntime-directml", backend_name="onnx"
        ) from exc
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise BackendError(
            "DmlExecutionProvider is not registered; install onnxruntime-directml",
            backend_name="onnx",
        )
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise BackendError(
            "DirectML's offline validation probe requires the optional onnx package",
            backend_name="onnx",
        ) from exc

    inputs = np.asarray([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=np.float32)
    weights = np.asarray([[2.0, 0.25], [-1.0, 3.0], [0.5, -2.0]], dtype=np.float32)
    expected = inputs @ weights
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="directml_matmul")],
        "aether_directml_validation",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, list(inputs.shape))],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, list(expected.shape))],
        initializer=[numpy_helper.from_array(weights, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    with tempfile.TemporaryDirectory(prefix="aether-directml-validation-") as temporary:
        root = Path(temporary)
        model_path = root / "matmul.onnx"
        onnx.save(model, str(model_path))
        backend = ONNXBackend()
        model_id = str(model_path)
        try:
            session = backend.load_model(
                model_id,
                providers=[("DmlExecutionProvider", {"device_id": str(device_index)})],
                allow_provider_fallback=False,
                enable_profiling=True,
                profile_file_prefix=str(root / "provider-profile"),
            )
            started = time.perf_counter()
            observed = np.asarray(session.run(["output"], {"input": inputs})[0])
            elapsed = time.perf_counter() - started
            if observed.shape != expected.shape or not np.isfinite(observed).all():
                raise BackendError(
                    "DirectML workload returned an invalid output tensor", backend_name="onnx"
                )
            if not np.allclose(observed, expected, rtol=1e-5, atol=1e-5):
                raise BackendError(
                    "DirectML workload output does not match the NumPy reference",
                    backend_name="onnx",
                )
            evidence = backend.provider_execution_evidence(model_id, "DmlExecutionProvider")
            return {
                **evidence,
                "execution_scope": "onnxruntime_directml_inference",
                "device_index": device_index,
                "precision": "fp32",
                "workload": "offline_onnx_matmul_2x3_by_3x2",
                "duration_seconds": elapsed,
                "maximum_absolute_error": float(np.max(np.abs(observed - expected))),
                "observed_output": observed.tolist(),
                "reference_output": expected.tolist(),
            }
        finally:
            # A failing run can retain the session in its traceback. Finalize
            # its optional profile before removing Windows temporary files,
            # without replacing the original execution failure with cleanup.
            loaded = backend._models.get(model_id)
            if loaded is not None and backend._model_options[model_id].get("profiling_enabled"):
                try:
                    loaded.end_profiling()
                except Exception:
                    pass
            backend.unload_model(model_id)
