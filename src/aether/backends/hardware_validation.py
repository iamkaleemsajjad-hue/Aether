"""Explicit environment and real AEG execution checks, separate from detection."""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from aether.backends.capabilities import HardwareCapabilities, ValidationResult


def _compatible(requested: str, cap: HardwareCapabilities) -> bool:
    if requested == cap.target_id:
        return True
    if requested in {"cpu", "cuda", "rocm", "metal"}:
        return cap.target_id.startswith(requested) or (requested == "rocm" and cap.target_id.startswith("amd_mi"))
    if requested.startswith("cpu_"):
        features = cap.extra.get("cpu_features", {})
        required = {"cpu_generic": None, "cpu_avx2": "avx2", "cpu_avx512": "avx512f", "cpu_neon": "neon"}
        if requested not in required:
            return False
        feature = required[requested]
        return feature is None or bool(features.get(feature) or (feature == "neon" and features.get("asimd")))
    if requested.startswith("cuda_sm"):
        try:
            # Baseline cubin compatibility: same major, device minor >= target.
            req = int(requested.removeprefix("cuda_sm"))
            major, minor = map(int, str(cap.extra["compute_capability"]).split("."))
            return req // 10 == major and req % 10 <= minor
        except (ValueError, KeyError, TypeError):
            return False
    return False


def validate_backend_environment(
    target_id: str, *, execute: bool = False, backend_name: str | None = None,
    device_index: int = 0,
) -> ValidationResult:
    # Import lazily to retain the public detector API and allow its probes to be
    # substituted in deterministic tests. Substituted probes are never execution evidence.
    from aether.backends import hardware_detector as detector
    result = ValidationResult(backend_name=target_id, available=False)
    if device_index < 0:
        result.checks_failed.append("device_index must be nonnegative")
        return result
    if target_id.startswith("cpu"):
        caps = [detector.detect_cpu()]
    elif target_id.startswith("cuda"):
        caps = detector.detect_cuda_devices()
    elif target_id.startswith(("rocm", "amd_mi")):
        caps = detector.detect_rocm_devices()
    elif target_id.startswith("metal"):
        caps = detector.detect_metal_devices()
    elif target_id == "directml":
        caps = [detector.detect_directml()]
    elif target_id == "openvino_npu":
        caps = [cap for cap in detector.detect_openvino_devices() if cap.target_id == target_id]
    elif target_id == "openvino_gpu":
        caps = [cap for cap in detector.detect_openvino_devices() if cap.target_id == target_id]
    elif target_id == "qualcomm_qnn":
        caps = [detector.detect_qualcomm_npu()]
    elif target_id == "apple_ane":
        caps = [detector.detect_apple_ane()]
    else:
        result.checks_failed.append(f"Unknown or unimplemented execution target {target_id!r}")
        return result
    if target_id.startswith(("rocm", "amd_mi")) and backend_name in {None, "pytorch", "torch"}:
        # The HIP C device-property structure changes across ABI versions.
        # Explicit validation may inspect PyTorch's correctly bound properties;
        # discovery remains framework-free and never upgrades execution status.
        result.details["driver_devices"] = [cap.to_dict() for cap in caps]
        try:
            import torch
            framework_caps = detector.inspect_torch_hip_devices(torch)
            if framework_caps:
                caps = framework_caps
        except (ImportError, OSError, RuntimeError) as exc:
            result.details["framework_probe_error"] = str(exc)
    result.details["devices"] = [cap.to_dict() for cap in caps]
    if target_id == "directml" and execute:
        if backend_name not in {None, "onnx", "onnxruntime"}:
            result.checks_failed.append("DirectML validation requires the ONNX Runtime adapter")
            return result
        result.details["execution_backend"] = "onnx"
        try:
            from aether.backends.onnx_backend import validate_directml_execution
            workload = validate_directml_execution(device_index)
            result.available = True
            result.backend_available = True
            result.execution_tested = True
            result.execution_scope = workload["execution_scope"]
            result.details["workload"] = workload
            record = next((record for record in caps[0].extra.get("physical_adapters", [])
                           if record.get("directml_device_index") == device_index), None)
            selected = caps[0].to_dict()
            selected.update(available=True, runtime_available=True, hardware_present=True,
                            backend_available=True, execution_tested=True, supports_fp32=True, unavailable_reason=None)
            selected["extra"].update(execution_device=f"directml:{device_index}", device_index=device_index,
                                     device_index_scope="DXGI adapter index verified by DirectML execution",
                                     execution_scope=workload["execution_scope"], precision_scope="validated_runtime_formats")
            if record:
                selected.update(device=record["device"], vendor=record["vendor"],
                                driver_version=record.get("driver_version", "unknown"),
                                memory_bytes=record.get("memory_bytes", 0), memory_kind=record.get("memory_kind"),
                                memory_source=record.get("memory_source"), memory_bytes_known="memory_bytes" in record)
                selected["extra"].update(record)
            else:
                selected["extra"]["physical_identity_unconfirmed"] = True
            selected["precision_semantics"] = "FP32 ONNX execution verified; other formats and AEG export unconfirmed"
            selected["precision_summary"]["fp32"] = True
            result.details["selected_device"] = selected
            result.checks_passed.append("Real DirectML FP32 ONNX inference matched NumPy and was attributed to DmlExecutionProvider")
            result.warnings.append("This validates ONNX DirectML inference only. AEG DirectML export, other precisions, and production workloads are not validated.")
        except Exception as exc:
            result.backend_available = False
            result.checks_failed.append(f"DirectML execution validation failed: {exc}")
        return result
    def ordinal(cap: HardwareCapabilities) -> int | None:
        value = cap.extra.get("device_index", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    matches = [cap for cap in caps if cap.available and _compatible(target_id, cap)
               and ordinal(cap) == device_index]
    if not matches:
        result.checks_failed.append(
            f"No usable device at index {device_index} compatible with target {target_id!r}: "
            + "; ".join(cap.unavailable_reason or f"detected {cap.target_id}" for cap in caps)
        )
        return result
    cap = matches[0]
    result.available = True
    result.checks_passed.append(f"Hardware and runtime detected: {cap.device} ({cap.target_id})")
    result.details["selected_device"] = cap.to_dict()
    from aether.core.constants import SUPPORTED_TARGET_IDS
    if not cap.implemented or cap.target_id not in SUPPORTED_TARGET_IDS:
        result.backend_available = False
        result.checks_failed.append(f"Detected device target {cap.target_id!r} has no compatible implemented Aether execution target")
        return result
    selected_backend = backend_name or ("aether_cpu" if target_id.startswith("cpu") else "pytorch")
    result.details["execution_backend"] = selected_backend
    if target_id.startswith("cpu"):
        try:
            import numpy as np
            result.checks_passed.append(f"NumPy {np.__version__} available")
            from aether.kernels.native_cpu import get_native_kernels
            kernels = get_native_kernels()
            native = kernels.ensure_compiled()
            result.details["native_cpu"] = {
                "available": bool(native), "build_error": kernels.build_error,
                "compiler_optional": True, "fallback": None if native else "NumPy FP32 reference",
                "build_identity": kernels.build_identity(),
            }
            if native:
                result.checks_passed.append(f"Native CPU kernels compiled: {kernels.available_kernels()}")
            else:
                result.warnings.append(f"Native CPU kernel compilation failed: {kernels.build_error}; NumPy fallback is available. A C++ compiler is optional.")
        except Exception as exc:
            result.checks_failed.append(f"CPU runtime check failed: {exc}")
            return result
    try:
        from aether.backends.registry import BackendRegistry
        backend = BackendRegistry().get_backend(selected_backend)
        if backend is None:
            raise RuntimeError(f"Backend {selected_backend!r} could not be loaded")
        if not backend.available_for_target(cap.target_id):
            reason = getattr(backend, "unavailable_reason", None)
            raise RuntimeError(reason or f"Backend {selected_backend!r} is not ready for {cap.target_id}; install its matching runtime dependencies")
        # Validate actual framework/driver agreement separately from detector evidence.
        if selected_backend in {"pytorch", "torch"} and not target_id.startswith("cpu"):
            import torch
            if target_id.startswith(("cuda", "rocm", "amd_mi")):
                if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
                    raise RuntimeError("PyTorch cannot initialize the requested accelerator device")
                hip = bool(getattr(torch.version, "hip", None))
                if hip != target_id.startswith(("rocm", "amd_mi")):
                    raise RuntimeError("Installed PyTorch uses a different GPU runtime family")
                props = torch.cuda.get_device_properties(device_index)
                if target_id.startswith("cuda") and f"{props.major}.{props.minor}" != cap.extra.get("compute_capability"):
                    raise RuntimeError("CUDA Driver API and PyTorch disagree on the selected device compute capability")
                from aether.backends.hardware_inventory import device_identity_keys
                framework_identity = {key: str(getattr(props, key)) for key in ("uuid", "pci_bus_id")
                                      if getattr(props, key, None) is not None}
                detected_keys = device_identity_keys(cap.extra)
                framework_keys = device_identity_keys(framework_identity)
                for namespace in {key.split(":", 1)[0] for key in detected_keys}:
                    detected = {key for key in detected_keys if key.startswith(namespace + ":")}
                    confirmed = {key for key in framework_keys if key.startswith(namespace + ":")}
                    if confirmed and not detected.intersection(confirmed):
                        raise RuntimeError(f"Driver and PyTorch disagree on the selected device {namespace} identity")
                result.details["framework_device"] = {"name": props.name, "index": device_index,
                    "memory_bytes": int(props.total_memory), "runtime": "hip" if hip else "cuda",
                    "runtime_version": getattr(torch.version, "hip" if hip else "cuda", None),
                    **framework_identity}
            elif not torch.backends.mps.is_available():
                raise RuntimeError("PyTorch MPS cannot initialize this Metal device")
            elif cap.extra.get("execution_device") != "mps":
                raise RuntimeError("PyTorch MPS cannot select this non-default Metal adapter")
        result.backend_available = True
        result.checks_passed.append(f"Execution backend ready: {backend.name}")
        if execute:
            result.details["workload"] = _execute_aeg_probe(backend, cap, device_index)
            result.execution_tested = True
            result.execution_scope = "aether_aeg_compile_load_prefill_decode"
            result.checks_passed.append("Real AEG prefill and cached decode matched the independent NumPy reference")
        else:
            result.warnings.append("Execution was not tested. Use --execute to compile, load, and run an offline AEG workload.")
    except Exception as exc:
        result.backend_available = False if not result.backend_available else result.backend_available
        result.checks_failed.append(f"Backend validation failed: {exc}")
    return result


def _write_probe_checkpoint(path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 32,
        "rms_norm_eps": 1e-5, "rope_theta": 10000., "torch_dtype": "float32",
        "tie_word_embeddings": False,
    }), encoding="utf-8")
    rng = np.random.default_rng(718)
    matrices = {"model.embed_tokens.weight": (32, 16), "lm_head.weight": (32, 16)}
    prefix = "model.layers.0."
    for name, shape in {"self_attn.q_proj": (16, 16), "self_attn.k_proj": (8, 16),
                        "self_attn.v_proj": (8, 16), "self_attn.o_proj": (16, 16),
                        "mlp.gate_proj": (32, 16), "mlp.up_proj": (32, 16), "mlp.down_proj": (16, 32)}.items():
        matrices[prefix + name + ".weight"] = shape
    tensors = {name: (rng.normal(size=shape) * .1).astype(np.float32) for name, shape in matrices.items()}
    for name in ("model.norm.weight", prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"):
        tensors[name] = np.ones(16, dtype=np.float32)
    save_file(tensors, str(path / "model.safetensors"))
    vocab = {"<unk>": 0, "hello": 1, "world": 2, **{f"token{i}": i + 3 for i in range(29)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>", "model_max_length": 128}), encoding="utf-8")
    (path / "special_tokens_map.json").write_text('{"unk_token": "<unk>"}', encoding="utf-8")


def _execute_aeg_probe(backend: Any, cap: HardwareCapabilities, device_index: int) -> dict[str, Any]:
    """Offline real model, authenticated AEG, selected backend, numerical check.

    This is a small functional test, not model quality, performance or production
    certification. Random checkpoint weights are deliberate test inputs, never
    a substitute for loading a user's model or simulated hardware execution.
    """
    import numpy as np
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig
    from aether.runtime.aeg_loader import load_engine_from_path
    started = time.perf_counter()
    device = "cpu" if cap.target_id.startswith("cpu") else "mps" if cap.target_id.startswith("metal") else f"cuda:{device_index}"
    model_id = "aether-hardware-validation"
    with tempfile.TemporaryDirectory(prefix="aether-hardware-") as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        _write_probe_checkpoint(checkpoint)
        config = CompilerConfig(
            quality_mode="diagnostic",
            targets=[cap.target_id],
            precision="FP32",
            optimization_level=0,
            cache_dir=str(root / "cache"),
        )
        for item in fields(config):
            if item.name.startswith("enable_"):
                setattr(config, item.name, item.name == "enable_precision_assignment")
        artifact = root / "probe.aeg"
        Compiler(config).compile(str(checkpoint), output_path=artifact)
        reference = load_engine_from_path(
            artifact, prepare_native_kernels=False, allow_uncertified=True
        )
        try:
            handle = backend.load_model(
                model_id, aeg_path=str(artifact), execution_devices=[device],
                offline=True, allow_uncertified=True,
            )
            engine = handle.engine
            if engine is None:
                raise RuntimeError("Backend did not load an executable AEG engine")
            native_device_id = getattr(engine, "device_id", None)
            actual_device = str(getattr(engine, "device", f"cuda:{native_device_id}" if native_device_id is not None else "cpu"))
            if actual_device == "cuda":
                actual_device = "cuda:0"
            if actual_device != device:
                raise RuntimeError(f"Backend ran on {actual_device}, requested {device}; fallback is not validation")
            prompt = np.array([1, 2, 3], dtype=np.int64)
            expected, ref_cache = reference.forward(prompt)
            measured, cache = engine.forward(prompt)
            expected_decode, ref_cache = reference.forward(np.array([4]), ref_cache)
            measured_decode, cache = engine.forward(np.array([4]), cache)
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize(device)
            elif device == "mps":
                import torch
                torch.mps.synchronize()
            errors = []
            # Bound parity by the actual accumulation dtype. The probe uses
            # the backend's normal fast path rather than forcing every GPU to
            # FP32, and reports those bounds with the measured errors.
            compute_dtype = str(getattr(engine, "compute_dtype", "float32"))
            dtype_name = compute_dtype.rsplit(".", 1)[-1]
            bounds = {"float32": (2e-4, 2e-5), "float16": (5e-3, 2e-3), "bfloat16": (3e-2, 1e-2)}
            if dtype_name not in bounds:
                raise RuntimeError(f"No numerical validation bounds for backend dtype {compute_dtype}")
            rtol, atol = bounds[dtype_name]
            for expected_logits, actual_logits in ((expected, measured), (expected_decode, measured_decode)):
                if actual_logits.shape != expected_logits.shape or not np.isfinite(actual_logits).all():
                    raise RuntimeError("AEG produced invalid shape or non-finite logits")
                np.testing.assert_allclose(actual_logits, expected_logits, rtol=rtol, atol=atol)
                errors.append(float(np.max(np.abs(actual_logits - expected_logits))))
            if cache.length != 4:
                raise RuntimeError(f"Cached decode did not extend KV state: {cache.length}")
            tensor_devices: dict[str, str] = {}
            if hasattr(engine, "torch"):
                tensors = {name: getattr(engine, name, None) for name in ("embedding", "lm_head", "final_norm")}
                for index, layer in enumerate(getattr(engine, "layers", ())):
                    if isinstance(layer, dict):
                        tensors.update({f"layer_{index}.{name}": tensor for name, tensor in layer.items()})
                for kind in ("keys", "values"):
                    tensors.update({f"cache.{kind}.{index}": value for index, value in enumerate(getattr(cache, kind, ()))})
                tensor_devices = {name: str(value.device) for name, value in tensors.items() if engine.torch.is_tensor(value)}
                if not tensor_devices or any(placed != device for placed in tensor_devices.values()):
                    raise RuntimeError(f"AEG model or KV tensors were placed outside the requested device: {tensor_devices}")
            kernels = getattr(engine, "kernels", None)
            return {"model": "offline one-layer Llama functional fixture", "backend": backend.name,
                    "device": actual_device, "prefill_tokens": 3, "decode_tokens": 1,
                    "prefill_logits_shape": list(measured.shape), "decode_logits_shape": list(measured_decode.shape),
                    "kv_cache_length": cache.length, "storage_precision": "FP32",
                    "compute_dtype": compute_dtype,
                    "compilation_quality_mode": "diagnostic",
                    "original_hf_quality_validated": False,
                    "native_cpu": bool(kernels.is_native) if kernels is not None else None,
                    "tensor_devices": tensor_devices,
                    "reference": "independent NumPy FP32 CPU engine", "max_absolute_errors": errors,
                    "rtol": rtol, "atol": atol, "elapsed_seconds": time.perf_counter() - started}
        finally:
            unload = getattr(backend, "unload_model", None)
            if unload is not None:
                unload(model_id)
            else:
                unload_all = getattr(backend, "unload", None)
                if unload_all is not None:
                    unload_all()
