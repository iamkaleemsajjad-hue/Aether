"""Portable identities for cache state produced by validated execution weights.

Token prefixes alone cannot identify KV or cached logits: static adapters,
task vectors and arithmetic precision also affect their values. No tensor
bytes are copied or rehashed while comparing request contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping
from numbers import Real
from typing import Any

SCHEMA = "aether_session_execution_v1"


def validate_task_weights(
    weights: Any, *, available: Collection[str] | None = None, normalize: bool = False
) -> dict[str, float]:
    """Validate every coordinate, then return sorted effective coefficients.

    None and empty mappings select the base weights. Zero coordinates are
    checked before omission. Normalization avoids overflowing a finite sum.
    """
    if weights is None:
        return {}
    if not isinstance(weights, Mapping):
        raise ValueError("task weights must be a mapping")
    values: dict[str, float] = {}
    for name, coefficient in weights.items():
        if not isinstance(name, str) or not name or len(name) > 256 or "\0" in name:
            raise ValueError("task weight names must be nonempty bounded strings")
        if available is not None and name not in available:
            raise ValueError(f"task weights reference unknown vector {name!r}")
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
            raise ValueError(f"task weight for {name!r} must be a finite non-negative number")
        try:
            value = float(coefficient)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"task weight for {name!r} must be finite") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"task weight for {name!r} must be a finite non-negative number")
        if value:
            values[name] = value
    if normalize and values:
        largest = max(values.values())
        scaled = {name: value / largest for name, value in values.items()}
        total = math.fsum(scaled.values())
        values = {name: value / total for name, value in scaled.items() if value}
    return dict(sorted(values.items()))


def session_execution_contract(
    handle: Any, *, adapter_id: Any = None, task_weights: Any = None, engine: Any = None
) -> dict[str, Any]:
    """Describe validated static execution inputs, independently of object IDs."""
    manifest = getattr(handle, "manifest", None)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("cache contracts require a compiled artifact manifest")
    if adapter_id is not None and (
        not isinstance(adapter_id, str) or not adapter_id or adapter_id not in handle.lora_adapters
    ):
        raise ValueError("cache adapter identity is not present in the compiled artifact")
    metadata = getattr(handle, "metadata", {})
    descriptor = metadata.get("task_vectors", {})
    vectors = descriptor.get("vectors", []) if isinstance(descriptor, dict) else []
    if not isinstance(vectors, list):
        raise ValueError("compiled task-vector descriptors must be a list")
    available = {item.get("name") for item in vectors if isinstance(item, dict)}
    coefficients = validate_task_weights(task_weights, available=available)
    epoch = metadata.get("_precision_epoch", 0)
    if type(epoch) is not int or epoch < 0:
        raise ValueError("cache precision epoch must be a non-negative integer")
    execution = engine if engine is not None else handle.engine
    dtype = getattr(execution, "compute_dtype", None)
    if dtype is not None:
        dtype = str(dtype).removeprefix("torch.")
    elif getattr(execution, "weights", None) is not None:
        # Existing NumPy engines explicitly perform arithmetic in FP32 even
        # when projection storage has been compacted to FP16.
        dtype = "float32"
    elif (
        getattr(getattr(execution, "bundle", None), "metadata", {}).get("arithmetic_precision")
        == "fp32"
    ):
        dtype = "float32"
    architecture = manifest.get("architecture", {})
    geometry = {name: (list(architecture[name]) if isinstance(architecture.get(name), list)
                       else architecture.get(name)) for name in (
        "family", "layers", "hidden_size", "num_attention_heads", "num_kv_heads",
        "head_dim", "vocab_size", "ssm_variant", "hybrid_layer_types",
    )}
    fingerprint = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return {"schema": SCHEMA, "artifact_fingerprint": fingerprint,
            "precision_epoch": epoch, "arithmetic_dtype": dtype,
            "geometry": geometry, "adapter_id": adapter_id, "task_weights": coefficients}


def contract_key(contract: Mapping[str, Any]) -> str:
    """Hash typed contract fields without ambiguous string concatenation."""
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shared_cache_contract(contract: Mapping[str, Any], *, engine: Any) -> dict[str, Any]:
    """Partition opaque shared state by executor representation and device.

    Typed persistence can restore compatible state on another device. R2
    exchanges raw cache objects directly and must retain their storage domain.
    """
    from aether.runtime.execution_features import _PooledEngine
    from aether.runtime.spec_generation import SpeculativeGenerationAdapter

    visited: set[int] = set()
    while isinstance(engine, (SpeculativeGenerationAdapter, _PooledEngine)):
        if id(engine) in visited:
            raise ValueError("shared cache executor wrappers contain a cycle")
        visited.add(id(engine))
        engine = (
            engine.target if isinstance(engine, SpeculativeGenerationAdapter) else engine.engine
        )
    device = getattr(engine, "device", None)
    if device is None:
        # Known CPU families declare NumPy host state. Do
        # not infer CPU residency for an unknown adapter merely from absence.
        known_cpu = {
            "aether.runtime.cpu_engine.CPUExecutionEngine",
            "aether.runtime.hybrid_engine.HybridExecutionEngine",
            "aether.runtime.mamba_engine.MambaExecutionEngine",
            "aether.runtime.mamba2_engine.Mamba2ExecutionEngine",
            "aether.runtime.rwkv_engine.RWKVExecutionEngine",
            "aether.runtime.mla_engine.MLAExecutionEngine",
        }
        identity = f"{type(engine).__module__}.{type(engine).__qualname__}"
        if identity not in known_cpu:
            raise ValueError("shared cache executor does not declare its storage device")
        device = "cpu"
    return {"execution": dict(contract), "cache_storage": {
        "engine_type": f"{type(engine).__module__}.{type(engine).__qualname__}",
        "device": str(device),
    }}


def validate_persisted_contract(handle: Any, contract: Any) -> dict[str, Any]:
    """Validate portable provenance without activating its saved adapter/tasks."""
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ValueError("persisted session has no verified execution weight contract")
    expected = session_execution_contract(
        handle, adapter_id=contract.get("adapter_id"), task_weights=contract.get("task_weights")
    )
    if contract_key(contract) != contract_key(expected):
        raise ValueError("persisted session execution weight or precision contract is incompatible")
    return expected
