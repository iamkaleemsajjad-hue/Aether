"""Exact portable Torch cache snapshots and independent phase-worker adapters.

Transfers use typed owned host bytes, not pickle, RDMA or dtype conversion.
Device visibility and successful transfers remain real framework operations.
Only explicitly supported nonsharded attention/hybrid engines are accepted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.runtime.kv_payload import CachePayload, TensorPayload

SCHEMA = "aether_torch_cache_v1"


def _types() -> tuple[Any, Any, Any, Any]:
    from aether.runtime.torch_engine import (
        TorchAEGEngine,
        TorchHybridAEGEngine,
        TorchHybridCache,
        TorchKVCache,
    )

    return TorchAEGEngine, TorchHybridAEGEngine, TorchKVCache, TorchHybridCache


def _geometry(engine: Any) -> dict[str, Any]:
    dense, hybrid, _, _ = _types()
    if type(engine) not in {dense, hybrid} or len(getattr(engine, "devices", [engine.device])) != 1:
        raise ValueError(
            "Torch cache handoff supports exact nonsharded attention/hybrid engines; "
            "collective meshes require their own adapter"
        )
    result = {
        "num_layers": int(engine.num_layers),
        "num_heads": int(engine.num_heads),
        "num_kv_heads": int(engine.num_kv_heads),
        "head_dim": int(engine.head_dim),
        "vocab_size": int(engine.embedding.shape[0]),
        "hidden_size": int(engine.embedding.shape[1]),
        "compute_dtype": str(getattr(engine, "compute_dtype", engine.embedding.dtype)),
    }
    if type(engine) is hybrid:
        result.update(
            layer_types=list(engine.layer_types),
            inner_size=int(engine.inner_size),
            state_size=int(engine.state_size),
            conv_kernel=int(engine.conv_kernel),
        )
    return result


def _check_tensor(value: Any, torch: Any) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_floating_point()
    ):
        raise ValueError("Torch cache coordinates must be dense floating tensors")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise ValueError("Torch cache coordinates must be finite")


def export_torch_cache(cache: Any, *, engine: Any, model_fingerprint: str) -> CachePayload:
    """Snapshot valid rows, aliases and recurrent state with exact original dtype."""
    _, hybrid_engine, attention, hybrid = _types()
    geometry = _geometry(engine)
    expected = hybrid if type(engine) is hybrid_engine else attention
    if type(cache) is not expected or type(cache.length) is not int or cache.length < 0:
        raise ValueError("Torch cache type/length differs from the engine contract")
    length, layers = cache.length, geometry["num_layers"]
    if len(cache.keys) != layers or len(cache.values) != layers:
        raise ValueError("Torch cache layer count differs from model geometry")
    tensors: dict[str, TensorPayload] = {}
    aliases: dict[tuple[int, int | None], str] = {}
    fields: dict[str, Any] = {"keys": [], "values": []}

    def store(value: Any, rows: int | None = None) -> str | None:
        if value is None:
            return None
        source = value if rows is None else value[:rows]
        _check_tensor(source, engine.torch)
        identity = (id(value), rows)
        if identity not in aliases:
            name = f"tensor_{len(tensors):06d}"
            aliases[identity] = name
            tensors[name] = TensorPayload.from_tensor(source)
        return aliases[identity]

    for index, (key, value) in enumerate(zip(cache.keys, cache.values, strict=True)):
        active = expected is attention or engine.layer_types[index] == "attention"
        if key is None or value is None:
            if key is not None or value is not None or (length and active):
                raise ValueError("Torch attention cache has incomplete active layer state")
        else:
            shape = (geometry["num_kv_heads"], geometry["head_dim"])
            if (
                key.ndim != 3
                or value.ndim != 3
                or key.shape != value.shape
                or key.shape[0] < length
                or tuple(key.shape[1:]) != shape
                or key.dtype != value.dtype
            ):
                raise ValueError("Torch KV geometry/dtype differs from the execution contract")
        fields["keys"].append(store(key, length))
        fields["values"].append(store(value, length))
    fields["last_logits"] = store(cache.last_logits)
    if length and (
        cache.last_logits is None or tuple(cache.last_logits.shape) != (geometry["vocab_size"],)
    ):
        raise ValueError("Populated Torch cache requires final vocabulary logits")
    if expected is hybrid:
        if len(cache.states) != layers or len(cache.conv_history) != layers:
            raise ValueError("Hybrid recurrent state layer count differs from model schedule")
        fields["states"], fields["conv_history"] = [], []
        for state, history in zip(cache.states, cache.conv_history, strict=True):
            if tuple(state.shape) != (1, engine.inner_size, engine.state_size) or tuple(
                history.shape
            ) != (1, engine.inner_size, max(engine.conv_kernel - 1, 0)):
                raise ValueError("Hybrid recurrent state geometry differs from model schedule")
            fields["states"].append(store(state))
            fields["conv_history"].append(store(history))
        if cache.last_hidden is not None and tuple(cache.last_hidden.shape) != (
            geometry["hidden_size"],
        ):
            raise ValueError("Hybrid final hidden state has invalid geometry")
        fields["last_hidden"] = store(cache.last_hidden)
    return CachePayload(
        model_fingerprint,
        "hybrid_v1" if expected is hybrid else "attention_kv_v1",
        length,
        tensors,
        {"schema": SCHEMA, "cache_type": expected.__name__, "geometry": geometry, "fields": fields},
    )


def _cpu_to_torch(payload: CachePayload, engine: Any, fingerprint: str) -> Any:
    from aether.runtime.cache_handoff import restore_cpu_cache
    from aether.runtime.cpu_engine import KVCache

    dense, _, attention, _ = _types()
    if type(engine) is not dense or payload.cache_kind != "attention_kv_v1":
        raise ValueError("CPU/Torch interoperability currently requires contiguous attention KV")
    cache = restore_cpu_cache(payload, expected_model_fingerprint=fingerprint)
    if type(cache) is not KVCache or cache.num_layers != engine.num_layers:
        raise ValueError("CPU cache family/layers differ from destination engine")
    keys, values = [], []
    for index in range(cache.num_layers):
        key, value, positions = cache.layer_view(index)
        if cache.length and (
            key is None
            or key.shape != (cache.length, engine.num_kv_heads, engine.head_dim)
            or value.shape != key.shape
            or not np.array_equal(positions, np.arange(cache.length))
        ):
            raise ValueError("CPU/Torch handoff requires complete contiguous uncompressed KV state")
        keys.append(None if key is None else engine.torch.from_numpy(key.copy()).to(engine.device))
        values.append(
            None if value is None else engine.torch.from_numpy(value.copy()).to(engine.device)
        )
    last = (
        None
        if cache.last_logits is None
        else engine.torch.from_numpy(cache.last_logits.copy()).to(engine.device)
    )
    result = attention(keys, values, cache.length, last)
    # Validate dtype/geometry without silently casting a CPU-produced cache.
    export_torch_cache(result, engine=engine, model_fingerprint=fingerprint)
    expected_dtype = getattr(engine, "compute_dtype", engine.embedding.dtype)
    if any(value is not None and value.dtype != expected_dtype for value in (*keys, *values)):
        raise ValueError(
            "CPU cache dtype differs from destination execution dtype; "
            "conversion is not a lossless handoff"
        )
    return result


def restore_torch_cache(
    payload: CachePayload, *, engine: Any, expected_model_fingerprint: str
) -> Any:
    """Restore owned device coordinates, rejecting identity/layout/dtype changes."""
    if payload.model_fingerprint != expected_model_fingerprint or not expected_model_fingerprint:
        raise ValueError("Cache model fingerprint differs from destination execution model")
    if payload.metadata.get("schema") == "aether_cpu_cache_v1":
        return _cpu_to_torch(payload, engine, expected_model_fingerprint)
    geometry = _geometry(engine)
    _, hybrid_engine, attention, hybrid = _types()
    expected = hybrid if type(engine) is hybrid_engine else attention
    kind = "hybrid_v1" if expected is hybrid else "attention_kv_v1"
    if (
        payload.cache_kind != kind
        or payload.metadata.get("schema") != SCHEMA
        or payload.metadata.get("cache_type") != expected.__name__
        or payload.metadata.get("geometry") != geometry
    ):
        raise ValueError(
            "Torch cache schema/kind/geometry/dtype differs from destination execution contract"
        )
    fields = payload.metadata.get("fields")
    names = {"keys", "values", "last_logits"} | (
        {"states", "conv_history", "last_hidden"} if expected is hybrid else set()
    )
    if not isinstance(fields, dict) or set(fields) != names:
        raise ValueError("Torch cache fields differ from the supported schema")
    restored: dict[str, Any] = {}

    def get(name: Any) -> Any:
        if name is None:
            return None
        if not isinstance(name, str) or name not in payload.tensors:
            raise ValueError("Torch cache tensor reference is missing")
        if name not in restored:
            restored[name] = payload.tensors[name].to_tensor(engine.device)
        return restored[name]

    values: dict[str, Any] = {}
    for name, references in fields.items():
        if name in {"keys", "values", "states", "conv_history"}:
            if not isinstance(references, list) or len(references) != geometry["num_layers"]:
                raise ValueError("Torch cache layer references differ from model geometry")
            values[name] = [get(reference) for reference in references]
        else:
            values[name] = get(references)
    if set(restored) != set(payload.tensors):
        raise ValueError("Torch cache payload contains undeclared semantic tensors")
    result = expected(**values, length=payload.logical_length)
    export_torch_cache(result, engine=engine, model_fingerprint=expected_model_fingerprint)
    dtype = getattr(engine, "compute_dtype", engine.embedding.dtype)
    if any(value is not None and value.dtype != dtype for value in (*result.keys, *result.values)):
        raise ValueError(
            "KV cache dtype differs from destination execution dtype; "
            "implicit conversion is forbidden"
        )
    if expected is hybrid and any(
        value.dtype != dtype for value in (*result.states, *result.conv_history)
    ):
        raise ValueError("Hybrid recurrent dtype differs from destination execution dtype")
    return result


def torch_payload_to_cpu_attention(
    payload: CachePayload, *, expected_model_fingerprint: str
) -> Any:
    """Exact NumPy-equivalent attention state only; BF16/FP8 are rejected."""
    from aether.runtime.cache_handoff import export_cpu_cache
    from aether.runtime.cpu_engine import KVCache

    if (
        payload.model_fingerprint != expected_model_fingerprint
        or payload.cache_kind != "attention_kv_v1"
        or payload.metadata.get("schema") != SCHEMA
        or payload.metadata.get("cache_type") != "TorchKVCache"
    ):
        raise ValueError("Torch/CPU attention payload identity/schema is incompatible")
    geometry, fields = payload.metadata.get("geometry", {}), payload.metadata.get("fields", {})
    count = geometry.get("num_layers")
    if (
        type(count) is not int
        or count < 1
        or set(fields) != {"keys", "values", "last_logits"}
        or any(
            not isinstance(fields.get(name), list) or len(fields[name]) != count
            for name in ("keys", "values")
        )
    ):
        raise ValueError("Torch attention payload has invalid field geometry")
    arrays: dict[str, np.ndarray] = {}

    def get(name: Any) -> Any:
        if name is None:
            return None
        if not isinstance(name, str) or name not in payload.tensors:
            raise ValueError("Torch attention payload tensor reference is missing")
        if name not in arrays:
            arrays[name] = payload.array(name)
        return arrays[name]

    keys = [get(name) for name in fields["keys"]]
    values = [get(name) for name in fields["values"]]
    cache = KVCache(
        count,
        keys=keys,
        values=values,
        positions=[
            None if key is None else np.arange(payload.logical_length, dtype=np.int64)
            for key in keys
        ],
        logical_length=payload.logical_length,
        stored_lengths=[0 if key is None else payload.logical_length for key in keys],
        last_logits=get(fields["last_logits"]),
    )
    if set(arrays) != set(payload.tensors):
        raise ValueError("Torch attention payload contains undeclared semantic tensors")
    for key in keys:
        if key is not None and key.shape != (
            payload.logical_length,
            geometry["num_kv_heads"],
            geometry["head_dim"],
        ):
            raise ValueError("Torch attention payload KV geometry is invalid")
    export_cpu_cache(cache, model_fingerprint=expected_model_fingerprint)
    return cache


def clone_torch_engine_for_worker(engine: Any, *, device_id: str | None = None) -> Any:
    """Own mutable execution state while sharing only immutable resident tensors.

    A different device receives exact tensor copies without changing dtypes.
    It requires retained source weights and framework-supported destination
    operations. Collective/sharded engines are explicitly excluded.
    """
    _geometry(engine)
    torch = engine.torch
    from aether.runtime.disaggregated_execution import clone_cpu_engine_for_worker
    from aether.runtime.torch_engine import _resolve_device

    destination = _resolve_device(torch, device_id or str(engine.device))
    same_device = destination == engine.device
    if not same_device and getattr(engine, "host_weights_released", False):
        raise ValueError(
            "Different-device workers require retained authenticated source weights; "
            "reload the artifact with host weights retained"
        )
    result = object.__new__(type(engine))
    tensor_copies: dict[int, Any] = {}

    def owned(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if same_device:
                return value
            if id(value) not in tensor_copies:
                tensor_copies[id(value)] = value.detach().to(destination).clone()
            return tensor_copies[id(value)]
        if isinstance(value, dict):
            return {key: owned(item) for key, item in value.items()}
        if isinstance(value, list):
            return [owned(item) for item in value]
        if isinstance(value, tuple):
            return tuple(owned(item) for item in value)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    result.__dict__ = {
        name: owned(value)
        for name, value in engine.__dict__.items()
        if name not in {"source_engine", "_strategies", "_readback_slots"}
    }
    source = engine.source_engine
    result.source_engine = (
        clone_cpu_engine_for_worker(source)
        if not getattr(engine, "host_weights_released", False)
        else source
    )
    result.device = destination
    if hasattr(result, "devices"):
        result.devices = [destination]
        result.layer_devices = [destination] * result.num_layers
    for name in ("_cos", "_sin", "_local_cos", "_local_sin", "_positions_cache"):
        if hasattr(result, name):
            setattr(result, name, None)
    result._readback_slots = False
    result._readback_slot = 0
    result._projection_choices = {}
    result._projection_table = {}
    result._projection_choice = None
    result._projection_rows = None
    if callable(getattr(result, "_reference_projection_factory", None)):
        result._reference_projection = result._reference_projection_factory(torch)
        result._projection = result._reference_projection
        result._strategies = result._build_strategy_calibrator()
    return result


class TorchDecoderAdapter:
    """Real framework execution with explicit, exact cache transfer boundaries."""

    def __init__(
        self, engine: Any, *, model_fingerprint: str, cache_kind: str | None = None
    ) -> None:
        _, hybrid, _, _ = _types()
        _geometry(engine)
        self.engine = engine
        self.model_fingerprint = model_fingerprint
        self.cache_kind = "hybrid_v1" if type(engine) is hybrid else "attention_kv_v1"
        if cache_kind is not None and cache_kind != self.cache_kind:
            raise ValueError("Configured cache kind differs from the Torch model contract")
        self.device_id = str(engine.device)

    def forward(self, token_ids: np.ndarray, cache: Any = None) -> tuple[np.ndarray, Any]:
        """Execute on the adapter's actual framework device."""
        return self.engine.forward(token_ids, cache)

    def export_cache(self, cache: Any) -> CachePayload:
        """Fence device work before snapshotting owned typed cache bytes."""
        self.synchronize()
        return export_torch_cache(
            cache, engine=self.engine, model_fingerprint=self.model_fingerprint
        )

    def import_cache(self, payload: CachePayload) -> Any:
        """Restore compatible owned tensors without changing precision."""
        return restore_torch_cache(
            payload, engine=self.engine, expected_model_fingerprint=self.model_fingerprint
        )

    def synchronize(self) -> None:
        """Fence CUDA/HIP or MPS execution; CPU operators are synchronous."""
        torch, device = self.engine.torch, self.engine.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def cache_nbytes(self, cache: Any) -> int:
        """Measure unique allocated tensor storage, including spare capacity."""
        # Report actual allocated storage including spare KV capacity, unlike
        # snapshots, which intentionally exclude uninitialized spare rows.
        owners: dict[tuple[str, int], int] = {}
        for name in ("keys", "values", "states", "conv_history", "last_logits", "last_hidden"):
            values = getattr(cache, name, None)
            values = values if isinstance(values, list) else [values]
            for value in values:
                if isinstance(value, self.engine.torch.Tensor):
                    storage = value.untyped_storage()
                    owners[(str(value.device), storage.data_ptr())] = storage.nbytes()
        return sum(owners.values())

    def close(self) -> None:
        """Finish outstanding adapter work before worker shutdown."""
        self.synchronize()


def make_torch_adapter_factory(
    engine: Any,
    *,
    model_fingerprint: str,
    cache_kind: str | None = None,
    device_id: str | None = None,
) -> Any:
    """Create independent actual Torch workers without deepcopying frameworks."""
    _geometry(engine)
    if not model_fingerprint:
        raise ValueError("Torch adapter factory requires an authenticated model fingerprint")
    return lambda: TorchDecoderAdapter(
        clone_torch_engine_for_worker(engine, device_id=device_id),
        model_fingerprint=model_fingerprint,
        cache_kind=cache_kind,
    )
