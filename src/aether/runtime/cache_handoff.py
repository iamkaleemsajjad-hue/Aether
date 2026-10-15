"""Lossless, explicitly typed CPU decoder state handoff.

This codec snapshots actual attention, recurrent and hybrid inference state. It
does not serialize arbitrary Python objects or infer a model identity from its
name. The authenticated loader supplies the fingerprint; receivers must match
it before restoring state. Spare attention capacity is excluded from snapshots.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from aether.hybrid.state import MambaState
from aether.runtime.cpu_engine import KVCache
from aether.runtime.hybrid_engine import HybridCache
from aether.runtime.mamba_engine import MambaCache
from aether.runtime.mamba2_engine import Mamba2Cache
from aether.runtime.mla_engine import MLACache
from aether.runtime.rwkv_engine import RWKVCache
from aether.runtime.kv_payload import CachePayload

_TYPES = {cls.__name__: cls for cls in
          (KVCache, MambaState, MambaCache, Mamba2Cache, RWKVCache, HybridCache, MLACache)}
_KINDS = {KVCache: "attention_kv_v1", MambaCache: "mamba_v1", Mamba2Cache: "mamba2_v1",
          RWKVCache: "rwkv_v1", HybridCache: "hybrid_v1", MLACache: "mla_replay_v1"}


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate(cache: Any, *, require_logits: bool = True) -> None:
    if type(cache) not in _KINDS:
        raise ValueError(f"Unsupported CPU cache type: {type(cache).__name__}")
    length = _integer(cache.length, "cache logical length")
    last = getattr(cache, "last_logits", None)
    if require_logits and length and (not isinstance(last, np.ndarray) or last.ndim != 1 or not last.size):
        raise ValueError("A populated cache must include its final logits")
    if isinstance(cache, KVCache):
        layers = _integer(cache.num_layers, "cache layer count")
        if not layers or any(not isinstance(values, list) or len(values) != layers for values in
            (cache.keys, cache.values, cache.positions, cache.stored_lengths)):
            raise ValueError("KV cache layer descriptors do not match num_layers")
        _integer(cache.reserve_length, "cache reserve length")
        for index in range(layers):
            stored = _integer(cache.stored_lengths[index], "stored KV length")
            key, value, position = cache.keys[index], cache.values[index], cache.positions[index]
            if key is None or value is None or position is None:
                if not all(item is None for item in (key, value, position)) or stored:
                    raise ValueError("KV cache has incomplete layer state")
                continue
            if (not all(isinstance(array, np.ndarray) for array in (key, value, position))
                or key.ndim != 3 or value.ndim != 3 or key.shape != value.shape
                or key.dtype.kind != "f" or value.dtype.kind != "f" or not all(key.shape[1:])
                or position.ndim != 1 or position.dtype.kind not in "iu"
                or key.shape[0] < stored or position.size < stored or stored > length):
                raise ValueError("KV cache has invalid tensor geometry or stored length")
            positions = position[:stored]
            if stored and (np.any(positions < 0) or np.any(positions >= length)
                           or np.any(np.diff(positions) <= 0)):
                raise ValueError("KV positions must be increasing and within the logical prefix")
    elif isinstance(cache, HybridCache):
        # Inactive attention layers legitimately have no rows; recurrent state
        # remains required even for those layers so the original schedule survives.
        _validate(cache.kv, require_logits=False)
        _validate(cache.ssm, require_logits=False)
        if cache.kv.length != length or cache.ssm.length != length:
            raise ValueError("Hybrid KV/recurrent lengths do not match the logical prefix")
    elif isinstance(cache, (MambaCache, Mamba2Cache)):
        if not cache.states or len(cache.states) != len(cache.conv_history):
            raise ValueError("Recurrent state and convolution-history layers must match")
        for state, history in zip(cache.states, cache.conv_history):
            if not isinstance(history, np.ndarray) or history.ndim != 3:
                raise ValueError("Recurrent convolution history must be a rank-three tensor")
            if isinstance(cache, MambaCache):
                if not isinstance(state, MambaState) or not isinstance(state.h, np.ndarray) or state.h.ndim != 3:
                    raise ValueError("Mamba state must include its rank-three recurrent tensor")
                _integer(state.layer_idx, "Mamba layer index")
                _integer(state.step, "Mamba step count")
            elif not isinstance(state, np.ndarray) or state.ndim != 4:
                raise ValueError("Mamba-2 state must be a rank-four tensor")
    elif isinstance(cache, RWKVCache):
        if not cache.aa or len({len(cache.aa), len(cache.bb), len(cache.pp), len(cache.previous)}) != 1:
            raise ValueError("RWKV recurrent layer descriptors must match")
        for index in range(len(cache.aa)):
            group = (cache.aa[index], cache.bb[index], cache.pp[index], cache.previous[index])
            if not all(isinstance(array, np.ndarray) for array in group) or len({array.shape for array in group[:3]}) != 1:
                raise ValueError("RWKV accumulator tensors must have matching geometry")
    elif isinstance(cache, MLACache):
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in cache.token_ids):
            raise ValueError("MLA replay state contains invalid token IDs")


def export_cpu_cache(cache: Any, *, model_fingerprint: str) -> CachePayload:
    """Snapshot supported state without sharing writable arrays with the source."""
    _validate(cache)
    tensors: dict[str, np.ndarray] = {}
    aliases: dict[tuple[int, int | None], str] = {}

    def encode(value: Any, valid_rows: int | None = None, allow_negative_inf: bool = False) -> Any:
        if isinstance(value, np.ndarray):
            array = value if valid_rows is None else value[:valid_rows]
            if array.dtype.kind not in "fiub" or np.isnan(array).any() or np.isposinf(array).any() or (not allow_negative_inf and np.isneginf(array).any()):
                raise ValueError("Cache tensor must contain finite numeric coordinates")
            identity = (id(value), valid_rows)
            if identity not in aliases:
                name = f"tensor_{len(tensors):06d}"
                aliases[identity] = name
                tensors[name] = np.ascontiguousarray(array)
            return {"tensor": aliases[identity]}
        if type(value).__name__ in _TYPES and type(value) is _TYPES[type(value).__name__]:
            members = {}
            for item in fields(value):
                field_value = getattr(value, item.name)
                if isinstance(value, KVCache) and item.name in {"keys", "values", "positions"}:
                    members[item.name] = [encode(array, value.stored_lengths[index])
                                          for index, array in enumerate(field_value)]
                else:
                    members[item.name] = encode(field_value, allow_negative_inf=isinstance(value, RWKVCache) and item.name == "pp")
            return {"type": type(value).__name__, "fields": members}
        if isinstance(value, list):
            return [encode(item, allow_negative_inf=allow_negative_inf) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError(f"Unsupported cache field type: {type(value).__name__}")

    root = encode(cache)
    return CachePayload(model_fingerprint=model_fingerprint, cache_kind=_KINDS[type(cache)],
        logical_length=cache.length, tensors=tensors, metadata={"schema": "aether_cpu_cache_v1", "root": root})


def restore_cpu_cache(payload: CachePayload, *, expected_model_fingerprint: str) -> Any:
    """Restore owned state and aliases after strict identity/schema checks."""
    if not expected_model_fingerprint or payload.model_fingerprint != expected_model_fingerprint:
        raise ValueError("Cache model fingerprint does not match the execution model")
    if payload.metadata.get("schema") != "aether_cpu_cache_v1":
        raise ValueError("Unsupported CPU cache handoff schema")
    arrays: dict[str, np.ndarray] = {}

    def decode(node: Any, depth: int = 0, allow_negative_inf: bool = False) -> Any:
        if depth > 32:
            raise ValueError("Cache state nesting exceeds the supported schema")
        if isinstance(node, list):
            return [decode(value, depth + 1, allow_negative_inf) for value in node]
        if isinstance(node, dict):
            if set(node) == {"tensor"}:
                name = node["tensor"]
                if not isinstance(name, str) or name not in payload.tensors:
                    raise ValueError("Cache tensor reference is missing")
                if name not in arrays:
                    arrays[name] = payload.array(name, copy=True)
                    if arrays[name].dtype.kind not in "fiub" or np.isnan(arrays[name]).any() or np.isposinf(arrays[name]).any():
                        raise ValueError("Cache tensor must contain finite numeric coordinates")
                if not allow_negative_inf and np.isneginf(arrays[name]).any():
                    raise ValueError("Negative infinity is only valid for empty RWKV log accumulators")
                return arrays[name]
            if set(node) != {"type", "fields"} or node["type"] not in _TYPES:
                raise ValueError("Unknown typed cache state")
            cls = _TYPES[node["type"]]
            members = node["fields"]
            if not isinstance(members, dict) or set(members) != {item.name for item in fields(cls)}:
                raise ValueError("Typed cache state fields do not match the supported schema")
            try:
                return cls(**{name: decode(value, depth + 1, cls is RWKVCache and name == "pp") for name, value in members.items()})
            except (TypeError, AttributeError, IndexError) as exc:
                raise ValueError("Invalid typed cache state fields") from exc
        if node is None or isinstance(node, (str, int, float, bool)):
            return node
        raise ValueError("Invalid cache state field")

    cache = decode(payload.metadata.get("root"))
    try:
        _validate(cache)
    except (TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Invalid cache state geometry") from exc
    if _KINDS[type(cache)] != payload.cache_kind or cache.length != payload.logical_length:
        raise ValueError("Cache kind or logical length differs from its envelope")
    if set(arrays) != set(payload.tensors):
        raise ValueError("Cache snapshot contains undeclared semantic tensors")
    return cache
