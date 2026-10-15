"""Semantic AEG lowering to standard ONNX prefill and cached decode graphs.

The exporter consumes authenticated executable weights, never a model-name
template or a source-framework trace. Unsupported state/optimizer semantics
fail before writing output. ONNX and ONNX Runtime remain optional dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import BackendError
from aether.runtime.cpu_engine import CPUExecutionEngine, KVCache

CONTRACT = "aether_onnx_decoder_v1"


def onnx_export_eligibility(package: Any) -> dict[str, Any]:
    """Check declared semantic support without importing ORT or running a model.

    This is eligibility only. Integrity, tensors and vendor operator coverage
    are checked independently during actual export and session creation.
    """
    architecture = getattr(getattr(package, "manifest", None), "architecture", None)
    data = architecture.to_dict() if hasattr(architecture, "to_dict") else dict(architecture or {})
    reason = None
    if not data:
        reason = "missing architecture metadata"
    elif (
        data.get("is_encoder")
        or data.get("is_encoder_decoder")
        or data.get("ssm_variant")
        or str(data.get("attention_type", "")).upper() == "MLA"
    ):
        reason = "specialized encoder/SSM/MLA operator contract has no ONNX lowering"
    elif data.get("is_moe") or data.get("num_experts", 0):
        reason = "routed-expert operator contract has no ONNX lowering"
    elif str(data.get("norm_type", "RMSNorm")).lower() not in {"rmsnorm", "layernorm"}:
        reason = "unsupported normalization"
    elif str(data.get("ffn_type", "SwiGLU")).lower() not in {
        "swiglu",
        "geglu",
        "gelu",
        "relu",
        "relu2",
    }:
        reason = "unsupported activation"
    metadata = getattr(package, "metadata", {})
    if isinstance(metadata, dict) and any(
        metadata.get(key)
        for key in ("sparse_attention_plan", "semantic_kv_plan", "cross_layer_kv_plan")
    ):
        reason = "active sparse/semantic/cross-layer execution plan has no ONNX lowering"
    scaling = data.get("rope_scaling")
    if isinstance(scaling, dict) and str(scaling.get("rope_type", scaling.get("type", ""))) in {
        "dynamic",
        "longrope",
    }:
        reason = "length-dependent rotary scaling has no ONNX lowering"
    return {
        "supported": reason is None,
        "reason": reason,
        "contract": CONTRACT,
        "execution_tested": False,
    }


def _error(message: str) -> BackendError:
    return BackendError(message, backend_name="onnx")


def _supported(engine: Any) -> None:
    if type(engine) is not CPUExecutionEngine:
        raise _error(
            "ONNX export currently requires the dense decoder operator contract; "
            "specialized encoder/SSM/MLA engines are unsupported"
        )
    if any(layer.experts for layer in engine.weights.layers):
        raise _error("ONNX export has no routed-expert lowering; MoE export is unsupported")
    if (
        engine.active_lora_adapter
        or engine.sparse_attention_plan
        or engine.semantic_kv_plan
        or engine.cross_layer_kv_plan
    ):
        raise _error(
            "ONNX export cannot discard active LoRA, sparse attention, semantic KV "
            "or cross-layer KV execution plans"
        )
    if engine._rope_is_length_sensitive():
        raise _error("ONNX export has no length-dependent dynamic/LongRoPE lowering")
    weights = engine.weights
    if str(weights.norm_type).lower() not in {"rmsnorm", "layernorm"}:
        raise _error(f"unsupported ONNX normalization: {weights.norm_type}")
    if str(weights.ffn_type).lower() not in {"swiglu", "geglu", "gelu", "relu", "relu2"}:
        raise _error(f"unsupported ONNX activation: {weights.ffn_type}")
    if str(weights.position_type).lower() not in {
        "rope",
        "rotary",
        "rotary_embedding",
        "absolute",
        "learned",
        "learned_absolute",
        "alibi",
        "alibi_bias",
        "none",
    }:
        raise _error(f"unsupported ONNX positional contract: {weights.position_type}")


class _Graph:
    def __init__(self, engine: CPUExecutionEngine, context: int, decode: bool) -> None:
        import onnx

        self.onnx = onnx
        self.e = engine
        self.w = engine.weights
        self.context = context
        self.decode = decode
        self.nodes: list[Any] = []
        self.initializers: list[Any] = []
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []
        self.serial = 0

    def const(self, value: Any, dtype: Any = np.float32) -> str:
        name = f"constant_{len(self.initializers)}"
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        self.initializers.append(self.onnx.numpy_helper.from_array(array, name))
        return name

    def op(self, kind: str, *inputs: str, **attributes: Any) -> str:
        self.serial += 1
        output = f"{kind.lower()}_{self.serial}"
        self.nodes.append(
            self.onnx.helper.make_node(kind, list(inputs), [output], name=output, **attributes)
        )
        return output

    def input(self, name: str, shape: list[Any], dtype: int | None = None) -> str:
        self.inputs.append(
            self.onnx.helper.make_tensor_value_info(
                name, dtype or self.onnx.TensorProto.FLOAT, shape
            )
        )
        return name

    def output(self, value: str, name: str, shape: list[Any]) -> None:
        self.nodes.append(self.onnx.helper.make_node("Identity", [value], [name], name=name))
        self.outputs.append(
            self.onnx.helper.make_tensor_value_info(name, self.onnx.TensorProto.FLOAT, shape)
        )

    def reshape(self, value: str, shape: list[int]) -> str:
        return self.op("Reshape", value, self.const(shape, np.int64))

    def unsqueeze(self, value: str, axes: list[int]) -> str:
        return self.op("Unsqueeze", value, self.const(axes, np.int64))

    def linear(self, value: str, matrix: np.ndarray, bias: np.ndarray | None = None) -> str:
        result = self.op("MatMul", value, self.const(np.asarray(matrix).T))
        return self.op("Add", result, self.const(bias)) if bias is not None else result

    def norm(
        self, value: str, weight: np.ndarray, bias: np.ndarray | None = None, *, head: bool = False
    ) -> str:
        centered = value
        if not head and str(self.w.norm_type).lower() == "layernorm":
            centered = self.op("Sub", value, self.op("ReduceMean", value, axes=[-1], keepdims=1))
        variance = self.op("ReduceMean", self.op("Mul", centered, centered), axes=[-1], keepdims=1)
        denominator = self.op("Sqrt", self.op("Add", variance, self.const(self.w.norm_eps)))
        result = self.op(
            "Mul", self.op("Div", centered, denominator), self.const(self.e._norm_weight(weight))
        )
        # CPU RMSNorm ignores a bias; LayerNorm carries it explicitly.
        if bias is not None and not head and str(self.w.norm_type).lower() == "layernorm":
            result = self.op("Add", result, self.const(bias))
        return result

    def scaled(self, value: str, scalar: float | None) -> str:
        return value if scalar is None else self.op("Mul", value, self.const(scalar))

    def softcap(self, value: str, scalar: float | None) -> str:
        return (
            value
            if scalar is None or scalar <= 0
            else self.scaled(self.op("Tanh", self.op("Div", value, self.const(scalar))), scalar)
        )

    def gelu(self, value: str) -> str:
        if self.w.gelu_approximate:
            cubic = self.op("Mul", self.op("Mul", value, value), value)
            argument = self.scaled(
                self.op("Add", value, self.scaled(cubic, 0.044715)), float(np.sqrt(2.0 / np.pi))
            )
            cdf = self.op("Tanh", argument)
        else:
            cdf = self.op("Erf", self.scaled(value, float(1.0 / np.sqrt(2.0))))
        return self.scaled(self.op("Mul", value, self.op("Add", cdf, self.const(1.0))), 0.5)

    def activation(self, gate: str, up: str | None) -> str:
        kind = str(self.w.ffn_type).lower()
        if up is None:
            if kind == "gelu":
                return self.gelu(gate)
            if kind in {"relu", "relu2"}:
                value = self.op("Relu", gate)
                return self.op("Mul", value, value) if kind == "relu2" else value
            raise _error(f"ONNX activation {kind} requires an up projection")
        activated = (
            self.gelu(gate)
            if kind in {"geglu", "gelu"}
            else self.op("Mul", gate, self.op("Sigmoid", gate))
        )
        return self.op("Mul", activated, up)

    def rope(self, value: str, positions: str, *, local: bool) -> str:
        cosine, sine = self.e._rope_tables_for_theta(
            float(
                self.w.rope_local_theta if local and self.w.rope_local_theta else self.w.rope_theta
            ),
            self.context,
        )
        cos = self.unsqueeze(self.op("Gather", self.const(cosine), positions, axis=0), [1])
        sin = self.unsqueeze(self.op("Gather", self.const(sine), positions, axis=0), [1])
        rotary, dim = self.e._rotary_dim, self.e.head_dim
        half = rotary // 2
        first = np.arange(0, rotary, 2) if self.w.rope_interleaved else np.arange(half)
        second = np.arange(1, rotary, 2) if self.w.rope_interleaved else np.arange(half, rotary)
        low = self.op("Gather", value, self.const(first, np.int64), axis=2)
        high = self.op("Gather", value, self.const(second, np.int64), axis=2)
        left = self.op("Sub", self.op("Mul", low, cos), self.op("Mul", high, sin))
        right = self.op("Add", self.op("Mul", high, cos), self.op("Mul", low, sin))
        rotated = self.op("Concat", left, right, axis=2)
        if self.w.rope_interleaved:
            indices = np.column_stack([np.arange(half), np.arange(half) + half]).reshape(-1)
            rotated = self.op("Gather", rotated, self.const(indices, np.int64), axis=2)
        if rotary < dim:
            tail = self.op("Gather", value, self.const(np.arange(rotary, dim), np.int64), axis=2)
            rotated = self.op("Concat", rotated, tail, axis=2)
        return rotated

    def build(self) -> Any:
        e, w = self.e, self.w
        sequence: str | int = 1 if self.decode else "sequence"
        total: str = "total_sequence"
        ids = self.input("input_ids", [1, sequence], self.onnx.TensorProto.INT64)
        positions = self.input("position_ids", [sequence], self.onnx.TensorProto.INT64)
        hidden = self.op(
            "Squeeze",
            self.op("Gather", self.const(w.embedding), ids, axis=0),
            self.const([0], np.int64),
        )
        hidden = self.scaled(hidden, w.embedding_scale)
        if w.embedding_norm is not None:
            hidden = self.norm(hidden, w.embedding_norm, w.embedding_norm_bias)
        if w.position_embedding is not None:
            hidden = self.op(
                "Add",
                hidden,
                self.op("Gather", self.const(w.position_embedding), positions, axis=0),
            )
        for index, layer in enumerate(w.layers):
            block = hidden
            post = e._norm_placement == "post"
            normalized = (
                hidden
                if post
                else self.norm(hidden, layer.attention_norm, layer.attention_norm_bias)
            )
            projections: list[str] = []
            local = bool(
                w.attention_layers
                and index < len(w.attention_layers)
                and str(w.attention_layers[index]).lower() in {"local", "window", "sliding_window"}
            )
            for role, heads in (("q", e.num_heads), ("k", e.num_kv_heads), ("v", e.num_kv_heads)):
                value = self.linear(
                    normalized, getattr(layer, f"{role}_proj"), getattr(layer, f"{role}_proj_bias")
                )
                norm_weight = getattr(layer, f"{role}_norm", None)
                if norm_weight is not None and e._qk_norm_is_full:
                    value = self.norm(value, norm_weight)
                value = self.reshape(value, [-1, heads, e.head_dim])
                if norm_weight is not None and not e._qk_norm_is_full:
                    value = self.norm(value, norm_weight, head=True)
                if (
                    role != "v"
                    and str(w.position_type).lower() in {"rope", "rotary", "rotary_embedding"}
                    and e._layer_uses_rope(index)
                ):
                    value = self.rope(value, positions, local=local)
                projections.append(value)
            query, key, value = projections
            key = self.op("Transpose", key, perm=[1, 0, 2])
            value = self.op("Transpose", value, perm=[1, 0, 2])
            if self.decode:
                key = self.op(
                    "Concat",
                    self.input(f"past_key_{index}", [e.num_kv_heads, "past_sequence", e.head_dim]),
                    key,
                    axis=1,
                )
                value = self.op(
                    "Concat",
                    self.input(
                        f"past_value_{index}", [e.num_kv_heads, "past_sequence", e.head_dim]
                    ),
                    value,
                    axis=1,
                )
            self.output(key, f"present_key_{index}", [e.num_kv_heads, total, e.head_dim])
            self.output(value, f"present_value_{index}", [e.num_kv_heads, total, e.head_dim])
            # Grouped broadcast MatMul shares K/V across query groups without
            # materializing repeated full-history key/value tensors.
            repeats = e.num_heads // e.num_kv_heads
            query = self.reshape(
                self.op("Transpose", query, perm=[1, 0, 2]),
                [e.num_kv_heads, repeats, -1, e.head_dim],
            )
            scores = self.op(
                "MatMul", query, self.unsqueeze(self.op("Transpose", key, perm=[0, 2, 1]), [1])
            )
            scores = self.scaled(scores, e._layer_attention_scale(index))
            if str(w.position_type).lower() in {"alibi", "alibi_bias"}:
                distance = (
                    self.input("position_distance", [sequence, total])
                    if not any(v.name == "position_distance" for v in self.inputs)
                    else "position_distance"
                )
                slope = self.const(e._alibi_slopes.reshape(e.num_kv_heads, repeats, 1, 1))
                scores = self.op("Add", scores, self.op("Mul", slope, distance))
            scores = self.softcap(scores, w.attn_logit_softcap)
            mask = self.input(f"attention_mask_{index}", [1, 1, sequence, total])
            probabilities = self.op("Softmax", self.op("Add", scores, mask), axis=-1)
            context = self.op("MatMul", probabilities, self.unsqueeze(value, [1]))
            context = self.reshape(context, [e.num_heads, -1, e.head_dim])
            context = self.reshape(
                self.op("Transpose", context, perm=[1, 0, 2]), [-1, e.num_heads * e.head_dim]
            )
            attention = self.linear(context, layer.o_proj, layer.o_proj_bias)
            if layer.post_attention_norm is not None:
                attention = self.norm(
                    attention, layer.post_attention_norm, layer.post_attention_norm_bias
                )
            elif post:
                attention = self.norm(attention, layer.attention_norm, layer.attention_norm_bias)
            attention = self.scaled(attention, w.residual_scale)
            hidden = self.op("Add", block, attention)
            if w.parallel_residual:
                ffn_input = self.norm(block, layer.ffn_norm, layer.ffn_norm_bias)
            else:
                ffn_input = (
                    hidden if post else self.norm(hidden, layer.ffn_norm, layer.ffn_norm_bias)
                )
            gate = self.linear(ffn_input, layer.gate_proj, layer.gate_proj_bias)
            up = (
                self.linear(ffn_input, layer.up_proj, layer.up_proj_bias)
                if layer.up_proj is not None
                else None
            )
            ffn = self.linear(self.activation(gate, up), layer.down_proj, layer.down_proj_bias)
            if layer.post_ffn_norm is not None:
                ffn = self.norm(ffn, layer.post_ffn_norm, layer.post_ffn_norm_bias)
            elif post:
                ffn = self.norm(ffn, layer.ffn_norm, layer.ffn_norm_bias)
            hidden = self.op("Add", hidden, self.scaled(ffn, w.residual_scale))
        hidden = self.norm(hidden, w.final_norm, w.final_norm_bias)
        logits = self.softcap(
            self.scaled(self.linear(hidden, w.lm_head), w.logit_scale), w.final_logit_softcap
        )
        self.output(self.unsqueeze(logits, [0]), "logits", [1, sequence, w.vocab_size])
        self.output(hidden, "hidden_states", [sequence, w.hidden_size])
        graph = self.onnx.helper.make_graph(
            self.nodes,
            "aether_cached_decode" if self.decode else "aether_prefill",
            self.inputs,
            self.outputs,
            self.initializers,
        )
        model = self.onnx.helper.make_model(
            graph, producer_name="aether", opset_imports=[self.onnx.helper.make_opsetid("", 17)]
        )
        model.ir_version = 9
        self.onnx.helper.set_model_props(
            model,
            {"aether.contract": CONTRACT, "aether.phase": "decode" if self.decode else "prefill"},
        )
        self.onnx.checker.check_model(model)
        return model


@dataclass(frozen=True)
class ONNXExport:
    """An exported prefill/decode pair sharing authenticated model weights."""

    root: Path
    metadata: dict[str, Any]

    @property
    def prefill_path(self) -> Path:
        """Return the model accepting the complete initial prompt."""
        return self.root / "prefill.onnx"

    @property
    def decode_path(self) -> Path:
        """Return the model consuming one token and the existing KV state."""
        return self.root / "decode.onnx"


def export_engine_to_onnx(
    engine: CPUExecutionEngine,
    output_dir: str | Path,
    *,
    max_context: int | None = None,
    model_identity: str | None = None,
) -> ONNXExport:
    """Write a reusable dense-operator bundle; never mutate the source engine.

    Quantized artifact weights are already dequantized by the authenticated
    loader and are exported as FP32 arithmetic, not as native INT4/FP8 claims.
    Both graphs reference one deduplicated external weight file.
    """
    import onnx

    _supported(engine)
    context = (
        int(engine.weights.context_length or 4096) if max_context is None else int(max_context)
    )
    if context <= 0:
        raise _error("ONNX max_context must be positive")
    if engine.weights.position_embedding is not None:
        context = min(context, engine.weights.position_embedding.shape[0])
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"ONNX export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aether-onnx-", dir=destination.parent))
    fingerprint = hashlib.sha256()
    seen: dict[str, tuple[int, int]] = {}
    try:
        with (staging / "weights.data").open("wb") as weights_file:
            for decode in (False, True):
                model = _Graph(engine, context, decode).build()
                for tensor in model.graph.initializer:
                    data = tensor.raw_data
                    digest = hashlib.sha256(data).hexdigest()
                    identity = f"{tensor.data_type}:{list(tensor.dims)}:{digest}"
                    fingerprint.update(identity.encode())
                    # Shape inference needs axes/reshape/index constants inline;
                    # ORT cannot parse external shape tensors at model load.
                    if tensor.data_type != onnx.TensorProto.FLOAT or len(data) < 1024:
                        continue
                    if identity not in seen:
                        offset = weights_file.tell()
                        weights_file.write(data)
                        seen[identity] = (offset, len(data))
                    offset, size = seen[identity]
                    onnx.external_data_helper.set_external_data(
                        tensor, location="weights.data", offset=offset, length=size
                    )
                    tensor.ClearField("raw_data")
                    tensor.data_location = onnx.TensorProto.EXTERNAL
                name = "decode.onnx" if decode else "prefill.onnx"
                fingerprint.update(model.SerializeToString())
                (staging / name).write_bytes(model.SerializeToString())
        fingerprint.update(
            json.dumps(
                {
                    "attention_layers": engine.weights.attention_layers,
                    "attention_window": engine.weights.attention_window,
                },
                sort_keys=True,
            ).encode()
        )
        metadata = {
            "format": CONTRACT,
            "model_identity": model_identity or fingerprint.hexdigest(),
            "weights_fingerprint": fingerprint.hexdigest(),
            "max_context": context,
            "num_layers": engine.weights.num_layers,
            "num_heads": engine.num_heads,
            "num_kv_heads": engine.num_kv_heads,
            "head_dim": engine.head_dim,
            "vocab_size": engine.weights.vocab_size,
            "hidden_size": engine.weights.hidden_size,
            "arithmetic_precision": "fp32",
            "position_type": engine.weights.position_type,
            "attention_layers": engine.weights.attention_layers,
            "attention_window": engine.weights.attention_window,
        }
        (staging / "contract.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for name in ("prefill.onnx", "decode.onnx"):
            onnx.checker.check_model(str(staging / name))
        os.rename(staging, destination)
        return ONNXExport(destination, metadata)
    except BaseException:
        shutil.rmtree(staging)
        raise


def export_aeg_to_onnx(
    aeg: Any, output_dir: str | Path, *, max_context: int | None = None
) -> ONNXExport:
    """Explicitly export an integrity-checked AEG into standard ONNX operators."""
    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import load_engine_from_package

    package = aeg if isinstance(aeg, AEGPackage) else AEGPackage(Path(aeg))
    if package.manifest is None:
        package.load()
    engine = load_engine_from_package(package, prepare_native_kernels=False)
    return export_engine_to_onnx(engine, output_dir, max_context=max_context)


class ONNXAEGEngine:
    """Real provider execution of exported prefill and incremental KV decode.

    NumPy holds request cache tensors and masks; model operators run through
    the configured ORT providers. Provider placement requires separate profiling.
    DirectML calls are serialized because its sessions prohibit concurrent Run.
    """

    def __init__(self, bundle: ONNXExport, prefill: Any, decode: Any) -> None:
        self.bundle = bundle
        self.prefill_session = prefill
        self.decode_session = decode
        self._lock = threading.Lock()
        self.device = "onnxruntime"
        self.num_layers = int(bundle.metadata["num_layers"])

    def get_providers(self) -> list[str]:
        """Return initialized providers without claiming node execution."""
        return list(self.prefill_session.get_providers())

    def forward(
        self, token_ids: np.ndarray, cache: KVCache | None = None
    ) -> tuple[np.ndarray, KVCache]:
        """Execute prompt or cached suffix operators and update owned KV state."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        meta = self.bundle.metadata
        if ids.size == 0 or ids.min() < 0 or ids.max() >= meta["vocab_size"]:
            raise _error("ONNX AEG requires nonempty in-vocabulary token IDs")
        cache = cache or KVCache(self.num_layers)
        if cache.num_layers != self.num_layers:
            raise _error("ONNX KV cache layer count is incompatible")
        if cache.length and ids.size != 1:
            # The decode graph is optimized for a one-token step. Chunked
            # suffixes retain exact cached semantics through repeated steps.
            rows = []
            for token in ids:
                row, cache = self.forward(np.asarray([token]), cache)
                rows.append(row)
            return np.concatenate(rows, axis=0), cache
        past = cache.length
        total = past + ids.size
        if total > meta["max_context"]:
            raise _error(
                f"ONNX sequence length {total} exceeds exported context {meta['max_context']}"
            )
        positions = np.arange(past, total, dtype=np.int64)
        key_positions = np.arange(total, dtype=np.int64)
        feeds = {"input_ids": ids[None, :], "position_ids": positions}
        for index in range(self.num_layers):
            if past:
                key, value, cached_positions = cache.layer_view(index)
                expected = (past, meta["num_kv_heads"], meta["head_dim"])
                if (
                    key is None
                    or value is None
                    or key.shape != expected
                    or value.shape != expected
                    or not np.array_equal(cached_positions, np.arange(past))
                ):
                    raise _error(
                        "ONNX cache requires complete, contiguous uncompressed KV "
                        "positions and matching geometry"
                    )
                feeds[f"past_key_{index}"] = np.ascontiguousarray(
                    key.transpose(1, 0, 2), dtype=np.float32
                )
                feeds[f"past_value_{index}"] = np.ascontiguousarray(
                    value.transpose(1, 0, 2), dtype=np.float32
                )
            allowed = key_positions[None, :] <= positions[:, None]
            layers = meta.get("attention_layers")
            local = bool(
                layers
                and index < len(layers)
                and str(layers[index]).lower() in {"local", "window", "sliding_window"}
            )
            window = int(meta.get("attention_window") or 0)
            if local and window > 0:
                allowed &= key_positions[None, :] >= positions[:, None] - window + 1
            feeds[f"attention_mask_{index}"] = np.where(
                allowed, np.float32(0.0), np.float32(-np.inf)
            )[None, None]
        if str(meta["position_type"]).lower() in {"alibi", "alibi_bias"}:
            feeds["position_distance"] = (key_positions[None, :] - positions[:, None]).astype(
                np.float32
            )
        session = self.decode_session if past else self.prefill_session
        with self._lock:
            outputs = dict(
                zip(
                    (spec.name for spec in session.get_outputs()),
                    session.run(None, feeds),
                    strict=True,
                )
            )
        logits = np.asarray(outputs["logits"][0], dtype=np.float32)
        if not np.isfinite(logits).all():
            raise _error("ONNX AEG produced nonfinite logits")
        for index in range(self.num_layers):
            cache.replace_layer(
                index,
                np.ascontiguousarray(outputs[f"present_key_{index}"].transpose(1, 0, 2)),
                np.ascontiguousarray(outputs[f"present_value_{index}"].transpose(1, 0, 2)),
                key_positions,
            )
        cache.logical_length = total
        cache.last_logits = logits[-1].copy()
        cache.last_hidden = np.asarray(outputs["hidden_states"][-1], dtype=np.float32).copy()
        return logits, cache

    def forward_step(
        self,
        token_ids: np.ndarray,
        kv_cache: KVCache | None = None,
        return_all_logits: bool = False,
    ) -> tuple[np.ndarray, KVCache]:
        """Return final or complete logits through the common decoder interface."""
        logits, cache = self.forward(token_ids, kv_cache)
        return (logits if return_all_logits else logits[-1]), cache

    def generate_iter(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        *,
        cache: KVCache | None = None,
        cache_callback: Any | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Any = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Yield sampled tokens while retaining real incremental decoder state."""
        from aether.runtime.sampling import SamplingConfig, sample_token
        from aether.runtime.stopping import stop_token_set

        grammar = kwargs.get("grammar_session")
        if max_tokens < 1:
            raise _error("max_tokens must be positive")
        ids = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        if ids.size:
            logits, cache = self.forward(ids, cache)
            current = logits[-1]
        elif cache is not None and cache.last_logits is not None:
            current = cache.last_logits
        else:
            raise _error("generation requires prompt IDs or populated cache")
        rng = np.random.default_rng(seed)
        sampling = SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p)
        stops = stop_token_set(eos_token_id)
        try:
            for _ in range(max_tokens):
                token = int(sample_token(current, sampling, rng=rng, grammar_session=grammar))
                yield token
                if token in stops:
                    break
                if grammar is not None and getattr(grammar, "is_accepting", lambda: False)():
                    break
                logits, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
                current = logits[-1]
        finally:
            if cache_callback is not None:
                cache_callback(cache)

    def generate_with_cache(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        *,
        cache: KVCache | None = None,
        **kwargs: Any,
    ) -> tuple[list[int], KVCache]:
        """Complete generation and return its last consumed-token cache."""
        holder = [cache]
        tokens = list(
            self.generate_iter(
                prompt_ids,
                max_tokens,
                cache=cache,
                cache_callback=lambda value: holder.__setitem__(0, value),
                **kwargs,
            )
        )
        return tokens, holder[0]

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        """Collect generated token IDs through the same incremental iterator."""
        return list(self.generate_iter(prompt_ids, max_tokens, **kwargs))

    def create_phase_adapter_factory(self, model_fingerprint: str) -> Any:
        """Independent Python execution state sharing serialized ORT sessions.

        ORT model tensors are immutable. DirectML Run calls use the shared lock;
        every phase/request owns its own KV cache and snapshot buffers.
        """
        from aether.runtime.disaggregated_execution import CPUDecoderAdapter

        def factory() -> Any:
            engine = ONNXAEGEngine(self.bundle, self.prefill_session, self.decode_session)
            engine._lock = self._lock
            engine.device = self.device
            adapter = CPUDecoderAdapter(
                engine, model_fingerprint=model_fingerprint, cache_kind="attention_kv_v1"
            )
            adapter.device_id = self.device
            return adapter

        return factory
