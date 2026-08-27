"""Single-process tensor-parallel execution for portable decoder AEGs.

The executor deliberately shards weights instead of using ``device_map=auto``.
For a device count ``P`` and a tensor dimension of length ``N`` the partition
boundaries are

    b_i = floor(i*N/P),  i = 0 .. P

so every row/column is owned exactly once.  A decoder layer uses the standard
tensor-parallel decomposition:

* Q/K/V and gated-MLP input projections are row (output) sharded;
* attention output and MLP down projections are column (input) sharded;
* the vocabulary embedding and LM head are vocabulary-row sharded.

Hidden states and KV cache are activations, not model weights.  They may be
copied between devices for a local GEMM, but no complete weight tensor is
materialized on more than one GPU.  This is a correctness-first local
implementation; vendor collectives can replace the explicit copies later.
"""

from __future__ import annotations

import time
import os
from types import SimpleNamespace
from typing import Any

import numpy as np

from aether.parallelism.sharding import balanced_partition, capacity_weighted_partition
from aether.runtime.torch_engine import (
    BatchedKVCache,
    TorchAEGEngine,
    TorchKVCache,
    execution_numerics,
    _resolve_device,
)
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def _boundaries(length: int, parts: int) -> list[tuple[int, int]]:
    if length < 0 or parts <= 0:
        raise ValueError(f"cannot partition dimension {length} across {parts} devices")
    # The floor-of-cumulative-boundaries rule is lossless when dimensions are
    # not divisible by the mesh size.  A repeated ``length // parts`` shard
    # size silently dropped the remainder rows/columns in real checkpoints.
    return balanced_partition(length, parts)


class TorchTensorParallelAEGEngine(TorchAEGEngine):
    """Execute a dense decoder AEG over all local CUDA/ROCm devices.

    This class is selected only for a standard dense decoder graph.  The
    existing specialised engines (MLA, hybrid SSM, encoder and seq2seq) are
    intentionally not routed here until their cross-device state contracts
    are implemented.
    """

    def __init__(self, cpu_engine: Any, devices: list[str]) -> None:
        if len(devices) < 2:
            raise ValueError("tensor parallel execution requires at least two devices")
        # Do not call TorchAEGEngine.__init__: it would upload complete weights
        # to the primary device before sharding them.
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by backend
            raise RuntimeError("PyTorch is required for tensor-parallel execution") from exc
        self.torch = torch
        self.devices = [_resolve_device(torch, value) for value in devices]
        self.device = self.devices[0]
        requested_dtype = str(os.environ.get("AETHER_TORCH_DTYPE", "auto")).lower()
        if requested_dtype in {"fp16", "float16", "half"}:
            self.compute_dtype = torch.float16
        elif requested_dtype in {"bf16", "bfloat16"}:
            self.compute_dtype = torch.bfloat16
        elif requested_dtype in {"fp32", "float32"} or self.device.type != "cuda":
            self.compute_dtype = torch.float32
        else:
            self.compute_dtype = torch.float16
        self._shard_weights = self._calibrate_mixed_mesh()
        total_capacity = float(sum(self._shard_weights or ()))
        self.shard_fractions = (
            [value / total_capacity for value in self._shard_weights]
            if total_capacity > 0.0 else
            [1.0 / len(self.devices)] * len(self.devices)
        )
        logger.info(
            "Tensor-parallel shard fractions across %s: %s",
            [str(device) for device in self.devices],
            [round(value, 6) for value in self.shard_fractions],
        )

        # ── Hardware topology + collective strategy ────────────────────────
        # Detect NVLink / PCIe / XGMI interconnects and log the recommended
        # AllReduce algorithm.  This is informational at init; the actual
        # collectives use torch.distributed or explicit device copies.
        # Reference: Megatron-LM §3.3 (Shoeybi et al. 2019, arXiv:1909.08053).
        try:
            from aether.parallelism.hardware_topology import detect_hardware_topology
            _device_strs = [str(d) for d in devices]
            self._topology = detect_hardware_topology(_device_strs)
            _strat = self._topology.recommend_strategy()
            # Estimate allreduce cost for a typical hidden_size=4096 layer
            # payload = 2 * hidden_size * 4 bytes (float32)
            _sample_payload = 2 * 4096 * 4
            _latency_ms = self._topology.allreduce_latency_ms(_sample_payload)
            logger.info(
                "TP collective strategy: %s | %s | "
                "estimated allreduce latency per layer (H=4096): %.3f ms",
                _strat.value,
                self._topology.summary(),
                _latency_ms,
            )
        except Exception:  # noqa: BLE001 — topology is advisory, never fatal
            self._topology = None

        source_weights = cpu_engine.weights
        # Keep only scalar graph metadata after sharding. Holding the CPU
        # engine or its weight container would retain a complete second model
        # copy in host memory.
        self.source_engine = None
        self.weights = SimpleNamespace(
            rope_theta=float(source_weights.rope_theta),
            norm_eps=float(source_weights.norm_eps),
            norm_type=source_weights.norm_type,
            ffn_type=source_weights.ffn_type,
            position_type=source_weights.position_type,
            parallel_residual=bool(getattr(source_weights, "parallel_residual", False)),
            attention_layers=getattr(source_weights, "attention_layers", None),
            attention_window=getattr(source_weights, "attention_window", None),
            final_norm_bias=getattr(source_weights, "final_norm_bias", None),
            # The execution-numerics contract must survive sharding: dropping
            # any of it here would silently execute the model as a Llama-style
            # block regardless of what it actually is.
            **execution_numerics(source_weights),
        )
        self.num_heads = int(cpu_engine.num_heads)
        self.num_kv_heads = int(cpu_engine.num_kv_heads)
        self.head_dim = int(cpu_engine.head_dim)
        self.num_layers = len(source_weights.layers)
        self._resolve_execution_numerics()
        embedding = source_weights.embedding
        self.embedding_vocab_size = int(np.asarray(embedding).shape[0])
        self.embedding_hidden_size = int(np.asarray(embedding).shape[1])
        position_embedding = source_weights.position_embedding
        self.position_hidden_size = (
            None if position_embedding is None else int(np.asarray(position_embedding).shape[1])
        )
        self._alibi_slopes = self._tensor_primary(self._source_alibi())
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        # Uneven and empty local head shards are valid for GQA/MQA.  The
        # concatenation collectives reconstruct Q/K/V before attention, so a
        # mesh wider than the KV-head count does not require replicating the
        # complete projection or silently dropping heads.
        # These plans are optional memory/latency optimizations persisted by
        # the compiler.  They must never make a portable artifact unloadable
        # on an accelerator: the exact dense attention/cache semantics remain
        # valid when an accelerator executor does not implement a particular
        # optimization.  In that case we deliberately retain the complete
        # cache for correctness and warn about the additional memory cost.
        self.persisted_optimization_plans = {
            name: value
            for name, value in (
                ("sparse_attention", cpu_engine.sparse_attention_plan),
                ("semantic_kv", cpu_engine.semantic_kv_plan),
                ("cross_layer_kv", cpu_engine.cross_layer_kv_plan),
            )
            if value is not None
        }
        if self.persisted_optimization_plans:
            logger.warning(
                "Tensor-parallel executor is using dense attention/full KV semantics; "
                "persisted optimizations are not implemented on this accelerator path: %s. "
                "Model remains runnable with higher memory use.",
                sorted(self.persisted_optimization_plans),
            )

        self.embedding, self._embedding_ranges = self._row_shards(embedding)
        self.final_norm = self._norm_primary(source_weights.final_norm)
        self.final_norm_bias = self._optional_primary(getattr(source_weights, "final_norm_bias", None))
        self.lm_head, self._lm_head_ranges = self._row_shards(
            source_weights.lm_head, self.embedding_hidden_size, "lm_head"
        )
        self.embedding_norm = self._optional_norm_primary(source_weights.embedding_norm)
        self.embedding_norm_bias = self._optional_primary(source_weights.embedding_norm_bias)
        self.position_embedding, self._position_ranges = self._optional_row_shards(position_embedding)
        self.layers = [self._convert_sharded_layer(layer) for layer in source_weights.layers]
        self._cos: Any | None = None
        self._sin: Any | None = None
        self._local_cos: Any | None = None
        self._local_sin: Any | None = None
        self._positions_cache: Any | None = None

    def _source_alibi(self) -> np.ndarray:
        # Use the same canonical helper as the base executor without creating
        # any model-sized tensor on a GPU.
        from aether.runtime.positional import alibi_slopes

        return alibi_slopes(self.num_heads)

    def _tensor_primary(self, value: Any) -> Any:
        return self.torch.as_tensor(
            np.asarray(value, dtype=np.float32), device=self.device, dtype=self.compute_dtype
        )

    def _optional_primary(self, value: Any | None) -> Any | None:
        return None if value is None else self._tensor_primary(value)

    def _norm_primary(self, value: Any) -> Any:
        """Materialize a normalization *scale* in its effective form.

        Gemma stores normalization weights as offsets from unity, so the scale
        actually applied is ``1 + w``.  Materializing the raw weight instead
        collapses the residual stream — and because it only affects this
        executor, it would appear exclusively on a multi-device host.  Biases
        are additive and must not receive the offset.
        """
        array = np.asarray(value, dtype=np.float32)
        if self.norm_offset_one:
            array = array + np.float32(1.0)
        return self.torch.as_tensor(
            array, device=self.device, dtype=self.compute_dtype
        )

    def _optional_norm_primary(self, value: Any | None) -> Any | None:
        return None if value is None else self._norm_primary(value)

    def _tensor_on(self, value: Any, device: Any) -> Any:
        return self.torch.as_tensor(
            np.asarray(value, dtype=np.float32), device=device, dtype=self.compute_dtype
        )

    def _synchronize(self, device: Any) -> None:
        if device.type == "cuda":
            self.torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(self.torch, "mps"):
            self.torch.mps.synchronize()

    def _calibrate_mixed_mesh(self) -> list[float] | None:
        """Return physical-memory capacities or measure mixed-device throughput.

        GPU model parallelism is capacity constrained: a 36 GiB card and a
        64 GiB card cannot safely receive equal weight ranges.  Total device
        memory therefore determines the default partition for homogeneous
        accelerator meshes.  A mixed CPU/accelerator mesh has no meaningful
        common memory-performance unit, so a small measured GEMM supplies a
        conservative compute-capacity signal instead.  Both probes happen
        before model shards are uploaded.
        """
        torch = self.torch
        if all(device.type == "cuda" for device in self.devices):
            capacities: list[float] = []
            try:
                for device in self.devices:
                    properties = torch.cuda.get_device_properties(device)
                    total_memory = float(getattr(properties, "total_memory", 0))
                    if total_memory <= 0.0:
                        return None
                    capacities.append(total_memory)
                return capacities
            except Exception:  # noqa: BLE001 - equal partition is safe fallback
                return None
        if len({device.type for device in self.devices}) <= 1:
            return None
        rows = 256
        columns = 512
        weights: list[float] = []
        try:
            with torch.inference_mode():
                for device in self.devices:
                    left = torch.randn((rows, columns), device=device, dtype=torch.float32)
                    right = torch.randn((columns, columns), device=device, dtype=torch.float32)
                    torch.mm(left, right)
                    self._synchronize(device)
                    samples: list[float] = []
                    for _ in range(2):
                        started = time.perf_counter()
                        torch.mm(left, right)
                        self._synchronize(device)
                        samples.append(time.perf_counter() - started)
                    weights.append(1.0 / max(float(np.median(samples)), 1e-9))
        except Exception as exc:  # noqa: BLE001 - calibration is an optimization
            # A failed probe must not make a valid mesh unusable. Equal
            # partitions remain lossless and are safer than an unmeasured
            # vendor-specific estimate.
            del exc
            return None
        return weights

    def _partition_ranges(self, length: int) -> list[tuple[int, int]]:
        if self._shard_weights is None:
            return _boundaries(length, len(self.devices))
        return capacity_weighted_partition(length, self._shard_weights)

    def _canonical_projection(self, value: Any, input_features: int, name: str) -> np.ndarray:
        """Return a projection in ``(out_features, in_features)`` layout.

        AEG ingestion preserves enough source layout information for the
        single-device engines to accept both PyTorch ``Linear`` matrices and
        source-checkpoint ``Conv1D`` matrices. Tensor parallelism must normalize the
        matrix before partitioning: row sharding always means output-feature
        sharding and column sharding always means input-feature sharding.
        This is shape-driven, so it is independent of a model-family name.
        """
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"{name} weight must be rank-2, got shape {array.shape}")
        if int(array.shape[1]) == int(input_features):
            return np.ascontiguousarray(array)
        if int(array.shape[0]) == int(input_features):
            return np.ascontiguousarray(array.T)
        raise ValueError(
            f"{name} weight/input mismatch: input has {input_features} features "
            f"but weight shape is {array.shape}"
        )

    def _row_shards(
        self, value: Any, input_features: int | None = None, name: str = "projection"
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        array = (
            self._canonical_projection(value, input_features, name)
            if input_features is not None else np.asarray(value)
        )
        ranges = self._partition_ranges(int(array.shape[0]))
        return [self._tensor_on(array[start:end], device) for device, (start, end) in zip(self.devices, ranges)], ranges

    def _column_shards(
        self, value: Any, input_features: int | None = None, name: str = "projection"
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        array = (
            self._canonical_projection(value, input_features, name)
            if input_features is not None else np.asarray(value)
        )
        ranges = self._partition_ranges(int(array.shape[1]))
        return [self._tensor_on(array[:, start:end], device) for device, (start, end) in zip(self.devices, ranges)], ranges

    def _optional_row_shards(self, value: Any | None) -> tuple[list[Any] | None, list[tuple[int, int]] | None]:
        if value is None:
            return None, None
        return self._row_shards(value)

    def _row_bias_shards(self, value: Any | None) -> list[Any] | None:
        if value is None:
            return None
        array = np.asarray(value)
        ranges = self._partition_ranges(int(array.shape[0]))
        return [self._tensor_on(array[start:end], device) for device, (start, end) in zip(self.devices, ranges)]

    def _convert_sharded_layer(self, layer: Any) -> dict[str, Any]:
        # Normalization weights are small and are replicated on the primary
        # device; only the projections are sharded.  The sandwich/post-norm
        # slots must be carried too, or a Gemma-2/3, OLMo-2 or EXAONE-4 model
        # would silently execute as a plain pre-norm block on a multi-GPU mesh.
        norm_scale_names = (
            "attention_norm", "q_norm", "k_norm", "ffn_norm",
            "post_attention_norm", "post_ffn_norm",
        )
        primary_names = (
            "attention_norm_bias", "ffn_norm_bias", "o_proj_bias", "down_proj_bias",
            "post_attention_norm_bias", "post_ffn_norm_bias",
        )
        result: dict[str, Any] = {
            name: self._optional_norm_primary(getattr(layer, name, None))
            for name in norm_scale_names
        }
        result.update(
            {name: self._optional_primary(getattr(layer, name, None)) for name in primary_names}
        )
        result["q_proj"], result["q_ranges"] = self._row_shards(
            layer.q_proj, self.embedding_hidden_size, "q_proj"
        )
        result["k_proj"], result["k_ranges"] = self._row_shards(
            layer.k_proj, self.embedding_hidden_size, "k_proj"
        )
        result["v_proj"], result["v_ranges"] = self._row_shards(
            layer.v_proj, self.embedding_hidden_size, "v_proj"
        )
        result["o_proj"], result["o_ranges"] = self._column_shards(
            layer.o_proj, self.num_heads * self.head_dim, "o_proj"
        )
        result["q_proj_bias"] = self._row_bias_shards(getattr(layer, "q_proj_bias", None))
        result["k_proj_bias"] = self._row_bias_shards(getattr(layer, "k_proj_bias", None))
        result["v_proj_bias"] = self._row_bias_shards(getattr(layer, "v_proj_bias", None))
        gate_features = None
        if layer.gate_proj is None:
            result["gate_proj"], result["gate_ranges"] = None, None
        else:
            gate_matrix = self._canonical_projection(
                layer.gate_proj, self.embedding_hidden_size, "gate_proj"
            )
            gate_features = int(gate_matrix.shape[0])
            result["gate_proj"], result["gate_ranges"] = self._row_shards(
                gate_matrix, self.embedding_hidden_size, "gate_proj"
            )
        result["up_proj"] = (
            None if layer.up_proj is None else
            self._row_shards(layer.up_proj, self.embedding_hidden_size, "up_proj")[0]
        )
        result["up_ranges"] = (
            None if layer.up_proj is None else
            self._partition_ranges(int(self._canonical_projection(
                layer.up_proj, self.embedding_hidden_size, "up_proj"
            ).shape[0]))
        )
        if layer.down_proj is None:
            result["down_proj"], result["down_ranges"] = None, None
        else:
            if gate_features is None:
                raise ValueError("down_proj requires a gate/up projection to define its input width")
            result["down_proj"], result["down_ranges"] = self._column_shards(
                layer.down_proj, gate_features,
                "down_proj",
            )
        result["gate_proj_bias"] = self._row_bias_shards(getattr(layer, "gate_proj_bias", None))
        result["up_proj_bias"] = self._row_bias_shards(getattr(layer, "up_proj_bias", None))

        router = getattr(layer, "router", None)
        result["router"], result["router_ranges"] = (
            self._row_shards(router, self.embedding_hidden_size, "router")
            if router is not None else (None, None)
        )
        result["experts"] = []
        for expert in (getattr(layer, "experts", None) or []):
            gate_matrix = self._canonical_projection(
                expert.gate_proj, self.embedding_hidden_size, "expert.gate_proj"
            )
            gate, gate_ranges = self._row_shards(
                gate_matrix, self.embedding_hidden_size, "expert.gate_proj"
            )
            up = (
                None if expert.up_proj is None else
                self._row_shards(expert.up_proj, self.embedding_hidden_size, "expert.up_proj")[0]
            )
            down, _ = self._column_shards(
                expert.down_proj, int(gate_matrix.shape[0]), "expert.down_proj"
            )
            result["experts"].append({
                "gate_proj": gate, "up_proj": up, "down_proj": down,
                "gate_proj_bias": self._row_bias_shards(getattr(expert, "gate_proj_bias", None)),
                "up_proj_bias": self._row_bias_shards(getattr(expert, "up_proj_bias", None)),
                "down_proj_bias": self._optional_primary(getattr(expert, "down_proj_bias", None)),
                "gate_ranges": gate_ranges,
            })
        result["num_activated_experts"] = int(getattr(layer, "num_activated_experts", 1) or 1)
        return result

    def _gather(self, values: list[Any]) -> Any:
        return self.torch.cat([value.to(self.device) for value in values], dim=-1)

    def _embedding_lookup(self, ids: Any) -> Any:
        """Gather rows of a vocabulary-sharded embedding table.

        Rank-generic: ``ids`` may be ``(seq,)`` or ``(batch, seq)``.  The gather
        itself is per token and carries no positional meaning, so flattening the
        leading axes is lossless here — each id is looked up independently, and the
        result is restored to the caller's shape.  Contrast attention, where the
        same flatten would splice independent sequences together.
        """
        lead = tuple(ids.shape)
        flat = ids.reshape(-1)
        output = self.torch.empty(
            (int(flat.numel()), self.embedding_hidden_size),
            device=self.device,
            dtype=self.embedding[0].dtype,
        )
        for shard, (start, end), device in zip(self.embedding, self._embedding_ranges, self.devices):
            mask = (flat >= start) & (flat < end)
            positions = self.torch.where(mask)[0]
            if positions.numel():
                local_ids = flat.index_select(0, positions).to(device) - start
                output.index_copy_(0, positions, shard.index_select(0, local_ids).to(self.device))
        return output.reshape(*lead, self.embedding_hidden_size)

    def _position_lookup(self, positions: Any) -> Any:
        """Gather rows of a position-sharded learned table.

        Rank-generic for the same reason as :meth:`_embedding_lookup`.  In a batch
        the positions are per row — a padded row's real tokens still start at 0 —
        so this must be driven by the layout's positions, never by padded indices.
        """
        assert self.position_embedding is not None and self._position_ranges is not None
        lead = tuple(positions.shape)
        flat = positions.reshape(-1)
        width = int(self.position_hidden_size)
        output = self.torch.empty(
            (int(flat.numel()), width),
            device=self.device,
            dtype=self.position_embedding[0].dtype,
        )
        for shard, (start, end), device in zip(self.position_embedding, self._position_ranges, self.devices):
            mask = (flat >= start) & (flat < end)
            indexes = self.torch.where(mask)[0]
            if indexes.numel():
                local = flat.index_select(0, indexes).to(device) - start
                output.index_copy_(0, indexes, shard.index_select(0, local).to(self.device))
        return output.reshape(*lead, width)

    def _parallel_row_linear(self, x: Any, weights: list[Any], biases: list[Any] | None) -> Any:
        values = []
        for index, weight in enumerate(weights):
            local_x = x.to(self.devices[index])
            bias = None if biases is None else biases[index]
            values.append(self._linear(local_x, weight, bias))
        return self._gather(values)

    def _parallel_column_linear(self, x: Any, weights: list[Any], bias: Any | None) -> Any:
        result = self.torch.zeros((*x.shape[:-1], int(weights[0].shape[0])), device=self.device, dtype=x.dtype)
        for index, weight in enumerate(weights):
            start = sum(int(item.shape[1]) for item in weights[:index])
            local = self._linear(x[..., start:start + int(weight.shape[1])].to(self.devices[index]), weight)
            result = result + local.to(self.device)
        return result if bias is None else result + bias

    def _moe_ffn(self, hidden: Any, layer: dict[str, Any]) -> Any:
        if layer["router"] is None or not layer["experts"]:
            raise ValueError("portable MoE layer is missing its router or experts")
        # Expert routing is strictly per token: a token's expert choice and output
        # depend on nothing but that token.  Collapsing a batch's leading axes into
        # one token axis is therefore lossless, and it keeps the scatter/gather
        # dispatch rank-2 for every batch width.  The same flatten applied to
        # attention would splice independent sequences together; that is why it is
        # confined to this one operation.
        lead = tuple(hidden.shape[:-1])
        flat = hidden.reshape(-1, int(hidden.shape[-1])) if len(lead) > 1 else hidden
        router_logits = self._parallel_row_linear(flat, layer["router"], None)
        top_k = min(layer["num_activated_experts"], len(layer["experts"]))
        # Probabilities come from a softmax over the complete expert set; only
        # architectures declaring ``norm_topk_prob`` renormalize the selected k.
        probabilities = self.torch.softmax(router_logits.float(), dim=-1)
        routing, selected = self.torch.topk(probabilities, top_k, dim=-1)
        if self.moe_renormalize_topk:
            routing = routing / routing.sum(dim=-1, keepdim=True)
        routing = routing.to(dtype=flat.dtype)
        output = self.torch.zeros_like(flat)
        for expert_index, expert in enumerate(layer["experts"]):
            rows, slots = self.torch.where(selected == expert_index)
            if not rows.numel():
                continue
            source = flat.index_select(0, rows)
            gate = self._parallel_row_linear(source, expert["gate_proj"], expert["gate_proj_bias"])
            up = None if expert["up_proj"] is None else self._parallel_row_linear(source, expert["up_proj"], expert["up_proj_bias"])
            activated = self._activation(gate, up)
            value = self._parallel_column_linear(activated, expert["down_proj"], expert["down_proj_bias"])
            output.index_add_(0, rows, value * routing[rows, slots].unsqueeze(-1))
        return output.reshape(*lead, int(output.shape[-1])) if len(lead) > 1 else output

    def _forward_device(
        self,
        token_ids: np.ndarray | Any,
        cache: TorchKVCache | BatchedKVCache | None = None,
        *,
        validate_ids: bool = False,
        reserve: int = 0,
        batched: bool = False,
        logits: str = "all",
    ) -> tuple[Any, TorchKVCache | BatchedKVCache]:
        """Run one sharded step, for a single sequence or for a batch.

        The signature must stay compatible with
        :meth:`TorchAEGEngine._forward_device`: the inherited generation loop
        calls it directly, so a missing keyword here is a runtime failure that
        only appears on a multi-device host.

        Batching and sharding are orthogonal, and this method is where that becomes
        true.  Sharding splits each weight matrix across devices; batching adds a
        leading axis to the activations.  Neither touches the other, because every
        collective here reduces or concatenates over the *feature* axis — the
        sharded dimension — and never over the batch or sequence axes:

        * ``_parallel_row_linear`` splits output features, so it concatenates the
          shards' results along the last axis. Rank-agnostic.
        * ``_parallel_column_linear`` splits input features, so each device
          contributes a partial sum over its slice and the results are added. This
          is the all-reduce of Megatron-LM §3.3 (Shoeybi et al. 2019,
          arXiv:1909.08053), and a sum over feature slices commutes with any
          leading batch axis.

        So the batched sharded pass is the single-device batched pass with the two
        linear primitives swapped, and the KV cache stays local to the device that
        owns those heads.  The five rank-dependent sites branch on ``batched``; the
        architecture-variant handling is shared with the single-sequence path rather
        than duplicated, for the same reason it is shared in the base class.
        """
        torch = self.torch
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids if batched else token_ids.reshape(-1)
            if ids.device != self.device or ids.dtype != torch.long:
                ids = ids.to(device=self.device, dtype=torch.long)
        else:
            array = np.asarray(token_ids, dtype=np.int64)
            ids = torch.as_tensor(
                array if batched else array.reshape(-1), device=self.device
            )
        if batched and ids.dim() != 2:
            raise ValueError(
                "a batched forward pass requires rank-2 (batch, seq) ids, got rank "
                f"{ids.dim()}"
            )
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if validate_ids and (int(ids.min()) < 0 or int(ids.max()) >= self.embedding_vocab_size):
            raise ValueError("token id is outside the compiled vocabulary")
        if cache is None:
            cache = (
                self._new_batched_cache(batch_size=int(ids.shape[0]), reserve=reserve)
                if batched
                else TorchKVCache([None] * self.num_layers, [None] * self.num_layers)
            )
        past = int(cache.length)
        lead = tuple(ids.shape)
        seq_len = int(ids.shape[-1])
        total = past + seq_len
        span = torch.arange(past, total, device=self.device, dtype=torch.long)
        if batched:
            # A row's position is its padded index minus its own pad count, so a
            # short row is not rotated as though it began mid-sequence.
            positions = (span.unsqueeze(0) - cache.pad_counts).clamp_(min=0)
            key_positions = (
                self._key_positions(total).unsqueeze(0) - cache.pad_counts
            ).clamp_(min=0)
            live_view = cache.live_view(total)
        else:
            positions = span
            key_positions = self._key_positions(total)
            live_view = None
        uses_rope = self.uses_rope
        if uses_rope:
            self._ensure_rope(total)
        # Rotation factors depend only on the positions and the rotary base, so
        # gather them once per step rather than twice per layer.
        rope_global = self._rope_slice(positions, local=False) if uses_rope else None
        rope_local = (
            self._rope_slice(positions, local=True)
            if uses_rope and self._local_cos is not None
            else rope_global
        )
        post_norm = self.norm_placement == "post"
        parallel = self.parallel_residual
        residual_scale = self.residual_scale
        hidden = self._embedding_lookup(ids)
        if self.embedding_scale is not None:
            hidden = hidden * self.embedding_scale
        if self.embedding_norm is not None:
            hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
        if self.position_embedding is not None:
            hidden = hidden + self._position_lookup(positions)

        with torch.inference_mode():
            for index, layer in enumerate(self.layers):
                local_attention, attention_window, attention_scale, layer_uses_rope = (
                    self.layer_plan[index]
                )
                layer_uses_rope = layer_uses_rope and uses_rope
                rope_factors = rope_local if local_attention else rope_global
                block_input = hidden
                if post_norm:
                    # OLMo-2 feeds the raw residual into each sublayer and
                    # normalizes the sublayer output instead.
                    normed = hidden
                else:
                    normed = self._norm(
                        hidden, layer["attention_norm"], layer["attention_norm_bias"]
                    )
                q = self._parallel_row_linear(normed, layer["q_proj"], layer["q_proj_bias"])
                k = self._parallel_row_linear(normed, layer["k_proj"], layer["k_proj_bias"])
                if self.qk_norm_is_full:
                    if layer["q_norm"] is not None:
                        q = self._norm(q, layer["q_norm"])
                    if layer["k_norm"] is not None:
                        k = self._norm(k, layer["k_norm"])
                q = q.reshape(*lead, self.num_heads, self.head_dim)
                k = k.reshape(*lead, self.num_kv_heads, self.head_dim)
                if not self.qk_norm_is_full:
                    if layer["q_norm"] is not None:
                        q = self._norm(q, layer["q_norm"])
                    if layer["k_norm"] is not None:
                        k = self._norm(k, layer["k_norm"])
                if layer_uses_rope:
                    q = self._rope(q, *rope_factors)
                    k = self._rope(k, *rope_factors)
                v = self._parallel_row_linear(
                    normed, layer["v_proj"], layer["v_proj_bias"]
                ).reshape(*lead, self.num_kv_heads, self.head_dim)
                k_all = self._append_kv(cache.keys[index], k, past, total, reserve)
                v_all = self._append_kv(cache.values[index], v, past, total, reserve)
                context = self._attention(
                    q,
                    k_all[:, :total] if batched else k_all[:total],
                    v_all[:, :total] if batched else v_all[:total],
                    positions,
                    key_positions,
                    attention_window,
                    attention_scale,
                    live=live_view,
                )
                cache.keys[index] = k_all
                cache.values[index] = v_all
                attention_out = self._parallel_column_linear(
                    context.reshape(*lead, self.num_heads * self.head_dim),
                    layer["o_proj"],
                    layer["o_proj_bias"],
                )
                # Sandwich blocks normalize the sublayer output as well
                # (Gemma-2/3, EXAONE-4); post-norm blocks normalize it instead
                # of the input (OLMo-2).  Both precede the residual add.
                if layer["post_attention_norm"] is not None:
                    attention_out = self._norm(
                        attention_out,
                        layer["post_attention_norm"],
                        layer["post_attention_norm_bias"],
                    )
                elif post_norm:
                    attention_out = self._norm(
                        attention_out, layer["attention_norm"], layer["attention_norm_bias"]
                    )
                if residual_scale is not None:
                    attention_out = attention_out * residual_scale
                if parallel:
                    # GPT-J, GPT-NeoX, Falcon and Cohere evaluate the
                    # feed-forward branch from the block input.
                    normed = self._norm(
                        block_input, layer["ffn_norm"], layer["ffn_norm_bias"]
                    )
                elif post_norm:
                    hidden = block_input + attention_out
                    normed = hidden
                else:
                    hidden = hidden + attention_out
                    normed = self._norm(hidden, layer["ffn_norm"], layer["ffn_norm_bias"])
                if layer["experts"]:
                    ffn_out = self._moe_ffn(normed, layer)
                else:
                    gate = self._parallel_row_linear(normed, layer["gate_proj"], layer["gate_proj_bias"])
                    up = None if layer["up_proj"] is None else self._parallel_row_linear(normed, layer["up_proj"], layer["up_proj_bias"])
                    ffn_out = self._parallel_column_linear(
                        self._activation(gate, up), layer["down_proj"], layer["down_proj_bias"]
                    )
                if layer["post_ffn_norm"] is not None:
                    ffn_out = self._norm(
                        ffn_out, layer["post_ffn_norm"], layer["post_ffn_norm_bias"]
                    )
                elif post_norm:
                    ffn_out = self._norm(ffn_out, layer["ffn_norm"], layer["ffn_norm_bias"])
                if residual_scale is not None:
                    ffn_out = ffn_out * residual_scale
                hidden = (
                    block_input + attention_out + ffn_out if parallel else hidden + ffn_out
                )
            # Honour the caller's logits request here too, for the same reason the
            # single-device executor does: the sharded vocabulary projection is the
            # widest GEMM in the pass, and generation reads only its final row.
            # Slicing before the projection is exact, since the projection is
            # per-position.
            if logits == "last":
                hidden = hidden[:, -1:] if batched else hidden[-1:]
            elif logits != "all":
                raise ValueError(f"logits mode must be 'all' or 'last', got {logits!r}")
            hidden = self._norm(hidden, self.final_norm, self.final_norm_bias)
            projected = self._parallel_row_linear(hidden, self.lm_head, None)
            if self.logit_scale is not None:
                projected = projected * self.logit_scale
            if self.final_logit_softcap:
                projected = self._softcap(projected, self.final_logit_softcap)
            cache.length = total
            cache.last_logits = (
                projected[:, -1] if batched else projected[-1]
            ).detach()
        return projected, cache

    def forward(self, token_ids: np.ndarray | Any, cache: TorchKVCache | None = None) -> tuple[np.ndarray, TorchKVCache]:
        """Run a public forward pass and materialize logits as NumPy."""
        logits, cache = self._forward_device(token_ids, cache, validate_ids=True)
        return logits.detach().float().cpu().numpy(), cache
