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
from types import SimpleNamespace
from typing import Any

import numpy as np

from aether.parallelism.sharding import balanced_partition, capacity_weighted_partition
from aether.runtime.torch_engine import TorchAEGEngine, TorchKVCache


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
        self.devices = [torch.device(value) for value in devices]
        self.device = self.devices[0]
        self._shard_weights = self._calibrate_mixed_mesh()

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
            import logging as _logging
            _logging.getLogger(__name__).info(
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
        )
        self.num_heads = int(cpu_engine.num_heads)
        self.num_kv_heads = int(cpu_engine.num_kv_heads)
        self.head_dim = int(cpu_engine.head_dim)
        self.num_layers = len(source_weights.layers)
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
        if cpu_engine.sparse_attention_plan or cpu_engine.semantic_kv_plan or cpu_engine.cross_layer_kv_plan:
            raise ValueError("tensor-parallel execution does not yet implement persisted sparse/KV alias plans")

        self.embedding, self._embedding_ranges = self._row_shards(embedding)
        self.final_norm = self._tensor_primary(source_weights.final_norm)
        self.lm_head, self._lm_head_ranges = self._row_shards(source_weights.lm_head)
        self.embedding_norm = self._optional_primary(source_weights.embedding_norm)
        self.embedding_norm_bias = self._optional_primary(source_weights.embedding_norm_bias)
        self.position_embedding, self._position_ranges = self._optional_row_shards(position_embedding)
        self.layers = [self._convert_sharded_layer(layer) for layer in source_weights.layers]
        self._cos: Any | None = None
        self._sin: Any | None = None

    def _source_alibi(self) -> np.ndarray:
        # Use the same canonical helper as the base executor without creating
        # any model-sized tensor on a GPU.
        from aether.runtime.positional import alibi_slopes

        return alibi_slopes(self.num_heads)

    def _tensor_primary(self, value: Any) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)

    def _optional_primary(self, value: Any | None) -> Any | None:
        return None if value is None else self._tensor_primary(value)

    def _tensor_on(self, value: Any, device: Any) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)

    def _synchronize(self, device: Any) -> None:
        if device.type == "cuda":
            self.torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(self.torch, "mps"):
            self.torch.mps.synchronize()

    def _calibrate_mixed_mesh(self) -> list[float] | None:
        """Measure relative GEMM throughput only for heterogeneous meshes.

        Equal GPU meshes intentionally use equal tensor partitions.  A CPU
        plus accelerator mesh has no portable peak-throughput constant, so a
        short measured GEMM is used as the capacity signal instead of
        inventing cross-vendor FLOP conversion factors.  The calibration is
        model-size independent and is run before any model shard is uploaded.
        """
        if len({device.type for device in self.devices}) <= 1:
            return None
        torch = self.torch
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

    def _row_shards(self, value: Any) -> tuple[list[Any], list[tuple[int, int]]]:
        array = np.asarray(value)
        ranges = self._partition_ranges(int(array.shape[0]))
        return [self._tensor_on(array[start:end], device) for device, (start, end) in zip(self.devices, ranges)], ranges

    def _column_shards(self, value: Any) -> tuple[list[Any], list[tuple[int, int]]]:
        array = np.asarray(value)
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
        primary_names = ("attention_norm", "attention_norm_bias", "q_norm", "k_norm", "ffn_norm", "ffn_norm_bias", "o_proj_bias", "down_proj_bias")
        result: dict[str, Any] = {
            name: self._optional_primary(getattr(layer, name, None)) for name in primary_names
        }
        result["q_proj"], result["q_ranges"] = self._row_shards(layer.q_proj)
        result["k_proj"], result["k_ranges"] = self._row_shards(layer.k_proj)
        result["v_proj"], result["v_ranges"] = self._row_shards(layer.v_proj)
        result["o_proj"], result["o_ranges"] = self._column_shards(layer.o_proj)
        result["q_proj_bias"] = self._row_bias_shards(getattr(layer, "q_proj_bias", None))
        result["k_proj_bias"] = self._row_bias_shards(getattr(layer, "k_proj_bias", None))
        result["v_proj_bias"] = self._row_bias_shards(getattr(layer, "v_proj_bias", None))
        if layer.gate_proj is None:
            result["gate_proj"], result["gate_ranges"] = None, None
        else:
            result["gate_proj"], result["gate_ranges"] = self._row_shards(layer.gate_proj)
        result["up_proj"] = None if layer.up_proj is None else self._row_shards(layer.up_proj)[0]
        result["up_ranges"] = None if layer.up_proj is None else self._partition_ranges(np.asarray(layer.up_proj).shape[0])
        if layer.down_proj is None:
            result["down_proj"], result["down_ranges"] = None, None
        else:
            result["down_proj"], result["down_ranges"] = self._column_shards(layer.down_proj)
        result["gate_proj_bias"] = self._row_bias_shards(getattr(layer, "gate_proj_bias", None))
        result["up_proj_bias"] = self._row_bias_shards(getattr(layer, "up_proj_bias", None))

        router = getattr(layer, "router", None)
        result["router"], result["router_ranges"] = (self._row_shards(router) if router is not None else (None, None))
        result["experts"] = []
        for expert in (getattr(layer, "experts", None) or []):
            gate, gate_ranges = self._row_shards(expert.gate_proj)
            up = None if expert.up_proj is None else self._row_shards(expert.up_proj)[0]
            down, _ = self._column_shards(expert.down_proj)
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
        output = self.torch.empty((int(ids.numel()), self.embedding_hidden_size), device=self.device, dtype=self.embedding[0].dtype)
        for shard, (start, end), device in zip(self.embedding, self._embedding_ranges, self.devices):
            mask = (ids >= start) & (ids < end)
            positions = self.torch.where(mask)[0]
            if positions.numel():
                local_ids = ids.index_select(0, positions).to(device) - start
                output.index_copy_(0, positions, shard.index_select(0, local_ids).to(self.device))
        return output

    def _position_lookup(self, positions: Any) -> Any:
        assert self.position_embedding is not None and self._position_ranges is not None
        output = self.torch.empty((int(positions.numel()), int(self.position_hidden_size)), device=self.device, dtype=self.position_embedding[0].dtype)
        for shard, (start, end), device in zip(self.position_embedding, self._position_ranges, self.devices):
            mask = (positions >= start) & (positions < end)
            indexes = self.torch.where(mask)[0]
            if indexes.numel():
                local = positions.index_select(0, indexes).to(device) - start
                output.index_copy_(0, indexes, shard.index_select(0, local).to(self.device))
        return output

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
        router_logits = self._parallel_row_linear(hidden, layer["router"], None)
        top_k = min(layer["num_activated_experts"], len(layer["experts"]))
        selected_logits, selected = self.torch.topk(router_logits, top_k, dim=-1)
        routing = self.torch.softmax(selected_logits, dim=-1)
        output = self.torch.zeros_like(hidden)
        for expert_index, expert in enumerate(layer["experts"]):
            rows, slots = self.torch.where(selected == expert_index)
            if not rows.numel():
                continue
            source = hidden.index_select(0, rows)
            gate = self._parallel_row_linear(source, expert["gate_proj"], expert["gate_proj_bias"])
            up = None if expert["up_proj"] is None else self._parallel_row_linear(source, expert["up_proj"], expert["up_proj_bias"])
            activated = self._activation(gate, up)
            value = self._parallel_column_linear(activated, expert["down_proj"], expert["down_proj_bias"])
            output.index_add_(0, rows, value * routing[rows, slots].unsqueeze(-1))
        return output

    def forward(self, token_ids: np.ndarray | Any, cache: TorchKVCache | None = None) -> tuple[np.ndarray, TorchKVCache]:
        torch = self.torch
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64).reshape(-1), device=self.device)
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.embedding_vocab_size:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or TorchKVCache([None] * self.num_layers, [None] * self.num_layers)
        past = int(cache.length)
        seq_len = int(ids.numel())
        positions = torch.arange(past, past + seq_len, device=self.device, dtype=torch.long)
        uses_rope = str(self.weights.position_type or "RoPE").lower() in {"rope", "rotary", "rotary_embedding"}
        if uses_rope:
            self._ensure_rope(past + seq_len)
        hidden = self._embedding_lookup(ids)
        if self.embedding_norm is not None:
            hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
        if self.position_embedding is not None:
            hidden = hidden + self._position_lookup(positions)

        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                normed = self._norm(hidden, layer["attention_norm"], layer["attention_norm_bias"])
                q = self._parallel_row_linear(normed, layer["q_proj"], layer["q_proj_bias"]).reshape(seq_len, self.num_heads, self.head_dim)
                if layer["q_norm"] is not None:
                    q = self._norm(q, layer["q_norm"])
                if uses_rope:
                    q = self._rope(q, positions)
                k = self._parallel_row_linear(normed, layer["k_proj"], layer["k_proj_bias"]).reshape(seq_len, self.num_kv_heads, self.head_dim)
                if layer["k_norm"] is not None:
                    k = self._norm(k, layer["k_norm"])
                if uses_rope:
                    k = self._rope(k, positions)
                v = self._parallel_row_linear(normed, layer["v_proj"], layer["v_proj_bias"]).reshape(seq_len, self.num_kv_heads, self.head_dim)
                if cache.keys[index] is not None:
                    k_all = torch.cat((cache.keys[index], k), dim=0)
                    v_all = torch.cat((cache.values[index], v), dim=0)
                else:
                    k_all, v_all = k, v
                context = self._attention(q, k_all, v_all, positions, torch.arange(past + seq_len, device=self.device))
                cache.keys[index] = k_all
                cache.values[index] = v_all
                attention_out = self._parallel_column_linear(
                    context.reshape(seq_len, self.num_heads * self.head_dim), layer["o_proj"], layer["o_proj_bias"]
                )
                hidden = hidden + attention_out
                normed = self._norm(hidden, layer["ffn_norm"], layer["ffn_norm_bias"])
                if layer["experts"]:
                    hidden = hidden + self._moe_ffn(normed, layer)
                else:
                    gate = self._parallel_row_linear(normed, layer["gate_proj"], layer["gate_proj_bias"])
                    up = None if layer["up_proj"] is None else self._parallel_row_linear(normed, layer["up_proj"], layer["up_proj_bias"])
                    hidden = hidden + self._parallel_column_linear(self._activation(gate, up), layer["down_proj"], layer["down_proj_bias"])
            hidden = self._norm(hidden, self.final_norm)
            logits = self._parallel_row_linear(hidden, self.lm_head, None)
            cache.length = past + seq_len
            cache.last_logits = logits[-1].detach()
        return logits.detach().float().cpu().numpy(), cache
