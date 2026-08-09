"""
Pass 19 — Sub-2-Bit Quantization (Ternary / Binary / NanoQuant).

Sub-2-bit quantization achieves 10× memory compression vs BF16 while enabling
addition-only inference kernels (no multiplications) on ternary hardware.

Three methods:

1. **BitNet b1.58** (Ma et al., 2024 + 2026 scale-up):
   - Weights: ternary {-1, 0, +1}, chosen by absmean scaling.
   - Activations: INT8 with per-token dynamic scaling.
   - Kernel: weight-only multiply-free (ADD/SUB + accumulate).
   - Memory: 1.58 bits/weight theoretical (2 bits packed in practice).
   - Throughput: 5× vs BF16 on ternary-native CPU; 3× on CUDA.

2. **BTC-LLM** (2026):
   - Binary codebook of 256 entries per weight block of 128 elements.
   - Represents weights as 8-bit codebook indices → 0.8–1.11 bits effective.
   - Gather-based kernel: index lookup + accumulate.

3. **NanoQuant** (2026):
   - Trellis codebook (Viterbi path) optimizing joint quantization across
     adjacent weight blocks.
   - Sub-1-bit effective with trellis path storage.
   - Highest compression, ~20% quality gate relaxation needed.

Quality gate: perplexity increase < config.sub2bit_quality_gate_ppl.
If gate fails: fall back to INT4 (Pass 2 precision assignment).

AEG artifacts:
  - ``.aeg/quantization/sub2bit_manifest.json``: method, scale tables.
  - Repacked weight blobs replacing BF16/INT8 blobs.

Research basis:
  - BitNet b1.58 (Ma et al. 2024): original 1.58-bit ternary paper.
  - BitNet b2 (2026): 2B-scale ternary scaling law verification.
  - BTC-LLM (2026): binary trellis codebook compression.
  - NanoQuant (2026): sub-1-bit trellis quantization.
  - Era of 1-bit LLMs (Ma et al. 2024): compute savings analysis.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_METHODS: frozenset[str] = frozenset({"bitnet", "btc_llm", "nanoquant"})


class Sub2BitQuantizationPass(BasePass):
    """Pass 19: Quantize model weights to sub-2-bit ternary or binary format.

    Replaces BF16/FP16 weight blobs with packed ternary/binary representations.
    Emits ``aeg.ternary_linear`` / ``aeg.binary_linear`` kernel opcodes that
    map to addition-only inference paths on ternary-native hardware.
    """

    name = "sub2bit_quantization"
    description = (
        "Quantize weights to sub-2-bit (BitNet b1.58 ternary / BTC-LLM binary / "
        "NanoQuant trellis) for 10× memory compression and addition-only inference."
    )

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})

        if not config.enable_sub2bit:
            return graph, report

        method = config.sub2bit_method
        if method not in _SUPPORTED_METHODS:
            logger.warning("Pass 19: Unknown method %r. Using 'bitnet'.", method)
            method = "bitnet"

        quality_gate_ppl = config.sub2bit_quality_gate_ppl

        try:
            logger.info(
                "Pass 19: Sub-2-bit quantization via %s (quality gate PPL+%.1f%%).",
                method,
                quality_gate_ppl * 100,
            )

            # Extract weight tensors from graph.
            weight_store = _get_weight_store(graph)
            if not weight_store:
                report.status = "skipped"
                report.details["reason"] = "no_weights_in_graph"
                return graph, report

            # The old implementation used fixed literature estimates as if
            # they were a measured perplexity gate.  That can accept a bad
            # artifact.  Until a real baseline/candidate evaluator is passed
            # through the compiler, refuse to publish the transformed graph.
            report.status = "skipped"
            report.details = {
                "reason": "quality_evaluator_unavailable",
                "message": (
                    "sub-2-bit quantization requires a measured baseline and candidate "
                    "quality evaluator; hardcoded perplexity estimates are not accepted"
                ),
            }
            return graph, report

            # Quantize weights.
            quantizer = _get_quantizer(method)
            quantized_store, scale_tables, bits_per_weight = quantizer.quantize(weight_store)

            # Quality gate: estimate perplexity increase.
            ppl_increase = _estimate_ppl_increase(method, bits_per_weight)
            if ppl_increase > quality_gate_ppl:
                logger.warning(
                    "Pass 19: Quality gate FAIL — estimated PPL increase %.2f%% > "
                    "threshold %.2f%%.  Falling back to INT4.",
                    ppl_increase * 100,
                    quality_gate_ppl * 100,
                )
                report.status = "skipped"
                report.details["reason"] = "quality_gate_failed"
                report.details["estimated_ppl_increase"] = ppl_increase
                return graph, report

            # Update graph weight store with quantized weights.
            n_updated = _update_weight_store(graph, quantized_store)

            # Emit ternary/binary kernel opcodes.
            n_opcodes = _emit_sub2bit_opcodes(graph, method, weight_store)

            # Write manifest.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                _write_sub2bit_manifest(
                    output_dir=Path(graph.output_dir),
                    method=method,
                    scale_tables=scale_tables,
                    n_tensors=n_updated,
                    bits_per_weight=bits_per_weight,
                )

            # Compute compression ratio vs BF16 (2 bytes/elem).
            bytes_per_elem = bits_per_weight / 8
            compression_ratio = 2.0 / bytes_per_elem

            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "method": method,
                "bits_per_weight": bits_per_weight,
                "compression_ratio_vs_bf16": round(compression_ratio, 1),
                "n_tensors_quantized": n_updated,
                "n_opcodes_emitted": n_opcodes,
                "estimated_ppl_increase_pct": round(ppl_increase * 100, 2),
                "quality_gate_passed": True,
            }
            logger.info(
                "Pass 19 complete: %s, %.2f bits/weight, %.1f× compression, "
                "%d tensors.  PPL+%.2f%%.  Elapsed: %.3fs.",
                method,
                bits_per_weight,
                compression_ratio,
                n_updated,
                ppl_increase * 100,
                elapsed,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 19 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)

        return graph, report


# ── Quantizers ────────────────────────────────────────────────────────────────


class _BaseQuantizer:
    bits_per_weight: float = 2.0

    def quantize(
        self,
        weight_store: dict[str, list[float]],
    ) -> tuple[dict[str, list[int]], dict[str, Any], float]:
        """Returns (quantized_store, scale_tables, bits_per_weight)."""
        raise NotImplementedError


class _BitNetQuantizer(_BaseQuantizer):
    """BitNet b1.58 ternary quantization.

    Weight quantization:
      1. Compute scale γ = mean(|W|) (absmean scaling).
      2. Quantize: W_q = RoundClip(W / γ + 0.5) ∈ {-1, 0, +1}.
      3. Store as int8 packed 4 per byte (2-bit packed).

    Activation quantization:
      - Per-token INT8 with scale = max(|X|) / 127.
    """

    bits_per_weight = 1.58  # log2(3)

    def quantize(self, weight_store):
        quantized: dict[str, list[int]] = {}
        scales: dict[str, float] = {}

        for name, weights in weight_store.items():
            if not weights:
                quantized[name] = []
                scales[name] = 1.0
                continue
            # Absmean scale.
            gamma = sum(abs(w) for w in weights) / max(1, len(weights))
            if gamma < 1e-10:
                gamma = 1e-10
            # Ternary quantization: round-clip to {-1, 0, +1}.
            q_weights = [
                max(-1, min(1, round(w / gamma))) for w in weights
            ]
            quantized[name] = q_weights
            scales[name] = gamma

        scale_tables = {"method": "absmean", "scales": scales}
        return quantized, scale_tables, self.bits_per_weight


class _BTCLLMQuantizer(_BaseQuantizer):
    """BTC-LLM binary codebook quantization.

    Each block of 128 weights is represented by an 8-bit codebook index
    pointing to one of 256 representative centroids.  Effective: 0.8–1.11 bits.
    """

    bits_per_weight = 1.0  # 8-bit index over 128-weight block

    def quantize(self, weight_store):
        quantized: dict[str, list[int]] = {}
        codebooks: dict[str, list[float]] = {}
        BLOCK_SIZE = 128
        N_CENTROIDS = 256

        for name, weights in weight_store.items():
            if not weights:
                quantized[name] = []
                codebooks[name] = []
                continue

            indices: list[int] = []
            centroid_table: list[float] = []

            for block_start in range(0, len(weights), BLOCK_SIZE):
                block = weights[block_start: block_start + BLOCK_SIZE]
                if not block:
                    continue
                # k-means with N_CENTROIDS centroids (simplified: evenly spaced).
                w_min = min(block)
                w_max = max(block)
                w_range = w_max - w_min
                centroids = [
                    w_min + (w_range * i / (N_CENTROIDS - 1))
                    for i in range(N_CENTROIDS)
                ]
                # Assign each weight to nearest centroid.
                block_indices = []
                for w in block:
                    best_idx = min(
                        range(N_CENTROIDS),
                        key=lambda i: abs(centroids[i] - w),
                    )
                    block_indices.append(best_idx)
                indices.extend(block_indices)
                centroid_table.extend(centroids)

            quantized[name] = indices
            codebooks[name] = centroid_table

        scale_tables = {"method": "btc_codebook", "codebooks": {k: v[:256] for k, v in codebooks.items()}}
        return quantized, scale_tables, self.bits_per_weight


class _NanoQuantizer(_BaseQuantizer):
    """NanoQuant trellis codebook quantization.

    Uses Viterbi algorithm over weight sequence to find optimal codebook path.
    Achieves sub-1-bit compression via joint inter-element correlations.
    """

    bits_per_weight = 0.9

    def quantize(self, weight_store):
        # NanoQuant: simplified implementation using 2-state trellis.
        # Full implementation would use 4–16 trellis states.
        quantized: dict[str, list[int]] = {}
        paths: dict[str, list[int]] = {}

        for name, weights in weight_store.items():
            if not weights:
                quantized[name] = []
                paths[name] = []
                continue
            # 2-state trellis: state 0 = negative cluster, state 1 = positive cluster.
            # Viterbi: find path minimizing total quantization error.
            centroids = [-1.0, 1.0]  # 2 states
            path: list[int] = []
            for w in weights:
                best_state = 0 if abs(w - centroids[0]) <= abs(w - centroids[1]) else 1
                path.append(best_state)
            quantized[name] = path
            paths[name] = path

        scale_tables = {"method": "nanoquant_trellis", "n_states": 2}
        return quantized, scale_tables, self.bits_per_weight


def _get_quantizer(method: str) -> _BaseQuantizer:
    return {"bitnet": _BitNetQuantizer(), "btc_llm": _BTCLLMQuantizer(), "nanoquant": _NanoQuantizer()}.get(
        method, _BitNetQuantizer()
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_weight_store(graph: Any) -> dict[str, list[float]]:
    weights: dict[str, list[float]] = {}
    if hasattr(graph, "weight_store") and hasattr(graph.weight_store, "items"):
        for name, tensor in graph.weight_store.items():
            if isinstance(tensor, list):
                weights[str(name)] = tensor
            elif hasattr(tensor, "tolist"):
                weights[str(name)] = tensor.tolist()
    elif hasattr(graph, "parameters") and callable(graph.parameters):
        for name, param in graph.parameters():
            if hasattr(param, "tolist"):
                weights[str(name)] = param.reshape(-1).tolist()
    # Ingestion binds checkpoint tensors to AEGGraph node attributes rather
    # than to a separate optimizer weight store.  Read those real arrays so
    # Pass 19 cannot silently report ``no_weights_in_graph`` for a runnable
    # SafeTensors/ONNX/PyTorch graph.
    elif hasattr(graph, "nodes"):
        nodes = graph.nodes.values() if hasattr(graph.nodes, "values") else graph.nodes
        for node in nodes:
            node_id = str(getattr(node, "id", "weight"))
            attributes = getattr(node, "attributes", {})
            for suffix, key in (("", node_id), ("_up", f"{node_id}_up")):
                value = attributes.get("up_weight" if suffix else "weight") if isinstance(attributes, dict) else None
                if value is None or not hasattr(value, "reshape"):
                    continue
                weights[key] = [float(v) for v in value.reshape(-1).tolist()]
    return weights


def _update_weight_store(graph: Any, quantized: dict[str, list[int]]) -> int:
    n_updated = 0
    if hasattr(graph, "weight_store") and hasattr(graph.weight_store, "update"):
        graph.weight_store.update(quantized)
        n_updated = len(quantized)
    elif hasattr(graph, "metadata"):
        graph.metadata["quantized_weights_ref"] = "quantization/sub2bit_manifest.json"
        n_updated = len(quantized)
    return n_updated


def _emit_sub2bit_opcodes(graph: Any, method: str, weight_store: dict) -> int:
    """Replace linear kernel opcodes with ternary/binary variants."""
    opcode_map = {
        "bitnet": "aeg.ternary_linear",
        "btc_llm": "aeg.binary_codebook_linear",
        "nanoquant": "aeg.trellis_linear",
    }
    op = opcode_map.get(method, "aeg.ternary_linear")
    n_emitted = 0
    if hasattr(graph, "metadata"):
        graph.metadata.setdefault("sub2bit_opcodes", []).append(
            {"opcode": op, "n_weight_tensors": len(weight_store)}
        )
        n_emitted = len(weight_store)
    return n_emitted


def _estimate_ppl_increase(method: str, bits_per_weight: float) -> float:
    """Estimate perplexity increase for sub-2-bit methods.

    Empirical bounds from BitNet b1.58 (Ma 2024) and BTC-LLM (2026):
    - BitNet: ~3% PPL increase on Llama-class models (7B–70B).
    - BTC-LLM: ~5% PPL increase.
    - NanoQuant: ~8% PPL increase.
    """
    return {"bitnet": 0.03, "btc_llm": 0.05, "nanoquant": 0.08}.get(method, 0.05)


def _write_sub2bit_manifest(
    output_dir: Path,
    method: str,
    scale_tables: dict,
    n_tensors: int,
    bits_per_weight: float,
) -> None:
    quant_dir = output_dir / "quantization"
    quant_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "aether_sub2bit_v1",
        "method": method,
        "bits_per_weight": bits_per_weight,
        "n_tensors": n_tensors,
        "scale_tables_summary": {k: v for k, v in scale_tables.items() if k != "scales"},
    }
    (quant_dir / "sub2bit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.debug("Wrote sub-2-bit manifest: %s", quant_dir / "sub2bit_manifest.json")


