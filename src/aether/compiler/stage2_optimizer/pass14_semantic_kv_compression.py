"""
Pass 14 — Semantic KV Cache Compression.

KV cache is the dominant memory consumer in long-context inference.  Naive
eviction strategies (sliding window, H₂O) discard tokens without considering
semantic content.  Semantic compression retains semantically distinct KV pairs
and deduplicates near-identical ones.

Three complementary strategies:

1. **ChunkKV** (2026): Divide KV sequence into fixed-size chunks (default 16
   tokens).  Compute mean pooled key-vector per chunk.  Cluster chunks by
   cosine similarity (k-means or greedy merge).  Retain one representative KV
   per cluster.  Reduction: 40–60%.

2. **SentenceKV** (EMNLP 2025): Detect sentence boundaries using punctuation /
   special token markers.  Retain full KV for the first token of each sentence.
   Drop mid-sentence tokens beyond a retention budget.  Reduction: 30–50%.

3. **Hybrid**: ChunkKV for bulk compression + SentenceKV for boundary retention.
   Achieves best quality/compression tradeoff.

Cross-sequence deduplication via ANN (HNSW) is handled at runtime by the
Semantic Cache (R11) — this pass only plans the per-sequence compression policy.

Research basis:
  - ChunkKV (2026): chunk-level cosine clustering.
  - SentenceKV (EMNLP 2025): sentence-boundary aware KV retention.
  - SemantiCache (2026): cross-request semantic deduplication via HNSW.
  - H₂O (NeurIPS 2023): heavy-hitter KV eviction.
  - SnapKV (2024): compress on the prefill path.
  - PyramidKV (2024): per-layer variable compression rates.

AEG artifacts:
  - ``.aeg/graph/kv_compression_plan.json``: per-layer compression policy.
  - ``aeg.semantic_kv_compress(kv_block, ratio, strategy)`` opcodes.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticKVCompressionPass(BasePass):
    """Pass 14: Plan and annotate semantic KV compression for each model layer.

    This pass does NOT perform runtime compression (that happens in the KV
    cache manager).  It computes the per-layer compression policy and emits
    ``aeg.semantic_kv_compress`` opcodes into the AEG graph.

    PyramidKV insight: lower layers need higher KV retention (syntactic) while
    upper layers tolerate more compression (semantic).  We implement a pyramid
    schedule: retention_rate(layer) = base_rate + (1-base_rate) * (1 - l/L)^α
    where l is layer index, L is total layers, and α=0.5.
    """

    name = "semantic_kv_compression"
    description = (
        "Plan per-layer semantic KV compression ratios and emit "
        "aeg.semantic_kv_compress IR opcodes."
    )

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})

        if not config.enable_semantic_kv:
            return graph, report

        try:
            strategy = config.semantic_kv_strategy
            base_ratio = config.semantic_kv_compression_ratio  # fraction to RETAIN

            n_layers = _count_layers(architecture, graph)
            head_dim = _infer_head_dim(architecture)
            n_heads = _infer_n_kv_heads(architecture)

            if n_layers == 0:
                report.status = "skipped"
                report.details["reason"] = "no_layers_detected"
                return graph, report

            logger.info(
                "Pass 14: Planning %s KV compression — %d layers, base retention %.0f%%.",
                strategy,
                n_layers,
                base_ratio * 100,
            )

            # Compute per-layer retention rates using pyramid schedule.
            alpha = 0.5  # PyramidKV exponent.
            layer_plans: list[dict[str, Any]] = []
            total_saved_kv_pairs = 0.0

            for l_idx in range(n_layers):
                # Pyramid schedule: lower layers retain more KV.
                pyramid_bonus = (1.0 - base_ratio) * (1.0 - l_idx / n_layers) ** alpha
                retention = min(1.0, base_ratio + pyramid_bonus)
                drop_rate = 1.0 - retention

                # Estimate KV pairs saved (per 1000 context tokens).
                kv_pairs_per_1k = 1000 * n_heads  # approximate
                saved = kv_pairs_per_1k * drop_rate
                total_saved_kv_pairs += saved

                # Choose chunk size: smaller for upper layers (more aggressive).
                chunk_size = max(4, int(16 * retention))

                plan = {
                    "layer_index": l_idx,
                    "strategy": strategy,
                    "retention_ratio": round(retention, 4),
                    "drop_ratio": round(drop_rate, 4),
                    "chunk_size": chunk_size,
                    "n_kv_heads": n_heads,
                    "head_dim": head_dim,
                    "estimated_kv_saved_per_1k_tokens": round(saved, 1),
                }
                layer_plans.append(plan)

                # Emit opcode into graph.
                _emit_kv_compress_opcode(graph, l_idx, retention, strategy, chunk_size)

            avg_retention = sum(p["retention_ratio"] for p in layer_plans) / n_layers
            avg_drop = 1.0 - avg_retention

            # Write compression plan to AEG.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                from pathlib import Path
                _write_kv_compression_plan(
                    output_dir=Path(graph.output_dir),
                    layer_plans=layer_plans,
                    strategy=strategy,
                    avg_retention=avg_retention,
                )

            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "n_layers": n_layers,
                "strategy": strategy,
                "avg_retention_ratio": round(avg_retention, 4),
                "avg_drop_ratio": round(avg_drop, 4),
                "estimated_kv_reduction_pct": round(avg_drop * 100, 1),
                "layer_plans": layer_plans,
            }
            logger.info(
                "Pass 14 complete: avg %.0f%% KV retained (%.0f%% compressed) "
                "across %d layers.  Elapsed: %.3fs.",
                avg_retention * 100,
                avg_drop * 100,
                n_layers,
                elapsed,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 14 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)

        return graph, report


# ── Semantic clustering helpers ───────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two flat vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_kv_compress(
    key_vectors: list[list[float]],
    value_vectors: list[list[float]],
    retention_ratio: float,
    chunk_size: int = 16,
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Compress KV pairs using ChunkKV semantic clustering.

    Args:
        key_vectors: List of key vectors, each of shape (head_dim,).
        value_vectors: Corresponding value vectors.
        retention_ratio: Fraction of chunks to retain (0.0–1.0).
        chunk_size: Number of tokens per chunk.

    Returns:
        (compressed_keys, compressed_values, retained_indices)

    Algorithm:
      1. Divide sequence into non-overlapping chunks of ``chunk_size`` tokens.
      2. Compute mean-pooled key vector per chunk.
      3. Greedy clustering: merge chunks whose mean key cosine similarity
         exceeds (1 - retention_ratio) threshold.
      4. Retain one representative (centroid's closest actual KV) per cluster.
    """
    n = len(key_vectors)
    if n == 0 or retention_ratio >= 1.0:
        return key_vectors, value_vectors, list(range(n))

    # Step 1: Build chunks.
    chunks: list[list[int]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(list(range(start, end)))

    # Step 2: Compute mean key per chunk.
    chunk_means: list[list[float]] = []
    for chunk in chunks:
        d = len(key_vectors[0])
        mean_vec = [0.0] * d
        for idx in chunk:
            for j, v in enumerate(key_vectors[idx]):
                mean_vec[j] += v
        count = len(chunk)
        chunk_means.append([v / count for v in mean_vec])

    # Step 3: Greedy merge — merge chunk i+1 into i if similarity > threshold.
    merge_threshold = 1.0 - retention_ratio  # higher threshold = less merging
    # Clamp: cosine similarity is in [-1, 1]; use 0.9 as max useful threshold.
    merge_threshold = max(0.0, min(0.99, merge_threshold))

    cluster_members: list[list[int]] = [[c] for c in range(len(chunks))]
    i = 0
    while i < len(cluster_members) - 1:
        mean_i = chunk_means[cluster_members[i][0]]
        mean_j = chunk_means[cluster_members[i + 1][0]]
        if cosine_similarity(mean_i, mean_j) > merge_threshold:
            # Merge cluster i+1 into i.
            cluster_members[i].extend(cluster_members[i + 1])
            del cluster_members[i + 1]
            # Recompute mean for merged cluster.
            d = len(chunk_means[0])
            new_mean = [0.0] * d
            for c_idx in cluster_members[i]:
                for j, v in enumerate(chunk_means[c_idx]):
                    new_mean[j] += v
            count = len(cluster_members[i])
            new_mean = [v / count for v in new_mean]
            chunk_means[cluster_members[i][0]] = new_mean
        else:
            i += 1

    # Step 4: Retain one representative token per cluster (the first chunk's first token).
    retained_indices: list[int] = []
    for cluster in cluster_members:
        first_chunk_idx = cluster[0]
        representative_token = chunks[first_chunk_idx][0]
        retained_indices.append(representative_token)

    retained_indices = sorted(set(retained_indices))  # deduplicate, preserve order
    compressed_keys = [key_vectors[i] for i in retained_indices]
    compressed_values = [value_vectors[i] for i in retained_indices]
    return compressed_keys, compressed_values, retained_indices


def sentence_kv_compress(
    key_vectors: list[list[float]],
    value_vectors: list[list[float]],
    sentence_boundary_mask: list[bool],
    retention_ratio: float,
) -> tuple[list[list[float]], list[list[float]], list[int]]:
    """Compress KV using SentenceKV sentence-boundary aware retention.

    Args:
        key_vectors: Key vectors.
        value_vectors: Value vectors.
        sentence_boundary_mask: True at position i iff token i is a sentence
            boundary (first token of a new sentence).
        retention_ratio: Target fraction of tokens to retain.

    Returns:
        (compressed_keys, compressed_values, retained_indices)

    Algorithm:
      - Always retain sentence-boundary tokens (they carry high information).
      - From non-boundary tokens, retain a fraction proportional to the budget
        remaining after boundary tokens are accounted for.
    """
    n = len(key_vectors)
    if n == 0 or retention_ratio >= 1.0:
        return key_vectors, value_vectors, list(range(n))

    boundary_indices = [i for i, is_b in enumerate(sentence_boundary_mask) if is_b]
    non_boundary = [i for i in range(n) if i not in set(boundary_indices)]

    # Budget: total retained = n * retention_ratio.
    budget = max(len(boundary_indices), int(n * retention_ratio))
    non_boundary_budget = max(0, budget - len(boundary_indices))

    # From non-boundary, sample uniformly (stride-based).
    if non_boundary and non_boundary_budget > 0:
        stride = max(1, len(non_boundary) // non_boundary_budget)
        sampled_non_boundary = non_boundary[::stride][:non_boundary_budget]
    else:
        sampled_non_boundary = []

    retained_indices = sorted(set(boundary_indices) | set(sampled_non_boundary))
    compressed_keys = [key_vectors[i] for i in retained_indices]
    compressed_values = [value_vectors[i] for i in retained_indices]
    return compressed_keys, compressed_values, retained_indices


# ── Graph annotation helpers ──────────────────────────────────────────────────


def _emit_kv_compress_opcode(
    graph: Any,
    layer_idx: int,
    retention_ratio: float,
    strategy: str,
    chunk_size: int,
) -> None:
    """Emit a semantic_kv_compress opcode annotation into the graph."""
    opcode = {
        "opcode": "aeg.semantic_kv_compress",
        "layer_index": layer_idx,
        "retention_ratio": retention_ratio,
        "strategy": strategy,
        "chunk_size": chunk_size,
    }
    if hasattr(graph, "add_kv_annotation"):
        graph.add_kv_annotation(layer_idx, opcode)
    elif hasattr(graph, "metadata"):
        kv_plan = graph.metadata.setdefault("kv_compression_opcodes", [])
        kv_plan.append(opcode)


def _write_kv_compression_plan(
    output_dir: Any,
    layer_plans: list[dict[str, Any]],
    strategy: str,
    avg_retention: float,
) -> None:
    from pathlib import Path
    graph_dir = Path(output_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "format": "aether_kv_compression_v1",
        "strategy": strategy,
        "avg_retention_ratio": round(avg_retention, 4),
        "pyramid_alpha": 0.5,
        "layers": layer_plans,
    }
    (graph_dir / "kv_compression_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    logger.debug("Wrote KV compression plan: %s", graph_dir / "kv_compression_plan.json")


def _count_layers(architecture: Any, graph: Any) -> int:
    if isinstance(architecture, dict):
        for k in ("num_hidden_layers", "n_layers", "num_layers"):
            if k in architecture:
                return int(architecture[k])
    elif hasattr(architecture, "num_hidden_layers"):
        return int(architecture.num_hidden_layers)
    if hasattr(graph, "n_layers"):
        return int(graph.n_layers)
    return 32  # safe default for 7B-class models


def _infer_head_dim(architecture: Any) -> int:
    if isinstance(architecture, dict):
        if "head_dim" in architecture:
            return int(architecture["head_dim"])
        h = int(architecture.get("hidden_size", 4096))
        n = int(architecture.get("num_attention_heads", 32))
        return h // n if n > 0 else 128
    elif hasattr(architecture, "head_dim"):
        return int(architecture.head_dim)
    return 128


def _infer_n_kv_heads(architecture: Any) -> int:
    if isinstance(architecture, dict):
        for k in ("num_key_value_heads", "n_kv_heads", "num_attention_heads"):
            if k in architecture:
                return int(architecture[k])
    elif hasattr(architecture, "num_key_value_heads"):
        return int(architecture.num_key_value_heads)
    return 8  # GQA default


