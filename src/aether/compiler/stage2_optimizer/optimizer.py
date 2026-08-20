"""
OptimizerPipeline — pass manager for Stage 2 optimizer passes.

The OptimizerPipeline orchestrates all 22 compiler passes in the PRD-defined
order.  It also handles pass registration, dependency checking, and report
generation.
"""

from __future__ import annotations

import time
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CalibrationError, CompilerPassError
from aether.utils.logging import get_logger

# PRD v4.0 + v5.0 pass imports
from aether.compiler.stage2_optimizer.pass10_mtp_head import MTPHeadCompilationPass
from aether.compiler.stage2_optimizer.pass11_grammar_constraint import GrammarConstraintCompilerPass
from aether.compiler.stage2_optimizer.pass12_model_merging import ModelMergingPass
from aether.compiler.stage2_optimizer.pass13_ttt_fast_weight import TTTFastWeightInjectionPass
from aether.compiler.stage2_optimizer.pass14_semantic_kv_compression import SemanticKVCompressionPass
from aether.compiler.stage2_optimizer.pass15_cross_layer_kv import CrossLayerKVSharingPass
from aether.compiler.stage2_optimizer.pass16_green_energy import GreenEnergyCompilationPass
from aether.compiler.stage2_optimizer.pass17_tee_wrapping import TEEKernelWrappingPass
from aether.compiler.stage2_optimizer.pass18_mdlm_drafter import MDLMDrafterCompilationPass
from aether.compiler.stage2_optimizer.pass19_sub2bit_quant import Sub2BitQuantizationPass
from aether.compiler.stage2_optimizer.pass20_video_compression import VideoTokenCompressionPass
from aether.compiler.stage2_optimizer.pass21_advanced_peft import AdvancedPEFTCompilationPass
from aether.compiler.stage2_optimizer.pass22_rlvr_verifier import RLVRVerifierHeadInjectionPass

logger = get_logger(__name__)

# BasePass lives in base_pass.py to avoid circular imports.
# (passes import BasePass, optimizer imports passes — circular if both in optimizer.py)
from aether.compiler.stage2_optimizer.base_pass import BasePass  # noqa: E402  re-export


class OperatorFusionPass(BasePass):
    """Pass 1: Operator Fusion.

    Identifies fuseable operation sequences (e.g., RMSNorm + QKV + RoPE) and
    merges them into single megakernel operations. This reduces the number of
    GPU kernel launches and eliminates intermediate DRAM round-trips.
    """

    name = "operator_fusion"
    description = "Fuse sequential operations into megakernels."

    # Fuseable sequences: each entry is a tuple of op-type groups that must appear
    # in the same transformer layer (in any order) to qualify for fusion.
    _FUSION_SEQUENCES: list[tuple[tuple[str, ...], ...]] = [
        # Full attention pre-block: RMSNorm → QKV → RoPE
        (("rmsnorm",), ("qkv_proj",), ("rope",)),
        # FFN block: RMSNorm → SwiGLU FFN
        (("rmsnorm",), ("swiglu_ffn",)),
        # GQA Projection
        (("gqa",), ("gate_proj",)),
    ]

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            if not hasattr(graph, "fuse_subgraph"):
                report = PassReport(
                    pass_name=self.name,
                    status="skipped",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"reason": "Graph does not support fuse_subgraph"},
                )
                logger.info("Pass 1: Fused 0 operation groups")
                return graph, report

            fused_count = 0
            merged_ops_total = 0
            patterns: dict[str, int] = {}
            # Track already-fused node IDs to prevent double-fusion.
            already_fused_ids: set[str] = set()

            # iter_layers() groups nodes by transformer layer_index.
            # This is always present on AEGGraph and is the canonical way to walk layers.
            layer_iter = graph.iter_layers() if hasattr(graph, "iter_layers") else iter([[n for n in graph]])

            for layer_nodes in layer_iter:
                # Build op_type → node list for this layer (skip already-fused nodes).
                # Within each op_type list, preserve ordering so the first entry is the
                # first topologically — e.g. the pre-attention rmsnorm comes before the
                # pre-FFN rmsnorm so the attention fusion picks up the right one.
                op_map: dict[str, list] = {}
                for n in layer_nodes:
                    nid = getattr(n, "id", "")
                    if (
                        n.op_type
                        and not n.attributes.get("is_fused_away", False)
                        and nid not in already_fused_ids
                        and n.op_type not in ("kv_cache", "moe_router", "input", "output")
                    ):
                        op_map.setdefault(n.op_type, []).append(n)

                # Try each candidate fusion sequence in order (highest-priority first).
                for seq in self._FUSION_SEQUENCES:
                    candidates: list[Any] = []
                    for op_group in seq:
                        found = None
                        for op in op_group:
                            # Skip nodes already consumed by a previous fusion this layer.
                            available = [
                                n for n in op_map.get(op, [])
                                if getattr(n, "id", "") not in already_fused_ids
                            ]
                            if available:
                                found = available[0]
                                break
                        if found is None:
                            break  # This sequence not present in layer.
                        candidates.append(found)
                    else:
                        # All ops in sequence found — fuse them.
                        if len(candidates) >= 2:
                            # Matching operation types in the same layer is not
                            # enough: a layer contains two RMSNorms, and choosing
                            # the first one for the FFN pattern can fuse the
                            # attention branch into the later residual branch.
                            # Require each adjacent pair to be a real data edge.
                            # This also keeps Qwen3's Q/K norm nodes between QKV
                            # and RoPE instead of manufacturing an invalid cycle.
                            if any(
                                not any(
                                    edge.source == getattr(candidates[index], "id", "")
                                    and edge.target == getattr(candidates[index + 1], "id", "")
                                    for edge in graph.get_output_edges(getattr(candidates[index], "id", ""))
                                )
                                for index in range(len(candidates) - 1)
                            ):
                                continue
                            # Qwen3 applies parameterized Q/K head norms
                            # between QKV projection and RoPE. Fusing across
                            # those nodes changes graph topology and can create
                            # a cycle when the norm parameters are retained.
                            if any(
                                bool(getattr(candidate, "attributes", {}).get("qk_norm"))
                                for candidate in candidates
                            ):
                                continue
                            candidate_ids = [getattr(c, "id", f"node_{i}") for i, c in enumerate(candidates)]
                            first_op = candidates[0].op_type or "op"
                            last_op = candidates[-1].op_type or "op"
                            pattern_key = f"{first_op}+{last_op}"
                            # Build a deterministic unique fused_name from candidate IDs.
                            fused_name = "+".join(c.op_type or "op" for c in candidates)
                            fused_op_type = f"aeg.fused_{'_'.join(c.op_type or 'op' for c in candidates)}"
                            try:
                                graph.fuse_subgraph(
                                    candidate_ids,
                                    fused_name,
                                    fused_op_type,
                                )
                                fused_count += 1
                                merged_ops_total += len(candidates)
                                patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
                                # Mark fused nodes so they aren't reused in subsequent sequences.
                                already_fused_ids.update(candidate_ids)
                                for c in candidates:
                                    if c.op_type in op_map:
                                        op_map[c.op_type] = [
                                            n for n in op_map[c.op_type]
                                            if getattr(n, "id", "") not in already_fused_ids
                                        ]
                            except Exception as fuse_exc:  # noqa: BLE001
                                logger.debug("Pass 1: fuse_subgraph failed for %s: %s", fused_name, fuse_exc)

            if hasattr(graph, "set_metadata"):
                graph.set_metadata("fusion_accounting", {
                    "fused_groups": fused_count,
                    "merged_ops_total": merged_ops_total,
                    "launches_saved": max(0, merged_ops_total - fused_count),
                    "patterns": dict(patterns),
                })
            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=fused_count * 3,
                details={
                    "fused_count": fused_count,
                    "merged_ops_total": merged_ops_total,
                    "fusion_patterns": patterns,
                },
            )
            logger.info(f"Pass 1: Fused {fused_count} operation groups")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 1 failed: {exc}")
        return graph, report


class SensitivityAnalysisPass(BasePass):
    """Pass 2: Sensitivity Analysis.

    Computes a per-layer sensitivity score: the change in model perplexity
    per saved bit when a layer is quantized. This is the mathematical
    foundation for Aether's mixed-precision quantization.
    """

    name = "sensitivity_analysis"
    description = "Compute d(perplexity)/d(precision) per layer."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            from aether.compiler.calibration import SensitivityCalibration, get_dataset

            base_precision_map = {f"layer_{i}": "BF16" for i in range(architecture.layers)}
            base_precision_map["embedding"] = "BF16"
            base_precision_map["lm_head"] = "BF16"
            dataset = get_dataset(config.calibration_dataset, max_tokens=config.calibration_tokens)
            calibration = SensitivityCalibration(architecture)
            layer_weights = self._collect_layer_weights(graph, architecture.layers)
            method = SensitivityCalibration.scoring_method(layer_weights)
            sensitivity_map = calibration.score_by_layer(
                dataset, base_precision_map, layer_weights=layer_weights
            )
            summary = calibration.score_summary(sensitivity_map)
            summary["method"] = method
            annotated_count = self._annotate_graph(graph, sensitivity_map)
            if hasattr(graph, "set_metadata"):
                graph.set_metadata("sensitivity_method", method)

            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=annotated_count,
                details={
                    "sensitivity_map": sensitivity_map,
                    "summary": summary,
                    "method": method,
                    "dataset": dataset.name,
                    # Weight-backed calibration uses measured reconstruction
                    # error and does not consume text.  Do not force a named
                    # corpus lookup merely to populate an informational
                    # counter; text calibration remains explicit for
                    # weightless graphs.
                    "calibration_tokens": (
                        dataset.token_count()
                        if not layer_weights and hasattr(dataset, "token_count")
                        else 0
                    ),
                    "high_sensitivity_layers": sum(1 for v in sensitivity_map.values() if v > 0.7),
                    "low_sensitivity_layers": sum(1 for v in sensitivity_map.values() if v < 0.4),
                },
            )
            logger.info(
                "Pass 2: Computed sensitivity for %d graph nodes (method: %s)",
                annotated_count,
                method,
            )
        except Exception as exc:
            # A named calibration corpus is an optional input for graphs that
            # carry no measurable tensors.  If it is unavailable locally,
            # skip this pass rather than failing compilation or inventing
            # benchmark prose; real weight-backed graphs use the measured
            # reconstruction path above and do not need text samples.
            if isinstance(exc, CalibrationError):
                report = PassReport(
                    pass_name=self.name,
                    status="skipped",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"reason": str(exc), "calibration_required": True},
                )
                logger.warning("Pass 2 skipped: %s", exc)
                return graph, report
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 2 failed: {exc}")
        return graph, report

    @staticmethod
    def _collect_layer_weights(graph: Any, num_layers: int) -> dict[str, list[Any]]:
        """Gather real bound weight tensors per layer from the graph.

        Returns ``{layer_i: [ndarray, ...]}`` plus ``embedding``/``lm_head``
        entries.  Layers with no bound tensors are simply absent, which keeps
        ``score_by_layer`` on the honest fallback path when the checkpoint
        carried no weights.
        """
        import numpy as np

        layer_weights: dict[str, list[Any]] = {}
        node_iter = (
            graph.nodes.values()
            if hasattr(graph, "nodes")
            else (graph if hasattr(graph, "__iter__") else [])
        )
        for node in node_iter:
            attributes = getattr(node, "attributes", {}) or {}
            candidate = attributes.get("weight") if isinstance(attributes, dict) else None
            if candidate is None:
                continue
            try:
                array = np.asarray(candidate)
            except (TypeError, ValueError):
                continue
            if array.ndim == 0 or array.size == 0:
                continue
            node_id = str(getattr(node, "id", ""))
            layer_index = getattr(node, "layer_index", None)
            if layer_index is not None and 0 <= int(layer_index) < num_layers:
                layer_weights.setdefault(f"layer_{int(layer_index)}", []).append(array)
            elif node_id == "embedding" or node_id == "lm_head":
                layer_weights.setdefault(node_id, []).append(array)
        return layer_weights

    def _annotate_graph(self, graph: Any, sensitivity_map: dict[str, float]) -> int:
        """Attach sensitivity scores to every graph node with a layer index."""
        annotated_count = 0
        # Use graph.nodes.values() instead of `for node in graph:` to avoid
        # triggering topological_order() which fails on fused graphs.
        node_iter = (
            graph.nodes.values()
            if hasattr(graph, "nodes")
            else (graph if hasattr(graph, "__iter__") else [])
        )
        for node in node_iter:
            layer_index = getattr(node, "layer_index", None)
            if layer_index is None or layer_index < 0:
                continue
            score = sensitivity_map.get(f"layer_{layer_index}")
            if score is None:
                continue
            if hasattr(node, "add_attribute"):
                node.add_attribute("sensitivity", score)
                node.add_attribute("sensitivity_class", self._classify(score))
                annotated_count += 1
        if hasattr(graph, "set_metadata"):
            graph.set_metadata("sensitivity_map", sensitivity_map)
        return annotated_count

    def _classify(self, score: float) -> str:
        if score >= 0.9:
            return "critical"
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"


class PrecisionAssignmentPass(BasePass):
    """Pass 3: Precision Assignment.

    Using the sensitivity map from Pass 2, assigns an optimal precision
    format to each layer subject to the quality budget constraint.
    """

    name = "precision_assignment"
    description = "Assign mixed precision using sensitivity map."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        precision_map: dict[str, str] = {}
        try:
            if config.precision_assignment_mode == "manual":
                precision_map.update(config.manual_precision_map)
            elif config.precision_assignment_mode == "uniform":
                for i in range(architecture.layers):
                    precision_map[f"layer_{i}"] = "Q4_K_M"
            else:
                sensitivity_map: dict[str, float] = {}
                if hasattr(graph, "metadata"):
                    sensitivity_map.update(getattr(graph, "metadata", {}).get("sensitivity_map", {}))
                if not sensitivity_map and hasattr(graph, "nodes"):
                    # Use graph.nodes.values() to avoid topological_order() on fused graphs.
                    for node in graph.nodes.values():
                        layer_index = getattr(node, "layer_index", None)
                        if layer_index is None or layer_index < 0:
                            continue
                        sens = node.get_attribute("sensitivity") if hasattr(node, "get_attribute") else None
                        if sens is not None:
                            sensitivity_map[f"layer_{layer_index}"] = max(
                                float(sens),
                                sensitivity_map.get(f"layer_{layer_index}", 0.0),
                            )
                if not sensitivity_map:
                    for i in range(architecture.layers):
                        # Weight-reconstruction sensitivity is not a model
                        # quality evaluation.  Until a perplexity/task gate has
                        # actually run, keep the source precision rather than
                        # silently accumulating lossy Q4/Q3 error across all
                        # transformer layers.
                        precision_map[f"layer_{i}"] = "BF16"
                else:
                    for i in range(architecture.layers):
                        layer_key = f"layer_{i}"
                        # The current sensitivity signal measures weight
                        # reconstruction only; it does not establish the PRD's
                        # perplexity/task-quality budget.  Approximate formats
                        # are therefore reserved for explicit manual/uniform
                        # requests until a real evaluation gate is available.
                        precision_map[layer_key] = "BF16"

            # Always keep embedding and LM head at BF16
            precision_map["embedding"] = "BF16"
            precision_map["lm_head"] = "BF16"

            # Annotate nodes with precision
            if hasattr(graph, "iter_layers"):
                for i, node_group in enumerate(graph.iter_layers()):
                    precision = precision_map.get(f"layer_{i}", "BF16")
                    for node in node_group:
                        if node.attributes and "precision" not in node.attributes:
                            node.set_precision(__import__("aether.core.types", fromlist=["Precision"]).Precision.from_string(precision))

            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=len(precision_map),
                details={
                    "precision_map": precision_map,
                    "bf16_layers": sum(1 for v in precision_map.values() if v == "BF16"),
                    "fp8_layers": sum(1 for v in precision_map.values() if v == "FP8"),
                    "q4_k_m_layers": sum(1 for v in precision_map.values() if v == "Q4_K_M"),
                    "q3_k_layers": sum(1 for v in precision_map.values() if v == "Q3_K"),
                },
            )
            logger.info(f"Pass 3: Assigned precision to {len(precision_map)} layers")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 3 failed: {exc}")
        return graph, report


class KVCacheStructuringPass(BasePass):
    """Pass 4: KV Cache Structuring.

    Annotates the AEG-IR with explicit KV cache graph nodes: paged block sizes,
    radix-tree prefix hints, memory tier offload thresholds, and cache-sharing
    policies.
    """

    name = "kv_cache_structuring"
    description = "Structure KV cache with paged blocks and tiering."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        from aether.core.aeg_ir import AEGOpCode

        start = time.perf_counter()
        try:
            kv_cache_nodes_added = 0
            if hasattr(graph, "add_node"):
                from aether.core.graph import AEGGraphNode, AEGGraphNodeType

                # Add KV cache nodes for each layer
                for i in range(architecture.layers):
                    cache_name = f"kv_cache_layer_{i}"
                    cache_node = AEGGraphNode(
                        id=cache_name,
                        node_type=AEGGraphNodeType.KV_CACHE,
                        name=cache_name.replace("_", " ").title(),
                        attributes={
                            "dtype": config.kv_cache_dtype,
                            "num_heads": architecture.num_kv_heads or architecture.num_attention_heads,
                            "head_dim": architecture.head_dim or (architecture.hidden_size // architecture.num_attention_heads),
                            "block_size": 16,
                            "cpu_offload_gb": config.kv_cache_cpu_gb,
                            "nvme_offload_gb": config.kv_cache_nvme_gb,
                            "prefix_hints": True,
                        },
                    )
                    graph.add_node(cache_node)
                    kv_cache_nodes_added += 1

            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=kv_cache_nodes_added,
                details={
                    "kv_cache_nodes_added": kv_cache_nodes_added,
                    "kv_cache_dtype": config.kv_cache_dtype,
                    "cpu_offload_gb": config.kv_cache_cpu_gb,
                    "nvme_offload_gb": config.kv_cache_nvme_gb,
                },
            )
            logger.info(f"Pass 4: Added {kv_cache_nodes_added} KV cache nodes")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 4 failed: {exc}")
        return graph, report


class MoERoutingPass(BasePass):
    """Pass 5: MoE Expert Routing Optimization.

    For MoE models, this pass classifies experts into hot/warm/cold tiers,
    adds threshold-based routing hints, and emits intra-expert sparsity
    annotations.  Tier assignment prefers *measured* per-expert activation
    frequencies (recorded on the graph by profiling loaders or calibration);
    without measurements it falls back to the documented Zipf-like empirical
    prior and labels the router node with the tier basis so downstream
    consumers can distinguish measured from prior estimates.
    """

    name = "moe_routing"
    description = "Optimize MoE expert routing and tiering."

    #: Zipf-like activation prior (Zoph et al. 2022; Fedus et al. 2022):
    #: ~20% of experts handle ~50% of tokens.  Mirrors
    #: ``moe_loader._classify_experts`` so both classification sites agree.
    _PRIOR_HOT_FRACTION = 0.20
    _PRIOR_WARM_FRACTION = 0.30

    @staticmethod
    def _collect_activation_frequencies(graph: Any, total_experts: int) -> Any:
        """Return per-expert activation frequencies when the graph has them.

        Accepts either normalized frequencies (``expert_activation_frequencies``)
        or raw counts (``expert_activation_counts``); counts are normalized to
        frequencies here.  Returns ``None`` when no complete, valid profile is
        attached — never a fabricated distribution.
        """
        metadata = getattr(graph, "metadata", {}) or {}
        for key in ("expert_activation_frequencies", "expert_activation_counts"):
            raw = metadata.get(key)
            if raw is None:
                continue
            try:
                import numpy as np

                values = np.asarray(raw, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                continue
            if values.size != total_experts or bool((values < 0).any()):
                continue
            if key == "expert_activation_counts":
                total = float(values.sum())
                if total <= 0:
                    continue
                values = values / total
            return values
        return None

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            hot_experts = 0
            warm_experts = 0
            cold_experts = 0
            tier_basis = "none"
            total_experts = architecture.num_experts if architecture.is_moe else 0

            if total_experts > 0 and hasattr(graph, "add_node"):
                from aether.core.graph import AEGGraphNode, AEGGraphNodeType

                hot_threshold = config.moe_hot_threshold
                warm_threshold = config.moe_warm_threshold
                frequencies = self._collect_activation_frequencies(graph, total_experts)
                if frequencies is not None:
                    hot_mask = frequencies > hot_threshold
                    warm_mask = (frequencies > warm_threshold) & ~hot_mask
                    hot_experts = int(hot_mask.sum())
                    warm_experts = int(warm_mask.sum())
                    cold_experts = total_experts - hot_experts - warm_experts
                    tier_basis = "measured_activation_frequency"
                else:
                    hot_experts = min(
                        total_experts,
                        max(1, round(total_experts * self._PRIOR_HOT_FRACTION)),
                    )
                    warm_experts = min(
                        total_experts - hot_experts,
                        max(1, round(total_experts * self._PRIOR_WARM_FRACTION)),
                    )
                    cold_experts = total_experts - hot_experts - warm_experts
                    tier_basis = "empirical_zipf_prior"

                router_node = AEGGraphNode(
                    id="moe_router",
                    node_type=AEGGraphNodeType.EXPERT_ROUTER,
                    name="MoE Router",
                    op_type="aeg.moe_router",
                    attributes={
                        "num_experts": total_experts,
                        "hot_threshold": hot_threshold,
                        "warm_threshold": warm_threshold,
                        "routing_strategy": "threshold",
                        "total_experts": total_experts,
                        "hot_count": hot_experts,
                        "warm_count": warm_experts,
                        "cold_count": cold_experts,
                        "tier_basis": tier_basis,
                    },
                )
                graph.add_node(router_node)

            report = PassReport(
                pass_name=self.name,
                status="applied" if total_experts > 0 else "skipped",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=1 if total_experts > 0 else 0,
                details={
                    "total_experts": total_experts,
                    "hot_experts": hot_experts,
                    "warm_experts": warm_experts,
                    "cold_experts": cold_experts,
                    "tier_basis": tier_basis,
                    "threshold_hot": config.moe_hot_threshold,
                    "threshold_warm": config.moe_warm_threshold,
                },
            )
            if total_experts > 0:
                logger.info(
                    f"Pass 5: Classified {total_experts} experts "
                    f"({hot_experts} hot, {warm_experts} warm, {cold_experts} cold; "
                    f"basis: {tier_basis})"
                )
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 5 failed: {exc}")
        return graph, report


class ParallelismDiscoveryPass(BasePass):
    """Pass 6: Automatic Parallelism Discovery.

    Searches over the parallelism strategy space (tensor parallel degree,
    pipeline stages, expert parallel degree, context parallel degree) and
    produces separate prefill and decode sharding plans. These plans are stored
    in the AEG artifact for zero-configuration multi-GPU deployment.
    """

    name = "parallelism_discovery"
    description = "Discover optimal parallelism strategy."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        from aether.core.aeg_format import create_default_sharding_plans

        start = time.perf_counter()
        try:
            plans = create_default_sharding_plans(architecture)
            # Set sharding annotations on graph
            if hasattr(graph, "set_metadata"):
                graph.set_metadata("sharding_plans", {k: v.to_dict() for k, v in plans.items()})
                graph.set_metadata("parallelism_config", {
                    "degrees_evaluated": config.parallelism_degrees,
                })

            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=len(plans),
                details={
                    "plans_generated": len(plans),
                    "gpu_counts": list(plans.keys()),
                    "prefill_plans": {k: v.phase for k, v in plans.items() if v.phase == "prefill"},
                    "decode_plans": {k: v.phase for k, v in plans.items() if v.phase == "decode"},
                },
            )
            logger.info(f"Pass 6: Generated {len(plans)} parallelism plans")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 6 failed: {exc}")
        return graph, report


class ReasoningGraphPass(BasePass):
    """Pass 7: Reasoning Graph Compilation.

    Emits a budget-controlled reasoning graph description for models used in
    chain-of-thought or tool-augmented workflows. The graph is metadata today,
    but it is executable by downstream runtimes because every node has explicit
    budget, confidence, and transition semantics.
    """

    name = "reasoning_graph"
    description = "Compile budget-controlled reasoning graph metadata."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            budget = max(1, config.reasoning_budget_tokens)
            reasoning_graph = {
                "version": "reasoning_graph/1.0",
                "entrypoint": "reasoning_step",
                "budget_tokens": budget,
                "early_exit_threshold": 0.95,
                "nodes": [
                    {"id": "context_encode", "op": "aeg.reasoning_context_encode", "budget_cost": 0},
                    {"id": "draft_thought", "op": "aeg.reasoning_forward", "budget_cost": "dynamic"},
                    {"id": "verify_thought", "op": "aeg.reasoning_verify", "confidence_threshold": 0.95},
                    {"id": "budget_decrement", "op": "aeg.budget_decrement", "max_tokens": budget},
                ],
                "edges": [
                    ["context_encode", "draft_thought"],
                    ["draft_thought", "verify_thought"],
                    ["verify_thought", "budget_decrement"],
                ],
                "speculative_cot": {
                    "enabled": True,
                    "draft_family": architecture.family,
                    "acceptance_floor": 0.70,
                },
            }
            if hasattr(graph, "set_metadata"):
                graph.set_metadata("reasoning_graph", reasoning_graph)
            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=len(reasoning_graph["nodes"]),
                details=reasoning_graph,
            )
            logger.info("Pass 7: Compiled reasoning graph metadata")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 7 failed: {exc}")
        return graph, report


class SparseAttentionPass(BasePass):
    """Pass 8: MInference-style sparse attention pattern planning."""

    name = "sparse_attention"
    description = "Compile sparse long-context attention head patterns."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            calibrated_maps = self._collect_calibration_maps(graph)
            patterns, pattern_basis = self._assign_head_patterns(architecture, calibrated_maps)
            enabled = architecture.context_length >= config.sparse_attention_context_threshold
            plan = {
                "version": "sparse_attention/1.0",
                "enabled": enabled,
                "activation_context_length": config.sparse_attention_context_threshold,
                "model_context_length": architecture.context_length,
                "patterns": patterns,
                "pattern_basis": pattern_basis,
                "runtime_fallback": "dense_attention",
                "research_basis": (
                    ["MInference", "MMInference", "Ring Attention"]
                    if pattern_basis == "minference_calibrated"
                    else ["MInference (pattern taxonomy)", "deterministic rotation assignment"]
                ),
            }
            if hasattr(graph, "set_metadata"):
                graph.set_metadata("attention_head_patterns", plan)
            report = PassReport(
                pass_name=self.name,
                status="applied" if enabled else "skipped",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=len(patterns),
                details=plan,
            )
            logger.info(
                "Pass 8: Planned %d sparse attention head groups (basis: %s)",
                len(patterns),
                pattern_basis,
            )
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 8 failed: {exc}")
        return graph, report

    @staticmethod
    def _collect_calibration_maps(graph: Any) -> dict[int, Any]:
        """Return per-layer calibration attention maps recorded on the graph.

        Calibration forward passes may attach ``attention_calibration_maps`` as
        ``{layer_idx: {head_idx: 2-D attention array}}``.  Returns ``{}`` when
        absent — the pass then falls back to a deterministic, labelled
        rotation assignment instead of pretending MInference profiling ran.
        """
        metadata = getattr(graph, "metadata", {}) or {}
        raw = metadata.get("attention_calibration_maps")
        if not isinstance(raw, dict):
            return {}
        maps: dict[int, Any] = {}
        for layer_key, head_map in raw.items():
            if isinstance(head_map, dict):
                try:
                    maps[int(layer_key)] = head_map
                except (TypeError, ValueError):
                    continue
        return maps

    def _assign_head_patterns(
        self,
        architecture: Any,
        calibration_maps: dict[int, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Assign a sparse pattern to every attention head.

        When calibration attention maps are available the real MInference
        classifier (``AttentionPatternClassifier``) scores each head's map.
        Without calibration data a deterministic rotation over the MInference
        pattern taxonomy is used so every pattern class is represented; the
        returned basis string records which path produced the plan.
        """
        head_count = max(1, architecture.num_attention_heads)
        maps = calibration_maps or {}
        if maps:
            from aether.compiler.stage2_optimizer.pass8_minference import AttentionPatternClassifier

            classifier = AttentionPatternClassifier()
            patterns: list[dict[str, Any]] = []
            for head in range(head_count):
                head_map = maps.get(0, {}).get(head) or maps.get(head)
                if head_map is None:
                    raise ValueError(
                        "partial attention calibration maps: head "
                        f"{head} has no map; refusing to mix measured and "
                        "fabricated patterns"
                    )
                classified = classifier.classify_head(
                    layer_idx=0,
                    head_idx=head,
                    num_layers=1,
                    num_heads=head_count,
                    attention_map=head_map,
                )
                patterns.append({
                    "head": head,
                    "pattern": str(classified.pattern_type),
                    "sink_tokens": classified.num_sink_tokens,
                    "local_window": classified.local_window_size,
                    "stride": 128 if classified.pattern_type == "vertical_slash" else 0,
                })
            return patterns, "minference_calibrated"

        rotation = ("vertical_slash", "block_sparse", "a_shape")
        patterns = []
        for head in range(head_count):
            pattern = rotation[head % 3]
            patterns.append({
                "head": head,
                "pattern": pattern,
                "sink_tokens": 128,
                "local_window": 4096,
                "stride": 128 if pattern == "vertical_slash" else 0,
            })
        return patterns, "deterministic_rotation_fallback"


class PruningSparsityPass(BasePass):
    """Pass 9: Wanda/SparseGPT-inspired sparsity mask planning."""

    name = "pruning_sparsity"
    description = "Plan sparsity masks and sparse kernel eligibility."

    #: Ops whose weight matrices are eligible for pruning.
    PRUNABLE_OPS = frozenset({"linear", "qkv_proj", "gate_proj", "swiglu_ffn", "expert_ffn"})

    #: Minimum sparsity at which 2:4 semi-structured pruning is preferred. A
    #: tolerance is applied because the sensitivity-scaled target lands on
    #: 0.44999999999999996 at default settings, which would otherwise never
    #: select 2:4 despite 2:4 being exactly the 50%-sparsity pattern requested.
    NM_SPARSITY_THRESHOLD = 0.45
    NM_SPARSITY_TOLERANCE = 1e-6

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            target_sparsity = config.pruning_target_sparsity
            masks: dict[str, dict[str, Any]] = {}
            computed = 0
            # Iterate via graph.nodes.values() (safe dict iteration) rather than
            # `for node in graph` which calls topological_order() and will raise
            # ValueError("Graph contains a cycle") on graphs that have been through
            # Pass 1 fusion (fused subgraph nodes produce apparent cycles in the
            # edge list). Direct node-dict iteration avoids this entirely.
            node_iter = (
                graph.nodes.values()
                if hasattr(graph, "nodes")
                else (graph if hasattr(graph, "__iter__") else [])
            )
            for node in node_iter:
                op_type = getattr(node, "op_type", None)
                if op_type not in self.PRUNABLE_OPS:
                    continue
                # Skip nodes that were absorbed into a fused megakernel by Pass 1;
                # the fused node itself will carry the pruning annotation instead.
                if getattr(node, "attributes", {}).get("is_fused_away", False):
                    continue
                entry = self._plan_node(node, op_type, target_sparsity, config)
                if entry.get("mask_computed"):
                    computed += 1
                masks[getattr(node, "id", f"node_{len(masks)}")] = entry

            plan = {
                "version": "sparsity/1.1",
                "method": config.pruning_metric,
                "target_sparsity": target_sparsity,
                "masks": masks,
                "masks_computed": computed,
                "masks_planned_only": len(masks) - computed,
                "sparse_kernel_targets": ["cuda_sm80", "cuda_sm90", "cuda_sm100", "cuda_sm120"],
            }
            if hasattr(graph, "set_metadata"):
                graph.set_metadata("sparsity_plan", plan)
            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=len(masks),
                details=plan,
            )
            logger.info(
                f"Pass 9: {computed} masks computed, {len(masks) - computed} planned "
                f"across {len(masks)} prunable nodes"
            )
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 9 failed: {exc}")
        return graph, report

    def _node_weights(self, node: Any) -> Any:
        """Extract the weight tensor from a graph node, or None if not attached.

        The ingestion pipeline stores weights in node attributes under several
        possible keys. This method probes each in priority order.
        """
        if not hasattr(node, "get_attribute"):
            return None
        for attr_name in ("weight", "weights", "W", "weight_tensor", "kernel"):
            w = node.get_attribute(attr_name)
            if w is not None:
                return w
        # Fallback: check a weight_store dict on the node itself
        weight_store = node.get_attribute("weight_store")
        if isinstance(weight_store, dict):
            node_id = getattr(node, "id", None)
            node_name = getattr(node, "name", None)
            for key in (node_id, node_name):
                if key and key in weight_store:
                    return weight_store[key]
        return None

    def _node_activation_norms(self, node: Any, in_features: int) -> Any:
        """Return per-column L2 activation norms for Wanda importance scoring.

        Returns a list/array of length `in_features`, or None if not available.
        If calibration data recorded norms in node attributes we return them;
        otherwise the caller falls back to magnitude pruning.
        """
        if not hasattr(node, "get_attribute"):
            return None
        for attr_name in ("activation_norms", "act_norms", "input_norms", "wanda_norms"):
            v = node.get_attribute(attr_name)
            if v is not None:
                return v
        return None

    def _compute_magnitude_mask(
        self, weights: Any, sparsity: float, pattern: str
    ) -> dict[str, Any]:
        """Compute a sparsity mask using weight magnitude as importance score.

        For 2:4 semi-structured: keeps the 2 largest-magnitude weights in every
        group of 4 consecutive elements along the output dimension.
        For unstructured: globally thresholds by magnitude percentile.

        Returns a dict with mask metadata (not a binary tensor, to avoid
        requiring numpy/torch in the compiler path).
        """
        result: dict[str, Any] = {"pattern": pattern, "method": "magnitude"}
        try:
            if hasattr(weights, "tolist"):
                flat = weights.tolist() if hasattr(weights, "tolist") else list(weights)
                if isinstance(flat[0], list):
                    flat = [x for row in flat for x in row]
            elif isinstance(weights, (list, tuple)):
                flat = [float(x) for x in weights]
            else:
                result["mask_computed"] = False
                result["reason"] = "unsupported weight type"
                return result

            abs_flat = [abs(v) for v in flat]
            n_total = len(abs_flat)

            if pattern == "2:4" and n_total >= 4:
                # 2:4: in each group of 4, zero the 2 smallest.
                n_zeroed = 0
                for g in range(0, n_total - 3, 4):
                    group = sorted(range(4), key=lambda i: abs_flat[g + i])
                    n_zeroed += 2  # always zero 2 of 4
                achieved = n_zeroed / n_total
            else:
                # Unstructured: threshold by percentile.
                sorted_abs = sorted(abs_flat)
                threshold_idx = max(0, int(sparsity * n_total) - 1)
                threshold = sorted_abs[threshold_idx]
                n_zeroed = sum(1 for v in abs_flat if v <= threshold)
                achieved = n_zeroed / max(1, n_total)

            result["mask_computed"] = True
            result["achieved_sparsity"] = round(achieved, 4)
            result["n_weights"] = n_total
            result["n_zeroed"] = n_zeroed
        except Exception as exc:  # noqa: BLE001
            result["mask_computed"] = False
            result["reason"] = f"mask computation error: {exc}"
        return result

    def _plan_node(
        self, node: Any, op_type: str, target_sparsity: float, config: CompilerConfig
    ) -> dict[str, Any]:
        """Plan — and where weights are attached, actually compute — a node's mask.

        Sensitivity from Pass 2 modulates the per-node target so fragile layers are
        pruned less aggressively. When the node carries a real weight matrix the
        mask is computed (magnitude or Wanda) and its achieved sparsity recorded;
        otherwise the entry stays a plan for a later stage that has the weights.
        """
        sensitivity = 0.5
        if hasattr(node, "get_attribute"):
            sensitivity = float(node.get_attribute("sensitivity", 0.5))
        # Fragile layers (high sensitivity) pruned less aggressively.
        sparsity = max(0.0, min(target_sparsity, target_sparsity * (1.15 - sensitivity * 0.5)))
        pattern = (
            "2:4"
            if sparsity >= self.NM_SPARSITY_THRESHOLD - self.NM_SPARSITY_TOLERANCE
            else "unstructured"
        )

        entry: dict[str, Any] = {
            "layer_index": getattr(node, "layer_index", None),
            "op_type": op_type,
            "target_sparsity": round(sparsity, 4),
            "pattern": pattern,
            "importance_metric": config.pruning_metric,
            "sensitivity": round(sensitivity, 4),
            "mask_computed": False,
        }

        weights = self._node_weights(node)
        if weights is None:
            entry["reason"] = "no weight tensor attached to graph node"
            return entry

        # 2:M kernels require aligned input dimensions. If the real tensor is
        # not aligned, use a truthful unstructured mask instead of emitting an
        # invalid structured-kernel claim.
        weight_shape = getattr(weights, "shape", ())
        if pattern != "unstructured" and len(weight_shape) >= 2:
            group_size = int(pattern.split(":", 1)[1])
            if int(weight_shape[1]) % group_size:
                pattern = "unstructured"
                entry["pattern"] = pattern
                entry["pattern_fallback"] = "unstructured"
                entry["reason"] = "input dimension is not aligned for the requested N:M pattern"

        # Determine effective metric: fall back to magnitude if Wanda norms missing.
        metric = config.pruning_metric
        if metric in ("wanda", "sparsegpt"):
            in_features = (
                weights.shape[1]
                if hasattr(weights, "shape") and len(weights.shape) >= 2
                else (len(weights[0]) if isinstance(weights, (list, tuple)) and weights else 1)
            )
            activation_norms = self._node_activation_norms(node, in_features)
            if activation_norms is None:
                metric = "magnitude"  # Degrade gracefully without calibration norms.
                entry["metric_fallback"] = "magnitude"

        # Compute and retain the actual boolean mask. A planning-only report
        # is insufficient: the runtime/backend needs the same verified mask
        # that the quality report describes.
        try:
            import numpy as np
            from aether.quantization.pruning import build_mask, verify_nm_pattern

            mask = build_mask(
                np.asarray(weights, dtype=np.float32),
                target_sparsity=sparsity,
                pattern=pattern,
                metric=metric,
                activation_norms=(
                    np.asarray(self._node_activation_norms(node, int(np.asarray(weights).shape[1])), dtype=np.float32)
                    if self._node_activation_norms(node, int(np.asarray(weights).shape[1])) is not None
                    else None
                ),
            )
            if hasattr(node, "add_attribute"):
                node.add_attribute("pruning_mask", mask)
            entry.update(mask.to_dict())
            entry["mask_computed"] = True
            entry["n_weights"] = int(mask.mask.size)
            entry["n_zeroed"] = int(mask.pruned_count)
            entry["achieved_sparsity"] = round(mask.achieved_sparsity, 4)
            if pattern != "unstructured":
                n_keep, m_group = (int(part) for part in pattern.split(":", 1))
                entry["nm_pattern_valid"] = bool(verify_nm_pattern(mask.mask, n_keep, m_group))
        except Exception as exc:  # noqa: BLE001
            entry["mask_computed"] = False
            entry["reason"] = f"mask computation error: {exc}"
        entry["importance_metric"] = metric
        return entry

class OptimizerPipeline:
    """Orchestrates all 22 Aether optimizer passes (PRD v3.1 passes 1–9 + PRD v4.0–v5.0 passes 10–22).

    Pass ordering (PRD v2 canonical order):
      1  OperatorFusionPass             2  SensitivityAnalysisPass
      3  PrecisionAssignmentPass        4  KVCacheStructuringPass
      5  MoERoutingPass                 6  ParallelismDiscoveryPass
      7  ReasoningGraphPass             8  SparseAttentionPass
      9  PruningSparsityPass            10 MTPHeadCompilationPass
      11 GrammarConstraintCompilerPass  12 ModelMergingPass
      13 TTTFastWeightInjectionPass     14 SemanticKVCompressionPass
      15 CrossLayerKVSharingPass        16 GreenEnergyCompilationPass
      17 TEEKernelWrappingPass          18 MDLMDrafterCompilationPass
      19 Sub2BitQuantizationPass        20 VideoTokenCompressionPass
      21 AdvancedPEFTCompilationPass    22 RLVRVerifierHeadInjectionPass
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        self.config = config or CompilerConfig()
        self._passes: list[BasePass] = [
            # PRD v3.1 passes 1–9
            OperatorFusionPass(),
            SensitivityAnalysisPass(),
            PrecisionAssignmentPass(),
            KVCacheStructuringPass(),
            MoERoutingPass(),
            ParallelismDiscoveryPass(),
            ReasoningGraphPass(),
            SparseAttentionPass(),
            PruningSparsityPass(),
            # PRD v4.0 passes 10–17
            MTPHeadCompilationPass(),
            GrammarConstraintCompilerPass(),
            ModelMergingPass(),
            TTTFastWeightInjectionPass(),
            SemanticKVCompressionPass(),
            CrossLayerKVSharingPass(),
            GreenEnergyCompilationPass(),
            TEEKernelWrappingPass(),
            # PRD v5.0 passes 18–22
            MDLMDrafterCompilationPass(),
            Sub2BitQuantizationPass(),
            VideoTokenCompressionPass(),
            AdvancedPEFTCompilationPass(),
            RLVRVerifierHeadInjectionPass(),
        ]
        self._pass_enabled: dict[str, bool] = {
            # PRD v3.1 — all on by default
            "operator_fusion": True,
            "sensitivity_analysis": True,
            "precision_assignment": True,
            "kv_cache_structuring": True,
            "moe_routing": True,
            "parallelism_discovery": True,
            "reasoning_graph": True,
            "sparse_attention": True,
            "pruning_sparsity": True,
            # PRD v4.0 — opt-in passes; ``_sync_enabled_from_config`` applies
            # the CompilerConfig flags before every run, and constants.py
            # keeps every v4/v5 pass disabled by default.  The values here
            # mirror those defaults so a directly-constructed pipeline (used
            # by tests) cannot silently run v4/v5 passes unconfigured.
            "mtp_head_compilation": False,          # opt-in (enable_mtp_head)
            "grammar_constraint_compilation": False,   # opt-in
            "model_merging": False,                    # opt-in
            "ttt_fast_weight_injection": False,        # opt-in
            "semantic_kv_compression": False,       # opt-in (enable_semantic_kv)
            "cross_layer_kv_sharing": False,        # opt-in (enable_cross_layer_kv)
            "green_energy_compilation": False,         # opt-in
            "tee_kernel_wrapping": False,              # opt-in
            # PRD v5.0
            "mdlm_drafter_compilation": False,         # opt-in
            "sub2bit_quantization": False,             # opt-in
            "video_token_compression": False,      # opt-in (enable_video_compression)
            "advanced_peft_compilation": False,    # opt-in (enable_advanced_peft)
            "rlvr_verifier_head_injection": False,     # opt-in
        }

    @property
    def pass_count(self) -> int:
        return len(self._passes)

    def register_pass(self, pass_instance: BasePass) -> None:
        """Register a custom pass at the end of the pipeline."""
        self._passes.append(pass_instance)
        self._pass_enabled[pass_instance.name] = True

    def enable_pass(self, name: str) -> None:
        """Enable a pass by name."""
        if name in self._pass_enabled:
            self._pass_enabled[name] = True

    def disable_pass(self, name: str) -> None:
        """Disable a pass by name."""
        if name in self._pass_enabled:
            self._pass_enabled[name] = False

    def run(
        self,
        graph: Any,
        architecture: Any,
    ) -> tuple[Any, list[PassReport]]:
        """Run all enabled passes sequentially.

        Args:
            graph: Input computation graph or AEG-IR module.
            architecture: Model architecture metadata.

        Returns:
            Tuple of (optimized_graph, list of PassReports).
        """
        self._sync_enabled_from_config()
        pass_reports: list[PassReport] = []
        current_graph = graph

        for pass_instance in self._passes:
            if not self._pass_enabled.get(pass_instance.name, True):
                # Keep the metadata contract stable when an approximate pass is
                # disabled for quality safety.  Consumers can distinguish an
                # explicit no-op from a missing/old artifact without assuming
                # that the optimizer silently ran the pass.
                if pass_instance.name == "pruning_sparsity" and hasattr(current_graph, "set_metadata"):
                    current_graph.set_metadata(
                        "sparsity_plan",
                        {
                            "enabled": False,
                            "reason": "disabled_pending_quality_gate",
                            "masks": {"status": "not_applied"},
                        },
                    )
                pass_reports.append(
                    PassReport(
                        pass_name=pass_instance.name,
                        status="skipped",
                        details={"reason": "Disabled in configuration"},
                    )
                )
                logger.debug(f"Pass '{pass_instance.name}' skipped (disabled)")
                continue

            logger.info(f"Running pass: {pass_instance.name}")
            current_graph, report = pass_instance.run(current_graph, architecture, self.config)
            pass_reports.append(report)

        return current_graph, pass_reports

    def _sync_enabled_from_config(self) -> None:
        """Sync pass-enabled flags from CompilerConfig."""
        c = self.config
        # PRD v3.1
        self._pass_enabled["operator_fusion"] = c.enable_fusion
        self._pass_enabled["sensitivity_analysis"] = c.enable_sensitivity
        self._pass_enabled["precision_assignment"] = c.enable_precision_assignment
        self._pass_enabled["kv_cache_structuring"] = c.enable_kv_cache_structuring
        self._pass_enabled["moe_routing"] = c.enable_moe_routing
        self._pass_enabled["parallelism_discovery"] = c.enable_parallelism_discovery
        self._pass_enabled["reasoning_graph"] = c.enable_reasoning_graph
        self._pass_enabled["sparse_attention"] = c.enable_sparse_attention
        self._pass_enabled["pruning_sparsity"] = c.enable_pruning
        # PRD v4.0
        self._pass_enabled["mtp_head_compilation"] = c.enable_mtp_head
        self._pass_enabled["grammar_constraint_compilation"] = c.enable_grammar_constraint
        self._pass_enabled["model_merging"] = c.enable_model_merging
        self._pass_enabled["ttt_fast_weight_injection"] = c.enable_ttt
        self._pass_enabled["semantic_kv_compression"] = c.enable_semantic_kv
        self._pass_enabled["cross_layer_kv_sharing"] = c.enable_cross_layer_kv
        self._pass_enabled["green_energy_compilation"] = c.enable_green_energy
        self._pass_enabled["tee_kernel_wrapping"] = c.enable_tee
        # PRD v5.0
        self._pass_enabled["mdlm_drafter_compilation"] = c.enable_mdlm_drafter
        self._pass_enabled["sub2bit_quantization"] = c.enable_sub2bit
        self._pass_enabled["video_token_compression"] = c.enable_video_compression
        self._pass_enabled["advanced_peft_compilation"] = c.enable_advanced_peft
        self._pass_enabled["rlvr_verifier_head_injection"] = c.enable_rlvr_verifier

    def __repr__(self) -> str:
        enabled = sum(1 for v in self._pass_enabled.values() if v)
        return f"OptimizerPipeline({enabled}/{self.pass_count} passes enabled)"
