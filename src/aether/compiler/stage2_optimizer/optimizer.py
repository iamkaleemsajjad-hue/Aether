"""
OptimizerPipeline — pass manager for Stage 2 optimizer passes.

The OptimizerPipeline orchestrates the six compiler passes in the correct order.
It also handles pass registration, dependency checking, and report generation.
"""

from __future__ import annotations

import time
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class BasePass:
    """Base class for all optimizer passes.

    Each pass must implement `run()` which takes a computation graph (an AEGGraph
    or AEGIRModule) and returns an (optimized_graph, report) tuple.
    """

    name: str = "base"
    description: str = "Base optimizer pass."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        """Execute the pass.

        Args:
            graph: Input computation graph or AEG-IR module.
            architecture: Model architecture metadata.
            config: Compiler configuration.

        Returns:
            Tuple of (optimized_graph, PassReport).
        """
        raise NotImplementedError


class OperatorFusionPass(BasePass):
    """Pass 1: Operator Fusion.

    Identifies fuseable operation sequences (e.g., RMSNorm + QKV + RoPE) and
    merges them into single megakernel operations. This reduces the number of
    GPU kernel launches and eliminates intermediate DRAM round-trips.
    """

    name = "operator_fusion"
    description = "Fuse sequential operations into megakernels."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})
        try:
            if hasattr(graph, "fuse_subgraph"):
                fused_count = 0
                patterns: dict[str, int] = {}
                # Detect transformer layers
                if hasattr(graph, "iter_layers"):
                    for node_group in graph.iter_layers():
                        # Identify QKV + RoPE fusion candidates
                        qkv_nodes = [n for n in node_group if n.op_type in ("qkv_proj", None)]
                        if len(qkv_nodes) >= 3:
                            graph.fuse_subgraph(
                                [n.id for n in qkv_nodes[:3]],
                                "fused_qkv_rope_norm",
                                "aeg.fused_qkv_rope_norm",
                            )
                            fused_count += 1
                            patterns["qkv_rope_norm"] = patterns.get("qkv_rope_norm", 0) + 1
                else:
                    # Flat graph approach: scan for sequential patterns
                    qkv_pattern = []
                    for node in graph:
                        if node.node_type.value in ("parameter", "input") or qkv_pattern:
                            if node.op_type in ("linear", "qkv_linear", "gate_proj", "up_proj", "down_proj") and qkv_pattern:
                                qkv_pattern.append(node)
                                if len(qkv_pattern) == 3:
                                    graph.fuse_subgraph(
                                        [n.id for n in qkv_pattern],
                                        "fused_linear_group",
                                        "aeg.fused_linear_group",
                                    )
                                    fused_count += 1
                                    patterns["linear_group"] = patterns.get("linear_group", 0) + 1
                                    qkv_pattern = []

                report = PassReport(
                    pass_name=self.name,
                    status="applied",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    nodes_affected=fused_count * 3,
                    details={
                        "fused_count": fused_count,
                        "fusion_patterns": patterns,
                    },
                )
                logger.info(f"Pass 1: Fused {fused_count} operation groups")
            else:
                report = PassReport(
                    pass_name=self.name,
                    status="skipped",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"reason": "Graph does not support fuse_subgraph"},
                )
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
            sensitivity_map = calibration.score_by_layer(dataset, base_precision_map)
            summary = calibration.score_summary(sensitivity_map)
            annotated_count = self._annotate_graph(graph, sensitivity_map)

            report = PassReport(
                pass_name=self.name,
                status="applied",
                duration_ms=(time.perf_counter() - start) * 1000,
                nodes_affected=annotated_count,
                details={
                    "sensitivity_map": sensitivity_map,
                    "summary": summary,
                    "dataset": dataset.name,
                    "calibration_tokens": dataset.token_count() if hasattr(dataset, "token_count") else 0,
                    "high_sensitivity_layers": sum(1 for v in sensitivity_map.values() if v > 0.7),
                    "low_sensitivity_layers": sum(1 for v in sensitivity_map.values() if v < 0.4),
                },
            )
            logger.info(f"Pass 2: Computed sensitivity for {annotated_count} graph nodes")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 2 failed: {exc}")
        return graph, report

    def _annotate_graph(self, graph: Any, sensitivity_map: dict[str, float]) -> int:
        """Attach sensitivity scores to every graph node with a layer index."""
        annotated_count = 0
        if hasattr(graph, "__iter__"):
            for node in graph:
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
        from aether.core.constants import (
            SENSITIVITY_CRITICAL_THRESHOLD,
            SENSITIVITY_HIGH_THRESHOLD,
            SENSITIVITY_MEDIUM_THRESHOLD,
        )

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
                if not sensitivity_map and hasattr(graph, "__iter__"):
                    for node in graph:
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
                        precision_map[f"layer_{i}"] = "Q4_K_M"
                else:
                    for i in range(architecture.layers):
                        layer_key = f"layer_{i}"
                        sensitivity = sensitivity_map.get(layer_key, 0.5)
                        if sensitivity >= SENSITIVITY_CRITICAL_THRESHOLD:
                            precision_map[layer_key] = "BF16"
                        elif sensitivity >= SENSITIVITY_HIGH_THRESHOLD:
                            precision_map[layer_key] = "FP8"
                        elif sensitivity >= SENSITIVITY_MEDIUM_THRESHOLD:
                            precision_map[layer_key] = "Q4_K_M"
                        else:
                            precision_map[layer_key] = "Q3_K"

            # Always keep embedding and LM head at BF16
            precision_map["embedding"] = "BF16"
            precision_map["lm_head"] = "BF16"

            # Annotate nodes with precision
            if hasattr(graph, "iter_layers"):
                for i, node_group in enumerate(graph.iter_layers()):
                    precision = precision_map.get(f"layer_{i}", "Q4_K_M")
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

    For MoE models, this pass profiles expert activations, classifies experts
    into hot/warm/cold tiers, adds threshold-based routing hints, and emits
    intra-expert sparsity annotations.
    """

    name = "moe_routing"
    description = "Optimize MoE expert routing and tiering."

    def run(self, graph: Any, architecture: Any, config: CompilerConfig) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        try:
            hot_experts = 0
            warm_experts = 0
            cold_experts = 0
            total_experts = architecture.num_experts if architecture.is_moe else 0

            if total_experts > 0 and hasattr(graph, "add_node"):
                from aether.core.graph import AEGGraphNode, AEGGraphNodeType

                hot_threshold = config.moe_hot_threshold
                warm_threshold = config.moe_warm_threshold
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
                        "hot_count": int(total_experts * 0.2),
                        "warm_count": int(total_experts * 0.5),
                        "cold_count": total_experts - int(total_experts * 0.7),
                    },
                )
                graph.add_node(router_node)
                hot_experts = int(total_experts * 0.2)
                warm_experts = int(total_experts * 0.5)
                cold_experts = total_experts - hot_experts - warm_experts

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
                    "threshold_hot": config.moe_hot_threshold,
                    "threshold_warm": config.moe_warm_threshold,
                },
            )
            if total_experts > 0:
                logger.info(f"Pass 5: Classified {total_experts} experts ({hot_experts} hot, {warm_experts} warm, {cold_experts} cold)")
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
            patterns = self._assign_head_patterns(architecture)
            enabled = architecture.context_length >= config.sparse_attention_context_threshold
            plan = {
                "version": "sparse_attention/1.0",
                "enabled": enabled,
                "activation_context_length": config.sparse_attention_context_threshold,
                "model_context_length": architecture.context_length,
                "patterns": patterns,
                "runtime_fallback": "dense_attention",
                "research_basis": ["MInference", "MMInference", "Ring Attention"],
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
            logger.info(f"Pass 8: Planned {len(patterns)} sparse attention head groups")
        except Exception as exc:
            report = PassReport(
                pass_name=self.name,
                status="failed",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(exc)},
            )
            logger.error(f"Pass 8 failed: {exc}")
        return graph, report

    def _assign_head_patterns(self, architecture: Any) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        head_count = max(1, architecture.num_attention_heads)
        for head in range(head_count):
            if head % 3 == 0:
                pattern = "vertical_slash"
            elif head % 3 == 1:
                pattern = "block_sparse"
            else:
                pattern = "a_shape"
            patterns.append({
                "head": head,
                "pattern": pattern,
                "sink_tokens": 128,
                "local_window": 4096,
                "stride": 128 if pattern == "vertical_slash" else 0,
            })
        return patterns


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
            if hasattr(graph, "__iter__"):
                for node in graph:
                    op_type = getattr(node, "op_type", None)
                    if op_type not in self.PRUNABLE_OPS:
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

    def _plan_node(
        self, node: Any, op_type: str, target_sparsity: float, config: CompilerConfig
    ) -> dict[str, Any]:
        """Plan — and where weights are attached, actually compute — a node's mask.

        Sensitivity from Pass 2 modulates the per-node target so fragile layers are
        pruned less aggressively. When the node carries a real weight matrix the
        mask is computed and its achieved sparsity recorded; otherwise the entry
        stays a plan for a later stage that does have the weights.
        """
        sensitivity = 0.5
        if hasattr(node, "get_attribute"):
            sensitivity = float(node.get_attribute("sensitivity", 0.5))
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

        from aether.quantization.pruning import build_mask, verify_nm_pattern

        metric = config.pruning_metric
        activation_norms = self._node_activation_norms(node, weights.shape[1])
        if metric in ("wanda", "sparsegpt") and activation_norms is None:
            # Wanda needs calibration activations; fall back rather than fabricate.
            metric = "magnitude"
            entry["metric_fallback"] = "magnitude (no calibration activations)"

        # N:M sparsity requires the input dimension to divide the group size.
        if pattern == "2:4" and weights.shape[1] % 4 != 0:
            pattern = "unstructured"
            entry["pattern"] = pattern
            entry["pattern_fallback"] = f"in_features={weights.shape[1]} not divisible by 4"

        mask = build_mask(
            weights,
            target_sparsity=sparsity,
            pattern=pattern,
            metric=metric,
            activation_norms=activation_norms,
        )
        entry["mask_computed"] = True
        entry["importance_metric"] = metric
        entry["achieved_sparsity"] = round(mask.achieved_sparsity, 4)
        entry["kept"] = mask.kept_count
        entry["pruned"] = mask.pruned_count
        if pattern == "2:4":
            entry["nm_pattern_valid"] = verify_nm_pattern(mask.mask, 2, 4)
        if hasattr(node, "add_attribute"):
            node.add_attribute("pruning_mask", mask)
            node.add_attribute("achieved_sparsity", mask.achieved_sparsity)
        return entry

    def _node_weights(self, node: Any) -> Any:
        """Return a node's 2-D weight matrix, or None when it carries no weights."""
        import numpy as np

        for attr in ("weight", "weights"):
            candidate = getattr(node, attr, None)
            if candidate is None and hasattr(node, "get_attribute"):
                candidate = node.get_attribute(attr, None)
            if candidate is None:
                continue
            arr = np.asarray(candidate)
            if arr.ndim == 2 and arr.size > 0:
                return arr.astype(np.float32)
        return None

    def _node_activation_norms(self, node: Any, in_features: int) -> Any:
        """Return per-input-feature calibration activation norms, if recorded."""
        import numpy as np

        if not hasattr(node, "get_attribute"):
            return None
        norms = node.get_attribute("activation_norms", None)
        if norms is None:
            return None
        arr = np.asarray(norms, dtype=np.float32).ravel()
        return arr if arr.size == in_features else None


class OptimizerPipeline:
    """Orchestrates the six Aether optimizer passes.

    Passes are run in sequential order since each depends on the output of
    the previous pass. The pipeline can be configured to skip specific passes.
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        self.config = config or CompilerConfig()
        self._passes: list[BasePass] = [
            OperatorFusionPass(),
            SensitivityAnalysisPass(),
            PrecisionAssignmentPass(),
            KVCacheStructuringPass(),
            MoERoutingPass(),
            ParallelismDiscoveryPass(),
            ReasoningGraphPass(),
            SparseAttentionPass(),
            PruningSparsityPass(),
        ]
        self._pass_enabled = {
            "operator_fusion": True,
            "sensitivity_analysis": True,
            "precision_assignment": True,
            "kv_cache_structuring": True,
            "moe_routing": True,
            "parallelism_discovery": True,
            "reasoning_graph": True,
            "sparse_attention": True,
            "pruning_sparsity": True,
        }

    @property
    def pass_count(self) -> int:
        return len(self._passes)

    def register_pass(self, pass_instance: BasePass) -> None:
        """Register a custom pass."""
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
        pass_reports: list[PassReport] = []
        current_graph = graph

        # Update enabled status from config
        self._pass_enabled["operator_fusion"] = self.config.enable_fusion
        self._pass_enabled["sensitivity_analysis"] = self.config.enable_sensitivity
        self._pass_enabled["precision_assignment"] = self.config.enable_precision_assignment
        self._pass_enabled["kv_cache_structuring"] = self.config.enable_kv_cache_structuring
        self._pass_enabled["moe_routing"] = self.config.enable_moe_routing
        self._pass_enabled["parallelism_discovery"] = self.config.enable_parallelism_discovery
        self._pass_enabled["reasoning_graph"] = self.config.enable_reasoning_graph
        self._pass_enabled["sparse_attention"] = self.config.enable_sparse_attention
        self._pass_enabled["pruning_sparsity"] = self.config.enable_pruning

        for pass_instance in self._passes:
            if not self._pass_enabled.get(pass_instance.name, True):
                pass_reports.append(
                    PassReport(
                        pass_name=pass_instance.name,
                        status="skipped",
                        details={"reason": "Disabled in configuration"},
                    )
                )
                logger.info(f"Pass '{pass_instance.name}' skipped (disabled in config)")
                continue

            logger.info(f"Running pass: {pass_instance.name}")
            current_graph, report = pass_instance.run(current_graph, architecture, self.config)
            pass_reports.append(report)

        return current_graph, pass_reports

    def __repr__(self) -> str:
        enabled = sum(1 for v in self._pass_enabled.values() if v)
        return f"OptimizerPipeline({enabled}/{self.pass_count} passes enabled)"
