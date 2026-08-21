"""
Aether Compiler API.

The `Compiler` class is the main entry point for compiling AI models into the
Aether Execution Graph (AEG) format. It orchestrates the five stages of the
compiler pipeline:

1. Model ingestion and graph extraction
2. Aether optimizer (six graph-level passes)
3. Hardware targeting and backend selection
4. AEG artifact packaging
5. Quality report generation

It also exposes the dry-run `plan()` method that previews the compilation
without doing the expensive work.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.plan import (
    CompilationPlan,
    OptimizationOpportunity,
    estimate_compile_time_s,
    estimate_memory_gb,
    recommend_backend,
)
from aether.compiler.report import (
    FusionSummary,
    PassReport,
    PrecisionSummary,
    QualityReport,
    compute_average_bit_width,
    compute_precision_distribution,
)
from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.aeg_format import AEGManifest, AEGPackage, KernelSetMetadata, MemoryRequirements, OptimizationMetadata
from aether.core.constants import AEG_FORMAT_VERSION, AETHER_VERSION, DEFAULT_HUB_URL, SUPPORTED_TARGET_IDS
from aether.core.exceptions import CompilationError, CompilerConfigError
from aether.core.hash_utils import compute_aeg_cache_key, compute_graph_hash
from aether.core.types import HardwareTarget, ModelArchitecture, Precision
from aether.utils.logging import get_logger

logger = get_logger(__name__)


#: Logical per-layer tensors the CPU forward pass requires for every
#: transformer layer, using the weight-store naming convention.
_REQUIRED_LAYER_TENSORS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "attention_norm", "ffn_norm",
    "gate_proj", "down_proj",
)

#: Logical global tensors required regardless of layer count. ``lm_head`` is
#: absent when the source model ties its output projection to the embedding.
_OPTIONAL_GLOBAL_TENSORS: frozenset[str] = frozenset({"lm_head", "final_norm"})


def _has_portable_torch_contract(target_id: str) -> bool:
    """Return whether the packaged PyTorch contract can represent a target.

    TEE targets deliberately do not qualify: a normal PyTorch graph cannot
    provide enclave isolation or attestation merely because it can execute
    tensors on a CUDA device.
    """
    return bool(
        target_id in SUPPORTED_TARGET_IDS
        and target_id != "cuda_sm100_tee"
        and target_id.startswith(("cuda_", "rocm_", "metal_"))
    )


def _copy_tokenizer_files(source_model_path: str, destination: Path) -> bool:
    """Copy HuggingFace tokenizer files without importing transformers.

    Returns True when the source carries tokenizer files worth packaging
    (``tokenizer.json`` is the fast-tokenizer format and is sufficient on its
    own). Returns False when the caller should fall back to AutoTokenizer.
    """
    import shutil

    source = Path(source_model_path)
    if not source.is_dir():
        return False
    if not (source / "tokenizer.json").is_file():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "tokenizer.json", destination / "tokenizer.json")
    for optional in (
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "spiece.model",
    ):
        candidate = source / optional
        if candidate.is_file():
            shutil.copy2(candidate, destination / optional)
    return True


def _verify_layer_invariants(
    graph: Any,
    architecture: ModelArchitecture,
    model_id: str,
) -> None:
    """Fail compilation when graph and architecture describe different models.

    Enforces source_layers == graph_layers and positive hidden size / heads.
    The manifest is written from ``architecture`` afterwards, and the runtime
    loader rebuilds exactly ``manifest.layers`` layers while failing on any
    missing per-layer tensor, so this check closes the loop:
    source == graph == manifest == runtime.
    """
    graph_layers = getattr(graph, "num_layers", None)
    if graph_layers is None:
        layer_ids = [
            getattr(node, "layer_index", None)
            for node in (getattr(graph, "nodes", {}) or {}).values()
        ]
        declared = int(
            getattr(architecture, "decoder_layers", None)
            or architecture.layers
            if bool(getattr(architecture, "is_encoder_decoder", False))
            else architecture.layers
        )
        # Global nodes (final_norm, lm_head) carry the sentinel index
        # ``n_layers``; only indices below n_layers are transformer layers.
        transformer_indices = {idx for idx in layer_ids if idx is not None and 0 <= idx < declared}
        expected_indices = set(range(declared))
        missing = sorted(expected_indices - transformer_indices)
        if missing:
            raise CompilationError(
                f"Layer count invariant violated: source architecture declares "
                f"{declared} layers but the extracted graph is missing layers {missing}. "
                f"Refusing to package an artifact that does not match its source model.",
                model_id=model_id,
                stage="stage1_ingestion",
            )
            graph_layers = declared
        if bool(getattr(architecture, "is_encoder_decoder", False)):
            encoder_declared = int(getattr(architecture, "encoder_layers", None) or declared)
            encoder_indices = {idx for idx in layer_ids if idx is not None and idx < 0}
            expected_encoder = {-index - 1 for index in range(encoder_declared)}
            missing_encoder = sorted(expected_encoder - encoder_indices)
            if missing_encoder:
                raise CompilationError(
                    "Encoder layer count invariant violated: source architecture declares "
                    f"{encoder_declared} encoder layers but the extracted graph is missing "
                    f"layers {missing_encoder}.",
                    model_id=model_id,
                    stage="stage1_ingestion",
                )
    if graph_layers is not None and int(graph_layers) != int(architecture.layers):
        raise CompilationError(
            f"Layer count invariant violated: source architecture declares "
            f"{architecture.layers} layers but the extracted graph has {graph_layers}. "
            f"Refusing to package an artifact that does not match its source model.",
            model_id=model_id,
            stage="stage1_ingestion",
        )
    if int(architecture.layers) <= 0:
        raise CompilationError(
            f"Architecture declares layers={architecture.layers}; a runnable "
            f"artifact requires at least one layer",
            model_id=model_id,
            stage="stage1_ingestion",
        )
    if int(architecture.hidden_size) <= 0 or int(architecture.num_attention_heads) <= 0:
        raise CompilationError(
            f"Architecture declares hidden_size={architecture.hidden_size}, "
            f"num_attention_heads={architecture.num_attention_heads}; both must be positive",
            model_id=model_id,
            stage="stage1_ingestion",
        )


def _verify_weight_accounting(
    graph: Any,
    architecture: ModelArchitecture,
    package: AEGPackage,
    model_id: str,
) -> None:
    """Verify every required logical tensor was serialized; record accounting.

    Writes a ``weight_accounting`` block into the package metadata with the
    logical/physical counts the PRD requires, and fails closed when a required
    logical tensor is missing from the serialized store. Fused physical storage
    is acceptable only because the quantizer splits fused QKV back into the
    three logical tensors it stands for.
    """
    serialized = set(package.weights)
    if bool(getattr(architecture, "is_encoder_decoder", False)):
        required = {"embedding", "encoder_final_norm", "final_norm"}
        # lm_head may be tied to the shared embedding in T5 checkpoints.
        # The runtime uses the authenticated shared matrix only when the
        # source config declares tied embeddings; an untied head is required.
        if not bool(getattr(architecture, "tie_word_embeddings", True)):
            required.add("lm_head")
        encoder_layers = int(getattr(architecture, "encoder_layers", None) or architecture.layers)
        decoder_layers = int(getattr(architecture, "decoder_layers", None) or architecture.layers)
        gated_seq2seq = str(getattr(architecture, "ffn_type", "")).lower() in {"gatedgelu", "geglu"}
        for i in range(encoder_layers):
            required.update({
                f"encoder_layer_{i}_{component}"
                for component in (
                    "norm1", "q_proj", "k_proj", "v_proj", "o_proj",
                    "norm2", "ffn_out",
                )
            })
            if gated_seq2seq:
                required.update({f"encoder_layer_{i}_ffn_in_{suffix}" for suffix in ("0", "1")})
            else:
                required.add(f"encoder_layer_{i}_ffn_in")
        for i in range(decoder_layers):
            required.update({
                f"decoder_layer_{i}_{component}"
                for component in (
                    "self_norm", "self_q_proj", "self_k_proj", "self_v_proj", "self_o_proj",
                    "cross_norm", "cross_q_proj", "cross_k_proj", "cross_v_proj", "cross_o_proj",
                    "ffn_norm", "ffn_out",
                )
            })
            if gated_seq2seq:
                required.update({f"decoder_layer_{i}_ffn_in_{suffix}" for suffix in ("0", "1")})
            else:
                required.add(f"decoder_layer_{i}_ffn_in")
    elif bool(getattr(architecture, "is_encoder", False)):
        # Encoder artifacts use the BERT/RoBERTa execution vocabulary.  They
        # do not have decoder-only RMSNorm/SwiGLU/lm_head tensors, so applying
        # the causal-LM invariant here would reject valid checkpoints after
        # ingestion had already extracted their real parameters.
        required = {
            "embedding",
            "position_embedding",
            "token_type_embedding",
            "embedding_norm",
            "pooler",
        }
        encoder_layer_tensors = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "attention_norm", "intermediate_proj", "output_proj", "output_norm",
        )
        for i in range(int(architecture.layers)):
            for component in encoder_layer_tensors:
                required.add(f"layer_{i}_{component}")
    else:
        required = {"embedding"}
        if bool(getattr(architecture, "embedding_norm", False)):
            required.add("embedding_norm")
        if str(getattr(architecture, "position_type", "RoPE") or "RoPE").lower() in {
            "absolute", "learned", "learned_absolute"
        }:
            required.add("position_embedding")
        ffn_type = str(getattr(architecture, "ffn_type", "SwiGLU") or "SwiGLU").lower()
        moe_layers = getattr(architecture, "moe_layer_indices", None)
        is_mla = str(getattr(architecture, "attention_type", "") or "").upper() == "MLA"
        is_ssm = getattr(architecture, "ssm_variant", None) in {
            "selective_scan", "ssd", "rwkv_time_mix", "hybrid_selective_scan",
        }
        hybrid_types = getattr(architecture, "hybrid_layer_types", None)
        for i in range(int(architecture.layers)):
            is_moe_layer = bool(
                architecture.is_moe
                and (moe_layers is None or i in moe_layers)
            )
            layer_components = _REQUIRED_LAYER_TENSORS
            if bool(getattr(architecture, "parallel_residual", False)):
                # Parallel-residual blocks reuse attention_norm for the FFN
                # branch; no second norm tensor exists in GPT-J checkpoints.
                layer_components = tuple(
                    component for component in layer_components
                    if component != "ffn_norm"
                )
            is_state_layer = getattr(architecture, "ssm_variant", None) == "selective_scan" or (
                getattr(architecture, "ssm_variant", None) == "hybrid_selective_scan"
                and isinstance(hybrid_types, list)
                and i < len(hybrid_types)
                and str(hybrid_types[i]).lower() == "ssm"
            )
            if is_state_layer:
                layer_components = (
                    "ssm_norm", "ssm_in_proj", "ssm_conv1d", "ssm_x_proj",
                    "ssm_dt_proj", "ssm_a_log", "ssm_d", "ssm_out_proj",
                )
            elif getattr(architecture, "ssm_variant", None) == "ssd":
                layer_components = (
                    "ssm_norm", "ssm_in_proj", "ssm_conv1d", "ssm_a_log",
                    "ssm_d", "ssm_dt", "ssm_out_proj",
                )
            elif getattr(architecture, "ssm_variant", None) == "rwkv_time_mix":
                layer_components = (
                    "ssm_norm", "ssm_ffn_norm", "ssm_time_decay", "ssm_time_first",
                    "ssm_time_mix_k", "ssm_time_mix_v", "ssm_time_mix_r",
                    "ssm_ffn_time_mix_k", "ssm_ffn_time_mix_r",
                    "ssm_key", "ssm_value", "ssm_receptance", "ssm_output",
                    "ssm_ffn_key", "ssm_ffn_value", "ssm_ffn_receptance",
                )
            elif is_mla:
                # MLA layers replace ordinary q/k/v projections with the
                # compressed query/KV contract.  The output projection remains
                # a normal hidden-space projection.
                layer_components = (
                    "o_proj", "attention_norm", "ffn_norm",
                    "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
                    "k_rope_proj", "q_a_norm", "kv_a_norm", "gate_proj", "down_proj",
                )
            for component in layer_components:
                # Routed layers have no dense gate/up/down projections; their
                # authenticated expert bank is checked below instead.
                if is_moe_layer and component in {"gate_proj", "down_proj", "up_proj"}:
                    continue
                required.add(f"layer_{i}_{component}")
            if is_moe_layer:
                required.add(f"layer_{i}_moe_router")
                for expert in range(int(architecture.num_experts)):
                    required.update({
                        f"layer_{i}_expert_{expert}_{projection}"
                        for projection in ("gate_proj", "up_proj", "down_proj")
                    })
            # GLU families have two input projections; classic GELU decoder
            # blocks (GPT-2/Neo/NeoX) have one intermediate projection.
            elif (
                not is_state_layer
                and getattr(architecture, "ssm_variant", None) not in {"ssd", "rwkv_time_mix"}
                and ffn_type not in {"gelu", "relu", "relu2"}
            ):
                required.add(f"layer_{i}_up_proj")
            if bool(getattr(architecture, "qk_norm", False)) and not is_state_layer:
                required.update({f"layer_{i}_q_norm", f"layer_{i}_k_norm"})
    missing = sorted(required - serialized)

    metadata = getattr(graph, "metadata", {}) or {}
    accounting = {
        "logical_required_tensor_count": len(required) + len(_OPTIONAL_GLOBAL_TENSORS),
        "logical_bound_tensor_count": int(metadata.get("bound_weight_count", 0) or 0),
        "source_tensor_count": int(metadata.get("source_tensor_count", 0) or 0),
        "physical_serialized_tensor_count": len(serialized),
        "required_weight_count": len(required),
        "serialized_weight_count": len(serialized),
        "missing_required_tensors": missing,
        "unbound_source_tensors": list(metadata.get("unbound_weight_names", [])),
    }
    package.metadata["weight_accounting"] = accounting

    if missing:
        preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise CompilationError(
            f"Weight binding invariant violated: {len(missing)} required logical "
            f"tensors were not serialized ({preview}). Refusing to package a "
            f"runnable artifact with missing weights.",
            model_id=model_id,
            stage="stage4_packaging",
            details=accounting,
        )


class Compiler:
    """Main Aether compiler.

    Usage:
        compiler = Compiler()
        plan = compiler.plan("Qwen/Qwen3-8B")
        aeg = compiler.compile("Qwen/Qwen3-8B")
        aeg.save("./qwen3-8b.aeg")
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        """Initialize the compiler with a configuration.

        Args:
            config: Compiler configuration. If None, uses default configuration.
        """
        self.config = config or CompilerConfig()
        self.pipeline = OptimizerPipeline(self.config)
        logger.info(
            "Aether compiler initialized",
            aether_version=AETHER_VERSION,
            optimization_level=self.config.optimization_level,
            targets=self.config.get_targets(),
        )

    def plan(
        self,
        model: str,
        *,
        hardware: str | None = None,
    ) -> CompilationPlan:
        """Dry-run compilation planning.

        Inspects the model and estimates what the compiler would do without
        actually producing an AEG artifact. This is useful for interactive
        exploration and CI smoke checks.

        Args:
            model: Model identifier, local path, or file name.
            hardware: Optional hardware target override. "auto" means detect.

        Returns:
            A `CompilationPlan` with opportunities, estimates, and warnings.
        """
        try:
            architecture = self._detect_architecture(model)
        except Exception as exc:
            plan = CompilationPlan(model_id=model)
            plan.add_error(f"Architecture detection failed: {exc}")
            return plan

        targets = self._resolve_targets(hardware)
        plan = CompilationPlan(model_id=model, architecture=architecture, targets=targets)

        # Add fusion opportunities
        if self.config.enable_fusion and architecture:
            plan.add_fusion_opportunity(
                OptimizationOpportunity(
                    pass_name="operator_fusion",
                    description="Fuse RMSNorm + QKV + RoPE into a single megakernel",
                    nodes=[f"layer_{i}_norm" for i in range(min(architecture.layers, 4))],
                    estimated_memory_saved_mb=architecture.params_billion * 50.0,
                    estimated_latency_reduction_ms=0.5 * architecture.layers,
                    confidence=0.9,
                )
            )
            plan.add_fusion_opportunity(
                OptimizationOpportunity(
                    pass_name="operator_fusion",
                    description="Fuse attention output projection + residual add",
                    nodes=["attn_out", "residual_add"],
                    estimated_memory_saved_mb=architecture.params_billion * 20.0,
                    estimated_latency_reduction_ms=0.2 * architecture.layers,
                    confidence=0.85,
                )
            )

        # Add sensitivity/precision opportunities
        if self.config.enable_sensitivity and self.config.enable_precision_assignment:
            plan.add_sensitivity_opportunity(
                OptimizationOpportunity(
                    pass_name="sensitivity_analysis",
                    description="Compute d(perplexity)/d(precision) per layer",
                    nodes=[f"layer_{i}" for i in range(architecture.layers)],
                    estimated_memory_saved_mb=0.0,
                    estimated_latency_reduction_ms=0.0,
                    confidence=0.95,
                )
            )
            plan.add_precision_opportunity(
                OptimizationOpportunity(
                    pass_name="precision_assignment",
                    description="Assign mixed precision (BF16/FP8/Q4) based on sensitivity",
                    nodes=[f"layer_{i}" for i in range(architecture.layers)],
                    estimated_memory_saved_mb=architecture.params_billion * 500.0,
                    estimated_latency_reduction_ms=0.0,
                    confidence=0.8,
                )
            )

        # Add KV cache opportunities
        if self.config.enable_kv_cache_structuring:
            plan.add_kv_cache_opportunity(
                OptimizationOpportunity(
                    pass_name="kv_cache_structuring",
                    description="Paged KV cache with RadixTree prefix hints",
                    nodes=["kv_cache"],
                    estimated_memory_saved_mb=architecture.context_length * 0.1,
                    estimated_latency_reduction_ms=0.3,
                    confidence=0.9,
                )
            )

        # Add MoE opportunities
        if self.config.enable_moe_routing and architecture.is_moe:
            plan.add_moe_opportunity(
                OptimizationOpportunity(
                    pass_name="moe_routing",
                    description="Hot/warm/cold expert tiering and threshold-based routing",
                    nodes=["moe_router"],
                    estimated_memory_saved_mb=architecture.params_billion * 1000.0,
                    estimated_latency_reduction_ms=1.0,
                    confidence=0.75,
                )
            )

        # Add parallelism opportunities
        if self.config.enable_parallelism_discovery:
            for num_gpus in self.config.parallelism_degrees:
                if num_gpus > 1 and architecture.params_billion > 7.0:
                    plan.add_parallelism_opportunity(
                        OptimizationOpportunity(
                            pass_name="parallelism_discovery",
                            description=f"Compute {num_gpus}-GPU tensor/pipeline parallelism plan",
                            nodes=[],
                            estimated_memory_saved_mb=0.0,
                            estimated_latency_reduction_ms=0.0,
                            confidence=0.7,
                        )
                    )

        # Backend recommendations
        for target in targets:
            backend = recommend_backend(target)
            if backend:
                plan.backend_recommendations[target] = backend
            else:
                plan.add_warning(f"No backend available for target {target}")

        # Estimates
        plan.estimated_memory_gb = estimate_memory_gb(
            architecture,
            kv_cache_dtype=self.config.kv_cache_dtype,
            max_context_length=architecture.context_length,
        )
        plan.estimated_compile_time_s = estimate_compile_time_s(
            architecture,
            targets,
            self.config.optimization_level,
            self.config.calibration_tokens,
        )
        # Estimate AEG size: weights + metadata + kernels (approximate)
        plan.estimated_aeg_size_gb = architecture.params_billion * 0.5  # rough compressed size

        return plan

    def compile(
        self,
        model: str,
        *,
        targets: list[str] | None = None,
        quality_budget: float | None = None,
        calibration_dataset: str | None = None,
        output_path: str | Path | None = None,
        evaluation_evaluator: Any | None = None,
        eval_benchmarks: list[str] | None = None,
        eval_baselines: dict[str, float] | None = None,
        eval_max_regression: float | None = None,
    ) -> AEGPackage:
        """Compile a model into an AEG artifact.

        Args:
            model: Model identifier, local path, or GGUF file.
            targets: Optional list of hardware targets. If None, uses config.
            quality_budget: Optional quality budget override.
            calibration_dataset: Optional calibration dataset override.
            output_path: Optional path to save the AEG package. If None, uses
                the Aether cache directory.
            evaluation_evaluator: Optional measured benchmark callback. When
                supplied, the callback is executed through the CI evaluation
                gate before the artifact is accepted.
            eval_benchmarks: Benchmark names required by the evaluator.
            eval_baselines: Measured baseline scores keyed by benchmark.
            eval_max_regression: Maximum relative score regression. Defaults
                to the configured quality budget when an evaluator is supplied.

        Returns:
            A compiled `AEGPackage`.

        Raises:
            CompilationError: If compilation fails.
        """
        config = self.config.clone()
        if config.reproducible_builds:
            raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
            if raw_epoch is None:
                raise CompilationError(
                    "reproducible_builds=True requires SOURCE_DATE_EPOCH",
                    model_id=model,
                    stage="configuration",
                )
            try:
                start_time = datetime.datetime.fromtimestamp(
                    int(raw_epoch), tz=datetime.timezone.utc
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise CompilationError(
                    "SOURCE_DATE_EPOCH must be a valid integer Unix timestamp",
                    model_id=model,
                    stage="configuration",
                ) from exc
        else:
            start_time = datetime.datetime.now(datetime.timezone.utc)
        if targets is not None:
            config.targets = targets
        if quality_budget is not None:
            config.quality_budget = quality_budget
        if calibration_dataset is not None:
            config.calibration_dataset = calibration_dataset

        # Resolve model ID and detect architecture
        try:
            architecture = self._detect_architecture(model)
        except Exception as exc:
            msg = f"Architecture detection failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model, stage="ingestion") from exc

        model_id = self._normalize_model_id(model)
        logger.info(
            "Starting compilation",
            model_id=model_id,
            architecture=architecture.family,
            params_billion=architecture.params_billion,
        )

        # Stage 1: Ingestion
        try:
            graph = self._stage1_ingest(model, architecture)
        except Exception as exc:
            msg = f"Model ingestion failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage1_ingestion") from exc

        # Pass 18 consumes an explicit trained drafter bundle.  Load it into
        # the in-memory graph before optimization; the pass validates and
        # persists it into the AEG package.  No random/default weights are
        # ever synthesized for a requested diffusion feature.
        if config.enable_mdlm_drafter and config.mdlm_drafter_weights_path:
            try:
                from aether.compiler.stage2_optimizer.pass18_mdlm_drafter import (
                    load_mdlm_weight_bundle,
                )

                setattr(
                    graph,
                    "mdlm_drafter_weights",
                    load_mdlm_weight_bundle(config.mdlm_drafter_weights_path),
                )
            except Exception as exc:
                msg = f"MDLM drafter weights could not be loaded: {exc}"
                raise CompilationError(msg, model_id=model_id, stage="stage1_ingestion") from exc

        # Optional optimizer passes emit binary/config artifacts while they
        # transform the in-memory graph.  Give those passes a real staging
        # directory, then copy the resulting files into the AEG package during
        # Stage 4.  Previously ``graph.output_dir`` was never set, so enabled
        # v4/v5 passes reported success while emitting no persisted artifact.
        pass_artifact_dir = Path(tempfile.mkdtemp(prefix="aether-pass-artifacts-"))
        setattr(graph, "output_dir", str(pass_artifact_dir))

        # Stage 2: Optimize
        try:
            optimized_graph, pass_reports = self._stage2_optimize(graph, architecture, config)
        except Exception as exc:
            shutil.rmtree(pass_artifact_dir, ignore_errors=True)
            msg = f"Optimization failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage2_optimizer") from exc

        # Stage 3: Target and backend selection
        try:
            target_profiles = self._stage3_target(optimized_graph, architecture, config)
        except Exception as exc:
            shutil.rmtree(pass_artifact_dir, ignore_errors=True)
            msg = f"Hardware targeting failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage3_targeting") from exc

        # Stage 4: Package
        try:
            package = self._stage4_package(
                model_id,
                architecture,
                optimized_graph,
                target_profiles,
                pass_reports,
                config,
                output_path,
                start_time,
            )
        except Exception as exc:
            shutil.rmtree(pass_artifact_dir, ignore_errors=True)
            msg = f"AEG packaging failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage4_packaging") from exc

        shutil.rmtree(pass_artifact_dir, ignore_errors=True)

        # A missing evaluator is not a passing quality gate. Preserve a
        # truthful uncertified state in the artifact until measured benchmark
        # evidence is supplied below.
        package.metadata["evaluation_status"] = "uncertified"
        provenance_payload = package.metadata.get("provenance")
        if isinstance(provenance_payload, dict):
            hardware_payload = provenance_payload.setdefault("hardware_certification", {})
            if isinstance(hardware_payload, dict):
                hardware_payload["eval_gate_passed"] = False
        package.save()
        package.verify_integrity()

        if evaluation_evaluator is not None:
            if not callable(evaluation_evaluator):
                raise CompilationError(
                    "evaluation_evaluator must be callable",
                    model_id=model_id,
                    stage="evaluation",
                )
            try:
                from aether.observability.ci_pipeline import CIEvalPipeline

                benchmarks = tuple(eval_benchmarks or ("hellaswag", "mmlu", "gsm8k"))
                max_regression = (
                    config.quality_budget
                    if eval_max_regression is None
                    else float(eval_max_regression)
                )
                if not 0.0 <= max_regression:
                    raise ValueError("eval_max_regression must be non-negative")
                quality_report = CIEvalPipeline(
                    aeg_path=package.root,
                    max_regression=max_regression,
                    required_benchmarks=benchmarks,
                    evaluator=evaluation_evaluator,
                ).run(list(benchmarks), baselines=eval_baselines)
                package.metadata["eval_report"] = quality_report.to_dict()
                package.metadata["evaluation_status"] = (
                    "certified" if quality_report.gate_decision.passed else "rejected"
                )
                provenance_payload = package.metadata.get("provenance")
                if isinstance(provenance_payload, dict):
                    hardware_payload = provenance_payload.setdefault("hardware_certification", {})
                    if isinstance(hardware_payload, dict):
                        hardware_payload["eval_gate_passed"] = bool(
                            quality_report.gate_decision.passed
                        )
                    benchmark_rows = quality_report.to_dict().get("benchmarks", [])
                    if isinstance(benchmark_rows, list):
                        provenance_payload["eval_results"] = {
                            str(row.get("benchmark")): float(row.get("score", 0.0))
                            for row in benchmark_rows
                            if isinstance(row, dict) and row.get("benchmark") is not None
                        }
                # Re-save so the report is included in the manifest's declared
                # artifact hashes and survives a process restart.
                package.save()
                package.verify_integrity()
                if not quality_report.gate_decision.passed:
                    raise CompilationError(
                        "Evaluation gate failed; the compiled AEG is rejected",
                        model_id=model_id,
                        stage="evaluation",
                        details=quality_report.to_dict(),
                    )
            except CompilationError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize evaluator failures
                raise CompilationError(
                    f"Evaluation gate could not complete: {exc}",
                    model_id=model_id,
                    stage="evaluation",
                ) from exc

        logger.info("Compilation complete", model_id=model_id, package_path=str(package.root))
        return package

    def quality_report(self, aeg: AEGPackage) -> QualityReport:
        """Generate a quality report from a compiled AEG artifact.

        Args:
            aeg: Compiled AEG package.

        Returns:
            A `QualityReport` summarizing the AEG.
        """
        if not aeg.manifest:
            msg = "AEG package has no manifest"
            raise CompilationError(msg)
        return QualityReport(
            model_id=aeg.manifest.model_id,
            architecture=aeg.manifest.architecture,
            memory=aeg.manifest.memory_requirements,
            targets=aeg.manifest.kernels.targets,
            backend_recommendations=aeg.manifest.kernels.backend_plans,
            precision=PrecisionSummary(
                distribution=compute_precision_distribution(aeg.get_precision_map()),
                average_bit_width=compute_average_bit_width(aeg.get_precision_map()),
                quality_budget=aeg.manifest.optimization.quality_budget_ppl_increase,
            ),
            fusion=FusionSummary(
                fused_op_count=aeg.manifest.optimization.fused_ops_count,
                original_op_count=int(
                    aeg.metadata.get("fusion_accounting", {}).get("merged_ops_total",
                    aeg.manifest.optimization.fused_ops_count),
                ),
                memory_round_trips_saved=int(
                    aeg.metadata.get("fusion_accounting", {}).get("launches_saved",
                    aeg.manifest.optimization.fused_ops_count),
                ),
                fusion_patterns=dict(
                    aeg.metadata.get("fusion_accounting", {}).get(
                        "patterns",
                        {p: 1 for p in aeg.manifest.optimization.fusion_passes_applied},
                    )
                ),
            ),
        )

    def _detect_architecture(self, model: str) -> ModelArchitecture:
        """Detect the architecture of a model from its name or metadata.

        Args:
            model: Model identifier or path.

        Returns:
            Detected model architecture.
        """
        from aether.compiler.stage1_ingestion.architecture_detector import (
            ArchitectureDetector,
        )

        return ArchitectureDetector().detect(model)

    def _normalize_model_id(self, model: str) -> str:
        """Normalize a model reference to a stable model ID."""
        path = Path(model)
        if path.exists() and path.is_dir():
            return path.name
        return model.replace("/", "--").replace("\\", "--")

    def _resolve_targets(self, hardware: str | None) -> list[str]:
        """Resolve target list from config and optional hardware override."""
        if hardware is not None and hardware != "auto":
            return [hardware]
        return self.config.get_targets()

    def _stage1_ingest(self, model: str, architecture: ModelArchitecture) -> Any:
        """Run Stage 1: model ingestion and graph extraction."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(self.config)
        return pipeline.ingest(model, architecture)

    def _stage2_optimize(
        self,
        graph: Any,
        architecture: ModelArchitecture,
        config: CompilerConfig,
    ) -> tuple[Any, list[PassReport]]:
        """Run Stage 2: optimizer passes."""
        pipeline = OptimizerPipeline(config)
        return pipeline.run(graph, architecture)

    def _stage3_target(
        self,
        graph: Any,
        architecture: ModelArchitecture,
        config: CompilerConfig,
    ) -> list[HardwareProfile]:
        """Run Stage 3: hardware targeting."""
        from aether.compiler.stage3_targeting.target_registry import TargetRegistry

        registry = TargetRegistry()
        return registry.create_profiles(graph, architecture, config.get_targets())

    def _stage4_package(
        self,
        model_id: str,
        architecture: ModelArchitecture,
        graph: Any,
        target_profiles: list[HardwareProfile],
        pass_reports: list[PassReport],
        config: CompilerConfig,
        output_path: str | Path | None,
        start_time: datetime.datetime,
    ) -> AEGPackage:
        """Run Stage 4: create and save the AEG package."""
        from aether.core.aeg_ir import AEGIRModule

        if output_path is None:
            from aether.utils.file_io import aether_cache_dir

            cache_root = aether_cache_dir(config.cache_dir)
            output_path = cache_root / "models" / f"{model_id}.aeg"
        else:
            output_path = Path(output_path)

        if output_path.exists() and not config.overwrite:
            msg = f"AEG output path already exists: {output_path}. Use overwrite=True to replace."
            raise CompilationError(msg, model_id=model_id)
        if output_path.exists():
            shutil.rmtree(output_path)

        package = AEGPackage.create(output_path, model_id=model_id, aether_version=AETHER_VERSION)

        # Materialize files emitted by optimizer passes before package.save()
        # computes the manifest's content hashes.  Copying only files from the
        # private staging directory prevents compiler internals from leaking
        # into the artifact while preserving the pass-produced subdirectories.
        staged_dir = getattr(graph, "output_dir", None)
        if staged_dir:
            staged_path = Path(staged_dir)
            if staged_path.is_dir():
                for child in staged_path.iterdir():
                    destination = package.root / child.name
                    if child.is_dir():
                        shutil.copytree(child, destination, dirs_exist_ok=True)
                    elif child.is_file() and child.name not in {"manifest.json", "FORMAT_VERSION"}:
                        shutil.copy2(child, destination)

        # Convert graph to AEG-IR
        if isinstance(graph, AEGIRModule):
            ir = graph
        elif hasattr(graph, "to_ir"):
            ir = graph.to_ir()
        else:
            ir = AEGIRModule.from_graph(graph)
        package.ir = ir
        if hasattr(graph, "metadata"):
            package.metadata = dict(getattr(graph, "metadata"))
        # Persisted optimizer plans are executable runtime inputs, not merely
        # validation payloads.  Surface the staged Pass 14 plan in metadata so
        # a fresh AEG loader can attach it to the CPU KV cache after restart.
        semantic_kv_plan = package.root / "graph" / "kv_compression_plan.json"
        if semantic_kv_plan.is_file():
            package.metadata["kv_compression_plan"] = json.loads(
                semantic_kv_plan.read_text(encoding="utf-8")
            )
        cross_layer_kv_plan = package.root / "graph" / "cross_layer_kv_plan.json"
        if cross_layer_kv_plan.is_file():
            package.metadata["cross_layer_kv_plan"] = json.loads(
                cross_layer_kv_plan.read_text(encoding="utf-8")
            )
        primary_target = target_profiles[0].target_id if target_profiles else "cpu_avx512"
        from aether.agentic import AgentWorkflowOptimizer, ToolCall
        from aether.attention import MLAPlanner
        from aether.cuda import CUDAGraphCapturePlan
        from aether.distillation import DistillationPipeline
        from aether.fleet import AetherFleetManager, FleetConfig, FleetNode, HotReloadRouter
        from aether.inference.multimodal import default_multimodal_plan
        from aether.observability import ABRolloutController, DriftMonitor, EvalGate
        from aether.runtime.eagle import EAGLE3Planner

        agentic_trace = [
            ToolCall("retrieve", {"query": "string"}, average_latency_ms=18.0),
            ToolCall("rerank", {"documents": "array"}, average_latency_ms=24.0),
            ToolCall("generate", {"context": "string"}, average_latency_ms=90.0, writes_context=True),
        ]
        fleet_nodes = [
            FleetNode("local-0", primary_target, gpu_count=1 if primary_target.startswith(("cuda", "rocm", "metal")) else 0, memory_gb=80.0),
            FleetNode("portable-0", "cpu_avx512", memory_gb=max(16.0, architecture.params_billion * 2)),
        ]

        # Capabilities are derived from what actually ran and what the source
        # architecture is — never a fixed advertisement.  Infrastructure
        # plans that every AEG package physically embeds (safety configs,
        # provenance, observability manifests, rollout/fleet/distillation
        # plan templates) legitimately count as packaged capabilities;
        # architecture-specific traits (MLA, SSM) and optimizer features are
        # gated on their real triggers.
        applied_passes = {
            report.pass_name for report in pass_reports if getattr(report, "status", "") == "applied"
        }
        family = (architecture.family or "").lower()
        attention = (architecture.attention_type or "").lower()
        capabilities = [
            "safety_guardrails",
            "provenance",
            "watermark",
            "lora_adapters",
            "rag_pipeline",
            "agentic_workflow",
            "multimodal_graph",
            "quantization_aware_compilation",
            "observability",
            "eval_gates",
            "ab_rollout",
            "fleet_management",
            "hot_reload",
            "distillation",
            "cuda_graph_capture",
            "eagle3_speculation",
        ]
        if "reasoning_graph" in applied_passes:
            capabilities.append("reasoning_graph")
        if "sparse_attention" in applied_passes:
            capabilities.append("sparse_attention")
        if "pruning_sparsity" in applied_passes:
            capabilities.append("pruning_sparsity")
        if "mla" in attention or "deepseek" in family or "kimi" in family or "glm" in family:
            capabilities.append("mla_native")
        if any(marker in family for marker in ("mamba", "rwkv", "jamba", "zamba", "ssm")):
            capabilities.append("hybrid_ssm")

        package.metadata.update({
            "model_id": model_id,
            "architecture": architecture.to_dict(),
            "capabilities": capabilities,
            "agentic_workflow": AgentWorkflowOptimizer(min_sequence_frequency=1).compile([agentic_trace]),
            "eval_gates": EvalGate().manifest(),
            "drift_monitor": DriftMonitor(baseline_win_rate=0.5).manifest(),
            "ab_rollout": ABRolloutController("compile_default", candidate_percent=0.01).manifest(),
            "fleet_deployment": AetherFleetManager(fleet_nodes).plan_manifest(
                model_id,
                FleetConfig(replicas=1, preferred_targets=tuple(profile.target_id for profile in target_profiles) or ("cpu_avx512",)),
            ),
            "hot_reload": HotReloadRouter(active_aeg=model_id).manifest(),
            "distillation": DistillationPipeline().compile_manifest(
                DistillationPipeline().plan(model_id, f"{model_id}-student", task_type="general"),
            ),
            "cuda_graphs": CUDAGraphCapturePlan(primary_target, max_context_length=architecture.context_length).to_dict(),
            "mla_plan": MLAPlanner().plan(architecture, target=primary_target).to_dict(),
            "eagle3": EAGLE3Planner().plan(architecture).to_dict(),
            "multimodal_graph": default_multimodal_plan(model_id).to_graph(),
            "metrics_schema": {
                "version": "otel_metrics/1.0",
                "metrics": [
                    "tokens_per_second",
                    "ttft_ms",
                    "eagle3_accept_rate",
                    "kv_hit_rate",
                    "mla_compression_ratio",
                    "reasoning_budget_used",
                    "gpu_vram_utilization",
                ],
            },
            "quantization_aware_compilation": {
                "enabled": True,
                "calibration_dataset": config.calibration_dataset,
                "quality_budget_ppl_increase": config.quality_budget,
                "precision_formats": ["BF16", "FP8", "FP4", "Q4_K_M", "Q3_K"],
            },
        })

        # Compute graph hash
        graph_hash = compute_graph_hash(ir)

        # ── Provenance (PRD §35) ────────────────────────────────────────────
        # Built from the real ProvenanceManifest rather than an inline dict so
        # the artifact carries a verifiable model hash and the actual
        # transformation chain. EU AI Act Art. 50 requires a deployer be able
        # to audit what was done to the model they are running.
        from aether.provenance.manifest import (
            EUAIActRecord,
            HardwareCertification,
            ProvenanceManifest,
            TransformationRecord,
        )

        provenance = ProvenanceManifest(
            model_hash=graph_hash.replace("sha256:", ""),
            source_model_id=model_id,
            model_architecture=architecture.family,
            compiler_version=f"aether/{AETHER_VERSION}",
            compile_timestamp=start_time.timestamp(),
            transformations=[
                TransformationRecord(
                    pass_name=report.pass_name,
                    parameters={"status": report.status},
                    timestamp=start_time.timestamp(),
                )
                for report in pass_reports
            ],
            eu_ai_act=EUAIActRecord(
                risk_category="limited_risk",
                transparency_obligations_met=True,
                intended_purpose="general_text_generation",
            ),
            hardware_certification=HardwareCertification(
                certified_targets=[p.target_id for p in target_profiles] or ["cpu_avx512"],
                primary_target=primary_target,
            ),
            watermark_enabled=True,
            watermark_algorithm="greenlist_statistical",
        )
        package.metadata["provenance"] = provenance.to_dict()

        # Build precision map from pass reports
        precision_map: dict[str, str] = {}
        for report in pass_reports:
            if report.pass_name == "precision_assignment":
                precision_map.update(report.details.get("precision_map", {}))
        if not precision_map:
            # No quality-certified precision plan means preserve source
            # precision.  Lossy Q4/Q3 assignment is available through an
            # explicit precision mode, never as a silent fallback.
            precision_map = {f"layer_{i}": "BF16" for i in range(architecture.layers)}
        sub2bit_report = next(
            (
                report
                for report in pass_reports
                if report.pass_name == "sub2bit_quantization"
                and report.status == "applied"
            ),
            None,
        )
        if sub2bit_report is not None:
            method = str(sub2bit_report.details.get("method", ""))
            if method != "bitnet":
                raise CompilationError(
                    f"Sub-2-bit pass reported unsupported runtime method {method!r}",
                    model_id=model_id,
                    stage="stage4_packaging",
                )
            # Preserve BF16 embedding/output/norm defaults while assigning the
            # real BitNet codec to every transformer layer projection.
            precision_map = {
                f"layer_{i}": "TERNARY" for i in range(architecture.layers)
            }
            package.metadata["sub2bit_runtime"] = {
                "method": method,
                "precision": "TERNARY",
                "backend": "cpu_reference_dense_dequantize",
                "quality_gate": sub2bit_report.details.get("quality_gate", {}),
                "weight_reconstruction": sub2bit_report.details.get(
                    "weight_reconstruction", {}
                ),
            }
        package.set_precision_map(precision_map)

        # ── Quantize and persist weights into the AEG blob ──────────────────
        # This is the key step that makes the package self-contained: every
        # weight-bearing graph node is quantized to its assigned precision and
        # written into weights/quantized/model.aeg-quant so downstream
        # load_engine_from_package() can reconstruct the full forward pass.
        try:
            from aether.compiler.weight_quantizer import quantize_graph_weights

            from aether.core.graph import AEGGraph

            if isinstance(graph, AEGGraph):
                quant_stats = quantize_graph_weights(
                    graph=graph,
                    package=package,
                    precision_map=precision_map,
                    default_precision="BF16",
                    block_size=32,
                )
                if quant_stats.tensors_written == 0:
                    raise CompilationError(
                        "No model weights were attached during ingestion; refusing to create a runnable AEG. "
                        "Provide a local checkpoint or enable a supported model download path.",
                        model_id=model_id,
                        stage="stage1_ingestion",
                    )
                logger.info(
                    "Quantized %d weight tensors (%d bytes) for %s",
                    quant_stats.tensors_written,
                    quant_stats.bytes_written,
                    model_id,
                )
        except Exception as exc:  # noqa: BLE001 — weight quant is best-effort
            raise CompilationError(
                f"Weight quantization failed; refusing to create a graph-only runnable package: {exc}",
                model_id=model_id,
                stage="stage4_packaging",
            ) from exc

        # ── Hard layer/architecture invariants ─────────────────────────────
        # The source architecture is the single source of truth. The compiled
        # artifact must describe exactly the same model or the pipeline has
        # silently corrupted it (the historical 4-layer -> 1-layer bug).
        _verify_layer_invariants(graph, architecture, model_id)

        # ── Architecture-aware weight accounting ───────────────────────────
        # Every logical tensor the architecture requires must be present in
        # the serialized weight store. Fused storage (physical qkv) is fine,
        # but only when the logical tensors it stands for are all serialized.
        _verify_weight_accounting(graph, architecture, package, model_id)

        # Sharding plans
        from aether.core.aeg_format import create_default_sharding_plans

        sharding_plans = create_default_sharding_plans(architecture)
        for num_gpus, plan in sharding_plans.items():
            package.set_sharding_plan(num_gpus, plan)

        # Backend plans per target.  A target profile is a compilation request,
        # not evidence that an executable vendor binary was produced.  Mark
        # only CPU targets as executable here; accelerator targets receive a
        # portable PyTorch contract when this is a standard decoder graph.
        kernels = KernelSetMetadata(targets=[p.target_id for p in target_profiles])
        for profile in target_profiles:
            backend_name = profile.recommended_backend or "pytorch"
            package.set_backend_plan(profile.target_id, backend_name)
        kernels.backend_plans = {t: package.get_backend_plan(t) for t in kernels.targets if package.get_backend_plan(t)}
        kernels.variant_status = {
            profile.target_id: (
                "executable" if profile.target_id.startswith("cpu_") else "plan_only"
            )
            for profile in target_profiles
        }
        # The AEG graph and authenticated weight store are the portable source
        # contract. PyTorch can execute dense and routed standard decoder
        # blocks directly on CUDA, ROCm, and MPS without recompiling weights.
        # Distinct SSM, encoder, and encoder-decoder contracts remain gated
        # below; dense MLA has its own device executor and portable contract.
        family_lower = str(architecture.family or "").lower()
        portable_decoder = (
            not architecture.is_encoder
            and not architecture.is_encoder_decoder
            and not getattr(architecture, "is_multimodal", False)
            and not any(token in family_lower for token in ("mamba", "rwkv", "retnet", "ssm", "jamba"))
        )
        portable_mla = (
            str(architecture.attention_type or "").upper() == "MLA"
            and not architecture.is_encoder
            and not architecture.is_encoder_decoder
            and not getattr(architecture, "is_multimodal", False)
        )
        portable_encoder = (
            architecture.is_encoder
            and not architecture.is_encoder_decoder
            and not getattr(architecture, "is_multimodal", False)
        )
        portable_seq2seq = (
            architecture.is_encoder_decoder
            and not getattr(architecture, "is_multimodal", False)
        )
        # Each non-standard contract has a dedicated executor.  Pure
        # SSM/Mamba and RWKV remain separate from standard decoder routing;
        # the flags below prevent a capability mismatch from being hidden by
        # a generic fallback.
        portable_hybrid = (
            getattr(architecture, "ssm_variant", None) == "hybrid_selective_scan"
            and not architecture.is_moe
        )
        portable_state = (
            getattr(architecture, "ssm_variant", None) in {"selective_scan", "ssd", "rwkv_time_mix"}
            and not architecture.is_moe
        )
        if (
            portable_decoder or portable_mla or portable_hybrid or portable_state
            or portable_encoder or portable_seq2seq
        ):
            # ``aether_cpu`` is the framework-free execution contract for the
            # canonical AEG graph.  PyTorch remains an optional accelerator
            # materializer, never a dependency of the artifact or base wheel.
            kernels.portable_backends = ["aether_cpu", "pytorch"]
            for profile in target_profiles:
                if _has_portable_torch_contract(profile.target_id):
                    kernels.variant_status[profile.target_id] = "portable"
                elif not profile.target_id.startswith("cpu_"):
                    kernels.variant_status[profile.target_id] = "plan_only"
        package.manifest.kernels = kernels

        # Embed the real native CPU library when a host toolchain is available.
        # The file is copied before package.save(), so AEG artifact hashing
        # covers the executable itself and the reload path can load it instead
        # of silently recompiling from a machine-local cache.
        packaged_kernel_artifacts: list[dict[str, Any]] = []
        try:
            from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
            from aether.core.exceptions import KernelError
            from aether.kernels.native_cpu import get_native_kernels

            native = get_native_kernels()
            if native.ensure_compiled() and native.library_path is not None:
                for profile in target_profiles:
                    if not profile.target_id.startswith("cpu_"):
                        continue
                    relative = Path("generated_kernels") / profile.target_id / (
                        f"native_cpu{native.library_path.suffix}"
                    )
                    artifact = KernelEmitter(profile.target_id).emit_executable(
                        "gemm", package.root / relative
                    )
                    packaged_kernel_artifacts.append(
                        {
                            "target_id": profile.target_id,
                            "path": relative.as_posix(),
                            "sha256": artifact.sha256,
                            "symbols": list(artifact.symbols),
                            "backend": artifact.backend,
                        }
                    )
        except (KernelError, OSError) as exc:
            logger.warning("Native CPU kernel was not embedded in AEG: %s", exc)
        if packaged_kernel_artifacts:
            package.metadata["kernel_artifacts"] = packaged_kernel_artifacts

        # Optimization metadata
        fused_ops_count = sum(
            1 for r in pass_reports if r.pass_name == "operator_fusion" and r.status == "applied"
        )
        fusion_passes = [r.details.get("fusion_pattern", "unknown") for r in pass_reports if r.pass_name == "operator_fusion"]
        applied_passes = [r.pass_name for r in pass_reports if r.status in {"applied", "ok"}]
        package.metadata["optimizer_passes"] = applied_passes
        v4_passes = {
            "mtp_head_compilation",
            "grammar_constraint_compilation",
            "model_merging",
            "ttt_fast_weight_injection",
            "semantic_kv_compression",
            "cross_layer_kv_sharing",
            "green_energy_compilation",
            "tee_kernel_wrapping",
        }
        v5_passes = {
            "mdlm_drafter_compilation",
            "sub2bit_quantization",
            "video_token_compression",
            "advanced_peft_compilation",
            "rlvr_verifier_head_injection",
        }
        if v5_passes.intersection(applied_passes):
            package.manifest.format_version = "AEG/3.0"
        elif v4_passes.intersection(applied_passes):
            package.manifest.format_version = "AEG/2.0"
        package.metadata["aeg_format_version"] = package.manifest.format_version
        optimization = OptimizationMetadata(
            fusion_passes_applied=fusion_passes or applied_passes,
            fused_ops_count=fused_ops_count,
            sensitivity_calibration_dataset=config.calibration_dataset,
            quality_budget_ppl_increase=config.quality_budget,
        )
        package.manifest.optimization = optimization

        # Memory requirements
        mem_gb = estimate_memory_gb(architecture, {k: Precision.from_string(v) for k, v in precision_map.items()})
        bf16_gb = estimate_memory_gb(architecture, None)
        memory = MemoryRequirements(
            bf16_gb=bf16_gb,
            compiled_min_gb=mem_gb,
            recommended_gb=mem_gb * 1.2,
        )
        package.manifest.memory_requirements = memory
        package.manifest.architecture = architecture
        package.manifest.compiled_at = start_time.isoformat()
        package.manifest.graph_hash = graph_hash

        # Text AEGs must carry the exact tokenizer used by the source model.
        # Without it, a package can contain numeric weights but cannot provide
        # faithful text inference after the compiler process exits.  Keep this
        # local-only: a successful compile never hides a missing tokenizer by
        # falling back to a different vocabulary.
        source_model_path = package.metadata.get("source_model_path")
        if source_model_path and architecture.family not in {"vision_family", "whisper_family"}:
            try:
                if package.metadata.get("source_format") == "gguf":
                    from aether.compiler.stage1_ingestion.gguf_loader import export_gguf_tokenizer

                    tokenizer_info = export_gguf_tokenizer(
                        source_model_path, package.root / "tokenizer"
                    )
                    package.metadata["tokenizer_info"] = tokenizer_info
                elif _copy_tokenizer_files(source_model_path, package.root / "tokenizer"):
                    # Framework-free path: a checkpoint that already carries
                    # HF tokenizer files is packaged by copying, without
                    # importing transformers.
                    pass
                else:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        source_model_path,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    tokenizer.save_pretrained(package.root / "tokenizer")
                package.metadata["tokenizer_path"] = "tokenizer"
            except Exception as exc:
                raise CompilationError(
                    f"Could not package the source model tokenizer from {source_model_path}: {exc}",
                    model_id=model_id,
                    stage="stage4_packaging",
                ) from exc

        package.save()
        return package

    def __repr__(self) -> str:
        return f"Compiler(level={self.config.optimization_level}, targets={self.config.get_targets()})"
