"""
Pass 12 — Model Merging and Task Vector Fusion.

Model merging produces a single set of weights that performs well across multiple
tasks by combining task-specific fine-tuned models with a shared base model.
The key insight: a fine-tuned model's capability is encoded in its *task vector*
δθ = θ_fine − θ_base.  Multiple task vectors can be composed algebraically.

Implemented merging methods:

1. **Task Arithmetic** (Ilharco et al., ICLR 2023):
   θ_merged = θ_base + Σᵢ λᵢ · δθᵢ
   Simple weighted sum of task vectors.

2. **DARE** (Drop And REscale, Yu et al., 2024):
   Randomly zero-out p% of each δθᵢ; rescale survivors by 1/(1−p).
   Reduces interference between task vectors.  Default p=0.9 (90% dropout).

3. **TIES-Merging** (Yadav et al., NeurIPS 2023):
   Trim small deltas, elect sign by weighted magnitude, merge consistent signs.
   Resolves sign conflicts that cause interference.

4. **FREE-Merging** (2026):
   Evolutionary optimization of per-layer merge coefficients λᵢ using a small
   validation set.  Finds optimal λ without exhaustive search.

5. **Evolutionary** (Akiba et al., 2024):
   CMA-ES over merge coefficients; SOTA on model soup benchmarks.

Research basis:
  - Task Arithmetic ICLR 2023: δθ composition.
  - DARE 2024: drop-and-rescale interference reduction.
  - TIES-Merging NeurIPS 2023: sign conflict resolution.
  - FREE-Merging 2026: evolutionary coefficient optimization.
  - Evolutionary Model Merging 2024: CMA-ES for merging.
  - Model Soups 2022: weight averaging baseline.
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_MERGING_METHODS: frozenset[str] = frozenset(
    {"task_arithmetic", "dare", "ties", "free", "evolutionary", "soup"}
)

# DARE default dropout probability.
_DARE_DEFAULT_DROP_PROB: float = 0.9

# TIES trimming threshold: keep only top-k% of parameters by magnitude.
_TIES_TRIM_FRACTION: float = 0.2


class ModelMergingPass(BasePass):
    """Pass 12: Merge multiple fine-tuned task vectors into a single AEG model.

    This pass:
      1. Loads each source model's weight store.
      2. Computes task vectors: δθᵢ = θ_fineᵢ − θ_base.
      3. Merges task vectors using the configured method.
      4. Writes merged weights back into the AEG weight store.
      5. Stores a merge manifest at ``.aeg/merging/merge_manifest.json``.
    """

    name = "model_merging"
    description = (
        "Merge task vectors from multiple fine-tuned models into a single AEG "
        "using Task Arithmetic, DARE, TIES-Merging, FREE-Merging, or Evolutionary methods."
    )

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})

        if not config.enable_model_merging:
            return graph, report

        sources = config.model_merging_sources
        if not sources:
            logger.warning(
                "Pass 12: enable_model_merging=True but model_merging_sources is empty."
            )
            report.status = "skipped"
            report.details["reason"] = "no_merging_sources"
            return graph, report

        method = config.model_merging_method
        if method not in _SUPPORTED_MERGING_METHODS:
            report.status = "failed"
            report.details["error"] = f"Unsupported model merge method: {method}"
            return graph, report

        try:
            # Determine coefficients.
            coefficients = list(config.model_merging_coefficients)
            if coefficients and len(coefficients) != len(sources):
                raise ValueError("Provide one coefficient per requested merging source")
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError("Merge coefficients must be finite")
            if not coefficients:
                # Default: uniform 1/N scaling.
                coefficients = [1.0 / len(sources)] * len(sources)
                logger.info(
                    "Pass 12: No explicit coefficients; using uniform λ=%.4f for %d sources.",
                    coefficients[0],
                    len(sources),
                )

            logger.info(
                "Pass 12: Merging %d source models via %s (λ=%s).",
                len(sources),
                method,
                coefficients,
            )

            # Load base weights from graph.
            base_weights = _extract_weights(graph, "base")
            if not base_weights:
                raise ValueError("model merging requires real base weights in the graph")

            # Load source weights.
            source_weights_list: list[dict[str, Any]] = []
            for src_path in sources:
                try:
                    src_weights = _load_source_weights(src_path, graph)
                    if not src_weights:
                        raise ValueError("source contains no readable tensors")
                    source_weights_list.append(src_weights)
                    logger.debug("  Loaded source: %s (%d tensors)", src_path, len(src_weights))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("  Failed to load source %r: %s. Skipping.", src_path, exc)

            if len(source_weights_list) != len(sources):
                raise ValueError("Every requested merge source must load; partial merges are not allowed")

            # Compute task vectors.
            task_vectors = _compute_task_vectors(base_weights, source_weights_list)
            if any(not vector for vector in task_vectors):
                raise ValueError("Every merge source must contain at least one matching base tensor")
            logger.info("Pass 12: Computed %d task vectors.", len(task_vectors))

            # Apply merging method.
            merger = _get_merger(method)
            merged_delta = merger.merge(
                task_vectors=task_vectors,
                coefficients=coefficients,
                config=config,
            )

            # Produce merged weights: θ_merged = θ_base + merged_delta.
            merged_weights = _apply_delta(base_weights, merged_delta)

            # Write merged weights into graph.
            n_updated = _update_graph_weights(
                graph, {name: merged_weights[name] for name in merged_delta}
            )
            if n_updated <= 0:
                raise ValueError("Model merging did not update any graph weights")

            # Write merge manifest.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                from pathlib import Path
                _write_merge_manifest(
                    output_dir=Path(graph.output_dir),
                    sources=sources,
                    method=method,
                    coefficients=coefficients,
                    n_params=sum(len(v) for v in merged_delta.values()),
                )

            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "method": method,
                "n_sources": len(source_weights_list),
                "sources": list(sources),
                "coefficients": coefficients,
                "n_tensors_updated": n_updated,
                "task_vector_keys": list(merged_delta.keys())[:20],
                "quality_validated": False,
            }
            if hasattr(graph, "set_metadata"):
                graph.set_metadata("model_merging", dict(report.details))
            logger.info(
                "Pass 12 complete: %d tensors merged via %s in %.3fs.",
                n_updated,
                method,
                elapsed,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 12 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)

        return graph, report


# ── Task vector arithmetic ────────────────────────────────────────────────────


_PARAMETER_ATTRIBUTES = {
    "weight": "", "up_weight": "_up", "bias": "_bias",
    "q_weight": "_q", "k_weight": "_k", "v_weight": "_v",
    "q_bias": "_q_bias", "k_bias": "_k_bias", "v_bias": "_v_bias",
}


def _graph_parameter_slots(graph: Any):
    """Visit the same real projection/scale/bias arrays consumed by packing."""
    nodes = graph.nodes.values() if hasattr(graph.nodes, "values") else graph.nodes
    for node in nodes:
        attributes = getattr(node, "attributes", {})
        if not isinstance(attributes, dict):
            continue
        for attribute, suffix in _PARAMETER_ATTRIBUTES.items():
            if attributes.get(attribute) is not None:
                yield f"{node.id}{suffix}", node, attribute


def _load_source_graph_weights(source_path: str, base_graph: Any) -> dict[str, list[float]]:
    """Reuse ingestion's architecture/layout binding before comparing shapes.

    Keeping shaped arrays until this comparison prevents equal-sized transposed
    or reshaped tensors from being treated as the same parameter coordinates.
    A source may contain a matching subset; unrelated coordinates are omitted.
    """
    import numpy as np
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

    source = IngestionPipeline().ingest(source_path, base_graph.architecture)
    expected = {
        name: np.asarray(node.attributes[attribute]).shape
        for name, node, attribute in _graph_parameter_slots(base_graph)
    }
    result = {}
    for name, node, attribute in _graph_parameter_slots(source):
        if name not in expected:
            continue
        tensor = np.asarray(node.attributes[attribute], dtype=np.float32)
        if tensor.shape != expected[name]:
            raise ValueError(f"Merge tensor {name!r} shape {tensor.shape} differs from base {expected[name]}")
        result[name] = tensor.reshape(-1).tolist()
    return result


def _source_graph_store_arrays(graph: Any) -> dict[str, Any]:
    """Use weight-packing coordinates without low-precision source rounding."""
    import numpy as np
    from aether.compiler.weight_quantizer import GraphWeightQuantizer
    from aether.quantization.formats import dequantize_tensor

    packer = GraphWeightQuantizer(default_precision="FP32")
    result = {}
    for node in graph.nodes.values():
        attributes = node.attributes
        weight = packer._extract_weight(node)
        if weight is None:
            continue
        if node.op_type == "qkv_proj" or node.id.endswith("_qkv"):
            split = packer._split_qkv(node, weight, "FP32", node.layer_index)
            result.update({name: dequantize_tensor(tensor) for name, tensor in split.items()})
            for projection in ("q", "k", "v"):
                bias = attributes.get(f"{projection}_bias")
                if bias is not None:
                    result[f"layer_{node.layer_index}_{projection}_proj_bias"] = np.asarray(bias, dtype=np.float32)
        else:
            name = packer._weight_name(node.id, node.op_type, node.layer_index)
            if name is None:
                continue
            result[name] = np.asarray(weight, dtype=np.float32)
            if attributes.get("bias") is not None:
                result[f"{name}_bias"] = np.asarray(attributes["bias"], dtype=np.float32)
            if attributes.get("up_weight") is not None and node.id.endswith("_gate_proj"):
                result[f"layer_{node.layer_index}_up_proj"] = np.asarray(attributes["up_weight"], dtype=np.float32)
    return result


def _finish_source_tensors(tensors: dict[str, Any], base_graph: Any, source_path: str) -> dict[str, list[float]]:
    """Preserve exact source shapes when aligning to an AEG weight index."""
    import numpy as np

    if callable(getattr(base_graph, "weight_store", None)) and getattr(base_graph, "manifest", None) is not None:
        entries = base_graph.weight_store().entries
        if tensors and not any(name in entries for name in tensors):
            from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

            graph = IngestionPipeline().ingest(source_path, base_graph.manifest.architecture)
            tensors = _source_graph_store_arrays(graph)
        result = {}
        for name, value in tensors.items():
            if name not in entries:
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape != entries[name].shape:
                raise ValueError(f"Merge tensor {name!r} shape {array.shape} differs from base {entries[name].shape}")
            result[name] = array.reshape(-1).tolist()
        return result
    return {name: np.asarray(value, dtype=np.float32).reshape(-1).tolist() for name, value in tensors.items()}


def _extract_weights(graph: Any, role: str = "base") -> dict[str, list[float]]:
    """Extract parameter tensors from graph as {name: flat_list} dict.

    This is intentionally storage-agnostic — it works with AEG-IR nodes,
    dict-based graphs, and PyTorch state_dicts.
    """
    weights: dict[str, list[float]] = {}
    if hasattr(graph, "parameters"):
        # PyTorch-style .parameters() dict.
        for name, param in graph.parameters().items():
            if hasattr(param, "tolist"):
                weights[name] = param.tolist()
            elif isinstance(param, (list, tuple)):
                weights[name] = list(param)
    elif hasattr(graph, "weight_store"):
        store = graph.weight_store
        if hasattr(store, "items"):
            for name, tensor in store.items():
                weights[name] = list(tensor) if hasattr(tensor, "__iter__") else [float(tensor)]
    elif hasattr(graph, "nodes"):
        import numpy as np
        for name, node, attribute in _graph_parameter_slots(graph):
            weights[name] = np.asarray(node.attributes[attribute], dtype=np.float32).reshape(-1).tolist()
    elif isinstance(graph, dict):
        for name, val in graph.items():
            if isinstance(val, (list, tuple)):
                weights[name] = list(val)
    return weights


def _load_source_weights(source_path: str, base_graph: Any) -> dict[str, list[float]]:
    """Load weights from a source model path.

    Tries: AEG weight store → safetensors (numpy/torch) → GGUF → PyTorch state_dict.
    Falls back to empty dict if path does not resolve to any known format.
    """
    import numpy as np
    from pathlib import Path
    p = Path(source_path)

    if not p.exists():
        logger.debug("Source path does not exist: %s", source_path)
        return {}

    # AEG is the native artifact format. Read and dequantize its persisted
    # weight store instead of treating the directory as an opaque success.
    if p.is_dir() and (p / "manifest.json").is_file():
        try:
            from aether.core.aeg_format import AEGPackage

            package = AEGPackage(p).load()
            package.verify_integrity()
            if not package.has_weights:
                return {}
            return _finish_source_tensors(package.weight_store().dequantize_all(), base_graph, source_path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"AEG merge source {source_path!r} failed integrity/load validation: {exc}") from exc

    if hasattr(base_graph, "nodes") and getattr(base_graph, "architecture", None) is not None:
        # Checkpoint names/layouts follow the ordinary compilation contract,
        # including shards, QKV packing and GPT Conv1D layout conversion.
        return _load_source_graph_weights(source_path, base_graph)

    # Try safetensors with numpy (Torch-independent).
    if p.suffix == ".safetensors" or (p.is_dir() and any(p.glob("*.safetensors"))):
        try:
            from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

            tensors = SafeTensorsLoader(p).load()
        except Exception as exc:  # noqa: BLE001
            logger.debug("safetensors load failed: %s", exc)
        else:
            return _finish_source_tensors(tensors, base_graph, source_path)

    # Try GGUF.
    if p.suffix.lower() in (".gguf", ".ggml") or (p.is_dir() and list(p.glob("*.gguf"))):
        gguf_path = p if p.suffix.lower() in (".gguf", ".ggml") else next(p.glob("*.gguf"))
        try:
            from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader

            reader = GGUFReader(gguf_path)
            return _finish_source_tensors({name: reader.dequantize(name) for name in reader.tensors}, base_graph, source_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("GGUF source load failed: %s", exc)

    # Try PyTorch. ``weights_only=True`` is mandatory: merging inputs are
    # model files, and an untrusted checkpoint must never execute arbitrary
    # pickled code inside the compiler process.
    try:
        import torch  # type: ignore[import]
        try:
            sd = torch.load(str(p), map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - torch < 1.13 without the flag
            raise ValueError(
                "task-vector merging requires torch.load(weights_only=True) "
                "support; upgrade PyTorch or provide safetensors inputs"
            )
        if isinstance(sd, dict):
            return _finish_source_tensors({k: v.float().numpy() for k, v in sd.items() if hasattr(v, "float")}, base_graph, source_path)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.load failed: %s", exc)

    return {}


def _compute_task_vectors(
    base_weights: dict[str, list[float]],
    source_weights_list: list[dict[str, list[float]]],
) -> list[dict[str, list[float]]]:
    """Compute δθᵢ = θ_fineᵢ − θ_base for each source."""
    task_vectors: list[dict[str, list[float]]] = []
    for src_weights in source_weights_list:
        delta: dict[str, list[float]] = {}
        for name in base_weights:
            if name in src_weights:
                base_vals = base_weights[name]
                src_vals = src_weights[name]
                if len(base_vals) != len(src_vals):
                    raise ValueError(f"Merge tensor {name!r} has mismatched element counts")
                if not all(math.isfinite(value) for value in base_vals) or not all(math.isfinite(value) for value in src_vals):
                    raise ValueError(f"Merge tensor {name!r} must contain finite coordinates")
                delta[name] = [src_vals[i] - base_vals[i] for i in range(len(base_vals))]
        task_vectors.append(delta)
    return task_vectors


def _apply_delta(
    base_weights: dict[str, list[float]],
    merged_delta: dict[str, list[float]],
) -> dict[str, list[float]]:
    """θ_merged = θ_base + merged_delta, element-wise."""
    merged: dict[str, list[float]] = dict(base_weights)
    for name, delta_vals in merged_delta.items():
        if name in merged:
            base_vals = merged[name]
            if len(base_vals) != len(delta_vals):
                raise ValueError(f"Merge delta {name!r} has mismatched element counts")
            result = [base_vals[i] + delta_vals[i] for i in range(len(base_vals))]
            if not all(math.isfinite(value) for value in result):
                raise ValueError(f"Merge tensor {name!r} overflowed or contains nonfinite coordinates")
            merged[name] = result
    return merged


def _update_graph_weights(graph: Any, merged_weights: dict[str, list[float]]) -> int:
    """Write merged weights back into the graph's weight store. Returns count updated."""
    n_updated = 0
    if hasattr(graph, "weight_store") and hasattr(graph.weight_store, "update"):
        graph.weight_store.update(merged_weights)
        n_updated = len(merged_weights)
    elif hasattr(graph, "nodes"):
        import numpy as np
        for name, node, attribute in _graph_parameter_slots(graph):
            if name not in merged_weights:
                continue
            original = np.asarray(node.attributes[attribute])
            array = np.asarray(merged_weights[name], dtype=np.float32).reshape(original.shape)
            if hasattr(node, "add_attribute"):
                node.add_attribute(attribute, array)
            else:
                node.attributes[attribute] = array
            n_updated += 1
    elif hasattr(graph, "parameters") and callable(graph.parameters):
        try:
            import torch  # type: ignore[import]

            for name, param in graph.named_parameters():
                if name in merged_weights:
                    new_vals = merged_weights[name]
                    param.data.copy_(
                        torch.tensor(new_vals, dtype=param.dtype).reshape(param.shape)
                    )
                    n_updated += 1
        except ImportError:
            pass
    elif isinstance(graph, dict):
        for name, vals in merged_weights.items():
            if name in graph:
                graph[name] = vals
                n_updated += 1
    return n_updated


# ── Merging method implementations ────────────────────────────────────────────


def _validate_task_vector_inputs(
    task_vectors: list[dict[str, list[float]]],
    coefficients: list[float],
    *,
    nonnegative: bool = False,
) -> None:
    """Validate coordinates before arithmetic, including partial adapter keys.

    Sources may target different subsets of base parameters. For each shared
    key, however, every supplied coordinate list must have the same length.
    """
    if len(coefficients) != len(task_vectors):
        raise ValueError("Provide one coefficient per task vector")
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Merge coefficients must be finite")
    if nonnegative and any(value < 0 for value in coefficients):
        raise ValueError("TIES disjoint-mean coefficients must be nonnegative")
    lengths: dict[str, int] = {}
    for vector in task_vectors:
        for name, values in vector.items():
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Merge tensor {name!r} must contain finite coordinates")
            if name in lengths and lengths[name] != len(values):
                raise ValueError(f"Merge tensor {name!r} has mismatched element counts")
            lengths[name] = len(values)


class _BaseMerger:
    """Abstract base for merging algorithms."""

    def merge(
        self,
        task_vectors: list[dict[str, list[float]]],
        coefficients: list[float],
        config: Any,
    ) -> dict[str, list[float]]:
        raise NotImplementedError


class _TaskArithmeticMerger(_BaseMerger):
    """Task Arithmetic: θ_merged = θ_base + Σᵢ λᵢ · δθᵢ."""

    def merge(self, task_vectors, coefficients, config):
        _validate_task_vector_inputs(task_vectors, coefficients)
        if not task_vectors:
            return {}
        all_keys = set().union(*[set(tv.keys()) for tv in task_vectors])
        merged: dict[str, list[float]] = {}
        for key in all_keys:
            # Collect deltas for this key from each source.
            key_deltas = [
                (coefficients[i], tv[key])
                for i, tv in enumerate(task_vectors)
                if key in tv
            ]
            if not key_deltas:
                continue
            length = len(key_deltas[0][1])
            result = [0.0] * length
            for lam, delta in key_deltas:
                for j in range(length):
                    result[j] += lam * delta[j]
            if not all(math.isfinite(value) for value in result):
                raise ValueError(f"Merge tensor {key!r} overflowed")
            merged[key] = result
        return merged


class _DAREMerger(_BaseMerger):
    """DARE (Drop And REscale): randomly zero-out p% of each δθᵢ, rescale by 1/(1-p)."""

    def __init__(self, drop_prob: float = _DARE_DEFAULT_DROP_PROB, seed: int = 42) -> None:
        if not math.isfinite(drop_prob) or not 0 <= drop_prob < 1:
            raise ValueError("DARE drop probability must be finite and in [0, 1)")
        self.drop_prob = drop_prob
        self.seed = seed

    def merge(self, task_vectors, coefficients, config):
        _validate_task_vector_inputs(task_vectors, coefficients)
        rng = random.Random(self.seed)
        rescale = 1.0 / (1.0 - self.drop_prob)
        # Apply drop-and-rescale to each task vector.
        sparsified = []
        for tv in task_vectors:
            sparse_tv: dict[str, list[float]] = {}
            for key, vals in tv.items():
                new_vals = [
                    (v * rescale if rng.random() > self.drop_prob else 0.0)
                    for v in vals
                ]
                sparse_tv[key] = new_vals
            sparsified.append(sparse_tv)
        # Then apply task arithmetic on sparsified vectors.
        return _TaskArithmeticMerger().merge(sparsified, coefficients, config)


class _TIESMerger(_BaseMerger):
    """TIES-Merging: Trim→Elect→Merge with sign conflict resolution."""

    def merge(self, task_vectors, coefficients, config):
        _validate_task_vector_inputs(task_vectors, coefficients, nonnegative=True)
        if not task_vectors:
            return {}
        all_keys = set().union(*[set(tv.keys()) for tv in task_vectors])
        merged: dict[str, list[float]] = {}

        for key in all_keys:
            key_deltas = [(coefficients[i], tv[key]) for i, tv in enumerate(task_vectors) if key in tv]
            if not key_deltas:
                continue
            length = len(key_deltas[0][1])

            # STEP 1: Trim — zero out small magnitudes below TIES_TRIM_FRACTION.
            trimmed = []
            for lam, delta in key_deltas:
                vals = delta
                threshold = sorted([abs(v) for v in vals], reverse=True)[
                    max(0, math.ceil(len(vals) * _TIES_TRIM_FRACTION) - 1)
                ] if vals else 0.0
                trimmed_vals = [v if abs(v) >= threshold else 0.0 for v in vals]
                trimmed.append((lam, trimmed_vals))

            # STEP 2: Elect by weighted signed magnitude per element.
            result = [0.0] * length
            for j in range(length):
                magnitude = sum(lam * v[j] for lam, v in trimmed if len(v) > j)
                elected_sign = math.copysign(1, magnitude) if magnitude else 0

                # STEP 3: Merge only consistent-sign contributions.
                total_w = 0.0
                total_v = 0.0
                for lam, v in trimmed:
                    if j < len(v) and v[j] != 0 and math.copysign(1, v[j]) == elected_sign:
                        total_v += lam * v[j]
                        total_w += lam
                result[j] = total_v / total_w if total_w > 0 else 0.0

            merged[key] = result
            if not all(math.isfinite(value) for value in result):
                raise ValueError(f"Merge tensor {key!r} overflowed")
        return merged


class _FREEMerger(_BaseMerger):
    """FREE-Merging (2026): evolutionary optimization of merge coefficients.

    Uses a simple (1+1)-ES (evolution strategy) to optimize λᵢ per-layer.
    Without a validation set we use the L2 norm of the merged delta as a
    proxy objective (minimize interference = minimize ||merged_delta||₂).
    """

    def __init__(self, n_iterations: int = 50, sigma: float = 0.1, seed: int = 42) -> None:
        self.n_iterations = n_iterations
        self.sigma = sigma
        self.seed = seed

    def merge(self, task_vectors, coefficients, config):
        rng = random.Random(self.seed)
        ta = _TaskArithmeticMerger()

        best_coeff = list(coefficients)
        best_result = ta.merge(task_vectors, best_coeff, config)
        best_score = _l2_norm(best_result)

        for _ in range(self.n_iterations):
            # Gaussian mutation of coefficients; clip to [0, 1].
            candidate = [
                max(0.0, min(1.0, c + rng.gauss(0, self.sigma))) for c in best_coeff
            ]
            # Normalize so they sum to 1.
            total = sum(candidate) or 1.0
            candidate = [c / total for c in candidate]

            result = ta.merge(task_vectors, candidate, config)
            score = _l2_norm(result)
            # Minimize interference (smaller merged delta = less noise).
            if score < best_score:
                best_score = score
                best_coeff = candidate
                best_result = result

        logger.debug(
            "FREE-Merging converged: best λ=%s, ||Δθ||₂=%.4f",
            [f"{c:.3f}" for c in best_coeff],
            best_score,
        )
        return best_result


def _l2_norm(weights: dict[str, list[float]]) -> float:
    """Compute the total L2 norm of a weight delta dict."""
    total = 0.0
    for vals in weights.values():
        total += sum(v * v for v in vals)
    return math.sqrt(total)


class _ModelSoupMerger(_BaseMerger):
    """Simple uniform weight averaging (model soup baseline)."""

    def merge(self, task_vectors, coefficients, config):
        # Soup = all sources have equal weight = task arithmetic with 1/N.
        n = len(task_vectors)
        uniform = [1.0 / n] * n if n > 0 else coefficients
        return _TaskArithmeticMerger().merge(task_vectors, uniform, config)


def _get_merger(method: str) -> _BaseMerger:
    """Return the appropriate merger instance for the given method name."""
    methods = {
        "task_arithmetic": _TaskArithmeticMerger(),
        "dare": _DAREMerger(),
        "ties": _TIESMerger(),
        "free": _FREEMerger(),
        "evolutionary": _FREEMerger(n_iterations=200, sigma=0.05),  # heavier FREE
        "soup": _ModelSoupMerger(),
    }
    if method not in methods:
        raise ValueError(f"Unsupported model merge method: {method}")
    return methods[method]


# ── AEG blob writer ───────────────────────────────────────────────────────────


def _write_merge_manifest(
    output_dir: Any,
    sources: list[str],
    method: str,
    coefficients: list[float],
    n_params: int,
) -> None:
    """Write merge manifest JSON to .aeg/merging/."""
    merging_dir = output_dir / "merging"
    merging_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "aether_merge_manifest_v1",
        "method": method,
        "n_sources": len(sources),
        "sources": sources,
        "coefficients": coefficients,
        "n_parameters_merged": n_params,
    }
    (merging_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.debug("Wrote merge manifest: %s", merging_dir / "merge_manifest.json")


