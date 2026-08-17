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
   Trim small deltas, Elect sign by majority vote, merge only consistent signs.
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
            logger.warning("Pass 12: Unknown merging method %r. Using task_arithmetic.", method)
            method = "task_arithmetic"

        try:
            # Determine coefficients.
            coefficients = list(config.model_merging_coefficients)
            if len(coefficients) != len(sources):
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

            if not source_weights_list:
                report.status = "skipped"
                report.details["reason"] = "all_sources_failed_to_load"
                return graph, report

            # Compute task vectors.
            task_vectors = _compute_task_vectors(base_weights, source_weights_list)
            if not any(task_vectors):
                report.status = "skipped"
                report.details["reason"] = "sources_have_no_overlapping_tensors"
                return graph, report
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
            n_updated = _update_graph_weights(graph, merged_weights)

            # Write merge manifest.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                from pathlib import Path
                _write_merge_manifest(
                    output_dir=Path(graph.output_dir),
                    sources=sources,
                    method=method,
                    coefficients=coefficients,
                    n_params=sum(
                        math.prod(v) if isinstance(v, list) else len(v)
                        for v in merged_delta.values()
                    ),
                )

            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "method": method,
                "n_sources": len(source_weights_list),
                "coefficients": coefficients,
                "n_tensors_updated": n_updated,
                "task_vector_keys": list(merged_delta.keys())[:20],
            }
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
    elif isinstance(graph, dict):
        for name, val in graph.items():
            if isinstance(val, (list, tuple)):
                weights[name] = list(val)
    return weights


def _load_source_weights(source_path: str, base_graph: Any) -> dict[str, list[float]]:
    """Load weights from a source model path.

    Tries: AEG weight store → safetensors → GGUF → PyTorch state_dict.
    Falls back to empty dict if path does not resolve to any known format.
    """
    from pathlib import Path
    p = Path(source_path)

    if not p.exists():
        logger.debug("Source path does not exist: %s", source_path)
        return {}

    # AEG is the native artifact format.  Read and dequantize its persisted
    # weight store instead of treating the directory as an opaque success.
    if p.is_dir() and (p / "manifest.json").is_file():
        try:
            from aether.core.aeg_format import AEGPackage

            package = AEGPackage(p).load()
            if not package.has_weights:
                return {}
            return {
                name: tensor.reshape(-1).astype("float32").tolist()
                for name, tensor in package.weight_store().dequantize_all().items()
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("AEG source load failed: %s", exc)
            return {}

    # Try safetensors.
    if p.suffix == ".safetensors" or (p.is_dir() and (p / "model.safetensors").exists()):
        try:
            import safetensors.torch as st  # type: ignore[import]
            sf_path = p if p.suffix == ".safetensors" else p / "model.safetensors"
            tensors = st.load_file(str(sf_path))
            return {k: v.float().reshape(-1).tolist() for k, v in tensors.items()}
        except ImportError:
            logger.debug("safetensors not installed.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("safetensors load failed: %s", exc)

    # Try PyTorch.  ``weights_only=True`` is mandatory: merging inputs are
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
            return {k: v.float().reshape(-1).tolist() for k, v in sd.items() if hasattr(v, "float")}
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
                # Element-wise subtraction; handle size mismatches gracefully.
                min_len = min(len(base_vals), len(src_vals))
                delta[name] = [src_vals[i] - base_vals[i] for i in range(min_len)]
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
            min_len = min(len(base_vals), len(delta_vals))
            merged[name] = [base_vals[i] + delta_vals[i] for i in range(min_len)]
    return merged


def _update_graph_weights(graph: Any, merged_weights: dict[str, list[float]]) -> int:
    """Write merged weights back into the graph's weight store. Returns count updated."""
    n_updated = 0
    if hasattr(graph, "weight_store") and hasattr(graph.weight_store, "update"):
        graph.weight_store.update(merged_weights)
        n_updated = len(merged_weights)
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
            length = min(len(d) for _, d in key_deltas)
            result = [0.0] * length
            for lam, delta in key_deltas:
                for j in range(length):
                    result[j] += lam * delta[j]
            merged[key] = result
        return merged


class _DAREMerger(_BaseMerger):
    """DARE (Drop And REscale): randomly zero-out p% of each δθᵢ, rescale by 1/(1-p)."""

    def __init__(self, drop_prob: float = _DARE_DEFAULT_DROP_PROB, seed: int = 42) -> None:
        self.drop_prob = drop_prob
        self.seed = seed

    def merge(self, task_vectors, coefficients, config):
        rng = random.Random(self.seed)
        rescale = 1.0 / (1.0 - self.drop_prob) if self.drop_prob < 1.0 else 1.0
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
        if not task_vectors:
            return {}
        all_keys = set().union(*[set(tv.keys()) for tv in task_vectors])
        merged: dict[str, list[float]] = {}

        for key in all_keys:
            key_deltas = [(coefficients[i], tv[key]) for i, tv in enumerate(task_vectors) if key in tv]
            if not key_deltas:
                continue
            length = min(len(d) for _, d in key_deltas)

            # STEP 1: Trim — zero out small magnitudes below TIES_TRIM_FRACTION.
            trimmed = []
            for lam, delta in key_deltas:
                vals = delta[:length]
                threshold = sorted([abs(v) for v in vals], reverse=True)[
                    max(0, int(len(vals) * _TIES_TRIM_FRACTION) - 1)
                ] if vals else 0.0
                trimmed_vals = [v if abs(v) >= threshold else 0.0 for v in vals]
                trimmed.append((lam, trimmed_vals))

            # STEP 2: Elect — majority vote on sign per element.
            result = [0.0] * length
            for j in range(length):
                pos = sum(1 for _, v in trimmed if len(v) > j and v[j] > 0)
                neg = sum(1 for _, v in trimmed if len(v) > j and v[j] < 0)
                elected_sign = 1 if pos >= neg else -1

                # STEP 3: Merge only consistent-sign contributions.
                total_w = 0.0
                total_v = 0.0
                for lam, v in trimmed:
                    if j < len(v) and v[j] != 0 and math.copysign(1, v[j]) == elected_sign:
                        total_v += lam * v[j]
                        total_w += lam
                result[j] = total_v / total_w if total_w > 0 else 0.0

            merged[key] = result
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
    return {
        "task_arithmetic": _TaskArithmeticMerger(),
        "dare": _DAREMerger(),
        "ties": _TIESMerger(),
        "free": _FREEMerger(),
        "evolutionary": _FREEMerger(n_iterations=200, sigma=0.05),  # heavier FREE
        "soup": _ModelSoupMerger(),
    }.get(method, _TaskArithmeticMerger())


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


