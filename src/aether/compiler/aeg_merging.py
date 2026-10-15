"""Merge authenticated AEG weights without re-ingesting compiled graph files.

Merging preserves the base graph, tokenizer and target contracts. It invalidates
weight-dependent certification; model quality needs a new evaluation after any
merge. Tensor work is vectorized and performed one tensor at a time.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.core.hash_utils import compute_file_hash
from aether.core.weight_store import WeightStore
from aether.quantization.formats import QuantizedTensor, dequantize_tensor, quantize_tensor
from aether.utils.artifact_publication import publish_staged_directory


def _validate_input_tree(root: Path) -> None:
    """Reject unmanifested links, junctions and special files before copying."""
    resolved_root = root.resolve(strict=True)
    pending = [resolved_root]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            details = path.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or getattr(details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                or not path.resolve(strict=True).is_relative_to(resolved_root)
            ):
                raise ValueError(f"Linked AEG package members cannot be merged: {path}")
            if stat.S_ISDIR(details.st_mode):
                pending.append(path)
            elif not stat.S_ISREG(details.st_mode):
                raise ValueError(f"Non-regular AEG package member: {path}")


def invalidate_weight_dependent_evidence(
    package: AEGPackage, *, transformation: str = "model_merge"
) -> None:
    """Clear stale weight evidence on an already staged package copy.

    Auxiliary execution payloads and their configuration are retained. Their
    quality must be evaluated again against the changed canonical weights.
    """
    if package.manifest is None:
        raise ValueError("Evidence invalidation requires a loaded AEG manifest")
    for directory in ("provenance", "observability", "security", "tee", "watermark"):
        for file in (package.root / directory).rglob("*"):
            if file.is_file() and (
                directory in {"provenance", "security", "tee", "watermark"}
                or file.name == "eval_report.json"
            ):
                file.unlink()
    for key in ("eval_report", "provenance", "fingerprint", "watermark", "tee", "tee_attestation"):
        package.metadata.pop(key, None)
    package.manifest.provenance = type(package.manifest.provenance)()
    package.manifest.optimization.actual_ppl_increase = None
    package.metadata["evaluation_status"] = "uncertified"
    package.metadata["evidence_invalidation"] = {
        "reason": transformation,
        "quality_validated": False,
        "auxiliary_policy": "preserve_base_requires_reevaluation",
    }


def merge_tensor_arrays(
    base: np.ndarray,
    sources: list[np.ndarray],
    coefficients: list[float],
    method: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Merge matching tensor coordinates; no truncation or inferred alignment.

    TIES uses magnitude sign election and a weighted disjoint mean, with top
    20% trimming per tensor. DARE uses seeded 90% drop/rescale task vectors.
    These are explicit variants, not quality-certified defaults for every model.
    """
    if not sources or len(sources) != len(coefficients):
        raise ValueError("Provide one coefficient per nonempty source list")
    if any(array.shape != base.shape for array in sources):
        raise ValueError("Merge tensor shapes must match exactly")
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Merge coefficients must be finite")
    if not np.isfinite(base).all() or any(not np.isfinite(array).all() for array in sources):
        raise ValueError("Merge tensors must be finite")
    if method == "linear":
        if not math.isclose(sum(coefficients), 1.0, abs_tol=1e-7):
            raise ValueError("Linear interpolation coefficients must sum to 1")
        result = np.zeros(base.shape, dtype=np.float32)
        for coefficient, array in zip(coefficients, sources, strict=True):
            result += coefficient * array.astype(np.float32)
    elif method in {"task_arithmetic", "dare", "ties"}:
        result = base.astype(np.float32).copy()
        if method == "ties":
            if any(value < 0 for value in coefficients):
                raise ValueError("TIES disjoint-mean coefficients must be nonnegative")
            deltas = []
            for array in sources:
                delta = array.astype(np.float32) - base
                flat = np.abs(delta).reshape(-1)
                keep = max(1, math.ceil(flat.size * 0.2))
                threshold = (
                    np.partition(flat, max(0, flat.size - keep))[max(0, flat.size - keep)]
                    if flat.size
                    else 0
                )
                deltas.append(np.where(np.abs(delta) >= threshold, delta, 0))
            elected = np.sign(
                sum(
                    coefficient * delta
                    for coefficient, delta in zip(coefficients, deltas, strict=True)
                )
            )
            numerator = np.zeros(base.shape, dtype=np.float32)
            denominator = np.zeros(base.shape, dtype=np.float32)
            for coefficient, delta in zip(coefficients, deltas, strict=True):
                aligned = (np.sign(delta) == elected) & (delta != 0)
                numerator += coefficient * np.where(aligned, delta, 0)
                denominator += coefficient * aligned
            result += np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
            )
        else:
            rng = rng if rng is not None else np.random.default_rng(42)
            for coefficient, array in zip(coefficients, sources, strict=True):
                delta = array.astype(np.float32) - base
                if method == "dare":
                    delta = delta * (rng.random(delta.shape) >= 0.9) / 0.1
                result += coefficient * delta
    else:
        raise ValueError(f"Unsupported model merge method: {method}")
    if not np.isfinite(result).all():
        raise ValueError("Merge overflowed the FP32 output format")
    return result


def merge_aeg_packages(
    models: list[str | Path],
    output: str | Path,
    *,
    method: str = "linear",
    coefficients: list[float] | None = None,
) -> AEGPackage:
    """Write a separate runnable AEG from compatible authenticated inputs.

    All sources are required. Destination creation is staged and verified before
    publication, and an existing destination or overlapping input is rejected.
    """
    if len(models) < 2:
        raise ValueError("At least two AEG inputs are required")
    if method not in {"linear", "task_arithmetic", "ties", "dare"}:
        raise ValueError(f"Unsupported model merge method: {method}")
    coefficients = (
        list(coefficients) if coefficients is not None else [1 / len(models)] * len(models)
    )
    if len(coefficients) != len(models) or not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Provide one finite coefficient per AEG input")
    packages = [load_aeg_package(path) for path in models]
    base = packages[0]
    if base.manifest is None or base.ir is None or not base.has_weights:
        raise ValueError("Base AEG must contain a real graph and weight store")
    destination = Path(output).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Merged destination already exists: {destination}")
    resolved_destination = destination.resolve()
    for package in packages:
        _validate_input_tree(package.root)
        package.verify_integrity()
        source = package.root.resolve()
        if (
            resolved_destination == source
            or resolved_destination.is_relative_to(source)
            or source.is_relative_to(resolved_destination)
        ):
            raise ValueError("Merged output must not overlap any input package")
        if package.manifest is None or package.ir is None or not package.has_weights:
            raise ValueError("Every AEG input needs a real graph and weight store")
        if "tee_kernel_wrapping" in package.metadata.get("optimizer_passes", []):
            raise ValueError(
                "TEE-bound AEG weights require a fresh attested compilation after merging"
            )
        if package.manifest.architecture.to_dict() != base.manifest.architecture.to_dict():
            raise ValueError("AEG architectures must match exactly")
        if package.ir.to_dict() != base.ir.to_dict():
            raise ValueError("AEG computation graphs must match exactly")
        if any(
            package.metadata.get(key) != base.metadata.get(key)
            for key in ("nodes", "normalization", "source_config")
        ):
            raise ValueError("AEG operator semantics must match exactly")
        tokenizers = {
            file.relative_to(package.root / "tokenizer").as_posix(): compute_file_hash(file)
            for file in (package.root / "tokenizer").rglob("*")
            if file.is_file()
        }
        base_tokenizers = {
            file.relative_to(base.root / "tokenizer").as_posix(): compute_file_hash(file)
            for file in (base.root / "tokenizer").rglob("*")
            if file.is_file()
        }
        if not tokenizers or tokenizers != base_tokenizers:
            raise ValueError("AEG tokenizer files and token IDs must match exactly")
        store = package.weight_store()
        base_store = base.weight_store()
        if set(store.entries) != set(base_store.entries) or any(
            store.entries[name].shape != entry.shape for name, entry in base_store.entries.items()
        ):
            raise ValueError("AEG weight names and tensor shapes must match exactly")
    if destination.exists():
        raise FileExistsError(f"Merged destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stores = [package.weight_store() for package in packages]

    class MergedTensors(Mapping[str, QuantizedTensor]):
        def __len__(self):
            return len(stores[0].entries)

        def __iter__(self):
            return iter(stores[0].entries)

        def __getitem__(self, name):
            entry = stores[0].entries[name]
            arrays = [dequantize_tensor(store.load_tensor(name)) for store in stores]
            seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little") ^ 42
            merged = merge_tensor_arrays(
                arrays[0], arrays, coefficients, method, np.random.default_rng(seed)
            )
            return quantize_tensor(merged, entry.precision, block_size=entry.block_size)

    with tempfile.TemporaryDirectory(prefix=".aether-merge-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "merged.aeg"
        shutil.copytree(base.root, staging)
        package = AEGPackage(staging).load()
        invalidate_weight_dependent_evidence(package)
        package.manifest.model_id = f"merged/{base.manifest.model_id}"
        if package.manifest.format_version not in {"AEG/2.0", "AEG/3.0"}:
            package.manifest.format_version = "AEG/2.0"
        package.manifest.compiled_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record: dict[str, Any] = {
            "format": "aether_merge_manifest_v1",
            "method": method,
            "sources": [str(path) for path in models],
            "coefficients": coefficients,
            "n_sources": len(models),
            "n_tensors_updated": len(stores[0].entries),
            "precision_policy": "preserve_base_storage_formats",
            "auxiliary_policy": "preserve_base_requires_reevaluation",
            "quality_validated": False,
            "seed": 42,
            "ties_trim_scope": "per_tensor",
        }
        package.metadata["merge"] = record
        package.metadata["model_merging"] = record
        passes = list(package.metadata.get("optimizer_passes", []))
        if "model_merging" not in passes:
            passes.append("model_merging")
        package.metadata["optimizer_passes"] = passes
        package.metadata.setdefault("pass_reports", {})["model_merging"] = {
            "status": "applied",
            "details": record,
        }
        package.metadata["weight_bytes"] = WeightStore(staging / "weights" / "quantized").save(
            MergedTensors()
        )
        (staging / "merging").mkdir(exist_ok=True)
        (staging / "merging" / "merge_manifest.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        package.save()
        package.verify_integrity()
        publish_staged_directory(staging, destination)
    return AEGPackage(destination).load()
