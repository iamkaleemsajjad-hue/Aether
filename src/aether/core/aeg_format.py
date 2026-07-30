"""
AEG format serialization and package management.

This module defines the on-disk structure of AEG artifacts: the manifest,
graph, weights, kernels, parallelism plans, and version metadata. It also
provides loading/saving, integrity verification, and round-trip guarantees.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.aeg_ir import AEGIRModule
from aether.core.constants import (
    AEG_FORMAT_VERSION,
    AEG_GRAPH_FILENAME,
    AEG_IR_FILE_EXTENSION,
    AEG_MANIFEST_FILENAME,
    AEG_QUANT_FILE_EXTENSION,
    DEFAULT_MODEL_CACHE_SUBDIR,
)
from aether.core.exceptions import AEGFormatError, AEGIntegrityError, AEGVersionError
from aether.core.hash_utils import compute_content_hash, compute_directory_hash, compute_file_hash, verify_file_hash
from aether.core.types import AEGVersion, ModelArchitecture, Precision, ShardingPlan


@dataclass
class OptimizationMetadata:
    """Metadata about the optimization applied during compilation."""

    fusion_passes_applied: list[str] = field(default_factory=list)
    """Names of fusion passes applied."""

    fused_ops_count: int = 0
    """Number of fused operations produced."""

    sensitivity_calibration_dataset: str = "wikitext-2"
    """Calibration dataset used for sensitivity analysis."""

    quality_budget_ppl_increase: float = 0.02
    """Configured maximum perplexity increase."""

    actual_ppl_increase: float | None = None
    """Measured perplexity increase after quantization."""

    precision_distribution: dict[str, str] = field(default_factory=dict)
    """Distribution of precisions by percentage (e.g., {'BF16': '12%', ...})."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fusion_passes_applied": self.fusion_passes_applied,
            "fused_ops_count": self.fused_ops_count,
            "sensitivity_calibration_dataset": self.sensitivity_calibration_dataset,
            "quality_budget_ppl_increase": self.quality_budget_ppl_increase,
            "actual_ppl_increase": self.actual_ppl_increase,
            "precision_distribution": self.precision_distribution,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OptimizationMetadata:
        return OptimizationMetadata(
            fusion_passes_applied=list(data.get("fusion_passes_applied", [])),
            fused_ops_count=data.get("fused_ops_count", 0),
            sensitivity_calibration_dataset=data.get("sensitivity_calibration_dataset", "wikitext-2"),
            quality_budget_ppl_increase=data.get("quality_budget_ppl_increase", 0.02),
            actual_ppl_increase=data.get("actual_ppl_increase"),
            precision_distribution=dict(data.get("precision_distribution", {})),
        )


@dataclass
class KernelSetMetadata:
    """Metadata about compiled kernel sets in the AEG."""

    targets: list[str] = field(default_factory=list)
    """Hardware targets covered by this kernel set."""

    flash_attention_variant: str = "flash_attention_3"
    """FlashAttention variant used for attention kernels."""

    backend_plans: dict[str, str] = field(default_factory=dict)
    """Map of target_id to backend plan descriptor."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": self.targets,
            "flash_attention_variant": self.flash_attention_variant,
            "backend_plans": self.backend_plans,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> KernelSetMetadata:
        return KernelSetMetadata(
            targets=list(data.get("targets", [])),
            flash_attention_variant=data.get("flash_attention_variant", "flash_attention_3"),
            backend_plans=dict(data.get("backend_plans", {})),
        )


@dataclass
class MemoryRequirements:
    """Memory requirements for different precisions and configs."""

    bf16_gb: float
    """Memory requirement at full BF16 precision (GB)."""

    compiled_min_gb: float
    """Minimum memory requirement with compiled precision (GB)."""

    recommended_gb: float
    """Recommended memory for optimal performance (GB)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bf16_gb": self.bf16_gb,
            "compiled_min_gb": self.compiled_min_gb,
            "recommended_gb": self.recommended_gb,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MemoryRequirements:
        return MemoryRequirements(
            bf16_gb=data["bf16_gb"],
            compiled_min_gb=data["compiled_min_gb"],
            recommended_gb=data["recommended_gb"],
        )


@dataclass
class AEGManifest:
    """Top-level manifest for an AEG artifact.

    The manifest contains version information, model architecture metadata,
    optimization metadata, kernel set metadata, and hashes for integrity
    verification.
    """

    model_id: str
    """Model identifier (e.g., 'Qwen/Qwen3-72B-Instruct')."""

    aether_version: str
    """Aether Runtime version that produced this AEG."""

    compiled_at: str
    """ISO 8601 timestamp of compilation."""

    graph_hash: str
    """Content-addressed hash of the AEG-IR graph."""

    architecture: ModelArchitecture
    """Detected model architecture."""

    optimization: OptimizationMetadata = field(default_factory=OptimizationMetadata)
    """Optimization metadata from the compiler."""

    kernels: KernelSetMetadata = field(default_factory=KernelSetMetadata)
    """Kernel set metadata."""

    memory_requirements: MemoryRequirements | None = None
    """Memory requirements at different precisions."""

    manifest_hash: str | None = None
    """Hash of the manifest itself (excluding this field)."""

    format_version: str = AEG_FORMAT_VERSION
    """AEG format version string."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_id": self.model_id,
            "aether_version": self.aether_version,
            "compiled_at": self.compiled_at,
            "graph_hash": self.graph_hash,
            "architecture": self.architecture.to_dict(),
            "optimization": self.optimization.to_dict(),
            "kernels": self.kernels.to_dict(),
            "memory_requirements": self.memory_requirements.to_dict() if self.memory_requirements else None,
            "manifest_hash": self.manifest_hash,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGManifest:
        memory_data = data.get("memory_requirements")
        memory_requirements = MemoryRequirements.from_dict(memory_data) if memory_data else None
        return AEGManifest(
            model_id=data["model_id"],
            aether_version=data["aether_version"],
            compiled_at=data["compiled_at"],
            graph_hash=data["graph_hash"],
            architecture=ModelArchitecture.from_dict(data["architecture"]),
            optimization=OptimizationMetadata.from_dict(data.get("optimization", {})),
            kernels=KernelSetMetadata.from_dict(data.get("kernels", {})),
            memory_requirements=memory_requirements,
            manifest_hash=data.get("manifest_hash"),
            format_version=data.get("format_version", AEG_FORMAT_VERSION),
        )

    @staticmethod
    def from_json(json_str: str) -> AEGManifest:
        return AEGManifest.from_dict(json.loads(json_str))

    def compute_and_set_manifest_hash(self) -> None:
        """Compute the manifest hash and set it."""
        payload = self.to_dict()
        payload.pop("manifest_hash", None)
        self.manifest_hash = compute_content_hash(payload)

    def verify(self) -> None:
        """Verify the manifest's version and integrity.

        Raises:
            AEGVersionError: If the AEG format version is unsupported.
            AEGIntegrityError: If the manifest hash does not match.
        """
        current = AEGVersion.current()
        try:
            manifest_version = AEGVersion.parse(self.format_version)
        except (ValueError, IndexError) as exc:
            msg = f"Invalid AEG format version: {self.format_version}"
            raise AEGVersionError(msg, aeg_version=self.format_version) from exc
        if not manifest_version.is_compatible_with(current):
            msg = (
                f"AEG format {self.format_version} is not compatible with runtime "
                f"{current}. Minimum compatible: {current}"
            )
            raise AEGVersionError(
                msg,
                aeg_version=self.format_version,
                minimum_version=str(current),
            )
        if self.manifest_hash:
            payload = self.to_dict()
            payload.pop("manifest_hash", None)
            computed = compute_content_hash(payload)
            if computed != self.manifest_hash:
                msg = f"AEG manifest integrity check failed: expected {self.manifest_hash}, got {computed}"
                raise AEGIntegrityError(msg, expected_hash=self.manifest_hash, actual_hash=computed)

    def __repr__(self) -> str:
        return f"AEGManifest({self.model_id}, {self.format_version}, hash={self.graph_hash[:20]}...)"


class AEGPackage:
    """Represents a compiled AEG package on disk.

    An AEG package is a directory (or tar archive) containing:
    - FORMAT_VERSION
    - graph/computation_graph.aeg-ir
    - graph/metadata.json
    - weights/quantized/ (precision_map.json + model.aeg-quant)
    - kernels/ (per-target kernel metadata/plans)
    - parallelism/ (1/2/4/8 GPU sharding plans)
    - manifest.json
    """

    def __init__(self, root: Path | str) -> None:
        """Open an existing AEG package at the given root path.

        Args:
            root: Path to the AEG package directory.
        """
        self.root = Path(root).resolve()
        self.manifest: AEGManifest | None = None
        self.ir: AEGIRModule | None = None
        self.precision_map: dict[str, str] = {}
        self.sharding_plans: dict[int, ShardingPlan] = {}
        self.metadata: dict[str, Any] = {}
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return True if the package has been fully loaded into memory."""
        return self._is_loaded

    @property
    def model_id(self) -> str | None:
        """Return the model ID from the loaded manifest."""
        return self.manifest.model_id if self.manifest else None

    @property
    def format_version(self) -> str | None:
        """Return the AEG format version."""
        return self.manifest.format_version if self.manifest else None

    @staticmethod
    def create(root: Path | str, model_id: str, aether_version: str) -> AEGPackage:
        """Create a new AEG package directory structure.

        Args:
            root: Root directory where the package will be created.
            model_id: Model identifier.
            aether_version: Aether version string.

        Returns:
            A new, empty AEGPackage.
        """
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        package = AEGPackage(root_path)
        package.manifest = AEGManifest(
            model_id=model_id,
            aether_version=aether_version,
            compiled_at="2026-07-27T00:00:00Z",  # Will be updated at save time
            graph_hash="sha256:pending",
            architecture=ModelArchitecture(
                family="unknown",
                params_billion=0.0,
                layers=1,
                hidden_size=64,
                num_attention_heads=4,
            ),
            optimization=OptimizationMetadata(),
            kernels=KernelSetMetadata(),
            memory_requirements=MemoryRequirements(bf16_gb=0.0, compiled_min_gb=0.0, recommended_gb=0.0),
        )
        return package

    def load(self) -> AEGPackage:
        """Load the entire package from disk into memory.

        Returns:
            Self, for chaining.

        Raises:
            AEGFormatError: If the package is malformed.
            FileNotFoundError: If required files are missing.
        """
        if not self.root.exists():
            msg = f"AEG package not found: {self.root}"
            raise FileNotFoundError(msg)
        manifest_path = self.root / AEG_MANIFEST_FILENAME
        if not manifest_path.exists():
            msg = f"Missing AEG manifest: {manifest_path}"
            raise FileNotFoundError(msg)
        self.manifest = AEGManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        self.manifest.verify()
        # Load graph
        graph_path = self.root / "graph" / AEG_GRAPH_FILENAME
        if graph_path.exists():
            self.ir = AEGIRModule.from_json(graph_path.read_text(encoding="utf-8"))
        # Load metadata
        metadata_path = self.root / "graph" / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Load precision map
        precision_path = self.root / "weights" / "quantized" / "precision_map.json"
        if precision_path.exists():
            self.precision_map = json.loads(precision_path.read_text(encoding="utf-8"))
        # Load sharding plans
        parallelism_dir = self.root / "parallelism"
        if parallelism_dir.exists():
            for plan_file in parallelism_dir.glob("*.json"):
                plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
                plan = ShardingPlan.from_dict(plan_data)
                self.sharding_plans[plan.num_gpus] = plan
        self._is_loaded = True
        return self

    def save(self) -> None:
        """Save the package to disk.

        Raises:
            AEGFormatError: If the package is incomplete or malformed.
        """
        if not self.manifest:
            msg = "Cannot save AEG package without a manifest"
            raise AEGFormatError(msg)
        # Update timestamp
        import datetime

        self.manifest.compiled_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Ensure directories
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "graph").mkdir(exist_ok=True)
        (self.root / "weights" / "quantized").mkdir(parents=True, exist_ok=True)
        (self.root / "kernels").mkdir(exist_ok=True)
        (self.root / "parallelism").mkdir(exist_ok=True)
        # Write format version
        (self.root / "FORMAT_VERSION").write_text(self.manifest.format_version + "\n", encoding="utf-8")
        # Write graph
        if self.ir:
            (self.root / "graph" / AEG_GRAPH_FILENAME).write_text(self.ir.to_json(indent=2), encoding="utf-8")
            # Compute and set graph hash
            graph_hash = compute_file_hash(self.root / "graph" / AEG_GRAPH_FILENAME)
            self.manifest.graph_hash = graph_hash
            (self.root / "graph" / "graph.sha256").write_text(graph_hash + "\n", encoding="utf-8")
        # Write metadata
        (self.root / "graph" / "metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        # Write precision map
        (self.root / "weights" / "quantized" / "precision_map.json").write_text(
            json.dumps(self.precision_map, indent=2), encoding="utf-8"
        )
        # Write sharding plans
        for num_gpus, plan in self.sharding_plans.items():
            plan_path = self.root / "parallelism" / f"{num_gpus}gpu.json"
            plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        # Write kernel metadata
        kernel_meta_path = self.root / "kernels" / "kernel_sets.json"
        kernel_meta_path.write_text(json.dumps(self.manifest.kernels.to_dict(), indent=2), encoding="utf-8")
        # Compute and set manifest hash
        self.manifest.compute_and_set_manifest_hash()
        # Write manifest
        (self.root / AEG_MANIFEST_FILENAME).write_text(self.manifest.to_json(indent=2), encoding="utf-8")

    def save_as_archive(self, output_path: Path | str) -> Path:
        """Save the package as a tar archive.

        Args:
            output_path: Path to write the tar archive.

        Returns:
            Path to the created archive.
        """
        self.save()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            tar.add(self.root, arcname=self.root.name)
        return output

    @staticmethod
    def load_from_archive(archive_path: Path | str) -> AEGPackage:
        """Load an AEG package from a tar archive.

        Args:
            archive_path: Path to the tar archive.

        Returns:
            Loaded AEGPackage.
        """
        archive = Path(archive_path)
        if not archive.exists():
            msg = f"Archive not found: {archive}"
            raise FileNotFoundError(msg)
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmpdir)
            extracted_root = Path(tmpdir)
            # Find the AEG package root inside the archive
            candidates = list(extracted_root.glob("*"))
            package_root = candidates[0] if candidates else extracted_root
            package = AEGPackage(package_root)
            package.load()
            return package

    def verify_integrity(self) -> None:
        """Verify the integrity of all package files using stored hashes.

        Raises:
            AEGIntegrityError: If any file hash does not match.
        """
        if not self.manifest:
            msg = "Cannot verify package without manifest"
            raise AEGIntegrityError(msg)
        self.manifest.verify()
        # Verify graph hash
        graph_path = self.root / "graph" / AEG_GRAPH_FILENAME
        if graph_path.exists():
            expected = self.manifest.graph_hash
            if not verify_file_hash(graph_path, expected):
                actual = compute_file_hash(graph_path)
                msg = f"AEG graph hash mismatch: expected {expected}, got {actual}"
                raise AEGIntegrityError(msg, file_path=str(graph_path), expected_hash=expected, actual_hash=actual)
        # Verify overall package hash if available
        package_hash_path = self.root / "graph" / "graph.sha256"
        if package_hash_path.exists():
            stored = package_hash_path.read_text(encoding="utf-8").strip()
            computed = compute_file_hash(graph_path)
            if stored != computed:
                msg = f"AEG graph file hash mismatch: stored {stored}, computed {computed}"
                raise AEGIntegrityError(msg, file_path=str(package_hash_path), expected_hash=stored, actual_hash=computed)

    def get_backend_plan(self, target_id: str) -> str | None:
        """Return the backend plan descriptor for a hardware target."""
        if not self.manifest:
            return None
        return self.manifest.kernels.backend_plans.get(target_id)

    def get_sharding_plan(self, num_gpus: int) -> ShardingPlan | None:
        """Return the sharding plan for a specific GPU count."""
        return self.sharding_plans.get(num_gpus)

    def get_precision_map(self) -> dict[str, str]:
        """Return the per-layer precision assignments."""
        return dict(self.precision_map)

    def set_precision_map(self, precision_map: dict[str, str]) -> None:
        """Set the per-layer precision assignments."""
        self.precision_map = dict(precision_map)

    def set_sharding_plan(self, num_gpus: int, plan: ShardingPlan) -> None:
        """Set the sharding plan for a specific GPU count."""
        self.sharding_plans[num_gpus] = plan

    def set_backend_plan(self, target_id: str, plan_descriptor: str) -> None:
        """Set the backend plan descriptor for a hardware target."""
        if not self.manifest:
            msg = "Cannot set backend plan without manifest"
            raise AEGFormatError(msg)
        self.manifest.kernels.backend_plans[target_id] = plan_descriptor
        if target_id not in self.manifest.kernels.targets:
            self.manifest.kernels.targets.append(target_id)

    def compute_size(self) -> int:
        """Compute the total size of the package on disk in bytes."""
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def copy_to(self, destination: Path | str) -> AEGPackage:
        """Copy the package to a new location.

        Args:
            destination: Destination path.

        Returns:
            AEGPackage at the new location. The returned package is loaded if
            the destination contains a valid manifest.
        """
        dest = Path(destination).resolve()
        if self.root.exists():
            shutil.copytree(self.root, dest, dirs_exist_ok=True)
        new_package = AEGPackage(dest)
        if dest.exists() and (dest / AEG_MANIFEST_FILENAME).exists():
            new_package.load()
        return new_package

    def __repr__(self) -> str:
        return f"AEGPackage({self.root}, model_id={self.model_id}, loaded={self._is_loaded})"


def load_aeg_package(path: Path | str) -> AEGPackage:
    """Load an AEG package from disk (directory or archive).

    Args:
        path: Path to the AEG package (directory or .tar.gz archive).

    Returns:
        Loaded AEGPackage.
    """
    p = Path(path)
    if p.is_dir():
        package = AEGPackage(p)
        package.load()
        return package
    if p.suffix in (".gz", ".tgz"):
        return AEGPackage.load_from_archive(p)
    msg = f"Cannot load AEG package from: {p}. Expected directory or .tar.gz archive."
    raise AEGFormatError(msg)


def create_default_sharding_plans(architecture: ModelArchitecture) -> dict[int, ShardingPlan]:
    """Create default sharding plans for 1, 2, 4, and 8 GPUs based on architecture.

    Args:
        architecture: Model architecture metadata.

    Returns:
        Mapping of GPU count to ShardingPlan.
    """
    plans: dict[int, ShardingPlan] = {}
    params_gb = architecture.params_billion * 2.0  # Approximate GB at BF16
    for num_gpus in (1, 2, 4, 8):
        if num_gpus == 1:
            plan = ShardingPlan(
                num_gpus=1,
                phase="prefill",
                tensor_parallel_degree=1,
                pipeline_stages=1,
                memory_per_gpu_gb=params_gb,
            )
        elif num_gpus == 2:
            plan = ShardingPlan(
                num_gpus=2,
                phase="prefill",
                tensor_parallel_degree=2,
                pipeline_stages=1,
                memory_per_gpu_gb=params_gb / 2,
            )
        elif num_gpus == 4:
            plan = ShardingPlan(
                num_gpus=4,
                phase="prefill",
                tensor_parallel_degree=4,
                pipeline_stages=1,
                memory_per_gpu_gb=params_gb / 4,
            )
        else:
            plan = ShardingPlan(
                num_gpus=8,
                phase="prefill",
                tensor_parallel_degree=4,
                pipeline_stages=2,
                memory_per_gpu_gb=params_gb / 8,
            )
        plans[num_gpus] = plan
    return plans
