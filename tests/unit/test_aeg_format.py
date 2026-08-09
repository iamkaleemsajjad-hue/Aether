"""
Tests for AEG format serialization, loading, and integrity verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aether.core.aeg_format import AEGManifest, AEGPackage, AEGVersion, create_default_sharding_plans, load_aeg_package
from aether.core.exceptions import AEGFormatError, AEGIntegrityError, AEGVersionError
from aether.core.types import ModelArchitecture, ShardingPlan


class TestAEGVersion:
    """Tests for AEGVersion parsing and compatibility."""

    def test_parse_full_version(self) -> None:
        version = AEGVersion.parse("AEG/1.0")
        assert version.major == 1
        assert version.minor == 0

    def test_parse_minor_only(self) -> None:
        version = AEGVersion.parse("AEG/2.5")
        assert version.major == 2
        assert version.minor == 5

    def test_parse_without_prefix(self) -> None:
        version = AEGVersion.parse("1.0")
        assert version.major == 1
        assert version.minor == 0

    def test_version_string(self) -> None:
        version = AEGVersion(1, 0)
        assert version.version_string == "AEG/1.0"

    def test_compatible_same_major(self) -> None:
        v1 = AEGVersion(1, 0)
        v2 = AEGVersion(1, 3)
        assert v2.is_compatible_with(v1)
        assert v1.is_compatible_with(v1)

    def test_incompatible_different_major(self) -> None:
        v1 = AEGVersion(1, 0)
        v2 = AEGVersion(2, 0)
        assert not v2.is_compatible_with(v1)

    def test_incompatible_older_minor(self) -> None:
        v1 = AEGVersion(1, 5)
        v2 = AEGVersion(1, 0)
        assert not v2.is_compatible_with(v1)


class TestAEGManifest:
    """Tests for AEG manifest creation, verification, and serialization."""

    def test_create_manifest(self, small_architecture: ModelArchitecture) -> None:
        manifest = AEGManifest(
            model_id="test-model",
            aether_version="0.1.0",
            compiled_at="2026-07-27T00:00:00Z",
            graph_hash="sha256:abc123",
            architecture=small_architecture,
        )
        assert manifest.model_id == "test-model"
        assert manifest.format_version == "AEG/1.1"
        assert manifest.architecture.family == "llama_family"

    def test_manifest_json_roundtrip(self, small_architecture: ModelArchitecture) -> None:
        manifest = AEGManifest(
            model_id="test-model",
            aether_version="0.1.0",
            compiled_at="2026-07-27T00:00:00Z",
            graph_hash="sha256:abc123",
            architecture=small_architecture,
        )
        manifest.compute_and_set_manifest_hash()
        json_str = manifest.to_json()
        loaded = AEGManifest.from_json(json_str)
        assert loaded.model_id == manifest.model_id
        assert loaded.format_version == manifest.format_version
        assert loaded.manifest_hash == manifest.manifest_hash

    def test_manifest_verify_ok(self, small_architecture: ModelArchitecture) -> None:
        manifest = AEGManifest(
            model_id="test-model",
            aether_version="0.1.0",
            compiled_at="2026-07-27T00:00:00Z",
            graph_hash="sha256:abc123",
            architecture=small_architecture,
        )
        manifest.compute_and_set_manifest_hash()
        manifest.verify()

    def test_manifest_verify_version_error(self, small_architecture: ModelArchitecture) -> None:
        manifest = AEGManifest(
            model_id="test-model",
            aether_version="0.1.0",
            compiled_at="2026-07-27T00:00:00Z",
            graph_hash="sha256:abc123",
            architecture=small_architecture,
            format_version="AEG/99.0",
        )
        manifest.compute_and_set_manifest_hash()
        with pytest.raises(AEGVersionError):
            manifest.verify()

    def test_manifest_verify_hash_mismatch(self, small_architecture: ModelArchitecture) -> None:
        manifest = AEGManifest(
            model_id="test-model",
            aether_version="0.1.0",
            compiled_at="2026-07-27T00:00:00Z",
            graph_hash="sha256:abc123",
            architecture=small_architecture,
            manifest_hash="sha256:tampered",
        )
        with pytest.raises(AEGIntegrityError):
            manifest.verify()


class TestAEGPackage:
    """Tests for AEG package creation, saving, loading, and verification."""

    def test_create_package(self, tmp_path: Path) -> None:
        package = AEGPackage.create(
            tmp_path / "my_model.aeg",
            model_id="Qwen/Qwen3-8B",
            aether_version="0.1.0",
        )
        assert package.model_id == "Qwen/Qwen3-8B"
        assert not package.is_loaded

    def test_save_and_load_package(self, minimal_aeg_package: AEGPackage) -> None:
        package = minimal_aeg_package
        loaded = AEGPackage(package.root).load()
        assert loaded.is_loaded
        assert loaded.model_id == "test-model"
        assert loaded.manifest is not None
        assert loaded.manifest.graph_hash == package.manifest.graph_hash

    def test_integrity_check(self, minimal_aeg_package: AEGPackage) -> None:
        minimal_aeg_package.verify_integrity()

    def test_integrity_fail_on_tampered_manifest(self, tmp_path: Path, minimal_aeg_package: AEGPackage) -> None:
        # Tamper with the manifest
        manifest_path = minimal_aeg_package.root / "manifest.json"
        original = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(original.replace("test-model", "tampered-model"), encoding="utf-8")
        package = AEGPackage(minimal_aeg_package.root)
        with pytest.raises(AEGIntegrityError):
            package.verify_integrity()

    def test_set_precision_map(self, minimal_aeg_package: AEGPackage) -> None:
        pkg = minimal_aeg_package
        precision_map = {"embedding": "BF16", "layer_0": "Q4_K_M", "lm_head": "BF16"}
        pkg.set_precision_map(precision_map)
        pkg.save()
        loaded = AEGPackage(pkg.root).load()
        assert loaded.get_precision_map() == precision_map

    def test_load_from_directory(self, minimal_aeg_package: AEGPackage) -> None:
        loaded = load_aeg_package(minimal_aeg_package.root)
        assert loaded.model_id == "test-model"

    def test_copy_package(self, tmp_path: Path, minimal_aeg_package: AEGPackage) -> None:
        dest = tmp_path / "copied_model.aeg"
        copied = minimal_aeg_package.copy_to(dest)
        assert copied.root.exists()
        assert copied.model_id == "test-model"

    def test_compute_size(self, minimal_aeg_package: AEGPackage) -> None:
        size = minimal_aeg_package.compute_size()
        assert size > 0


class TestShardingPlans:
    """Tests for default sharding plan creation."""

    def test_default_plans_small_model(self) -> None:
        arch = ModelArchitecture(
            family="llama_family",
            params_billion=7.0,
            layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
        )
        plans = create_default_sharding_plans(arch)
        assert 1 in plans
        assert 2 in plans
        assert 4 in plans

    def test_sharding_plan_validation(self) -> None:
        plan = ShardingPlan(num_gpus=4, phase="prefill", tensor_parallel_degree=4)
        assert plan.num_gpus == 4
        assert plan.phase == "prefill"

    def test_sharding_plan_invalid_tp(self) -> None:
        with pytest.raises(ValueError):
            ShardingPlan(num_gpus=1, phase="prefill", tensor_parallel_degree=0)

    def test_sharding_plan_serialization(self) -> None:
        plan = ShardingPlan(
            num_gpus=8,
            phase="decode",
            tensor_parallel_degree=4,
            pipeline_stages=2,
            memory_per_gpu_gb=24.0,
        )
        data = plan.to_dict()
        loaded = ShardingPlan.from_dict(data)
        assert loaded.num_gpus == plan.num_gpus
        assert loaded.phase == plan.phase
        assert loaded.tensor_parallel_degree == plan.tensor_parallel_degree
