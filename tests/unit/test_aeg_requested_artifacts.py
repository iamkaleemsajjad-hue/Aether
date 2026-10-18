"""Optional package payloads require a producer, not a format default."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aether.core.aeg_format import AEGPackage, KernelSetMetadata
from aether.core.constants import AEG_GRAPH_FILENAME
from aether.core.exceptions import AEGIntegrityError
from aether.core.hash_utils import compute_file_hash

if TYPE_CHECKING:
    from pathlib import Path

OPTIONAL_ARTIFACTS = {
    "reasoning_graph": "graph/reasoning_graph.aeg-ir",
    "rag_pipeline": "graph/rag_pipeline.aeg-ir",
    "attention_head_patterns": "graph/attention_head_patterns.json",
    "prefill_decode_split": "parallelism/prefill_decode_split.json",
    "compute_profiles": "inference/compute_profiles.json",
    "prompt_guard": "safety/prompt_guard.json",
    "output_filter": "safety/output_filter.json",
    "audit_log": "safety/audit_log.json",
    "provenance": "provenance/manifest.json",
    "fingerprint": "provenance/fingerprint.json",
    "watermark": "watermark/config.json",
    "adapters": "adapters/manifest.json",
    "cuda_graphs": "cuda_graphs/capture_manifest.json",
    "eval_gates": "observability/eval_gates.json",
    "drift_monitor": "observability/drift_monitor.json",
    "metrics_schema": "observability/metrics_schema.json",
    "ab_rollout": "rollout/ab_config.json",
    "fleet_deployment": "fleet/deployment_plan.json",
    "hot_reload": "fleet/hot_reload.json",
    "distillation": "distillation/plan.json",
    "agentic_workflow": "agentic/workflow_cache.json",
    "mla_plan": "mla/plan.json",
    "eagle3": "speculation/eagle3.json",
    "multimodal_graph": "multimodal/graph.json",
}


def test_missing_source_hash_license_and_compliance_remain_unknown() -> None:
    """Identifiers and evaluation scores cannot supply source-weight evidence."""
    from aether.provenance.manifest import ProvenanceManifest

    manifest = ProvenanceManifest(source_model_id="named-model", eval_results={"score": 0.8})
    assert manifest.model_hash == ""
    payload = manifest.to_dict()
    assert payload["model_hash"] == ""
    assert payload["source_model"]["license"] == ""
    assert payload["eu_ai_act"]["risk_category"] == "unknown"
    assert payload["eu_ai_act"]["transparency_obligations_met"] is False
    assert payload["hardware_certification"]["certified_targets"] == []


def test_minimal_package_omits_unrequested_artifacts(minimal_aeg_package: AEGPackage) -> None:
    """An ordinary save must contain no optional feature templates."""
    package = minimal_aeg_package
    expected = {
        "manifest.json",
        "FORMAT_VERSION",
        f"graph/{AEG_GRAPH_FILENAME}",
        "graph/graph.sha256",
        "graph/metadata.json",
        "weights/precision_map.json",
        "weights/quantized/precision_map.json",
    }
    files = {
        path.relative_to(package.root).as_posix()
        for path in package.root.rglob("*")
        if path.is_file()
    }
    assert files == expected
    assert set(package.manifest.artifacts) == expected - {"manifest.json"}
    AEGPackage(package.root).load().verify_integrity()


@pytest.mark.parametrize("key,relative_path", OPTIONAL_ARTIFACTS.items())
def test_explicit_metadata_payload_is_preserved(
    minimal_aeg_package: AEGPackage,
    key: str,
    relative_path: str,
) -> None:
    """Explicit producer data remains serializable and integrity protected."""
    package = minimal_aeg_package
    payload = {"producer": key, "requested": True}
    package.metadata[key] = payload
    package.save()
    assert json.loads((package.root / relative_path).read_text(encoding="utf-8")) == payload
    loaded = AEGPackage(package.root).load()
    loaded.verify_integrity()
    assert relative_path in loaded.manifest.artifacts


@pytest.mark.parametrize("payload", [{}, [], False, 0, ""])
def test_explicit_empty_payload_is_not_treated_as_missing(
    minimal_aeg_package: AEGPackage, payload: object
) -> None:
    """False and empty producer values must survive without invented defaults."""
    package = minimal_aeg_package
    package.metadata["compute_profiles"] = payload
    package.save()
    text = (package.root / "inference/compute_profiles.json").read_text(encoding="utf-8")
    assert text == "" if payload == "" else json.loads(text) == payload


def test_none_payloads_do_not_create_artifacts(minimal_aeg_package: AEGPackage) -> None:
    """A missing producer value must not create a payload."""
    package = minimal_aeg_package
    package.metadata.update(dict.fromkeys(OPTIONAL_ARTIFACTS))
    package.save()
    for relative_path in OPTIONAL_ARTIFACTS.values():
        assert not (package.root / relative_path).exists(), relative_path


def test_library_writer_files_survive_save(minimal_aeg_package: AEGPackage) -> None:
    """Saving preserves real library-writer output and its readers."""
    from aether.inference.multimodal import ModalityEncoder, MultiModalGraphPlan
    from aether.inference.rag import RAGPipelinePlan, RetrievalSource
    from aether.provenance.manifest import ProvenanceManifest
    from aether.safety.policy import SafetyManifestWriter

    package = minimal_aeg_package
    rag = RAGPipelinePlan("embed.aeg", "llm.aeg", sources=[RetrievalSource("docs", "vector")])
    multimodal = MultiModalGraphPlan("llm.aeg", encoders=(ModalityEncoder("image", "vit.aeg"),))
    paths = [rag.save(package.root), multimodal.save(package.root)]
    paths.extend(SafetyManifestWriter().write(package.root).values())
    paths.append(
        ProvenanceManifest(source_model_id="explicit", model_hash="ab" * 32).save(package.root)
    )
    before = {path: path.read_bytes() for path in paths}
    package.save()
    loaded = AEGPackage(package.root).load()
    loaded.verify_integrity()
    assert {path: path.read_bytes() for path in paths} == before
    assert RAGPipelinePlan.load(package.root).sources == rag.sources
    assert MultiModalGraphPlan.load(package.root).encoders == multimodal.encoders
    assert loaded.manifest.provenance.model_hash == "ab" * 32


def test_legacy_auxiliary_files_survive_load_resave_and_archive(
    tmp_path: Path,
    minimal_aeg_package: AEGPackage,
) -> None:
    """Existing optional payload bytes remain readable across a resave and archive."""
    package = minimal_aeg_package
    before = {}
    for key, relative_path in OPTIONAL_ARTIFACTS.items():
        path = package.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"legacy_producer": key}, indent=2, sort_keys=True).encode()
        path.write_bytes(payload)
        before[relative_path] = payload
    # Construct the old layout independently of the optional-artifact writer.
    package.manifest.artifacts = {
        path.relative_to(package.root).as_posix(): compute_file_hash(path)
        for path in package.root.rglob("*")
        if path.is_file() and path != package.root / "manifest.json"
    }
    package.manifest.compute_and_set_manifest_hash()
    (package.root / "manifest.json").write_text(package.manifest.to_json(), encoding="utf-8")
    loaded = AEGPackage(package.root).load()
    loaded.verify_integrity()
    archive = loaded.save_as_archive(tmp_path / "legacy.tar.gz")
    restored = AEGPackage.load_from_archive(archive)
    restored.verify_integrity()
    for relative_path, payload in before.items():
        assert (restored.root / relative_path).read_bytes() == payload


@pytest.mark.parametrize("remove", [False, True])
def test_optional_payloads_still_require_integrity(
    minimal_aeg_package: AEGPackage,
    remove: bool,
) -> None:
    """Removing or altering an explicitly supplied payload must fail integrity."""
    package = minimal_aeg_package
    path = package.root / "fleet/deployment_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"explicit": true}', encoding="utf-8")
    package.save()
    loaded = AEGPackage(package.root).load()
    if remove:
        path.unlink()
    else:
        path.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(AEGIntegrityError):
        loaded.verify_integrity()


def test_invalidated_evidence_is_not_resurrected(minimal_aeg_package: AEGPackage) -> None:
    """Weight changes must not resurrect deleted evaluation or provenance evidence."""
    from aether.compiler.aeg_merging import invalidate_weight_dependent_evidence

    package = minimal_aeg_package
    package.metadata.update(
        {
            "provenance": {},
            "fingerprint": {},
            "watermark": {"enabled": True},
            "eval_report": {"gate_decision": {"passed": True}},
        }
    )
    package.save()
    invalidate_weight_dependent_evidence(package)
    package.save()
    for relative_path in (
        "provenance/manifest.json",
        "provenance/fingerprint.json",
        "watermark/config.json",
        "observability/eval_report.json",
    ):
        assert not (package.root / relative_path).exists(), relative_path
        assert relative_path not in package.manifest.artifacts
    package.verify_integrity()


def test_architecture_alone_does_not_claim_mla_weight_absorption(
    minimal_aeg_package: AEGPackage,
) -> None:
    """A declared architecture does not prove compressed weights were produced."""
    package = minimal_aeg_package
    package.manifest.architecture.attention_type = "MLA"
    package.save()
    assert not (package.root / "weights/mla_compressed/latent_kv.manifest.json").exists()


def test_kernel_defaults_do_not_claim_flash_attention() -> None:
    """Only an explicit declaration may name a FlashAttention implementation."""
    assert KernelSetMetadata().flash_attention_variant == ""
    assert KernelSetMetadata.from_dict({}).flash_attention_variant == ""
    # Explicit legacy declarations remain readable.
    assert (
        KernelSetMetadata.from_dict(
            {"flash_attention_variant": "flash_attention_3"}
        ).flash_attention_variant
        == "flash_attention_3"
    )


def test_explicit_kernel_contract_roundtrips(minimal_aeg_package: AEGPackage) -> None:
    """Runtime-consumed target authorization survives save, reload and integrity."""
    package = minimal_aeg_package
    package.manifest.kernels = KernelSetMetadata(
        targets=["cpu_generic"],
        backend_plans={"cpu_generic": "aether_cpu"},
        portable_backends=["aether_cpu"],
        variant_status={"cpu_generic": "executable"},
    )
    package.save()
    loaded = AEGPackage(package.root).load()
    loaded.verify_integrity()
    assert loaded.get_backend_plan("cpu_generic") == "aether_cpu"
    assert loaded.supports_runtime_target("cpu_generic")
    assert (
        json.loads((package.root / "kernels/kernel_sets.json").read_text(encoding="utf-8"))
        == package.manifest.kernels.to_dict()
    )


def test_cpu_compile_has_execution_contract_without_demo_claims(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real CPU compile runs with honest provenance and no demo feature payloads."""
    from aether import Compiler
    from aether.compiler.config import CompilerConfig
    from aether.kernels.native_cpu import NativeCPUKernels
    from aether.runtime.aeg_loader import load_engine_from_package

    # This test never starts a compiler/toolchain process or requires a native DLL.
    monkeypatch.setattr(NativeCPUKernels, "ensure_compiled", lambda self: False)
    config = CompilerConfig(
        targets=["cpu_generic"],
        overwrite=True,
        calibration_tokens=8,
        enable_sensitivity=False,
        enable_reasoning_graph=False,
        enable_sparse_attention=False,
        enable_pruning=False,
    )
    package = Compiler(config).compile(
        str(tiny_local_safetensors_model), output_path=tmp_path / "cpu.aeg"
    )
    for key, relative_path in OPTIONAL_ARTIFACTS.items():
        if key != "provenance":
            assert key not in package.metadata, key
            assert not (package.root / relative_path).exists(), relative_path
    assert "capabilities" not in package.metadata
    assert "quantization_aware_compilation" not in package.metadata
    provenance = package.metadata["provenance"]
    assert provenance["source_model"]["id"] == package.model_id
    assert not provenance["source_model"].get("license")
    assert not provenance.get("model_hash")  # The IR hash is not a source-weight hash.
    assert provenance["watermark"]["enabled"] is False
    assert provenance["eu_ai_act"]["transparency_obligations_met"] is False
    assert provenance["hardware_certification"]["certified_targets"] == []
    assert "eval_report" not in package.metadata
    kernels = package.manifest.kernels
    assert kernels.flash_attention_variant == ""
    assert kernels.backend_plans == {"cpu_generic": "aether_cpu"}

    engine = load_engine_from_package(AEGPackage(package.root).load(), prepare_native_kernels=False)
    prompt = np.asarray([1, 2, 3], dtype=np.int64)
    logits, _ = engine.forward(prompt)
    expected_tokens = engine.generate(prompt, max_tokens=3, temperature=0.0)
    # Old template files must remain readable without becoming execution inputs.
    for key in ("watermark", "fleet_deployment", "compute_profiles", "cuda_graphs"):
        path = package.root / OPTIONAL_ARTIFACTS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"enabled": true, "legacy_template": true}', encoding="utf-8")
    package.save()
    restored = AEGPackage.load_from_archive(package.save_as_archive(tmp_path / "cpu.tar.gz"))
    reloaded_engine = load_engine_from_package(restored, prepare_native_kernels=False)
    actual_logits, _ = reloaded_engine.forward(prompt)
    np.testing.assert_array_equal(actual_logits, logits)
    assert reloaded_engine.generate(prompt, max_tokens=3, temperature=0.0) == expected_tokens
