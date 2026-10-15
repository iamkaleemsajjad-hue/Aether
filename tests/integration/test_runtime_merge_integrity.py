"""Runtime merging publishes authenticated weights and retains executable task vectors."""

from __future__ import annotations

import json
import shutil
from dataclasses import fields

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.core.aeg_format import AEGPackage
from aether.core.exceptions import RuntimeError as AetherRuntimeError


@pytest.fixture(scope="module")
def original_aeg(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-merge-source")
    checkpoint = root / "checkpoint"
    _write_probe_checkpoint(checkpoint)
    config = CompilerConfig(
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(root / "cache"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    return Compiler(config).compile(str(checkpoint), output_path=root / "original.aeg").root


@pytest.fixture
def base_aeg(original_aeg, tmp_path):
    path = tmp_path / "base.aeg"
    shutil.copytree(original_aeg, path)
    return AEGPackage(path).load()


@pytest.fixture
def runtime(tmp_path):
    instance = Runtime(
        RuntimeConfig(
            model_cache_dir=str(tmp_path / "cache"),
            backend_name="aether_cpu",
            execution_devices=["cpu"],
            hf_offline=True,
            enable_semantic_cache=False,
        )
    )
    yield instance
    for model_id, backend in list(instance._loaded_backends.items()):
        unload = getattr(backend, "unload_model", None)
        if callable(unload):
            unload(model_id)


def _source(base, path, changes):
    from safetensors.numpy import save_file

    weights = base.weight_store().dequantize_all()
    payload = {name: (weights[name] + delta).astype(np.float32) for name, delta in changes.items()}
    save_file(payload, str(path))
    return {"name": path.stem, "path": str(path), "coefficient": 1.0}


def _output(base):
    return base.root.with_name(base.root.name + ".merged")


def _no_partial_output(base):
    assert not _output(base).exists()
    assert not list(base.root.parent.glob(".aether-runtime-merge-*"))


def test_every_source_must_overlap_before_output_is_published(base_aeg, runtime, tmp_path):
    from safetensors.numpy import save_file

    valid = _source(base_aeg, tmp_path / "valid.safetensors", {"embedding": 0.01})
    unrelated = tmp_path / "unrelated.safetensors"
    save_file({"unrelated_parameter": np.ones((2, 2), dtype=np.float32)}, str(unrelated))
    with pytest.raises(AetherRuntimeError, match="overlap|matching|persisted tensor"):
        runtime.merge(str(base_aeg.root), [valid, {"name": "unrelated", "path": str(unrelated)}])
    _no_partial_output(base_aeg)


def test_duplicate_vector_names_fail_without_partial_output(base_aeg, runtime, tmp_path):
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    with pytest.raises(ValueError, match="duplicate task vector name"):
        runtime.merge(str(base_aeg.root), [source, source])
    _no_partial_output(base_aeg)


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), -float("inf"), True])
def test_coefficients_are_rejected_before_loading_inputs(runtime, coefficient):
    with pytest.raises(ValueError, match="coefficient"):
        runtime.merge(
            "nonexistent-model", [{"path": "nonexistent-source", "coefficient": coefficient}]
        )


def test_corrupt_base_is_rejected_before_any_output(base_aeg, runtime, tmp_path):
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    blob = base_aeg.root / "weights/quantized/model.aeg-quant"
    data = bytearray(blob.read_bytes())
    data[-1] ^= 0xFF
    blob.write_bytes(data)
    with pytest.raises(Exception, match="[Hh]ash|integrity|checksum"):
        runtime.merge(str(base_aeg.root), [source])
    _no_partial_output(base_aeg)


def _merged(runtime, base, sources, **options):
    result = runtime.merge(str(base.root), sources, **options)
    assert result["status"] == "merged"
    assert result["quality_validated"] is False
    package = AEGPackage(result["output_model"]).load()
    package.verify_integrity()
    return package


def test_partial_sources_persist_deltas_preserve_untouched_codes_and_execute(
    base_aeg, runtime, tmp_path
):
    from aether.quantization.formats import dequantize_tensor, quantize_tensor

    before = base_aeg.weight_store().dequantize_all()
    sources = [
        _source(base_aeg, tmp_path / "embedding.safetensors", {"embedding": 0.01}),
        _source(base_aeg, tmp_path / "head.safetensors", {"lm_head": -0.02}),
    ]
    sources[0]["coefficient"] = 2.0
    sources[1]["coefficient"] = 0.5
    merged = _merged(runtime, base_aeg, sources, method="task-arithmetic")
    after = merged.weight_store().dequantize_all()
    deltas = {}
    for name, change, coefficient in [("embedding", 0.01, 2.0), ("lm_head", -0.02, 0.5)]:
        deltas[name] = (before[name] + change).astype(np.float32).astype(np.float64) - before[
            name
        ].astype(np.float64)
        expected = (before[name].astype(np.float64) + coefficient * deltas[name]).astype(np.float32)
        entry = base_aeg.weight_store().entries[name]
        # Storage precision is part of the integration contract. The head is
        # BF16 in this real compiler fixture; it must retain that rounding.
        expected = dequantize_tensor(
            quantize_tensor(expected, entry.precision, block_size=entry.block_size)
        )
        np.testing.assert_array_equal(after[name], expected)
    for name in before.keys() - {"embedding", "lm_head"}:
        left = base_aeg.weight_store().load_tensor(name)
        right = merged.weight_store().load_tensor(name)
        np.testing.assert_array_equal(left.data, right.data)
        np.testing.assert_array_equal(left.scales, right.scales)
        np.testing.assert_array_equal(left.zero_points, right.zero_points)
    descriptors = merged.metadata["task_vectors"]["vectors"]
    assert [item["name"] for item in descriptors] == ["embedding", "head"]
    for item, name in zip(descriptors, ["embedding", "lm_head"]):
        assert item["tensors"][0]["shape"] == list(before[name].shape)
        with np.load(merged.root / item["path"], allow_pickle=False) as arrays:
            # Stored task vectors retain the raw source delta, not applied coefficients.
            np.testing.assert_array_equal(
                arrays[item["tensors"][0]["key"]], deltas[name].astype(np.float32).reshape(-1)
            )
    response = runtime.generate(str(merged.root), "hello", max_tokens=2, temperature=0.0)
    assert response.usage["completion_tokens"] > 0
    runtime.set_task_weights(str(merged.root), embedding=0.2, head=0.8)
    reweighted = runtime.generate(str(merged.root), "hello", max_tokens=2, temperature=0.0)
    assert reweighted.metrics.extra["task_reweighting"]["vectors"] == ["embedding", "head"]


def test_nondefault_quantization_block_size_is_preserved(base_aeg, runtime, tmp_path):
    from aether.quantization.formats import quantize_tensor

    store = base_aeg.weight_store()
    base_aeg.weights = {name: store.load_tensor(name) for name in store.load_index()}
    base_aeg.weights["embedding"] = quantize_tensor(
        store.dequantize_all()["embedding"], "INT8", block_size=16
    )
    base_aeg.save()
    base_aeg = AEGPackage(base_aeg.root).load()
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    merged = _merged(runtime, base_aeg, [source])
    entry = merged.weight_store().load_index()["embedding"]
    assert (entry.precision, entry.block_size) == ("INT8", 16)
    assert not np.array_equal(
        merged.weight_store().dequantize_all()["embedding"],
        base_aeg.weight_store().dequantize_all()["embedding"],
    )


def test_merge_invalidates_weight_evidence_and_preserves_auxiliary_payloads(
    base_aeg, runtime, tmp_path
):
    # These are stale test fixtures, not claims of real model-quality validation.
    base_aeg.manifest.optimization.actual_ppl_increase = 0.05
    base_aeg.metadata["eval_report"] = {"fixture": "stale evaluation", "gate": {"passed": True}}
    base_aeg.metadata["fingerprint"] = {"enabled": True, "fixture": "stale fingerprint"}
    base_aeg.metadata["provenance"] = {"source_model_hash": "stale-weight-hash"}
    old_signature = base_aeg.root / "provenance/old-signature.c2pa"
    old_signature.write_bytes(b"stale test fixture")
    auxiliary = base_aeg.root / "speculation/base-auxiliary.bin"
    auxiliary.write_bytes(b"preserve configured auxiliary payload")
    base_aeg.metadata["auxiliary_fixture"] = {"path": "speculation/base-auxiliary.bin"}
    base_aeg.save()
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    merged = _merged(runtime, base_aeg, [source])
    assert merged.manifest.optimization.actual_ppl_increase is None
    assert merged.metadata["evaluation_status"] == "uncertified"
    assert merged.metadata["evidence_invalidation"]["quality_validated"] is False
    for key in ("eval_report", "provenance", "fingerprint", "watermark", "tee", "tee_attestation"):
        assert key not in merged.metadata
    assert not (merged.root / "observability/eval_report.json").exists()
    assert not (merged.root / "provenance/old-signature.c2pa").exists()
    assert (merged.root / "speculation/base-auxiliary.bin").read_bytes() == auxiliary.read_bytes()
    assert merged.metadata["auxiliary_fixture"] == base_aeg.metadata["auxiliary_fixture"]
    assert (
        merged.metadata["model_merge"]["auxiliary_policy"] == "preserve_base_requires_reevaluation"
    )


def test_quantization_failure_cleans_staging_and_never_publishes(
    base_aeg, runtime, tmp_path, monkeypatch
):
    import aether.quantization.formats as formats

    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})

    def fail_quantization(*args, **kwargs):
        raise ValueError("injected quantization failure")

    monkeypatch.setattr(formats, "quantize_tensor", fail_quantization)
    with pytest.raises(ValueError, match="injected quantization failure"):
        runtime.merge(str(base_aeg.root), [source])
    _no_partial_output(base_aeg)


def test_final_verification_failure_cleans_staging(base_aeg, runtime, tmp_path, monkeypatch):
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    original_verify = AEGPackage.verify_integrity

    def verify(package):
        if package.metadata.get("model_merge"):
            raise ValueError("injected final verification failure")
        return original_verify(package)

    monkeypatch.setattr(AEGPackage, "verify_integrity", verify)
    with pytest.raises(ValueError, match="injected final verification failure"):
        runtime.merge(str(base_aeg.root), [source])
    _no_partial_output(base_aeg)


def test_equal_element_count_wrong_shape_is_rejected(base_aeg, runtime, tmp_path):
    from safetensors.numpy import save_file

    values = base_aeg.weight_store().dequantize_all()["embedding"]
    source = tmp_path / "wrong.safetensors"
    save_file({"embedding": values.reshape(-1)}, str(source))
    with pytest.raises(ValueError, match="shape"):
        runtime.merge(str(base_aeg.root), [{"path": str(source)}])
    _no_partial_output(base_aeg)


@pytest.mark.parametrize("density", [0.0, -1.0, 1.1, float("nan"), float("inf"), True])
def test_density_is_validated_before_input_loading(runtime, density):
    with pytest.raises(ValueError, match="density"):
        runtime.merge("missing-base", [{"path": "missing-source"}], density=density, method="dare")


@pytest.mark.parametrize("method", ["task_arithmetic", "soup", "free", "evolutionary"])
def test_unsupported_density_is_explicitly_rejected(runtime, method):
    with pytest.raises(ValueError, match="density is not supported"):
        runtime.merge("missing-base", [{"path": "missing-source"}], density=0.5, method=method)


def test_soup_does_not_silently_discard_nonuniform_coefficients(runtime):
    with pytest.raises(ValueError, match="uniform coefficients"):
        runtime.merge(
            "missing-base",
            [{"path": "a", "coefficient": 0.2}, {"path": "b", "coefficient": 0.8}],
            method="soup",
        )


@pytest.mark.parametrize(
    "method",
    ["task_arithmetic", "dare", "ties", "dare_ties", "free_merging", "evolutionary", "soup"],
)
def test_supported_methods_publish_loadable_uncertified_outputs(
    base_aeg, runtime, tmp_path, method
):
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    merged = _merged(runtime, base_aeg, [source], method=method)
    assert merged.metadata["model_merge"]["method"] == (
        "free" if method == "free_merging" else method
    )
    assert merged.metadata["model_merge"]["quality_validated"] is False
    np.testing.assert_allclose(
        merged.weight_store().dequantize_all()["embedding"],
        (base_aeg.weight_store().dequantize_all()["embedding"] + 0.01).astype(np.float32),
        atol=2e-8,
    )
    manifest = json.loads((merged.root / "merging/merge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "aether_merge_manifest_v1"


def test_ties_density_retains_exact_top_fraction(base_aeg, runtime, tmp_path):
    before = base_aeg.weight_store().dequantize_all()["embedding"]
    changes = np.linspace(0.01, 1.0, before.size, dtype=np.float32).reshape(before.shape)
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": changes})
    merged = _merged(runtime, base_aeg, [source], method="ties", density=0.25)
    delta = merged.weight_store().dequantize_all()["embedding"] - before
    keep = int(np.ceil(before.size * 0.25))
    assert np.count_nonzero(delta) == keep
    np.testing.assert_array_equal(delta.reshape(-1)[:-keep], np.zeros(before.size - keep))
    expected = ((before + changes).astype(np.float32) - before).reshape(-1)
    np.testing.assert_allclose(delta.reshape(-1)[-keep:], expected[-keep:], atol=5e-8)


def test_ties_density_tie_breaking_retains_exact_fraction(base_aeg, runtime, tmp_path):
    before = base_aeg.weight_store().dequantize_all()["embedding"]
    # Zero the base to make every task coordinate's magnitude exactly equal.
    from aether.quantization.formats import quantize_tensor

    store = base_aeg.weight_store()
    base_aeg.weights = {name: store.load_tensor(name) for name in store.load_index()}
    base_aeg.weights["embedding"] = quantize_tensor(np.zeros_like(before), "FP32")
    base_aeg.save()
    base_aeg = AEGPackage(base_aeg.root).load()
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 1.0})
    merged = _merged(runtime, base_aeg, [source], method="ties", density=0.25)
    values = merged.weight_store().dequantize_all()["embedding"].reshape(-1)
    keep = int(np.ceil(before.size * 0.25))
    np.testing.assert_array_equal(values[:keep], np.ones(keep))
    np.testing.assert_array_equal(values[keep:], np.zeros(before.size - keep))


def test_dare_density_controls_drop_rescale_and_is_reproducible(base_aeg, runtime, tmp_path):
    import random

    before = base_aeg.weight_store().dequantize_all()["embedding"]
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.125})
    merged = _merged(runtime, base_aeg, [source], method="dare", density=0.5)
    original_delta = (
        (before + 0.125).astype(np.float32).astype(np.float64) - before.astype(np.float64)
    ).reshape(-1)
    rng = random.Random(42)
    expected_delta = np.asarray(
        [2.0 * value if rng.random() > 0.5 else 0.0 for value in original_delta]
    )
    expected = (before.astype(np.float64) + expected_delta.reshape(before.shape)).astype(np.float32)
    np.testing.assert_array_equal(merged.weight_store().dequantize_all()["embedding"], expected)
    second_root = tmp_path / "second.aeg"
    shutil.copytree(base_aeg.root, second_root)
    second = _merged(runtime, AEGPackage(second_root).load(), [source], method="dare", density=0.5)
    np.testing.assert_array_equal(second.weight_store().dequantize_all()["embedding"], expected)


def test_dare_ties_performs_sign_election_instead_of_plain_dare_alias(base_aeg, runtime, tmp_path):
    before = base_aeg.weight_store().dequantize_all()["embedding"]
    sources = [
        _source(base_aeg, tmp_path / "positive.safetensors", {"embedding": 1.0}),
        _source(base_aeg, tmp_path / "negative.safetensors", {"embedding": -2.0}),
    ]
    for source in sources:
        source["coefficient"] = 0.5
    merged = _merged(runtime, base_aeg, sources, method="dare_ties", density=1.0)
    # Weighted sign election chooses the negative vector, then disjoint mean
    # retains that full delta. Plain DARE would have produced -.5 instead.
    np.testing.assert_array_equal(
        merged.weight_store().dequantize_all()["embedding"], (before - 2.0).astype(np.float32)
    )
    assert merged.metadata["model_merge"]["density_policy"] == "drop_then_per_tensor_trim"


def test_existing_destination_is_never_overwritten(base_aeg, runtime, tmp_path):
    source = _source(base_aeg, tmp_path / "task.safetensors", {"embedding": 0.01})
    destination = _output(base_aeg)
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_bytes(b"existing artifact")
    with pytest.raises(AetherRuntimeError, match="already exists"):
        runtime.merge(str(base_aeg.root), [source])
    assert sentinel.read_bytes() == b"existing artifact"
    assert list(destination.iterdir()) == [sentinel]
    assert not list(base_aeg.root.parent.glob(".aether-runtime-merge-*"))


def test_corrupt_aeg_source_is_rejected_before_output(base_aeg, runtime, tmp_path):
    source_root = tmp_path / "task.aeg"
    shutil.copytree(base_aeg.root, source_root)
    blob = source_root / "weights/quantized/model.aeg-quant"
    data = bytearray(blob.read_bytes())
    data[-1] ^= 0xFF
    blob.write_bytes(data)
    with pytest.raises(ValueError, match="[Hh]ash|integrity|checksum"):
        runtime.merge(str(base_aeg.root), [{"name": "task", "path": str(source_root)}])
    _no_partial_output(base_aeg)
