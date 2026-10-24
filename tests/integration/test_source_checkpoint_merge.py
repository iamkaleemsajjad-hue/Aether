"""Actual source-checkpoint merging follows ingestion and stored coordinates."""

from __future__ import annotations

import shutil

import numpy as np
import pytest


def test_actual_source_cli_merge_updates_all_checkpoint_projections(tiny_local_safetensors_model, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from safetensors.numpy import load_file, save_file
    from aether import Compiler, CompilerConfig
    from aether.cli import cli
    from aether.core.aeg_format import load_aeg_package
    from aether.quantization.formats import dequantize_tensor, quantize_tensor
    from aether.runtime.aeg_loader import load_engine_from_package

    monkeypatch.setenv("AETHER_BACKEND", "aether_cpu")
    monkeypatch.setenv("AETHER_EXECUTION_DEVICES", "cpu")
    monkeypatch.setenv("AETHER_HF_OFFLINE", "1")
    first = tiny_local_safetensors_model
    second = tmp_path / "fine-tuned"
    reference_source = tmp_path / "reference-source"
    shutil.copytree(first, second)
    shutil.copytree(first, reference_source)
    originals = load_file(str(first / "model.safetensors"))
    edited = {name: values * 1.02 for name, values in originals.items()}
    save_file(edited, str(second / "model.safetensors"))
    expected = {name: (0.25 * values.astype(np.float64) + 0.75 * edited[name]).astype(np.float32) for name, values in originals.items()}
    save_file(expected, str(reference_source / "model.safetensors"))
    destination = tmp_path / "merged.aeg"
    response = CliRunner().invoke(cli, [
        "--diagnostic",
        "--cache-dir", str(tmp_path / "cli-cache"), "merge", str(first), str(second),
        "--method", "linear", "--weights", "0.25,0.75", "--output", str(destination),
    ])
    assert response.exit_code == 0, response.output
    merged = load_aeg_package(destination)
    merged.verify_integrity()
    assert merged.metadata["model_merging"]["n_tensors_updated"] == len(originals)
    assert merged.metadata["model_merging"]["quality_validated"] is False
    # Compile an independently interpolated source; reproduce only storage
    # formats, so sensitivity choices cannot obscure a coordinate mismatch.
    reference = Compiler(CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_avx2"], enable_sensitivity=False,
        cache_dir=tmp_path / "reference-cache",
    )).compile(str(reference_source), output_path=tmp_path / "reference.aeg")
    actual = merged.weight_store().dequantize_all()
    expected_weights = reference.weight_store().dequantize_all()
    for name, entry in merged.weight_store().entries.items():
        rounded = dequantize_tensor(quantize_tensor(expected_weights[name], entry.precision, entry.block_size))
        np.testing.assert_allclose(actual[name], rounded, rtol=1e-6, atol=1e-6)
    engine = load_engine_from_package(merged, prepare_native_kernels=False, allow_uncertified=True)
    assert len(engine.generate(np.asarray([1, 2], dtype=np.int64), max_tokens=2, temperature=0)) > 0


def test_source_alignment_preserves_partial_keys_and_rejects_same_size_wrong_shape(tiny_local_safetensors_model, tmp_path):
    from safetensors.numpy import load_file, save_file
    from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
    from aether.compiler.stage2_optimizer.pass12_model_merging import _load_source_weights

    architecture = ArchitectureDetector().detect(str(tiny_local_safetensors_model))
    graph = IngestionPipeline().ingest(str(tiny_local_safetensors_model), architecture)
    originals = load_file(str(tiny_local_safetensors_model / "model.safetensors"))
    partial = tmp_path / "adapter.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": originals["model.layers.0.mlp.down_proj.weight"] * 1.1}, str(partial))
    aligned = _load_source_weights(str(partial), graph)
    assert len(aligned) == 1
    wrong = tmp_path / "wrong.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": originals["model.layers.0.mlp.down_proj.weight"].T.copy()}, str(wrong))
    with pytest.raises(ValueError, match="shape"):
        _load_source_weights(str(wrong), graph)


def test_aeg_base_source_alignment_reuses_pack_names_and_exact_shapes(tiny_local_safetensors_model, tmp_path):
    from safetensors.numpy import save_file
    from aether import Compiler, CompilerConfig
    from aether.compiler.stage2_optimizer.pass12_model_merging import _load_source_weights

    base = Compiler(CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_avx2"], precision="FP32", enable_sensitivity=False,
        cache_dir=tmp_path / "cache",
    )).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "base.aeg")
    aligned = _load_source_weights(str(tiny_local_safetensors_model), base)
    assert set(aligned) == set(base.weight_store().entries)
    partial = tmp_path / "canonical.safetensors"
    values = base.weight_store().dequantize_all()["embedding"]
    save_file({"embedding": values}, str(partial))
    assert set(_load_source_weights(str(partial), base)) == {"embedding"}
    save_file({"embedding": values.T.copy()}, str(partial))
    with pytest.raises(ValueError, match="shape"):
        _load_source_weights(str(partial), base)
