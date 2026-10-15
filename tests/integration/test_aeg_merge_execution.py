"""Real AEG merging, persisted weights, integrity and CPU numerical execution."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aether.compiler.aeg_merging import merge_aeg_packages, merge_tensor_arrays


def test_merge_numerics_are_shape_checked_and_magnitude_based():
    base = np.zeros(1, dtype=np.float32)
    sources = [np.asarray([1.0]), np.asarray([1.0]), np.asarray([-10.0])]
    # Sign election is by total magnitude, not the count of signs.
    assert merge_tensor_arrays(base, sources, [1, 1, 1], "ties")[0] == -10
    np.testing.assert_allclose(merge_tensor_arrays(base, sources[:2], [0.2, 0.8], "linear"), [1])
    with pytest.raises(ValueError, match="shapes"):
        merge_tensor_arrays(base, [np.zeros(2)], [1], "linear")
    with pytest.raises(ValueError, match="sum to 1"):
        merge_tensor_arrays(base, [base], [2], "linear")
    with pytest.raises(ValueError, match="Unsupported"):
        merge_tensor_arrays(base, [base], [1], "unknown")


def test_actual_linear_aeg_merge_matches_reference_logits_and_cli(
    tiny_local_safetensors_model, tmp_path, monkeypatch
):
    from click.testing import CliRunner

    from aether import Compiler, CompilerConfig
    from aether.cli import cli
    from aether.core.aeg_format import AEGPackage
    from aether.quantization.formats import quantize_tensor
    from aether.runtime.aeg_loader import load_engine_from_package

    monkeypatch.setenv("AETHER_BACKEND", "aether_cpu")
    monkeypatch.setenv("AETHER_EXECUTION_DEVICES", "cpu")
    monkeypatch.setenv("AETHER_HF_OFFLINE", "1")
    first = Compiler(
        CompilerConfig(
            targets=["cpu_avx2"],
            precision="FP32",
            enable_sensitivity=False,
            cache_dir=tmp_path / "cache",
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "first.aeg")
    second_path = tmp_path / "second.aeg"
    first.metadata["evaluation_status"] = "certified"
    first.metadata["eval_report"] = {"fixture_score": 1.0}
    first.metadata["fingerprint"] = {"fixture_weight_hash": "old-weights"}
    first.save()
    shutil.copytree(first.root, second_path)
    second = AEGPackage(second_path).load()
    original = first.weight_store().dequantize_all()
    edited = {name: array * 1.02 for name, array in original.items()}
    entries = first.weight_store().entries
    second.weights = {
        name: quantize_tensor(array, entries[name].precision, entries[name].block_size)
        for name, array in edited.items()
    }
    second.save()
    second.verify_integrity()
    runner = CliRunner()
    output = tmp_path / "merged.aeg"
    reply = runner.invoke(
        cli,
        [
            "merge",
            str(first.root),
            str(second_path),
            "--method",
            "linear",
            "--weights",
            "0.25,0.75",
            "--output",
            str(output),
        ],
    )
    assert reply.exit_code == 0, reply.output
    merged = AEGPackage(output).load()
    merged.verify_integrity()
    assert merged.metadata["merge"]["quality_validated"] is False
    assert merged.metadata["evaluation_status"] == "uncertified"
    assert "eval_report" not in merged.metadata
    assert "fingerprint" not in merged.metadata
    assert not (merged.root / "observability" / "eval_report.json").exists()
    assert not merged.manifest.provenance.c2pa_binding
    actual_weights = merged.weight_store().dequantize_all()
    persisted_second = second.weight_store().dequantize_all()
    expected = {name: 0.25 * original[name] + 0.75 * persisted_second[name] for name in original}
    for name in original:
        # Preserving base BF16 storage applies its documented format rounding.
        from aether.quantization.formats import dequantize_tensor

        rounded = dequantize_tensor(
            quantize_tensor(expected[name], entries[name].precision, entries[name].block_size)
        )
        np.testing.assert_array_equal(actual_weights[name], rounded)
    reference_path = tmp_path / "reference.aeg"
    shutil.copytree(first.root, reference_path)
    reference = AEGPackage(reference_path).load()
    reference.weights = {
        name: quantize_tensor(array, entries[name].precision, entries[name].block_size)
        for name, array in expected.items()
    }
    reference.save()
    engine = load_engine_from_package(merged, prepare_native_kernels=False)
    reference_engine = load_engine_from_package(reference, prepare_native_kernels=False)
    prompt = np.asarray([1, 2], dtype=np.int64)
    np.testing.assert_allclose(
        engine.forward(prompt)[0], reference_engine.forward(prompt)[0], atol=1e-6, rtol=1e-6
    )
    assert engine.generate(prompt, max_tokens=2, temperature=0) == reference_engine.generate(
        prompt, max_tokens=2, temperature=0
    )
    assert runner.invoke(cli, ["merge-info", str(output)]).exit_code == 0
    with pytest.raises(FileExistsError):
        merge_aeg_packages([first.root, second_path], output)
    with pytest.raises(ValueError, match="overlap"):
        merge_aeg_packages([first.root, second_path], first.root / "inside.aeg")
    (second_path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    second.save()
    with pytest.raises(ValueError, match="tokenizer"):
        merge_aeg_packages([first.root, second_path], tmp_path / "bad.aeg")


def test_copy_validation_rejects_undeclared_windows_reparse_points(tmp_path, monkeypatch):
    from aether.compiler.aeg_merging import _validate_input_tree

    member = tmp_path / "undeclared-member"
    member.write_text("local placeholder", encoding="utf-8")
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == member:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="Linked AEG"):
        _validate_input_tree(tmp_path)
