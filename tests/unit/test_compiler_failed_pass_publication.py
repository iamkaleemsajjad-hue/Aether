"""Real failed optimizer passes must not publish or replace runnable artifacts."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import fields
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

from aether import Compiler, CompilerConfig
from aether.core.exceptions import CompilationError
from aether.runtime.aeg_loader import load_engine_from_path

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _config(root: Path, *, merging: bool, sources: list[str]) -> CompilerConfig:
    config = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"], optimization_level=0, skip_download=True,
        overwrite=True, cache_dir=str(root / "cache"),
        model_merging_sources=sources, model_merging_coefficients=[0.5] if sources else [],
    )
    for option in fields(config):
        if option.name.startswith("enable_"):
            setattr(config, option.name, False)
    config.enable_model_merging = merging
    return config


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _capture(root: Path) -> tuple[NDArray[np.float32], list[int]]:
    engine = load_engine_from_path(root, prepare_native_kernels=False, allow_uncertified=True)
    prompt = np.asarray([1, 2], dtype=np.int64)
    logits, _ = engine.forward(prompt)
    tokens = list(engine.generate_iter(prompt, max_tokens=3, temperature=0.0))
    return np.asarray(logits), tokens


@pytest.mark.parametrize("existing_output", [False, True])
def test_failed_real_merge_preserves_destination_and_cleans_staging(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_output: bool,
) -> None:
    """Missing requested source fails in the real pass before targeting or writes."""
    output = tmp_path / "model.aeg"
    original_files = {}
    original_logits = None
    original_tokens = None
    if existing_output:
        Compiler(_config(tmp_path, merging=False, sources=[])).compile(
            str(tiny_local_safetensors_model), output_path=output
        )
        original_files = _inventory(output)
        original_logits, original_tokens = _capture(output)

    staging = tmp_path / "private-staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    compiler = Compiler(_config(tmp_path, merging=True, sources=[str(tmp_path / "missing")]))
    target_spy = Mock(wraps=compiler._stage3_target)
    monkeypatch.setattr(compiler, "_stage3_target", target_spy)

    with pytest.raises(CompilationError, match="Every requested merge source must load") as caught:
        compiler.compile(str(tiny_local_safetensors_model), output_path=output)

    assert caught.value.stage == "stage2_optimizer"
    assert caught.value.model_id == tiny_local_safetensors_model.name
    failed = caught.value.details["failed_passes"]
    assert len(failed) == 1
    assert failed[0]["pass_name"] == "model_merging"
    assert failed[0]["status"] == "failed"
    assert "partial merges are not allowed" in failed[0]["details"]["error"]
    target_spy.assert_not_called()
    assert not list(staging.glob("aether-pass-artifacts-*"))
    if existing_output:
        assert _inventory(output) == original_files
        assert original_logits is not None
        assert original_tokens is not None
        logits, tokens = _capture(output)
        np.testing.assert_array_equal(logits, original_logits)
        assert tokens == original_tokens
    else:
        assert not output.exists()


def test_benign_skipped_merge_preserves_real_weights_and_generation(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """An explicitly enabled pass with no sources may still report a benign skip."""
    baseline = Compiler(_config(tmp_path, merging=False, sources=[])).compile(
        str(tiny_local_safetensors_model), output_path=tmp_path / "baseline.aeg"
    )
    skipped = Compiler(_config(tmp_path, merging=True, sources=[])).compile(
        str(tiny_local_safetensors_model), output_path=tmp_path / "skipped.aeg"
    )
    skipped.verify_integrity()
    statuses = {
        entry["pass"]: entry["parameters"]["status"]
        for entry in skipped.metadata["provenance"]["transformations"]
    }
    assert statuses["model_merging"] == "skipped"
    weights_a = baseline.weight_store().dequantize_all()
    weights_b = skipped.weight_store().dequantize_all()
    assert weights_a.keys() == weights_b.keys()
    for name in weights_a:
        np.testing.assert_array_equal(weights_b[name], weights_a[name])
    logits_a, tokens_a = _capture(baseline.root)
    logits_b, tokens_b = _capture(skipped.root)
    np.testing.assert_array_equal(logits_b, logits_a)
    assert tokens_b == tokens_a


def test_real_source_merge_cli_failure_does_not_publish_an_artifact(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """The CLI's failure status must also mean no partially requested AEG exists."""
    from click.testing import CliRunner

    from aether.cli import cli

    output = tmp_path / "cli-output.aeg"
    result = CliRunner().invoke(
        cli,
        ["--diagnostic", "--cache-dir", str(tmp_path / "cli-cache"), "merge",
         str(tiny_local_safetensors_model), str(tmp_path / "missing-source"),
         "--method", "linear", "--output", str(output)],
        env={"AETHER_HF_OFFLINE": "1"},
    )
    assert result.exit_code == 1
    assert "Optimization failed" in result.output
    assert "model_merging" in result.output
    assert "Every requested merge source must load" in result.output
    assert not output.exists()
