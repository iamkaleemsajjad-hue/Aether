"""Real compiler failures must preserve the published artifact and its execution."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aether import Compiler, CompilerConfig
from aether.core.aeg_format import AEGPackage
from aether.core.exceptions import CompilationError
from aether.runtime.aeg_loader import load_engine_from_path


def _config(root: Path) -> CompilerConfig:
    return CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        skip_download=True,
        overwrite=True,
        cache_dir=str(root / "cache"),
        precision="FP32",
    )


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _capture(root: Path) -> tuple[Any, list[int]]:
    engine = load_engine_from_path(root, prepare_native_kernels=False, allow_uncertified=True)
    ids = np.asarray([1, 2], dtype=np.int64)
    logits, _ = engine.forward(ids)
    tokens = list(engine.generate_iter(ids, max_tokens=3, temperature=0))
    return logits, tokens


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("reason", ["rejection", "exception", "invalid_callback"])
def test_failed_evaluation_preserves_public_destination(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    existing: bool,
    reason: str,
) -> None:
    """A rejected/incomplete gate cannot create output or displace valid bytes."""
    output = tmp_path / "model.aeg"
    before, execution = {}, None
    if existing:
        Compiler(_config(tmp_path)).compile(str(tiny_local_safetensors_model), output_path=output)
        before, execution = _inventory(output), _capture(output)
    observed: list[dict[str, Any]] = []

    def evaluator(benchmark: str, spec: Any) -> dict[str, Any]:
        observed.append({"published": _inventory(output), "spec": dict(spec)})
        if reason == "exception":
            raise RuntimeError("controlled evaluator outage")
        return {"score": 0, "num_correct": 0, "num_total": 1, "latency_ms": 0}

    callback = object() if reason == "invalid_callback" else evaluator
    config = _config(tmp_path)
    config.precision = "Q8_0"  # A distinct candidate, never the prior bytes.
    with pytest.raises(CompilationError) as caught:
        Compiler(config).compile(
            str(tiny_local_safetensors_model),
            output_path=output,
            evaluation_evaluator=callback,
            eval_benchmarks=["mmlu"],
            eval_baselines={"mmlu": 1.0},
        )
    assert caught.value.stage == "evaluation"
    assert _inventory(output) == before
    assert output.exists() is existing
    for observation in observed:
        assert observation["published"] == before
        candidate = Path(observation["spec"]["aeg_path"])
        assert candidate != output and candidate.parent.parent == output.parent
        assert not candidate.exists()
    if existing:
        assert execution is not None
        logits, tokens = _capture(output)
        np.testing.assert_array_equal(logits, execution[0])
        assert tokens == execution[1]
        AEGPackage(output).load().verify_integrity()
    assert not list(tmp_path.glob(".aether-compile-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_real_missing_projection_packaging_failure_preserves_destination(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    existing: bool,
) -> None:
    """A real missing source tensor must fail before any public output changes."""
    st = pytest.importorskip("safetensors.numpy")
    output = tmp_path / "model.aeg"
    before = {}
    if existing:
        Compiler(_config(tmp_path)).compile(str(tiny_local_safetensors_model), output_path=output)
        before = _inventory(output)
    broken = tmp_path / "broken-source"
    shutil.copytree(tiny_local_safetensors_model, broken)
    tensors = st.load_file(str(broken / "model.safetensors"))
    del tensors["model.layers.0.mlp.down_proj.weight"]
    st.save_file(tensors, str(broken / "model.safetensors"))

    with pytest.raises(CompilationError, match="required logical tensors"):
        Compiler(_config(tmp_path)).compile(str(broken), output_path=output)
    assert _inventory(output) == before
    assert output.exists() is existing
    assert not list(tmp_path.glob(".aether-compile-*"))


@pytest.mark.parametrize("existing", [False, True])
def test_accepted_evaluator_executes_private_candidate_then_returns_public_package(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    existing: bool,
) -> None:
    """The callback runs the candidate while readers still see the old artifact.

    The score is a controlled callback-contract value, not measured MMLU quality.
    Real CPU logits and tokens separately prove which artifact was executed.
    """
    output = tmp_path / "model.aeg"
    before = {}
    if existing:
        Compiler(_config(tmp_path)).compile(str(tiny_local_safetensors_model), output_path=output)
        before = _inventory(output)
    seen: list[tuple[Path, Any, list[int]]] = []

    def evaluator(benchmark: str, spec: Any) -> dict[str, Any]:
        assert benchmark == "mmlu"
        assert _inventory(output) == before
        candidate = Path(spec["aeg_path"])
        assert candidate != output and candidate.parent.parent == output.parent
        assert Path(spec["publication_path"]) == output
        AEGPackage(candidate).load().verify_integrity()
        logits, tokens = _capture(candidate)
        seen.append((candidate, logits, tokens))
        return {"score": 1, "num_correct": 1, "num_total": 1, "latency_ms": 0}

    config = _config(tmp_path)
    config.precision = "Q8_0"
    package = Compiler(config).compile(
        str(tiny_local_safetensors_model),
        output_path=output,
        evaluation_evaluator=evaluator,
        eval_benchmarks=["mmlu"],
        eval_baselines={"mmlu": 1.0},
    )
    assert len(seen) == 1
    assert not seen[0][0].exists()
    assert package.root == output
    assert package.weights
    store = package.weight_store()
    assert store.directory == output / "weights" / "quantized"
    assert store.entries
    for name, tensor in package.weights.items():
        np.testing.assert_array_equal(store.load_tensor(name).data, tensor.data)
    package.verify_integrity()
    reloaded = AEGPackage(output).load()
    reloaded.verify_integrity()
    assert reloaded.metadata["eval_report"]["aeg_path"] == str(output)
    assert reloaded.metadata["evaluation_status"] == "certified"
    logits, tokens = _capture(output)
    np.testing.assert_array_equal(logits, seen[0][1])
    assert tokens == seen[0][2]
    assert not list(tmp_path.glob(".aether-compile-*"))


def test_default_cache_publication_remains_discoverable_by_original_reference(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
) -> None:
    """A private staging name cannot change the shared compiler/runtime cache key."""
    from aether.runtime.config import RuntimeConfig
    from aether.runtime.runtime import Runtime
    from aether.utils.file_io import aeg_cache_path

    config = _config(tmp_path)
    model = str(tiny_local_safetensors_model)
    package = Compiler(config).compile(model)
    expected = aeg_cache_path(model, config.cache_dir)
    assert package.root == expected
    runtime = Runtime(RuntimeConfig(model_cache_dir=config.cache_dir, hf_offline=True))
    assert runtime.list() == [expected.name]
    resolved = runtime._resolve_aeg_path(model)
    assert resolved is not None
    assert Path(resolved) == expected
    package.verify_integrity()
    logits, tokens = _capture(expected)
    assert np.isfinite(logits).all() and len(tokens) == 3
    assert not list(expected.parent.glob(".aether-compile-*"))
