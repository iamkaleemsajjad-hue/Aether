"""A real compiled artifact must retain its source's generation stop settings."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mamba1_fidelity_fixtures import checkpoint_data, matrix_logits, write_checkpoint

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.core.aeg_format import load_aeg_package
from aether.core.exceptions import CompilationError

if TYPE_CHECKING:
    from pathlib import Path


def _compile(source: Path, artifact: Path, *, overwrite: bool = False) -> None:
    Compiler(
        CompilerConfig(
            quality_mode="diagnostic",
            targets=["cpu_generic"],
            skip_download=True,
            precision_assignment_mode="uniform",
            precision="FP32",
            calibration_tokens=4,
            cache_dir=str(artifact.parent / "compiler-cache"),
            overwrite=overwrite,
        )
    ).compile(str(source), output_path=artifact)


@pytest.mark.parametrize("backend", ["aether_cpu", "pytorch"])
@pytest.mark.parametrize(
    "generation",
    [None, {"eos_token_id": 12}, {"eos_token_id": [16, 12]}, {"eos_token_id": None}, {}],
    ids=["model-fallback", "scalar-override", "list-override", "disabled", "empty-override"],
)
def test_public_generation_and_stream_retain_source_stop_settings(
    tmp_path: Path,
    backend: str,
    generation: dict[str, Any] | None,
) -> None:
    """Match the source's stop boundary using independently computed greedy logits."""
    if backend == "pytorch":
        pytest.importorskip("torch")
    source = tmp_path / "source"
    config = write_checkpoint(source)
    _, tensors = checkpoint_data()
    declared = config if generation is None else generation
    stop_value = declared.get("eos_token_id")
    stops = set(stop_value if isinstance(stop_value, list) else [stop_value])
    prefix, expected = [1], []
    for _ in range(8):
        token = int(np.argmax(matrix_logits(config, tensors, np.asarray(prefix, np.int64))[-1]))
        expected.append(token)
        prefix.append(token)
        if token in stops:
            break
    if generation is not None:
        original_bytes = (json.dumps(generation, indent=3) + "\n").encode("utf-8")
        (source / "generation_config.json").write_bytes(original_bytes)
    artifact = tmp_path / "model.aeg"
    _compile(source, artifact)
    with Runtime(
        RuntimeConfig(
            backend_name=backend,
            execution_devices=["cpu"],
            hf_offline=True,
            allow_uncertified=True,
            speculative_decoding=False,
            enable_semantic_cache=False,
            enable_prefix_caching=False,
            model_cache_dir=str(tmp_path / "runtime-cache"),
        )
    ) as runtime:
        response = runtime.generate(str(artifact), "hello", max_tokens=8, temperature=0)
        streamed = "".join(
            runtime.generate_stream(str(artifact), "hello", max_tokens=8, temperature=0)
        )
    expected_text = " ".join(f"tok{token}" for token in expected)
    assert response.text == streamed == expected_text
    assert response.usage["completion_tokens"] == len(expected)
    packaged = artifact / "tokenizer/generation_config.json"
    if generation is not None:
        assert packaged.read_bytes() == original_bytes
    else:
        assert json.loads(packaged.read_text()) == {"eos_token_id": 16}
    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert package.manifest is not None
    assert "tokenizer/generation_config.json" in package.manifest.artifacts


@pytest.mark.parametrize("backend", ["aether_cpu", "pytorch"])
def test_slow_tokenizer_materialization_retains_model_stop_id(tmp_path: Path, backend: str) -> None:
    """The actual AutoTokenizer fallback needs the same generation settings copy."""
    transformers = pytest.importorskip("transformers")
    if backend == "pytorch":
        pytest.importorskip("torch")
    source = tmp_path / "source"
    config = write_checkpoint(source)
    _, tensors = checkpoint_data()
    (source / "tokenizer.json").unlink()
    vocabulary = ["[UNK]", "hello", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocabulary.extend(f"tok{index}" for index in range(6, config["vocab_size"]))
    (source / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab_file=str(source / "vocab.txt"))
    tokenizer.save_pretrained(source)
    assert not (source / "tokenizer.json").exists()
    ids = np.asarray(tokenizer.encode("hello"), np.int64)
    first = int(np.argmax(matrix_logits(config, tensors, ids)[-1]))
    config["eos_token_id"] = first
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    artifact = tmp_path / "model.aeg"
    _compile(source, artifact)
    with Runtime(
        RuntimeConfig(
            backend_name=backend,
            execution_devices=["cpu"],
            allow_uncertified=True,
            hf_offline=True,
            speculative_decoding=False,
            enable_semantic_cache=False,
            enable_prefix_caching=False,
            model_cache_dir=str(tmp_path / "runtime-cache"),
        )
    ) as runtime:
        response = runtime.generate(str(artifact), "hello", max_tokens=8, temperature=0)
        streamed = "".join(
            runtime.generate_stream(str(artifact), "hello", max_tokens=8, temperature=0)
        )
    assert response.usage["completion_tokens"] == 1
    assert response.text == streamed == tokenizer.decode([first], skip_special_tokens=True)


@pytest.mark.parametrize("malformed", ["{", "[]"])
def test_invalid_generation_config_refuses_before_replacing_destination(
    tmp_path: Path,
    malformed: str,
) -> None:
    """Diagnostics cannot publish a file the runtime would ignore as malformed."""
    source = tmp_path / "source"
    write_checkpoint(source)
    (source / "generation_config.json").write_text(malformed, encoding="utf-8")
    artifact = tmp_path / "model.aeg"
    artifact.mkdir()
    (artifact / "prior.txt").write_bytes(b"previous artifact")
    with pytest.raises(CompilationError, match="tokenizer"):
        _compile(source, artifact, overwrite=True)
    assert [path.name for path in artifact.iterdir()] == ["prior.txt"]
    assert (artifact / "prior.txt").read_bytes() == b"previous artifact"
