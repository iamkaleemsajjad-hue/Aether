"""Compare real packaged tokenization with the pinned original HF loader."""

from __future__ import annotations

import builtins
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mamba1_fidelity_fixtures import checkpoint_data, matrix_logits, write_checkpoint

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.native_cpu_backend import PackagedTokenizer
from aether.core.exceptions import BackendError

if TYPE_CHECKING:
    from pathlib import Path


def _source(path: Path, config: dict[str, Any], special: dict[str, Any]) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    path.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel({"<unk>": 0, "<marker>": 1, "hello": 2, "world": 3}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", **config}),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")


def _framework_free(path: Path, monkeypatch: pytest.MonkeyPatch) -> PackagedTokenizer:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] in {"torch", "transformers"}:
            raise AssertionError("packaged tokenization imported an optional model framework")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", guarded)
        return PackagedTokenizer(path / "tokenizer.json")


@pytest.mark.parametrize("location", ["config", "special_map"])
@pytest.mark.parametrize(
    "role",
    [
        "unk_token",
        "bos_token",
        "eos_token",
        "pad_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ],
)
def test_declared_special_roles_match_original_hf_without_framework_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    role: str,
) -> None:
    """A declaration outside tokenizer.json still affects encoding and special-token removal."""
    transformers = pytest.importorskip("transformers")
    marker = "<unk>" if role == "unk_token" else "<marker>"
    declaration = {role: [marker] if role == "additional_special_tokens" else marker}
    source = tmp_path / "source"
    _source(
        source,
        declaration if location == "config" else {},
        declaration if location == "special_map" else {},
    )
    reference = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
    packaged = _framework_free(source, monkeypatch)
    marker_id = reference.convert_tokens_to_ids(marker)
    assert packaged.decode([marker_id], skip_special_tokens=True) == reference.decode(
        [marker_id], skip_special_tokens=True
    )
    assert packaged.decode([marker_id], skip_special_tokens=False) == marker
    text = f"hello{marker}world"
    assert packaged(text, add_special_tokens=False)["input_ids"][0].tolist() == reference.encode(
        text, add_special_tokens=False
    )
    assert packaged._tokenizer.get_vocab() == reference.get_vocab()


@pytest.mark.parametrize("special_flag", [False, True])
def test_added_token_decoder_preserves_declared_flags_and_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    special_flag: bool,
) -> None:
    """An existing vocabulary entry can have matching/whitespace flags in HF metadata."""
    transformers = pytest.importorskip("transformers")
    descriptor = {
        "content": "<marker>",
        "special": special_flag,
        "normalized": False,
        "single_word": True,
        "lstrip": True,
        "rstrip": True,
    }
    source = tmp_path / "source"
    _source(source, {"added_tokens_decoder": {"1": descriptor}}, {})
    reference = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
    packaged = _framework_free(source, monkeypatch)
    assert packaged._tokenizer.get_vocab() == reference.get_vocab()
    assert packaged.decode([1], skip_special_tokens=True) == reference.decode(
        [1], skip_special_tokens=True
    )
    for text in ("hello <marker> world", "hello<marker>world", "<marker> world"):
        np.testing.assert_array_equal(
            packaged(text, add_special_tokens=False)["input_ids"][0],
            reference.encode(text, add_special_tokens=False),
        )


@pytest.mark.parametrize("backend", ["aether_cpu", "pytorch"])
def test_public_special_token_output_matches_original_hf_tokenizer(
    tmp_path: Path, backend: str
) -> None:
    """The actual finite zero-gate checkpoint emits UNK, which HF omits from text."""
    transformers = pytest.importorskip("transformers")
    if backend == "pytorch":
        pytest.importorskip("torch")
    source = tmp_path / "source"
    config = write_checkpoint(source, with_bias=True, large_silu="gate")
    _, tensors = checkpoint_data(with_bias=True, large_silu="gate")
    prefix, expected_ids = [1], []
    for _ in range(8):
        token = int(np.argmax(matrix_logits(config, tensors, np.asarray(prefix, np.int64))[-1]))
        expected_ids.append(token)
        prefix.append(token)
        if token == config["eos_token_id"]:
            break
    reference = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
    artifact = tmp_path / "model.aeg"
    Compiler(
        CompilerConfig(
            quality_mode="diagnostic",
            targets=["cpu_generic"],
            skip_download=True,
            precision_assignment_mode="uniform",
            precision="FP32",
            calibration_tokens=4,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    ).compile(str(source), output_path=artifact)
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
    assert response.usage["completion_tokens"] == len(expected_ids)
    assert response.text == streamed == reference.decode(expected_ids, skip_special_tokens=True)


def test_serialized_token_flags_are_stable_across_repeated_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical string declarations must preserve flags already in tokenizer.json."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import AddedToken, Tokenizer

    source = tmp_path / "source"
    _source(source, {"eos_token": "<marker>"}, {})
    serialized = Tokenizer.from_file(str(source / "tokenizer.json"))
    serialized.add_special_tokens(
        [
            AddedToken(
                "<marker>",
                lstrip=True,
                rstrip=True,
                single_word=True,
                normalized=False,
                special=True,
            )
        ]
    )
    serialized.save(str(source / "tokenizer.json"))
    saved = (source / "tokenizer.json").read_bytes()
    reference = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
    for _ in range(3):
        packaged = _framework_free(source, monkeypatch)
        assert packaged._tokenizer.get_vocab() == reference.get_vocab()
        for text in ("hello <marker> world", "hello<marker>world"):
            assert packaged(text)["input_ids"][0].tolist() == reference.encode(text)
    assert (source / "tokenizer.json").read_bytes() == saved


@pytest.mark.parametrize(
    "config",
    [
        {"added_tokens_decoder": {"3": {"content": "<marker>", "special": True}}},
        {"additional_special_tokens": "<marker>"},
    ],
)
def test_malformed_token_declarations_refuse_instead_of_renumbering_or_splitting(
    tmp_path: Path,
    config: dict[str, Any],
) -> None:
    """Unusable declared IDs and malformed marker lists cannot authorize a tokenizer."""
    source = tmp_path / "source"
    _source(source, config, {})
    with pytest.raises(BackendError, match="AddedToken"):
        PackagedTokenizer(source / "tokenizer.json")
