"""The compile -> discover -> run contract, by the original model ID.

A compiled artifact must be findable by the same reference it was compiled from.
That invariant was broken in a way no existing test could catch: artifact naming
lived in two modules with two different conventions, and the only agreement between
them was accidental.

    compiler._stage4_package   ->  models/microsoft--Phi-3.5-mini-instruct.aeg
    runtime._resolve_aeg_path  ->  models/microsoft_Phi-3.5-mini-instruct

Two independent divergences — the separator and the suffix — so *no* Hugging Face
ID containing ``/`` could ever resolve after compiling. Local directory paths took
a different branch and worked, which is exactly why the suite stayed green.

These tests pin the contract itself rather than either side's spelling, so the two
cannot drift apart again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether.utils.file_io import (
    AEG_SUFFIX,
    aeg_artifact_name,
    aeg_cache_candidates,
    aeg_cache_path,
)

#: Reference shapes real Hugging Face IDs take: an org prefix, dots in version
#: numbers, single and double dashes, mixed case, and a bare name with no org.
MODEL_IDS = [
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen3-0.6B",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "SummerSigh/GPTNeo350M-Instruct-SFT",
    "deepseek-ai/DeepSeek-V3",
    "tiiuae/falcon-7b",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "gpt2",
    "org/name--with--doubles",
]


def _fake_artifact(root: Path, name: str, *, model_id: str = "m") -> Path:
    """Create a directory that looks like an AEG package to the resolver."""
    package = root / "models" / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "manifest.json").write_text(
        json.dumps({"model_id": model_id}), encoding="utf-8"
    )
    return package


def _runtime(cache: Path):
    from aether.runtime.config import RuntimeConfig
    from aether.runtime.runtime import Runtime

    return Runtime(RuntimeConfig(model_cache_dir=str(cache), hf_offline=True))


# ── The naming contract itself ──────────────────────────────────────────────


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_artifact_name_is_a_safe_single_path_component(model_id: str) -> None:
    name = aeg_artifact_name(model_id)
    assert name.endswith(AEG_SUFFIX)
    assert "/" not in name and "\\" not in name, "must be one path component"
    assert not name.startswith("."), "a leading dot would hide the directory"
    assert Path(name).name == name


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_naming_is_deterministic(model_id: str) -> None:
    assert aeg_artifact_name(model_id) == aeg_artifact_name(model_id)


def test_distinct_model_ids_get_distinct_artifact_names() -> None:
    """A collision would make one model silently serve another's weights."""
    names = [aeg_artifact_name(model_id) for model_id in MODEL_IDS]
    assert len(set(names)) == len(names)


def test_naming_preserves_the_characters_that_make_an_id_recognizable() -> None:
    """Dots and single dashes are ordinary in repo names and are left alone."""
    assert aeg_artifact_name("microsoft/Phi-3.5-mini-instruct") == (
        "microsoft--Phi-3.5-mini-instruct.aeg"
    )
    assert aeg_artifact_name("Qwen/Qwen3-0.6B") == "Qwen--Qwen3-0.6B.aeg"


def test_naming_handles_degenerate_input() -> None:
    assert aeg_artifact_name("") == f"unnamed_model{AEG_SUFFIX}"
    assert aeg_artifact_name("   ") == f"unnamed_model{AEG_SUFFIX}"
    assert not aeg_artifact_name(".hidden").startswith(".")
    # Colons and spaces are unsafe on Windows paths.
    assert ":" not in aeg_artifact_name("a:b/c")
    assert " " not in aeg_artifact_name("a b/c")


def test_the_writer_and_the_reader_agree(tmp_path: Path) -> None:
    """The single assertion the original bug would have failed.

    ``aeg_cache_path`` is what the compiler writes to and the first candidate the
    runtime resolves. If those two ever diverge again, this fails.
    """
    for model_id in MODEL_IDS:
        written = aeg_cache_path(model_id, tmp_path)
        candidates = aeg_cache_candidates(model_id, tmp_path)
        assert candidates[0] == written, model_id


# ── Discovery by the original model ID ──────────────────────────────────────


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_a_compiled_artifact_is_discoverable_by_its_model_id(
    tmp_path: Path, model_id: str
) -> None:
    """Write where the compiler writes; resolve by the ID the user typed."""
    expected = aeg_cache_path(model_id, tmp_path)
    _fake_artifact(tmp_path, expected.name, model_id=model_id)

    runtime = _runtime(tmp_path)
    resolved = runtime._resolve_aeg_path(model_id)
    assert resolved is not None, f"{model_id} was written but is not discoverable"
    assert Path(resolved) == expected


def test_an_unknown_model_still_resolves_to_nothing(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert runtime._resolve_aeg_path("nobody/never-compiled") is None


def test_a_directory_without_a_manifest_is_not_an_artifact(tmp_path: Path) -> None:
    """A half-written package must not resolve.

    The previous resolver accepted any directory that existed, so a partial or
    unrelated directory resolved here and failed later with an error about
    execution rather than about the artifact.
    """
    (tmp_path / "models" / aeg_artifact_name("org/model")).mkdir(parents=True)
    runtime = _runtime(tmp_path)
    assert runtime._resolve_aeg_path("org/model") is None


def test_an_explicit_path_still_resolves(tmp_path: Path) -> None:
    """Passing a directory directly must keep working, including a relative one."""
    package = _fake_artifact(tmp_path, "explicit.aeg")
    runtime = _runtime(tmp_path)
    assert Path(runtime._resolve_aeg_path(str(package))) == package.resolve()


@pytest.mark.parametrize(
    "legacy_name",
    [
        "microsoft_Phi-3.5-mini-instruct",          # the old reader convention
        "microsoft_Phi-3.5-mini-instruct.aeg",
        "microsoft--Phi-3.5-mini-instruct",         # the old writer convention
    ],
)
def test_artifacts_from_earlier_versions_remain_discoverable(
    tmp_path: Path, legacy_name: str
) -> None:
    """Upgrading Aether must not orphan a package already on disk."""
    _fake_artifact(tmp_path, legacy_name)
    runtime = _runtime(tmp_path)
    resolved = runtime._resolve_aeg_path("microsoft/Phi-3.5-mini-instruct")
    assert resolved is not None, f"{legacy_name} became undiscoverable"
    assert Path(resolved).name == legacy_name


def test_a_canonical_artifact_wins_over_a_legacy_one(tmp_path: Path) -> None:
    """A freshly compiled package must take precedence over a stale layout."""
    _fake_artifact(tmp_path, "microsoft_Phi-3.5-mini-instruct")
    canonical = _fake_artifact(
        tmp_path, aeg_artifact_name("microsoft/Phi-3.5-mini-instruct")
    )
    runtime = _runtime(tmp_path)
    assert Path(runtime._resolve_aeg_path("microsoft/Phi-3.5-mini-instruct")) == canonical


# ── Every command resolves through the same path ────────────────────────────


def test_list_output_is_accepted_back_as_input(tmp_path: Path) -> None:
    """``aether list`` must not print a name the other commands reject.

    ``--`` stands in for ``/`` and a repo name may itself contain ``--``, so
    reversing the mangling is ambiguous. Instead every name ``list`` emits must be
    resolvable, which is the property a user actually needs.
    """
    for model_id in MODEL_IDS:
        _fake_artifact(tmp_path, aeg_cache_path(model_id, tmp_path).name)

    runtime = _runtime(tmp_path)
    listed = runtime.list()
    assert len(listed) == len(MODEL_IDS)
    for name in listed:
        assert runtime._resolve_aeg_path(name) is not None, name


def test_info_and_graph_share_the_resolver(tmp_path: Path) -> None:
    """A single resolver means no command can disagree about what exists.

    ``info`` reads the manifest and ``graph`` loads the IR, but both must first
    find the artifact through the same function, or one can succeed while another
    reports the model missing.
    """
    import inspect

    from aether.runtime.runtime import Runtime

    assert "_resolve_aeg_path" in inspect.getsource(Runtime.info)

    from aether import cli

    # Click wraps the function in a Command; the body is on ``.callback``.
    assert "_resolve_aeg_path" in inspect.getsource(cli.graph.callback)


def test_resolution_survives_a_fresh_runtime(tmp_path: Path) -> None:
    """Discovery must be a property of the filesystem, not of process state."""
    model_id = "microsoft/Phi-3.5-mini-instruct"
    expected = _fake_artifact(tmp_path, aeg_cache_path(model_id, tmp_path).name)

    first = _runtime(tmp_path)
    assert Path(first._resolve_aeg_path(model_id)) == expected
    del first

    second = _runtime(tmp_path)
    assert Path(second._resolve_aeg_path(model_id)) == expected


def test_repeated_resolution_is_idempotent(tmp_path: Path) -> None:
    model_id = "Qwen/Qwen3-0.6B"
    _fake_artifact(tmp_path, aeg_cache_path(model_id, tmp_path).name)
    runtime = _runtime(tmp_path)
    answers = {runtime._resolve_aeg_path(model_id) for _ in range(5)}
    assert len(answers) == 1


def test_the_compiler_derives_its_output_path_from_the_shared_contract() -> None:
    """Pin the writer side by source, since compiling a real model needs weights."""
    import inspect

    from aether.compiler.compiler import Compiler

    source = inspect.getsource(Compiler._stage4_package)
    assert "aeg_cache_path" in source, "the compiler no longer uses the shared contract"
    # And it must derive the name from the caller's reference, not a mangled one.
    assert "source_model_id" in source


def test_pull_does_not_reintroduce_its_own_path(tmp_path: Path) -> None:
    """``Runtime.pull`` previously computed a cache path of its own."""
    import inspect

    from aether.runtime.runtime import Runtime

    source = inspect.getsource(Runtime.pull)
    assert "safe_model_id_path" not in source
