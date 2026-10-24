"""Real target cache persistence rebuilds only speculative acceleration state."""

import json
from dataclasses import fields

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.runtime.execution_features import model_fingerprint
from aether.runtime.spec_generation import SpeculativeSessionCache


@pytest.fixture
def speculative_artifact(tmp_path, request):
    """Compile complete dense or mixed selective-scan/attention CPU graphs."""
    from safetensors.numpy import load_file, save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    source = tmp_path / "checkpoint"
    _write_probe_checkpoint(source)
    # Plain word tokens allow an exact text -> token-prefix roundtrip, including
    # the unknown token, independently of the model's chosen output IDs.
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={"unknown": 0, "hello": 1, "world": 2, **{f"token{i}": i + 3 for i in range(29)}},
            unk_token="unknown",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "special_tokens_map.json").write_text('{"unk_token":"unknown"}', encoding="utf-8")
    token_config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    token_config["unk_token"] = "unknown"
    (source / "tokenizer_config.json").write_text(json.dumps(token_config), encoding="utf-8")
    if request.param == "hybrid":
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config.update(
            architectures=["JambaForCausalLM"],
            model_type="jamba",
            num_hidden_layers=2,
            d_model=16,
            d_inner=16,
            d_state=2,
            dt_rank=1,
            d_conv=3,
            layers_block_type=["mamba", "attention"],
        )
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        tensors = {
            name.replace("model.layers.0.", "model.layers.1."): value
            for name, value in load_file(str(source / "model.safetensors")).items()
        }
        rng = np.random.default_rng(923)
        prefix = "model.layers.0.mamba."
        for name, shape in {
            "in_proj.weight": (32, 16),
            "conv1d.weight": (16, 3),
            "x_proj.weight": (5, 16),
            "dt_proj.weight": (16, 1),
            "A_log": (16, 2),
            "out_proj.weight": (16, 16),
        }.items():
            tensors[prefix + name] = rng.normal(0, 0.05, shape).astype(np.float32)
        for name in ("norm.weight", "D"):
            tensors[prefix + name] = np.ones(16, dtype=np.float32)
        for name in ("conv1d.bias", "dt_proj.bias"):
            tensors[prefix + name] = np.zeros(16, dtype=np.float32)
        save_file(tensors, str(source / "model.safetensors"))
    artifact = tmp_path / "model.aeg"
    _compile(source, artifact, tmp_path / "compiler-cache")
    return str(artifact)


def _compile(source, artifact, cache_dir):
    config = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"], precision="FP32", optimization_level=0, cache_dir=str(cache_dir)
    )
    for field in fields(config):
        if field.name.startswith("enable_"):
            setattr(config, field.name, field.name == "enable_precision_assignment")
    Compiler(config).compile(str(source), output_path=artifact)


def _config(tmp_path, artifact, backend):
    return RuntimeConfig(
        allow_uncertified=True,
        backend_name=backend,
        execution_devices=["cpu"],
        draft_model_id=artifact,
        draft_backend_name=backend,
        draft_execution_devices=["cpu"],
        enable_semantic_cache=False,
        speculative_decoding=False,
        dynamic_precision=False,
        model_cache_dir=str(tmp_path / "runtime-cache"),
        kv_cache_path=str(tmp_path / "stored-sessions"),
    )


def _snapshot(cache, handle):
    if hasattr(handle.engine, "torch"):
        from aether.runtime.torch_cache_handoff import export_torch_cache

        return export_torch_cache(
            cache, engine=handle.engine, model_fingerprint=model_fingerprint(handle)
        )
    from aether.runtime.cache_handoff import export_cpu_cache

    return export_cpu_cache(cache, model_fingerprint=model_fingerprint(handle))


@pytest.mark.parametrize(
    "backend,speculative_artifact",
    [
        ("aether_cpu", "dense"),
        ("aether_cpu", "hybrid"),
        ("pytorch", "dense"),
        ("pytorch", "hybrid"),
    ],
    indirect=["speculative_artifact"],
    ids=["cpu-dense", "cpu-hybrid", "torch-dense", "torch-hybrid"],
)
def test_speculative_target_state_persists_and_resumes_after_restart(
    speculative_artifact, tmp_path, backend, monkeypatch
):
    """Restart preserves target bytes and matches uninterrupted speculative decoding."""
    if backend == "pytorch":
        pytest.importorskip("torch")
        monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
        monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    config = _config(tmp_path, speculative_artifact, backend)
    with Runtime(config) as runtime:
        runtime.generate(
            speculative_artifact,
            "hello world",
            max_tokens=3,
            temperature=0,
            aether_kv_session_id="spec",
        )
        handle = runtime._loaded_models[speculative_artifact]
        prefix, composite = handle.session_caches["spec"]
        assert isinstance(composite, SpeculativeSessionCache)
        before = _snapshot(composite.target_cache, handle)
        continued_prompt = (
            handle.tokenizer.decode(prefix.tolist(), skip_special_tokens=False) + " hello"
        )
        encoded = handle.tokenizer(continued_prompt, return_tensors="np")["input_ids"][0]
        np.testing.assert_array_equal(encoded[: len(prefix)], prefix)
        handle.save_session_cache("uninterrupted", prefix.copy(), composite.clone())
        expected = runtime.generate(
            speculative_artifact,
            continued_prompt,
            max_tokens=4,
            temperature=0,
            aether_kv_session_id="uninterrupted",
        )
        expected_prefix, expected_state = handle.session_caches["uninterrupted"]
        expected_logits = _snapshot(expected_state.target_cache, handle).tensors
        saved = runtime.persist_session(speculative_artifact, "spec")
        assert saved["logical_length"] == len(prefix)
        block = runtime.kv_cache.block_for_payload_key(saved["key"])
        stored = runtime.kv_cache.export_payload(block.block_id)
        assert stored.tensors == before.tensors
        assert stored.metadata["speculative_resume"] == {
            "target_state": "lossless",
            "draft_state": "rebuild_from_token_prefix",
        }
        assert "spec" not in handle.session_caches
        # Automatic local restoration exercises the same raw-target bootstrap
        # path that a fresh Runtime will use after explicit restoration.
        automatic = runtime.generate(
            speculative_artifact,
            continued_prompt,
            max_tokens=4,
            temperature=0,
            aether_kv_session_id="spec",
        )
        assert automatic.text == expected.text
        assert automatic.metrics.extra["kv_reused_tokens"] == len(prefix)
        assert (
            _snapshot(handle.session_caches["spec"][1].target_cache, handle).tensors
            == expected_logits
        )
    with Runtime(config) as restarted:
        restarted.restore_session(speculative_artifact, "spec", saved["key"])
        handle = restarted._loaded_models[speculative_artifact]
        restored_prefix, restored = handle.session_caches["spec"]
        assert not isinstance(restored, SpeculativeSessionCache)
        np.testing.assert_array_equal(restored_prefix, prefix)
        assert _snapshot(restored, handle).serialize() == before.serialize()
        resumed = restarted.generate(
            speculative_artifact,
            continued_prompt,
            max_tokens=4,
            temperature=0,
            aether_kv_session_id="spec",
        )
        actual_prefix, actual_state = handle.session_caches["spec"]
        assert isinstance(actual_state, SpeculativeSessionCache)
        assert actual_state.source_identity == (
            id(handle.engine),
            id(restarted._draft_runtime._loaded_models[speculative_artifact].engine),
        )
        assert resumed.text == expected.text
        assert resumed.metrics.extra["kv_reused_tokens"] == len(prefix)
        np.testing.assert_array_equal(actual_prefix, expected_prefix)
        assert _snapshot(actual_state.target_cache, handle).tensors == expected_logits


@pytest.mark.parametrize("speculative_artifact", ["dense"], indirect=True)
@pytest.mark.parametrize("mismatch", ["target", "prefix"])
def test_speculative_persistence_rejects_mismatched_target_or_prefix(
    speculative_artifact, tmp_path, mismatch
):
    """Reject incoherent composite state without clearing its live session."""
    with Runtime(_config(tmp_path, speculative_artifact, "aether_cpu")) as runtime:
        runtime.generate(
            speculative_artifact,
            "hello world",
            max_tokens=2,
            temperature=0,
            aether_kv_session_id="spec",
        )
        handle = runtime._loaded_models[speculative_artifact]
        original = handle.session_caches["spec"]
        composite = original[1]
        if mismatch == "target":
            composite.source_identity = (id(handle.engine) + 1, composite.source_identity[1])
        else:
            composite.token_ids = (*composite.token_ids[:-1], (composite.token_ids[-1] + 1) % 32)
        with pytest.raises(ValueError, match=r"different execution engine|recorded token prefix"):
            runtime.persist_session(speculative_artifact, "spec")
        assert handle.session_caches["spec"] is original
        assert not runtime.kv_cache._payloads


@pytest.mark.parametrize("speculative_artifact", ["dense"], indirect=True)
def test_persisted_target_resumes_with_a_different_compatible_draft(speculative_artifact, tmp_path):
    """Changing the acceleration draft retains exact authoritative target results."""
    from safetensors.numpy import load_file, save_file

    from aether.runtime.spec_decode import numpy_logits

    config = _config(tmp_path, speculative_artifact, "aether_cpu")
    with Runtime(config) as runtime:
        runtime.generate(
            speculative_artifact,
            "hello world",
            max_tokens=3,
            temperature=0,
            aether_kv_session_id="spec",
        )
        handle = runtime._loaded_models[speculative_artifact]
        prefix, composite = handle.session_caches["spec"]
        target_snapshot = _snapshot(composite.target_cache, handle)
        continued_prompt = (
            handle.tokenizer.decode(prefix.tolist(), skip_special_tokens=False) + " hello"
        )
        handle.save_session_cache("uninterrupted", prefix.copy(), composite.clone())
        expected = runtime.generate(
            speculative_artifact,
            continued_prompt,
            max_tokens=4,
            temperature=0,
            aether_kv_session_id="uninterrupted",
        )
        expected_prefix, expected_cache = handle.session_caches["uninterrupted"]
        expected_logits = numpy_logits(expected_cache.last_logits).copy()
        saved = runtime.persist_session(speculative_artifact, "spec")
    source = tmp_path / "checkpoint"
    tensors = load_file(str(source / "model.safetensors"))
    tensors["lm_head.weight"] *= -1  # Actual changed draft predictions, same tokenizer contract.
    save_file(tensors, str(source / "model.safetensors"))
    draft = tmp_path / "different-draft.aeg"
    _compile(source, draft, tmp_path / "draft-compiler-cache")
    config.draft_model_id = str(draft)
    with Runtime(config) as restarted:
        restarted.restore_session(speculative_artifact, "spec", saved["key"])
        handle = restarted._loaded_models[speculative_artifact]
        assert (
            _snapshot(handle.session_caches["spec"][1], handle).serialize()
            == target_snapshot.serialize()
        )
        actual = restarted.generate(
            speculative_artifact,
            continued_prompt,
            max_tokens=4,
            temperature=0,
            aether_kv_session_id="spec",
        )
        actual_prefix, actual_cache = handle.session_caches["spec"]
        assert actual.text == expected.text
        assert actual.metrics.extra["kv_reused_tokens"] == len(prefix)
        assert actual.metrics.spec_accept_rate == 0
        np.testing.assert_array_equal(actual_prefix, expected_prefix)
        np.testing.assert_allclose(
            numpy_logits(actual_cache.last_logits), expected_logits, rtol=0, atol=5e-7
        )
        new_draft = restarted._draft_runtime._loaded_models[str(draft)]
        assert actual_cache.source_identity == (id(handle.engine), id(new_draft.engine))
