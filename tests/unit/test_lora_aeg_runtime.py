"""Real Pass 21 artifact loading and CPU projection execution tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether.adapters.lora import load_compiled_lora_adapters
from aether.compiler.stage2_optimizer.pass21_advanced_peft import _write_lora_blob
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights


def _write_adapter_artifact(root: Path) -> None:
    adapter_root = root / "adapters" / "demo"
    adapter_root.mkdir(parents=True)
    name_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    name_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    A = np.arange(8, dtype=np.float32).reshape(2, 4) / 10.0
    B = np.arange(8, dtype=np.float32).reshape(4, 2) / 20.0
    _write_lora_blob(adapter_root / "lora_A.bin", {name_a: A.reshape(-1).tolist()}, {name_a: [2, 4]}, 2, True)
    _write_lora_blob(adapter_root / "lora_B.bin", {name_b: B.reshape(-1).tolist()}, {name_b: [4, 2]}, 2, False)
    (root / "adapters" / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "format": "aether_adapter_manifest_v1",
                "n_adapters": 1,
                "adapters": [
                    {
                        "name": "demo",
                        "rank": 2,
                        "lora_A_ref": "demo/lora_A.bin",
                        "lora_B_ref": "demo/lora_B.bin",
                        "runtime_scale": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _engine() -> CPUExecutionEngine:
    rng = np.random.default_rng(11)
    hidden = 4
    layer = LayerWeights(
        attention_norm=np.ones(hidden, dtype=np.float32),
        q_proj=rng.normal(size=(4, hidden)).astype(np.float32),
        k_proj=rng.normal(size=(2, hidden)).astype(np.float32),
        v_proj=rng.normal(size=(2, hidden)).astype(np.float32),
        o_proj=rng.normal(size=(hidden, hidden)).astype(np.float32),
        ffn_norm=np.ones(hidden, dtype=np.float32),
        gate_proj=rng.normal(size=(8, hidden)).astype(np.float32),
        up_proj=rng.normal(size=(8, hidden)).astype(np.float32),
        down_proj=rng.normal(size=(hidden, 8)).astype(np.float32),
    )
    return CPUExecutionEngine(
        ModelWeights(
            embedding=rng.normal(size=(16, hidden)).astype(np.float32),
            layers=[layer],
            final_norm=np.ones(hidden, dtype=np.float32),
            lm_head=rng.normal(size=(16, hidden)).astype(np.float32),
        ),
        num_heads=2,
        num_kv_heads=1,
    )


def test_compiled_lora_blob_is_loaded_and_changes_real_projection(tmp_path: Path) -> None:
    _write_adapter_artifact(tmp_path)
    adapters = load_compiled_lora_adapters(tmp_path)
    assert set(adapters) == {"demo"}
    assert set(adapters["demo"]) == {(0, "q_proj")}

    engine = _engine()
    selected = engine.with_lora_adapter(adapters, "demo")
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    base = engine._linear(x, engine.weights.layers[0].q_proj, (0, "q_proj"))
    applied = selected._linear(x, selected.weights.layers[0].q_proj, (0, "q_proj"))
    A = adapters["demo"][(0, "q_proj")][0]
    B = adapters["demo"][(0, "q_proj")][1]
    expected = base + ((x @ A.T) @ B.T) * 0.5
    np.testing.assert_allclose(applied, expected, rtol=2e-2, atol=2e-2)

    base_logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    adapted_logits, _ = selected.forward(
        np.asarray([1, 2], dtype=np.int64), adapter_id="demo"
    )
    assert not np.allclose(base_logits, adapted_logits)
    generated, _ = selected.generate_with_cache(
        np.asarray([1, 2], dtype=np.int64), max_tokens=1, temperature=0.0, adapter_id="demo"
    )
    assert len(generated) == 1


def test_cpu_decode_does_not_copy_transposed_projection_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-token projection must keep the zero-copy transpose view.

    A decode step visits every projection in every layer.  Materialising a
    contiguous transpose at each visit turns a normal model into repeated
    model-sized memory copies and makes CPU generation appear hung.
    """
    import aether.runtime.cpu_engine as cpu_engine_module

    engine = _engine()
    original = cpu_engine_module.np.ascontiguousarray
    calls: list[tuple[int, ...]] = []

    def record(value, *args, **kwargs):
        calls.append(tuple(np.asarray(value).shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(cpu_engine_module.np, "ascontiguousarray", record)
    engine._linear(np.ones((1, 4), dtype=np.float32), engine.weights.layers[0].q_proj)

    assert calls == [(1, 4)]


def test_compiled_lora_loader_rejects_legacy_or_tampered_blob(tmp_path: Path) -> None:
    _write_adapter_artifact(tmp_path)
    blob = tmp_path / "adapters" / "demo" / "lora_A.bin"
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(data)
    with pytest.raises(ValueError, match="unsupported or truncated"):
        load_compiled_lora_adapters(tmp_path)


def test_pass21_writes_runtime_consumable_safetensors_artifact(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    from aether.compiler.config import CompilerConfig
    from aether.compiler.stage2_optimizer.pass21_advanced_peft import AdvancedPEFTCompilationPass

    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8"
    )
    safetensors.save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((2, 4), dtype=np.float32),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones((4, 2), dtype=np.float32),
        },
        str(source / "adapter_model.safetensors"),
    )

    class Graph:
        output_dir = tmp_path / "aeg"
        metadata: dict[str, object] = {}

    Graph.output_dir.mkdir()
    _, report = AdvancedPEFTCompilationPass().run(
        Graph(), {"hidden_size": 4},
        CompilerConfig(enable_advanced_peft=True, peft_adapter_paths=[str(source)]),
    )
    assert report.status == "applied"
    loaded = load_compiled_lora_adapters(Graph.output_dir)
    assert set(loaded["adapter"]) == {(0, "q_proj")}
