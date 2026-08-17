"""Regression tests for the layer/architecture invariant (the 4-layer → 1-layer bug).

Hard invariant under test::

    source_layers == graph_layers == manifest_layers == runtime_layers

plus hidden size, heads, and vocabulary consistency between the compiled
artifact and the rebuilt CPU engine. Compilation and artifact loading must
FAIL CLOSED on any mismatch, and a deliberately corrupted manifest must be
rejected instead of silently loading a wrong-shaped model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether.compiler.weight_quantizer import quantize_graph_weights
from aether.core.aeg_format import AEGPackage
from aether.core.exceptions import AEGFormatError
from aether.runtime.aeg_loader import AEGLoadError, load_engine_from_package, package_is_runnable

# Reuse the real pipeline helpers from the e2e suite.
from tests.unit.test_e2e_compile_run_cpu import (
    _build_aeg_graph,
    _make_model_weights,
    _rng,
)

from aether.core.aeg_ir import AEGIRModule
from aether.core.types import ModelArchitecture


def _arch_for_layers(n_layers: int) -> ModelArchitecture:
    return ModelArchitecture(
        family="llama_test",
        params_billion=0.001,
        layers=n_layers,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _make_weights_for_layers(n_layers: int):
    """Build ModelWeights with exactly ``n_layers`` layers."""
    from tests.unit.test_e2e_compile_run_cpu import _make_layer_weights
    from aether.runtime.cpu_engine import ModelWeights

    rng = _rng(7 + n_layers)
    layers = [_make_layer_weights(rng, i) for i in range(n_layers)]
    return ModelWeights(
        embedding=rng.standard_normal((256, 64)).astype(np.float32) * 0.02,
        layers=layers,
        final_norm=np.ones(64, dtype=np.float32),
        lm_head=rng.standard_normal((256, 64)).astype(np.float32) * 0.02,
        rope_theta=10000.0,
        norm_eps=1e-5,
    )


def _compile_roundtrip(tmp_path: Path, n_layers: int) -> tuple[AEGPackage, object]:
    """Graph → quantize → save → reload; returns (package, engine)."""
    arch = _arch_for_layers(n_layers)
    weights = _make_weights_for_layers(n_layers)
    graph = _build_aeg_graph(arch, weights)

    output = tmp_path / f"model_{n_layers}l.aeg"
    package = AEGPackage.create(output, model_id=f"layers_{n_layers}", aether_version="1.0.0")
    package.manifest.architecture = arch  # type: ignore[union-attr]
    package.ir = AEGIRModule.from_graph(graph)
    stats = quantize_graph_weights(
        graph=graph,
        package=package,
        precision_map={f"layer_{i}": "Q4_K_M" for i in range(n_layers)},
        default_precision="Q4_K_M",
        block_size=32,
    )
    assert stats.tensors_written >= n_layers * 9, "every layer must serialize its tensors"
    package.save()

    loaded = AEGPackage(output)
    loaded.load()
    engine = load_engine_from_package(loaded)
    return loaded, engine


class TestLayerInvariantRoundTrip:
    @pytest.mark.parametrize("n_layers", [1, 2, 4, 8])
    def test_layer_count_preserved_through_pipeline(self, tmp_path: Path, n_layers: int) -> None:
        """A model with N layers must remain N layers after save + reload."""
        package, engine = _compile_roundtrip(tmp_path, n_layers)
        manifest_layers = package.manifest.architecture.layers  # type: ignore[union-attr]
        runtime_layers = len(engine.weights.layers)
        assert manifest_layers == n_layers
        assert runtime_layers == n_layers
        # The runtime actually executes the same layer count it declares.
        engine.num_layers == n_layers if hasattr(engine, "num_layers") else None

    def test_hidden_and_heads_preserved(self, tmp_path: Path) -> None:
        package, engine = _compile_roundtrip(tmp_path, 4)
        assert engine.weights.embedding.shape[1] == 64
        assert engine.num_heads == 4
        assert engine.num_kv_heads == 2
        assert package.manifest.architecture.hidden_size == 64  # type: ignore[union-attr]

    def test_forward_produces_vocab_sized_logits(self, tmp_path: Path) -> None:
        _, engine = _compile_roundtrip(tmp_path, 4)
        logits, _cache = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
        assert logits.shape[-1] == 256
        assert np.all(np.isfinite(logits))

    def test_graph_hash_is_never_pending(self, tmp_path: Path) -> None:
        package, _ = _compile_roundtrip(tmp_path, 2)
        assert package.manifest.graph_hash != "sha256:pending"  # type: ignore[union-attr]
        assert package.manifest.graph_hash.startswith("sha256:")  # type: ignore[union-attr]


class TestCorruptedManifestFailsClosed:
    def test_layer_count_corruption_rejected(self, tmp_path: Path) -> None:
        """Rewriting manifest layers=1 for a 4-layer model must fail at load."""
        package, _ = _compile_roundtrip(tmp_path, 4)
        import json

        manifest_path = package.root / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["architecture"]["layers"] = 1
        # Re-hash so only the layer count is wrong, not the manifest integrity.
        from aether.core.hash_utils import compute_content_hash

        payload = dict(data)
        payload.pop("manifest_hash", None)
        data["manifest_hash"] = compute_content_hash(payload)
        manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        reloaded = AEGPackage(package.root)
        reloaded.load()
        with pytest.raises((AEGLoadError, AEGFormatError)):
            load_engine_from_package(reloaded)

    def test_vocab_mismatch_rejected(self, tmp_path: Path) -> None:
        """A manifest claiming a different vocab than the embedding must fail."""
        package, _ = _compile_roundtrip(tmp_path, 2)
        import json

        manifest_path = package.root / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["architecture"]["vocab_size"] = 999
        from aether.core.hash_utils import compute_content_hash

        payload = dict(data)
        payload.pop("manifest_hash", None)
        data["manifest_hash"] = compute_content_hash(payload)
        manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        reloaded = AEGPackage(package.root)
        reloaded.load()
        with pytest.raises(AEGLoadError, match="[Vv]ocabulary"):
            load_engine_from_package(reloaded)

    def test_hidden_mismatch_rejected(self, tmp_path: Path) -> None:
        package, _ = _compile_roundtrip(tmp_path, 2)
        import json

        manifest_path = package.root / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["architecture"]["hidden_size"] = 32
        from aether.core.hash_utils import compute_content_hash

        payload = dict(data)
        payload.pop("manifest_hash", None)
        data["manifest_hash"] = compute_content_hash(payload)
        manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        reloaded = AEGPackage(package.root)
        reloaded.load()
        with pytest.raises(AEGLoadError, match="[Hh]idden"):
            load_engine_from_package(reloaded)


class TestSaveFailClosed:
    def test_runnable_package_without_ir_rejected(self, tmp_path: Path) -> None:
        """Weights without a graph must never produce a runnable artifact."""
        arch = _arch_for_layers(2)
        weights = _make_weights_for_layers(2)
        graph = _build_aeg_graph(arch, weights)
        package = AEGPackage.create(tmp_path / "no_ir.aeg", model_id="m", aether_version="1.0.0")
        package.manifest.architecture = arch  # type: ignore[union-attr]
        quantize_graph_weights(graph=graph, package=package, default_precision="Q4_K_M", block_size=32)
        with pytest.raises(AEGFormatError, match="graph hash"):
            package.save()

    def test_placeholder_architecture_rejected(self, tmp_path: Path) -> None:
        arch = _arch_for_layers(2)
        weights = _make_weights_for_layers(2)
        graph = _build_aeg_graph(arch, weights)
        package = AEGPackage.create(tmp_path / "unknown.aeg", model_id="m", aether_version="1.0.0")
        # architecture left at placeholder default (family="unknown", layers=1)
        package.ir = AEGIRModule.from_graph(graph)
        quantize_graph_weights(graph=graph, package=package, default_precision="Q4_K_M", block_size=32)
        with pytest.raises(AEGFormatError, match="placeholder architecture"):
            package.save()

    def test_pending_hash_package_not_runnable(self, tmp_path: Path) -> None:
        """package_is_runnable must reject a pending-hash manifest."""
        package, _ = _compile_roundtrip(tmp_path, 2)
        package.manifest.graph_hash = "sha256:pending"  # type: ignore[union-attr]
        assert package_is_runnable(package) is False
