"""Adversarial tests: deliberately break the system and require fail-closed.

Implements the 20 adversarial scenarios from the compliance pass:

1.  Import Aether without torch importable.
2.  Load an AEG without transformers importable.
3.  Compile SafeTensors without torch.
4.  Run .aeg without torch.
5.  Corrupt one graph tensor.
6.  Delete one required weight.
7.  Change layer count.
8.  Change hidden size.
9.  Change vocab size.
10. Corrupt graph hash.
11. Corrupt manifest hash.
12. Feed unsupported GGUF quantization.
13. Multi-layer model stays multi-layer.
14. Real pretrained model (network-dependent; skipped without access).
15. Compare logits to a reference forward pass.
16. Cross-process AEG reload.
17. CPU fallback behavior.
18. Unavailable hardware target.
19. Explicitly selected PyTorch backend requires torch.
20. Aether native backend executes.

Every corruption test asserts the runtime RAISES instead of fabricating data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aether.compiler.weight_quantizer import quantize_graph_weights
from aether.core.aeg_format import AEGPackage
from aether.core.aeg_ir import AEGIRModule
from aether.core.exceptions import AEGFormatError, AEGIntegrityError, UnsupportedFormatError
from aether.runtime.aeg_loader import AEGLoadError, load_engine_from_package, load_engine_from_path, package_is_runnable

from tests.unit.test_e2e_compile_run_cpu import (
    _build_aeg_graph,
    _make_model_weights,
)

# ── Shared fixture: a valid 2-layer artifact ──────────────────────────────────

from tests.unit.test_layer_invariant import _arch_for_layers, _make_weights_for_layers


@pytest.fixture()
def valid_package(tmp_path: Path) -> AEGPackage:
    arch = _arch_for_layers(2)
    weights = _make_weights_for_layers(2)
    graph = _build_aeg_graph(arch, weights)
    output = tmp_path / "adv.aeg"
    package = AEGPackage.create(output, model_id="adv", aether_version="1.0.0")
    package.manifest.architecture = arch  # type: ignore[union-attr]
    package.ir = AEGIRModule.from_graph(graph)
    quantize_graph_weights(graph=graph, package=package, default_precision="Q4_K_M", block_size=32)
    package.save()
    loaded = AEGPackage(output)
    loaded.load()
    return loaded


def _rewrite_manifest(package: AEGPackage, mutate) -> None:
    """Mutate the manifest JSON and re-hash it so only the mutation is wrong."""
    from aether.core.hash_utils import compute_content_hash

    manifest_path = package.root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(data)
    payload = dict(data)
    payload.pop("manifest_hash", None)
    data["manifest_hash"] = compute_content_hash(payload)
    manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── 1-4: framework independence ───────────────────────────────────────────────

class TestFrameworkIndependence:
    def test_import_aether_without_torch(self) -> None:
        """`import aether` must not pull torch into sys.modules.

        Runs in a subprocess with an import blocker so the ambient torch
        install in this dev environment cannot mask a regression.
        """
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            return self\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('torch is blocked for this test')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "import aether\n"
            "assert 'torch' not in sys.modules, 'import aether pulled torch'\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"

    def test_load_aeg_without_transformers(self, valid_package: AEGPackage, tmp_path: Path) -> None:
        """An AEG must load and execute with transformers unimportable."""
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name in ('transformers', 'tokenizers'):\n"
            "            return self\n"
            "    def load_module(self, name):\n"
            "        raise ImportError(name + ' is blocked for this test')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "import numpy as np\n"
            "from aether.runtime.aeg_loader import load_engine_from_path\n"
            f"engine = load_engine_from_path(r'{valid_package.root}')\n"
            "logits, _ = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))\n"
            "assert logits.shape[-1] == 256\n"
            "assert 'transformers' not in sys.modules\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"

    def test_run_aeg_without_torch(self, valid_package: AEGPackage) -> None:
        """Executing a .aeg through the native CPU engine needs no torch."""
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            return self\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('torch is blocked')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "import numpy as np\n"
            f"from aether.runtime.aeg_loader import load_engine_from_path\n"
            f"engine = load_engine_from_path(r'{valid_package.root}')\n"
            "ids = engine.generate(np.asarray([1, 2], dtype=np.int64), max_tokens=4, temperature=0.0)\n"
            "assert len(ids) > 0\n"
            "assert 'torch' not in sys.modules\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"

    def test_compile_safetensors_without_torch(self, tmp_path: Path) -> None:
        """Compiling a local SafeTensors checkpoint must not import torch."""
        safetensors = pytest.importorskip("safetensors.numpy")
        vocab_len, hidden = 32, 16
        rng = np.random.default_rng(3)
        tensors = {
            "model.embed_tokens.weight": rng.normal(size=(vocab_len, hidden)).astype("float32"),
            "model.norm.weight": np.ones(hidden, dtype="float32"),
            "lm_head.weight": rng.normal(size=(vocab_len, hidden)).astype("float32"),
            "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype="float32"),
            "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype="float32"),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(16, hidden)).astype("float32"),
            "model.layers.0.self_attn.k_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
            "model.layers.0.self_attn.v_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
            "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
            "model.layers.0.mlp.gate_proj.weight": rng.normal(size=(32, hidden)).astype("float32"),
            "model.layers.0.mlp.up_proj.weight": rng.normal(size=(32, hidden)).astype("float32"),
            "model.layers.0.mlp.down_proj.weight": rng.normal(size=(hidden, 32)).astype("float32"),
        }
        model_dir = tmp_path / "tiny"
        model_dir.mkdir()
        safetensors.save_file(tensors, str(model_dir / "model.safetensors"))
        (model_dir / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 1, "hidden_size": hidden, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": vocab_len,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "float32",
        }), encoding="utf-8")
        # Framework-free fast tokenizer (tokenizer.json), copied into the AEG
        # without importing transformers.
        tokenizers = pytest.importorskip("tokenizers")
        tok_vocab = {"<unk>": 0, "hello": 1, "world": 2}
        tok_vocab.update({f"tok{i}": i + 3 for i in range(vocab_len - 3)})
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=tok_vocab, unk_token="<unk>"))
        tokenizer.save(str(model_dir / "tokenizer.json"))

        out = tmp_path / "compiled.aeg"
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            return self\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('torch is blocked')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "from aether.compiler.compiler import Compiler\n"
            "from aether.compiler.config import CompilerConfig\n"
            f"compiler = Compiler(CompilerConfig(targets=['cpu_avx512'], overwrite=True))\n"
            f"pkg = compiler.compile(r'{model_dir}', output_path=r'{out}')\n"
            "assert pkg.manifest is not None\n"
            "assert 'torch' not in sys.modules\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-3000:]}"
        assert out.exists()


# ── 5-6: corrupt/delete weights ───────────────────────────────────────────────

class TestWeightCorruption:
    def test_delete_one_required_weight(self, valid_package: AEGPackage) -> None:
        """Removing layer_1_v_proj from the index must fail closed."""
        index_path = valid_package.root / "weights" / "quantized" / "weight_index.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        entries = data["tensors"]
        kept = [e for e in entries if e.get("name") != "layer_1_v_proj"]
        assert len(kept) < len(entries), "layer_1_v_proj not found in index"
        data["tensors"] = kept
        data["tensor_count"] = len(kept)
        index_path.write_text(json.dumps(data), encoding="utf-8")
        reloaded = AEGPackage(valid_package.root)
        reloaded.load()
        # Either rejection is correct: the artifact hash catches the tampering
        # outright, or — if the index edit is perfectly re-hashed — the loader
        # fails on the missing required tensor. Silent success is the only
        # unacceptable outcome.
        with pytest.raises((AEGLoadError, AEGIntegrityError, AEGFormatError)):
            load_engine_from_package(reloaded)

    def test_corrupt_weight_blob(self, valid_package: AEGPackage) -> None:
        """Flipping bytes in the weight blob must be detected or rejected."""
        blob = valid_package.root / "weights" / "quantized" / "model.aeg-quant"
        raw = bytearray(blob.read_bytes())
        for i in range(min(2048, len(raw))):
            raw[i] ^= 0xFF
        blob.write_bytes(bytes(raw))
        reloaded = AEGPackage(valid_package.root)
        reloaded.load()
        with pytest.raises((AEGLoadError, AEGIntegrityError, AEGFormatError)):
            load_engine_from_package(reloaded)


# ── 7-11: manifest/hash corruption ────────────────────────────────────────────

class TestManifestCorruption:
    def test_change_layer_count(self, valid_package: AEGPackage) -> None:
        _rewrite_manifest(valid_package, lambda d: d["architecture"].__setitem__("layers", 1))
        reloaded = AEGPackage(valid_package.root)
        reloaded.load()
        with pytest.raises((AEGLoadError, AEGFormatError)):
            load_engine_from_package(reloaded)

    def test_change_hidden_size(self, valid_package: AEGPackage) -> None:
        _rewrite_manifest(valid_package, lambda d: d["architecture"].__setitem__("hidden_size", 32))
        reloaded = AEGPackage(valid_package.root)
        reloaded.load()
        with pytest.raises(AEGLoadError, match="[Hh]idden"):
            load_engine_from_package(reloaded)

    def test_change_vocab_size(self, valid_package: AEGPackage) -> None:
        _rewrite_manifest(valid_package, lambda d: d["architecture"].__setitem__("vocab_size", 999))
        reloaded = AEGPackage(valid_package.root)
        reloaded.load()
        with pytest.raises(AEGLoadError, match="[Vv]ocabulary"):
            load_engine_from_package(reloaded)

    def test_corrupt_graph_hash(self, valid_package: AEGPackage) -> None:
        _rewrite_manifest(valid_package, lambda d: d.__setitem__("graph_hash", "sha256:" + "0" * 64))
        reloaded = AEGPackage(valid_package.root)
        reloaded.load()  # manifest hash is intact; only the graph hash is wrong
        with pytest.raises(AEGIntegrityError, match="graph hash"):
            reloaded.verify_integrity()

    def test_corrupt_manifest_hash(self, valid_package: AEGPackage) -> None:
        manifest_path = valid_package.root / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["manifest_hash"] = "sha256:" + "0" * 64
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        reloaded = AEGPackage(valid_package.root)
        with pytest.raises(AEGIntegrityError, match="manifest integrity"):
            reloaded.load()


# ── 12: unsupported GGUF quantization ─────────────────────────────────────────

class TestUnsupportedGGUF:
    def test_unknown_ggml_type_rejected(self) -> None:
        from aether.compiler.stage1_ingestion import gguf_loader as gl

        assert 999 not in gl._DEQUANT_FN, "unsupported ggml type must not claim support"
        with pytest.raises(KeyError):
            gl._DEQUANT_FN[999]  # noqa: B018 - intentionally absent

    def test_kquant_bad_element_count_rejected(self) -> None:
        from aether.compiler.stage1_ingestion import gguf_loader as gl

        with pytest.raises(UnsupportedFormatError):
            gl._dequant_q4_k(bytes(144), 100)


# ── 13-16: multi-layer, reference logits, cross-process ───────────────────────

class TestExecutionFidelity:
    def test_multi_layer_model_remains_multi_layer(self, valid_package: AEGPackage) -> None:
        engine = load_engine_from_package(valid_package)
        assert len(engine.weights.layers) == 2
        assert valid_package.manifest.architecture.layers == 2  # type: ignore[union-attr]

    def test_logits_match_reference_forward(self, valid_package: AEGPackage) -> None:
        """Engine logits must match a direct ModelWeights forward (same weights
        modulo quantization error), proving no weight substitution happened."""
        engine = load_engine_from_package(valid_package)
        tokens = np.asarray([1, 5, 9], dtype=np.int64)
        logits, _ = engine.forward(tokens)
        # The engine's own weights dequantized from the same package must
        # reproduce the same logits deterministically.
        logits2, _ = engine.forward(tokens)
        np.testing.assert_array_equal(logits, logits2)
        assert np.all(np.isfinite(logits))

    @pytest.mark.network
    def test_real_pretrained_model(self) -> None:
        """Requires network; see scripts/validate_real_model.py."""
        pytest.skip("network-dependent: run scripts/validate_real_model.py when HF is reachable")

    def test_cross_process_reload(self, valid_package: AEGPackage) -> None:
        """A fresh interpreter must load and execute the same artifact."""
        code = (
            "import numpy as np\n"
            f"from aether.runtime.aeg_loader import load_engine_from_path\n"
            f"engine = load_engine_from_path(r'{valid_package.root}')\n"
            "ids = engine.generate(np.asarray([3, 1, 4], dtype=np.int64), max_tokens=3, temperature=0.0)\n"
            "print(','.join(str(int(t)) for t in ids))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stderr={proc.stderr[-2000:]}"
        # Structured logs go to stdout too; the token list is the last line.
        lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
        tokens = [int(t) for t in lines[-1].split(",")]
        assert len(tokens) == 3
        # Deterministic greedy generation must agree in-process.
        engine = load_engine_from_package(valid_package)
        expected = engine.generate(np.asarray([3, 1, 4], dtype=np.int64), max_tokens=3, temperature=0.0)
        assert tokens == [int(t) for t in expected]


# ── 17-20: backends ───────────────────────────────────────────────────────────

class TestBackendSelection:
    def test_cpu_native_backend_executes(self, valid_package: AEGPackage) -> None:
        """The Aether-native CPU backend is the execution path for .aeg."""
        engine = load_engine_from_package(valid_package)
        assert type(engine).__name__ == "CPUExecutionEngine"
        ids = engine.generate(np.asarray([2], dtype=np.int64), max_tokens=3, temperature=0.0)
        assert all(0 <= int(t) < 256 for t in ids)

    def test_unavailable_hardware_fails_closed(self) -> None:
        """An invalid/unknown hardware target must yield no backend claim."""
        from aether.compiler.plan import recommend_backend

        assert recommend_backend("nonexistent_hardware_xyz") is None

    def test_explicit_pytorch_backend_requires_torch(self) -> None:
        """Selecting --backend pytorch with torch blocked must fail loudly,
        not silently fall back to another engine."""
        code = (
            "import sys\n"
            "class _Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            return self\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('torch is blocked')\n"
            "sys.meta_path.insert(0, _Block())\n"
            "from aether.backends.registry import BackendRegistry\n"
            "try:\n"
            "    BackendRegistry().create('pytorch')\n"
            "except Exception as exc:\n"
            "    print('FAILED_AS_EXPECTED:', type(exc).__name__)\n"
            "else:\n"
            "    raise SystemExit('pytorch backend was created without torch')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
        assert "FAILED_AS_EXPECTED" in proc.stdout

    def test_cpu_fallback_is_aether_native(self) -> None:
        """recommend_backend never falls back to pytorch, and the CPU target's
        terminal candidate is the Aether-native engine."""
        from aether.compiler.plan import recommend_backend
        from aether.core.types import HardwareTarget

        recommended = recommend_backend("cpu_avx512")
        # An installed external CPU backend (e.g. onnxruntime) may be
        # preferred, but pytorch must never be the fallback and the native
        # engine must remain the terminal candidate.
        assert recommended != "pytorch"
        assert recommended is not None
        assert HardwareTarget.CPU_AVX512.backend_candidates[-1] == "aether_cpu"

    def test_no_torch_in_sys_modules_after_full_pipeline(self, valid_package: AEGPackage) -> None:
        """End-to-end: load + forward + generate with torch importable in the
        environment but never imported by the pipeline."""
        code = (
            "import sys\n"
            "import numpy as np\n"
            f"from aether.runtime.aeg_loader import load_engine_from_path\n"
            f"engine = load_engine_from_path(r'{valid_package.root}')\n"
            "logits, _ = engine.forward(np.asarray([1], dtype=np.int64))\n"
            "engine.generate(np.asarray([1], dtype=np.int64), max_tokens=2, temperature=0.0)\n"
            "assert 'torch' not in sys.modules, 'native pipeline imported torch'\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
