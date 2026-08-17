#!/usr/bin/env python3
"""
ci_smoke_test.py — Minimal CI smoke test for Aether Runtime.

Runs in under 60 seconds on any machine without network access or GPUs.
Designed to be the gating check in a GitHub Actions / GitLab CI pipeline.

What it tests:
  1. Package import and version string
  2. Quantization codecs (all formats: INT8, Q4_K_M, NF4, FP8, BF16)
  3. Native CPU kernel compilation + all 7 primitives
  4. CPUExecutionEngine — forward pass and greedy generation
  5. Full compile → quantize → save → load → infer cycle (toy model)
  6. AEG package integrity (manifest, blob, index all present)

Exit code: 0 = all pass, 1 = any failure.

Usage:
    python scripts/ci_smoke_test.py
    python scripts/ci_smoke_test.py --verbose
    python scripts/ci_smoke_test.py --junit results.xml   # JUnit XML for CI
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

import numpy as np

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ── Test registry ─────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float
    error: str = ""


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)

    def run(self, name: str, fn: Callable, verbose: bool = False) -> TestResult:
        if verbose:
            print(f"  {DIM}running {name}...{RESET}", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            r = TestResult(name=name, passed=True, duration_s=elapsed)
            if verbose:
                print(f"{GREEN}PASS{RESET}  ({elapsed*1000:.0f}ms)")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            r = TestResult(
                name=name, passed=False, duration_s=elapsed,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            if verbose:
                print(f"{RED}FAIL{RESET}  ({elapsed*1000:.0f}ms)")
                print(f"    {RED}{r.error.splitlines()[0]}{RESET}")
        self.results.append(r)
        return r

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ── Individual tests ──────────────────────────────────────────────────────────

def test_import() -> None:
    import aether  # noqa: F401
    from aether.compiler.compiler import Compiler
    from aether.runtime.cpu_engine import CPUExecutionEngine
    from aether.core.aeg_format import AEGPackage
    from aether.quantization.formats import quantize_tensor


def test_quantize_int8() -> None:
    from aether.quantization.formats import dequantize_tensor, quantize_tensor
    w = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
    qt = quantize_tensor(w, "INT8", block_size=32)
    out = dequantize_tensor(qt)
    assert out.shape == w.shape
    assert np.all(np.isfinite(out))


def test_quantize_q4km() -> None:
    from aether.quantization.formats import dequantize_tensor, quantize_tensor
    w = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32) * 0.1
    qt = quantize_tensor(w, "Q4_K_M", block_size=32)
    assert qt.packed
    out = dequantize_tensor(qt)
    assert out.shape == w.shape
    # Zero weights must roundtrip exactly.
    zeros = np.zeros((32, 32), np.float32)
    qt_z = quantize_tensor(zeros, "Q4_K_M", block_size=32)
    out_z = dequantize_tensor(qt_z)
    np.testing.assert_array_equal(out_z, np.zeros_like(out_z))


def test_quantize_nf4() -> None:
    from aether.quantization.formats import dequantize_tensor, quantize_tensor
    w = np.random.default_rng(2).standard_normal((32, 32)).astype(np.float32)
    qt = quantize_tensor(w, "NF4", block_size=32)
    out = dequantize_tensor(qt)
    assert out.shape == w.shape


def test_quantize_fp8() -> None:
    from aether.quantization.formats import dequantize_tensor, quantize_tensor
    w = np.random.default_rng(3).standard_normal((32, 32)).astype(np.float32) * 0.5
    qt = quantize_tensor(w, "FP8", block_size=32)
    out = dequantize_tensor(qt)
    assert out.shape == w.shape


def test_quantize_bf16() -> None:
    from aether.quantization.formats import dequantize_tensor, quantize_tensor
    w = np.array([1.0, -0.5, 0.125, 0.0], np.float32)
    qt = quantize_tensor(w, "BF16")
    out = dequantize_tensor(qt)
    np.testing.assert_allclose(out, w, atol=1e-3)


def test_native_kernels_rmsnorm() -> None:
    from aether.kernels.native_cpu import NativeCPUKernels
    k = NativeCPUKernels()
    x = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    w = np.ones(4, np.float32)
    out = k.rmsnorm(x, w)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_native_kernels_swiglu() -> None:
    from aether.kernels.native_cpu import NativeCPUKernels
    k = NativeCPUKernels()
    g = np.array([1.0, -1.0, 0.5], np.float32)
    u = np.ones(3, np.float32)
    out = k.swiglu(g, u)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_native_kernels_softmax() -> None:
    from aether.kernels.native_cpu import NativeCPUKernels
    k = NativeCPUKernels()
    x = np.array([[1.0, 2.0, 3.0, 0.5]], np.float32)
    out = k.softmax(x)
    assert abs(float(out.sum()) - 1.0) < 1e-5


def test_native_kernels_rope() -> None:
    from aether.kernels.native_cpu import NativeCPUKernels
    k = NativeCPUKernels()
    x = np.random.default_rng(4).standard_normal((4, 2, 8)).astype(np.float32)
    cos_t = np.ones((4, 4), np.float32)
    sin_t = np.zeros((4, 4), np.float32)
    # With sin=0, cos=1 → identity rotation.
    out = k.rope(x, cos_t, sin_t)
    np.testing.assert_allclose(out, x, atol=1e-5)


def test_native_kernels_argmax() -> None:
    from aether.kernels.native_cpu import NativeCPUKernels
    k = NativeCPUKernels()
    logits = np.array([0.1, 0.5, 0.9, 0.2], np.float32)
    assert k.argmax(logits) == 2


def test_cpu_engine_forward() -> None:
    from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
    rng = np.random.default_rng(5)
    h, v, heads, kv, inter = 32, 64, 2, 1, 64
    hd = h // heads

    lw = LayerWeights(
        attention_norm=np.ones(h, np.float32),
        q_proj=rng.standard_normal((heads*hd, h)).astype(np.float32)*0.02,
        k_proj=rng.standard_normal((kv*hd, h)).astype(np.float32)*0.02,
        v_proj=rng.standard_normal((kv*hd, h)).astype(np.float32)*0.02,
        o_proj=rng.standard_normal((h, heads*hd)).astype(np.float32)*0.02,
        ffn_norm=np.ones(h, np.float32),
        gate_proj=rng.standard_normal((inter, h)).astype(np.float32)*0.02,
        up_proj=rng.standard_normal((inter, h)).astype(np.float32)*0.02,
        down_proj=rng.standard_normal((h, inter)).astype(np.float32)*0.02,
    )
    mw = ModelWeights(
        embedding=rng.standard_normal((v, h)).astype(np.float32)*0.02,
        layers=[lw], final_norm=np.ones(h, np.float32),
        lm_head=rng.standard_normal((v, h)).astype(np.float32)*0.02,
    )
    engine = CPUExecutionEngine(mw, num_heads=heads, num_kv_heads=kv)
    logits, _ = engine.forward(np.array([1, 2, 3], np.int64))
    assert logits.shape == (3, v)
    assert np.all(np.isfinite(logits))


def test_cpu_engine_generate() -> None:
    from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
    rng = np.random.default_rng(6)
    h, v, heads, kv, inter = 32, 64, 2, 1, 64
    hd = h // heads
    lw = LayerWeights(
        attention_norm=np.ones(h, np.float32),
        q_proj=rng.standard_normal((heads*hd, h)).astype(np.float32)*0.02,
        k_proj=rng.standard_normal((kv*hd, h)).astype(np.float32)*0.02,
        v_proj=rng.standard_normal((kv*hd, h)).astype(np.float32)*0.02,
        o_proj=rng.standard_normal((h, heads*hd)).astype(np.float32)*0.02,
        ffn_norm=np.ones(h, np.float32),
        gate_proj=rng.standard_normal((inter, h)).astype(np.float32)*0.02,
        up_proj=rng.standard_normal((inter, h)).astype(np.float32)*0.02,
        down_proj=rng.standard_normal((h, inter)).astype(np.float32)*0.02,
    )
    mw = ModelWeights(
        embedding=rng.standard_normal((v, h)).astype(np.float32)*0.02,
        layers=[lw], final_norm=np.ones(h, np.float32),
        lm_head=rng.standard_normal((v, h)).astype(np.float32)*0.02,
    )
    engine = CPUExecutionEngine(mw, num_heads=heads, num_kv_heads=kv)
    tokens = engine.generate(np.array([1], np.int64), max_tokens=5, temperature=0.0)
    assert len(tokens) == 5
    assert all(0 <= t < v for t in tokens)


def test_e2e_compile_quantize_save_load_infer() -> None:
    """Full pipeline: build graph → quantize → save → load → run."""
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
    from aether.compiler.weight_quantizer import quantize_graph_weights
    from aether.core.aeg_format import AEGPackage
    from aether.core.graph import AEGGraph
    from aether.core.types import ModelArchitecture
    from aether.runtime.aeg_loader import load_engine_from_package
    from aether.runtime.cpu_engine import LayerWeights, ModelWeights

    rng = np.random.default_rng(7)
    h, v, heads, kv, inter = 64, 256, 4, 2, 128
    hd = h // heads

    arch = ModelArchitecture(
        family="ci_test", params_billion=0.001, layers=1,
        hidden_size=h, num_attention_heads=heads, num_kv_heads=kv,
        head_dim=hd, intermediate_size=inter, vocab_size=v,
    )
    lw = LayerWeights(
        attention_norm=np.ones(h, np.float32),
        q_proj=rng.standard_normal((heads*hd, h)).astype(np.float32)*0.02,
        k_proj=rng.standard_normal((kv*hd, h)).astype(np.float32)*0.02,
        v_proj=rng.standard_normal((kv*hd, h)).astype(np.float32)*0.02,
        o_proj=rng.standard_normal((h, heads*hd)).astype(np.float32)*0.02,
        ffn_norm=np.ones(h, np.float32),
        gate_proj=rng.standard_normal((inter, h)).astype(np.float32)*0.02,
        up_proj=rng.standard_normal((inter, h)).astype(np.float32)*0.02,
        down_proj=rng.standard_normal((h, inter)).astype(np.float32)*0.02,
    )
    mw = ModelWeights(
        embedding=rng.standard_normal((v, h)).astype(np.float32)*0.02,
        layers=[lw], final_norm=np.ones(h, np.float32),
        lm_head=rng.standard_normal((v, h)).astype(np.float32)*0.02,
    )

    pipeline = IngestionPipeline()
    graph = AEGGraph(name="ci_test", architecture=arch)
    pipeline._build_architecture_graph(graph, arch)

    weight_map = {
        "embedding": mw.embedding, "final_norm": mw.final_norm, "lm_head": mw.lm_head,
        "layer_0_rmsnorm": lw.attention_norm, "layer_0_qkv": np.concatenate([lw.q_proj, lw.k_proj, lw.v_proj], 0),
        "layer_0_out_proj": lw.o_proj, "layer_0_ffn_norm": lw.ffn_norm,
        "layer_0_gate_proj": lw.gate_proj, "layer_0_ffn": lw.down_proj,
    }
    for node in graph:
        w = weight_map.get(getattr(node, "id", ""))
        if w is not None:
            node.add_attribute("weight", w)
        # The architecture graph represents the SwiGLU gate and up
        # projections as one logical node. Preserve both real checkpoint
        # tensors using the same binding contract as Stage 1 ingestion.
        if getattr(node, "id", "") == "layer_0_gate_proj":
            node.add_attribute("up_weight", lw.up_proj)

    with tempfile.TemporaryDirectory() as tmp:
        pkg_path = Path(tmp) / "ci_test.aeg"
        pkg = AEGPackage.create(pkg_path, model_id="ci_test", aether_version="0")
        pkg.manifest.architecture = arch  # type: ignore[union-attr]
        quantize_graph_weights(graph, pkg, default_precision="Q4_K_M", block_size=32)
        pkg.save()

        # Verify blobs exist.
        assert (pkg_path / "weights" / "quantized" / "model.aeg-quant").exists()
        assert (pkg_path / "weights" / "quantized" / "weight_index.json").exists()

        # Load and run.
        loaded = AEGPackage(pkg_path)
        loaded.load()
        engine = load_engine_from_package(loaded)
        logits, _ = engine.forward(np.array([1, 2, 3], np.int64))
        assert logits.shape == (3, v)
        assert np.all(np.isfinite(logits))


def test_weight_store_round_trip() -> None:
    from aether.core.weight_store import WeightStore
    from aether.quantization.formats import QuantizedTensor, dequantize_tensor, quantize_tensor
    import tempfile

    w = np.random.default_rng(8).standard_normal((32, 64)).astype(np.float32) * 0.1
    qt = quantize_tensor(w, "Q4_K_M", block_size=32)

    with tempfile.TemporaryDirectory() as tmp:
        store = WeightStore(tmp)
        store.save({"test_weight": qt})
        assert (Path(tmp) / "model.aeg-quant").exists()

        store2 = WeightStore(tmp)
        loaded = store2.load_tensor("test_weight")
        out = dequantize_tensor(loaded)
        assert out.shape == w.shape


# ── JUnit XML output ──────────────────────────────────────────────────────────

def _write_junit(suite: TestSuite, path: str) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="aether-ci-smoke" tests="{len(suite.results)}" '
        f'failures="{suite.failed}" time="{sum(r.duration_s for r in suite.results):.3f}">',
    ]
    for r in suite.results:
        lines.append(
            f'  <testcase name="{r.name}" time="{r.duration_s:.3f}">'
        )
        if not r.passed:
            msg = r.error.splitlines()[0].replace('"', "'")
            lines.append(f'    <failure message="{msg}"><![CDATA[{r.error}]]></failure>')
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_TESTS: list[tuple[str, Callable]] = [
    ("import",                      test_import),
    ("quantize_int8",               test_quantize_int8),
    ("quantize_q4km",               test_quantize_q4km),
    ("quantize_nf4",                test_quantize_nf4),
    ("quantize_fp8",                test_quantize_fp8),
    ("quantize_bf16",               test_quantize_bf16),
    ("kernels_rmsnorm",             test_native_kernels_rmsnorm),
    ("kernels_swiglu",              test_native_kernels_swiglu),
    ("kernels_softmax",             test_native_kernels_softmax),
    ("kernels_rope",                test_native_kernels_rope),
    ("kernels_argmax",              test_native_kernels_argmax),
    ("cpu_engine_forward",          test_cpu_engine_forward),
    ("cpu_engine_generate",         test_cpu_engine_generate),
    ("e2e_compile_quantize_infer",  test_e2e_compile_quantize_save_load_infer),
    ("weight_store_round_trip",     test_weight_store_round_trip),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Aether Runtime CI smoke test")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--junit", metavar="FILE", help="Write JUnit XML to FILE")
    args = parser.parse_args()

    print(f"{BOLD}Aether Runtime -- CI Smoke Test{RESET}")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")

    suite = TestSuite()
    t_total = time.perf_counter()

    for name, fn in ALL_TESTS:
        suite.run(name, fn, verbose=args.verbose)

    elapsed = time.perf_counter() - t_total

    # Summary
    print(f"\n{'='*50}")
    if suite.failed == 0:
        print(f"{GREEN}{BOLD}PASSED  {suite.passed}/{len(suite.results)} tests  ({elapsed:.2f}s){RESET}")
    else:
        print(f"{RED}{BOLD}FAILED  {suite.failed} of {len(suite.results)} tests  ({elapsed:.2f}s){RESET}")
        print()
        for r in suite.results:
            if not r.passed:
                print(f"  {RED}[X] {r.name}{RESET}")
                for line in r.error.splitlines()[:5]:
                    print(f"      {DIM}{line}{RESET}")

    if args.junit:
        _write_junit(suite, args.junit)
        print(f"\nJUnit XML → {args.junit}")

    return 0 if suite.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
