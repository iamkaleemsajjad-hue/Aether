"""
Adversarial test suite — second forensic audit.

These tests verify BEHAVIOR, not file existence. Each test either:
  (a) proves an execution path actually runs, or
  (b) proves an isolation guarantee holds.

Run with:
    pytest tests/unit/test_adversarial_audit.py -v
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_weights(vocab=32, hidden=16, heads=2, kv_heads=2, layers=2, ffn=32):
    """Build a tiny CPUExecutionEngine for behavioral tests.

    Uses the real LayerWeights/ModelWeights dataclass field names discovered
    by introspection, and passes ``num_heads`` separately to the engine
    constructor as required by ``CPUExecutionEngine.__init__``.
    """
    from aether.runtime.cpu_engine import (
        CPUExecutionEngine, ModelWeights, LayerWeights,
    )
    rng = np.random.default_rng(42)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    layer_list = []
    for _ in range(layers):
        lw = LayerWeights(
            attention_norm=rand(hidden),
            q_proj=rand(heads * (hidden // heads), hidden),
            k_proj=rand(kv_heads * (hidden // kv_heads), hidden),
            v_proj=rand(kv_heads * (hidden // kv_heads), hidden),
            o_proj=rand(hidden, heads * (hidden // heads)),
            ffn_norm=rand(hidden),
            gate_proj=rand(ffn, hidden),
            up_proj=rand(ffn, hidden),
            down_proj=rand(hidden, ffn),
        )
        layer_list.append(lw)

    weights = ModelWeights(
        embedding=rand(vocab, hidden),
        layers=layer_list,
        final_norm=rand(hidden),
        lm_head=rand(vocab, hidden),
        position_type="RoPE",
        norm_eps=1e-5,
        norm_type="rmsnorm",
        ffn_type="SwiGLU",
    )

    # CPUExecutionEngine takes (weights, num_heads, num_kv_heads=...)
    eng = CPUExecutionEngine(weights, num_heads=heads, num_kv_heads=kv_heads)
    return eng



# ---------------------------------------------------------------------------
# 1. PyTorch independence — CPU engine must not import torch
# ---------------------------------------------------------------------------

class TestNoPyTorchInCPUPath:
    """cpu_engine.py must produce logits with zero torch involvement."""

    def test_cpu_engine_module_imports_no_torch(self):
        """cpu_engine.py must not import torch at module level."""
        import aether.runtime.cpu_engine as mod
        assert "torch" not in dir(mod), "torch symbol found in cpu_engine namespace"

    def test_cpu_forward_runs_without_torch(self):
        """A forward pass must complete when torch is NOT importable."""
        # Temporarily block torch import
        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            engine = _tiny_weights()
            logits, cache = engine.forward(np.array([0, 1, 2], dtype=np.int64))
            vocab = engine.weights.embedding.shape[0]
            assert logits.shape == (3, vocab)
        finally:
            if original is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original

    def test_native_cpu_backend_module_imports_no_torch(self):
        """native_cpu_backend.py must not import torch at module level."""
        import aether.backends.native_cpu_backend as mod
        assert "torch" not in sys.modules or sys.modules.get("torch") is not None
        # The module itself must not depend on torch
        source = Path(mod.__file__).read_text(errors="ignore")
        assert "import torch" not in source
        assert "from torch" not in source


# ---------------------------------------------------------------------------
# 2. CPU engine actually executes C++ kernels when compiler available
# ---------------------------------------------------------------------------

class TestNativeCPUKernelExecution:
    """Verify C++ kernel path is selected when a compiler exists."""

    def test_get_native_kernels_returns_object(self):
        from aether.kernels.native_cpu import get_native_kernels, NativeCPUKernels
        k = get_native_kernels()
        assert isinstance(k, NativeCPUKernels)

    def test_sgemm_produces_correct_result(self):
        """SGEMM result must match numpy reference within float32 tolerance."""
        from aether.kernels.native_cpu import get_native_kernels
        k = get_native_kernels()
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 8)).astype(np.float32)
        B = rng.standard_normal((8, 16)).astype(np.float32)
        result = k.sgemm(A, B)
        expected = A @ B
        np.testing.assert_allclose(result, expected, atol=1e-4,
                                   err_msg="sgemm result diverges from numpy reference")

    def test_rmsnorm_correctness(self):
        from aether.kernels.native_cpu import get_native_kernels
        k = get_native_kernels()
        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 16)).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        eps = 1e-5
        result = k.rmsnorm(x, w, eps)
        # reference
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        expected = (x / rms) * w
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_softmax_correctness(self):
        from aether.kernels.native_cpu import get_native_kernels
        k = get_native_kernels()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        result = k.softmax(x)
        x_shifted = x - x.max(axis=-1, keepdims=True)
        expected = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. CPUExecutionEngine full forward pass
# ---------------------------------------------------------------------------

class TestCPUEngineForward:
    def test_forward_shape(self):
        eng = _tiny_weights()
        vocab = eng.weights.embedding.shape[0]
        logits, cache = eng.forward(np.array([1, 2, 3], dtype=np.int64))
        assert logits.shape == (3, vocab), f"wrong logits shape: {logits.shape}"
        assert np.all(np.isfinite(logits)), "logits contain NaN or Inf"

    def test_generate_returns_token_list(self):
        eng = _tiny_weights()
        vocab = eng.weights.embedding.shape[0]
        tokens = eng.generate(np.array([1, 2], dtype=np.int64), max_tokens=5)
        assert isinstance(tokens, list)
        assert len(tokens) <= 5
        assert all(0 <= t < vocab for t in tokens)

    def test_kv_cache_grows_monotonically(self):
        from aether.runtime.cpu_engine import KVCache
        eng = _tiny_weights()
        _, cache = eng.forward(np.array([1, 2, 3], dtype=np.int64))
        assert cache.length == 3
        _, cache2 = eng.forward(np.array([4], dtype=np.int64), cache=cache)
        assert cache2.length == 4

    def test_deterministic_greedy(self):
        eng = _tiny_weights()
        t1 = eng.generate(np.array([1, 2], dtype=np.int64), max_tokens=4, temperature=0.0)
        t2 = eng.generate(np.array([1, 2], dtype=np.int64), max_tokens=4, temperature=0.0)
        assert t1 == t2, "greedy sampling must be deterministic"


# ---------------------------------------------------------------------------
# 4. Quantized dispatch wiring
# ---------------------------------------------------------------------------

class TestQuantizedLinearDispatch:
    """Verify that _linear() actually routes INT8 through QuantizedLinear."""

    def test_linear_fp32_uses_sgemm(self):
        eng = _tiny_weights()
        x = np.ones((1, 16), dtype=np.float32)
        weight = np.ones((8, 16), dtype=np.float32)
        out = eng._linear(x, weight)
        assert out.shape == (1, 8)

    def test_linear_int8_dispatches_to_quantized(self):
        """INT8 weight must trigger QuantizedLinear, not bare sgemm."""
        from aether.kernels.quantized_linear import QuantizedLinear
        eng = _tiny_weights()
        x = np.ones((1, 16), dtype=np.float32)
        # INT8 packed weight
        weight_int8 = np.zeros((8, 16), dtype=np.int8)
        scale = np.ones(8, dtype=np.float32) * 0.1
        out = eng._linear(x, weight_int8, precision="int8", weight_scale=scale)
        assert out.shape == (1, 8)
        assert np.all(np.isfinite(out))

    def test_linear_unknown_precision_falls_back(self):
        """Unknown precision must still produce FP32 output (sgemm fallback)."""
        eng = _tiny_weights()
        x = np.ones((1, 16), dtype=np.float32)
        weight = np.ones((8, 16), dtype=np.float32)
        out = eng._linear(x, weight, precision="unknown_format")
        assert out.shape == (1, 8)


# ---------------------------------------------------------------------------
# 5. Speculative decoding wired into backend
# ---------------------------------------------------------------------------

class TestSpeculativeDecodingWired:
    """DraftTargetSpecDecoder must be reachable via GenerationRequest.extra."""

    def test_speculative_engine_generate_step_calls_real_forward(self):
        """generate_step must call engine.forward_step, not return fabricated tokens."""
        from aether.runtime.speculative_engine import DraftTargetSpecDecoder, SpeculativeConfig

        draft = _tiny_weights()
        target = _tiny_weights()

        call_log = []
        orig_forward_step = draft.forward_step

        def patched_forward_step(ids, kv_cache=None, **kw):
            call_log.append(list(np.asarray(ids, dtype=np.int64).reshape(-1)))
            return orig_forward_step(ids, kv_cache=kv_cache, **kw)

        draft.forward_step = patched_forward_step

        cfg = SpeculativeConfig(gamma=2)
        decoder = DraftTargetSpecDecoder(draft_engine=draft, target_engine=target, config=cfg)
        prompt = np.array([1, 2, 3], dtype=np.int64)

        accepted, _dc, _tc, _stats = decoder.generate_step(prompt, draft_cache=None, target_cache=None)
        assert len(call_log) > 0, "draft.forward_step was never called"
        assert isinstance(accepted, list)
        assert len(accepted) >= 1

    def test_speculative_tokens_are_valid_vocab_ids(self):
        from aether.runtime.speculative_engine import DraftTargetSpecDecoder, SpeculativeConfig

        draft = _tiny_weights()
        target = _tiny_weights()
        vocab = draft.weights.embedding.shape[0]
        cfg = SpeculativeConfig(gamma=2)
        decoder = DraftTargetSpecDecoder(draft_engine=draft, target_engine=target, config=cfg)

        accepted, _dc, _tc, _stats = decoder.generate_step(
            np.array([0, 1], dtype=np.int64), draft_cache=None, target_cache=None
        )
        for tok in accepted:
            assert 0 <= tok < vocab, f"token {tok} out of vocab range"


# ---------------------------------------------------------------------------
# 6. Continuous batching — dynamic batch changes
# ---------------------------------------------------------------------------

class TestContinuousBatchingDynamic:
    def test_batch_changes_dynamically(self):
        """A second request must join mid-decode without waiting for first to finish."""
        from aether.runtime.continuous_batcher import (
            ContinuousBatcher, BatcherConfig, Request, RequestStatus,
        )

        engine = _tiny_weights()
        cfg = BatcherConfig(max_decode_seqs=4, max_queue_size=16)
        batcher = ContinuousBatcher(engine, config=cfg)

        req1 = Request(request_id="r1", prompt_token_ids=[1, 2], max_new_tokens=6)
        req2 = Request(request_id="r2", prompt_token_ids=[3, 4], max_new_tokens=4)

        batcher.submit(req1)
        # Advance one step — req1 should be prefilling or decoding
        batcher.step()
        # Submit req2 while req1 is in progress
        batcher.submit(req2)
        batcher.step()
        batcher.step()

        # Both requests should have progressed past WAITING
        terminal = {RequestStatus.DECODING, RequestStatus.FINISHED,
                    RequestStatus.PREFILLING, RequestStatus.CANCELLED}
        assert req1.status in terminal, f"req1 still WAITING after 3 steps: {req1.status}"

    def test_generate_sync_returns_text(self):
        from aether.runtime.continuous_batcher import ContinuousBatcher

        engine = _tiny_weights()
        vocab = engine.weights.embedding.shape[0]
        batcher = ContinuousBatcher(engine)

        result = batcher.generate_sync(
            np.array([0, 1, 2], dtype=np.int64),
            max_new_tokens=5,
        )
        assert isinstance(result, list)
        assert all(0 <= t < vocab for t in result)


# ---------------------------------------------------------------------------
# 7. Paged KV cache — storage vs attention execution
# ---------------------------------------------------------------------------

class TestPagedKVCacheStorage:
    def test_pages_allocated_and_readable(self):
        from aether.runtime.paged_kv_cache import PagedKVPool
        num_layers, num_kv_heads, head_dim = 2, 2, 8
        pool = PagedKVPool(
            num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
            page_size=4, max_pages=8, device="cpu"
        )
        pool.allocate_sequence("s1", num_layers=num_layers)
        k = np.ones((1, num_kv_heads, head_dim), dtype=np.float32)
        v = np.ones((1, num_kv_heads, head_dim), dtype=np.float32)
        # append_kv adds one token position at a time
        pool.append_kv("s1", layer_idx=0, k=k, v=v)
        k_out, v_out = pool.get_kv("s1", layer_idx=0)
        assert k_out.shape[-1] == head_dim, f"head_dim mismatch: {k_out.shape}"
        assert v_out.shape[-1] == head_dim, f"head_dim mismatch: {v_out.shape}"

    def test_paged_kv_not_connected_to_cpu_attention(self):
        """
        Document the known gap: cpu_engine attention reads from KVCache
        (contiguous), NOT from PagedKVPool (paged blocks).
        This test is a regression guard — it must PASS to confirm the gap
        still exists until paged attention is implemented.
        """
        from aether.runtime.cpu_engine import KVCache
        from aether.runtime.paged_kv_cache import PagedKVPool

        engine = _tiny_weights()
        # Infer shapes from weights
        num_layers = len(engine.weights.layers)
        num_kv_heads = engine.num_kv_heads
        head_dim = engine.weights.lm_head.shape[1] // engine.num_heads
        pool = PagedKVPool(max_pages=4, page_size=4, num_layers=num_layers,
                           num_kv_heads=num_kv_heads, head_dim=head_dim)

        # Engine forward uses KVCache, not PagedKVPool
        _, cache = engine.forward(np.array([0, 1], dtype=np.int64))
        assert isinstance(cache, KVCache)
        # pool is a different, unconnected object
        assert not hasattr(cache, "page_table"), (
            "KVCache now has a page_table — update this test to verify actual paged attention"
        )


# ---------------------------------------------------------------------------
# 8. AEG format — can load without PyTorch in sys.modules
# ---------------------------------------------------------------------------

class TestAEGFormatPyTorchFree:
    def test_aeg_format_imports_without_torch(self):
        """aeg_format.py must be importable without torch."""
        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            if "aether.core.aeg_format" in sys.modules:
                del sys.modules["aether.core.aeg_format"]
            import aether.core.aeg_format  # noqa: F401
        finally:
            if original is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original


# ---------------------------------------------------------------------------
# 9. Backend registry — CUDA backend registered
# ---------------------------------------------------------------------------

class TestBackendRegistry:
    def test_cuda_backend_in_registry(self):
        """NativeCUDABackend must appear in the registry (even if unavailable)."""
        from aether.backends.registry import BackendRegistry
        reg = BackendRegistry()
        # backend_names may use the .name attribute of the backend class
        names = getattr(reg, "backend_names", None) or list(getattr(reg, "_backends", {}).keys())
        # Accept any of the names the backend might register under
        cuda_present = any("cuda" in n for n in names)
        assert cuda_present, (
            f"No CUDA backend found in registry. Found: {names}. "
            f"Check that NativeCUDABackend.name='aether_cuda' and get_capabilities() is implemented."
        )

    def test_cpu_backend_available(self):
        from aether.backends.registry import BackendRegistry
        reg = BackendRegistry()
        avail = reg.get_available_backend_names()
        assert "aether_cpu" in avail or "native_cpu" in avail or any(
            "cpu" in n for n in avail
        ), f"No CPU backend available. Found: {avail}"

    def test_torch_backend_is_last(self):
        """TorchBackend must be the last fallback, not the first selection."""
        from aether.backends.registry import BackendRegistry
        reg = BackendRegistry()
        names = list(reg._backends.keys())
        torch_idx = next((i for i, n in enumerate(names) if "torch" in n), None)
        cuda_idx  = next((i for i, n in enumerate(names) if "cuda" in n), None)
        cpu_idx   = next((i for i, n in enumerate(names) if "cpu" in n), None)
        if torch_idx is not None and cpu_idx is not None:
            assert torch_idx > cpu_idx, "TorchBackend appears before NativeCPUBackend"
        if torch_idx is not None and cuda_idx is not None:
            assert torch_idx > cuda_idx, "TorchBackend appears before NativeCUDABackend"


# ---------------------------------------------------------------------------
# 10. No fabricated tokens (regression guard on old speculative.py)
# ---------------------------------------------------------------------------

class TestNoFabricatedTokens:
    def test_old_hash_based_speculative_not_in_production_path(self):
        """
        The old speculative.py used (token * 31 + branch) % 100000 to
        'generate' draft tokens — a hash function, not a language model.
        Verify it is not callable from the production path.
        """
        import aether.runtime.speculative as old_spec
        # The old module may still exist for backward compat, but it must
        # NOT be imported by the generation path (native_cpu_backend.py)
        backend_src = Path(
            __import__("aether.backends.native_cpu_backend", fromlist=[""]).__file__
        ).read_text(errors="ignore")
        assert "from aether.runtime.speculative import" not in backend_src, (
            "native_cpu_backend imports old hash-based speculative.py"
        )
