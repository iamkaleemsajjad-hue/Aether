"""Tests for the kernels package."""

from __future__ import annotations

import numpy as np

from aether.kernels import (
    AttentionKernel,
    FFNKernel,
    GEMMKernel,
    LayerNormKernel,
    RMSNormKernel,
    RoPEKernel,
)


class TestAttentionKernel:
    def test_forward(self) -> None:
        kernel = AttentionKernel(head_dim=64)
        q = np.random.randn(1, 4, 8, 64).astype(np.float32)
        k = np.random.randn(1, 4, 8, 64).astype(np.float32)
        v = np.random.randn(1, 4, 8, 64).astype(np.float32)
        out = kernel.forward(q, k, v)
        assert out.shape == (1, 4, 8, 64)

    def test_forward_with_mask(self) -> None:
        kernel = AttentionKernel(head_dim=64)
        q = np.random.randn(1, 2, 4, 64).astype(np.float32)
        k = np.random.randn(1, 2, 4, 64).astype(np.float32)
        v = np.random.randn(1, 2, 4, 64).astype(np.float32)
        mask = np.zeros((4, 4), dtype=np.float32)
        out = kernel.forward(q, k, v, mask)
        assert out.shape == (1, 2, 4, 64)


class TestGEMMKernel:
    def test_bf16_gemm(self) -> None:
        kernel = GEMMKernel()
        a = np.random.randn(4, 64).astype(np.float32)
        b = np.random.randn(8, 64).astype(np.float32)
        out = kernel.forward(a, b, dtype="bf16")
        assert out.shape == (4, 8)


class TestFFNKernel:
    def test_swiglu(self) -> None:
        kernel = FFNKernel(activation="swiglu")
        x = np.random.randn(2, 64).astype(np.float32)
        out = kernel.forward(x)
        assert out.shape == (2, 64)

    def test_gelu(self) -> None:
        kernel = FFNKernel(activation="gelu")
        x = np.random.randn(2, 64).astype(np.float32)
        out = kernel.forward(x)
        assert out.shape == (2, 64)


class TestRoPEKernel:
    def test_compute_freqs(self) -> None:
        kernel = RoPEKernel(theta=10000.0)
        freqs = kernel.compute_freqs(head_dim=64, seq_len=10)
        assert freqs.shape == (10, 32)

    def test_apply(self) -> None:
        kernel = RoPEKernel()
        freqs = kernel.compute_freqs(head_dim=64, seq_len=4)
        x = np.random.randn(1, 4, 4, 64).astype(np.float32)
        rotated = kernel.apply(x, freqs)
        assert rotated.shape == x.shape

    def test_batch_apply(self) -> None:
        kernel = RoPEKernel()
        q = np.random.randn(1, 4, 8, 64).astype(np.float32)
        k = np.random.randn(1, 4, 8, 64).astype(np.float32)
        qr, kr = kernel.batch_apply(q, k, seq_len=8, head_dim=64)
        assert qr.shape == q.shape
        assert kr.shape == k.shape


class TestNormKernels:
    def test_rms_norm(self) -> None:
        kernel = RMSNormKernel(eps=1e-6)
        x = np.random.randn(2, 64).astype(np.float32)
        out = kernel.forward(x)
        assert out.shape == (2, 64)

    def test_layer_norm(self) -> None:
        kernel = LayerNormKernel(eps=1e-5)
        x = np.random.randn(2, 64).astype(np.float32)
        out = kernel.forward(x)
        assert out.shape == (2, 64)
