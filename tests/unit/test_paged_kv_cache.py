"""
Tests for aether.runtime.paged_kv_cache — PagedKVPool

Verifies:
- Pool allocates actual tensors (not metadata-only)
- BlockAllocator reference counting
- KV write/read roundtrip
- Prefix sharing (attach_prefix, insert_prefix)
- CPU offload path (numpy mirror)
- Pool exhaustion raises RuntimeError
- Stats reporting
- Multi-sequence isolation
"""

from __future__ import annotations

import pytest
import numpy as np

from aether.runtime.paged_kv_cache import (
    BlockAllocator,
    KVPageRef,
    PagedKVPool,
    PrefixHash,
)


# ---------------------------------------------------------------------------
# BlockAllocator
# ---------------------------------------------------------------------------

class TestBlockAllocator:
    def test_basic_alloc_free(self):
        alloc = BlockAllocator(10)
        pages = alloc.allocate(3)
        assert len(pages) == 3
        assert alloc.free_count == 7
        alloc.free(pages)
        assert alloc.free_count == 10

    def test_exhaustion(self):
        alloc = BlockAllocator(2)
        alloc.allocate(2)
        with pytest.raises(RuntimeError, match="exhausted"):
            alloc.allocate(1)

    def test_ref_count_retain(self):
        alloc = BlockAllocator(4)
        pages = alloc.allocate(1)
        alloc.retain(pages[0])
        # Now ref_count = 2; free once → not returned to pool
        alloc.free(pages)
        assert alloc.free_count == 3  # only 3 returned, page still held
        alloc.free(pages)  # now ref_count = 0
        assert alloc.free_count == 4

    def test_stats_tracking(self):
        alloc = BlockAllocator(5)
        pages = alloc.allocate(3)
        assert alloc.allocations == 3
        alloc.free([pages[0]])   # free one of the allocated pages
        assert alloc.frees == 1


# ---------------------------------------------------------------------------
# PrefixHash
# ---------------------------------------------------------------------------

class TestPrefixHash:
    def test_deterministic(self):
        toks = [1, 2, 3, 4, 5]
        assert PrefixHash.of(toks) == PrefixHash.of(toks)

    def test_different_inputs_different_hashes(self):
        assert PrefixHash.of([1, 2, 3]) != PrefixHash.of([1, 2, 4])

    def test_empty(self):
        h = PrefixHash.of([])
        assert isinstance(h, str) and len(h) == 32


# ---------------------------------------------------------------------------
# PagedKVPool — basic lifecycle
# ---------------------------------------------------------------------------

POOL_KWARGS = dict(
    num_layers=2,
    num_kv_heads=2,
    head_dim=4,
    page_size=4,
    max_pages=16,
    dtype="float32",
    device="cpu",
)


class TestPagedKVPoolLifecycle:
    def test_pool_is_allocated(self):
        pool = PagedKVPool(**POOL_KWARGS)
        assert pool.pool_bytes() > 0
        assert pool._np_pool.shape == (16, 4, 2, 2, 2, 4)

    def test_allocate_free_sequence(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("seq0")
        assert pool._allocator.used_count == 0  # no pages yet
        pool.free_sequence("seq0")

    def test_stats_empty(self):
        pool = PagedKVPool(**POOL_KWARGS)
        s = pool.stats()
        assert s.total_pages == 16
        assert s.free_pages == 16
        assert s.used_pages == 0


class TestPagedKVPoolWriteRead:
    def test_write_single_token(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("s1")
        k = np.ones((2, 4), dtype=np.float32)  # (num_kv_heads, head_dim)
        v = np.ones((2, 4), dtype=np.float32) * 2.0
        pool.append_kv("s1", layer_idx=0, k=k, v=v)
        # One page should be allocated
        assert pool._allocator.used_count == 1
        k_out, v_out = pool.get_kv("s1", layer_idx=0, as_torch=False)
        assert k_out.shape == (1, 2, 4)
        assert v_out.shape == (1, 2, 4)
        np.testing.assert_allclose(k_out[0], k, atol=1e-5)
        np.testing.assert_allclose(v_out[0], v, atol=1e-5)

    def test_write_multiple_tokens_spans_pages(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("s2")
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        # page_size=4, write 6 tokens → should use 2 pages
        for i in range(6):
            pool.append_kv("s2", layer_idx=0, k=k * i, v=v * i)
        assert pool._allocator.used_count == 2
        k_out, v_out = pool.get_kv("s2", layer_idx=0, as_torch=False)
        assert k_out.shape == (6, 2, 4)
        np.testing.assert_allclose(k_out[0], np.zeros((2, 4)), atol=1e-5)
        np.testing.assert_allclose(k_out[5], np.ones((2, 4)) * 5, atol=1e-5)

    def test_get_total_tokens(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("s3")
        k = np.ones((2, 4), dtype=np.float32)
        for _ in range(3):
            pool.append_kv("s3", layer_idx=0, k=k, v=k)
        assert pool.get_total_tokens("s3", layer_idx=0) == 3

    def test_two_layers_independent(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("s4")
        k0 = np.ones((2, 4), dtype=np.float32)
        k1 = np.ones((2, 4), dtype=np.float32) * 99.0
        pool.append_kv("s4", layer_idx=0, k=k0, v=k0)
        pool.append_kv("s4", layer_idx=1, k=k1, v=k1)
        k_l0, _ = pool.get_kv("s4", 0, as_torch=False)
        k_l1, _ = pool.get_kv("s4", 1, as_torch=False)
        np.testing.assert_allclose(k_l0[0], k0, atol=1e-5)
        np.testing.assert_allclose(k_l1[0], k1, atol=1e-5)

    def test_two_sequences_isolated(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("a")
        pool.allocate_sequence("b")
        ka = np.ones((2, 4), dtype=np.float32)
        kb = np.ones((2, 4), dtype=np.float32) * 5.0
        pool.append_kv("a", layer_idx=0, k=ka, v=ka)
        pool.append_kv("b", layer_idx=0, k=kb, v=kb)
        k_a, _ = pool.get_kv("a", 0, as_torch=False)
        k_b, _ = pool.get_kv("b", 0, as_torch=False)
        np.testing.assert_allclose(k_a[0], ka, atol=1e-5)
        np.testing.assert_allclose(k_b[0], kb, atol=1e-5)


class TestPagedKVPoolPrefixSharing:
    def test_insert_and_lookup_prefix(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("base")
        k = np.ones((2, 4), dtype=np.float32)
        token_ids = [1, 2, 3, 4]
        for _ in token_ids:
            pool.append_kv("base", layer_idx=0, k=k, v=k)
        phash = pool.insert_prefix(token_ids, "base", layer_idx=0)
        assert isinstance(phash, str)

        # Exact lookup
        found_hash, matched = pool.lookup_prefix(token_ids)
        assert found_hash == phash
        assert matched == len(token_ids)

    def test_prefix_miss(self):
        pool = PagedKVPool(**POOL_KWARGS)
        phash, matched = pool.lookup_prefix([99, 100, 101])
        assert phash is None
        assert matched == 0

    def test_attach_prefix_increases_token_count(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("src")
        k = np.ones((2, 4), dtype=np.float32)
        token_ids = [1, 2]
        for _ in token_ids:
            pool.append_kv("src", 0, k=k, v=k)
        phash = pool.insert_prefix(token_ids, "src", 0)

        pool.allocate_sequence("dst")
        n = pool.attach_prefix("dst", phash, 0)
        assert n == len(token_ids)


class TestPagedKVPoolExhaustion:
    def test_exhaustion_raises_on_alloc(self):
        # tiny pool: 2 pages, page_size=1 → each token needs a page
        pool = PagedKVPool(
            num_layers=1, num_kv_heads=1, head_dim=2,
            page_size=1, max_pages=2, dtype="float32", device="cpu"
        )
        pool.allocate_sequence("s")
        k = np.ones((1, 2), dtype=np.float32)
        pool.append_kv("s", 0, k=k, v=k)
        pool.append_kv("s", 0, k=k, v=k)  # 2nd page
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.append_kv("s", 0, k=k, v=k)  # would need 3rd page


class TestPagedKVPoolStats:
    def test_stats_after_writes(self):
        pool = PagedKVPool(**POOL_KWARGS)
        pool.allocate_sequence("s")
        k = np.ones((2, 4), dtype=np.float32)
        pool.append_kv("s", 0, k=k, v=k)
        s = pool.stats()
        assert s.used_pages == 1
        assert s.free_pages == 15
        assert s.active_sequences == 1

    def test_repr(self):
        pool = PagedKVPool(**POOL_KWARGS)
        r = repr(pool)
        assert "PagedKVPool" in r
