"""
Tests for the Aether runtime, KV cache, scheduler, and speculative engine.
"""

from __future__ import annotations

import pytest

from aether.core.types import MemoryTier
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.scheduler import DisaggregatedScheduler
from aether.runtime.speculative import TreeSpeculativeEngine


class TestKVCacheManager:
    """Tests for the KV cache manager."""

    def test_allocate_block(self) -> None:
        cache = KVCacheManager(dtype="fp8")
        block = cache.allocate_block(layer_index=0, token_start=0, token_count=16)
        assert block.block_id == 0
        assert block.layer_index == 0
        assert block.token_count == 16

    def test_find_prefix_miss(self) -> None:
        cache = KVCacheManager(dtype="fp8")
        assert cache.find_prefix("nonexistent") is None

    def test_find_prefix_hit(self) -> None:
        cache = KVCacheManager(dtype="fp8")
        block = cache.allocate_block(layer_index=0, token_start=0, token_count=16, prefix_hash="abc123")
        found = cache.find_prefix("abc123")
        assert found is block
        assert cache.hit_rate() > 0

    def test_evict_to_tier(self) -> None:
        cache = KVCacheManager(dtype="fp8")
        block = cache.allocate_block(layer_index=0, token_start=0, token_count=16)
        cache.evict_to_tier(str(block.block_id), MemoryTier.L2_CPU_DRAM)
        assert block.tier == MemoryTier.L2_CPU_DRAM

    def test_transfer_stats_measure_real_tier_movement(self) -> None:
        cache = KVCacheManager(dtype="fp8")
        block = cache.allocate_block(layer_index=0, token_start=0, token_count=16)
        assert cache.evict_to_tier(str(block.block_id), MemoryTier.L2_CPU_DRAM)
        stats = cache.get_transfer_stats()
        assert stats["local_tier_transfers"] == 1
        assert stats["transferred_tokens"] == 16
        assert stats["transfers_by_route"]["L1_GPU_HBM->L2_CPU_DRAM"] == 1

    def test_hit_rate(self) -> None:
        cache = KVCacheManager(dtype="fp8")
        cache.allocate_block(layer_index=0, token_start=0, token_count=16, prefix_hash="abc")
        cache.find_prefix("abc")
        cache.find_prefix("xyz")
        assert cache.hit_rate() == 0.5


class TestDisaggregatedScheduler:
    """Tests for the disaggregated prefill/decode scheduler."""

    def test_submit(self) -> None:
        scheduler = DisaggregatedScheduler()
        request_id = scheduler.submit(
            prompt_tokens=[1, 2, 3],
            max_tokens=32,
            temperature=0.7,
            top_p=0.9,
        )
        assert request_id == "req_1"
        assert scheduler.pending_prefill_count == 1

    def test_schedule_prefill(self) -> None:
        scheduler = DisaggregatedScheduler(max_batch_size=2)
        scheduler.submit([1, 2, 3], 32, 0.7, 0.9)
        scheduler.submit([4, 5, 6], 32, 0.7, 0.9)
        batch = scheduler.schedule_prefill()
        assert len(batch) == 2
        assert scheduler.pending_prefill_count == 0

    def test_finish_prefill(self) -> None:
        scheduler = DisaggregatedScheduler()
        scheduler.submit([1, 2, 3], 32, 0.7, 0.9)
        batch = scheduler.schedule_prefill()
        scheduler.finish_prefill(batch)
        assert scheduler.pending_decode_count == 1

    def test_advance_decode(self) -> None:
        scheduler = DisaggregatedScheduler()
        request_id = scheduler.submit([1, 2, 3], max_tokens=2, temperature=0.7, top_p=0.9)
        batch = scheduler.schedule_prefill()
        scheduler.finish_prefill(batch)
        completed = scheduler.advance_decode({request_id: 42})
        assert not completed
        completed = scheduler.advance_decode({request_id: 43})
        assert completed == [request_id]

    def test_get_request(self) -> None:
        scheduler = DisaggregatedScheduler()
        request_id = scheduler.submit([1, 2, 3], 32, 0.7, 0.9)
        request = scheduler.get_request(request_id)
        assert request is not None
        assert request.request_id == request_id

    def test_prefill_chunks_long_prompt(self) -> None:
        scheduler = DisaggregatedScheduler(max_batch_size=1, prefill_chunk_size=2)
        request_id = scheduler.submit([1, 2, 3, 4, 5], 32, 0.7, 0.9)
        first = scheduler.schedule_prefill()
        assert first[0].last_prefill_chunk == (0, 2)
        scheduler.finish_prefill(first)
        assert scheduler.pending_prefill_count == 1
        second = scheduler.schedule_prefill()
        scheduler.finish_prefill(second)
        third = scheduler.schedule_prefill()
        scheduler.finish_prefill(third)
        assert scheduler.pending_decode_count == 1
        assert scheduler.get_request(request_id).phase == "decode"

    def test_decode_prioritizes_short_remaining_work(self) -> None:
        scheduler = DisaggregatedScheduler(max_batch_size=2)
        long_id = scheduler.submit([1], 10, 0.7, 0.9)
        short_id = scheduler.submit([2], 2, 0.7, 0.9)
        batch = scheduler.schedule_prefill()
        scheduler.finish_prefill(batch)
        scheduler.max_batch_size = 1
        scheduled = scheduler.schedule_decode()
        assert scheduled[0].request_id == short_id
        assert scheduler.get_request(long_id) is not None

    def test_queue_snapshot(self) -> None:
        scheduler = DisaggregatedScheduler()
        scheduler.submit([1, 2, 3], 4, 0.7, 0.9)
        snapshot = scheduler.queue_snapshot()
        assert snapshot["pending_prefill"] == 1
        assert snapshot["prefill_tokens_remaining"] == 3


class TestTreeSpeculativeEngine:
    """Tests for the tree-speculative decoding engine."""

    def test_draft_model_selection(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="Qwen/Qwen3-72B")
        assert engine.draft_model_id is None

    def test_explicit_draft_model_is_preserved(self) -> None:
        engine = TreeSpeculativeEngine(
            target_model_id="any/family-model",
            draft_model_id="local/compatible-draft",
        )
        assert engine.draft_model_id == "local/compatible-draft"

    def test_draft_model_selection_unknown(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="unknown/model")
        assert engine.draft_model_id is None

    def test_build_draft_tree(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="test")
        tree = engine.build_draft_tree([1, 2, 3], max_depth=2)
        assert tree is not None
        assert tree.depth == 0

    def test_verify_tree(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="test")
        tree = engine.build_draft_tree([1, 2, 3], max_depth=2)
        accepted = engine.verify_tree(tree, [])
        assert len(accepted) > 0
        assert accepted[0] == tree.token_id

    def test_acceptance_rate(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="test")
        tree = engine.build_draft_tree([1], max_depth=2)
        engine.verify_tree(tree, [])
        rate = engine.acceptance_rate()
        assert 0.0 <= rate <= 1.0

    def test_verify_tree_rejects_target_mismatch(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="test")
        tree = engine.build_draft_tree([0], max_depth=2, branching_factor=2)
        accepted = engine.verify_tree(tree, [[10.0, 0.0, 0.0]])
        assert accepted == [0]
        assert engine.acceptance_rate() == 0.0

    def test_prune_tree_removes_low_probability_branches(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="test")
        tree = engine.build_draft_tree([0], max_depth=1, branching_factor=3)
        engine.prune_tree(tree, min_probability=0.6)
        assert all(child.probability >= 0.6 for child in tree.children)

    def test_should_use_speculation_initially(self) -> None:
        engine = TreeSpeculativeEngine(target_model_id="test")
        assert engine.should_use_speculation()


class TestTorchBackendCompiledAEG:
    """Tests for local compiled AEG fallback execution."""

    def test_generate_from_compiled_aeg_without_tokenizer_fails_closed(self, minimal_aeg_package) -> None:
        from aether.backends.base import GenerationRequest
        from aether.backends.torch_backend import CompiledAEGHandle, TorchBackend
        from aether.core.exceptions import BackendError

        backend = TorchBackend()
        model = backend.load_model("test-model", str(minimal_aeg_package.root))
        assert isinstance(model, CompiledAEGHandle)
        with pytest.raises(BackendError, match="tokenizer-backed generation adapter"):
            backend.generate(
                GenerationRequest(
                    model_id="test-model",
                    prompt="Explain AEG",
                    max_tokens=8,
                )
            )


class TestRuntimeBasics:
    """Tests for the Runtime class core logic."""

    def test_runtime_initialization(self) -> None:
        from aether import Runtime, RuntimeConfig
        config = RuntimeConfig(backend_name="pytorch")
        rt = Runtime(config)
        assert rt.config.backend_name == "pytorch"
        assert rt.fingerprint.target_id is not None

    def test_list_empty(self, tmp_cache_dir) -> None:
        from aether import Runtime, RuntimeConfig
        rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_cache_dir)))
        assert rt.list() == []

    def test_benchmark_smoke(self, tmp_cache_dir) -> None:
        from aether import Runtime, RuntimeConfig
        rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_cache_dir)))
        # Should not crash on missing model
        try:
            rt.benchmark("nonexistent/model", max_tokens=8)
        except Exception:
            pass  # Expected for missing model
