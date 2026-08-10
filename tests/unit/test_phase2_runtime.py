"""
Tests for Phase 2 runtime components:
  - EAGLE-3 engine
  - Dynamic precision manager
  - Model registry
  - Attention kernels (FlashAttention-2, GQA, Sliding Window, Paged)
"""

from __future__ import annotations

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# EAGLE-3 Engine
# ---------------------------------------------------------------------------

class TestEAGLE3Planner:
    def test_plan_small_model(self):
        from aether.runtime.eagle import EAGLE3Planner

        class FakeArch:
            layers = 4
            context_length = 4096

        planner = EAGLE3Planner()
        plan = planner.plan(FakeArch())
        assert plan.tree_depth >= 3
        assert plan.branching_factor >= 2
        assert len(plan.fusion_layers) > 0

    def test_plan_large_model(self):
        from aether.runtime.eagle import EAGLE3Planner

        class FakeArch:
            layers = 80
            context_length = 131072

        plan = EAGLE3Planner().plan(FakeArch())
        assert plan.attention_drift_correction is True
        assert plan.tree_depth == 5

    def test_plan_to_dict(self):
        from aether.runtime.eagle import EAGLE3Planner

        class FakeArch:
            layers = 8
            context_length = 8192

        plan = EAGLE3Planner().plan(FakeArch())
        d = plan.to_dict()
        assert d["version"] == "eagle3/1.0"
        assert "fusion_layers" in d
        assert "tree_depth" in d

    def test_verify_acceptance_above_floor(self):
        from aether.runtime.eagle import EAGLE3Planner

        class FakeArch:
            layers = 4
            context_length = 4096

        plan = EAGLE3Planner().plan(FakeArch(), target_acceptance=0.8)
        result = EAGLE3Planner().verify_acceptance(8, 10, plan)
        assert "acceptance_rate" in result
        assert result["acceptance_rate"] == pytest.approx(0.8)

    def test_verify_acceptance_below_floor(self):
        from aether.runtime.eagle import EAGLE3Planner

        class FakeArch:
            layers = 4
            context_length = 4096

        plan = EAGLE3Planner().plan(FakeArch(), target_acceptance=0.9)
        result = EAGLE3Planner().verify_acceptance(2, 10, plan)
        assert result["use_speculation"] is False
        assert result["fallback"] == "standard_decode"


class TestEAGLE3Engine:
    def _make_engine(self, vocab_size: int = 512, layers: int = 4) -> "EAGLE3Engine":
        from aether.runtime.eagle import EAGLE3Plan, EAGLE3Engine

        plan = EAGLE3Plan(
            draft_model_id=None,
            fusion_layers=tuple(range(layers)),
            tree_depth=3,
            branching_factor=3,
            acceptance_floor=0.6,
        )
        projection = np.linspace(
            -0.25, 0.25, num=64 * vocab_size, dtype=np.float32
        ).reshape(64, vocab_size)
        return EAGLE3Engine(
            plan=plan,
            hidden_size=64,
            vocab_size=vocab_size,
            output_projection=projection,
        )

    @staticmethod
    def _hidden_states() -> list[np.ndarray]:
        return [
            np.linspace(-1.0, 1.0, num=64, dtype=np.float32),
            np.linspace(1.0, -1.0, num=64, dtype=np.float32),
        ]

    def test_build_draft_tree_basic(self):
        engine = self._make_engine()
        roots = engine.build_draft_tree([1, 2, 3], hidden_states=self._hidden_states())
        assert len(roots) > 0
        for root in roots:
            assert root.depth == 0
            assert root.token_id >= 0

    def test_draft_tree_depth(self):
        engine = self._make_engine()
        roots = engine.build_draft_tree([10], hidden_states=self._hidden_states())
        # BFS check for children at depth > 0
        has_children = any(len(r.children) > 0 for r in roots)
        assert has_children

    def test_path_to_root(self):
        engine = self._make_engine()
        roots = engine.build_draft_tree([5, 10, 15], hidden_states=self._hidden_states())
        for root in roots:
            path = root.path_to_root()
            assert path[-1] == root.token_id  # leaf ends with its own id

    def test_verify_returns_tokens(self):
        engine = self._make_engine()
        roots = engine.build_draft_tree([1, 2, 3], hidden_states=self._hidden_states())
        tokens, accepted, proposed = engine.verify(roots, temperature=1.0)
        assert isinstance(tokens, list)
        assert isinstance(accepted, int)
        assert isinstance(proposed, int)
        assert proposed > 0

    def test_verify_empty_roots(self):
        engine = self._make_engine()
        tokens, accepted, proposed = engine.verify([])
        assert tokens == []
        assert accepted == 0
        assert proposed == 0

    def test_acceptance_rate_tracking(self):
        engine = self._make_engine()
        for _ in range(5):
            roots = engine.build_draft_tree([42], hidden_states=self._hidden_states())
            engine.verify(roots, temperature=0.5)
        rate = engine.acceptance_rate()
        assert 0.0 <= rate <= 1.0

    def test_should_use_speculation_initial(self):
        engine = self._make_engine()
        # No acceptance evidence exists yet, so the engine must not claim that
        # speculation is safe before a verified draft step.
        assert engine.should_use_speculation() is False

    def test_stats_dict(self):
        engine = self._make_engine()
        stats = engine.stats()
        assert "acceptance_rate" in stats
        assert "plan" in stats

    def test_repr(self):
        engine = self._make_engine()
        r = repr(engine)
        assert "EAGLE3Engine" in r

    def test_feature_extrapolator_top_k(self):
        from aether.runtime.eagle import FeatureExtrapolator
        projection = np.eye(64, 256, dtype=np.float32)
        fe = FeatureExtrapolator(
            hidden_size=64,
            vocab_size=256,
            fusion_layers=(0, 1),
            output_projection=projection,
        )
        logits = fe.extrapolate([np.ones(64, dtype=np.float32)], last_token_id=7, temperature=1.0)
        assert logits.shape == (256,)
        indices, probs = fe.top_k_probs(logits, k=5)
        assert len(indices) == 5
        assert len(probs) == 5
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_missing_draft_projection_fails_closed(self):
        from aether.core.exceptions import RuntimeError as AetherRuntimeError
        from aether.runtime.eagle import FeatureExtrapolator

        fe = FeatureExtrapolator(hidden_size=4, vocab_size=8, fusion_layers=())
        with pytest.raises(AetherRuntimeError, match="refusing synthetic logits"):
            fe.extrapolate([np.ones(4, dtype=np.float32)], last_token_id=0)


# ---------------------------------------------------------------------------
# Dynamic Precision Manager
# ---------------------------------------------------------------------------

class TestDynamicPrecisionManager:
    def _make_mgr(self):
        from aether.runtime.precision_manager import DynamicPrecisionManager

        prec_map = {
            "embedding": "BF16",
            "layer_0_attn": "BF16",
            "layer_1_attn": "BF16",
            "layer_0_ffn": "Q4_K_M",
            "layer_1_ffn": "Q4_K_M",
            "lm_head": "BF16",
        }
        sensitivity = {
            "embedding": 1.0,
            "layer_0_attn": 0.8,
            "layer_1_attn": 0.8,
            "layer_0_ffn": 0.3,
            "layer_1_ffn": 0.3,
            "lm_head": 1.0,
        }
        return DynamicPrecisionManager(prec_map, sensitivity, max_ppl_delta=0.1)

    def test_initial_precision(self):
        mgr = self._make_mgr()
        assert mgr.get_active_precision("embedding") == "BF16"
        assert mgr.get_active_precision("layer_0_ffn") == "Q4_K_M"

    def test_unknown_layer_returns_bf16(self):
        mgr = self._make_mgr()
        assert mgr.get_active_precision("nonexistent") == "BF16"

    def test_downgrade_on_high_pressure(self):
        mgr = self._make_mgr()
        snap = mgr.update(memory_pressure=0.95)
        # Some layer should have been downgraded (or budget exhausted)
        assert isinstance(snap.downgrades, int)

    def test_locked_layers_not_downgraded(self):
        mgr = self._make_mgr()
        mgr.update(memory_pressure=0.99)
        # embedding and lm_head must remain BF16
        assert mgr.get_active_precision("embedding") == "BF16"
        assert mgr.get_active_precision("lm_head") == "BF16"

    def test_upgrade_on_low_pressure(self):
        mgr = self._make_mgr()
        mgr.update(memory_pressure=0.95)
        snap = mgr.update(memory_pressure=0.4)
        assert isinstance(snap.upgrades, int)

    def test_reset(self):
        mgr = self._make_mgr()
        mgr.update(memory_pressure=0.95)
        mgr.reset()
        assert mgr.get_active_precision("layer_0_attn") == "BF16"

    def test_estimated_ppl_delta_zero_initial(self):
        mgr = self._make_mgr()
        delta = mgr.estimated_ppl_delta()
        assert delta >= 0.0

    def test_active_map(self):
        mgr = self._make_mgr()
        amap = mgr.get_active_map()
        assert "embedding" in amap
        assert len(amap) == 6

    def test_stats_dict(self):
        mgr = self._make_mgr()
        stats = mgr.stats()
        assert "memory_pressure" in stats
        assert "estimated_ppl_delta" in stats
        assert "active_map" in stats

    def test_repr(self):
        mgr = self._make_mgr()
        assert "DynamicPrecisionManager" in repr(mgr)

    def test_load_from_precision_map(self):
        from aether.runtime.precision_manager import DynamicPrecisionManager
        mgr = DynamicPrecisionManager()
        mgr.load_from_precision_map({"layer_0": "BF16", "layer_1": "Q8_0"})
        assert mgr.get_active_precision("layer_0") == "BF16"


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_get(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry(max_loaded_models=4)
        handle = object()
        entry = reg.register("model/A", handle, "pytorch")
        assert entry.model_id == "model/A"
        assert reg.get("model/A") is not None
        assert reg.is_loaded("model/A") is True

    def test_require_missing_raises(self):
        from aether.runtime.model_registry import ModelRegistry
        from aether.core.exceptions import ModelNotFoundError

        reg = ModelRegistry()
        with pytest.raises(ModelNotFoundError):
            reg.require("missing/model")

    def test_unload(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        reg.register("A", object(), "pytorch")
        assert reg.is_loaded("A")
        removed = reg.unload("A")
        assert removed is True
        assert not reg.is_loaded("A")

    def test_unload_busy_model(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        reg.register("B", object(), "pytorch")
        reg.acquire("B")
        removed = reg.unload("B")
        assert removed is False  # still in use
        reg.release("B")
        removed = reg.unload("B")
        assert removed is True

    def test_lru_eviction(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry(max_loaded_models=2)
        reg.register("A", object(), "pytorch")
        time.sleep(0.01)
        reg.register("B", object(), "pytorch")
        # Register C should evict A (LRU)
        reg.register("C", object(), "pytorch")
        assert reg.is_loaded("C")
        # A should have been evicted
        assert not reg.is_loaded("A")

    def test_hot_reload(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        h1 = object()
        h2 = object()
        reg.register("M", h1, "pytorch")
        reg.register("M", h2, "llamacpp")
        entry = reg.require("M")
        assert entry.handle is h2
        assert entry.backend_name == "llamacpp"

    def test_list_models(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        reg.register("Z", object(), "p")
        reg.register("A", object(), "p")
        models = reg.list_models()
        assert models == sorted(models)

    def test_touch_increments_request_count(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        reg.register("X", object(), "pytorch")
        entry = reg.acquire("X")
        assert entry.request_count == 1
        reg.release("X")

    def test_total_request_count(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        reg.register("M1", object(), "p")
        reg.register("M2", object(), "p")
        reg.acquire("M1")
        reg.release("M1")
        reg.acquire("M2")
        reg.release("M2")
        assert reg.total_request_count() == 2

    def test_len(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry()
        assert len(reg) == 0
        reg.register("A", object(), "p")
        assert len(reg) == 1

    def test_repr(self):
        from aether.runtime.model_registry import ModelRegistry
        reg = ModelRegistry(max_loaded_models=8)
        assert "ModelRegistry" in repr(reg)


# ---------------------------------------------------------------------------
# Attention Kernels
# ---------------------------------------------------------------------------

class TestVanillaAttention:
    def test_output_shape(self):
        from aether.kernels.attention import VanillaAttention
        attn = VanillaAttention(head_dim=16)
        q = np.random.randn(2, 4, 8, 16).astype(np.float32)
        k = np.random.randn(2, 4, 8, 16).astype(np.float32)
        v = np.random.randn(2, 4, 8, 16).astype(np.float32)
        out = attn.forward(q, k, v, causal=True)
        assert out.shape == q.shape

    def test_causal_masking(self):
        """Future tokens should not attend to future tokens."""
        from aether.kernels.attention import VanillaAttention
        attn = VanillaAttention(head_dim=4)
        # Set keys/values such that only position 0 has a signal
        q = np.ones((1, 1, 3, 4), dtype=np.float32)
        k = np.zeros((1, 1, 3, 4), dtype=np.float32)
        k[0, 0, 0, :] = 10.0  # Strong signal only at position 0
        v = np.eye(3, 4, dtype=np.float32)[np.newaxis, np.newaxis]
        out = attn.forward(q, k, v, causal=True)
        assert out.shape == (1, 1, 3, 4)

    def test_no_nan(self):
        from aether.kernels.attention import VanillaAttention
        attn = VanillaAttention(head_dim=32)
        q = np.random.randn(1, 8, 16, 32).astype(np.float32)
        k = np.random.randn(1, 8, 16, 32).astype(np.float32)
        v = np.random.randn(1, 8, 16, 32).astype(np.float32)
        out = attn.forward(q, k, v)
        assert not np.isnan(out).any()


class TestFlashAttention2:
    def test_output_shape(self):
        from aether.kernels.attention import FlashAttention2
        fa = FlashAttention2(head_dim=16, block_size=8)
        q = np.random.randn(1, 2, 8, 16).astype(np.float32)
        k = np.random.randn(1, 2, 8, 16).astype(np.float32)
        v = np.random.randn(1, 2, 8, 16).astype(np.float32)
        out = fa._tiled_forward(q, k, v, mask=None, causal=True)
        assert out.shape == q.shape

    def test_matches_vanilla_noncausal(self):
        """FA-2 tiled output should closely match VanillaAttention (non-causal)."""
        from aether.kernels.attention import FlashAttention2, VanillaAttention
        np.random.seed(42)
        q = np.random.randn(1, 2, 4, 8).astype(np.float32)
        k = np.random.randn(1, 2, 4, 8).astype(np.float32)
        v = np.random.randn(1, 2, 4, 8).astype(np.float32)
        fa = FlashAttention2(head_dim=8, block_size=4)
        vanilla = VanillaAttention(head_dim=8)
        out_fa = fa._tiled_forward(q, k, v, mask=None, causal=False)
        out_v = vanilla.forward(q, k, v, causal=False)
        np.testing.assert_allclose(out_fa, out_v, rtol=1e-3, atol=1e-4)

    def test_no_nan(self):
        from aether.kernels.attention import FlashAttention2
        fa = FlashAttention2(head_dim=16, block_size=16)
        q = np.random.randn(1, 1, 4, 16).astype(np.float32)
        k = np.random.randn(1, 1, 4, 16).astype(np.float32)
        v = np.random.randn(1, 1, 4, 16).astype(np.float32)
        out = fa._tiled_forward(q, k, v, mask=None, causal=True)
        assert np.isfinite(out).all()


class TestGroupedQueryAttention:
    def test_output_shape_gqa(self):
        from aether.kernels.attention import GroupedQueryAttention
        gqa = GroupedQueryAttention(num_q_heads=8, num_kv_heads=2, head_dim=16)
        q = np.random.randn(1, 8, 6, 16).astype(np.float32)
        k = np.random.randn(1, 2, 6, 16).astype(np.float32)
        v = np.random.randn(1, 2, 6, 16).astype(np.float32)
        out = gqa.forward(q, k, v)
        assert out.shape == (1, 8, 6, 16)

    def test_mqa_single_kv_head(self):
        from aether.kernels.attention import GroupedQueryAttention
        gqa = GroupedQueryAttention(num_q_heads=4, num_kv_heads=1, head_dim=8)
        q = np.random.randn(1, 4, 3, 8).astype(np.float32)
        k = np.random.randn(1, 1, 3, 8).astype(np.float32)
        v = np.random.randn(1, 1, 3, 8).astype(np.float32)
        out = gqa.forward(q, k, v)
        assert out.shape == (1, 4, 3, 8)

    def test_invalid_divisor_raises(self):
        from aether.kernels.attention import GroupedQueryAttention
        with pytest.raises(ValueError):
            GroupedQueryAttention(num_q_heads=7, num_kv_heads=3, head_dim=16)


class TestSlidingWindowAttention:
    def test_output_shape(self):
        from aether.kernels.attention import SlidingWindowAttention
        swa = SlidingWindowAttention(head_dim=16, window_size=4)
        q = np.random.randn(1, 2, 8, 16).astype(np.float32)
        k = np.random.randn(1, 2, 8, 16).astype(np.float32)
        v = np.random.randn(1, 2, 8, 16).astype(np.float32)
        out = swa.forward(q, k, v)
        assert out.shape == q.shape

    def test_no_nan(self):
        from aether.kernels.attention import SlidingWindowAttention
        swa = SlidingWindowAttention(head_dim=8, window_size=2)
        q = np.random.randn(1, 1, 4, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        out = swa.forward(q, k, v)
        assert np.isfinite(out).all()


class TestPagedAttention:
    def test_allocate_and_write_read(self):
        from aether.kernels.attention import PagedAttention
        pa = PagedAttention(head_dim=4, block_size=4)
        block_id = pa.allocate_block(num_heads=2)
        key = np.ones((2, 4), dtype=np.float32)
        val = np.ones((2, 4), dtype=np.float32) * 2.0
        pa.write_kv(block_id, slot=0, key=key, value=val)
        assert block_id in pa._kv_store

    def test_decode_forward(self):
        from aether.kernels.attention import PagedAttention
        pa = PagedAttention(head_dim=8, block_size=4)
        H = 2
        block_id = pa.allocate_block(H)
        # Write 3 tokens
        for i in range(3):
            k = np.random.randn(H, 8).astype(np.float32)
            v = np.random.randn(H, 8).astype(np.float32)
            pa.write_kv(block_id, i, k, v)
        q = np.random.randn(1, H, 1, 8).astype(np.float32)
        out = pa.forward(q, block_table=[[block_id]], seq_lens=[3])
        assert out.shape == (1, H, 1, 8)
        assert np.isfinite(out).all()


class TestAttentionDispatcher:
    def test_selects_fa2_default(self):
        from aether.kernels.attention import AttentionDispatcher
        d = AttentionDispatcher(num_q_heads=8, num_kv_heads=8, head_dim=16)
        assert "flash" in d.kernel_name or "vanilla" in d.kernel_name

    def test_selects_gqa(self):
        from aether.kernels.attention import AttentionDispatcher
        d = AttentionDispatcher(num_q_heads=8, num_kv_heads=2, head_dim=16)
        assert d.kernel_name == "grouped_query_attention"

    def test_selects_sliding_window(self):
        from aether.kernels.attention import AttentionDispatcher
        d = AttentionDispatcher(num_q_heads=4, num_kv_heads=4, head_dim=8, window_size=16)
        assert d.kernel_name == "sliding_window_attention"

    def test_forward(self):
        from aether.kernels.attention import AttentionDispatcher
        d = AttentionDispatcher(num_q_heads=4, num_kv_heads=4, head_dim=8)
        q = np.random.randn(1, 4, 4, 8).astype(np.float32)
        k = np.random.randn(1, 4, 4, 8).astype(np.float32)
        v = np.random.randn(1, 4, 4, 8).astype(np.float32)
        out = d.forward(q, k, v)
        assert out.shape == q.shape
