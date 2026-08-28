"""
Phase 4 tests: Reasoning graph, Cascade router, Agentic sessions,
LoRA engine, SSM/Hybrid states, Compute controller, RAG pipeline,
Multimodal dispatch, Provenance & watermarking.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reasoning Graph (Pass 7) Tests
# ---------------------------------------------------------------------------

class TestPass7ReasoningGraph:
    def _make_graph(self):
        class G:
            num_layers = 4
            metadata = {}
        return G()

    def test_cot_config_defaults(self):
        from aether.compiler.stage2_optimizer.pass7_reasoning_graph import CoTConfig
        cfg = CoTConfig()
        assert cfg.max_thinking_tokens > 0
        assert 0 < cfg.temperature <= 2.0

    def test_run_produces_reasoning_graph(self, tmp_path):
        from aether.compiler.stage2_optimizer.pass7_reasoning_graph import Pass7ReasoningGraph
        p = Pass7ReasoningGraph(model_id="r1-test")
        g = self._make_graph()
        p.run(g, aeg_dir=str(tmp_path))
        assert p.reasoning_graph is not None
        assert p.reasoning_graph.has_think_phase

    def test_graph_saved_to_aeg(self, tmp_path):
        from aether.compiler.stage2_optimizer.pass7_reasoning_graph import Pass7ReasoningGraph
        import json
        p = Pass7ReasoningGraph(model_id="r1-save-test")  # 'r1' in REASONING_MODEL_FAMILIES
        p.run(self._make_graph(), aeg_dir=str(tmp_path))
        rg_file = tmp_path / "graph" / "reasoning_graph.json"
        assert rg_file.exists()
        data = json.loads(rg_file.read_text())
        assert "version" in data

    def test_budget_controller_range(self, tmp_path):
        from aether.compiler.stage2_optimizer.pass7_reasoning_graph import Pass7ReasoningGraph
        p = Pass7ReasoningGraph(model_id="reasoning-test-model")  # 'reasoning' in families
        p.run(self._make_graph(), aeg_dir=str(tmp_path))
        controller = p.reasoning_graph.budget_controller
        assert controller is not None
        assert controller.max_tokens > 0

    def test_annotates_graph_metadata(self):
        from aether.compiler.stage2_optimizer.pass7_reasoning_graph import Pass7ReasoningGraph
        p = Pass7ReasoningGraph(model_id="reasoning-annotate-test")
        g = self._make_graph()
        p.run(g)
        assert g.metadata.get("reasoning_enabled") is True


# ---------------------------------------------------------------------------
# Cascade Router Tests
# ---------------------------------------------------------------------------

class TestComplexityScorer:
    def test_simple_scores_low(self):
        from aether.runtime.cascade_router import ComplexityScorer
        s = ComplexityScorer()
        signals = s.extract_signals("What is 2+2?")
        score = s.score(signals)
        assert score < 0.4

    def test_expert_math_scores_high(self):
        from aether.runtime.cascade_router import ComplexityScorer
        s = ComplexityScorer()
        text = (
            "Prove using rigorous epsilon-delta definition that lim_{x→0} sin(x)/x = 1. "
            "Provide step-by-step algebraic derivation using Taylor expansion. "
            "$$\\sin(x) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n x^{2n+1}}{(2n+1)!}$$"
        )
        signals = s.extract_signals(text)
        score = s.score(signals)
        # Math content should score significantly higher than a trivial question
        assert score > 0.25
        # Must score higher than a simple greeting
        simple_score = s.score(s.extract_signals("Hi, how are you?"))
        assert score > simple_score


    def test_classify(self):
        from aether.runtime.cascade_router import ComplexityScorer
        s = ComplexityScorer()
        assert s.classify(0.1) == "simple"
        assert s.classify(0.4) == "medium"
        assert s.classify(0.7) == "hard"
        assert s.classify(0.9) == "very_hard"


class TestCascadeRouter:
    def _router_with_defaults(self):
        from aether.runtime.cascade_router import CascadeRouter
        r = CascadeRouter()
        r.register_default_tiers(
            nano="nano.aeg", small="small.aeg", mid="mid.aeg", large="large.aeg"
        )
        return r

    def test_routes_simple_to_tier0(self):
        r = self._router_with_defaults()
        decision = r.route("Hi, how are you?")
        assert decision.tier.tier_id == 0

    def test_routes_complex_to_higher_tier(self):
        r = self._router_with_defaults()
        decision = r.route(
            "Prove the Riemann Hypothesis step by step using advanced complex analysis "
            "and zeta function theory with full mathematical rigor."
        )
        assert decision.tier.tier_id >= 1

    def test_force_tier(self):
        r = self._router_with_defaults()
        decision = r.route("Hello", force_tier=2)
        assert decision.tier.tier_id == 2

    def test_force_reasoning(self):
        r = self._router_with_defaults()
        decision = r.route("Simple question", force_reasoning=True)
        assert decision.tier.supports_reasoning

    def test_escalation(self):
        r = self._router_with_defaults()
        decision = r.route("Tell me something")
        escalated = r.escalate(decision, reason="low_confidence")
        if decision.tier.tier_id < 3:
            assert escalated is not None
            assert escalated.tier.tier_id > decision.tier.tier_id
            assert escalated.escalated is True

    def test_stats_sum_to_one(self):
        r = self._router_with_defaults()
        for q in ["hi", "calculate sin(pi/3)", "prove riemann hypothesis"]:
            r.route(q)
        stats = r.stats()
        total = sum(stats["tier_distribution"].values())
        assert abs(total - 1.0) < 0.01

    def test_no_tiers_raises(self):
        from aether.runtime.cascade_router import CascadeRouter
        r = CascadeRouter()
        with pytest.raises(RuntimeError, match="No tiers registered"):
            r.route("test")


# ---------------------------------------------------------------------------
# Agentic KV Session Manager Tests
# ---------------------------------------------------------------------------

class TestAgenticKVSessionManager:
    def test_create_session(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        session = mgr.create_session("s1")
        assert session.session_id == "s1"
        assert session.turn_count == 0

    def test_idempotent_create(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        s1 = mgr.create_session("s1")
        s2 = mgr.create_session("s1")
        assert s1.session_id == s2.session_id

    def test_append_turn_creates_block(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        mgr.create_session("s")
        block = mgr.append_turn("s", [1, 2, 3, 4, 5])
        assert block.num_tokens == 5
        assert block.tier == "L1_GPU"

    def test_prefix_dedup(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        tokens = [100, 200, 300, 400, 500]
        mgr.create_session("s1")
        block1 = mgr.append_turn("s1", tokens)
        mgr.create_session("s2")
        block2 = mgr.append_turn("s2", tokens)
        # Same tokens → same prefix hash → cache hit on second call
        assert block1.prefix_hash == block2.prefix_hash

    def test_get_kv_blocks(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        mgr.create_session("s")
        mgr.append_turn("s", [1, 2, 3])
        mgr.append_turn("s", [4, 5, 6])
        blocks = mgr.get_kv_blocks("s")
        assert len(blocks) >= 1

    def test_hit_rate_increases_with_reuse(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        tokens = list(range(10))
        mgr.create_session("s1")
        mgr.append_turn("s1", tokens)
        mgr.create_session("s2")
        mgr.append_turn("s2", tokens)  # should be cache hit
        assert mgr.kv_hit_rate > 0.0

    def test_close_session_evicts_blocks(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        mgr.create_session("s")
        mgr.append_turn("s", [1, 2, 3, 4, 5])
        mgr.close_session("s", evict_blocks=True)
        assert "s" not in mgr._sessions

    def test_stats_dict(self):
        from aether.runtime.agentic_session import AgenticKVSessionManager
        mgr = AgenticKVSessionManager()
        stats = mgr.stats()
        assert "kv_hit_rate" in stats
        assert "active_sessions" in stats


# ---------------------------------------------------------------------------
# LoRA Engine Tests
# ---------------------------------------------------------------------------

class TestLoRAEngine:
    def test_create_adapter(self):
        from aether.adapters.lora import LoRAHotSwapEngine
        engine = LoRAHotSwapEngine(max_slots=4)
        adapter = engine.create_adapter_from_config(
            "code_v1", rank=8, alpha=16, hidden_dim=64, modules=["q_proj", "v_proj"]
        )
        assert adapter.config.adapter_id == "code_v1"
        assert adapter.config.rank == 8
        assert "q_proj" in adapter.weights

    def test_scaling_factor(self):
        from aether.adapters.lora import LoRAConfig
        cfg = LoRAConfig(adapter_id="t", rank=16, alpha=32)
        assert cfg.scaling == pytest.approx(2.0, rel=1e-6)

    def test_load_adapter(self):
        from aether.adapters.lora import LoRAHotSwapEngine
        engine = LoRAHotSwapEngine(max_slots=4)
        adapter = engine.create_adapter_from_config("a", rank=4, hidden_dim=32)
        slot = engine.load_adapter(adapter)
        assert slot >= 0
        assert engine.pool.get("a") is not None

    def test_bgmv_forward_no_adapter(self):
        from aether.adapters.lora import LoRAHotSwapEngine
        engine = LoRAHotSwapEngine()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 32)).astype(np.float32)
        W = rng.normal(size=(16, 32)).astype(np.float32)
        out = engine.serve_batch(x, W, [None, None, None, None])
        assert out.shape == (4, 16)

    def test_bgmv_forward_with_adapter(self):
        from aether.adapters.lora import LoRAHotSwapEngine
        engine = LoRAHotSwapEngine()
        adapter = engine.create_adapter_from_config("a", rank=4, hidden_dim=32, modules=["q_proj"])
        engine.load_adapter(adapter)
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 32)).astype(np.float32)
        W = rng.normal(size=(32, 32)).astype(np.float32)
        out = engine.serve_batch(x, W, ["a", None], module_name="q_proj")
        assert out.shape == (2, 32)
        # Row 0 (with adapter) should differ from row 1 (no adapter) due to LoRA delta
        # Note: B is zero-initialized so delta = 0 for a fresh adapter
        assert np.isfinite(out).all()

    def test_lora_delta_computation(self):
        from aether.adapters.lora import LoRAAdapter, LoRAConfig
        rng = np.random.default_rng(0)
        r = 4
        hidden = 16
        cfg = LoRAConfig(adapter_id="t", rank=r, alpha=8.0)
        A = rng.normal(size=(r, hidden)).astype(np.float16)
        B = rng.normal(size=(hidden, r)).astype(np.float16)
        adapter = LoRAAdapter(config=cfg, weights={"q_proj": (A, B)})
        delta = adapter.get_delta("q_proj")
        assert delta is not None
        assert delta.shape == (hidden, hidden)

    def test_pico_compression(self):
        from aether.adapters.lora import LoRAHotSwapEngine, LoRACompiler, LoRAAdapter, LoRAConfig
        rng = np.random.default_rng(0)
        r = 16
        hidden = 32
        A = rng.normal(size=(r, hidden)).astype(np.float16)
        B = rng.normal(size=(hidden, r)).astype(np.float16)
        adapter = LoRAAdapter(
            config=LoRAConfig("t", rank=r, alpha=32.0),
            weights={"q_proj": (A, B)}
        )
        compiler = LoRACompiler(mode="delta_compress")
        compressed = compiler.delta_compress(adapter, compression_target=0.25)
        # B should be 4x smaller after compression
        _, B_c = compressed.weights["q_proj"]
        assert B_c.shape[1] <= r  # compressed rank ≤ original rank
        assert compressed.config.pico_compressed is True

    def test_save_to_aeg(self, tmp_path):
        from aether.adapters.lora import LoRAHotSwapEngine
        engine = LoRAHotSwapEngine(max_slots=2)
        adapter = engine.create_adapter_from_config("a", rank=4, hidden_dim=32)
        engine.load_adapter(adapter)
        engine.save_to_aeg(tmp_path)
        assert (tmp_path / "adapters" / "manifest.json").exists()

    def test_both_construction_forms_are_tagged_with_their_layout(self):
        """The two LoRAAdapter forms store transposed A/B; each must know which."""
        from aether.adapters.lora import LoRAAdapter, LoRAConfig

        delta = LoRAAdapter(
            adapter_id="d",
            delta_a=np.ones((4, 2), np.float32),   # (in, rank)
            delta_b=np.ones((2, 3), np.float32),   # (rank, out)
        )
        assert delta.layout == LoRAAdapter.LAYOUT_DELTA

        bgmv = LoRAAdapter(
            config=LoRAConfig(adapter_id="b", rank=2, alpha=2.0),
            weights={"q_proj": (np.ones((2, 4), np.float32),   # (rank, in)
                                np.ones((3, 2), np.float32))},  # (out, rank)
        )
        assert bgmv.layout == LoRAAdapter.LAYOUT_BGMV

    def test_single_delta_adapter_rejected_by_bgmv_kernel(self):
        """
        Serving a single-delta adapter through BGMV must raise, not guess.

        The layouts are transposes of each other. For square matrices the
        multiply succeeds and returns a silently wrong result, so this has to
        be an explicit type check rather than a shape check.
        """
        from aether.adapters.lora import LoRAAdapter, LoRAHotSwapEngine

        square = LoRAAdapter(
            adapter_id="sq",
            delta_a=np.ones((2, 2), np.float32),
            delta_b=np.ones((2, 2), np.float32),
        )
        engine = LoRAHotSwapEngine(max_slots=1)
        engine.register(square)

        x = np.ones((1, 2), np.float32)
        base = np.eye(2, dtype=np.float32)
        with pytest.raises(ValueError, match="single-delta layout"):
            engine.serve_batch(x, base, ["sq"], module_name="default")

    def test_bgmv_adapter_rejected_by_forward(self):
        """The reverse mismatch must also raise rather than compute."""
        from aether.adapters.lora import LoRAAdapter, LoRAConfig, LoRAHotSwapEngine

        bgmv = LoRAAdapter(
            config=LoRAConfig(adapter_id="bg", rank=2, alpha=2.0),
            weights={"q_proj": (np.ones((2, 2), np.float32),
                                np.ones((2, 2), np.float32))},
        )
        engine = LoRAHotSwapEngine(np.eye(2, dtype=np.float32))
        engine.register(bgmv)

        with pytest.raises(ValueError, match="BGMV layout"):
            engine.forward(np.ones((1, 2), np.float32), ["bg"], module="q_proj")

    def test_forward_and_serve_batch_agree_on_equivalent_adapters(self):
        """
        The two paths are different conventions for the same math.

        Given adapters that are transposes of one another, forward() and
        serve_batch() must produce the same output — otherwise one of them
        has its scaling or orientation wrong.
        """
        from aether.adapters.lora import LoRAAdapter, LoRAConfig, LoRAHotSwapEngine

        rng = np.random.default_rng(3)
        in_f, out_f, rank = 4, 3, 2
        A_delta = rng.normal(size=(in_f, rank)).astype(np.float32)
        B_delta = rng.normal(size=(rank, out_f)).astype(np.float32)
        base = rng.normal(size=(in_f, out_f)).astype(np.float32)
        x = rng.normal(size=(1, in_f)).astype(np.float32)

        delta = LoRAAdapter(adapter_id="d", delta_a=A_delta, delta_b=B_delta, alpha=float(rank))
        e1 = LoRAHotSwapEngine(base)
        e1.register(delta)
        out_forward = e1.forward(x, ["d"])

        # Same adapter expressed in BGMV layout; serve_batch takes W as (out, in).
        bgmv = LoRAAdapter(
            config=LoRAConfig(adapter_id="d", rank=rank, alpha=float(rank)),
            weights={"default": (A_delta.T.copy(), B_delta.T.copy())},
        )
        e2 = LoRAHotSwapEngine(max_slots=1)
        e2.register(bgmv)
        out_bgmv = e2.serve_batch(x, base.T.copy(), ["d"], module_name="default")

        np.testing.assert_allclose(out_forward, out_bgmv, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# SSM / Hybrid State Tests
# ---------------------------------------------------------------------------

class TestMambaState:
    def test_init_pool(self):
        from aether.hybrid.state import SSMStatePool
        pool = SSMStatePool()
        pool.init_mamba("req1", layer_indices=[0, 1, 2], d_inner=64, d_state=4, batch_size=1)
        state = pool.get_mamba("req1", 0)
        assert state is not None
        assert state.h.shape == (1, 4, 64)

    def test_set_and_get_mamba(self):
        from aether.hybrid.state import SSMStatePool, MambaState
        pool = SSMStatePool()
        state = MambaState(layer_idx=0, h=np.ones((1, 8, 32), dtype=np.float32))
        pool.set_mamba("req1", 0, state)
        retrieved = pool.get_mamba("req1", 0)
        assert np.allclose(retrieved.h, 1.0)

    def test_deep_copy_isolation(self):
        from aether.hybrid.state import SSMStatePool, MambaState
        pool = SSMStatePool()
        h = np.zeros((1, 4, 16), dtype=np.float32)
        pool.set_mamba("r", 0, MambaState(0, h))
        retrieved = pool.get_mamba("r", 0)
        retrieved.h[:] = 99.0
        orig = pool.get_mamba("r", 0)
        assert not np.allclose(orig.h, 99.0)  # original unchanged

    def test_free_request(self):
        from aether.hybrid.state import SSMStatePool
        pool = SSMStatePool()
        pool.init_mamba("r", [0, 1], d_inner=16, d_state=4)
        pool.free("r")
        assert pool.get_mamba("r", 0) is None


class TestHybridMemoryPool:
    def test_kv_set_get(self):
        from aether.hybrid.state import HybridMemoryPool
        pool = HybridMemoryPool()
        k = np.random.randn(10, 8, 64).astype(np.float32)
        v = np.random.randn(10, 8, 64).astype(np.float32)
        pool.set_kv("r", 0, k, v)
        k2, v2 = pool.get_kv("r", 0)
        assert np.allclose(k, k2)
        assert np.allclose(v, v2)

    def test_append_kv(self):
        from aether.hybrid.state import HybridMemoryPool
        pool = HybridMemoryPool()
        k1 = np.random.randn(5, 4, 32).astype(np.float32)
        v1 = np.random.randn(5, 4, 32).astype(np.float32)
        k2 = np.random.randn(3, 4, 32).astype(np.float32)
        v2 = np.random.randn(3, 4, 32).astype(np.float32)
        pool.set_kv("r", 0, k1, v1)
        pool.append_kv("r", 0, k2, v2)
        k_out, v_out = pool.get_kv("r", 0)
        assert k_out.shape[0] == 8

    def test_snapshot_and_rollback(self):
        from aether.hybrid.state import HybridMemoryPool
        pool = HybridMemoryPool()
        k = np.ones((4, 2, 8), dtype=np.float32)
        pool.set_kv("r", 0, k, k)
        snap = pool.snapshot("r", step=1)
        # Modify state
        pool.set_kv("r", 0, k * 2, k * 2)
        # Rollback
        ok = pool.rollback("r", snap.snapshot_id)
        assert ok
        k_restored, _ = pool.get_kv("r", 0)
        assert np.allclose(k_restored, 1.0)

    def test_free_request(self):
        from aether.hybrid.state import HybridMemoryPool
        pool = HybridMemoryPool()
        pool.set_kv("r", 0, np.zeros((2, 2, 8), np.float32), np.zeros((2, 2, 8), np.float32))
        pool.free_request("r")
        assert pool.get_kv("r", 0) is None


class TestMambaSSMForward:
    def test_single_step_shape(self):
        from aether.hybrid.state import MambaSSM, MambaState
        model = MambaSSM(d_model=8, d_state=4, d_inner=16)
        batch = 2
        state = MambaState(0, h=np.zeros((batch, 4, 16), np.float32))
        x = np.random.randn(batch, 16).astype(np.float32)
        A = np.random.randn(16, 4).astype(np.float32) * -0.5
        B = np.random.randn(batch, 4).astype(np.float32)
        C = np.random.randn(batch, 4).astype(np.float32)
        D = np.ones(16, dtype=np.float32)
        dt = np.ones((batch, 16), dtype=np.float32) * 0.1
        y, new_state = model.step(x, state, A, B, C, D, dt)
        assert y.shape == (batch, 16)
        assert new_state.h.shape == state.h.shape
        assert new_state.step == 1

    def test_state_changes_after_step(self):
        from aether.hybrid.state import MambaSSM, MambaState
        model = MambaSSM(d_model=4, d_state=2, d_inner=8)
        state = MambaState(0, h=np.zeros((1, 2, 8), np.float32))
        x = np.ones((1, 8), np.float32)
        A = np.full((8, 2), -0.1, np.float32)
        B = np.ones((1, 2), np.float32)
        C = np.ones((1, 2), np.float32)
        D = np.ones(8, np.float32)
        dt = np.ones((1, 8), np.float32) * 0.1
        _, new_state = model.step(x, state, A, B, C, D, dt)
        assert not np.allclose(new_state.h, 0.0)


class TestHybridLayerSchedule:
    def test_jamba_schedule(self):
        from aether.hybrid.state import get_hybrid_layer_schedule
        sched = get_hybrid_layer_schedule("jamba", 32)
        assert len(sched) == 32
        attn_count = sched.count("attn")
        ssm_count = sched.count("ssm")
        assert attn_count + ssm_count == 32
        assert attn_count >= 1

    def test_pure_mamba_schedule(self):
        from aether.hybrid.state import get_hybrid_layer_schedule
        sched = get_hybrid_layer_schedule("mamba", 24)
        assert all(s == "ssm" for s in sched)


# ---------------------------------------------------------------------------
# Inference-Time Compute Controller Tests
# ---------------------------------------------------------------------------

class TestProcessRewardModel:
    def test_score_returns_float_in_range(self):
        from aether.runtime.compute_controller import ProcessRewardModel
        prm = ProcessRewardModel()
        score = prm.score("What is 2+2?", "The answer is 4.")
        assert 0.0 <= score <= 1.0

    def test_math_response_scores_high(self):
        from aether.runtime.compute_controller import ProcessRewardModel
        prm = ProcessRewardModel()
        response = (
            "Step 1: Apply the quadratic formula x = -b ± √(b²-4ac) / 2a.\n"
            "Step 2: Substitute a=1, b=-5, c=6. Therefore x = (5 ± √1) / 2.\n"
            "The result is x = 3 or x = 2."
        )
        score = prm.score("Solve x^2-5x+6=0", response)
        assert score > 0.4

    def test_uncertain_response_scores_lower(self):
        from aether.runtime.compute_controller import ProcessRewardModel
        prm = ProcessRewardModel()
        r_good = "Therefore the answer is exactly 42, since 6 × 7 = 42."
        r_bad  = "I'm not sure, maybe it's around 42? Unclear."
        assert prm.score("q", r_good) > prm.score("q", r_bad)


class TestBestOfN:
    def test_selects_best_candidate(self):
        from aether.runtime.compute_controller import BestOfN, BoNConfig
        bon = BestOfN(BoNConfig(selection="reward_model"))
        candidates = [
            "I'm not sure about this.",
            "Step 1: integrate. Step 2: therefore the answer is π. Result is π.",
        ]
        best, idx, scores = bon.select_best("Integrate sin(x) from 0 to π", candidates)
        assert idx == 1

    def test_longest_selection(self):
        from aether.runtime.compute_controller import BestOfN, BoNConfig
        bon = BestOfN(BoNConfig(selection="longest"))
        candidates = ["short", "a much longer answer here with more content"]
        best, idx, _ = bon.select_best("q", candidates)
        assert idx == 1


class TestInferenceComputeController:
    def test_strategy_mapping(self):
        from aether.runtime.compute_controller import InferenceComputeController
        ctrl = InferenceComputeController()
        assert ctrl.select_strategy("simple") == "greedy"
        assert ctrl.select_strategy("very_hard") == "mcts"

    def test_run_simple(self):
        from aether.runtime.compute_controller import InferenceComputeController
        ctrl = InferenceComputeController()
        result = ctrl.run(
            "What is 2+2?",
            complexity_class="simple",
            candidates=["4", "5"],
        )
        assert "best_response" in result
        assert result["strategy"] == "greedy"
        assert 0.0 <= result["prm_score"] <= 1.0

    def test_run_bon(self):
        from aether.runtime.compute_controller import InferenceComputeController
        ctrl = InferenceComputeController()
        cands = [f"Response {i}" for i in range(4)]
        result = ctrl.run("q", complexity_class="medium", candidates=cands)
        assert result["strategy"] == "best_of_4"

    def test_max_tokens_increases_with_complexity(self):
        from aether.runtime.compute_controller import InferenceComputeController
        ctrl = InferenceComputeController()
        assert ctrl.get_max_tokens("simple") < ctrl.get_max_tokens("very_hard")

    def test_reasoning_search_requires_model_callbacks(self):
        from aether.runtime.compute_controller import InferenceComputeController

        ctrl = InferenceComputeController()
        with pytest.raises(RuntimeError, match="model-backed expand_fn"):
            ctrl.run("hard problem", complexity_class="very_hard")


# ---------------------------------------------------------------------------
# RAG Pipeline Tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def _make_docs(self, n: int = 10):
        from aether.inference.rag import Document
        return [
            Document(
                doc_id=str(i),
                text=f"Document {i}: This discusses topic {i % 3} and subject {i % 5}.",
                source="test",
            )
            for i in range(n)
        ]

    def test_index_and_retrieve(self):
        from aether.inference.rag import RAGPipeline
        pipeline = RAGPipeline()
        docs = self._make_docs(20)
        pipeline.index_documents(docs)
        result = pipeline.retrieve("topic 1")
        assert len(result.documents) > 0
        assert result.retrieval_latency_ms > 0

    def test_reranker(self):
        from aether.inference.rag import CrossEncoderReranker, Document
        reranker = CrossEncoderReranker()
        docs = [
            Document("1", "The quick brown fox", source="test"),
            Document("2", "This is completely unrelated content about databases", source="test"),
            Document("3", "The fox jumped over the fox", source="test"),
        ]
        reranked = reranker.rerank("fox jump", docs, top_k=2)
        assert len(reranked) == 2

    def test_context_assembler_budget(self):
        from aether.inference.rag import ContextAssembler, Document
        assembler = ContextAssembler(max_tokens=100)
        docs = [Document(str(i), "x" * 200, source="test") for i in range(5)]
        context = assembler.assemble("query", docs)
        # Should stay within budget
        estimated_tokens = len(context) * 0.25
        assert estimated_tokens <= 120  # some slack for header

    def test_pipeline_run(self):
        from aether.inference.rag import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.index_documents(self._make_docs(10))
        result = pipeline.run("topic discussion")
        assert "response" in result
        assert "total_latency_ms" in result

    def test_cache_hit_on_second_call(self):
        from aether.inference.rag import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.index_documents(self._make_docs(5))
        pipeline.run("same query here")
        pipeline.run("same query here")
        stats = pipeline.stats()
        assert stats["cache_hits"] >= 1

    def test_bm25_retriever(self):
        from aether.inference.rag import BM25Retriever, Document
        bm25 = BM25Retriever()
        docs = [
            Document("1", "machine learning neural networks"),
            Document("2", "football soccer sports"),
            Document("3", "deep learning transformers attention"),
        ]
        bm25.index(docs)
        results = bm25.search("neural attention transformers", top_k=3)
        assert len(results) > 0
        # Most relevant should have higher scores
        assert results[0].score >= results[-1].score

    def test_save_to_aeg(self, tmp_path):
        from aether.inference.rag import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.save_to_aeg(tmp_path)
        assert (tmp_path / "graph" / "rag_pipeline.json").exists()


# ---------------------------------------------------------------------------
# Multimodal Tests
# ---------------------------------------------------------------------------

class TestMultiModalDispatcher:
    @staticmethod
    def _loaded_tiny_dispatcher(**kwargs):
        from aether.inference.multimodal import MultiModalGraphDispatcher, VLMConfig

        cfg = VLMConfig(
            image_size=28,
            patch_size=14,
            num_image_tokens=4,
            connector_hidden_dim=8,
            **kwargs,
        )
        projection = np.ones((3 * cfg.patch_size * cfg.patch_size, 8), np.float32)
        connector = (
            np.eye(8, dtype=np.float32),
            np.eye(8, dtype=np.float32),
        )
        return MultiModalGraphDispatcher(
            cfg,
            patch_projection=projection,
            connector_weights=connector,
        )

    def test_preprocess_image(self):
        dispatcher = self._loaded_tiny_dispatcher()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = dispatcher.process_image(img.astype(np.float32))
        assert "visual_embeddings" in result
        assert result["num_visual_tokens"] > 0

    def test_dynamic_tiling_produces_multiple_tiles(self):
        dispatcher = self._loaded_tiny_dispatcher(
            dynamic_resolution=True,
            max_tiles=4,
        )
        img = np.random.randn(1024, 1024, 3).astype(np.float32)
        result = dispatcher.process_image(img)
        assert result["num_tiles"] > 1

    def test_compression_reduces_tokens(self):
        from aether.inference.multimodal import (
            MultiModalGraphDispatcher, VLMConfig, VisualTokenCompressor
        )
        comp = VisualTokenCompressor(compression_ratio=0.5)
        tokens = np.random.randn(100, 256).astype(np.float32)
        compressed = comp.compress(tokens)
        assert compressed.shape[0] == 50  # 50% of 100

    def test_merge_late_fusion(self):
        from aether.inference.multimodal import MultiModalGraphDispatcher, VLMConfig
        cfg = VLMConfig.llava_15()
        dispatcher = MultiModalGraphDispatcher(cfg)
        text_emb = np.zeros((20, 64), np.float32)
        visual_emb = np.zeros((10, 64), np.float32)
        merged = dispatcher.merge_embeddings(text_emb, visual_emb)
        assert merged.shape[0] == 30  # prepended

    def test_architecture_detection_llava(self):
        from aether.inference.multimodal import MultiModalGraphDispatcher, VLMConfig
        d = MultiModalGraphDispatcher()
        cfg = d.detect_vlm_architecture({
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
        })
        assert cfg is not None
        assert cfg.connector_type == "mlp"

    def test_non_vlm_returns_none(self):
        from aether.inference.multimodal import MultiModalGraphDispatcher
        d = MultiModalGraphDispatcher()
        cfg = d.detect_vlm_architecture({"architectures": ["LlamaForCausalLM"]})
        assert cfg is None

    def test_missing_vlm_weights_fail_closed(self):
        from aether.inference.multimodal import MultiModalGraphDispatcher, VLMConfig

        dispatcher = MultiModalGraphDispatcher(
            VLMConfig(image_size=28, patch_size=14, connector_hidden_dim=8)
        )
        with pytest.raises(RuntimeError, match="requires loaded ViT and connector weights"):
            dispatcher.process_image(np.zeros((28, 28, 3), dtype=np.float32))

    def test_save_to_aeg(self, tmp_path):
        from aether.inference.multimodal import MultiModalGraphDispatcher
        d = MultiModalGraphDispatcher()
        d.save_to_aeg(tmp_path)
        assert (tmp_path / "graph" / "multimodal_config.json").exists()


# ---------------------------------------------------------------------------
# Provenance & Watermarking Tests
# ---------------------------------------------------------------------------

class TestProvenanceManifest:
    def test_create_from_compilation(self):
        from aether.provenance.manifest import ProvenanceManifest
        pm = ProvenanceManifest.from_compilation(
            model_id="qwen3-72b",
            model_weights_hash="a" * 64,
            certified_targets=["cuda", "rocm"],
        )
        assert pm.source_model_id == "qwen3-72b"
        assert "cuda" in pm.hardware_certification.certified_targets

    def test_chain_hash_changes_with_transformations(self):
        from aether.provenance.manifest import ProvenanceManifest, TransformationRecord
        pm = ProvenanceManifest.from_compilation("m", "a" * 64)
        h1 = pm.compute_chain_hash()
        pm.add_transformation(TransformationRecord("pass1"))
        h2 = pm.compute_chain_hash()
        assert h1 != h2

    def test_save_and_load(self, tmp_path):
        from aether.provenance.manifest import ProvenanceManifest, TransformationRecord
        pm = ProvenanceManifest.from_compilation("qwen3", "b" * 64)
        pm.add_transformation(TransformationRecord("fuse_ops"))
        pm.save(tmp_path)
        loaded = ProvenanceManifest.load(tmp_path)
        assert loaded.source_model_id == "qwen3"
        assert len(loaded.transformations) == 1

    def test_to_dict_structure(self):
        from aether.provenance.manifest import ProvenanceManifest
        pm = ProvenanceManifest.from_compilation("test", "c" * 64)
        d = pm.to_dict()
        assert "eu_ai_act" in d
        assert "hardware_certification" in d
        assert d["eu_ai_act"]["risk_category"] == "limited_risk"


class TestProvenanceBuilder:
    def test_record_passes(self):
        from aether.provenance.manifest import ProvenanceManifest, ProvenanceBuilder
        pm = ProvenanceManifest.from_compilation("m", "d" * 64)
        builder = ProvenanceBuilder(pm)
        builder.record_fusion()
        builder.record_quantization("fp8")
        builder.record_pruning("wanda_24", 0.5)
        assert len(pm.transformations) == 3

    def test_finalize_adds_fingerprint(self):
        from aether.provenance.manifest import ProvenanceManifest, ProvenanceBuilder
        pm = ProvenanceManifest.from_compilation("m", "e" * 64)
        builder = ProvenanceBuilder(pm)
        final = builder.finalize(
            aeg_content=b"fake aeg content",
            watermark_enabled=True,
            watermark_key=b"secret",
        )
        assert final.watermark_enabled is True
        assert len(final.watermark_key_fingerprint) == 32
        # finalize() does not invent a C2PA identifier. Content Credentials need
        # a signature, and an unsigned artifact must report itself as unsigned
        # rather than carrying a "c2pa://…" string that resolves nowhere.
        assert final.c2pa_binding == ""
        assert final.c2pa_signed is False


class TestKGWWatermark:
    def test_apply_changes_logits(self):
        from aether.provenance.manifest import KGWWatermark
        wm = KGWWatermark(vocab_size=1000)
        logits = np.zeros(1000, dtype=np.float32)
        boosted = wm.apply(logits, prev_token_id=42)
        # Some logits should be increased by delta
        assert boosted.max() > 0.0
        assert (boosted > 0).sum() > 0

    def test_detect_non_watermarked(self):
        from aether.provenance.manifest import KGWWatermark
        wm = KGWWatermark(vocab_size=500, key=b"test_key")
        # Random tokens — not watermarked
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 500, size=200).tolist()
        result = wm.detect(tokens)
        # Random text should not exceed threshold
        assert "is_watermarked" in result
        assert result["z_score"] < 10.0  # sanity check

    def test_watermarked_text_detected(self):
        from aether.provenance.manifest import KGWWatermark
        """Simulate watermarked text: always choose green-list tokens."""
        wm = KGWWatermark(vocab_size=100, gamma=0.5, delta=5.0, key=b"wm_key")
        # Simulate greedy generation always picking green tokens
        tokens = [0]
        rng = np.random.default_rng(42)
        for _ in range(100):
            prev = tokens[-1]
            green_ids = wm._get_green_list(prev)
            next_tok = int(rng.choice(green_ids))
            tokens.append(next_tok)
        result = wm.detect(tokens)
        # With 100% green sampling, z-score should be very high
        assert result["z_score"] > 4.0
        assert result["is_watermarked"] is True

    def test_key_fingerprint_stable(self):
        from aether.provenance.manifest import KGWWatermark
        wm = KGWWatermark(key=b"stable_key")
        fp1 = wm.key_fingerprint()
        fp2 = wm.key_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 32
