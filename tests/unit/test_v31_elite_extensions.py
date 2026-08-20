"""Tests for v3.1 Elite Extensions (Sections 28–40).

Covers:
- Section 28: Long-Context Engine (SalienceKVEvictor, RingAttentionPlanner, YaRNConfig, LongContextProfile)
- Section 33: Distillation PRM (ProcessRewardModel, ReasoningChainAligner, SelfDistillationConfig)
- Section 35: Provenance Fingerprint (AEGModelFingerprint, ZKOwnershipProof)
- Section 36: Hot-Reload (AetherHotReload, AutoRolloutController)
- Section 37: CUDA Graphs (CUDAGraphManifestWriter, PersistentKernelRegistry)
- Section 38: Fleet Health (FleetHealthMonitor, AutoScaler, MultiRegionTopology)
- Section 39: AEG Format v3.1 (AEGPackageV31, AEGManifestV31)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Section 28: Long-Context Engine
# ---------------------------------------------------------------------------

class TestSalienceKVEvictor:
    def _make_block(self, request_id="req1", start=0, end=100, is_anchor=False, attention_mass=0.1):
        from aether.runtime.long_context import KVBlock
        return KVBlock(request_id=request_id, layer_idx=0, start_token=start,
                       end_token=end, is_anchor=is_anchor, attention_mass=attention_mass)

    def test_anchor_block_max_score(self):
        from aether.runtime.long_context import SalienceKVEvictor
        evictor = SalienceKVEvictor(window_size=1000)
        anchor = self._make_block(is_anchor=True)
        score = evictor.score_salience(anchor, current_seq_len=5000)
        # Anchor gets 0.2 * 1.0 = 0.2 + recency + attention — should be high
        assert score > 0.0

    def test_recent_block_higher_score_than_old(self):
        from aether.runtime.long_context import SalienceKVEvictor
        evictor = SalienceKVEvictor(window_size=1000)
        recent = self._make_block(start=4500, end=4600)
        old = self._make_block(start=0, end=100)
        score_recent = evictor.score_salience(recent, current_seq_len=5000)
        score_old = evictor.score_salience(old, current_seq_len=5000)
        assert score_recent > score_old

    def test_eviction_order_sorted_ascending(self):
        from aether.runtime.long_context import SalienceKVEvictor, KVBlock
        evictor = SalienceKVEvictor(window_size=1000)
        blocks = [
            self._make_block(start=0, end=50, is_anchor=False),
            self._make_block(start=4500, end=4600, is_anchor=False),
            self._make_block(start=1, end=5, is_anchor=True),
        ]
        result = evictor.eviction_order(blocks, current_seq_len=5000)
        scores = [s for _, s, _ in result]
        assert scores == sorted(scores)  # Ascending (lowest first = evict first)

    def test_eviction_order_tiers_assigned(self):
        from aether.runtime.long_context import SalienceKVEvictor
        evictor = SalienceKVEvictor()
        blocks = [self._make_block(start=i*100, end=(i+1)*100) for i in range(8)]
        result = evictor.eviction_order(blocks, current_seq_len=1000)
        tiers = {tier for _, _, tier in result}
        # At least 2 distinct tiers across 8 blocks
        assert len(tiers) >= 2

    def test_attention_weights_influence_score(self):
        from aether.runtime.long_context import SalienceKVEvictor, KVBlock
        evictor = SalienceKVEvictor()
        block = KVBlock("req1", 0, 0, 10, False, 0.0)
        weights_high = np.array([0.9] * 100)
        weights_low = np.array([0.01] * 100)
        score_high = evictor.score_salience(block, 100, weights_high)
        score_low = evictor.score_salience(block, 100, weights_low)
        assert score_high > score_low

    def test_config_dict(self):
        from aether.runtime.long_context import SalienceKVEvictor
        config = SalienceKVEvictor().config_dict()
        assert "evictor" in config
        assert "tiers" in config
        assert len(config["tiers"]) == 4


class TestRingAttentionPlanner:
    def test_plan_4gpu_1m_tokens(self):
        from aether.runtime.long_context import RingAttentionPlanner
        planner = RingAttentionPlanner()
        plan = planner.plan(total_tokens=1_000_000, num_gpus=4, target="cuda_sm90")
        assert plan.num_gpus == 4
        assert plan.total_tokens == 1_000_000
        assert plan.tokens_per_gpu == 250_000
        assert plan.topology in ("ring", "striped", "ulysses_ring")

    def test_plan_insufficient_gpus_raises(self):
        from aether.runtime.long_context import RingAttentionPlanner
        planner = RingAttentionPlanner()
        with pytest.raises(ValueError, match="Insufficient GPUs"):
            planner.plan(total_tokens=10_000_000, num_gpus=1, target="cuda_sm90")

    def test_ulysses_ring_for_large_clusters(self):
        from aether.runtime.long_context import RingAttentionPlanner
        planner = RingAttentionPlanner()
        plan = planner.plan(total_tokens=1_000_000, num_gpus=128, target="cuda_sm100")
        assert plan.topology == "ulysses_ring"

    def test_plan_to_dict(self):
        from aether.runtime.long_context import RingAttentionPlanner
        planner = RingAttentionPlanner()
        plan = planner.plan(1000, 2, target="cuda_sm90")
        d = plan.to_dict()
        assert "topology" in d
        assert "tokens_per_gpu" in d
        assert "research" in d

    def test_write_plans_creates_files(self):
        from aether.runtime.long_context import RingAttentionPlanner
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = RingAttentionPlanner()
            files = planner.write_plans(tmpdir, target="cuda_sm90")
            assert len(files) > 0
            for f in files:
                assert f.exists()
                json.loads(f.read_text())  # Valid JSON


class TestYaRNConfig:
    def test_scale_factor(self):
        from aether.runtime.long_context import YaRNConfig
        cfg = YaRNConfig(original_max_position=4096, target_max_position=131072)
        assert cfg.scale_factor == pytest.approx(32.0, abs=0.1)

    def test_extended_theta_larger_than_original(self):
        from aether.runtime.long_context import YaRNConfig
        cfg = YaRNConfig(original_max_position=4096, target_max_position=131072)
        assert cfg.extended_theta > cfg.rope_theta

    def test_all_methods_produce_dict(self):
        from aether.runtime.long_context import YaRNConfig
        for method in ("yarn", "longrope", "dynamic_ntk"):
            cfg = YaRNConfig(method=method)
            d = cfg.to_dict()
            assert d["method"] == method
            assert "scale_factor" in d


class TestLongContextProfile:
    def test_to_dict_structure(self):
        from aether.runtime.long_context import LongContextProfile
        profile = LongContextProfile(max_context_tokens=1_000_000)
        d = profile.to_dict()
        assert d["max_context_tokens"] == 1_000_000
        assert "rope_extension" in d
        assert "kv_eviction_policy" in d
        assert d["sparse_attention_enabled"] is True

    def test_write_to_manifest(self):
        from aether.runtime.long_context import LongContextProfile
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = LongContextProfile(max_context_tokens=131072)
            path = profile.write_to_manifest(tmpdir)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "long_context_profile" in data
            assert data["long_context_profile"]["max_context_tokens"] == 131072


# ---------------------------------------------------------------------------
# Section 33: Process Reward Model and Distillation
# ---------------------------------------------------------------------------

class TestProcessRewardModel:
    def test_score_returns_float(self):
        from aether.distillation.reward_model import ProcessRewardModel
        prm = ProcessRewardModel()
        score = prm.score("What is 2+2?", "2+2=4, therefore the answer is 4.")
        assert 0.0 <= score <= 1.0

    def test_step_scorer_returns_float(self):
        from aether.distillation.reward_model import ProcessRewardModel
        prm = ProcessRewardModel()
        s = prm.step_scorer("prompt", "Therefore, x = 5 because 3+2=5.")
        assert 0.0 <= s <= 1.0

    def test_good_reasoning_scores_higher(self):
        from aether.distillation.reward_model import ProcessRewardModel
        prm = ProcessRewardModel()
        good_step = "Therefore, since 3 * 4 = 12, the answer is 12."
        bad_step = "I think maybe probably the answer could be 12 or so."
        assert prm.step_scorer("", good_step) > prm.step_scorer("", bad_step)

    def test_score_detailed_returns_step_list(self):
        from aether.distillation.reward_model import ProcessRewardModel
        prm = ProcessRewardModel()
        response = "1. First we compute 2+2=4.\n2. Therefore the answer is 4."
        steps = prm.score_detailed("What is 2+2?", response)
        assert len(steps) >= 1
        for step in steps:
            assert 0.0 <= step.score <= 1.0
            assert isinstance(step.is_correct, bool)

    def test_math_error_detection(self):
        from aether.distillation.reward_model import ProcessRewardModel
        prm = ProcessRewardModel()
        bad = "The result is = undefined because the formula breaks."
        steps = prm.score_detailed("", bad)
        assert any(s.error_type == "math_error" for s in steps)

    def test_empty_response(self):
        from aether.distillation.reward_model import ProcessRewardModel
        prm = ProcessRewardModel()
        score = prm.score("prompt", "")
        assert score == 0.0

    def test_to_dict(self):
        from aether.distillation.reward_model import ProcessRewardModel
        d = ProcessRewardModel().to_dict()
        assert "aggregation" in d
        assert d["aggregation"] == "minimum_step_score"


class TestReasoningChainAligner:
    def test_align_similar_chains(self):
        from aether.distillation.reward_model import ReasoningChainAligner
        aligner = ReasoningChainAligner()
        teacher = "1. First step: compute x.\n2. Therefore x = 5."
        student = "1. We compute x.\n2. So x equals 5."
        result = aligner.align("prompt", teacher, student)
        assert result.alignment_score >= 0.0
        assert isinstance(result.missing_steps, list)
        assert isinstance(result.extra_steps, list)

    def test_align_perfect_match(self):
        from aether.distillation.reward_model import ReasoningChainAligner
        aligner = ReasoningChainAligner()
        chain = "1. The sky is blue.\n2. Therefore it is daytime."
        result = aligner.align("", chain, chain)
        assert result.alignment_score > 0.5

    def test_align_empty_student(self):
        from aether.distillation.reward_model import ReasoningChainAligner
        aligner = ReasoningChainAligner()
        teacher = "1. Step one.\n2. Step two."
        result = aligner.align("", teacher, "")
        # Should not crash
        assert result.alignment_score >= 0.0

    def test_result_to_dict(self):
        from aether.distillation.reward_model import ReasoningChainAligner
        aligner = ReasoningChainAligner()
        result = aligner.align("p", "Step 1. Step 2.", "Step 1.")
        d = result.to_dict()
        assert "alignment_score" in d
        assert "num_teacher_steps" in d


class TestSelfDistillationConfig:
    def test_default_config(self):
        from aether.distillation.reward_model import SelfDistillationConfig
        cfg = SelfDistillationConfig()
        assert cfg.num_generations >= 1
        assert cfg.teacher_context_length > cfg.student_context_length

    def test_reasoning_preset(self):
        from aether.distillation.reward_model import SelfDistillationConfig
        cfg = SelfDistillationConfig.for_reasoning_model()
        assert cfg.teacher_context_length >= 32768
        assert cfg.selection_strategy == "prm_top1"

    def test_general_preset(self):
        from aether.distillation.reward_model import SelfDistillationConfig
        cfg = SelfDistillationConfig.for_general_model()
        assert cfg.selection_strategy == "majority_vote"

    def test_to_dict(self):
        from aether.distillation.reward_model import SelfDistillationConfig
        d = SelfDistillationConfig().to_dict()
        assert "expected_quality_retention" in d
        assert "cost_reduction_vs_external_teacher" in d
        assert d["method"] == "self_distillation_via_fine_tuning"


# ---------------------------------------------------------------------------
# Section 35: Provenance Fingerprinting
# ---------------------------------------------------------------------------

class TestAEGModelFingerprint:
    @staticmethod
    def _runner(trigger: str) -> str:
        return f"model-response:{trigger}"

    def test_embed_returns_dict(self):
        from aether.provenance.fingerprint import AEGModelFingerprint
        fp = AEGModelFingerprint()
        result = fp.embed("model.aeg", "owner-123", n_triggers=10, generate=self._runner)
        assert "n_triggers" in result
        assert result["n_triggers"] == 10
        assert len(result["trigger_records"]) == 10

    def test_verify_same_model_matches(self):
        from aether.provenance.fingerprint import AEGModelFingerprint
        fp = AEGModelFingerprint()
        fingerprint = fp.embed("model.aeg", "owner-123", n_triggers=10, generate=self._runner)
        result = fp.verify("model.aeg", "owner-123", fingerprint, generate=self._runner)
        assert result.is_derived
        assert result.match_rate == 1.0

    def test_verify_wrong_owner_no_match(self):
        from aether.provenance.fingerprint import AEGModelFingerprint
        fp = AEGModelFingerprint()
        fingerprint = fp.embed("model.aeg", "owner-123", n_triggers=10, generate=self._runner)
        result = fp.verify("model.aeg", "wrong-owner", fingerprint, generate=self._runner)
        assert not result.is_derived
        assert result.match_rate == 0.0

    def test_fingerprint_result_to_dict(self):
        from aether.provenance.fingerprint import AEGModelFingerprint
        fp = AEGModelFingerprint()
        fingerprint = fp.embed("model.aeg", "owner-abc", n_triggers=5, generate=self._runner)
        result = fp.verify("model.aeg", "owner-abc", fingerprint, generate=self._runner)
        d = result.to_dict()
        assert "is_derived" in d
        assert "verdict" in d
        assert d["verdict"] in ("IP_DERIVED", "NOT_DERIVED")

    def test_write_creates_fingerprint_json(self):
        from aether.provenance.fingerprint import AEGModelFingerprint
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = AEGModelFingerprint()
            path = fp.write(tmpdir, "owner-456", n_triggers=5, generate=self._runner)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["n_triggers"] == 5
            assert "owner_id_hash" in data


class TestZKOwnershipProof:
    def test_create_proof(self):
        from aether.provenance.fingerprint import ZKOwnershipProof
        proof = ZKOwnershipProof.create("owner-xyz", "sha256:abc123")
        assert proof.owner_commitment
        assert proof.proof_hash
        assert proof.model_binding

    def test_verify_binding_correct_hash(self):
        from aether.provenance.fingerprint import ZKOwnershipProof
        model_hash = "sha256:" + "a" * 64
        proof = ZKOwnershipProof.create("owner-1", model_hash)
        assert proof.verify_binding(model_hash)

    def test_verify_binding_wrong_hash(self):
        from aether.provenance.fingerprint import ZKOwnershipProof
        proof = ZKOwnershipProof.create("owner-1", "sha256:abc")
        assert not proof.verify_binding("sha256:xyz_different")

    def test_to_dict(self):
        from aether.provenance.fingerprint import ZKOwnershipProof
        proof = ZKOwnershipProof.create("owner-1", "sha256:abc")
        d = proof.to_dict()
        assert "protocol" in d
        assert d["protocol"] == "groth16"
        assert "owner_commitment" in d


# ---------------------------------------------------------------------------
# Section 36: Hot-Reload
# ---------------------------------------------------------------------------

class TestAetherHotReload:
    def test_start_reload_creates_experiment(self):
        from aether.runtime.hot_reload import AetherHotReload, RolloutState
        reloader = AetherHotReload()
        exp = reloader.start_reload("new-model.aeg", active_aeg="old-model.aeg")
        assert exp.experiment_id
        assert exp.state == RolloutState.CANARY
        assert exp.candidate_percent == pytest.approx(0.01, abs=0.001)

    def test_route_returns_valid_variant(self):
        from aether.runtime.hot_reload import AetherHotReload
        reloader = AetherHotReload()
        exp = reloader.start_reload("new.aeg", active_aeg="old.aeg")
        for i in range(20):
            assignment = reloader.route_request(exp.experiment_id, f"user-{i}")
            # ABRolloutController returns 'candidate' or 'control'
            assert assignment in ("active", "candidate", "control")

    def test_record_telemetry_no_alert(self):
        from aether.runtime.hot_reload import AetherHotReload
        from aether.observability.gates import TelemetrySnapshot
        reloader = AetherHotReload()
        exp = reloader.start_reload("new.aeg", baseline_win_rate=0.80)
        snap = TelemetrySnapshot(100.0, 50.0, 0.85, 0.70, 0.30, 0.5, 0.75, win_rate=0.79)
        for _ in range(5):
            status = reloader.record_telemetry(exp.experiment_id, snap)
        assert not status.get("alert", False)

    def test_rollback(self):
        from aether.runtime.hot_reload import AetherHotReload, RolloutState
        reloader = AetherHotReload()
        exp = reloader.start_reload("new.aeg", active_aeg="old.aeg")
        active = reloader.rollback(exp.experiment_id)
        assert active == "old.aeg"
        assert exp.state == RolloutState.ROLLED_BACK
        assert exp.candidate_percent == 0.0

    def test_promote(self):
        from aether.runtime.hot_reload import AetherHotReload, RolloutState
        reloader = AetherHotReload()
        exp = reloader.start_reload("new.aeg", active_aeg="old.aeg")
        result = reloader.promote(exp.experiment_id)
        assert result == "new.aeg"
        assert exp.state == RolloutState.FULL_ROLLOUT

    def test_status_returns_dict(self):
        from aether.runtime.hot_reload import AetherHotReload
        reloader = AetherHotReload()
        exp = reloader.start_reload("new.aeg")
        status = reloader.status(exp.experiment_id)
        assert "state" in status
        assert "candidate_percent" in status

    def test_all_experiments(self):
        from aether.runtime.hot_reload import AetherHotReload
        reloader = AetherHotReload()
        reloader.start_reload("a.aeg")
        reloader.start_reload("b.aeg")
        experiments = reloader.all_experiments()
        assert len(experiments) == 2


# ---------------------------------------------------------------------------
# Section 37: CUDA Graphs
# ---------------------------------------------------------------------------

class TestCUDAGraphManifestWriter:
    def test_write_creates_directory(self):
        from aether.cuda.graph_manifest import CUDAGraphManifestWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CUDAGraphManifestWriter(target="cuda_sm90")
            files = writer.write(tmpdir)
            cuda_dir = Path(tmpdir) / "cuda_graphs"
            assert cuda_dir.exists()
            assert len(files) > 0

    def test_writes_per_batch_files(self):
        from aether.cuda.graph_manifest import CUDAGraphManifestWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CUDAGraphManifestWriter(
                target="cuda_sm90",
                decode_batch_sizes=(1, 2, 4, 8),
            )
            files = writer.write(tmpdir)
            names = {f.name for f in files}
            assert "sm90_decode_b1.json" in names
            assert "sm90_decode_b8.json" in names

    def test_manifest_is_valid_json(self):
        from aether.cuda.graph_manifest import CUDAGraphManifestWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CUDAGraphManifestWriter(target="cuda_sm90")
            files = writer.write(tmpdir)
            manifest = Path(tmpdir) / "cuda_graphs" / "manifest.json"
            assert manifest.exists()
            data = json.loads(manifest.read_text())
            assert "target" in data
            assert "capture_plan" in data
            assert "persistent_kernels" in data

    def test_non_cuda_target_no_files(self):
        from aether.cuda.graph_manifest import CUDAGraphManifestWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CUDAGraphManifestWriter(target="metal_m3")
            files = writer.write(tmpdir)
            assert files == []

    def test_decode_graph_metadata_structure(self):
        from aether.cuda.graph_manifest import CUDAGraphManifestWriter
        writer = CUDAGraphManifestWriter(target="cuda_sm100")
        meta = writer._decode_graph_metadata(8)
        assert meta["batch_size"] == 8
        assert "kernel_sequence" in meta
        assert len(meta["kernel_sequence"]) > 0


class TestPersistentKernelRegistry:
    def test_sm90_has_kernels(self):
        from aether.cuda.graph_manifest import PersistentKernelRegistry
        registry = PersistentKernelRegistry(sm_version="sm90")
        specs = registry.get_specs()
        assert len(specs) > 0

    def test_to_dict_structure(self):
        from aether.cuda.graph_manifest import PersistentKernelRegistry
        registry = PersistentKernelRegistry(sm_version="sm90")
        d = registry.to_dict()
        assert "kernels" in d
        assert "persistent_count" in d
        assert "overhead_reduction" in d

    def test_sm100_has_fp4_gemm(self):
        from aether.cuda.graph_manifest import PersistentKernelRegistry
        registry = PersistentKernelRegistry(sm_version="sm100")
        specs = registry.get_specs()
        kernel_names = [s.kernel_name for s in specs]
        assert any("fp4" in name or "sm100" in name for name in kernel_names)


# ---------------------------------------------------------------------------
# Section 38: Fleet Health
# ---------------------------------------------------------------------------

class TestFleetHealthMonitor:
    def _make_metrics(self, node_id="n1", p95=200.0, error_rate=0.0, gpu_util=0.7, queue=0):
        from aether.fleet.health import NodeMetrics
        return NodeMetrics(
            node_id=node_id,
            p50_latency_ms=80.0,
            p95_latency_ms=p95,
            p99_latency_ms=p95 * 1.5,
            error_rate=error_rate,
            tokens_per_second=120.0,
            gpu_utilization=gpu_util,
            queue_depth=queue,
        )

    def test_healthy_node(self):
        from aether.fleet.health import FleetHealthMonitor, NodeHealth
        monitor = FleetHealthMonitor()
        metrics = self._make_metrics(p95=200.0, error_rate=0.001, gpu_util=0.7)
        health = monitor.record(metrics)
        assert health == NodeHealth.HEALTHY

    def test_degraded_on_high_latency(self):
        from aether.fleet.health import FleetHealthMonitor, NodeHealth
        monitor = FleetHealthMonitor()
        metrics = self._make_metrics(p95=600.0)  # > 500ms SLO
        health = monitor.record(metrics)
        assert health in (NodeHealth.DEGRADED, NodeHealth.UNHEALTHY)

    def test_fleet_summary(self):
        from aether.fleet.health import FleetHealthMonitor
        monitor = FleetHealthMonitor()
        for i in range(3):
            monitor.record(self._make_metrics(node_id=f"n{i}"))
        summary = monitor.fleet_summary()
        assert summary["total_nodes"] == 3
        assert "health_distribution" in summary

    def test_unhealthy_nodes_listed(self):
        from aether.fleet.health import FleetHealthMonitor
        monitor = FleetHealthMonitor()
        monitor.record(self._make_metrics("good", p95=100.0, error_rate=0.0))
        monitor.record(self._make_metrics("bad", p95=5000.0, error_rate=0.2))
        unhealthy = monitor.unhealthy_nodes()
        assert "bad" in unhealthy


class TestAutoScaler:
    def _make_metrics(self, queue=0, p95=100.0, gpu_util=0.5, p50=50.0):
        from aether.fleet.health import NodeMetrics
        return NodeMetrics(
            node_id="n1",
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p95 * 2,
            error_rate=0.0,
            tokens_per_second=100.0,
            gpu_utilization=gpu_util,
            queue_depth=queue,
        )

    def test_scale_up_on_high_queue(self):
        from aether.fleet.health import AutoScaler
        scaler = AutoScaler(min_replicas=1, max_replicas=8, scale_up_queue_threshold=5)
        decision = scaler.evaluate(2, [self._make_metrics(queue=20)])
        assert decision.action == "scale_up"
        assert decision.delta_replicas > 0

    def test_no_change_within_slo(self):
        from aether.fleet.health import AutoScaler
        scaler = AutoScaler()
        decision = scaler.evaluate(2, [self._make_metrics(queue=0, p95=200.0, gpu_util=0.5)])
        assert decision.action in ("no_change", "scale_down")

    def test_scale_down_when_idle(self):
        from aether.fleet.health import AutoScaler
        scaler = AutoScaler(min_replicas=1, max_replicas=8, scale_down_idle_steps=1)
        for _ in range(2):
            decision = scaler.evaluate(3, [self._make_metrics(queue=0, p50=5.0, p95=10.0)])
        assert decision.action in ("scale_down", "no_change")

    def test_cannot_exceed_max_replicas(self):
        from aether.fleet.health import AutoScaler
        scaler = AutoScaler(min_replicas=1, max_replicas=4, scale_up_queue_threshold=5)
        decision = scaler.evaluate(4, [self._make_metrics(queue=100)])
        assert decision.target_replicas <= 4


class TestMultiRegionTopology:
    def _make_region(self, region_id, target="cuda_sm90", compliance=()):
        from aether.fleet.health import RegionSpec
        from aether.fleet.manager import FleetNode
        node = FleetNode(node_id=f"{region_id}-n1", target=target, gpu_count=2, memory_gb=80, region=region_id)
        return RegionSpec(region_id=region_id, nodes=(node,), compliance_zones=compliance)

    def test_assign_region_returns_region(self):
        from aether.fleet.health import MultiRegionTopology
        regions = [self._make_region("us-east-1"), self._make_region("eu-west-2")]
        topo = MultiRegionTopology(regions)
        region = topo.assign_region("req-001")
        assert region.region_id in ("us-east-1", "eu-west-2")

    def test_compliance_filter(self):
        from aether.fleet.health import MultiRegionTopology
        eu = self._make_region("eu-west-2", compliance=("GDPR",))
        us = self._make_region("us-east-1", compliance=("CCPA",))
        topo = MultiRegionTopology([eu, us])
        region = topo.assign_region("req-1", compliance_required="GDPR")
        assert region.region_id == "eu-west-2"

    def test_topology_manifest(self):
        from aether.fleet.health import MultiRegionTopology
        regions = [self._make_region("us-east-1"), self._make_region("eu-west-2")]
        topo = MultiRegionTopology(regions)
        manifest = topo.topology_manifest()
        assert manifest["region_count"] == 2
        assert "routing" in manifest


# ---------------------------------------------------------------------------
# Section 39: AEG Format v3.1
# ---------------------------------------------------------------------------

class TestAEGPackageV31:
    def test_build_creates_directory_structure(self):
        from aether.core.aeg_format_v31 import AEGPackageV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "test-model.aeg"
            pkg = AEGPackageV31(model_id="test-model")
            pkg.build(aeg_dir)
            assert aeg_dir.exists()
            assert (aeg_dir / "FORMAT_VERSION").exists()
            assert (aeg_dir / "manifest.json").exists()

    def test_build_creates_v31_directories(self):
        from aether.core.aeg_format_v31 import AEGPackageV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            pkg = AEGPackageV31(model_id="qwen3-72b", target="cuda_sm90")
            pkg.build(aeg_dir)
            # v3.1 NEW directories
            assert (aeg_dir / "cuda_graphs").exists()
            assert (aeg_dir / "inference").exists()
            assert (aeg_dir / "provenance").exists()
            assert (aeg_dir / "watermark").exists()
            assert (aeg_dir / "safety").exists()

    def test_format_version_correct(self):
        from aether.core.aeg_format_v31 import AEGPackageV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test").build(aeg_dir)
            version = (aeg_dir / "FORMAT_VERSION").read_text()
            assert version.startswith("AEG/")

    def test_manifest_has_features(self):
        from aether.core.aeg_format_v31 import AEGPackageV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test-model", target="cuda_sm90").build(aeg_dir)
            manifest = json.loads((aeg_dir / "manifest.json").read_text())
            assert "features" in manifest
            assert manifest["artifact_kind"] == "metadata_skeleton"
            assert manifest["executable"] is False
            assert manifest["features"]["safety_guardrails"] is False
            assert manifest["features"]["eu_ai_act_compliant"] is False

    def test_provenance_manifest_written(self):
        from aether.core.aeg_format_v31 import AEGPackageV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("qwen3-72b").build(aeg_dir)
            prov = json.loads((aeg_dir / "provenance" / "manifest.json").read_text())
            assert "eu_ai_act" in prov
            assert "model_hash" in prov
            assert prov["model_hash"] is None
            assert prov["source_hash_status"] == "unavailable"

    def test_provenance_hashes_real_source_bytes(self, tmp_path):
        from aether.core.aeg_format_v31 import ProvenanceManifest

        weights = tmp_path / "weights.bin"
        weights.write_bytes(b"real-model-bytes")
        provenance = ProvenanceManifest.from_compile_run(
            "local-model", model_weights_path=str(weights)
        )
        assert provenance.model_hash == "sha256:" + __import__("hashlib").sha256(
            b"weights.binreal-model-bytes"
        ).hexdigest()
        assert provenance.source_hash_status == "verified"
        assert provenance.eval_gate_passed is False

    def test_watermark_config_written(self):
        from aether.core.aeg_format_v31 import AEGPackageV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test").build(aeg_dir)
            wm = json.loads((aeg_dir / "watermark" / "config.json").read_text())
            assert wm["method"] == "green_list_token"
            assert "delta" in wm


class TestAEGManifestV31:
    def test_load_existing_manifest(self):
        from aether.core.aeg_format_v31 import AEGPackageV31, AEGManifestV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test").build(aeg_dir)
            reader = AEGManifestV31(aeg_dir)
            manifest = reader.load()
            assert "model_id" in manifest

    def test_verify_format_version(self):
        from aether.core.aeg_format_v31 import AEGPackageV31, AEGManifestV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test").build(aeg_dir)
            reader = AEGManifestV31(aeg_dir)
            assert reader.verify_format_version()

    def test_list_available_targets(self):
        from aether.core.aeg_format_v31 import AEGPackageV31, AEGManifestV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test").build(aeg_dir)
            reader = AEGManifestV31(aeg_dir)
            targets = reader.list_available_targets()
            assert targets == []

    def test_has_cuda_graphs(self):
        from aether.core.aeg_format_v31 import AEGPackageV31, AEGManifestV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("test", target="cuda_sm90").build(aeg_dir)
            reader = AEGManifestV31(aeg_dir)
            assert not reader.has_cuda_graphs()

    def test_summary_structure(self):
        from aether.core.aeg_format_v31 import AEGPackageV31, AEGManifestV31
        with tempfile.TemporaryDirectory() as tmpdir:
            aeg_dir = Path(tmpdir) / "model.aeg"
            AEGPackageV31("summary-test").build(aeg_dir)
            reader = AEGManifestV31(aeg_dir)
            summary = reader.summary()
            assert summary["model_id"] == "summary-test"
            assert "available_targets" in summary
            assert "features" in summary
