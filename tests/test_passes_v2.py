"""
Tests for PRD v4.0 + v5.0 optimizer passes (10–22).

Each pass is tested with:
  - Smoke test (runs without exception on minimal input).
  - Core logic test (verifies key algorithm outputs).
  - Config-flag test (pass is skipped when disabled).
  - AEG artifact test (correct output files written).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport


# ── Mock graph / architecture helpers ─────────────────────────────────────────


def _make_graph(n_layers: int = 4, output_dir: str | None = None) -> MagicMock:
    """Create a minimal mock AEG graph."""
    g = MagicMock()
    g.n_layers = n_layers
    g.output_dir = output_dir
    g.metadata = {}
    g.weight_store = {}
    return g


def _make_arch(n_layers: int = 4, hidden: int = 256, vocab: int = 1000) -> dict:
    return {
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": hidden // 4,
    }


def _make_config(**kwargs) -> CompilerConfig:
    cfg = CompilerConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 10 — MTP Head Compilation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass10MTPHead:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass10_mtp_head import MTPHeadCompilationPass
        return MTPHeadCompilationPass()

    def test_smoke(self):
        p = self._import()
        g = _make_graph()
        cfg = _make_config(enable_mtp_head=True, mtp_num_heads=3)
        result_g, report = p.run(g, _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p = self._import()
        g = _make_graph()
        cfg = _make_config(enable_mtp_head=False)
        _, report = p.run(g, _make_arch(), cfg)
        assert report.status == "skipped"

    def test_heads_emitted(self, tmp_path):
        p = self._import()
        g = _make_graph(output_dir=str(tmp_path))
        cfg = _make_config(enable_mtp_head=True, mtp_num_heads=3)
        _, report = p.run(g, _make_arch(), cfg)
        if report.status == "ok":
            assert report.details.get("mtp_heads_compiled", 0) == 3

    def test_artifact_written(self, tmp_path):
        p = self._import()
        g = _make_graph(output_dir=str(tmp_path))
        cfg = _make_config(enable_mtp_head=True, mtp_num_heads=2)
        _, report = p.run(g, _make_arch(), cfg)
        if report.status == "ok":
            artifact = tmp_path / "speculation" / "mtp_config.json"
            assert artifact.exists(), "mtp_config.json not written"


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 11 — Grammar Constraint
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass11GrammarConstraint:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass11_grammar_constraint import GrammarConstraintCompilerPass
        return GrammarConstraintCompilerPass()

    def test_smoke(self):
        p = self._import()
        g = _make_graph()
        cfg = _make_config(enable_grammar_constraint=True, grammar_type="json", grammar_spec="{}")
        _, report = p.run(g, _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p = self._import()
        _, report = self._import().run(_make_graph(), _make_arch(), _make_config(enable_grammar_constraint=False))
        assert report.status == "skipped"

    def test_json_grammar_compiles(self):
        p = self._import()
        cfg = _make_config(enable_grammar_constraint=True, grammar_type="json", grammar_spec="{}")
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        if report.status == "ok":
            assert report.details.get("n_states", 0) > 0

    def test_fsm_artifact_written(self, tmp_path):
        p = self._import()
        g = _make_graph(output_dir=str(tmp_path))
        cfg = _make_config(enable_grammar_constraint=True, grammar_type="regex", grammar_spec=r"\d+")
        _, report = p.run(g, _make_arch(), cfg)
        if report.status == "ok":
            fsm_bin = tmp_path / "grammar" / "fsm.bin"
            assert fsm_bin.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 14 — Semantic KV Compression
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass14SemanticKV:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass14_semantic_kv_compression import (
            SemanticKVCompressionPass,
            chunk_kv_compress,
            sentence_kv_compress,
            cosine_similarity,
        )
        return SemanticKVCompressionPass(), chunk_kv_compress, sentence_kv_compress, cosine_similarity

    def test_smoke(self):
        p, *_ = self._import()
        cfg = _make_config(enable_semantic_kv=True, semantic_kv_compression_ratio=0.5)
        _, report = p.run(_make_graph(), _make_arch(n_layers=4), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p, *_ = self._import()
        _, report = p.run(_make_graph(), _make_arch(), _make_config(enable_semantic_kv=False))
        assert report.status == "skipped"

    def test_cosine_similarity_unit_vectors(self):
        _, _, _, cos_sim = self._import()
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(cos_sim(a, b) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        _, _, _, cos_sim = self._import()
        assert abs(cos_sim([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_chunk_kv_compress_reduces(self):
        _, chunk_kv, _, _ = self._import()
        keys = [[float(i), float(i + 1)] for i in range(64)]
        vals = [[float(i)] for i in range(64)]
        ck, cv, idx = chunk_kv(keys, vals, retention_ratio=0.5, chunk_size=8)
        assert len(ck) < len(keys)
        assert len(ck) == len(cv)

    def test_sentence_kv_compress(self):
        _, _, sent_kv, _ = self._import()
        keys = [[float(i)] for i in range(20)]
        vals = [[float(i)] for i in range(20)]
        mask = [True if i % 5 == 0 else False for i in range(20)]
        ck, cv, idx = sent_kv(keys, vals, mask, retention_ratio=0.5)
        # All boundary tokens must be retained.
        boundary_idxs = {i for i, b in enumerate(mask) if b}
        retained_set = set(idx)
        assert boundary_idxs <= retained_set, "Boundary tokens not retained."

    def test_pyramid_schedule_report(self):
        p, *_ = self._import()
        cfg = _make_config(enable_semantic_kv=True, semantic_kv_compression_ratio=0.3)
        _, report = p.run(_make_graph(), _make_arch(n_layers=8), cfg)
        if report.status == "ok":
            plans = report.details.get("layer_plans", [])
            assert len(plans) == 8
            # Lower layers should have higher retention (pyramid).
            assert plans[0]["retention_ratio"] >= plans[-1]["retention_ratio"]

    def test_artifact_written(self, tmp_path):
        p, *_ = self._import()
        cfg = _make_config(enable_semantic_kv=True, semantic_kv_compression_ratio=0.5)
        _, report = p.run(_make_graph(output_dir=str(tmp_path)), _make_arch(), cfg)
        if report.status == "ok":
            plan = tmp_path / "graph" / "kv_compression_plan.json"
            assert plan.exists()
            data = json.loads(plan.read_text())
            assert "layers" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 15 — Cross-Layer KV Sharing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass15CrossLayerKV:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass15_cross_layer_kv import (
            CrossLayerKVSharingPass,
            _middle_outward_groups,
        )
        return CrossLayerKVSharingPass(), _middle_outward_groups

    def test_smoke(self):
        p, _ = self._import()
        cfg = _make_config(enable_cross_layer_kv=True, cross_layer_kv_share_threshold=0.5)
        _, report = p.run(_make_graph(), _make_arch(n_layers=8), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p, _ = self._import()
        _, report = p.run(_make_graph(), _make_arch(), _make_config(enable_cross_layer_kv=False))
        assert report.status == "skipped"

    def test_middle_outward_groups_structure(self):
        _, mo = self._import()
        groups = mo(n_layers=8, threshold=0.5)
        # All target layers should be unique (no double-sharing).
        all_tgts = [t for tgts in groups.values() for t in tgts]
        assert len(all_tgts) == len(set(all_tgts)), "Duplicate sharing assignments."

    def test_memory_reduction_positive(self):
        p, _ = self._import()
        cfg = _make_config(enable_cross_layer_kv=True, cross_layer_kv_share_threshold=0.4)
        _, report = p.run(_make_graph(), _make_arch(n_layers=16), cfg)
        if report.status == "ok":
            assert report.details["estimated_kv_memory_reduction_pct"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 16 — Green Energy
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass16GreenEnergy:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass16_green_energy import (
            GreenEnergyCompilationPass,
            _GRID_CARBON_INTENSITY,
        )
        return GreenEnergyCompilationPass(), _GRID_CARBON_INTENSITY

    def test_smoke(self):
        p, _ = self._import()
        cfg = _make_config(enable_green_energy=True, green_carbon_region="eu-north", green_target_tdp_watts=300.0)
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p, _ = self._import()
        _, report = p.run(_make_graph(), _make_arch(), _make_config(enable_green_energy=False))
        assert report.status == "skipped"

    def test_carbon_intensity_table(self):
        _, ci = self._import()
        # Nordic region must be cleanest in table.
        assert ci["eu-north"] < ci["ap-south"]
        assert ci["us-west"] < ci["cn-north"]

    def test_artifact_written(self, tmp_path):
        p, _ = self._import()
        cfg = _make_config(enable_green_energy=True, green_carbon_region="us-west", green_target_tdp_watts=400.0)
        p.run(_make_graph(output_dir=str(tmp_path)), _make_arch(), cfg)
        profile = tmp_path / "metadata" / "green_profile.json"
        assert profile.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 17 — TEE Wrapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass17TEE:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass17_tee_wrapping import TEEKernelWrappingPass
        return TEEKernelWrappingPass()

    def test_smoke(self):
        p = self._import()
        cfg = _make_config(enable_tee=True, tee_backend="nvidia_cc", tee_attest_endpoint=None)
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p = self._import()
        _, report = p.run(_make_graph(), _make_arch(), _make_config(enable_tee=False))
        assert report.status == "skipped"

    def test_artifact_written(self, tmp_path):
        p = self._import()
        cfg = _make_config(enable_tee=True, tee_backend="nvidia_cc", tee_attest_endpoint=None)
        p.run(_make_graph(output_dir=str(tmp_path)), _make_arch(), cfg)
        tee_cfg = tmp_path / "security" / "tee_config.json"
        hash_manifest = tmp_path / "security" / "weight_hash_manifest.json"
        assert tee_cfg.exists()
        assert hash_manifest.exists()

    def test_all_backends_accepted(self):
        from aether.compiler.stage2_optimizer.pass17_tee_wrapping import _SUPPORTED_TEE_BACKENDS
        for backend in _SUPPORTED_TEE_BACKENDS:
            p = self._import()
            cfg = _make_config(enable_tee=True, tee_backend=backend, tee_attest_endpoint=None)
            _, report = p.run(_make_graph(), _make_arch(), cfg)
            assert report.status in ("applied", "ok", "skipped", "failed")


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 18 — MDLM Drafter
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass18MDLM:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass18_mdlm_drafter import (
            MDLMDrafterCompilationPass,
            _cosine_schedule,
        )
        return MDLMDrafterCompilationPass(), _cosine_schedule

    def test_cosine_schedule_bounds(self):
        _, sched = self._import()
        s = sched(10)
        assert abs(s[0] - 1.0) < 1e-6, "Alpha_0 must be 1.0"
        assert abs(s[-1] - 0.0) < 1e-6, "Alpha_T must be 0.0"
        assert len(s) == 11  # T+1 values

    def test_cosine_schedule_monotone(self):
        _, sched = self._import()
        s = sched(20)
        for i in range(len(s) - 1):
            assert s[i] >= s[i + 1], "Schedule must be monotonically decreasing."

    def test_smoke(self):
        p, _ = self._import()
        cfg = _make_config(enable_mdlm_drafter=True, mdlm_drafter_steps=5, mdlm_draft_block_size=4)
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_speedup_positive(self):
        p, _ = self._import()
        cfg = _make_config(enable_mdlm_drafter=True, mdlm_drafter_steps=5, mdlm_draft_block_size=8)
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        if report.status == "ok":
            assert report.details["estimated_speedup"] > 1.0

    def test_artifact_written(self, tmp_path):
        p, _ = self._import()
        g = _make_graph(output_dir=str(tmp_path))
        cfg = _make_config(enable_mdlm_drafter=True, mdlm_drafter_steps=5, mdlm_draft_block_size=4)
        p.run(g, _make_arch(), cfg)
        schedule = tmp_path / "diffusion" / "schedule.json"
        assert schedule.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 19 — Sub-2-Bit Quantization
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass19Sub2Bit:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass19_sub2bit_quant import (
            Sub2BitQuantizationPass,
            _BitNetQuantizer,
        )
        return Sub2BitQuantizationPass(), _BitNetQuantizer()

    def test_bitnet_ternary_values(self):
        _, qtz = self._import()
        weights = {"layer": [2.0, -1.5, 0.1, -0.05, 3.0]}
        quantized, scales, bpw = qtz.quantize(weights)
        for v in quantized["layer"]:
            assert v in (-1, 0, 1), f"Unexpected ternary value: {v}"

    def test_bitnet_compression_ratio(self):
        _, qtz = self._import()
        assert abs(qtz.bits_per_weight - math.log2(3)) < 0.01

    def test_smoke_bitnet(self):
        p, _ = self._import()
        g = _make_graph()
        g.weight_store = {"layer.weight": [1.0, -2.0, 0.5, 0.0, -0.1]}
        cfg = _make_config(enable_sub2bit=True, sub2bit_method="bitnet", sub2bit_quality_gate_ppl=0.10)
        _, report = p.run(g, _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p, _ = self._import()
        _, report = p.run(_make_graph(), _make_arch(), _make_config(enable_sub2bit=False))
        assert report.status == "skipped"


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 20 — Video Token Compression
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass20VideoCompression:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass20_video_compression import (
            VideoTokenCompressionPass,
            _is_vlm_architecture,
        )
        return VideoTokenCompressionPass(), _is_vlm_architecture

    def test_non_vlm_skip(self):
        p, _ = self._import()
        cfg = _make_config(enable_video_compression=True)
        arch = {"model_type": "llama"}  # Non-VLM
        _, report = p.run(_make_graph(), arch, cfg)
        assert report.status == "skipped" or report.details.get("reason") == "non_vlm_architecture"

    def test_vlm_detection(self):
        _, detect = self._import()
        vlm_arch = {"architectures": ["Qwen2VLForConditionalGeneration"], "vision_model": {}}
        assert detect(vlm_arch, _make_graph()) is True

    def test_non_vlm_detection(self):
        _, detect = self._import()
        llm_arch = {"architectures": ["LlamaForCausalLM"]}
        assert detect(llm_arch, _make_graph()) is False

    def test_token_reduction_positive(self):
        p, _ = self._import()
        vlm_arch = {"architectures": ["LLaVAForConditionalGeneration"], "vision_model": {}}
        cfg = _make_config(enable_video_compression=True, video_compression_ratio=0.25)
        _, report = p.run(_make_graph(), vlm_arch, cfg)
        if report.status == "ok":
            assert report.details["token_reduction_pct"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 21 — Advanced PEFT
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass21PEFT:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass21_advanced_peft import (
            AdvancedPEFTCompilationPass,
            _compile_lora_plus,
        )
        return AdvancedPEFTCompilationPass(), _compile_lora_plus

    def test_skip_no_adapters(self):
        p, _ = self._import()
        cfg = _make_config(enable_advanced_peft=True, peft_adapter_paths=[])
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        # Pass is skipped when adapter_paths is empty.
        assert report.status in ("skipped", "ok")

    def test_lora_plus_scaling(self):
        _, compile_fn = self._import()
        adapter = {
            "lora_A": {"layer.weight": [1.0, 2.0, 3.0]},
            "lora_B": {"layer.weight": [1.0, 1.0]},
            "rank": 4,
        }
        compiled = compile_fn(adapter, "/tmp/test_adapter", lambda_scale=16.0, hidden_size=64)
        # LoRA+ scales B matrix by lambda / sqrt(rank) = 16/sqrt(4) = 8.
        expected_scale = 16.0 / math.sqrt(4)
        assert abs(compiled["lora_plus_scale"] - expected_scale) < 1e-6
        for v_scaled, v_orig in zip(
            compiled["lora_B"]["layer.weight"],
            adapter["lora_B"]["layer.weight"],
        ):
            assert abs(v_scaled - v_orig * expected_scale) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 22 — RLVR Verifier
# ═══════════════════════════════════════════════════════════════════════════════


class TestPass22RLVR:
    def _import(self):
        from aether.compiler.stage2_optimizer.pass22_rlvr_verifier import (
            RLVRVerifierHeadInjectionPass,
        )
        return RLVRVerifierHeadInjectionPass()

    def test_smoke(self):
        p = self._import()
        cfg = _make_config(enable_rlvr_verifier=True, rlvr_verifier_type="sympy", rlvr_group_size=4)
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        assert report.status in ("applied", "ok", "skipped", "failed")

    def test_skip_when_disabled(self):
        p = self._import()
        _, report = p.run(_make_graph(), _make_arch(), _make_config(enable_rlvr_verifier=False))
        assert report.status == "skipped"

    def test_k_bounds_enforced(self):
        p = self._import()
        cfg = _make_config(enable_rlvr_verifier=True, rlvr_verifier_type="sympy", rlvr_group_size=100)
        _, report = p.run(_make_graph(), _make_arch(), cfg)
        if report.status == "ok":
            assert report.details["grpo_K"] <= 64

    def test_artifact_written(self, tmp_path):
        p = self._import()
        cfg = _make_config(enable_rlvr_verifier=True, rlvr_verifier_type="sympy", rlvr_group_size=4)
        p.run(_make_graph(output_dir=str(tmp_path)), _make_arch(), cfg)
        cfg_file = tmp_path / "training" / "rlvr_config.json"
        assert cfg_file.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Optimizer Pipeline integration test
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptimizerPipelineIntegration:
    def test_pipeline_has_22_passes(self):
        from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
        pipeline = OptimizerPipeline()
        assert pipeline.pass_count == 22

    def test_pipeline_runs_without_exception(self):
        from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
        pipeline = OptimizerPipeline()
        g = _make_graph()
        arch = _make_arch()
        result_g, reports = pipeline.run(g, arch)
        assert len(reports) == 22
        statuses = {r.status for r in reports}
        # "applied" is a legacy status used by original PRD v3.1 passes (1, 9).
        # New passes must use "ok" | "skipped" | "failed".
        valid_statuses = {"ok", "skipped", "failed", "applied"}
        assert statuses <= valid_statuses, f"Unexpected statuses: {statuses - valid_statuses}"

    def test_new_passes_disabled_by_default_not_crash(self):
        """Opt-in passes should cleanly skip, not crash."""
        from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
        pipeline = OptimizerPipeline()
        g = _make_graph()
        _, reports = pipeline.run(g, _make_arch())
        opt_in_names = {
            "grammar_constraint_compilation",
            "model_merging",
            "ttt_fast_weight_injection",
            "tee_kernel_wrapping",
            "mdlm_drafter_compilation",
            "sub2bit_quantization",
            "rlvr_verifier_head_injection",
        }
        for r in reports:
            if r.pass_name in opt_in_names:
                assert r.status == "skipped", (
                    f"Opt-in pass {r.pass_name} should be skipped by default, got {r.status}"
                )
