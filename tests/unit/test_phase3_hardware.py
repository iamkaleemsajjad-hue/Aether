"""
Phase 3 tests: Multi-hardware targets, MLA, FP4/MXFP4, MInference, Pruning.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MLA Tests
# ---------------------------------------------------------------------------

class TestMLAConfig:
    def test_deepseek_v3_preset(self):
        from aether.attention.mla import MLAConfig
        cfg = MLAConfig.deepseek_v3()
        assert cfg.kv_lora_rank == 512
        assert cfg.num_heads == 128
        assert cfg.rope_decoupled is True

    def test_compression_ratio(self):
        from aether.attention.mla import MLAConfig
        cfg = MLAConfig(
            kv_lora_rank=512,
            num_heads=128,
            v_head_dim=128,
            num_kv_heads=128,
        )
        # standard_kv_dim = 2 * 128 * 128 = 32768
        # compression = 32768 / 512 = 64
        assert cfg.compression_ratio == pytest.approx(64.0, rel=0.01)

    def test_to_from_dict(self):
        from aether.attention.mla import MLAConfig
        cfg = MLAConfig.kimi_k2()
        d = cfg.to_dict()
        cfg2 = MLAConfig.from_dict(d)
        assert cfg2.kv_lora_rank == cfg.kv_lora_rank
        assert cfg2.num_heads == cfg.num_heads


class TestMLAWeightAbsorber:
    def test_absorb_returns_dict(self):
        from aether.attention.mla import MLAConfig, MLAWeightAbsorber
        cfg = MLAConfig(
            kv_lora_rank=8, num_heads=2, num_kv_heads=2,
            qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
        )
        absorber = MLAWeightAbsorber(cfg)
        rng = np.random.default_rng(0)
        weights = {
            # kv_b: (kv_lora_rank=8, num_kv_heads*(qk_nope_head_dim+v_head_dim)=2*(4+4)=16)
            "model.layers.0.kv_b_proj.weight": rng.normal(size=(8, 16)).astype(np.float32),
            # q:    (any_out, kv_lora_rank=8) — last dim must equal kv_lora_rank for absorption
            "model.layers.0.q_proj.weight":    rng.normal(size=(4, 8)).astype(np.float32),
            # o:    (num_kv_heads*v_head_dim=8, out_dim=16)
            "model.layers.0.o_proj.weight":    rng.normal(size=(8, 16)).astype(np.float32),
        }
        result = absorber.absorb(weights, layer_prefix="model.layers.0")
        assert isinstance(result, dict)
        assert len(result) >= len(weights)

    def test_estimate_kv_savings(self):
        from aether.attention.mla import MLAConfig, MLAWeightAbsorber
        cfg = MLAConfig.deepseek_v3()
        absorber = MLAWeightAbsorber(cfg)
        savings = absorber.estimate_kv_savings(seq_len=131072, num_layers=61)
        assert savings["savings_pct"] > 90.0   # DeepSeek ~98% savings
        assert savings["compression_ratio"] > 50.0


class TestMLACompressedKVCache:
    def test_init_and_append(self):
        from aether.attention.mla import MLAConfig, MLACompressedKVCache
        # num_kv_heads must match rope_k dim-1=2
        cfg = MLAConfig(kv_lora_rank=8, num_heads=2, num_kv_heads=2, qk_rope_head_dim=4)
        cache = MLACompressedKVCache(cfg, max_seq_len=32)
        cache.init_request("req1")
        latent = np.random.randn(3, 8).astype(np.float32)
        rope_k = np.random.randn(3, 2, 4).astype(np.float32)
        cache.append("req1", latent, rope_k)
        assert cache.seq_len("req1") == 3

    def test_reconstruct_shape(self):
        from aether.attention.mla import MLAConfig, MLACompressedKVCache
        # Explicitly set num_kv_heads=2 to match rope_k dim-1
        cfg = MLAConfig(kv_lora_rank=8, num_heads=2, num_kv_heads=2,
                        qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)
        cache = MLACompressedKVCache(cfg, max_seq_len=64)
        cache.init_request("r")
        cache.append("r", np.zeros((5, 8), np.float32), np.zeros((5, 2, 4), np.float32))
        # W_kv_b: (kv_lora_rank=8, num_kv_heads*(qk_nope_head_dim+v_head_dim)=2*(4+4)=16)
        W_kv_b = np.random.randn(8, 2 * (4 + 4)).astype(np.float32)
        k_nope, k_rope, v = cache.reconstruct("r", W_kv_b)
        assert k_nope.shape == (5, 2, 4)
        assert v.shape == (5, 2, 4)
        assert k_rope.shape == (5, 2, 4)

    def test_eviction_on_overflow(self):
        from aether.attention.mla import MLAConfig, MLACompressedKVCache
        cfg = MLAConfig(kv_lora_rank=4, num_heads=1, num_kv_heads=1, qk_rope_head_dim=4)
        cache = MLACompressedKVCache(cfg, max_seq_len=10)
        cache.init_request("r")
        for _ in range(15):
            cache.append("r", np.zeros((1, 4), np.float32), np.zeros((1, 1, 4), np.float32))
        assert cache.seq_len("r") == 10  # capped at max_seq_len

    def test_free_request(self):
        from aether.attention.mla import MLAConfig, MLACompressedKVCache
        cfg = MLAConfig(kv_lora_rank=4, num_heads=1, num_kv_heads=1, qk_rope_head_dim=4)
        cache = MLACompressedKVCache(cfg, max_seq_len=64)
        cache.init_request("r")
        cache.append("r", np.zeros((2, 4), np.float32), np.zeros((2, 1, 4), np.float32))
        cache.free_request("r")
        assert cache.seq_len("r") == 0


class TestMLADetector:
    def test_detect_deepseek_from_config(self):
        from aether.attention.mla import MLADetector
        d = MLADetector()
        cfg = d.detect_from_config({
            "architectures": ["DeepseekV3ForCausalLM"],
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
        })
        assert cfg is not None
        assert cfg.kv_lora_rank == 512

    def test_detect_from_weights(self):
        from aether.attention.mla import MLADetector
        d = MLADetector()
        keys = [
            "model.layers.0.kv_a_proj.weight",
            "model.layers.0.kv_b_proj.weight",
            "model.layers.0.k_rope.weight",
        ]
        cfg = d.detect_from_weights(keys)
        assert cfg is not None
        assert cfg.rope_decoupled is True

    def test_no_mla_for_llama(self):
        from aether.attention.mla import MLADetector
        d = MLADetector()
        cfg = d.detect_from_config({
            "architectures": ["LlamaForCausalLM"],
            "num_attention_heads": 32,
        })
        assert cfg is None


# ---------------------------------------------------------------------------
# MXFP4 Tests
# ---------------------------------------------------------------------------

class TestMXFP4Codec:
    def test_encode_decode_roundtrip(self):
        from aether.quantization.codecs import MXFP4Codec
        codec = MXFP4Codec()
        rng = np.random.default_rng(0)
        blocks = rng.normal(0, 1, (8, 32)).astype(np.float32)
        codes, scales, zp = codec.encode(blocks)
        decoded = codec.decode(codes, scales, zp)
        assert decoded.shape == blocks.shape
        # Reconstruction error should be small relative to input scale
        rel_err = np.abs(decoded - blocks).mean() / (np.abs(blocks).mean() + 1e-9)
        assert rel_err < 0.5

    def test_zero_block_roundtrip(self):
        from aether.quantization.codecs import MXFP4Codec
        codec = MXFP4Codec()
        blocks = np.zeros((4, 32), dtype=np.float32)
        codes, scales, zp = codec.encode(blocks)
        decoded = codec.decode(codes, scales, zp)
        assert np.allclose(decoded, 0.0, atol=1e-6)

    def test_4bit_codes(self):
        from aether.quantization.codecs import MXFP4Codec
        codec = MXFP4Codec()
        blocks = np.random.randn(4, 32).astype(np.float32)
        codes, _, _ = codec.encode(blocks)
        # FP4 codes must fit in 4 bits (0-15)
        assert codes.max() <= 15
        assert codes.min() >= 0

    def test_get_codec_mxfp4(self):
        from aether.quantization.codecs import get_codec, MXFP4Codec
        codec = get_codec("MXFP4")
        assert isinstance(codec, MXFP4Codec)

    def test_quantize_dequantize_tensor(self):
        from aether.quantization.formats import quantize_tensor, dequantize_tensor
        w = np.random.randn(64, 64).astype(np.float32)
        qt = quantize_tensor(w, "MXFP4", block_size=32)
        assert qt.bits == 4
        reconstructed = dequantize_tensor(qt)
        assert reconstructed.shape == w.shape


# ---------------------------------------------------------------------------
# MInference Tests
# ---------------------------------------------------------------------------

class TestMInferencePass:
    def _make_graph(self, num_layers: int = 8):
        class G:
            pass
        g = G()
        g.num_layers = num_layers
        g.metadata = {}
        return g

    def test_skips_short_context(self):
        from aether.compiler.stage2_optimizer.pass8_minference import Pass8MInference
        p = Pass8MInference(
            model_config={"max_position_embeddings": 4096},
            model_id="small-model"
        )
        g = self._make_graph()
        p.run(g)
        assert p.profile is None  # skipped

    def test_classifies_long_context(self):
        from aether.compiler.stage2_optimizer.pass8_minference import Pass8MInference
        p = Pass8MInference(
            model_config={
                "max_position_embeddings": 131072,
                "num_attention_heads": 8,
                "num_hidden_layers": 4,
            },
            model_id="qwen3-72b"
        )
        g = self._make_graph(num_layers=4)
        p.run(g)
        assert p.profile is not None
        assert len(p.profile.patterns) == 4 * 8  # 4 layers × 8 heads

    def test_sparsity_in_range(self):
        from aether.compiler.stage2_optimizer.pass8_minference import Pass8MInference
        p = Pass8MInference(
            model_config={
                "max_position_embeddings": 65536,
                "num_attention_heads": 4,
                "num_hidden_layers": 4,
            }
        )
        p.run(self._make_graph(4))
        for pattern in p.profile.patterns:
            assert 0.0 <= pattern.sparsity_ratio <= 1.0

    def test_profile_save_load(self, tmp_path):
        from aether.compiler.stage2_optimizer.pass8_minference import (
            Pass8MInference, MInferenceProfile
        )
        p = Pass8MInference(
            model_config={
                "max_position_embeddings": 65536,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
            },
            model_id="test-model"
        )
        p.run(self._make_graph(2), aeg_dir=str(tmp_path))
        loaded = MInferenceProfile.load(tmp_path)
        assert loaded.model_id == "test-model"
        assert len(loaded.patterns) == 4  # 2 layers × 2 heads


class TestSparseAttentionKernel:
    def test_a_shape_forward(self):
        from aether.compiler.stage2_optimizer.pass8_minference import (
            SparseAttentionKernel, HeadPattern, SparsePattern
        )
        pattern = HeadPattern(
            layer_idx=0, head_idx=0,
            pattern_type=SparsePattern.A_SHAPE,
            sparsity_ratio=0.8,
            local_window_size=4,
            num_sink_tokens=2,
        )
        kernel = SparseAttentionKernel(pattern)
        q = np.random.randn(16, 32).astype(np.float32)
        k = np.random.randn(16, 32).astype(np.float32)
        v = np.random.randn(16, 32).astype(np.float32)
        out = kernel.forward(q, k, v)
        assert out.shape == (16, 32)
        assert np.isfinite(out).all()

    def test_vertical_slash_forward(self):
        from aether.compiler.stage2_optimizer.pass8_minference import (
            SparseAttentionKernel, HeadPattern, SparsePattern
        )
        pattern = HeadPattern(
            layer_idx=1, head_idx=0,
            pattern_type=SparsePattern.VERTICAL_SLASH,
            sparsity_ratio=0.7,
            slash_width=2,
            slash_count=2,
        )
        kernel = SparseAttentionKernel(pattern)
        S = 12
        q = np.random.randn(S, 16).astype(np.float32)
        k = np.random.randn(S, 16).astype(np.float32)
        v = np.random.randn(S, 16).astype(np.float32)
        out = kernel.forward(q, k, v)
        assert out.shape == (S, 16)


# ---------------------------------------------------------------------------
# Pruning / Pass 9 Tests
# ---------------------------------------------------------------------------

class TestWandaPruner:
    def test_unstructured_sparsity(self):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import WandaPruner
        pruner = WandaPruner(sparsity_ratio=0.5)
        W = np.random.randn(32, 64).astype(np.float32)
        pruned, mask = pruner.prune_unstructured(W)
        actual_sparsity = 1.0 - mask.mean()
        assert abs(actual_sparsity - 0.5) < 0.05  # within 5%

    def test_24_semi_structured_ratio(self):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import WandaPruner
        pruner = WandaPruner(sparsity_ratio=0.5)
        W = np.random.randn(16, 64).astype(np.float32)
        pruned, mask = pruner.prune_24_semi_structured(W)
        # 2:4 = exactly 50% zeros
        actual_sparsity = 1.0 - mask.mean()
        assert abs(actual_sparsity - 0.5) < 0.02

    def test_24_every_group_has_2_nonzero(self):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import WandaPruner
        pruner = WandaPruner(sparsity_ratio=0.5)
        W = np.random.randn(4, 8).astype(np.float32)
        _, mask = pruner.prune_24_semi_structured(W)
        # Each group of 4 should have exactly 2 ones
        mask_groups = mask.reshape(4, 2, 4)
        for row in range(4):
            for grp in range(2):
                assert mask_groups[row, grp].sum() == 2.0

    def test_layer_mask_skips_1d(self):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import WandaPruner
        pruner = WandaPruner(0.5)
        bias = np.zeros(128, dtype=np.float32)
        mask = pruner.compute_layer_mask("bias", bias)
        assert mask.actual_sparsity == 0.0


class TestPass9PruningSparsity:
    def _make_graph(self):
        class G:
            num_layers = 4
            metadata = {}
        return G()

    def test_speed_strategy(self, tmp_path):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import Pass9PruningSparsity
        p = Pass9PruningSparsity(
            strategy="speed",
            model_config={"num_hidden_layers": 4},
            model_id="test"
        )
        g = self._make_graph()
        p.run(g, aeg_dir=str(tmp_path))
        assert p.manifest is not None
        assert p.manifest.strategy == "speed"
        assert p.manifest.estimated_throughput_multiplier > 1.0

    def test_blackwell_strategy(self):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import Pass9PruningSparsity
        p = Pass9PruningSparsity(strategy="blackwell", model_id="test")
        g = self._make_graph()
        p.run(g)
        assert p.manifest.estimated_throughput_multiplier >= 1.5

    def test_unknown_strategy_raises(self):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import Pass9PruningSparsity
        with pytest.raises(ValueError, match="Unknown strategy"):
            Pass9PruningSparsity(strategy="nonexistent")

    def test_wanda_on_real_weights(self, tmp_path):
        from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import Pass9PruningSparsity
        rng = np.random.default_rng(0)
        weights = {
            "model.layers.0.q_proj.weight": rng.normal(size=(64, 64)).astype(np.float32),
            "model.layers.0.v_proj.weight": rng.normal(size=(64, 64)).astype(np.float32),
        }
        p = Pass9PruningSparsity(strategy="quality", model_id="test")
        g = self._make_graph()
        p.run(g, weights=weights, aeg_dir=str(tmp_path))
        assert len(p.manifest.layer_masks) == 2
        for mask in p.manifest.layer_masks:
            assert 0.4 < mask.actual_sparsity < 0.6  # ~50% sparsity


# ---------------------------------------------------------------------------
# Metal MSL emitter Tests
# ---------------------------------------------------------------------------

class TestMetalKernelEmitter:
    def test_emit_gemm_source(self):
        from aether.compiler.stage3_targeting.target_metal import MetalKernelEmitter
        emitter = MetalKernelEmitter()
        kernel = emitter.emit_gemm(dtype="bf16")
        assert "gemm_simdgroup_bf16" in kernel.source
        assert "threadgroup" in kernel.source
        assert kernel.dtype == "bf16"

    def test_emit_flash_attention(self):
        from aether.compiler.stage3_targeting.target_metal import MetalKernelEmitter
        emitter = MetalKernelEmitter()
        kernel = emitter.emit_flash_attention(dtype="fp16")
        assert "flash_attention_fp16" in kernel.source
        assert "INFINITY" in kernel.source or "online" in kernel.source.lower() or "m_i" in kernel.source

    def test_emit_rmsnorm(self):
        from aether.compiler.stage3_targeting.target_metal import MetalKernelEmitter
        emitter = MetalKernelEmitter()
        kernel = emitter.emit_rmsnorm()
        assert "rsqrt" in kernel.source

    def test_emit_all_standard(self):
        from aether.compiler.stage3_targeting.target_metal import MetalKernelEmitter
        emitter = MetalKernelEmitter()
        kernels = emitter.emit_all_standard_kernels(dtype="bf16")
        assert len(kernels) == 4

    def test_save_kernels(self, tmp_path):
        from aether.compiler.stage3_targeting.target_metal import MetalKernelEmitter
        emitter = MetalKernelEmitter()
        emitter.emit_all_standard_kernels()
        saved = emitter.save(tmp_path)
        assert (tmp_path / "kernel_manifest.json").exists()
        assert len(saved) >= 2


# ---------------------------------------------------------------------------
# ROCm HIP emitter Tests
# ---------------------------------------------------------------------------

class TestROCmKernelEmitter:
    def test_emit_gemm_source(self):
        from aether.compiler.stage3_targeting.target_rocm import ROCmKernelEmitter
        emitter = ROCmKernelEmitter()
        kernel = emitter.emit_gemm(dtype="fp16")
        assert "gemm_fp16" in kernel.source
        assert "__shared__" in kernel.source
        assert "__syncthreads" in kernel.source

    def test_emit_flash_attention_hip(self):
        from aether.compiler.stage3_targeting.target_rocm import ROCmKernelEmitter
        emitter = ROCmKernelEmitter()
        kernel = emitter.emit_flash_attention(dtype="fp16")
        assert "flash_attention_fp16" in kernel.source
        assert "expf" in kernel.source

    def test_mi300x_profile(self):
        from aether.compiler.stage3_targeting.target_rocm import ROCmTargetProfile
        profile = ROCmTargetProfile.mi300x()
        assert profile.gfx_arch == "gfx942"
        assert profile.hbm_capacity_gb == 192.0

    def test_save_kernels(self, tmp_path):
        from aether.compiler.stage3_targeting.target_rocm import ROCmKernelEmitter
        emitter = ROCmKernelEmitter()
        emitter.emit_all_standard_kernels(dtype="fp16")
        saved = emitter.save(tmp_path)
        assert (tmp_path / "kernel_manifest.json").exists()
        # Check .hip files were created
        hip_files = list(tmp_path.glob("*.hip"))
        assert len(hip_files) >= 4
