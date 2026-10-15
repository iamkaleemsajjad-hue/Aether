"""Small structurally complete CPU decoder fixtures without toolchain discovery."""
import numpy as np


def engine_for_family(family):
    from aether.kernels.native_cpu import NativeCPUKernels
    if family in {"dense", "hybrid", "mamba"}:
        if family == "dense":
            from test_disaggregated_execution import _setup
            return _setup()[0]()
        from test_hybrid_rope_scaling import _weights, HEADS, KV_HEADS, STATE, INNER, DT_RANK, CONV
        from aether.runtime.hybrid_engine import HybridExecutionEngine
        engine = HybridExecutionEngine(_weights(None), layer_types=["ssm", "attention"],
            num_heads=HEADS, num_kv_heads=KV_HEADS, state_size=STATE, inner_size=INNER,
            dt_rank=DT_RANK, conv_kernel=CONV, kernels=NativeCPUKernels.reference())
        return engine if family == "hybrid" else engine._mamba
    rng = np.random.default_rng(147)
    hidden, vocab, intermediate = 8, 17, 12
    def matrix(*shape): return rng.normal(0, .08, shape).astype(np.float32)
    def ones(size): return np.ones(size, np.float32)
    embedding, head = matrix(vocab, hidden), matrix(vocab, hidden)
    if family == "mamba2":
        from aether.runtime.mamba2_engine import Mamba2ExecutionEngine, Mamba2ModelWeights, Mamba2LayerWeights
        inner, heads, groups, state, conv = 8, 2, 1, 2, 3
        layer = Mamba2LayerWeights(norm=ones(hidden), in_proj=matrix(2 * inner + 2 * groups * state + heads, hidden),
            conv1d=matrix(inner + 2 * groups * state, conv), a_log=matrix(heads),
            d=ones(heads), dt=matrix(heads), out_proj=matrix(hidden, inner))
        return Mamba2ExecutionEngine(Mamba2ModelWeights(embedding, [layer], ones(hidden), head),
            state_size=state, inner_size=inner, num_heads=heads, num_groups=groups, conv_kernel=conv)
    if family == "rwkv":
        from aether.runtime.rwkv_engine import RWKVExecutionEngine, RWKVModelWeights, RWKVLayerWeights
        layer = RWKVLayerWeights(norm=ones(hidden), ffn_norm=ones(hidden), time_decay=matrix(hidden),
            time_first=matrix(hidden), time_mix_k=ones(hidden) * .5, time_mix_v=ones(hidden) * .6,
            time_mix_r=ones(hidden) * .7, ffn_time_mix_k=ones(hidden) * .4, ffn_time_mix_r=ones(hidden) * .3,
            key=matrix(hidden, hidden), value=matrix(hidden, hidden), receptance=matrix(hidden, hidden),
            output=matrix(hidden, hidden), ffn_key=matrix(intermediate, hidden),
            ffn_value=matrix(hidden, intermediate), ffn_receptance=matrix(hidden, hidden))
        return RWKVExecutionEngine(RWKVModelWeights(embedding, [layer], ones(hidden), head))
    if family == "mla":
        from aether.attention.mla import MLAConfig
        from aether.runtime.mla_engine import MLAExecutionEngine, MLAModelWeights, MLALayerWeights
        heads, q_rank, kv_rank, nope, rope, value = 2, 4, 3, 2, 2, 2
        mla = {"q_a_proj.weight": matrix(q_rank, hidden), "q_a_norm.weight": ones(q_rank),
            "q_b_proj.weight": matrix(heads * (nope + rope), q_rank),
            "kv_a_proj.weight": matrix(kv_rank, hidden), "kv_a_norm.weight": ones(kv_rank),
            "kv_b_proj.weight": matrix(heads * (nope + value), kv_rank),
            "k_rope_proj.weight": matrix(heads * rope, hidden)}
        layer = MLALayerWeights(attention_norm=ones(hidden), ffn_norm=ones(hidden),
            o_proj=matrix(hidden, heads * value), ffn_in=matrix(intermediate, hidden),
            ffn_up=matrix(intermediate, hidden), ffn_out=matrix(hidden, intermediate), mla=mla)
        return MLAExecutionEngine(MLAModelWeights(embedding, [layer], ones(hidden), head),
            MLAConfig(kv_lora_rank=kv_rank, q_lora_rank=q_rank, qk_nope_head_dim=nope,
                qk_rope_head_dim=rope, v_head_dim=value, num_heads=heads, num_kv_heads=heads))
    raise ValueError(f"Unknown CPU test family: {family}")
