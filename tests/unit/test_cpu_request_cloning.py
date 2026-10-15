"""Real request-local cloning preserves dense execution numerics and untouched storage."""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_aeg_onnx_export import make_engine

from aether.runtime.cpu_engine import CPUExecutionEngine, ExpertWeights


@pytest.mark.parametrize(
    "variant",
    [
        "learned",
        "alibi",
        "scaled_sandwich",
        "partial_interleaved",
        "qk_post",
        "exact_geglu",
    ],
)
def test_zero_task_delta_preserves_real_prefill_and_decode(variant):
    """A materialized zero delta must not change any computed token state."""
    base = make_engine(variant)
    selected = base.with_task_deltas(
        {"layer_0_q_proj": np.zeros_like(base.weights.layers[0].q_proj)}
    )
    prefix = np.asarray([1, 2, 3], dtype=np.int64)
    expected, expected_cache = base.forward(prefix)
    actual, actual_cache = selected.forward(prefix)
    np.testing.assert_array_equal(actual, expected)
    expected, _ = base.forward(np.asarray([4], dtype=np.int64), cache=expected_cache)
    actual, _ = selected.forward(np.asarray([4], dtype=np.int64), cache=actual_cache)
    np.testing.assert_array_equal(actual, expected)


def test_clone_preserves_all_weight_fields_and_shares_only_untouched_tensors():
    """Future dataclass fields survive cloning without model-sized copies."""
    base = make_engine("scaled_sandwich")
    weights = replace(
        base.weights,
        attention_scale=0.9,
        attention_scale_by_layer_index=True,
        rope_scaling={"rope_type": "linear", "factor": 2.0},
        original_context_length=8,
        no_rope_layers=[1],
        gelu_approximate=False,
        moe_renormalize_topk=False,
    )
    base = CPUExecutionEngine(weights, base.num_heads, base.num_kv_heads, kernels=base.kernels)
    selected = base.with_task_deltas({"layer_0_q_proj": np.zeros_like(weights.layers[0].q_proj)})
    for item in fields(weights):
        if item.name != "layers":
            assert getattr(selected.weights, item.name) is getattr(weights, item.name)
    for index, (original, cloned) in enumerate(
        zip(weights.layers, selected.weights.layers, strict=True)
    ):
        for item in fields(original):
            before, after = getattr(original, item.name), getattr(cloned, item.name)
            if index == 0 and item.name == "q_proj":
                assert not np.shares_memory(before, after)
                np.testing.assert_array_equal(after, before)
            else:
                assert after is before


@pytest.mark.parametrize(
    "tensor_name",
    [
        "q_norm",
        "k_norm",
        "post_attention_norm",
        "post_ffn_norm",
        "post_attention_norm_bias",
        "post_ffn_norm_bias",
    ],
)
def test_requested_norm_delta_matches_explicit_reference(tensor_name):
    """Previously omitted canonical tensors must be applied, not silently ignored."""
    base = make_engine("qk_post" if tensor_name in {"q_norm", "k_norm"} else "scaled_sandwich")
    layer = base.weights.layers[0]
    if getattr(layer, tensor_name) is None:
        setattr(layer, tensor_name, np.zeros(base.weights.hidden_size, np.float32))
    original = getattr(layer, tensor_name).copy()
    delta = np.linspace(0.03, 0.25, original.size, dtype=np.float32)
    selected = base.with_task_deltas({f"layer_0_{tensor_name}": delta})
    layers = [replace(layer, **{tensor_name: original + delta}), *base.weights.layers[1:]]
    expected = CPUExecutionEngine(
        replace(base.weights, layers=layers),
        base.num_heads,
        base.num_kv_heads,
        kernels=base.kernels,
    )
    for ids in (np.asarray([1, 2, 3]), np.asarray([4])):
        actual_logits, _ = selected.forward(ids)
        expected_logits, _ = expected.forward(ids)
        np.testing.assert_array_equal(actual_logits, expected_logits)
    np.testing.assert_array_equal(getattr(layer, tensor_name), original)


def test_moe_clone_retains_expert_biases_and_updates_named_bias():
    """Routed experts use the same field-preserving copy-on-write operation."""
    base = make_engine("scaled_sandwich")
    layer = base.weights.layers[0]
    hidden = base.weights.hidden_size
    intermediate = layer.gate_proj.shape[0]
    expert = ExpertWeights(
        layer.gate_proj,
        layer.up_proj,
        layer.down_proj,
        gate_proj_bias=np.linspace(0.01, 0.08, intermediate, dtype=np.float32),
        up_proj_bias=np.linspace(0.02, 0.09, intermediate, dtype=np.float32),
        down_proj_bias=np.linspace(0.03, 0.1, hidden, dtype=np.float32),
    )
    layer = replace(
        layer,
        gate_proj=None,
        up_proj=None,
        down_proj=None,
        router=np.ones((1, hidden), dtype=np.float32),
        experts=[expert],
    )
    base = CPUExecutionEngine(
        replace(base.weights, layers=[layer, base.weights.layers[1]]),
        base.num_heads,
        base.num_kv_heads,
        kernels=base.kernels,
    )
    delta = np.linspace(0.04, 0.12, hidden, dtype=np.float32)
    selected = base.with_task_deltas({"layer_0_expert_0_down_proj_bias": delta})
    reference_layer = replace(
        layer, experts=[replace(expert, down_proj_bias=expert.down_proj_bias + delta)]
    )
    reference = CPUExecutionEngine(
        replace(base.weights, layers=[reference_layer, base.weights.layers[1]]),
        base.num_heads,
        base.num_kv_heads,
        kernels=base.kernels,
    )
    for ids in (np.asarray([1, 2, 3]), np.asarray([4])):
        actual, _ = selected.forward(ids)
        expected, _ = reference.forward(ids)
        np.testing.assert_array_equal(actual, expected)
    assert selected.weights.layers[0].experts[0].gate_proj_bias is expert.gate_proj_bias
    assert selected.weights.layers[0].experts[0].up_proj_bias is expert.up_proj_bias
    assert selected.weights.layers[0].experts[0].gate_proj is expert.gate_proj


@pytest.mark.parametrize(
    "variant", ["scaled_sandwich", "partial_interleaved", "qk_post", "exact_geglu"]
)
def test_zero_lora_clone_preserves_actual_numerics_and_runtime_plans(variant):
    """Selecting a zero-effect adapter retains all source execution semantics."""
    base = make_engine(variant)
    q_proj = base.weights.layers[0].q_proj
    adapters = {
        "zero": {
            (0, "q_proj"): (
                np.ones((1, q_proj.shape[1]), np.float32),
                np.zeros((q_proj.shape[0], 1), np.float32),
                1.0,
            )
        }
    }
    base.sparse_attention_plan = {"enabled": False}
    base.semantic_kv_plan = {
        "format": "aether_kv_compression_v1",
        "strategy": "chunk",
        "layers": [{"retention_ratio": 1.0, "chunk_size": 4}] * len(base.weights.layers),
    }
    base.cross_layer_kv_plan = {
        "format": "aether_cross_layer_kv_v1",
        "n_layers": len(base.weights.layers),
        "sharing_groups": [],
    }
    selected = base.with_lora_adapter(adapters, "zero")
    assert selected.weights is base.weights
    assert selected.kernels is base.kernels
    for name in ("sparse_attention_plan", "semantic_kv_plan", "cross_layer_kv_plan"):
        assert getattr(selected, name) is getattr(base, name)
    expected, expected_cache = base.forward(np.asarray([1, 2, 3]))
    actual, actual_cache = selected.forward(np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(actual, expected)
    expected, _ = base.forward(np.asarray([4]), cache=expected_cache)
    actual, _ = selected.forward(np.asarray([4]), cache=actual_cache)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "name,delta",
    [
        ("layer_0_nonexistent", np.zeros(1)),
        ("layer_0_q_norm", np.zeros(4)),
        ("context_length", np.asarray(4)),
        ("lm_head", np.full((29, 16), np.nan)),
        ("lm_head", np.full((29, 16), np.inf)),
        ("lm_head", np.full((29, 16), 1e308)),
    ],
)
def test_invalid_or_nonfinite_delta_fails_before_execution(name, delta):
    """A malformed direct engine request cannot silently select a different model."""
    base = make_engine()
    with pytest.raises(ValueError):
        base.with_task_deltas({name: delta})


def test_finite_delta_addition_cannot_overflow_output():
    """Finite inputs still fail if their sum does not fit the execution dtype."""
    base = make_engine()
    base.weights.lm_head = np.full_like(base.weights.lm_head, np.finfo(np.float32).max)
    with pytest.raises(ValueError, match="overflowed"):
        base.with_task_deltas({"lm_head": base.weights.lm_head.copy()})
