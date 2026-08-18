"""Runtime GRPO callback integration tests."""

from __future__ import annotations

import json

from aether.runtime import Runtime, RuntimeConfig


def test_runtime_grpo_executes_supplied_policy_and_optimizer(tmp_path):
    aeg = tmp_path / "trainable.aeg"
    (aeg / "training").mkdir(parents=True)
    (aeg / "training" / "rlvr_config.json").write_text(
        json.dumps(
            {
                "format": "aether_rlvr_v1",
                "verifier_type": "sympy",
                "grpo_K": 2,
                "verifier_config": {"clip_ratio": 0.2, "normalize_rewards": True},
            }
        ),
        encoding="utf-8",
    )
    (aeg / "manifest.json").write_text("{}", encoding="utf-8")
    sampled = []
    optimized = []

    def policy(*, prompt, max_new_tokens, temperature, sample_idx):
        sampled.append((prompt, sample_idx))
        return "4" if sample_idx == 0 else "3"

    def optimizer(loss):
        optimized.append(float(loss))
        return 0.25

    runtime = Runtime(RuntimeConfig(enable_semantic_cache=False))
    report = runtime.grpo_train_step(
        str(aeg),
        ["solve"],
        group_size=2,
        domain="math",
        ground_truths=["4"],
        model_forward_fn=policy,
        optimizer_step_fn=optimizer,
    )
    assert report["status"] == "ok"
    assert report["optimizer_steps"] == 1
    assert len(sampled) == 2
    assert optimized == [0.0]
    assert report["steps"][0]["pass_at_k"] == 1.0


def test_runtime_grpo_without_gradient_backend_fails_closed(tmp_path):
    runtime = Runtime(RuntimeConfig(enable_semantic_cache=False))
    report = runtime.grpo_train_step(str(tmp_path / "model.aeg"), ["solve"], group_size=2)
    assert report["status"] == "failed"
    assert "model_forward_fn" in report["error"]
