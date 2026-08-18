"""Public SDK extension contract tests with real local runtime objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from aether import AetherClient


def test_sdk_exposes_runtime_extension_methods() -> None:
    client = AetherClient("missing-model.aeg", enable_safety=False)
    for name in (
        "generate_with_tools",
        "generate_video",
        "grpo_train_step",
        "get_attestation_report",
        "quantization_report",
        "semantic_cache_stats",
        "kv_transfer_stats",
        "set_task_weights",
        "multi_agent_session",
    ):
        assert callable(getattr(client, name, None)), name


def test_sdk_grpo_returns_explicit_unavailable_training_result() -> None:
    result = AetherClient("missing-model.aeg", enable_safety=False).grpo_train_step(
        ["2+2=?", "3+3=?"]
    )
    assert result["status"] == "failed"
    assert "gradient" in result["error"].lower()


def test_sdk_attestation_is_explicit_when_tee_is_not_loaded() -> None:
    report = AetherClient("missing-model.aeg", enable_safety=False).get_attestation_report()
    assert report["enabled"] is False
    assert report["hardware_backed"] is False
    assert report["reason"]


def test_sdk_video_rejects_without_a_runtime_video_encoder(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not a video")
    with pytest.raises(Exception, match="video"):
        AetherClient("missing-model.aeg", enable_safety=False).generate_video(
            video, "describe this video"
        )

