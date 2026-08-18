"""Executable Pass 18/R9 CPU contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from aether.compiler.config import CompilerConfig
from aether.compiler.stage2_optimizer.pass18_mdlm_drafter import (
    MDLMDrafterCompilationPass,
    load_mdlm_weight_bundle,
)
from aether.runtime.r9_diffusion_spec_engine import DiffusionSpecEngine


def _bundle(vocab: int = 8, hidden: int = 4, draft_hidden: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(21)
    return {
        "token_embedding": rng.normal(size=(vocab, draft_hidden)).astype(np.float32),
        "context_projection": rng.normal(size=(hidden, draft_hidden)).astype(np.float32),
        "output_projection": rng.normal(size=(draft_hidden, vocab)).astype(np.float32),
        "output_bias": np.zeros(vocab, dtype=np.float32),
        "time_embedding": rng.normal(size=(7, draft_hidden)).astype(np.float32),
    }


def test_pass18_persists_and_r9_executes_real_cpu_head(tmp_path):
    weights = _bundle()
    weight_path = tmp_path / "mdlm.npz"
    np.savez(weight_path, **weights)
    loaded = load_mdlm_weight_bundle(weight_path)

    graph = SimpleNamespace(metadata={}, output_dir=str(tmp_path), mdlm_drafter_weights=loaded)
    config = CompilerConfig(
        enable_mdlm_drafter=True,
        mdlm_drafter_steps=6,
        mdlm_draft_block_size=3,
    )
    _, report = MDLMDrafterCompilationPass().run(
        graph,
        {"vocab_size": 8, "hidden_size": 4},
        config,
    )
    assert report.status == "applied"
    assert (tmp_path / "diffusion" / "schedule.json").is_file()
    assert (tmp_path / "graph" / "mdlm_draft_head.npz").is_file()
    assert (tmp_path / "graph" / "mdlm_draft_head_config.json").is_file()

    # A fresh engine must load executable weights from disk, not compiler
    # process state or a synthetic fallback.
    engine = DiffusionSpecEngine(
        vocab_size=8,
        hidden_size=4,
        initial_K=3,
        initial_T=6,
        mask_token_id=99,
    )
    assert engine.load_from_aeg(str(tmp_path)) is True
    draft = engine.draft(np.ones((2, 4), dtype=np.float32), position=0)
    assert len(draft.tokens) == 3
    assert len(draft.logits) == 3
    assert all(0 <= token < 8 for token in draft.tokens)

    target_logits = [[0.0] * 8 for _ in draft.tokens]
    accepted, accepted_count = engine.verify(draft, target_logits)
    assert accepted
    assert accepted_count == len(accepted)
    assert engine.get_stats()["draft_head_loaded"] is True


def test_pass18_rejects_incomplete_bundle(tmp_path):
    graph = SimpleNamespace(
        metadata={},
        output_dir=str(tmp_path),
        mdlm_drafter_weights={"token_embedding": np.zeros((8, 3), dtype=np.float32)},
    )
    report = MDLMDrafterCompilationPass().run(
        graph,
        {"vocab_size": 8, "hidden_size": 4},
        CompilerConfig(enable_mdlm_drafter=True),
    )[1]
    assert report.status == "failed"
    assert "missing required tensors" in report.details["error"]
