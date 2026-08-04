"""
Pass 22 — RLVR Verifier Head Injection.

Reinforcement Learning from Verifiable Rewards (RLVR) attaches a deterministic
verifier to the model that evaluates whether a generated solution is correct.
This avoids the need for a human-preference reward model and enables training
on domains where ground truth can be checked programmatically.

GRPO (Group Relative Policy Optimization):
  - Sample K solutions per prompt (default K=8).
  - Compute reward r_i ∈ {0, 1} from verifier for each solution.
  - Policy gradient: ∇θ Σᵢ (r_i - mean(r)) / std(r) · log π_θ(s_i).
  - No critic model needed (relative reward).

Four verifier types compiled into AEG:

1. **sympy** (math): Symbolic algebra verification via SymPy.
   - Parse LLM output as SymPy expression.
   - Compare to ground truth with simplify(LLM_ans - truth) == 0.

2. **pytest** (code): Execute Python code in sandboxed subprocess.
   - Run LLM output through provided test suite.
   - Reward = fraction of tests passing.

3. **llm_judge**: Small LM verifier (e.g., Qwen-2.5-1.5B judge).
   - Call judge model, parse "correct"/"incorrect" output.

4. **human**: Human preference feedback loop.
   - Placeholder: deferred to training pipeline.

AEG artifacts:
  - ``.aeg/training/rlvr_config.json``: verifier type, GRPO K, reward schema.
  - ``.aeg/training/verifier_head.bin``: compiled verifier weights (for LM judge).
  - ``aeg.rlvr_sample(K)``, ``aeg.rlvr_verify(type)``, ``aeg.grpo_update()`` opcodes.

Research basis:
  - GRPO (DeepSeek-R1, 2025): group relative policy optimization.
  - RLVR (2025): verifiable reward learning without preference data.
  - K2V (2026): sub-task DAG decomposition for dense reward shaping.
  - RLSVR (2026): multi-agent self-play RLHF for open-ended tasks.
  - OpenRLHF (2025): scalable RLHF framework (PPO + GRPO).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_VERIFIER_TYPES: frozenset[str] = frozenset(
    {"sympy", "pytest", "llm_judge", "human"}
)


class RLVRVerifierHeadInjectionPass(BasePass):
    """Pass 22: Inject RLVR verifier head and GRPO training opcodes into AEG.

    This pass is opt-in and intended for training workflows only.
    It annotates the model graph with RLVR sampling and verification opcodes
    that the training framework reads at fine-tuning time.
    """

    name = "rlvr_verifier_head_injection"
    description = (
        "Inject RLVR verifier head (GRPO K sampling + deterministic verification) "
        "into AEG training subgraph.  Supports sympy / pytest / llm_judge verifiers."
    )

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})

        if not config.enable_rlvr_verifier:
            return graph, report

        verifier_type = config.rlvr_verifier_type
        if verifier_type not in _SUPPORTED_VERIFIER_TYPES:
            logger.warning("Pass 22: Unknown verifier type %r. Using 'sympy'.", verifier_type)
            verifier_type = "sympy"

        K = config.rlvr_group_size

        try:
            logger.info(
                "Pass 22: Injecting RLVR verifier head (type=%s, GRPO K=%d).",
                verifier_type,
                K,
            )

            # Validate GRPO group size.
            if K < 2:
                logger.warning("Pass 22: GRPO K=%d is < 2. Setting K=2.", K)
                K = 2
            if K > 64:
                logger.warning("Pass 22: GRPO K=%d is > 64. Capping at K=64.", K)
                K = 64

            # Build RLVR config.
            verifier_config = _build_verifier_config(verifier_type, K, architecture)

            # Emit RLVR training opcodes.
            n_opcodes = _emit_rlvr_opcodes(graph, verifier_type, K, verifier_config)

            # Compute GRPO memory overhead.
            # Each GRPO step needs K × (model activations) in memory.
            # Estimate: K × 1 GPU-forward pass memory.
            memory_overhead_factor = K  # K concurrent rollouts.

            # Write RLVR training artifacts.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                _write_rlvr_artifacts(
                    output_dir=Path(graph.output_dir),
                    verifier_type=verifier_type,
                    K=K,
                    verifier_config=verifier_config,
                )

            elapsed = time.perf_counter() - start
            report.status = "ok"
            report.elapsed_s = elapsed
            report.details = {
                "verifier_type": verifier_type,
                "grpo_K": K,
                "n_opcodes_emitted": n_opcodes,
                "memory_overhead_factor": memory_overhead_factor,
                "training_mode_only": True,
                "verifier_config": verifier_config,
            }
            logger.info(
                "Pass 22 complete: RLVR %s verifier, GRPO K=%d, "
                "%d opcodes.  Memory factor: %dx.  Elapsed: %.3fs.",
                verifier_type,
                K,
                n_opcodes,
                memory_overhead_factor,
                elapsed,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 22 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)

        return graph, report


def _build_verifier_config(
    verifier_type: str,
    K: int,
    architecture: Any,
) -> dict[str, Any]:
    """Build verifier-specific configuration dict."""
    base = {
        "verifier_type": verifier_type,
        "grpo_K": K,
        "reward_schema": "binary",  # r_i ∈ {0, 1} for deterministic verifiers
        "normalize_rewards": True,
        "clip_ratio": 0.2,  # PPO-style clipping for GRPO policy update
    }

    if verifier_type == "sympy":
        base.update(
            {
                "sympy_timeout_s": 5.0,
                "simplification_method": "simplify",  # sympy.simplify
                "tolerance": 1e-6,
                "parse_as": "expression",  # or "equation", "inequality"
            }
        )
    elif verifier_type == "pytest":
        base.update(
            {
                "sandbox": "subprocess",
                "timeout_s": 10.0,
                "memory_limit_mb": 512,
                "reward_mode": "fraction_passing",  # or "all_or_nothing"
                "python_executable": "python3",
            }
        )
    elif verifier_type == "llm_judge":
        base.update(
            {
                "judge_model": "qwen2.5-1.5b-instruct",
                "judge_prompt_template": (
                    "Problem: {problem}\nAnswer: {answer}\n"
                    "Is the answer correct? Reply 'correct' or 'incorrect'."
                ),
                "parse_positive_keyword": "correct",
                "judge_temperature": 0.0,
            }
        )
    elif verifier_type == "human":
        base.update(
            {
                "feedback_endpoint": None,  # Set at training time
                "timeout_s": 60.0,
                "reward_schema": "float",  # Human can give continuous scores
            }
        )

    # K2V sub-task decomposition config (always injected).
    base["k2v"] = {
        "enabled": True,
        "max_subtasks": 8,
        "dag_decomposition": True,
        "dense_reward_shaping": True,
    }

    return base


def _emit_rlvr_opcodes(
    graph: Any,
    verifier_type: str,
    K: int,
    verifier_config: dict,
) -> int:
    """Emit RLVR training opcodes into the graph's training subgraph."""
    opcodes = [
        {
            "opcode": "aeg.rlvr_sample",
            "K": K,
            "description": "Sample K solutions per prompt for GRPO rollout",
        },
        {
            "opcode": "aeg.rlvr_verify",
            "verifier_type": verifier_type,
            "description": f"Evaluate solutions with {verifier_type} verifier",
        },
        {
            "opcode": "aeg.grpo_update",
            "K": K,
            "clip_ratio": verifier_config.get("clip_ratio", 0.2),
            "normalize": verifier_config.get("normalize_rewards", True),
            "description": "Compute GRPO policy gradient and update model weights",
        },
        {
            "opcode": "aeg.k2v_decompose",
            "max_subtasks": verifier_config["k2v"]["max_subtasks"],
            "description": "K2V sub-task DAG decomposition for dense reward shaping",
        },
    ]

    n_emitted = 0
    if hasattr(graph, "add_training_subgraph"):
        for op in opcodes:
            graph.add_training_subgraph(op)
            n_emitted += 1
    elif hasattr(graph, "metadata"):
        graph.metadata.setdefault("rlvr_opcodes", []).extend(opcodes)
        n_emitted = len(opcodes)

    return n_emitted


def _write_rlvr_artifacts(
    output_dir: Path,
    verifier_type: str,
    K: int,
    verifier_config: dict,
) -> None:
    """Write RLVR config to .aeg/training/."""
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    rlvr_config = {
        "format": "aether_rlvr_v1",
        "verifier_type": verifier_type,
        "grpo_K": K,
        "verifier_config": verifier_config,
        "opcodes": [
            "aeg.rlvr_sample",
            "aeg.rlvr_verify",
            "aeg.grpo_update",
            "aeg.k2v_decompose",
        ],
    }
    (training_dir / "rlvr_config.json").write_text(
        json.dumps(rlvr_config, indent=2), encoding="utf-8"
    )
    logger.debug("Wrote RLVR config: %s", training_dir / "rlvr_config.json")
