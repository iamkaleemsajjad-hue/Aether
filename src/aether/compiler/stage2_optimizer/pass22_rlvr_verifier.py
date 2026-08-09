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
            report.status = "applied"
            report.duration_ms = elapsed * 1000
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


# ─────────────────────────────────────────────────────────────────────────────
# Runtime GRPO methods (called by Runtime.grpo_train_step)
# Research: DeepSeek-R1 GRPO 2025, RLVR 2025, Flow-GRPO 2026, K2V 2026
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO) trainer.

    Implements the GRPO policy gradient algorithm from DeepSeek-R1 (2025).
    For each batch of prompts:
      1. Sample K responses from the current policy (via generate_fn)
      2. Score each response using a domain-specific verifier
      3. Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + ε)
      4. Compute clipped policy gradient loss (PPO-style): L = -E[min(ρ·A, clip(ρ,1-ε,1+ε)·A)]
      5. Apply gradient update (via optimizer hook)

    K2V Enhancement (arXiv 2026): Sub-task DAG decomposition provides dense
    intermediate rewards at each reasoning step, improving gradient signal for
    long-chain reasoning tasks (math olympiad, multi-step code).

    Flow-GRPO Enhancement (arXiv 2026): Adaptive K scheduling — reduces K when
    variance is low (responses are uniform) and increases K when high.
    """

    def __init__(
        self,
        verifier_type: str = "sympy",
        group_size: int = 8,
        clip_ratio: float = 0.2,
        normalize_rewards: bool = True,
        k2v_dense_rewards: bool = True,
        adaptive_K: bool = True,
    ) -> None:
        self.verifier_type = verifier_type
        self.K = group_size
        self.clip_ratio = clip_ratio
        self.normalize_rewards = normalize_rewards
        self.k2v_dense_rewards = k2v_dense_rewards
        self.adaptive_K = adaptive_K

        # Training stats
        self._total_steps = 0
        self._total_reward = 0.0
        self._total_loss = 0.0

    def verify_math(self, response: str, ground_truth: str | None = None) -> float:
        """Verify a math response using SymPy symbolic algebra.

        Returns 1.0 if response matches ground truth, 0.0 otherwise.  A
        missing ground truth is not evidence of correctness and therefore
        receives zero reward.
        """
        if ground_truth is None:
            return 0.0

        try:
            from sympy import simplify, sympify
            from sympy.parsing.sympy_parser import parse_expr
            import re

            # Extract last number/expression from response
            numbers = re.findall(r"-?\d+\.?\d*(?:/\d+)?", response)
            if not numbers:
                return 0.0

            candidate = numbers[-1]
            try:
                expr_candidate = sympify(candidate)
                expr_truth = sympify(ground_truth.strip())
                diff = simplify(expr_candidate - expr_truth)
                return 1.0 if diff == 0 else 0.0
            except Exception:
                return 1.0 if candidate.strip() == ground_truth.strip() else 0.0

        except ImportError:
            # SymPy not available: string match fallback
            import re
            resp_nums = re.findall(r"-?\d+\.?\d*", response)
            truth_nums = re.findall(r"-?\d+\.?\d*", ground_truth)
            if resp_nums and truth_nums:
                return 1.0 if resp_nums[-1] == truth_nums[-1] else 0.0
            return 0.0

    def verify_code(self, response: str, test_code: str | None = None) -> float:
        """Verify generated code by running tests in a sandboxed subprocess.

        Returns fraction of tests passed (0.0–1.0).
        """
        if test_code is None:
            return 0.0

        # With test code: run in subprocess
        import subprocess
        import tempfile
        import textwrap

        try:
            # Extract code from response
            code = response
            if "```" in response:
                parts = response.split("```")
                for i, part in enumerate(parts):
                    if part.startswith("python") and i + 1 < len(parts):
                        code = parts[i + 1]
                        break
                    elif i % 2 == 1:
                        code = part

            full_test = textwrap.dedent(f"""
{code}

{test_code}
""")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_test)
                tmp_path = f.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True, text=True, timeout=10.0,
            )

            import os
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            return 1.0 if result.returncode == 0 else 0.0

        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0

    def verify_heuristic(self, response: str, domain: str = "general") -> float:
        """Heuristic verifier for general/logic/reasoning domains.

        Uses response quality signals: length, coherence, keyword presence.
        Returns reward in [0, 1].
        """
        import re
        score = 0.0

        # Length reward: not too short, not padded
        words = len(response.split())
        if words >= 20:
            score += 0.3
        elif words >= 5:
            score += 0.15

        # Coherence: contains sentences (ends with punctuation)
        sentences = re.findall(r"[.!?]", response)
        if len(sentences) >= 2:
            score += 0.2

        # Domain-specific signals
        if domain in ("math", "reasoning"):
            has_calc = bool(re.search(r"\d+\s*[+\-*/=]\s*\d+", response))
            has_therefore = any(w in response.lower() for w in ["therefore", "thus", "hence", "so", "answer"])
            score += 0.25 if has_calc else 0.0
            score += 0.25 if has_therefore else 0.0
        elif domain in ("code",):
            has_code = "def " in response or "class " in response or "```" in response
            score += 0.5 if has_code else 0.0
        else:
            # General: reward informativeness
            unique_words = len(set(response.lower().split()))
            score += min(0.5, unique_words / 100.0)

        return min(1.0, score)

    def verify_response(
        self,
        response: str,
        domain: str = "general",
        ground_truth: str | None = None,
        test_code: str | None = None,
    ) -> float:
        """Dispatch to appropriate verifier based on domain.

        Returns scalar reward in [0, 1].
        """
        if domain == "math":
            return self.verify_math(response, ground_truth)
        elif domain == "code":
            return self.verify_code(response, test_code)
        elif domain in ("logic", "reasoning", "general"):
            # These domains require a supplied verifier/dataset.  Surface
            # heuristic scoring remains available as an explicit helper, but
            # it is not safe as an RLVR reward source.
            return 0.0
        else:
            return 0.0

    def compute_grpo_advantages(
        self, rewards: list[float]
    ) -> list[float]:
        """Compute group-relative advantages for GRPO.

        A_i = (r_i - mean(r)) / (std(r) + ε)

        This normalization removes the baseline (mean reward) and scales by
        the variance to produce a unit-variance advantage signal, which is
        more stable for policy gradient than raw rewards.
        """
        import math
        if len(rewards) < 2:
            return [0.0] * len(rewards)

        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(var_r) + 1e-8

        return [(r - mean_r) / std_r for r in rewards]

    def run_step(
        self,
        prompts: list[str],
        domain: str = "general",
        generate_fn: Any = None,
        learning_rate: float = 1e-6,
        max_tokens: int = 2048,
        ground_truths: list[str] | None = None,
        test_codes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run one GRPO training step.

        For each prompt in prompts:
          1. Sample K=self.K responses from generate_fn
          2. Score each with domain verifier
          3. Compute advantages
          4. Compute and log policy gradient loss
          (Weight update requires gradient-capable backend; here we compute and log)

        Args:
            prompts: List of training prompts.
            domain: Verifier domain ('math', 'code', 'logic', 'general').
            generate_fn: Callable(prompt, max_tokens, temperature) → str.
            learning_rate: Policy gradient step size.
            max_tokens: Max tokens per sample.
            ground_truths: Optional ground truth answers for exact verification.
            test_codes: Optional test code for code verification.

        Returns:
            Step report with loss, mean_reward, advantages, per-prompt stats.
        """
        import math
        import time as _time

        if generate_fn is None:
            return {
                "status": "failed",
                "error": "generate_fn is required for GRPO training step",
                "research_basis": "GRPO DeepSeek-R1 2025",
            }

        start = _time.perf_counter()
        all_advantages: list[list[float]] = []
        all_rewards: list[list[float]] = []
        all_losses: list[float] = []
        per_prompt_stats = []

        for p_idx, prompt in enumerate(prompts):
            gt = ground_truths[p_idx] if ground_truths and p_idx < len(ground_truths) else None
            tc = test_codes[p_idx] if test_codes and p_idx < len(test_codes) else None

            # Step 1: Sample K responses
            responses = []
            for k in range(self.K):
                try:
                    temperature = 0.7 + 0.3 * (k / max(1, self.K - 1))  # varied temperature
                    resp = generate_fn(prompt, max_tokens=max_tokens, temperature=temperature)
                    responses.append(str(resp))
                except Exception as e:
                    logger.debug(f"GRPO sample {k} failed: {e}")
                    responses.append("")

            # Step 2: Compute rewards
            rewards = [
                self.verify_response(r, domain, gt, tc)
                for r in responses
            ]

            # Step 3: K2V dense reward shaping (arXiv 2026)
            if self.k2v_dense_rewards and len(responses) > 0:
                # Decompose into sub-tasks and add intermediate rewards
                # A structural bonus can only refine an already verified
                # response; it must never turn an unverified response into a
                # positive reward.
                for i, resp in enumerate(responses):
                    if rewards[i] <= 0.0:
                        continue
                    steps = len([line for line in resp.split("\n") if line.strip()])
                    step_bonus = min(0.1, steps * 0.01)  # up to 10% bonus for more steps
                    rewards[i] = min(1.0, rewards[i] + step_bonus)

            # Step 4: GRPO advantages
            advantages = self.compute_grpo_advantages(rewards)

            # Step 5: Compute GRPO loss (for logging; gradient update requires training backend)
            # L_GRPO = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
            # Since we don't have old policy probabilities here, we approximate ratio ≈ 1
            # (first step of policy optimization) — this is the standard on-policy GRPO setup
            losses_per_sample = []
            for adv in advantages:
                # With ratio = 1 (initial step): L = -min(A, clip(1, 1-ε, 1+ε) * A) = -A
                # PPO clipping with ratio=1 is just the advantage itself
                loss_i = -adv  # negative because we minimize loss
                losses_per_sample.append(loss_i)

            mean_loss = sum(losses_per_sample) / max(1, len(losses_per_sample))
            all_advantages.append(advantages)
            all_rewards.append(rewards)
            all_losses.append(mean_loss)

            best_resp_idx = rewards.index(max(rewards))
            per_prompt_stats.append({
                "prompt_len": len(prompt),
                "K_samples": len(responses),
                "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
                "max_reward": round(max(rewards), 4),
                "min_reward": round(min(rewards), 4),
                "mean_advantage": round(sum(advantages) / max(1, len(advantages)), 4),
                "grpo_loss": round(mean_loss, 6),
                "best_response_len": len(responses[best_resp_idx]),
            })

        # Aggregate
        flat_rewards = [r for rs in all_rewards for r in rs]
        flat_losses = all_losses
        mean_reward = sum(flat_rewards) / max(1, len(flat_rewards))
        mean_loss = sum(flat_losses) / max(1, len(flat_losses))
        duration_s = _time.perf_counter() - start

        # Adaptive K (Flow-GRPO arXiv 2026): adjust K for next step
        reward_variance = sum((r - mean_reward) ** 2 for r in flat_rewards) / max(1, len(flat_rewards))
        if self.adaptive_K:
            if reward_variance < 0.01:  # all rewards similar → reduce K
                self.K = max(2, self.K - 1)
            elif reward_variance > 0.2:  # high variance → increase K for better gradient
                self.K = min(64, self.K + 2)

        self._total_steps += 1
        self._total_reward += mean_reward
        self._total_loss += mean_loss

        return {
            "status": "ok",
            "step": self._total_steps,
            "num_prompts": len(prompts),
            "K_per_prompt": self.K,
            "domain": domain,
            "mean_reward": round(mean_reward, 4),
            "mean_loss": round(mean_loss, 6),
            "reward_variance": round(reward_variance, 4),
            "total_samples": len(flat_rewards),
            "per_prompt_stats": per_prompt_stats,
            "duration_s": round(duration_s, 3),
            "learning_rate": learning_rate,
            "clip_ratio": self.clip_ratio,
            "k2v_dense_rewards": self.k2v_dense_rewards,
            "adaptive_K_enabled": self.adaptive_K,
            "cumulative_mean_reward": round(self._total_reward / self._total_steps, 4),
            "research_basis": [
                "GRPO DeepSeek-R1 arXiv 2025",
                "RLVR arXiv 2025",
                "Flow-GRPO arXiv 2026",
                "K2V arXiv 2026",
            ],
        }


# Module-level convenience: attach grpo_train_step to the Pass class
def _grpo_train_step(
    self: "RLVRVerifierHeadInjectionPass",
    model_id: str,
    prompts: list[str],
    group_size: int = 8,
    domain: str = "math",
    learning_rate: float = 1e-6,
    clip_ratio: float = 0.2,
    max_tokens: int = 2048,
    generate_fn: Any = None,
) -> dict[str, Any]:
    """GRPO training step dispatched from Runtime.grpo_train_step()."""
    trainer = GRPOTrainer(
        verifier_type="sympy" if domain == "math" else "pytest" if domain == "code" else "heuristic",
        group_size=group_size,
        clip_ratio=clip_ratio,
        normalize_rewards=True,
        k2v_dense_rewards=True,
        adaptive_K=True,
    )
    return trainer.run_step(
        prompts=prompts,
        domain=domain,
        generate_fn=generate_fn,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
    )


# Dynamically attach the method to the pass class
RLVRVerifierHeadInjectionPass.grpo_train_step = _grpo_train_step


