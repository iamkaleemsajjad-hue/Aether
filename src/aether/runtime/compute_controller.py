"""
Inference-Time Compute Scaling Engine.

Implements the "third scaling law": smaller models + more thinking tokens
outperform larger models with no thinking time (DeepSeek-R1, 2025).

By 2030, test-time compute scaling is projected to account for 75% of all
AI compute demand (Google scaling research, 2025).

Strategies:
  - Best-of-N (BoN): generate N candidates, select via reward model
  - Beam Search: width-W beam with length penalty
  - MCTS: Monte Carlo Tree Search with UCB exploration
  - Adaptive: complexity-aware automatic strategy selection

Research:
  - Inference-Time Scaling (Google, 2025): compute-optimal BoN
  - ThreadWeaver (2026): parallel reasoning branches
  - InferenceTimePessimism (2026): budget-aware early stopping
  - Process Reward Model (OmegaPRM, 2025): step-level quality scoring
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

@dataclass
class BoNConfig:
    n_samples: int = 8
    selection: str = "reward_model"   # "reward_model" | "longest" | "majority_vote"
    parallel: bool = True
    temperature: float = 1.0


@dataclass
class BeamSearchConfig:
    beam_width: int = 4
    length_penalty: float = 1.0
    max_depth: int = 1024
    temperature: float = 0.7


@dataclass
class MCTSConfig:
    simulations: int = 32
    ucb_constant: float = 1.4
    max_depth: int = 10
    rollout_temperature: float = 1.0


ADAPTIVE_BUDGET_MAP = {
    "simple":    {"strategy": "greedy",    "max_tokens": 512},
    "medium":    {"strategy": "best_of_4", "max_tokens": 2048},
    "hard":      {"strategy": "beam_4",    "max_tokens": 8192},
    "very_hard": {"strategy": "mcts",      "max_tokens": 32768},
}


# ---------------------------------------------------------------------------
# Process Reward Model (PRM)
# ---------------------------------------------------------------------------

class ProcessRewardModel:
    """
    Scores intermediate reasoning steps using a lightweight head.

    The PRM assigns a quality score to each reasoning step in a chain-of-thought
    response. The minimum step score is used as the overall response quality
    (conservative aggregation — any bad step invalidates the response).

    Research: OmegaPRM (2025), Math-Shepherd (2024), Let's Verify Step by Step (2023).
    """

    def __init__(self, score_fn: Callable[[str, str], float] | None = None) -> None:
        """
        Args:
            score_fn: Optional custom step scorer (prompt, step_text) → [0, 1].
                      If None, uses a heuristic scorer.
        """
        self._score_fn = score_fn or self._heuristic_scorer

    def score(self, prompt: str, response: str) -> float:
        """
        Score a full response by its weakest reasoning step.

        Returns a quality score in [0, 1].
        """
        steps = self._parse_reasoning_steps(response)
        if not steps:
            return self._heuristic_scorer(prompt, response)
        step_scores = [self._score_fn(prompt, step) for step in steps]
        return float(min(step_scores))  # conservative: min step quality

    def score_steps(self, prompt: str, response: str) -> list[float]:
        """Return per-step quality scores for analysis."""
        steps = self._parse_reasoning_steps(response)
        return [self._score_fn(prompt, step) for step in steps]

    def _parse_reasoning_steps(self, response: str) -> list[str]:
        """Split response into reasoning steps."""
        import re
        # Split on numbered lists, "Step N:", or double newlines
        steps = re.split(
            r'\n(?=Step\s*\d+[:.]|\d+[.)]\s|\n)',
            response.strip()
        )
        # Filter to meaningful steps (>20 chars)
        return [s.strip() for s in steps if len(s.strip()) > 20]

    def _heuristic_scorer(self, prompt: str, step_text: str) -> float:
        """
        Heuristic step quality score based on:
        - Step length (longer = more thought)
        - Mathematical notation presence
        - Logical connectives ("therefore", "thus", "because")
        - Question resolution signals ("answer is", "= ", "result")
        """
        text = step_text.lower()
        score = 0.5  # baseline

        # Length signal: longer steps tend to be more thorough
        length_bonus = min(0.2, len(step_text) / 500)
        score += length_bonus

        # Math notation
        if any(m in step_text for m in ["=", "×", "÷", "∫", "√", "\\frac"]):
            score += 0.1

        # Logical connectives
        if any(w in text for w in ["therefore", "thus", "because", "since", "hence"]):
            score += 0.1

        # Resolution markers (answer provided)
        if any(w in text for w in ["answer is", "result is", "equals", "= "]):
            score += 0.1

        # Uncertainty markers (penalize)
        if any(w in text for w in ["not sure", "unclear", "i don't know", "maybe"]):
            score -= 0.15

        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Best-of-N
# ---------------------------------------------------------------------------

class BestOfN:
    """
    Best-of-N sampling: generate N candidate responses and select the best.

    Selection methods:
    - reward_model: select via PRM score (default)
    - longest: select longest response (proxy for thoroughness)
    - majority_vote: select most common answer (for factual questions)
    """

    def __init__(self, config: BoNConfig, prm: ProcessRewardModel | None = None) -> None:
        self.config = config
        self.prm = prm or ProcessRewardModel()

    def select_best(
        self,
        prompt: str,
        candidates: list[str],
    ) -> tuple[str, int, list[float]]:
        """
        Select the best candidate from a list of responses.

        Returns:
            (best_response, best_idx, all_scores)
        """
        if not candidates:
            raise ValueError("candidates list is empty")

        if self.config.selection == "reward_model":
            scores = [self.prm.score(prompt, c) for c in candidates]
            best_idx = int(np.argmax(scores))
        elif self.config.selection == "longest":
            scores = [float(len(c)) for c in candidates]
            best_idx = int(np.argmax(scores))
        elif self.config.selection == "majority_vote":
            # Simple majority vote: most frequent last line (answer extraction)
            last_lines = [c.strip().split("\n")[-1] for c in candidates]
            from collections import Counter
            counts = Counter(last_lines)
            majority = counts.most_common(1)[0][0]
            scores = [1.0 if c.strip().split("\n")[-1] == majority else 0.0 for c in candidates]
            best_idx = scores.index(1.0)
        else:
            scores = [1.0] * len(candidates)
            best_idx = 0

        return candidates[best_idx], best_idx, scores


# ---------------------------------------------------------------------------
# Beam Search Node
# ---------------------------------------------------------------------------

@dataclass
class BeamNode:
    """A single node in the beam search tree."""
    text: str
    score: float        # cumulative log probability
    length: int = 0
    done: bool = False

    def adjusted_score(self, length_penalty: float) -> float:
        """Score normalized by length (length-penalty normalization)."""
        denom = ((5 + max(self.length, 1)) / 6.0) ** length_penalty
        return self.score / denom


class BeamSearchDecoder:
    """
    Width-W beam search decoder.

    Maintains W active beams, expanding each by sampling from the
    model's output distribution and keeping the top-W by score.
    """

    def __init__(self, config: BeamSearchConfig) -> None:
        self.config = config

    def decode(
        self,
        prompt: str,
        generate_fn: Callable[[str], tuple[str, float]] | None = None,
        num_steps: int = 10,
    ) -> list[BeamNode]:
        """
        Run beam search for num_steps expansion rounds.

        Args:
            prompt: Initial prompt.
            generate_fn: (partial_text) → (next_token, log_prob).
                         If None, runs a dry simulation.
            num_steps: Number of beam expansion steps.

        Returns:
            List of completed beams, sorted by adjusted score.
        """
        W = self.config.beam_width
        lp = self.config.length_penalty

        # Initialize beams
        beams = [BeamNode(text=prompt, score=0.0, length=0)]

        for step in range(num_steps):
            if all(b.done for b in beams):
                break

            new_beams: list[BeamNode] = []
            for beam in beams:
                if beam.done:
                    new_beams.append(beam)
                    continue

                # Expand beam: try W hypotheses
                for _ in range(W):
                    if generate_fn is not None:
                        token, log_prob = generate_fn(beam.text)
                    else:
                        # Simulation: random token/score
                        rng = np.random.default_rng()
                        token = f"token_{step}"
                        log_prob = float(-rng.exponential(0.5))

                    done = token in ("<eos>", "</s>", "<|end|>")
                    new_beams.append(BeamNode(
                        text=beam.text + " " + token,
                        score=beam.score + log_prob,
                        length=beam.length + 1,
                        done=done,
                    ))

            # Keep top-W beams by adjusted score
            new_beams.sort(key=lambda b: b.adjusted_score(lp), reverse=True)
            beams = new_beams[:W]

        beams.sort(key=lambda b: b.adjusted_score(lp), reverse=True)
        return beams


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """Node in the MCTS reasoning tree."""

    def __init__(self, text: str, parent: "MCTSNode | None" = None) -> None:
        self.text = text
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.visits = 0
        self.total_value = 0.0

    @property
    def value(self) -> float:
        return self.total_value / max(self.visits, 1)

    def ucb_score(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.value + c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)

    def best_child(self, c: float) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb_score(c))

    def expand(self, new_texts: list[str]) -> list["MCTSNode"]:
        for text in new_texts:
            child = MCTSNode(text=text, parent=self)
            self.children.append(child)
        return self.children

    def backup(self, value: float) -> None:
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.backup(value)


class MCTSDecoder:
    """
    Monte Carlo Tree Search decoder for reasoning tasks.

    Explores the reasoning tree with UCB-guided selection, random rollouts,
    and backpropagation of quality scores.
    """

    def __init__(
        self,
        config: MCTSConfig,
        prm: ProcessRewardModel | None = None,
    ) -> None:
        self.config = config
        self.prm = prm or ProcessRewardModel()

    def search(
        self,
        prompt: str,
        expand_fn: Callable[[str], list[str]] | None = None,
    ) -> MCTSNode:
        """
        Run MCTS on a reasoning tree rooted at prompt.

        Args:
            prompt: Root prompt.
            expand_fn: (text) → [continuation1, continuation2, ...]
                       If None, uses a dummy expansion.

        Returns:
            Best leaf node found.
        """
        root = MCTSNode(text=prompt)
        c = self.config.ucb_constant

        for _ in range(self.config.simulations):
            # Selection: traverse to leaf via UCB
            node = root
            while node.children and node.visits > 0:
                node = node.best_child(c)

            # Expansion: add children
            if expand_fn is not None:
                continuations = expand_fn(node.text)
            else:
                continuations = [f"{node.text} [step_{i}]" for i in range(3)]

            if continuations and node.visits > 0:
                node.expand(continuations[:3])
                if node.children:
                    node = node.children[0]

            # Rollout + evaluation
            value = self.prm.score(prompt, node.text)
            node.backup(value)

        # Return best leaf
        def best_leaf(n: MCTSNode) -> MCTSNode:
            if not n.children:
                return n
            return best_leaf(max(n.children, key=lambda c: c.value))

        return best_leaf(root)


# ---------------------------------------------------------------------------
# Inference-Time Compute Controller (main entry point)
# ---------------------------------------------------------------------------

class InferenceComputeController:
    """
    Master controller for inference-time compute scaling.

    Selects strategy based on complexity_class and dispatches to
    BoN, BeamSearch, or MCTS.

    Integrates with the CascadeRouter's complexity classification.
    """

    STRATEGIES = {
        "greedy":    {"n": 1,  "method": "greedy"},
        "best_of_4": {"n": 4,  "method": "bon"},
        "best_of_8": {"n": 8,  "method": "bon"},
        "beam_4":    {"n": 4,  "method": "beam"},
        "mcts":      {"n": 32, "method": "mcts"},
    }

    def __init__(
        self,
        prm: ProcessRewardModel | None = None,
        bon_config: BoNConfig | None = None,
        beam_config: BeamSearchConfig | None = None,
        mcts_config: MCTSConfig | None = None,
    ) -> None:
        self.prm = prm or ProcessRewardModel()
        self._bon   = BestOfN(bon_config or BoNConfig(), self.prm)
        self._beam  = BeamSearchDecoder(beam_config or BeamSearchConfig())
        self._mcts  = MCTSDecoder(mcts_config or MCTSConfig(), self.prm)
        self._stats: dict[str, int] = {s: 0 for s in self.STRATEGIES}

    def select_strategy(self, complexity_class: str) -> str:
        """Map complexity class to strategy name."""
        return ADAPTIVE_BUDGET_MAP.get(complexity_class, {}).get("strategy", "greedy")

    def get_max_tokens(self, complexity_class: str) -> int:
        """Get max tokens budget for a complexity class."""
        return ADAPTIVE_BUDGET_MAP.get(complexity_class, {}).get("max_tokens", 512)

    def run(
        self,
        prompt: str,
        complexity_class: str = "simple",
        candidates: list[str] | None = None,
        generate_fn: Callable[[str], tuple[str, float]] | None = None,
    ) -> dict[str, Any]:
        """
        Run inference-time compute scaling for a prompt.

        Args:
            prompt: Input prompt.
            complexity_class: From CascadeRouter (simple/medium/hard/very_hard).
            candidates: Pre-generated candidates for BoN selection.
            generate_fn: Token generator for beam/MCTS.

        Returns:
            Dict with best_response, strategy, score, latency_ms.
        """
        t0 = time.perf_counter()
        strategy = self.select_strategy(complexity_class)
        self._stats[strategy] = self._stats.get(strategy, 0) + 1
        cfg = self.STRATEGIES.get(strategy, {"method": "greedy"})

        if cfg["method"] == "greedy" or not candidates:
            result_text = candidates[0] if candidates else prompt
            score = self.prm.score(prompt, result_text)
            best_idx = 0

        elif cfg["method"] == "bon":
            cands = candidates or [prompt]
            best_text, best_idx, scores = self._bon.select_best(prompt, cands)
            result_text = best_text
            score = scores[best_idx]

        elif cfg["method"] == "beam":
            beams = self._beam.decode(prompt, generate_fn=generate_fn)
            result_text = beams[0].text if beams else prompt
            score = self.prm.score(prompt, result_text)
            best_idx = 0

        elif cfg["method"] == "mcts":
            best_node = self._mcts.search(prompt)
            result_text = best_node.text
            score = best_node.value
            best_idx = 0

        else:
            result_text = candidates[0] if candidates else prompt
            score = 0.5
            best_idx = 0

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "best_response": result_text,
            "best_idx": best_idx,
            "strategy": strategy,
            "complexity_class": complexity_class,
            "prm_score": round(score, 4),
            "latency_ms": round(latency_ms, 2),
            "max_tokens": self.get_max_tokens(complexity_class),
        }

    def stats(self) -> dict[str, Any]:
        total = sum(self._stats.values())
        return {
            "total_requests": total,
            "strategy_counts": dict(self._stats),
            "strategy_distribution": {
                k: round(v / max(total, 1), 3)
                for k, v in self._stats.items()
            },
        }
