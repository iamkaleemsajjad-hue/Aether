"""
Pass 7 — Reasoning Graph Compiler.

Extracts a Chain-of-Thought (CoT) execution plan from reasoning models
(DeepSeek-R1, QwQ-32B, o1/o3-style models) and compiles it into the AEG
graph as explicit reasoning-graph nodes:

  - Budget tokens: maximum tokens to spend on thinking
  - Early-exit nodes: confidence-based exit from reasoning loop
  - Reflection nodes: backtracking + re-evaluation points
  - Speculative CoT: draft reasoning steps verified by target model

AEG-IR nodes emitted:
  aeg.reasoning_budget(max_thinking_tokens=int, adaptive=bool)
  aeg.reasoning_step(step_idx=int, can_exit=bool)
  aeg.reflection_checkpoint(depth=int)
  aeg.early_exit(confidence_threshold=float)

This pass is a no-op for non-reasoning models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Stable public entry point for Pass 7 within the optimizer pipeline. The
# pipeline-facing pass lives in ``optimizer``; this module additionally hosts
# the CoT graph compiler and its runtime budget controller.
from aether.compiler.stage2_optimizer.optimizer import ReasoningGraphPass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ReasoningGraphPass",
    "Pass7ReasoningGraph",
    "ReasoningGraph",
    "ReasoningBudget",
    "ReasoningBudgetController",
    "ReasoningStep",
    "CoTConfig",
    "is_reasoning_model",
    "REASONING_MODEL_FAMILIES",
    "REASONING_CONFIG_KEYS",
]


# ---------------------------------------------------------------------------
# Reasoning model detection
# ---------------------------------------------------------------------------

#: Model families known to use explicit CoT / thinking tokens.
REASONING_MODEL_FAMILIES = frozenset([
    "deepseek-r1",
    "deepseekr1",
    "deepseek-r2",
    "r1",                 # generic r1 prefix (e.g., r1-test, r1-distill)
    "qwq",
    "qvq",
    "qwen3",              # Qwen3 supports extended thinking mode
    "skywork-o1",
    "marco-o1",
    "o1",
    "o3",
    "s1",
    "reasoning",
    "thinking",           # any model with 'thinking' in the name
    "cot",               # chain-of-thought annotated models
    "extended-thinking",
])


#: Config keys that signal a reasoning model.
REASONING_CONFIG_KEYS = frozenset([
    "thinking_mode",
    "reasoning_effort",
    "max_thinking_tokens",
    "enable_thinking",
    "cot_mode",
    "budget_tokens",
])


def is_reasoning_model(model_config: dict[str, Any], model_id: str = "") -> bool:
    """Return True if the model is a reasoning model requiring Pass 7."""
    mid_lower = model_id.lower()
    for family in REASONING_MODEL_FAMILIES:
        if family in mid_lower:
            return True
    for key in REASONING_CONFIG_KEYS:
        if key in model_config:
            return True
    if model_config.get("architectures"):
        arch = " ".join(model_config["architectures"]).lower()
        for family in REASONING_MODEL_FAMILIES:
            if family in arch:
                return True
    return False


# ---------------------------------------------------------------------------
# Reasoning graph data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningBudget:
    """Budget controller for inference-time token spending."""

    # Hard maximum thinking tokens
    max_thinking_tokens: int = 32768
    # Whether to adapt budget to problem complexity (see InferenceComputeController)
    adaptive: bool = True
    # Budget levels by complexity
    budget_map: dict[str, int] = field(default_factory=lambda: {
        "simple":    512,
        "medium":    2048,
        "hard":      8192,
        "very_hard": 32768,
    })
    # Confidence threshold to trigger early exit from reasoning
    early_exit_confidence: float = 0.92
    # Maximum reflection depth (backtrack up to N times)
    max_reflection_depth: int = 3
    # Speculative CoT: accept rate threshold below which draft reasoning is discarded
    spec_cot_accept_floor: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_thinking_tokens": self.max_thinking_tokens,
            "adaptive": self.adaptive,
            "budget_map": self.budget_map,
            "early_exit_confidence": self.early_exit_confidence,
            "max_reflection_depth": self.max_reflection_depth,
            "spec_cot_accept_floor": self.spec_cot_accept_floor,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningBudget":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_model_config(cls, config: dict[str, Any]) -> "ReasoningBudget":
        max_tokens = config.get("max_thinking_tokens",
                     config.get("max_new_tokens_think", 32768))
        return cls(
            max_thinking_tokens=min(max_tokens, 65536),
            adaptive=config.get("adaptive_thinking", True),
        )


@dataclass
class ReasoningStep:
    """A compiled reasoning step node in the CoT graph."""

    step_idx: int
    can_exit: bool = False          # True if an early-exit check is compiled here
    is_reflection: bool = False     # True if this is a backtracking point
    confidence_threshold: float = 0.92
    token_budget_fraction: float = 1.0   # fraction of total budget for this step

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "can_exit": self.can_exit,
            "is_reflection": self.is_reflection,
            "confidence_threshold": self.confidence_threshold,
            "token_budget_fraction": self.token_budget_fraction,
        }


@dataclass
class ReasoningGraph:
    """
    The compiled reasoning graph embedded into the AEG package.

    Stored as .aeg/graph/reasoning_graph.json after Pass 7.
    """

    model_id: str
    budget: ReasoningBudget
    steps: list[ReasoningStep]
    supports_speculative_cot: bool = False
    speculative_cot_draft_layers: list[int] = field(default_factory=list)
    version: str = "reasoning_graph/1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "budget": self.budget.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "supports_speculative_cot": self.supports_speculative_cot,
            "speculative_cot_draft_layers": self.speculative_cot_draft_layers,
            "num_steps": len(self.steps),
        }

    def save(self, aeg_dir: str | Path) -> Path:
        out = Path(aeg_dir) / "graph" / "reasoning_graph.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Reasoning graph saved", path=str(out), steps=len(self.steps))
        return out

    @classmethod
    def load(cls, aeg_dir: str | Path) -> "ReasoningGraph":
        p = Path(aeg_dir) / "graph" / "reasoning_graph.json"
        if not p.exists():
            raise FileNotFoundError(f"Reasoning graph not found at {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            model_id=d["model_id"],
            budget=ReasoningBudget.from_dict(d["budget"]),
            steps=[ReasoningStep(**s) for s in d["steps"]],
            supports_speculative_cot=d.get("supports_speculative_cot", False),
            speculative_cot_draft_layers=d.get("speculative_cot_draft_layers", []),
            version=d.get("version", "reasoning_graph/1.0"),
        )

    @property
    def has_think_phase(self) -> bool:
        """True when the graph contains at least one reasoning step (think phase)."""
        return len(self.steps) > 0

    @property
    def budget_controller(self) -> "ReasoningBudgetController":
        """Return a runtime budget controller bound to this graph."""
        return ReasoningBudgetController(self)

    @property
    def max_tokens(self) -> int:
        """Maximum thinking tokens for this graph."""
        return self.budget.max_thinking_tokens




# ---------------------------------------------------------------------------
# Pass 7 — Optimizer pass implementation
# ---------------------------------------------------------------------------

class Pass7ReasoningGraph:
    """
    Optimizer Pass 7: Reasoning Graph Compiler.

    For reasoning models:
      1. Detects CoT architecture (thinking tokens, early-exit, reflection)
      2. Computes step decomposition based on model depth
      3. Identifies layers that can serve as speculative CoT draft layers
      4. Emits reasoning_graph.json into the AEG package
      5. Annotates the AEG-IR graph with reasoning op nodes

    For non-reasoning models: no-op (returns graph unchanged).
    """

    name = "pass7_reasoning_graph"

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        model_id: str = "",
    ) -> None:
        self.model_config = model_config or {}
        self.model_id = model_id
        self._graph: ReasoningGraph | None = None

    @property
    def reasoning_graph(self) -> ReasoningGraph | None:
        return self._graph

    def run(self, graph: Any, aeg_dir: str | Path | None = None) -> Any:
        """
        Execute Pass 7 on the AEG-IR graph.

        Args:
            graph: AEGGraph (or any object with .nodes / .layers attributes).
            aeg_dir: Output directory for saving reasoning_graph.json.

        Returns:
            Annotated graph (modified in-place or returned as-is for non-reasoning models).
        """
        if not is_reasoning_model(self.model_config, self.model_id):
            logger.debug("Pass 7: non-reasoning model — skipping")
            return graph

        logger.info("Pass 7: compiling reasoning graph", model_id=self.model_id)

        budget = ReasoningBudget.from_model_config(self.model_config)
        steps = self._build_steps(graph, budget)
        draft_layers = self._identify_draft_layers(graph)

        self._graph = ReasoningGraph(
            model_id=self.model_id,
            budget=budget,
            steps=steps,
            supports_speculative_cot=len(draft_layers) > 0,
            speculative_cot_draft_layers=draft_layers,
        )

        # Annotate graph nodes with reasoning ops
        self._annotate_graph(graph, self._graph)

        if aeg_dir is not None:
            self._graph.save(aeg_dir)

        logger.info(
            "Pass 7 complete",
            steps=len(steps),
            draft_layers=draft_layers,
            max_tokens=budget.max_thinking_tokens,
        )
        return graph

    def _build_steps(self, graph: Any, budget: ReasoningBudget) -> list[ReasoningStep]:
        """
        Decompose the reasoning process into compiled steps.

        For a typical reasoning model with N transformer layers:
        - Steps are aligned with groups of layers (every N//8 layers = one step)
        - The first and last steps always have early-exit checks
        - Every 3rd step is a reflection checkpoint
        """
        num_layers = self._get_num_layers(graph)
        num_steps = max(4, num_layers // 8)
        steps = []

        for i in range(num_steps):
            is_first = (i == 0)
            is_last  = (i == num_steps - 1)
            is_mid   = (i == num_steps // 2)
            can_exit = is_first or is_last or is_mid or (i % 4 == 0)
            is_reflection = (i % 3 == 2)

            # Earlier steps get more budget fraction (warm-up is expensive)
            frac = 1.0 / num_steps * (1.5 if i < 2 else 1.0)
            frac = min(frac, 1.0)

            steps.append(ReasoningStep(
                step_idx=i,
                can_exit=can_exit,
                is_reflection=is_reflection,
                confidence_threshold=budget.early_exit_confidence,
                token_budget_fraction=round(frac, 4),
            ))

        return steps

    def _identify_draft_layers(self, graph: Any) -> list[int]:
        """
        Identify transformer layers suitable as speculative CoT draft layers.

        For EAGLE-3 speculative CoT, the draft model uses the top N//4 layers
        to generate reasoning draft tokens. Middle layers offer best
        accuracy/speed tradeoff for draft generation.
        """
        num_layers = self._get_num_layers(graph)
        if num_layers < 4:
            return []
        start = num_layers // 4
        end   = num_layers * 3 // 4
        step  = max(1, (end - start) // 4)
        return list(range(start, end, step))[:4]

    def _get_num_layers(self, graph: Any) -> int:
        """Extract number of transformer layers from graph or config."""
        if hasattr(graph, "num_layers"):
            return int(graph.num_layers)
        if hasattr(graph, "layers"):
            return len(graph.layers)
        return int(self.model_config.get("num_hidden_layers", 32))

    def _annotate_graph(self, graph: Any, reasoning_graph: ReasoningGraph) -> None:
        """
        Annotate the AEG-IR graph with reasoning op nodes.

        Inserts virtual nodes that the runtime budget controller reads to
        decide when to trigger early exits or reflections.
        """
        if not hasattr(graph, "metadata"):
            return
        graph.metadata["reasoning_enabled"] = True
        graph.metadata["reasoning_graph"] = reasoning_graph.to_dict()
        graph.metadata["max_thinking_tokens"] = reasoning_graph.budget.max_thinking_tokens
        graph.metadata["speculative_cot"] = reasoning_graph.supports_speculative_cot


# ---------------------------------------------------------------------------
# Runtime: Reasoning Budget Controller
# ---------------------------------------------------------------------------

class ReasoningBudgetController:
    """
    Runtime reasoning budget controller.

    Runs alongside the token generation loop and enforces:
    - Hard token limit (budget.max_thinking_tokens)
    - Confidence-based early exit
    - Reflection triggers (backtrack and re-evaluate)
    - Speculative CoT acceptance/rejection
    """

    def __init__(self, reasoning_graph: ReasoningGraph) -> None:
        self.graph = reasoning_graph
        self.budget = reasoning_graph.budget
        self._tokens_spent = 0
        self._step_idx = 0
        self._reflection_count = 0
        self._last_confidence = 0.0

    @property
    def max_tokens(self) -> int:
        """Maximum thinking token budget."""
        return self.budget.max_thinking_tokens



    def should_continue(self, logits: np.ndarray | None = None) -> bool:
        """Return True if generation should continue (budget not exhausted)."""
        if self._tokens_spent >= self.budget.max_thinking_tokens:
            logger.debug("Reasoning: hard budget exhausted at %d tokens", self._tokens_spent)
            return False
        return True

    def record_token(self, logit: float | None = None) -> None:
        """Record one generated token and update internal state."""
        self._tokens_spent += 1

    def check_early_exit(self, logits: np.ndarray) -> bool:
        """
        Check if we should exit reasoning early based on confidence.

        Confidence is estimated as the softmax probability of the top-1 token.
        Returns True if we should STOP reasoning.
        """
        current_step = self._current_step()
        if current_step is None or not current_step.can_exit:
            return False

        # Compute softmax confidence
        shifted = logits - logits.max()
        exp_logits = np.exp(shifted)
        probs = exp_logits / (exp_logits.sum() + 1e-9)
        top1_prob = float(probs.max())
        self._last_confidence = top1_prob

        if top1_prob >= current_step.confidence_threshold:
            logger.debug(
                "Reasoning: early exit triggered",
                confidence=top1_prob,
                tokens=self._tokens_spent,
            )
            return True
        return False

    def check_reflection(self) -> bool:
        """Return True if a reflection (backtrack) should be triggered."""
        if self._reflection_count >= self.budget.max_reflection_depth:
            return False
        current_step = self._current_step()
        if current_step and current_step.is_reflection:
            self._reflection_count += 1
            return True
        return False

    def advance_step(self) -> None:
        """Move to the next reasoning step."""
        self._step_idx = min(self._step_idx + 1, len(self.graph.steps) - 1)

    def remaining_budget(self) -> int:
        """Tokens remaining in the thinking budget."""
        return max(0, self.budget.max_thinking_tokens - self._tokens_spent)

    def stats(self) -> dict[str, Any]:
        return {
            "tokens_spent": self._tokens_spent,
            "step_idx": self._step_idx,
            "reflection_count": self._reflection_count,
            "last_confidence": self._last_confidence,
            "remaining_budget": self.remaining_budget(),
        }

    def _current_step(self) -> ReasoningStep | None:
        steps = self.graph.steps
        if not steps:
            return None
        return steps[min(self._step_idx, len(steps) - 1)]


# ---------------------------------------------------------------------------
# CoTConfig — Chain-of-Thought configuration alias
# ---------------------------------------------------------------------------

@dataclass
class CoTConfig:
    """
    Chain-of-Thought (CoT) generation configuration.

    Alias / superset of ReasoningBudget that includes generation
    hyperparameters: temperature, top_p, and thinking token budget.

    Used by Pass7ReasoningGraph and InferenceComputeController to
    control reasoning depth.
    """

    max_thinking_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    adaptive: bool = True
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    enable_budget_forcing: bool = True
    min_thinking_tokens: int = 256

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_thinking_tokens": self.max_thinking_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "adaptive": self.adaptive,
            "think_start_token": self.think_start_token,
            "think_end_token": self.think_end_token,
            "enable_budget_forcing": self.enable_budget_forcing,
            "min_thinking_tokens": self.min_thinking_tokens,
        }

    @classmethod
    def from_reasoning_budget(cls, budget: ReasoningBudget) -> "CoTConfig":
        return cls(
            max_thinking_tokens=budget.max_thinking_tokens,
            adaptive=budget.adaptive,
        )

    def to_reasoning_budget(self) -> ReasoningBudget:
        return ReasoningBudget(
            max_thinking_tokens=self.max_thinking_tokens,
            adaptive=self.adaptive,
        )

