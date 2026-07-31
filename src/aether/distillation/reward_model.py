"""Process Reward Model and reasoning chain distillation (Section 33 — v3.1 Elite).

Extends the distillation pipeline with:
- ProcessRewardModel: step-level quality scorer for chain-of-thought
- ReasoningChainAligner: aligns teacher reasoning chains with student outputs
- SelfDistillationConfig: ICL-based self-teacher configuration (SDFT 2026)

Research:
- Let's Verify Step by Step (2023) — PRM foundation
- Math-Shepherd (2024) — automated PRM
- OmegaPRM (2025) — MCTS-based PRM data collection
- DeepSeek-R1 Distillation (2025) — reasoning chain transfer
- SDFT (2026) — Self-Distillation via Fine-Tuning, autonomous optimization
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Step parsing utilities
# ---------------------------------------------------------------------------

def _parse_reasoning_steps(response: str) -> list[str]:
    """
    Parse a chain-of-thought response into individual reasoning steps.

    Handles common CoT formats:
    - Numbered lists (1. ... 2. ...)
    - Bullet points (- ... - ...)
    - <think>...</think> blocks (DeepSeek-R1 style)
    - Step: ... delimiters
    """
    # Strip <think> tags if present (DeepSeek-R1 format)
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        response = think_match.group(1).strip()

    # Try numbered steps first: "1. ...", "Step 1:", etc.
    numbered = re.split(r"\n(?:\d+\.\s+|Step\s+\d+:?\s*)", response.strip())
    if len(numbered) > 1:
        return [s.strip() for s in numbered if s.strip()]

    # Try bullet points
    bullets = re.split(r"\n[-•*]\s+", response.strip())
    if len(bullets) > 1:
        return [s.strip() for s in bullets if s.strip()]

    # Fallback: split on double newlines (paragraph-level steps)
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    # Single step
    return [response.strip()] if response.strip() else []


# ---------------------------------------------------------------------------
# Process Reward Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepScore:
    """Score for a single reasoning step."""

    step_idx: int
    step_text: str
    score: float          # [0, 1] — higher is better
    is_correct: bool
    error_type: str | None = None   # "logical_error" | "math_error" | "hallucination" | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "step_text": self.step_text[:120] + "..." if len(self.step_text) > 120 else self.step_text,
            "score": round(self.score, 4),
            "is_correct": self.is_correct,
            "error_type": self.error_type,
        }


class ProcessRewardModel:
    """
    Step-level quality scorer for chain-of-thought reasoning.

    Scores individual reasoning steps in a response, then aggregates using
    the conservative minimum (a chain is only as strong as its weakest step).

    The default scorer uses heuristic signals (suitable for testing and
    offline evaluation). In production, replace `step_scorer` with a fine-tuned
    reward model (e.g., Qwen3-0.6B fine-tuned on Math-Shepherd data).

    Research:
    - Let's Verify Step by Step (Lightman et al. 2023) — PRM foundation paper
    - Math-Shepherd (2024) — automated PRM data annotation
    - OmegaPRM (2025) — MCTS-based PRM training data collection
    """

    # Heuristic signals for step quality (lexical patterns)
    CORRECTNESS_SIGNALS: list[tuple[str, float]] = [
        (r"\btherefore\b|\bthus\b|\bhence\b",             +0.10),   # Logical conclusion markers
        (r"\bbecause\b|\bsince\b|\bgiven that\b",          +0.08),   # Causal reasoning
        (r"\d+\s*[+\-×÷*/]\s*\d+\s*=\s*\d+",             +0.12),   # Math computation
        (r"\bwe know that\b|\brecall that\b",              +0.06),   # Knowledge recall
        (r"\bcontradiction\b|\bincorrect\b|\bwrong\b",     -0.15),   # Error detection
        (r"\bi think\b|\bmaybe\b|\bperhaps\b|\bpossibly\b",-0.10),   # Hedging (uncertainty)
        (r"\bactually\b|\bwait\b|\blet me reconsider\b",   -0.08),   # Self-correction (neutral-to-negative)
        (r"=\s*(?:undefined|error|nan|∞)",                 -0.20),   # Mathematical errors
    ]

    def __init__(self, min_step_score: float = 0.0, max_step_score: float = 1.0) -> None:
        self.min_step_score = min_step_score
        self.max_step_score = max_step_score

    def step_scorer(self, prompt: str, step: str) -> float:
        """
        Score a single reasoning step in [0, 1].

        In production: call a fine-tuned PRM model here.
        Default: heuristic lexical scorer.
        """
        score = 0.5  # Base score (neutral)
        for pattern, delta in self.CORRECTNESS_SIGNALS:
            if re.search(pattern, step, re.IGNORECASE):
                score += delta

        # Penalize very short or very long steps (likely incomplete or repetitive)
        word_count = len(step.split())
        if word_count < 5:
            score -= 0.15
        elif word_count > 300:
            score -= 0.08

        return round(min(self.max_step_score, max(self.min_step_score, score)), 4)

    def score(self, prompt: str, response: str) -> float:
        """
        Score a full chain-of-thought response.

        Returns the minimum step score (conservative — a chain is only as
        strong as its weakest link). This matches the Math-Shepherd approach.
        """
        steps = _parse_reasoning_steps(response)
        if not steps:
            return 0.0
        step_scores = [self.step_scorer(prompt, step) for step in steps]
        return min(step_scores)  # Conservative: minimum step quality

    def score_detailed(self, prompt: str, response: str) -> list[StepScore]:
        """Score each step individually and return detailed breakdown."""
        steps = _parse_reasoning_steps(response)
        results = []
        for i, step in enumerate(steps):
            s = self.step_scorer(prompt, step)
            error_type = None
            if re.search(r"=\s*(?:undefined|error|nan)", step, re.IGNORECASE):
                error_type = "math_error"
            elif re.search(r"\bcontradiction\b|\bincorrect\b", step, re.IGNORECASE):
                error_type = "logical_error"
            elif re.search(r"\bi think\b|\bmaybe\b", step, re.IGNORECASE) and s < 0.4:
                error_type = "hallucination"

            results.append(StepScore(
                step_idx=i,
                step_text=step,
                score=s,
                is_correct=s >= 0.5,
                error_type=error_type,
            ))
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "prm/1.0",
            "type": "heuristic_lexical",
            "aggregation": "minimum_step_score",
            "min_step_score": self.min_step_score,
            "max_step_score": self.max_step_score,
            "research": ["LetsVerifyStepByStep:2023", "MathShepherd:2024", "OmegaPRM:2025"],
        }


# ---------------------------------------------------------------------------
# Reasoning chain aligner
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Result of aligning a student response to a teacher reasoning chain."""

    teacher_steps: list[str]
    student_steps: list[str]
    step_scores: list[float]
    alignment_score: float         # Overall alignment quality [0, 1]
    missing_steps: list[int]       # Indices of teacher steps not in student
    extra_steps: list[int]         # Indices of student steps with no teacher match

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_teacher_steps": len(self.teacher_steps),
            "num_student_steps": len(self.student_steps),
            "step_scores": [round(s, 4) for s in self.step_scores],
            "alignment_score": round(self.alignment_score, 4),
            "missing_steps": self.missing_steps,
            "extra_steps": self.extra_steps,
        }


class ReasoningChainAligner:
    """
    Aligns student reasoning chains to teacher chains for distillation.

    Computes step-level alignment between teacher and student CoT traces,
    identifying missing steps, extra steps, and alignment quality.

    Research: DeepSeek-R1 Distillation (2025), Feature-Based Distillation (IEEE 2025).
    """

    def __init__(self, prm: ProcessRewardModel | None = None) -> None:
        self.prm = prm or ProcessRewardModel()

    def _step_similarity(self, step_a: str, step_b: str) -> float:
        """
        Compute lexical similarity between two reasoning steps.

        Uses word overlap (Jaccard similarity) as a fast proxy.
        In production, replace with embedding cosine similarity.
        """
        words_a = set(step_a.lower().split())
        words_b = set(step_b.lower().split())
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def align(
        self,
        prompt: str,
        teacher_response: str,
        student_response: str,
        similarity_threshold: float = 0.25,
    ) -> AlignmentResult:
        """
        Align student reasoning to teacher reasoning chain.

        Args:
            prompt: Original prompt (for PRM context).
            teacher_response: Teacher model's full CoT response.
            student_response: Student model's full CoT response.
            similarity_threshold: Minimum similarity for step match.

        Returns:
            AlignmentResult with step-level alignment scores.
        """
        teacher_steps = _parse_reasoning_steps(teacher_response)
        student_steps = _parse_reasoning_steps(student_response)

        if not teacher_steps:
            teacher_steps = [teacher_response]
        if not student_steps:
            student_steps = [student_response]

        # Greedy alignment: for each teacher step, find best-matching student step
        matched_student = set()
        step_scores: list[float] = []
        missing: list[int] = []

        for t_idx, t_step in enumerate(teacher_steps):
            best_score = 0.0
            best_s_idx = -1
            for s_idx, s_step in enumerate(student_steps):
                if s_idx in matched_student:
                    continue
                sim = self._step_similarity(t_step, s_step)
                if sim > best_score:
                    best_score = sim
                    best_s_idx = s_idx

            if best_score >= similarity_threshold and best_s_idx >= 0:
                matched_student.add(best_s_idx)
                step_scores.append(best_score)
            else:
                step_scores.append(0.0)
                missing.append(t_idx)

        extra = [i for i in range(len(student_steps)) if i not in matched_student]
        alignment_score = sum(step_scores) / max(len(teacher_steps), 1)

        return AlignmentResult(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            step_scores=step_scores,
            alignment_score=alignment_score,
            missing_steps=missing,
            extra_steps=extra,
        )


# ---------------------------------------------------------------------------
# Self-distillation configuration (SDFT 2026)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelfDistillationConfig:
    """
    Self-Distillation via Fine-Tuning (SDFT) configuration.

    Uses the model itself as the teacher via in-context learning (ICL):
    the larger-context version of the same model generates high-quality
    examples that are then used to fine-tune the base model.

    Research: SDFT (2026) — autonomous optimization, 5-30x cost reduction
    vs external teacher distillation with 95-97% quality retention.
    """

    teacher_context_length: int = 32768    # Teacher uses long context for ICL
    student_context_length: int = 4096    # Student operates in shorter context
    icl_examples_per_task: int = 8        # Number of ICL demonstrations
    temperature: float = 0.8              # Sampling temperature for self-generation
    num_generations: int = 4              # Candidates per prompt (best kept)
    selection_strategy: str = "prm_top1" # "prm_top1" | "majority_vote" | "min_edit"
    anti_forgetting: bool = True          # Mix original data to prevent forgetting

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "sdft/1.0",
            "method": "self_distillation_via_fine_tuning",
            "teacher_context_length": self.teacher_context_length,
            "student_context_length": self.student_context_length,
            "icl_examples_per_task": self.icl_examples_per_task,
            "temperature": self.temperature,
            "num_generations": self.num_generations,
            "selection_strategy": self.selection_strategy,
            "anti_forgetting": self.anti_forgetting,
            "expected_quality_retention": "95-97%",
            "cost_reduction_vs_external_teacher": "5-30x",
            "research": "SDFT:2026 — Self-Distillation via Fine-Tuning (ICL teacher)",
        }

    @classmethod
    def for_reasoning_model(cls) -> "SelfDistillationConfig":
        """Config optimized for reasoning models (R1/QwQ style)."""
        return cls(
            teacher_context_length=65536,
            student_context_length=8192,
            icl_examples_per_task=4,
            temperature=0.6,
            num_generations=8,
            selection_strategy="prm_top1",
            anti_forgetting=True,
        )

    @classmethod
    def for_general_model(cls) -> "SelfDistillationConfig":
        """Config optimized for general-purpose models."""
        return cls(
            teacher_context_length=16384,
            student_context_length=4096,
            icl_examples_per_task=8,
            temperature=0.8,
            num_generations=4,
            selection_strategy="majority_vote",
            anti_forgetting=True,
        )
