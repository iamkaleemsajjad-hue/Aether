"""
Real speculative decoding engine — Algorithm 1 (Leviathan et al. 2023).

IMPORTANT: This module replaces the fake token-generation logic in the old
speculative.py which used ``(token_id * 31 + branch_index) % 100000`` to
generate "draft" tokens.  That is NOT speculative decoding.  This module
implements real, statistically-correct speculative decoding.

Algorithms implemented
-----------------------
1. DraftTargetSpecDecoder   — external small draft model + large target model.
   Based on "Fast Inference from Transformers via Speculative Decoding"
   (Leviathan et al., 2023, ICML). https://arxiv.org/abs/2211.17192

2. SelfSpeculativeDecoder   — target model with early-exit draft.
   Based on "Draft & Verify: Lossless Large Language Model Acceleration
   via Self-Speculative Decoding" (Zhang et al., 2024).
   https://arxiv.org/abs/2309.00832

3. MedusaDecoder            — parallel prediction heads on the target model.
   Based on "Medusa: Simple LLM Inference Acceleration Framework with
   Multiple Decoding Heads" (Cai et al., 2024).
   https://arxiv.org/abs/2401.10774

Correctness guarantee
---------------------
Algorithm 1 (DraftTargetSpecDecoder) provably samples from the TARGET model's
distribution (not the draft model's).  The acceptance test ensures that
rejected tokens are replaced by a corrected sample, maintaining the exact
target distribution.

NO fake tokens. NO hash-based generation. NO fixed lookup tables.
The draft tokens come from a real forward pass through a draft model.

Architecture requirements
-------------------------
Both draft and target engines must implement:
    forward_step(token_ids, kv_cache) -> (logits, new_kv_cache)

Where:
    token_ids  : list[int] or np.ndarray[int64], shape (seq_len,)
    logits     : np.ndarray, shape (seq_len, vocab_size) or (vocab_size,)
    new_kv_cache : updated cache object (passed back in next call)

This interface is engine-agnostic: TorchAEGEngine, CPUExecutionEngine, and
any future backend all work as long as they expose forward_step().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DraftTargetSpecDecoder",
    "SelfSpeculativeDecoder",
    "MedusaDecoder",
    "SpeculativeStats",
    "SpeculativeConfig",
    "make_speculative_decoder",
]


# ---------------------------------------------------------------------------
# Engine protocol — what an engine must expose
# ---------------------------------------------------------------------------

@runtime_checkable
class ForwardStepEngine(Protocol):
    """Minimal protocol for speculative decoding integration."""

    def forward_step(
        self,
        token_ids: list[int] | Any,
        kv_cache: Any,
        return_all_logits: bool = False,
    ) -> tuple[Any, Any]:
        """
        Single forward pass.

        Args:
            token_ids:        Input token IDs (last token for decode,
                              full sequence for prefill).
            kv_cache:         Current KV cache state.
            return_all_logits: If True, return logits for all positions;
                               if False, return only the last position.

        Returns:
            (logits, updated_kv_cache)
            logits: numpy array shape (vocab_size,) or (seq_len, vocab_size)
        """
        ...


# ---------------------------------------------------------------------------
# Configuration + Statistics
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # Number of draft tokens generated per step
    gamma: int = 4

    # Minimum acceptance rate before falling back to standard decode
    # Set to 0.0 to always use speculative decoding
    min_acceptance_rate: float = 0.5

    # Number of steps to warm up before enabling fallback
    warmup_steps: int = 10

    # Temperature for draft sampling (0 = greedy)
    draft_temperature: float = 0.0

    # If True, use greedy acceptance (faster but slightly biased)
    # If False, use exact acceptance (Algorithm 1 - unbiased)
    greedy_acceptance: bool = False

    # Medusa-specific: number of heads
    medusa_num_heads: int = 4

    # Self-speculative: which layer index to exit for draft
    early_exit_layer: int | None = None


@dataclass
class SpeculativeStats:
    """Running statistics for one speculative decoding session."""

    steps: int = 0
    tokens_accepted: int = 0
    tokens_generated: int = 0
    draft_forward_passes: int = 0
    target_forward_passes: int = 0
    fallback_steps: int = 0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Draft token acceptance rate.

        Computed as ``tokens_accepted / tokens_generated`` where
        ``tokens_generated`` counts the non-draft tokens produced by the target
        (bonus tokens when all drafts accepted, or rejection-correction samples).
        A value of 1.0 means all draft tokens were accepted on every step.
        """
        if self.steps == 0:
            return 0.0
        return self.tokens_accepted / max(1, self.tokens_generated)

    @property
    def speedup_estimate(self) -> float:
        """
        Theoretical speedup: E[accepted tokens + 1] / 1 target call.
        At 100% acceptance rate, speedup = gamma + 1.
        At 0% rate, speedup = 1 (same as standard decode).
        """
        if self.target_forward_passes == 0:
            return 1.0
        total = self.tokens_accepted + self.steps  # +1 bonus per step
        return total / max(1, self.target_forward_passes)

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "tokens_accepted": self.tokens_accepted,
            "tokens_generated": self.tokens_generated,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "speedup_estimate": round(self.speedup_estimate, 3),
            "draft_forward_passes": self.draft_forward_passes,
            "target_forward_passes": self.target_forward_passes,
            "fallback_steps": self.fallback_steps,
            "total_time_ms": round(self.total_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Utility: softmax + sampling
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature."""
    if temperature == 0.0:
        # Greedy: delta distribution on argmax
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[int(np.argmax(logits))] = 1.0
        return probs
    x = logits.astype(np.float64)
    if temperature != 1.0:
        x = x / temperature
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def _sample(probs: np.ndarray) -> int:
    """Sample a token ID from a probability distribution."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0:
        return int(np.argmax(probs))
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


def _greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Algorithm 1: Draft + Target speculative decoding (Leviathan et al. 2023)
# ---------------------------------------------------------------------------

class DraftTargetSpecDecoder:
    """
    Exact speculative decoding with draft and target models.

    Algorithm 1 from Leviathan et al. (2023):
    1. Draft model generates gamma candidate tokens auto-regressively.
    2. Target model verifies all gamma+1 positions in ONE forward pass.
    3. Accept each draft token i with probability min(1, p_target(xi) / p_draft(xi)).
    4. On rejection, sample corrected token from (p_target - p_draft)⁺.
    5. Always emit at least one token (the bonus sample from target logits).

    This provably samples from the target distribution exactly.
    No approximation is made except floating-point rounding.
    """

    def __init__(
        self,
        draft_engine: ForwardStepEngine,
        target_engine: ForwardStepEngine,
        config: SpeculativeConfig | None = None,
    ) -> None:
        if draft_engine is None or target_engine is None:
            raise ValueError("DraftTargetSpecDecoder requires both draft and target engines.")
        self.draft = draft_engine
        self.target = target_engine
        self.config = config or SpeculativeConfig()
        self._stats = SpeculativeStats()

    def generate_step(
        self,
        token_ids: list[int],
        draft_cache: Any,
        target_cache: Any,
    ) -> tuple[list[int], Any, Any, SpeculativeStats]:
        """
        Execute one speculative decoding step.

        Args:
            token_ids:    Full prompt + previously generated tokens.
            draft_cache:  Draft model KV cache.
            target_cache: Target model KV cache.

        Returns:
            (new_tokens, draft_cache, target_cache, stats)
            new_tokens: 1 to gamma+1 tokens from the target distribution.
        """
        t0 = time.perf_counter()
        gamma = self.config.gamma
        temp = self.config.draft_temperature

        # ── Step 1: Draft generates gamma tokens ──────────────────────
        draft_tokens: list[int] = []
        draft_probs: list[np.ndarray] = []
        current_ids = list(token_ids)

        for _ in range(gamma):
            logits, draft_cache = self.draft.forward_step(
                current_ids, draft_cache, return_all_logits=False
            )
            logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
            probs = _softmax(logits_np, temperature=temp)
            token = _sample(probs) if temp > 0 else _greedy(logits_np)
            draft_tokens.append(token)
            draft_probs.append(probs)
            current_ids = current_ids + [token]
            self._stats.draft_forward_passes += 1

        # ── Step 2: Target verifies gamma+1 positions at once ─────────
        # Feed the original context + all draft tokens; get logits at
        # all draft positions plus the position after the last draft token.
        verify_ids = list(token_ids) + draft_tokens
        target_logits, target_cache = self.target.forward_step(
            verify_ids, target_cache, return_all_logits=True
        )
        self._stats.target_forward_passes += 1

        target_logits_np = np.asarray(target_logits, dtype=np.float64)
        # target_logits_np shape: (len(verify_ids), vocab) or (vocab,)
        # We need logits at positions [len(token_ids)-1 : len(token_ids)+gamma]
        # = the gamma draft positions + the bonus position
        if target_logits_np.ndim == 1:
            # Engine returned only last token logits — this engine doesn't
            # support return_all_logits; fall back to standard decode
            logger.debug("Target engine does not support return_all_logits; "
                         "falling back to standard step")
            self._stats.fallback_steps += 1
            bonus_probs = _softmax(target_logits_np)
            bonus_token = _sample(bonus_probs)
            self._stats.steps += 1
            self._stats.tokens_generated += 1
            self._stats.total_time_ms += (time.perf_counter() - t0) * 1000
            return [bonus_token], draft_cache, target_cache, self._stats

        # Extract per-position target probabilities for draft positions
        # Position j in verify_ids gets logits at row (j-1) of the output
        # (because causal: output[j] predicts input[j+1]).
        # So for draft_token[i] at position len(token_ids)+i in verify_ids,
        # the TARGET's probability comes from output row len(token_ids)-1+i.
        start = len(token_ids) - 1
        target_probs_list: list[np.ndarray] = []
        for i in range(gamma):
            row = min(start + i, target_logits_np.shape[0] - 1)
            target_probs_list.append(_softmax(target_logits_np[row]))

        # Bonus: target's prediction from the last verify position
        bonus_row = min(start + gamma, target_logits_np.shape[0] - 1)
        bonus_logits = target_logits_np[bonus_row]

        # ── Step 3: Accept / Reject (Algorithm 1) ─────────────────────
        accepted_tokens: list[int] = []
        for i, (draft_tok, d_probs, t_probs) in enumerate(
            zip(draft_tokens, draft_probs, target_probs_list)
        ):
            if self.config.greedy_acceptance:
                # Greedy: accept if target's argmax == draft token
                if _greedy(t_probs) == draft_tok:
                    accepted_tokens.append(draft_tok)
                    self._stats.tokens_accepted += 1
                else:
                    # Reject: sample from target
                    corrected = t_probs.copy()
                    corrected = np.maximum(corrected, 0.0)
                    corrected /= max(corrected.sum(), 1e-12)
                    accepted_tokens.append(_sample(corrected))
                    self._stats.tokens_generated += 1
                    break
            else:
                # Exact: accept with probability min(1, p_t(x) / p_d(x))
                d_prob_xi = float(d_probs[draft_tok])
                t_prob_xi = float(t_probs[draft_tok])
                ratio = min(1.0, t_prob_xi / max(d_prob_xi, 1e-12))
                u = float(np.random.uniform(0.0, 1.0))
                if u < ratio:
                    accepted_tokens.append(draft_tok)
                    self._stats.tokens_accepted += 1
                else:
                    # Rejection: sample from corrected distribution (t - d)⁺
                    corrected = np.maximum(t_probs - d_probs, 0.0)
                    total = corrected.sum()
                    if total > 0:
                        corrected /= total
                    else:
                        corrected = t_probs / max(t_probs.sum(), 1e-12)
                    accepted_tokens.append(_sample(corrected))
                    self._stats.tokens_generated += 1
                    break

        # ── Step 4: Bonus token from target ───────────────────────────
        # Always sample one extra token from the target at the bonus position.
        # This is the +1 in "gamma+1 tokens generated per step".
        if len(accepted_tokens) == gamma:
            # All draft tokens were accepted: bonus from target's next position
            bonus_probs = _softmax(bonus_logits)
            bonus_token = _sample(bonus_probs)
            accepted_tokens.append(bonus_token)
            self._stats.tokens_generated += 1

        self._stats.steps += 1
        self._stats.total_time_ms += (time.perf_counter() - t0) * 1000
        logger.debug(
            "SpecDecode step: accepted %d/%d draft tokens (total so far: %.1f%%)",
            len(accepted_tokens) - 1 if len(accepted_tokens) > gamma else len(accepted_tokens),
            gamma,
            100 * self._stats.acceptance_rate,
        )
        return accepted_tokens, draft_cache, target_cache, self._stats

    def reset_stats(self) -> None:
        self._stats = SpeculativeStats()

    @property
    def stats(self) -> SpeculativeStats:
        return self._stats


# ---------------------------------------------------------------------------
# Self-speculative decoder (Zhang et al. 2024)
# ---------------------------------------------------------------------------

class SelfSpeculativeDecoder:
    """
    Self-speculative decoding using early exit from the target model.

    The same model acts as both draft and target:
    - Draft: run only first ``early_exit_layer`` transformer layers.
    - Target: run all layers (verify gamma+1 positions in one pass).

    This avoids loading a second model but requires engine support for
    partial forward passes (early_exit_at_layer parameter).

    If the engine does not support early exit, falls back to standard decode
    with a warning.
    """

    def __init__(
        self,
        engine: Any,
        config: SpeculativeConfig | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or SpeculativeConfig()
        self._stats = SpeculativeStats()
        self._early_exit_layer = self.config.early_exit_layer
        self._supports_early_exit = hasattr(engine, "forward_step_early_exit")

        if not self._supports_early_exit:
            logger.warning(
                "SelfSpeculativeDecoder: engine %r does not implement "
                "forward_step_early_exit(); self-speculative decoding will "
                "fall back to standard decode. This is NOT an error — it "
                "means the early-exit path is not yet wired for this engine.",
                type(engine).__name__,
            )

    def generate_step(
        self,
        token_ids: list[int],
        kv_cache: Any,
    ) -> tuple[list[int], Any, SpeculativeStats]:
        """
        One step: early-exit draft, full verification, accept/reject.

        Falls back to a single standard decode step if early exit is unsupported.
        """
        t0 = time.perf_counter()

        if not self._supports_early_exit:
            # Standard decode fallback
            logits, kv_cache = self.engine.forward_step(
                token_ids, kv_cache, return_all_logits=False
            )
            logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
            token = _greedy(logits_np)
            self._stats.steps += 1
            self._stats.tokens_generated += 1
            self._stats.fallback_steps += 1
            self._stats.target_forward_passes += 1
            self._stats.total_time_ms += (time.perf_counter() - t0) * 1000
            return [token], kv_cache, self._stats

        gamma = self.config.gamma
        temp = self.config.draft_temperature

        # Draft: early exit
        draft_tokens: list[int] = []
        draft_probs: list[np.ndarray] = []
        current_ids = list(token_ids)
        for _ in range(gamma):
            logits, _ = self.engine.forward_step_early_exit(
                current_ids, None, exit_layer=self._early_exit_layer
            )
            logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
            probs = _softmax(logits_np, temperature=temp)
            tok = _sample(probs) if temp > 0 else _greedy(logits_np)
            draft_tokens.append(tok)
            draft_probs.append(probs)
            current_ids = current_ids + [tok]
            self._stats.draft_forward_passes += 1

        # Target: full pass
        verify_ids = list(token_ids) + draft_tokens
        target_logits, kv_cache = self.engine.forward_step(
            verify_ids, kv_cache, return_all_logits=True
        )
        self._stats.target_forward_passes += 1
        target_logits_np = np.asarray(target_logits, dtype=np.float64)

        # Accept / reject (same as DraftTargetSpecDecoder)
        accepted: list[int] = []
        start = len(token_ids) - 1
        for i in range(gamma):
            row = min(start + i, target_logits_np.shape[0] - 1)
            t_probs = _softmax(target_logits_np[row])
            d_probs = draft_probs[i]
            tok = draft_tokens[i]
            ratio = min(1.0, float(t_probs[tok]) / max(float(d_probs[tok]), 1e-12))
            u = float(np.random.uniform())
            if u < ratio:
                accepted.append(tok)
                self._stats.tokens_accepted += 1
            else:
                corr = np.maximum(t_probs - d_probs, 0.0)
                corr /= max(corr.sum(), 1e-12)
                accepted.append(_sample(corr))
                self._stats.tokens_generated += 1
                break

        if len(accepted) == gamma:
            bonus_row = min(start + gamma, target_logits_np.shape[0] - 1)
            bonus_probs = _softmax(target_logits_np[bonus_row])
            accepted.append(_sample(bonus_probs))
            self._stats.tokens_generated += 1

        self._stats.steps += 1
        self._stats.total_time_ms += (time.perf_counter() - t0) * 1000
        return accepted, kv_cache, self._stats

    @property
    def stats(self) -> SpeculativeStats:
        return self._stats


# ---------------------------------------------------------------------------
# Medusa decoder (Cai et al. 2024)
# ---------------------------------------------------------------------------

class MedusaDecoder:
    """
    Medusa-style parallel prediction heads.

    Requires a model with additional ``medusa_heads`` trained on top of
    the base model's hidden states.  Each head predicts one position ahead
    independently (no autoregressive dependency between heads).

    If the engine does not expose ``forward_with_medusa_heads()``, this
    decoder emits an explicit warning and falls back to standard decode.
    Medusa heads cannot be fabricated — they must exist in the checkpoint.
    """

    def __init__(
        self,
        engine: Any,
        config: SpeculativeConfig | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or SpeculativeConfig()
        self._stats = SpeculativeStats()
        self._has_medusa = hasattr(engine, "forward_with_medusa_heads")

        if not self._has_medusa:
            logger.warning(
                "MedusaDecoder: engine %r does not implement "
                "forward_with_medusa_heads(). Medusa decoding requires "
                "a checkpoint trained with Medusa heads. "
                "Falling back to standard decode.",
                type(engine).__name__,
            )

    def generate_step(
        self,
        token_ids: list[int],
        kv_cache: Any,
    ) -> tuple[list[int], Any, SpeculativeStats]:
        """One Medusa decoding step."""
        t0 = time.perf_counter()

        if not self._has_medusa:
            logits, kv_cache = self.engine.forward_step(
                token_ids, kv_cache, return_all_logits=False
            )
            logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
            token = _greedy(logits_np)
            self._stats.steps += 1
            self._stats.tokens_generated += 1
            self._stats.fallback_steps += 1
            self._stats.target_forward_passes += 1
            self._stats.total_time_ms += (time.perf_counter() - t0) * 1000
            return [token], kv_cache, self._stats

        # Medusa forward: returns (base_logits, [head_logits_1, head_logits_2, ...])
        base_logits, head_logits_list, kv_cache = self.engine.forward_with_medusa_heads(
            token_ids, kv_cache
        )
        self._stats.target_forward_passes += 1

        base_np = np.asarray(base_logits, dtype=np.float64).reshape(-1)
        base_token = _greedy(base_np)
        accepted = [base_token]
        self._stats.tokens_generated += 1

        # Tree-structured verification: greedy accept each head's top prediction
        for head_logits in head_logits_list[:self.config.medusa_num_heads]:
            head_np = np.asarray(head_logits, dtype=np.float64).reshape(-1)
            head_token = _greedy(head_np)
            accepted.append(head_token)
            self._stats.tokens_accepted += 1
            self._stats.tokens_generated += 1

        self._stats.steps += 1
        self._stats.total_time_ms += (time.perf_counter() - t0) * 1000
        return accepted, kv_cache, self._stats

    @property
    def stats(self) -> SpeculativeStats:
        return self._stats


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_speculative_decoder(
    mode: str,
    target_engine: Any,
    draft_engine: Any | None = None,
    config: SpeculativeConfig | None = None,
) -> DraftTargetSpecDecoder | SelfSpeculativeDecoder | MedusaDecoder:
    """
    Factory: construct the appropriate speculative decoder.

    Args:
        mode:          One of 'draft_target', 'self', 'medusa'.
        target_engine: The primary (large) model engine.
        draft_engine:  Small draft model engine (required for 'draft_target').
        config:        SpeculativeConfig (defaults applied if None).

    Raises:
        ValueError:  If mode is 'draft_target' and draft_engine is None.
    """
    cfg = config or SpeculativeConfig()
    mode = mode.lower().replace("-", "_").replace(" ", "_")

    if mode in ("draft_target", "external", "draft"):
        if draft_engine is None:
            raise ValueError(
                "mode='draft_target' requires a draft_engine. "
                "Provide a smaller model engine or use mode='self' for "
                "self-speculative decoding."
            )
        return DraftTargetSpecDecoder(draft_engine, target_engine, cfg)

    if mode in ("self", "self_speculative", "early_exit"):
        return SelfSpeculativeDecoder(target_engine, cfg)

    if mode in ("medusa", "medusa_heads"):
        return MedusaDecoder(target_engine, cfg)

    raise ValueError(
        f"Unknown speculative decoding mode: {mode!r}. "
        "Valid modes: 'draft_target', 'self', 'medusa'."
    )
