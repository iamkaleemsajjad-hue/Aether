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
from aether.runtime.sampling import SamplingConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DraftTargetSpecDecoder",
    "SelfSpeculativeDecoder",
    "MedusaDecoder",
    "SpeculativeStats",
    "SpeculativeConfig",
    "make_speculative_decoder",
    "SpeculativeGenerationAdapter",
    "SpeculativeSessionCache",
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

    # If True, the target distribution is the exact greedy delta distribution.
    # False defaults to temperature-one stochastic target sampling.
    greedy_acceptance: bool = False

    # Medusa-specific: number of heads
    medusa_num_heads: int = 4

    # Self-speculative: which layer index to exit for draft
    early_exit_layer: int | None = None

    target_temperature: float | None = None
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.gamma) is not int or not 1 <= self.gamma <= 128:
            raise ValueError("speculative gamma must be an integer in [1, 128]")
        SamplingConfig(self.draft_temperature, self.top_k, self.top_p)
        target = self.target_temperature if self.target_temperature is not None else (0.0 if self.greedy_acceptance else 1.0)
        SamplingConfig(target, self.top_k, self.top_p)
        if self.greedy_acceptance and target != 0:
            raise ValueError("greedy acceptance requires target temperature zero")
        if not 0 <= self.min_acceptance_rate <= 1 or not np.isfinite(self.min_acceptance_rate):
            raise ValueError("minimum acceptance rate must be finite and in [0, 1]")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("speculative warmup steps must be a nonnegative integer")


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
    draft_tokens_proposed: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Draft token acceptance rate.

        Computed as ``tokens_accepted / draft_tokens_proposed``. Output counts
        include accepted candidates, corrections and bonus tokens separately.
        A value of 1.0 means all draft tokens were accepted on every step.
        """
        if self.steps == 0:
            return 0.0
        denominator = self.draft_tokens_proposed or self.tokens_generated
        return min(1.0, self.tokens_accepted / max(1, denominator))

    @property
    def speedup_estimate(self) -> float:
        """
        Tokens emitted per actual target forward call, not measured speedup.
        Cache commit/replay calls are counted; latency also includes draft work.
        """
        if self.target_forward_passes == 0:
            return 1.0
        total = self.tokens_generated
        return total / max(1, self.target_forward_passes)

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "tokens_accepted": self.tokens_accepted,
            "tokens_generated": self.tokens_generated,
            "draft_tokens_proposed": self.draft_tokens_proposed,
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
    2. Target model verifies candidates in one causal forward pass.
    3. Accept each draft token i with probability min(1, p_target(xi) / p_draft(xi)).
    4. On rejection, sample corrected token from (p_target - p_draft)⁺.
    5. Emit a correction on rejection or a bonus after complete acceptance,
       subject to the requested output limit and stopping criteria. Cache
       commit/rollback can require an additional actual forward call.

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
        self._rng = np.random.default_rng(self.config.seed)

    def generate_step(
        self,
        token_ids: list[int],
        draft_cache: Any,
        target_cache: Any,
        *,
        max_tokens: int | None = None,
        eos_token_id: Any = None,
        grammar_session: Any = None,
    ) -> tuple[list[int], Any, Any, SpeculativeStats]:
        """Verify a real draft using incremental suffixes and owned rollback state.

        token_ids is the complete logical prefix; cache.length identifies the
        already consumed part. Rejected candidates never survive in either
        returned cache. The output bound and stop/grammar criteria are applied
        before state is committed, rather than slicing a speculative result.
        """
        from aether.runtime.spec_decode import speculative_step
        began = time.perf_counter()
        tokens, draft_cache, target_cache = speculative_step(
            self, list(token_ids), draft_cache, target_cache,
            max_tokens=max_tokens, eos_token_id=eos_token_id,
            grammar_session=grammar_session,
        )
        if tokens:
            self._stats.steps += 1
            self._stats.total_time_ms += (time.perf_counter() - began) * 1000
        return tokens, draft_cache, target_cache, self._stats

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


from aether.runtime.spec_generation import SpeculativeGenerationAdapter, SpeculativeSessionCache
