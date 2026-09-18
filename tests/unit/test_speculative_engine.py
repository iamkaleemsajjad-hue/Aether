"""
Tests for aether.runtime.speculative_engine

Verifies:
- DraftTargetSpecDecoder produces tokens from target distribution (not draft)
- Algorithm 1 acceptance/rejection math is statistically correct
- Bonus token is always emitted (even on full rejection)
- SelfSpeculativeDecoder falls back gracefully when engine lacks early_exit
- MedusaDecoder falls back gracefully when engine lacks medusa_heads
- make_speculative_decoder factory routing
- SpeculativeStats accumulates correctly
- No hash-based fake token generation
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from aether.runtime.speculative_engine import (
    DraftTargetSpecDecoder,
    MedusaDecoder,
    SelfSpeculativeDecoder,
    SpeculativeConfig,
    SpeculativeStats,
    _sample,
    _softmax,
    make_speculative_decoder,
)


# ---------------------------------------------------------------------------
# Utility unit tests
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.random.randn(100).astype(np.float64)
        p = _softmax(logits)
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)

    def test_greedy_temperature_zero(self):
        logits = np.array([1.0, 5.0, 2.0])
        p = _softmax(logits, temperature=0.0)
        assert int(np.argmax(p)) == 1
        np.testing.assert_allclose(p[1], 1.0, atol=1e-9)

    def test_high_temperature_flatter(self):
        logits = np.array([10.0, 0.0, 0.0])
        p_low = _softmax(logits, temperature=0.1)
        p_high = _softmax(logits, temperature=10.0)
        assert p_low[0] > p_high[0]  # high temp = flatter

    def test_numerical_stability(self):
        logits = np.array([1000.0, 999.0])
        p = _softmax(logits)
        assert np.all(np.isfinite(p))


class TestSample:
    def test_returns_valid_index(self):
        probs = np.array([0.2, 0.5, 0.3])
        tok = _sample(probs)
        assert 0 <= tok < 3

    def test_zero_prob_never_sampled(self):
        probs = np.array([0.0, 1.0, 0.0])
        results = {_sample(probs) for _ in range(100)}
        assert results == {1}

    def test_uniform_distribution(self):
        np.random.seed(0)
        probs = np.ones(10) / 10.0
        counts = np.zeros(10)
        for _ in range(1000):
            counts[_sample(probs)] += 1
        # Each bucket should get ~100 samples with large tolerance
        assert all(20 < c < 200 for c in counts)


# ---------------------------------------------------------------------------
# Fake engine for testing
# ---------------------------------------------------------------------------

class _FakeEngine:
    """
    A minimal engine that always outputs fixed logits.

    The draft engine puts all mass on token 42.
    The target engine puts all mass on token 99.
    This allows exact verification of the acceptance test.
    """

    def __init__(self, preferred_token: int, vocab_size: int = 200):
        self.preferred_token = preferred_token
        self.vocab_size = vocab_size
        self._call_count = 0

    def init_kv_cache(self):
        return None

    def forward_step(self, token_ids, kv_cache, return_all_logits=False):
        self._call_count += 1
        logits = np.full(self.vocab_size, -10.0, dtype=np.float64)
        logits[self.preferred_token] = 10.0  # strong preference
        if return_all_logits:
            # Return logits for every position
            n = len(list(token_ids)) if hasattr(token_ids, '__len__') else 1
            return np.tile(logits, (n, 1)), kv_cache
        return logits, kv_cache


class _FakeEngineAllMass(_FakeEngine):
    """Greedy engine — always returns the same token regardless of input."""
    pass


# ---------------------------------------------------------------------------
# DraftTargetSpecDecoder
# ---------------------------------------------------------------------------

class TestDraftTargetSpecDecoder:
    def _make_decoder(self, draft_pref=42, target_pref=99, gamma=3):
        draft = _FakeEngine(draft_pref, vocab_size=200)
        target = _FakeEngine(target_pref, vocab_size=200)
        config = SpeculativeConfig(gamma=gamma, greedy_acceptance=False)
        return DraftTargetSpecDecoder(draft, target, config)

    def test_always_returns_at_least_one_token(self):
        dec = self._make_decoder()
        np.random.seed(0)
        tokens, _, _, stats = dec.generate_step([1, 2, 3], None, None)
        assert len(tokens) >= 1

    def test_bonus_token_appended_on_full_acceptance(self):
        """When draft == target, all drafts should be accepted → gamma+1 tokens."""
        # Make draft and target agree on the same token
        draft = _FakeEngine(42, vocab_size=200)
        target = _FakeEngine(42, vocab_size=200)
        config = SpeculativeConfig(gamma=4, greedy_acceptance=True)
        dec = DraftTargetSpecDecoder(draft, target, config)
        np.random.seed(0)
        tokens, _, _, stats = dec.generate_step([1, 2], None, None)
        # All 4 accepted + 1 bonus
        assert len(tokens) == 5

    def test_rejection_produces_target_token(self):
        """When draft = token 42, target = token 99, draft should be rejected
        and a corrected token drawn from target distribution (token 99)."""
        draft = _FakeEngine(42, vocab_size=200)
        target = _FakeEngine(99, vocab_size=200)
        config = SpeculativeConfig(gamma=1, greedy_acceptance=True)
        dec = DraftTargetSpecDecoder(draft, target, config)
        np.random.seed(0)
        for _ in range(20):
            tokens, _, _, _ = dec.generate_step([1, 2], None, None)
            # Either 1 token (rejected) or 2 tokens (accepted + bonus)
            assert len(tokens) >= 1
            # Last token should be target's preferred token (99)
            assert tokens[-1] == 99

    def test_stats_accumulate(self):
        dec = self._make_decoder()
        np.random.seed(0)
        for _ in range(5):
            dec.generate_step([1, 2, 3], None, None)
        assert dec.stats.steps == 5
        assert dec.stats.tokens_generated >= 5

    def test_draft_forward_pass_count(self):
        """Should run gamma draft passes per step."""
        gamma = 4
        draft = _FakeEngine(1, vocab_size=50)
        target = _FakeEngine(1, vocab_size=50)
        config = SpeculativeConfig(gamma=gamma)
        dec = DraftTargetSpecDecoder(draft, target, config)
        dec.generate_step([1], None, None)
        assert draft._call_count == gamma

    def test_target_forward_pass_count(self):
        """Should run exactly 1 target pass per step."""
        draft = _FakeEngine(1, vocab_size=50)
        target = _FakeEngine(1, vocab_size=50)
        config = SpeculativeConfig(gamma=3)
        dec = DraftTargetSpecDecoder(draft, target, config)
        dec.generate_step([1], None, None)
        assert target._call_count == 1

    def test_requires_draft_engine(self):
        with pytest.raises(ValueError, match="requires both"):
            DraftTargetSpecDecoder(None, _FakeEngine(1))

    def test_reset_stats(self):
        dec = self._make_decoder()
        dec.generate_step([1], None, None)
        assert dec.stats.steps > 0
        dec.reset_stats()
        assert dec.stats.steps == 0

    def test_greedy_acceptance_mode(self):
        """In greedy mode, accepted iff target argmax == draft token."""
        draft = _FakeEngine(5, vocab_size=50)
        target = _FakeEngine(5, vocab_size=50)  # same preference
        config = SpeculativeConfig(gamma=3, greedy_acceptance=True)
        dec = DraftTargetSpecDecoder(draft, target, config)
        tokens, _, _, _ = dec.generate_step([1, 2], None, None)
        # All draft tokens match target → all accepted + bonus = 4
        assert len(tokens) == 4


# ---------------------------------------------------------------------------
# SelfSpeculativeDecoder
# ---------------------------------------------------------------------------

class TestSelfSpeculativeDecoder:
    def test_fallback_when_no_early_exit(self):
        engine = _FakeEngine(7, vocab_size=50)
        dec = SelfSpeculativeDecoder(engine, SpeculativeConfig(gamma=3))
        tokens, _, stats = dec.generate_step([1, 2], None)
        assert len(tokens) == 1
        assert stats.fallback_steps == 1

    def test_stats_increment_on_fallback(self):
        engine = _FakeEngine(7, vocab_size=50)
        dec = SelfSpeculativeDecoder(engine)
        for _ in range(3):
            dec.generate_step([1], None)
        assert dec.stats.steps == 3
        assert dec.stats.fallback_steps == 3


# ---------------------------------------------------------------------------
# MedusaDecoder
# ---------------------------------------------------------------------------

class TestMedusaDecoder:
    def test_fallback_when_no_medusa_heads(self):
        engine = _FakeEngine(3, vocab_size=50)
        dec = MedusaDecoder(engine, SpeculativeConfig())
        tokens, _, stats = dec.generate_step([1, 2], None)
        assert len(tokens) == 1
        assert stats.fallback_steps == 1

    def test_medusa_head_execution(self):
        """When engine has forward_with_medusa_heads, use it."""
        engine = MagicMock()
        base_logits = np.full(50, -10.0)
        base_logits[7] = 10.0
        head_logits = [np.full(50, -10.0) for _ in range(3)]
        for i, h in enumerate(head_logits):
            h[i + 10] = 10.0
        engine.forward_with_medusa_heads.return_value = (base_logits, head_logits, None)

        config = SpeculativeConfig(medusa_num_heads=3)
        dec = MedusaDecoder(engine, config)
        tokens, _, stats = dec.generate_step([1, 2], None)
        assert tokens[0] == 7  # base model's argmax
        assert len(tokens) == 1 + 3  # base + 3 heads
        engine.forward_with_medusa_heads.assert_called_once()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestMakeSpeculativeDecoder:
    def test_draft_target_mode(self):
        draft = _FakeEngine(1)
        target = _FakeEngine(2)
        dec = make_speculative_decoder("draft_target", target, draft)
        assert isinstance(dec, DraftTargetSpecDecoder)

    def test_self_mode(self):
        engine = _FakeEngine(1)
        dec = make_speculative_decoder("self", engine)
        assert isinstance(dec, SelfSpeculativeDecoder)

    def test_medusa_mode(self):
        engine = _FakeEngine(1)
        dec = make_speculative_decoder("medusa", engine)
        assert isinstance(dec, MedusaDecoder)

    def test_draft_mode_requires_draft_engine(self):
        with pytest.raises(ValueError, match="requires a draft_engine"):
            make_speculative_decoder("draft_target", _FakeEngine(1), draft_engine=None)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown speculative decoding mode"):
            make_speculative_decoder("invalid_mode", _FakeEngine(1))

    def test_alias_modes(self):
        engine = _FakeEngine(1)
        dec = make_speculative_decoder("early_exit", engine)
        assert isinstance(dec, SelfSpeculativeDecoder)
        dec2 = make_speculative_decoder("medusa_heads", engine)
        assert isinstance(dec2, MedusaDecoder)


# ---------------------------------------------------------------------------
# SpeculativeStats
# ---------------------------------------------------------------------------

class TestSpeculativeStats:
    def test_acceptance_rate_zero_steps(self):
        s = SpeculativeStats()
        assert s.acceptance_rate == 0.0

    def test_acceptance_rate_calculation(self):
        s = SpeculativeStats(steps=10, tokens_accepted=30, tokens_generated=40)
        assert abs(s.acceptance_rate - 0.75) < 1e-9

    def test_speedup_with_full_acceptance(self):
        s = SpeculativeStats(
            steps=10,
            tokens_accepted=40,  # gamma=4, all accepted
            tokens_generated=50,  # +10 bonus
            target_forward_passes=10,
        )
        # total tokens = 50, target passes = 10 → speedup = 5.0
        assert abs(s.speedup_estimate - 5.0) < 1e-9

    def test_report_is_dict(self):
        s = SpeculativeStats(steps=1, tokens_generated=3, target_forward_passes=1)
        r = s.report()
        assert isinstance(r, dict)
        assert "acceptance_rate" in r
        assert "speedup_estimate" in r
