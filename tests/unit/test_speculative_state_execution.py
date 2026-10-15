"""Exact draft sampling exercised with actual CPU attention and recurrent state."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from aether.runtime.speculative_engine import (
    DraftTargetSpecDecoder, SpeculativeConfig, SpeculativeGenerationAdapter,
)
from aether.runtime.sampling import SamplingConfig, token_probabilities
from test_disaggregated_execution import PROMPT, _setup


@pytest.mark.parametrize("budget", [1, 2, 4, 5, 11])
@pytest.mark.parametrize("reject", [False, True])
def test_actual_attention_greedy_tokens_and_cache_are_exactly_bounded(budget, reject):
    factory, _, _ = _setup()
    target, draft = factory(), factory()
    if reject:
        draft.weights = copy.deepcopy(draft.weights)
        draft.weights.lm_head *= -1
    expected, expected_cache = factory().generate_with_cache(PROMPT, budget, temperature=0)
    adapter = SpeculativeGenerationAdapter(target, draft, gamma=4, min_acceptance_rate=0)
    actual, session = adapter.generate_with_cache(PROMPT, budget, temperature=0)
    assert actual == expected
    assert session.length == len(PROMPT) + budget
    assert len(session.token_ids) == session.length
    np.testing.assert_allclose(session.last_logits, expected_cache.last_logits, rtol=0, atol=5e-7)
    stats = adapter.speculative_stats()
    assert stats["tokens_generated"] == budget
    assert 0 <= stats["acceptance_rate"] <= 1
    assert stats["target_forward_passes"] >= 1
    if reject:
        assert stats["tokens_accepted"] < stats["draft_tokens_proposed"]


def test_real_incremental_suffix_session_does_not_repeat_prefix_or_mutate_input():
    factory, _, _ = _setup()
    adapter = SpeculativeGenerationAdapter(factory(), factory(), gamma=3)
    first, session = adapter.generate_with_cache(PROMPT, 4)
    original = session.clone()
    original_keys = session.keys[0].copy()
    second, updated = adapter.generate_with_cache([], 3, cache=session)
    expected, expected_cache = factory().generate_with_cache(PROMPT, 7)
    assert first + second == expected
    assert updated.length == len(PROMPT) + 7
    np.testing.assert_array_equal(session.last_logits, original.last_logits)
    np.testing.assert_array_equal(session.keys[0], original_keys)
    np.testing.assert_allclose(updated.last_logits, expected_cache.last_logits, rtol=0, atol=5e-7)


def test_rollback_clones_live_rows_and_preserves_cross_layer_aliases():
    from aether.runtime.cpu_engine import KVCache
    from aether.runtime.spec_decode import clone_cache
    keys = np.zeros((100, 2, 4), dtype=np.float32)
    position = np.arange(100, dtype=np.int64)
    cache = KVCache(2, keys=[keys, keys], values=[keys, keys], positions=[position, position],
                    logical_length=3, stored_lengths=[3, 3], last_logits=np.zeros(17, np.float32))
    cloned = clone_cache(cache)
    assert cloned.keys[0] is cloned.keys[1] is cloned.values[0]
    assert cloned.keys[0].shape[0] == 3
    assert cloned.positions[0] is cloned.positions[1]
    assert not np.shares_memory(cloned.keys[0], keys)


def test_raw_cache_bootstrap_requires_actual_prefix_then_prefills_only_missing_positions():
    factory, _, _ = _setup()
    _, cache = factory().forward(PROMPT)
    adapter = SpeculativeGenerationAdapter(factory(), factory(), gamma=2)
    with pytest.raises(ValueError, match="cached_prefix_ids"):
        adapter.generate_with_cache([], 2, cache=cache)
    actual, session = adapter.generate_with_cache([], 2, cache=cache, cached_prefix_ids=PROMPT)
    expected = factory().generate(PROMPT, 2)
    assert actual == expected
    assert session.length == len(PROMPT) + 2
    assert cache.length == len(PROMPT)


def test_zero_budget_really_prefills_and_eos_does_not_enter_cache():
    factory, _, _ = _setup()
    adapter = SpeculativeGenerationAdapter(factory(), factory(), gamma=4)
    tokens, prefetched = adapter.generate_with_cache(PROMPT, 0)
    assert tokens == [] and prefetched.length == len(PROMPT)
    assert prefetched.draft_cache.length == prefetched.target_cache.length
    expected = factory().generate(PROMPT, 1)[0]
    tokens, stopped = adapter.generate_with_cache([], 9, cache=prefetched, eos_token_id=[expected])
    assert tokens == [expected]
    assert stopped.length == len(PROMPT)
    assert stopped.draft_cache.length == len(PROMPT)


def test_actual_hybrid_rejection_rolls_back_both_attention_and_recurrent_state():
    from cpu_family_fixtures import engine_for_family
    target, draft = engine_for_family("hybrid"), engine_for_family("hybrid")
    draft.weights = copy.deepcopy(draft.weights)
    draft.weights.lm_head *= -1
    adapter = SpeculativeGenerationAdapter(target, draft, gamma=4, min_acceptance_rate=0)
    expected, expected_cache = engine_for_family("hybrid").generate_with_cache(PROMPT, 7)
    actual, cache = adapter.generate_with_cache(PROMPT, 7)
    assert actual == expected
    assert cache.length == cache.target_cache.kv.length == cache.target_cache.ssm.length == len(PROMPT) + 7
    np.testing.assert_allclose(cache.last_logits, expected_cache.last_logits, rtol=0, atol=5e-7)
    np.testing.assert_allclose(cache.target_cache.ssm.states[0].h, expected_cache.ssm.states[0].h, rtol=0, atol=5e-7)


def test_real_stochastic_target_marginal_with_rejections_topk_and_nucleus():
    factory, _, _ = _setup()
    target, draft = factory(), factory()
    draft.weights = copy.deepcopy(draft.weights)
    draft.weights.lm_head = np.roll(draft.weights.lm_head, 1, axis=0)
    logits, _ = target.forward(PROMPT)
    expected = token_probabilities(logits[-1], SamplingConfig(.7, 8, .8))
    decoder = DraftTargetSpecDecoder(draft, target, SpeculativeConfig(
        gamma=1, target_temperature=.7, draft_temperature=1.3, top_k=8, top_p=.8,
        min_acceptance_rate=0, seed=193))
    trials = 2000
    counts = np.zeros(expected.size, dtype=np.int64)
    for _ in range(trials):
        tokens, _, cache, _ = decoder.generate_step(PROMPT.tolist(), None, None, max_tokens=1)
        assert len(tokens) == 1 and cache.length == len(PROMPT) + 1
        counts[tokens[0]] += 1
    # Six binomial standard deviations plus a two-count discrete allowance:
    # this tests the actual target marginal, including q != p and rejection.
    tolerance = 6 * np.sqrt(trials * expected * (1 - expected)) + 2
    assert np.all(np.abs(counts - trials * expected) <= tolerance)
    assert np.all(counts[expected == 0] == 0)
    assert 0 < decoder.stats.tokens_accepted < decoder.stats.draft_tokens_proposed


class _Grammar:
    def __init__(self, tokens=(2, 4)):
        self.tokens, self.position, self.advances = tokens, 0, []
    def clone(self): return copy.deepcopy(self)
    def get_token_mask(self):
        mask = bytearray(3)
        token = self.tokens[self.position]
        mask[token // 8] |= 1 << (token % 8)
        return bytes(mask)
    def advance(self, token):
        if token != self.tokens[self.position]: return -1
        self.position += 1
        self.advances.append(token)
        return self.position
    def is_accepting(self): return self.position == len(self.tokens)


def test_grammar_verified_state_advances_once_and_terminating_token_is_not_cached():
    factory, _, _ = _setup()
    grammar = _Grammar()
    adapter = SpeculativeGenerationAdapter(factory(), factory(), gamma=4)
    tokens, cache = adapter.generate_with_cache(PROMPT, 10, grammar_session=grammar)
    assert tokens == grammar.advances == [2, 4]
    assert cache.length == len(PROMPT) + 1
    assert cache.draft_cache.length == cache.length


def test_speculative_cache_contract_and_active_hooks_fail_explicitly():
    factory, _, _ = _setup()
    adapter = SpeculativeGenerationAdapter(factory(), factory())
    with pytest.raises(ValueError, match="unsupported active"):
        adapter.generate(PROMPT, 3, ttt_slots=[{}])
    with pytest.raises(ValueError, match="token limit"):
        adapter.generate(PROMPT, -1)
    with pytest.raises(ValueError, match="different target/draft"):
        _, cache = adapter.generate_with_cache(PROMPT, 1)
        SpeculativeGenerationAdapter(factory(), factory()).generate([], 1, cache=cache)


def test_single_step_rejection_keeps_caller_caches_owned_and_prefix_lengths_consistent():
    factory, _, _ = _setup()
    target, draft = factory(), factory()
    draft.weights = copy.deepcopy(draft.weights)
    draft.weights.lm_head *= -1
    _, original_target = target.forward(PROMPT)
    _, original_draft = draft.forward(PROMPT)
    keys_before = original_target.keys[0].copy()
    decoder = DraftTargetSpecDecoder(draft, target, SpeculativeConfig(gamma=4, greedy_acceptance=True))
    tokens, updated_draft, updated_target, _ = decoder.generate_step(
        PROMPT.tolist(), original_draft, original_target, max_tokens=3)
    assert original_target.length == original_draft.length == len(PROMPT)
    np.testing.assert_array_equal(original_target.keys[0], keys_before)
    assert updated_target.length == updated_draft.length == len(PROMPT) + len(tokens)
    expected, _ = factory().forward(np.asarray(PROMPT.tolist() + tokens))
    np.testing.assert_allclose(updated_target.last_logits, expected[-1], rtol=0, atol=5e-7)
