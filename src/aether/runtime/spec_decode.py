"""Incremental speculative sampling with owned verification/rollback state."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from aether.runtime.sampling import SamplingConfig, token_probabilities
from aether.runtime.stopping import stop_token_set


def numpy_logits(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def clone_cache(cache: Any) -> Any:
    if cache is None:
        return None
    from aether.runtime.cpu_engine import KVCache
    from aether.runtime.hybrid_engine import HybridCache
    if type(cache) is KVCache:
        # Copy live rows only and retain cross-layer aliases. Copying unused
        # geometric capacity costs context-sized memory without useful state.
        arrays = {}
        def owned(value, rows):
            if value is None:
                return None
            key = (id(value), rows)
            if key not in arrays:
                arrays[key] = value[:rows].copy()
            return arrays[key]
        return KVCache(cache.num_layers,
            keys=[owned(value, rows) for value, rows in zip(cache.keys, cache.stored_lengths)],
            values=[owned(value, rows) for value, rows in zip(cache.values, cache.stored_lengths)],
            positions=[owned(value, rows) for value, rows in zip(cache.positions, cache.stored_lengths)],
            logical_length=cache.length, stored_lengths=list(cache.stored_lengths),
            reserve_length=cache.reserve_length,
            last_logits=None if cache.last_logits is None else cache.last_logits.copy(),
            last_hidden=None if cache.last_hidden is None else cache.last_hidden.copy())
    if type(cache) is HybridCache:
        return HybridCache(kv=clone_cache(cache.kv), ssm=clone_cache(cache.ssm), length=cache.length,
                           last_logits=None if cache.last_logits is None else cache.last_logits.copy(),
                           last_hidden=None if cache.last_hidden is None else cache.last_hidden.copy())
    if callable(getattr(cache, "clone", None)):
        cloned = cache.clone()
        if cloned is cache:
            raise ValueError("Cache clone must return owned rollback state")
        return cloned
    from aether.runtime.mla_engine import MLACache
    if type(cache) is MLACache:
        return MLACache(list(cache.token_ids), None if cache.last_logits is None else cache.last_logits.copy())
    # Existing Torch recurrent cache records use class-level initial fields.
    # deepcopy alone does not copy those fields, so copy each recognized state.
    known = {"TorchMambaCache": ("states", "conv_history", "length", "last_logits"),
             "TorchMamba2Cache": ("states", "conv_history", "length", "last_logits"),
             "TorchRWKVCache": ("aa", "bb", "pp", "previous", "length", "last_logits"),
             "TorchMLACache": ("token_ids", "length", "last_logits")}
    if type(cache).__module__ == "aether.runtime.torch_state_engine" and type(cache).__name__ in known:
        def owned(value):
            if callable(getattr(value, "detach", None)):
                return value.detach().clone()
            if isinstance(value, list):
                return [owned(item) for item in value]
            return copy.deepcopy(value)
        cloned = copy.copy(cache)
        for name in known[type(cache).__name__]:
            setattr(cloned, name, owned(getattr(cache, name)))
        return cloned
    raise ValueError(f"No speculative rollback contract for {type(cache).__name__}")


def clone_grammar(grammar: Any) -> Any:
    if grammar is None:
        return None
    clone = getattr(grammar, "clone", None)
    if not callable(clone):
        raise ValueError("Speculative grammar sessions require an owned clone() rollback contract")
    result = clone()
    if result is grammar:
        raise ValueError("Grammar clone must return independent request state")
    return result


def forward(decoder: Any, role: str, ids: list[int], cache: Any, *, all_logits: bool) -> tuple[np.ndarray, Any]:
    engine = decoder.draft if role == "draft" else decoder.target
    step = getattr(engine, "forward_step", None)
    if callable(step):
        logits, updated = step(ids, cache, return_all_logits=all_logits)
    elif callable(getattr(engine, "forward", None)):
        logits, updated = engine.forward(np.asarray(ids, dtype=np.int64), cache)
        if not all_logits:
            logits = numpy_logits(logits)[-1]
    else:
        raise ValueError("Speculative execution requires an actual causal forward hook")
    setattr(decoder._stats, f"{role}_forward_passes", getattr(decoder._stats, f"{role}_forward_passes") + 1)
    return numpy_logits(logits), updated


def prefix_length(cache: Any, ids: list[int]) -> int:
    if cache is None:
        return 0
    length = getattr(cache, "length", None)
    if not isinstance(length, (int, np.integer)) or isinstance(length, bool) or not 0 <= length <= len(ids):
        raise ValueError("Speculative cache length is incompatible with the supplied full prefix")
    if length == len(ids) and getattr(cache, "last_logits", None) is None:
        raise ValueError("A fully cached prefix requires its actual final logits")
    return int(length)


def speculative_step(decoder: Any, ids: list[int], draft_cache: Any, target_cache: Any, *,
                     max_tokens: int | None = None, eos_token_id: Any = None,
                     grammar_session: Any = None) -> tuple[list[int], Any, Any]:
    if not ids or any(isinstance(token, bool) or not isinstance(token, (int, np.integer)) or token < 0 for token in ids):
        raise ValueError("Speculative decoding requires a nonempty integer token prefix")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
        raise ValueError("Speculative output limit must be a nonnegative integer")
    if max_tokens == 0:
        return [], draft_cache, target_cache
    limit = max_tokens if max_tokens is not None else decoder.config.gamma + 1
    stops = stop_token_set(eos_token_id)
    cfg = decoder.config
    target_temperature = cfg.target_temperature if cfg.target_temperature is not None else (0.0 if cfg.greedy_acceptance else 1.0)
    target_sampling = SamplingConfig(target_temperature, cfg.top_k, cfg.top_p)
    draft_sampling = SamplingConfig(cfg.draft_temperature, cfg.top_k, cfg.top_p)
    target_length = prefix_length(target_cache, ids)
    suffix = ids[target_length:]
    original_target = clone_cache(target_cache)
    rng = decoder._rng

    def choose(probabilities):
        return int(np.argmax(probabilities)) if target_temperature == 0 else int(rng.choice(probabilities.size, p=probabilities))

    def ordinary():
        cache = clone_cache(original_target)
        if suffix:
            logits, cache = forward(decoder, "target", suffix, cache, all_logits=False)
            last = logits.reshape(-1)
        else:
            last = numpy_logits(cache.last_logits).reshape(-1)
        token = choose(token_probabilities(last, target_sampling, grammar_session=grammar_session))
        stop = token in stops
        if grammar_session is not None:
            if grammar_session.advance(token) < 0:
                raise ValueError("Speculative correction was rejected by grammar")
            stop |= bool(grammar_session.is_accepting())
        if cache is not None and not stop:
            _, cache = forward(decoder, "target", [token], cache, all_logits=False)
        decoder._stats.fallback_steps += 1
        decoder._stats.tokens_generated += 1
        return [token], draft_cache, cache

    if (decoder._stats.steps >= cfg.warmup_steps and decoder._stats.draft_tokens_proposed
        and decoder._stats.acceptance_rate < cfg.min_acceptance_rate):
        return ordinary()

    draft_length = prefix_length(draft_cache, ids)
    base_draft = clone_cache(draft_cache)
    if draft_length < len(ids):
        draft_logits, base_draft = forward(decoder, "draft", ids[draft_length:], base_draft, all_logits=False)
        draft_logits = draft_logits.reshape(-1)
    else:
        draft_logits = numpy_logits(base_draft.last_logits).reshape(-1)
    work_draft = clone_cache(base_draft)
    draft_grammar = clone_grammar(grammar_session)
    proposed, distributions = [], []
    for index in range(min(cfg.gamma, limit)):
        q = token_probabilities(draft_logits, draft_sampling, grammar_session=draft_grammar)
        token = int(np.argmax(q)) if cfg.draft_temperature == 0 else int(rng.choice(q.size, p=q))
        proposed.append(token)
        distributions.append(q)
        complete = token in stops
        if draft_grammar is not None:
            if draft_grammar.advance(token) < 0:
                raise ValueError("Draft candidate was rejected by its grammar mask")
            complete |= bool(draft_grammar.is_accepting())
        if complete or index + 1 == min(cfg.gamma, limit):
            break
        input_ids = [token] if work_draft is not None else ids + proposed
        draft_logits, work_draft = forward(decoder, "draft", input_ids, work_draft, all_logits=False)
        draft_logits = draft_logits.reshape(-1)

    decoder._stats.draft_tokens_proposed += len(proposed)
    verification, work_target = forward(decoder, "target", suffix + proposed,
                                         clone_cache(original_target), all_logits=True)
    if verification.ndim != 2 or verification.shape[0] != len(suffix) + len(proposed):
        # A last-logit-only executor cannot validate earlier candidates. Discard
        # its speculative state and sample from the genuine original prefix.
        return ordinary()
    target_grammar = clone_grammar(grammar_session)
    emitted = []
    accepted = 0
    terminated = False
    for index, (candidate, q) in enumerate(zip(proposed, distributions)):
        if suffix:
            logits = verification[len(suffix) - 1 + index]
        elif index:
            logits = verification[index - 1]
        else:
            logits = numpy_logits(original_target.last_logits).reshape(-1)
        p = token_probabilities(logits, target_sampling, grammar_session=target_grammar)
        if p.shape != q.shape:
            raise ValueError("Draft and target vocabularies differ")
        ratio = min(1.0, float(p[candidate]) / float(q[candidate]))
        is_accepted = ratio >= 1.0 or float(rng.random()) < ratio
        if is_accepted:
            token = candidate
            accepted += 1
        else:
            corrected = np.maximum(p - q, 0)
            mass = corrected.sum()
            if mass <= 0:
                raise ValueError("Rejected speculative candidate has no positive residual probability")
            token = choose(corrected / mass)
        emitted.append(token)
        if target_grammar is not None:
            if target_grammar.advance(token) < 0:
                raise ValueError("Target token was rejected by its grammar mask")
            terminated |= bool(target_grammar.is_accepting())
        terminated |= token in stops
        if terminated or len(emitted) == limit or not is_accepted:
            break
    if accepted == len(proposed) and not terminated and len(emitted) < limit:
        p = token_probabilities(verification[-1], target_sampling, grammar_session=target_grammar)
        token = choose(p)
        emitted.append(token)
        if target_grammar is not None:
            if target_grammar.advance(token) < 0:
                raise ValueError("Bonus token was rejected by its grammar mask")
            terminated |= bool(target_grammar.is_accepting())
        terminated |= token in stops
    if grammar_session is not None:
        for token in emitted:
            if grammar_session.advance(token) < 0:
                raise ValueError("Verified token was rejected by the original grammar state")
    # Ordinary engine sessions exclude a terminating EOS/grammar token from
    # the cache. Every other emitted position is committed exactly once.
    committed = emitted[:-1] if terminated else emitted
    if work_target is None:
        updated_target = None
    elif accepted == len(proposed) and len(committed) >= len(proposed):
        updated_target = work_target
        extra = committed[len(proposed):]
        if extra:
            _, updated_target = forward(decoder, "target", extra, updated_target, all_logits=False)
    else:
        updated_target = clone_cache(original_target)
        if suffix or committed:
            _, updated_target = forward(decoder, "target", suffix + committed, updated_target, all_logits=False)
    if base_draft is None:
        updated_draft = None
    else:
        stored = max(0, len(proposed) - 1)
        if accepted == len(proposed) and len(committed) >= stored:
            updated_draft = work_draft
            extra = committed[stored:]
        else:
            updated_draft = clone_cache(base_draft)
            extra = committed
        if extra:
            _, updated_draft = forward(decoder, "draft", extra, updated_draft, all_logits=False)
    decoder._stats.tokens_accepted += accepted
    decoder._stats.tokens_generated += len(emitted)
    return emitted, updated_draft, updated_target
