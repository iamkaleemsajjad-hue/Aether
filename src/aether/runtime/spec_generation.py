"""Engine-shaped exact draft/target sessions with explicit consumed prefixes."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading
from typing import Any, Callable, Iterator

import numpy as np

from aether.runtime.sampling import SamplingConfig
from aether.runtime.spec_decode import clone_cache, forward, prefix_length
from aether.runtime.stopping import stop_token_set


@dataclass
class SpeculativeSessionCache:
    """Owned target/draft state and the exact token prefix both states consume.

    Target cache fields remain inspectable for ordinary session metrics.
    Persistence stores only the exact target cache and consumed prefix; the
    draft can be rebuilt. Separate phase workers still require a draft adapter.
    """
    target_cache: Any
    draft_cache: Any
    token_ids: tuple[int, ...]
    source_identity: tuple[int, int]

    @property
    def length(self) -> int:
        return int(getattr(self.target_cache, "length", len(self.token_ids)))

    @property
    def last_logits(self) -> Any:
        return getattr(self.target_cache, "last_logits", None)

    def clone(self) -> "SpeculativeSessionCache":
        return SpeculativeSessionCache(clone_cache(self.target_cache), clone_cache(self.draft_cache),
                                       self.token_ids, self.source_identity)

    def __getattr__(self, name: str) -> Any:
        # Dataclass/copy introspection can happen before fields are assigned.
        target = self.__dict__.get("target_cache")
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)


class SpeculativeGenerationAdapter:
    """Exact request sampler over real draft and target forward hooks.

    Known CPU engines get fresh worker state sharing immutable weights. Other
    engines may provide explicit factories; otherwise a shared execution lock
    serializes their mutable executor state without duplicating device weights.
    Both tokenizers must be verified by the caller before constructing this
    adapter. A vocabulary-size check alone cannot establish token identity.
    """
    _locks_guard = threading.Lock()
    _locks: dict[int, threading.Lock] = {}

    def __init__(self, target_engine: Any, draft_engine: Any, gamma: int = 4, *,
                 target_factory: Callable[[], Any] | None = None,
                 draft_factory: Callable[[], Any] | None = None,
                 draft_temperature: float | None = None,
                 min_acceptance_rate: float = .5, warmup_steps: int = 10) -> None:
        from aether.runtime.speculative_engine import SpeculativeConfig
        SpeculativeConfig(gamma=gamma, min_acceptance_rate=min_acceptance_rate,
                          warmup_steps=warmup_steps,
                          draft_temperature=0 if draft_temperature is None else draft_temperature)
        if target_engine is None or draft_engine is None:
            raise ValueError("Speculative generation requires real target and draft engines")
        self.target, self.draft, self.gamma = target_engine, draft_engine, gamma
        self.target_factory, self.draft_factory = target_factory, draft_factory
        self.draft_temperature = draft_temperature
        self.min_acceptance_rate, self.warmup_steps = min_acceptance_rate, warmup_steps
        self._local = threading.local()
        self._source_identity = (id(target_engine), id(draft_engine))
        with self._locks_guard:
            self._execution_locks = [self._locks.setdefault(identity, threading.Lock())
                                     for identity in sorted(set(self._source_identity))]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    @staticmethod
    def _tokens(value: Any, *, label: str) -> list[int]:
        ids = np.asarray(value)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in "iu" or np.any(ids < 0)
                                         or np.any(ids > np.iinfo(np.int64).max))):
            raise ValueError(f"{label} must be a one-dimensional nonnegative integer token sequence")
        return ids.astype(np.int64).tolist()

    @staticmethod
    def _engine(source: Any, factory: Callable[[], Any] | None) -> tuple[Any, bool]:
        if factory is not None:
            engine = factory()
            if engine is source:
                raise ValueError("Speculative worker factory must return owned engine state")
            return engine, True
        from aether.runtime.disaggregated_execution import clone_cpu_engine_for_worker
        from aether.runtime.cpu_engine import CPUExecutionEngine
        from aether.runtime.hybrid_engine import HybridExecutionEngine
        from aether.runtime.mamba_engine import MambaExecutionEngine
        from aether.runtime.mamba2_engine import Mamba2ExecutionEngine
        from aether.runtime.rwkv_engine import RWKVExecutionEngine
        from aether.runtime.mla_engine import MLAExecutionEngine
        if type(source) in (CPUExecutionEngine, HybridExecutionEngine, MambaExecutionEngine,
                            Mamba2ExecutionEngine, RWKVExecutionEngine, MLAExecutionEngine):
            return clone_cpu_engine_for_worker(source), True
        return source, False

    def generate_iter(self, prompt_ids: Any, max_tokens: int = 16, *, temperature: float = 0,
                      top_k: int = 0, top_p: float = 1, eos_token_id: Any = None,
                      seed: int | None = None, grammar_session: Any = None,
                      cache: Any = None, cached_prefix_ids: Any = None,
                      cache_callback: Callable[[SpeculativeSessionCache], Any] | None = None,
                      **kwargs: Any) -> Iterator[int]:
        from aether.runtime.speculative_engine import DraftTargetSpecDecoder, SpeculativeConfig
        SamplingConfig(temperature, top_k, top_p)
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("Speculative token limit must be a nonnegative integer")
        unsupported = sorted(key for key, value in kwargs.items() if value is not None)
        if unsupported:
            raise ValueError(f"Speculative request has unsupported active execution hooks: {unsupported}")
        suffix = self._tokens(prompt_ids, label="Prompt suffix")
        if isinstance(cache, SpeculativeSessionCache):
            if cache.source_identity != self._source_identity:
                raise ValueError("Speculative session belongs to a different target/draft pair")
            if cached_prefix_ids is not None and self._tokens(cached_prefix_ids, label="Cached prefix") != list(cache.token_ids):
                raise ValueError("Explicit cached prefix differs from the speculative session")
            ids = list(cache.token_ids) + suffix
            target_cache, draft_cache = clone_cache(cache.target_cache), clone_cache(cache.draft_cache)
            if cache.length != len(cache.token_ids):
                raise ValueError("Speculative session token prefix differs from target cache length")
        elif cache is not None:
            if cached_prefix_ids is None:
                raise ValueError("Raw target cache requires exact cached_prefix_ids to initialize the draft")
            prefix = self._tokens(cached_prefix_ids, label="Cached prefix")
            if prefix_length(cache, prefix) != len(prefix):
                raise ValueError("Raw target cache does not consume the complete supplied cached prefix")
            ids = prefix + suffix
            target_cache, draft_cache = clone_cache(cache), None
        else:
            if cached_prefix_ids is not None:
                raise ValueError("cached_prefix_ids requires an existing target cache")
            ids, target_cache, draft_cache = suffix, None, None
        if not ids:
            raise ValueError("Speculative generation requires a populated token prefix")
        self._local.stats, self._local.cache = {}, None
        target, target_owned = self._engine(self.target, self.target_factory)
        draft, draft_owned = self._engine(self.draft, self.draft_factory)
        locks = [] if target_owned and draft_owned else self._execution_locks
        @contextmanager
        def executing():
            # Generator consumption/close may occur on a different thread.
            # No executor lock may remain held across a token yield. Cache
            # state is request-owned; each real verification cycle serializes
            # the borrowed executor's mutable tables/calibration state.
            for lock in locks:
                lock.acquire()
            try:
                yield
            finally:
                for lock in reversed(locks):
                    lock.release()
        with executing():
            decoder = DraftTargetSpecDecoder(draft, target, SpeculativeConfig(
                gamma=self.gamma, target_temperature=temperature, top_k=top_k, top_p=top_p,
                draft_temperature=temperature if self.draft_temperature is None else self.draft_temperature,
                min_acceptance_rate=self.min_acceptance_rate, warmup_steps=self.warmup_steps,
                greedy_acceptance=temperature == 0, seed=seed))
            if max_tokens == 0:
                for role, current in (("target", target_cache), ("draft", draft_cache)):
                    missing = ids[prefix_length(current, ids):]
                    if missing:
                        _, current = forward(decoder, role, missing, current, all_logits=False)
                    if role == "target":
                        target_cache = current
                    else:
                        draft_cache = current
        generated = 0
        stops = stop_token_set(eos_token_id)
        while generated < max_tokens:
            with executing():
                tokens, draft_cache, target_cache, _ = decoder.generate_step(
                    ids, draft_cache, target_cache, max_tokens=max_tokens - generated,
                    eos_token_id=eos_token_id, grammar_session=grammar_session)
            if not tokens:
                raise RuntimeError("Speculative executor made no generation progress")
            generated += len(tokens)
            ids += tokens
            yield from tokens
            if tokens[-1] in stops or (grammar_session is not None and grammar_session.is_accepting()):
                break
        consumed = int(getattr(target_cache, "length", len(ids)))
        if not 0 <= consumed <= len(ids):
            raise ValueError("Target cache returned an invalid consumed prefix length")
        session = SpeculativeSessionCache(target_cache, draft_cache, tuple(ids[:consumed]), self._source_identity)
        self._local.cache = session
        self._local.stats = decoder.stats.report()
        # Alias consumed by current backend metrics, retaining the more
        # precise proposed count in the complete decoder report.
        self._local.stats["draft_tokens"] = decoder.stats.draft_tokens_proposed
        self._local.stats["algorithm"] = "exact_draft_target"
        if cache_callback is not None:
            cache_callback(session)

    def generate_with_cache(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> tuple[list[int], SpeculativeSessionCache]:
        generated = list(self.generate_iter(prompt_ids, max_tokens, **kwargs))
        return generated, self._local.cache

    def generate(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens, **kwargs))

    def speculative_stats(self) -> dict[str, Any]:
        return dict(getattr(self._local, "stats", {}))
