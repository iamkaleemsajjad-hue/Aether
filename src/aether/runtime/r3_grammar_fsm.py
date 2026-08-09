"""
R3 — Grammar FSM Runtime Engine.

Structured output generation (JSON, code, XML, regex-constrained text) is
enforced at decode time by masking invalid tokens before sampling.  The FSM
engine loads the pre-compiled FSA blob from ``.aeg/grammar/fsm.bin`` (produced
by Pass 11) and provides O(1) mask lookups per decode step.

Core guarantee: 100% syntactic validity with < 50 µs overhead per step.

Engine lifecycle:
  1. ``load(fsm_bin_path)``: Load FSA from disk into memory.
  2. ``reset()``: Reset FSA to initial state for a new request.
  3. ``get_token_mask(state) → bytearray``: Return the bitmask of valid tokens
     for the current state.
  4. ``advance(token_id) → next_state``: Update FSA state after token acceptance.
  5. ``is_accepting() → bool``: Check if current state is an accepting state.

Token mask format:
  - bytearray of length ceil(vocab_size / 8).
  - Bit i is set iff token i is valid from the current state.
  - Consumed directly by sampling kernels (no conversion needed).

Multi-grammar support:
  - Each request carries an optional grammar_id.
  - The engine maintains a per-request FSA state machine instance.
  - Multiple concurrent requests can use different grammars.

Research basis:
  - XGrammar (MLC 2026): <50µs structured output enforcement.
  - LLGuidance (MSR 2026): regex + CFG constraint decoding.
  - CRANE (ICML 2026): structure-aligned beam decoding.
  - Synchromesh (2023): constraint decoding via completion engine.
"""

from __future__ import annotations

import json
import math
import struct
import threading
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

# FSA binary format magic (must match Pass 11 writer).
_FSA_MAGIC = b"AETHER_FSA_v1\x00\x00\x00"
_HEADER_SIZE = 64  # bytes


class GrammarFSMEngine:
    """Runtime R3: Grammar FSM enforcement engine.

    Loads pre-compiled FSA blobs and enforces token masks at decode time.
    Thread-safe for concurrent multi-request use via per-request state instances.
    """

    def __init__(self) -> None:
        self._grammars: dict[str, "_LoadedFSA"] = {}
        self._grammar_metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stats = _FSMStats()

    def load(self, fsm_bin_path: str, grammar_id: str = "default") -> bool:
        """Load a pre-compiled FSA binary blob.

        Args:
            fsm_bin_path: Path to the ``.aeg/grammar/fsm.bin`` file.
            grammar_id: Identifier for this grammar (used in per-request lookup).

        Returns:
            True if loaded successfully, False on error.
        """
        p = Path(fsm_bin_path)
        if not p.exists():
            logger.warning("R3: FSM binary not found at %s.", fsm_bin_path)
            return False

        try:
            fsa = _LoadedFSA.from_binary(p)
            with self._lock:
                self._grammars[grammar_id] = fsa
            logger.info(
                "R3: Loaded grammar %r — %d states, vocab=%d, ~%.1f µs/step.",
                grammar_id,
                fsa.n_states,
                fsa.vocab_size,
                fsa.estimated_mask_lookup_us,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("R3: Failed to load FSM from %s: %s", fsm_bin_path, exc)
            return False

    def load_from_config(self, aeg_dir: str, grammar_id: str = "default") -> bool:
        """Load FSA using the grammar config JSON in an AEG directory."""
        config_path = Path(aeg_dir) / "grammar" / "fsm_config.json"
        if not config_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            bin_file = config.get("blob_file", "fsm.bin")
            bin_path = Path(aeg_dir) / "grammar" / bin_file
            loaded = self.load(str(bin_path), grammar_id)
            if loaded:
                with self._lock:
                    self._grammar_metadata[grammar_id] = config
            return loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("R3: Failed to load from config %s: %s", config_path, exc)
            return False

    def create_session(self, grammar_id: str = "default") -> "_FSMSession":
        """Create a new per-request FSA session.

        Each request gets its own session with independent state tracking.
        The underlying FSA data is shared (read-only, zero-copy).

        Args:
            grammar_id: Which grammar to use.

        Returns:
            FSMSession for use during decoding.

        Raises:
            KeyError: If grammar_id is not loaded.
        """
        with self._lock:
            if grammar_id not in self._grammars:
                raise KeyError(
                    f"Grammar {grammar_id!r} not loaded. Call load() first."
                )
            fsa = self._grammars[grammar_id]

        session = _FSMSession(fsa=fsa, engine_stats=self._stats)
        self._stats.sessions_created += 1
        return session

    def is_loaded(self, grammar_id: str = "default") -> bool:
        """Check if a grammar is loaded."""
        with self._lock:
            return grammar_id in self._grammars

    @property
    def loaded_grammars(self) -> list[str]:
        """List of loaded grammar IDs."""
        with self._lock:
            return list(self._grammars.keys())

    def matches_compiled_constraint(self, source: str, grammar_id: str = "default") -> bool:
        """Check that a request matches a trusted, tokenizer-aware compiled FSA."""
        import hashlib

        with self._lock:
            metadata = self._grammar_metadata.get(grammar_id)
        expected = str(metadata.get("schema_hash", "")) if metadata else ""
        if not metadata or metadata.get("tokenizer_aware") is not True:
            return False
        actual = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return bool(expected) and expected == actual

    @property
    def stats(self) -> "_FSMStats":
        return self._stats


class _FSMSession:
    """Per-request FSA state machine session.

    Not thread-safe: intended for use by a single request's decode loop.
    """

    def __init__(self, fsa: "_LoadedFSA", engine_stats: "_FSMStats") -> None:
        self._fsa = fsa
        self._current_state: int = fsa.initial_state
        self._step_count: int = 0
        self._engine_stats = engine_stats

    def reset(self) -> None:
        """Reset to initial FSA state (for request reuse)."""
        self._current_state = self._fsa.initial_state
        self._step_count = 0

    def get_token_mask(self) -> bytearray:
        """Return the bitmask of valid next tokens for the current FSA state.

        Performance: O(1) lookup from pre-built mask table.
        Returns a bytearray of length ceil(vocab_size / 8).
        """
        mask = self._fsa.get_mask(self._current_state)
        self._engine_stats.mask_lookups += 1
        return mask

    def advance(self, token_id: int) -> int:
        """Advance the FSA by consuming token_id.

        Args:
            token_id: The token ID that was sampled (must be valid in current mask).

        Returns:
            New state index (-1 if invalid transition).
        """
        next_state = self._fsa.transition(self._current_state, token_id)
        if next_state >= 0:
            self._current_state = next_state
            self._step_count += 1
            self._engine_stats.tokens_advanced += 1
        else:
            logger.debug(
                "R3: Invalid transition from state %d on token %d.",
                self._current_state,
                token_id,
            )
            self._engine_stats.invalid_transitions += 1
        return next_state

    def is_accepting(self) -> bool:
        """Return True if the current state is an accepting (terminal) state."""
        return self._current_state in self._fsa.accepting_states

    def is_valid_token(self, token_id: int) -> bool:
        """Check if a single token is valid from the current state."""
        mask = self.get_token_mask()
        if token_id < 0 or token_id >= self._fsa.vocab_size:
            return False
        return bool(mask[token_id // 8] & (1 << (token_id % 8)))

    @property
    def current_state(self) -> int:
        return self._current_state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def vocab_size(self) -> int:
        return self._fsa.vocab_size


class _LoadedFSA:
    """An FSA loaded from a binary blob.  Read-only after construction."""

    def __init__(
        self,
        n_states: int,
        vocab_size: int,
        initial_state: int,
        accepting_states: frozenset[int],
        transitions: dict[tuple[int, int], int],
        token_masks: dict[int, bytearray],
        estimated_mask_lookup_us: float = 5.0,
    ) -> None:
        self.n_states = n_states
        self.vocab_size = vocab_size
        self.initial_state = initial_state
        self.accepting_states = accepting_states
        self._transitions = transitions
        self._token_masks = token_masks
        self.estimated_mask_lookup_us = estimated_mask_lookup_us
        self._empty_mask = bytearray(math.ceil(vocab_size / 8))

    def get_mask(self, state: int) -> bytearray:
        """Return token bitmask for state.  O(1) dict lookup."""
        return self._token_masks.get(state, self._empty_mask)

    def transition(self, state: int, token_id: int) -> int:
        """Return next state, or -1 for invalid transition."""
        return self._transitions.get((state, token_id), -1)

    @classmethod
    def from_binary(cls, path: Path) -> "_LoadedFSA":
        """Deserialize FSA from binary blob (format defined by Pass 11)."""
        data = path.read_bytes()

        # Verify magic.
        if data[:16] != _FSA_MAGIC:
            raise ValueError(f"Invalid FSA magic in {path}")

        # Parse header.
        (n_states, n_transitions, vocab_size, initial_state,
         n_accepting, mask_bytes_per_state, _version) = struct.unpack_from("<7I", data, 16)

        offset = _HEADER_SIZE

        # Parse accepting states.
        accepting_states: set[int] = set()
        for _ in range(n_accepting):
            s_id, = struct.unpack_from("<I", data, offset)
            accepting_states.add(s_id)
            offset += 4

        # Parse transitions.
        transitions: dict[tuple[int, int], int] = {}
        for _ in range(n_transitions):
            state, token, next_state = struct.unpack_from("<III", data, offset)
            transitions[(state, token)] = next_state
            offset += 12

        # Parse token masks.
        token_masks: dict[int, bytearray] = {}
        for state_idx in range(n_states):
            mask_bytes = data[offset: offset + mask_bytes_per_state]
            token_masks[state_idx] = bytearray(mask_bytes)
            offset += mask_bytes_per_state

        # Load estimated latency from config json if present.
        config_path = path.parent / "fsm_config.json"
        est_us = 5.0
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                est_us = float(cfg.get("estimated_mask_lookup_us", est_us))
            except Exception:  # noqa: BLE001
                pass

        return cls(
            n_states=n_states,
            vocab_size=vocab_size,
            initial_state=initial_state,
            accepting_states=frozenset(accepting_states),
            transitions=transitions,
            token_masks=token_masks,
            estimated_mask_lookup_us=est_us,
        )


class _FSMStats:
    __slots__ = ("mask_lookups", "tokens_advanced", "invalid_transitions", "sessions_created")

    def __init__(self) -> None:
        self.mask_lookups = 0
        self.tokens_advanced = 0
        self.invalid_transitions = 0
        self.sessions_created = 0
