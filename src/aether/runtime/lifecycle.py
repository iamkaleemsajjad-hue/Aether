"""Concurrent request admission and owned-stream shutdown for Runtime."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, Concatenate, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class RuntimeLifecycle:
    """Readers run concurrently; closing permanently rejects new admission."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._local = threading.local()
        self._active = 0
        self._exclusive_owner: int | None = None
        self._close_owner: int | None = None
        self._closing = self._closed = False
        self._streams: set[RuntimeStream[Any]] = set()
        self._sessions: dict[tuple[str, str], _SessionEntry] = {}

    @contextmanager
    def session(
        self, model_id: str, session_id: Any, *, nested: bool = False
    ) -> Iterator[tuple[str, str] | None]:
        """Serialize one mutable session without copying its full tensor state.

        Locks are not thread-owned: a paused iterator may be closed by the
        Runtime owner on another thread. Entry references include waiters and
        disappear after the final operation, preventing an unbounded lock map.
        """
        if not isinstance(session_id, str):
            yield None
            return
        key = (model_id, session_id)
        owner = threading.current_thread()
        with self._condition:
            entry = self._sessions.get(key)
            if entry is not None and entry.owner is owner:
                if nested and getattr(self._local, "session_preparation", None) == key:
                    reentrant = True
                else:
                    raise RuntimeError(
                        "This session already has an active operation or stream; "
                        "close it before reusing the session"
                    )
            else:
                reentrant = False
                if entry is None:
                    entry = self._sessions[key] = _SessionEntry()
                entry.users += 1
        if reentrant:
            yield key
            return
        acquired = False
        try:
            entry.lock.acquire()
            acquired = True
            with self._condition:
                entry.owner = owner
            yield key
        finally:
            with self._condition:
                if acquired:
                    entry.owner = None
                entry.users -= 1
                if entry.users == 0:
                    self._sessions.pop(key, None)
            if acquired:
                entry.lock.release()

    @contextmanager
    def preparing_session(self, key: tuple[str, str] | None) -> Iterator[None]:
        """Allow automatic restoration to reuse its already-owned session lease."""
        previous = getattr(self._local, "session_preparation", None)
        self._local.session_preparation = key
        try:
            yield
        finally:
            self._local.session_preparation = previous

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Admit concurrent work, reusing this thread's existing outer lease."""
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth = depth
            return
        with self._condition:
            while self._exclusive_owner is not None and not self._closing:
                self._condition.wait()
            if self._closing or self._closed:
                raise RuntimeError("Runtime is closing or closed")
            self._active += 1
            self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @contextmanager
    def exclusive(self, before_wait: Callable[[], None] | None = None) -> Iterator[None]:
        """Rare destructive mutations wait for existing readers without lock inversion."""
        if getattr(self._local, "depth", 0):
            raise RuntimeError("Cannot remove a model from inside an active Runtime operation")
        with self._condition:
            while self._exclusive_owner is not None and not self._closing:
                self._condition.wait()
            if self._closing or self._closed:
                raise RuntimeError("Runtime is closing or closed")
            self._exclusive_owner = threading.get_ident()
        try:
            if before_wait is not None:
                before_wait()
            with self._condition:
                while self._active:
                    self._condition.wait()
                if self._closing:
                    raise RuntimeError("Runtime is closing or closed")
                self._active = 1
                self._local.depth = 1
            try:
                yield
            finally:
                self._local.depth = 0
                with self._condition:
                    self._active -= 1
        finally:
            with self._condition:
                self._exclusive_owner = None
                self._condition.notify_all()

    def register(self, stream: RuntimeStream[Any]) -> None:
        """Retain an iterator until completion or explicit owner shutdown."""
        with self._condition:
            self._streams.add(stream)

    def unregister(self, stream: RuntimeStream[Any]) -> None:
        """Release the owner reference after iterator cleanup."""
        with self._condition:
            self._streams.discard(stream)

    def cancel_streams(self, model_ids: set[str] | None = None) -> list[BaseException]:
        """Close matching paused iterators and cancel currently executing steps."""
        with self._condition:
            streams = tuple(self._streams)
        errors = []
        for stream in streams:
            if model_ids is not None and stream.model_id not in model_ids:
                continue
            try:
                stream.request_close()
            except BaseException as exc:
                errors.append(exc)
        return errors

    def begin_close(self, timeout: float) -> tuple[bool, list[BaseException]]:
        """Claim shutdown, stop new admission, and wait boundedly for current work."""
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout < 0
            or not math.isfinite(timeout)
        ):
            raise ValueError("Runtime close timeout must be finite and nonnegative")
        if getattr(self._local, "depth", 0):
            raise RuntimeError("Cannot close Runtime from inside an active operation")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._close_owner is not None:
                if self._close_owner == threading.get_ident():
                    raise RuntimeError("Runtime close cannot recursively close itself")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Runtime shutdown is still in progress")
                self._condition.wait(remaining)
            if self._closed:
                return False, []
            self._closing = True
            self._close_owner = threading.get_ident()
            self._condition.notify_all()
        try:
            errors = self.cancel_streams()
            with self._condition:
                while self._active or self._exclusive_owner is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Runtime has an executing operation; shutdown has not completed"
                        )
                    self._condition.wait(remaining)
            return True, errors
        except BaseException:
            self.finish_close(False)
            raise

    def finish_close(self, succeeded: bool) -> None:
        """Publish completion or permit a retry after genuine cleanup failure."""
        with self._condition:
            if succeeded:
                self._closed = True
            self._close_owner = None
            self._condition.notify_all()


class RuntimeStream(Iterator[R], Generic[R]):
    """Owned iterator whose paused backend state can be closed without waiting for next()."""

    def __init__(
        self, lifecycle: RuntimeLifecycle, iterator: Iterator[R], model_id: str | None
    ) -> None:
        self._lifecycle, self._iterator = lifecycle, iterator
        self.model_id = model_id
        self._step_lock = threading.RLock()
        self._cancelled = threading.Event()
        self._closed = False
        lifecycle.register(self)

    def __iter__(self) -> RuntimeStream[R]:
        return self

    def __next__(self) -> R:
        with self._step_lock:
            if self._closed or self._cancelled.is_set():
                self._close_inner()
                raise StopIteration
            try:
                with self._lifecycle.operation():
                    try:
                        return next(self._iterator)
                    except BaseException:
                        self._close_inner()
                        raise
                    finally:
                        if self._cancelled.is_set():
                            self._close_inner()
            finally:
                if self._cancelled.is_set():
                    self._close_inner()

    def _close_inner(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        finally:
            self._lifecycle.unregister(self)

    def request_close(self) -> None:
        """Cancel a running step without blocking its owner on the kernel."""
        self._cancelled.set()
        if self._step_lock.acquire(blocking=False):
            try:
                self._close_inner()
            finally:
                self._step_lock.release()

    def close(self) -> None:
        """Close the iterator once its executing step has returned."""
        self._cancelled.set()
        with self._step_lock:
            self._close_inner()


_initialization_lock = threading.Lock()


class _SessionEntry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0
        self.owner: threading.Thread | None = None


def runtime_lifecycle(runtime: Any) -> RuntimeLifecycle:
    """Return the instance admission owner, initializing legacy test instances."""
    state = getattr(runtime, "_lifecycle", None)
    if state is None:
        # Some existing low-level tests construct Runtime with __new__. Lazy
        # initialization gives those instances the same admission contract.
        with _initialization_lock:
            state = getattr(runtime, "_lifecycle", None)
            if state is None:
                state = runtime._lifecycle = RuntimeLifecycle()
    return state


def runtime_operation(method: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
    """Preserve a method's signature while admitting its complete operation."""

    @wraps(method)
    def guarded(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        with runtime_lifecycle(self).operation():
            return method(self, *args, **kwargs)

    return guarded


def runtime_exclusive(method: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
    """Guard cached-model removal and close only the affected paused streams."""

    @wraps(method)
    def guarded(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        lifecycle = runtime_lifecycle(self)

        def cancel_model_streams() -> None:
            from pathlib import Path

            from aether.core.exceptions import ModelNotFoundError
            from aether.utils.file_io import cached_model_paths

            model = args[0] if args else kwargs.get("model_id")
            paths = cached_model_paths(model, self.config.model_cache_dir)
            if not paths:
                raise ModelNotFoundError(f"Cached model {model} not found", model_id=model)
            aliases = {model}
            with self._lock:
                for alias, (_, resolved) in self._aeg_paths.items():
                    if resolved and Path(resolved) in paths:
                        aliases.add(alias)
            # A blocked same-session request already owns an admission lease.
            # Releasing its paused stream must precede waiting for active=0.
            errors = lifecycle.cancel_streams(aliases)
            if errors:
                raise RuntimeError(f"Runtime stream cleanup failed: {errors[0]}") from errors[0]

        with lifecycle.exclusive(before_wait=cancel_model_streams):
            return method(self, *args, **kwargs)

    return guarded


def runtime_session_operation(
    method: Callable[Concatenate[T, P], R],
) -> Callable[Concatenate[T, P], R]:
    """Guard persistence/restoration under the same mutable session lease."""

    @wraps(method)
    def guarded(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        lifecycle = runtime_lifecycle(self)
        model = args[0] if args else kwargs.get("model_id")
        session = args[1] if len(args) > 1 else kwargs.get("session_id")
        with lifecycle.operation(), lifecycle.session(model, session, nested=True):
            return method(self, *args, **kwargs)

    return guarded


def runtime_stream(
    method: Callable[Concatenate[T, P], Iterator[R]],
) -> Callable[Concatenate[T, P], RuntimeStream[R]]:
    """Preserve typed iteration while registering the Runtime-owned iterator."""

    @wraps(method)
    def guarded(self: T, *args: P.args, **kwargs: P.kwargs) -> RuntimeStream[R]:
        lifecycle = runtime_lifecycle(self)
        with lifecycle.operation():
            model_id = args[0] if args else kwargs.get("model_id")
            return RuntimeStream(lifecycle, method(self, *args, **kwargs), model_id)

    return guarded
