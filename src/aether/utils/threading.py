"""
Threading and concurrency utilities.

Provides a small set of helpers for safe concurrent execution: a worker pool for
background tasks, a thread-safe rate limiter, and an async bridge for running
synchronous backend calls without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from aether.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class BackgroundWorker:
    """Simple thread-pool wrapper for background tasks."""

    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "aether") -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._shutdown = False

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> "asyncio.Future[T]":
        """Submit a function to the background pool."""
        if self._shutdown:
            raise RuntimeError("Worker has been shut down")
        return self._executor.submit(fn, *args, **kwargs)  # type: ignore[return-value]

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)

    def __repr__(self) -> str:
        return f"BackgroundWorker(max_workers={self._executor._max_workers})"


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, rate: float, burst: float | None = None) -> None:
        """Initialize a rate limiter.

        Args:
            rate: Tokens per second.
            burst: Maximum burst size. Defaults to rate.
        """
        self.rate = rate
        self.burst = burst or rate
        self._tokens = self.burst
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, blocking if necessary."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now
            if tokens > self._tokens:
                deficit = tokens - self._tokens
                wait_time = deficit / self.rate
                self._tokens = 0.0
        if wait_time := max(0.0, deficit / self.rate if "deficit" in dir() else 0.0):  # noqa: B023
            time.sleep(wait_time)
        return wait_time

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without blocking."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now
            if tokens <= self._tokens:
                self._tokens -= tokens
                return True
            return False

    def __repr__(self) -> str:
        return f"RateLimiter(rate={self.rate}, burst={self.burst})"


async def run_in_thread(
    fn: Callable[..., T],
    *args: Any,
    executor: ThreadPoolExecutor | None = None,
    **kwargs: Any,
) -> T:
    """Run a synchronous function in a thread pool without blocking the event loop.

    Args:
        fn: Synchronous function to execute.
        args: Positional arguments.
        executor: Optional executor. Uses asyncio default if not provided.
        kwargs: Keyword arguments.

    Returns:
        The function return value.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


class AtomicCounter:
    """Thread-safe atomic counter."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        """Atomically increment and return the new value."""
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta: int = 1) -> int:
        """Atomically decrement and return the new value."""
        with self._lock:
            self._value -= delta
            return self._value

    @property
    def value(self) -> int:
        """Return the current value."""
        with self._lock:
            return self._value

    def __repr__(self) -> str:
        return f"AtomicCounter({self.value})"
