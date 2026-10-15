"""Estimator failure/convergence contracts; these fixtures do not validate hardware."""

import itertools
import sys
import time
from types import SimpleNamespace

import pytest

from aether.placement import census


class _Clock:
    def __init__(self, durations, tensor):
        self.durations = iter(durations)
        self.tensor = tensor
        self.previous_calls = 0
        self.current = 0.0
        self.start = True

    def __call__(self):
        if not self.start:
            count = self.tensor.calls - self.previous_calls
            self.current += next(self.durations) * count / 8
            self.previous_calls = self.tensor.calls
        self.start = not self.start
        return self.current


def _probe(monkeypatch, durations, *, device="cpu", passes=20):
    tensor = SimpleNamespace(calls=0)

    def add(_value):
        tensor.calls += 1

    tensor.add_ = add
    framework = SimpleNamespace(
        device=lambda name: SimpleNamespace(type=name),
        ones=lambda *_args, **_kwargs: tensor,
        float32="float32",
    )
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, "torch", framework)
        scope.setattr(time, "perf_counter", _Clock(durations, tensor))
        scope.setattr(census, "_DISPATCH_PROBE_PASSES", passes)
        scope.setattr(census, "_DISPATCH_PROBE_MIN_SECONDS", 0.03)
        result = census.measure_dispatch_seconds(device, operations=8)
    return result, tensor.calls


def test_probe_discards_cold_passes_and_does_not_publish_an_isolated_minimum(monkeypatch):
    """Warmup and one fast clock sample cannot become the settled estimate."""
    result, calls = _probe(monkeypatch, [0.1] * 3 + [0.005, 0.0025] + [0.005] * 15)
    assert result == pytest.approx(0.005 / 8)
    assert calls >= 8 * (3 + 1 + 7)


def test_unsettled_probe_returns_unknown_instead_of_an_unqualified_measurement(monkeypatch):
    """An oscillating scheduler cannot silently replace a measured ledger entry."""
    durations = itertools.chain([0.1] * 3 + [0.005], itertools.cycle([0.002, 0.01]))
    result, calls = _probe(monkeypatch, durations)
    assert result == 0
    assert calls >= 3 + 1 + 20


def test_a_slow_pilot_cannot_prevent_sufficient_actual_timed_work(monkeypatch):
    """Re-size blocks using measured duration after a contaminated pilot."""
    result, _ = _probe(monkeypatch, [0.1] * 3 + [0.05] + [0.005] * 20)
    assert result == pytest.approx(0.005 / 8)


@pytest.mark.parametrize("bad_duration", [0.0, float("nan"), float("inf")])
def test_nonfinite_or_zero_clock_intervals_are_not_measurements(monkeypatch, bad_duration):
    """Invalid timing never reaches calibration as a positive measured cost."""
    result, _ = _probe(monkeypatch, [0.01] * 3 + [bad_duration])
    assert result == 0


def test_async_device_without_a_completion_barrier_is_not_probed(monkeypatch):
    """An unobservable work queue must not be timed as synchronous execution."""
    result, calls = _probe(monkeypatch, [], device="unregistered_async")
    assert result == calls == 0


@pytest.mark.parametrize("operations", [0, -1, True, 1.5])
def test_probe_rejects_invalid_operation_counts(operations):
    """Reject invalid control values independently of optional framework presence."""
    with pytest.raises(ValueError, match="positive integer"):
        census.measure_dispatch_seconds("cpu", operations=operations)
