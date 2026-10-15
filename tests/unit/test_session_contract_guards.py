"""Storage-domain guards inspect declarations; they do not execute CUDA hardware."""
import pytest

from aether.runtime.execution_features import _PooledEngine
from aether.runtime.session_contract import contract_key, shared_cache_contract
from aether.runtime.spec_generation import SpeculativeGenerationAdapter


class DeclaredStorage:
    """Only a storage descriptor, with no fabricated forward/validation method."""
    def __init__(self, device):
        self.device = device


def test_known_nested_wrappers_preserve_underlying_device_and_representation():
    """Two declared device domains remain distinct through both known wrappers."""
    contract = {"schema": "test_execution_identity"}
    first, second = DeclaredStorage("cuda:0"), DeclaredStorage("cuda:1")
    wrapped_first = _PooledEngine(SpeculativeGenerationAdapter(first, first), None, None)
    wrapped_second = _PooledEngine(SpeculativeGenerationAdapter(second, second), None, None)
    a = shared_cache_contract(contract, engine=wrapped_first)
    b = shared_cache_contract(contract, engine=wrapped_second)
    assert a == shared_cache_contract(contract, engine=first)
    assert b == shared_cache_contract(contract, engine=second)
    assert a["cache_storage"]["device"] == "cuda:0"
    assert b["cache_storage"]["device"] == "cuda:1"
    assert contract_key(a) != contract_key(b)


def test_unknown_missing_storage_device_is_not_assumed_cpu():
    """An absent declaration must not manufacture a CPU storage claim."""
    with pytest.raises(ValueError, match="does not declare"):
        shared_cache_contract({}, engine=DeclaredStorage(None))


def test_wrapper_cycles_fail_without_unbounded_recursion():
    """Malformed wrapper ownership fails finitely instead of recursing."""
    target = DeclaredStorage("cpu")
    wrapped = SpeculativeGenerationAdapter(target, target)
    wrapped.target = wrapped
    with pytest.raises(ValueError, match="cycle"):
        shared_cache_contract({}, engine=wrapped)
