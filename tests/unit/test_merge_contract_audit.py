"""Merging must preserve coordinates and require every requested source."""

from types import SimpleNamespace

import pytest

from aether.compiler.config import CompilerConfig
from aether.compiler.stage2_optimizer import pass12_model_merging as merging


@pytest.mark.parametrize("merger", [merging._TaskArithmeticMerger(), merging._DAREMerger(drop_prob=0), merging._TIESMerger()])
@pytest.mark.parametrize("vectors,coefficients,error", [
    ([{"weight": [1, 2]}, {"weight": [3]}], [1, 1], "mismatched"),
    ([{"weight": [1]}], [], "one coefficient"),
    ([{"weight": [1]}], [float("nan")], "finite"),
    ([{"weight": [float("inf")]}], [1], "finite"),
])
def test_public_mergers_reject_invalid_coordinate_contracts(merger, vectors, coefficients, error):
    with pytest.raises(ValueError, match=error):
        merger.merge(vectors, coefficients, None)


def test_task_arithmetic_preserves_partial_adapter_subsets():
    merged = merging._TaskArithmeticMerger().merge(
        [{"attention": [1, 2]}, {"ffn": [3, 4]}], [0.25, 0.5], None
    )
    assert merged == {"attention": [0.25, 0.5], "ffn": [1.5, 2]}


def test_ties_elects_weighted_magnitude_and_rejects_negative_mean_weights():
    assert merging._TIESMerger().merge([{"w": [1]}, {"w": [1]}, {"w": [-10]}], [1, 1, 1], None) == {"w": [-10]}
    with pytest.raises(ValueError, match="nonnegative"):
        merging._TIESMerger().merge([{"w": [1]}], [-1], None)


@pytest.mark.parametrize("probability", [-0.1, 1, 2, float("nan"), float("inf")])
def test_dare_rejects_invalid_drop_probability(probability):
    with pytest.raises(ValueError, match="probability"):
        merging._DAREMerger(drop_prob=probability)


def test_unknown_method_cannot_silently_use_task_arithmetic():
    with pytest.raises(ValueError, match="Unsupported"):
        merging._get_merger("slerp")


def test_nonfinite_base_and_final_weights_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        merging._compute_task_vectors({"w": [float("inf")]}, [{"w": [2]}])
    with pytest.raises(ValueError, match="overflowed"):
        merging._apply_delta({"w": [1e308]}, {"w": [1e308]})


@pytest.mark.parametrize("sources", [
    {"first": {"w": [2, 3]}, "second": {}},
    {"first": {"w": [2, 3]}, "second": {"unrelated": [4, 5]}},
    {"first": {}, "second": {}},
])
def test_pass_requires_each_requested_source_to_contribute_readable_matching_weights(monkeypatch, sources):
    monkeypatch.setattr(merging, "_load_source_weights", lambda source, graph: sources[source])
    graph = {"w": [1, 1]}
    config = CompilerConfig(enable_model_merging=True, model_merging_sources=list(sources))
    _, report = merging.ModelMergingPass().run(graph, None, config)
    assert report.status == "failed"
    assert graph == {"w": [1, 1]}


def test_pass_reports_actual_subset_updates_and_rejects_graph_without_writer(monkeypatch):
    monkeypatch.setattr(merging, "_load_source_weights", lambda source, graph: {"w": [3, 5]})
    config = CompilerConfig(enable_model_merging=True, model_merging_sources=["source"], model_merging_coefficients=[0.5])
    graph = {"w": [1, 1], "untouched": [7, 8]}
    _, report = merging.ModelMergingPass().run(graph, None, config)
    assert report.status == "applied"
    assert report.details["n_tensors_updated"] == 1
    assert graph == {"w": [2, 3], "untouched": [7, 8]}
    readonly = SimpleNamespace(parameters=lambda: {"w": [1, 1]}, named_parameters=lambda: [])
    _, report = merging.ModelMergingPass().run(readonly, None, config)
    assert report.status == "failed"
    assert "did not update" in report.details["error"]
