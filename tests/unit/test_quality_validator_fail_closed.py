"""Quality evidence, exact metric oracles and actual append-only engine execution."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aether.evaluation import QualityReport, QualityValidator, ValidationResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("passed", [False, True])
def test_empty_result_cannot_assert_a_pass(passed: bool) -> None:
    """A boolean without measurements is not quality evidence."""
    result = ValidationResult(passed=passed)
    assert not result.passed
    assert not QualityReport(runs=[result]).all_passed


def test_empty_report_does_not_pass_by_vacuous_truth() -> None:
    """An unexecuted report cannot certify an optimization."""
    assert not QualityReport().all_passed


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"reference_logits": [1.0, 2.0]},
        {"candidate_logits": [1.0, 2.0]},
        {"reference_tokens": [1]},
        {"candidate_tokens": [1]},
        {"reference_nll": [1.0]},
        {"candidate_nll": [1.0]},
        {"reference_activations": {"layer": [1.0]}},
        {"candidate_activations": {"layer": [1.0]}},
    ],
)
def test_missing_or_unpaired_evidence_is_rejected(evidence: dict[str, Any]) -> None:
    """A missing counterpart must not silently remove a requested check."""
    with pytest.raises(ValueError):
        QualityValidator().validate_all(**evidence)


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        ([], []),
        (1.0, 1.0),
        (np.zeros((0, 3)), np.zeros((0, 3))),
        (np.zeros((1, 0)), np.zeros((1, 0))),
        (np.ones((2, 3)), np.ones((1, 3))),
        (np.ones((1, 3)), np.ones((1, 1))),
        (np.ones((1, 2, 3)), np.ones((1, 2, 3))),
        ([1.0, 2.0], [1.0, float("nan")]),
        ([1.0, float("inf")], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, -float("inf")]),
        ([1j, 2j], [1j, 2j]),
        ([True, False], [True, False]),
        (["1", "2"], ["1", "2"]),
    ],
)
def test_logits_require_equal_nonempty_real_finite_shapes(reference: Any, candidate: Any) -> None:
    """Broadcasting, zip truncation and nonfinite data are not measurements."""
    with pytest.raises(ValueError):
        QualityValidator().validate_logits(reference, candidate)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), True])
def test_temperature_must_be_positive_finite(temperature: Any) -> None:
    """Invalid temperature must not become a clamped or NaN distribution."""
    with pytest.raises(ValueError):
        QualityValidator().validate_logits([1.0, 2.0], [1.0, 2.0], temperature)


@pytest.mark.parametrize(
    "field",
    [
        "max_kl_div",
        "max_mae",
        "min_cosine",
        "min_token_agreement",
        "max_perplexity_delta",
        "max_activation_error",
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_invalid_thresholds_cannot_disable_a_gate(field: str, invalid: Any) -> None:
    """All thresholds must be actual finite scalar measurements/policies."""
    with pytest.raises(ValueError):
        QualityValidator(**{field: invalid})


def test_kl_uses_log_probabilities_without_an_epsilon_ceiling() -> None:
    """KL([1,0] || [exp(-1000),1]) tends to 1000, not -log(epsilon)."""
    result = QualityValidator(max_mae=2000, min_cosine=-1).validate_logits(
        [1000.0, 0.0], [0.0, 1000.0]
    )
    assert result.kl_divergence == pytest.approx(1000.0, abs=1e-12)
    assert not result.passed


def test_localized_logit_regression_cannot_be_averaged_away() -> None:
    """One bad position must still fail the KL/cosine gate among 999 good ones."""
    reference = np.tile([3.0, 1.0], (1000, 1))
    candidate = reference.copy()
    candidate[-1] = [1.0, 3.0]
    result = QualityValidator().validate_logits(reference, candidate)
    assert result.kl_divergence < 0.01
    assert result.cosine_similarity > 0.999
    assert not result.passed


@pytest.mark.parametrize("values", [[0.0, 0.0], [1e300, -1e300], [1e-300, -1e-300]])
def test_exact_logits_have_stable_cosine_and_real_topk_width(values: list[float]) -> None:
    """Zero/equal and very large/small vectors must not lose exact agreement."""
    result = QualityValidator().validate_logits(values, values)
    assert result.passed
    assert result.cosine_similarity == pytest.approx(1.0)
    assert result.top5_agreement_rate == 1.0


def test_logits_match_torch_reference_kl_and_cross_entropy() -> None:
    """PyTorch's log-target KL implementation supplies an independent oracle."""
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    reference = torch.tensor([[12.0, -8.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
    candidate = torch.tensor([[10.0, -6.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    expected = functional.kl_div(
        functional.log_softmax(candidate, dim=-1),
        functional.log_softmax(reference, dim=-1),
        log_target=True,
        reduction="batchmean",
    ).item()
    result = QualityValidator().validate_logits(reference.numpy(), candidate.numpy())
    assert result.kl_divergence == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_missing_greedy_suffix_counts_as_disagreement() -> None:
    """Missing generated IDs cannot be dropped from the denominator."""
    result = QualityValidator().validate_greedy_agreement([1, 2], [1])
    assert (result.total_tokens, result.matched_tokens, result.first_divergence_pos) == (2, 1, 1)
    assert result.agreement_rate == 0.5
    assert not result.passed
    empty_candidate = QualityValidator().validate_greedy_agreement([1, 2], [])
    assert empty_candidate.agreement_rate == 0.0
    assert not empty_candidate.passed


@pytest.mark.parametrize("tokens", [[], [True], [1.0], [-1], ["1"]])
def test_greedy_evidence_cannot_be_empty_or_noninteger(tokens: list[Any]) -> None:
    """Token equality cannot validate malformed or absent generated sequences."""
    with pytest.raises(ValueError):
        QualityValidator().validate_greedy_agreement(tokens, tokens)


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([1.0], [float("nan")]),
        ([float("inf")], [1.0]),
        ([1.0], [-1.0]),
        ([[1.0]], [[1.0]]),
        ([True], [True]),
        (["1"], ["1"]),
    ],
)
def test_nll_evidence_requires_equal_nonempty_finite_nonnegative_vectors(
    reference: Any,
    candidate: Any,
) -> None:
    """PPL compares the same scored labels; no truncation or invalid loss is valid."""
    with pytest.raises(ValueError):
        QualityValidator().validate_perplexity(reference, candidate)


def test_perplexity_regression_above_the_old_cap_is_visible() -> None:
    """PPL ratio equals exp(candidate mean NLL - reference mean NLL)."""
    result = QualityValidator().validate_perplexity([101.0], [102.0])
    assert result.relative_delta == pytest.approx(math.expm1(1.0))
    assert not result.passed


def test_overflowed_absolute_ppl_preserves_exact_log_ratio_and_strict_json(tmp_path: Path) -> None:
    """Representability of absolute PPL must not cap or erase the relative test."""
    validator = QualityValidator()
    result = validator.validate_all(reference_nll=[1000.0], candidate_nll=[1000.02])
    assert result.perplexity is not None
    assert not result.passed
    assert result.perplexity.relative_delta == pytest.approx(math.expm1(0.02))
    assert result.perplexity.reference_perplexity is None
    assert result.perplexity.candidate_perplexity is None
    report = QualityReport(runs=[result])
    report.save(tmp_path / "report.json")
    encoded = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert not json.loads(encoded)["all_passed"]


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        ({}, {}),
        ({"x": [1.0]}, {}),
        ({"x": [1.0]}, {"other": [1.0]}),
        ({"x": [1.0]}, {"x": [1.0, 2.0]}),
        ({"x": []}, {"x": []}),
        ({"x": [1.0]}, {"x": [float("nan")]}),
        ({"x": [1.0]}, {"x": [float("inf")]}),
    ],
)
def test_missing_mismatched_or_nonfinite_activations_cannot_pass(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """A layer cannot disappear from the check because its evidence is invalid."""
    with pytest.raises(ValueError):
        QualityValidator().validate_activations(reference, candidate)


def test_changed_zero_reference_activation_is_a_real_failure() -> None:
    """An undefined nonzero/zero relative error must not be skipped as agreement."""
    validator = QualityValidator()
    assert validator.validate_activations({"x": [0.0]}, {"x": [0.0]}).passed
    assert not validator.validate_activations({"x": [0.0]}, {"x": [1.0]}).passed


@pytest.fixture
def actual_cpu_engine(tmp_path: Path, tiny_local_safetensors_model: Path) -> Any:
    """Compile, verify and load real offline weights; this is an AEG oracle."""
    from aether import Compiler, CompilerConfig
    from aether.runtime.aeg_loader import load_engine_from_path

    package = Compiler(
        CompilerConfig(
            quality_mode="diagnostic",
            targets=["cpu_generic"],
            skip_download=True,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "model.aeg")
    package.verify_integrity()
    return load_engine_from_path(package.root, prepare_native_kernels=False, allow_uncertified=True)


class TracedEngine:
    """Record the real engine's request inputs and actual resulting cache lengths."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.requests: list[list[tuple[list[int], int]]] = []

    def forward_step(self, ids: list[int], cache: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Delegate execution and retain actual cache growth for each request."""
        if cache is None:
            self.requests.append([])
        logits, updated = self.engine.forward_step(ids, cache, **kwargs)
        self.requests[-1].append((list(ids), updated.length))
        return logits, updated


def test_engine_comparison_prefills_once_and_appends_only_one_token(actual_cpu_engine: Any) -> None:
    """Real KV lengths must be 2,3,4, not the old repeated-prefix 2,5,9."""
    reference, candidate = TracedEngine(actual_cpu_engine), TracedEngine(actual_cpu_engine)
    result = QualityValidator().compare_engines(reference, candidate, [1, 2], max_tokens=3)
    for request in reference.requests + candidate.requests:
        assert [length for _, length in request] == [2, 3, 4]
        assert [len(ids) for ids, _ in request] == [2, 1, 1]
    assert result.logits is not None
    assert result.tokens is not None and result.tokens.agreement_rate == 1.0
    assert result.perplexity is None


def test_engine_ppl_scores_fixed_labels_against_fresh_full_prefix_oracle(
    actual_cpu_engine: Any,
) -> None:
    """Each supplied label is scored under its actual preceding source history."""
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    prompt, labels = [1, 2], [3, 5, 7]
    oracle = np.stack(
        [
            actual_cpu_engine.forward_step(prompt + labels[:index], None)[0]
            for index in range(len(labels))
        ]
    )
    expected = functional.cross_entropy(
        torch.tensor(oracle, dtype=torch.float64), torch.tensor(labels), reduction="mean"
    ).item()
    result = QualityValidator().compare_engines(
        actual_cpu_engine,
        actual_cpu_engine,
        prompt,
        max_tokens=3,
        continuation_ids=labels,
    )
    assert result.passed
    assert result.logits is not None
    assert result.perplexity is not None and result.perplexity.num_tokens == len(labels)
    assert result.perplexity.reference_mean_nll == pytest.approx(expected, abs=1e-6)
    assert result.perplexity.candidate_mean_nll == pytest.approx(expected, abs=1e-6)
    assert result.perplexity.relative_delta == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [{"token_ids": []}, {"token_ids": [True]}, {"token_ids": [1], "max_tokens": 0}],
)
def test_invalid_engine_request_is_rejected_before_execution(kwargs: dict[str, Any]) -> None:
    """Malformed validation requests must fail before calling either engine."""
    with pytest.raises(ValueError):
        QualityValidator().compare_engines(object(), object(), **kwargs)


def test_derived_ppl_percentage_overflow_is_explicit_strict_json(tmp_path: Path) -> None:
    """A finite ratio can overflow only when converted to percentage units."""
    result = QualityValidator().validate_all(reference_nll=[0.0], candidate_nll=[709.0])
    report = QualityReport(runs=[result])
    report.save(tmp_path / "report.json")
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["runs"][0]
    assert payload["perplexity"]["relative_delta"] > 1e307
    assert payload["perplexity"]["relative_delta_pct"] is None
    assert payload["nonfinite_metrics"]["perplexity.relative_delta_pct"] == "positive_infinity"
    assert not payload["passed"]


def test_fixed_labels_outside_vocabulary_are_rejected_before_teacher_forcing(
    actual_cpu_engine: Any,
) -> None:
    """Invalid labels must not become inputs to the engine's next cached step."""
    reference, candidate = TracedEngine(actual_cpu_engine), TracedEngine(actual_cpu_engine)
    with pytest.raises(ValueError, match="vocab"):
        QualityValidator().compare_engines(
            reference,
            candidate,
            [1, 2],
            max_tokens=2,
            continuation_ids=[1000, 3],
        )
    assert len(reference.requests) == len(candidate.requests) == 1
    assert all(len(request) == 2 for request in reference.requests + candidate.requests)


@pytest.mark.parametrize("buffer_dtype", [np.float32, np.float64])
def test_reused_engine_output_buffer_cannot_rewrite_prior_quality_evidence(
    actual_cpu_engine: Any,
    buffer_dtype: Any,
) -> None:
    """Preallocated logit scratch storage must be snapshotted at measurement time."""

    class ReusingEngine:
        def __init__(self) -> None:
            self.buffer: Any = None

        def forward_step(self, ids: list[int], cache: Any, **kwargs: Any) -> tuple[Any, Any]:
            """Return real engine logits through one intentionally reused buffer."""
            logits, cache = actual_cpu_engine.forward_step(ids, cache, **kwargs)
            if self.buffer is None:
                self.buffer = np.empty(logits.shape, dtype=buffer_dtype)
            np.copyto(self.buffer, logits)
            return self.buffer, cache

    result = QualityValidator().compare_engines(
        actual_cpu_engine,
        ReusingEngine(),
        [1, 2],
        max_tokens=3,
        continuation_ids=[3, 5, 7],
    )
    assert result.passed
    assert result.logits is not None and result.logits.max_absolute_error == 0.0


def test_mutated_invalid_policy_cannot_disable_later_perplexity_gate() -> None:
    """Public threshold changes must retain the same validation as construction."""
    validator = QualityValidator()
    with pytest.raises(ValueError):
        validator.max_perplexity_delta = float("inf")
        validator.validate_perplexity([1.0], [2.0])


def test_required_evidence_cannot_be_replaced_by_a_passing_subset() -> None:
    """Production callers can require complete metrics instead of a token-only pass."""
    with pytest.raises(ValueError, match="Required quality evidence"):
        QualityValidator().validate_all(
            reference_tokens=[1, 2],
            candidate_tokens=[1, 2],
            required_checks=("logits", "tokens", "perplexity"),
        )


def test_result_boolean_string_cannot_become_a_report_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mutated boolean string remains malformed, even with passing measurements."""
    result = QualityValidator().validate_all(reference_tokens=[1], candidate_tokens=[1])
    monkeypatch.setattr(result, "passed", "false")
    assert result.to_dict()["passed"] is False
    report = QualityReport(runs=[result])
    assert not report.all_passed
    assert report.pass_rate == 0.0


def test_removing_a_required_measurement_invalidates_a_previously_passing_result() -> None:
    """The complete required check set must survive beyond validate_all()."""
    result = QualityValidator().validate_all(
        reference_logits=[1.0, 2.0],
        candidate_logits=[1.0, 2.0],
        reference_tokens=[1],
        candidate_tokens=[1],
        reference_nll=[1.0],
        candidate_nll=[1.0],
        required_checks=("logits", "tokens", "perplexity"),
    )
    assert result.passed
    result.perplexity = None
    assert not result.evidence_passed
    assert not result.to_dict()["passed"]
    assert not QualityReport(runs=[result]).all_passed


def test_empty_token_dto_cannot_assert_executed_evidence() -> None:
    """Public result constructors cannot turn zero observations into a pass."""
    from aether.evaluation.quality_validator import TokenValidation

    tokens = TokenValidation(0, 0, 1.0, None, True, 0.95)
    assert not ValidationResult(tokens=tokens, passed=True).evidence_passed


def test_wrong_result_component_type_cannot_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passing logit DTO cannot substitute for required token evidence."""
    validator = QualityValidator()
    result = validator.validate_all(reference_tokens=[1], candidate_tokens=[1])
    monkeypatch.setattr(result, "tokens", validator.validate_logits([1.0, 2.0], [1.0, 2.0]))
    assert not result.evidence_passed


def test_mutated_nonfinite_token_rate_cannot_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reports must reject nonfinite result fields rather than trust a stale flag."""
    result = QualityValidator().validate_all(reference_tokens=[1], candidate_tokens=[1])
    assert result.tokens is not None
    monkeypatch.setattr(result.tokens, "agreement_rate", float("nan"))
    assert not result.evidence_passed


@pytest.mark.parametrize("kind", ["logits", "nll", "activations"])
def test_masked_observations_cannot_become_complete_evidence(kind: str) -> None:
    """NumPy conversion must not silently discard a mask denoting missing data."""
    missing = np.array([1.0, 2.0]).view(type=np.ma.MaskedArray)
    missing.mask = [False, True]
    validator = QualityValidator()
    with pytest.raises(ValueError, match="masked"):
        if kind == "logits":
            validator.validate_logits(missing, missing)
        elif kind == "nll":
            validator.validate_perplexity(missing, missing)
        else:
            validator.validate_activations({"x": missing}, {"x": missing})


def test_zero_agreement_threshold_still_requires_both_generated_sequences() -> None:
    """A permissive threshold cannot turn an absent generation into evidence."""
    assert not QualityValidator(min_token_agreement=0).validate_greedy_agreement([1], []).passed


@pytest.mark.parametrize("kind", ["logits", "perplexity", "activations"])
def test_inconsistent_finite_metric_summaries_cannot_pass(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite fields must still agree with extrema/representability and means."""
    result = QualityValidator().validate_all(
        reference_logits=[1.0, 2.0],
        candidate_logits=[1.0, 2.0],
        reference_nll=[1.0],
        candidate_nll=[1.0],
        reference_activations={"x": [1.0]},
        candidate_activations={"x": [1.0]},
    )
    if kind == "logits":
        assert result.logits is not None
        monkeypatch.setattr(result.logits, "kl_divergence", 1.0)
    elif kind == "perplexity":
        assert result.perplexity is not None
        monkeypatch.setattr(result.perplexity, "reference_perplexity", None)
    else:
        assert result.activations is not None
        monkeypatch.setattr(result.activations, "mean_layer_error", 1.0)
    assert not result.evidence_passed


def test_same_aeg_cpu_and_torch_cpu_pass_complete_fixed_label_comparison(
    actual_cpu_engine: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One real artifact executes on both engines with all three measured checks."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
    portable = TorchAEGEngine(actual_cpu_engine, "cpu", retain_host_weights=True)
    result = QualityValidator().compare_engines(
        actual_cpu_engine,
        portable,
        [1, 2],
        max_tokens=3,
        continuation_ids=[3, 5, 7],
    )
    assert result.passed and result.evidence_passed
    assert result.required_checks == ("logits", "tokens", "perplexity")
    assert result.logits is not None and result.logits.min_cosine_similarity >= 0.999
    assert result.tokens is not None and result.tokens.agreement_rate == 1.0
    assert result.perplexity is not None and result.perplexity.passed
