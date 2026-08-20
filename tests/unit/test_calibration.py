"""Tests for the calibration package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether.compiler.calibration import (
    CalibrationDataset,
    CustomJsonlDataset,
    HellaswagDataset,
    InlineCalibrationDataset,
    PerplexityEvaluator,
    SensitivityCalibration,
    WikiText2Dataset,
    get_dataset,
)
from aether.core.exceptions import CalibrationError
from aether.core.types import ModelArchitecture


def _require_local_dataset(dataset: CalibrationDataset) -> list[str]:
    """Use real named data when cached; mark the test environment-limited otherwise."""
    try:
        return list(dataset.iter_text())
    except CalibrationError as exc:
        pytest.skip(str(exc))


class TestDatasets:
    def test_wikitext2(self) -> None:
        ds = WikiText2Dataset(max_tokens=1000)
        texts = _require_local_dataset(ds)
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_hellaswag(self) -> None:
        ds = HellaswagDataset(max_tokens=500)
        texts = _require_local_dataset(ds)
        assert len(texts) > 0

    def test_custom_jsonl(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "calib.jsonl"
        with jsonl_file.open("w") as f:
            f.write(json.dumps({"text": "sample calibration text"}) + "\n")
            f.write(json.dumps({"text": "another sample"}) + "\n")
        ds = CustomJsonlDataset(str(jsonl_file), max_tokens=500)
        texts = list(ds.iter_text())
        assert len(texts) == 2

    def test_get_dataset_wikitext(self) -> None:
        ds = get_dataset("wikitext-2")
        assert isinstance(ds, WikiText2Dataset)

    def test_inline_dataset_respects_token_budget(self) -> None:
        ds = InlineCalibrationDataset(["one two three", "four five"], max_tokens=4)
        texts = list(ds.iter_limited_text())
        assert " ".join(texts).split() == ["one", "two", "three", "four"]
        assert ds.token_count() == 4


class TestPerplexityEvaluator:
    def test_evaluate_default(self) -> None:
        evaluator = PerplexityEvaluator(vocab_size=32000, model_params_b=1.0)
        dataset = InlineCalibrationDataset(["real test calibration text for perplexity"], max_tokens=100)
        result = evaluator.evaluate(dataset)
        assert result.perplexity > 1.0
        assert result.loss > 0.0
        assert result.num_tokens > 0

    def test_evaluate_with_precision(self) -> None:
        evaluator = PerplexityEvaluator(model_params_b=1.0)
        dataset = InlineCalibrationDataset(["real test calibration text for perplexity"], max_tokens=100)
        precision_map = {"layer_0": "BF16", "layer_1": "Q3_K"}
        result = evaluator.evaluate(dataset, precision_map)
        assert result.perplexity > 0
        assert result.details["precision_penalty"] > 0

    def test_more_aggressive_precision_has_higher_loss(self) -> None:
        evaluator = PerplexityEvaluator(model_params_b=1.0)
        dataset = InlineCalibrationDataset(["real test calibration text for perplexity"], max_tokens=100)
        bf16 = evaluator.evaluate(dataset, {"layer_0": "BF16", "layer_1": "BF16"})
        q3 = evaluator.evaluate(dataset, {"layer_0": "Q3_K", "layer_1": "Q3_K"})
        assert q3.loss > bf16.loss

    def test_compare(self) -> None:
        evaluator = PerplexityEvaluator(model_params_b=1.0)
        ds = InlineCalibrationDataset(["real test calibration text for perplexity"], max_tokens=100)
        baseline = evaluator.evaluate(ds)
        quantized = evaluator.evaluate(ds, {"layer_0": "Q3_K", "layer_1": "Q3_K"})
        comparison = evaluator.compare(baseline, quantized)
        assert "baseline_ppl" in comparison
        assert "relative_delta" in comparison


class TestSensitivityCalibration:
    def test_score_by_layer(self) -> None:
        arch = ModelArchitecture(
            family="llama_family",
            params_billion=1.0,
            layers=4,
            hidden_size=128,
            num_attention_heads=4,
        )
        cal = SensitivityCalibration(architecture=arch)
        ds = InlineCalibrationDataset(["real test calibration text for perplexity"], max_tokens=100)
        base_map = {f"layer_{i}": "BF16" for i in range(arch.layers)}
        base_map["embedding"] = "BF16"
        base_map["lm_head"] = "BF16"
        scores = cal.score_by_layer(ds, base_map)
        assert len(scores) == arch.layers + 2
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_score_summary(self) -> None:
        arch = ModelArchitecture(family="test", layers=4, hidden_size=64, num_attention_heads=4, params_billion=0.01)
        cal = SensitivityCalibration(architecture=arch)
        summary = cal.score_summary({"layer_0": 0.9, "layer_1": 0.5, "layer_2": 0.3, "layer_3": 0.1})
        assert summary["mean"] == pytest.approx(0.45, rel=0.1)
