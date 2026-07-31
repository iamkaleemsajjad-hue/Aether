"""Tests for weight pruning and sparsity mask computation.

Pass 9 previously emitted only a plan dict; these tests cover the real mask math
that replaced it — importance metrics, exact N:M structure, and the fallbacks
that keep the pass usable when weights or calibration data are missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.quantization.pruning import (
    PruningMask,
    apply_mask,
    build_mask,
    build_nm_mask,
    build_unstructured_mask,
    compute_importance,
    verify_nm_pattern,
)

METRICS = ["magnitude", "wanda", "sparsegpt"]


@pytest.fixture
def weights() -> np.ndarray:
    """A (64, 128) weight matrix; 128 is divisible by 4 and 8 for N:M patterns."""
    return np.random.RandomState(0).randn(64, 128).astype(np.float32)


@pytest.fixture
def activation_norms() -> np.ndarray:
    """Strictly positive per-input-feature activation norms."""
    return np.abs(np.random.RandomState(1).randn(128)).astype(np.float32) + 0.1


class TestComputeImportance:
    def test_magnitude_is_absolute_value(self, weights: np.ndarray) -> None:
        np.testing.assert_allclose(compute_importance(weights, "magnitude"), np.abs(weights))

    def test_wanda_scales_columns_by_activation_norm(
        self, weights: np.ndarray, activation_norms: np.ndarray
    ) -> None:
        expected = np.abs(weights) * activation_norms[None, :]
        actual = compute_importance(weights, "wanda", activation_norms)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_sparsegpt_uses_squared_terms(
        self, weights: np.ndarray, activation_norms: np.ndarray
    ) -> None:
        expected = np.abs(weights) ** 2 * activation_norms[None, :] ** 2
        actual = compute_importance(weights, "sparsegpt", activation_norms)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    @pytest.mark.parametrize("metric", METRICS)
    def test_importance_is_non_negative(
        self, weights: np.ndarray, activation_norms: np.ndarray, metric: str
    ) -> None:
        assert np.all(compute_importance(weights, metric, activation_norms) >= 0)

    def test_rejects_non_2d_weights(self) -> None:
        with pytest.raises(ValueError, match="2-D weight matrix"):
            compute_importance(np.zeros((2, 2, 2), dtype=np.float32), "magnitude")

    def test_rejects_unknown_metric(self, weights: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown importance metric"):
            compute_importance(weights, "voodoo")  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric", ["wanda", "sparsegpt"])
    def test_activation_metrics_require_norms(self, weights: np.ndarray, metric: str) -> None:
        with pytest.raises(ValueError, match="requires activation_norms"):
            compute_importance(weights, metric)  # type: ignore[arg-type]

    def test_rejects_mismatched_activation_norms(self, weights: np.ndarray) -> None:
        with pytest.raises(ValueError, match="input features"):
            compute_importance(weights, "wanda", np.ones(7, dtype=np.float32))


class TestUnstructuredMask:
    @pytest.mark.parametrize("target", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    def test_achieved_sparsity_tracks_target(self, weights: np.ndarray, target: float) -> None:
        mask = build_mask(weights, target_sparsity=target)
        # Per-row pruning quantises to whole columns, so allow one column of slack.
        assert abs(mask.achieved_sparsity - target) <= 1.0 / weights.shape[1]

    def test_zero_sparsity_keeps_everything(self, weights: np.ndarray) -> None:
        assert build_mask(weights, target_sparsity=0.0).achieved_sparsity == 0.0

    def test_prunes_the_smallest_weights(self) -> None:
        w = np.array([[1.0, 2.0, 3.0, 100.0]], dtype=np.float32)
        mask = build_unstructured_mask(compute_importance(w, "magnitude"), 0.5)
        # The two largest (3.0, 100.0) must survive.
        assert mask.tolist() == [[False, False, True, True]]

    def test_per_row_prunes_each_row_equally(self) -> None:
        w = np.array([[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
        mask = build_unstructured_mask(compute_importance(w, "magnitude"), 0.5, per_row=True)
        assert mask.sum(axis=1).tolist() == [2, 2]

    def test_global_mode_can_prune_a_whole_row(self) -> None:
        """Global comparison lets a uniformly small row be pruned entirely."""
        w = np.array([[0.001, 0.002], [100.0, 200.0]], dtype=np.float32)
        mask = build_unstructured_mask(compute_importance(w, "magnitude"), 0.5, per_row=False)
        assert mask[0].sum() == 0
        assert mask[1].sum() == 2

    def test_rejects_out_of_range_sparsity(self, weights: np.ndarray) -> None:
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError, match=r"target_sparsity must be in \[0, 1\)"):
                build_unstructured_mask(np.abs(weights), bad)


class TestNMMask:
    @pytest.mark.parametrize(("pattern", "n", "m"), [("2:4", 2, 4), ("4:8", 4, 8), ("1:2", 1, 2)])
    def test_structure_holds_exactly(
        self, weights: np.ndarray, activation_norms: np.ndarray, pattern: str, n: int, m: int
    ) -> None:
        mask = build_mask(weights, pattern=pattern, metric="wanda", activation_norms=activation_norms)
        assert verify_nm_pattern(mask.mask, n, m)
        assert mask.achieved_sparsity == pytest.approx(1.0 - n / m)

    @pytest.mark.parametrize("metric", METRICS)
    def test_structure_holds_for_every_metric(
        self, weights: np.ndarray, activation_norms: np.ndarray, metric: str
    ) -> None:
        mask = build_mask(weights, pattern="2:4", metric=metric, activation_norms=activation_norms)
        assert verify_nm_pattern(mask.mask, 2, 4)

    def test_keeps_the_two_largest_per_group(self) -> None:
        w = np.array([[1.0, 2.0, 3.0, 4.0, 40.0, 30.0, 20.0, 10.0]], dtype=np.float32)
        mask = build_nm_mask(compute_importance(w, "magnitude"), 2, 4)
        assert mask.tolist() == [[False, False, True, True, True, True, False, False]]

    def test_rejects_misaligned_input_dimension(self) -> None:
        w = np.random.RandomState(2).randn(4, 30).astype(np.float32)
        with pytest.raises(ValueError, match="not divisible by group size"):
            build_nm_mask(np.abs(w), 2, 4)

    @pytest.mark.parametrize(("n", "m"), [(4, 4), (5, 4), (0, 4), (-1, 4), (2, 0)])
    def test_rejects_invalid_patterns(self, weights: np.ndarray, n: int, m: int) -> None:
        with pytest.raises(ValueError, match="invalid N:M pattern"):
            build_nm_mask(np.abs(weights), n, m)

    def test_ties_still_keep_exactly_n(self) -> None:
        """All-equal groups must not keep more or fewer than n."""
        w = np.ones((4, 16), dtype=np.float32)
        mask = build_nm_mask(compute_importance(w, "magnitude"), 2, 4)
        assert verify_nm_pattern(mask, 2, 4)

    def test_zeros_still_keep_exactly_n(self) -> None:
        w = np.zeros((4, 16), dtype=np.float32)
        assert verify_nm_pattern(build_nm_mask(compute_importance(w, "magnitude"), 2, 4), 2, 4)

    def test_verify_rejects_wrong_count(self) -> None:
        bad = np.ones((2, 8), dtype=bool)  # keeps 4 of every 4
        assert not verify_nm_pattern(bad, 2, 4)

    def test_verify_rejects_misaligned_shape(self) -> None:
        assert not verify_nm_pattern(np.ones((2, 6), dtype=bool), 2, 4)


class TestWandaQuality:
    def test_wanda_differs_from_magnitude(
        self, weights: np.ndarray, activation_norms: np.ndarray
    ) -> None:
        """If activation weighting changed nothing, the metric would be pointless."""
        mag = build_mask(weights, pattern="2:4", metric="magnitude")
        wanda = build_mask(weights, pattern="2:4", metric="wanda", activation_norms=activation_norms)
        assert not np.array_equal(mag.mask, wanda.mask)

    def test_wanda_lowers_activation_weighted_error(
        self, weights: np.ndarray, activation_norms: np.ndarray
    ) -> None:
        """Wanda's whole claim: better preservation of what reaches the output."""
        mag = build_mask(weights, pattern="2:4", metric="magnitude")
        wanda = build_mask(weights, pattern="2:4", metric="wanda", activation_norms=activation_norms)
        scale = activation_norms[None, :]
        mag_err = np.sqrt(np.mean(((weights - apply_mask(weights, mag)) * scale) ** 2))
        wanda_err = np.sqrt(np.mean(((weights - apply_mask(weights, wanda)) * scale) ** 2))
        assert wanda_err < mag_err

    def test_uniform_activations_reduce_wanda_to_magnitude(self, weights: np.ndarray) -> None:
        uniform = np.ones(weights.shape[1], dtype=np.float32)
        mag = build_mask(weights, pattern="2:4", metric="magnitude")
        wanda = build_mask(weights, pattern="2:4", metric="wanda", activation_norms=uniform)
        np.testing.assert_array_equal(mag.mask, wanda.mask)


class TestApplyMask:
    def test_pruned_entries_are_exactly_zero(
        self, weights: np.ndarray, activation_norms: np.ndarray
    ) -> None:
        mask = build_mask(weights, pattern="2:4", metric="wanda", activation_norms=activation_norms)
        pruned = apply_mask(weights, mask)
        assert np.all(pruned[~mask.mask] == 0.0)

    def test_kept_entries_are_untouched(self, weights: np.ndarray) -> None:
        mask = build_mask(weights, target_sparsity=0.5)
        pruned = apply_mask(weights, mask)
        np.testing.assert_array_equal(pruned[mask.mask], weights[mask.mask])

    def test_does_not_mutate_input(self, weights: np.ndarray) -> None:
        original = weights.copy()
        apply_mask(weights, build_mask(weights, target_sparsity=0.5))
        np.testing.assert_array_equal(weights, original)

    def test_accepts_raw_boolean_array(self, weights: np.ndarray) -> None:
        raw = np.ones_like(weights, dtype=bool)
        raw[:, 0] = False
        assert np.all(apply_mask(weights, raw)[:, 0] == 0.0)

    def test_rejects_shape_mismatch(self, weights: np.ndarray) -> None:
        with pytest.raises(ValueError, match="does not match weight shape"):
            apply_mask(weights, np.ones((2, 2), dtype=bool))


class TestPruningMaskMetadata:
    def test_counts_are_consistent(self, weights: np.ndarray) -> None:
        mask = build_mask(weights, target_sparsity=0.5)
        assert mask.kept_count + mask.pruned_count == weights.size
        assert mask.kept_count == int(mask.mask.sum())

    def test_to_dict_reports_achieved_sparsity(self, weights: np.ndarray) -> None:
        payload = build_mask(weights, pattern="2:4", metric="magnitude").to_dict()
        assert payload["pattern"] == "2:4"
        assert payload["achieved_sparsity"] == pytest.approx(0.5)
        assert payload["n"] == 2 and payload["m"] == 4
        assert payload["structured"] is True

    def test_repr_is_informative(self, weights: np.ndarray) -> None:
        assert "2:4" in repr(build_mask(weights, pattern="2:4", metric="magnitude"))

    def test_empty_mask_reports_zero_sparsity(self) -> None:
        empty = PruningMask(
            mask=np.zeros((0, 0), dtype=bool),
            pattern="unstructured",
            metric="magnitude",
            target_sparsity=0.5,
            shape=(0, 0),
        )
        assert empty.achieved_sparsity == 0.0

    def test_rejects_unknown_pattern(self, weights: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown sparsity pattern"):
            build_mask(weights, pattern="3:7")  # type: ignore[arg-type]


class TestSparsityInteractsWithQuantization:
    @pytest.mark.parametrize("precision", ["Q4_K_M", "NF4", "FP8", "Q8_0"])
    def test_pruned_zeros_survive_quantization(
        self, weights: np.ndarray, activation_norms: np.ndarray, precision: str
    ) -> None:
        """A pruned-then-quantized tensor must keep its structural zeros."""
        from aether.quantization.formats import dequantize_tensor, quantize_tensor

        mask = build_mask(weights, pattern="2:4", metric="wanda", activation_norms=activation_norms)
        pruned = apply_mask(weights, mask)
        restored = dequantize_tensor(quantize_tensor(pruned, precision, 32))
        assert np.all(restored[~mask.mask] == 0.0)

    def test_quantization_never_exceeds_the_nm_budget(self, weights: np.ndarray) -> None:
        """Sparse tensor cores need *at most* n non-zeros per group of m.

        Quantization can round a small kept weight down to zero, which yields a
        group with fewer than n survivors. That is extra sparsity and is
        acceptable; a group with *more* than n would break the kernel contract.
        """
        from aether.quantization.formats import dequantize_tensor, quantize_tensor

        mask = build_mask(weights, pattern="2:4", metric="magnitude")
        restored = dequantize_tensor(quantize_tensor(apply_mask(weights, mask), "Q4_K_M", 32))
        rows, cols = restored.shape
        per_group = (restored != 0.0).reshape(rows, cols // 4, 4).sum(axis=2)
        assert per_group.max() <= 2

    def test_mask_zeros_survive_quantization_exactly(self, weights: np.ndarray) -> None:
        """Regression: fp16 scale rounding used to reintroduce ~5e-4 at pruned positions."""
        from aether.quantization.formats import dequantize_tensor, quantize_tensor

        mask = build_mask(weights, pattern="2:4", metric="magnitude")
        restored = dequantize_tensor(quantize_tensor(apply_mask(weights, mask), "Q4_K_M", 32))
        assert np.all(restored[~mask.mask] == 0.0)
