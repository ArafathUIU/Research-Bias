"""
Tests for masking gap and masking ratio (Phase 3).

Mathematical correctness tests for the aggregation masking metrics.
No GPU required.

Covers:
- Masking Gap ≥ 0 (Jensen's inequality)
- Masking Ratio edge cases (zero denominator → NaN)
- Known exact values
- Positive fraction / negative fraction
- Sign balance examples from the paper
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairlens.masking import compute_masking, MaskingResult, MASKING_EPSILON


# ---------------------------------------------------------------------------
# Mathematical guarantee: Masking Gap ≥ 0
# ---------------------------------------------------------------------------

class TestMaskingGapNonNegative:
    """
    MG = mean(|CDS|) - |mean(CDS)| ≥ 0 always.
    Proof: by Jensen's inequality, for f(x) = |x| convex:
    E[f(X)] ≥ f(E[X]), so mean(|CDS|) ≥ |mean(CDS)|.
    """

    def test_random_inputs_100_trials(self):
        rng = np.random.default_rng(seed=0)
        for trial in range(100):
            n = rng.integers(2, 50)
            cds_vals = rng.normal(0, 0.05, n).tolist()
            result = compute_masking(cds_vals)
            assert result.masking_gap >= -1e-10, (
                f"Trial {trial}: MG = {result.masking_gap} < 0 for {cds_vals[:5]}"
            )

    def test_all_positive_zero_gap(self):
        """If all CDS > 0, there is no cancellation → MG = 0."""
        result = compute_masking([0.1, 0.2, 0.3])
        assert abs(result.masking_gap) < 1e-9, (
            f"All positive CDS → MG should be 0, got {result.masking_gap}"
        )

    def test_all_negative_zero_gap(self):
        """If all CDS < 0, no cancellation → MG = 0."""
        result = compute_masking([-0.1, -0.2, -0.3])
        assert abs(result.masking_gap) < 1e-9, (
            f"All negative CDS → MG should be 0, got {result.masking_gap}"
        )

    def test_single_element(self):
        result = compute_masking([0.05])
        assert result.masking_gap >= 0.0

    def test_two_equal_opposite(self):
        """[+x, -x] → signed_mean = 0, mean_abs = x → MG = x."""
        x = 0.10
        result = compute_masking([x, -x])
        assert abs(result.masking_gap - x) < 1e-9


# ---------------------------------------------------------------------------
# Known exact values
# ---------------------------------------------------------------------------

class TestKnownValues:
    """Verify exact values from the Part 13 masking example."""

    def test_paper_example(self):
        """
        Example from Part 13: [+0.10, -0.10, +0.12, -0.12]
        signed_mean = 0
        mean_abs = 0.11
        MG = 0.11
        MR = 1.0
        """
        result = compute_masking([0.10, -0.10, 0.12, -0.12])
        assert abs(result.signed_mean) < 1e-9, f"signed_mean should be 0, got {result.signed_mean}"
        assert abs(result.absolute_mean - 0.11) < 1e-9, (
            f"mean_abs should be 0.11, got {result.absolute_mean}"
        )
        assert abs(result.masking_gap - 0.11) < 1e-9, (
            f"MG should be 0.11, got {result.masking_gap}"
        )
        assert abs(result.masking_ratio - 1.0) < 1e-9, (
            f"MR should be 1.0, got {result.masking_ratio}"
        )

    def test_all_same_value(self):
        """[+v, +v, +v] → signed_mean = v, mean_abs = v → MG = 0, MR = 0."""
        v = 0.05
        result = compute_masking([v, v, v])
        assert abs(result.masking_gap) < 1e-9
        assert abs(result.masking_ratio) < 1e-9

    def test_fractions_correct(self):
        result = compute_masking([0.1, -0.2, 0.3, 0.0])
        n = 4
        # pos: 0.1, 0.3 → 2; neg: -0.2 → 1; zero: 0.0 → 1
        assert abs(result.positive_fraction - 2/4) < 1e-9
        assert abs(result.negative_fraction - 1/4) < 1e-9
        assert abs(result.zero_fraction - 1/4) < 1e-9

    def test_n_correct(self):
        result = compute_masking([0.1, 0.2, 0.3, 0.4, 0.5])
        assert result.n == 5

    def test_median_correct(self):
        result = compute_masking([0.1, 0.3, 0.5])
        assert abs(result.median_cds - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_masking([])

    def test_zero_mean_abs_masking_ratio_nan(self):
        """
        If all CDS = 0, mean(|CDS|) = 0 → MR is undefined (NaN).
        masking_ratio_nan must be True.
        """
        result = compute_masking([0.0, 0.0, 0.0])
        assert result.masking_ratio_nan, "Expected masking_ratio_nan=True when all CDS=0"
        assert math.isnan(result.masking_ratio), (
            f"Expected MR=NaN, got {result.masking_ratio}"
        )

    def test_masking_ratio_in_unit_interval(self):
        """MR must be in [0, 1] when defined."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            cds_vals = rng.normal(0, 0.05, 20).tolist()
            result = compute_masking(cds_vals)
            if not result.masking_ratio_nan:
                assert -1e-9 <= result.masking_ratio <= 1.0 + 1e-9, (
                    f"MR out of [0,1]: {result.masking_ratio}"
                )

    def test_single_zero_element(self):
        result = compute_masking([0.0])
        assert result.masking_ratio_nan

    def test_to_dict_returns_none_for_nan_ratio(self):
        result = compute_masking([0.0, 0.0])
        d = result.to_dict()
        assert d["masking_ratio"] is None
        assert d["masking_ratio_undefined"] is True

    def test_to_dict_returns_float_for_valid_ratio(self):
        result = compute_masking([0.1, -0.1, 0.2])
        d = result.to_dict()
        assert isinstance(d["masking_ratio"], float)
        assert not d["masking_ratio_undefined"]

    def test_large_values(self):
        """MG ≥ 0 must hold even for large CDS values."""
        result = compute_masking([10.0, -10.0, 5.0, -5.0])
        assert result.masking_gap >= -1e-9

    def test_single_element_nonzero(self):
        result = compute_masking([0.05])
        assert not result.masking_ratio_nan
        assert abs(result.masking_ratio) < 1e-9  # |mean| = mean_abs → MR = 0
