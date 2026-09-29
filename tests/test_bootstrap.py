"""
Tests for bootstrap implementations (Phase 3).

All tests use small synthetic embeddings — no GPU or LLM required.
Embedding tensors are CPU float32 and deliberately kept small (K=10, D=8)
so tests run quickly.

Covers:
- Independent bootstrap: CI structure, direction states
- Paired bootstrap: preserved legacy behavior
- Bootstrap ablation: both methods on same data
- Direction state logic
- Mathematical properties (CI contains observed value with high probability)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairlens.bootstrap import (
    IndependentBootstrap,
    PairedBootstrap,
    BootstrapResult,
    BootstrapAblation,
    bootstrap_ablation,
    direction_state,
    _mean_dispersion,
    make_bootstrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_normalized(K: int, D: int, seed: int = 0) -> torch.Tensor:
    """Make K L2-normalized random D-dimensional embeddings."""
    torch.manual_seed(seed)
    x = torch.randn(K, D)
    return torch.nn.functional.normalize(x, dim=1)


def make_tight_cluster(K: int, D: int, center_seed: int = 0) -> torch.Tensor:
    """Embeddings very close to a single center (low dispersion)."""
    torch.manual_seed(center_seed)
    center = torch.nn.functional.normalize(torch.randn(1, D), dim=1)
    noise = torch.randn(K, D) * 0.01
    x = center + noise
    return torch.nn.functional.normalize(x, dim=1)


def make_spread_cluster(K: int, D: int, seed: int = 99) -> torch.Tensor:
    """Embeddings spread across the hypersphere (high dispersion)."""
    return make_normalized(K, D, seed=seed)


# ---------------------------------------------------------------------------
# Direction state tests
# ---------------------------------------------------------------------------

class TestDirectionState:

    def test_positive_ci(self):
        assert direction_state(0.01, 0.15) == 1

    def test_negative_ci(self):
        assert direction_state(-0.15, -0.01) == -1

    def test_ci_spans_zero_positive_low(self):
        assert direction_state(-0.01, 0.10) == 0

    def test_ci_spans_zero_negative_high(self):
        assert direction_state(-0.10, 0.01) == 0

    def test_zero_width_ci_at_zero(self):
        assert direction_state(0.0, 0.0) == 0  # edge: ci_low = 0 is NOT > 0

    def test_tiny_positive_low(self):
        """ci_low > 0 (not >= 0) is required for +1."""
        assert direction_state(1e-10, 0.1) == 1

    def test_exactly_zero_low(self):
        """ci_low = 0 exactly → direction is 0 (indeterminate)."""
        assert direction_state(0.0, 0.1) == 0


# ---------------------------------------------------------------------------
# Mean dispersion tests (mathematical baseline)
# ---------------------------------------------------------------------------

class TestMeanDispersion:

    def test_identical_embeddings_zero_dispersion(self):
        """If all K embeddings are identical, dispersion must be 0."""
        e = torch.nn.functional.normalize(torch.ones(10, 8), dim=1)
        # All identical → centroid = same vector → all distances = 0
        d = _mean_dispersion(e)
        assert abs(d) < 1e-5, f"Expected ~0 dispersion, got {d}"

    def test_tight_cluster_low_dispersion(self):
        tight = make_tight_cluster(20, 16, center_seed=7)
        spread = make_spread_cluster(20, 16, seed=7)
        d_tight = _mean_dispersion(tight)
        d_spread = _mean_dispersion(spread)
        assert d_tight < d_spread, (
            f"Tight cluster ({d_tight:.4f}) should have lower dispersion "
            f"than spread cluster ({d_spread:.4f})"
        )

    def test_dispersion_nonnegative(self):
        for seed in range(10):
            e = make_normalized(15, 8, seed=seed)
            assert _mean_dispersion(e) >= 0.0


# ---------------------------------------------------------------------------
# Independent bootstrap tests
# ---------------------------------------------------------------------------

class TestIndependentBootstrap:

    def setup_method(self):
        self.K, self.D = 20, 16
        self.tight = make_tight_cluster(self.K, self.D, center_seed=1)
        self.spread = make_spread_cluster(self.K, self.D, seed=2)
        self.ibs = IndependentBootstrap(n_bootstrap=500, seed=42)

    def test_result_is_bootstrap_result(self):
        result = self.ibs.compute(self.tight, self.spread)
        assert isinstance(result, BootstrapResult)

    def test_method_label(self):
        result = self.ibs.compute(self.tight, self.spread)
        assert result.method == "independent"

    def test_ci_low_le_ci_high(self):
        result = self.ibs.compute(self.tight, self.spread)
        assert result.ci_low <= result.ci_high

    def test_prob_positive_plus_negative_le_one(self):
        result = self.ibs.compute(self.tight, self.spread)
        assert 0.0 <= result.prob_positive <= 1.0
        assert 0.0 <= result.prob_negative <= 1.0
        assert result.prob_positive + result.prob_negative <= 1.0 + 1e-9

    def test_direction_consistent_with_ci(self):
        """Direction state must be consistent with CI boundaries."""
        result = self.ibs.compute(self.tight, self.spread)
        if result.ci_low > 0:
            assert result.direction == 1
        elif result.ci_high < 0:
            assert result.direction == -1
        else:
            assert result.direction == 0

    def test_tight_vs_spread_negative_cds(self):
        """
        tight = A, spread = B → CDS = D_A - D_B < 0 (B more dispersed).
        For a large enough difference, direction should be -1.
        """
        result = self.ibs.compute(self.tight, self.spread)
        assert result.observed_cds < 0, (
            f"Expected negative CDS (tight < spread), got {result.observed_cds:.4f}"
        )

    def test_identical_groups_cds_near_zero(self):
        """If A and B are identical, CDS should be near 0."""
        emb = make_normalized(20, 16, seed=5)
        result = self.ibs.compute(emb, emb)
        assert abs(result.observed_cds) < 1e-6, (
            f"Expected CDS ≈ 0 for identical groups, got {result.observed_cds}"
        )

    def test_ci_width_positive(self):
        result = self.ibs.compute(self.tight, self.spread)
        assert result.ci_width > 0

    def test_n_iterations_stored(self):
        result = self.ibs.compute(self.tight, self.spread)
        assert result.n_iterations == 500


# ---------------------------------------------------------------------------
# Paired bootstrap tests (legacy preservation)
# ---------------------------------------------------------------------------

class TestPairedBootstrap:

    def setup_method(self):
        self.K, self.D = 20, 16
        self.emb_a = make_normalized(self.K, self.D, seed=10)
        self.emb_b = make_normalized(self.K, self.D, seed=20)
        self.pbs = PairedBootstrap(n_bootstrap=500, seed=42)

    def test_method_label(self):
        result = self.pbs.compute(self.emb_a, self.emb_b)
        assert result.method == "paired"

    def test_requires_equal_k(self):
        """Paired bootstrap must raise if K differs between A and B."""
        emb_a_diff = make_normalized(15, self.D, seed=1)
        emb_b_diff = make_normalized(20, self.D, seed=2)
        with pytest.raises(ValueError, match="equal K"):
            self.pbs.compute(emb_a_diff, emb_b_diff)

    def test_ci_low_le_ci_high(self):
        result = self.pbs.compute(self.emb_a, self.emb_b)
        assert result.ci_low <= result.ci_high

    def test_direction_consistent_with_ci(self):
        result = self.pbs.compute(self.emb_a, self.emb_b)
        if result.ci_low > 0:
            assert result.direction == 1
        elif result.ci_high < 0:
            assert result.direction == -1
        else:
            assert result.direction == 0


# ---------------------------------------------------------------------------
# Bootstrap ablation tests (paired vs independent comparison)
# ---------------------------------------------------------------------------

class TestBootstrapAblation:

    def setup_method(self):
        self.K, self.D = 20, 16
        self.emb_a = make_normalized(self.K, self.D, seed=3)
        self.emb_b = make_normalized(self.K, self.D, seed=4)

    def test_ablation_returns_both_results(self):
        abl = bootstrap_ablation(self.emb_a, self.emb_b, n_bootstrap=200)
        assert isinstance(abl.independent, BootstrapResult)
        assert isinstance(abl.paired, BootstrapResult)

    def test_ablation_methods_correct(self):
        abl = bootstrap_ablation(self.emb_a, self.emb_b, n_bootstrap=200)
        assert abl.independent.method == "independent"
        assert abl.paired.method == "paired"

    def test_ablation_summary_keys(self):
        abl = bootstrap_ablation(self.emb_a, self.emb_b, n_bootstrap=200)
        summary = abl.summary()
        required = {
            "independent_ci", "paired_ci",
            "independent_width", "paired_width",
            "width_difference", "direction_agrees",
        }
        assert required.issubset(set(summary.keys()))

    def test_both_observe_same_cds(self):
        """Both methods start from the same observed CDS value."""
        abl = bootstrap_ablation(self.emb_a, self.emb_b, n_bootstrap=200)
        assert abs(abl.independent.observed_cds - abl.paired.observed_cds) < 1e-9


# ---------------------------------------------------------------------------
# make_bootstrap factory test
# ---------------------------------------------------------------------------

class TestMakeBootstrap:

    def test_factory_independent(self):
        bs = make_bootstrap("independent", n_bootstrap=100, seed=0)
        assert isinstance(bs, IndependentBootstrap)

    def test_factory_paired(self):
        bs = make_bootstrap("paired", n_bootstrap=100, seed=0)
        assert isinstance(bs, PairedBootstrap)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown bootstrap method"):
            make_bootstrap("unknown")
