"""
Tests for FAIRLens CDS metrics (Phase 3).

Mathematical correctness tests using synthetic embeddings.
No GPU or LLM required.

Covers:
- CDS sign (A more dispersed → positive CDS)
- Identical groups → CDS = 0, centroid_separation = 0
- Absolute CDS (ADM) always ≥ 0
- Centroid separation bounds (cosine distance ∈ [0, 2])
- PairMetrics field population
- aggregate_metrics computation
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairlens.cds import (
    FAIRLensCDS,
    PairMetrics,
    centroid_separation,
    aggregate_metrics,
)
from fairlens.bootstrap import _mean_dispersion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalized(K, D, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(K, D)
    return torch.nn.functional.normalize(x, dim=1)


def tight_cluster(K, D, center_seed=0):
    torch.manual_seed(center_seed)
    center = torch.nn.functional.normalize(torch.randn(1, D), dim=1)
    noise = torch.randn(K, D) * 0.005
    x = center.expand(K, D) + noise
    return torch.nn.functional.normalize(x, dim=1)


# ---------------------------------------------------------------------------
# Centroid separation tests
# ---------------------------------------------------------------------------

class TestCentroidSeparation:

    def test_identical_embeddings_zero_separation(self):
        """Same embeddings for A and B → cosine separation = 0."""
        emb = normalized(10, 8, seed=1)
        cs = centroid_separation(emb, emb, metric="cosine")
        assert abs(cs) < 1e-5, f"Expected CS ≈ 0, got {cs}"

    def test_orthogonal_centroids(self):
        """Orthogonal centroids → cosine similarity = 0 → separation = 1."""
        D = 4
        # Construct two orthogonal unit vectors as centroids
        c_a = torch.zeros(10, D)
        c_a[:, 0] = 1.0  # all point in direction 0
        c_b = torch.zeros(10, D)
        c_b[:, 1] = 1.0  # all point in direction 1
        c_a = torch.nn.functional.normalize(c_a, dim=1)
        c_b = torch.nn.functional.normalize(c_b, dim=1)
        cs = centroid_separation(c_a, c_b, metric="cosine")
        assert abs(cs - 1.0) < 1e-4, f"Expected CS ≈ 1 for orthogonal, got {cs}"

    def test_cs_nonnegative(self):
        for seed in range(20):
            a = normalized(10, 8, seed=seed)
            b = normalized(10, 8, seed=seed + 100)
            cs = centroid_separation(a, b, metric="cosine")
            assert cs >= -1e-6, f"CS must be ≥ 0, got {cs} at seed {seed}"

    def test_cs_at_most_two(self):
        """Cosine distance ∈ [0, 2] for unit vectors."""
        for seed in range(20):
            a = normalized(10, 8, seed=seed)
            b = normalized(10, 8, seed=seed + 200)
            cs = centroid_separation(a, b, metric="cosine")
            assert cs <= 2.0 + 1e-6

    def test_unknown_metric_raises(self):
        emb = normalized(5, 4, seed=0)
        with pytest.raises(ValueError, match="Unknown metric"):
            centroid_separation(emb, emb, metric="manhattan")


# ---------------------------------------------------------------------------
# CDS sign and absolute value tests
# ---------------------------------------------------------------------------

class TestCDSValues:

    def setup_method(self):
        self.K, self.D = 20, 16
        self.tight = tight_cluster(self.K, self.D, center_seed=1)
        self.spread = normalized(self.K, self.D, seed=99)
        self.calc = FAIRLensCDS(
            bootstrap_method="independent",
            n_bootstrap=200,
            bootstrap_seed=42,
        )

    def test_identical_groups_cds_zero(self):
        """A and B identical → CDS = 0."""
        emb = normalized(self.K, self.D, seed=5)
        pm = self.calc.compute(emb, emb)
        assert abs(pm.cds) < 1e-6, f"Expected CDS = 0 for identical groups, got {pm.cds}"

    def test_identical_groups_centroid_separation_zero(self):
        """A and B identical → centroid_separation = 0."""
        emb = normalized(self.K, self.D, seed=5)
        pm = self.calc.compute(emb, emb)
        assert abs(pm.centroid_separation) < 1e-5, (
            f"Expected CS = 0 for identical groups, got {pm.centroid_separation}"
        )

    def test_cds_sign_tight_a_spread_b(self):
        """
        A = tight cluster (low dispersion), B = spread.
        CDS = D_A - D_B should be negative (B more dispersed).
        """
        pm = self.calc.compute(self.tight, self.spread)
        assert pm.cds < 0, (
            f"Expected negative CDS (tight A, spread B), got {pm.cds:.4f}"
        )

    def test_cds_sign_spread_a_tight_b(self):
        """
        A = spread, B = tight cluster.
        CDS = D_A - D_B should be positive (A more dispersed).
        """
        pm = self.calc.compute(self.spread, self.tight)
        assert pm.cds > 0, (
            f"Expected positive CDS (spread A, tight B), got {pm.cds:.4f}"
        )

    def test_absolute_cds_equals_abs_cds(self):
        for seed in range(10):
            a = normalized(self.K, self.D, seed=seed)
            b = normalized(self.K, self.D, seed=seed + 50)
            pm = self.calc.compute(a, b)
            assert abs(pm.adm - abs(pm.cds)) < 1e-9, (
                f"ADM ({pm.adm}) != |CDS| ({abs(pm.cds)})"
            )

    def test_adm_nonnegative(self):
        for seed in range(10):
            a = normalized(self.K, self.D, seed=seed)
            b = normalized(self.K, self.D, seed=seed + 50)
            pm = self.calc.compute(a, b)
            assert pm.adm >= 0.0

    def test_dispersion_a_equals_mean_dispersion(self):
        """dispersion_a must match direct calculation."""
        emb_a = normalized(self.K, self.D, seed=7)
        emb_b = normalized(self.K, self.D, seed=8)
        pm = self.calc.compute(emb_a, emb_b)
        direct = _mean_dispersion(emb_a)
        assert abs(pm.dispersion_a - direct) < 1e-7


# ---------------------------------------------------------------------------
# PairMetrics field population
# ---------------------------------------------------------------------------

class TestPairMetricsFields:

    def test_identity_fields_set_via_dict(self):
        emb = normalized(10, 8, seed=0)
        calc = FAIRLensCDS(n_bootstrap=100, bootstrap_seed=0)
        identity = {
            "pair_id": "P12345ABCDE",
            "model_id": "phi3_mini",
            "domain": "hiring",
            "group_a": "Black_Female",
            "group_b": "White_Male",
        }
        pm = calc.compute(emb, emb, identity=identity)
        assert pm.pair_id == "P12345ABCDE"
        assert pm.model_id == "phi3_mini"
        assert pm.domain == "hiring"

    def test_to_dict_serialisable(self):
        import json
        emb = normalized(10, 8, seed=0)
        calc = FAIRLensCDS(n_bootstrap=50, bootstrap_seed=0)
        pm = calc.compute(emb, emb)
        # Should not raise
        d = pm.to_dict()
        json.dumps(d)  # must be JSON-serializable


# ---------------------------------------------------------------------------
# aggregate_metrics tests
# ---------------------------------------------------------------------------

class TestAggregateMetrics:

    def _make_pair_metrics(self, cds: float) -> PairMetrics:
        """Create a minimal PairMetrics with given CDS."""
        pm = PairMetrics(cds=cds, adm=abs(cds))
        return pm

    def test_empty_list_returns_empty(self):
        result = aggregate_metrics([])
        assert result == {}

    def test_signed_mean_correct(self):
        pairs = [self._make_pair_metrics(c) for c in [0.1, -0.1, 0.2, -0.2]]
        agg = aggregate_metrics(pairs)
        assert abs(agg["signed_mean_cds"]) < 1e-9

    def test_mean_absolute_correct(self):
        pairs = [self._make_pair_metrics(c) for c in [0.1, -0.1, 0.2, -0.2]]
        agg = aggregate_metrics(pairs)
        assert abs(agg["mean_absolute_cds"] - 0.15) < 1e-9

    def test_masking_gap_correct(self):
        """For [+0.1, -0.1, +0.2, -0.2], MG = mean(|CDS|) - |mean(CDS)| = 0.15."""
        pairs = [self._make_pair_metrics(c) for c in [0.1, -0.1, 0.2, -0.2]]
        agg = aggregate_metrics(pairs)
        assert abs(agg["masking_gap"] - 0.15) < 1e-9

    def test_masking_gap_nonnegative(self):
        """MG must be ≥ 0 for any input (Jensen's inequality)."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            cds_vals = rng.normal(0, 0.05, 20).tolist()
            pairs = [self._make_pair_metrics(c) for c in cds_vals]
            agg = aggregate_metrics(pairs)
            assert agg["masking_gap"] >= -1e-10, (
                f"MG must be ≥ 0, got {agg['masking_gap']}"
            )

    def test_fractions_sum_to_one(self):
        """positive + negative + indeterminate fractions must sum to 1."""
        pairs = [PairMetrics(cds=c, adm=abs(c), direction=d)
                 for c, d in [(0.1, 1), (-0.1, -1), (0.05, 0)]]
        agg = aggregate_metrics(pairs)
        total = agg["positive_fraction"] + agg["negative_fraction"] + agg["indeterminate_fraction"]
        assert abs(total - 1.0) < 1e-9

    def test_n_count(self):
        pairs = [self._make_pair_metrics(0.0) for _ in range(7)]
        agg = aggregate_metrics(pairs)
        assert agg["n"] == 7
