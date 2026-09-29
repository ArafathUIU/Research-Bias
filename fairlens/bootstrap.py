"""
FAIRLens Bootstrap Module
=========================

Implements two bootstrap resampling methods for CDS confidence intervals.

Methods
-------
IndependentBootstrap (FAIRLens v1 default)
    A and B conditions are resampled using SEPARATE random index vectors.
    This is scientifically correct when condition A and condition B responses
    are generated independently by the LLM — response index k in group A has
    no natural correspondence with response index k in group B.

PairedBootstrap (legacy_reproduction only)
    A and B conditions are resampled using the SAME random index vector.
    This was the original CONSIST implementation. It creates artificial
    pairing between responses that were generated independently. It is
    preserved here ONLY for exact reproduction of the original pilot results.
    Do not use for new FAIRLens v1 analyses.

Ablation
--------
bootstrap_ablation() runs both methods on the same embeddings and returns
a comparison of CI widths and direction classifications. This is used to
quantify the methodological correction in the Phase 1-4 Implementation Report.

Direction States
----------------
Given a bootstrap CI [ci_low, ci_high] for CDS:
  +1 : ci_low > 0   (A-dispersion statistically larger, CI entirely positive)
   0 : ci_low <= 0 <= ci_high   (direction indeterminate)
  -1 : ci_high < 0  (B-dispersion statistically larger, CI entirely negative)

This is conservative: a CDS of any sign is labelled 0 unless the full CI
excludes zero. Small effects and wide CIs will correctly be labelled 0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Direction state
# ---------------------------------------------------------------------------

def direction_state(ci_low: float, ci_high: float) -> int:
    """
    Return the uncertainty-aware direction of CDS.

        +1  if ci_low > 0    (A consistently more dispersed than B)
         0  if ci spans zero (direction indeterminate)
        -1  if ci_high < 0   (B consistently more dispersed than A)
    """
    if ci_low > 0.0:
        return 1
    elif ci_high < 0.0:
        return -1
    else:
        return 0


# ---------------------------------------------------------------------------
# Shared dispersion helper (same formula used by both bootstrap methods)
# ---------------------------------------------------------------------------

def _mean_dispersion(embeddings: torch.Tensor) -> float:
    """
    Mean Euclidean distance from each normalized embedding to the group centroid.

    Parameters
    ----------
    embeddings : Tensor of shape (K, D) — L2-normalized

    Returns
    -------
    float — mean within-group dispersion D_{i,g}

    Note: Even though embeddings are L2-normalized (unit vectors), we compute
    Euclidean distance to the centroid (which is NOT necessarily a unit vector
    after averaging). The metric is still bounded and interpretable:
    for unit vectors, Euclidean distance to centroid ∈ [0, sqrt(2)].
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {embeddings.shape}")
    centroid = embeddings.mean(dim=0, keepdim=True)
    dists = torch.cdist(embeddings, centroid, p=2).squeeze(-1)
    return float(dists.mean().item())


# ---------------------------------------------------------------------------
# BootstrapResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    method: str                  # "independent" or "paired"
    n_iterations: int
    observed_cds: float
    ci_low: float
    ci_high: float
    mean_bootstrap_cds: float
    std_bootstrap_cds: float
    prob_positive: float         # fraction of bootstrap replicates with CDS > 0
    prob_negative: float         # fraction of bootstrap replicates with CDS < 0
    direction: int               # direction_state(ci_low, ci_high)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


# ---------------------------------------------------------------------------
# Independent bootstrap (FAIRLens v1)
# ---------------------------------------------------------------------------

class IndependentBootstrap:
    """
    Bootstrap CI for CDS using INDEPENDENT resampling of A and B.

    Recommended for FAIRLens v1. Matches the generative process where
    K_A and K_B responses are drawn independently from the LLM.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed

    def compute(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> BootstrapResult:
        """
        Compute bootstrap CI for CDS = dispersion(A) - dispersion(B).

        Parameters
        ----------
        embeddings_a : Tensor (K, D) — L2-normalized embeddings for condition A
        embeddings_b : Tensor (K, D) — L2-normalized embeddings for condition B

        Returns
        -------
        BootstrapResult
        """
        k_a = embeddings_a.shape[0]
        k_b = embeddings_b.shape[0]

        observed_cds = _mean_dispersion(embeddings_a) - _mean_dispersion(embeddings_b)

        rng = torch.Generator()
        rng.manual_seed(self.seed)

        diffs: list[float] = []
        for _ in range(self.n_bootstrap):
            # Separate index vectors — independent resampling
            idx_a = torch.randint(0, k_a, (k_a,), generator=rng)
            idx_b = torch.randint(0, k_b, (k_b,), generator=rng)
            boot_a = embeddings_a[idx_a]
            boot_b = embeddings_b[idx_b]
            diffs.append(_mean_dispersion(boot_a) - _mean_dispersion(boot_b))

        diffs_arr = np.array(diffs, dtype=np.float64)
        alpha = 1.0 - self.confidence_level
        ci_low = float(np.percentile(diffs_arr, 100 * alpha / 2))
        ci_high = float(np.percentile(diffs_arr, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            method="independent",
            n_iterations=self.n_bootstrap,
            observed_cds=observed_cds,
            ci_low=ci_low,
            ci_high=ci_high,
            mean_bootstrap_cds=float(diffs_arr.mean()),
            std_bootstrap_cds=float(diffs_arr.std()),
            prob_positive=float((diffs_arr > 0).mean()),
            prob_negative=float((diffs_arr < 0).mean()),
            direction=direction_state(ci_low, ci_high),
        )


# ---------------------------------------------------------------------------
# Paired bootstrap (legacy_reproduction only)
# ---------------------------------------------------------------------------

class PairedBootstrap:
    """
    Bootstrap CI for CDS using the ORIGINAL CONSIST paired resampling.

    PRESERVED FOR LEGACY REPRODUCTION ONLY. The same random index vector
    is applied to both groups A and B, creating artificial pairing that
    does not reflect the independent generative process.

    Do NOT use for new FAIRLens v1 analyses. Use IndependentBootstrap instead.
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
        seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed

    def compute(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> BootstrapResult:
        """
        Compute bootstrap CI using the original paired-index methodology.

        Assumes embeddings_a.shape[0] == embeddings_b.shape[0] = K.
        """
        k = embeddings_a.shape[0]
        if embeddings_b.shape[0] != k:
            raise ValueError(
                f"Paired bootstrap requires equal K: got {k} vs {embeddings_b.shape[0]}"
            )

        observed_cds = _mean_dispersion(embeddings_a) - _mean_dispersion(embeddings_b)

        rng = torch.Generator()
        rng.manual_seed(self.seed)

        diffs: list[float] = []
        for _ in range(self.n_bootstrap):
            # SAME index for both — original CONSIST implementation
            idx = torch.randint(0, k, (k,), generator=rng)
            boot_a = embeddings_a[idx]
            boot_b = embeddings_b[idx]
            diffs.append(_mean_dispersion(boot_a) - _mean_dispersion(boot_b))

        diffs_arr = np.array(diffs, dtype=np.float64)
        alpha = 1.0 - self.confidence_level
        ci_low = float(np.percentile(diffs_arr, 100 * alpha / 2))
        ci_high = float(np.percentile(diffs_arr, 100 * (1 - alpha / 2)))

        return BootstrapResult(
            method="paired",
            n_iterations=self.n_bootstrap,
            observed_cds=observed_cds,
            ci_low=ci_low,
            ci_high=ci_high,
            mean_bootstrap_cds=float(diffs_arr.mean()),
            std_bootstrap_cds=float(diffs_arr.std()),
            prob_positive=float((diffs_arr > 0).mean()),
            prob_negative=float((diffs_arr < 0).mean()),
            direction=direction_state(ci_low, ci_high),
        )


# ---------------------------------------------------------------------------
# Bootstrap ablation: paired vs independent comparison
# ---------------------------------------------------------------------------

@dataclass
class BootstrapAblation:
    independent: BootstrapResult
    paired: BootstrapResult

    @property
    def ci_width_difference(self) -> float:
        """Positive = independent CI is wider."""
        return self.independent.ci_width - self.paired.ci_width

    @property
    def direction_agrees(self) -> bool:
        return self.independent.direction == self.paired.direction

    def summary(self) -> dict:
        return {
            "independent_ci": [self.independent.ci_low, self.independent.ci_high],
            "paired_ci": [self.paired.ci_low, self.paired.ci_high],
            "independent_width": self.independent.ci_width,
            "paired_width": self.paired.ci_width,
            "width_difference": self.ci_width_difference,
            "independent_direction": self.independent.direction,
            "paired_direction": self.paired.direction,
            "direction_agrees": self.direction_agrees,
            "independent_prob_positive": self.independent.prob_positive,
            "paired_prob_positive": self.paired.prob_positive,
        }


def bootstrap_ablation(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapAblation:
    """
    Run both independent and paired bootstrap on the same embeddings.

    Returns a BootstrapAblation for quantitative comparison of how much
    the methodological correction changes CI widths and direction labels.
    """
    ind = IndependentBootstrap(n_bootstrap=n_bootstrap, seed=seed)
    paired = PairedBootstrap(n_bootstrap=n_bootstrap, seed=seed)
    return BootstrapAblation(
        independent=ind.compute(embeddings_a, embeddings_b),
        paired=paired.compute(embeddings_a, embeddings_b),
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_bootstrap(method: str, n_bootstrap: int = 1000, seed: int = 42):
    """Return the appropriate bootstrap class for the given method name."""
    if method == "independent":
        return IndependentBootstrap(n_bootstrap=n_bootstrap, seed=seed)
    elif method == "paired":
        return PairedBootstrap(n_bootstrap=n_bootstrap, seed=seed)
    else:
        raise ValueError(f"Unknown bootstrap method: {method!r}. Use 'independent' or 'paired'.")
