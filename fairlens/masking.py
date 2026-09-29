"""
FAIRLens Masking Analysis
=========================

Implements the Aggregation Masking metrics described in Parts 13-14.

Problem
-------
Signed CDS values can cancel in the aggregate even when local disparities
are substantial. A model that generates +0.10 disparity for some prompts
and -0.10 for others produces mean(CDS) ≈ 0 but is NOT neutral locally.

Metrics
-------
Masking Gap (MG)
    MG = mean(|CDS|) - |mean(CDS)|

    MG ≥ 0 always (by Jensen's inequality: E[|X|] ≥ |E[X]|).
    A large MG means local effects are cancelling in the signed aggregate.
    MG = 0 only when all CDS values have the same sign (no cancellation).

Masking Ratio (MR)
    MR = 1 - |mean(CDS)| / mean(|CDS|)

    MR ∈ [0, 1] when mean(|CDS|) > 0.
    MR ≈ 0  → little cancellation (signed aggregate captures local effects)
    MR ≈ 1  → strong cancellation (signed aggregate hides local effects)

    Special case: if mean(|CDS|) < epsilon, MR is undefined (return NaN).

Signed Aggregate
    |mean(CDS)| — the absolute value of the signed mean.
    This is what a naive aggregate analysis would report.

All three quantities are needed together for a complete characterisation.
Reporting only mean(CDS) or only |mean(CDS)| is insufficient.

Reference
---------
These measures are proposed FAIRLens metrics. The relationship between
signed and unsigned aggregates is related to the literature on variance
decomposition but the specific MG/MR formulation should be verified
against any equivalent existing measures before claiming novelty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


MASKING_EPSILON = 1e-9  # Minimum mean(|CDS|) before MR is declared undefined


@dataclass
class MaskingResult:
    """
    Full masking analysis for a set of CDS values.

    Attributes
    ----------
    n                 : number of CDS values
    signed_mean       : mean(CDS) — may cancel positive and negative
    absolute_mean     : mean(|CDS|) — always ≥ |signed_mean|
    signed_aggregate  : |mean(CDS)| — what naive aggregation reports
    masking_gap       : MG = absolute_mean - signed_aggregate   (≥ 0)
    masking_ratio     : MR = 1 - signed_aggregate / absolute_mean  (∈ [0,1])
    masking_ratio_nan : True if absolute_mean < epsilon (MR undefined)
    positive_fraction : fraction of CDS > 0
    negative_fraction : fraction of CDS < 0
    zero_fraction     : fraction of CDS == 0 exactly
    """
    n: int
    signed_mean: float
    absolute_mean: float
    signed_aggregate: float     # |mean(CDS)|
    masking_gap: float          # MG
    masking_ratio: float        # MR (NaN if undefined)
    masking_ratio_nan: bool
    positive_fraction: float
    negative_fraction: float
    zero_fraction: float
    std_cds: float
    median_cds: float

    def to_dict(self) -> dict:
        import math
        return {
            "n": self.n,
            "signed_mean_cds": self.signed_mean,
            "mean_absolute_cds": self.absolute_mean,
            "signed_aggregate": self.signed_aggregate,
            "masking_gap": self.masking_gap,
            "masking_ratio": None if self.masking_ratio_nan else self.masking_ratio,
            "masking_ratio_undefined": self.masking_ratio_nan,
            "positive_fraction": self.positive_fraction,
            "negative_fraction": self.negative_fraction,
            "zero_fraction": self.zero_fraction,
            "std_cds": self.std_cds,
            "median_cds": self.median_cds,
        }


def compute_masking(cds_values: List[float]) -> MaskingResult:
    """
    Compute all masking metrics for a list of signed CDS values.

    Parameters
    ----------
    cds_values : list of signed CDS values for a set of prompt pairs

    Returns
    -------
    MaskingResult with all metrics populated.

    Raises
    ------
    ValueError if cds_values is empty.

    Mathematical guarantee
    ----------------------
    masking_gap >= 0 always. This follows from Jensen's inequality:
    for the convex function f(x) = |x|, E[f(X)] >= f(E[X]),
    so mean(|CDS|) >= |mean(CDS)|, therefore MG = mean(|CDS|) - |mean(CDS)| >= 0.
    The unit test test_masking_gap_nonnegative verifies this for random inputs.
    """
    if not cds_values:
        raise ValueError("cds_values must not be empty")

    arr = np.array(cds_values, dtype=np.float64)
    n = len(arr)

    signed_mean = float(arr.mean())
    absolute_mean = float(np.abs(arr).mean())
    signed_aggregate = float(abs(signed_mean))
    masking_gap = float(absolute_mean - signed_aggregate)

    # Numerical safety: MG can be tiny-negative due to floating-point
    # arithmetic. Clamp to 0.
    if masking_gap < 0.0 and masking_gap > -1e-12:
        masking_gap = 0.0

    masking_ratio_nan = absolute_mean < MASKING_EPSILON
    masking_ratio = (
        float(1.0 - signed_aggregate / absolute_mean)
        if not masking_ratio_nan
        else float("nan")
    )

    return MaskingResult(
        n=n,
        signed_mean=signed_mean,
        absolute_mean=absolute_mean,
        signed_aggregate=signed_aggregate,
        masking_gap=masking_gap,
        masking_ratio=masking_ratio,
        masking_ratio_nan=masking_ratio_nan,
        positive_fraction=float((arr > 0).mean()),
        negative_fraction=float((arr < 0).mean()),
        zero_fraction=float((arr == 0).mean()),
        std_cds=float(arr.std()),
        median_cds=float(np.median(arr)),
    )


def illustrate_masking_example() -> str:
    """
    Return a string illustrating the masking problem with a concrete example.

    Used in the Implementation Report and methodology document.
    """
    example_cds = [+0.10, -0.10, +0.12, -0.12]
    result = compute_masking(example_cds)
    lines = [
        "Masking Example: [+0.10, -0.10, +0.12, -0.12]",
        f"  signed_mean(CDS) = {result.signed_mean:+.4f}",
        f"  |signed_mean|    = {result.signed_aggregate:.4f}  ← what naive aggregation shows",
        f"  mean(|CDS|)      = {result.absolute_mean:.4f}  ← actual local effect magnitude",
        f"  Masking Gap      = {result.masking_gap:.4f}  ← hidden by cancellation",
        f"  Masking Ratio    = {result.masking_ratio:.4f}  ← {result.masking_ratio*100:.0f}% of local effect masked",
    ]
    return "\n".join(lines)
