"""
FAIRLens Core CDS Metrics
=========================

Extends the original CONSIST CDSCalculator with:
  - Absolute CDS (|CDS|) — disparity magnitude regardless of direction
  - Direction state (±1, 0) from bootstrap CI
  - Probability CDS > 0 and CDS < 0 from bootstrap distribution
  - Multi-signal profile: CDS + centroid separation together
  - Choice of bootstrap method (independent vs paired)

Relationship to consist/cds.py
-------------------------------
This module reimplements and extends the core formulas. It does NOT import
from consist/cds.py to avoid the finetune/peft circular dependency.
The mathematical formulas are identical to the original implementation.

Metric naming (final terminology used throughout FAIRLens)
----------------------------------------------------------
CDS   = Contextual Disparity Score  (signed: D_A - D_B)
ADM   = Absolute Disparity Magnitude  (|CDS|)
CS    = Centroid Separation  (cosine distance between group centroids)

These names are used consistently in all modules, schemas, and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

import torch
import numpy as np

from .bootstrap import (
    IndependentBootstrap,
    PairedBootstrap,
    BootstrapResult,
    direction_state,
    _mean_dispersion,
    make_bootstrap,
)


# ---------------------------------------------------------------------------
# Centroid separation
# ---------------------------------------------------------------------------

def centroid_separation(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    metric: str = "cosine",
) -> float:
    """
    Cosine (or alternative) distance between group centroids.

    CS = 1 - cosine_similarity(mean(emb_A), mean(emb_B))

    This measures whether the two groups' average semantic representations
    are located in different regions of embedding space. It is orthogonal
    to CDS: two groups can have CDS ≈ 0 (similar within-group spread) but
    large CS (their centroids point in different directions).

    Parameters
    ----------
    metric : "cosine" | "angular" | "euclidean"
    """
    c_a = embeddings_a.mean(dim=0, keepdim=True)
    c_b = embeddings_b.mean(dim=0, keepdim=True)

    if metric == "cosine":
        sim = torch.nn.functional.cosine_similarity(c_a, c_b, dim=1)
        return float((1.0 - sim).item())
    elif metric == "angular":
        sim = torch.nn.functional.cosine_similarity(c_a, c_b, dim=1)
        return float((torch.acos(sim.clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi).item())
    elif metric == "euclidean":
        return float(torch.cdist(c_a, c_b, p=2).item())
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'cosine', 'angular', or 'euclidean'.")


# ---------------------------------------------------------------------------
# Pair-level result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PairMetrics:
    """
    All FAIRLens metrics for one counterfactual prompt pair × model.

    Computed fields
    ---------------
    cds         : signed disparity (D_A - D_B)
    adm         : absolute disparity magnitude (|CDS|)
    dispersion_a, dispersion_b : within-condition mean dispersion
    centroid_separation : cosine distance between group centroids
    bootstrap   : full bootstrap result (CI, direction, probabilities)
    direction   : convenience accessor for bootstrap.direction (−1/0/+1)
    """

    # --- Identity fields (set by caller) ---
    pair_id: str = ""
    model_id: str = ""
    domain: str = ""
    template_idx: int = 0
    group_a: str = ""
    group_b: str = ""
    prompt_a: str = ""
    prompt_b: str = ""

    # --- Computed dispersion metrics ---
    dispersion_a: float = 0.0
    dispersion_b: float = 0.0
    cds: float = 0.0
    adm: float = 0.0               # |CDS|

    # --- Centroid separation ---
    centroid_separation: float = 0.0
    centroid_metric: str = "cosine"

    # --- Bootstrap results ---
    bootstrap_method: str = "independent"
    ci_low: float = 0.0
    ci_high: float = 0.0
    prob_positive: float = 0.0
    prob_negative: float = 0.0
    direction: int = 0             # −1 / 0 / +1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# FAIRLens CDS Calculator
# ---------------------------------------------------------------------------

class FAIRLensCDS:
    """
    Compute all per-pair FAIRLens metrics from embedding tensors.

    Parameters
    ----------
    bootstrap_method : "independent" (FAIRLens v1) | "paired" (legacy)
    n_bootstrap      : number of bootstrap replicates (1000 for v1, 500 for legacy)
    confidence_level : CI confidence level (default 0.95)
    centroid_metric  : metric for centroid separation
    bootstrap_seed   : seed for bootstrap RNG
    """

    def __init__(
        self,
        bootstrap_method: str = "independent",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        centroid_metric: str = "cosine",
        bootstrap_seed: int = 42,
    ):
        self.bootstrap_method = bootstrap_method
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.centroid_metric = centroid_metric
        self.bootstrapper = make_bootstrap(
            method=bootstrap_method,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )

    def compute(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        identity: Optional[dict] = None,
    ) -> PairMetrics:
        """
        Compute all metrics for one prompt pair.

        Parameters
        ----------
        embeddings_a : (K, D) L2-normalized tensor for condition A
        embeddings_b : (K, D) L2-normalized tensor for condition B
        identity     : optional dict with pair_id, model_id, domain, etc.

        Returns
        -------
        PairMetrics with all fields populated.
        """
        disp_a = _mean_dispersion(embeddings_a)
        disp_b = _mean_dispersion(embeddings_b)
        cds_val = disp_a - disp_b
        adm_val = abs(cds_val)

        cs = centroid_separation(embeddings_a, embeddings_b, self.centroid_metric)

        boot: BootstrapResult = self.bootstrapper.compute(embeddings_a, embeddings_b)

        pm = PairMetrics(
            dispersion_a=disp_a,
            dispersion_b=disp_b,
            cds=cds_val,
            adm=adm_val,
            centroid_separation=cs,
            centroid_metric=self.centroid_metric,
            bootstrap_method=self.bootstrap_method,
            ci_low=boot.ci_low,
            ci_high=boot.ci_high,
            prob_positive=boot.prob_positive,
            prob_negative=boot.prob_negative,
            direction=boot.direction,
        )

        if identity:
            for key, val in identity.items():
                if hasattr(pm, key):
                    setattr(pm, key, val)

        return pm


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def aggregate_metrics(pair_metrics: List[PairMetrics]) -> dict:
    """
    Compute all FAIRLens aggregation metrics over a list of PairMetrics.

    Returns a dict suitable for use in the aggregate result schema (Part 34).
    """
    if not pair_metrics:
        return {}

    cds_vals = np.array([p.cds for p in pair_metrics], dtype=np.float64)
    adm_vals = np.array([p.adm for p in pair_metrics], dtype=np.float64)
    cs_vals = np.array([p.centroid_separation for p in pair_metrics], dtype=np.float64)
    directions = [p.direction for p in pair_metrics]

    n = len(cds_vals)
    signed_mean = float(cds_vals.mean())
    mean_abs = float(adm_vals.mean())
    masking_gap = mean_abs - abs(signed_mean)         # always ≥ 0
    masking_ratio = (
        1.0 - abs(signed_mean) / mean_abs
        if mean_abs > 1e-9
        else 0.0
    )

    n_pos = sum(1 for d in directions if d == 1)
    n_neg = sum(1 for d in directions if d == -1)
    n_ind = sum(1 for d in directions if d == 0)

    return {
        "n": n,
        "signed_mean_cds": signed_mean,
        "median_cds": float(np.median(cds_vals)),
        "mean_absolute_cds": mean_abs,
        "std_cds": float(cds_vals.std()),
        "ci_mean_cds": [
            float(np.percentile(cds_vals, 2.5)),
            float(np.percentile(cds_vals, 97.5)),
        ],
        "positive_fraction": n_pos / n,
        "negative_fraction": n_neg / n,
        "indeterminate_fraction": n_ind / n,
        "masking_gap": float(masking_gap),
        "masking_ratio": float(masking_ratio),
        "mean_centroid_separation": float(cs_vals.mean()),
    }
