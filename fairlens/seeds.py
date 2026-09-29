"""
FAIRLens Deterministic Seed Strategy
====================================

Implements the SHA-256-derived per-sample pseudorandom seed generation.

Scientific Rationale & Requirements
-----------------------------------
- Identity tuple contains at minimum:
    (experiment_id, model_id, pair_id, condition, sample_idx)
- Every generated sample record explicitly stores its derived integer seed.
- Reproducibility statement:
    "Each sample uses a distinct deterministically derived pseudorandom seed."
  Note: This does not claim that SHA-256-derived seeds make stochastic samples
  statistically independent. Furthermore, hardware- and library-level
  nondeterminism (such as non-deterministic GPU kernel execution order in PyTorch/CUDA)
  means bitwise exact reproducibility across differing hardware cannot be guaranteed.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Any


def derive_sample_seed(
    experiment_id: str,
    model_id: str,
    pair_id: str,
    condition: str,
    sample_idx: int,
) -> int:
    """
    Derive a deterministic integer seed from sample identity fields.

    Parameters
    ----------
    experiment_id : str
        Unique identifier of the experiment run.
    model_id : str
        Canonical model short identifier (e.g. 'phi3_mini').
    pair_id : str
        Corpus pair ID (e.g. 'P70815598687').
    condition : str
        Demographic condition arm ('A' or 'B').
    sample_idx : int
        Sample index within condition (0 to K-1).

    Returns
    -------
    int
        A 31-bit non-negative integer pseudorandom seed in [0, 2**31 - 2].
    """
    identity = f"{experiment_id}:{model_id}:{pair_id}:{condition}:{sample_idx}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    # 31-bit positive int, guaranteed non-negative and compatible with PyTorch/NumPy
    return int(digest[:8], 16) % 2147483647


def make_sample_identity_dict(
    experiment_id: str,
    model_id: str,
    pair_id: str,
    condition: str,
    sample_idx: int,
) -> Dict[str, Any]:
    """Return dictionary with sample identity and its derived seed."""
    seed = derive_sample_seed(
        experiment_id=experiment_id,
        model_id=model_id,
        pair_id=pair_id,
        condition=condition,
        sample_idx=sample_idx,
    )
    return {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "pair_id": pair_id,
        "condition": condition,
        "sample_idx": sample_idx,
        "seed": seed,
    }
