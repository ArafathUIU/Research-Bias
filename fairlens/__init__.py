"""
FAIRLens Package
================

Provides the FAIRLens framework for demographic representational disparity
auditing in generative LLMs.

Two experiment tracks:
  fairlens_v1         — corrected methodology (independent bootstrap, validated corpus)
  legacy_reproduction — preserves original CONSIST pilot methodology for comparison

The original consist/ package is preserved unchanged (except surgical bug fixes)
for backward compatibility and legacy reproduction.
"""

from .config import FAIRLensConfig, LegacyConfig, CANONICAL_MODELS, MODEL_SHORT_IDS
from .corpus import (
    CorpusValidator,
    CorpusPair,
    AuditResult,
    make_pair_id,
    sha256_of_pair,
    sha256_of_file,
    CANONICAL_GROUP_PAIRS,
    SES_GROUP_PAIRS,
)
from .bootstrap import (
    IndependentBootstrap,
    PairedBootstrap,
    BootstrapResult,
    BootstrapAblation,
    bootstrap_ablation,
    direction_state,
    make_bootstrap,
)
from .cds import (
    FAIRLensCDS,
    PairMetrics,
    centroid_separation,
    aggregate_metrics,
)
from .masking import (
    compute_masking,
    MaskingResult,
    illustrate_masking_example,
)
from .manifest import (
    ExperimentManifest,
    make_manifest,
    collect_hardware,
    collect_software,
)
from .storage import (
    GenerationStore,
    atomic_write_json,
    atomic_append_jsonl,
    create_result_structure,
    generation_db_path,
    generation_jsonl_path,
    validate_sample_record,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "FAIRLensConfig", "LegacyConfig", "CANONICAL_MODELS", "MODEL_SHORT_IDS",
    # Corpus
    "CorpusValidator", "CorpusPair", "AuditResult",
    "make_pair_id", "sha256_of_pair", "sha256_of_file",
    "CANONICAL_GROUP_PAIRS", "SES_GROUP_PAIRS",
    # Bootstrap
    "IndependentBootstrap", "PairedBootstrap",
    "BootstrapResult", "BootstrapAblation",
    "bootstrap_ablation", "direction_state", "make_bootstrap",
    # CDS
    "FAIRLensCDS", "PairMetrics",
    "centroid_separation", "aggregate_metrics",
    # Masking
    "compute_masking", "MaskingResult", "illustrate_masking_example",
    # Manifest
    "ExperimentManifest", "make_manifest", "collect_hardware", "collect_software",
    # Storage
    "GenerationStore", "atomic_write_json", "atomic_append_jsonl",
    "create_result_structure", "generation_db_path", "generation_jsonl_path",
    "validate_sample_record",
]
