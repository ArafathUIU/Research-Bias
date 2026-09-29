"""
FAIRLens Configuration
======================

Two experiment tracks are defined:

  fairlens_v1 (DEFAULT):
    - primary_temperature explicitly set to 0.7
    - independent within-condition bootstrap resampling
    - SES contrast excluded pending redesign approval
    - validated corpus with pair IDs and hashes
    - full reproducibility manifests

  legacy_reproduction:
    - reproduces original CONSIST paper methodology
    - paired bootstrap (same index vector for A and B)
    - 500 bootstrap iterations
    - all 240 pairs including SES (with known rendering bug preserved as-is)
    - temperature 0.7 explicitly set (fixing the original silent T=0.3 bug)

These tracks must NEVER share result directories. No result from one track
may be used to validate or compare against the other without explicit labelling.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical model suite (final cross-model paper)
# Do not alter without updating the experiment manifest schema.
# ---------------------------------------------------------------------------
CANONICAL_MODELS: List[str] = [
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]

# Short identifiers for result filenames and manifest keys
MODEL_SHORT_IDS = {
    "microsoft/Phi-3-mini-4k-instruct": "phi3_mini",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b",
    "google/gemma-2-9b-it": "gemma2_9b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek_r1_8b",
}


# ---------------------------------------------------------------------------
# Demographic contrast configurations
# ---------------------------------------------------------------------------
#: Main Race/Gender Module (5 contrasts x 10 domains x 4 templates = 200 pairs)
FAIRLENS_V1_MAIN_CONTRASTS: List[Tuple[str, str]] = [
    ("Black_Female", "White_Male"),
    ("Black_Male", "White_Male"),
    ("Asian_Female", "White_Female"),
    ("Hispanic_Male", "White_Male"),
    ("Black_Female", "Black_Male"),
]

#: 2x2 Race x SES Factorial Module (gender fixed to male for FAIRLens v1)
#: 4 controlled pairwise contrasts constructed from four factorial cells:
#:   1. Black Low SES vs Black High SES -> SES comparison with race held constant
#:   2. White Low SES vs White High SES -> SES comparison with race held constant
#:   3. Black Low SES vs White Low SES  -> race comparison with SES held constant
#:   4. Black High SES vs White High SES -> race comparison with SES held constant
#: (4 contrasts x 10 domains x 4 templates = 160 pairs)
#: NOTE ON TERMINOLOGY:
#: These four CDS values are four pairwise contrasts constructed from four factorial cells,
#: NOT "CDS values for four cells". The descriptive difference between the two SES contrasts
#: across racial groups is a "descriptive comparison of SES disparity across racial groups"
#: or "descriptive moderation signal" — NOT a formal factorial interaction or ANOVA effect.
FAIRLENS_V1_SES_CONTRASTS: List[Tuple[str, str]] = [
    ("Black_LowSES_Male", "Black_HighSES_Male"),
    ("White_LowSES_Male", "White_HighSES_Male"),
    ("Black_LowSES_Male", "White_LowSES_Male"),
    ("Black_HighSES_Male", "White_HighSES_Male"),
]

#: Complete FAIRLens v1 audit corpus (5 + 4 = 9 contrasts x 40 = 360 prompt pairs)
FAIRLENS_V1_GROUP_PAIRS: List[Tuple[str, str]] = (
    FAIRLENS_V1_MAIN_CONTRASTS + FAIRLENS_V1_SES_CONTRASTS
)

#: Legacy reproduction track (6 contrasts x 40 = 240 prompt pairs)
#: Includes historical unrendered LowSES_Black vs HighSES_White
LEGACY_GROUP_PAIRS: List[Tuple[str, str]] = [
    ("Black_Female", "White_Male"),
    ("Black_Male", "White_Male"),
    ("Asian_Female", "White_Female"),
    ("Hispanic_Male", "White_Male"),
    ("LowSES_Black", "HighSES_White"),
    ("Black_Female", "Black_Male"),
]


@dataclass
class FAIRLensConfig:
    """
    Configuration for a FAIRLens v1 experiment run.

    Corpus structure (360 prompt pairs total):
    ------------------------------------------
    - Main Race/Gender module: 5 contrasts x 10 domains x 4 templates = 200 pairs
    - SES Factorial module:    4 contrasts x 10 domains x 4 templates = 160 pairs
      (gender fixed to male as a deliberate scope decision for FAIRLens v1)

    Interaction and contrast terminology:
    ------------------------------------
    - The 4 SES comparisons are four pairwise contrasts constructed from four factorial cells.
      Do NOT describe them as "CDS values for four cells".
    - Descriptive comparison between racial groups of SES disparity:
      "We compute a descriptive comparison of SES disparity across racial groups.
       This quantity is not interpreted as a formal factorial interaction effect."

    Seed strategy:
    --------------
    - Deterministic per-sample seeds derived via SHA-256 from:
      (experiment_id, model_id, pair_id, condition, sample_idx)
    - "Each sample uses a distinct deterministically derived pseudorandom seed."
      Hardware/library-level nondeterminism is acknowledged.
    """

    # -----------------------------------------------------------------------
    # Experiment identity
    # -----------------------------------------------------------------------
    experiment_track: str = "fairlens_v1"   # "fairlens_v1" | "legacy_reproduction"
    framework_version: str = "1.0.0"

    # -----------------------------------------------------------------------
    # Generation parameters
    # -----------------------------------------------------------------------
    #: Temperature for all main-experiment generations.
    #: MUST be set explicitly — do NOT read from robustness_temperatures[0].
    primary_temperature: float = 0.7

    #: Used ONLY for the temperature-robustness ablation (Phase 11).
    #: Never use as a fallback for primary generation.
    robustness_temperatures: List[float] = field(
        default_factory=lambda: [0.3, 0.7, 1.0]
    )

    num_samples: int = 20        # K responses per demographic condition
    top_p: float = 0.95
    max_new_tokens: int = 128
    do_sample: bool = True

    # -----------------------------------------------------------------------
    # Bootstrap parameters
    # -----------------------------------------------------------------------
    #: "independent" → separate resampling index for A and B (FAIRLens v1)
    #: "paired"      → same index for A and B (legacy_reproduction only)
    bootstrap_method: str = "independent"
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    bootstrap_seed: int = 42

    # -----------------------------------------------------------------------
    # Embedding parameters
    # -----------------------------------------------------------------------
    primary_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Encoders for the encoder-robustness ablation (Phase 9).
    robustness_encoders: List[str] = field(
        default_factory=lambda: [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "intfloat/e5-small-v2",
        ]
    )
    normalize_embeddings: bool = True

    # -----------------------------------------------------------------------
    # Distance metrics
    # -----------------------------------------------------------------------
    #: Metric for CENTROID SEPARATION (between-group).
    #: Valid values: "cosine" | "angular" | "euclidean"
    centroid_distance_metric: str = "cosine"

    #: NOTE: Within-group DISPERSION always uses Euclidean distance on
    #: L2-normalized embeddings, regardless of centroid_distance_metric.
    #: This is intentional: Euclidean on normalized vectors is equivalent to
    #: sqrt(2 * (1 - cosine_similarity)) and provides a bounded, interpretable
    #: distance measure. This distinction must be documented in all reports.

    # -----------------------------------------------------------------------
    # Model loading (GPU environments only)
    # -----------------------------------------------------------------------
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"   # "float16" | "bfloat16"
    device_map: str = "auto"

    # -----------------------------------------------------------------------
    # Corpus parameters
    # -----------------------------------------------------------------------
    #: Total controlled prompt pairs in FAIRLens v1 (200 main + 160 SES factorial)
    corpus_pair_count: int = 360
    #: SES module is fully approved and active in FAIRLens v1 as a 2x2 factorial design
    exclude_ses_contrast: bool = False

    # -----------------------------------------------------------------------
    # Randomness
    # -----------------------------------------------------------------------
    python_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42
    generation_seed: Optional[int] = None    # None = non-reproducible CUDA
    bootstrap_seed: int = 42
    subsample_seed: int = 42

    # -----------------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------------
    output_dir: str = "results"

    # -----------------------------------------------------------------------
    # Canonical model suite
    # -----------------------------------------------------------------------
    canonical_models: List[str] = field(
        default_factory=lambda: list(CANONICAL_MODELS)
    )


@dataclass
class LegacyConfig:
    """
    Configuration for the legacy_reproduction experiment track.

    Preserves every methodological choice from the original CONSIST pilot,
    EXCEPT: primary_temperature is now explicitly 0.7 (fixing the original
    silent T=0.3 default bug). This is the ONLY intentional change from
    the original methodology; it matches what the paper states was used.

    All other legacy choices are preserved:
    - Paired bootstrap (same index for A and B)
    - 500 bootstrap iterations
    - All 240 pairs including unvalidated SES contrast
    - MiniLM-L6-v2 encoder
    - Euclidean within-group dispersion
    - Cosine centroid separation
    """

    experiment_track: str = "legacy_reproduction"
    framework_version: str = "0.1.0-legacy"

    # Generation
    primary_temperature: float = 0.7   # explicitly set; paper used 0.7
    num_samples: int = 20
    top_p: float = 0.95
    max_new_tokens: int = 128
    do_sample: bool = True

    # Bootstrap — PAIRED (original methodology)
    bootstrap_method: str = "paired"
    n_bootstrap: int = 500            # original pilot used 500
    confidence_level: float = 0.95
    bootstrap_seed: int = 42

    # Embedding
    primary_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True

    # Distance
    centroid_distance_metric: str = "cosine"

    # Corpus — ALL 240 pairs including SES
    exclude_ses_contrast: bool = False

    # Model loading
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    device_map: str = "auto"

    # Randomness
    python_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42

    # Storage
    output_dir: str = "results"
