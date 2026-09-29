from .config import CONSISTConfig, GroupConfig, INTERSECTIONAL_GROUPS, CDS_DISTANCE_METRICS
from .prompts import PromptGenerator, PromptPair, PromptSet, BIAS_DOMAINS, TEMPLATES
from .generate import GenerationHarness
from .embed import EmbeddingExtractor
from .cds import CDSCalculator, CDSResult, PairCDS
from .stats import StatisticalAnalyzer
from .validate import ValidationSuite, ValidationReport
from .pipeline import CONSISTPipeline

# BUG FIX: finetune.py requires peft which is not installed in local analysis
# environments. Make this import optional — peft is only needed on GPU cloud nodes.
try:
    from .finetune import FineTuningIntervention, InterventionResult
except ImportError:
    FineTuningIntervention = None  # type: ignore
    InterventionResult = None      # type: ignore

__all__ = [
    "CONSISTConfig", "GroupConfig", "INTERSECTIONAL_GROUPS", "CDS_DISTANCE_METRICS",
    "PromptGenerator", "PromptPair", "PromptSet", "BIAS_DOMAINS", "TEMPLATES",
    "GenerationHarness",
    "EmbeddingExtractor",
    "CDSCalculator", "CDSResult", "PairCDS",
    "StatisticalAnalyzer",
    "ValidationSuite", "ValidationReport",
    "FineTuningIntervention", "InterventionResult",
    "CONSISTPipeline",
]
