from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GroupConfig:
    race: Optional[str] = None
    gender: Optional[str] = None
    ses: Optional[str] = None


INTERSECTIONAL_GROUPS = {
    "Black_Female": GroupConfig(race="Black", gender="female"),
    "Black_Male": GroupConfig(race="Black", gender="male"),
    "White_Female": GroupConfig(race="White", gender="female"),
    "White_Male": GroupConfig(race="White", gender="male"),
    "Asian_Female": GroupConfig(race="Asian", gender="female"),
    "Asian_Male": GroupConfig(race="Asian", gender="male"),
    "Hispanic_Female": GroupConfig(race="Hispanic", gender="female"),
    "Hispanic_Male": GroupConfig(race="Hispanic", gender="male"),
    "LowSES_Black": GroupConfig(race="Black", ses="low"),
    "HighSES_Black": GroupConfig(race="Black", ses="high"),
    "LowSES_White": GroupConfig(race="White", ses="low"),
    "HighSES_White": GroupConfig(race="White", ses="high"),
}


CDS_DISTANCE_METRICS = ["cosine", "euclidean", "angular"]


@dataclass
class CONSISTConfig:
    num_samples: int = 20
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.7, 1.0])
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    distance_metric: str = "cosine"
    batch_size: int = 4
    max_new_tokens: int = 128
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    output_dir: str = "results"
    seed: int = 42
