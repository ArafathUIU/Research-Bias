from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from .config import CONSISTConfig


class EmbeddingExtractor:
    def __init__(self, cfg: Optional[CONSISTConfig] = None):
        self.cfg = cfg or CONSISTConfig()
        self.model = None

    def load_model(self, model_name: Optional[str] = None):
        model_name = model_name or self.cfg.embedding_model
        self.model = SentenceTransformer(model_name, device=self.cfg.device)

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_samples(
        self, samples: List[List[str]]
    ) -> List[torch.Tensor]:
        return [self.embed_texts(group_samples) for group_samples in samples]
