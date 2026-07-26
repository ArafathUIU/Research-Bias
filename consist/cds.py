from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
import numpy as np

from .config import CONSISTConfig
from .prompts import PromptPair, PromptSet


@dataclass
class PairCDS:
    group_a: str
    group_b: str
    domain: str
    template: str
    cds_value: float
    dispersion_a: float
    dispersion_b: float
    distance_between_centroids: float
    ci_lower: float
    ci_upper: float
    p_value: Optional[float] = None


@dataclass
class CDSResult:
    per_pair: List[PairCDS] = field(default_factory=list)
    config: Optional[CONSISTConfig] = None

    def aggregate_by_domain(self) -> Dict[str, float]:
        domains = {}
        for pair in self.per_pair:
            domains.setdefault(pair.domain, []).append(pair.cds_value)
        return {d: float(np.mean(vals)) for d, vals in domains.items()}

    def aggregate_by_group_pair(self) -> Dict[str, float]:
        pairs = {}
        for p in self.per_pair:
            key = f"{p.group_a}_vs_{p.group_b}"
            pairs.setdefault(key, []).append(p.cds_value)
        return {k: float(np.mean(vals)) for k, vals in pairs.items()}

    def overall_cds(self) -> float:
        return float(np.mean([p.cds_value for p in self.per_pair]))


class CDSCalculator:
    def __init__(self, cfg: Optional[CONSISTConfig] = None):
        self.cfg = cfg or CONSISTConfig()

    @staticmethod
    def centroid(embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.mean(dim=0)

    @staticmethod
    def mean_dispersion(embeddings: torch.Tensor) -> float:
        c = embeddings.mean(dim=0, keepdim=True)
        dists = torch.cdist(embeddings, c, p=2).squeeze()
        return float(dists.mean().item())

    @staticmethod
    def centroid_distance(
        emb_a: torch.Tensor, emb_b: torch.Tensor, metric: str = "cosine"
    ) -> float:
        c_a = emb_a.mean(dim=0, keepdim=True)
        c_b = emb_b.mean(dim=0, keepdim=True)
        if metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(c_a, c_b)
            return float((1 - sim).item())
        elif metric == "angular":
            sim = torch.nn.functional.cosine_similarity(c_a, c_b)
            return float((torch.acos(sim.clamp(-1, 1)) / torch.pi).item())
        else:
            return float(torch.cdist(c_a, c_b, p=2).item())

    def compute_cds(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> float:
        disp_a = self.mean_dispersion(embeddings_a)
        disp_b = self.mean_dispersion(embeddings_b)
        return disp_a - disp_b

    def bootstrap_ci(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> tuple:
        n = self.cfg.n_bootstrap
        k = embeddings_a.shape[0]
        diffs = []
        for _ in range(n):
            idx = torch.randint(0, k, (k,))
            boot_a = embeddings_a[idx]
            boot_b = embeddings_b[idx]
            diffs.append(self.compute_cds(boot_a, boot_b))
        diffs = torch.tensor(diffs)
        alpha = 1 - self.cfg.confidence_level
        lower = float(diffs.quantile(alpha / 2).item())
        upper = float(diffs.quantile(1 - alpha / 2).item())
        return lower, upper

    def bootstrap_p_value(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> float:
        observed = self.compute_cds(embeddings_a, embeddings_b)
        n = self.cfg.n_bootstrap
        k = embeddings_a.shape[0]
        combined = torch.cat([embeddings_a, embeddings_b], dim=0)
        count = 0
        half = k
        for _ in range(n):
            idx = torch.randperm(2 * k)
            perm_a = combined[idx[:half]]
            perm_b = combined[idx[half:]]
            null_cds = self.compute_cds(perm_a, perm_b)
            if abs(null_cds) >= abs(observed):
                count += 1
        return count / n

    def evaluate_pair(
        self,
        pair: PromptPair,
        samples_a: List[str],
        samples_b: List[str],
        embed_fn,
    ) -> PairCDS:
        emb_a = embed_fn(samples_a)
        emb_b = embed_fn(samples_b)
        cds_val = self.compute_cds(emb_a, emb_b)
        disp_a = self.mean_dispersion(emb_a)
        disp_b = self.mean_dispersion(emb_b)
        centroid_dist = self.centroid_distance(emb_a, emb_b, self.cfg.distance_metric)
        ci_lower, ci_upper = self.bootstrap_ci(emb_a, emb_b)

        return PairCDS(
            group_a=pair.group_a,
            group_b=pair.group_b,
            domain=pair.domain,
            template=pair.template,
            cds_value=cds_val,
            dispersion_a=disp_a,
            dispersion_b=disp_b,
            distance_between_centroids=centroid_dist,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def evaluate_prompt_set(
        self,
        prompt_set: PromptSet,
        sample_fn,
        embed_fn,
    ) -> CDSResult:
        pairs = []
        for pair in prompt_set.pairs:
            samples_a = sample_fn(pair.prompt_a)
            samples_b = sample_fn(pair.prompt_b)
            cds = self.evaluate_pair(pair, samples_a, samples_b, embed_fn)
            pairs.append(cds)
        return CDSResult(per_pair=pairs, config=self.cfg)
