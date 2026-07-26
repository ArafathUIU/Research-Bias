from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
from scipy import stats

from .cds import CDSResult


@dataclass
class ValidationReport:
    correlation_results: Dict[str, float] = field(default_factory=dict)
    correlation_pvalues: Dict[str, float] = field(default_factory=dict)
    novel_detection: Dict[str, bool] = field(default_factory=dict)
    cross_model_agreement: Optional[float] = None
    incremental_validity: Optional[Dict] = None


class ValidationSuite:
    def __init__(self, cds_result: CDSResult):
        self.result = cds_result

    def correlate_with_benchmark(
        self,
        benchmark_scores: Dict[str, float],
    ) -> Dict[str, float]:
        cds_by_group = self.result.aggregate_by_group_pair()
        shared_keys = set(cds_by_group.keys()) & set(benchmark_scores.keys())
        if len(shared_keys) < 3:
            return {"spearman_r": 0.0, "p_value": 1.0, "n": len(shared_keys)}

        cds_vals = [cds_by_group[k] for k in shared_keys]
        bench_vals = [benchmark_scores[k] for k in shared_keys]
        r, p = stats.spearmanr(cds_vals, bench_vals)
        return {"spearman_r": float(r), "p_value": float(p), "n": len(shared_keys)}

    def correlate_with_bbq(self, bbq_scores: Dict[str, float]) -> Dict:
        return self.correlate_with_benchmark(bbq_scores)

    def test_novel_detection(
        self, bbq_scores: Dict[str, float], threshold: float = 0.8
    ) -> Dict[str, bool]:
        cds_by_group = self.result.aggregate_by_group_pair()
        result = {}
        for group_key, cds_val in cds_by_group.items():
            bbq_val = bbq_scores.get(group_key, None)
            if bbq_val is None:
                continue
            if bbq_val < threshold and abs(cds_val) > 0.05:
                result[group_key] = True
            else:
                result[group_key] = False
        return result

    def cross_model_agreement(
        self, other_result: CDSResult
    ) -> float:
        cds_1 = [p.cds_value for p in self.result.per_pair]
        cds_2 = [p.cds_value for p in other_result.per_pair]
        if len(cds_1) != len(cds_2):
            return 0.0
        r, _ = stats.pearsonr(cds_1, cds_2)
        return float(r)

    def incremental_validity(
        self,
        bbq_scores: Dict[str, float],
        downstream_fairness_scores: Dict[str, float],
    ) -> Dict:
        cds_by_group = self.result.aggregate_by_group_pair()
        shared = (
            set(cds_by_group.keys())
            & set(bbq_scores.keys())
            & set(downstream_fairness_scores.keys())
        )
        if len(shared) < 4:
            return {"error": "too few shared groups", "n": len(shared)}

        cds_arr = np.array([cds_by_group[k] for k in shared])
        bbq_arr = np.array([bbq_scores[k] for k in shared])
        downstream_arr = np.array([downstream_fairness_scores[k] for k in shared])

        r_bbq_only, _ = stats.pearsonr(bbq_arr, downstream_arr)

        cds_residual = stats.linregress(cds_arr, bbq_arr).residual
        r_cds_residual, p_cds_residual = stats.pearsonr(cds_residual, downstream_arr)

        return {
            "r_bbq_only": float(r_bbq_only),
            "r_cds_incremental": float(r_cds_residual),
            "p_cds_incremental": float(p_cds_residual),
            "n": len(shared),
        }

    def run_all(
        self,
        bbq_scores: Optional[Dict[str, float]] = None,
        downstream_scores: Optional[Dict[str, float]] = None,
        other_model_result: Optional[CDSResult] = None,
    ) -> ValidationReport:
        report = ValidationReport()

        if bbq_scores:
            corr = self.correlate_with_bbq(bbq_scores)
            report.correlation_results["bbq_vs_cds"] = corr.get("spearman_r", 0)
            report.correlation_pvalues["bbq_vs_cds"] = corr.get("p_value", 1)

        if bbq_scores:
            report.novel_detection = self.test_novel_detection(bbq_scores)

        if other_model_result:
            report.cross_model_agreement = self.cross_model_agreement(other_model_result)

        if bbq_scores and downstream_scores:
            report.incremental_validity = self.incremental_validity(
                bbq_scores, downstream_scores
            )

        return report
