from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from .cds import CDSResult


class StatisticalAnalyzer:
    def __init__(self, result: CDSResult):
        self.result = result

    def domain_summary(self) -> Dict[str, dict]:
        by_domain = {}
        for pair in self.result.per_pair:
            by_domain.setdefault(pair.domain, []).append(pair.cds_value)
        summary = {}
        for domain, vals in by_domain.items():
            summary[domain] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
        return summary

    def group_pair_summary(self) -> Dict[str, dict]:
        by_pair = {}
        for p in self.result.per_pair:
            key = f"{p.group_a}_vs_{p.group_b}"
            by_pair.setdefault(key, []).append(p.cds_value)
        summary = {}
        for key, vals in by_pair.items():
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
        return summary

    def effect_sizes(self) -> Dict[str, float]:
        by_pair = {}
        for p in self.result.per_pair:
            key = f"{p.group_a}_vs_{p.group_b}"
            by_pair.setdefault(key, []).append(p.cds_value)
        es = {}
        for key, vals in by_pair.items():
            vals_arr = np.array(vals)
            es[key] = float(vals_arr.mean() / (vals_arr.std() + 1e-8))
        return es

    def significant_pairs(self, alpha: float = 0.05) -> List[str]:
        sig = []
        for p in self.result.per_pair:
            if p.p_value is not None and p.p_value < alpha:
                sig.append(
                    f"{p.group_a}_vs_{p.group_b}|{p.domain}: CDS={p.cds_value:.4f} "
                    f"(p={p.p_value:.4f})"
                )
        return sig

    def plot_cds_by_domain(self, save_path: Optional[str] = None):
        summary = self.domain_summary()
        domains = list(summary.keys())
        means = [summary[d]["mean"] for d in domains]
        stds = [summary[d]["std"] for d in domains]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(domains, means, xerr=stds, capsize=4)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Mean CDS")
        ax.set_title("CDS by Bias Domain")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_cds_by_group_pair(self, save_path: Optional[str] = None):
        summary = self.group_pair_summary()
        labels = list(summary.keys())
        means = [summary[l]["mean"] for l in labels]
        stds = [summary[l]["std"] for l in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels, means, xerr=stds, capsize=4)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Mean CDS")
        ax.set_title("CDS by Group Pair")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_temperature_comparison(
        self, results_by_temp: Dict[float, CDSResult], save_path: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(self.result.per_pair))
        width = 0.25
        for i, (temp, res) in enumerate(sorted(results_by_temp.items())):
            vals = [p.cds_value for p in res.per_pair]
            ax.bar(x + i * width, vals, width, label=f"T={temp}")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{p.group_a[:4]}vs{p.group_b[:4]}" for p in self.result.per_pair])
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("CDS")
        ax.set_title("CDS Across Temperatures")
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
