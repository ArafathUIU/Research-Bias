from typing import Optional, Dict, List
import os
import json
import numpy as np

from .config import CONSISTConfig
from .prompts import PromptGenerator, PromptSet
from .generate import GenerationHarness
from .embed import EmbeddingExtractor
from .cds import CDSCalculator, CDSResult
from .stats import StatisticalAnalyzer
from .validate import ValidationSuite, ValidationReport
from .finetune import FineTuningIntervention


class CONSISTPipeline:
    def __init__(self, cfg: Optional[CONSISTConfig] = None):
        self.cfg = cfg or CONSISTConfig()
        self.prompt_generator = PromptGenerator()
        self.generation_harness = GenerationHarness(self.cfg)
        self.embedding_extractor = EmbeddingExtractor(self.cfg)
        self.cds_calculator = CDSCalculator(self.cfg)
        self.prompt_set: Optional[PromptSet] = None
        self.results: Dict[float, CDSResult] = {}
        self.analyzer = None
        self.validation_report: Optional[ValidationReport] = None

    def setup(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.prompt_set = self.prompt_generator.generate()
        print(f"Generated {len(self.prompt_set)} prompt pairs across "
              f"{len(set(p.domain for p in self.prompt_set.pairs))} domains")
        return self

    def run_generation(
        self,
        model_name: Optional[str] = None,
    ):
        self.generation_harness.load_model(model_name)
        print(f"Model loaded: {model_name or self.cfg.model_name}")
        return self

    def _sample_fn(self, prompt: str) -> List[str]:
        return self.generation_harness.generate_samples(
            prompt,
            num_samples=self.cfg.num_samples,
            temperature=self.current_temperature,
        )

    def _embed_fn(self, texts: List[str]):
        return self.embedding_extractor.embed_texts(texts)

    def compute_cds(self, temperature: float) -> CDSResult:
        self.current_temperature = temperature
        self.embedding_extractor.load_model()
        result = self.cds_calculator.evaluate_prompt_set(
            self.prompt_set, self._sample_fn, self._embed_fn
        )
        self.results[temperature] = result
        print(f"CDS computed at T={temperature}: overall CDS = {result.overall_cds():.4f}")
        return result

    def compute_all_temperatures(self) -> Dict[float, CDSResult]:
        for t in self.cfg.temperatures:
            self.compute_cds(t)
        return self.results

    def analyze(self, temperature: Optional[float] = None) -> StatisticalAnalyzer:
        if temperature is not None:
            result = self.results.get(temperature)
            if result is None:
                raise ValueError(f"No results for T={temperature}")
        else:
            temp = self.cfg.temperatures[0]
            result = self.results.get(temp)
            if result is None:
                raise ValueError("No results computed yet")
        self.analyzer = StatisticalAnalyzer(result)
        return self.analyzer

    def validate(
        self,
        bbq_scores: Optional[Dict[str, float]] = None,
        downstream_scores: Optional[Dict[str, float]] = None,
        other_model_result: Optional[CDSResult] = None,
    ) -> ValidationReport:
        if not self.results:
            raise ValueError("Compute CDS before validation")
        main_result = self.results[self.cfg.temperatures[0]]
        suite = ValidationSuite(main_result)
        self.validation_report = suite.run_all(
            bbq_scores=bbq_scores,
            downstream_scores=downstream_scores,
            other_model_result=other_model_result,
        )
        return self.validation_report

    def run_intervention(
        self,
        model,
        tokenizer,
        n_per_group: int = 100,
        num_epochs: int = 3,
    ) -> dict:
        intervention = FineTuningIntervention(self.cfg)
        result = intervention.run_intervention(
            model=model,
            tokenizer=tokenizer,
            cds_calculator=self.cds_calculator,
            prompt_set=self.prompt_set,
            sample_fn=self._sample_fn,
            embed_fn=self._embed_fn,
            n_per_group=n_per_group,
            num_epochs=num_epochs,
        )
        return {
            "cds_before": result.cds_before.overall_cds(),
            "cds_after": result.cds_after.overall_cds(),
            "cds_change": result.cds_change,
            "n_train": result.n_train_examples,
        }

    def save_results(self, path: Optional[str] = None):
        path = path or os.path.join(self.cfg.output_dir, "cds_results.json")
        data = {
            "config": {
                "model": self.cfg.model_name,
                "num_samples": self.cfg.num_samples,
                "temperatures": self.cfg.temperatures,
                "distance_metric": self.cfg.distance_metric,
            },
            "results": {},
        }
        for temp, result in self.results.items():
            data["results"][str(temp)] = {
                "overall_cds": result.overall_cds(),
                "by_domain": result.aggregate_by_domain(),
                "by_group_pair": result.aggregate_by_group_pair(),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")

    def run_full(
        self,
        model_name: Optional[str] = None,
        bbq_scores: Optional[Dict[str, float]] = None,
        downstream_scores: Optional[Dict[str, float]] = None,
    ):
        self.setup()
        self.run_generation(model_name)
        self.compute_all_temperatures()
        self.analyze()
        self.validate(
            bbq_scores=bbq_scores,
            downstream_scores=downstream_scores,
        )
        self.save_results()
        return self
