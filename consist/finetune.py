from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from .config import CONSISTConfig
from .cds import CDSCalculator, CDSResult


@dataclass
class InterventionResult:
    cds_before: CDSResult
    cds_after: CDSResult
    cds_change: float
    n_train_examples: int


class FineTuningIntervention:
    def __init__(self, cfg: Optional[CONSISTConfig] = None):
        self.cfg = cfg or CONSISTConfig()

    def build_balanced_dataset(self, n_per_group: int = 100) -> List[Dict]:
        from .prompts import PromptGenerator, TEMPLATES, INTERSECTIONAL_GROUPS

        examples = []
        groups = list(INTERSECTIONAL_GROUPS.keys())
        import random
        rng = random.Random(self.cfg.seed)

        for group_name in groups:
            group = INTERSECTIONAL_GROUPS[group_name]
            templates_flat = []
            for domain_templates in TEMPLATES.values():
                templates_flat.extend(domain_templates)
            rng.shuffle(templates_flat)
            selected = templates_flat[:n_per_group]
            for t in selected:
                filled = t.format(
                    race=group.race or "the",
                    gender=group.gender or "person",
                )
                examples.append({
                    "text": filled,
                    "group": group_name,
                })
        return examples

    def prepare_dataset(self, examples: List[Dict]) -> Dataset:
        texts = [ex["text"] for ex in examples]
        ds = Dataset.from_dict({"text": texts})
        def tokenize(ex):
            return self.tokenizer(
                ex["text"], truncation=True, padding="max_length", max_length=256
            )
        ds = ds.map(tokenize, batched=True)
        return ds

    def apply_lora(self, model, r: int = 8, alpha: int = 16):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        return model

    def finetune(
        self,
        train_dataset: Dataset,
        output_dir: str = "finetune_output",
        num_epochs: int = 3,
        lr: float = 2e-4,
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )
        trainer.train()
        return trainer

    def run_intervention(
        self,
        model,
        tokenizer,
        cds_calculator: CDSCalculator,
        prompt_set,
        sample_fn,
        embed_fn,
        n_per_group: int = 100,
        num_epochs: int = 3,
    ) -> InterventionResult:
        self.model = model
        self.tokenizer = tokenizer

        cds_before = cds_calculator.evaluate_prompt_set(
            prompt_set, sample_fn, embed_fn
        )

        examples = self.build_balanced_dataset(n_per_group)
        train_ds = self.prepare_dataset(examples)
        self.model = self.apply_lora(self.model)
        self.finetune(train_ds, num_epochs=num_epochs)

        cds_after = cds_calculator.evaluate_prompt_set(
            prompt_set, sample_fn, embed_fn
        )

        cds_change = cds_after.overall_cds() - cds_before.overall_cds()
        return InterventionResult(
            cds_before=cds_before,
            cds_after=cds_after,
            cds_change=cds_change,
            n_train_examples=len(examples),
        )
