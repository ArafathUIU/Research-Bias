from typing import List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from .config import CONSISTConfig


class GenerationHarness:
    def __init__(self, cfg: Optional[CONSISTConfig] = None):
        self.cfg = cfg or CONSISTConfig()
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def load_model(self, model_name: Optional[str] = None):
        model_name = model_name or self.cfg.model_name
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_samples(
        self,
        prompt: str,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        num_samples = num_samples or self.cfg.num_samples
        temperature = temperature if temperature is not None else self.cfg.temperatures[0]
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens

        outputs = []
        for _ in range(num_samples):
            result = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                return_full_text=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            outputs.append(result[0]["generated_text"])
        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        return [
            self.generate_samples(p, num_samples, temperature)
            for p in prompts
        ]
