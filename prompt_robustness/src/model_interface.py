import os
import time
import torch
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

class ModelInterface:
    def __init__(self, model_name: str, config: Config):
        self.config = config
        self.model_name = model_name
        
        # Device map
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"\n--- Running model: {model_name} on {self.device} ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad_token_id manually if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Hard mapped to Seq2Seq architectures seamlessly 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_responses(self, prompts: List[str]) -> List[str]:
        start_t = time.time()
        
        # Apply specific prompt syntax formatting per model arrays
        formatted_prompts = []
        for p in prompts:
            if "flan-t5" in self.model_name.lower():
                if not p.lower().startswith("summarize:"):
                    formatted_prompts.append(f"summarize: {p}")
                else:
                    formatted_prompts.append(p)
            else:
                formatted_prompts.append(p)
                
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"[{self.model_name}] Generated {len(prompts)} responses in {time.time() - start_t:.2f}s")
        return [r.strip() for r in responses]

    def compute_perplexity_variance(self, prompts, responses):
        """Compute perplexity variance across (prompt + response) pairs.

        For Seq2Seq models (e.g., BART, T5), this correctly uses the prompt as
        encoder input and the response as decoder labels, which mirrors how the
        model processes text during generation. This yields a more faithful
        perplexity estimate than computing on responses alone.

        For each pair (prompt_i, response_i):
            PPL_i = exp(cross_entropy_loss(prompt_i → response_i))

        Returns:
            Normalized variance of PPL values across all pairs, clamped to [0, 1].
        """
        import math
        ppl_values = []

        n = min(len(prompts), len(responses))
        for i in range(n):
            prompt = prompts[i]
            resp = responses[i]

            # Tokenize prompt as encoder input, response as decoder labels
            encoder_inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            decoder_inputs = self.tokenizer(
                resp, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            if encoder_inputs["input_ids"].shape[1] == 0 or decoder_inputs["input_ids"].shape[1] == 0:
                continue

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs.get("attention_mask"),
                    labels=decoder_inputs["input_ids"]
                )
                loss = outputs.loss.item()
                # Guard against extreme loss values that would overflow exp()
                loss = min(loss, 20.0)
                ppl = math.exp(loss)
                ppl_values.append(ppl)

        if len(ppl_values) < 2:
            return 0.0

        mean = sum(ppl_values) / len(ppl_values)
        var = sum((x - mean) ** 2 for x in ppl_values) / (len(ppl_values) - 1)
        # Normalize: divide by 1000 to bring variance into [0, 1] range
        return max(0.0, min(1.0, var / 1000.0))

    def compute_branching_factor(self, responses):
        import math
        entropies = []
        for resp in responses:
            inputs = self.tokenizer(resp, return_tensors="pt").to(self.device)
            if inputs["input_ids"].shape[1] == 0:
                 continue
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs["input_ids"]
                )
                logits = outputs.logits.squeeze(0)  
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token_entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1)
                entropies.append(token_entropy.mean().item())
        if not entropies:
            return 0.0
        avg_entropy = sum(entropies) / len(entropies)
        bf = 2 ** avg_entropy
        vocab_size = self.tokenizer.vocab_size
        max_bf = 2 ** math.log2(vocab_size) if vocab_size > 0 else 1.0
        return max(0.0, min(1.0, bf / max_bf))
