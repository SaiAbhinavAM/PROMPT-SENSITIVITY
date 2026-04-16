from typing import List
from .model_interface import ModelInterface

def compute_ppl_variance(prompts: List[str], responses: List[str], model_interface: ModelInterface) -> float:
    """Perplexity variance metric proxy.
    Computes perplexity variance via ModelInterface.
    """
    return model_interface.compute_perplexity_variance(prompts, responses)
