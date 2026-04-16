from typing import List
from .model_interface import ModelInterface

def compute_branching_factor(responses: List[str], model_interface: ModelInterface) -> float:
    """Branching Factor (BF) metric proxy."""
    return model_interface.compute_branching_factor(responses)
