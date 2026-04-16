import random
from typing import Dict, List

def generate_prompt_variants(input_text: str, seed: int = 42, max_per_level: int = None, task_type: str = "summarization") -> Dict[str, List[str]]:
    """
    Generate structured, reproducible prompt variants aligned with the ORI framework.
    Uses perturbation levels for AUC-E computation to evaluate robustness.
    
    Levels:
    - d1: Literal (low perturbation)
    - d2: Stylistic (medium perturbation)
    - d3: Creative/complex (high perturbation)
    """
    import random
    import warnings
    # Deterministic generation
    rng = random.Random(seed)
    
    # Strip whitespace and handle empty
    text = input_text.strip()
    if not text:
        return {"d1": [], "d2": [], "d3": []}

    if task_type == "summarization":
        d1_templates = [
            "Summarize the following text: {}",
            "Provide a concise summary of the passage: {}",
            "Write a brief summary capturing key points: {}"
        ]
        
        d2_templates = [
            "Condense the text into a short summary: {}",
            "Summarize the main ideas clearly: {}",
            "Give a factual summary of the content: {}"
        ]
        
        d3_templates = [
            "Produce a neutral summary of the passage in 2-3 sentences: {}",
            "Summarize this article in a few sentences: {}",
            "Create a factual, non-stylized summary of this text: {}"
        ]
        
        # Validation Pass
        for template_set in [d1_templates, d2_templates, d3_templates]:
            for t in template_set:
                if "summar" not in t.lower():
                    warnings.warn(f"[Validation Warning] Prompt mismatch detected: '{t}' does not strictly ask for a summary.")
    elif task_type == "creative":
        d1_templates = ["Write a simple story about: {}", "Draft a creative text on: {}"]
        d2_templates = ["Write a poetic interpretation of: {}", "Create an emotional narrative for: {}"]
        d3_templates = ["Offer a humorous, sarcastic take on: {}", "Explain this concept as if we are in the year 2100: {}"]
    else:
        # Fallback to defaults
        d1_templates, d2_templates, d3_templates = [], [], []
    
    if max_per_level:
        d1_templates = rng.sample(d1_templates, min(max_per_level, len(d1_templates))) if d1_templates else []
        d2_templates = rng.sample(d2_templates, min(max_per_level, len(d2_templates))) if d2_templates else []
        d3_templates = rng.sample(d3_templates, min(max_per_level, len(d3_templates))) if d3_templates else []
        
    # Generate prompts avoiding duplicates
    prompts = {
        "d1": list(dict.fromkeys(t.format(text) for t in d1_templates)),
        "d2": list(dict.fromkeys(t.format(text) for t in d2_templates)),
        "d3": list(dict.fromkeys(t.format(text) for t in d3_templates))
    }
    
    return prompts

def flatten_prompt_variants(prompt_groups: Dict[str, List[str]]) -> List[str]:
    """Combine all perturbation levels into a single list of prompts."""
    flat_list = []
    for level_prompts in prompt_groups.values():
        flat_list.extend(level_prompts)
    return flat_list
