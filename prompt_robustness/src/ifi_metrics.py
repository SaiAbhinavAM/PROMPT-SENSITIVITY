"""
IFI (Internal Fidelity Index) Metrics for LL-PIRC Pipeline.

Near-zero cost add-on metrics computed during baseline forward passes:

1. PPL_var: Standard deviation of sequence-level perplexity across K
   paraphrases per article. High PPL_var → internal processing is
   inconsistent across prompt variants.

2. PC_stab: Mean variance of branching factor B = exp(H) across K
   paraphrases, averaged over generation timesteps. Uses temperature=1
   softmax and torch.distributions.Categorical entropy.
   High PC_stab → model uncertainty varies strongly with prompt wording.

Reference: method_analysis_prompt_sensitivity.md, Section Q4-Q5 (lines
           59-82), IFI metrics specification
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_sequence_perplexity(
    model,
    tokenizer,
    prompt: str,
    device: Optional[str] = None
) -> float:
    """
    Compute sequence-level perplexity for a single prompt.

    Uses the model's cross-entropy loss (mean negative log-likelihood
    per token) to compute perplexity:
        PPL = exp(loss)

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text string.
        device: Device for computation.

    Returns:
        ppl: Sequence-level perplexity (scalar).
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss  # mean cross-entropy per token

    ppl = torch.exp(loss).item()
    return ppl


def compute_ppl_var(
    model,
    tokenizer,
    paraphrases: List[str],
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute PPL_var: standard deviation of sequence-level perplexity
    across K paraphrase variants.

    PPL_var = std({PPL(prompt_k) for k=1..K})

    This measures how much the model's overall confidence varies when
    the same intent is expressed with different surface forms.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        paraphrases: List of K prompt strings.
        device: Device for computation.

    Returns:
        result: Dict with:
            - 'ppl_var': standard deviation of PPL across K variants
            - 'ppl_values': list of individual PPL values
            - 'ppl_mean': mean PPL across variants
    """
    ppl_values = []
    for k, prompt in enumerate(paraphrases):
        ppl = compute_sequence_perplexity(model, tokenizer, prompt, device)
        ppl_values.append(ppl)
        logger.debug(f"Paraphrase {k+1}/{len(paraphrases)}: PPL = {ppl:.3f}")

    ppl_tensor = torch.tensor(ppl_values)
    ppl_var = ppl_tensor.std().item()
    ppl_mean = ppl_tensor.mean().item()

    logger.info(
        f"PPL_var: std={ppl_var:.4f}, mean={ppl_mean:.3f}, "
        f"values={[f'{v:.2f}' for v in ppl_values]}"
    )

    return {
        'ppl_var': ppl_var,
        'ppl_values': ppl_values,
        'ppl_mean': ppl_mean
    }


def compute_per_token_entropy_and_branching(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 1.0,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token entropy H and branching factor B = exp(H) for
    a single prompt.

    Uses temperature=1 softmax to get calibrated probabilities, then
    computes entropy via torch.distributions.Categorical.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text string.
        temperature: Temperature for softmax. Must be 1.0 for meaningful
            branching factor (B → 1 trivially at temperature → 0).
        device: Device for computation.

    Returns:
        entropy: Tensor of shape (seq_len - 1,). Per-token entropy.
        branching_factor: Tensor of shape (seq_len - 1,). B = exp(H).
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Remove batch dimension and exclude last position (no next token)
    logits = logits.squeeze(0)[:-1]  # (seq_len-1, vocab_size)

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Compute entropy using Categorical distribution
    # Categorical expects logits (unnormalized log-probs)
    dist = torch.distributions.Categorical(logits=scaled_logits.float())
    entropy = dist.entropy()  # (seq_len-1,)

    # Branching factor B = exp(H)
    branching_factor = torch.exp(entropy)

    return entropy, branching_factor


def compute_pc_stab(
    model,
    tokenizer,
    paraphrases: List[str],
    temperature: float = 1.0,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute PC_stab: mean variance of branching factor B = exp(H)
    across K paraphrases, averaged over generation timesteps.

    For each paraphrase k and each token position t:
        B_k(t) = exp(H(p(·|context_t; prompt_k)))

    PC_stab = mean_t[Var_k[B_k(t)]]

    This measures how much the model's per-token uncertainty varies
    across prompt paraphrases, averaged over the generation process.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        paraphrases: List of K prompt strings.
        temperature: Temperature for softmax (must be 1.0).
        device: Device for computation.

    Returns:
        result: Dict with:
            - 'pc_stab': mean variance of branching factor across variants
            - 'mean_branching_factors': list of mean B per variant
            - 'min_seq_len': minimum sequence length used for alignment
    """
    K = len(paraphrases)
    branching_all = []
    seq_lengths = []

    for k, prompt in enumerate(paraphrases):
        _, bf = compute_per_token_entropy_and_branching(
            model, tokenizer, prompt, temperature, device
        )
        branching_all.append(bf)
        seq_lengths.append(len(bf))
        logger.debug(
            f"Paraphrase {k+1}/{K}: mean B = {bf.mean().item():.3f}, "
            f"seq_len = {len(bf)}"
        )

    # Align to minimum sequence length
    min_len = min(seq_lengths)
    bf_stack = torch.stack(
        [bf[:min_len].float() for bf in branching_all]
    )  # (K, min_len)

    # Var_k[B_k(t)] for each position t, then mean over positions
    var_per_position = bf_stack.var(dim=0)  # (min_len,)
    pc_stab = var_per_position.mean().item()

    mean_bfs = [bf.mean().item() for bf in branching_all]

    logger.info(
        f"PC_stab: {pc_stab:.6f}, mean B per variant: "
        f"{[f'{v:.2f}' for v in mean_bfs]}, min_seq_len={min_len}"
    )

    return {
        'pc_stab': pc_stab,
        'mean_branching_factors': mean_bfs,
        'min_seq_len': min_len
    }


def compute_ifi_metrics(
    model,
    tokenizer,
    paraphrases: List[str],
    temperature: float = 1.0,
    device: Optional[str] = None
) -> Dict:
    """
    Compute both IFI metrics (PPL_var and PC_stab) in one call.

    This is the main entry point for IFI metric computation. It combines
    both metrics for efficiency.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        paraphrases: List of K prompt strings.
        temperature: Temperature for PC_stab computation.
        device: Device for computation.

    Returns:
        result: Dict with all IFI metrics and sub-results.
    """
    logger.info(f"Computing IFI metrics for {len(paraphrases)} paraphrases...")

    ppl_result = compute_ppl_var(model, tokenizer, paraphrases, device)
    pc_result = compute_pc_stab(
        model, tokenizer, paraphrases, temperature, device
    )

    return {
        'ppl_var': ppl_result['ppl_var'],
        'ppl_mean': ppl_result['ppl_mean'],
        'ppl_values': ppl_result['ppl_values'],
        'pc_stab': pc_result['pc_stab'],
        'mean_branching_factors': pc_result['mean_branching_factors'],
    }
