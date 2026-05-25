# Technical Analysis: PRI Evaluation Framework & Layer-wise Mitigation
## Novelty Check · Feasibility Assessment · Refined Methodology
### Continuation of: "Adaptive Prompt Sensitivity in Complex LLM Tasks"

---

## EXECUTIVE SUMMARY

| | Method 1 (PRI Framework) | Method 2 (Layer-wise Mitigation) |
|---|---|---|
| **Novelty verdict** | Partially novel — framework framing + 2×2 matrix + harmonic synthesis are new; individual metrics have partial precedents | Partially novel — the perplexity-as-proxy idea is new in this context; "preserve low-perplexity tokens" is mechanically undefined and needs refinement |
| **Feasibility verdict** | Mostly feasible; 2 of 6 sub-metrics need tightening (TRD, KPIG) | Feasible in principle; Step 3 needs the Logit Lens mechanism; Step 6 needs one of 3 concrete implementations |
| **Closest competitor** | POSIX (composite index) + ProSA (internal confidence) — but neither uses dual-pillar architecture or harmonic synthesis | Liu & Chu 2604.18389 (layer-wise ‖Δh‖ + oracle layer swap) — different diagnostic and different intervention |
| **Recommendation** | Keep Method 1 with 3 targeted refinements | Refine Method 2: replace "perplexity proxy" with Logit Lens + specify Step 6 as Paraphrase-Invariant Residual Clamping (PIRC) |

---

## TASK 1 — NOVELTY CHECK AGAINST LITERATURE REVIEW

### FOR METHOD 1: PRI Evaluation Framework

#### Q1. Has Wasserstein distance been used to compare LLM output distributions across prompt variants?

**Short answer: Not in any prompt-sensitivity paper found in the review.**

Wasserstein distance (Earth Mover's Distance) is used in:
- Text generation diversity evaluation (Fu et al. 2021, "A Theoretical Analysis of the Repetition Problem in Text Generation")
- Generative-model quality evaluation (FID and Wasserstein-based extensions for image generation)
- Domain-shift robustness in NLP (Sinha et al., "UnNatural Language Inference," ACL 2021)

But none of the papers surveyed — POSIX, ProSA, PromptBench, FormatSpread, PromptSET, PromptEval, Hua et al., Liu & Chu — use W1 or any Wasserstein variant to compare output embedding distributions across prompt paraphrases. **SMS as defined is novel.**

The most adjacent work is the within-model variability paper (arXiv 2601.21339), which measures variance of outputs across prompts for creative tasks, but uses standard statistical variance on embedding distances, not W1. The Hua et al. "Flaw or Artifact" paper (EMNLP 2025 Main) uses output similarity metrics but cosine similarity aggregates, not optimal transport.

**Practical note**: W1 between output embedding distributions requires either (a) a set of N outputs per prompt variant (expensive) or (b) treating the single output's token embeddings as an empirical measure (more tractable but changes the interpretation). The precise operationalization matters for novelty claims.

#### Q2. Has any paper proposed a dual-pillar (behavioral + internal) evaluation specifically for prompt sensitivity?

**No — this architecture is novel.**

- **POSIX** (Chatterjee 2410.02185) combines log-likelihood discrepancy (internal signal) and response diversity (output signal) but treats them as co-equal components in a flat weighted sum, not as two conceptually distinct pillars with diagnostic separation.
- **ProSA** (Zhuo 2410.12405) explicitly links decoding confidence to sensitivity but again treats it as a predictor/correlate, not as a second pillar with its own index.
- **PromptBench** and **FormatSpread** are purely output-behavioral — no internal metrics.
- **Hua et al.** "Flaw or Artifact" (EMNLP 2025) arguably makes the closest argument — they show that low ORI (output instability) can be an evaluation artifact rather than genuine sensitivity — which is exactly what the Low ORI + High IFI cell of the 2×2 matrix captures. But they do not formalize this as a dual-pillar index.

The dual-pillar architecture (ORI vs IFI as separate sub-scores) with explicit interpretive separation is new.

#### Q3. Has the 2×2 diagnosis matrix been used in any LLM evaluation paper?

**No published paper uses a 2×2 (ORI × IFI) causal diagnosis matrix for prompt sensitivity.**

The closest structural analogies are:
- Calibration vs accuracy 2×2 plots in model evaluation (Guo et al. "On Calibration of Modern Neural Networks," ICML 2017) — but not applied to sensitivity.
- Hua et al. "Flaw or Artifact" implicitly reasons about the four cells but never formalizes them as a grid.
- Reliability diagrams in confidence calibration literature.

**The 2×2 matrix is the most clearly novel contribution of Method 1.** It provides a diagnostic taxonomy that is directly actionable (each cell has a different implication for researchers and practitioners) and does not exist anywhere in the prompt-sensitivity literature.

#### Q4. Is PPL_var (perplexity variance across templates) used as a prompt sensitivity metric?

**Not as a named, standalone metric. Partially adjacent work exists.**

- ProSA (Zhuo 2410.12405) notes that "higher decoding confidence correlates with lower sensitivity" and uses sequence log-probability, but does not compute its variance across templates as a sensitivity metric.
- POSIX includes a "confidence variance" component but operationalizes it as variance of token-level log-likelihoods across response tokens for a single prompt, not as sequence-level perplexity variance across multiple prompt templates.
- Razavi et al. (2502.06065) do not use perplexity at all.

Defining σ_PPL as the standard deviation of sequence-level perplexity across K prompt templates for the same semantic intent is a concrete, novel operationalization. It is technically computable on any open-source model via HuggingFace's `model(**inputs).loss` which returns mean cross-entropy (= log perplexity per token).

**PPL_var as defined is novel as a named prompt-sensitivity metric.**

#### Q5. Is PC_stab (branching factor / token entropy stability) defined anywhere?

**Not as a prompt-sensitivity metric. The underlying quantity exists in language modeling theory.**

- Shannon entropy of the next-token distribution H(p(·|context)) is a standard quantity in LM evaluation.
- The "branching factor" B = exp(H) is occasionally mentioned in IR and information theory but is not standard in LM evaluation.
- Holtzman et al. ("The Curious Case of Neural Text Degeneration," ICLR 2020) analyze token probability distributions but for generation quality, not cross-prompt stability.
- Liu & Chu (2604.18389) analyze log π(y|h₀) (log-probability of the full response) but not per-token entropy.

Computing the variance of the branching factor B = exp(H(p_t)) across K prompt variants and averaging over generation timesteps t is not found in any paper reviewed. **PC_stab is novel.**

**Important caveat**: B = exp(H) is sensitive to temperature and sampling strategy. You should specify whether this is computed with greedy decoding (temperature → 0, in which case B → 1 always trivially) or with temperature = 1 (default softmax). The meaningful regime is temperature = 1 or a "calibrated" temperature set per model.

#### Q6. Has anyone combined output-level + internal confidence metrics into a composite score like PRI?

**Partially — POSIX is the closest, but the combination architecture differs.**

POSIX synthesizes: log-likelihood discrepancy, response diversity, semantic coherence, and confidence variance — using a weighted average. This is a flat combination of heterogeneous signals including both internal (log-likelihood, confidence) and external (diversity, coherence) components.

PRI differs in three important ways:
1. It separates components into two explicitly named sub-indices (ORI / IFI) before combining.
2. It uses a **weighted harmonic mean** rather than a weighted arithmetic mean — this penalizes catastrophic failure in any single dimension, which a simple average does not.
3. The 2×2 matrix from the sub-index scores creates a diagnostic layer that POSIX does not have.

**The synthesis architecture (harmonic mean of two sub-indices, diagnostic matrix) is novel even if individual metrics have partial precedents.**

#### Q7. VERDICT — Is Method 1 Novel?

| Component | Novelty | Precedent |
|---|---|---|
| SMS (W1 on output embedding distributions) | **Yes** | W1 used elsewhere in NLP, not here |
| AUC-E (performance elasticity AUC) | **Partial** — the AUC-over-perturbation curve idea exists in adversarial robustness (PromptBench uses accuracy-vs-attack curves) | PromptBench robustness curves |
| TRD (JS-divergence between topic distributions) | **Partial** — Manakul et al. 2406.03993 uses topic coherence for summarization sensitivity, but not JS-divergence between topic clusters | Hallucination/faithfulness papers |
| KPIG (Information Gain variance on critical spans) | **Largely novel** — measuring sensitivity on sub-spans is new | No direct precedent found |
| PPL_var | **Yes** | ProSA uses confidence but not σ_PPL across templates |
| PC_stab | **Yes** | Branching factor used in IR, not here |
| ORI/IFI dual-pillar architecture | **Yes** | Flat combinations exist; two-pillar architecture does not |
| 2×2 Diagnosis Matrix | **Yes** — most clearly novel single contribution | No precedent in prompt sensitivity |
| Weighted Harmonic Mean synthesis | **Partial** — WHM used in IR (F1) and NLP composites; novel application here | F-score, composite IR metrics |

**Overall: Method 1 is substantially novel.** The 2×2 matrix is the strongest claim. The dual-pillar architecture with harmonic synthesis is the second. Individual metrics have varying degrees of precedent but the combination does not exist.

---

### FOR METHOD 2: Layer-wise Mitigation

#### Q1. Has any paper specifically identified "sensitive layers" for prompt sensitivity?

**Yes — Liu & Chu (2604.18389) is the direct competitor, but with a different diagnostic.**

Liu & Chu define "sensitive layers" as those where ‖Δh^(ℓ)‖ (L2 norm of hidden-state difference between two prompt variants) is largest, and show via first-order Taylor expansion that this norm grows monotonically toward deeper layers in transformer LMs. Their identification method is **vector norm of activation difference**, not perplexity.

The proposed Method 2 uses **perplexity (via the forward pass) as a proxy** for confidence divergence across layers. This is a different diagnostic signal — more intuitive but less precisely defined without specifying how per-layer perplexity is computed (this requires the Logit Lens mechanism — see Q5).

**Sensitivity-Positional Co-Localization** (arXiv 2604.07766) identifies sensitive layers via a **correctness-differential hidden-state metric** δ_ℓ = cosine_distance(h^(ℓ)[correct], h^(ℓ)[incorrect]) — again, a different signal from perplexity.

The perplexity-based layer-sensitivity identification is **novel in mechanism** even if the goal is shared.

#### Q2. Has hidden state anchoring or token preservation at specific layers been used to reduce output variance?

**Not for prompt-paraphrase sensitivity. Closest work is mechanistically different.**

- **ITI** (Li et al. 2306.03341) shifts activations along a *learned direction* (truthfulness direction) at specific attention heads. It does not "anchor" specific tokens; it shifts the full activation vector.
- **CAA** (Rimsky et al. ACL 2024) adds a mean-difference steering vector at a chosen layer. Again, a full-vector shift, not token-level anchoring.
- **Liu & Chu Appendix G** performs an oracle layer-swap: h_A^(ℓ) ← h_B^(ℓ) (replace one prompt's hidden state with another's at a chosen layer). This is the closest mechanistic analog — but it (a) swaps full hidden states, not token-specific, (b) is oracle (requires access to the "correct" variant's hidden state), and (c) is only tested on MCQ.

The concept of identifying high-confidence (low-perplexity) **tokens** at sensitive layers and specifically anchoring those token positions (rather than the full hidden state) is **novel in design**, though underspecified in its current form (see Feasibility, Q2).

#### Q3. Is there a training-free method that intervenes at layer level specifically for prompt sensitivity?

**Liu & Chu Appendix G is the only paper with this exact combination, but it uses an oracle intervention on MCQ only.**

All other training-free layer-level interventions (ITI, RepE, CAA, ActAdd, CAST, Angular Steering) are designed for attribute control (truthfulness, harmlessness, style, bias) — not for prompt-paraphrase robustness. None of them is framed as a prompt sensitivity mitigation.

CREME (2507.16407) does localize sensitivity to specific layers and intervenes for robustness, but it is training-based (weight editing of W_V) and code-specific.

#### Q4. Do ITI or RepE apply their methods to prompt sensitivity reduction?

**No.** Both are explicitly framed as attribute-steering methods:
- ITI targets the "truthful" direction defined by a linear probe trained on TruthfulQA.
- RepE targets concepts like "honesty," "power-seeking," "race bias."
- CAA steers between behaviorally defined contrast pairs (e.g., sycophantic vs non-sycophantic responses).

None of them frames the steering target as "the direction that makes a model's response invariant to paraphrase of the same intent." The closest conceptual bridge is: define the target direction as the *paraphrase-invariance* direction (mean Δh between paraphrase pairs should be zero after steering) — but this has not been implemented or published.

#### Q5. Has layer-by-layer perplexity been used to locate sensitivity onset?

**Not in the prompt-sensitivity literature. The enabling mechanism (Logit Lens / TunedLens) exists.**

- **Logit Lens** (nostalgebraist, 2020 — blog post; formalized in Belrose et al. "Eliciting Latent Predictions from Transformers with the Tuned Lens," arXiv 2303.08112, ICML 2023) projects intermediate hidden states h^(ℓ) through the unembedding matrix U to get a token probability distribution at each layer: p^(ℓ)(w) ∝ exp(U · h^(ℓ)). From this, one can compute a per-layer perplexity.
- Skean et al. (2502.02013) uses information-theoretic analysis of layer-wise representations but not perplexity specifically.
- Sun et al. (2602.04297) probes hidden states across layers to study prompt underspecification, but uses probing classifiers, not perplexity.

**Using Logit Lens-derived per-layer perplexity as a sensitivity onset detector across prompt paraphrases is novel.** The TunedLens paper is not in the prompt-sensitivity literature at all.

#### Q6. Is "preserving low-perplexity tokens" mechanically defined anywhere?

**No — this is the main underspecified gap in Method 2.**

The concept has no precedent in the literature as described. The challenge is that "preserving" a token's influence during the forward pass has no unique implementation — there are at least three plausible meanings, each with different mathematical and computational properties (see Feasibility Q2 below for the full breakdown).

This underspecification is a weakness but also an opportunity: the specific implementation chosen becomes part of the novel contribution.

#### Q7. VERDICT — Is Method 2 Novel?

**The overall pipeline is novel; the core diagnostic (Step 3) needs mechanical grounding via Logit Lens; Step 6 needs one of three defined implementations.**

| Component | Novelty | Issue |
|---|---|---|
| Two-prompt forward pass to extract hidden states (Step 2) | Existing (Liu & Chu, ITI) | Fine as setup |
| Per-layer perplexity as sensitivity proxy (Step 3) | **Novel in context** | Must specify Logit Lens mechanism |
| Sensitive layer detection by divergence in per-layer PPL (Step 4) | **Novel in mechanism** | Liu & Chu uses ‖Δh‖ instead |
| "Anchor token" concept from high-confidence positions (Step 5) | **Novel** | No precedent |
| Token-selective residual anchoring to reduce sensitivity (Step 6) | **Novel** | Mechanically underspecified — needs one of 3 implementations |
| Training-free, generative task focus | **Cleanly differentiates from all existing work** | Core novelty claim |

**Closest competitor: Liu & Chu (2604.18389).** Distance: they do steps 2 and 4 differently (‖Δh‖ not PPL), don't do step 5 at all, and their step 6 analog is oracle-only and MCQ-only. The proposed method is differentiated on: (a) diagnostic signal (PPL-based via logit lens vs norm-based), (b) token-selective anchoring vs full-vector swap, (c) generative-task application, (d) deployable non-oracle operation.

---

## TASK 2 — FEASIBILITY ASSESSMENT

### FOR METHOD 1: PRI Evaluation Framework

#### Q1. Are all 6 sub-metrics technically computable on open-source models?

**Yes, all 6 are computable on HuggingFace models, but with varying implementation effort.**

Quick feasibility table:

| Metric | Computable | Effort | Access needed |
|---|---|---|---|
| SMS (W1, output embeddings) | ✅ | Medium | Sentence embedding model (e.g. SBERT) + scipy.stats.wasserstein_distance |
| AUC-E (performance-vs-perturbation AUC) | ✅ | Medium | Reference metric (ROUGE/BERTScore) + parametric perturbation control |
| TRD (JS between topic distributions) | ⚠️ Needs scoping | High | BERTopic or LDA; "intended topics" definition is circular |
| KPIG (Information Gain variance on critical spans) | ⚠️ Needs scoping | High | Requires span identification + conditional generation |
| PPL_var | ✅ | Low | model(**inputs).loss |
| PC_stab | ✅ | Low | torch.distributions.Categorical(logits=logits).entropy() |

#### Q2. Which metrics are hardest to implement and why?

**TRD is the hardest.** The fundamental problem is defining "intended topics" independently of the model's output. If you use BERTopic/LDA on the reference document or gold summary, you get a ground-truth topic distribution p_ref. If you use it on the generated response, you get p_gen. D_JS(p_ref, p_gen) is a faithfulness measure — standard in summarization evaluation — but using it *across two prompt variants' outputs* as a sensitivity measure requires p_gen₁ and p_gen₂, where the reference is the same document. This is feasible but the "intended topics" framing is ambiguous: are intended topics from the source document or from the prompt text itself?

**Recommendation**: Replace "intended topics" with "source-document topics" (BERTopic on the input document, giving p_src), and measure:
```
TRD = D_JS(p_gen₁ || p_src) - D_JS(p_gen₂ || p_src)
```
This quantifies the differential faithfulness drift across prompt variants and is well-defined.

**KPIG is the second hardest.** "Information Gain on critical instruction spans" is not a standard NLP primitive. The most natural interpretation: extract spans from the prompt that contain explicit constraints (e.g., "in 100 words," "using bullet points"), then compute how well the model's output satisfies those constraints (binary or graded). Variance of this satisfaction score across 10 paraphrased prompts = KPIG.

**Recommendation**: Implement KPIG as constraint satisfaction variance. Identify constraint spans via regex or a small NER model trained on instruction keywords; score compliance (word count within 10% of target, presence of requested format, etc.); compute variance across K paraphrases.

#### Q3. Computational costs per metric

| Metric | Cost model | Per-sample cost (rough) |
|---|---|---|
| SMS | K outputs × SBERT encode + W1 solve (1D or 1-D projected) | O(K·L·d) for K outputs, L tokens, d embed dim; W1 in O(K log K) | ~1–2s/sample on CPU with SBERT |
| AUC-E | K perturbation levels × 1 forward pass each + reference metric | Dominant cost is K×LLM inference; reference metric cheap | K × inference time |
| TRD | K outputs × BERTopic cluster + JS compute | BERTopic is O(N²) at fit time; amortize across samples | BERTopic fit: minutes; per-sample: <0.1s after fit |
| KPIG | K outputs × constraint satisfaction score | Regex-based: negligible; LLM-judge: K × inference | Regex: near-zero; LLM-judge: K × inference |
| PPL_var | K forward passes | model.loss already computed during generation | Near zero marginal if K outputs already generated |
| PC_stab | K forward passes, one entropy computation per token per pass | Same as PPL_var | Near zero marginal |

**Key insight**: SMS and AUC-E are the expensive metrics; PPL_var and PC_stab are essentially free given that you already ran K forward passes to generate the K outputs. This means IFI (Pillar 2) is much cheaper than ORI (Pillar 1) per sample.

#### Q4. Is the Weighted Harmonic Mean a valid synthesis approach?

**Yes — both mathematically sound and precedented in NLP.**

The harmonic mean is the appropriate aggregator when you want to penalize catastrophic failure in any single component. In NLP:
- F1 score = harmonic mean of precision and recall (MachineLearning literature)
- mHAP (Multi-dimensional Human Annotation Protocol) uses harmonic means for multi-criterion scoring

For PRI, the harmonic mean property ensures: if PPL_var → ∞ (catastrophic internal brittleness), then PRI → 0 even if all ORI metrics are high — which correctly identifies the "Stochastic Luck" cell. This is mathematically superior to arithmetic averaging for this diagnostic purpose.

**One issue**: harmonic mean is only defined for positive values. All 6 metrics must be normalized to (0, 1] before synthesis. Specify normalization functions for each (e.g., σ_PPL uses a soft-max normalization: IFI_PPL = exp(-α · σ_PPL) for some α calibrated on the dataset).

---

### FOR METHOD 2: Layer-wise Mitigation

#### Q1. Is extracting layer-wise perplexity feasible using HuggingFace forward hooks?

**Yes, but requires the Logit Lens mechanism — forward hooks alone are insufficient.**

A standard HuggingFace forward hook extracts the hidden state tensor h^(ℓ) ∈ ℝ^(seq_len × d_model) at layer ℓ. To convert this to a token probability distribution (and hence perplexity), you need to apply the model's unembedding matrix U ∈ ℝ^(vocab_size × d_model) and the layer norm:

```python
# Logit Lens at layer ℓ
def logit_lens(h_ell, model):
    # Apply final layer norm (LN_f) from the model
    h_normed = model.model.norm(h_ell)  # Llama-style; model.transformer.ln_f for GPT-2
    # Project through unembedding matrix
    logits = model.lm_head(h_normed)   # shape: (seq_len, vocab_size)
    return logits

# Per-layer perplexity for token sequence y = [y_1, ..., y_T]
def layer_perplexity(logits, labels):
    log_probs = F.log_softmax(logits[:-1], dim=-1)  # exclude last position
    token_log_probs = log_probs.gather(-1, labels[1:].unsqueeze(-1)).squeeze(-1)
    return torch.exp(-token_log_probs.mean())
```

**Note on TunedLens (Belrose et al. 2303.08112, ICML 2023)**: TunedLens fits a small affine transformation per layer to improve the logit-lens accuracy. For Method 2, the raw Logit Lens suffices for *sensitivity detection* (you only need to detect divergence, not absolute accuracy). TunedLens would give more accurate per-layer distributions if you need them.

**Complete hook-based extraction**:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

hidden_states = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output[0] is the hidden state tensor for decoder layers
        hidden_states[layer_idx] = output[0].detach()
    return hook

# Register hooks on all decoder layers
hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

# Forward pass
with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])

# hidden_states[i] is now h^(i) for i = 0..N_layers-1
# Remove hooks
for h in hooks: h.remove()
```

This is fully feasible and standard practice in mechanistic interpretability.

#### Q2. What does "preserve low-perplexity tokens" mean mechanically — 3 most realistic implementations?

This is the most critical underspecification in Method 2. Here are the three well-defined implementations, ordered by elegance and novelty:

---

**Implementation A: Paraphrase-Invariant Residual Clamping (PIRC)** ← Recommended

*What it does*: At the identified sensitive layer ℓ*, for each anchor token position i (those with logit-lens perplexity below threshold τ), replace the hidden state of prompt variant A with the mean hidden state across all K paraphrase variants:

```
h_A^(ℓ*)[i] ← (1/K) Σ_k h_k^(ℓ*)[i]   for all anchor positions i
```

where anchor positions i are identified as tokens t_i where PPL^(ℓ*)(t_i; prompt_A) < τ (the low-perplexity / high-confidence tokens in the sensitive layer).

*Why this is the best choice*:
- It is non-oracle: the mean is computed across K generated paraphrases, all of which are available at inference time after Step 1.
- It directly implements "preserve" semantically: the clamped position carries the consensus representation, not any single variant's idiosyncratic hidden state.
- It is differentiable and invertible — you can compute an exact correction vector: Δh_i = h_mean^(ℓ*)[i] - h_A^(ℓ*)[i], and add it back.
- Closely related to Liu & Chu's oracle swap but uses mean(K variants) instead of a single oracle variant — this is the key non-oracle upgrade.

*Implementation*:

```python
def pirc_forward(model, prompts, sensitive_layer, anchor_threshold=2.0):
    """
    prompts: list of K paraphrase strings (same intent)
    sensitive_layer: int, the identified sensitive layer ℓ*
    anchor_threshold: PPL threshold below which a token is an anchor
    """
    K = len(prompts)
    all_hidden = []  # shape: [K, seq_len, d_model]
    all_ppl = []     # shape: [K, seq_len]

    for prompt in prompts:
        h, ppl = extract_layer_hidden_and_ppl(model, prompt, sensitive_layer)
        all_hidden.append(h)
        all_ppl.append(ppl)

    all_hidden = torch.stack(all_hidden)   # [K, seq_len, d_model]
    all_ppl = torch.stack(all_ppl)         # [K, seq_len]

    # Anchor mask: positions with low PPL (high confidence) in majority of variants
    anchor_mask = (all_ppl < anchor_threshold).float().mean(0) > 0.5  # [seq_len]

    # Mean hidden state at sensitive layer across variants
    mean_hidden = all_hidden.mean(0)   # [seq_len, d_model]

    # For each variant, clamp anchor positions to mean
    stabilized_hidden = all_hidden.clone()
    stabilized_hidden[:, anchor_mask, :] = mean_hidden[anchor_mask, :]

    return stabilized_hidden  # Inject back via hooks for continued decoding
```

---

**Implementation B: Selective Attention Masking (SAM)**

*What it does*: At sensitive layer ℓ*, modify the attention weight matrix A^(ℓ*) to increase the attention weight from all query positions to anchor token positions (the high-confidence tokens), effectively forcing the model to "read" the most stable representations more heavily:

```
Ã^(ℓ*)[q, i] = A^(ℓ*)[q, i] + α · anchor_mask[i]   (renormalized)
```

*Pros*: Operates on the attention mechanism itself; softer than clamping; preserves the residual stream structure.
*Cons*: Modifying attention matrices mid-forward-pass requires custom attention implementations or patching the scaled_dot_product_attention call; more brittle in practice; the α hyperparameter is hard to set.

---

**Implementation C: Paraphrase-Contrastive Direction Subtraction (PCDS)**

*What it does*: Treat the difference between two prompt variants' hidden states at ℓ* as a "sensitivity direction" d = h_A^(ℓ*) - h_B^(ℓ*). Project this direction out of the hidden states of all subsequent variants:

```
h_A^(ℓ*) ← h_A^(ℓ*) - (h_A^(ℓ*) · d̂) · d̂   where d̂ = d / ‖d‖
```

*Pros*: Directly cancels the inter-variant divergence direction; analogous to CAA/RepE steering in its mathematical form.
*Cons*: Requires a reference pair (A, B) to define d; with K > 2 variants, requires averaging over multiple d vectors; projection may remove semantically meaningful variation.

---

**Recommendation**: Implement A (PIRC) as the primary method, B and C as ablations. The comparison between these three implementations across generative tasks is itself a contribution.

#### Q3. Does running 2+ prompts at inference (to find sensitive layers) create a deployment bottleneck?

**Yes, this is a real bottleneck. Here is a tiered analysis:**

| Deployment scenario | Cost | Severity | Mitigation |
|---|---|---|---|
| Research / offline evaluation | Run K=5 paraphrases, identify ℓ* per task | K× inference cost | Acceptable |
| Online inference with pre-known sensitive layer | Run only 2 prompts (original + 1 paraphrase) to compute PIRC | 2× cost | Manageable |
| Real-time user-facing deployment | Need ℓ* in advance | 1× cost if ℓ* is pre-cached | Best option |

**Practical mitigation**: In practice, ℓ* can be calibrated offline on a representative sample of the target task (e.g., 100 summarization examples × 5 paraphrases each) and then **fixed** at deployment time. This is analogous to how ITI probes are pre-trained and fixed. For deployment:
1. Offline phase: identify ℓ* for each task class (summarization, code, dialogue).
2. Online phase: run K=2 paraphrases (original + 1 fast paraphrase from a small model like Phi-3-mini), clamp anchor positions at pre-cached ℓ*, decode.
3. Cost: 2× inference per query (same as Liu & Chu's setup).

For the first experiment, cost is not a concern — run offline with K=5.

#### Q4. Top 3 technical risks

1. **Logit Lens perplexity is noisy in early layers**: In early transformer layers, h^(ℓ) has not yet been processed by the attention mechanism sufficiently to produce meaningful token distributions via logit lens. The per-layer PPL signal may be flat/uninformative for ℓ < L/3. This means sensitive-layer detection may fail to differentiate early layers. **Mitigation**: Only scan layers ℓ ∈ [L/4, L] (the informative range per Skean et al. and Liu & Chu).

2. **The "low-perplexity tokens" at a sensitive layer may not be semantically meaningful anchor tokens**: If the high-confidence tokens at ℓ* are function words (the, a, is) rather than content words (summarize, story, code), clamping their hidden states adds little sensitivity-reducing signal. **Mitigation**: Filter anchor tokens to content words only (POS tagging); or use BPE token importance scores (token log-prob × IDF) as the anchor criterion instead of raw logit-lens perplexity.

3. **PIRC requires the K paraphrases to be generated before decoding**: Generating K paraphrases with a small LM (Phi-3-mini) before the main LM forward pass adds latency. If the paraphrase model is too dissimilar in phrasing, the mean hidden state at ℓ* may not represent the "true intent" well. **Mitigation**: Generate paraphrases with T5-paraphrase or a simple back-translation pipeline; validate that paraphrases are semantically similar (cosine sim > 0.85 in SBERT space) before including.

#### Q5. Which open-source models are best suited?

| Model | Reason to use | Caveat |
|---|---|---|
| **LLaMA-3-8B-Instruct** | Most-studied in mechanistic interpretability; large community; good hook support in HuggingFace | May require approval token |
| **Mistral-7B-Instruct-v0.3** | Good instruction following; Apache 2.0 license; widely tested with RepE and CAA | Slightly less characterized per-layer than LLaMA-3 |
| **Phi-3-mini-4k-instruct (3.8B)** | Small enough to run multiple forward passes cheaply; Microsoft-backed; well-characterized | Architectural quirks (shared embeddings) may require adaptation |
| **Qwen2.5-7B-Instruct** | Liu & Chu used Qwen1.5; directly replicates their setup for comparison; strong multilingual | Less community interpretability work |
| **Gemma-2-9B-it** | Strong performance; DeepMind interpretability team's primary model | Less mechanistic interpretability tooling |

**Best starting choice for the first experiment**: **LLaMA-3-8B-Instruct** — maximum comparability to prior work, best tooling support (TransformerLens, nnsight, baukit), and directly comparable to Liu & Chu results.

---

## TASK 3 — REFINED METHODOLOGY (Where Original Methods Need Updating)

Both methods are novel enough to keep. What follows are the targeted refinements needed to make them fully specified and defensible.

### METHOD 1 Refinements

**Refinement 1: Operationalize SMS precisely.**

Original: "1-Wasserstein distance between output embedding distributions."

Problem: A single model output is a *string*, not a distribution. Clarify which distribution is being compared:

Option A (Distributional over samples): Generate N=20 outputs per prompt variant using temperature sampling. Embed each with SBERT. Compute the empirical measure μ_A = {e₁,...,e_N} and μ_B similarly. Compute W1(μ_A, μ_B) via linear programming or the 1D Wasserstein formula after projecting onto the top principal component. **This is the most rigorous interpretation.**

Option B (Token-level distribution): Treat the sequence of token embeddings within a single generated output as an empirical measure. Faster but less meaningful for prompt sensitivity (it measures output diversity at the token level, not at the response level).

**Recommendation**: Use Option A with N=10 samples (greedy would give W1=0 trivially). Specify SBERT model (e.g., all-mpnet-base-v2, 768-dim).

**Refinement 2: Tighten TRD.**

Replace "intended topics" with "source-document topics." Define:
```
TRD(variant_A, variant_B) = |D_JS(p_topic(output_A) || p_topic(src)) 
                              - D_JS(p_topic(output_B) || p_topic(src))|
```
where p_topic is the topic distribution from BERTopic (or LDA K=10) fitted on the source document. This makes TRD directly a faithfulness-drift metric and avoids the circular "intended topics" problem.

**Refinement 3: Operationalize KPIG.**

Replace with a cleaner formulation: **Instruction Constraint Variance (ICV)** — for each of the K paraphrased prompts, extract explicit constraints (word count, format, language, perspective) using a small regex+rule system or a classification model. Score compliance for each constraint c_j ∈ {0, 1} in the model output. Define:

```
KPIG_j = Var_k[compliance(output_k, constraint_j)]   across K paraphrases
KPIG = mean_j[KPIG_j]
```

**Normalization**: All 6 metrics must be mapped to [0, 1] before harmonic mean synthesis. Specify the mapping for each (e.g., KPIG → 1 - tanh(α · KPIG) for sensitivity, so high variance → low score → pulls PRI down).

---

### METHOD 2 Refinements

**The refined pipeline: Logit Lens Sensitivity Detection + Paraphrase-Invariant Residual Clamping (LL-PIRC)**

Step 1 (unchanged): Create K=5 paraphrase variants of the same prompt.

Step 2 (unchanged): Run all K through the LLM with forward hooks extracting h^(ℓ) at all layers.

Step 3 (UPDATED — replace "perplexity as proxy" with Logit Lens):
- At each layer ℓ, apply the Logit Lens: compute logits^(ℓ) = LN_f(h^(ℓ)) @ U^T.
- Compute per-token log-probabilities and sequence-level PPL^(ℓ) for each variant k.
- Define the layer-wise sensitivity signal: S(ℓ) = Var_k[PPL^(ℓ)(output_k)]   (variance of logit-lens PPL across K variants at layer ℓ).

Step 4 (UPDATED — formalize sensitive layer identification):
- Plot S(ℓ) vs ℓ. The "sensitive layer onset" is ℓ* = argmax_ℓ [ΔS(ℓ)] = the layer where the variance curve inflects upward most sharply.
- Alternatively, identify ℓ* as the first ℓ where S(ℓ) > μ + 2σ (where μ, σ are computed over all layers) — a z-score threshold.

Step 5 (UPDATED — anchor token identification):
- At layer ℓ*, compute per-token logit-lens PPL^(ℓ*)(t_i; variant_k) for each token position i and variant k.
- Define anchor tokens: positions i where mean_k[PPL^(ℓ*)(t_i; variant_k)] < τ AND Var_k[PPL^(ℓ*)(t_i; variant_k)] < τ_var.
- This identifies tokens that are both confident AND consistently confident across variants — the most stable anchors.

Step 6 (UPDATED — PIRC):
```python
# For a new test prompt at inference time:
# 1. Generate K=3-5 paraphrases using a fast paraphrase LM
# 2. Run all K through LLM up to layer ℓ* (with hooks)
# 3. Compute mean hidden state at ℓ* across K variants
mean_h = torch.stack([hidden[ell_star] for hidden in all_hidden]).mean(0)
# 4. For anchor positions i, replace h^(ℓ*) with mean_h[i]
for k in range(K):
    all_hidden[k][ell_star][anchor_mask] = mean_h[anchor_mask]
# 5. Continue forward pass from ℓ* with stabilized hidden states
# 6. Decode — the output is now anchored to the consensus representation
```

---

## TASK 4 — FINAL RECOMMENDED METHODOLOGY

### 1. FINAL EVALUATION FRAMEWORK

**Adopt Method 1 with the three refinements above (SMS Option A, TRD redefined on source-document topics, KPIG as Instruction Constraint Variance).**

**What makes it novel:**
- First dual-pillar (ORI + IFI) evaluation framework for prompt sensitivity — no existing benchmark separates behavioral and internal signals into two sub-indices.
- 2×2 causal diagnosis matrix is unique and actionable — directly maps ORI/IFI combinations to researcher-interpretable conditions (True Robustness, Evaluation Artifact, Stochastic Luck, Knowledge Boundary).
- W1 (Wasserstein) distance for output embedding distributions replaces cosine-similarity aggregates used by POSIX/ProSA, providing a geometrically grounded sensitivity measure.
- Weighted harmonic mean synthesis penalizes catastrophic single-dimension failure, unlike POSIX's arithmetic mean.
- PPL_var and PC_stab are novel named primitives for the internal pillar.

**Metrics to prioritize for first experiment:**
Implement IFI (PPL_var + PC_stab) first — they are near-zero marginal cost given existing forward passes and provide immediate differentiation from POSIX. Then add SMS (requires N=10 samples per prompt variant but is straightforward with SBERT). TRD and KPIG can follow in a second experiment.

**PRI formula (finalized):**

```
ORI = HM(norm(SMS), norm(AUC-E), norm(TRD), norm(KPIG))
IFI = HM(norm(PPL_var), norm(PC_stab))
PRI = HM(ORI, IFI)   [or weighted: PRI = HM(w_ORI · ORI, w_IFI · IFI)]
```

where HM denotes harmonic mean and norm(·) maps each metric to [0,1] via a monotone calibrated transform.

---

### 2. FINAL MITIGATION METHOD

**Adopt Method 2 refined as LL-PIRC (Logit Lens Sensitivity Detection + Paraphrase-Invariant Residual Clamping).**

**What makes it novel:**
1. **First training-free mitigation for prompt sensitivity in generative tasks** — Liu & Chu's Appendix G is oracle-only and MCQ-only; this is deployable and generative.
2. **Logit Lens as the sensitivity onset detector** — using variance of per-layer logit-lens perplexity across paraphrases (S(ℓ)) to locate ℓ* is new in the prompt sensitivity context.
3. **Token-selective residual clamping to consensus** — rather than swapping full hidden states (Liu & Chu oracle) or adding a full steering vector (CAA/ITI), this selectively anchors only the high-confidence token positions to the cross-paraphrase mean — mechanically less disruptive.
4. **Non-oracle**: uses the mean of K sampled paraphrases rather than any single "correct" variant.

**Exact HuggingFace implementation steps:**

```python
# ============================================================
# LL-PIRC: Full Implementation
# ============================================================

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 0. Load model ---
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# --- 1. Logit Lens PPL at layer ell ---
def logit_lens_ppl(hidden, model, labels):
    """hidden: (seq_len, d_model); labels: (seq_len,) token IDs"""
    h_normed = model.model.norm(hidden)             # final layer norm
    logits = model.lm_head(h_normed)               # (seq_len, vocab)
    log_p = F.log_softmax(logits[:-1], dim=-1)     # (seq_len-1, vocab)
    token_lp = log_p.gather(-1, labels[1:].unsqueeze(-1)).squeeze(-1)  # (seq_len-1,)
    ppl_per_token = torch.exp(-token_lp)           # (seq_len-1,)
    return ppl_per_token                            # per-token logit-lens PPL

# --- 2. Extract hidden states at all layers for one prompt ---
def extract_all_hidden(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hidden_by_layer = {}
    hooks = []
    def make_hook(i):
        def hook(module, input, output):
            hidden_by_layer[i] = output[0].detach().float()
        return hook
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model(**inputs)
    for h in hooks: h.remove()
    return hidden_by_layer, inputs["input_ids"].squeeze(0)

# --- 3. Compute S(ell) = Var_k[PPL^ell] across K paraphrases ---
def find_sensitive_layer(model, tokenizer, paraphrases, device="cuda"):
    N_layers = model.config.num_hidden_layers
    all_ppl_by_layer = {ell: [] for ell in range(N_layers)}
    for prompt in paraphrases:
        hidden_by_layer, labels = extract_all_hidden(model, tokenizer, prompt, device)
        for ell in range(N_layers):
            ppl_tok = logit_lens_ppl(hidden_by_layer[ell], model, labels)
            all_ppl_by_layer[ell].append(ppl_tok.mean().item())
    S = {}
    for ell in range(N_layers):
        vals = torch.tensor(all_ppl_by_layer[ell])
        S[ell] = vals.var().item()
    # Sensitive layer = highest inflection in S
    S_vals = torch.tensor([S[ell] for ell in range(N_layers)])
    ell_star = (S_vals[1:] - S_vals[:-1]).argmax().item() + 1  # inflection point
    return ell_star, S

# --- 4. Identify anchor token positions at ell* ---
def find_anchor_tokens(model, tokenizer, paraphrases, ell_star, ppl_threshold=3.0, var_threshold=1.0, device="cuda"):
    K = len(paraphrases)
    ppl_per_tok_all = []
    for prompt in paraphrases:
        hidden_by_layer, labels = extract_all_hidden(model, tokenizer, prompt, device)
        ppl_tok = logit_lens_ppl(hidden_by_layer[ell_star], model, labels)
        ppl_per_tok_all.append(ppl_tok)
    ppl_stack = torch.stack(ppl_per_tok_all)   # (K, seq_len-1)
    mean_ppl = ppl_stack.mean(0)
    var_ppl  = ppl_stack.var(0)
    anchor_mask = (mean_ppl < ppl_threshold) & (var_ppl < var_threshold)
    return anchor_mask

# --- 5. PIRC: run clamped forward pass ---
def pirc_generate(model, tokenizer, paraphrases, ell_star, anchor_mask, device="cuda", max_new_tokens=200):
    K = len(paraphrases)
    all_hidden_at_ell = []
    # Collect hidden states at ell_star for all paraphrases
    for prompt in paraphrases:
        hidden_by_layer, _ = extract_all_hidden(model, tokenizer, prompt, device)
        all_hidden_at_ell.append(hidden_by_layer[ell_star])
    all_hidden_at_ell = torch.stack(all_hidden_at_ell)   # (K, seq_len, d)
    mean_h = all_hidden_at_ell.mean(0)                   # (seq_len, d)

    # Now do a forward pass for the primary prompt (paraphrases[0])
    # with a hook that clamps anchor positions at ell_star
    def clamp_hook(module, input, output):
        h = output[0]     # (batch, seq_len, d_model)
        # Clamp anchor positions to mean across paraphrases
        h[:, anchor_mask, :] = mean_h[anchor_mask, :].unsqueeze(0).to(h.device)
        return (h,) + output[1:]

    primary_inputs = tokenizer(paraphrases[0], return_tensors="pt").to(device)
    target_layer = list(model.model.layers)[ell_star]
    hook = target_layer.register_forward_hook(clamp_hook)

    with torch.no_grad():
        output_ids = model.generate(
            **primary_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False    # greedy for reproducibility in experiments
        )
    hook.remove()
    return tokenizer.decode(output_ids[0][primary_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
```

---

### 3. COMBINED NOVELTY CLAIM

We present two contributions for understanding and mitigating prompt sensitivity in generative LLM tasks. First, we introduce PRI (Prompt Robustness Index), a dual-pillar evaluation framework that separately measures observational output stability (via Wasserstein-based semantic drift, performance elasticity, thematic alignment, and constraint satisfaction variance) and intrinsic model confidence (via perplexity variance and branching-factor stability across templates), synthesized via weighted harmonic mean into a single composite score with an interpretive 2×2 causal diagnosis matrix — the first framework to distinguish genuine robustness from evaluation artifacts, stochastic luck, and knowledge boundary effects. Second, we introduce LL-PIRC, the first training-free, generative-task mitigation method for prompt-paraphrase sensitivity, which uses Logit Lens-based per-layer perplexity variance across sampled paraphrases to localize the "sensitive layer" ℓ* and then applies Paraphrase-Invariant Residual Clamping — anchoring high-confidence token positions at ℓ* to the cross-paraphrase consensus hidden state — to stabilize generation without any model training, weight access, or oracle knowledge of a "correct" prompt variant.

---

### 4. SUGGESTED PAPER STRUCTURE

```
Title: PRI and LL-PIRC: Evaluation and Training-Free Mitigation of 
       Prompt Sensitivity in Generative LLM Tasks

1. Introduction (2 pages)
   - Motivation: prompt sensitivity in real-world generative tasks
   - Two contributions: (1) benchmark+metric, (2) mitigation method
   - Summary of findings and claimed novelty

2. Background and Related Work (1.5 pages)
   - Prompt sensitivity in classification vs generative tasks
   - Existing benchmarks: POSIX, ProSA, PromptBench, ReCode
   - Layer-level analysis: Liu & Chu, Skean et al., Kaplan et al.
   - Training-free interventions: ITI, CAA, RepE (positioned as adjacent)
   
3. The PRI Evaluation Framework (2.5 pages)
   - 3.1: Pillar 1 — ORI (SMS, AUC-E, TRD, KPIG)
   - 3.2: Pillar 2 — IFI (PPL_var, PC_stab)
   - 3.3: PRI synthesis via weighted harmonic mean
   - 3.4: 2×2 Causal Diagnosis Matrix
   - 3.5: Comparison with POSIX/ProSA (Table: which properties each captures)

4. The GenSens Benchmark (1.5 pages)
   - 4 task families: summarization, code gen, creative writing, dialogue
   - Prompt variation construction methodology
   - Datasets: CNN/DM, HumanEval, WritingPrompts (story), MultiWOZ
   - K=10 paraphrase variants per instance, 100 instances per task = 4,000 (prompt, variant) pairs
   
5. LL-PIRC: Training-Free Sensitivity Mitigation (2.5 pages)
   - 5.1: Logit Lens per-layer perplexity extraction (Step 3)
   - 5.2: Sensitive layer identification via S(ℓ) inflection (Step 4)
   - 5.3: Anchor token identification (Step 5)
   - 5.4: Paraphrase-Invariant Residual Clamping (Step 6)
   - 5.5: Comparison with 3 ablations (PIRC vs SAM vs PCDS)
   - 5.6: Non-oracle deployment protocol

6. Experiments (3 pages)
   - 6.1: PRI on GenSens benchmark — 8 models, 4 tasks
          (Table: PRI scores; 2×2 distribution per model per task)
   - 6.2: Correlation of IFI with ORI across tasks 
          (validates dual-pillar diagnostic)
   - 6.3: LL-PIRC mitigation results
          Primary: ROUGE-L variance reduction on summarization (LLaMA-3-8B)
          Secondary: Pass@k variance on HumanEval code
          Extended: BERTScore variance on creative writing
   - 6.4: Sensitive layer location per task (does ℓ* differ by task?)
   - 6.5: Ablation: PIRC vs SAM vs PCDS
   - 6.6: Ablation: K=1,2,5,10 paraphrases vs mitigation quality
   - 6.7: Comparison with POSIX, ProSA, Hua et al. calibration-corrected baseline
   
7. Analysis (1 page)
   - When does PIRC fail? (Knowledge Boundary cell of 2×2)
   - Efficiency: 2× inference cost vs mitigation gain tradeoff
   - Layer ℓ* analysis: task-specific vs universal?

8. Conclusion (0.5 page)
```

**Experiments needed per section:**
- Section 3: No new experiments — PRI is defined analytically; compute it on existing benchmark subsets (POSIX's open-ended subset, ProSA's coding subset) for comparison.
- Section 4: GenSens benchmark construction (annotation + paraphrase generation — 1 GPU-week with Phi-3-mini as paraphrase LM).
- Section 6.1–6.2: Run PRI on GenSens — main evaluation experiment (1–2 GPU-weeks on 8 models × 4 tasks × 4,000 pairs).
- Section 6.3: LL-PIRC on CNN/DM summarization — first experiment (1–2 GPU-days on LLaMA-3-8B; see below).
- Section 6.4: Run LL-PIRC across tasks to extract ℓ* per task (included in 6.3 compute).
- Section 6.5: Ablations on PIRC variants — extra 1 GPU-day.
- Section 6.6: K ablation — included in baseline runs.

---

### 5. FIRST EXPERIMENT TO RUN

**Goal**: Validate that LL-PIRC reduces ROUGE-L variance across prompt paraphrases on CNN/DailyMail summarization with LLaMA-3-8B-Instruct — *before* building the full GenSens benchmark.

**Dataset**: CNN/DailyMail test set; sample N=100 articles.

**Prompt variants**: For each article, generate K=5 paraphrase variants of the summarization instruction using T5-paraphrase (Hegde & Patil 2020) or a Phi-3-mini one-shot paraphraser. Verify all 5 variants score > 0.85 cosine similarity (SBERT) with the original instruction.

**Model**: LLaMA-3-8B-Instruct on single A100 80GB (fits in fp16).

**Procedure**:
1. Baseline: Generate 5 outputs (one per paraphrase) with greedy decoding. Compute ROUGE-L for each output vs gold summary. Record ROUGE-L variance across 5 outputs per article → `var_baseline[n]` for n=1..100.
2. Run LL-PIRC: extract S(ℓ) for each article's 5 paraphrases; identify ℓ*; identify anchor tokens; run PIRC-clamped generation; compute ROUGE-L. Record → `var_pirc[n]`.
3. Compute: relative variance reduction = 1 - mean(var_pirc) / mean(var_baseline).
4. Compute: mean ROUGE-L baseline vs mean ROUGE-L PIRC (ensure quality is preserved).
5. Statistical test: Wilcoxon signed-rank test on var_baseline vs var_pirc across 100 articles (paired, non-parametric).

**Expected result**: ROUGE-L variance reduction ≥ 25% relative (e.g., from std≈4.5 to std≈3.4 across paraphrases), with mean ROUGE-L change < 1 absolute point. If this holds, the core claim is validated and the full benchmark + 8-model study is justified.

**Expected ℓ* finding** (based on Liu & Chu, Skean et al.): ℓ* ≈ L × 0.6–0.75 = layer 19–24 of 32 for LLaMA-3-8B. If ℓ* falls consistently in this range across articles, it also validates that sensitivity localizes to late-middle layers (as predicted by theory) and that ℓ* can be pre-cached for deployment.

**Compute budget**: ~6 hours on single A100 for 100 articles × 5 variants × 2 forward passes each (baseline + PIRC). Total ≈ 1,000 forward passes.

**Success criteria**:
- p < 0.01 on Wilcoxon signed-rank test for variance reduction.
- Mean ROUGE-L within ±1 point of baseline (no quality regression).
- ℓ* is consistent across articles (coefficient of variation of ℓ* across 100 articles < 0.2).

**Failure modes and fallbacks**:
- If variance reduction < 10%: try anchor threshold τ ∈ {1.5, 2.0, 3.0} and report sensitivity to threshold.
- If mean ROUGE-L drops > 2 points: use a softer clamping (interpolation α ∈ (0,1) instead of full replacement: h_clamped = α · mean_h + (1-α) · h_A).
- If ℓ* varies wildly: shift to task-level ℓ* estimation (identify ℓ* by averaging S(ℓ) over 20 articles, then fix for the remaining 80) — this is the practical deployment protocol anyway.

---

*Document prepared as part of the "Adaptive Prompt Sensitivity in Complex LLM Tasks" research program. Builds on the literature review completed in the prior session.*
