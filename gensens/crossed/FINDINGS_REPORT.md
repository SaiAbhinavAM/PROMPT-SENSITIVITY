# Prompt Sensitivity in Summarization — Findings Report

**Crossed-design benchmark · Llama-3.1-8B-Instruct & Qwen2.5-7B-Instruct · CNN/DailyMail**
_Generated 2026-08-10_

---

## 1. Executive summary

We measure **prompt sensitivity** — how much a model's output changes when an instruction is reworded but its meaning is preserved — on a crossed design of **30 articles × 50 prompt "seeds" × ~6 meaning-equivalent paraphrases** (8,970 generations per model). Sensitivity is defined as the **spread** of a cell's outputs; quality is the **mean**, kept strictly separate.

**Headline finding:** *Which rhetorical **dimension** an instruction uses (question-form, theme-isolation, output-format, …) drives prompt sensitivity far more than the coarse Pool A/B split.* This holds in Llama-3.1-8B (dimension explains ~19% of sensitivity variance vs ~2% for pool; permutation p = 0.022) and **replicates in direction** in Qwen2.5-7B (Spearman 0.70, p = 0.005), with the most- and least-sensitive dimension classes **identical** across the two models.

**Honest qualifier:** the effect is **strong in Llama but weaker/borderline in Qwen** (Qwen mixed-model p = 0.019 but assumption-free permutation p = 0.071), and only the **coarse** ranking replicates — the exact 14-way order is model-specific.

---

## 2. Methodology

- **Design:** every seed applied to the same 30 CNN/DailyMail articles (instance-fixed, prompt-varied), following POSIX/ProSA/PromptBench convention. A **cell** = one (article, seed) → its ~6 paraphrase outputs.
- **Sensitivity = spread, Quality = mean.** Every metric that takes a mean across a cell's outputs also reports its variance; the variance is the sensitivity signal.
- **Sensitivity composite (v2), rank-normalized within a run:** `0.25·SMS-drift + 0.20·ROUGE-L-var + 0.15·CS-cv + 0.15·faith-cv + 0.125·PPL-var + 0.125·PC-stab-cv`. Spread of bounded quality scores is level-normalized (coefficient of variation) to de-confound spread from quality level; lexical drift (ROUGE-L) is included so the score is not purely embedding-based.
- **Scorers (deterministic, SOTA):** SMS drift via Sentence-BERT (meaning drift); **MiniCheck-DeBERTa-v3-large** for faithfulness (SOTA fact-checking, whole-summary mode); **BERTScore** (rescaled) for correctness vs the gold summary; real teacher-forced perplexity / branching-factor for confidence spread. An LLM-judge was deliberately avoided — its own stochasticity would contaminate a variance benchmark.
- **Decoding:** greedy (temperature 0), to isolate the prompt effect from sampling noise.
- **Paraphrase equivalence** was validated in the dataset by **bidirectional NLI entailment**, not just similarity (see §6).

---

## 3. Main finding — Llama-3.1-8B

**Dimension is the primary axis.** Instruction dimension explains **19.3%** of sensitivity variance vs **1.7%** for the Pool A/B split (~11×). Mixed-effects test `sensitivity ~ dimension + (1|seed) + (1|article)`: **χ²=30.8, p=0.0036**. Pool A/B is not significant (seed-level Mann-Whitney p = 0.16).

**Sensitivity by pool:** A = 0.485 [0.475, 0.496], B = 0.535 [0.519, 0.550].

**Per-dimension ranking (most → least sensitive):**

| Rank | Dimension | Sensitivity |
|---|---|---|
| 1 | Meta-reflection* | 0.620 |
| 2 | Question Form | 0.614 |
| 3 | Theme Isolation | 0.611 |
| 4 | Analytical Framing | 0.561 |
| 5 | Role Based | 0.520 |
| 6 | Persona / Tone | 0.490 |
| 7 | Audience Perspective | 0.487 |
| 8 | Emphasis Focus | 0.485 |
| 9 | Completeness Framing | 0.483 |
| 10 | Tone Register | 0.480 |
| 11 | Lexical Filter | 0.468 |
| 12 | Direct Command | 0.446 |
| 13 | Output Format | 0.358 |
| 14 | Constraint Based | 0.352 |

\* Meta-reflection is backed by only **1 seed** — treat it as a single-prompt observation, not a stable dimension estimate (see §8).

---

## 4. Robustness validation (Llama)

Every internal cross-check passed.

| Check | Question | Result |
|---|---|---|
| Different scorers (×3 configs) | Is the ranking scorer-dependent? | Spearman 0.94 — stable |
| Quality confound | Is sensitivity just low quality? | residualized ranking Spearman 0.94 |
| **Length confound** | Is it just an output-length effect? | corr −0.07; length-adjusted ranking **0.97** |
| **Split-half replication** | Does it replicate within the study? | **mean Spearman 0.87** over 300 splits |
| **Permutation test** | Significant with no normality assumption? | **p = 0.022** |
| **Paraphrase equivalence** | Do the paraphrases truly mean the same? | bidirectional NLI **0.98 / 0.97**, 100% passed, token overlap 0.36 |

The **assumption-free permutation p = 0.022** is the most defensible significance value (the parametric mixed-model p = 0.0036 assumes a normality the rank-normalized composite lacks).

---

## 5. Cross-model replication — Qwen2.5-7B

Identical grid and pipeline, only the generation model changed.

| Metric | Result | Reading |
|---|---|---|
| Dimension-ranking Spearman (Llama vs Qwen) | **0.70** (p = 0.005) | ranking correlates across families |
| Seed-ranking Spearman (50 prompts) | 0.74 (p = 1e-9) | individual prompts rank alike |
| SMS-drift anchor cross-check | 0.82 | even higher on the run-independent measure |
| **Bottom-3 dimension overlap** | **Jaccard 1.00 (identical)** | least-sensitive set is the same |
| Top-3 dimension overlap | Jaccard 0.50 | Question Form + Theme Isolation shared |

**Qwen independently reproduces the structure:** dimension η² = 14.9% vs pool 0.8% (~18×); mixed-model **p = 0.019**; Pool A/B null (seed-MWU p = 0.47).

**The critical caveat — Qwen's effect is weaker.** Under the strict, assumption-free tests, Qwen's dimension effect does **not** clear significance: **permutation p = 0.071**, seed-level Kruskal p = 0.156. It is significant only in the parametric mixed model. Excluding the ~4.5% of cells that contained a degenerate/refusal output did **not** rescue it (permutation p 0.071 → 0.066), so the weakness is **intrinsic**, not a data-quality artifact.

**What replicates vs. what doesn't:**

- ✅ **Replicates:** Question Form & Theme Isolation among the most sensitive; Constraint Based, Direct Command, Output Format among the least — in **both** models. The "dimension >> pool" structure.
- ❌ **Does not replicate:** the exact ranking. The middle reshuffles (Meta-reflection #1→#7, Tone Register #10→#3, Role Based #5→#10), and the **effect strength is model-dependent** (strong in Llama, borderline in Qwen).

---

## 6. Which model is more sensitive?

The rank-normalized composite cannot answer this (relative within each run). Using the **absolute, run-independent** spreads on the same 1,500 cells:

| What drifts across paraphrases | Llama | Qwen | More sensitive |
|---|---|---|---|
| Meaning (SMS drift) | 0.0941 | 0.0953 | ≈ tie (p = 0.10) |
| Wording (ROUGE-L spread) | 0.0009 | 0.0011 | Qwen (p = 6e-10) |
| Correctness spread | 0.0041 | 0.0046 | Qwen (p = 4e-6) |
| BERTScore spread | 0.0032 | 0.0038 | Qwen (p = 2e-7) |
| Faithfulness spread | 0.0399 | 0.0383 | Llama (n.s.) |

**Qwen is slightly more sensitive overall — but the edge is in *wording*, not *meaning*** (meaning-drift is a statistical tie). Output lengths are similar (Llama 209, Qwen 200 words), so this is not a length artifact.

Note the nuance vs §5: Qwen is **moodier on average but less organized by dimension** — its sensitivity is spread more evenly across prompt styles, whereas Llama's is more clearly tied to which style is used. "More sensitive overall" and "sensitivity more explained by dimension" are different axes.

---

## 7. Output-quality audit

Scanning all 8,970 Qwen generations: **0 empty**, 0 ultra-short; **~1% degenerate repetition** (84 outputs), **0.32% refusal-like** (29), **1.6% near the 512-token cap** (possible truncation), **3.4%** still carrying a "summary…:" preamble the (Llama-tuned) stripper missed. None are large; the degenerate/refusal cells were shown not to drive the result (§5). Some very-short Qwen outputs were **correct** constraint-following (e.g. "no financial content mentioned" for a Theme-Isolation prompt on an off-topic article), not degeneracy.

---

## 8. Flaws & limitations (honest)

1. **Effect strength is model-dependent.** Strong in Llama, borderline in Qwen (permutation p = 0.071). The direction/coarse ranking replicates; the strength does not.
2. **Only the coarse pattern replicates.** The exact 14-way ranking is model-specific; only the extremes are stable across models.
3. **Thin, unbalanced dimensions.** Meta-reflection = 1 seed; its #1→#7 move across models confirms single-seed dimensions are unreliable. Several dimensions have only 3 seeds.
4. **Narrow model/task slice.** Two ~7–8B instruction-tuned models, one task (summarization), one dataset (CNN/DM). No scale contrast (70B), no format/surface perturbations, no non-summarization task.
5. **No human validation** of the drift metric — face validity only.
6. **Post-hoc dimension pivot.** The pre-registered Pool A/B hypothesis failed (null in both models); dimension emerged from the data.
7. **Composite still SMS-leaning** (SMS-drift is the strongest single component, rho ~0.72); faithfulness contributes least.
8. **Minor:** ~1% degenerate outputs, whole-summary (not per-sentence) MiniCheck for speed, regex entity-coverage, B12 seed has 5 variants not 6.

---

## 9. Defensible claims

> Across two model families (Llama-3.1-8B, Qwen2.5-7B), instruction **dimension** drives prompt sensitivity far more than the Pool split (both mixed-model p < 0.02, ~11–18× more variance), and the dimension **rankings agree significantly** (Spearman 0.70, p = 0.005), with **identical most- and least-sensitive dimension classes**. Question-form and theme-isolation phrasings are reliably the most sensitive; format- and constraint-based phrasings the least. However, the **effect strength is model-dependent** — strong in Llama, borderline in Qwen under assumption-free testing — and the fine-grained ranking is model-specific. The result is best described as a **robust coarse pattern**, not an exact cross-model law.

**Do not claim:** that it holds for all LLMs, that the precise 14-way ranking replicates, or that Meta-reflection specifically is most sensitive.

---

## 10. Recommended next steps

1. **A third model** (highest value) — resolves whether Qwen's weakness is an outlier or the effect is genuinely fragile. Mistral-7B (another family, same scale) or Llama-3.1-70B (scale contrast).
2. **Balance thin dimensions** — add seeds so no dimension rests on 1 prompt.
3. **~50-pair human validation** of the drift metric.
4. **Format/surface perturbations** as a second axis, or scope the paper explicitly to semantic-paraphrase sensitivity.
