# Methodology Changelog

> **This file tracks every change to the project's methodology, metrics, formulas, algorithms, and evaluation procedures.**
> New entries go at the **top** (newest first). Read this file to understand the full history of design decisions.

---

## [2026-05-25] — Initial Methodology Baseline (Documentation)

**Files:** All project files (initial snapshot)

**What Changed:**
- Documented the complete current-state methodology as the baseline for future change tracking
- No code changes — this entry establishes the starting point

**Current Methodology Summary:**

### Phase 1: GenSens Dataset Generation
- **Paraphrase Model:** LLaMA-3-8B-Instruct (bfloat16)
- **Similarity Filter:** SBERT (`all-mpnet-base-v2`), cosine similarity ≥ 0.82
- **Scale:** 800 instances × 8 variants across 4 tasks (Summarization, Code, Creative, Dialogue)
- **Source Datasets:** CNN/DailyMail, HumanEval + MBPP, WritingPrompts, MultiWOZ 2.2
- **Strategies:** 16 diverse paraphrase strategies

### Phase 2: PRI Benchmark
- **Core Metrics (ORI):** SMS, AUC-E, TRD, KPIG, PPL Variance, Branching Factor
- **Advanced Metrics:** Semantic TRD, Wasserstein SMS, Advanced KPIG, USD
- **Quality Metrics:** Correctness Score (CS), Hallucination Score (HS)
- **PRI Formula:** `0.40 × Consistency + 0.35 × CS + 0.25 × exp(−HS)`
- **Penalties:** HS > 0.5 → PRI × 0.6; avg_len < 12 → PRI × 0.85
- **Final Score:** `0.6 × PRI + 0.4 × Human_Score` (static mode)
- **Dynamic Weighting:** `trust_human = min(0.7, 0.4 + 0.3 × USD)` (optional)
- **LLM Judge:** Flan-T5-base zero-shot scorer (1-5 scale, normalized to [0,1])
- **Embeddings:** all-MiniLM-L6-v2 (Sentence-BERT)
- **Prompt Levels:** d1 (Literal/Low), d2 (Stylistic/Medium), d3 (Creative/High) — 3 templates each
- **PRI Math:** Also available as weighted harmonic mean: `Σw_i / Σ(w_i / S_i)`
- **Attribution:** 4 categories — True Robustness, Stochastic Luck, Evaluation Artifact, Knowledge Boundary

### Phase 3: LL-PIRC Intervention
- **Logit Lens:** `logits^(ℓ) = lm_head(LayerNorm(h^(ℓ)))` → per-token PPL
- **Sensitivity Signal:** `S(ℓ) = Var_k[mean_token_PPL at layer ℓ]`
- **ℓ* Detection:** Inflection method (default): `argmax_ℓ [S(ℓ) − S(ℓ−1)]`; Z-score fallback: first ℓ where S(ℓ) > mean + 2×std
- **Layer Scan Range:** 25% to 100% of total layers
- **Anchor Tokens:** Percentile-based (bottom 30% by `rank(mean_ppl) + rank(var_ppl)`), content-word filtering, guaranteed ≥1 anchor
- **Clamping:** Full (α=1.0): `h[anchors] = mean_h`; Soft (α<1.0): `h = α×mean_h + (1−α)×h_original`
- **Generation:** Greedy decoding (`do_sample=False`), max_new_tokens=200
- **Supported Architectures:** LLaMA-style (`model.model.norm`) and GPT-2-style (`model.transformer.ln_f`)

### Phase 4: Statistical Evaluation
- **Variance Metric:** `Δ_var = 1 − mean(var_pirc) / mean(var_baseline)`
- **Wilcoxon (Variance):** One-sided (baseline > PIRC), α=0.01
- **Wilcoxon (ROUGE-L):** Two-sided, α=0.01
- **ℓ* Stats:** Mean, std, min, max, coefficient of variation
- **Plots:** S(ℓ) sensitivity curves, variance scatter + histogram

**Why:**
- Establishing a documented baseline so all future changes can be tracked relative to this snapshot

**Impact:**
- No behavioral change — documentation only

---

## [2026-05-25] — Transformers v5.x Compatibility Fix (pirc.py)

**Files Modified:** `prompt_robustness/src/pirc.py`

**What Changed:**
- Updated `clamp_hook()` in `PIRCGenerator.generate_with_clamping()` to handle both tuple and plain Tensor outputs from transformer blocks
- Previously assumed output was always a `tuple`; transformers v5.x's `output_capturing` wrapper returns a plain `Tensor` during `model.generate()`

**Before:**
```python
h = output[0]
rest = output[1:]
# ... clamping ...
return (h,) + rest
```

**After:**
```python
if isinstance(output, tuple):
    h = output[0]
    rest = output[1:]
else:
    h = output
    rest = None
# ... clamping ...
if rest is not None:
    return (h,) + rest
else:
    return h
```

**Why:**
- Fatal crash (`TypeError: can only concatenate tuple (not "Tensor") to tuple`) on transformers ≥ 5.x during PIRC clamped generation
- This was a blocking bug for any GPU deployment

**Impact:**
- **Critical fix** — without this, PIRC generation crashes on modern transformers
- No change to numerical outputs when running on compatible transformer versions
- Backward compatible with older transformers that return tuples

---

## [2026-05-25] — Anchor Token Selection: Threshold → Percentile (anchor_tokens.py)

**Files Modified:** `prompt_robustness/src/anchor_tokens.py`

**What Changed:**
- Replaced the default anchor selection from **absolute PPL thresholds** (`mean_ppl < τ AND var_ppl < τ_var`) to **percentile-based stability ranking**
- New default: select bottom 30% most-stable tokens by `rank(mean_ppl) + rank(var_ppl)`
- Added content-word filtering (exclude punctuation + stopwords)
- Added guarantee: always ≥ 1 anchor, at most ⌊min_len/2⌋
- Legacy threshold method preserved as `identify_anchors_threshold()` for ablations

**Why:**
- The absolute threshold approach was **brittle across architectures**: different models have wildly different PPL scales (GPT-2 baselines ~50-100 PPL vs LLaMA ~5-15 PPL)
- A poorly tuned τ silently produced 0 anchors → PIRC clamping became a complete no-op with no warning
- Percentile-based selection is model-agnostic and self-calibrating

**Impact:**
- **Breaking change** for anchor selection behavior — PIRC results will differ from any previous runs using the threshold method
- The percentile method is strictly more reliable (never returns 0 anchors)
- Previous results generated with the threshold method should be re-validated

---

## [2026-05-25] — GenSens Stats Table Fix (generate_dataset.py)

**Files Modified:** `gensens/scripts/generate_dataset.py`

**What Changed:**
- Added `if task_name.startswith("_"): continue` check in `print_stats_table()` to skip internal keys like `_meta`

**Why:**
- Crash (`KeyError: 'sbert_similarity'`) when the stats dict contained the `_meta` key and the printer tried to process it as a task entry

**Impact:**
- Cosmetic fix only — no change to generated data, just prevents a crash during the final summary print

---
