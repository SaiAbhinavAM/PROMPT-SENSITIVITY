# Data Preparation — Prompt Variant Generation and Evaluation Role

> End-to-end reference for how prompt variants are created, filtered, stored, and consumed by every stage of the evaluation pipeline.

---

## 1. Overview

The entire research hypothesis rests on a single question: *if you ask the same thing in a different way, does the model answer consistently?* To test this, every input instance must be paired with multiple **semantically equivalent but lexically distinct** prompt variants. This document explains how those variants are produced and how each downstream metric uses them.

There are **two independent variant generation mechanisms** in this project:

| Mechanism | Module | When Used | How Generated |
|-----------|--------|-----------|---------------|
| **GenSens** | `gensens/scripts/` | Primary (H100, research runs) | LLM-driven, 16 strategies, SBERT filtered |
| **Synthetic d1/d2/d3** | `prompt_robustness/src/prompt_generator.py` | Fallback (no GenSens data supplied) | Hard-coded template strings, 3 perturbation levels |

When `evaluate_sample` receives a sample that already carries `prompt_variants` (loaded from a GenSens JSONL/CSV), it uses those directly. Only when no variants are supplied does it fall back to the d1/d2/d3 templates.

---

## 2. Source Datasets (Phase 1 Input)

GenSens draws raw instances from four publicly available datasets, one per task domain. **Every task now carries a non-empty `reference_output`** (gold) and an `input_text` (grounding) — both written into `metadata` by the loader and bridged to the CSV columns of the same name. Rows whose source lacks a reference are skipped.

> **Verified HF paths (inspected before coding):** `cnn_dailymail` only exposes configs `1.0.0/2.0.0/3.0.0` — the names *sum_in_brief* and *generate_story* are **PromptSource template framings**, not HF configs. DREAM loads from `dream` (`trust_remote_code=True`). ELI5 loads from `sentence-transformers/eli5`, config `pair` (columns `question`/`answer`, **train split only**).

### 2.1 Summarization — CNN/DailyMail (`sum_in_brief`)

- **Dataset:** `cnn_dailymail` v3.0.0, `test` split
- **Sampling:** Stratified by article length — short < 400, medium 400–800, long > 800 words; ⌊n/3⌋ per bucket, remainder to long, seed 42
- **What gets paraphrased:** The fixed instruction — *"Summarize the following news article in 3-4 sentences, capturing the main events and key details."*
- **What stays fixed:** The full article body
- **`input_text` (grounding):** the article
- **`reference_output` (gold):** the `highlights` field

### 2.2 Creative — CNN/DailyMail (`generate_story`, the inverse)

- **Dataset:** `cnn_dailymail` v3.0.0, `test` split — **same length-stratified sampler as summarization** (seed 42), so it covers the same article/highlights pairs inverted
- **What gets paraphrased:** The fixed instruction — *"Write a detailed news story that expands on the following summary points, elaborating them into a complete article."*
- **What stays fixed:** The summary premise (the highlights)
- **`input_text` (grounding):** the highlights (the short premise)
- **`reference_output` (gold):** the full **article** (the gold long-form "story")

> **Why the inversion:** `cnn_dailymail` has no separate story field, and making `input_text == reference_output` would be circular. The PromptSource *generate_story* framing inverts summarization (premise → full story), which yields a non-empty, non-circular reference and keeps creative distinct from summarization.

### 2.3 Dialogue — DREAM

- **Dataset:** `dream` (reading-comprehension over dialogues), `train` split
- **Filtering:** non-empty dialogue, question, and answer (deterministic shuffle, seed 42)
- **What gets paraphrased:** The **question**
- **What stays fixed:** The dialogue turns and the multiple-choice options
- **`input_text` (grounding):** the dialogue turns (joined)
- **`reference_output` (gold):** the correct **answer choice**

### 2.4 QA — ELI5

- **Dataset:** `sentence-transformers/eli5`, config `pair`, `train` split
- **Filtering:** English question of 5–60 words, answer ≥ 10 words (light deterministic quality gate, seed 42)
- **What gets paraphrased:** The **question**
- **What stays fixed:** Only the answering instruction frame (the question *is* the content)
- **`input_text` (grounding):** the question
- **`reference_output` (gold):** the answer/explanation

---

## 3. GenSens: LLM-Driven Variant Generation

### 3.1 Pipeline Architecture

```
Source Instance
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  ParaphraseGenerator.process_instance()                 │
│                                                         │
│  For each of 16 strategies (round-robin):               │
│    1. Build LLM prompt (strategy instruction + base)    │
│    2. Generate candidate via LLaMA / Flan-T5 / vLLM    │
│    3. Clean output (strip echoed prefixes, metadata)    │
│    4. Quality gates:                                    │
│         a. min 5 words                                  │
│         b. not an exact copy                            │
│         c. not a near-duplicate (first-8-word sig)      │
│         d. SBERT sim ∈ [0.82, 0.98]  ← meaning band    │
│         e. Jaccard overlap < 0.85 vs base               │
│         f. Jaccard overlap < 0.85 vs already-accepted   │
│         g. max 2 variants per strategy                  │
│    5. Accept or reject                                  │
│                                                         │
│  Retry up to 3 times if n_variants not reached          │
└─────────────────────────────────────────────────────────┘
      │
      ▼
build_prompt_variants()
  Replace base_text in base_prompt with each paraphrased_text
      │
      ▼
    JSONL + CSV  →  gensens/data/
```

### 3.2 Generation Backends

Three backends are available, selected via `--model`:

| Backend | Model | Hardware | Speed | Use Case |
|---------|-------|----------|-------|----------|
| `local` | `google/flan-t5-large` (~770MB) | CPU / Apple MPS | Slow | Dev / smoke test |
| `llama` | `meta-llama/Meta-Llama-3.1-8B-Instruct` (bf16) | H100 | Medium | Single-GPU production |
| `vllm` | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | H100 | Fast (batched) | Full research run |

The `vllm` backend uses `SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, seed=42)` for batched generation. `local` and `llama` generate one candidate at a time with `temperature=0.85, top_p=0.92`.

**Task-specific system prompts** tell the generator exactly what kind of rewrite is expected (e.g., for summarization: *"Do NOT change what is being asked — only change the wording"*). This keeps the instruction semantics stable while the surface form changes.

### 3.3 The 16 Paraphrase Strategies

Each strategy targets a specific axis of linguistic variation. No more than 2 variants from any single strategy are accepted per instance (`max_per_strategy=2`), ensuring the final variant set spans multiple dimensions.

| # | Strategy Name | What Changes |
|---|--------------|-------------|
| 1 | `formal_tone` | Register → more formal, professional |
| 2 | `casual_tone` | Register → conversational, informal |
| 3 | `reordered_clauses` | Syntactic order of phrases/clauses |
| 4 | `synonyms` | Key content words replaced with synonyms |
| 5 | `concise` | Length reduced (fewer words, same meaning) |
| 6 | `different_structure` | Sentence structure completely different |
| 7 | `different_opening` | First word/phrase entirely changed |
| 8 | `role_prefix` | Adds role context ("As a reader, …") |
| 9 | `passive_voice` | Active constructions → passive where possible |
| 10 | `question_form` | Rewritten as a question or request |
| 11 | `imperative` | Rewritten as a direct command |
| 12 | `elaborated` | Adds brief detail/elaboration |
| 13 | `technical_vocab` | More precise / technical word choices |
| 14 | `simple_vocab` | Simpler, everyday vocabulary |
| 15 | `split_sentences` | Long sentences split into shorter ones |
| 16 | `merged_sentences` | Short phrases merged into longer sentences |

### 3.4 The SBERT Similarity Band (Quality Gate)

Every candidate is scored against the original `base_text` using SBERT (`all-mpnet-base-v2`):

```
Lower bound: sim ≥ 0.82  →  rejects meaning drift ("asks something different")
Upper bound: sim ≤ 0.98  →  rejects near-identical restatements ("no real perturbation")
```

A candidate outside the band `[0.82, 0.98]` is discarded and the next strategy is tried. This band ensures every variant is both *semantically equivalent* (within tolerance) and *lexically distinct* (worth testing).

Additionally, a **Jaccard token-overlap** check rejects candidates that reuse > 85% of the base text's words (`max_token_overlap=0.85`), catching cases where SBERT similarity is acceptable but lexical surface form barely changed.

### 3.5 Output Schema

Each instance is stored in JSONL with the following structure:

```jsonl
{
  "instance_id": "summ_0000",
  "task": "summarization",
  "base_text": "Summarize the following news article in 3-4 sentences, ...",
  "base_prompt": "Summarize the following news article...\n\nArticle:\n<full article text>\n\nSummary:",
  "metadata": {
    "article": "<full article text>",
    "gold_summary": "<reference highlights>",
    "word_count": 542
  },
  "variants": [
    {
      "variant_idx": 0,
      "paraphrased_text": "Provide a concise 3-4 sentence overview of the news article below, ...",
      "sbert_similarity": 0.9134,
      "strategy": "formal_tone",
      "full_prompt": "Provide a concise 3-4 sentence overview...\n\nArticle:\n<same article>\n\nSummary:"
    },
    {
      "variant_idx": 1,
      "paraphrased_text": "Can you sum up this article in a few sentences?",
      "sbert_similarity": 0.8541,
      "strategy": "question_form",
      "full_prompt": "Can you sum up this article...\n\nArticle:\n<same article>\n\nSummary:"
    },
    ...
  ],
  "n_variants_generated": 4,
  "generation_time_s": 12.4,
  "diversity": {
    "distinct_2": 0.7812,
    "mean_pairwise_overlap": 0.2341,
    "n_strategies": 4
  }
}
```

A companion **flat CSV** (one row per variant) is also written, making dataset loading trivial without re-running generation.

### 3.6 Diversity Metrics Logged Per Instance

After generation, two corpus-level diversity stats are recorded:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `distinct_2` | `unique bigrams / total bigrams` across all variants | Higher = more lexically varied |
| `mean_pairwise_overlap` | Mean Jaccard overlap between all variant pairs | Lower = more diverse |

These are logged but not used as hard filters — they serve as diagnostic signals.

### 3.7 Checkpointing

Generation is crash-safe. A `.checkpoint.jsonl` file is updated every `--checkpoint_every` instances (default: 20). On restart, already-completed `instance_id`s are skipped. The checkpoint is deleted once the final JSONL is written.

---

## 4. Synthetic d1/d2/d3 Variants (Fallback)

When GenSens data is not available, `prompt_generator.py` generates variants from hard-coded templates organised into three **perturbation levels**:

| Level | Name | Perturbation | Purpose |
|-------|------|-------------|---------|
| d1 | Literal | Low — minimal wording change | Near-baseline; tests lexical sensitivity |
| d2 | Stylistic | Medium — phrasing and style shifted | Tests moderate surface variation |
| d3 | Creative/Complex | High — restructured, different framing | Tests robustness under significant rewording |

For summarization, each level has 3 template strings (9 total). The key difference from GenSens is that d1/d2/d3 templates are **pre-written and static** — they do not call an LLM and carry no SBERT guarantee. They exist for fast local runs and smoke tests.

AUC-E (Performance Elasticity) specifically uses these perturbation levels to build a robustness curve: it measures how model quality degrades as perturbation level increases from d1 → d2 → d3.

---

## 5. Post-Generation Validation (`validate_dataset.py`)

Six checks are run automatically after generation:

| Check | What It Catches | Gate |
|-------|----------------|------|
| **Completeness** | Wrong instance count or variant count | Hard error |
| **Similarity distribution** | Variants below 0.82 threshold that slipped through | Warning |
| **Diversity** | Mean pairwise word-edit-distance < 10 (too similar) | Warning |
| **Length** | Variants < 5 words or > 3× base word count | Warning |
| **Exact copies** | `paraphrased_text == base_text` | Warning |
| **Prompt substitution** | `full_prompt` still contains original `base_text` | Warning |

The validation report is saved to `gensens/data/gensens_validation_report.json`.

---

## 6. How Variants Feed Into Evaluation (End-to-End)

Once generated, the K variants per instance drive **every metric** in Phase 2 (PRI Benchmark) and Phase 3 (PIRC). Here is the exact role each metric assigns to the variant set.

### 6.1 Step-by-Step Evaluation Flow

```
K prompt variants  →  Subject Model  →  K responses
                                              │
            ┌─────────────────────────────────┴────────────────────────────────────┐
            │                                                                      │
            ▼                                                                      ▼
    CONSISTENCY / ORI metrics                                           QUALITY metrics
    (how different are the K responses?)                                (how good are the responses?)
            │                                                                      │
     SMS  AUC-E  TRD  KPIG                                          CS  Faithfulness  HS  LLM-Judge
            │                                                                      │
            └──────────────────────────┬───────────────────────────────────────────┘
                                       │
                                 PRI  Diagnostic_PRI  ORI  IFI
                                       │
                                  Final_Score
```

### 6.2 Metric-by-Metric Breakdown

#### SMS — Semantic Manifold Stability
`mean(cosine_sim(embeddings)) − 0.5 × Var(L2_normed_embeddings)`

Encodes ALL K response embeddings at once. A high SMS means the K outputs cluster tightly in embedding space — the model consistently produced semantically similar responses regardless of which paraphrase was used.

#### AUC-E — Performance Elasticity
Area under the quality-vs-perturbation-level curve (d1 → d2 → d3). Built from the d1/d2/d3 sub-groups of variants. A flat curve (high AUC-E) means the model degrades gracefully as prompt wording becomes more distant from the original. A steep drop means the model is brittle to phrasing changes.

#### TRD — Thematic Robustness Drift (semantic)
`Var(embeddings)` expressed as mean pairwise semantic distance between the K responses. Low TRD means all K responses stayed on the same semantic track; high TRD means the model's responses drifted in meaning across paraphrases.

#### KPIG — Key Point Information Gain (coverage)
`mean(|facts_i ∩ all_facts| / |all_facts|)` where `facts` are n-gram units extracted from each response against the reference. Measures how consistently the model extracts the key information from the source, across all K prompt variants.

#### PPL Variance (IFI component)
The model's token-level perplexity is measured for each of the K prompt+response pairs. High variance across the K inputs means the model finds some phrasings internally more "surprising" than others — a signal of internal sensitivity even when the output text looks similar.

#### Branching Factor (IFI component)
Entropy of the top-k token probability distribution at each generation step, averaged across the K inputs. High branching factor variance means the model is uncertain and takes different generation paths for different paraphrases.

#### CS — Correctness Score
SBERT cosine similarity between each of the K responses and the reference output, averaged. Not used in PRI directly — it feeds the quality axis.

#### Faithfulness (PRI axis)
NLI entailment probability: `P(entail) + 0.5 × P(neutral)` from a DeBERTa cross-encoder, computed for each of the K `(source_article, response_i)` pairs and averaged. Tests whether each variant's response is factually grounded in the input, not just whether it matches the reference.

#### LLM-as-a-Judge (Human Score)
The judge scores ALL K responses against the input (not just responses[0]). The average judge score becomes `human_score`. This ensures the quality signal spans every variant's output, not just the best or first one.

### 6.3 PRI Composite

```
PRI = 0.40 × SMS  +  0.35 × CS  +  0.25 × Faithfulness
```

All three inputs are derived from the K variants. SMS captures cross-paraphrase output consistency. CS captures reference-matching quality. Faithfulness captures source grounding.

### 6.4 Diagnostic_PRI Composite

```
Diagnostic_ORI = HM(SMS,  AUC-E,  1−TRD,  KPIG)
Diagnostic_IFI = HM(1−PPL_var,  1−BF)
Diagnostic_PRI = HM(Diagnostic_ORI,  Diagnostic_IFI)
```

The harmonic mean is strict: if any single pillar is near zero, the composite collapses. This lets the 2×2 diagnosis matrix identify *where* the failure is:

| ORI ≥ 0.70 | IFI ≥ 0.70 | Diagnosis |
|------------|------------|-----------|
| Yes | Yes | **Robust** — consistent outputs, stable internals |
| Yes | No | **Externally Stable / Internally Fragile** — outputs look consistent, but internals show high PPL/BF variance |
| No | Yes | **Internally Stable / Output-Sensitive** — internals are stable, but outputs diverge across paraphrases |
| No | No | **Fragile** — fails on both axes |

### 6.5 PIRC — Phase 3 (uses variants for stabilisation)

PIRC uses the K variant prompts differently — not just for measurement but for **intervention**:

1. All K variants are fed through the model simultaneously with Logit Lens hooks active.
2. The **sensitive layer ℓ\*** is detected by finding the layer where per-layer PPL variance across the K variants spikes most.
3. **Anchor tokens** (bottom 30% by PPL rank) are identified at ℓ\*.
4. A **consensus hidden state** is computed as the mean of the K anchor-token hidden states at ℓ\*.
5. Each of the K variants is then re-generated with its anchor token activations clamped toward the consensus: `h[anchors] = α × mean_h + (1−α) × h_original`.
6. This produces K PIRC-stabilised outputs — one per variant.
7. ROUGE-L is computed for each of the K PIRC outputs, and `pirc_rouge_var` is the **real variance** across those K scores (not zero-by-construction as in the old single-output design).

Phase 4 then tests whether `pirc_rouge_var` is significantly lower than `var_baseline` using the Wilcoxon signed-rank test, 95% bootstrap CI, and paired Cohen's dz effect size.

---

## 7. Data Flow Summary (All Phases)

```
Phase 1 — GenSens
═══════════════════════════════════════════════════════════
  CNN/DailyMail (sum_in_brief + generate_story) / DREAM / ELI5
         │
         ▼
  LLaMA-3 / Flan-T5 paraphrase generation
  (16 strategies, SBERT band [0.82, 0.98])
         │
         ▼
  gensens/data/gensens_<task>_<N>inst_<K>var.jsonl
  gensens/data/gensens_<task>_<N>inst_<K>var.csv

Phase 2 — PRI Benchmark
═══════════════════════════════════════════════════════════
  Load JSONL/CSV  (or fallback: generate d1/d2/d3 templates)
         │
         ▼  K prompt variants per instance
  Subject model generates K responses
         │
         ├─► responses.csv  (Layer B — one row per variant)
         │
         ▼  K responses
  Score: SMS, AUC-E, TRD, KPIG, PPL_var, BF, CS, Faith, HS
         │
         ├─► scored_samples.csv  (Layer C — one row per instance)
         │     raw components + diagnostic_ori/ifi/pri + diagnosis
         │
         ▼
  reaggregate.py  (CPU-only, no model)
    recomputes: PRI, Diagnostic_PRI, ORI, IFI, Final_Score
         │
         ▼
  results/benchmark_reaggregated.csv
  results/final_results.csv

Phase 3 — PIRC
═══════════════════════════════════════════════════════════
  Load K variants from baseline.json
         │
         ▼  K prompts fed simultaneously
  Logit Lens → detect ℓ*  →  find anchors  →  clamp
         │
         ▼  K PIRC-stabilised outputs
  ROUGE-L for each → pirc_rouge_mean, pirc_rouge_var
         │
         ▼
  results/pirc.json

Phase 4 — Statistical Evaluation
═══════════════════════════════════════════════════════════
  Compare var_baseline vs pirc_rouge_var
         │
         ▼
  Wilcoxon signed-rank test  (α = 0.01)
  95% bootstrap CI on mean variance reduction
  Paired Cohen's dz effect size
         │
         ▼
  results/evaluation_summary.json
```

---

## 8. Key Parameters Reference

| Parameter | Default | Where Set | Effect |
|-----------|---------|-----------|--------|
| `n_variants` (K) | 8 (GenSens), 5 (PIRC config) | CLI / `config.yaml` | Number of prompt variants per instance |
| `similarity_threshold` | 0.82 | `config.yaml`, CLI | Lower SBERT band — rejects meaning drift |
| `similarity_upper` | 0.98 | CLI | Upper SBERT band — rejects near-identical restatements |
| `max_token_overlap` | 0.85 | CLI | Lexical diversity floor (Jaccard) |
| `max_per_strategy` | 2 | CLI | Cap per strategy to ensure variety |
| `min_word_count` | 5 | Hard-coded in `ParaphraseGenerator` | Rejects degenerate short outputs |
| `max_retries` | 3 | `generate_paraphrases()` | Retries full strategy loop if not enough valid variants |
| `checkpoint_every` | 20 | CLI | Crash-safe checkpoint frequency |
| SBERT model (GenSens) | `all-mpnet-base-v2` | `ParaphraseGenerator.__init__` | Used for similarity gate |
| SBERT model (PRI SMS) | `all-MiniLM-L6-v2` | `config.yaml` / env | Used for embedding-based metrics |
