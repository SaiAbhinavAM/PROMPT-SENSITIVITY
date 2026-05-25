# CLAUDE.md — Project Instructions for AI Assistants

> **Read this file first. It is the single source of truth for how this project works and how changes must be documented.**

---

## Project Overview

**Project Name:** Prompt Sensitivity — Adaptive Prompt Sensitivity in Complex LLM Tasks

**What this project does:**
This project investigates and mitigates **prompt sensitivity** in large language models — the phenomenon where semantically equivalent but lexically different prompts produce inconsistent outputs. It implements a complete research pipeline with 4 phases:

1. **GenSens (Dataset Generation)** — Generates benchmark datasets of semantically equivalent prompt paraphrases across 4 task domains (Summarization, Code, Creative, Dialogue) using LLaMA-3, filtered through SBERT similarity (cosine ≥ 0.82).

2. **PRI Benchmark (Prompt Robustness Evaluation)** — Quantifies how robust a model's outputs are using 6 core metrics (SMS, AUC-E, TRD, KPIG, PPL Variance, Branching Factor), 4 advanced metrics (Semantic TRD, Wasserstein SMS, Advanced KPIG, USD), plus quality metrics (Correctness, Hallucination, LLM-as-a-Judge).

3. **LL-PIRC (Logit-Lens Paraphrase-Invariant Residual Clamping)** — An inference-time intervention that detects the model's sensitive layer (ℓ*) via Logit Lens per-layer PPL analysis, identifies anchor token positions, and clamps their hidden states to a cross-paraphrase consensus representation.

4. **Statistical Evaluation** — Validates the intervention using Wilcoxon signed-rank tests on ROUGE-L variance and quality, plus ℓ* distribution analysis.

---

## Project Structure

```
PROMPT SENSITIVITY/
├── gensens/                          # Phase 1: Dataset generation
│   ├── scripts/
│   │   ├── generate_dataset.py       # Main dataset generation pipeline
│   │   ├── validate_dataset.py       # Post-generation validation
│   │   ├── paraphrase_generator.py   # LLaMA-3 paraphrase generation + SBERT
│   │   └── data_loaders.py           # Task-specific data loading
│   ├── data/                         # Generated datasets (JSONL)
│   └── README.md
│
├── prompt_robustness/                # Phases 2-4: Evaluation & Intervention
│   ├── main.py                       # PRI benchmark entry point
│   ├── evaluate.py                   # Phase 4 statistical evaluation
│   ├── experiment_baseline.py        # Baseline experiment runner
│   ├── experiment_pirc.py            # PIRC experiment runner
│   ├── config.yaml                   # Central configuration
│   ├── src/
│   │   ├── evaluator.py              # Core per-sample evaluation pipeline
│   │   ├── benchmark.py              # Multi-model benchmark orchestration
│   │   ├── pri_calculator.py         # PRI weighted harmonic mean
│   │   ├── prompt_generator.py       # d1/d2/d3 prompt perturbation levels
│   │   ├── model_interface.py        # HuggingFace model wrapper
│   │   ├── embeddings.py             # Sentence-BERT embeddings
│   │   ├── sms_metric.py             # Semantic Manifold Stability
│   │   ├── auc_e_metric.py           # Performance Elasticity (AUC-E)
│   │   ├── trd_metric.py             # Thematic Robustness Drift
│   │   ├── kpig_metric.py            # Key Point Information Gain
│   │   ├── ppl_variance.py           # Perplexity Variance
│   │   ├── branching_factor.py       # Token-level Branching Factor
│   │   ├── correctness_metric.py     # Correctness Score (CS)
│   │   ├── hallucination_metric.py   # Hallucination Score (HS)
│   │   ├── metrics_advanced.py       # Advanced metrics (TRD_sem, SMS_W, KPIG_adv, USD, ROUGE, BERTScore)
│   │   ├── llm_judge.py              # Flan-T5 LLM-as-a-Judge
│   │   ├── logit_lens.py             # Logit Lens per-layer PPL extraction
│   │   ├── sensitive_layer.py        # ℓ* detection (inflection / z-score)
│   │   ├── anchor_tokens.py          # Anchor token identification (percentile)
│   │   ├── pirc.py                   # PIRC clamped generation
│   │   ├── attribution_matrix.py     # Robustness diagnosis classification
│   │   ├── scoring_aggregator.py     # Alternative aggregation methods
│   │   ├── cache_manager.py          # Disk-based response caching
│   │   ├── table_utils.py            # Publication-quality table rendering
│   │   └── data_loader.py            # Dataset loading
│   ├── results/                      # Output CSVs, plots, JSON summaries
│   └── dashboard/                    # Streamlit dashboard
│
├── local_pirc_smoke_test.py          # Standalone LL-PIRC test (GPT-2)
├── run_local_smoke_test.sh           # 4-phase local test runner
├── CLAUDE.md                         # ← YOU ARE HERE (project instructions)
└── method.md                         # ← Methodology changelog (MUST maintain)
```

---

## ⚠️ MANDATORY RULE: Maintaining `method.md`

**Every time you make a change to any methodology, metric, formula, algorithm, pipeline logic, or evaluation procedure in this project, you MUST update `method.md`.**

### What counts as a "methodology change":

- Adding, removing, or modifying any metric (SMS, TRD, KPIG, PRI, etc.)
- Changing the PRI formula, weights, or composition
- Modifying the Final Score computation or weighting
- Changing the Logit Lens, sensitive layer detection, or anchor token algorithm
- Changing the PIRC clamping logic (alpha, soft/hard, hook behavior)
- Modifying the GenSens paraphrase generation strategy, SBERT threshold, or filtering
- Adding new evaluation steps, statistical tests, or baselines
- Changing prompt templates (d1/d2/d3) or perturbation strategies
- Modifying the LLM-as-a-Judge setup
- Adding or changing hallucination/correctness penalty formulas
- Changing the attribution/diagnosis classification logic
- Any config.yaml parameter changes that affect evaluation behavior
- Adding new models or changing model configurations
- Modifying the dataset structure or validation rules
- Bug fixes that change numerical outputs (not cosmetic fixes)

### How to update `method.md`:

1. Add a new entry at the **top** of the changelog (newest first)
2. Use this exact format:

```markdown
## [YYYY-MM-DD] — Brief Title of Change

**Files Modified:** `path/to/file1.py`, `path/to/file2.py`

**What Changed:**
- Bullet-point description of each specific change

**Why:**
- The reasoning and motivation behind this change

**Impact:**
- How this affects outputs, scores, or behavior
- Whether this is a breaking change
- Whether results need to be re-run

---
```

### What NOT to log in `method.md`:

- Pure cosmetic/formatting changes (comments, whitespace, variable names)
- README or documentation-only updates
- Dependency version bumps that don't change behavior
- Adding print statements or logging

---

## Key Formulas (Current)

Keep these in mind — if any of these change, `method.md` must be updated:

| Formula | Definition | Location |
|---------|-----------|----------|
| **PRI** | `0.40 × Consistency + 0.35 × CS + 0.25 × exp(−HS)` | `src/evaluator.py:180` |
| **Final Score** | `0.6 × PRI + 0.4 × Human_Score` | `src/evaluator.py:82` |
| **SMS** | `mean(cosine_sim) − 0.5 × Var(embeddings)` | `src/sms_metric.py` |
| **TRD** | `Var(lengths) / mean(lengths)²` | `src/trd_metric.py` |
| **KPIG** | `mean(|facts_i ∩ all_facts| / |all_facts|)` | `src/kpig_metric.py` |
| **ORI** | `(SMS + AUC-E + (1−TRD) + KPIG) / 4` | `src/evaluator.py:244` |
| **IFI** | `1 − (PPL_var + BF) / 2` | `src/evaluator.py:240` |
| **S(ℓ)** | `Var_k[mean_token_PPL at layer ℓ]` | `src/sensitive_layer.py` |
| **ℓ*** | `argmax_ℓ [S(ℓ) − S(ℓ−1)]` | `src/sensitive_layer.py` |
| **Anchor Selection** | Bottom 30% by `rank(mean_ppl) + rank(var_ppl)` | `src/anchor_tokens.py` |
| **Clamping** | `h[anchors] = α × mean_h + (1−α) × h_original` | `src/pirc.py` |
| **Variance Reduction** | `1 − mean(var_pirc) / mean(var_baseline)` | `evaluate.py` |

---

## Key Thresholds & Constants

| Parameter | Value | Location |
|-----------|-------|----------|
| SBERT similarity gate | ≥ 0.82 | `gensens/scripts/paraphrase_generator.py` |
| Hallucination penalty threshold | HS > 0.5 → PRI × 0.6 | `src/evaluator.py:183` |
| Short output penalty | avg_len < 12 → PRI × 0.85 | `src/evaluator.py:188` |
| Entity coverage penalty | coverage < 20% | `src/correctness_metric.py` |
| Anchor percentile | 30% | `src/anchor_tokens.py` |
| Layer scan start | 25% depth | `src/sensitive_layer.py` |
| Z-score fallback threshold | 2.0 | `src/sensitive_layer.py` |
| Wilcoxon significance level | α = 0.01 | `evaluate.py` |
| PRI weights | Consistency=0.40, CS=0.35, exp(-HS)=0.25 | `src/evaluator.py` |
| Final Score weights | PRI=0.60, Human=0.40 | `src/evaluator.py` |
| Dynamic weighting cap | trust_human ≤ 0.70 | `src/evaluator.py` |

---

## Coding Conventions

- **Python 3.9+** with type hints
- All metrics must be clamped to **[0, 1]** range
- Use `max(0.0, min(1.0, value))` for all score normalization
- HuggingFace Transformers is the model interface layer
- Sentence-BERT (`all-MiniLM-L6-v2`) for embeddings in PRI benchmark
- SBERT (`all-mpnet-base-v2`) for GenSens paraphrase filtering
- `config.yaml` is the central configuration; CLI args override it
- Disk-based caching via `CacheManager` (MD5 hash keys)
- Greedy decoding (`do_sample=False`) for all PIRC generation
- Forward hooks must handle both tuple and plain Tensor outputs (transformers v5.x compatibility)

---

## How to Run

```bash
# Phase 1: Generate dataset
cd gensens
python scripts/generate_dataset.py --task all --n_instances 200 --n_variants 8

# Phase 2: PRI Benchmark
cd prompt_robustness
python main.py

# Phase 3: PIRC Experiment
cd prompt_robustness
python experiment_pirc.py

# Phase 4: Statistical Evaluation
cd prompt_robustness
python evaluate.py

# Local smoke test (all phases, small scale)
./run_local_smoke_test.sh
```

---

## Remember

1. **Always read `method.md` first** before making any methodology changes — it contains the full history of what was tried and why.
2. **Always update `method.md`** after making methodology changes — future sessions depend on it.
3. **Never silently change a formula or threshold** — document the old value, new value, and reasoning.
4. **If a change affects numerical outputs**, note whether previously generated results need to be re-run.
