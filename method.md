# Methodology Changelog

> **This file tracks every change to the project's methodology, metrics, formulas, algorithms, and evaluation procedures.**
> New entries go at the **top** (newest first). Read this file to understand the full history of design decisions.

---

## [2026-05-26] — Single source of truth for PRI formula + central .env

**Files Modified:** new `prompt_robustness/src/scores.py`, `prompt_robustness/src/evaluator.py`, `prompt_robustness/reaggregate.py`, `prompt_robustness/src/config.py`, new `.env` / `.env.example`, `run_h100_eval.sh`, `start.sh`

**What Changed:**
- **`src/scores.py`** now holds the canonical composite formulas — `compute_pri` (0.40·consistency + 0.35·quality + 0.25·faithfulness, short-output gate), `compute_ori`, `compute_ifi`, `compute_final_static`, plus the weight constants. `evaluator.py` and `reaggregate.py` both import these, so the live and offline PRI can no longer drift. Verified numerically identical (flan-t5-base PRI = 0.590 before and after).
- **Central `.env`**: one file at repo root for `HF_TOKEN` + all model-role config (subjects, judge, embedder, faithfulness, decoding). `config.py` loads it via `find_dotenv(usecwd=True)`; `run_h100_eval.sh` and `start.sh` `source` it so gensens/vLLM/huggingface-cli also get `HF_TOKEN`. `.env.example` is the committed template; `.env` is gitignored.

**Why:**
- The PRI formula was duplicated in evaluator.py and reaggregate.py (drift risk flagged earlier). Keys/config were scattered across CLI flags and ad-hoc env vars.

**Impact:**
- No numerical change (refactor verified identical). Change the PRI composition in ONE place (`scores.py`) now. NOTE: `.env` ships with `MODEL_NAME` set to the 3 H100 subjects, so a bare local `python main.py` would try to load them — override `MODEL_NAME` for laptop runs.

---

## [2026-05-26] — Fault-tolerant CSV: incremental + resume + generate-only + H100 runner

> Guarantees that computed values are never lost on a crash, and that the single
> 80GB card only ever holds one big model at a time.

**Files Modified:** `prompt_robustness/src/csv_io.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/main.py`, new `run_h100_eval.sh`

**What Changed:**
- **`IncrementalCSVWriter`**: appends rows and `flush()`+`os.fsync()` after *every* write, so whatever is computed is already durable on disk. Header written once; opens in append mode when resuming.
- **Resume**: `existing_response_keys()` / `existing_scored_keys()` read the `(model,instance_id)` pairs already on disk; both the benchmark and the scorer **skip** them, so a re-run after a crash continues where it stopped (verified: "Scoring 0/2 groups, 2 already done").
- **`benchmark_models`** now writes responses + scored rows **per completed sample** (was a single bulk write at the very end) — a mid-run OOM no longer loses everything. Samples get a stable `instance_id` (`idx_N` for legacy datasets).
- **`generate_responses_to_csv` + `main.py --generate-only`**: loads ONLY the subject model, writes `responses.csv` incrementally, no scorer models — so an 8B subject never co-resides with the 70B judge on one 80GB card.
- **`score_from_responses_csv`** rewritten to write `scored_samples.csv` incrementally + resume; no longer rewrites the input responses.csv.
- **`run_h100_eval.sh`**: orchestrates Phase 1 (70B generator) → 2a generate-only per subject → 2b score-from-CSV (7B embedder + 70B judge/NLI) → 2c reaggregate, one big model per process.

**Why:**
- H100 time is paid and jobs are long; an OOM/crash at sample 150/200 must not discard the first 149. And a 70B + 8B + 7B cannot co-reside on 80GB, so generation and scoring must be separate processes.

**Impact:**
- No change to metric values. Behavioural: runs are now resumable and crash-safe; `results/responses.csv` and `results/scored_samples.csv` are written progressively. Verified locally end-to-end (generate-only → resume-skip → score → resume-skip → reaggregate).

---

## [2026-05-26] — P2 scoring instruments scaled to 7B/70B (env-flagged)

> Completes Phase 2 (evaluation) for the strict no-<2B H100 policy: the three
> scoring instruments (embedder, judge, faithfulness) can now run at 7B/70B.
> All are additive backends defaulting to the small local models, so laptop
> runs are unchanged.

**Files Modified:** `prompt_robustness/src/embeddings.py`, new `prompt_robustness/src/llm_backend.py`, `prompt_robustness/src/llm_judge.py`, `prompt_robustness/src/faithfulness_metric.py`

**What Changed:**
- **Embedder** (`EmbeddingHelper`): model is now `EMBEDDER_MODEL` (default `all-MiniLM-L6-v2`); set to `Alibaba-NLP/gte-Qwen2-7B-instruct` on the H100. `trust_remote_code=True` so 7B embedders load. Drives SMS / CS / TRD_semantic / KPIG_advanced.
- **Shared causal backend** (`llm_backend.ChatLLM`): loads a large instruct model ONCE (cached by id), vLLM-or-HF (lazy imports), batched greedy `chat_batch`. Judge and faithfulness reuse the same resident model.
- **Judge** (`llm_judge`): seq2seq ids (t5/flan/bart) keep the local T5 pipeline; any other id is treated as a causal instruct judge and routed through `ChatLLM` with a 1–5 scoring prompt. Set `JUDGE_MODEL` to a 70B-AWQ repo (+ `JUDGE_QUANTIZATION=awq_marlin`).
- **Faithfulness** (`faithfulness_metric`): `FAITHFULNESS_BACKEND=llm` uses LLM-as-NLI (entailment/neutral/contradiction → 1.0/0.5/0.0) via the shared backend (`FAITHFULNESS_MODEL`, default = `JUDGE_MODEL`); default `nli` keeps the independent deberta cross-encoder.

**Why:**
- Strict "no <2B anywhere" policy for the H100 evaluation, while keeping the judge/faithfulness model INDEPENDENT of the 8B subjects (avoids self-eval bias).

**Impact:**
- No change to default (laptop) behaviour — verified: routing booleans correct, default score-from-CSV + reaggregate unchanged. Big-model paths (gte-Qwen2-7B, 70B judge/NLI) are implemented but H100-verify only. **Switching the embedder rescales SMS/CS/TRD** → re-baseline. Env knobs: `EMBEDDER_MODEL`, `JUDGE_MODEL`, `JUDGE_QUANTIZATION`, `FAITHFULNESS_BACKEND`, `FAITHFULNESS_MODEL`, `LLM_BACKEND`.

---

## [2026-05-26] — CSV persistence: generate-once, re-evaluate cheaply

> Lets the expensive H100 generation run ONCE and be re-evaluated with little or
> no GPU. Three CSV layers, all round-trip losslessly (pandas QUOTE_MINIMAL, so
> article text with commas/quotes/newlines survives).

**Files Modified:** new `prompt_robustness/src/csv_io.py`, new `prompt_robustness/reaggregate.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/src/evaluator.py`, `prompt_robustness/src/data_loader.py`, `prompt_robustness/main.py`, `gensens/scripts/generate_dataset.py`

**What Changed:**
- **Layer A — dataset CSV:** `generate_dataset.py` now writes a flat `<name>.csv` (one row per variant) beside every JSONL. `csv_io.gensens_jsonl_to_csv()` exports existing JSONLs.
- **Layer B — responses.csv:** the benchmark persists every model×instance×variant prompt+response. New `score_from_responses_csv()` and `main.py --responses-csv PATH` re-score from it with **no generation model loaded** (verified: 0 model loads). `evaluator.evaluate_sample` accepts `precomputed_responses` and tolerates `model_interface=None` (PPL/BF — IFI-only diagnostics — are skipped in this mode).
- **Layer C — scored_samples.csv:** raw per-(model,instance) metric components (sms, auc_e, trd, kpig, cs, hs, faithfulness, human_score, avg_length, rouge…). New `reaggregate.py` recomputes PRI/ORI/IFI/Final and honest correlations from it with **zero models** — tune PRI weights (`--w-consistency/--w-quality/--w-faith`) and re-rank in seconds.
- Validators (`validate_dataset_csv` / `validate_responses_csv` / `validate_scored_csv`) check columns, non-null required fields, and contiguous variant indices.
- `data_loader.load_gensens_dataset` now also carries per-variant `strategies`.

**Why:**
- H100 time is costly; iterating on the *evaluation methodology* (PRI composition) should not require regenerating 70B paraphrases or 8B responses. This separates the three cost tiers so only the changed tier is recomputed.

**Impact:**
- No change to metric values — purely persistence + a no-GPU re-aggregation path. `reaggregate.py` mirrors the `evaluator.py` PRI/ORI/Final formulas; keep the two in sync if the composition changes. Verified locally end-to-end (dataset→CSV→benchmark→responses.csv/scored_samples.csv→score-from-CSV→reaggregate).

---

## [2026-05-26] — H100 Phase 1: diversity controls, vLLM generator, model registry

> First implementation step of the H100 scale-up (generator → 70B-AWQ, subjects → 3×8B,
> judge+faithfulness → 70B, embedder → gte-Qwen2-7B). This entry covers Phase 1 only.

**Files Modified:** `gensens/scripts/paraphrase_generator.py`, `gensens/scripts/generate_dataset.py`, `prompt_robustness/config.yaml`

**What Changed:**
- **Diversity controls** in `generate_paraphrases()` (the core fix for "stimulus too easy"):
  - Similarity **band** `[similarity_threshold, similarity_upper]` (default 0.82–0.98) — rejects both meaning-drift (too low) and trivial restatements (too high). Previously only a lower bound.
  - **Lexical-divergence floor**: reject a candidate whose token-Jaccard overlap with the base text, or with any already-accepted variant, exceeds `max_token_overlap` (default 0.85).
  - **Per-strategy cap** (`max_per_strategy`, default 2) so the kept variants span multiple strategy families.
  - Per-instance **diversity stats** recorded (`distinct_2`, `mean_pairwise_overlap`, `n_strategies`).
- **vLLM generator backend** (`--model vllm`, `--model-id`, `--quantization`): batched generation for H100; lazy `import vllm` so the laptop flan-t5 path is unaffected. Default 70B id is an AWQ-INT4 repo with `awq_marlin`.
- **`config.yaml` model registry** (`roles:`) assigning one model per role; fixed the `Meta-Llama-3` → `Llama-3.1` mismatch; annotated the legacy `back_translation` keys as unused by the current generator.
- CLI flags added to `generate_dataset.py` for all diversity knobs and the vLLM backend.

**Why:**
- The `<2B` pilot showed variants clustered at SBERT ~0.87 with low lexical diversity, so the consistency axis saturated (0.74–0.93) and PRI could not discriminate models. A bigger generator alone does not guarantee diversity; explicit band + lexical + strategy controls do.

**Impact:**
- Generated datasets will have **higher lexical diversity** and span more strategies; some near-duplicate variants that previously passed are now rejected (so a given instance may need more retries to reach `n_variants`). **Breaking** for dataset comparability; regenerate. Verified locally on the flan-t5 path (3 instances: mean pairwise token overlap dropped to ~0.39–0.49). The vLLM/70B path is implemented but must be verified on the H100 (vLLM not installable on the Mac).

---

## [2026-05-25] — Evaluation Rectifications R1–R7 (flaw fixes after <2B pilot)

> A `<2B`-model pilot (flan-t5-base/large, distilbart-cnn, bart-large-cnn; n=40) exposed
> structural flaws in the evaluation. Empirical evidence from that run:
> `Consistency(SMS)↔PRI r=0.949`, `SMS_Wasserstein↔SMS r=0.981`,
> SMS/CS/SMS_W all correlate with the judge at ~0.307–0.309 (one latent signal under three names),
> and `Final↔Human r=0.609` was circular (Final contains Human). The seven changes below
> address these. They are grouped under one date but are independent.

### R1 — Close the loop: evaluate real GenSens paraphrases

**Files Modified:** `prompt_robustness/src/data_loader.py`, `prompt_robustness/src/evaluator.py`

**What Changed:**
- `data_loader.load_dataset()` now routes `.jsonl` → new `load_gensens_dataset()`, mapping each GenSens record to `{input_text←metadata.article, reference_output←metadata.gold_summary, prompt_variants←[variant.full_prompt], topic_label←task}`.
- `evaluate_sample()` uses `sample["prompt_variants"]` when present; only falls back to the synthetic d1/d2/d3 templates for legacy `.json` datasets.

**Why:**
- Phase 1 generated SBERT-filtered paraphrase variants (the actual prompt-sensitivity stimulus) but `evaluator.py` discarded them and re-wrapped the article in 9 near-synonymous templates. Phases 1 and 2 were disconnected; "prompt sensitivity" was never measured on the generated paraphrases.

**Impact:**
- Robustness is now measured against the real paraphrase set. Variant count is now data-driven (e.g. 4) instead of a fixed 9. **Breaking** for cross-run comparability; re-run required.

### R2 — Deterministic decoding by default

**Files Modified:** `prompt_robustness/src/config.py`, `prompt_robustness/src/model_interface.py`

**What Changed:**
- `do_sample` now defaults to **False** (env `DO_SAMPLE=true` to opt back in); added `seed` (env `SEED`, default 42).
- `generate_responses()` only passes `temperature`/`top_p` when sampling, and seeds the RNG when it does.

**Why:**
- With `do_sample=True, temperature=0.7`, output variance across prompt variants was confounded with sampling noise — variance could not be attributed to prompt wording. (The repo even had a "Stochastic Luck" label acknowledging this but never controlled it.)

**Impact:**
- Output differences across variants now reflect prompt sensitivity, not RNG. Consistency-family metrics (SMS, AUC-E) shift upward and become interpretable. **Breaking**; re-run required.

### R3 — De-collinear PRI: SMS ⟂ Quality ⟂ NLI Faithfulness

**Files Modified:** `prompt_robustness/src/evaluator.py`, new `prompt_robustness/src/faithfulness_metric.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/src/analysis.py`, `prompt_robustness/src/table_utils.py`, `prompt_robustness/main.py`

**What Changed:**
- **Old:** `PRI = 0.40·Consistency(SMS) + 0.35·CS + 0.25·(1−HS)`, with CS further multiplied by cosine-KPIG and `exp(−1.5·HS)`. SMS effectively entered three times.
- **New:** `PRI = 0.40·Consistency(SMS) + 0.35·Quality(CS) + 0.25·Faithfulness`, three non-collinear axes.
- New **Faithfulness** = NLI entailment of each response by the source (`cross-encoder/nli-deberta-v3-small`, ~140M; `entail + 0.5·neutral`, averaged over variants), with graceful fallback to `1 − HS` if the model can't load.
- Canonical **KPIG** = reference-coverage (`kpig_advanced`), not the cosine version (which equaled SMS). Removed **SMS_Wasserstein** entirely (r≈0.98 with SMS). Reporting columns now show **Faithfulness** in place of SMS_Wasserstein.

**Why:**
- PRI was effectively a rescaled SMS (r=0.949) and three "metrics" measured the same quantity. Faithfulness adds a genuinely independent, semantically grounded axis.

**Impact:**
- PRI now has three separable components; SMS_Wasserstein and cosine-KPIG no longer reported/used. **Breaking**; re-run required.

### R4 — Count hallucination exactly once

**Files Modified:** `prompt_robustness/src/evaluator.py`

**What Changed:**
- Removed `cs_score *= exp(−1.5·HS)` and the `if HS>0.5: PRI *= 0.6` override, and the cosine-KPIG `cs_score *= (0.5 + 0.5·kpig)` multiplier.
- HS is now a **diagnostic only**; faithfulness is the single faithfulness signal in PRI.

**Why:**
- HS hit the score three times (CS multiplier, PRI `(1−HS)` term, and the `>0.5` override) — triple-penalizing one noisy bag-of-words signal.

**Impact:**
- CS and PRI no longer compounded by HS. Scores for high-HS rows rise relative to before. **Breaking**; re-run required.

### R5 — Honest validation + stronger judge over all variants

**Files Modified:** `prompt_robustness/src/llm_judge.py`, `prompt_robustness/src/evaluator.py`, `prompt_robustness/src/analysis.py`, `prompt_robustness/src/table_utils.py`

**What Changed:**
- Judge upgraded from `flan-t5-base` → `flan-t5-large` (env `JUDGE_MODEL`); new `llm_judge_mean()` scores **all** variants and averages (was `responses[0]` only).
- Correlation analysis **drops `Final_Score_vs_Human`** (circular: Final contains Human) and adds `Faithfulness_vs_Human`. `PRI_vs_Human` is the reported validity signal.

**Why:**
- The "quality" half ignored robustness (one variant), the judge under-discriminated (std 0.128), and `Final↔Human r=0.609` was self-correlation.

**Impact:**
- Human_Score spans all variants; reported correlations are non-circular. Re-run required.

### R6 — Semantic TRD canonical; PPL/BF intra-model only

**Files Modified:** `prompt_robustness/src/evaluator.py`

**What Changed:**
- Canonical `metrics["trd"]` (used by ORI and attribution) is now embedding-based `trd_semantic`; length-based TRD kept as `trd_length` diagnostic.
- PPL variance & branching factor documented as **intra-model diagnostics** feeding IFI only — never the cross-model PRI ranking.

**Why:**
- Length-variance TRD called two semantically different but equal-length summaries "no drift". PPL/BF are model-internal and not comparable across seq2seq vs causal architectures.

**Impact:**
- ORI now reflects semantic drift. No change to PRI (PPL/BF were already outside it). Re-run required for ORI.

### R7 — Validator selects the newest dataset file

**Files Modified:** `gensens/scripts/validate_dataset.py`

**What Changed:**
- File selection per task now sorts candidates by mtime (newest first) instead of taking an arbitrary `os.listdir` entry.

**Why:**
- With multiple `gensens_<task>_*var.jsonl` files present, the validator silently validated a stale file instead of the freshly generated one.

**Impact:**
- Validation now reflects the latest run. No numerical impact on metrics.

---

## [2026-05-25] — PRI Formula: Replace exp(−HS) with Linear (1 − HS)

**Files Modified:** `prompt_robustness/src/evaluator.py`

**What Changed:**
- **Old:** `PRI = 0.40 × Consistency + 0.35 × CS + 0.25 × exp(−HS)`
- **New:** `PRI = 0.40 × Consistency + 0.35 × CS + 0.25 × (1 − HS)`

**Why:**
- The `exp(−HS)` term created a guaranteed floor: even with Consistency=0 and CS=0, PRI ≥ 0.25 × exp(0) = 0.25
- This caused **ceiling compression**: 50% of rows had PRI ≥ 0.95 in the evaluation dataset
- The linear `(1 − HS)` term allows PRI to reach 0.0 when all components are bad
- The HS > 0.5 penalty (`pri *= 0.6`) and short-output penalty (`pri *= 0.85`) are kept unchanged

**Impact:**
- **Breaking change** — PRI scores will be lower for models with high hallucination rates
- All previous PRI results should be re-run to get comparable numbers
- New PRI range: theoretically [0.0, 1.0] with no guaranteed floor

---

## [2026-05-25] — Hallucination Metric: Add Common Word Exclusion & Reference Grounding

**Files Modified:** `prompt_robustness/src/hallucination_metric.py`, `prompt_robustness/src/evaluator.py`

**What Changed:**
- Added a set of ~200 common English words (`_COMMON_WORDS`) that are excluded from the hallucination check: connectors, common verbs, adverbs, generic nouns, pronouns
- Added optional `reference` parameter: words from the gold reference are now part of the grounding set and are not flagged as hallucinated
- `evaluator.py` now passes `reference=reference` to the hallucination metric
- Convention unchanged: HS = 0.0 (no hallucination, good) → 1.0 (fully hallucinated, bad)

**Why:**
- The old metric flagged standard English words like "becomes", "reaching", "expected", "dramatically" as hallucinations simply because they didn't appear in the input text
- For summarization, the model's output may use words from the reference summary that aren't in the source article — these are valid, not hallucinated
- This caused a systematic overestimation of hallucination (mean HS was ~0.6 for faithful summaries)

**Impact:**
- HS scores will be **significantly lower** (closer to 0.0) for faithful summaries
- Models that previously appeared to have high hallucination but were actually using common English will see corrected scores
- All previous HS and HS-dependent scores (CS, PRI, Final Score) should be re-run

---

## [2026-05-25] — SMS Metric: Fix Zero-Score for Identical Outputs

**Files Modified:** `prompt_robustness/src/sms_metric.py`

**What Changed:**
- L2-normalize embeddings **before** computing the diversity penalty
- Added a near-zero variance fast-path: if `diversity_penalty < 1e-6`, return raw cosine similarity without applying the penalty

**Before:** Raw embeddings → diversity penalty could be large (e.g., 2.5) due to embedding magnitude → `SMS = 0.95 - 0.5 × 2.5 = -0.3 → clamped to 0.0`

**After:** Normalized embeddings → diversity penalty on unit sphere (max ~2.0) → near-identical outputs give `penalty ≈ 0.0` → `SMS ≈ 0.999`

**Why:**
- 29/60 rows (48%) had SMS = 0.0 in the evaluation dataset
- 24 of these were cases where the model produced **perfectly identical** outputs for all 5 prompt variants — the most robust behavior possible — yet scored 0.0 for consistency
- The raw embedding L2 norms varied by model (range 3–15), making the diversity penalty scale-dependent and unreliable

**Impact:**
- **Breaking change** — SMS scores will increase substantially for models with consistent outputs
- Since SMS feeds directly into PRI as the Consistency term (weight 0.40), PRI will also shift
- Previous rankings may change; re-run required

---

## [2026-05-25] — Attribution Matrix: Fix Metric Polarity

**Files Modified:** `prompt_robustness/src/attribution_matrix.py`

**What Changed:**
- Lower-is-better metrics (`trd`, `ppl_var`, `bf`) are now inverted (`1 - v`) before applying thresholds
- Thresholds relaxed: True Robustness from `all ≥ 0.7` to `all ≥ 0.65`; Evaluation Artifact from `any < 0.3` to `any < 0.2`
- Stochastic Luck now uses spread-based detection: `avg ≥ 0.55 AND max-min spread > 0.35`
- Knowledge Boundary is now the default fallback (was unreachable before)

**Why:**
- The old code treated all metrics as higher-is-better, but TRD = 0.0 (low drift = good) was being counted as "below threshold" and preventing True Robustness classification
- Only 2 of 4 diagnosis categories ever fired (True Robustness and Stochastic Luck)
- Evaluation Artifact and Knowledge Boundary were unreachable

**Impact:**
- Diagnosis labels will change for many samples
- All 4 categories should now fire appropriately
- No impact on PRI scores (attribution is a post-hoc label, not a score component)

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
