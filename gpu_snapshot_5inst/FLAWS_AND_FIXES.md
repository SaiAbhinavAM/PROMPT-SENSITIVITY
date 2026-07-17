# Prompt Sensitivity Benchmark: Comprehensive Flaws, Fixes & Improvement Plan

A complete audit of the PRI/ORI benchmark and the LL-PIRC mitigation pipeline as observed in the 5-instance run on the A100 instance (2026-05-29 to 2026-05-30). This document consolidates every bug, methodological flaw, and improvement opportunity discussed across the engineering session, organized by severity and ownership.

---

## Table of Contents

1. Executive Summary
2. Critical Bugs (blocking, fix in days)
3. Methodological Gaps (1-3 weeks)
4. Reproducibility & Infrastructure (1 week)
5. LL-PIRC Mitigation (parallel track, 2-3 weeks)
6. Dataset Quality (1 week)
7. Paper-Readiness Items (2-4 weeks)
8. Alternatives to Human Validation
9. Publication Paths
10. Sequencing Plan
11. What Is Already Done Correctly

---

# 1. Executive Summary

The pipeline is **engineering-sound** (single-source formulas, decoupled generate/score, reproducible runs) but **not yet scientifically defensible** as a standalone benchmark contribution. The decomposed-axis idea behind PRI is genuinely novel and supportable by the data (pairwise correlation among SMS, CS, and Faithfulness is below 0.15). Three engineering bugs (IFI saturation, anchor-selector NaN/inf handling, TRD collinearity) make several headline numbers misleading; all three are fixable in days. One structural issue (self-judging by the 70B AWQ model used both as judge and as a subject) will require a multi-judge panel before any reviewer will accept the benchmark. The best near-term publication path is a **method paper centered on LL-PIRC** using PRI/ORI as the evaluation harness, not a standalone benchmark paper.

Key empirical findings from the 5-instance run that drive this assessment:

- PRI and ORI are correctly de-collinearized (Pearson r = 0.148).
- The three PRI axes (SMS, CS, Faithfulness) are pairwise independent (|r| < 0.15).
- TRD inside ORI is essentially redundant with SMS (r = -0.98).
- CS is dominated by entity coverage (r = 0.96).
- The judge model rewards verbosity (judge vs avg_length r = 0.83); ROUGE-L punishes it (r = -0.85).
- All four subject models cluster within a PRI spread of 0.030 — beneath sampling noise at n = 19 evaluations per model.

---

# 2. Critical Bugs (blocking)

These must be fixed before any further benchmarking. None take more than a day.

## 2.1 Anchor Selector NaN/inf Handling

**File:** `prompt_robustness/src/anchor_tokens.py`

**Symptom:** Logged anchors include positions with `mean_ppl = inf` and `var_ppl = nan`. These are the *most* pathological tokens in the sequence, yet the algorithm picks them as the *most stable* anchors.

**Root cause:** `torch.argsort` on a tensor containing `inf` or `nan` places those values at the front of the rank order on most PyTorch versions. The "bottom 30 % by combined rank-sum" selection consequently grabs exactly the worst tokens. The percentile gate then forwards 76-122 broken positions for clamping at layer 9.

**Fix:**

```python
# In _extract_ppl_across_variants(), before computing ranks:
finite_mask = torch.isfinite(mean_ppl) & torch.isfinite(var_ppl)
# Replace non-finite with a sentinel that sorts to the END
LARGE = torch.finfo(mean_ppl.dtype).max
mean_ppl = torch.where(finite_mask, mean_ppl, torch.tensor(LARGE))
var_ppl  = torch.where(finite_mask, var_ppl,  torch.tensor(LARGE))
# Then compute argsort safely
mean_ranks = torch.argsort(torch.argsort(mean_ppl))
var_ranks  = torch.argsort(torch.argsort(var_ppl))
# Exclude non-finite positions from the candidate pool entirely
sorted_positions = [p for p in torch.argsort(stability).tolist() if finite_mask[p]]
```

**Impact:** This alone explains a large part of the LL-PIRC mitigation failure. Combined with the prefill-only hook fix already in `pirc.py`, the mitigation should produce coherent stabilized outputs (the latest snapshot already shows this — `pirc.json` produces real summaries with ROUGE-L preserved within 0.5 % of baseline).

## 2.2 IFI Saturation in scored_samples.csv

**File:** `prompt_robustness/src/evaluator.py` (where `ppl_var` and `bf` get written to rows)

**Symptom:** Every row of every model has `ifi = 1.000` in `scored_samples.csv`. Diagnostic_PRI = HM(Diagnostic_ORI, Diagnostic_IFI) collapses to Diagnostic_ORI because the IFI multiplicand is constant.

**Root cause:** The raw `ppl_var` and `bf` values in `ifi_metrics.json` are in their natural scales (PPL variance 0.07-0.13, BF 48-60 for Llama-8B). When written to `scored_samples.csv` they appear as 0.0 (rounded to printing precision but functionally zero), so `IFI = 1 - (0 + 0)/2 = 1.0`. Either the per-model normalization step is suppressed, or the natural values are being scaled to a near-zero range before storage.

**Fix:** Pull a per-model min-max normalization step. Before writing each row:

```python
# After all rows for one model are collected, normalize within model
m_ppl_var = max(0.001, max(rows.ppl_var))
m_bf      = max(0.001, max(rows.bf))
for r in rows:
    r.ppl_var_norm = r.ppl_var / m_ppl_var
    r.bf_norm      = r.bf      / m_bf
    r.ifi          = 1 - (r.ppl_var_norm + r.bf_norm) / 2
```

Or alternatively: drop IFI from the cross-model column entirely, document it as intra-model-only, and remove it from Diagnostic_PRI.

## 2.3 PRI vs Diagnostic_PRI Rank Anti-Correlation

**Files:** `scores.py`, paper draft

**Symptom:** Pearson correlation between arithmetic PRI and harmonic-mean Diagnostic_PRI is -0.13. Two metrics that ostensibly measure the same underlying construct (robustness) actively disagree on model ranking.

**Root cause:** Harmonic mean approaches the smallest component; arithmetic mean rewards balanced moderates. Llama-70B-AWQ and Llama-8B have one mediocre axis each (AUC-E, CS respectively), so harmonic punishes them while arithmetic rewards them.

**Fix:** Pick ONE for publication. Recommendation:

- **Use arithmetic PRI** as the primary ranking score (forgiving, intuitive, common practice).
- **Report Diagnostic_PRI in an appendix table** as a strict-evaluation companion, labeled as such.
- **Do not ship both as if they measure the same construct.**

## 2.4 CS Composition vs Reality

**File:** `src/correctness_metric.py`

**Symptom:** CS is documented as a blend of `length-adequacy + entity-coverage + cosine-to-reference`. In practice CS correlates with avg_coverage at r = 0.96 — CS is essentially coverage.

**Root cause:** Either the other terms have low variance on this dataset, or the blending weights are skewed toward coverage.

**Fix options:**

1. Re-balance the CS formula so length-adequacy and cosine each contribute at least 15 % of the variance.
2. Rename CS to `Coverage` and remove the marketing claim.
3. Replace CS with a composite that includes fluency (perplexity) and grammaticality (a small classifier).

The current state misleads the reader.

---

# 3. Methodological Gaps

## 3.1 Statistical Power

**Current:** n = 5 instances per task × 4 tasks = 20 unique instances. Per-model rows: 19 (one dialogue instance dropped). PRI spread across 4 models is 0.030; Wilcoxon p > 0.10 in Phase 4.

**Required:** ≥ 200 instances per task × ≥ 4 tasks = 800+ unique instances. Per-model rows: ≥ 200. Spread of 0.030 is then resolvable.

**Estimated cost:** 12-18 hours on a single A100, dominated by 70B-AWQ generation. The pipeline already runs this scale; only GPU time is needed.

## 3.2 Subject Model Diversity

**Current:** Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3, Llama-3.1-70B-AWQ. Two of these are Llama-family.

**Recommended additions:** Gemma-2-9B, Phi-3.5-mini, DeepSeek-V2.5-Lite, Yi-9B. Minimum for benchmark publication: 8 distinct model families.

**Why:** Reviewers cannot accept rankings derived from 4 models, 2 of which share an architecture lineage.

## 3.3 Judge Independence

**Current:** Llama-3.1-70B-AWQ serves as (a) a subject model, (b) the LLM-as-Judge for Human_Score, (c) the LLM-as-NLI for Faithfulness. This is a triple self-evaluation.

**Empirical evidence of bias:** Llama-70B-AWQ scores itself Human_Score = 0.866 (highest) and Faithfulness = 0.905 (high). Without a panel, this is indistinguishable from genuine quality.

**Fix:** Add a multi-judge panel. Minimum:

- **Mixtral-8x7B-Instruct** (different architecture, fits on A100).
- **Qwen2.5-72B-Instruct** (different training data).
- Optionally **GPT-4o** or **Claude Sonnet** via API for cross-family validation (~$20 for 200 samples).

Report:

- **Krippendorff's α** or **Fleiss' κ** across the panel.
- **Per-judge PRI ranking** with Spearman ρ between judges (should be > 0.8).
- **Final Human_Score as average of the panel**, not from a single judge.

## 3.4 AUC-E Definition

**Current:** `compute_auc_e_metric(reference, responses)` returns a single scalar with no documented underlying curve.

**Fix:** Publish the exact curve construction. What is on the x-axis? Perturbation strength? Variant index? What is on the y-axis? ROUGE-L? Cosine similarity? Without this, AUC-E is a magic number that reviewers cannot verify.

**Recommended:** Return the AUC plus the underlying curve points as a JSON column so the metric is auditable.

## 3.5 Cross-Embedder Validation

**Current:** SMS, CS-cosine, KPIG, and TRD_semantic all use BGE-large-en-v1.5. The benchmark's discriminative power may be an artifact of this embedder choice.

**Fix:** Run the full pipeline with at least two other embedders:

- `Alibaba-NLP/gte-Qwen2-7B-instruct` (7B, MTEB ~72) — known to require the patched native loader.
- `intfloat/e5-mistral-7b-instruct` (7B, different family).
- `mixedbread-ai/mxbai-embed-large-v1` (335M, similar to BGE-large).

Report **PRI ranking stability across embedders** (Spearman ρ across embedders should be > 0.7). If rankings change with embedder, the benchmark is reporting embedder choice as much as model robustness.

## 3.6 Weight Ablations

**Current:** PRI weights `0.40 / 0.35 / 0.25` are asserted, not justified.

**Fix:** Run a 50-row pilot with three weight settings:

- Current `(0.40, 0.35, 0.25)`
- Equal `(0.333, 0.333, 0.334)`
- Learned (fit weights to maximize correlation with downstream task accuracy)

Report which setting gives the best agreement with downstream metrics.

## 3.7 TRD Redundancy in ORI

**Current:** TRD vs SMS correlation is r = -0.98. ORI is effectively a 3-axis arithmetic mean masquerading as 4-axis.

**Fix options:**

1. Replace TRD with TRD_semantic (already computed, embedding-based, less length-coupled).
2. Reweight ORI to `0.5 * SMS + 0.5 * AUC-E` and report TRD / KPIG separately as diagnostics.
3. Keep TRD but document that on most paraphrase datasets it carries no marginal information beyond SMS.

Recommended: option 1.

## 3.8 Prior-Work Comparison

**Current:** No baseline comparison against existing prompt-sensitivity benchmarks.

**Required:** Map PRI/ORI rankings against PromptBench, RobustLR, and BIG-bench prompt-perturbation subsets on a shared subset of models. If Spearman ρ > 0.85, the contribution is the orthogonal decomposition (legitimately novel framing). If ρ < 0.5, the contribution is harder to defend without external validation showing PRI is right and they are wrong.

---

# 4. Reproducibility & Infrastructure

## 4.1 Dependency Pinning

**Current:** `requirements_gpu.txt` uses ranges (`vllm>=0.8.0`). The CUDA/torch/vllm compat debugging in this session proves wide ranges break.

**Fix:** Pin every dependency.

```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0
transformers==5.7.0
vllm==0.8.5.post1
autoawq==0.2.9
gptqmodel==7.0.0
sentence-transformers==3.0.0
datasets==4.8.5
```

## 4.2 Deterministic Seeding

**Current:** `do_sample=False` is set in `ModelInterface`, but numpy and python `random` are not centrally seeded.

**Fix:** Add a `set_global_seed(seed=42)` helper that seeds numpy, torch, torch.cuda, random in every `experiment_*.py` entry point.

## 4.3 Containerization

**Current:** The bootstrap on a fresh A100 took 30+ minutes and hit four separate issues (CUDA driver mismatch, missing gptqmodel, dataset script-loader deprecation, embedder swap).

**Fix:** Publish a Docker image with the pinned dependencies. Future reviewers and re-runners get a `docker run` workflow instead.

## 4.4 Crash-Safe Checkpointing

**Current:** `IncrementalCSVWriter` already provides crash-safe row writes for `responses.csv` and `scored_samples.csv`. But `baseline.json`, `pirc.json`, and `ifi_metrics.json` are written all-at-once at the end.

**Fix:** Stream JSON-Lines or per-article JSON files; aggregate into the final JSON only at the end. A crash in article 4 of 5 then preserves articles 0-3.

## 4.5 CI Smoke Test

**Current:** No automated regression test.

**Fix:** A 2-instance smoke test that runs phases 1-4 in ~10 minutes, gating PRs against changes that break the formulas. The smoke test should compare PRI / ORI of one fixed sample against a snapshot value.

---

# 5. LL-PIRC Mitigation

## 5.1 Verify Prefill-Only Hook at Scale

**Status:** Prefill-only fix already in `pirc.py`. At n = 5 instances it produces coherent outputs and 37 % variance reduction with quality flat. Needs validation at n = 200.

## 5.2 Article 1 Outlier

**Symptom:** PIRC variance went UP 23 % on article 1 while mean ROUGE-L also went up. The intervention moved that article's distribution in an unexpected direction.

**Likely cause:** Anchor-selector NaN/inf bug (§2.1) producing different anchor sets per article. Re-check after §2.1 fix.

## 5.3 Ablate ℓ\*

**Current:** ℓ\* = 9 for every article (CV = 0). Could be a real property of Llama-3.1-8B, or could be an artifact of `scan_start_fraction = 0.25`.

**Fix:** Force ℓ\* ∈ {6, 9, 12, 18, 24, 28} and run the mitigation at each. Show layer 9 is empirically optimal (lowest output variance while maintaining quality). If a deeper layer works equally well, the safety argument for clamping mid-network weakens significantly.

## 5.4 Ablate α

**Current:** α = 0.5 is asserted as "soft clamping" without ablation.

**Fix:** Sweep α ∈ {0.1, 0.25, 0.5, 0.75, 1.0} on a 50-article subset. Report variance reduction and ROUGE-L change for each. Argue for the chosen α from the Pareto front.

## 5.5 Baseline Comparisons

**Current:** LL-PIRC is compared only to a no-intervention baseline.

**Required for method paper:**

1. **Higher temperature smoothing** (T = 0.7, average K samples).
2. **Self-consistency voting** across paraphrases (modal answer).
3. **System-prompt stabilization** ("Be stable across rephrasings...").
4. **In-context learning** with K paraphrase examples shown.

If LL-PIRC isn't significantly better than at least the strongest of these baselines, the method paper does not ship.

## 5.6 Mechanistic Story for ℓ\* (optional but valuable)

**Question:** Why does Llama-3.1-8B have its sensitivity peak at layer 9 of 32? Probing literature suggests layers 6-8 are the "lexical → semantic transition" in similar-scale Llamas. A short interpretability section linking ℓ\* to a known structural transition makes the method paper much stronger.

---

# 6. Dataset Quality

## 6.1 GenSens Paraphrase Quality Audit

**Current:** SBERT cosine in [0.82, 0.98] is the filter. This is necessary but not sufficient for semantic equivalence — adversarial paraphrases can pass while changing meaning.

**Fix:** Take a 100-pair sample. Run a second NLI model (DeBERTa-v3-MNLI) on each pair in both directions. Report the fraction with bidirectional entailment as a paraphrase quality score. This becomes an appendix table.

## 6.2 Adversarial Paraphrase Subset

**Current:** All paraphrases are LLM-generated "easy" rewrites.

**Fix:** Add a hard subset:

- Typos and misspellings.
- Sentence reordering.
- Negation with double-negation.
- Hedged language ("might be" instead of "is").
- Politeness / formality shifts.

Report PRI separately on easy vs hard. This is a clean "robustness under attack" story.

## 6.3 Dialogue Task Density

**Current:** 5 instances generating only 2.8 variants per instance (the SBERT diversity filter is killing most candidates).

**Fix:** Either loosen the band for dialogue (`similarity_threshold` to 0.78), or generate 3× as many candidates and filter harder. Dialogue currently contributes proportionally less than other tasks to the benchmark.

## 6.4 Task Coverage

**Current:** Summarization + Creative + QA + Dialogue.

**Recommended additions for benchmark publication:**

- Code generation (HumanEval paraphrased docstrings).
- Math reasoning (GSM8K paraphrased questions).
- Multi-step reasoning (MMLU-Pro paraphrased).
- Translation (FLORES paraphrased prompts).
- Classification (SST-2, AG News paraphrased prompts).

Standard benchmarks cover 10-20 task families; you currently cover 4.

---

# 7. Paper-Readiness Items

## 7.1 Pin the Construct Definition

**Current:** "Prompt sensitivity" is asserted but not formally defined.

**Fix:** State precisely:

> "PRI measures the degree to which a model's response distribution remains stable under semantics-preserving prompt perturbations, conditioned on response quality vs a known-correct reference."

Every metric must map back to this. SMS measures stability. CS measures quality vs reference. Faithfulness measures grounding.

## 7.2 Prior-Work Comparison Table

**Required:** A single comparison table with one column per benchmark (PromptBench, RobustLR, BIG-bench prompt-perturbation, this work) and rows for: number of axes, dataset size, task coverage, judge independence, embedder dependence, statistical validation method, public leaderboard.

## 7.3 Limitations Section

Must include:

- Embedder dependence (single embedder family).
- Judge dependence (single judge family).
- Single dataset language (English).
- Single corpus domain (mostly news / encyclopedic).
- Statistical power at n = 200 (compute Cohen's dz, not just p-values).
- Self-evaluation bias if multi-judge panel not used.

## 7.4 Ethical Considerations

CNN/DailyMail is copyrighted. The fair-use rationale (academic research, transformative use, no commercial deployment) must be explicit.

## 7.5 Datasheet for the Dataset

NeurIPS Datasets & Benchmarks track requires a datasheet in the Gebru et al. format. ~5 pages covering motivation, composition, collection process, preprocessing, uses, distribution, maintenance.

## 7.6 Public Leaderboard Infrastructure

Benchmark papers without a live leaderboard are routinely rejected from D&B tracks. Options:

- GitHub repo with submission scripts and a JSON results format.
- HuggingFace Space with automated evaluation on submitted model URLs.
- Papers with Code leaderboard entry.

---

# 8. Alternatives to Human Validation

You cannot run a human annotation pilot. Five credible alternatives, ranked by ROI:

## 8.1 Existing Human-Labeled Datasets as Proxy Ground Truth

Score datasets that already have human labels and report correlation with PRI:

- **PAWS / PAWS-X:** human-labeled paraphrase pairs. High-PRI ⇒ human-labeled-paraphrase; low-PRI ⇒ adversarial.
- **STS-Benchmark:** sentence-pair similarity 0-5. SMS should rank-correlate with human similarity.
- **MT-Bench / Chatbot Arena Elo:** show your Final_Score / PRI Spearman-correlates with Arena Elo across 30+ open models.

Best ROI: **MT-Bench Elo correlation across 30 models** — a single defensible number.

## 8.2 Downstream-Task Performance Correlation

If high-PRI means "more robust," then high-PRI models should achieve higher accuracy on **paraphrased** versions of standard benchmarks:

- **Paraphrased MMLU:** 500 questions × 5 paraphrases each. High-PRI ⇒ low accuracy variance per question.
- **Paraphrased BIG-Bench Hard:** harder benchmark, more discriminating.
- **Paraphrased GSM8K:** numeric answers give deterministic agreement signal.

This is the **strongest external validation** because accuracy is unarguable.

## 8.3 Multi-Judge Panel (Already Covered in §3.3)

Replaces inter-annotator agreement with inter-judge agreement. Standard in HELM, MT-Bench, Chatbot Arena Hard.

## 8.4 Synthetic Ground Truth via Controlled Constructions

Generate paraphrase pairs where the ground truth is deterministic by construction:

- Active ↔ passive voice.
- Number-word rewrites (3 ↔ "three").
- Acronym expansion.
- Independent-clause reordering.
- WordNet synonym substitution.

You can generate 10 000+ pairs in an afternoon. Reviewers prefer this to human annotation when constructions are rigorous.

## 8.5 Self-Consistency on Math/Reasoning

Same question, K phrasings, measure exact-match agreement on the numeric answer:

- **GSM8K:** numeric agreement.
- **MATH:** numeric agreement.
- **CSQA:** multiple-choice agreement on chosen option.

Then `corr(PRI, agreement_rate)` is your validation.

---

# 9. Publication Paths

## Path A — Benchmark Paper (12-18 months)

**Venues:** NeurIPS D&B, EMNLP Findings, ACL Demo.

**Required before submission:**

- Scale to ≥ 200 instances × 4-10 tasks.
- ≥ 6 model families.
- Cross-embedder ablation.
- Multi-judge panel.
- Comparison vs PromptBench, RobustLR, BIG-bench.
- All §2 critical bugs fixed.
- Datasheet + public leaderboard.

**Risk:** High. Benchmark papers face heavy scrutiny.

## Path B — Method Paper Centered on LL-PIRC (6-9 months)

**Venues:** ICLR, ACL, EMNLP main track.

**Required:**

- Anchor-selector bug fixed.
- LL-PIRC ablations on ℓ\*, α, anchor-percentile.
- Strong baselines (temperature smoothing, self-consistency, ICL).
- 200-instance scale.
- The PRI/ORI benchmark is the evaluation harness, not the contribution.

**Risk:** Moderate. The contribution is the method. The benchmark only has to be adequate.

**Recommendation:** This is the highest-probability publication path.

## Path C — Position / Survey Paper (3 months)

**Venues:** TMLR, Findings, workshop.

**Required:**

- Clear argument for the 3-axis decomposition.
- Empirical demonstration of the collinearity audit.
- Survey of prior prompt-sensitivity work.

**Risk:** Low. But citation impact is lower.

---

# 10. Sequencing Plan

## Month 1

- Fix all §2 critical bugs (anchor selector, IFI saturation, PRI/Diagnostic_PRI choice, CS composition).
- Scale to 200 instances × 4 tasks on a fresh A100 (~12 hours).
- Run on 8 model families (Llama-8B, Llama-70B-AWQ, Qwen-7B, Mistral-7B, Gemma-9B, Phi-3.5-mini, DeepSeek-V2.5-Lite, Yi-9B).
- Add MT-Bench Elo correlation (no humans needed).
- Ship first arXiv preprint as a method paper around LL-PIRC.

## Month 2

- Multi-judge panel (Llama-70B + Mixtral-8x7B + Qwen-72B).
- Paraphrased-MMLU accuracy validation.
- Weight ablations (PRI and ORI).
- AUC-E definition.
- Submit to ACL / EMNLP method track.

## Month 3+

- Reproducibility infrastructure (Docker, pinned deps, CI smoke test).
- Dataset audits (GenSens NLI check, adversarial subset).
- Datasheet + leaderboard.
- Re-frame as a benchmark paper for NeurIPS D&B if reception is good.

## Skip For Now

- Mechanistic story for ℓ\*.
- Wasserstein-attribution divergence as a new axis.
- Cross-language evaluation.

---

# 11. What Is Already Done Correctly

Do not redo these.

- **Decoupled generation / scoring** with `--generate-only` + `--responses-csv`. The two-step pipeline is correct and crash-safe.
- **Single-source formulas** in `src/scores.py`. Mirrored by `reaggregate.py`.
- **Disk-resume safety** via `IncrementalCSVWriter` for `responses.csv` and `scored_samples.csv`.
- **Dual-pillar diagnosis matrix** in `compute_dual_pillar_diagnosis`. The 2×2 framing on (ORI ≥ 0.70, IFI ≥ 0.70) is intuitive.
- **Bootstrap CIs + Cohen's dz + Wilcoxon** in `evaluate.py`. The right stack of statistics for paired comparisons.
- **GenSens paraphrase pipeline** with LLM strategies + SBERT band [0.82, 0.98]. Working at 5-instance scale; ready for 200.
- **LL-PIRC infrastructure**: baseline + PIRC + evaluate phases. The scaffolding is sound.
- **Logit Lens fp32 dtype fix** in `logit_lens.py` (already applied).
- **PIRC prefill-only hook fix** in `pirc.py` (already applied).
- **GenSens DREAM dataset GitHub fallback** in `data_loaders.py` (already applied).
- **Multi-task merge logic** in `run_on_gpu.sh` (already applied).
- **CUDA-aware torch reinstall** via `ensure_torch()` in `run_on_gpu.sh` (already applied).
- **gptqmodel installation** for AWQ Marlin backend (already applied on remote; should be added to requirements).

---

# 12. Final Verdict

The pipeline has the right bones. The decomposed-axis framing is genuinely novel and survives the collinearity audit. The engineering is solid where it matters most (formulas in one place, crash-safe writes, decoupled scoring). The empirical claims are not yet defensible at n = 5 with three engineering bugs and a self-judging configuration, but every blocker is fixable in days-to-weeks.

**Recommended path: method paper on LL-PIRC using PRI/ORI as evaluation harness.** Submit to ACL / EMNLP within 6-9 months. Frame the benchmark as the evaluation infrastructure, not the contribution. Reserve the benchmark-paper route for after the method paper lands and external interest emerges.

If LL-PIRC mitigation does not survive the anchor-selector fix and ablations, fall back to **Path C: position paper on the 3-axis decomposition** as a principle of prompt-sensitivity measurement. That ships in 3 months and establishes priority on the framing.
