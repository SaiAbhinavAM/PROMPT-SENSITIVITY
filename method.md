# Methodology Changelog

> **This file tracks every change to the project's methodology, metrics, formulas, algorithms, and evaluation procedures.**
> New entries go at the **top** (newest first). Read this file to understand the full history of design decisions.

---

## [2026-08-09] — Robustness validation script + cross-checks on the 30-article run (all passed)

**Files Added:** `gensens/crossed/scripts/validate_robustness.py`, `results_30article_bestscorers/VALIDATION.md`, `.../results/validation.json`

**What Changed:**
- New pure-CPU `validate_robustness.py` (operates on `cell_metrics_scored.jsonl` + the seed dataset; no GPU) running four post-hoc cross-checks and writing `validation.json`:
  1. **Length confound** — corr(Sensitivity, mean_output_len) and dimension ranking before/after regressing out length.
  2. **Split-half replication** — N random article half-splits; mean Spearman of the two independent dimension rankings.
  3. **Permutation test** — assumption-free significance of the dimension effect at the SEED level (shuffle dimension labels; no normality assumption).
  4. **Paraphrase equivalence** — dataset-wide SBERT similarity + bidirectional NLI entailment + `nli_passed` rate + token overlap.

**Why:**
- Stress-test the dimension finding against confounds (length), reliability (split-half), a stricter significance test (permutation, since the composite is rank-normalized ≈ uniform, violating the mixed model's normality assumption), and the foundational premise that the paraphrases are meaning-equivalent.

**Impact:**
- No change to any score. All four checks **PASSED** on the 30-article SOTA-scorer run: length ranking-Spearman **0.97** (not a length artifact), split-half **mean Spearman 0.87** (replicates within-study), permutation **p=0.022** (significant with no distributional assumption), paraphrase equivalence **NLI 0.98/0.97 fwd/bwd, 100% nli_passed, token overlap 0.36** (equivalent meaning, distinct wording).
- **Reporting note:** the assumption-free permutation **p=0.022** is the most defensible significance value (the parametric mixed-model p=0.0036 assumes normality the rank-normalized composite doesn't have). Details in `VALIDATION.md`. The one un-checkable item remains **cross-model replication** (single model/task).

---

## [2026-08-09] — Production 30-article run with SOTA scorers (result: dimension finding held & strengthened)

**Files Modified:** `gensens/crossed/run_crossed_h100.sh`; **Added:** `gensens/crossed/results_30article_bestscorers/`

**What Changed:**
- Added a `FAITH_WHOLE_SUMMARY` env passthrough to the runner. MiniCheck per-sentence scoring is ~4x slower (each summary → many doc/claim pairs; ~4.5 h for the faithfulness pass alone at 30 articles on an A30), so the production run used **whole-summary** mode (`FAITH_WHOLE_SUMMARY=1`): each summary scored as ONE claim by the same SOTA MiniCheck deberta-v3-large model. Minimal granularity loss on 3–4 sentence news summaries; the faithfulness pass dropped to ~40 min.
- Executed the full crossed-design 30-article benchmark on a Jarvis Labs A30, **reusing the existing 8,970 responses** (`--phases 2,3,4,5,6`, generation skipped — identical greedy outputs, saved ~1 h GPU) with MiniCheck faithfulness + BERTScore correctness + the PPL pass ON (full 6-metric composite incl. the de-confounded `pc_stab_cv`). Results committed under `results_30article_bestscorers/`.

**Why:**
- Validate the v2 composite + best scorers on the real crossed grid, and confirm the dimension finding is not an artifact of the (previously weaker) faithfulness/correctness scorers.

**Impact:**
- **The dimension finding HELD and strengthened.** Mixed-effects dimension test **p=0.0036** (was 0.011 with the old-scorer v2), dimension η²=19.3% vs pool 1.7% (~11×), seed-aggregated Kruskal now significant (p=0.036, was 0.063), robust to quality (Spearman 0.94) and to dropping single-seed dimensions (η² 0.187 vs 0.193, p=0.004). Ranking unchanged (Meta-reflection / Question Form / Theme Isolation most; Output Format / Constraint Based least). Sensitivity by pool A=0.485 / B=0.535 ≈ old-scorer v2 (0.488/0.529) — robust to the scorer upgrade. The finding now survives THREE scorer configurations (legacy → v2 → v2+SOTA).
- **PRI absolute values dropped** (Pool A 0.756→0.579, B 0.713→0.539): expected, because BERTScore-rescaled correctness and MiniCheck faithfulness are more calibrated/stricter than the old SBERT-cosine + weak-NLI. Not a regression — the Sensitivity headline is unaffected. PRI values are only comparable within the same scorer set.

---

## [2026-08-09] — Best-in-class scorers: MiniCheck faithfulness + BERTScore correctness (pre-GPU-run quality upgrade)

**Files Modified:** `gensens/crossed/scripts/compute_cell_metrics.py`, `run_crossed_h100.sh`, `requirements_crossed.txt`

**What Changed (two scorer upgrades chosen to maximize result validity before a paid production run):**

1. **Faithfulness → MiniCheck (default), NLI kept as fallback (`compute_cell_metrics.py`).** Replaced the `deberta-base` NLI cross-encoder (lightweight, the weakest link) with **MiniCheck** (`lytang/MiniCheck-DeBERTa-v3-Large`, EMNLP 2024) — a dedicated fact-checking model, SOTA on the LLM-AggreFact leaderboard, deterministic, and it **chunks long documents internally**. This eliminates the premise-truncation hack and the "output too long → default neutral 0.5" fallbacks entirely. Multi-sentence summaries are scored **per sentence** (MiniCheck's recommended usage, via a dependency-free regex splitter) and averaged; empty generations default to neutral 0.5. New flags: `--faithfulness_backend {minicheck,nli}` (default `minicheck`), `--faithfulness_model` (default `deberta-v3-large`), `--faith_whole_summary`, `--minicheck_cache_dir`. If the `minicheck` package can't be imported on the box, the code logs a clear install hint and **auto-falls back** to the NLI backend (`compute_faithfulness_nli`, the previous logic) — the run never fakes or crashes on a missing optional dep.

2. **Correctness-vs-gold → BERTScore (`compute_cell_metrics.py`).** The `cs` metric was `0.5·SBERT-cosine-to-gold + 0.5·entity-coverage`. BERTScore was already being computed but **discarded**. `cs` now = `0.5·BERTScore-F1-to-gold + 0.5·entity-coverage` — BERTScore's token-level match to a reference is exactly what it was designed for, and this gives the previously-wasted BERTScore pass a real purpose. SBERT is now used **only** for output-to-output `sms_drift` (its designed purpose), removing the double-use. Added `--bertscore_model` (default `roberta-large`, can be `microsoft/deberta-xlarge-mnli`) and `rescale_with_baseline=True` by default (spreads BERTScore from the compressed ~0.85–0.95 band to a usable [0,1] range); `--no_bertscore_rescale` to disable. Gold-summary SBERT embeddings are no longer computed (dead work removed).

**Why:**
- For a *sensitivity* benchmark, the best scorer is one that is both accurate AND deterministic — an LLM-judge would inject its own variance into the very quantity being measured, so it was deliberately avoided. Faithfulness was the least-reliable metric and feeds the composite; MiniCheck is the strongest deterministic option and also fixes long-article handling. BERTScore was redundant with SBERT for drift but is the right tool for reference-based correctness, so it was moved there instead of deleted.

**Impact:**
- Changes `faith_mean`/`faith_var` (→ `faith_cv`) and `cs_mean`/`cs_var` (→ `cs_cv`), therefore the Sensitivity composite, PRI, and diagnosis — a quality improvement, **re-baseline on the next run**. These do NOT retrofit onto existing `cell_metrics.jsonl` (per-output faith/BERTScore were never stored), so they take effect only when Phase 2 is re-run on the GPU; the completed 30-article result is unchanged and predates this.
- New dependency: `minicheck` (git install, added to `requirements_crossed.txt` with a note to install separately if it conflicts with the pinned transformers/torch — the NLI fallback covers that case). Verified locally: all scripts parse, `--help` exposes the new flags, and the custom logic (regex sentence-splitting, per-output averaging, whole-summary mode, empty→0.5, flat doc/output alignment) passed injected-mock unit tests. The MiniCheck/BERTScore model paths themselves are GPU-only and must be validated in the pre-flight smoke run (`N_ARTICLES=2 --phases 1,2`) before the full run — the smoke will surface any install/API/OOM issue in minutes.

**Follow-on (same day) — PPL pass re-enabled by default + Phase-2 memory hygiene (`run_crossed_h100.sh`, `compute_cell_metrics.py`):**
- `SKIP_PPL_ENTROPY` default flipped **1 → 0**: the production benchmark now RUNS the PPL/branching-factor pass, yielding the full 6-metric composite and activating the de-confounded `pc_stab_cv` (2026-08-09 composite-v2 entry). This reverses the 2026-07-14 "retired by default" decision for the production run (still overridable with `SKIP_PPL_ENTROPY=1` for a faster 4-metric run). No metric math changes — it only controls whether ppl_var/pc_stab_var are computed.
- Because Phase 2 now stacks SBERT + MiniCheck + BERTScore's roberta-large + the 8B PPL model on one card, each scorer is explicitly `del`-eted and `torch.cuda.empty_cache()`-d in its own scope before the next loads (`_empty_gpu_cache`; the earlier draft freed a function *parameter*, which does not drop the caller's binding — fixed). Pure infra: no numerical change, reduces OOM risk on the (most OOM-prone) PPL step.

---

## [2026-08-09] — Sensitivity composite v2: de-confounded quality spread, lexical drift, absolute anchor, first-class quality-residual

**Files Modified:** `gensens/crossed/scripts/common.py`, `aggregate_scores.py`, `summary_report.py`

**What Changed (five measurement fixes found by auditing the completed 30-article result; all pure-CPU, re-run on existing `cell_metrics.jsonl` with no re-inference):**

1. **#6 De-confound the quality spreads (`aggregate_scores.py`, `common.py`).** `cs_var` / `faith_var` were RAW variance of a bounded [0,1] score, which is mechanically entangled with the score's mean — on the 30-article data `faith_var` vs `faith_mean` was Spearman **+0.45** (and `cs_var` vs `cs_mean` −0.21), while `ppl_var` already used std/mean. The composite now uses **coefficient of variation** `cs_cv = std/cs_mean`, `faith_cv = std/faith_mean` (new `common.coeff_of_variation`, mean-guarded, capped at 5.0), derived from the stored per-cell var+mean. This removes the level dependence and makes cs/faith consistent with ppl_var.

2. **#5 Broaden beyond embedding space (`aggregate_scores.py`).** SBERT drift (`sms_drift`) and lexical drift (`rougeL_var`, already computed but unused) correlate only **Spearman 0.15** on the 30-article data — the old composite was nearly blind to pure word-choice variation. `rougeL_var` is now a first-class spread component. New composite: `Sensitivity = 0.25·sms_drift + 0.20·rougeL_var + 0.15·cs_cv + 0.15·faith_cv + 0.125·ppl_var + 0.125·pc_stab_var` (rank-normalized per component, weights renormalized when any metric is absent — verified on the `skipppl` run, which correctly renormalizes to 4 metrics: sms 0.333 / rougeL 0.267 / cs_cv 0.20 / faith_cv 0.20). SBERT's dominance of the ranking dropped from Spearman 0.755 → **0.703** (composite now genuinely multi-signal). Legacy composite kept behind `--composite legacy` for ablation/reproducing prior runs.

3. **#8 First-class quality-adjusted "pure spread" (`aggregate_scores.py`, `common.py`).** New `common.ols_residual` (pure numpy, no statsmodels) computes `sensitivity_resid` = Sensitivity with `cs_mean + faith_mean` regressed out; every per-seed / per-pool / per-dimension aggregate now reports `sensitivity_resid_mean`. Raw-vs-residualized dimension ranking is **Spearman 0.96** (was 0.93), confirming the finding is not a quality artifact.

4. **#7 Absolute, run-independent anchor (`aggregate_scores.py`, `summary_report.py`).** The rank-normalized Sensitivity is relative WITHIN a run (0.5 = median cell), so it is not comparable across runs/models. Aggregates now also report raw `sms_drift_mean` (1 − mean pairwise cosine), which has a fixed meaning and IS cross-run comparable; the report labels the composite as relative and prints the anchor + residual columns.

5. **#9 Pool-B constraint-satisfaction caveat (`summary_report.py`).** Added an explicit note that Pool-B prompts impose a constraint (no proper nouns / theme isolation), so some cross-paraphrase variation is legitimate degrees of freedom in satisfying the constraint, not model fragility — Pool-B sensitivity is an upper bound on fragility.

6. **Finish the de-confound for branching-factor spread (`compute_cell_metrics.py`, `aggregate_scores.py`).** `pc_stab_var` was the last spread metric still using RAW variance (of an unbounded quantity, so entangled with its magnitude). `compute_cell_metrics.py` now also stores `pc_stab_mean`, and the v2 composite uses `pc_stab_cv = std/mean` (via `common.coeff_of_variation`). Runs generated BEFORE this field existed have no `pc_stab_mean`, so `aggregate_scores.py` falls back to the raw variance value — verified the 30-article `Sensitivity` is byte-for-byte identical (Pool A 0.4878 / Pool B 0.5285). The de-confounded `pc_stab_cv` activates automatically on the next run that regenerates `cell_metrics.jsonl` (needs the GPU PPL pass).

7. **Log the thin-dimension robustness check (`dimension_analysis.py`, `summary_report.py`).** A dimension backed by a single seed conflates "dimension" with one specific prompt (only Meta-reflection, n_seeds=1). `dimension_analysis.py` now emits a `dimension_robustness` block that re-computes η² and the mixed-model LRT after dropping <2-seed dimensions, plus the surviving-dimension ranking Spearman; the report prints it. **Result: dropping Meta-reflection leaves η²=0.172 (vs 0.176 all) and mixed-model p=0.011 — the dimension effect is robust to thin dimensions.**

**Why:**
- These are measurement-validity flaws sitting on top of a conceptually sound design. Raw-variance-of-a-bounded-score confounds spread with quality (#6); a single embedding metric misses lexical variation (#5); a purely relative composite cannot be compared across runs (#7); the sensitivity↔quality entanglement (r=−0.24) needed to be shown removable, not just asserted (#8); and constrained prompts inflate apparent fragility (#9).

**Impact:**
- Sensitivity composite VALUES change (new formula + de-confound), but the **headline is unchanged and now better supported**: dimension ranking is **Spearman 0.95** vs the legacy composite (Question Form / Theme Isolation / Meta-reflection most sensitive; Output Format / Constraint Based least). Mixed-effects dimension test stays significant: **LRT χ²=27.4, df=13, p=0.011** (was p=0.006). η²: dimension 17.6% vs pool 1.1%. Pool A/B seed-level Mann-Whitney stays non-significant (p=0.22). Re-baseline any downstream artifacts to the v2 numbers.
- Re-ran the full analysis chain (aggregate → diagnosis → dimension_analysis → significance → summary_report) on BOTH `results_30article_full/` and `results_30article_skipppl/`; before-state backed up to `/tmp/preV2/`. No GPU / re-generation needed — `cell_metrics.jsonl` already stored the per-cell var, mean, and rougeL_var. **Still outstanding (unchanged by this entry):** thin per-dimension seed support (Meta-reflection = 1 seed), single-model/single-task scope, and B12's missing variant.

---

## [2026-07-14] — Dimension-first analysis: mixed-effects model, residualized sensitivity, seed-level dimension test, PPL retired by default

**Files Modified:** `gensens/crossed/scripts/significance.py`, `summary_report.py`, `run_crossed_h100.sh`; **Added:** `gensens/crossed/scripts/dimension_analysis.py`

**What Changed (four analysis-rigor improvements, all pure-CPU, run on the completed 30-article result):**

1. **Dimension effect is now the headline; pool demoted (`significance.py`, `summary_report.py`).** The 30-article result showed the instruction DIMENSION explains ~10x more sensitivity variance than the Pool A/B split (η² 0.18 vs 0.019). `significance.py` now emits a `dimension_effect` block: η² for dimension and pool, and a **seed-aggregated Kruskal-Wallis** across dimensions (each of the 50 seeds contributes one value = its mean sensitivity), which is the valid non-pseudoreplicated dimension test (mirrors the seed-level pool fix). The report leads with the dimension effect; the pool comparison is now a clearly-labeled secondary section.

2. **Mixed-effects model (`dimension_analysis.py`, new).** Fits `Sensitivity ~ C(dimension)` with CROSSED random intercepts for seed and article (statsmodels variance-components MixedLM, ML), plus a likelihood-ratio test vs an intercept-only model. This is the properly-powered dimension test — it uses all 1,500 cells while correctly modeling that cells are nested in seeds and crossed with articles. **Result: LRT χ²=29.3, df=13, p=0.0059 — the dimension effect IS significant** once seed/article structure is modeled. Variance components: seed=0.0029, article=0.0071, residual=0.0182 (article content matters more than which specific seed). The three tests now bracket the truth honestly: per-cell Kruskal p=1.5e-47 (pseudoreplicated, overstated), seed-aggregated Kruskal p=0.051 (valid but underpowered — 50 seeds / 14 dims), mixed model p=0.0059 (correct).

3. **Quality-residualized "pure sensitivity" (`dimension_analysis.py`).** Adds `sensitivity_resid` = Sensitivity with the quality covariates (CS_mean, faith_mean) regressed out via OLS, addressing the mild entanglement found at n=30 (r(Sensitivity, CS_mean) = −0.235). The dimension ranking is **Spearman 0.93** between raw and residualized Sensitivity, confirming the dimension finding is not a quality artifact. Writes `dimension_analysis.json`.

4. **PPL/branching-factor retired by default (`run_crossed_h100.sh`).** `ppl_var` and `pc_stab_var` correlate r=0.68 (mutually redundant), the ~30-min PPL pass did not change the 30-article conclusion (3-metric and 5-metric composites agree), so `SKIP_PPL_ENTROPY` now defaults to **1**. Set to 0 only when the full 5-metric composite is specifically wanted. `dimension_analysis.py` is also wired into Phase 5 of the runner (non-fatal if statsmodels is absent).

**Why:**
- The raw per-cell tests overstate significance (pseudoreplication), the composite is mildly quality-entangled, and the coarse Pool A/B framing hides the real (dimension-level) signal. These four changes report the finding in its honest, defensible form and remove an expensive metric pass that earns nothing.

**Impact:**
- No change to any per-cell metric or the Sensitivity composite values — this is analysis/reporting on top of existing scores, plus a new `dimension_analysis.json` and a re-framed `summary_report.md`. New dependency for the mixed model: `statsmodels` (pip; CPU-only). The substantive result is unchanged and now better supported: **prompt sensitivity is driven by instruction dimension (mixed-model p=0.006, survives quality-adjustment), not the Pool A/B split.** Known limitation surfaced by the new `n_seeds` column in the report: several dimensions have thin seed support (Meta-reflection = 1 seed), so single-seed dimensions conflate the dimension with one specific prompt — adding seeds to thin dimensions is the recommended next dataset improvement.

---

## [2026-07-09] — Strip instruction-echo preamble from outputs before content/quality scoring

**Files Modified:** `gensens/crossed/scripts/compute_cell_metrics.py`

**What Changed:**
- Added `strip_preamble()` and applied it to the model outputs before computing the **content/quality** metrics in the crossed-design pipeline. Instruction-tuned Llama-3.1 frequently prefixes summaries with a meta line that echoes the instruction — e.g. "Here is a summary of the article in 3-4 sentences:", "Here's a 3-4 sentence summary of the news article:", "**Main Events and Key Details:**", "Here are the key points:" — and the exact wording **varies across paraphrases** (some variants emit it, some don't; the phrasing differs when they do). On the 5-article pilot this preamble appeared in **~49% of the 1,495 outputs**.
- The stripper is conservative: it removes only a short (≤20-word) leading clause ending in a colon that is clearly a meta-announcement (opens with here is/here's/here are/these are/below is/the following/sure/certainly/… **or** contains a summary-ish keyword: summary/synopsis/rundown/overview/key points/main events/takeaways/…), up to 3 stacked header+announce layers, and **never** empties an output (falls back to the trimmed original). Validated on all 1,495 pilot outputs: 49% cleaned, 0 emptied, 0 strips > 220 chars, and the only "residual" meta-looking starts were genuine first-person summaries ("I'll never forget the day…") correctly left untouched.
- **Which metrics use which text:** cleaned output feeds SMS (SBERT embeddings), CS (cosine + entity coverage), ROUGE-L, BERTScore, faithfulness (NLI hypothesis), and `mean_output_len`. The **raw** generation is kept for the PPL / branching-factor pass, which measures the model's token-level confidence over what it actually produced. Each row now carries `output_clean` alongside `output`; a warning records how many outputs were stripped.

**Why:**
- Preamble is boilerplate, not summary content, and because its presence/wording varies across paraphrases it was a **confound** for a sensitivity benchmark: it inflated `sms_drift` (outputs "differ" partly due to preamble phrasing, not summary substance) and diluted the quality metrics (ROUGE / cosine to gold / entity coverage all degraded by the non-summary prefix). Stripping it isolates the summary content, so the measured sensitivity reflects real content variation.

**Impact:**
- Changes numerical outputs of SMS/CS/ROUGE/BERTScore/faithfulness and therefore the Sensitivity composite, PRI, and diagnosis — expected to **reduce** spurious sensitivity and **raise** quality scores. PPL_var/PC_stab_var unchanged (raw text). Prior crossed-design results (including the two 5-article pilots) predate this and are not comparable — re-baseline.
- Applied to the in-flight 30-article run: the fix was pushed to the box while the run was still in Phase 1 (generation), so its Phase 2 scoring uses the cleaned outputs with no restart. Raw generations in `responses_crossed.jsonl` are preserved verbatim; cleaning happens only at scoring time, so it is fully reversible/auditable.

---

## [2026-07-06] — Evaluation-rigor fixes: pseudoreplication, outlier-robust normalization, absolute diagnosis

**Files Modified:** `gensens/crossed/scripts/common.py`, `aggregate_scores.py`, `significance.py`, `diagnosis_matrix.py`, `summary_report.py`

**What Changed (three flaws found by stress-testing the crossed pipeline on its own 250-cell A30 pilot output):**

1. **Pseudoreplication in the Pool A vs B significance test (`significance.py`).** The test compared per-cell Sensitivity as if the cells were independent (175 Pool-A cells vs 75 Pool-B), but cells are 35 seeds × N_ARTICLES and 15 seeds × N_ARTICLES — cells sharing a seed (same prompt) or article (same content) are correlated, so the per-cell p-value is anticonservative. On the pilot this over-claimed: per-cell p=0.009 vs the honest **per-seed** p=0.054 (rank-normalized composite). Fix: the **headline test is now the seed-level Mann-Whitney** (aggregate each seed to its mean Sensitivity across the shared articles → 35 vs 15 seed means); the per-cell test is retained only as `pool_comparison_percell_ref` and explicitly flagged "pseudoreplicated — not for inference". Added `MIN_SEEDS_FOR_TEST=10` (separate from the per-cell `MIN_N_FOR_TEST=20`) because seed counts are fixed by the dataset (Pool B only has 15 seeds regardless of N_ARTICLES) and Mann-Whitney is valid at 15 vs 35; a `low_power_warning` is attached when the smaller group < 20.

2. **Outlier-dominated min-max normalization collapsed 3 of 5 sensitivity metrics (`common.py`, `aggregate_scores.py`).** The Sensitivity composite min-max-normalized each spread metric, but cs_var / faith_var / pc_stab_var are heavy-tailed (raw max/mean of 14× / 12× / 26×), so a single outlier cell set the max and squashed 79% / 72% / 94% of cells to ≈0 on those axes — their nominal weights (0.20/0.20/0.15 = 55% of the composite) contributed almost nothing for the typical cell. Fix: new **`rank_normalize`** (average-rank percentile to [0,1], outlier-robust) is now the **default** normalization; legacy min-max kept behind `--normalization minmax` for ablation. Effect on the pilot: each metric's Spearman with the composite rose to a meaningful level (pc_stab_var 0.71, cs_var 0.55, faith_var 0.44, ppl_var 0.66, sms_drift 0.77), and the composite's dependence on sms_drift-alone dropped (Spearman 0.81 → 0.77 — it now adds signal beyond sms_drift). `aggregate_scores.py` also logs per-component Spearman and the sms_drift-alone comparison so the composite's honesty is auditable each run.

3. **Diagnosis 2×2 labels were relative thresholds read as absolute (`diagnosis_matrix.py`).** Thresholds are the model's own medians, so "high quality" only means "above this model's median". On the pilot, "True Robustness" cells averaged abs CS_mean 0.642 (range 0.563–0.842) — stable but only mediocre in absolute terms, so the label overstates. Fix: the printed distribution and a new `diagnosis_quadrant_summary.csv` now report each quadrant's **absolute** CS_mean (mean/min/max) and Sensitivity, with an explicit note that the split is relative to the model's median. Removed the misleading "all 4 cells populated = good" log line (a median split fills both bins by construction).

**Why:**
- These are execution/statistics flaws sitting on top of a conceptually-correct approach (sensitivity=spread, quality=mean, crossed design — validated in the pilot). They matter for any inferential or publication claim: the pseudoreplication inflated significance, the normalization made the weighting scheme illusory, and the diagnosis labels overstated absolute quality.

**Impact:**
- Numerical outputs of the Sensitivity composite CHANGE (rank vs min-max rescales every cell; the composite is now centered ~0.5 rather than compressed low). Rankings are broadly preserved (same Pool B > Pool A direction) but absolute Sensitivity values are NOT comparable to pre-fix runs — re-baseline. PRI, SMS, and the raw per-cell metrics are unchanged.
- The headline Pool A vs B claim is now honestly **borderline at n=5 (seed-level p=0.054)** rather than falsely significant (per-cell p=0.009); the 30-article run is needed for adequate power (more articles tighten the per-seed means without changing the 35-vs-15 seed counts).
- Verified end-to-end on the downloaded `results_pilot5/` (pure-math, no GPU): `common.py` unit-checked (rank spreads a heavy-tailed vector to full [0,1] where min-max collapsed it), and aggregate→diagnosis→significance→summary_report regenerated cleanly. Not yet applied to a fresh full run (30-article run was stopped before completion).

---

## [2026-07-06] — Fix CUDA OOM in crossed-design PPL/entropy pass + A30 pilot validation

**Files Modified:** `gensens/crossed/scripts/compute_cell_metrics.py`

**What Changed:**
- Fixed a CUDA out-of-memory crash in the teacher-forced PPL/branching-factor pass (phase 2) that aborted the metrics stage on a 24 GB A30. Two compounding causes, both fixed:
  1. **Model stacking:** `compute_cell_metrics.py` runs as a single process that loads the SBERT embedder, the NLI cross-encoder, and BERTScore's roberta-large, and THEN the 16 GB Llama-8B for the PPL pass — all co-resident on the GPU (~18 GB used, ~4 GB free). Now the earlier scorer models are explicitly freed (`del embedder, nli; gc.collect(); torch.cuda.empty_cache()`) before the 8B is loaded. Measured effect on the pilot: free VRAM before the 8B load went from ~4 GB to ~25 GB.
  2. **Full-vocab softmax blow-up:** the pass computed `log_softmax`, `.exp()`, and entropy over the entire `(batch, seq_len, vocab≈128k)` logits tensor at once — several GB of fp32, a single 4.54 GB allocation that OOM'd at the `lm_head`. Now the logits are sliced to each row's RESPONSE positions FIRST (l ≤ max_tokens ≪ seq_len), and softmax/entropy run only on that small `(l, vocab)` slice; the full logits tensor and batch buffers are deleted with `torch.cuda.empty_cache()` each batch.
- Default `--ppl_batch_size` lowered 8 → 4 (caps the `(batch, seq_len, vocab)` logits allocation). The runner also exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

**Why:**
- The A30 pilot (5 articles) was designed to surface exactly this class of infra failure before a full run. It did: phase 1 (generation) completed fine, phase 2 OOM'd at the PPL pass. Phase 1 outputs are checkpointed, so only phases 2–6 needed re-running after the fix.

**Impact:**
- No change to the PPL/entropy math or any numerical output — identical per-token NLL and softmax-entropy, just computed memory-safely. Purely an infrastructure fix; results are unaffected and do not need re-interpretation.
- **A30 pilot (5 articles × 50 seeds × ~6 variants = 1,495 outputs) completed end-to-end and validated the methodology.** Real timings on one A30: phase 1 generation 11 min (≈2.24 it/s, 100% GPU); full phases 2–6 ≈ 9 min. Extrapolated: ~1.5 h for 30 articles, ~5 h for 100 (generation-dominated). Validation outcomes: (a) **Sensitivity is decoupled from quality** — Pearson r(Sensitivity, CS_mean) = 0.002 (p=0.98), confirming the spread/mean split works and Sensitivity is not a quality proxy; (b) **Pool B more sensitive than Pool A** (0.178 vs 0.151), Mann-Whitney p=0.009 even at n=5; (c) all spread metrics finite with genuine range, PPL/branching-factor 250/250 finite (custom teacher-forced path works); (d) `sms_drift` visually calibrated (high-drift cells show divergent wording, low-drift cells near-identical); (e) low-n guards fired correctly (Wilcoxon seed test auto-skipped, underpowered-n warning emitted).
- **Known observation to carry forward:** ~6% of outputs (89/1495) were long enough that after truncating the article premise to the NLI 512-token window, no premise context fit, so their faithfulness defaulted to neutral (0.5). This proportion will persist at larger N and slightly dampens `faith_var`/`faith_mean` for those cells; consider a longer-context NLI model or sentence-level premise selection if faithfulness precision becomes load-bearing.

---

## [2026-07-06] — Crossed-design inference + sensitivity/quality split for GenSens summarization (new benchmark module)

**Files Added:** `gensens/crossed/scripts/{common,run_inference_crossed,compute_cell_metrics,aggregate_scores,diagnosis_matrix,significance,summary_report}.py`, `gensens/crossed/run_crossed_h100.sh`, `gensens/crossed/requirements_crossed.txt`

**What Changed:**
- New, separate crossed-design harness for the 50-seed/5-paraphrase summarization dataset (`gensens/data/gensens_summ_50seed_5para.jsonl`), independent of `prompt_robustness/` (different experimental grid, not a replacement for the PRI benchmark there).
- **Design:** every one of the 50 seeds is applied to the SAME N_ARTICLES CNN/DailyMail articles (stratified by word count, seed=42) — instance-fixed, prompt-varied, following POSIX/ProSA/PromptBench convention. Never 1:1 paired. A "cell" = one (article, seed) x 6 outputs (1 base + 5 paraphrases).
- **Core principle enforced in the code, not just the docs: SENSITIVITY = spread, QUALITY = mean.** Every metric that takes a mean across the 6 cell outputs also reports its variance; the variance is the headline sensitivity signal, the mean is a quality covariate. This directly fixes a conflation risk in earlier composite scores (e.g. the existing `PRI` in `prompt_robustness/src/scores.py`, which is mean-based and can rate 6 identically-mediocre outputs as "robust").
- **New per-cell sensitivity metrics (spreads):** `sms_drift` (1 − mean pairwise SBERT cosine of the 6 outputs), `cs_var`/`faith_var` (variance of a per-output correctness/faithfulness score), `ppl_var` (coefficient of variation of per-output perplexity), `pc_stab_var` (variance of per-output branching factor = exp(mean token entropy)), `rougeL_var`, `bertscore_var`.
- **New per-cell quality metrics (means):** `sms_similarity`, `cs_mean`, `faith_mean`, `rougeL_mean`, `bertscore_mean`, `mean_output_len`. These are covariates for interpreting sensitivity, not sensitivity measures themselves.
- **New composite `Sensitivity` (headline):** empirical min-max normalize each spread metric across all cells, then `0.30*sms_drift_n + 0.20*cs_var_n + 0.20*faith_var_n + 0.15*ppl_var_n + 0.15*pc_stab_var_n`. If a metric is unavailable for any cell (e.g. `--skip_ppl_entropy`), it is dropped entirely and the remaining weights are renormalized to sum to 1 — never faked.
- **Relabeled `PRI` for this module:** `0.40*sms_similarity + 0.35*cs_mean + 0.25*faith_mean` (x0.85 if `mean_output_len < 12`), explicitly documented as *quality-gated robustness*, not pure sensitivity. `Sensitivity` above is the benchmark's headline number.
- **PPL/branching-factor are computed for real**, not estimated from vLLM top-k logprobs: `compute_cell_metrics.py` runs a second, separate-process HF `transformers` teacher-forced forward pass per (prompt, output) after the vLLM generation process has exited (avoids double-loading the 8B model on the GPU), computing exact per-token NLL (→ PPL) and full-softmax entropy (→ branching factor) — never approximated from a truncated top-k distribution.
- **Faithfulness** uses `cross-encoder/nli-deberta-v3-small` (lightweight, deterministic) as `P(entailment) + 0.5*P(neutral)` of article→output, with the article truncated (never the output) to fit the NLI context window. This is a different, cheaper faithfulness scorer than `prompt_robustness/src/faithfulness_metric.py`'s 70B-AWQ LLM-judge — appropriate here given the crossed design's 30,000-output scale.
- **Hierarchical aggregation:** cell → per-seed (mean/std/95% bootstrap CI over the shared articles) → per-pool and per-dimension, always computed cell-first (never flattened across articles before the per-cell composite). Pool A and Pool B are kept and reported separately in every aggregate.
- **Diagnosis 2x2** on two independent axes (`CS_mean` quality vs. `Sensitivity` spread), thresholds at the empirical median of each axis → True Robustness / Fragile / Consistently Poor / Unreliable.
- **Significance tests:** Mann-Whitney U (Pool A vs B) with rank-biserial effect size; per-dimension ranking with bootstrap CI; Wilcoxon signed-rank on paired per-article Sensitivity for the most- vs. least-sensitive seed (valid pairing because the design is crossed).
- Greedy decoding (`temperature=0.0`) hard-coded for the entire main generation run — no sampling option, to isolate prompt effect from decoding-randomness noise.

**Known upstream data-quality gap surfaced while validating this module:**
- Seed **B12** carries only 4 total variants (1 base + 3 paraphrases) instead of the target 6 — per `gensens/data/fix_changelog.json`, one paraphrase was flagged `oversim_replaced` (SBERT similarity 0.951, too close to base) and the replacement generation never succeeded (`unresolved: true`). `common.load_seed_prompts()` now tolerates per-seed variant counts down to a floor of 4 (raising only below that), logging a warning and recording the true `n_variants` per cell rather than padding/faking a 6th output. `run_inference_crossed.py`'s expected-task-count check and `compute_cell_metrics.py`'s cell-completeness check both use each seed's actual expected variant set (not a flat 6) accordingly. Re-running `gensens/scripts/regenerate_flagged.py` to close this gap is recommended before a publication-quality run, but the pipeline is correct either way.

**Why:**
- The existing repo's `PRI`/`ORI`/`IFI` composites (see Key Formulas table above) are quality-weighted means over variants and, by construction, cannot distinguish "the model is robust" from "the model is uniformly mediocre." A dedicated sensitivity-as-spread benchmark, run on the standard crossed grid used by POSIX/ProSA/PromptBench, closes that measurement gap for the summarization task specifically, using the already-audited 50-seed dataset.

**Impact:**
- Purely additive: no existing formula, threshold, or script in `prompt_robustness/` or the rest of `gensens/` is modified. `gensens/crossed/` is a new, independent module.
- Requires an H100/A30-class GPU with vLLM + HF `transformers` to actually execute (30,000 generations + a second teacher-forced pass + SBERT/NLI/BERTScore scoring); not runnable on a local CPU-only machine. The aggregation/diagnosis/significance scripts (`aggregate_scores.py`, `diagnosis_matrix.py`, `significance.py`) are pure numpy/pandas/scipy and were smoke-tested locally against a synthetic 30-article x 50-seed fixture (1,500 cells) — all four diagnosis cells populated, Mann-Whitney correctly recovered the injected Pool-B-more-sensitive signal (p≈3.6e-62), Wilcoxon most-vs-least-sensitive seed test ran cleanly. The GPU-dependent scripts (`run_inference_crossed.py`'s vLLM path, `compute_cell_metrics.py`'s embedding/NLI/PPL passes) were reviewed but not executed end-to-end — no results have been generated yet.

---

## [2026-07-05] — Surgical fix pass on the 50-seed summarization dataset + new 0.92 SBERT upper bound

**Files Modified:** `gensens/scripts/fix_flagged_seeds.py` (new)

**What Changed:**
- New standalone repair script that fixes specific flawed seeds/variants in `gensens_summ_50seed_5para.jsonl` in place, without regenerating the whole pool. Three fixes:
  1. **Top-up** under-target seeds A03, A15, A18 (pool A), B05 (pool B) from <5 to exactly 5 variants.
  2. **Full regenerate** B14 (must carry an explicit skepticism/doubt marker — stricter than the validator's lenient check that accepts "critical") and B15 (regenerated only when <4/5 current variants retain BOTH the 50-word-summary and self-critique parts). Both use a one-shot constraint-preserving system prompt.
  3. **Replace** every variant across all 50 seeds whose `sbert_similarity > 0.92`, using stronger transformation strategies (prefer `different_structure` → `technical_vocab` → `role_prefix`; `reordered_clauses` excluded because it yields near-identical text), keeping each seed at 5 variants.
- **New acceptance band: `0.82 ≤ SBERT cosine ≤ 0.92`** for all (re)generated variants. The upper bound of 0.92 is new — the original pipeline gated only on the 0.82 lower bound (and a separate 0.95/0.98 semantic-dedup ceiling), which let through variants that were lexically near-identical to the base and therefore too easy to serve as a real prompt-sensitivity test.
- The existing bidirectional NLI gate (`cross-encoder/nli-deberta-v3-small`, τ=0.50 both directions) is retained. Sampling for (re)generation: `temperature=0.85, top_p=0.92`, up to 3 retries per strategy, per-call vLLM seed varied so retries actually differ (the shared `generate_batch` hardcodes `seed=42`). Dedup by first-8-token signature within each seed.
- Writes a pre-fix backup (`gensens_summ_50seed_5para.backup.jsonl`) and a per-seed change log (`fix_changelog.json`). Untouched seeds are copied from the original file verbatim (byte-for-byte); only affected records are re-serialised.

**Why:**
- Post-generation analysis found three residual quality issues (under-target counts, constraint drift on the skeptical/self-critique Pool B seeds, and over-similar variants) that did not warrant a full pool regeneration. A targeted pass fixes only the affected records at far lower GPU cost while leaving the validated majority untouched.
- The 0.92 upper bound directly targets the prompt-sensitivity benchmark's validity: a paraphrase that is ~0.95+ cosine to the base is essentially the same surface form and does not exercise the model's robustness to rewording.

**Impact:**
- `gensens_summ_50seed_5para.jsonl` is modified in place for the affected seeds (all seeds still 5 variants where achievable; any seed that cannot reach 5 valid variants after retries is marked `unresolved: true` in `fix_changelog.json` rather than silently shipped short).
- After this pass, all variants in the dataset should satisfy `0.82 ≤ sim ≤ 0.92`; earlier analyses/plots that assumed the old (lower-bound-only) band are superseded. The regenerated PDF/export should be rebuilt from the updated JSONL.
- Scope is limited to this repair script; the generator's own default band/params are unchanged (the 0.92 upper bound is enforced by `fix_flagged_seeds.py`, not baked into `paraphrase_generator.py`).

**Run outcome (2026-07-05) + touched-detection bug fix:**
- First run: FIX 1 topped up A03/A15/A18/B05 to 5; FIX 2 regenerated B14 (5/5 task skepticism markers) and left B15 as-is (4/5 held both parts, ≥4 threshold); FIX 3 replaced 16 of 17 over-similar variants. Final dataset: 49 seeds at 5 variants, **B12 at 4 (unresolved)** — its lone over-similar variant could not be replaced within the band + bidirectional-NLI + tabloid-tone constraint after all retries, so the over-similar variant was dropped (best-available-in-band) and B12 flagged `unresolved: true`.
- **Bug fixed in `fix_oversimilar` / `main`:** the "seed was modified" test originally keyed on a *successful replacement* (`any _fix_type == oversim_replace`). For B12 the replacement failed, so the (correctly) mutated 4-variant in-memory record was discarded and the original 5-variant line (still containing the 0.951 over-similar variant) was written back — the report and the file disagreed. `fix_oversimilar` now returns `True` whenever an over-similar variant was present (it is dropped regardless of replacement success), and `main` marks the seed touched on that return value. The B12 line was corrected on disk to the 4-variant in-band version.
- `validate_summ_paraphrases.py` independently flags B02/B06/B08 for a single constraint-keyword miss each: B06/B08 still meet the task's ≥4/5 threshold; B02 is 1/5 but was **out of scope** (not in the fix list and had no over-similar variant, so left untouched per the task rule). The validator's Pool B keyword lists are also narrower than this task's (e.g. it matches `"skeptical"` but not `"skepticism"`), so several B14/B05 variants that satisfy the task's explicit marker list are false-flagged by the validator.

---

## [2026-07-05] — Fix Pool B one-shot example corrupting length-ratio/similarity/dedup checks in regenerate_flagged.py

**Files Modified:** `gensens/scripts/regenerate_flagged.py`

**What Changed:**
- `POOL_B_ONESHOT` (the one-shot constraint-preservation example) is now appended to `_pool_b_retry_system_prompt()`'s returned system-prompt string instead of being concatenated onto `record_for_gen["base_text"]`.
- `record_for_gen["base_text"]` is now always the seed's unmodified `base_text`.

**Why:**
- `paraphrase_generator.generate_paraphrases()` uses `record["base_text"]` for two different purposes: (1) what the LLM is asked to rephrase, and (2) the reference string for the length-ratio, SBERT-similarity, token-overlap, and dedup-signature gates. The old code concatenated a ~500-character one-shot example onto `base_text` for pool B, which inflated the reference length used by (2) while the LLM still (correctly) generated a normal-length rephrase of just the actual instruction — so every candidate's length ratio came out far below `length_ratio_min=0.70` and was rejected.
- Confirmed live during a regeneration run on 2026-07-05: seeds B01, B02, B05 each returned 0/5 variants across all 3 attempts, with `rejection breakdown` showing `length_ratio_oob=90` (100% of the 90 candidates tried). The run was killed mid-batch once the pattern was clear, rather than letting it burn GPU time on the remaining Pool B seeds, which would have failed identically.

**Impact:**
- Previous regeneration attempt for the 13 flagged seeds (A03, A15, A18, B01, B02, B05, B06, B07, B08, B10, B11, B12, B14) was aborted after 7 seeds; no output was written (the script only writes `gensens_summ_50seed_5para.jsonl` once, at the end, after all flagged seeds are processed) so the original file is untouched. Full regeneration re-run needed for all 13 seeds with this fix.

---

## [2026-07-05] — Fix regenerate_flagged.py's 70B AWQ fallback default (consistency w/ generator change)

**Files Modified:** `gensens/scripts/regenerate_flagged.py`

**What Changed:**
- Added `DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"`; `--model-id` now defaults to it instead of `None`.
- `gen_kw` now always passes `model_id`, instead of only when `--model-id` was explicitly given.

**Why:**
- Same trap as `generate_summ_paraphrases.py` before the entry below: `ParaphraseGenerator(model_id=None)` falls back to the class-level 70B AWQ default. Left as-is, retrying the 6 flagged seeds would have silently loaded a different (and unquantized-for-this-path) generator than the one that produced the other 44 seeds, defeating the point of the retry.

**Impact:**
- Regenerated seeds (A03, A15, A18, B05, B07, B11) now use the same 8B generator as the original run. No effect on already-accepted seeds.

---

## [2026-07-05] — 8B generator default for the 50-seed summarization paraphrase pipeline

**Files Modified:** `gensens/scripts/generate_summ_paraphrases.py`

**What Changed:**
- Changed the default `--model-id` for this script from the class-level `ParaphraseGenerator.VLLM_MODEL_ID` fallback (70B AWQ INT4) to `meta-llama/Meta-Llama-3.1-8B-Instruct`, still via the `vllm` backend (batched, bf16, no quantization).
- Added `DEFAULT_MODEL_ID` constant and always pass `model_id` into `ParaphraseGenerator(...)` (previously only passed when `--model-id` was explicitly given).
- Scope is limited to this script. `generate_dataset.py` (creative/dialogue/qa tasks) and `config.yaml`'s `models.generator` documentation still default to the 70B AWQ generator — unchanged.

**Why:**
- User chose to run the summarization paraphrase generation step on an 8B model instead of the 70B AWQ generator, primarily to simplify GPU requirements (single ~24GB card, no AWQ quantization, no tensor-parallel sharding needed).
- Llama-3.1-8B-Instruct (not Qwen2.5-7B-Instruct, the other 8B-class model already in this pipeline) was chosen to preserve the existing methodological separation between the instruction-pool generator (Qwen2.5-7B, used by `generate_summ_instructions.py`) and the paraphrase generator — using the same model for both would collapse that separation (see 2026-05-27 entry below).

**Impact:**
- Any `gensens_summ_50seed_5para.jsonl` generated after this change uses an 8B paraphraser instead of the 70B AWQ paraphraser; paraphrase quality/style may differ from what the 70B generator would have produced. Not yet run — no existing results are invalidated by this change.
- GPU requirement for this step drops from ~35GB (70B AWQ, single 80GB card or 2×A30 with tensor-parallel) to ~16-20GB (8B bf16, any single ~24GB card, `tensor_parallel_size=1`).
- To revert to the 70B AWQ generator for this script, pass `--model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --quantization awq_marlin` (note: this script has no `--quantization` CLI flag yet, unlike `generate_dataset.py`; vLLM will attempt to auto-detect AWQ from the model's `config.json` if you go this route).

---

## [2026-07-01] — Pool-aware system prompts + 50-seed summarization pipeline (H100 run prep)

**Files Modified:** `gensens/scripts/paraphrase_generator.py`, `gensens/scripts/generate_summ_paraphrases.py` (new), `gensens/scripts/validate_summ_paraphrases.py` (new), `gensens/scripts/regenerate_flagged.py` (new)

**What Changed:**
- Added `SUMM_SYSTEM_PROMPTS` dict to `paraphrase_generator.py` with separate entries for Pool A (open-style rewriting) and Pool B (constraint-preservation rewriting)
- Updated `_build_chat_prompt`, `_generate_single_vllm`, `_generate_single_llama`, `_generate_single`, `_generate_n_vllm` to accept `pool: str = "A"` and route to the correct system prompt for summarization tasks
- Updated `generate_paraphrases` to extract `pool` from the record dict and thread it through the entire generation pipeline
- Wrote `generate_summ_paraphrases.py`: orchestrator that loads 50 seeds, verifies 35A + 15B, runs generation with checkpoint/resume every 10 seeds, outputs `gensens_summ_50seed_5para.jsonl` and per-pool/per-dimension stats
- Wrote `validate_summ_paraphrases.py`: validator covering checks A–G (completeness, SBERT ≥ 0.82, no duplicates, length sanity, `{{article}}` placeholder, Pool B keyword constraint checks per prompt_id B01–B15, LLM-as-judge fallback if >30% Pool B variants fail keyword check)
- Wrote `regenerate_flagged.py`: reads validation JSON, re-runs generation for flagged seeds with tighter Pool B system prompt + one-shot example, up to 3 attempts; marks unresolvable seeds with `constraint_leakage_unresolved: true`; re-runs validation and prints before/after diff

**Why:**
- Summarization benchmark needs 50 distinct seed prompts (not one canonical instruction) paraphrased into 5 variants each
- Pool B instructions contain specific constraints (banned words, forced words, tone, persona, output structure) that must survive the paraphrase rewrite — a generic system prompt did not preserve these reliably
- End-to-end pipeline (generate → validate → regenerate) makes the H100 run self-contained

**Impact:**
- Breaking change for Pool B generation: Pool B paraphrases from any pre-existing run used the generic summarization system prompt and must be re-generated
- Pool A generation is semantically equivalent to previous behaviour (same intent, improved wording)
- No numerical metric changes; no benchmark results need re-running

---

## [2026-06-04] — CLI plumbing: expose Tier-1/Tier-2 methodology + tensor parallelism (PUBLICATION_SPEC §6, §7, §8)

**Files Modified:**
- `gensens/scripts/paraphrase_generator.py` (constructor + `_load_vllm`)
- `gensens/scripts/generate_dataset.py` (CLI parser + ParaphraseGenerator instantiation)
- `stage0a_smoke.sh` (production-ready environment variables for both Qwen-7B and 70B AWQ launches)

**What Changed:**

- **`ParaphraseGenerator.__init__` — new `tensor_parallel_size` parameter.** Defaults to `1` (single card). `_load_vllm` now passes `tensor_parallel_size=self.tensor_parallel_size` to vLLM's `LLM(...)` constructor instead of the hardcoded `1`. **Required** for production runs on A30 × 2 (or any dual 24 GB setup) with the 70B AWQ generator — a single 24 GB card cannot hold the ~35 GB AWQ weights, so vLLM must shard via tensor parallelism. Without this fix the 70B AWQ generator OOMs immediately at model-load.
- **`generate_dataset.py` — 18 new CLI flags expose every methodology knob** that PUBLICATION_SPEC §6, §7, §8 introduces. The CLI grew from 14 flags to 32. New flags:
  - `--tensor-parallel-size` (vLLM TP for A30 × 2)
  - `--disable-nli-gate` (bool; defaults to enabled to honour §6.1)
  - `--nli-entail-threshold` (default 0.50)
  - `--nli-model-name` (default `cross-encoder/nli-deberta-v3-small`)
  - `--enable-nli-ensemble` (Tier-2 §7.1 multi-NLI 2-of-3 majority — REQUIRED for publication runs)
  - `--nli-ensemble-models` (override ensemble members)
  - `--nli-ensemble-majority` (default 2)
  - `--length-ratio-min`, `--length-ratio-max` (§6.2 length-ratio filter, default 0.70–1.50)
  - `--semantic-dedup-threshold` (§6.3 SBERT semantic dedup, default 0.95)
  - `--best-of-n`, `--best-of-n-temperature`, `--best-of-n-top-p` (§6.5 best-of-N per strategy; defaults n=3, T=0.95, top_p=0.92)
  - `--max-per-family-{lexical,syntactic,pragmatic,length}` (§6.6 per-family budgets)
  - `--task-thresholds-json` (per-task SBERT threshold overrides via JSON string)
- **All 15 new ParaphraseGenerator kwargs are now forwarded** from the corresponding CLI args. End-to-end validation confirms: every methodology knob added in the Tier-1/Tier-2 commits is reachable from the command line, with sensible publication-grade defaults so a bare `--enable-nli-ensemble` invocation produces the recommended configuration.
- **`stage0a_smoke.sh` upgraded with parameterised environment variables**: `TENSOR_PARALLEL_SIZE`, `QUANTIZATION`, `ENABLE_NLI_ENSEMBLE`, `BEST_OF_N`. The same script now serves both the Qwen-7B smoke test (defaults: TP=1, no quantization, ensemble ON) and the 70B AWQ production launch (set `TENSOR_PARALLEL_SIZE=2 QUANTIZATION=awq_marlin` before invocation). `GEN_ARGS` is composed conditionally so passing `--quantization ""` is never attempted.

**Why:**
- Without this commit, the Tier-2 multi-NLI ensemble (§7.1) — a required acceptance-criterion item from PUBLICATION_SPEC.md — could not fire from the production CLI even though the underlying logic was in place. The ensemble defaults OFF in `ParaphraseGenerator.__init__` for backward compatibility; without a CLI surface to enable it, no production invocation would exercise it.
- Without `tensor_parallel_size` exposed, the 70B AWQ generator (the standard PUBLICATION_SPEC choice) could not run on Jarvis A30 × 2 instances — the cheapest GPU configuration available, ~17% cheaper total than A100 80 GB for our pipeline. The hardcoded `tensor_parallel_size=1` would have forced users onto more expensive single-card GPUs.

**Impact:**
- **Backward compatible for legacy invocations**: every new flag has a sensible default that matches the existing behaviour. Old scripts that don't pass the new flags behave identically to the pre-commit code.
- **Forward-compatible for production**: a publication-grade GenSens run is now invocable via:
  ```bash
  python gensens/scripts/generate_dataset.py \
      --task all --n_instances 200 --n_variants 8 \
      --model vllm \
      --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
      --quantization awq_marlin \
      --tensor-parallel-size 2 \
      --enable-nli-ensemble \
      --best-of-n 3
  ```
- **No methodology / formula change**: all four formula tables in CLAUDE.md remain unchanged. This commit is pure plumbing — it exposes the existing Tier-1 / Tier-2 methodology through the CLI surface so production runs can actually exercise it.

---

## [2026-06-04] — LL-PIRC experiment infrastructure: baselines comparison + ablation sweeps (PUBLICATION_SPEC §10)

**Files Modified / Added:**
- Added: `prompt_robustness/experiment_baselines_comparison.py`
- Added: `prompt_robustness/run_pirc_ablations.py`
- Modified: `prompt_robustness/experiment_pirc.py` (wired `anchor_percentile` to `AnchorTokenIdentifier`; added `--anchor-percentile` CLI flag)

**What Changed:**

- **§10 (Table 2) — `experiment_baselines_comparison.py` — the headline LL-PIRC method-paper result.**
  - Runs the four no-training mitigation baselines from `src/mitigation_baselines.py` (`temperature_smoothing`, `self_consistency_vote`, `system_prompt_stabilize`, `in_context_learning`) plus the previously-computed LL-PIRC results, all against the SAME article set used by `experiment_pirc.py`.
  - Reuses helpers from `experiment_pirc.py` (`load_config`, `load_baseline_results`, `load_target_model`, `build_prompts_from_baseline_result`, `score_outputs_rouge_l`, `compute_rouge_l`) so the article schedule, prompt formatting, and ROUGE scoring are byte-identical between the PIRC harness and the baselines harness.
  - New `build_generate_fn(model, tokenizer, max_new_tokens)` adapter wraps HuggingFace `model.generate` into the `(prompt, kwargs) -> str` signature expected by every function in `mitigation_baselines.py`. Handles: greedy vs sampled decoding, per-call seed for sampling reproducibility, optional `system_prompt` injection via chat template (with raw concat fallback).
  - PIRC results are READ from a prior `pirc.json` rather than re-run, so the comparison costs (4 baselines × N articles × K paraphrases) of model inference — no PIRC recompute.
  - Per-method per-article metrics: outputs, ROUGE-L scores, ROUGE mean, ROUGE variance, ROUGE min (worst-prompt), wall time.
  - `aggregate_comparison()` computes the canonical variance-reduction formula `1 - mean(var_method) / mean(var_baseline_no_intervention)` matching `experiment_pirc.py` semantics so methods are directly comparable across drivers.
  - `render_markdown()` produces the publication-ready headline table with method ordering (no-intervention → baselines → LL-PIRC) — verified on synthetic data.
  - Crash-safe: writes the full `baselines_comparison.json` after every article so a mid-run failure resumes cleanly.

- **§10 — `run_pirc_ablations.py` — three-knob ablation sweep driver.**
  - Sweeps `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` (clamping strength), `anchor_percentile ∈ {10, 20, 30, 50, 70}`, and optionally `ℓ* ∈ {user-supplied list}`. Defaults to `--knobs alpha anchor_percentile`; ℓ* is opt-in because it requires the user to know the layer count of their subject model.
  - For each (knob, value) pair: launches `experiment_pirc.py` as a subprocess with the override CLI flag, then copies the resulting `pirc.json` + `eval_summary.json` into `<results-dir>/<knob>=<value>/` so the next run does not clobber this run's artefacts.
  - `_extract_headline()` pulls per-run metrics from `eval_summary.json` + recomputes worst-prompt ROUGE-L (`mean_a [min_k ROUGE-L]`) from `pirc.json` per-article scores. Worst-prompt is the metric the field has standardised on (RobustAlpacaEval, Cao et al. ICLR 2024) and was previously missing from the eval summary.
  - Outputs per-knob CSV (`alpha_ablation.csv`, `anchor_percentile_ablation.csv`, `ell_star_ablation.csv`) + a consolidated `ablation_summary.md` Markdown table.
  - `--dry-run` prints the schedule without launching any subprocess — useful for cost estimation before committing to a GPU run.

- **`experiment_pirc.py` plumbing (in support of the ablation):**
  - `setup_pirc_pipeline()` now reads `anchor_tokens.anchor_percentile` from config (defaulting to 30.0 for backward compatibility) and forwards it to `AnchorTokenIdentifier`. The parameter existed in the class for some time but was never wired through the harness.
  - New CLI flag `--anchor-percentile <float>` (e.g. `10`, `20`, `30`, `50`, `70`) overrides the config value. Symmetric with the existing `--alpha` and `--ell-star` ablation flags.

**Why:**
- The baselines-comparison driver is the SINGLE most important publication asset for the LL-PIRC method-paper claim. Without it, "LL-PIRC reduces variance more than the baselines" is an unsupported claim and reviewers will (rightly) push back. The four baselines were already library-only in `mitigation_baselines.py` — what was missing was the driver that runs them on the same article set with consistent scoring.
- The three-knob ablation sweep is required for the "design choice ablations" section of the paper. Without α / anchor-percentile sweeps reviewers ask "did you tune these?" and "why does this configuration work best?" The driver makes the answers reproducible.
- The `anchor_percentile` wiring fix closes a latent bug where the config value was never honoured even when set — runs that set `anchor_tokens.anchor_percentile: 50` in the YAML silently still used 30.

**Impact:**
- No methodology changes to existing LL-PIRC, baselines, or ROUGE scoring — purely orchestration code.
- A `pirc.json` produced by `experiment_pirc.py` BEFORE this commit may have a different `anchor_percentile` than what the config requested (the value was ignored). Re-running the experiment with the same config now produces faithfully-percentile-controlled results; old runs are still valid AT the default percentile (30) but should be retired for ablation purposes.
- Subprocess-based ablation sweep: each `<knob>=<value>` run is an independent `experiment_pirc.py` invocation. On a single GPU box the runs are serialised; on multi-GPU setups the driver could be extended to dispatch in parallel (out of scope for paper #1).

---

## [2026-06-04] — Tier-2 strategy expansion: TextFooler + back-translation (PUBLICATION_SPEC §7.2, §7.3)

**Files Modified / Added:**
- Added: `gensens/scripts/adversarial_textfooler.py`
- Added: `gensens/scripts/back_translation_family.py`
- Modified: `gensens/scripts/paraphrase_generator.py` (registered `back_translation` family in `MAX_PER_FAMILY`)

**What Changed:**

- **§7.3 — TextFooler-style adversarial paraphrase module (`adversarial_textfooler.py`, new ~530 lines).**
  - End-to-end pipeline: BERT MLM fill-mask synonym proposer + SBERT semantic floor (≥ 0.85) + bidirectional NLI (single-model or ensemble) + length-ratio filter.
  - Algorithm: for each base prompt, iteratively (up to `max_substitutions`, default 3) find the content-word substitution that maximally reduces SBERT similarity to the base WHILE staying inside the paraphrase gate. The chosen substitution is the most "adversarial" without exiting the paraphrase manifold.
  - Implementation classes: `MLMSynonymProposer` (top-K context-aware synonyms via fill-mask pipeline, post-processed to strip WordPiece prefixes and reject single-token / non-alphabetic predictions), `SBERTSimilarity` (single embedder owned by the generator), `AdversarialTextFooler` (the main pipeline; mirrors the production filter stack semantics).
  - Output JSONL schema: per-prompt `{base, adversarial_paraphrase, substitutions:[{position, original, substitute, sbert_sim, nli_fwd, nli_bwd, adversariality}], num_substitutions, filter_metadata, perturbation_family: "adversarial_textfooler"}`.
  - CPU- and GPU-friendly: NLI helpers reuse production `_get_nli_model` / `_get_nli_ensemble` from `paraphrase_generator.py` when importable, falling back to a local CrossEncoder reimplementation otherwise. Same with the SBERT instance.
  - Reproducibility: deterministic top-K MLM ordering; no sampling-based generation. Resume-safe (`instance_id`-keyed dedup against existing output JSONL).
  - The existing `adversarial_paraphrases.py` (deterministic perturbations — typo / sentence_reorder / double_negation / hedged / formality_shift) is NOT deleted here. The PUBLICATION_SPEC plan is to rename it `noise_paraphrases.py` and report PRI in three separate buckets: (a) GenSens paraphrases, (b) noise paraphrases, (c) adversarial paraphrases. The rename will land in a follow-up commit so existing CLI flags do not break in the same commit as the strategy expansion.

- **§7.2 — Back-translation as the 5th strategy family (`back_translation_family.py`, new ~440 lines).**
  - Runs as an AUGMENTATION PASS over an existing GenSens JSONL (rather than being inlined into `paraphrase_generator.py`). This keeps the heavy NMT model loads amortised across all instances in one process.
  - Wraps `Helsinki-NLP/opus-mt-en-{de,fr,ru,es,zh}` and the reverse pairs. Default pivot languages: `de fr ru` (the same trio listed in `config.yaml paraphrase.pivot_languages` as legacy — now re-enabled).
  - Per-instance loop: for each pivot language, round-trip translate the base text and apply the FULL production filter stack: length ratio, SBERT band [0.82, 0.98], token Jaccard ≤ 0.85 vs base and vs accepted variants, SBERT semantic dedup ≤ 0.95, bidirectional NLI ≥ 0.50 (single-model or ensemble).
  - Accepts up to `max_per_instance` (default 2) back-translation variants per record. Each variant tagged `strategy="back_translation_<lang>"`, `strategy_family="back_translation"` and carries the same audit fields as a prompt-engineering variant (`sbert_similarity`, `length_ratio`, `token_overlap_to_base`, `max_cos_to_accepted`, `nli_entail_fwd`, `nli_entail_bwd`, `nli_passed`).
  - I/O is deliberately additive: the script READS one JSONL and WRITES a new JSONL where each record's `variants` list has the back-translation variants APPENDED. `instance_id`, `base_text`, original variants, and all metadata are preserved unchanged. Backward compatible with every downstream reader.

- **Family taxonomy registration (`paraphrase_generator.py`).**
  - `MAX_PER_FAMILY` now reads `{lexical: 2, syntactic: 2, pragmatic: 1, length: 1, back_translation: 2}` — total budget 8 variants, exactly matching the GenSens K=8 target.
  - The `back_translation` family cap is registered here even though the variants themselves are produced by a separate script — this ensures that any future code path that consults `MAX_PER_FAMILY` (e.g. a future unified pipeline) does not double-count the back-translation slots.

**Why:**
- Adversarial TextFooler: the current "adversarial" subset in `adversarial_paraphrases.py` is *noise injection* (typos, double-negation), not adversarial generation. Reviewers immediately spot this — adversarial means "the perturbation was chosen to maximally disrupt the target." MLM-based word substitution with semantic+NLI gating is the standard adversarial baseline for NLP robustness papers (PromptBench, TextFooler original, the `prompt-robustness` literature broadly).
- Back-translation: PromptBench's headline paraphrase strategy is round-trip MT, and the legacy `config.yaml` already listed the Helsinki-NLP model trio — we just hadn't wired it into the new diversity-controlled gate stack. Adding it as a *separate augmentation pass* avoids interleaving heavy NMT loads with the vLLM generator and keeps the architectural separation between "LLM rewrite strategies" (lexical/syntactic/pragmatic/length families) and "NMT round-trip strategies" (back_translation family) explicit in the methodology.

**Impact:**
- TextFooler script is additive: produces a separate `adversarial_textfooler_*.jsonl` that downstream Phase-2 / Phase-3 evaluation reads as a third paraphrase bucket alongside the standard GenSens variants and the noise variants. No change to existing JSONL schemas.
- Back-translation script is additive: increases variant counts on existing GenSens JSONLs without modifying any existing field. Records that already had K=8 variants will now have up to K=10 (8 + 2 BT) variants; downstream `K=8` consumers will simply use the first 8 in `variants_idx` order — no breakage but a behavioural shift to be aware of (consumers that truncate at K=8 will get the prompt-engineering variants by default; consumers that want a mixed set should reshuffle or explicitly select).
- `MAX_PER_FAMILY` change is forward-only: legacy code that iterates `MAX_PER_FAMILY` will now see a new family key. The cap-enforcement logic in `generate_paraphrases` uses `.get(family, 999)` so unknown families never block generation.

---

## [2026-06-04] — RobustAlpacaEval borrowed-validation pipeline (PUBLICATION_SPEC §8.4)

**Files Added:** `prompt_robustness/scripts/robustalpaca_crossvalidation.py`

**What Changed:**

- **§8.4 — Borrowed-human-validation pipeline (the "killer move").**
  - Three modes:
    1. `--mode prepare` — download `ZBWpro/RobustAlpacaEval` (or fall back through `Cao-Yifan/RobustAlpacaEval`) via the `datasets` library, flatten the (original_query, paraphrases) records into a canonical pair CSV.
    2. `--mode audit` — run the GenSens `FilterPipeline` (imported from `paws_negative_controls.py` so behaviour is byte-identical to the §8.3 audit) on RobustAlpacaEval's 1000 human-verified pairs. Reports per-filter pass rate, per-metric distributions, and the headline "filter pass rate ≥ 88% on human-verified paraphrases" claim.
    3. `--mode crossvalidate` — compare per-pair audit JSONLs from a GenSens audit and a RobustAlpacaEval audit using a two-sample Kolmogorov-Smirnov test on (SBERT cos, token Jaccard, length ratio, NLI fwd, NLI bwd). Reports KS statistic + p-value per metric and an overall "ALIGNED / PARTIAL / DIVERGENT" verdict.
  - Pure-Python KS implementation (Stephens 1970 asymptotic approximation, no SciPy dependency) — unit-tested on:
    - Identical N(0,1) distributions (n=200 each): KS=0.10, p=0.26 (correctly ALIGNED)
    - Shifted N(0,1) vs N(2,1) (n=200 each): KS=0.67, p<10⁻⁴ (correctly DIVERGENT)
    - Edge case empty input: returns (None, None)
  - Audit-JSONL key tolerance: the loader accepts both `sbert_sim` (PAWS audit JSONL schema) and `sbert_similarity` (GenSens variant JSONL schema) so cross-script comparisons work without an intermediate transformation pass.

**Why:**
- This is the most compelling reviewer-comfort element of the §8 4-pronged validation. RobustAlpacaEval is a published human-validated dataset (Cao et al., ICLR 2024). If the GenSens filter pipeline retains ≥88% of its human-verified paraphrases AND the GenSens paraphrase-metric distributions are statistically indistinguishable (KS p>0.05) from the RobustAlpacaEval distributions, we have transitive human validation that no purely-automated validation can otherwise produce.
- The KS test was chosen over t-test / Wilcoxon because it is distribution-free and sensitive to the entire CDF shape (not just location). Reviewers who push back on "you didn't do new human annotation" cannot also argue with a fully reproducible distributional alignment metric measured against a published human-verified reference.

**Impact:**
- No methodology change to the GenSens dataset itself — this is purely validation infrastructure.
- Adds a new dependency on `datasets>=2.0.0` for `--mode prepare` (lazy-imported so the audit/crossvalidate modes work without it).
- The `crossvalidate` JSONL loader needs an audit JSONL on each side — typically produced by running PAWS-style audits on (a) a sampled subset of accepted GenSens variants and (b) the prepared RobustAlpacaEval pairs.

---

## [2026-06-04] — Tier-2 validation infrastructure (PUBLICATION_SPEC §7.1, §8.1, §8.3)

**Files Modified / Added:**
- Modified: `gensens/scripts/paraphrase_generator.py`
- Added: `prompt_robustness/scripts/llm_judge_paraphrases.py`
- Added: `prompt_robustness/scripts/paws_negative_controls.py`
- Added: `gensens/data/paws_negative_controls.jsonl` (200 generated pairs)

**What Changed:**

- **§7.1 — Multi-NLI ensemble (3-of-3 majority voting) in `paraphrase_generator.py`**
  - New module-level constant `_NLI_ENSEMBLE_MODELS = ['cross-encoder/nli-deberta-v3-small', 'cross-encoder/nli-deberta-v3-base', 'cross-encoder/nli-roberta-base']` and helpers `_get_nli_ensemble()` (lazy loader with abstain-on-load-failure semantics) and `_nli_ensemble_entail()` (per-model vote tally + mean entailment).
  - New `ParaphraseGenerator.__init__` flags: `enable_nli_ensemble: bool=False`, `nli_ensemble_models: Optional[List[str]]=None`, `nli_ensemble_majority: int=2`. Default off so existing pre-§7.1 runs remain bit-stable.
  - `_bidirectional_nli` now returns a 4-tuple `(passes, p_fwd, p_bwd, ensemble_audit)`. The ensemble audit dict carries per-model probs + vote counts in both directions, persisted on each accepted variant under the new `nli_ensemble` field. All-abstain (every checkpoint failed to load) gracefully bypasses with a warning rather than crashing the run.
  - Members that fail to load are excluded from the vote majority — bypass is the safer behaviour for an infrastructure issue.

- **§8.1 — GPT-4o-mini paraphrase-quality judge (`llm_judge_paraphrases.py`, new)**
  - Two modes: `--mode rate` (score N stratified GenSens pairs on a 1–5 Likert scale) and `--mode calibrate` (measure Cohen's κ + Spearman ρ vs human gold ratings, e.g. RobustAlpacaEval's human-verified subset).
  - Deterministic judge call (`temperature=0.0, seed=42, max_tokens=4`) with a fixed system prompt + integer-extraction regex. Resume-safe (skips pair_ids already on disk).
  - Stratified sampling by `strategy` / `strategy_family` / `task` keys to control selection bias.
  - Local-only Spearman and Cohen's κ implementations so the script has no SciPy dependency. Verified on synthetic perfect-agreement / anti-agreement / partial-agreement inputs.

- **§8.3 — PAWS-style negative controls (`paws_negative_controls.py`, new)**
  - Two modes: `--mode generate` (template-generate 200 (base, NEAR-paraphrase) pairs across 5 perturbation types) and `--mode audit` (run the GenSens filter pipeline on the pairs and report rejection rate per type; target ≥ 90%).
  - Five perturbation generators with lexical-overlap-preserving rewrites:
    1. `subject_object_swap` (mean Jaccard 1.00)
    2. `polarity_flip` (mean Jaccard 0.80)
    3. `quantifier_swap` (mean Jaccard 0.65)
    4. `antonym_substitution` (mean Jaccard 0.72)
    5. `modifier_flip` (mean Jaccard 1.00)
  - Audit mode imports the production NLI helpers (`_get_nli_model`, `_nli_entail_prob`, `_get_nli_ensemble`, `_nli_ensemble_entail`) from `paraphrase_generator.py` when available so the audit faithfully mirrors the production filters. Falls back to a local CrossEncoder reimplementation when the import fails (e.g. running in a stripped environment).
  - CPU-friendly: no vLLM / no transformers AutoModel — only sentence-transformers + cross-encoder. Runs on a laptop.
  - Initial generated dataset (200 pairs) saved to `gensens/data/paws_negative_controls.jsonl` and ready for audit on the next GPU/laptop run.

**Why:**
- Multi-NLI ensemble: single-model NLI has ~5–10% false-positive rate on PAWS-style adversarial pairs (Zhang et al., NAACL 2019). A 2-of-3 majority across independently-trained checkpoints pushes the compound false-positive rate below 1%, at modest extra cost.
- LLM-as-judge: human annotation is out of scope per PUBLICATION_SPEC §8. GPT-4o-mini as the judge (with κ calibration against a human-verified subset) is the established substitute (Zheng et al. NeurIPS 2023; Liu et al. EMNLP 2023) and is the cheapest way to produce a defensible quality signal at scale.
- PAWS-style negative controls: validate that the filter pipeline is sensitive to semantic meaning, not just lexical overlap. Without this, "our filters retain 92% of human-verified paraphrases" is uninformative because we can't tell if they also retain 92% of intentional non-paraphrases. This is the "quality control on the quality control".

**Impact:**
- No change to existing GenSens datasets — both new scripts are additive validators, and the multi-NLI ensemble defaults OFF (existing runs reproduce bit-stable).
- Production GenSens runs that opt into the ensemble (`enable_nli_ensemble=True`) will have a slightly stricter filter; expect a small (~5–15%) drop in candidate yield, partly compensated by best-of-N (§6.5).
- New per-variant audit field `nli_ensemble` appears on accepted variants when the ensemble is active. JSONL-keyed downstream consumers are unaffected; positional readers will not break since variant dicts are written by key.

---

## [2026-06-04] — Tier-1 paraphrase generation improvements (PUBLICATION_SPEC §6)

**Files Modified:** `gensens/scripts/paraphrase_generator.py`, `PUBLICATION_SPEC.md` (new)

**What Changed:**

- **§6.1 — Bidirectional NLI gate now applies to ALL 4 tasks** (previously QA/dialogue only).
  - New module-level `_get_nli_model()` lazy-loads `cross-encoder/nli-deberta-v3-small` (cached singleton).
  - New `_nli_entail_prob(model, premise, hypothesis)` returns `P(entailment)` via softmax over [contradiction, entail, neutral] logits.
  - New `ParaphraseGenerator._bidirectional_nli(base, candidate)` runs both directions; rejects if either `P(entail) < τ_NLI` (default 0.50).
  - Graceful degradation: NLI model failures log once and return `(True, None, None)` so a missing-NLI environment never crashes generation; per-variant `nli_passed=False` flags it for downstream audit.
- **§6.2 — Length-ratio filter.** New `_passes_length_ratio()` rejects candidates whose word count falls outside `[0.70×, 1.50×]` of the base text. Default `length_ratio_min=0.70`, `length_ratio_max=1.50`.
- **§6.3 — SBERT semantic dedup.** New `_passes_semantic_dedup()` rejects candidates with cosine similarity > 0.95 to ANY already-accepted variant. The legacy first-8-words signature is retained as a fast pre-filter only.
- **§6.4 — Per-variant audit metadata.** Every accepted variant now carries `nli_entail_fwd`, `nli_entail_bwd`, `nli_passed`, `token_overlap_to_base`, `length_ratio`, `max_cos_to_accepted`, `strategy_family`, `best_of_n_index` alongside the existing `sbert_similarity` and `strategy` fields.
- **§6.5 — Best-of-N per strategy.** New `_generate_n_vllm()` requests `n=3` candidates per (instance, strategy) call to vLLM with `temperature=0.95, top_p=0.92, seed=42+strategy_idx`. All n candidates run through the full filter pipeline; the surviving candidate with the lowest mean token Jaccard to (base + already-accepted variants) is selected by `_select_most_diverse()`. Set `best_of_n=1` to recover legacy single-shot behaviour.
- **§6.6 — Strategy-family taxonomy.** New module-level `STRATEGY_FAMILY: Dict[str, str]` maps the 16 strategy names into 4 functional families (`lexical`, `syntactic`, `pragmatic`, `length`). New `MAX_PER_FAMILY = {'lexical': 2, 'syntactic': 2, 'pragmatic': 1, 'length': 1}` enforces per-family caps (total budget 6 prompt-engineering variants; slots 7–8 reserved for back-translation in a follow-up). The per-strategy `max_per_strategy` parameter is retained as a no-op for backward compatibility.
- **Reject diagnostics.** `generate_paraphrases()` now logs a per-instance breakdown of why each candidate was rejected (`empty_or_short`, `out_of_sbert_band`, `length_ratio_oob`, `semantic_dedup_collision`, `nli_failed`, `family_budget_exceeded`, …) so reviewers and us can audit filter behaviour.
- **PUBLICATION_SPEC.md (new)** at the repo root consolidates the full NeurIPS D&B 2026 submission plan: target venue, dual contribution framing, model roster, paraphrase quality validation strategy (4-pronged automated approach in lieu of human annotation), benchmark statistical properties, public release artefacts, budget, 6-week timeline, and acceptance criteria.

**Why:**
- The PAWS lesson (Zhang et al., NAACL 2019): SBERT cosine alone cannot distinguish paraphrases from high-overlap non-paraphrases. Single-task NLI was the only barrier on QA/dialogue; summarisation and creative were left unprotected.
- Length-ratio filter prevents truncation/expansion candidates from contaminating ROUGE-L-based metrics (AUC-E and KPIG-via-ROUGE are length-sensitive).
- Per-variant audit metadata is what reviewers need to verify our filters actually work; per-instance reject breakdown is what *we* need to tune them.
- Best-of-N (n=3) ~doubles effective diversity at essentially the same vLLM compute (continuous batching). This is the cheapest publication-grade improvement available.
- Strategy families enforce the methodological balance our "principled taxonomy" claim requires; the 16 ad-hoc strategies overlap heavily (~6 functional groups in practice).

**Impact:**
- Default behaviour CHANGES on existing GenSens datasets. Variants generated before this commit are not directly comparable — re-run the GenSens pipeline to get the post-spec dataset.
- New per-variant audit fields appear in every JSONL output. Downstream consumers that read by-key will not break; consumers that read by positional index will. The JSONL→CSV exporter in `csv_io.py` ignores unknown keys so existing CSV readers still work.
- Yield characteristics shift: more rejections (NLI now applies to all tasks) but better-quality variants. Dialogue/QA yield should improve slightly because best-of-N gives the NLI gate more candidates to evaluate per strategy.
- Backward compatibility: `enable_nli_gate=False` recovers the pre-spec QA/dialogue-only NLI behaviour; `best_of_n=1` recovers single-shot generation; `max_per_strategy` still works (per-family cap is the dominant constraint when both are active).

---

## [2026-06-04] — CRITICAL: Logit Lens NaN fix + IFI saturation fix (status-report blockers)

**Files Modified:** `prompt_robustness/src/logit_lens.py`, `prompt_robustness/src/sensitive_layer.py`, `prompt_robustness/src/csv_io.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/src/evaluator.py`, `prompt_robustness/tests/test_logit_lens_nan.py` (new), `prompt_robustness/diagnose_logit_lens.py` (new)

**What Changed:**

- **Logit Lens NaN fix (`logit_lens.py`).** The previous `compute_logits` cast `h_normed` BACK to `lm_head.weight.dtype` (typically fp16/bf16) before the matmul. For intermediate layers, the unembedding output routinely exceeded fp16's max (65504) → +inf → `log_softmax(inf)` = NaN → all per-layer PPL NaN. New code keeps the projection in fp32 when `cast_to_float32=True` by calling `F.linear(h_normed, self.lm_head.weight.float(), bias=...)`. Added a logit clamp at ±50 plus a token-log-prob clamp at [-50, 0] in `compute_per_token_ppl` so extreme but plausible cases never overflow `exp(-log_prob)`.
- **Sensitivity-curve robustness (`sensitive_layer.py`).** `compute_sensitivity_curve` now aggregates `Var_k[ log(mean_PPL_k) ]` instead of `Var_k[ mean_PPL_k ]`. Working in log space is (a) scale-invariant, (b) finite whenever PPL > 0, (c) what the LL-PIRC sensitivity story actually claims. Layers whose K paraphrases produce <2 finite PPL values are marked `NaN` and skipped downstream.
- **Inflection / z-score NaN handling (`sensitive_layer.py`).** `find_sensitive_layer_inflection` previously fed an all-NaN ΔS list into `max(deltas, key=...)`. Python's NaN comparisons always return False, so `max` returned the first delta — i.e. ℓ* = `scan_start + 1` for every article. New code skips any ΔS that depends on a non-finite endpoint and defers to the z-score fallback when no finite ΔS exists. `find_sensitive_layer_zscore` likewise filters out non-finite values before computing mean / std / threshold.
- **IFI saturation fix (`csv_io.py`, `benchmark.py`, `evaluator.py`).** `generate_responses_to_csv` (Phase 2a) now calls `mi.compute_perplexity_variance(...)` and `mi.compute_branching_factor(...)` while the subject model is still resident, and persists the two values on every variant row via new `ppl_var_inst` / `bf_inst` columns. `read_responses_grouped` surfaces them as `precomputed_ppl_var` / `precomputed_bf` on the sample dict. `evaluate_sample` prefers the precomputed values, then falls back to `model_interface.*`, then to `0.0`. Without this, the score-from-CSV phase had no generation model and `ppl_var = bf = 0.0` for every row → `IFI = 1.0` → diagnosis matrix collapsed to a single category (FLAWS §2.2 root cause).
- **Regression test (`tests/test_logit_lens_nan.py`, new).** Pytest fixture loads GPT-2 in fp16 and verifies: (1) per-token PPL is finite at every layer; (2) ≥50% of S(ℓ) values are finite for a 4-paraphrase scan; (3) the inflection detector defers when ΔS is all NaN (no silent `scan_start + 1` regression); (4) the inflection detector picks the largest finite ΔS in a partially-NaN curve.
- **Diagnostic CLI (`diagnose_logit_lens.py`, new).** Standalone script that loads any HF causal LM, runs a 4-paraphrase Logit Lens sweep, prints the S(ℓ) table, and detects the "ℓ* = scan_start + 1 AND mostly-NaN S" regression pattern. Exits non-zero on regression so it can be wired into CI later.

**Why:**

- The 5-instance pilot reported "variance reduction = 37.5%" but the underlying S(ℓ) curves were NaN for every layer in every article. Mechanistically there was no real ℓ* detection — the same default layer (`scan_start + 1 = 9`) was picked every time and the variance reduction was clamping at an arbitrary layer. Publishing the LL-PIRC mechanism claim without this fix would not survive review.
- The PRI benchmark pilot showed `PPL_var = BF = 0.0` for all 76 scored samples, forcing `Diagnostic_IFI = 1.0` and collapsing the 2×2 diagnosis matrix to one category ("Internally Stable / Output-Sensitive" for 71/76 = 93%). The dual-pillar diagnosis loses all power without genuine IFI.

**Impact:**

- **Breaking for any prior PIRC run.** All `pirc.json` / `eval_summary.json` files generated before this commit used a silently-degenerate ℓ* and should be re-run. The 5-instance pilot's "37.5% variance reduction" headline must be re-derived from a corrected run.
- **Breaking for any prior `scored_samples.csv` row** where IFI was used in diagnosis: the ppl_var / bf columns will be non-zero after re-running Phase 2a → 2b, and the diagnosis matrix will populate all four categories.
- **Backward-compatible for old responses.csv.** `_migrate_schema_if_needed()` (already present in `IncrementalCSVWriter`) auto-adds the new `ppl_var_inst` / `bf_inst` columns as blank to legacy files; missing values fall back to None → 0.0 on read, matching pre-fix behaviour.
- **S(ℓ) values are now log-scale variance, not raw-scale.** Absolute magnitudes are NOT comparable to any pre-fix curves, but the ranking of layers (which drives ℓ*) is the meaningful signal.

---

## [2026-05-30] — Phase 6: weighted harmonic mean sub-components for Diagnostic_ORI / Diagnostic_IFI

**Files Modified:** `prompt_robustness/src/scores.py`, `prompt_robustness/src/config.py`, `prompt_robustness/src/evaluator.py`

**What Changed:**

- `compute_diagnostic_ori` now accepts per-axis weights and defaults to:
  - SMS = 0.40 (primary semantic stability)
  - AUC-E = 0.30 (performance elasticity)
  - KPIG = 0.20 (reference coverage, independent of SMS)
  - 1-TRD = 0.10 (collinear with SMS at r=-0.98 per FLAWS §3.7, demoted to tie-breaker)
- `compute_diagnostic_ifi` now accepts per-axis weights and defaults to:
  - 1-PPL_var = 0.50 (cleanest intra-model stability signal)
  - 1-BF = 0.30 (noisier entropy-derived signal)
  - 1-PC_stab = 0.20 (optional confidence-stability axis when present)
- `Config` gains `diag_ori_w_sms/auc_e/kpig/trd` and `diag_ifi_w_ppl_var/bf/pc_stab`, each overridable via env vars (`DIAG_ORI_W_*`, `DIAG_IFI_W_*`).
- `evaluator.py` passes Config sub-weights into `compute_diagnostic_ori` / `compute_diagnostic_ifi`.
- `weighted_harmonic_mean` semantics unchanged (still returns 0 when any axis is exactly 0 — preserves the strict diagnostic property).

**Why:**
- Equal weighting let TRD's r=-0.98 redundancy with SMS double-count consistency in Diagnostic_ORI, and an isolated TRD spike tanked the composite (e.g. SMS/AUC-E/KPIG=0.85, TRD=0.9 dropped HM to 0.30 even though only one axis disagreed).
- Empirically motivated weights from the FLAWS §3 audit (TRD-SMS collinearity, PPL_var being cleaner than BF) improve discriminability while preserving the harmonic-mean "any zero → zero" property.

**Impact:**
- Uniform-input scores unchanged (HM with normalized weights on equal values returns the value itself).
- TRD-only failure mode now scores ~0.49 instead of ~0.30 — SMS/AUC-E/KPIG still dominate.
- Old equal-weight runs are reproducible by setting all `DIAG_*_W_*=1.0`.
- `pri` (arithmetic) ranking score is **unchanged** — Phase 6 only affects `diagnostic_ori`, `diagnostic_ifi`, `diagnostic_pri`, and `diagnosis` (the APPENDIX-only diagnostic stack).

---

## [2026-05-30] — Phase 5: dataset quality from FLAWS_AND_FIXES.pdf §6

**Files Modified:** `gensens/scripts/paraphrase_generator.py`, `gensens/scripts/audit_paraphrase_quality.py` (new), `gensens/scripts/adversarial_paraphrases.py` (new)

**What Changed:**

- **§6.1 GenSens paraphrase quality audit** (`audit_paraphrase_quality.py` — new)
  - Standalone CLI that samples N pairs from a GenSens JSONL dataset, runs DeBERTa-v3-MNLI (configurable) in BOTH directions on each (base, paraphrase), and reports the fraction with bidirectional entailment.
  - Output: per-pair CSV + summary breakdown (bidirectional / one-direction / neither).
  - Goes in the paper as the paraphrase-quality appendix table.
- **§6.2 Adversarial paraphrase subset** (`adversarial_paraphrases.py` — new)
  - Five deterministic no-model perturbation families: `typo` (8% char-swaps), `sentence_reorder`, `double_negation` (insert "not not" after a copula — ¬¬X ≡ X but heavy lexical shift), `hedged` (prepend "Arguably," / "It might be the case that"), `formality_shift` (flip contractions both ways).
  - `adversarial_paraphrases(text, n, seed)` driver returns up to n distinct variants tagged `adv:<strategy>`, filtering no-ops.
  - PRI should be reported separately on the easy GenSens subset vs the adversarial subset.
- **§6.3 Dialogue task density** (`paraphrase_generator.py`)
  - Added `task_thresholds` dict on `ParaphraseGenerator` with `DEFAULT_TASK_THRESHOLDS = {"dialogue": 0.78}`.
  - The variant filter looks up the per-task threshold, defaulting to the global `similarity_threshold` (0.82) for everything else. Restores dialogue variant yield without polluting summarization/QA/creative.
- **§6.4 Task coverage:** Code-gen / GSM8K / MMLU-Pro / FLORES / SST-2 additions are dataset-level extensions, not code changes here — tracked as a follow-up.

**Why:**
- SBERT cos-sim alone passes adversarial paraphrases that flip meaning; an NLI audit gives reviewers an additional defensible quality signal.
- "Easy" LLM rewrites do not stress-test robustness; the adversarial subset gives the paper a clean "robustness under attack" narrative.
- Dialogue's SBERT filter at 0.82 was killing most candidates (only ~2.8 of N survived) — loosening the band restores variant density without changing global behaviour.

**Impact:**
- No change to existing dataset generation behaviour by default. Pass `task_thresholds={"dialogue": 0.78}` (or rely on the default merge) to opt into the loosened dialogue band.
- New audit/adversarial scripts are opt-in and require no changes to downstream consumers.
- Adversarial bucket should be scored AND reported separately so easy-subset numbers stay comparable to prior runs.

---

## [2026-05-30] — Phase 4: LL-PIRC mitigation track from FLAWS_AND_FIXES.pdf §5

**Files Modified:** `prompt_robustness/experiment_pirc.py`, `prompt_robustness/src/sensitive_layer.py`, `prompt_robustness/src/mitigation_baselines.py` (new)

**What Changed:**

- **§5.3 ℓ\* ablation knob** (`sensitive_layer.py`, `experiment_pirc.py`)
  - `SensitiveLayerDetector` gains a `force_ell_star: Optional[int]` constructor argument; when set, `detect()` returns that layer directly and the sensitivity curve still gets computed for logging.
  - `experiment_pirc.py` exposes `--ell-star L` on the CLI which writes `config['sensitive_layer']['force_ell_star']` and threads it through to the detector. Enables sweeping ℓ\* ∈ {6, 9, 12, 18, 24, 28} from the shell.
- **§5.4 α ablation knob** (`experiment_pirc.py`)
  - Added `--alpha A` CLI flag that overrides `config['pirc']['alpha']`. The existing dev-tuning code path also already supports `alpha_values` lists, so both manual single-α runs and grid sweeps are now first-class.
- **§5.5 Mitigation baselines for the method paper** (`src/mitigation_baselines.py` — new module)
  - Four no-training inference-time baselines that LL-PIRC must beat:
    1. `temperature_smoothing` — sample n=4 outputs per paraphrase at T=0.7, aggregate (default: longest).
    2. `self_consistency_vote` — classification mode returns the modal answer; generation mode picks the ROUGE-L centroid response.
    3. `system_prompt_stabilize` — prepends a "be stable across rephrasings" system prompt to every paraphrase.
    4. `in_context_learning` — builds K-1 paraphrase exemplars in context before the real prompt.
  - All four share a common `(prompts, generate_fn, **kwargs) -> List[str]` signature so they can be plugged into the same evaluation harness as PIRC.
  - `BASELINES` dict registry for the future `experiment_baselines_comparison.py` driver.
- **§5.1 / §5.2 / §5.6 deferred:** Scale verification at n=200, article-1 outlier re-check, and the mechanistic story for ℓ\* all require GPU runs (no code changes) and are tracked as follow-ups.

**Why:**
- The method paper requires both ablations (ℓ\*, α) and comparisons against the strongest no-training baselines. Without these, reviewers cannot tell whether PIRC's gains come from clamping specifically or from "any test-time intervention helps."

**Impact:**
- No behavioural change for default `experiment_pirc.py` runs.
- New CLI knobs `--alpha` / `--ell-star` enable ablation matrices from the shell.
- `mitigation_baselines.py` is library-only — the corresponding experiment driver is still TBD; the four functions are unit-smoke-tested in this commit.

---

## [2026-05-30] — Phase 3: reproducibility & infrastructure from FLAWS_AND_FIXES.pdf §4

**Files Modified:** `requirements_gpu_pinned.txt` (new), `prompt_robustness/src/utils.py`, `prompt_robustness/main.py`, `prompt_robustness/experiment_baseline.py`, `prompt_robustness/experiment_pirc.py`, `prompt_robustness/evaluate.py`

**What Changed:**

- **§4.1 Dependency pinning** (`requirements_gpu_pinned.txt`)
  - New strict-pin requirements file alongside the existing range-based one. Locks transformers, vllm, autoawq, gptqmodel (for the AWQ Marlin kernel), sentence-transformers, datasets, rouge-score, scikit-learn, numpy, pandas, etc. to exact versions known to work with torch 2.6.0+cu124 on A100.
- **§4.2 Deterministic seeding** (`src/utils.py`, all entry points)
  - Added `set_global_seed(seed=42)` — seeds Python `random`, NumPy, torch (CPU + CUDA), enables cuDNN determinism + warn-only `torch.use_deterministic_algorithms`, sets `PYTHONHASHSEED`.
  - Wired into `main.py`, `experiment_baseline.py`, `experiment_pirc.py`, `evaluate.py` immediately after the config is loaded.
- **§4.4 Crash-safe JSON checkpointing** (`src/utils.py`, baseline + pirc experiments)
  - Added `JsonlCheckpointWriter` — append-only JSONL writer with `flush + os.fsync` after every record, plus a tolerant `read_all` that survives a truncated last line (the typical crash pattern).
  - `experiment_baseline.py` now streams `baseline.jsonl` + `ifi_metrics.jsonl` (one article per line) alongside the existing all-at-once `baseline.json` / `ifi_metrics.json` final aggregates. The legacy summary `baseline_checkpoint.json` is still consumed for backward compatibility but JSONL is preferred when present.
  - `experiment_pirc.py` does the same with `pirc.jsonl`.
- **§4.3 / §4.5 deferred:** Docker image and CI smoke test are deferred to a later commit — they require infra changes (Dockerfile + GitHub Actions workflow) beyond the scope of the code base.

**Why:**
- Range-pinned requirements broke during the A100 bootstrap (CUDA/torch/vllm lock-step). Exact pins eliminate that whole class of breakage for re-runners.
- Bootstrapping / NLI / sampling baselines all touch RNGs; without a single seed call, re-runs drift even with `do_sample=False`.
- All-at-once JSON writes lose every computed article on a single crash; per-article fsync'd JSONL preserves work.

**Impact:**
- No behavioural change in steady-state results.
- Re-runs are now bit-reproducible up to non-deterministic GPU kernels (cuDNN determinism is best-effort).
- Output filenames `baseline.jsonl`, `ifi_metrics.jsonl`, `pirc.jsonl` are new artifacts. Downstream tooling that hard-codes `baseline.json` is unaffected — the JSON aggregate is still written.

---

## [2026-05-30] — Phase 2: methodological gaps from FLAWS_AND_FIXES.pdf §3

**Files Modified:** `prompt_robustness/src/auc_e_metric.py`, `prompt_robustness/src/evaluator.py`, `prompt_robustness/src/config.py`, `prompt_robustness/src/embeddings.py`

**What Changed:**

- **§3.4 AUC-E definition / auditable curve** (`auc_e_metric.py`, `evaluator.py`)
  - `compute_auc_e_metric` now accepts `return_curve=True` and emits the underlying (variant_idx, rougeL) curve plus `mean`, `std`, `cv` summary.
  - Documented axes: x = variant index (discrete paraphrase order from the dataset, NOT continuous perturbation strength); y = ROUGE-L F-measure.
  - `evaluate_sample` calls `compute_auc_e_metric(..., return_curve=True)` and stores the curve under `result["auc_e_curve"]` — fully auditable now.
- **§3.6 PRI weight ablation hooks** (`config.py`, `evaluator.py`)
  - Added `Config.pri_w_consistency / pri_w_quality / pri_w_faithfulness` with env-var overrides `PRI_W_CONSISTENCY`, `PRI_W_QUALITY`, `PRI_W_FAITHFULNESS`.
  - `evaluate_sample` passes these into `compute_pri`. Recommended ablation matrix: default (0.40 / 0.35 / 0.25) vs equal (0.333 / 0.333 / 0.334) vs learned (fit to downstream task accuracy).
- **§3.5 Cross-embedder ablation documentation** (`embeddings.py`)
  - Documented the four-embedder ablation set: BAAI/bge-large-en-v1.5 (default), Alibaba-NLP/gte-Qwen2-7B-instruct, intfloat/e5-mistral-7b-instruct, mixedbread-ai/mxbai-embed-large-v1.
  - Note: ranking stability across these should hit Spearman ρ > 0.7 to defend against the "rankings are an embedder artifact" reviewer objection.

**Why:**
- AUC-E was a "magic number" without published axes; reviewers couldn't verify the construct.
- PRI weights `0.40 / 0.35 / 0.25` were asserted, not justified — ablations require switching weights without forking the score formula.
- Cross-embedder validation is a near-mandatory ablation for any embedding-based benchmark; the infrastructure already existed, just lacked guidance.

**Impact:**
- AUC-E scalar values unchanged (same formula). New `auc_e_curve` field appears in result JSON but is not in `SCORED_COLS` (intentional — curve goes in `<run_id>.json` only).
- PRI numbers are unchanged with defaults; running with the env vars produces ablation variants.
- No new dependencies.

**§3.7 already done in this codebase:** `evaluator.py` already substitutes `trd_semantic` for the legacy length-based TRD inside ORI (`metrics["trd"] = trd_semantic`). The original §3.7 recommendation is satisfied by R6 in earlier changelog entries.

---

## [2026-05-30] — Phase 1: critical bug fixes from FLAWS_AND_FIXES.pdf §2

**Files Modified:** `prompt_robustness/src/anchor_tokens.py`, `prompt_robustness/src/correctness_metric.py`, `prompt_robustness/src/csv_io.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/src/scores.py`

**What Changed:**

- **§2.1 Anchor selector NaN/inf handling** (`anchor_tokens.py`)
  - Added `_sanitize_for_ranking()` helper that masks non-finite mean_ppl / var_ppl with the dtype max (sentinel sorts to the END of argsort), and returns a `finite_mask` for hard exclusion.
  - Anchor candidate pool is now filtered to finite positions only; non-finite positions can never win the "most stable" lottery again.
  - Logs the count of non-finite positions excluded at layer ℓ\*.
- **§2.2 IFI saturation fix** (`csv_io.py`, `benchmark.py`)
  - Extended `SCORED_COLS` with `ppl_var_norm`, `bf_norm`, `ifi_norm` (filled in post-stream, blank during incremental writing).
  - Added `csv_io.normalize_ifi_per_model()` — reads `scored_samples.csv`, computes per-model min-max normalization of `ppl_var` and `bf`, recomputes `ifi_norm = 1 - (ppl_var_norm + bf_norm) / 2`, rewrites the CSV in place.
  - `benchmark_models()` calls the normalizer once after the streaming writers close.
- **§2.3 PRI vs Diagnostic_PRI policy** (`scores.py`)
  - Added a documentation block declaring `pri` (arithmetic) the PRIMARY ranking score for main tables and `diagnostic_pri` (harmonic) APPENDIX-ONLY. The two measure different constructs by design and must never be shipped as if they agree.
- **§2.4 CS composition rebalance** (`correctness_metric.py`)
  - OLD: `CS = 0.5 * semantic + 0.5 * coverage` (collinear with coverage at r ≈ 0.96).
  - NEW: `CS = 0.40 * semantic + 0.20 * length_adequacy + 0.40 * coverage`.
  - Added `_length_adequacy(resp_len, ref_len)` — full credit inside the band `[0.5 * ref_len, 2 * ref_len]`, linearly decayed outside (zero at len=0 and at 4×ref_len).
  - Retained the `coverage < 0.2 → CS × 0.6` gate so semantic+length cannot rescue a hallucinated response.

**Why:**
- Anchor selector NaN/inf bug let PIRC clamp on the *worst* tokens (FLAWS §2.1 — explains a large portion of mitigation failure).
- IFI saturated at 1.0 collapsed `Diagnostic_PRI = HM(ORI, IFI)` to `Diagnostic_ORI` (FLAWS §2.2).
- Arithmetic PRI and harmonic Diagnostic_PRI were Pearson-anti-correlated (r ≈ -0.13); shipping both unlabeled was misleading (FLAWS §2.3).
- CS was empirically `Coverage` in disguise (r=0.96), undermining the multi-axis quality framing (FLAWS §2.4).

**Impact:**
- All previously scored PRI / CS / IFI numbers are NUMERICALLY DIFFERENT under the new formula. Existing `scored_samples.csv` outputs must be re-aggregated (or re-generated) before publication.
- PIRC results that depended on the broken anchor pool must be re-run end-to-end.
- `scored_samples.csv` schema now includes three additional columns; `csv_io.IncrementalCSVWriter._migrate_schema_if_needed()` handles old-format files transparently.
- Breaking change for any downstream consumer that hard-codes the SCORED_COLS list.

---

## [2026-05-29] — Critical PIRC mitigation bug fix: prefill-only clamping

**Files Modified:** `prompt_robustness/src/pirc.py`

**What Changed:**
- `generate_with_clamping()` now applies the forward hook **only during the prefill pass** (the single forward where `seq_len == prompt_len`), not on every per-token decode step.
- Added a `clamp_state["prefilled"]` latch closed-over by the hook so subsequent calls (each decode step) short-circuit and pass `output` through untouched.

**Why:**
- Previously the hook ran on every forward call. During `model.generate(...)` with KV cache, decode steps see `h.shape[1] == 1` (just the new token). The hook then executed `h[:, :1][:, mask] = mean_h_device[mask]`, overwriting the new token's hidden state with `mean_h[0]` — the *prompt's first-anchor* consensus. Every newly generated token's representation got pinned to the same anchor, collapsing autoregressive generation into a degenerate loop ("Here://://://...").
- Symptom in the 5-instance test: PIRC at α=1.0 and α=0.5 produced byte-identical degenerate outputs, identical PRI-relative ROUGE-L (0.012), identical 47% "variance reduction" — the alpha knob didn't matter because the model output was destroyed before it could differ.

**Impact:**
- Mitigation is now actually functional. Previous PIRC results (any run before this commit) are INVALID and must be re-run.
- ℓ\* detection, baseline pipeline, and Phase 1/2 scoring are unaffected — only the clamped-generation step changed.
- Re-run `experiment_pirc.py` to regenerate `pirc.json` / `eval_summary.json` / plots.

---

## [2026-05-29] — A100 instance bootstrap hardening (env / generator / quantization)

**Files Modified:** `run_on_gpu.sh`, `gensens/scripts/generate_summ_instructions.py`

**What Changed:**
- `run_on_gpu.sh`:
  - Source `$ROOT/.env` after the helper functions are defined, so `HF_TOKEN`, `MODEL_NAME`, `GENERATOR_MODEL`, `JUDGE_MODEL`, `EMBEDDER_MODEL`, etc. propagate to every Python subprocess (vLLM, HF, huggingface-cli).
  - Mirror `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` for older `huggingface_hub` codepaths.
  - Export `VLLM_WORKER_MULTIPROC_METHOD=spawn` to eliminate the residual fork/CUDA risk in vLLM 0.8+.
  - Resolve the Phase 0/1 `MODEL_ID` *after* loading `.env` with precedence `--model-id CLI > MODEL_ID env > GENERATOR_MODEL (.env) > Qwen 7B fallback`. The previous order took the Qwen 7B fallback before `.env` had a chance to set `GENERATOR_MODEL=...Llama-3.1-70B-AWQ-INT4`.
  - Build `QUANT_ARG="--quantization $GENERATOR_QUANTIZATION"` and pass it to both `generate_summ_instructions.py` (Phase 0) and `generate_dataset.py` (Phase 1) so the 70B AWQ generator loads with the Marlin kernel instead of trying to allocate fp16 weights (instant OOM otherwise).
  - Simplified `ensure_torch()` to use a real `torch.zeros(1).cuda()` op as the compatibility probe instead of parsing driver/version strings.
- `generate_summ_instructions.py`:
  - Added `--quantization` CLI flag and threaded it through `load_vllm_model(..., quantization=...)` into `LLM(quantization=...)`.

**Why:**
- Without `.env` sourcing, gated-model downloads (Llama-3.1) returned `401 Unauthorized` because the bash environment never saw `HF_TOKEN`.
- Without the new `MODEL_ID` resolution order, Phase 0/1 silently fell back to Qwen 7B even when `.env` declared the 70B AWQ generator — a methodology mismatch vs. `config.yaml` which mandates the 70B paraphraser.
- Without `--quantization`, vLLM would attempt to load `Meta-Llama-3.1-70B-Instruct-AWQ-INT4` as fp16 (~140 GB) and OOM immediately on an 80 GB A100.

**Impact:**
- No metric / formula changes — purely orchestration hardening so the pipeline boots cleanly on a fresh A100 instance.
- Existing local results unaffected; no re-run required.

---

## [2026-05-29] — Sync remote-only fixes back to local repo; pin torch/vLLM stack

**Files Modified:** `gensens/scripts/paraphrase_generator.py`, `requirements_gpu.txt`, `run_on_gpu.sh`

**What Changed:**
- `paraphrase_generator.py`: swap SBERT / vLLM init order — load vLLM **before** `SentenceTransformer(...)`. (This fix had only been applied on the now-deleted remote instance and was missing from the repo.)
- `requirements_gpu.txt`: pin `vllm>=0.8.0,<0.9.0` (was `vllm>=0.5.0`). Updated install header to recommend `torch==2.6.0+cu124` instead of cu121.
- `run_on_gpu.sh`: new `ensure_torch()` helper called from `ensure_deps()`. It detects the case where the pre-installed torch is built for CUDA 13.0 but the driver only supports CUDA ≤ 12.x and force-reinstalls `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` / `torchaudio==2.6.0` from the cu124 index. The downstream `pip install` line also pins `vllm>=0.8.0,<0.9.0`.

**Why:**
- vLLM v1 (0.8+) uses multiprocessing `fork` by default. `SentenceTransformer(...)` initializes CUDA on construction; if it runs before vLLM forks, every worker crashes with *"Cannot re-initialize CUDA in forked subprocess"*. Loading vLLM first guarantees the fork happens before CUDA is touched in the parent.
- vLLM 0.21+ pins `torch==2.11.0+cu130`. That wheel needs an NVIDIA driver ≥ 570.124 (CUDA 12.8). The A100 instance we provisioned shipped driver 570.86 (CUDA 12.8 max ABI 12080), so every CUDA call failed with *"NVIDIA driver too old"*. Capping at vllm 0.8.x keeps us on torch 2.6/cu124, which works on essentially every A100/H100 driver in use today.

**Impact:**
- No metric / formula changes — these are environment-setup fixes only.
- Existing results remain valid; no re-run required.
- Fresh GPU instances should now bootstrap successfully via `bash run_on_gpu.sh`.

---

## [2026-05-29] — Upgrade default embedder to BAAI/bge-large-en-v1.5; fix kpig_metric dead code

**Files Modified:** `prompt_robustness/src/embeddings.py`, `prompt_robustness/src/kpig_metric.py`, `prompt_robustness/src/evaluator.py`, `.env.example`, `prompt_robustness/config.yaml`

**What Changed:**
- `EmbeddingHelper._DEFAULT` changed from `sentence-transformers/all-mpnet-base-v2` to `BAAI/bge-large-en-v1.5`.
- `kpig_metric.py`: Removed hardcoded global `SentenceTransformer("all-MiniLM-L6-v2")` and `get_model()`. `compute_kpig_metric()` now accepts a shared `embedder` parameter (lazy-creates `EmbeddingHelper()` if None).
- `evaluator.py`: Removed dead `from .kpig_metric import compute_kpig_metric` import — the pipeline uses `compute_kpig_advanced` from `metrics_advanced.py`; the old import never executed and was loading an unused 22M model definition.
- `.env.example`, `config.yaml`: Updated embedder references to `BAAI/bge-large-en-v1.5`.

**Why:**
- `bge-large-en-v1.5` (335M, 1024-dim, MTEB ~64) is a well-cited research embedder (Xiao et al. 2023) with a clear quality advantage over `all-mpnet-base-v2` (MTEB ~57) and negligible extra cost (1.3 GB vs 0.4 GB).
- The hardcoded `all-MiniLM-L6-v2` in `kpig_metric.py` would have loaded a separate 22M model inconsistent with the shared embedder if `compute_kpig_metric` were ever called.
- Dead import removed to keep the codebase clean.

**Impact:**
- Breaking change: SMS/KPIG/TRD scores will differ from any prior runs (different embedding geometry). Re-run required.
- No change to formulas, weights, or pipeline structure.
- GenSens paraphrase filtering (`paraphrase_generator.py`) still uses `all-mpnet-base-v2` — that is intentional and unchanged.

---

## [2026-05-29] — Fix Llama/gated model loading: HF token, AWQ config, seq2seq detection

**Files Modified:** `prompt_robustness/src/model_interface.py`

**What Changed:**
- Added `_hf_token()` helper — reads `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from env; passed explicitly to every `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, and `AutoConfig.from_pretrained` call. Logs a warning at startup if unset.
- Added `_KNOWN_CAUSAL` fast-path in `_is_seq2seq()` — Llama/Mistral/Qwen/etc. return `False` immediately without making a network call to HuggingFace. Previously every Llama load called `AutoConfig.from_pretrained` which hit the gated-model auth gate.
- Added `_apply_quantization_kwargs()` — wires `quantization="awq_marlin"` into `AwqConfig(bits=4, backend="marlin")` (transformers ≥4.44) or falls back to autoawq direct loading. Previously the `quantization` param was stored but never passed to `from_pretrained`, so the 70B AWQ model loaded as raw fp16 → OOM.
- Added `max_length=2048` to the tokenizer call in `generate_responses()` — prevents runaway padding on 128K-context Llama models when tokenizing long article prompts in a batch.

**Why:**
- Llama models are gated on HuggingFace — without an explicit `token=` they fail with `OSError: Access to model is restricted`.
- The `quantization="awq_marlin"` value was silently ignored; loading 70B without AWQ quantization requires ~140 GB fp16 → immediate OOM on an 80 GB card.
- `_is_seq2seq` calling `AutoConfig.from_pretrained` on every model caused a redundant auth round-trip and, for gated models with a missing token, an exception that hid the real error.

**Impact:**
- All 4 subject models (3×8B/7B + 1×70B AWQ) should now load correctly on A100/H100.
- No metric or formula changes. Re-run not required if prior results were generated with working models.

---

## [2026-05-29] — Fix pipeline for full 200-instance, 4-task run

**Files Modified:** `prompt_robustness/src/config.py`, `prompt_robustness/src/benchmark.py`, `run_on_gpu.sh`

**What Changed:**
- **`config.py`**: `max_new_tokens` default raised from 50 → 256 (env: `MAX_NEW_TOKENS`). Added `max_samples: int` field (env: `MAX_SAMPLES`, 0 = no limit).
- **`benchmark.py`**: Removed hardcoded `[:100]` slice in both `benchmark_models()` and `generate_responses_to_csv()`. Both now use `config.max_samples` (0 = process all instances).
- **`run_on_gpu.sh` Phase 2**: After Phase 1, all 4 task JONSLs (summarization, creative, dialogue, qa) are merged via `cat` into a single `gensens_all_Ninst_Kvar.jsonl`. Phase 2 receives the combined file so all 4 tasks are benchmarked in one run.

**Why:**
- 50 tokens was cutting off summarization outputs mid-sentence; 256 tokens fits full summaries.
- The `[:100]` hardcode silently truncated a 200-instance dataset to 100 — half the data was never evaluated.
- Only the summarization JSONL was passed to Phase 2; creative/dialogue/QA instances were never benchmarked.

**Impact:**
- All 200 instances × 4 tasks = 800 samples are now benchmarked per model.
- Outputs will be longer and more coherent (256 tokens vs 50).
- Re-run required — prior results used truncated data and short outputs.

---

## [2026-05-29] — Fix gte-Qwen2-7B-instruct under transformers 5.7 via native Qwen2 path + EOS pooling

**Files Modified:** `prompt_robustness/src/embeddings.py`

**What Changed:**
- Added `_is_gte_qwen2()` detector and `_GteQwen2Embedder` class in `embeddings.py`.
- When `model_name` contains `gte-qwen2`, `EmbeddingHelper` now uses `_GteQwen2Embedder` instead of `SentenceTransformer(trust_remote_code=True)`.
- `_GteQwen2Embedder` loads the model with `trust_remote_code=False` (uses transformers' native `Qwen2Model`) and implements EOS-token pooling manually (last non-padding token, right-padded batches, L2-normalised).
- Default embedder remains `all-mpnet-base-v2`; opt-in to 7B via `EMBEDDER_MODEL=Alibaba-NLP/gte-Qwen2-7B-instruct`.

**Why:**
- `trust_remote_code=True` uses the HF-cached `modeling_qwen.py` which has two bugs under transformers ≥5.7:
  (a) `inv_freq` (`persistent=False` buffer) is corrupted during `from_pretrained` → RoPE cos/sin all-zero → Q/K norms collapse → cosine sim ≈ 0.01 for all pairs.
  (b) Left-padded batch attention mask is inverted in the custom attention prep code → further breaks similarity.
- `trust_remote_code=False` falls back to the native `Qwen2Model` in transformers which is fully 5.7-compatible and has neither bug.
- EOS-token pooling (last non-padding token) is the correct pooling strategy for the GTE-Qwen2 model family.

**Impact:**
- `gte-Qwen2-7B-instruct` now produces correct cosine similarities when requested via `EMBEDDER_MODEL`.
- No change to default behaviour (`all-mpnet-base-v2` still default); existing results unaffected.
- If `EMBEDDER_MODEL=Alibaba-NLP/gte-Qwen2-7B-instruct` is set, previous results must be discarded (prior runs had SMS=0 due to the bugs above).

---

## [2026-05-29] — Switch default embedder from gte-Qwen2-7B-instruct to all-mpnet-base-v2

**Files Modified:** `prompt_robustness/src/embeddings.py`

**What Changed:**
- `EmbeddingHelper._DEFAULT` changed from `"Alibaba-NLP/gte-Qwen2-7B-instruct"` to `"sentence-transformers/all-mpnet-base-v2"`.
- Removed the now-unnecessary `_repair_rope_caches()` helper function and `_patch_qwen2_config()` invocation that attempted to fix corrupted RoPE buffers.
- `trust_remote_code` in `SentenceTransformer(...)` is now set dynamically: `True` only when the model name contains `"qwen"` or `"gte"`, `False` otherwise (covers `all-mpnet-base-v2` and other standard ST models correctly).

**Why:**
- `gte-Qwen2-7B-instruct` uses a custom `modeling_qwen.py` that is fundamentally incompatible with transformers ≥5.7. Two separate bugs were confirmed:
  1. The `inv_freq` buffer (declared `persistent=False`) is corrupted during `from_pretrained` in transformers 5.7, causing RoPE to produce all-zero cosine/sine caches → Q and K norms collapse to ~0 → all attention outputs are degenerate.
  2. Left-padded batches trigger an inverted attention mask path in the custom model code, further breaking similarity computation.
- As a result, all pairwise cosine similarities were ~0.01 regardless of semantic content → SMS clamped to 0.0 for every sample.
- `all-mpnet-base-v2` (768-dim) is fully compatible with transformers 5.7, already cached on the GPU instance, and produces correct cosine similarities for semantically equivalent paraphrases.

**Impact:**
- **Breaking change**: SMS scores are now meaningful and non-zero. All previous results generated with `gte-Qwen2-7B-instruct` as embedder (SMS=0.0 for most samples) must be discarded and re-run.
- PRI and Final Score will change wherever SMS drove the Consistency component.
- Scores are NOT comparable across embedder choices (different geometry/dimensionality). Always note the embedder in result filenames or metadata.

---

## [2026-05-29] — Remove all small models from project; GPU-only pipeline

**Files Modified:** `prompt_robustness/src/config.py`, `prompt_robustness/src/llm_judge.py`, `prompt_robustness/src/embeddings.py`, `prompt_robustness/src/faithfulness_metric.py`, `gensens/scripts/paraphrase_generator.py`, `gensens/scripts/generate_dataset.py`

**What Changed:**
- **`paraphrase_generator.py`**: Removed `local` backend (flan-t5-large), `_load_flan()`, `_generate_single_flan()`, `_build_flan_prompt()`, deberta NLI gate (`_NLI_MODEL_NAME`, `_get_nli_model()`, `_nli_entail_prob()`), and all NLI-gate logic in `generate_paraphrases()`. Only `"vllm"` and `"llama"` backends remain. Default backend changed from `"local"` → `"vllm"`.
- **`generate_dataset.py`**: `--model` default changed from `"local"` → `"vllm"`; `"local"` removed from choices.
- **`llm_judge.py`**: Removed seq2seq judge path entirely — `judge_model`/`judge_tokenizer`/`judge_device` globals, `_judge_is_seq2seq()`, `get_judge()`, `llm_judge()`, `AutoModelForSeq2SeqLM` import. `JUDGE_MODEL` now hardcodes `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` as default. `llm_judge_with_raw()` always uses the causal 70B path.
- **`embeddings.py`**: Removed `all-MiniLM-L6-v2` (22M) default and `_default_embedder()` conditional. Default is now always `Alibaba-NLP/gte-Qwen2-7B-instruct`.
- **`faithfulness_metric.py`**: Removed deberta NLI backend entirely — `_MODEL_NAME`, `_model`, `_load_failed`, `_get_model()`, `_softmax()`, `nli_available()`, `_resolve_faithfulness_backend()`, and all `"nli"` backend code paths in `compute_faithfulness()` and `compute_faithfulness_with_raw()`. Only the LLM-as-NLI path remains. `_LLM_MODEL` defaults to the 70B-AWQ judge.
- **`config.py`**: Removed `_LOCAL_DEFAULT_MODELS`, runtime CUDA check, and RuntimeError guard. `_resolve_default_models()` simply returns the GPU subject list when `MODEL_NAME` env is unset. `device` default changed from `"cpu"` → `"cuda"`.

**Why:**
- All small models are removed so they cannot accidentally activate on GPU — via missing env vars, wrong defaults, or code paths that were never explicitly disabled.
- Small-model code will be reconnected later when a local-dev / CPU path is needed again.

**Impact:**
- Local / CPU / MPS execution is broken until small models are reconnected. This is intentional.
- No formula or metric changes. Re-run required if any prior runs used small-model defaults.

---

## [2026-05-29] — Enforce ≥4B models on CUDA; eliminate all <4B silent fallbacks on GPU

**Files Modified:** `prompt_robustness/src/config.py`, `prompt_robustness/src/llm_judge.py`, `prompt_robustness/src/embeddings.py`, `prompt_robustness/src/faithfulness_metric.py`, `gensens/scripts/paraphrase_generator.py`

**What Changed:**
- **`config.py`**: `models` field no longer defaults to the <2B laptop list on CUDA. New `_resolve_default_models()` auto-detects `torch.cuda.is_available()`: on GPU it defaults to the four ≥7B subjects from `config.yaml roles.subjects`; on CPU/MPS it keeps the original laptop models. Raises `RuntimeError` at startup if `MODEL_NAME` is explicitly set to any <4B model on a CUDA machine.
- **`llm_judge.py`**: `JUDGE_MODEL` no longer hard-defaults to `flan-t5-large`. New `_resolve_judge_model()` returns `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` on CUDA and `flan-t5-large` on CPU/MPS.
- **`embeddings.py`**: `EmbeddingHelper` no longer hard-defaults to `all-MiniLM-L6-v2` (22M). New `_default_embedder()` returns `Alibaba-NLP/gte-Qwen2-7B-instruct` on CUDA and `all-MiniLM-L6-v2` on CPU/MPS.
- **`faithfulness_metric.py`**: `_BACKEND` no longer hard-defaults to `"nli"` (deberta-small, 140M). New `_resolve_faithfulness_backend()` returns `"llm"` on CUDA (70B judge as NLI) and `"nli"` on CPU/MPS.
- **`paraphrase_generator.py`**: Added `RuntimeError` in `__init__` if `model_backend == "local"` and `torch.cuda.is_available()` — flan-t5-large (770M) is forbidden on GPU machines.

**Why:**
- All five components had small-model defaults that would silently activate on GPU if the corresponding env var was not set in `.env`. This means a user who forgets `JUDGE_MODEL`, `EMBEDDER_MODEL`, or `FAITHFULNESS_BACKEND` in their Jarvis `.env` would unknowingly run flan-t5/MiniLM for scoring, producing scores incomparable to the GPU baseline.
- Auto-detection by `torch.cuda.is_available()` is the safest approach: it infers the correct model tier from hardware without requiring any extra config on the GPU machine.

**Impact:**
- On CUDA machines with no `.env`: pipeline now auto-loads the correct ≥7B models without any manual configuration.
- On CUDA machines with a misconfigured `.env` (small models): startup raises `RuntimeError` immediately, preventing a silent bad run.
- On laptop (no CUDA): behavior is identical to before — all small models still load as defaults.
- No formula or metric changes. Re-run required if you previously ran with small-model defaults on GPU (scores will differ due to embedder/judge quality change).

---

## [2026-05-29] — Tighten GenSens paraphrase gates: per-task strategy whitelist, bidirectional NLI gate, K=8

**Files Modified:** `gensens/scripts/paraphrase_generator.py`, `prompt_robustness/config.yaml`

**What Changed:**
- **Per-task strategy whitelist (improvement #3).** Added `STRATEGY_BY_TASK: Dict[str, List[str]]` in `paraphrase_generator.py`. Previously all 16 strategies were applied to every task. Now:
  - `summarization`: 15 strategies (drops `question_form` — base_text is an instruction, not a question, so converting to a question changes intent).
  - `creative`: 12 strategies (drops `question_form`, `imperative`, `role_prefix` — they reshape news highlights into commentary or commands, distorting facts).
  - `dialogue`, `qa`: 9 strategies (drops `question_form` (no-op on questions), `imperative` (turns question into command, changes expected answer form), `role_prefix` (leaks meta-text), `passive_voice` (can swap subject/object of the question), `concise`, `split_sentences`, `merged_sentences` (over-compress/fragment short questions, corrupting intent)).
  `generate_paraphrases` now iterates only over the task-appropriate `(strategy_name, instruction)` pairs; unknown tasks fall back to the full 16-strategy set.
- **Bidirectional NLI semantic gate (improvement #1).** Added a global lazy-loaded `cross-encoder/nli-deberta-v3-small` model (`_get_nli_model`) and a per-call helper `_nli_entail_prob`. For tasks in `_NLI_REQUIRED_TASKS = {"qa", "dialogue"}`, every candidate that passes the SBERT band and token-overlap floor must additionally satisfy `P(entail)(base→cand) ≥ τ` **and** `P(entail)(cand→base) ≥ τ`, with `τ = nli_entail_threshold = 0.50` (new `__init__` parameter). Candidates failing either direction are rejected as intent-changing paraphrases (e.g., passive↔active question flips that swap the answer). For `summarization`/`creative` the NLI model is not consulted (cost-saving; the SBERT band + overlap floor already work well for non-question text). If the NLI model fails to load, the gate silently degrades to SBERT-only filtering with a warning. Accepted variants now carry `nli_entail_fwd` and `nli_entail_bwd` fields for downstream auditing.
- **K standardized to 8.** `prompt_robustness/config.yaml` `paraphrase.K` raised from `5` → `8`, aligning the evaluator with the GenSens `--n_variants` CLI default and giving SMS / TRD / AUC-E enough degrees of freedom for stable variance estimates (df=7 instead of df=4).

**Why:**
- **NLI gate.** SBERT cosine ≥ 0.82 with mpnet-base is a weak semantic gate for QA/dialogue. A surface-level rewrite like "What did X tell Y?" → "What was Y told?" can score cos ≥ 0.85 while reversing the answer the model should produce. Such paraphrases inflate the SMS/Faithfulness variance — the model is being marked "inconsistent" when the paraphrase itself changed intent. Bidirectional entailment of base ⟷ candidate is the standard test for semantic equivalence and matches the NLI model already used for `faithfulness_metric.py`.
- **Per-task whitelist.** Several of the 16 strategies are nonsensical or harmful for specific tasks (e.g., applying `question_form` to a base_text that's already a question is a near-no-op; applying `imperative` to a QA question turns "What is X?" into "Tell me X" which changes the expected answer form). Removing these per task improves the *quality* of the surviving variants without harming diversity, because the dropped strategies were producing near-duplicates anyway.
- **K=8.** Three different defaults coexisted (GenSens CLI default 8, eval config 5, on-disk smoke files 4). K=5 is the bare minimum for variance; K=8 is the practical sweet spot — bigger K hits diminishing returns vs. compute, especially with a 70B judge. Aligning all three on K=8 removes a long-standing inconsistency.

**Impact:**
- **Breaking for QA/dialogue paraphrase regeneration.** Old QA/dialogue datasets generated before this change may contain intent-changing paraphrases; regenerate to benefit from the NLI gate. Existing summarization/creative datasets are unaffected (no NLI applied there).
- **Reject rate up ~10–20% for QA/dialogue.** The 3-retry loop in `generate_paraphrases` usually absorbs the extra rejections; monitor logs and raise `max_retries` if instance yield drops below `n_variants`.
- **Variant yield slightly lower on dialogue/qa** because the whitelist trims the strategy pool from 16 to 9 — `max_per_strategy` may need to be raised from 2 → 3 if you target n_variants ≥ 9 on those tasks.
- **Per-variant audit fields.** New `nli_entail_fwd` / `nli_entail_bwd` in QA/dialogue variants; the JSONL→CSV exporter ignores unknown keys so existing readers still load.
- **Re-run required.** PRI / SMS / Faithfulness results on QA/dialogue should be re-run after regenerating the dataset; summarization/creative results are unchanged. K=5 → 8 also implies re-running the evaluator (more variants per sample = more model calls).
- **No formula changes** — PRI / ORI / IFI definitions are untouched. Only the paraphrase generation gate and variant count change.

---

## [2026-05-27] — Add 70B AWQ as a subject model; two-step generate→score pipeline

**Files Modified:** `prompt_robustness/src/config.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/src/model_interface.py`, `prompt_robustness/config.yaml`, `run_on_gpu.sh`

**What Changed:**
- **70B as subject**: `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` added to the subjects list alongside the three 8B/7B models. Its responses (summaries, QA answers, etc.) are now generated, stored, and scored — not only used as a judge/faithfulness scorer. This gives a four-model comparison: LLaMA-3.1-8B, Qwen2.5-7B, Mistral-7B, and LLaMA-3.1-70B-AWQ.
- **`subject_quantizations` config**: New `Dict[str, str]` field in `Config` (default: `{"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": "awq_marlin"}`). `benchmark.py` looks up the quantization for each model via `config.subject_quantizations.get(model_name)` and passes it to `ModelInterface(model_name, config, quantization=quant)`. Models absent from the dict load in bf16/fp16.
- **`_input_device` property**: Added to `ModelInterface` — returns `cuda:0` on CUDA (correct first-layer device for `device_map="auto"` multi-GPU or AWQ sharding), otherwise `self.device`. All internal tensor `.to()` calls updated from `self.device` to `self._input_device`.
- **Chat-template formatting**: `generate_responses()` now applies `tokenizer.apply_chat_template()` for any instruct model that ships a chat template — ensuring the 70B (and all instruct-tuned models) receive properly formatted `<|user|>` / `<|assistant|>` prompts instead of raw text.
- **Memory-safe two-step Phase 2**: `run_on_gpu.sh` Phase 2 now runs `--generate-only` (subject model, no judge) then `--responses-csv` (judge + NLI, no subject) so the 70B AWQ subject never co-resides on GPU with the 70B judge.

**Why:**
- Including the 70B as a subject allows a direct comparison of prompt sensitivity across model sizes (7–8B vs 70B) under identical prompt perturbations. The 70B AWQ quantization is the only way to fit this model on a single 80 GB card.
- The two-step generate→score split was already available via CLI flags; making it the default for Phase 2 prevents OOM when both the subject 70B and judge 70B would otherwise be loaded simultaneously.

**Impact:**
- `responses.csv` and `scored_samples.csv` will now contain rows for a fourth model (`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`).
- The `benchmark.csv` ranking table gains a fourth row. Previously generated results with only 3 models are still valid for those models; re-run Phase 2 to add 70B rows.
- No formula or metric changes — the four-model comparison uses identical PRI/ORI/IFI computations.

---

## [2026-05-27] — Qwen 8B simple-to-complex summarization instruction generator

**Files Modified:** `gensens/scripts/generate_summ_instructions.py` (new), `gensens/scripts/generate_dataset.py`

**What Changed:**
- Added standalone script `generate_summ_instructions.py` that uses **Qwen/Qwen2.5-7B-Instruct** (or any HF instruct model via `--model-id`) to generate 100 unique summarization instructions across 4 complexity levels ordered simple → complex:
  - Level 1 **simple** (≤12 words): plain, direct phrasing — 25 instructions
  - Level 2 **moderate** (12–20 words): standard with basic specificity — 25 instructions
  - Level 3 **detailed** (20–35 words): professional with explicit content requirements — 25 instructions
  - Level 4 **complex** (35–60 words): elaborate multi-clause framing — 25 instructions
- Each level uses a separate targeted system prompt; the model is asked to output one instruction per line (no bullets/numbering). Per-level retry loop (default 4 retries) compensates for SBERT filter loss.
- All candidates are SBERT-filtered against the canonical instruction (band 0.82–0.98, token Jaccard overlap ≤ 0.85, min 6 words). Deduplication is cross-level via a shared `seen_sigs` set.
- Output saved to `gensens/data/summ_instruction_pool.jsonl` — one record per line: `{instruction, complexity_level, complexity_label, sbert_similarity}`. Instructions sorted by `complexity_level` ascending so assignments go simple→complex across dataset instances.
- `generate_dataset.py` now checks for `summ_instruction_pool.jsonl` first (`_load_pool_file`). If found, loads instructions from it (Qwen-generated, level-sorted). If absent, falls back to on-the-fly LLM/strategy generation as before.
- Supports both `--backend hf` (HF Transformers, bfloat16, device_map=auto) and `--backend vllm` (H100 batched inference).

**Why:**
- A complexity gradient in the instruction pool tests whether model outputs degrade or change character as prompt complexity increases — a more systematic sensitivity probe than uniformly-diverse instructions.
- Using a dedicated open-source 8B model (Qwen) for pool generation separates the instruction-design step from the paraphrase-generation step, enabling the pool to be generated once and reused across multiple dataset runs.

**Impact:**
- Run `python scripts/generate_summ_instructions.py --backend vllm` on H100 once before `generate_dataset.py` to populate the pool file.
- If pool file is absent, behaviour is unchanged from the previous session (on-the-fly freeform/strategy generation).
- No changes to evaluation pipeline, metrics, or PIRC.

---

## [2026-05-27] — Summarization unique-instruction pool + Creative highlights paraphrasing

**Files Modified:** `gensens/scripts/data_loaders.py`, `gensens/scripts/paraphrase_generator.py`, `gensens/scripts/generate_dataset.py`

**What Changed:**
- **Summarization** — instead of all instances sharing the same canonical instruction, a pool of up to 100 unique instructions is pre-generated at the start of the task run using an open-source LLM. Each instance is assigned one unique instruction from this pool (cycling if `n_instances > 100`). The canonical instruction ("Summarize the following news article in 3-4 sentences...") is always index 0.
  - For `llama`/`vllm` backends (H100): uses `ParaphraseGenerator.generate_instruction_pool → _instruction_pool_freeform`, which sends a **single freeform prompt** to the LLM asking it to generate N diverse instructions in one call, then SBERT-filters the parsed lines (`sim ∈ [0.82, 0.98]`, token overlap ≤ 0.85). Falls back to strategy-based generation if the freeform yield is insufficient.
  - For `local` backend (Flan-T5, CPU/MPS): uses `_instruction_pool_strategies` which applies all 16 strategy prompts (same as normal paraphrase generation) with a lifted per-strategy cap to reach the target pool size.
- **Creative** — `base_text` changed from the fixed story-writing instruction to `item["highlights"]` (the summary premise). The instruction is now a hardcoded prefix in `base_prompt` and is no longer paraphrased. The paraphrase generator now receives the highlights as the thing to rephrase. `SYSTEM_PROMPTS["creative"]` updated to "rephrase the news summary/highlights while keeping all facts intact". The Flan task-context label for creative changed from "a story-writing instruction" to "a news summary/highlights". `metadata["instruction"]` added to creative records to preserve the fixed instruction string for downstream readers.

**Why:**
- **Summarization**: using a single fixed instruction for all instances creates an artificial uniformity — every instance's `base_text` is identical, making cross-instance diversity purely article-driven. Unique instructions per instance make the benchmark more realistic (real prompts vary at instruction level) and increase the population of instruction phrasings seen across the dataset.
- **Creative**: paraphrasing the instruction for story-writing is less meaningful than paraphrasing the premise (highlights), because the highlights carry the actual content the model must expand. Different phrasings of the same highlights are the more interesting perturbation for creative generation — they directly test whether lexically varied premises produce different stories.

**Impact:**
- Existing summarization datasets must be regenerated: `base_text` now varies per instance instead of being constant.
- Existing creative datasets must be regenerated: `base_text` is now the highlights, not the instruction; `full_prompt` structure changes accordingly.
- The `metadata["instruction"]` key is new for creative records; downstream readers that use only canonical bridge keys (`input_text`, `reference_output`) are unaffected.
- Evaluation pipeline (`evaluator.py`, `benchmark.py`, metric modules) is **unchanged** — it reads `base_text`, `variants`, and `metadata` fields, all of which still conform to the same schema.

---

## [2026-05-27] — Swap GenSens source datasets to a new four-task set (DATA-LAYER ONLY)

**Files Modified:** `gensens/scripts/data_loaders.py`, `gensens/scripts/generate_dataset.py`, `gensens/scripts/paraphrase_generator.py`, `gensens/scripts/validate_dataset.py`, `prompt_robustness/src/data_loader.py`, `prompt_robustness/src/csv_io.py`, `prompt_robustness/src/prompt_generator.py`, `DATA_PREPARATION.md`, `DATASET_GENERATION.md`

**What Changed:**
- Replaced the four GenSens source datasets. The new task set is:
  - **summarization** — `cnn_dailymail` 3.0.0 (*sum_in_brief* framing): paraphrase the summarize instruction; fixed = article; `reference_output` = gold highlights; `input_text` = article.
  - **creative** — `cnn_dailymail` 3.0.0 (*generate_story* framing, the inverse of summarization): paraphrase the story-writing instruction; fixed = the summary premise (highlights); `reference_output` = the full article (the gold "story"); `input_text` = highlights.
  - **dialogue** — replaced **MultiWOZ 2.2** with **DREAM** (HF id `dream`): paraphrase the question; fixed = dialogue turns + answer options; `reference_output` = the correct answer choice; `input_text` = the dialogue turns.
  - **qa** — **new task** (HF id `sentence-transformers/eli5`, config `pair`): paraphrase the question; `reference_output` = the answer/explanation; `input_text` = the question.
- **Removed the `code` task** (HumanEval + MBPP) entirely, including its loader and docstring extractor.
- **All four tasks now carry a non-empty `reference_output`** (the old creative/dialogue tasks had `reference_output=None`). Source rows lacking a reference are skipped at load time (hard requirement).
- Introduced canonical metadata bridge keys **`input_text`** (grounding) and **`reference_output`** (gold), emitted by every loader. The three data-layer readers — `generate_dataset.save_final` CSV writer, `csv_io.gensens_jsonl_to_csv`, and `data_loader.load_gensens_dataset` — now read canonical-first with a legacy `article`/`gold_summary` fallback (so older summarization datasets still load). Summarization additionally keeps `article`/`gold_summary` aliases.
- Verified real HF paths during inspection: `sum_in_brief`/`generate_story` are **PromptSource template names, not HF configs** (`cnn_dailymail` only exposes `1.0.0`/`2.0.0`/`3.0.0`); DREAM loads from `dream` (`trust_remote_code=True`); ELI5 loads from `sentence-transformers/eli5` config `pair` (columns `question`/`answer`, train split only).
- Updated `SYSTEM_PROMPTS` and the Flan task-context map to the new four tasks (reword only the instruction/question, preserve exact meaning, return only the rewrite).
- Added a **7th validator check** (`check_reference_present`) that flags any instance with an empty `reference_output`; updated the validator / generator / CLI task lists to `summarization, creative, dialogue, qa`.
- Added matching **d1/d2/d3 synthetic fallback templates for `qa` and `dialogue`** in `prompt_robustness/src/prompt_generator.py` so the fallback variant-generation approach still functions for every task.

**Why:**
- The old creative (WritingPrompts) and dialogue (MultiWOZ) tasks had no reference output, so reference-grounded signals (CS, ROUGE, AUC-E quality, the reference path of faithfulness) had nothing to score against on half the benchmark. The new four-task set closes that gap — every task now has a gold reference — while still spanning four distinct task domains.

**Impact:**
- **DATA-LAYER ONLY — the evaluation method is unchanged.** No edits to PRI / ORI / IFI / Diagnostic_PRI formulas, weights, thresholds, the faithfulness backends, the LLM-judge, the PIRC/Logit-Lens algorithms, or any statistical test. `scores.py`, `evaluator.py`, `benchmark.py`, `evaluate.py`, the `*_metric.py` modules, and the PIRC stack were not touched.
- **Breaking for dataset/result comparability:** previously generated GenSens datasets, `responses.csv`, `scored_samples.csv`, and all Phase 2–4 results must be **regenerated**. The `code` task is gone; a new `qa` task appears in every per-task output and table.
- Creative is now long-form generation (premise → article) with a real reference, not open-ended story writing. Creative and summarization draw from the **same** stratified CNN/DM test selection (seed 42), so they cover the same article/highlights pairs inverted (summarize vs. expand) — input ≠ reference for both.
- **Ambiguity resolved:** the task spec's creative row listed both "input_text: article / fixed: article" and "reference_output: the story/continuation field". CNN/DM has no separate story field, and input == reference would be circular. Resolved to the PromptSource *generate_story* inversion (grounding = highlights premise, reference = article) to guarantee a non-empty, non-circular reference. ELI5's `pair` split contains some low-quality rows, so the qa loader applies light deterministic quality filters (English question of 5–60 words, answer ≥ 10 words); this only affects which source rows are *selected*, not how anything is scored.

---

## [2026-05-26] — Dual-Pillar Diagnostic PRI + Publishable PIRC Variance

**Files Modified:** `prompt_robustness/src/scores.py`, `prompt_robustness/src/evaluator.py`, `prompt_robustness/src/csv_io.py`, `prompt_robustness/src/benchmark.py`, `prompt_robustness/src/analysis.py`, `prompt_robustness/reaggregate.py`, `prompt_robustness/src/pirc.py`, `prompt_robustness/experiment_pirc.py`, `prompt_robustness/evaluate.py`, `prompt_robustness/config.yaml`, `prompt_robustness/tests/test_scores.py`, `AGENTS.md`

**What Changed:**
- Added a separate **Diagnostic_PRI** for publication framing while keeping the existing fair-ranking `PRI = 0.40·SMS + 0.35·CS + 0.25·Faithfulness` unchanged.
- Added harmonic dual-pillar diagnostics: `Diagnostic_ORI = HM(SMS, AUC-E, 1−TRD, KPIG)`, `Diagnostic_IFI = HM(1−PPL_var, 1−BF[, 1−PC_stab])`, and `Diagnostic_PRI = HM(Diagnostic_ORI, Diagnostic_IFI)`.
- Added a 2×2 diagnosis matrix: Robust, Externally Stable / Internally Fragile, Internally Stable / Output-Sensitive, and Fragile.
- Wired diagnostic score outputs through live evaluation, sample CSVs, correlation summaries, benchmark aggregation, and no-GPU reaggregation.
- Added CSV schema migration before resume appends so older `scored_samples.csv` files are not corrupted when new diagnostic columns are introduced.
- Changed LL-PIRC evaluation to generate one PIRC-stabilized output per paraphrase variant and compute real post-PIRC ROUGE-L variance across variants.
- Removed test-time per-article tau/alpha selection by ROUGE; PIRC now uses fixed preregistered alpha/tau by default, with optional global dev-prefix tuning via `pirc_selection.dev_tune_articles`.
- Added bootstrap confidence intervals and paired effect sizes to Phase 4 evaluation.

**Why:**
- The paper needs novelty in both evaluation and mitigation without corrupting the fair primary ranking metric. A separate harmonic diagnostic score restores the dual-pillar novelty while preserving the de-collinearized PRI.
- Previous PIRC variance was set to zero because only one PIRC output was produced per article, making variance reduction trivial and not publishable.
- Per-article tau/alpha selection on test ROUGE was oracle-like and risked overfitting the mitigation result.

**Impact:**
- Existing `PRI` and `Final_Score` rankings remain comparable.
- New `Diagnostic_PRI`, `Diagnostic_ORI`, `Diagnostic_IFI`, and `Diagnosis` columns are added to outputs and should be reported as diagnostic/paper-novelty metrics.
- PIRC results generated by older code are no longer sufficient for Phase 4 variance claims; rerun `experiment_pirc.py` so `pirc.json` contains `pirc_rouge_var` and per-variant `pirc_rouge_scores`.
- Numerical outputs for diagnostic scores and PIRC variance are new; results need to be rerun for publication tables.

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
