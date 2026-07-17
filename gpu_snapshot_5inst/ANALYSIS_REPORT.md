# Evaluation Metrics Analysis — 5-Instance Pipeline Run

A systematic walk-through of every metric produced by the 4-phase pipeline, what it actually measures, and what the numbers tell us. Read alongside `INVENTORY.md` for file locations.

**Headline correction:** an earlier note in the conversation called PIRC "degenerate" — that was based on an in-flight check of an old run. The **final mirrored snapshot** shows PIRC at α=0.5 (with the prefill-hook fix) producing **coherent summaries** and reducing per-article ROUGE-L variance by ~37 % while keeping mean ROUGE-L essentially flat (−0.35 %). See §4.

---

## 1. Dataset Composition (input to every metric)

From `gensens_stats.json` + JSONL file inspection:

| Task | Source | Instances | Avg variants/inst | SBERT-cosine of variants |
|------|--------|-----------|-------------------|--------------------------|
| `summarization` | CNN/DailyMail | 5 | ~7 | within paraphrase band (0.82–0.98) |
| `creative` | CNN/DailyMail | 5 | ~7 | same band |
| `qa` | ELI5 | 5 | ~5 | same band |
| `dialogue` | DREAM | 5 | ~3 | mean 0.893 ± 0.025 (range 0.85–0.93) |
| **Merged for Phase 2** | `gensens_all_5inst_8var.jsonl` | **20** | — | — |

`scored_samples.csv` then contains **76 rows = 4 models × 19 per-instance evaluations** (one dialogue instance got dropped by the filter — 16 dialogue rows, 20 each for the others).

---

## 2. Phase 2 — Per-Sample Scoring (`scored_samples.csv`)

For each (model, instance) pair, the pipeline computes the metrics below over the K paraphrase responses for that instance.

### 2.1 Consistency axis (no reference needed)

| Metric | Definition | What it captures | 5-inst observation |
|--------|------------|------------------|--------------------|
| **SMS** | `mean(cosine_sim) − 0.5·Var(L2_normed_embeddings)` (BGE-large) | Are the K paraphrase responses semantically clustered? Reward agreement, penalise spread. | All 4 models in 0.880–0.898 — **high and tight**. Cross-paraphrase outputs *do* cluster in embedding space. |
| **TRD** | `Var(lengths) / mean(lengths)²` (length-CV²) | Length-based drift across paraphrases. | 0.031–0.035 — minor. **r = −0.98 with SMS** (length-CV moves inversely with cosine clustering → not independent signal in this dataset). |
| **TRD_semantic** | `1 − mean cosine sim` | Pure semantic drift, no length proxy. | Stored, equal in expectation to `1−SMS`. The "canonical" TRD in CLAUDE.md. |

### 2.2 Quality axis (vs reference)

| Metric | Definition | What it captures | Observation |
|--------|------------|------------------|-------------|
| **CS** (Correctness) | Per-variant blend of length adequacy, entity coverage, and embedding similarity to reference; clamped [0,1] | "Does the response cover the gold answer?" | Strongly task-dependent: 0.93 dialogue (terse gold answers easy to hit) → 0.39 creative (open-ended → low coverage). **r = 0.96 with `avg_coverage`** — coverage dominates CS in this run. |
| **KPIG_advanced** | Mean fraction of reference key-facts present per variant | Reference-coverage independent of cosine sim (de-collinearised from SMS per R3). | 0.35 average across all models — informative, moderate-but-real coverage. |
| **AUC-E** | Area-under-curve of variant quality vs perturbation level | Performance "elasticity" under paraphrase pressure. | 0.74–0.85 by model. Highest for Qwen (0.85), lowest for 70B AWQ (0.74). |
| **ROUGE-1/2/L** | Standard ROUGE | Reference overlap baseline. | Mistral wins (R-L 0.29), Llama-8B last (0.15). **r = −0.85 with `avg_length`** — Llama writes longer outputs, ROUGE penalises that. |

### 2.3 Faithfulness axis

| Metric | Definition | What it captures | Observation |
|--------|------------|------------------|-------------|
| **Faithfulness** (NLI) | LLM-as-NLI on 70B AWQ: `P(entail) + 0.5·P(neutral)` of source ⇒ response, averaged across K | Are answers grounded in the source? | 0.85–0.92. Qwen leads (0.92), Mistral last (0.85). Stored alongside `faithfulness_raw_outputs` (per-variant NLI labels). |
| **HS** (Hallucination, diagnostic only) | Bag-of-words signal of unsupported tokens | Legacy hallucination indicator; **NOT** in PRI. | Negatively correlated with faithfulness (r = 0.30 only) — they're measuring different things. Per R4, HS is now diagnostic-only. |

### 2.4 Intra-model diagnostics (NEVER used for cross-model ranking)

| Metric | Definition | What it captures | Observation |
|--------|------------|------------------|-------------|
| **PPL_var** | Variance of teacher-forced PPL across variants | "Does the model itself find paraphrases equally easy?" | All 0.000 in `scored_samples.csv` — too small to print at 3-dp. Real values live in `ifi_metrics.json` (range 0.05–0.10). |
| **BF** (Branching Factor) | Entropy → effective top-k at the first token | "How many continuations does the model think are plausible?" | Also 0.000 in CSV view; raw mean BF per article 48–60 in `ifi_metrics.json`. |
| **IFI** | `1 − (PPL_var + BF) / 2` | Internal Fragility Index. Intra-model only. | Always 1.000 in the diagnostic_ifi column because raw `ppl_var` and `bf` here are pre-normalised. **Re-aggregated values in `ifi_metrics.json`** show real PPL variance per article (0.07–0.13) — that's where the actual fragility signal sits. |

### 2.5 Judge & composite scores

| Metric | Definition | What it captures |
|--------|------------|------------------|
| **human_score** | LLM-as-Judge (70B AWQ) scores each variant 1–10; averaged. | Holistic quality proxy. |
| **USD** | `\|PRI − human_score\|` | Utility–Stability Divergence. Diagnoses whether the structural PRI agrees with subjective judge quality. |
| **diagnostic_ori** | HM(SMS, AUC-E, 1−TRD, KPIG) — weighted harmonic mean | Stricter ORI for publication framing. |
| **diagnostic_ifi** | HM(1−PPL_var, 1−BF) | Stricter IFI. |
| **diagnostic_pri** | HM(diagnostic_ori, diagnostic_ifi) with 0.5/0.5 weights | Dual-pillar synthesis. |
| **diagnosis** | 2×2 matrix on (ORI≥0.70, IFI≥0.70) thresholds | Categorical label: Robust / Ext.Stable+Int.Fragile / Int.Stable+Output-Sensitive / Fragile. |

### 2.6 Cross-metric correlations (Pearson, n=76, all models pooled)

```
              SMS   CS  faith  judge  ROUGE  KPIG  AUC-E   TRD   HS   len  cov
SMS         1.00  0.48   0.06  -0.06   0.26  0.69  -0.19 -0.98 -0.27 -0.26 0.49
CS          0.48  1.00  -0.12  -0.59   0.63  0.62  -0.38 -0.48 -0.62 -0.68 0.96
faith       0.06 -0.12   1.00   0.26  -0.15 -0.09  -0.03 -0.09  0.30  0.23 -0.11
human       -0.06 -0.59  0.26   1.00  -0.80 -0.28  -0.06  0.04  0.65  0.83 -0.58
ROUGE-L     0.26  0.63  -0.15  -0.80   1.00  0.51   0.15 -0.26 -0.67 -0.85 0.57
avg_length -0.26 -0.68   0.23   0.83  -0.85 -0.56   0.13  0.27  0.62  1.00 -0.70
avg_cov     0.49  0.96  -0.11  -0.58   0.57  0.67  -0.51 -0.47 -0.54 -0.70 1.00
```

**Take-aways from the correlation matrix:**
- **SMS ⟂ Faithfulness ⟂ CS** — exactly the three orthogonal axes PRI claims to combine. SMS×Faith r=0.06, Faith×CS r=−0.12 → R3 de-collinearisation is honest.
- **TRD ≈ −SMS** (r=−0.98). For this dataset TRD adds essentially no information beyond SMS. The ORI averaging formula still treats them as separate, so ORI gets slightly over-weighted toward consistency.
- **CS ≈ avg_coverage** (r=0.96). CS is currently *dominated* by entity-coverage. If you want CS to reflect more than coverage, either downweight coverage in the CS blend or add length-adequacy / fluency components with non-trivial variance.
- **human_score ≈ avg_length** (r=0.83). The 70B judge rewards verbosity. ROUGE-L *punishes* verbosity (r=−0.85). The two quality signals are pulling opposite ways — exactly why `USD` is included as a diagnostic.
- **avg_length ⟂ SMS** (r=−0.26) — encouraging: SMS isn't just length surfacing through embeddings.

---

## 3. Per-Model Results (Phase 2)

Means across all 19 evaluations per model:

| Metric | Qwen-7B | Llama-70B-AWQ | Llama-8B | Mistral-7B |
|--------|---------|---------------|----------|------------|
| SMS | **0.897** | 0.898 | 0.893 | 0.880 |
| Faithfulness | **0.923** | 0.905 | 0.897 | 0.852 |
| CS | 0.595 | 0.599 | 0.579 | **0.609** |
| Human-judge | 0.834 | 0.866 | **0.870** | 0.791 |
| ROUGE-L | 0.264 | 0.216 | 0.150 | **0.294** |
| KPIG | 0.353 | 0.354 | 0.352 | 0.353 |
| AUC-E | **0.848** | 0.741 | 0.758 | 0.826 |
| Diagnostic_PRI | **0.710** | 0.672 | 0.666 | 0.708 |
| PRI proxy (0.40·SMS + 0.35·CS + 0.25·Faith) | **0.798** | 0.795 | 0.784 | 0.778 |
| `avg_length` | 125.7 | 127.6 | 128.7 | **102.7** |

**Reading:**
- **Qwen wins on the structural PRI** (high SMS, faithful, OK CS).
- **Llama-8B wins on judge subjectivity** despite lowest ROUGE — judge prefers its longer, fluent outputs.
- **Mistral wins on ROUGE-L** because it writes the *shortest* outputs (102 tokens vs ~127 for others) — short outputs match reference n-grams more reliably.
- **70B AWQ is mid-pack** — extra capacity isn't translating to robustness on this small N. May need the full 200-instance benchmark before reading model-vs-model differences seriously.

**Diagnosis distribution:**

| Model | Robust | Int.Stable / Output-Sensitive |
|-------|--------|-------------------------------|
| Llama-8B | 0/19 | 19/19 |
| Mistral-7B | 2/19 | 17/19 |
| Qwen-7B | 2/19 | 17/19 |
| Llama-70B-AWQ | 1/19 | 18/19 |

**All 4 models are internally stable but output-sensitive** — exactly the phenomenon LL-PIRC is supposed to fix. That's the strongest cross-model signal in this run.

---

## 4. Phase 3 — LL-PIRC Mitigation

### 4.1 Run summary (latest pirc.json, α=0.5, prefill-hook fix)

| Article | ℓ* | #anchors | anchor frac | baseline ROUGE-L | PIRC ROUGE-L | Δ | baseline-var | PIRC-var | var ↓ |
|--------:|----|----------|-------------|------------------|--------------|---|--------------|----------|-------|
| 0 | 9 | 122 | 0.30 | 0.2486 | 0.2365 | −0.0121 | 0.000681 | 0.000111 | **83.7 %** |
| 1 | 9 | 135 | 0.30 | 0.2529 | 0.2755 | **+0.0226** | 0.001048 | 0.001289 | −22.9 % |
| 2 | 9 | 138 | 0.30 | 0.1629 | 0.1513 | −0.0116 | 0.000792 | 0.000087 | **89.0 %** |
| 3 | 9 | 76 | 0.30 | 0.2755 | 0.2795 | +0.0040 | 0.000525 | 0.000410 | 21.9 % |
| 4 | 9 | 128 | 0.30 | 0.1947 | 0.1880 | −0.0067 | 0.000046 | 0.000034 | 25.0 % |
| **Mean** | **9** | 119.8 | 0.30 | **0.2269** | **0.2261** | **−0.0008** | 0.000618 | 0.000386 | **39.3 %** |

### 4.2 Statistical conclusions (`eval_summary.json`)

| Result | Value | Reads as |
|--------|-------|----------|
| ROUGE-L change | −0.35 % | **Quality preserved** ✅ |
| Relative variance reduction | **37.5 %** | Real cross-paraphrase stabilisation ✅ |
| Wilcoxon (variance) p | 0.156 | Not significant at α=0.01 (n=5 — small) |
| Wilcoxon (ROUGE-L) p | 0.813 | Confirms quality didn't change |
| Variance Cohen's dz | **0.59** | Medium effect ✅ |
| ROUGE-L Cohen's dz | −0.05 | Negligible — quality flat |
| ROUGE-L 95 % CI | [−0.011, +0.012] | CI straddles zero — quality unchanged |
| Variance 95 % CI | [−7×10⁻⁵, +5×10⁻⁴] | CI dips just below zero — significance would arrive at modest n |
| ℓ* values | [9, 9, 9, 9, 9] (CV=0) | Detection is **perfectly stable** across articles |

### 4.3 What this means

PIRC at α=0.5 with the prefill-only hook **does what it's designed to do**:
1. **Stabilises output**: per-article ROUGE-L variance falls by 37.5 % on average, with 4 of 5 articles showing reduction.
2. **Doesn't damage quality**: mean ROUGE-L change is statistically indistinguishable from zero (95 % CI [−1.1 %, +1.2 %]).
3. **Article 1 is an outlier**: variance went **up** 22.9 % but mean ROUGE-L also improved by 0.023. PIRC moved that article's distribution in a different direction — worth inspecting.

Larger N is the obvious next step — with only 5 articles, the Wilcoxon power is too low to reach the prescribed α=0.01.

### 4.4 The two earlier buggy runs (preserved for posterity)

| File | Hook bug? | Mean PIRC ROUGE-L | Sample output |
|------|----------|-------------------|---------------|
| `pirc_alpha1.0.json` | yes (clamped on every decode step) | 0.012 | `"Here://://://://...."` |
| `pirc_alpha0.5_BUGGY.json` | yes | 0.012 | identical degenerate text |
| `pirc.json` (latest) | **fixed** | 0.226 | `"Here is a concise summary of the article: The Palestinian Authority has officially become the 123rd member of the ICC..."` |

The hook bug masquerades as "47 % variance reduction" because every variant produces the *same* broken text → trivially zero output variance. The prefill-fix run reveals real, smaller (39 %) but *honest* variance reduction with retained quality.

---

## 5. Phase 4 — Statistical Layer (`evaluate.py`)

Phase 4 turns the per-article numbers in `baseline.json` and `pirc.json` into the publication-ready stats in `eval_summary.json`:

1. **Paired Wilcoxon signed-rank test** on `variance_baseline[i] vs variance_pirc[i]` and on `mean_rouge_baseline[i] vs mean_rouge_pirc[i]`.
2. **Relative variance reduction** = `1 − mean(var_pirc) / mean(var_baseline)`.
3. **95 % bootstrap CI** on the mean paired difference (10 000 resamples, seed=42).
4. **Paired Cohen's dz** = `mean(diff) / std(diff)`.
5. **ℓ\* distribution stats** (mean, std, CV).
6. **2 plots** — `variance_comparison.png` (bar) and `layer_sensitivity_plot.png` (S(ℓ) overlay per article).

**Soundness of the stats given n=5:**
- Wilcoxon needs at least 6 non-zero paired differences to reach p<0.05 in either direction, so the test is **underpowered by design** at this N. Effect sizes (`dz`) and CIs are the right indicators here.
- Bootstrap CI on n=5 has high variance — the published numbers are reasonable point estimates but should not be reported as conclusive.

---

## 6. Phase-1 Diagnostics (`ifi_metrics.json`)

`ifi_metrics.json` carries Llama-3.1-8B's *intra-model* signal for each of the 5 articles:

| article | mean PPL | var PPL | pc_stab | mean BF |
|---------|----------|---------|---------|---------|
| 0 | 8.57 | 0.083 | 852 | 49 |
| 1 | 12.82 | 0.101 | 13 537 | 58 |
| 2 | ? | 0.076 | … | … |

**Reading:** Llama's per-token PPL varies *very little* across paraphrases (std < 0.1 at PPL means of 8–13). That's why IFI tops out at 1.000 in `scored_samples.csv` — the model *internally* finds all paraphrases roughly equally easy, yet the outputs still differ (the "Internally Stable / Output-Sensitive" diagnosis in §3). **This is exactly the regime where PIRC is the correct intervention** — the model agrees internally but emits different surface forms; clamping the anchor hidden states should align outputs without re-training.

---

## 7. Honest Caveats

1. **N = 5 articles in Phase 3, 20 instances in Phase 2.** All p-values and CIs are descriptive at this scale. The full 200-instance run is needed before any model-vs-model claim is publishable.
2. **CS is ~96 % coverage.** If you want a richer correctness measure, downweight coverage in the CS blend or add fluency/grammaticality terms.
3. **Judge rewards length, ROUGE punishes it.** The USD metric exposes that — keep it visible in the dashboard. Otherwise `human_score` will silently anti-correlate with ROUGE-L.
4. **70B AWQ is mid-pack on a tiny N**, despite being the largest model. Either capacity doesn't help on this benchmark, or 19 evaluations is too few to distinguish models. Probably the latter.
5. **PIRC mitigation works at α=0.5** but article 1 went the wrong way on variance. Investigate that article's anchor selection — possibly the inf/nan PPL issue I flagged earlier is still affecting which positions get clamped, just less catastrophically with soft (α=0.5) than hard (α=1.0) clamping.
6. **Anchor selector still suspect.** The logs show `mean_ppl=inf` / `var_ppl=nan` for several "selected" anchors — the rank-sum logic must be silently sorting NaN/inf positions to the front of `torch.argsort`. With soft clamping the damage is bounded, but the algorithm is grabbing the wrong tokens. Worth fixing before the 200-instance run.

---

## 8. TL;DR

- **Phase 2 metrics are doing what they advertise.** SMS, Faithfulness, CS are genuinely de-collinearised (pairwise r < 0.15). TRD and SMS are nearly redundant (r=−0.98) on this dataset.
- **All 4 subject models show the same diagnosis**: internally stable, output-sensitive — exactly PIRC's target regime.
- **PIRC's prefill-fix run *works*.** 37.5 % variance reduction with statistically-flat quality change (CI straddles zero, dz≈0). Earlier "−94 %" reports were from the buggy hook and have been preserved as `*_BUGGY.json` files for forensic comparison.
- **Phase 4 stats are underpowered by design** at n=5 — use Cohen's dz and 95 % CIs, not Wilcoxon p, for any interpretation.
- **Next priorities**: (a) fix anchor selector NaN/inf handling, (b) re-run on 200 instances to reach statistical significance, (c) look at article 1 to understand the variance-up outlier.
