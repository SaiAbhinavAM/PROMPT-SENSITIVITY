# PRI / ORI Benchmark Analysis — 5-instance run

Focused analysis of the **Phase 2 cross-model robustness benchmark** that produced `scored_samples.csv`. This document only covers PRI, ORI, IFI, Final_Score, and their diagnostic variants — not the LL-PIRC mitigation.

Source data: 76 rows of `scored_samples.csv` (4 models × 19 instances pooled from 4 tasks).

---

## 1. The Composite Scores — What Each One Is For

The pipeline produces **five composite scores** per (model, instance) pair. Each is computed by a single canonical formula in `src/scores.py`.

### 1.1 PRI — Prompt Robustness Index (primary cross-model ranking)

Arithmetic weighted mean over three **non-collinear axes**:

```
PRI = 0.40 · Consistency(SMS)
    + 0.35 · Quality(CS_vs_reference)
    + 0.25 · Faithfulness(NLI_entail+0.5·neutral)
```

- Plus a structural gate: if `avg_length < 12` tokens → `PRI *= 0.85` (penalise degenerate too-short outputs).
- Clamped to [0, 1].
- **Why arithmetic mean**: forgiving — a weak component can be partially compensated by strong ones, which is the correct behaviour for *ranking* models.

### 1.2 ORI — Observable Robustness Index (output-side only)

Arithmetic mean over four output-side metrics:

```
ORI = (SMS + AUC-E + (1 − TRD) + KPIG) / 4
```

- TRD is "lower-is-better", so it is inverted before averaging.
- No reference required → can be computed even when no gold answer exists.
- **Why averaged equally**: ORI is meant to be the most "observable" robustness signal — what you can see in the outputs themselves.

### 1.3 IFI — Intrinsic Fidelity Index (intra-model diagnostic ONLY)

```
IFI = 1 − (PPL_var + BF) / 2
```

- `PPL_var` = variance of teacher-forced perplexity across the K paraphrases.
- `BF` = first-token branching factor entropy.
- **Cross-model unusable** — different architectures have different PPL/BF scales. IFI is informative only when comparing one model against itself.

### 1.4 Final_Score — Production Composite

```
Final_Score = 0.6 · PRI + 0.4 · Human_Score
```

Human_Score = LLM-as-Judge rating (70B AWQ) of each variant, 1–10 normalised to [0, 1], averaged.

### 1.5 Diagnostic_* — Strict Publication Variants (harmonic mean)

```
Diagnostic_ORI = HM(SMS, AUC-E, 1 − TRD, KPIG)         # weighted harmonic
Diagnostic_IFI = HM(1 − PPL_var, 1 − BF)
Diagnostic_PRI = HM(Diagnostic_ORI, Diagnostic_IFI)    # equal weights 0.5/0.5
```

- Harmonic mean is **strict**: if any pillar approaches 0, the composite collapses to 0.
- Intended for publication figures where weak axes should not be hidden by strong ones.
- 2 × 2 diagnosis from (Diagnostic_ORI ≥ 0.70, Diagnostic_IFI ≥ 0.70).

---

## 2. Per-Model Results (means over all 19 evaluations)

```
                  Qwen-7B  Llama-70B-AWQ  Llama-8B  Mistral-7B
PRI                0.775       0.787       0.784      0.757
ORI                0.767       0.741       0.743      0.756
IFI                1.000       1.000       1.000      1.000     ← saturated
Final_Score        0.799       0.819       0.818      0.770
Diagnostic_ORI     0.567       0.517       0.504      0.564
Diagnostic_IFI     1.000       1.000       1.000      1.000
Diagnostic_PRI     0.710       0.672       0.666      0.708
Human_Score        0.834       0.866       0.870      0.791
```

### 2.1 Model rankings differ by score

| Score | 1st | 2nd | 3rd | 4th |
|-------|-----|-----|-----|-----|
| **PRI** | Llama-70B-AWQ (0.787) | Llama-8B (0.784) | Qwen-7B (0.775) | Mistral (0.757) |
| **ORI** | Qwen-7B (0.767) | Mistral (0.756) | Llama-8B (0.743) | Llama-70B (0.741) |
| **Final_Score** | Llama-70B (0.819) | Llama-8B (0.818) | Qwen (0.799) | Mistral (0.770) |
| **Diagnostic_PRI** | Qwen (0.710) | Mistral (0.708) | Llama-70B (0.672) | Llama-8B (0.666) |
| **Human_Score** | Llama-8B (0.870) | Llama-70B (0.866) | Qwen (0.834) | Mistral (0.791) |

**No single ranking dominates.** PRI and Diagnostic_PRI even disagree (the harmonic form punishes Llama-70B and Llama-8B because of their lower KPIG / AUC-E components, even though their consistency and faithfulness are strong).

### 2.2 Score-vs-score correlations across the 76 rows

```
corr(PRI, ORI)            = 0.148  ← they really do measure different things
corr(PRI, Final_Score)    = 0.456
corr(PRI, Human_Score)    = −0.185 ← judge disagrees with structural metric
corr(ORI, Human_Score)    = −0.302 ← stronger disagreement
corr(Diagnostic_PRI, PRI) = −0.134 ← harmonic and arithmetic forms ANTI-correlate
corr(Diagnostic_ORI, ORI) = 0.788
```

**Reading:**
- **PRI ⟂ ORI (r = 0.15).** PRI captures *agreement with reference* (CS + Faithfulness dominate); ORI captures *cross-paraphrase output stability*. By design they should *not* be redundant, and the data confirms it.
- **PRI vs Human_Score is slightly negative.** The judge rewards verbose, fluent outputs (`avg_length` correlates with Human at r=0.83); structural quality (CS / KPIG) doesn't track that. **USD metric exists precisely to flag this disagreement.**
- **Diagnostic_PRI vs PRI is anti-correlated.** The harmonic-mean form is sensitive to the weakest pillar, so models with one mediocre component get hammered. Useful as a *strict* publication score, not as a ranking score.
- **Diagnostic_ORI vs ORI r = 0.79.** Same components, different aggregator — harmonic ≈ arithmetic when no component is near zero, so they agree more than the PRI variants.

---

## 3. Per-Task Breakdown

### 3.1 PRI by task (pooled over models)

```
              creative  dialogue  qa   summarization
PRI            0.727     0.847   0.732    0.810
ORI            0.750     0.803   0.724    0.740
Final_Score    0.809     0.719   0.828    0.833
```

- **Dialogue has the highest PRI (0.847)** but the *lowest* Final_Score (0.719). The 70B judge marks dialogue responses down (they're terse one-token answers); structural metrics reward the SMS=1.0 perfect agreement.
- **Creative and QA flip the relationship**: Final_Score (driven by judge) is high, but PRI lower because reference-based CS suffers (open-ended tasks → no canonical answer to compare against).
- **Summarization is the most balanced**: PRI ≈ Final_Score ≈ 0.81–0.83 — both signals agree.

### 3.2 Per (model × task) PRI matrix

```
              Qwen-7B  Llama-70B  Llama-8B  Mistral-7B
creative       0.734    0.724      0.728     0.723
dialogue       0.824    0.864      0.905     0.798
qa             0.743    0.772      0.720     0.695
summarization  0.810    0.804      0.808     0.820
```

- **Llama-8B leads dialogue (PRI 0.905)** despite being the smallest of the four. Dialogue's gold-answer mechanic rewards "concise, faithful" outputs, which Llama-8B happens to produce.
- **Mistral leads summarization (0.820)** by writing the shortest outputs (avg_length 102 vs ~127 for others) — its summaries hit reference n-grams more often.
- **Models are tightly clustered** on creative (range 0.723–0.734). With n=5 instances per task, this is below the resolution we can reliably distinguish.

---

## 4. Component-Level Decomposition

### 4.1 PRI = 0.40·SMS + 0.35·CS + 0.25·Faithfulness — what drives the score?

Per-model means:

| Model | SMS (×0.40) | CS (×0.35) | Faith (×0.25) | Sum (= PRI) |
|-------|-------------|------------|---------------|-------------|
| Qwen-7B | 0.359 | 0.208 | 0.231 | 0.798* |
| Llama-70B | 0.359 | 0.210 | 0.226 | 0.795 |
| Llama-8B | 0.357 | 0.203 | 0.224 | 0.784 |
| Mistral | 0.352 | 0.213 | 0.213 | 0.778 |

*\*Tiny rounding mismatch with 0.775 above because of per-instance short-output gating.*

- **SMS is the dominant contributor** (~45% of every PRI). It's saturated near 0.89 for every model — i.e., embedding-level consistency is mostly a property of the **paraphrase set** and the **embedder (BGE-large)**, not the model.
- **CS is the discriminating axis at ~0.21 per row.** It's the only axis where models can meaningfully separate, but it's bottlenecked by `avg_coverage` (r = 0.96 between CS and coverage).
- **Faithfulness adds ~0.22**. Tight range across models (0.852–0.923) — the 70B-AWQ NLI judge is consistent.

### 4.2 ORI = (SMS + AUC-E + (1−TRD) + KPIG) / 4

Per-model means:

| Model | SMS | AUC-E | 1−TRD | KPIG | ORI |
|-------|-----|-------|-------|------|-----|
| Qwen-7B | 0.897 | 0.848 | 0.969 | 0.353 | **0.767** |
| Llama-70B | 0.898 | 0.741 | 0.969 | 0.354 | 0.741 |
| Llama-8B | 0.893 | 0.758 | 0.968 | 0.352 | 0.743 |
| Mistral | 0.880 | 0.826 | 0.965 | 0.353 | 0.756 |

- **(1−TRD) is saturated at ~0.97** for every model → contributes 0.24 of every ORI but ranks no model.
- **KPIG is anchored at 0.35** → contributes 0.09 of every ORI but doesn't differentiate models either.
- **AUC-E is the discriminating axis** in ORI (0.741–0.848). Qwen's strong AUC-E carries it to first place.
- Net: ORI is mostly **SMS + AUC-E** in disguise, with KPIG and (1−TRD) as near-constant ballast. If the goal of ORI is multi-axis output assessment, then either tune KPIG/TRD to have more dynamic range, or weight ORI like `0.5·SMS + 0.5·AUC-E`.

### 4.3 The structural collinearity problem

From the broader correlation matrix:

```
SMS ⟂ TRD: r = −0.98   (TRD adds almost no new information beyond SMS)
CS ⟂ avg_coverage: r = 0.96 (CS is dominated by coverage)
Human_Score ⟂ avg_length: r = 0.83 (judge rewards verbosity)
ROUGE-L ⟂ avg_length:     r = −0.85 (ROUGE punishes verbosity)
```

**Implications for PRI / ORI as research metrics:**
- ORI's TRD axis is redundant with SMS on this dataset. The 4-axis presentation may overstate ORI's coverage of "robustness facets."
- CS's apparent variability is mostly coverage variability. PRI's "Quality" pillar would be more informative if it blended length-adequacy + fluency + coverage rather than ~95 % coverage.
- USD (|PRI − Human_Score|) is the safety belt that surfaces the verbosity disagreement. Keep it visible.

---

## 5. Top / Bottom PRI Rows — What Drives Extremes

### 5.1 Top 5 PRI rows

```
model               topic       SMS    CS    Faith  len  PRI    Human  Final
Llama-70B-AWQ       dialogue    1.00   0.879 1.00    66  0.958  1.00   0.975
Llama-8B            qa          1.00   0.851 1.00   161  0.948  1.00   0.969
Llama-8B            dialogue    1.00   0.842 1.00    30  0.945  1.00   0.967
Llama-70B-AWQ       qa          1.00   0.838 1.00   160  0.943  1.00   0.966
Mistral-7B          qa          1.00   0.834 1.00   140  0.942  1.00   0.965
```

- **All top PRI rows have SMS = 1.00 and Faithfulness = 1.00.** This means all K paraphrase responses for that instance produced byte-identical (or embedding-identical) outputs.
- That's a degenerate-best case: the model just memorised the answer. PRI rewards it, which is correct for a *robustness* index.

### 5.2 Bottom 5 PRI rows

```
model              topic     SMS    CS    Faith  len    PRI    Human  Final
Mistral-7B         qa        0.799  0.245 0.333  112.5  0.489  0.733  0.587
Llama-8B           qa        0.792  0.371 0.583  160.5  0.592  0.733  0.649
Llama-70B-AWQ      creative  0.846  0.262 0.750  144.3  0.617  0.900  0.730
Mistral-7B         qa        0.780  0.200 1.000  147.8  0.632  1.000  0.779
Llama-8B           qa        0.829  0.209 0.938  154.0  0.639  1.000  0.783
```

- **All bottom rows are QA or creative** — open-ended tasks where references are weakest.
- **Faithfulness can be 1.0 while PRI is still low** (rows 4, 5): the response is grounded in the source but doesn't cover the *reference answer*. The CS pillar correctly penalises that.
- **The judge gives 1.0 to most of these rows** — exactly the PRI-vs-Human disagreement that USD surfaces. The judge thinks the response is fine; the structural metric says it doesn't match the reference.

---

## 6. IFI is Saturated — Why That Matters

`IFI = 1 − (PPL_var + BF) / 2` in `scored_samples.csv` is **1.000 for every row of every model**. That means either:
- (a) `ppl_var` and `bf` are reported pre-normalised on a near-zero scale (the raw values in `ifi_metrics.json` show PPL var ≈ 0.07–0.13 and BF ≈ 50 for Llama-8B), or
- (b) the normalisation step that should fold these into [0,1] was suppressed.

**Effect:** `Diagnostic_PRI = HM(Diagnostic_ORI, Diagnostic_IFI) = HM(Diagnostic_ORI, 1) ≈ Diagnostic_ORI`. The dual-pillar synthesis collapses to ORI alone.

**Fix to make IFI useful again:** either rescale PPL_var to a per-model min-max in [0,1] before computing IFI, or report IFI as a separate intra-model panel (as `ifi_metrics.json` already does) and drop it from Diagnostic_PRI for cross-model use.

---

## 7. What This 5-Instance Benchmark Can And Cannot Say

### ✅ Can say

1. The **three PRI axes are genuinely orthogonal** (pairwise |r| < 0.15) — the de-collinearisation effort behind PRI's R3 design holds up.
2. **PRI and ORI measure different things** (r = 0.148). Reporting both is informative, not redundant.
3. **All 4 models are diagnosed "Internally Stable / Output-Sensitive"** (output-side variability without internal disagreement) — the regime LL-PIRC is designed for.
4. **The 4 models cluster very tightly on PRI** (range 0.757–0.787, span 0.030). The benchmark currently doesn't have the dynamic range to distinguish strong-from-strong; it works best to detect catastrophically weak models.

### ❌ Cannot say

1. **Which model is best.** A 0.03 PRI spread across 19 rows per model is well within sampling noise. Need the 200-instance run.
2. **Anything about IFI.** Saturated at 1.000.
3. **Whether the harmonic Diagnostic_PRI is calibrated.** It anti-correlates with arithmetic PRI (r = −0.134); on n=76 that is enough to flag a definition mismatch but not enough to settle which form to publish.
4. **Whether PRI components other than SMS contribute usefully on this dataset.** SMS, (1−TRD), KPIG all saturate; only CS and AUC-E vary meaningfully across models.

---

## 8. Recommendations Before The 200-Instance Run

1. **Fix IFI normalisation.** Either feed pre-normalised `ppl_var` / `bf` from `ifi_metrics.json` into `scored_samples.csv`, or drop IFI from the cross-model report and present it per-model only.
2. **Decide whether TRD stays in ORI.** With r = −0.98 vs SMS it's redundant on this dataset. Either:
   - Replace TRD's slot with `TRD_semantic` (already in the CSV — measure semantic, not length, drift), or
   - Reweight ORI to `0.5·SMS + 0.5·AUC-E` and report TRD/KPIG separately.
3. **Increase CS dynamic range.** Today CS = 0.96·coverage. Blend in fluency / grammaticality / length-adequacy terms with non-trivial variance, or report CS = coverage explicitly.
4. **Keep USD visible** in the dashboard — it's the only metric currently flagging the PRI-vs-judge disagreement.
5. **Resolve PRI vs Diagnostic_PRI.** Pick one as the publication score. The arithmetic PRI is the better *ranking* score; the harmonic Diagnostic_PRI is the better *strict-evaluation* score; you can't really have both without a footnote.

---

## 9. TL;DR

- **PRI** (arithmetic 3-axis: SMS / CS / Faithfulness) is the primary cross-model robustness score. Llama-70B-AWQ marginally first; tight cluster.
- **ORI** (arithmetic 4-axis: SMS / AUC-E / 1−TRD / KPIG) is genuinely independent of PRI (r = 0.15). Qwen-7B marginally first; tight cluster.
- **Final_Score = 0.6·PRI + 0.4·Human_Score** elevates Llama-70B-AWQ (judge prefers it).
- **Diagnostic_*** harmonic-mean variants flip the rankings; arithmetic PRI and harmonic Diagnostic_PRI anti-correlate (r = −0.13).
- **IFI is broken (saturated at 1.000)** in `scored_samples.csv`; the real signal is in `ifi_metrics.json`.
- **n = 5 per task is too small** to settle inter-model differences; all-model PRI spread of 0.03 is within noise.
- **The collinearity audit confirms the R3 design**: SMS / CS / Faithfulness are genuinely independent axes for PRI; (1−TRD) and KPIG in ORI are dataset-saturated and don't differentiate models here.
