# EVALUATION.md — Complete Evaluation Process

> Exact definition of every metric, score, and statistical test in the pipeline,
> with the source location and current formula for each. This is the authoritative
> reference for *what the numbers mean*. For pipeline wiring see
> [`IMPLEMENTATION.md`](IMPLEMENTATION.md); for changelog see [`method.md`](method.md).
>
> **Convention:** every metric is clamped to **[0, 1]** with `max(0, min(1, x))`.
> Unless noted, "responses" = the list of model outputs, one per prompt variant of
> the same instance.

---

## 0. The big picture

For each `(subject_model, instance)` we run all K paraphrase variants and reduce
the K outputs to a handful of **axes**, then combine them into composite scores:

```
          ┌─ Consistency ── SMS            (are outputs semantically stable?)
 PRI ─────┼─ Quality ────── CS  (vs ref)   (are outputs correct?)
          └─ Faithfulness ─ NLI(src→resp)  (are outputs grounded?)
                    │
                    ├─ short-output gate (×0.85 if avg_len < 12)
                    ▼
 Final_Score = 0.6·PRI + 0.4·Human_Score   (Human = LLM-as-a-Judge, all variants)

 ORI = (SMS + AUC-E + (1−TRD) + KPIG)/4     (observable robustness, cross-model)
 IFI = 1 − (PPL_var + BF)/2                 (intra-model diagnostic only)
```

The three PRI axes are deliberately **non-collinear** (rectification R3):
consistency ⟂ quality ⟂ faithfulness. Metrics that turned out redundant or
unfair were demoted to diagnostics (see Section 9).

---

## 1. The three PRI axes

### 1.1 Consistency — SMS (Semantic Manifold Stability)
`src/sms_metric.py` · input: response embeddings `E ∈ ℝ^{K×d}`

```
SMS = mean(pairwise_cosine(E)) − 0.5 · mean(‖e_normed − centroid‖²)
```
- Embeddings are L2-normalised before the diversity penalty so it is
  scale-invariant. If the penalty < 1e-6 (near-identical outputs) the raw mean
  cosine is returned (SMS ≈ 1.0). Clamped to [0,1].
- **High = outputs cluster tightly** across paraphrases (robust).

### 1.2 Quality — CS (Correctness Score)
`src/correctness_metric.py` · input: responses, reference, embeddings

Per response: `cs = 0.5·cosine(resp, reference) + 0.5·entity_coverage`, where
`entity_coverage = |ref_entities ∩ resp_entities| / |ref_entities|`
(capitalised tokens). If `coverage < 0.2` → `cs ×= 0.6`. **CS = mean** over
responses. Also returns `avg_length` (mean word count) and `avg_coverage`.
- Kept **independent** of consistency and faithfulness (R3/R4): no KPIG multiplier,
  no hallucination multiplier.

### 1.3 Faithfulness — NLI entailment
`src/faithfulness_metric.py` · input: responses, source text

Two backends, selected by `FAITHFULNESS_BACKEND`:
- **`nli`** (default) — `cross-encoder/nli-deberta-v3-small`:
  `faithfulness = P(entailment) + 0.5·P(neutral)` of `source → response`,
  averaged over variants.
- **`llm`** (H100) — an independent large instruct model classifies each response
  as entailment / neutral / contradiction → 1.0 / 0.5 / 0.0, averaged.

Returns `None` if the model can't load → caller falls back to `1 − HS`.
- **High = the source supports the response** (not fabricated).

### 1.4 PRI (Prompt Robustness Index)
`src/scores.py::compute_pri` — the single source of truth (mirrored by
`reaggregate.py`):

```
PRI = 0.40·clamp(SMS) + 0.35·clamp(CS) + 0.25·clamp(Faithfulness)
if avg_length < 12:  PRI ×= 0.85            # short-output gate (R4)
PRI = clamp(PRI)
```

**Labels** (`evaluator.py`): PRI ≥ 0.70 → *Robust*; ≥ 0.50 → *Moderate*; else
*Unreliable*.

---

## 2. ORI (Observable Robustness Index)
`src/scores.py::compute_ori`

```
ORI = (SMS + AUC-E + (1 − TRD) + KPIG) / 4
```
Cross-model–comparable robustness from four observable (output-only) signals.
Here **TRD = semantic TRD** and **KPIG = coverage KPIG** (the canonical variants,
R6/R3).

### 2.1 AUC-E (Performance Elasticity)
`src/auc_e_metric.py` — `AUC-E = 1 − CV(ROUGE-L)` where `CV = std/mean` of ROUGE-L
(vs reference) across responses (ddof=1). High = ROUGE-L is **stable** across
paraphrases.

### 2.2 TRD (Thematic Robustness Drift) — canonical = semantic
`src/metrics_advanced.py::compute_trd_semantic` — mean cosine **distance** of
response embeddings from their centroid. **Lower = more coherent** (which is why
ORI uses `1 − TRD`). The legacy length-variance version
(`trd_metric.py`, `Var(len)/mean(len)²`) is retained only as the diagnostic
`trd_length`.

### 2.3 KPIG (Key-Point Information Gain) — canonical = coverage
`src/metrics_advanced.py::compute_kpig_advanced` — weighted coverage of the
reference's key units (named entities, keywords, bigrams) by the responses:
TF-IDF importance × inverse-frequency redundancy weight × semantic-match quality,
then a variance factor `max(0.3, 1 − e^{−3·std(coverage)})` that penalises
identical outputs. Designed **not to saturate** at 1.0. The constraint-aware
version in `kpig_metric.py` is the legacy/basic variant.

---

## 3. IFI (Intrinsic Fidelity Index) — diagnostic only
`src/scores.py::compute_ifi`

```
IFI = 1 − (PPL_var + BF) / 2
```
> **Intra-model only (R6):** PPL-variance and branching-factor are computed from a
> model's *own* internals, so they are **not comparable across architectures** and
> never enter the cross-model PRI ranking. In score-from-CSV mode (no generation
> model) both are 0 and IFI = 1.0.

- **PPL variance** (`model_interface.compute_perplexity_variance`): coefficient of
  variation of per-`(prompt,response)` perplexity across variants, ×2, clamped.
- **Branching Factor** (`model_interface.compute_branching_factor`): mean per-token
  softmax entropy of responses ÷ 8.0, clamped.

---

## 4. Quality / human-alignment metrics

### 4.1 Human_Score — LLM-as-a-Judge
`src/llm_judge.py::llm_judge_mean` — judges **every** prompt variant's response
(R5) on a 1–5 scale and returns `mean(score)/5 ∈ [0,1]`.
- seq2seq judges (flan-t5 / BART / Pegasus) use the local T5 pipeline;
- everything else (e.g. a 70B instruct judge) is routed through the shared
  `llm_backend.ChatLLM`. Unparseable outputs retry once, then default to 0.5/0.6
  (graceful, never crashes).

### 4.2 Final_Score
`src/scores.py::compute_final_static` (default, static):
```
Final_Score = 0.60·PRI + 0.40·Human_Score
```
Dynamic mode (`enable_dynamic_weighting`, `evaluator._compute_final_score`): when
PRI and the judge diverge (high USD), trust the human more:
`trust_human = min(0.70, 0.40 + 0.30·USD)`, `trust_pri = 1 − trust_human`.

### 4.3 USD (Utility-Stability Divergence)
`src/metrics_advanced.py::compute_usd` — `USD = |PRI − Human_Score|`. A large gap
flags calibration issues or evaluation artifacts; it drives the dynamic weighting.

### 4.4 iPRI
`evaluator.py` — `iPRI = clamp(PRI · CS)`: a correctness-gated PRI kept as an
auxiliary diagnostic.

---

## 5. Hallucination Score (HS) — diagnostic only (R4)
`src/hallucination_metric.py` — fraction of response content words (≥4 chars,
excluding a common-word stoplist) **absent** from the grounding vocabulary
(input + reference), averaged over responses. `HS = 0` → fully grounded,
`HS = 1` → fully fabricated.
> Since R4, HS is **diagnostic only**: it no longer multiplies into CS, and is not
> a PRI override. Faithfulness (NLI) is the single faithfulness signal in PRI. HS
> is retained for interpretability logs and as the `1 − HS` fallback when NLI is
> unavailable.

---

## 6. Baseline overlap metrics
`src/metrics_advanced.py` — reported for comparison, not part of PRI:
- **ROUGE-1/2/L** (`compute_rouge_scores`): mean n-gram/LCS F-measure of each
  response vs reference. Enabled by default (`enable_rouge`).
- **BERTScore** (`compute_bertscore`): contextual-embedding P/R/F1. Off by default
  (`enable_bertscore`, needs the `bert-score` package).

---

## 7. Attribution / robustness diagnosis
`src/attribution_matrix.py` — classifies *why* a sample is (un)robust from the ORI
sub-metrics. Lower-is-better metrics (`trd`, `ppl_var`, `bf`) are flipped to
`1 − v` first, then:

| Category | Rule |
|----------|------|
| **True Robustness** | all adjusted metrics ≥ 0.65 |
| **Evaluation Artifact** | any adjusted metric < 0.20 |
| **Stochastic Luck** | avg ≥ 0.55 **and** spread (max−min) > 0.35 |
| **Knowledge Boundary** | otherwise (moderate / mixed) |

---

## 8. Phase 3/4 — intervention evaluation

These quantify whether **LL-PIRC** reduces sensitivity (`evaluate.py`).

### 8.1 Baseline variance
`experiment_baseline.py` — per article, `var_baseline = Var_k(ROUGE-L)` across the
K=5 paraphrase outputs. Higher variance = more prompt-sensitive.

### 8.2 Relative variance reduction
```
Variance_Reduction = 1 − mean(var_pirc) / mean(var_baseline)
```
PIRC yields one deterministic output per article, so `var_pirc = 0` by
construction → the headline result is that PIRC **eliminates** cross-paraphrase
variance while keeping ROUGE-L within the baseline range.

### 8.3 Wilcoxon signed-rank tests (α = 0.01)
Non-parametric, paired (`evaluate.py`):
- **Variance** — one-sided (`alternative='greater'`, baseline > PIRC); pairs with
  zero difference are dropped; needs ≥ 2 non-zero pairs.
- **ROUGE-L quality** — two-sided, to check PIRC didn't significantly *change*
  quality.

### 8.4 ℓ\* distribution
mean / std / min / max / coefficient-of-variation of the detected sensitive layer
across articles — a low CV means the sensitive layer is stable across inputs.

### 8.5 Logit-Lens / sensitivity definitions (Phase 3 internals)
| Symbol | Definition | File |
|--------|-----------|------|
| per-layer logits | `lm_head(final_norm(h^ℓ))` | `logit_lens.py` |
| per-token PPL | `exp(−log_softmax(logits^ℓ)[next_token])` | `logit_lens.py` |
| `S(ℓ)` | `Var_k[mean per-token PPL at ℓ]` over K paraphrases | `sensitive_layer.py` |
| `ℓ*` | `argmax_ℓ [S(ℓ) − S(ℓ−1)]` (inflection); z-score>2 fallback | `sensitive_layer.py` |
| anchors | bottom 30 % by `rank(mean_ppl)+rank(var_ppl)`, content-filtered | `anchor_tokens.py` |
| clamp | `h[anchors] = α·mean_h + (1−α)·h_original` (α=1 hard, 0.5 soft) | `pirc.py` |

---

## 9. What changed and why (rectifications R1–R7)

The evaluation was audited and seven structural flaws were fixed. Summary (full
detail in `method.md`):

| ID | Flaw | Fix |
|----|------|-----|
| **R1** | GenSens dataset disconnected from evaluator (used synthetic prompts) | `.jsonl` bridge feeds real paraphrase variants into PRI |
| **R2** | Sampling noise confounded prompt sensitivity | greedy decoding (`do_sample=False`) is the default |
| **R3** | PRI axes collinear (SMS≈KPIG≈SMS_W) | PRI = consistency ⟂ quality ⟂ **NLI faithfulness**; drop SMS_W |
| **R4** | Hallucination penalty applied 3× (into CS, PRI override, gate) | HS counted **once**, demoted to diagnostic |
| **R5** | Judge saw only `responses[0]` | judge **all** variants, average |
| **R6** | TRD was length-only; PPL/BF mixed into cross-model rank | TRD = semantic; PPL/BF → IFI (intra-model) only |
| **R7** | Validator matched stale dataset files | sort by mtime, take newest |

### Legacy / superseded code (kept, not in the live path)
- `pri_calculator.py` — the original **weighted harmonic-mean** PRI over 6 ORI
  metrics; superseded by the 3-axis `scores.compute_pri`.
- `trd_metric.py` (length variance), `kpig_metric.py` (constraint-aware),
  `compute_sms_wasserstein` — retained as diagnostics / ablations only.
- `prompt_generator.py` d1/d2/d3 templates — fallback only when a sample has no
  GenSens variants.

---

## 10. Output files (where the numbers land)

| File | Produced by | Contents |
|------|-------------|----------|
| `results/responses.csv` | Phase 2 (full / generate-only) | every model×instance×variant response |
| `results/scored_samples.csv` | Phase 2 (full / score-from-CSV) | raw per-sample metric components (Layer C) |
| `results/benchmark.csv` | Phase 2 full run | per-model aggregated PRI/ORI/IFI/CS/HS/Faithfulness/Final |
| `results/benchmark_reaggregated.csv` | `reaggregate.py` | recomputed composites + honest correlations (no GPU) |
| `results/correlations.json` | Phase 2 | metric correlation analysis |
| `results/baseline.json` · `pirc.json` | Phase 3 | per-article ROUGE-L, variance, ℓ\*, anchors |
| `results/eval_summary.json` | Phase 4 | variance reduction, Wilcoxon, ℓ\* stats |
