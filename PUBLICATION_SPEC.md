# Publication Spec — GenSens + LL-PIRC

> **Single source of truth for the NeurIPS D&B 2026 submission.**
> Refer to this document during implementation. Update it (don't fork it) as
> decisions evolve. CLAUDE.md still governs methodology changelogging into
> `method.md`.

---

## 1. Submission target

| | |
|---|---|
| **Primary venue** | NeurIPS Datasets & Benchmarks 2026 |
| **Submission deadline** | mid-May 2026 (estimated) |
| **Backup venues** | EMNLP 2026 Findings · LREC-COLING 2026 · ACL 2026 (ARR) |
| **Page limit** | 9 pp main + appendix (NeurIPS D&B) |
| **Track** | Datasets & Benchmarks track (NOT main NeurIPS) |

## 2. Paper framing

**Title (working):** *GenSens: A Multi-Task Diversity-Controlled Benchmark for
Prompt Sensitivity Evaluation, with LL-PIRC as a Training-Free Mitigation Baseline*

**Dual contribution:**
1. **Primary (GenSens benchmark)** — multi-task prompt-robustness benchmark
   with bidirectional-NLI gates, principled paraphrase-strategy taxonomy,
   and automated quality validation calibrated to human-annotated reference data.
2. **Secondary (LL-PIRC method)** — training-free inference-time intervention
   that reduces prompt sensitivity via Logit-Lens-driven sensitive-layer
   detection and residual clamping.

**Page allocation (9 pp main):**
- 1.5 pp — Introduction
- 1.0 pp — Related work
- 3.0 pp — GenSens benchmark (PRIMARY)
- 1.5 pp — LL-PIRC method (SECONDARY)
- 1.5 pp — Experiments + ablations + quality validation
- 0.5 pp — Limitations + conclusion

## 3. Differentiation claims

GenSens differs from existing prompt-robustness benchmarks on four axes:

| Benchmark | # Tasks | # Instances | # Paraphrases | Gold ref? | Diversity controls? | Multi-axis score? |
|---|---|---|---|---|---|---|
| PromptBench (Zhu et al., 2023) | 8 | varies | attack-based | partial | no | no |
| RobustAlpacaEval (Cao et al., 2024) | 1 | 100 | 10 (human) | no | implicit | no |
| Razavi et al. (2025) | 2 | 11,469 | 9 | yes (QA) | length only | no |
| **GenSens (ours)** | **3** | **200** | **8 (filtered from 16+)** | **yes (all)** | **yes (5)** | **yes (PRI + diagnostic)** |

## 4. Subject roster (final)

| Tier | Model | Purpose |
|---|---|---|
| Small (1–3B) | `Qwen/Qwen2.5-1.5B-Instruct` | establish lower bound (REQUIRED for discriminability) |
| Mid (7–8B) | `meta-llama/Llama-3.1-8B-Instruct` | PIRC target + cross-arch |
| Mid (7–8B) | `Qwen/Qwen2.5-7B-Instruct` | cross-arch |
| Mid (7–8B) | `mistralai/Mistral-7B-Instruct-v0.3` | cross-arch |
| Closed (API) | `gpt-4o-mini` (OpenAI API) | frontier reference upper bound (~$15) |

Generator: `LLaMA-3.1-70B-AWQ`. Judge: 8B during iteration → 70B for final tables.
Embedder: `BAAI/bge-large-en-v1.5`.

**Drop the 70B subject** to save GPU time (paper #1; reserve for journal extension).

## 5. Task roster

Three tasks (drop QA):

| Task | Source | Variant target |
|---|---|---|
| Summarisation | CNN/DailyMail 3.0.0 | K=8, n=200 |
| Creative (article expansion) | CNN/DailyMail 3.0.0 (highlights→article) | K=8, n=200 |
| Dialogue (multi-turn QA) | DREAM | K=8, n=200 |

Total: 200 × 3 × 8 = 4 800 prompts per subject model.

---

## 6. Paraphrase generation — TIER 1 improvements (MUST)

These six items are **required** for any publication acceptance. Items 1–4
are pure code changes in `gensens/scripts/paraphrase_generator.py`.

### 6.1 Bidirectional NLI gate on ALL 4 tasks

Apply `cross-encoder/nli-deberta-v3-large` bidirectionally to every
(base, candidate) pair. Reject if `P(entail|base→cand) < τ_NLI` OR
`P(entail|cand→base) < τ_NLI`. Currently the gate fires only on QA/dialogue.

**Threshold:** τ_NLI = 0.50 (matches existing QA/dialogue threshold).
**Code surface:** new `_passes_bidirectional_nli()` helper; called inside
`generate_paraphrases()` after SBERT band + token overlap floor.

### 6.2 Length-ratio filter

Reject if `len(candidate_words) / len(base_words) ∉ [0.7, 1.5]`. Word count,
not chars.

**Code surface:** new `_passes_length_ratio()`; called immediately after
NLI gate.

### 6.3 SBERT semantic dedup (replace first-8-words hash)

Reject candidate if `cos_SBERT(candidate, accepted_v) > 0.95` for any
already-accepted variant `accepted_v`. Token Jaccard floor (0.85) retained
as a *second* gate.

**Code surface:** new `_passes_semantic_dedup()`; called after token Jaccard.
The `seen_sigs` first-8-words set is kept as a fast pre-filter but no longer
the primary dedup.

### 6.4 Per-variant audit metadata

Every accepted variant carries these new fields in the JSONL output:

```json
{
  "variant_idx": 3,
  "paraphrased_text": "...",
  "strategy": "passive_voice",
  "strategy_family": "syntactic",
  "sbert_similarity": 0.8742,
  "token_overlap_to_base": 0.71,
  "length_ratio": 1.12,
  "nli_entail_fwd": 0.91,
  "nli_entail_bwd": 0.88,
  "nli_passed": true,
  "best_of_n_index": 2
}
```

`nli_entail_fwd/bwd` already exist for QA/dialogue — extend to all variants.
The `strategy_family` field is added per item 6.6.

### 6.5 Best-of-N per strategy (n=3)

For each (instance, strategy) call to vLLM, request `n=3` candidates with
`temperature=0.95, top_p=0.92, seed=42+strategy_idx`. Among candidates that
pass ALL filters (SBERT band, length ratio, token Jaccard, semantic dedup,
NLI), select the one with **lowest token Jaccard to all already-accepted
variants** (max diversity).

**Code surface:** modify `generate_batch()` to accept `n=` kwarg; modify
`_generate_single_vllm()` to return a list of candidates; modify
`generate_paraphrases()` to pick best-of-N by diversity.

### 6.6 Strategy families taxonomy

Replace the 16-strategy per-strategy cap with a **4-family per-family cap**:

| Family | Strategies | `max_per_family` |
|---|---|---|
| **lexical** | `formal_tone`, `casual_tone`, `synonyms`, `technical_vocab`, `simple_vocab` | 2 |
| **syntactic** | `reordered_clauses`, `different_structure`, `different_opening`, `passive_voice`, `split_sentences`, `merged_sentences` | 2 |
| **pragmatic** | `question_form`, `imperative`, `role_prefix` | 1 |
| **length** | `concise`, `elaborated` | 1 |

Total budget: 2+2+1+1 = 6 from prompt-engineering strategies. Slot 7–8 reserved
for **back-translation** (Tier-2 family — added later).

**Code surface:** new `STRATEGY_FAMILY` mapping + `MAX_PER_FAMILY` dict;
`strategy_counts` becomes `family_counts`.

---

## 7. Paraphrase generation — TIER 2 improvements (STRONG ACCEPT)

### 7.1 Multi-NLI ensemble (3-of-3 majority)

Three NLI models in ensemble:
- `cross-encoder/nli-deberta-v3-large` (primary)
- `roberta-large-mnli`
- `facebook/bart-large-mnli`

Require **≥2 of 3** to assert bidirectional entailment.

### 7.2 Round-trip back-translation as a 5th family

Add `back_translation` family using `Helsinki-NLP/opus-mt-en-{de,fr,ru}` and
reverse models. Already in `config.yaml` as legacy — re-enable.

### 7.3 TextFooler-style adversarial paraphrase subset

Word-substitution attack with semantic guard:
1. POS-tag base prompt; iterate over content words (NOUN/VERB/ADJ).
2. For each word, find top-5 synonym candidates (CounterFitted GloVe or BERT MLM).
3. Filter: SBERT cos ≥ 0.85, bidirectional NLI passes, POS preserved.
4. Substitute the candidate that **maximally reduces subject-model output
   SBERT similarity** (this is the adversary).
5. Cap at 3–5 substitutions per prompt.

Output `adversarial_paraphrases_textfooler.jsonl`. Rename current
`adversarial_paraphrases.py` → `noise_paraphrases.py` (typos/negation/hedge
are noise, not adversaries).

### 7.4 GPT-4o-mini calibration

Sample 100 base prompts. Generate one paraphrase per prompt with GPT-4o-mini
via the OpenAI API using the same system prompt as the 70B AWQ generator.
Run bidirectional NLI ensemble on both sets. Report pass-rate parity.

### 7.5 Few-shot examples in system prompts

Add 2–3 positive paraphrase examples + 1 negative example per task in the
SYSTEM_PROMPTS dict.

---

## 8. Quality validation — 4-pronged automated approach (REPLACES human annotation)

Since fresh human annotation is out of scope, we use this four-pronged
automated validation, framed as **"comprehensive automated validation
calibrated to human-annotated reference data."**

### 8.1 Prong 1 — LLM-as-judge for paraphrase quality

Use `gpt-4o-mini` as the paraphrase-quality judge on 500 stratified
(base, paraphrase) pairs:

```
System: You are evaluating whether two sentences are paraphrases of each other.
A 5 means perfect paraphrase (identical meaning, different wording).
A 1 means not a paraphrase (different meaning).
Rate the following pair on a 1–5 scale. Output ONLY the integer.

Base:       {base}
Paraphrase: {candidate}

Rating:
```

**Calibration step (required):** also run on 100 RobustAlpacaEval paraphrases
with known human ratings. Report Cohen's κ between GPT-4o-mini and human
ratings as the LLM-as-judge legitimacy claim.

**Citations:** Zheng et al. (NeurIPS 2023, MT-Bench); Liu et al. (EMNLP 2023, G-Eval).

**Cost:** ~$2–5 in API. **Effort:** 1 day.

### 8.2 Prong 2 — Multi-NLI ensemble (item 7.1 above)

False-positive rate measured on PAWS-X test set: target ≤ 5%.

### 8.3 Prong 3 — PAWS-style negative controls

Construct 200 (base, NEAR-paraphrase) pairs that share high lexical overlap
but are NOT paraphrases:
- Subject/object swap
- Polarity flip (positive ↔ negative claim)
- Quantifier swap (all ↔ some)
- Antonym substitution
- Modifier flip

Template-generated, no human required. Report **filter rejection rate**;
target ≥ 90%.

**Cost:** $0. **Effort:** 1 day.

### 8.4 Prong 4 — Borrowed human validation (the killer move)

Run the GenSens filter pipeline on RobustAlpacaEval's 1000 human-verified
(base, paraphrase) pairs. Report:
- **Pass rate** — target ≥ 88% (human-verified paraphrases should mostly pass).
- **Kolmogorov-Smirnov distributional similarity** of (SBERT cos, length
  ratio, token Jaccard, bidirectional NLI) between GenSens and
  RobustAlpacaEval; target p > 0.05 on all four metrics.

This is **transitive human validation**: if our paraphrase distribution is
indistinguishable from a known human-verified one, our paraphrases inherit
that legitimacy.

**Cost:** $0. **Effort:** 1 day.

### 8.5 Paper-ready quality table

This goes verbatim into the paper:

| Validation prong | Metric | Target | Actual |
|---|---|---|---|
| 1. LLM-as-judge (GPT-4o-mini) | Mean rating (1–5) | ≥ 4.0 | TBD |
| 1a. Calibration | κ vs human (RobustAlpacaEval) | ≥ 0.60 | TBD |
| 2. Multi-NLI ensemble | False-positive rate (PAWS-X) | ≤ 5% | TBD |
| 3. PAWS-style negative controls | Filter rejection rate | ≥ 90% | TBD |
| 4. RobustAlpacaEval pass rate | Pass rate on human-verified | ≥ 88% | TBD |
| 4. Distributional alignment | KS test p-value (4 metrics) | all > 0.05 | TBD |

---

## 9. Benchmark statistical properties (REQUIRED for D&B)

Already implemented in `prompt_robustness/scripts/benchmark_properties.py`:

| Property | Healthy target | Pilot result (n=5) |
|---|---|---|
| PRI spread across models | ≥ 0.10 | 0.030 (NEEDS WIDER MODEL ROSTER) |
| Inter-task Spearman ρ | 0.30 ≤ \|ρ\| ≤ 0.85 | mean -0.10 (healthy, but n small) |
| Variance by task (η²) | ≥ 0.10 | 0.265 (healthy) |
| Variance by model (η²) | ≥ 0.05 | 0.015 (TOO LOW — fixed by wider roster) |
| Power for medium effect | n ≥ 45 per model | n = 19 (NEEDS SCALE-UP) |

Action: scale to n=200 × 3 tasks AND add 1.5B + GPT-4o-mini subjects.

---

## 10. LL-PIRC method (secondary contribution)

Already implemented; gaps to close before submission:

- ✅ Logit Lens NaN fix (commit 2026-06-04)
- ✅ IFI saturation fix (PPL/BF persistence)
- ✅ Prefill-only PIRC clamping
- ☐ ℓ* ablation sweep (CLI flag exists; run on n=200)
- ☐ α ablation sweep (CLI flag exists; run on n=200)
- ☐ Anchor percentile sweep (10%, 20%, 30%, 50%, 70%)
- ☐ Mitigation baseline comparison driver (`experiment_baselines_comparison.py`)
- ☐ Cross-validation on RobustAlpacaEval (100 queries)

---

## 11. Budget

| Item | Cost |
|---|---|
| GPU compute (~12 full runs on A6000 spot) | $60–100 |
| OpenAI API (GPT-4o-mini judge + calibration + subject) | $20–35 |
| HF Datasets/Spaces release infrastructure | $0 |
| **Total** | **$80–135** |

Compare: ~$500–800 with human annotation. ~85% cost reduction by using
the 4-pronged automated validation instead.

## 12. 6-week timeline

| Week | Focus |
|---|---|
| **1** | Tier-1 paraphrase generator (NLI all tasks, length filter, SBERT dedup, best-of-N, strategy families, audit metadata). Multi-NLI ensemble. Regenerate 5-instance pilot. |
| **2** | Validation pipeline: GPT-4o-mini judge + PAWS negative controls + RobustAlpacaEval borrowed validation + KS distributional alignment. |
| **3** | TextFooler adversarial module. Back-translation family. Full-pipeline regression test. |
| **4** | Run full 200-instance × 3-task × 4-subject benchmark. Persist `responses.csv` + `scored_samples.csv`. Run cross-model RobustAlpacaEval evaluation. |
| **5** | LL-PIRC + 4 mitigation baselines. All ablation sweeps (ℓ*, α, anchor%, embedder). Generate all paper tables/figures. |
| **6** | Write paper. Internal review. HuggingFace Datasets release. Submit. |

Buffer week (recommended): week 7 for revision/polish.

## 13. Public release artifacts (REQUIRED for D&B)

| Artifact | Hosting | License |
|---|---|---|
| GenSens dataset (JSONL + CSV) | HuggingFace Datasets `<user>/gensens-v1` | CC-BY-4.0 |
| Evaluation harness | GitHub + PyPI `pip install gensens-eval` | MIT |
| Leaderboard (optional) | HuggingFace Spaces | — |
| Reproducibility | `requirements_gpu_pinned.txt` + `Dockerfile` | — |

## 14. Acceptance criteria for "publication-ready" methodology

The paraphrase generation methodology is publication-ready when ALL of these are true:

- [ ] Bidirectional NLI gate fires on all 4 tasks (currently 2)
- [ ] Length-ratio filter applied to every candidate
- [ ] SBERT semantic dedup replaces first-8-words hash
- [ ] Best-of-N per strategy with diversity selection
- [ ] 4-family strategy taxonomy with per-family caps
- [ ] Per-variant audit metadata (NLI fwd/bwd, SBERT, length ratio, token overlap)
- [ ] Multi-NLI ensemble (≥2 of 3 majority vote)
- [ ] Back-translation as 5th strategy family
- [ ] TextFooler adversarial subset (separate from noise subset)
- [ ] GPT-4o-mini-as-judge quality validation on 500 sampled pairs
- [ ] LLM-as-judge calibration on RobustAlpacaEval (κ ≥ 0.60)
- [ ] PAWS-style negative controls (200 pairs, rejection rate ≥ 90%)
- [ ] RobustAlpacaEval borrowed human validation (pass rate ≥ 88%, KS p > 0.05)
- [ ] Reproducibility: pinned requirements, seeded RNGs, documented prompts

When all 14 are checked, the paraphrase generation methodology will clear
NeurIPS D&B reviewer scrutiny.

---

*Last updated: 2026-06-04*
