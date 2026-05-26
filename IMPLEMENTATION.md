# IMPLEMENTATION.md — End-to-End Implementation Details

> Single-document walkthrough of the **Prompt Sensitivity** pipeline: what each
> phase does, which files implement it, how data flows between phases, and how to
> run it. For the exact metric math see [`EVALUATION.md`](EVALUATION.md); for the
> H100/API deployment see [`H100_PIPELINE.md`](H100_PIPELINE.md); for the
> chronological changelog see [`method.md`](method.md).

---

## 1. What the project does

Large language models are **prompt-sensitive**: two prompts that mean the same
thing but are worded differently can produce inconsistent outputs. This project
(a) builds a benchmark that *measures* that sensitivity, and (b) implements an
inference-time intervention that *reduces* it.

It runs in **four phases**:

| Phase | Name | Entry point | Produces |
|-------|------|-------------|----------|
| 1 | **GenSens** — paraphrase dataset generation | `gensens/scripts/generate_dataset.py` | `gensens/data/*.jsonl` (+ `.csv`) |
| 2 | **PRI Benchmark** — robustness scoring | `prompt_robustness/main.py` | `results/responses.csv`, `results/scored_samples.csv`, `benchmark.csv` |
| 3 | **LL-PIRC** — mitigation intervention | `prompt_robustness/experiment_baseline.py` → `experiment_pirc.py` | `results/baseline.json`, `results/pirc.json` |
| 4 | **Statistical Evaluation** | `prompt_robustness/evaluate.py` | `results/eval_summary.json`, plots |

Phases 1→2 are the **measurement pipeline** (the part scaled for the H100 this
cycle). Phases 3→4 are the **intervention pipeline** (driven by `config.yaml`).

---

## 2. Repository map

```
PROMPT SENSITIVITY/
├── gensens/                          # PHASE 1
│   ├── scripts/
│   │   ├── generate_dataset.py       # orchestrator + CLI (writes .jsonl + .csv)
│   │   ├── paraphrase_generator.py   # 16 strategies, SBERT band, diversity caps, vLLM/HF/flan backends
│   │   ├── data_loaders.py           # source loaders for 4 tasks (CNN/DM, HumanEval+MBPP, WritingPrompts, MultiWOZ)
│   │   └── validate_dataset.py       # post-gen validation (newest file by mtime)
│   └── data/                         # generated datasets
│
├── prompt_robustness/                # PHASES 2-4
│   ├── main.py                       # PHASE 2 entry: full / --generate-only / --responses-csv
│   ├── reaggregate.py                # PHASE 2 re-aggregation from scored_samples.csv (no GPU)
│   ├── experiment_baseline.py        # PHASE 3 baseline (back-translation K=5, ROUGE-L variance)
│   ├── experiment_pirc.py            # PHASE 3 PIRC intervention (ℓ*, anchors, clamp)
│   ├── evaluate.py                   # PHASE 4 stats (variance reduction, Wilcoxon, ℓ* dist)
│   ├── config.yaml                   # central config for Phases 3-4 + H100 model roles registry
│   ├── src/
│   │   ├── scores.py                 # ★ canonical PRI/ORI/IFI/Final formulas (single source of truth)
│   │   ├── csv_io.py                 # ★ CSV schemas, incremental writer, validators, resume keys
│   │   ├── config.py                 # Config dataclass; loads repo-root .env
│   │   ├── data_loader.py            # routes .jsonl→GenSens bridge, .json→legacy
│   │   ├── benchmark.py              # orchestration: full / generate-only / score-from-CSV
│   │   ├── evaluator.py              # ★ per-sample evaluation pipeline
│   │   ├── model_interface.py        # HF subject-model wrapper (generate, PPL, BF)
│   │   ├── embeddings.py             # SentenceTransformer helper (EMBEDDER_MODEL)
│   │   ├── llm_backend.py            # shared causal instruct backend (vLLM-or-HF, cached)
│   │   ├── llm_judge.py              # LLM-as-a-Judge (flan-t5 seq2seq OR causal 70B)
│   │   ├── faithfulness_metric.py    # NLI faithfulness (deberta) or LLM-as-NLI (70B)
│   │   ├── sms_metric.py · auc_e_metric.py · trd_metric.py · kpig_metric.py
│   │   ├── correctness_metric.py · hallucination_metric.py · metrics_advanced.py
│   │   ├── ppl_variance.py · branching_factor.py · attribution_matrix.py
│   │   ├── logit_lens.py · sensitive_layer.py · anchor_tokens.py · pirc.py   # PHASE 3 core
│   │   ├── analysis.py · table_utils.py · cache_manager.py · prompt_generator.py
│   │   └── pri_calculator.py         # LEGACY harmonic-mean PRI (superseded by scores.py)
│   └── results/                      # all outputs (CSVs, JSON, plots)
│
├── local_pirc_smoke_test.py          # standalone GPT-2 LL-PIRC smoke test
├── start.sh                          # laptop runner (<2B models, all phases)
├── run_h100_eval.sh                  # H100 runner (70B/8B, Phase 1→2)
├── .env / .env.example               # central secrets/config (HF_TOKEN, model roles)
├── CLAUDE.md · method.md             # AI-assistant instructions + methodology changelog
└── IMPLEMENTATION.md · EVALUATION.md · H100_PIPELINE.md   # ← these docs
```

★ = files central to this cycle's architecture.

---

## 3. Phase 1 — GenSens (dataset generation)

**Goal:** for each source instance, produce K *semantically-equivalent but
lexically-different* paraphrases of the instruction, so Phase 2 can test whether
a model's output is stable across them.

**Source loaders** (`data_loaders.py`) — each returns the unified record schema
`{instance_id, task, base_text, base_prompt, metadata}`:

| Task | Dataset | Selection rule |
|------|---------|----------------|
| `summarization` | CNN/DailyMail 3.0.0 test | stratified by length (short<400 / 400–800 / >800 words) |
| `code` | HumanEval (164) + MBPP | docstring extracted as `base_text` |
| `creative` | WritingPrompts | English, 10–60 words, Reddit tags stripped |
| `dialogue` | MultiWOZ 2.2 | 4–8 turns, ends on a user turn |

**Paraphrase generation** (`paraphrase_generator.py`): for each instance it
walks **16 rewrite strategies** (formal/casual tone, reordered clauses, synonyms,
passive voice, question form, …) and keeps a candidate only if it passes **all**
diversity gates:

- SBERT (`all-mpnet-base-v2`) cosine in the band **[0.82, 0.98]** — too low = meaning drift, too high = trivial restatement;
- lexical Jaccard overlap ≤ **0.85** vs the base text **and** vs every already-accepted variant;
- ≤ **2** variants from any single strategy;
- not an exact copy, not a first-8-word duplicate, ≥ 5 words.

Backends (selected by `--model`): `local` = flan-t5-large (laptop), `llama` =
Llama-3.1-8B (HF), `vllm` = 70B-AWQ batched (H100).

**Output** — one JSONL record per instance:

```jsonc
{
  "instance_id": "summ_0001",
  "task": "summarization",
  "base_text": "Summarize the following news article…",
  "base_prompt": "…\n\nArticle:\n<full article>\n\nSummary:",
  "metadata": { "article": "<full article>", "gold_summary": "<highlights>" },
  "variants": [
    { "variant_idx": 0, "strategy": "formal_tone",
      "paraphrased_text": "…", "sbert_similarity": 0.91,
      "full_prompt": "<base_prompt with base_text replaced>" }
  ],
  "diversity": { "distinct_2": 0.74, "mean_pairwise_overlap": 0.31, "n_strategies": 4 }
}
```

`generate_dataset.py --save_final` also writes a flattened `.csv` alongside the
`.jsonl`. `validate_dataset.py` then checks the **newest** dataset file (sorted by
mtime, R7) for expected instance/variant counts and similarity-band compliance.

---

## 4. Phase 2 — PRI Benchmark (robustness scoring)

**Goal:** for each subject model, run all paraphrase variants of an instance and
score how *robust* (consistent + faithful + correct) its outputs are.

### 4.1 The GenSens bridge (R1)

`data_loader.load_dataset(path)` routes by extension:
- `.jsonl` → `load_gensens_dataset()` produces samples with **real** GenSens
  `prompt_variants` (no synthetic templates);
- `.json` → legacy list-of-dicts.

This closes the Phase 1 → Phase 2 loop: the evaluator scores the *actual*
SBERT-filtered paraphrases, not invented `d1/d2/d3` prompts.

### 4.2 Per-sample pipeline (`evaluator.evaluate_sample`)

For one `(model, instance)`:

1. **Variants** — use `sample["prompt_variants"]` (GenSens) or fall back to
   synthetic `d1/d2/d3` templates (`prompt_generator.py`).
2. **Responses** — either generate via `ModelInterface` (greedy, `do_sample=False`
   by default, R2) **or** read `sample["precomputed_responses"]` (score-from-CSV
   mode; no generation model needed).
3. **Metrics** — embed responses once, then compute SMS, AUC-E, semantic TRD,
   coverage KPIG, correctness (CS), hallucination (HS, diagnostic only), NLI
   faithfulness, and (only when a generation model is present) PPL-variance & BF.
4. **Composites** — `PRI`, `ORI`, `IFI` via `src/scores.py`; `Human_Score` via
   `llm_judge_mean` over **all** variants (R5); `Final_Score = 0.6·PRI + 0.4·Human`.

All metric definitions live in [`EVALUATION.md`](EVALUATION.md).

### 4.3 Three execution modes (`benchmark.py`, dispatched by `main.py`)

| Mode | CLI | Loads | Writes | Use |
|------|-----|-------|--------|-----|
| **Full** | `python main.py --dataset X.jsonl` | subject **+** scorers | responses.csv + scored_samples.csv | laptop / single small model |
| **Generate-only** | `python main.py --dataset X.jsonl --generate-only` | subject **only** | responses.csv | H100 Phase 2a (one big model at a time) |
| **Score-from-CSV** | `python main.py --responses-csv results/responses.csv` | scorers **only** | scored_samples.csv | H100 Phase 2b (no subject resident) |

Decoupling generate from score is what lets a **single 80 GB H100** run the 8B
subjects and the 70B judge without them ever co-residing in VRAM.

### 4.4 Re-aggregation (`reaggregate.py`)

Once `scored_samples.csv` (raw metric components) exists, recomputing
`PRI/ORI/IFI/Final` is **pure arithmetic** — no GPU. `reaggregate.py` imports the
exact same `src/scores.py` formulas as the live evaluator, so re-ranking models
or re-tuning PRI weights takes seconds and can never drift from production.

---

## 5. The CSV persistence layer (`src/csv_io.py`)

The expensive H100 generation runs **once**; everything downstream is recomputable
from CSV. Three layers:

| Layer | File | Grain | Schema constant |
|-------|------|-------|-----------------|
| A | `dataset.csv` | one row per prompt variant | `DATASET_COLS` |
| B | `responses.csv` | one row per model × instance × variant | `RESPONSES_COLS` |
| C | `scored_samples.csv` | one row per model × instance (raw metric components) | `SCORED_COLS` |

From **C**, all composite scores are derived arithmetically (Section 4.4).

**Fault tolerance** — `IncrementalCSVWriter` flushes **and `os.fsync`s** after every
write, so whatever has been computed is already durable on disk if the box crashes
or is killed. On restart, `existing_response_keys()` / `existing_scored_keys()`
return the `(model, instance_id)` pairs already present, and the orchestrator
**skips** them (resume). Writers open in append mode and write the header only for
a fresh file.

This was dry-run-verified: re-running a partially complete job skips all
already-done pairs; PRI recomputed offline equals the live value to machine
precision.

---

## 6. Phase 3 — LL-PIRC (mitigation)

> Driven by `config.yaml`. This is the deferred mitigation track (it is *not* part
> of the `run_h100_eval.sh` measurement runner). Core algorithm modules:
> `logit_lens.py`, `sensitive_layer.py`, `anchor_tokens.py`, `pirc.py`.

**Baseline** (`experiment_baseline.py`, Phase 2-style here): for 100 CNN/DM
articles, generate K=5 back-translation paraphrases of the summarization
instruction, greedily decode each, record **ROUGE-L variance** across the 5
outputs → `results/baseline.json` (+ `ifi_metrics.json`).

**PIRC** (`experiment_pirc.py`) — for each article's K paraphrases:

1. **Logit Lens** — at each decoder layer ℓ, project the hidden state through the
   final norm + `lm_head` and compute per-token perplexity.
2. **Sensitive layer ℓ\*** — `S(ℓ) = Var_k[mean per-token PPL at ℓ]` across the K
   paraphrases; `ℓ* = argmax_ℓ [S(ℓ) − S(ℓ−1)]` (inflection), scanning from 25 %
   depth, with a z-score>2 fallback.
3. **Anchor tokens** — at ℓ\*, rank positions by `rank(mean_ppl)+rank(var_ppl)`
   and select the bottom **30 %** most-stable (content-word filtered; guaranteed
   ≥1, ≤⌊min_len/2⌋ anchors).
4. **Clamp & generate** — compute the consensus hidden state `mean_h` across K
   variants; register a forward hook at ℓ\* that sets
   `h[anchors] = α·mean_h + (1−α)·h_original` (α=1 full, α=0.5 soft); greedy decode.

**Auto-retry:** try τ ∈ [1.5, 2.0, 3.0]; if mean ROUGE-L drops > 2 points vs
baseline, fall back to soft clamping (α=0.5). Output → `results/pirc.json`.

---

## 7. Phase 4 — Statistical Evaluation (`evaluate.py`)

Loads `baseline.json` + `pirc.json`, aligns by `article_idx`, and reports:

- **Relative variance reduction** = `1 − mean(var_pirc)/mean(var_baseline)`
  (PIRC produces one deterministic output → `var_pirc = 0` by construction);
- **Wilcoxon signed-rank tests** at α=0.01 — variance (one-sided, baseline>PIRC)
  and ROUGE-L quality (two-sided);
- **ℓ\* distribution** (mean / std / range / coefficient of variation);
- plots: `layer_sensitivity_plot.png`, `variance_comparison.png`; summary →
  `results/eval_summary.json`.

---

## 8. Configuration

Two layers, by design:

- **`.env`** (repo root, gitignored) — secrets + model roles for the
  **measurement** pipeline (Phases 1–2). Loaded by `python-dotenv`
  (`find_dotenv(usecwd=True)`, existing env vars win) and `source`d by both shell
  runners. Keys: `HF_TOKEN`, `MODEL_NAME` (subjects), `JUDGE_MODEL`,
  `JUDGE_QUANTIZATION`, `EMBEDDER_MODEL`, `FAITHFULNESS_BACKEND`,
  `FAITHFULNESS_MODEL`, `LLM_BACKEND`, `GENERATOR_MODEL`, `DO_SAMPLE`, `SEED`,
  `MAX_NEW_TOKENS`. `.env.example` is the committed template (empty `HF_TOKEN`).
- **`config.yaml`** — the **intervention** pipeline (Phases 3–4): target model,
  logit-lens / sensitive-layer / anchor / PIRC / retry hyperparameters, plus a
  `roles:` registry documenting the H100 model assignments.

`src/config.py` (`Config` dataclass) holds Phase-2 runtime defaults (decoding,
weights, paths, feature flags) and is overridden by CLI args / env.

---

## 9. How to run

```bash
# ── Laptop, <2B models, all phases (smoke) ──────────────────────────────
./start.sh                       # GenSens (flan-t5-large) → PRI (<2B) → LL-PIRC (gpt2-medium)
./start.sh --skip-gensens        # reuse an existing dataset

# ── H100, 70B/8B, Phases 1→2 ────────────────────────────────────────────
export HF_TOKEN=hf_...           # set in .env; accept Llama licenses on HF first
./run_h100_eval.sh               # 70B paraphrases → 8B generate-only → 70B score → reaggregate

# ── Manual, per phase ───────────────────────────────────────────────────
cd gensens && python scripts/generate_dataset.py --task all --n_instances 200 --n_variants 8
cd prompt_robustness
python main.py --dataset ../gensens/data/<dataset>.jsonl     # Phase 2 (full)
python reaggregate.py                                        # re-score from CSV (no GPU)
python experiment_baseline.py && python experiment_pirc.py   # Phase 3
python evaluate.py                                           # Phase 4
```

See [`H100_PIPELINE.md`](H100_PIPELINE.md) for the box setup, model roles, and the
single-card execution order.

---

## 10. Coding conventions & invariants

- Python 3.9+, type hints; every metric **clamped to [0,1]** via `max(0,min(1,x))`.
- **Greedy decoding** (`do_sample=False`) by default so output variance reflects
  *prompt* sensitivity, not sampling noise (R2).
- **Single source of truth:** PRI/ORI/IFI/Final live only in `src/scores.py`;
  `evaluator.py` and `reaggregate.py` both import it.
- Forward hooks handle **both** tuple and plain-Tensor layer outputs (transformers
  v5.x compatibility).
- Every CSV write is `flush+fsync`; every long run is **resume-safe**.
- **Any** change to a metric, formula, threshold, or numerical behavior MUST be
  logged in [`method.md`](method.md) (see CLAUDE.md for the mandatory format).
