# DATASET_GENERATION.md — GenSens (Phase 1) End-to-End

> Complete reference for **how the GenSens benchmark dataset is generated**: the
> source data, the generation models (with every hyperparameter), the 16
> paraphrase strategies, the filtering/diversity gates, the output schema, and how
> to run it locally or on the H100. For the rest of the system see
> [`IMPLEMENTATION.md`](IMPLEMENTATION.md); for how this dataset is consumed see
> [`EVALUATION.md`](EVALUATION.md).

---

## 1. What GenSens produces

For each source **instance** (a task input), GenSens generates up to *N* paraphrase
**variants** of the instruction — reworded but semantically equivalent — so that
Phase 2 can test whether a model's output stays stable across them.

```
source dataset ──▶ base_text + base_prompt
                      │
                      ▼  (16 rewrite strategies × paraphrase model)
                  candidate paraphrases
                      │
                      ▼  (SBERT band + lexical-diversity + per-strategy caps)
                  N accepted variants  ──▶  full_prompt (base_prompt with the paraphrase substituted)
                      │
                      ▼
            gensens_<task>_<N>inst_<K>var.jsonl  (+ .csv, + stats.json)
```

**Files involved** (all under `gensens/scripts/`):

| File | Role |
|------|------|
| `generate_dataset.py` | orchestrator + CLI; loads data, runs generation per task, checkpoints, writes JSONL/CSV/stats |
| `data_loaders.py` | source loaders for the 4 tasks (returns the unified record schema) |
| `paraphrase_generator.py` | the 16 strategies, the 3 model backends, the SBERT/diversity filters |
| `validate_dataset.py` | post-generation quality checks |

---

## 2. The four tasks and their source data (`data_loaders.py`)

Every loader returns records with the **unified schema**
`{instance_id, task, base_text, base_prompt, metadata}`, where **`base_text` is the
exact substring that gets paraphrased** and `base_prompt` is the full prompt that
embeds it.

**Every task now carries a non-empty reference** (the old creative/dialogue tasks
had none). Each loader writes the canonical bridge keys **`input_text`** (grounding)
and **`reference_output`** (gold) into `metadata`; rows whose source lacks a
reference are skipped.

> **Verified HF paths:** `cnn_dailymail` only exposes configs `1.0.0/2.0.0/3.0.0` —
> *sum_in_brief* and *generate_story* are **PromptSource template framings**, not HF
> configs. Dialogue uses `dream` (`trust_remote_code=True`); QA uses
> `sentence-transformers/eli5` config `pair` (`question`/`answer`, train split only).

| Task | Source dataset (HF) | Split / selection | `base_text` (paraphrased) | `instance_id` |
|------|--------------------|-------------------|---------------------------|---------------|
| `summarization` | `cnn_dailymail` 3.0.0 (*sum_in_brief*) | `test`, **stratified by article length**: short <400, medium 400–800, long >800 words (≈equal per bucket, seed 42) | the summarization instruction | `summ_0000…` |
| `creative` | `cnn_dailymail` 3.0.0 (*generate_story*, inverse) | `test`, **same length-stratified sampler** as summarization (seed 42) | the story-writing instruction | `crea_0000…` |
| `dialogue` | `dream` | `train`, non-empty dialogue/question/answer (shuffled, seed 42) | the **question** | `dial_0000…` |
| `qa` | `sentence-transformers/eli5` (`pair`) | `train`, EN question 5–60 words, answer ≥10 words (shuffled, seed 42) | the **question** | `qa_0000…` |

**`metadata` carried per task** (canonical `input_text` = grounding,
`reference_output` = gold; plus task extras):
- summarization → `{input_text: article, reference_output: highlights, source, word_count, +article/gold_summary aliases}`
- creative → `{input_text: highlights (premise), reference_output: article (gold story), source, word_count}`
- dialogue → `{input_text: dialogue turns, reference_output: correct answer choice, source, dialogue_id, choices}`
- qa → `{input_text: question, reference_output: answer, source}`

For summarization the `base_prompt` is
`"<instruction>\n\nArticle:\n<article>\n\nSummary:"`, so only the *instruction* is
reworded — the article stays fixed across variants. For creative the premise
(highlights) stays fixed; for dialogue the turns + options stay fixed. **Creative
is the inverse of summarization** (premise → full article), so `input_text ≠
reference_output` and the two CNN/DM tasks cover the same pairs inverted. Loaders
are deterministic (seed 42) and per-task failures are caught so one bad task
doesn't abort the rest.

---

## 3. The paraphrase models (`paraphrase_generator.py`)

Three interchangeable backends, selected by `--model`. **The model only writes
candidate paraphrases**; acceptance is decided by the SBERT/diversity filters
(Section 5), which are identical across backends.

### 3.1 `local` — google/flan-t5-large  (~770 M, laptop default)
- seq2seq; loads on **MPS** (Apple Silicon) else CPU.
- Decode (`_generate_single_flan`): `num_beams=4`, `early_stopping=True`,
  `do_sample=True`, `temperature=0.85`, `top_p=0.92`, `max_new_tokens=200`,
  input truncated to 512 tokens.
- Plain-text prompt (`_build_flan_prompt`): strategy instruction + task context +
  `Original: …` + `Rephrased version:`.

### 3.2 `llama` — meta-llama/Meta-Llama-3.1-8B-Instruct  (HF transformers)
- causal; `torch_dtype=bfloat16`, `device_map="auto"`, `local_files_only=True`,
  `padding_side="left"`.
- Decode (`_generate_single_llama`): `do_sample=True`, `temperature=0.85`,
  `top_p=0.92`, `max_new_tokens=300`.
- Uses the chat template with a task-specific **system prompt** + user message;
  single-sequence (slow) — intended for correctness, not throughput.

### 3.3 `vllm` — batched generator for the H100  (production)
- Default model: **`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`**
  (override with `--model-id`); `--quantization awq_marlin` (or `gptq_marlin`).
- vLLM engine: `tensor_parallel_size=1` (single 80 GB card),
  `gpu_memory_utilization=0.90`, `max_model_len=4096`, `dtype="auto"`,
  `trust_remote_code=True`.
- Sampling (`generate_batch`): `temperature=0.8`, `top_p=0.95`, `max_tokens=256`,
  `seed=42`. Batched → ~10–50× faster than the HF path.
- AWQ-INT4 keeps the 70B at ~40 GB so it fits one card with room for the KV cache.

> **SBERT filter model** (all backends): `sentence-transformers/all-mpnet-base-v2`,
> loaded first (small, always needed). Note this is a *different* embedder than the
> Phase-2 PRI embedder — they serve different roles.

### 3.4 Task system prompts (used by `llama`/`vllm` chat templates)
Each task has a dedicated system prompt instructing the model to **rephrase while
preserving exact meaning/intent** and return *only* the rewrite (no commentary):
`summarization`, `creative`, `dialogue`, `qa` (see `SYSTEM_PROMPTS`). Output is
post-cleaned (`_clean_output`) to strip echoed prefixes, quotes, and model
self-commentary (`Note:`, `Explanation:`, …).

---

## 4. The 16 paraphrase strategies

For each instance the generator iterates **all 16** strategies (in order), each a
distinct rewrite instruction, to maximise lexical/structural diversity:

| # | Strategy name | Intent |
|---|---------------|--------|
| 1 | `formal_tone` | more formal/professional |
| 2 | `casual_tone` | casual/conversational |
| 3 | `reordered_clauses` | reorder clauses/phrases |
| 4 | `synonyms` | swap key words for synonyms |
| 5 | `concise` | fewer words |
| 6 | `different_structure` | different sentence structure |
| 7 | `different_opening` | different opening word/phrase |
| 8 | `role_prefix` | add a brief role/context prefix |
| 9 | `passive_voice` | passive constructions |
| 10 | `question_form` | as a question/request |
| 11 | `imperative` | direct imperative command |
| 12 | `elaborated` | slightly more detail |
| 13 | `technical_vocab` | more technical vocabulary |
| 14 | `simple_vocab` | simpler everyday vocabulary |
| 15 | `split_sentences` | split into shorter sentences |
| 16 | `merged_sentences` | merge into longer sentences |

Each accepted variant records which `strategy` produced it.

---

## 5. Filtering & diversity gates (the quality controls)

A generated candidate is **accepted only if it passes every gate**:

| Gate | Rule | Default | Why |
|------|------|---------|-----|
| Min length | `≥ min_word_count` words | 5 | reject degenerate fragments |
| Not exact copy | `candidate ≠ base_text` (case-insensitive) | — | must actually be reworded |
| Dedup signature | first-8-word signature unseen | — | reject near-duplicate openings |
| **SBERT band (lower)** | cosine(base, cand) `≥ similarity_threshold` | **0.82** | below = meaning drift |
| **SBERT band (upper)** | cosine(base, cand) `≤ similarity_upper` | **0.98** | above = trivial restatement, no real perturbation |
| **Lexical floor (vs base)** | Jaccard word overlap `≤ max_token_overlap` | **0.85** | force lexical, not just cosmetic, change |
| **Lexical floor (vs accepted)** | Jaccard overlap vs every accepted variant `≤ max_token_overlap` | **0.85** | keep variants different *from each other* |
| **Per-strategy cap** | `≤ max_per_strategy` variants from one strategy | **2** | prevent one strategy dominating |

SBERT similarity uses `all-mpnet-base-v2` cosine; lexical overlap is the Jaccard of
lowercased word sets (`_token_overlap`).

---

## 6. Generation algorithm (per instance)

`ParaphraseGenerator.generate_paraphrases(record, n_variants, max_retries=3)`:

```
seen_signatures = { signature(base_text) }
strategy_counts = {}
for retry in 0..max_retries:          # up to 3 passes over all strategies
    if len(accepted) >= n_variants: break
    for strategy in STRATEGIES (16):
        if accepted >= n_variants: break
        if strategy_counts[strategy] >= max_per_strategy: skip
        cand = model.generate(strategy, base_text)
        apply all gates from §5
        if pass: accept (record strategy, sbert_similarity, variant_idx); update counts/signatures
re-index variant_idx 0..len-1
return accepted[:n_variants]
```

Then `build_prompt_variants` substitutes each accepted paraphrase back into
`base_prompt` (replacing `base_text`) to produce `full_prompt`; if `base_text`
isn't found verbatim it prepends the paraphrase. `process_instance` also computes
per-instance **diversity stats** (`distinct_2` bigram ratio, `mean_pairwise_overlap`,
`n_strategies`).

> Generation can yield **fewer than `n_variants`** for a hard instance (if not
> enough candidates clear the gates after 3 passes). That is expected; the
> validator (Section 9) flags it as a warning, not an error.

---

## 7. Orchestration, checkpointing & resume (`generate_dataset.py`)

1. Parse CLI; set up file+console logging under `gensens/logs/`.
2. Determine tasks (`all` → all four, else the named one).
3. **Instantiate the ParaphraseGenerator once** (shared across all tasks) with the
   chosen backend + diversity knobs.
4. For each task (`run_task`):
   - load source instances (seed 42);
   - load any existing `gensens_<task>.checkpoint.jsonl` and **skip already-done
     `instance_id`s** (resume);
   - process each instance; on failure, store an empty-variants record so the run
     continues;
   - **checkpoint every `--checkpoint_every` instances** (default 20);
   - on completion, `save_final` (JSONL **+** CSV) and delete the checkpoint.
5. Compute `compute_stats` across tasks → write `gensens_stats.json`, print a table.

---

## 8. Output formats

### 8.1 Per-instance JSONL record (`gensens_<task>_<N>inst_<K>var.jsonl`)
```jsonc
{
  "instance_id": "summ_0001",
  "task": "summarization",
  "base_text": "Summarize the following news article in 3-4 sentences…",
  "base_prompt": "Summarize…\n\nArticle:\n<full article>\n\nSummary:",
  "metadata": { "article": "<full article>", "gold_summary": "<highlights>", "word_count": 612 },
  "variants": [
    { "variant_idx": 0, "strategy": "formal_tone",
      "paraphrased_text": "Provide a concise 3-4 sentence synopsis of the article below…",
      "sbert_similarity": 0.9123,
      "full_prompt": "<base_prompt with base_text replaced by paraphrased_text>" }
  ],
  "n_variants_generated": 8,
  "generation_time_s": 14.2,
  "diversity": { "distinct_2": 0.74, "mean_pairwise_overlap": 0.31, "n_strategies": 5 }
}
```

### 8.2 Flat CSV (`gensens_<task>_<N>inst_<K>var.csv`) — one row per variant
Columns: `instance_id, task, variant_idx, strategy, base_text, paraphrased_text,
sbert_similarity, full_prompt, input_text, reference_output`
(`input_text` = `metadata.article`, `reference_output` = `metadata.gold_summary`).
This is the bridge that lets Phase 2 load the dataset **without re-generating**.

### 8.3 `gensens_stats.json`
Per-task: `n_instances`, `total_variants`, `avg_variants_per_instance`, SBERT
similarity distribution (mean/std/min/max/p25/p75), base- and variant-word-count
stats; plus a `_meta` block (`total_time_s`, `n_instances_per_task`, `n_variants`,
`seed`, `generated_at`).

---

## 9. Validation (`validate_dataset.py`)

Run after generation; picks the **newest** matching file per task (sorted by mtime,
rectification R7) so a fresh run is validated, not a stale one. Seven checks:

| # | Check | Flags |
|---|-------|-------|
| 1 | **Completeness** | wrong instance count (**error**); wrong variant count (warning) |
| 2 | **Similarity distribution** | variants with SBERT < 0.82; prints a histogram |
| 3 | **Diversity** | mean pairwise **word-level Levenshtein** distance < 10 between variants |
| 4 | **Length** | variant < 5 words or > 3× base length |
| 5 | **Exact copies** | a variant identical to `base_text` |
| 6 | **Prompt substitution** | `full_prompt` missing the paraphrase, or still containing the original `base_text` |
| 7 | **Reference present** | an instance with an empty `reference_output` (every task must carry a gold reference) |

A task **passes** if it has zero *errors* (warnings are non-fatal). Writes
`gensens_validation_report.json` (warnings capped at 50/task) and prints a
PASS/FAIL summary table.

---

## 10. CLI usage

```bash
cd gensens

# ── Laptop / development (flan-t5-large, MPS/CPU) ───────────────────────
python scripts/generate_dataset.py \
  --task summarization --n_instances 6 --n_variants 4 \
  --model local --checkpoint_every 2
python scripts/validate_dataset.py --expected-instances 6 --expected-variants 4

# ── Full local dataset, all 4 tasks ─────────────────────────────────────
python scripts/generate_dataset.py --task all --n_instances 200 --n_variants 8 --model local

# ── H100 production (70B-AWQ via vLLM, batched) ─────────────────────────
python scripts/generate_dataset.py \
  --model vllm \
  --model-id hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq_marlin \
  --task summarization --n_instances 200 --n_variants 8 --checkpoint_every 10
```

### CLI flags
| Flag | Default | Meaning |
|------|---------|---------|
| `--task` | `all` | `all` / `summarization` / `creative` / `dialogue` / `qa` |
| `--n_instances` | 200 | instances per task |
| `--n_variants` | 8 | target variants per instance |
| `--seed` | 42 | deterministic source sampling |
| `--checkpoint_every` | 20 | checkpoint frequency (instances) |
| `--model` | `local` | `local` (flan-t5-large) / `llama` (HF 8B) / `vllm` (batched) |
| `--model-id` | None | HF repo id for the vLLM backend |
| `--quantization` | None | vLLM quant: `awq_marlin` / `gptq_marlin` (None = bf16) |
| `--similarity-threshold` | 0.82 | lower SBERT band edge |
| `--similarity-upper` | 0.98 | upper SBERT band edge |
| `--max-token-overlap` | 0.85 | lexical Jaccard rejection threshold |
| `--max-per-strategy` | 2 | cap variants per strategy |

In the runners, **Phase 1 is driven by `start.sh`** (local, flan-t5-large) and
**`run_h100_eval.sh`** (vLLM 70B-AWQ); both pass these flags for you.

---

## 11. Model summary (quick reference)

| Role | Model | Backend | Key generation params |
|------|-------|---------|------------------------|
| Paraphraser (local) | `google/flan-t5-large` | HF seq2seq (MPS/CPU) | beams=4, temp 0.85, top_p 0.92, sample, max_new=200 |
| Paraphraser (HF) | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HF causal (bf16, auto) | temp 0.85, top_p 0.92, sample, max_new=300 |
| Paraphraser (prod) | `…Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | vLLM (awq_marlin) | temp 0.8, top_p 0.95, max_tokens 256, seed 42 |
| Similarity filter | `sentence-transformers/all-mpnet-base-v2` | sentence-transformers | cosine band [0.82, 0.98] |

> Any change to a strategy, gate, threshold, model, or output schema **must** be
> logged in [`method.md`](method.md) per the CLAUDE.md rule.
