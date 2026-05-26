# H100_PIPELINE.md — Working of Code, API Details & Pipeline to H100

> How the code actually runs on a rented **single H100 (80 GB)** box: the model
> roles, the API/auth (HuggingFace) details, why generation and scoring are
> decoupled, and the exact run order. For metric math see
> [`EVALUATION.md`](EVALUATION.md); for the full architecture see
> [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

---

## 1. Target environment

A **raw H100 box** (e.g. Jarvis Labs): you SSH in, run your own vLLM/HF processes,
weights stay private, no cold starts. This is *not* a hosted inference API — there
is no per-token billing endpoint. The only external API used is the **HuggingFace
Hub** (for downloading gated weights), authenticated by `HF_TOKEN`.

**The hard constraint:** one 80 GB card. A bf16 70B model needs >1 GPU, and you
cannot hold a 70B judge + an 8B subject + a 7B embedder resident at once. The
pipeline is architected around loading **one big model per process**.

---

## 2. Model roles (`config.yaml` → `roles:`, mirrored in `.env`)

| Role | Model | Backend | Phase |
|------|-------|---------|-------|
| **Generator** | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | vLLM (`awq_marlin`) | 1 — paraphrases |
| **Subjects** (under test) | `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` | HF (one at a time) | 2a — generate |
| **Embedder** | `Alibaba-NLP/gte-Qwen2-7B-instruct` | sentence-transformers (`trust_remote_code`) | 2b — SMS/CS/TRD |
| **Judge** | `…70B-Instruct-AWQ-INT4` | vLLM / `llm_backend.ChatLLM` | 2b — Human_Score |
| **Faithfulness (LLM-as-NLI)** | `…70B-Instruct-AWQ-INT4` (reuses judge) | `llm_backend.ChatLLM` | 2b — NLI axis |
| **PIRC target** | `meta-llama/Llama-3.1-8B-Instruct` | HF (needs hidden-state hooks) | 3 — mitigation |

Design rationale (the architecture you approved):
- **70B as generator** writes high-quality, diverse paraphrases.
- **3 cross-family 8B subjects** are the models whose prompt-robustness we rank —
  using different families (Llama/Qwen/Mistral) avoids same-family bias.
- **70B as judge + faithfulness**, independent of the 8B subjects, so the
  evaluator is not grading models from its own family.
- **AWQ-INT4** quantization shrinks the 70B to ~40 GB → fits one 80 GB card with
  room for the KV cache.

---

## 3. API / authentication details

```bash
# On the box, before anything:
export HF_TOKEN=hf_xxxxxxxx          # or set it in .env (gitignored)
huggingface-cli login --token "$HF_TOKEN"   # optional; token is read from env anyway
```

- **Accept the licenses** for the gated Llama repos on huggingface.co **with the
  same account** that owns the token — otherwise the download 403s.
- `HF_TOKEN` is read by `experiment_*.py` via `os.environ['HF_TOKEN']` and by the
  HF/vLLM loaders automatically. It is **never committed**: `.env` is gitignored
  and `.env.example` ships an empty placeholder.
- Where the token/config is loaded:
  - shell runners `source` `.env` (`set -a; . .env; set +a`);
  - `src/config.py` calls `load_dotenv(find_dotenv(usecwd=True))` so any process
    started anywhere under the repo picks up the root `.env`. **Existing env vars
    win**, so an explicit `EMBEDDER_MODEL=… python …` overrides `.env`.

```bash
# Prereqs on the box:
pip install vllm sentence-transformers transformers accelerate \
            rouge-score scipy scikit-learn pandas pyyaml tqdm python-dotenv
```

---

## 4. Why generate and score are decoupled

Loading a model in this codebase is role-specific:

- **Generate-only** (`main.py --generate-only` → `benchmark.generate_responses_to_csv`)
  loads **only** the subject model; no embedder, judge, or NLI. Writes
  `responses.csv`.
- **Score-from-CSV** (`main.py --responses-csv …` → `benchmark.score_from_responses_csv`)
  loads **only** the scorers (7B embedder + 70B judge/NLI); the subject is never
  loaded — it calls `evaluate_sample(model_interface=None)` and reads
  `precomputed_responses` from the CSV.

So at no point do an 8B subject and the 70B judge co-reside. The 70B is loaded
**once** for all subjects' responses (the judge model is cached by id in
`llm_backend.get_chat_llm`). This is the only way the whole evaluation fits on one
card.

---

## 5. How `ChatLLM` picks a backend (`src/llm_backend.py`)

The shared causal instruct backend (used by the judge and LLM-as-NLI) selects via
`LLM_BACKEND`:
- `auto` (default) → try **vLLM**, fall back to **HF transformers** if vLLM isn't
  importable;
- `vllm` / `hf` → force one.

vLLM is configured `tensor_parallel_size=1`, `gpu_memory_utilization=0.90`,
`max_model_len=4096`, and `quantization` from `JUDGE_QUANTIZATION` /
`LLM_QUANTIZATION` (e.g. `awq_marlin`). All heavy imports are lazy, so the same
module imports fine on a laptop without CUDA/vLLM. The judge is loaded once and
reused across every scored sample.

---

## 6. The H100 run order (`run_h100_eval.sh`)

```
PHASE 1   generate paraphrase dataset      (70B-AWQ via vLLM)         → dataset .jsonl/.csv
PHASE 2a  per subject: GENERATE responses  (8B only, no scorers)      → responses.csv
PHASE 2b  SCORE from responses.csv         (7B embedder + 70B judge/NLI) → scored_samples.csv
PHASE 2c  reaggregate                       (no GPU)                   → PRI/ORI/Final
```

Concretely, the script:

1. `source`s `.env`.
2. **Phase 1** — `gensens/scripts/generate_dataset.py --model vllm --model-id <70B>
   --quantization awq_marlin --task summarization --n_instances 200 --n_variants 8`
   (skippable with `--skip-gensens` to reuse a dataset).
3. **Phase 2a** — loops the 3 subjects, each:
   `MODEL_NAME=<subject> python main.py --dataset <dataset>.jsonl --generate-only`
   → all subjects' rows accumulate in one resume-safe `responses.csv`.
4. **Phase 2b** — once, with scorers only:
   `EMBEDDER_MODEL=<7B> JUDGE_MODEL=<70B> JUDGE_QUANTIZATION=awq_marlin
   FAITHFULNESS_BACKEND=llm FAITHFULNESS_MODEL=<70B> python main.py
   --responses-csv results/responses.csv` → `scored_samples.csv`.
5. **Phase 2c** — `python reaggregate.py --scored results/scored_samples.csv`
   (pure arithmetic, can also be run later on your laptop, free).

Override anything via env, e.g. `SUBJECTS="meta-llama/Llama-3.1-8B-Instruct"
./run_h100_eval.sh --skip-gensens`.

---

## 7. Crash safety on a paid box

Every CSV write is **`flush + os.fsync`** per sample (`csv_io.IncrementalCSVWriter`),
so computed values are durable the instant they're produced — an OOM, a killed
spot box, or a model error never loses prior work.

On restart the orchestrator reads `existing_response_keys()` /
`existing_scored_keys()` and **skips** the `(model, instance_id)` pairs already on
disk (writers reopen in append mode). So you can:

- re-run after a crash and it continues exactly where it stopped;
- pull `responses.csv` / `scored_samples.csv` back to your laptop and re-aggregate
  offline (Phase 2c needs no GPU) — so you only pay H100 time for generation +
  scoring, once.

This was dry-run-verified locally: re-running a partially complete job skipped all
done pairs, and offline-recomputed PRI matched the live value to machine precision.

---

## 8. Phase 3 (LL-PIRC) on the box

The mitigation track is separate from the measurement runner and is driven by
`config.yaml` (`model.name = Llama-3.1-8B-Instruct`, `device_map: auto`,
`dtype: float16`). It needs **HF transformers** (not vLLM) because PIRC registers
forward hooks on decoder layers to read/clamp hidden states — vLLM doesn't expose
those. Run:

```bash
cd prompt_robustness
python experiment_baseline.py     # K=5 back-translation paraphrases, ROUGE-L variance → baseline.json
python experiment_pirc.py         # ℓ* detection, anchor clamp → pirc.json  (auto-retry τ, soft-clamp fallback)
python evaluate.py                # variance reduction + Wilcoxon + ℓ* dist → eval_summary.json
```

`device_map="auto"` shards the 8B target across available memory; HF reads
`HF_TOKEN` from the env for the gated Llama weights.

---

## 9. Local development mirror (`start.sh`)

Everything above is exercisable on a laptop (Apple Silicon MPS / CPU) with **only
<2B models**, so you can validate wiring before paying for the box:

| Phase | Laptop model (<2B) | H100 model |
|-------|--------------------|------------|
| 1 GenSens | `flan-t5-large` (~770 M) | 70B-AWQ via vLLM |
| 2 PRI | flan-t5-base/large, distilbart, bart-large-cnn | 3× 8B subjects |
| 3 LL-PIRC | `gpt2-medium` (~355 M) | Llama-3.1-8B |

The `gpt2-medium` smoke test exercises the **identical** LL-PIRC code path
(Logit Lens → ℓ\* → anchors → clamp). `start.sh` sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` and `TOKENIZERS_PARALLELISM=false` for Mac.

---

## 10. Pre-flight checklist before renting the box

- [ ] `HF_TOKEN` set in `.env` (and **not** committed — verify `git status`).
- [ ] Llama-3.1 (8B + 70B-AWQ) and Qwen/Mistral licenses accepted on HF for that token.
- [ ] `pip install vllm sentence-transformers transformers accelerate rouge-score scipy scikit-learn pandas pyyaml tqdm python-dotenv`.
- [ ] Repo-wide imports + end-to-end dry run pass locally (already verified this cycle).
- [ ] Decide scale: `N_INSTANCES` / `N_VARIANTS` (defaults 200 / 8 in `run_h100_eval.sh`).
- [ ] Confirm `SUBJECTS`, `JUDGE_MODEL`, `EMBEDDER_MODEL` in `.env` match `config.yaml::roles`.

Then: `./run_h100_eval.sh`. Pull `results/responses.csv` + `results/scored_samples.csv`
back home and iterate on PRI weights with `reaggregate.py` for free.
