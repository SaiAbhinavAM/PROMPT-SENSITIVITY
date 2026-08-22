# Running the Crossed-Design Prompt-Sensitivity Benchmark on an RTX 4090

> Balanced **5-axis × 2 = 10 paraphrases/seed** scheme (method.md 2026-08-22),
> Llama-3.1-8B paraphraser, and 3 subject models: **Llama-3.1-8B, Qwen2.5-7B,
> Mistral-7B-v0.3**. This runbook is written for a single **24 GB RTX 4090**.

---

## 0. What fits on a 4090 (and what doesn't)

| Component | Model | Fits on 24 GB? |
|---|---|---|
| Paraphraser | Llama-3.1-8B (fp16, ~16 GB) | ✅ |
| Subject models | Llama-3.1-8B / Qwen2.5-7B / Mistral-7B (one at a time) | ✅ |
| Faithfulness scorer | MiniCheck `deberta-v3-large` | ✅ |
| Correctness scorer | BERTScore `roberta-large` | ✅ |
| NLI gate | `nli-deberta-v3-*` (paraphrase filtering) | ✅ |
| ~~70B judge / 70B faithfulness~~ | Llama-3.1-70B-AWQ (~40 GB) | ❌ **do not use** |

The crossed pipeline already uses the **deterministic scorers** (MiniCheck +
BERTScore) by default — **no 70B model is needed**. Everything here runs on one
4090.

---

## 1. Environment — pick ONE path

### Path A (recommended): **WSL2 (Ubuntu on Windows)**
vLLM is Linux-only; WSL2 gives CUDA passthrough to the 4090 so the *whole*
pipeline (paraphrase generation **and** response collection) runs unchanged,
and you stay on the same backend the existing results were validated on.

```powershell
# One-time, in an elevated PowerShell:
wsl --install -d Ubuntu
# reboot, set up your Ubuntu user, then work inside the Ubuntu shell.
```
Install the NVIDIA CUDA-on-WSL driver on the **Windows** side (not inside WSL);
`nvidia-smi` should then work inside Ubuntu.

### Path B: **native Windows (PowerShell)**
vLLM will **not** install. You can still run **paraphrase generation** with the
HF-transformers backend (`--backend llama`), but **Phase-1 response collection
(`run_inference_crossed.py`) currently requires vLLM** — see "Known limitations"
at the bottom. Use Path A unless you can't.

---

## 2. Install

```bash
# from the repo root: "PROMPT SENSITIVITY"
python -m venv .venv && source .venv/bin/activate      # (Windows: .venv\Scripts\activate)

# PyTorch matched to your CUDA (example: cu124):
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# Core stack:
pip install -r requirements_gpu.txt -r gensens/crossed/requirements_crossed.txt

# MiniCheck is a git install; if it conflicts, install separately — the pipeline
# falls back to the NLI faithfulness backend automatically:
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"

# Gated Llama weights need a HF login:
huggingface-cli login
```

> **Native Windows only:** skip `vllm` from `requirements_gpu.txt`
> (`pip install transformers accelerate sentence-transformers datasets rouge-score
> scikit-learn scipy nltk bert-score` + the MiniCheck line).

---

## 3. Step 1 — Generate the balanced 10-paraphrase summarization dataset

Uses the updated `ParaphraseGenerator` (5 balanced axes incl. FORMAT, gates
0.92 / 0.70, FORMAT diversity-exempt). Output → `gensens/data/gensens_summ_50seed_10para.jsonl`.
**Checkpoints every 10 seeds — resumable (`--resume` is on by default).**

```bash
# Path A (WSL2 / vLLM):
python gensens/scripts/generate_summ_paraphrases.py \
    --backend vllm --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.85

# Path B (native Windows / HF transformers):
python gensens/scripts/generate_summ_paraphrases.py --backend llama
```
Defaults already set for you: `--n-variants 10`, output filename, Llama-3.1-8B.

**Before the big run, smoke-test 3 seeds** to confirm all 5 axes fill:
```bash
python gensens/scripts/generate_summ_paraphrases.py --backend llama --no-resume \
    --output /tmp/summ_smoke.jsonl 2>&1 | head -40
# then eyeball: each seed should show lexical/syntactic/pragmatic/length/format
```

---

## 4. Step 2 — Collect responses for all 3 subject models

**WSL2 only** (uses vLLM, greedy, resumable; checkpoints every 2000 cells). Run
once per model — each writes to its own results dir. Start with `N_ARTICLES=30`
(matches the existing runs) or go to `100` for more power.

```bash
# tuned for 24 GB: lower mem fraction + skip the OOM-prone PPL reload if needed
export GPU_MEM_FRAC=0.85 N_ARTICLES=30 MAX_TOKENS=512

# 1) Llama-3.1-8B
MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  bash gensens/crossed/run_crossed_h100.sh

# 2) Qwen2.5-7B  → point results elsewhere so you don't overwrite Llama:
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
  bash gensens/crossed/run_crossed_h100.sh          # writes to gensens/crossed/results

# 3) Mistral-7B-v0.3
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 \
  bash gensens/crossed/run_crossed_h100.sh
```

> **Keep runs separate.** The script writes to `gensens/crossed/results/` by
> default. Before each new model, move the previous output aside
> (`mv gensens/crossed/results gensens/crossed/results_llama`, etc.) — mirror how
> `results_qwen/` was kept for the earlier run.

The script runs **all 6 phases** per model: (1) inference → (2) per-cell metrics
(SMS/CS/Faith/PPL/PC/ROUGE/BERTScore) → (3) composite + aggregation →
(4) diagnosis 2×2 → (5) significance → (6) `summary_report.md`.

### 4090 OOM knobs (if Phase 1 or the PPL step runs out of memory)
```bash
export GPU_MEM_FRAC=0.80          # give vLLM less, leave headroom for scorers
export SKIP_PPL_ENTROPY=1         # 4-metric run, skips the 8B reload (most OOM-prone)
export FAITH_WHOLE_SUMMARY=1      # ~4× faster MiniCheck, minimal accuracy loss
# and in run_inference: --max_model_len 4096 if articles overflow the KV cache
```

---

## 5. Step 3 — Cross-model comparison, robustness, significance

```bash
# pairwise replication comparison (repeat for each pair):
python gensens/crossed/scripts/compare_models.py \
    --a gensens/crossed/results_llama   --label_a Llama-3.1-8B \
    --b gensens/crossed/results_qwen    --label_b Qwen2.5-7B

# assumption-free robustness (length confound / split-half / permutation / paraphrase equivalence):
python gensens/crossed/scripts/validate_robustness.py --results_dir gensens/crossed/results_mistral
```
The 3rd model (Mistral) is the **tie-breaker** for the borderline Llama-strong /
Qwen-weak permutation result — that's the whole reason for this run.

---

## 6. Save-as-you-generate / resumability (already built in)

- **Paraphrase generation** checkpoints every 10 seeds → `gensens_summ_10para.checkpoint.jsonl`; re-run with `--resume` (default) to continue.
- **Response collection** checkpoints every 2000 cells → `responses_crossed.jsonl`; re-running the same command resumes from the last checkpoint.
- Nothing is lost if the box dies mid-run — just re-issue the same command.

---

## 7. Known limitations / open items

1. **Native Windows can't run Phase-1 response collection yet.**
   `run_inference_crossed.py` imports vLLM with no HF fallback. Options: use WSL2
   (Path A), or add an HF-transformers backend to that script (planned). Paraphrase
   generation already has the `--backend llama` (HF) path.
2. **Crossed pipeline is summarization-only.** QA / Dialogue / Creative datasets
   generate fine (`generate_dataset.py --task all`), but the crossed
   response+metric pipeline currently consumes the summarization seed schema
   (`prompt_id`/`pool`/`dimension`). Extending it to the other 3 tasks is a
   separate step.
3. **Per-axis analysis wiring.** The axis label (`strategy_family`) is now carried
   through `common.load_seed_prompts`, but the aggregation/plots don't yet break
   sensitivity down *by axis* — add that in `aggregate_scores.py` /
   `dimension_analysis.py` to report "which rewrite axis each model is most
   sensitive to."
4. **Regeneration invalidates prior results.** The new 10-paraphrase dataset
   replaces the old 5-paraphrase one; the existing Llama+Qwen crossed results were
   built on the old scheme and must be re-run to compare on equal footing.
